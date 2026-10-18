"""
Comparison with Cirq's built-in two-qubit decompositions and compilers.

Native gate sets and the Cirq routine each is compared with:

=============  ==========================================================
free FSim      the shortest of ``two_qubit_matrix_to_cz_operations`` with
               full CZs, with partial CZs (``allow_partial_czs=True``; a
               partial CZ is FSim(0, φ)) and
               ``two_qubit_matrix_to_sqrt_iswap_operations``.
Sycamore       ``cirq_google.SycamoreTargetGateset`` (when cirq_google is
               installed; it is not a dependency), given the named gate
               where there is one. The cirq-core baseline: local targets
               take no gate; controlled-phase classes (Weyl y = z = 0) go
               through a KAK change of basis and
               ``decompose_cphase_into_two_fsim`` (two gates); everything
               else through ``decompose_two_qubit_interaction_into_four_fsim_gates``.
√iSWAP         ``two_qubit_matrix_to_sqrt_iswap_operations``
CZ             ``two_qubit_matrix_to_cz_operations`` (``allow_partial_czs=False``)
=============  ==========================================================

Whole circuits are compiled with ``cirq.optimize_for_target_gateset`` and
``cirq.CZTargetGateset``, ``cirq.SqrtIswapTargetGateset`` or
``cirq_google.SycamoreTargetGateset``. All of those merge two-qubit blocks
before decomposing. The cirq-core Sycamore baseline merges blocks with
``cirq.merge_k_qubit_unitaries`` and applies the rule above to each.

Approximate Cirq route: ``cirq.two_qubit_gate_product_tabulation`` (also
behind ``SycamoreTargetGateset(tabulation=...)``) compiles to the Sycamore
gate approximately; :func:`tabulation_study` compares it with noise-aware
synthesis.

Timing: every routine, this package's included, is timed as the best of
three calls after warm-up.

All infidelities are process infidelities 1 − |Tr(U†V)|²/d². The estimated
fidelity of a compiled circuit is F_synthesis · (1 − ε)^{n_2q}, where ε is the
process infidelity of one two-qubit gate and single-qubit errors are ignored.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import weyl
from .calibration_map import (
    CouplerCalibration,
    SYCAMORE_GATE_ERROR,
    WILLOW_CZ_ERROR,
    SQRT_ISWAP_GATE_ERROR,
)
from .riemannian_optimizer import DifferentiableFSimSynthesizer, SynthesisWarning
from .cirq_transformer import FSimDecomposerTransformer, convert_decomposition_to_cirq_ops
from .unitary_ansatz import haar_random_unitary

try:
    import cirq

    HAS_CIRQ = True
except ImportError:  # pragma: no cover
    HAS_CIRQ = False
    cirq = None

GATE_SETS: Tuple[str, ...] = ("free", "sycamore", "sqrt_iswap", "cz")
LABELS = {"free": "free FSim", "sycamore": "Sycamore", "sqrt_iswap": "√iSWAP", "cz": "CZ"}
#: per-gate process infidelity used for each gate set in the benchmark (sources in
#: calibration_map). No published figure exists for a free-angle FSim family; the benchmark
#: gives it the Sycamore figure. The library itself asks for ``gate_error`` in that case.
GATE_ERRORS = {"free": SYCAMORE_GATE_ERROR, "sycamore": SYCAMORE_GATE_ERROR,
               "sqrt_iswap": SQRT_ISWAP_GATE_ERROR, "cz": WILLOW_CZ_ERROR}


def calibration_for(gate_set: str) -> Optional[CouplerCalibration]:
    return {"free": None, "sycamore": CouplerCalibration.sycamore(),
            "sqrt_iswap": CouplerCalibration.sqrt_iswap(), "cz": CouplerCalibration.cz()}[gate_set]


def process_infidelity(u: np.ndarray, v: np.ndarray) -> float:
    d = u.shape[0]
    return float(max(0.0, 1.0 - abs(np.trace(u.conj().T @ v)) ** 2 / d**2))


def _two_qubit_count(ops) -> int:
    ops = ops.all_operations() if isinstance(ops, cirq.AbstractCircuit) else ops
    return sum(1 for op in ops if len(op.qubits) == 2)


def _ops_unitary(ops, qubits) -> np.ndarray:
    c = ops if isinstance(ops, cirq.AbstractCircuit) else cirq.Circuit(ops)
    return c.unitary(qubit_order=list(qubits), qubits_that_should_be_present=list(qubits))


# --------------------------------------------------------------------------- #
# Two-qubit targets and Cirq baselines
# --------------------------------------------------------------------------- #
def two_qubit_targets(n_haar: int = 200, seed: int = 2026, cphase_steps: int = 8) -> List[Dict[str, Any]]:
    """Named gates, CPhase(kπ/steps) for k = 1..steps−1, and seeded Haar-random U(4)."""
    targets = []
    for name, gate in (("CNOT", cirq.CNOT), ("CZ", cirq.CZ), ("iSWAP", cirq.ISWAP),
                       ("√iSWAP", cirq.SQRT_ISWAP), ("SWAP", cirq.SWAP)):
        targets.append({"name": name, "group": name, "unitary": cirq.unitary(gate), "gate": gate})
    for k in range(1, cphase_steps):
        gate = cirq.CZPowGate(exponent=k / cphase_steps)          # diag(1, 1, 1, e^{iπk/steps})
        targets.append({"name": f"CPhase({k}π/{cphase_steps})", "group": "CPhase", "unitary": cirq.unitary(gate), "gate": gate})
    for j in range(n_haar):
        targets.append({"name": f"haar{j}", "group": "Haar", "unitary": haar_random_unitary(4, seed=seed + j), "gate": None})
    return targets


SYCAMORE_GATE = None


def _syc():
    global SYCAMORE_GATE
    if SYCAMORE_GATE is None:
        SYCAMORE_GATE = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)
    return SYCAMORE_GATE


def cirq_two_qubit(u: np.ndarray, gate_set: str, gate=None) -> Tuple[list, str]:
    """Cirq's closest built-in decomposition of ``u`` for ``gate_set``: (operations, routine name)."""
    q0, q1 = cirq.LineQubit.range(2)
    if gate_set == "cz":
        return list(cirq.two_qubit_matrix_to_cz_operations(q0, q1, u, allow_partial_czs=False)), "two_qubit_matrix_to_cz_operations"
    if gate_set == "sqrt_iswap":
        return list(cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u)), "two_qubit_matrix_to_sqrt_iswap_operations"
    if gate_set == "sycamore":
        return cirq_core_sycamore(u, (q0, q1))
    if gate_set == "free":
        cands = [(cirq_two_qubit(u, "cz")[0], "cz"),
                 (list(cirq.two_qubit_matrix_to_cz_operations(q0, q1, u, allow_partial_czs=True)), "cz_partial"),
                 (cirq_two_qubit(u, "sqrt_iswap")[0], "sqrt_iswap")]
        return min(cands, key=lambda c: _two_qubit_count(c[0]))          # ties go to the first
    raise ValueError(gate_set)


def cirq_core_sycamore(u: np.ndarray, qubits) -> Tuple[list, str]:
    """
    The cirq-core baseline for the Sycamore gate: no gate for a local
    target; two gates for a controlled-phase class (Weyl y = z = 0) through a
    KAK change of basis and ``decompose_cphase_into_two_fsim``; four gates
    through ``decompose_two_qubit_interaction_into_four_fsim_gates`` otherwise.
    """
    q0, q1 = qubits
    c = weyl.weyl_coordinates(u)
    if np.all(np.abs(c) <= 1e-9):
        g, f0, f1 = cirq.kron_factor_4x4_to_2x2s(u)
        return [cirq.MatrixGate(g * f0).on(q0), cirq.MatrixGate(f1).on(q1)], "local"
    if abs(c[1]) <= 1e-9 and abs(c[2]) <= 1e-9:
        cph = cirq.CZPowGate(exponent=4.0 * c[0] / np.pi)
        kt, kc = cirq.kak_decomposition(u), cirq.kak_decomposition(cirq.unitary(cph))
        before = [kc.single_qubit_operations_before[k].conj().T @ kt.single_qubit_operations_before[k] for k in (0, 1)]
        after = [kt.single_qubit_operations_after[k] @ kc.single_qubit_operations_after[k].conj().T for k in (0, 1)]
        ops = ([cirq.MatrixGate(before[0]).on(q0), cirq.MatrixGate(before[1]).on(q1)]
               + list(cirq.decompose_cphase_into_two_fsim(cph, fsim_gate=_syc(), qubits=[q0, q1]))
               + [cirq.MatrixGate(after[0]).on(q0), cirq.MatrixGate(after[1]).on(q1)])
        if process_infidelity(u, _ops_unitary(ops, [q0, q1])) < 1e-9:
            return ops, "decompose_cphase_into_two_fsim"
    circ = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(u, fsim_gate=_syc(), qubits=[q0, q1])
    return list(circ.all_operations()), "decompose_two_qubit_interaction_into_four_fsim_gates"


def _google_gateset():
    try:
        import cirq_google
    except ImportError:
        return None
    return cirq_google.SycamoreTargetGateset()


def cirq_google_two_qubit(u: np.ndarray, gateset, gate=None) -> list:
    """``SycamoreTargetGateset`` on the named gate when there is one (it has special cases), else a MatrixGate."""
    q0, q1 = cirq.LineQubit.range(2)
    op = gate.on(q0, q1) if gate is not None else cirq.MatrixGate(u).on(q0, q1)
    out = cirq.optimize_for_target_gateset(cirq.Circuit(op), gateset=gateset)
    return list(out.all_operations())


def _timed(fn: Callable, repeat: int = 3):
    best, out = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def reference_minimum(u: np.ndarray, gate_set: str) -> Tuple[Optional[int], str]:
    """Minimum count and how it is known: "closed form", "proven" (lower bound met), "numerical" or "not found"."""
    cal = calibration_for(gate_set)
    return weyl.minimum_count(u, None if cal is None else cal.angles)


def compare_two_qubit(targets: Sequence[Dict[str, Any]], gate_sets: Sequence[str] = GATE_SETS,
                      target_infidelity: float = 1e-9, n_restarts: int = 4, seed: int = 0,
                      unpruned: Optional[Callable[[Dict[str, Any]], bool]] = None,
                      include_cirq_google: bool = True, progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """
    One row per (target, gate set): this package's decomposition against Cirq's.

    ``unpruned(target)`` selects targets that are also synthesised with
    ``weyl_pruning=False`` (the 0.4 search from one stage) to measure the
    speed-up. Times are wall-clock seconds per decomposition after JIT
    warm-up, the best of three calls for every routine.
    """
    q = cirq.LineQubit.range(2)
    google = _google_gateset() if include_cirq_google else None
    rows: List[Dict[str, Any]] = []
    for gs in gate_sets:
        cal = calibration_for(gs)
        syn = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, seed=seed)
        syn_old = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, seed=seed, weyl_pruning=False)
        warm = haar_random_unitary(4, seed=12345)
        for s in (syn, syn_old):                                   # compile every stage count once
            for n in range(0, 5 if cal is not None else 3):
                s._optimize_stage(warm, n, 1, 5, fixed=None if cal is None else [cal.angles] * n)
        for t in targets:
            u = t["unitary"]
            res, dt = _timed(lambda: syn.decompose(u, n_restarts=n_restarts, calibration=cal))
            ours_u = _ops_unitary(convert_decomposition_to_cirq_ops(res, *q), q)
            (c_ops, routine), c_dt = _timed(lambda: cirq_two_qubit(u, gs, t["gate"]))
            ref, ref_kind = reference_minimum(u, gs)
            row = {
                "target": t["name"], "group": t["group"], "gate_set": gs,
                "ours_n": int(res.n_stages), "ours_infidelity": process_infidelity(u, ours_u), "ours_s": dt,
                "cirq_n": _two_qubit_count(c_ops), "cirq_infidelity": process_infidelity(u, _ops_unitary(c_ops, q)),
                "cirq_s": c_dt, "cirq_routine": routine,
                "min_n": ref, "min_kind": ref_kind,
            }
            if unpruned is not None and unpruned(t):
                old, old_dt = _timed(lambda: syn_old.decompose(u, n_restarts=n_restarts, calibration=cal))
                row.update({"unpruned_n": int(old.n_stages), "unpruned_s": old_dt,
                            "unpruned_infidelity": float(old.infidelity)})
            if gs == "sycamore" and google is not None:
                g_ops, g_dt = _timed(lambda: cirq_google_two_qubit(u, google, t["gate"]))
                row.update({"google_n": _two_qubit_count(g_ops), "google_s": g_dt,
                            "google_infidelity": process_infidelity(u, _ops_unitary(g_ops, q))})
            rows.append(row)
        if progress:
            progress(f"two-qubit targets done for {gs}")
    return rows


# --------------------------------------------------------------------------- #
# Noise-aware approximation
# --------------------------------------------------------------------------- #
def approximation_profile(u: np.ndarray, gate_set: str, synthesizer: Optional[DifferentiableFSimSynthesizer] = None,
                          n_restarts: int = 16, max_iter: int = 500) -> List[float]:
    """
    Best process infidelity found with n = 0, 1, …, n_exact native gates.

    The last entry is the exact decomposition. Entries for n < n_exact come
    from :meth:`DifferentiableFSimSynthesizer.best_fit`: closed forms for
    zero gates, one fixed native gate and one free FSim, otherwise
    ``n_restarts`` × ``max_iter`` multi-start search. Numerical entries are
    upper bounds on the true best infidelity, so fractions computed from
    them are lower bounds on how often fewer gates win.
    """
    cal = calibration_for(gate_set)
    syn = synthesizer or DifferentiableFSimSynthesizer(target_infidelity=1e-9)
    exact = syn.decompose(u, calibration=cal)
    prof = [float(syn.best_fit(u, n, cal, n_restarts, max_iter).infidelity) for n in range(exact.n_stages)]
    prof.append(float(exact.infidelity))
    return prof


def tabulation_study(haar: Sequence[np.ndarray], eps: float, profiles: Optional[Sequence[Sequence[float]]] = None,
                     max_infidelity: float = 1e-2, seed: int = 2026) -> Dict[str, Any]:
    """
    Cirq's approximate route to the Sycamore gate,
    ``two_qubit_gate_product_tabulation(SYC, max_infidelity)``, on the given
    targets: gate counts, process infidelities and the expected infidelity
    (1 − F) + n·ε, next to noise-aware synthesis when ``profiles`` are given.
    """
    t0 = time.perf_counter()
    tab = cirq.two_qubit_gate_product_tabulation(cirq.unitary(_syc()), max_infidelity, random_state=seed)
    build_s = time.perf_counter() - t0
    ns, infs, secs = [], [], []
    for u in haar:
        r, dt = _timed(lambda: tab.compile_two_qubit_gate(u))
        ns.append(len(r.local_unitaries) - 1)
        infs.append(process_infidelity(u, r.actual_gate))
        secs.append(dt)
    costs = [f + n * eps for f, n in zip(infs, ns)]
    out = {"max_infidelity_setting": max_infidelity, "build_s": build_s, "eps": eps, "n_targets": len(ns),
           "mean_n": float(np.mean(ns)), "n_histogram": {int(k): int(ns.count(k)) for k in sorted(set(ns))},
           "mean_infidelity": float(np.mean(infs)), "max_infidelity": float(np.max(infs)),
           "mean_cost": float(np.mean(costs)), "median_ms": 1e3 * float(np.median(secs))}
    if profiles is not None:
        picks = [noise_aware_choice(p, eps) for p in profiles]
        out["noise_aware_mean_cost"] = float(np.mean([c for _, c, _ in picks]))
        out["noise_aware_mean_n"] = float(np.mean([n for n, _, _ in picks]))
        out["exact_mean_cost"] = float(np.mean([ce for _, _, ce in picks]))
        out["tabulation_lower_cost"] = int(sum(c < pc - 1e-12 for c, (_, pc, _) in zip(costs, picks)))
    return out


def noise_aware_choice(profile: Sequence[float], eps: float) -> Tuple[int, float, float]:
    """(chosen n, its cost, cost of the exact decomposition) with cost(n) = infidelity(n) + n·ε."""
    costs = [inf + n * eps for n, inf in enumerate(profile)]
    n = int(np.argmin(costs))
    return n, float(costs[n]), float(costs[-1])


def noise_aware_sweep(profiles: Sequence[Sequence[float]], eps_grid: Sequence[float]) -> Dict[str, List[float]]:
    """For each ε: fraction of targets where fewer gates win, and the mean saving in expected infidelity."""
    frac, saved, saved_when = [], [], []
    for eps in eps_grid:
        picks = [noise_aware_choice(p, eps) for p in profiles]
        fewer = [n < len(p) - 1 for (n, _, _), p in zip(picks, profiles)]
        gains = [ce - c for (_, c, ce) in picks]
        frac.append(float(np.mean(fewer)))
        saved.append(float(np.mean(gains)))
        saved_when.append(float(np.mean([g for g, f in zip(gains, fewer) if f])) if any(fewer) else 0.0)
    return {"eps": [float(e) for e in eps_grid], "fraction_fewer": frac, "mean_saved": saved, "mean_saved_when_fewer": saved_when}


# --------------------------------------------------------------------------- #
# Circuits
# --------------------------------------------------------------------------- #
def qft_circuit(qubits: Sequence["cirq.Qid"]) -> "cirq.Circuit":
    """Textbook QFT (first qubit most significant) with the final qubit reversal."""
    n = len(qubits)
    ops = []
    for i in range(n):
        ops.append(cirq.H(qubits[i]))
        for j in range(i + 1, n):
            ops.append(cirq.CZPowGate(exponent=1.0 / 2 ** (j - i)).on(qubits[j], qubits[i]))
    for i in range(n // 2):
        ops.append(cirq.SWAP(qubits[i], qubits[n - 1 - i]))
    return cirq.Circuit(ops)


def cuccaro_adder_circuit(c, b, a, z) -> "cirq.Circuit":
    """
    One-bit ripple-carry adder of Cuccaro et al. (quant-ph/0410184): MAJ, carry
    out to z, UMA. Maps (c, b, a, z) to (c, a ⊕ b ⊕ c, a, z ⊕ carry).
    """
    def maj(x, y, w):
        return [cirq.CNOT(w, y), cirq.CNOT(w, x), cirq.CCNOT(x, y, w)]

    def uma(x, y, w):
        return [cirq.CCNOT(x, y, w), cirq.CNOT(w, x), cirq.CNOT(x, y)]

    return cirq.Circuit(maj(c, b, a) + [cirq.CNOT(a, z)] + uma(c, b, a))


def ghz_circuit(qubits: Sequence["cirq.Qid"]) -> "cirq.Circuit":
    return cirq.Circuit([cirq.H(qubits[0])] + [cirq.CNOT(qubits[k], qubits[k + 1]) for k in range(len(qubits) - 1)])


def random_circuit(n_qubits: int, depth: int, seed: int) -> "cirq.Circuit":
    """Layers of random single-qubit rotations and random two-qubit gates on random pairs."""
    rng = np.random.default_rng(seed)
    qs = cirq.LineQubit.range(n_qubits)
    ops = []
    for _ in range(depth):
        for qb in qs:
            ops.append(cirq.PhasedXZGate(x_exponent=rng.uniform(0, 1), z_exponent=rng.uniform(-1, 1),
                                         axis_phase_exponent=rng.uniform(-1, 1)).on(qb))
        perm = rng.permutation(n_qubits)
        for k in range(0, n_qubits - 1, 2):
            a, b = qs[perm[k]], qs[perm[k + 1]]
            choice = rng.integers(5)
            gate = [cirq.CNOT, cirq.CZ, cirq.ISWAP, cirq.SQRT_ISWAP, cirq.CZPowGate(exponent=rng.uniform(-1, 1))][choice]
            ops.append(gate.on(a, b))
    return cirq.Circuit(ops)


def circuit_suite(seed: int = 7) -> Dict[str, "cirq.Circuit"]:
    from .toffoli_decomposer import toffoli_circuit, fredkin_circuit, ccz_circuit

    q = cirq.LineQubit.range(4)
    return {
        "Toffoli": toffoli_circuit(*q[:3]),
        "Fredkin": fredkin_circuit(*q[:3]),
        "CCZ": ccz_circuit(*q[:3]),
        "QFT3": qft_circuit(q[:3]),
        "QFT4": qft_circuit(q),
        "Adder (4q)": cuccaro_adder_circuit(*q),
        "GHZ (4q)": ghz_circuit(q),
        "random 3q #1": random_circuit(3, 6, seed),
        "random 3q #2": random_circuit(3, 6, seed + 1),
        "random 4q #1": random_circuit(4, 6, seed + 2),
        "random 4q #2": random_circuit(4, 6, seed + 3),
    }


def _sycamore_cirq_core_compile(circuit: "cirq.Circuit") -> "cirq.Circuit":
    """The cirq-core Sycamore rule (:func:`cirq_core_sycamore`) on every merged two-qubit block."""
    expanded = cirq.Circuit(cirq.decompose(circuit, keep=lambda op: len(op.qubits) <= 2))
    merged = cirq.merge_k_qubit_unitaries(expanded, k=2, rewriter=lambda op: op)
    out = []
    for op in merged.all_operations():
        u = cirq.unitary(op)
        if len(op.qubits) == 2:
            out += cirq_core_sycamore(u, op.qubits)[0]
        else:
            out.append(cirq.MatrixGate(u).on(*op.qubits))
    return cirq.Circuit(out)


def cirq_compile(circuit: "cirq.Circuit", gate_set: str) -> "cirq.Circuit":
    if gate_set == "cz":
        return cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    if gate_set == "sqrt_iswap":
        return cirq.optimize_for_target_gateset(circuit, gateset=cirq.SqrtIswapTargetGateset())
    if gate_set == "sycamore":
        return _sycamore_cirq_core_compile(circuit)
    if gate_set == "sycamore_google":
        gs = _google_gateset()
        if gs is None:
            raise ImportError("cirq_google is not installed")
        return cirq.optimize_for_target_gateset(circuit, gateset=gs)
    raise ValueError(gate_set)


def compare_circuits(suite: Dict[str, "cirq.Circuit"], gate_sets: Sequence[str] = ("cz", "sqrt_iswap", "sycamore"),
                     target_infidelity: float = 1e-9, include_cirq_google: bool = True,
                     progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """
    Two-qubit counts and estimated fidelity for each circuit and gate set:
    Cirq, this package exact, and this package noise-aware at the gate set's ε.

    Every route is timed once per circuit, after one warm-up compilation of
    a small circuit with each route. Exact-mode synthesis failures (blocks
    left above the target infidelity) are counted from SynthesisWarning.
    """
    google = include_cirq_google and _google_gateset() is not None
    rows = []
    warm = ghz_circuit(cirq.LineQubit.range(3)) + cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2)))
    for gs in gate_sets:
        cirq_compile(warm, gs)
        if gs == "sycamore" and google:
            cirq_compile(warm, "sycamore_google")
        for mode in ("exact", "noise_aware"):
            FSimDecomposerTransformer(target_infidelity=target_infidelity, calibration=calibration_for(gs),
                                      approximation=mode, gate_error=GATE_ERRORS[gs]).optimize_circuit(warm)
    for name, circuit in suite.items():
        qs = sorted(circuit.all_qubits())
        ideal = _ops_unitary(circuit, qs)
        for gs in gate_sets:
            eps = GATE_ERRORS[gs]
            cal = calibration_for(gs)
            row: Dict[str, Any] = {"circuit": name, "gate_set": gs, "eps": eps, "n_qubits": len(qs)}
            out, dt = _timed(lambda: cirq_compile(circuit, gs), repeat=1)
            f = 1.0 - process_infidelity(ideal, _ops_unitary(out, qs))
            n2 = _two_qubit_count(out)
            row.update({"cirq_n": n2, "cirq_F": f, "cirq_est": f * (1 - eps) ** n2, "cirq_s": dt})
            if gs == "sycamore" and google:
                out, dt = _timed(lambda: cirq_compile(circuit, "sycamore_google"), repeat=1)
                f = 1.0 - process_infidelity(ideal, _ops_unitary(out, qs))
                n2 = _two_qubit_count(out)
                row.update({"google_n": n2, "google_F": f, "google_est": f * (1 - eps) ** n2, "google_s": dt})
            for mode in ("exact", "noise_aware"):
                tr = FSimDecomposerTransformer(target_infidelity=target_infidelity, calibration=cal,
                                               approximation=mode, gate_error=eps)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", SynthesisWarning)
                    out, dt = _timed(lambda: tr.optimize_circuit(circuit), repeat=1)
                f = 1.0 - process_infidelity(ideal, _ops_unitary(out, qs))
                n2 = _two_qubit_count(out)
                key = "ours" if mode == "exact" else "na"
                row.update({f"{key}_n": n2, f"{key}_F": f, f"{key}_est": f * (1 - eps) ** n2, f"{key}_s": dt,
                            f"{key}_blocks": len(tr.block_infidelities), f"{key}_bound": tr.infidelity_bound(),
                            f"{key}_failures": sum(issubclass(w.category, SynthesisWarning) for w in caught)})
                if mode == "noise_aware":
                    row["na_approximated_blocks"] = tr.stats["approximated_blocks"]
            rows.append(row)
        if progress:
            progress(f"circuit {name} done")
    return rows


# --------------------------------------------------------------------------- #
# Summaries and tables
# --------------------------------------------------------------------------- #
def summarize_two_qubit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for gs in sorted({r["gate_set"] for r in rows}, key=GATE_SETS.index):
        rs = [r for r in rows if r["gate_set"] == gs]
        haar = [r for r in rs if r["group"] == "Haar"]
        un = [r for r in rs if "unpruned_s" in r]
        s = {
            "n_targets": len(rs),
            "haar_ours_mean_n": float(np.mean([r["ours_n"] for r in haar])) if haar else None,
            "haar_cirq_mean_n": float(np.mean([r["cirq_n"] for r in haar])) if haar else None,
            "ours_median_infidelity": float(np.median([r["ours_infidelity"] for r in rs])),
            "ours_max_infidelity": float(np.max([r["ours_infidelity"] for r in rs])),
            "cirq_max_infidelity": float(np.max([r["cirq_infidelity"] for r in rs])),
            "ours_median_ms": 1e3 * float(np.median([r["ours_s"] for r in rs])),
            "cirq_median_ms": 1e3 * float(np.median([r["cirq_s"] for r in rs])),
            "at_minimum": float(np.mean([r["ours_n"] == r["min_n"] for r in rs if r["min_n"] is not None])),
            "minimum_kind_counts": {k: int(sum(r["min_kind"] == k for r in rs))
                                    for k in ("closed form", "proven", "numerical", "not found")
                                    if any(r["min_kind"] == k for r in rs)},
            "ours_above_minimum": int(sum(r["min_n"] is not None and r["ours_n"] > r["min_n"] for r in rs)),
            "ours_below_minimum": int(sum(r["min_n"] is not None and r["ours_n"] < r["min_n"] for r in rs)),
        }
        if un:
            s.update({
                "unpruned_targets": len(un),
                "unpruned_median_ms": 1e3 * float(np.median([r["unpruned_s"] for r in un])),
                "pruned_median_ms_same_targets": 1e3 * float(np.median([r["ours_s"] for r in un])),
                "speedup_total_time": float(np.sum([r["unpruned_s"] for r in un]) / np.sum([r["ours_s"] for r in un])),
                "unpruned_at_minimum": float(np.mean([r["unpruned_n"] == r["min_n"] for r in un if r["min_n"] is not None])),
            })
        if any("google_n" in r for r in rs):
            s["haar_google_mean_n"] = float(np.mean([r["google_n"] for r in haar])) if haar else None
            s["google_median_ms"] = 1e3 * float(np.median([r["google_s"] for r in rs]))
            s["google_max_infidelity"] = float(np.max([r["google_infidelity"] for r in rs]))
        out[gs] = s
    return out


def group_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean two-qubit counts per target group and gate set."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in rows:
        g = out.setdefault(r["group"], {}).setdefault(r["gate_set"], {"ours": [], "cirq": [], "min": [], "google": []})
        g["ours"].append(r["ours_n"]); g["cirq"].append(r["cirq_n"])
        if r["min_n"] is not None:
            g["min"].append(r["min_n"])
        if "google_n" in r:
            g["google"].append(r["google_n"])
    return {grp: {gs: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} for gs, d in by.items()}
            for grp, by in out.items()}


def _fmt_n(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def _cphase_label(rows) -> str:
    ks = sorted({r["target"] for r in rows if r["group"] == "CPhase"})
    steps = len(ks) + 1
    return f"CPhase(kπ/{steps}), k = 1..{steps - 1}"


def _kinds(s: Dict[str, Any]) -> str:
    return ", ".join(f"{k} {v}" for k, v in s["minimum_kind_counts"].items())


def markdown_tables(two_qubit_rows, circuit_rows=None, noise=None, check=None, tabulation=None) -> str:
    """The README tables, as markdown."""
    lines = []
    gc = group_counts(two_qubit_rows)
    gsets = [g for g in GATE_SETS if any(r["gate_set"] == g for r in two_qubit_rows)]
    has_google = any("google_n" in r for r in two_qubit_rows)
    syc_google = lambda g: g == "sycamore" and has_google

    def head2(g):
        return "Sycamore: ours / cirq_google / Cirq core" if syc_google(g) else f"{LABELS[g]}: ours / Cirq"

    lines += ["Two-qubit gate counts (mean over the group; minimum in brackets)", "",
              "| target | " + " | ".join(head2(g) for g in gsets) + " |", "|" + "---|" * (len(gsets) + 1)]
    for grp in ["CNOT", "CZ", "iSWAP", "√iSWAP", "SWAP", "CPhase", "Haar"]:
        if grp not in gc:
            continue
        cells = []
        for g in gsets:
            d = gc[grp].get(g)
            if d is None:
                cells.append("–"); continue
            keys = ["ours", "google", "cirq"] if syc_google(g) else ["ours", "cirq"]
            cells.append(" / ".join(_fmt_n(d[k]) for k in keys) + f" [{_fmt_n(d['min'])}]")
        label = {"CPhase": _cphase_label(two_qubit_rows), "Haar": "Haar-random U(4)"}.get(grp, grp)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    summ = summarize_two_qubit(two_qubit_rows)
    lines += ["", "Accuracy, speed and minimality over all targets (times: best of three calls)", "",
              "| gate set | at minimum (without pruning) | how the minimum is known | process infidelity, ours (median / max) | Cirq (max) | ms per decomposition, ours / Cirq | without Weyl pruning (ms) | speed-up |",
              "|---|---|---|---|---|---|---|---|"]
    for g in gsets:
        s = summ[g]
        un = f"{s['unpruned_median_ms']:.0f}" if "unpruned_median_ms" in s else "–"
        sp = f"{s['speedup_total_time']:.1f}×" if "speedup_total_time" in s else "–"
        um = f" ({100 * s['unpruned_at_minimum']:.1f}%)" if "unpruned_at_minimum" in s else ""
        cmax = f"{s['cirq_max_infidelity']:.0e}"
        cms = f"{s['cirq_median_ms']:.1f}"
        if syc_google(g):
            cmax = f"{s['google_max_infidelity']:.0e} (cirq_google), {cmax} (core)"
            cms = f"{s['google_median_ms']:.1f} (cirq_google), {cms} (core)"
        lines.append(f"| {LABELS[g]} | {100 * s['at_minimum']:.1f}%{um} | {_kinds(s)} | {s['ours_median_infidelity']:.0e} / "
                     f"{s['ours_max_infidelity']:.0e} | {cmax} | {s['ours_median_ms']:.0f} / {cms} | {un} | {sp} |")
    if circuit_rows:
        cgs = [g for g in ("cz", "sqrt_iswap", "sycamore") if any(r["gate_set"] == g for r in circuit_rows)]
        g_google = any("google_n" in r for r in circuit_rows)
        names = list(dict.fromkeys(r["circuit"] for r in circuit_rows))

        def keys(g):
            return (["google", "cirq"] if (g == "sycamore" and g_google) else ["cirq"]) + ["ours", "na"]

        def hdr(g):
            if g == "sycamore" and g_google:
                return "Sycamore: cirq_google / Cirq core / ours / noise-aware"
            return f"{LABELS[g]}: Cirq / ours / noise-aware"

        csum = summarize_circuits(circuit_rows)
        for title, fmt, total in (
                ("Whole circuits: two-qubit gate count", lambda r, k: f"{r[k + '_n']}",
                 lambda g, k: f"{csum[g][k + '_total_2q']}"),
                ("Whole circuits: estimated fidelity F · (1 − ε)^n", lambda r, k: f"{r[k + '_est']:.3f}",
                 lambda g, k: f"{csum[g]['mean_est_' + k]:.3f}")):
            lines += ["", title, "", "| circuit | " + " | ".join(hdr(g) for g in cgs) + " |", "|" + "---|" * (len(cgs) + 1)]
            for name in names:
                cells = []
                for g in cgs:
                    r = next(x for x in circuit_rows if x["circuit"] == name and x["gate_set"] == g)
                    cells.append(" / ".join(fmt(r, k) for k in keys(g)))
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            label = "total" if "count" in title else "mean"
            lines.append(f"| {label} | " + " | ".join(" / ".join(total(g, k) for k in keys(g)) for g in cgs) + " |")
    if noise:
        lines += ["", "Noise-aware synthesis on Haar-random U(4): fraction of targets where the best fit found "
                  "with fewer gates has lower (1 − F) + n·ε than the exact decomposition", "",
                  "| gate set | ε (process infidelity) | fewer gates better | mean saving (all) | mean saving (when fewer) | rerun check: same n / lower cost found |",
                  "|---|---|---|---|---|---|"]
        for g, d in noise.items():
            c = (check or {}).get(g)
            cc = f"{c['same_n']}/{c['checked']} / {c['check_lower_cost']}" if c else "–"
            lines.append(f"| {LABELS[g]} | {100 * d['eps']:.2f}% | {100 * d['fraction_fewer']:.1f}% | "
                         f"{d['mean_saved']:.2e} | {d['mean_saved_when_fewer']:.2e} | {cc} |")
    if tabulation:
        t = tabulation
        lines += ["", f"Sycamore, approximate routes at ε = {100 * t['eps']:.2f}% (mean over {t['n_targets']} Haar-random targets)", "",
                  "| route | mean gates | mean synthesis infidelity | mean (1 − F) + n·ε |", "|---|---|---|---|",
                  f"| Cirq tabulation (max_infidelity {t['max_infidelity_setting']:g}) | {t['mean_n']:.2f} | {t['mean_infidelity']:.2e} | {t['mean_cost']:.2e} |"]
        if "noise_aware_mean_cost" in t:
            lines += [f"| noise-aware synthesis | {t['noise_aware_mean_n']:.2f} | – | {t['noise_aware_mean_cost']:.2e} |",
                      f"| exact synthesis | – | – | {t['exact_mean_cost']:.2e} |"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Everything at once (benchmarks/compare_with_cirq.py and ``cirq-fsim compare``)
# --------------------------------------------------------------------------- #
def run_comparison(n_haar: int = 200, seed: int = 2026, circuits: bool = True, quick: bool = False,
                   eps_grid: Optional[Sequence[float]] = None,
                   progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Run the two-qubit comparison, the noise-aware study on the Haar-random
    targets and (optionally) the circuit comparison. Returns a JSON-ready dict.

    ``quick`` keeps it under a minute or two: 10 Haar targets, four CPhase
    angles and three circuits.
    """
    import platform
    from . import __version__

    if quick:
        n_haar = min(n_haar, 10)
    say = progress or (lambda msg: None)
    targets = two_qubit_targets(n_haar=n_haar, seed=seed, cphase_steps=4 if quick else 8)
    t0 = time.perf_counter()
    rows = compare_two_qubit(targets, unpruned=lambda t: True, progress=say)
    say(f"two-qubit comparison: {time.perf_counter() - t0:.0f} s")

    eps_grid = list(eps_grid) if eps_grid is not None else [float(e) for e in np.geomspace(1e-4, 3e-2, 31)]
    haar = [t for t in targets if t["group"] == "Haar"]
    noise: Dict[str, Any] = {"eps_grid": eps_grid, "profiles": {}, "sweep": {}, "at_cited": {}, "check": {}}
    t0 = time.perf_counter()
    for gs in GATE_SETS:
        syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
        profiles = [approximation_profile(t["unitary"], gs, syn, n_restarts=16, max_iter=500) for t in haar]
        noise["profiles"][gs] = profiles
        noise["sweep"][gs] = noise_aware_sweep(profiles, eps_grid)
        eps = GATE_ERRORS[gs]
        at = noise_aware_sweep(profiles, [eps])
        exact_cost = float(np.mean([noise_aware_choice(p, eps)[2] for p in profiles]))
        noise["at_cited"][gs] = {"eps": eps, "fraction_fewer": at["fraction_fewer"][0], "mean_saved": at["mean_saved"][0],
                                 "mean_saved_when_fewer": at["mean_saved_when_fewer"][0], "mean_exact_cost": exact_cost}
        # rerun check on a subset: the synthesiser's own noise-aware mode with another
        # seed and twice the restarts (32 per stage count); a lower cost there means the
        # profile was under-optimised
        check = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=1)
        n_check = min(len(haar), 40)
        agree = lower = higher = 0
        for t, p in zip(haar[:n_check], profiles[:n_check]):
            r = check.decompose(t["unitary"], calibration=calibration_for(gs), approximation="noise_aware",
                                gate_error=eps, n_restarts=8)
            n_p, c_p, _ = noise_aware_choice(p, eps)
            agree += int(r.n_stages == n_p)
            lower += int(r.expected_infidelity < c_p - 1e-9)
            higher += int(r.expected_infidelity > c_p + 1e-9)
        noise["check"][gs] = {"checked": n_check, "same_n": agree, "check_lower_cost": lower,
                              "check_higher_cost": higher, "seed": 1, "restarts_per_stage": 32}
        say(f"noise-aware profiles done for {gs}")
    noise["tabulation"] = tabulation_study([t["unitary"] for t in haar], GATE_ERRORS["sycamore"],
                                           noise["profiles"]["sycamore"], seed=seed)
    say(f"noise-aware study: {time.perf_counter() - t0:.0f} s")

    out: Dict[str, Any] = {
        "meta": {"version": __version__, "cirq": cirq.__version__, "python": platform.python_version(),
                 "machine": platform.machine(), "system": platform.system(), "n_haar": n_haar, "seed": seed,
                 "gate_errors": GATE_ERRORS, "quick": quick},
        "two_qubit": rows,
        "two_qubit_summary": summarize_two_qubit(rows),
        "group_counts": group_counts(rows),
        "haar_weyl": [weyl.weyl_coordinates(t["unitary"]).tolist() for t in haar],
        "noise": noise,
    }
    if circuits:
        suite = circuit_suite()
        if quick:
            suite = {k: suite[k] for k in ("Toffoli", "QFT3", "GHZ (4q)")}
        t0 = time.perf_counter()
        crow = compare_circuits(suite, progress=say)
        out["circuits"] = crow
        out["circuit_summary"] = summarize_circuits(crow)
        say(f"circuit comparison: {time.perf_counter() - t0:.0f} s")
    out["markdown"] = markdown_tables(rows, out.get("circuits"), noise["at_cited"], noise["check"], noise["tabulation"])
    return out


def summarize_circuits(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for gs in dict.fromkeys(r["gate_set"] for r in rows):
        rs = [r for r in rows if r["gate_set"] == gs]
        s = {
            "circuits": len(rs),
            "cirq_total_2q": int(sum(r["cirq_n"] for r in rs)),
            "ours_total_2q": int(sum(r["ours_n"] for r in rs)),
            "na_total_2q": int(sum(r["na_n"] for r in rs)),
            "ours_fewer_than_cirq": int(sum(r["ours_n"] < r["cirq_n"] for r in rs)),
            "ours_more_than_cirq": int(sum(r["ours_n"] > r["cirq_n"] for r in rs)),
            "blocks": int(sum(r["na_blocks"] for r in rs)),
            "approximated_blocks": int(sum(r["na_approximated_blocks"] for r in rs)),
            "circuits_with_approximation": int(sum(r["na_approximated_blocks"] > 0 for r in rs)),
            "mean_est_ours": float(np.mean([r["ours_est"] for r in rs])),
            "mean_est_na": float(np.mean([r["na_est"] for r in rs])),
            "mean_est_cirq": float(np.mean([r["cirq_est"] for r in rs])),
            "na_bound_holds": bool(all(1 - r["na_F"] <= r["na_bound"] + 1e-9 for r in rs)),
            "ours_median_s": float(np.median([r["ours_s"] for r in rs])),
            "cirq_median_s": float(np.median([r["cirq_s"] for r in rs])),
        }
        s["ours_min_F"] = float(min(r["ours_F"] for r in rs))
        s["ours_failures"] = int(sum(r["ours_failures"] for r in rs))
        if any("google_n" in r for r in rs):
            s["google_total_2q"] = int(sum(r["google_n"] for r in rs))
            s["mean_est_google"] = float(np.mean([r["google_est"] for r in rs]))
            s["google_min_F"] = float(min(r["google_F"] for r in rs))
            s["ours_fewer_than_google"] = int(sum(r["ours_n"] < r["google_n"] for r in rs))
            s["ours_more_than_google"] = int(sum(r["ours_n"] > r["google_n"] for r in rs))
        out[gs] = s
    return out


# --------------------------------------------------------------------------- #
# Compact results (what the README and benchmarks/make_figures.py need)
# --------------------------------------------------------------------------- #
def _sig(x: Any, digits: int = 4) -> Any:
    """Round floats to ``digits`` significant figures, recursively."""
    if isinstance(x, dict):
        return {k: _sig(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sig(v, digits) for v in x]
    if isinstance(x, (bool, np.bool_)) or x is None or isinstance(x, str):
        return bool(x) if isinstance(x, np.bool_) else x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return 0.0 if x == 0.0 else float(f"{x:.{digits}g}")
    return x


CIRCUIT_KEYS = ("circuit", "gate_set", "eps", "n_qubits", "cirq_n", "cirq_F", "cirq_est", "cirq_s",
                "google_n", "google_F", "google_est", "google_s", "ours_n", "ours_F", "ours_est", "ours_s",
                "ours_failures", "na_n", "na_F", "na_est", "na_s", "na_bound", "na_blocks", "na_approximated_blocks")
NAMED_KEYS = ("target", "gate_set", "ours_n", "ours_infidelity", "cirq_n", "cirq_routine", "google_n", "min_n", "min_kind")


def compact_results(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    The part of :func:`run_comparison`'s output that the README tables and
    the figures use, rounded to four significant figures: summaries, group
    counts, rows for the named two-qubit targets and the circuits, timing
    arrays, the noise-aware sweep, check and tabulation study, the
    approximation profiles and the Weyl coordinates of the Haar targets.
    """
    two = res["two_qubit"]
    timing = {}
    for gs in GATE_SETS:
        rs = [r for r in two if r["gate_set"] == gs]
        if not rs:
            continue
        t = {"ours_ms": [1e3 * r["ours_s"] for r in rs], "cirq_ms": [1e3 * r["cirq_s"] for r in rs]}
        if any("unpruned_s" in r for r in rs):
            t["unpruned_ms"] = [1e3 * r["unpruned_s"] for r in rs if "unpruned_s" in r]
        if any("google_s" in r for r in rs):
            t["google_ms"] = [1e3 * r["google_s"] for r in rs if "google_s" in r]
        timing[gs] = t
    noise = res["noise"]
    out = {
        "meta": res["meta"],
        "two_qubit_summary": res["two_qubit_summary"],
        "group_counts": res["group_counts"],
        "two_qubit_named": [{k: r[k] for k in NAMED_KEYS if k in r} for r in two if r["group"] not in ("Haar",)],
        "timing": _sig(timing, 3),
        "noise": {k: noise[k] for k in ("eps_grid", "sweep", "at_cited", "check", "tabulation") if k in noise},
        "profiles": _sig(noise["profiles"], 4),
        "haar_weyl": _sig(res["haar_weyl"], 5),
    }
    if "circuits" in res:
        out["circuits"] = [{k: r[k] for k in CIRCUIT_KEYS if k in r} for r in res["circuits"]]
        out["circuit_summary"] = res["circuit_summary"]
    out["markdown"] = res["markdown"]
    return _sig(out, 6)


def dump_json(obj: Any, path: str) -> None:
    """JSON with nested structure indented and lists of numbers kept on one line."""
    import json
    import re

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    text = json.dumps(obj, indent=1, default=default, ensure_ascii=False)
    num = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    flat = re.compile(r"\[\s*(" + num + r"(?:,\s*" + num + r")*)\s*\]")
    text = flat.sub(lambda m: "[" + ", ".join(re.split(r",\s*", m.group(1).strip())) + "]", text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
