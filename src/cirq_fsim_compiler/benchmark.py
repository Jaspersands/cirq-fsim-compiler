"""
Benchmark: free-angle vs calibrated-native synthesis of standard gates,
Euclidean vs Riemannian solvers, random U(4), three-qubit gates, batch stats.
Exposed as ``cirq-fsim benchmark``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import numpy as np

from .riemannian_optimizer import DifferentiableFSimSynthesizer
from .calibration_map import CouplerCalibration, SycamoreCalibrationMap
from .unitary_ansatz import haar_random_unitary
from .batch_compiler import BatchFSimCompiler


def run_compiler_benchmark(as_json: bool = False, quick: bool = False) -> int:
    import cirq
    out: Dict[str, Any] = {}
    log = [] if as_json else None

    def say(msg: str = "") -> None:
        (log.append(msg) if log is not None else print(msg))

    say("=" * 74)
    say("NATIVE FSIM COMPILER BENCHMARK (v0.3): free angles vs calibrated native gate")
    say("=" * 74)
    gates = {"CNOT": cirq.unitary(cirq.CNOT), "CZ": cirq.unitary(cirq.CZ), "iSWAP": cirq.unitary(cirq.ISWAP),
             "√iSWAP": cirq.unitary(cirq.SQRT_ISWAP), "SWAP": cirq.unitary(cirq.SWAP), "random U(4)": haar_random_unitary(4, seed=7)}
    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-8, seed=1)
    cal = CouplerCalibration.sycamore()
    say("\n1. Stage counts and infidelity")
    say("   gate         | free FSim(θ,φ): stages  1−F      ms | Sycamore FSim(π/2,π/6): stages  1−F      ms")
    rows = {}
    for name, u in gates.items():
        t0 = time.time(); r_free = synth.decompose(u, n_restarts=4); t_free = (time.time() - t0) * 1e3
        t0 = time.time(); r_nat = synth.decompose(u, n_restarts=6, calibration=cal); t_nat = (time.time() - t0) * 1e3
        say(f"   {name:<12s} |                {r_free.n_stages}     {r_free.infidelity:8.1e} {t_free:6.0f} |                       {r_nat.n_stages}     {r_nat.infidelity:8.1e} {t_nat:6.0f}")
        rows[name] = {"free": (r_free.n_stages, r_free.infidelity), "native": (r_nat.n_stages, r_nat.infidelity)}
    out["gates"] = rows

    say("\n2. Euclidean (L-BFGS on Euler angles) vs Riemannian (Cayley retraction) on CNOT / random U(4)")
    for name in ("CNOT", "random U(4)"):
        u = gates[name]
        t0 = time.time(); e = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=1).decompose(u, n_restarts=4, max_iter=400); te = time.time() - t0
        t0 = time.time(); r = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=1, method="riemannian").decompose(u, n_restarts=4, max_iter=600); tr = time.time() - t0
        say(f"   {name:<12s}: euclidean 1−F = {e.infidelity:.1e} ({te:.2f} s, {len(e.loss_history)} it) | riemannian 1−F = {r.infidelity:.1e} ({tr:.2f} s, {len(r.loss_history)} it)")
        out[f"solver_{name}"] = {"euclidean": e.infidelity, "riemannian": r.infidelity}

    say("\n3. Three-qubit gates (free angles)")
    from . import toffoli_decomposer as td
    q = cirq.LineQubit.range(3)
    for gname, fn, ref in (("Toffoli", td.decompose_toffoli_to_sycamore, cirq.unitary(cirq.CCNOT(*q))),
                           ("Fredkin", td.decompose_fredkin_to_sycamore, cirq.unitary(cirq.CSWAP(*q))),
                           ("CCZ", td.decompose_ccz_to_sycamore, cirq.unitary(cirq.CCZ(*q))),
                           ("QFT3", td.decompose_qft3_to_sycamore, td.qft_unitary(3))):
        t0 = time.time(); c = fn(*q, target_infidelity=1e-8); dt = time.time() - t0
        f = abs(np.trace(ref.conj().T @ cirq.unitary(c))) ** 2 / 64
        n_fsim = sum(1 for op in c.all_operations() if isinstance(op.gate, cirq.FSimGate))
        say(f"   {gname:<8s}: fidelity {f:.8f}  FSim gates {n_fsim:2d}  moments {len(c):3d}  ({dt:.2f} s)")
        out[gname] = {"fidelity": f, "fsim": n_fsim}

    say("\n4. Calibration drift: recompile CZ on a drifted coupler")
    cm = SycamoreCalibrationMap(seed=3)
    drifted = cm.get_coupler(0, 1)
    r0 = synth.decompose(gates["CZ"], n_restarts=6, calibration=cal)
    r1 = synth.decompose(gates["CZ"], n_restarts=6, calibration=drifted)
    say(f"   nominal FSim({cal.theta_cal:.4f}, {cal.phi_cal:.4f}): 1−F = {r0.infidelity:.1e} | drifted FSim({drifted.theta_cal:.4f}, {drifted.phi_cal:.4f}): 1−F = {r1.infidelity:.1e} (stages {r1.n_stages})")
    out["drift"] = {"nominal": r0.infidelity, "drifted": r1.infidelity}

    say("\n5. Batch of 12 random U(4) with the native gate (2 threads)")
    batch = [haar_random_unitary(4, seed=100 + k) for k in range(6 if quick else 12)]
    t0 = time.time()
    stats = BatchFSimCompiler(target_infidelity=1e-7, n_workers=2, calibration=cal).compute_summary_statistics(
        BatchFSimCompiler(target_infidelity=1e-7, n_workers=2, calibration=cal).compile_batch(batch))
    say(f"   {stats}  ({time.time() - t0:.1f} s)")
    out["batch"] = stats
    say("=" * 74)
    if as_json:
        print(json.dumps(out, indent=2, default=float))
    return 0
