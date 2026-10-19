"""
Differentiable synthesis of two-qubit unitaries into FSim + single-qubit layers.

Two solvers share the same loss  L = 1 − |Tr(U_t† U(p))|²/16:

* ``method="euclidean"`` — L-BFGS-B over Euler angles (a chart of the product
  manifold SU(2)^{2(n+1)} × T^{2n}); analytic JAX gradients, NumPy fallback.
* ``method="riemannian"`` — gradient descent directly on the SU(2) blocks with
  Cayley retractions (see :mod:`riemannian_manifold`).

Calibrated mode (``calibration=CouplerCalibration``) freezes every FSim to
the coupler's measured native angles and searches only the dressing layers,
returning the minimum number of native applications that reaches the target.

Stage search. With ``weyl_pruning=True`` (the default) the target's Weyl
coordinates decide where the search starts (:func:`weyl.predict_stages`).
The minimum is known for every native: in closed form for free angles,
CZ-class and √iSWAP-class natives, and from the exact n-gate regions of the
monodromy polytope for any other fixed FSim (:mod:`monodromy`). Synthesis
starts at that count. If the optimiser fails at that first count, it retries there once with
three times the restarts and five times the iterations, from new points.
If that also fails, it moves to the next count. With ``weyl_pruning=False``
the search starts at one stage, as in 0.4.

Analytic routes. For a fixed native gate two constructions from Cirq are
used where they apply, converted to this package's template: a
controlled-phase-class target (Weyl y = z = 0) with two native gates via
``cirq.decompose_cphase_into_two_fsim``, and any target with four gates via
``cirq.decompose_two_qubit_interaction_into_four_fsim_gates`` when the
native gate has 3π/8 ≤ |θ| ≤ 5π/8 and |φ| ≤ π/4 (Sycamore does). The first is
tried before the optimiser at n = 2 for every calibration (Cirq's routine
decides whether the native gate can make the controlled phase); the second
only if the optimiser fails at n = 4. Three Sycamore gates do not cover the Weyl chamber (they miss
x + y + |z| < π/8 with z ≠ 0), so n = 4 is part of the default search (``max_stages=4``).

Failure. If exact synthesis ends above ``target_infidelity``, the result is
not equivalent to the target. ``on_failure`` decides what happens: "warn"
(the default) issues a :class:`SynthesisWarning`, "raise" raises
:class:`SynthesisError`, "ignore" returns the result silently. In
noise-aware mode an approximation was asked for, so nothing is reported.

Tolerance. The Weyl-chamber tests use a coordinate tolerance of √ε for a
target infidelity ε, because the infidelity grows as the squared coordinate
distance; the zero- and one-gate tests use the closed-form infidelity
itself.

Noise-aware approximation. With ``approximation="noise_aware"`` and a
per-gate process infidelity ε, the synthesiser compares the exact
decomposition, n_exact gates, with the best approximations using fewer
gates. It returns the n that minimises (1 − F(n)) + n·ε, an estimate of the
total process infidelity when each native gate adds ε and single-qubit gates
are free. The exact decomposition is always one of the candidates, so the
returned estimate is never above the exact one (a property of the
construction, not a result). The best fit with fewer gates uses closed forms
where they exist (zero gates; one fixed native gate; one free FSim, by
maximising the closed-form class fidelity over (θ, φ)) and otherwise a
multi-start search with at least 16 restarts of 500 iterations. Numerical
fits can only overestimate the best infidelity, so the benefit of
approximating is, if anything, underestimated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from .unitary_ansatz import FSimCircuitTemplate, check_max_stages, fsim_matrix_np, euler_zxz_from_unitary
from .calibration_map import CouplerCalibration
from . import weyl

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False
    jax = None
    jnp = np


@dataclass
class DecompositionResult:
    """Outcome of synthesising a 4×4 unitary."""

    n_stages: int
    infidelity: float
    fidelity: float
    optimal_params: np.ndarray
    fsim_angles: List[Tuple[float, float]]
    single_qubit_angles: List[List[Tuple[float, float, float]]]
    synthesized_unitary: np.ndarray
    target_unitary: np.ndarray
    is_success: bool
    native: bool = False
    method: str = "euclidean"
    loss_history: List[float] = field(default_factory=list)
    n_restarts_used: int = 1
    calibration: Optional[CouplerCalibration] = None
    #: closed-form minimum stage count from the Weyl coordinates, when known
    predicted_stages: Optional[int] = None
    #: stage count of the exact decomposition (differs from n_stages when approximating)
    exact_n_stages: Optional[int] = None
    #: noise-aware mode: (1 − F) + n·ε for the returned decomposition
    expected_infidelity: Optional[float] = None
    gate_error: Optional[float] = None
    approximate: bool = False


#: FSim (θ, φ) points used to seed the first restarts: CZ-like, √iSWAP, iSWAP, Sycamore.
SEED_ANGLES: Tuple[Tuple[float, float], ...] = ((0.0, np.pi), (np.pi / 4, 0.0), (np.pi / 2, 0.0), (np.pi / 2, np.pi / 6))


def process_fidelity(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(np.trace(u.conj().T @ v)) ** 2 / (u.shape[0] ** 2))


APPROXIMATIONS = ("exact", "noise_aware")
ON_FAILURE = ("warn", "raise", "ignore")


class SynthesisWarning(UserWarning):
    """Exact synthesis did not reach the target infidelity."""


class SynthesisError(RuntimeError):
    """Exact synthesis did not reach the target infidelity (``on_failure="raise"``)."""

#: jitted loss and gradient per template, shared by all synthesisers (JAX compiles each once per process)
_COMPILED: Dict = {}


class DifferentiableFSimSynthesizer:
    """
    Parameters
    ----------
    target_infidelity : stop as soon as a stage count reaches this
    seed : restart RNG seed
    method : ``"euclidean"`` (L-BFGS-B on Euler angles) or ``"riemannian"``
    weyl_pruning : start the stage search where the Weyl coordinates allow
    approximation : ``"exact"`` or ``"noise_aware"`` (see the module docstring)
    gate_error : per-gate process infidelity ε for noise-aware mode; if None,
        the calibration's ``gate_error`` is used
    on_failure : "warn" (default), "raise" or "ignore" when exact synthesis
        misses ``target_infidelity``
    """

    def __init__(self, target_infidelity: float = 1e-6, seed: int = 42, method: str = "euclidean",
                 weyl_pruning: bool = True, approximation: str = "exact", gate_error: Optional[float] = None,
                 on_failure: str = "warn"):
        if method not in ("euclidean", "riemannian"):
            raise ValueError("method must be 'euclidean' or 'riemannian'")
        if approximation not in APPROXIMATIONS:
            raise ValueError(f"approximation must be one of {APPROXIMATIONS}")
        if on_failure not in ON_FAILURE:
            raise ValueError(f"on_failure must be one of {ON_FAILURE}")
        self.on_failure = on_failure
        self.target_infidelity = float(target_infidelity)
        self.seed = int(seed)
        self.method = method
        self.weyl_pruning = bool(weyl_pruning)
        self.approximation = approximation
        self.gate_error = None if gate_error is None else float(gate_error)
        self._compiled_templates: Dict = _COMPILED
        if HAS_JAX:
            for n in (0, 1, 2, 3):
                if n not in _COMPILED:
                    _COMPILED[n] = self._compile(FSimCircuitTemplate(n_stages=n))

    # ----------------------------------------------------------------- #
    @staticmethod
    def _compile(template: FSimCircuitTemplate):
        def loss_fn(p, u_t):
            u_a = template.evaluate_unitary_jax(p)
            ov = jnp.trace(jnp.conj(u_t).T @ u_a)
            return 1.0 - jnp.real(ov * jnp.conj(ov)) / 16.0

        return template, jax.jit(loss_fn), jax.jit(jax.grad(loss_fn))

    @staticmethod
    def _compile_fixed(n_stages: int):
        """Loss and gradient for ``n_stages`` frozen gates, with the angles as a run-time argument."""
        free = FSimCircuitTemplate(n_stages=n_stages)

        def loss_fn(p, u_t, angles):
            parts = [p[0:6]]
            for s in range(n_stages):
                parts += [angles[s], p[6 + 6 * s:12 + 6 * s]]
            u_a = free.evaluate_unitary_jax(jnp.concatenate(parts))
            ov = jnp.trace(jnp.conj(u_t).T @ u_a)
            return 1.0 - jnp.real(ov * jnp.conj(ov)) / 16.0

        return jax.jit(loss_fn), jax.jit(jax.grad(loss_fn))

    def _get_compiled(self, n_stages: int, fixed: Optional[Sequence[Tuple[float, float]]]):
        """(template, loss(p, u), grad(p, u)); one compilation per stage count, whatever the fixed angles."""
        if fixed is None:
            if n_stages not in self._compiled_templates:
                self._compiled_templates[n_stages] = self._compile(FSimCircuitTemplate(n_stages=n_stages))
            return self._compiled_templates[n_stages]
        key = ("fixed", n_stages)
        if key not in self._compiled_templates:
            self._compiled_templates[key] = self._compile_fixed(n_stages)
        loss_jit, grad_jit = self._compiled_templates[key]
        angles = jnp.asarray(np.array([[float(t), float(p)] for t, p in fixed], dtype=float).reshape(n_stages, 2))
        template = FSimCircuitTemplate(n_stages=n_stages, fixed_fsim=fixed)
        return template, (lambda p, u: loss_jit(p, u, angles)), (lambda p, u: grad_jit(p, u, angles))

    @staticmethod
    def _cost_and_grad_numpy(template: FSimCircuitTemplate, p: np.ndarray, target: np.ndarray, eps: float = 1e-6):
        def loss(q):
            return 1.0 - process_fidelity(target, template.evaluate_unitary_np(q))

        base = loss(p)
        g = np.zeros_like(p)
        for i in range(len(p)):
            pp = p.copy(); pp[i] += eps
            pm = p.copy(); pm[i] -= eps
            g[i] = (loss(pp) - loss(pm)) / (2 * eps)
        return base, g

    # ----------------------------------------------------------------- #
    def predict(self, target_unitary: np.ndarray, calibration: Optional[CouplerCalibration] = None,
                max_count: int = 6) -> "weyl.StagePrediction":
        """Weyl-chamber prediction used to choose the first stage count (exact counts searched up to ``max_count``)."""
        tol = np.sqrt(max(self.target_infidelity, 1e-16))
        angles = None if calibration is None else calibration.angles
        return weyl.predict_stages(np.asarray(target_unitary, dtype=np.complex128), fsim_angles=angles, tol=tol,
                                   max_count=max_count)

    def decompose(
        self,
        target_unitary: np.ndarray,
        max_stages: int = 4,
        n_restarts: int = 4,
        max_iter: int = 200,
        calibration: Optional[CouplerCalibration] = None,
        approximation: Optional[str] = None,
        gate_error: Optional[float] = None,
    ) -> DecompositionResult:
        """
        Lowest stage count (≤ max_stages) reaching ``target_infidelity``. If
        none does, the best result found, after a warning or an exception as
        set by ``on_failure``.

        ``approximation`` and ``gate_error`` override the values given to the
        constructor for this call.
        """
        target_unitary = np.asarray(target_unitary, dtype=np.complex128)
        if target_unitary.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {target_unitary.shape}")
        max_stages = check_max_stages(max_stages)
        approximation = approximation or self.approximation
        if approximation not in APPROXIMATIONS:
            raise ValueError(f"approximation must be one of {APPROXIMATIONS}")
        exact = self._decompose_exact(target_unitary, max_stages, n_restarts, max_iter, calibration)
        if exact.infidelity > self.target_infidelity and approximation == "exact":
            self._report_failure(exact, calibration, max_stages)
        if approximation == "exact":
            return exact
        eps = gate_error if gate_error is not None else self.gate_error
        if eps is None and calibration is not None:
            eps = calibration.gate_error
        if eps is None:
            raise ValueError("noise-aware synthesis needs gate_error (or a calibration with gate_error)")
        return self._noise_aware(target_unitary, exact, float(eps), n_restarts, max_iter, calibration)

    def _report_failure(self, res: DecompositionResult, calibration, max_stages) -> None:
        if self.on_failure == "ignore":
            return
        native = "free angles" if calibration is None else "native FSim(%.4f, %.4f)" % tuple(calibration.angles)
        msg = (f"exact synthesis reached process infidelity {res.infidelity:.2e} > target {self.target_infidelity:.1e} "
               f"with {res.n_stages} gate(s) ({native}, max_stages={max_stages}); the result is not equivalent to the "
               "target. Raise max_stages or n_restarts, or pass on_failure='raise' to stop on this.")
        if self.on_failure == "raise":
            raise SynthesisError(msg)
        warnings.warn(msg, SynthesisWarning, stacklevel=3)

    def _decompose_exact(self, target_unitary, max_stages, n_restarts, max_iter, calibration) -> DecompositionResult:
        max_stages = int(max_stages)
        predicted = None
        start = 1
        coords = weyl.weyl_coordinates(target_unitary) if calibration is not None else None
        if self.weyl_pruning:
            pred = self.predict(target_unitary, calibration, max_count=max_stages)
            predicted = pred.exact
            start = min(pred.exact if pred.exact is not None else pred.lower, max_stages)
        tol_c = np.sqrt(max(self.target_infidelity, 1e-16))
        best = None
        for n_stages in range(start, max_stages + 1):
            fixed = [calibration.angles] * n_stages if calibration is not None else None
            res = None
            if (calibration is not None and n_stages == 2 and abs(coords[1]) <= tol_c and abs(coords[2]) <= tol_c
                    and coords[0] > tol_c):
                res = self._analytic_cphase(target_unitary, calibration)
                if res is not None and res.infidelity > self.target_infidelity:
                    res = None
            if res is None:
                res = self._optimize_stage(target_unitary, n_stages, n_restarts, max_iter, fixed=fixed)
            if res.infidelity > self.target_infidelity and self.weyl_pruning and n_stages == start:
                # a proven lower bound (or the closed-form minimum): retry from fresh points with more restarts and
                # iterations before moving on (some targets converge slowly, e.g. near the x = π/4 face)
                retry = self._optimize_stage(target_unitary, n_stages, 3 * n_restarts, 5 * max_iter, fixed=fixed,
                                             seed=self.seed + 7919)
                retry.n_restarts_used += res.n_restarts_used
                if retry.infidelity < res.infidelity:
                    res = retry
            if res.infidelity > self.target_infidelity and calibration is not None and n_stages == 4:
                alt = self._analytic_four_fsim(target_unitary, calibration)
                if alt is not None and alt.infidelity < res.infidelity:
                    res = alt
            res.calibration = calibration
            res.predicted_stages = predicted
            if best is None or res.infidelity < best.infidelity:
                best = res
            if res.infidelity <= self.target_infidelity:
                best = res
                break
        best.exact_n_stages = best.n_stages
        return best

    # ----------------------------------------------------------------- #
    # analytic constructions from Cirq, converted to the template
    # ----------------------------------------------------------------- #
    def _result_from_ops(self, ops, target, calibration, label: str) -> Optional[DecompositionResult]:
        """Template parameters for a Cirq circuit made of the native gate and single-qubit gates."""
        import cirq

        q = cirq.LineQubit.range(2)
        native = fsim_matrix_np(*calibration.angles)
        acc = [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)]
        layers = []
        for op in ops:
            if len(op.qubits) == 0:
                continue
            if len(op.qubits) == 1:
                k = q.index(op.qubits[0])
                acc[k] = cirq.unitary(op) @ acc[k]
                continue
            u2 = cirq.Circuit(op).unitary(qubit_order=q)
            if not np.allclose(u2, native, atol=1e-9):
                return None
            layers.append(acc)
            acc = [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)]
        layers.append(acc)
        n = len(layers) - 1
        params = []
        for a0, a1 in layers:
            params += list(euler_zxz_from_unitary(a0)) + list(euler_zxz_from_unitary(a1))
        template = FSimCircuitTemplate(n_stages=n, fixed_fsim=[calibration.angles] * n)
        params = np.asarray(params, dtype=float)
        u_synth = template.evaluate_unitary_np(params)
        fid = process_fidelity(target, u_synth)
        fsim_angles, sq_angles = template.extract_angles(params)
        return DecompositionResult(
            n_stages=n, infidelity=float(max(0.0, 1.0 - fid)), fidelity=float(min(1.0, fid)), optimal_params=params,
            fsim_angles=fsim_angles, single_qubit_angles=sq_angles, synthesized_unitary=u_synth, target_unitary=target,
            is_success=bool(1.0 - fid <= self.target_infidelity), native=True, method=label, n_restarts_used=0)

    def _analytic_cphase(self, target, calibration) -> Optional[DecompositionResult]:
        """Controlled-phase class (Weyl y = z = 0) with two native gates, via cirq.decompose_cphase_into_two_fsim."""
        import cirq

        x = float(weyl.weyl_coordinates(target)[0])
        q = cirq.LineQubit.range(2)
        try:
            cph = cirq.CZPowGate(exponent=4.0 * x / np.pi)
            mid = list(cirq.decompose_cphase_into_two_fsim(cph, fsim_gate=cirq.FSimGate(*calibration.angles), qubits=q))
        except ValueError:
            return None
        kt, kc = cirq.kak_decomposition(target), cirq.kak_decomposition(cirq.unitary(cph))
        # target = g_t A_t E B_t and cphase = g_c A_c E B_c, so target ∝ (A_t A_c†) cphase (B_c† B_t)
        before = [kc.single_qubit_operations_before[k].conj().T @ kt.single_qubit_operations_before[k] for k in (0, 1)]
        after = [kt.single_qubit_operations_after[k] @ kc.single_qubit_operations_after[k].conj().T for k in (0, 1)]
        ops = [cirq.MatrixGate(before[k]).on(q[k]) for k in (0, 1)] + mid + [cirq.MatrixGate(after[k]).on(q[k]) for k in (0, 1)]
        return self._result_from_ops(ops, target, calibration, "analytic (decompose_cphase_into_two_fsim)")

    def _analytic_four_fsim(self, target, calibration) -> Optional[DecompositionResult]:
        """Any target with four native gates, via cirq.decompose_two_qubit_interaction_into_four_fsim_gates."""
        import cirq

        q = cirq.LineQubit.range(2)
        try:
            circ = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(
                target, fsim_gate=cirq.FSimGate(*calibration.angles), qubits=q)
        except ValueError:
            return None
        return self._result_from_ops(list(circ.all_operations()), target, calibration,
                                     "analytic (decompose_two_qubit_interaction_into_four_fsim_gates)")

    # ----------------------------------------------------------------- #
    # best approximations with a given number of gates (noise-aware mode)
    # ----------------------------------------------------------------- #
    @staticmethod
    def best_single_free_fsim(target) -> Tuple[float, float, float]:
        """
        (θ, φ, F): the FSim(θ, φ) whose class is closest to the target, with
        the closed-form class fidelity F, maximised over (θ, φ).
        """
        a = weyl._coords(target)

        def neg(v):
            b = np.array([-v[0], -v[0], -v[1]])            # FSim(θ, φ) ~ (−θ/2, −θ/2, −φ/4)
            c = weyl._weyl_images(b) - a
            return -float(np.max(np.prod(np.cos(c), axis=1) ** 2 + np.prod(np.sin(c), axis=1) ** 2))

        grid = [(t, p) for t in np.linspace(0.0, np.pi / 2, 25) for p in np.linspace(-np.pi / 2, np.pi / 2, 25)]
        vals = sorted((neg(g), g) for g in grid)[:4]
        best_v, best_x = vals[0]
        for v0, x0 in vals:
            r = scipy.optimize.minimize(neg, np.array(x0), method="Nelder-Mead",
                                        options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 4000})
            if r.fun < best_v:
                best_v, best_x = float(r.fun), tuple(r.x)
        return float(2.0 * best_x[0]), float(4.0 * best_x[1]), float(min(1.0, -best_v))

    def closed_form_best_infidelity(self, target, n: int, calibration: Optional[CouplerCalibration]) -> Optional[float]:
        """Best infidelity with exactly n gates where a closed form is known, else None."""
        coords = weyl.weyl_coordinates(target)
        if n == 0:
            return float(1.0 - weyl.class_fidelity(coords, np.zeros(3)))
        if n == 1 and calibration is not None:
            return float(1.0 - weyl.class_fidelity(coords, weyl.fsim_weyl_coordinates(*calibration.angles)))
        if n == 1:
            return float(1.0 - self.best_single_free_fsim(coords)[2])
        return None

    def best_fit(self, target, n: int, calibration: Optional[CouplerCalibration] = None, n_restarts: int = 16,
                 max_iter: int = 500, seed: Optional[int] = None) -> DecompositionResult:
        """The best approximation found with exactly ``n`` native gates."""
        target = np.asarray(target, dtype=np.complex128)
        if calibration is None and n == 1:
            theta, phi, f = self.best_single_free_fsim(target)
            res = self._optimize_stage(target, 1, max(n_restarts, 8), max_iter, fixed=[(theta, phi)], seed=seed,
                                       stop_at=1.0 - f)
            res.native = False
            return res
        floor = self.closed_form_best_infidelity(target, n, calibration) if n <= 1 else None
        fixed = None if calibration is None else [calibration.angles] * n
        return self._optimize_stage(target, n, n_restarts, max_iter, fixed=fixed, seed=seed, stop_at=floor)

    def _noise_aware(self, target, exact: DecompositionResult, eps: float, n_restarts: int, max_iter: int,
                     calibration: Optional[CouplerCalibration]) -> DecompositionResult:
        """Choose n ≤ n_exact minimising (1 − F(n)) + n·ε; the exact result is always a candidate."""
        best, best_cost = exact, exact.infidelity + exact.n_stages * eps
        n_r, it = max(16, 4 * int(n_restarts)), max(500, int(max_iter))
        for n in range(exact.n_stages - 1, -1, -1):
            floor = self.closed_form_best_infidelity(target, n, calibration) if n <= 1 else 0.0
            if floor + n * eps >= best_cost:
                continue
            res = self.best_fit(target, n, calibration, n_r, it)
            res.calibration = calibration
            cost = res.infidelity + n * eps
            if cost < best_cost:
                best, best_cost = res, cost
        best.predicted_stages = exact.predicted_stages
        best.exact_n_stages = exact.n_stages
        best.expected_infidelity = float(best_cost)
        best.gate_error = eps
        best.approximate = best is not exact
        return best

    def _optimize_stage(
        self,
        target: np.ndarray,
        n_stages: int,
        n_restarts: int,
        max_iter: int,
        fixed: Optional[Sequence[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        stop_at: Optional[float] = None,
    ) -> DecompositionResult:
        """Multi-start search with ``n_stages`` gates; stops early at the target, or at ``stop_at`` (a known optimum)."""
        seed = self.seed if seed is None else int(seed)
        stop = self.target_infidelity if stop_at is None else max(self.target_infidelity, float(stop_at) + 1e-12)
        if self.method == "riemannian":
            from .riemannian_manifold import RiemannianFSimSolver
            solver = RiemannianFSimSolver(n_stages=n_stages, fixed_fsim=fixed, max_iter=max(400, max_iter), seed=seed)
            return solver.solve(target, n_restarts=n_restarts, target_infidelity=self.target_infidelity)

        template = FSimCircuitTemplate(n_stages=n_stages, fixed_fsim=fixed)
        n_params = template.num_params()
        rng = np.random.default_rng(seed)

        if HAS_JAX:
            template, loss_jit, grad_jit = self._get_compiled(n_stages, fixed)
            u_t = jnp.asarray(target)

            def _cost_and_grad(p):
                pj = jnp.asarray(p)
                return float(loss_jit(pj, u_t)), np.asarray(grad_jit(pj, u_t), dtype=float)
        else:
            def _cost_and_grad(p):
                return self._cost_and_grad_numpy(template, p, target)

        last: Dict[str, Any] = {}

        def cost_and_grad(p):
            out = _cost_and_grad(p)
            last["p"], last["f"] = np.array(p, copy=True), out[0]
            return out

        def loss_at(p):                     # the callback's point is almost always the last one evaluated
            if "p" in last and np.array_equal(last["p"], p):
                return last["f"]
            return _cost_and_grad(p)[0]

        best_loss, best_p, best_hist, used = np.inf, None, [], 0
        bounds = [(-2 * np.pi, 2 * np.pi)] * n_params
        for r in range(int(n_restarts)):
            used = r + 1
            p0 = rng.uniform(-np.pi, np.pi, size=n_params)
            if not template.is_fixed and r < len(SEED_ANGLES):
                # cycle the FSim angles through a library of native points
                for s in range(n_stages):
                    idx = 6 + s * 8
                    p0[idx], p0[idx + 1] = SEED_ANGLES[r]
            hist: List[float] = []
            res = scipy.optimize.minimize(
                cost_and_grad, p0, method="L-BFGS-B", jac=True, bounds=bounds,
                callback=lambda xk: hist.append(loss_at(xk)),
                options={"maxiter": int(max_iter), "ftol": 1e-15, "gtol": 1e-12},
            )
            if res.fun < best_loss:
                best_loss, best_p, best_hist = float(res.fun), res.x, hist
            if best_loss <= stop:
                break

        u_synth = template.evaluate_unitary_np(best_p)
        fsim_angles, sq_angles = template.extract_angles(best_p)
        fid = process_fidelity(target, u_synth)
        return DecompositionResult(
            n_stages=n_stages,
            infidelity=float(max(0.0, 1.0 - fid)),
            fidelity=float(min(1.0, fid)),
            optimal_params=np.asarray(best_p),
            fsim_angles=fsim_angles,
            single_qubit_angles=sq_angles,
            synthesized_unitary=u_synth,
            target_unitary=target,
            is_success=bool(1.0 - fid <= self.target_infidelity),
            native=template.is_fixed,
            method="euclidean",
            loss_history=best_hist,
            n_restarts_used=used,
        )


def synthesize_unitary_to_fsim(
    unitary: np.ndarray,
    max_stages: int = 4,
    target_infidelity: float = 1e-6,
    calibration: Optional[CouplerCalibration] = None,
    method: str = "euclidean",
    seed: int = 42,
    approximation: str = "exact",
    gate_error: Optional[float] = None,
    weyl_pruning: bool = True,
    on_failure: str = "warn",
) -> DecompositionResult:
    """Convenience wrapper around :class:`DifferentiableFSimSynthesizer`."""
    synth = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, seed=seed, method=method,
                                          weyl_pruning=weyl_pruning, approximation=approximation, gate_error=gate_error,
                                          on_failure=on_failure)
    return synth.decompose(np.asarray(unitary), max_stages=max_stages, calibration=calibration)
