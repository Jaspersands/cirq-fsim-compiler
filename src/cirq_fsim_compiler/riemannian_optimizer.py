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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from .unitary_ansatz import FSimCircuitTemplate
from .calibration_map import CouplerCalibration

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


#: FSim (θ, φ) points used to seed the first restarts: CZ-like, √iSWAP, iSWAP, Sycamore.
SEED_ANGLES: Tuple[Tuple[float, float], ...] = ((0.0, np.pi), (np.pi / 4, 0.0), (np.pi / 2, 0.0), (np.pi / 2, np.pi / 6))


def process_fidelity(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(np.trace(u.conj().T @ v)) ** 2 / (u.shape[0] ** 2))


class DifferentiableFSimSynthesizer:
    """
    Parameters
    ----------
    target_infidelity : stop as soon as a stage count reaches this
    seed : restart RNG seed
    method : ``"euclidean"`` (L-BFGS-B on Euler angles) or ``"riemannian"``
    """

    def __init__(self, target_infidelity: float = 1e-6, seed: int = 42, method: str = "euclidean"):
        if method not in ("euclidean", "riemannian"):
            raise ValueError("method must be 'euclidean' or 'riemannian'")
        self.target_infidelity = float(target_infidelity)
        self.seed = int(seed)
        self.method = method
        self._compiled_templates: Dict = {}
        if HAS_JAX:
            for n in (1, 2, 3):
                self._compiled_templates[n] = self._compile(FSimCircuitTemplate(n_stages=n))

    # ----------------------------------------------------------------- #
    @staticmethod
    def _compile(template: FSimCircuitTemplate):
        def loss_fn(p, u_t):
            u_a = template.evaluate_unitary_jax(p)
            ov = jnp.trace(jnp.conj(u_t).T @ u_a)
            return 1.0 - jnp.real(ov * jnp.conj(ov)) / 16.0

        return template, jax.jit(loss_fn), jax.jit(jax.grad(loss_fn))

    def _get_compiled(self, n_stages: int, fixed: Optional[Sequence[Tuple[float, float]]]):
        key = n_stages if fixed is None else (n_stages, tuple(fixed))
        if key not in self._compiled_templates:
            self._compiled_templates[key] = self._compile(FSimCircuitTemplate(n_stages=n_stages, fixed_fsim=fixed))
        return self._compiled_templates[key]

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
    def decompose(
        self,
        target_unitary: np.ndarray,
        max_stages: int = 3,
        n_restarts: int = 4,
        max_iter: int = 200,
        calibration: Optional[CouplerCalibration] = None,
    ) -> DecompositionResult:
        """Lowest stage count (1..max_stages) reaching ``target_infidelity``; else the best found."""
        target_unitary = np.asarray(target_unitary, dtype=np.complex128)
        if target_unitary.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {target_unitary.shape}")
        best = None
        for n_stages in range(1, int(max_stages) + 1):
            fixed = [calibration.angles] * n_stages if calibration is not None else None
            res = self._optimize_stage(target_unitary, n_stages, n_restarts, max_iter, fixed=fixed)
            res.calibration = calibration
            if best is None or res.infidelity < best.infidelity:
                best = res
            if res.infidelity <= self.target_infidelity:
                return res
        return best

    def _optimize_stage(
        self,
        target: np.ndarray,
        n_stages: int,
        n_restarts: int,
        max_iter: int,
        fixed: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> DecompositionResult:
        if self.method == "riemannian":
            from .riemannian_manifold import RiemannianFSimSolver
            solver = RiemannianFSimSolver(n_stages=n_stages, fixed_fsim=fixed, max_iter=max(400, max_iter), seed=self.seed)
            return solver.solve(target, n_restarts=n_restarts, target_infidelity=self.target_infidelity)

        template = FSimCircuitTemplate(n_stages=n_stages, fixed_fsim=fixed)
        n_params = template.num_params()
        rng = np.random.default_rng(self.seed)

        if HAS_JAX:
            template, loss_jit, grad_jit = self._get_compiled(n_stages, fixed)
            u_t = jnp.asarray(target)

            def cost_and_grad(p):
                pj = jnp.asarray(p)
                return float(loss_jit(pj, u_t)), np.asarray(grad_jit(pj, u_t), dtype=float)
        else:
            def cost_and_grad(p):
                return self._cost_and_grad_numpy(template, p, target)

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
                callback=lambda xk: hist.append(cost_and_grad(xk)[0]),
                options={"maxiter": int(max_iter), "ftol": 1e-15, "gtol": 1e-12},
            )
            if res.fun < best_loss:
                best_loss, best_p, best_hist = float(res.fun), res.x, hist
            if best_loss < self.target_infidelity:
                break

        u_synth = template.evaluate_unitary_np(best_p)
        fsim_angles, sq_angles = template.extract_angles(best_p)
        fid = process_fidelity(target, u_synth)
        return DecompositionResult(
            n_stages=n_stages,
            infidelity=float(1.0 - fid),
            fidelity=float(min(1.0, fid)),
            optimal_params=np.asarray(best_p),
            fsim_angles=fsim_angles,
            single_qubit_angles=sq_angles,
            synthesized_unitary=u_synth,
            target_unitary=target,
            is_success=bool(1.0 - fid <= self.target_infidelity * 10),
            native=template.is_fixed,
            method="euclidean",
            loss_history=best_hist,
            n_restarts_used=used,
        )


def synthesize_unitary_to_fsim(
    unitary: np.ndarray,
    max_stages: int = 3,
    target_infidelity: float = 1e-6,
    calibration: Optional[CouplerCalibration] = None,
    method: str = "euclidean",
    seed: int = 42,
) -> DecompositionResult:
    """Convenience wrapper around :class:`DifferentiableFSimSynthesizer`."""
    synth = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, seed=seed, method=method)
    return synth.decompose(np.asarray(unitary), max_stages=max_stages, calibration=calibration)
