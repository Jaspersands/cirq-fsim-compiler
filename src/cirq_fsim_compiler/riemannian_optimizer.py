"""
Differentiable Riemannian Unitary Optimizer on SU(4).

Synthesizes minimum-depth FSim + 1Q sequences approximating arbitrary target unitaries
using JAX automatic differentiation and multi-start gradient optimization.
"""

from __future__ import annotations
import numpy as np
import scipy.optimize
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = np

from .unitary_ansatz import FSimCircuitTemplate


@dataclass
class DecompositionResult:
    """Stores the compilation results of synthesizing a 2-qubit unitary."""
    n_stages: int
    infidelity: float
    fidelity: float
    optimal_params: np.ndarray
    fsim_angles: List[Tuple[float, float]] # [(theta_1, phi_1), (theta_2, phi_2), ...]
    single_qubit_angles: List[List[Tuple[float, float, float]]] # [[(a0,b0,g0), (a1,b1,g1)], ...]
    synthesized_unitary: np.ndarray
    target_unitary: np.ndarray
    is_success: bool


class DifferentiableFSimSynthesizer:
    """
    JAX-accelerated unitary decomposer into native Sycamore FSim gates.
    """

    def __init__(self, target_infidelity: float = 1e-6):
        self.target_infidelity = target_infidelity
        self._compiled_templates = {}
        if HAS_JAX:
            self._compile_jax_solvers()

    def _compile_jax_solvers(self):
        def _make_solver(template):
            def loss_fn(p: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
                u_a = template.evaluate_unitary_jax(p)
                overlap = jnp.trace(jnp.conjugate(jnp.transpose(u_t)) @ u_a)
                fid = jnp.real(overlap * jnp.conjugate(overlap)) / 16.0
                return 1.0 - fid

            grad_fn = jax.jit(jax.grad(loss_fn))
            loss_jit = jax.jit(loss_fn)
            return loss_jit, grad_fn

        for n_stages in [1, 2, 3]:
            template = FSimCircuitTemplate(n_stages=n_stages)
            loss_jit, grad_fn = _make_solver(template)
            self._compiled_templates[n_stages] = (template, loss_jit, grad_fn)

    def decompose(
        self,
        target_unitary: np.ndarray,
        max_stages: int = 3,
        n_restarts: int = 4,
        max_iter: int = 150,
    ) -> DecompositionResult:
        """
        Attempts to decompose target_unitary into 1, 2, or 3 FSim stages.
        Returns the lowest-depth representation achieving the target infidelity.
        """
        if target_unitary.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {target_unitary.shape}")

        best_result = None

        for n_stages in range(1, max_stages + 1):
            res_stage = self._optimize_stage(target_unitary, n_stages, n_restarts, max_iter)
            if best_result is None or res_stage.infidelity < best_result.infidelity:
                best_result = res_stage

            if res_stage.infidelity <= self.target_infidelity:
                return res_stage

        return best_result

    def _optimize_stage(
        self,
        target_unitary: np.ndarray,
        n_stages: int,
        n_restarts: int,
        max_iter: int,
    ) -> DecompositionResult:
        template = FSimCircuitTemplate(n_stages=n_stages)
        n_params = template.num_params()
        rng = np.random.default_rng(42)

        best_loss = 1.0
        best_p = None

        if HAS_JAX and n_stages in self._compiled_templates:
            _, loss_jit, grad_fn = self._compiled_templates[n_stages]
            u_t_jax = jnp.array(target_unitary, dtype=jnp.complex128)

            def cost_and_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
                p_jax = jnp.array(p, dtype=jnp.float64)
                val = float(loss_jit(p_jax, u_t_jax))
                g = np.array(grad_fn(p_jax, u_t_jax), dtype=np.float64)
                return val, g
        else:
            def cost_and_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
                u_a = template.evaluate_unitary_np(p)
                overlap = np.trace(target_unitary.conj().T @ u_a)
                fid = float(np.real(overlap * np.conjugate(overlap)) / 16.0)
                loss = 1.0 - fid
                # Numerical gradient
                dp = 1e-5
                grad = np.zeros_like(p)
                for i in range(len(p)):
                    p_up = p.copy()
                    p_up[i] += dp
                    u_up = template.evaluate_unitary_np(p_up)
                    fid_up = float(np.real(np.trace(target_unitary.conj().T @ u_up) ** 2) / 16.0)
                    grad[i] = (fid - fid_up) / dp
                return loss, grad

        for r in range(n_restarts):
            p0 = rng.uniform(-np.pi, np.pi, size=n_params)
            # Give reasonable seeds for standard gates
            if r == 0 and n_stages >= 1:
                p0[6] = np.pi / 2.0 # theta
                p0[7] = np.pi / 6.0 # phi

            bounds = [(-2 * np.pi, 2 * np.pi) for _ in range(n_params)]
            opt_res = scipy.optimize.minimize(
                cost_and_grad,
                p0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": 1e-12, "disp": False},
            )

            if opt_res.fun < best_loss:
                best_loss = float(opt_res.fun)
                best_p = opt_res.x

            if best_loss < self.target_infidelity:
                break

        u_synth = template.evaluate_unitary_np(best_p)
        fid = float(np.real(np.trace(target_unitary.conj().T @ u_synth) * np.trace(u_synth.conj().T @ target_unitary)) / 16.0)

        # Parse angles
        fsim_angles = []
        sq_angles = []
        idx = 0
        # Layer 0 1Q
        sq_layer0 = [
            (float(best_p[0]), float(best_p[1]), float(best_p[2])),
            (float(best_p[3]), float(best_p[4]), float(best_p[5])),
        ]
        sq_angles.append(sq_layer0)
        idx += 6

        for s in range(n_stages):
            fsim_angles.append((float(best_p[idx]), float(best_p[idx + 1])))
            idx += 2
            sq_layer = [
                (float(best_p[idx]), float(best_p[idx + 1]), float(best_p[idx + 2])),
                (float(best_p[idx + 3]), float(best_p[idx + 4]), float(best_p[idx + 5])),
            ]
            sq_angles.append(sq_layer)
            idx += 6

        return DecompositionResult(
            n_stages=n_stages,
            infidelity=float(best_loss),
            fidelity=min(1.0, float(fid)),
            optimal_params=best_p,
            fsim_angles=fsim_angles,
            single_qubit_angles=sq_angles,
            synthesized_unitary=u_synth,
            target_unitary=target_unitary,
            is_success=bool(best_loss <= self.target_infidelity * 10),
        )


def synthesize_unitary_to_fsim(
    unitary: np.ndarray, max_stages: int = 3, target_infidelity: float = 1e-6
) -> DecompositionResult:
    """Convenience helper to synthesize a 4x4 unitary matrix."""
    synth = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity)
    return synth.decompose(unitary, max_stages=max_stages)
