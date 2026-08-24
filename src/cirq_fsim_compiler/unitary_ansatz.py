"""
Parameterized Unitary Circuit Templates for Google Sycamore FSim Architectures.

Provides exact matrix representations of:
1. FSim(theta, phi) two-qubit interaction.
2. Arbitrary SU(2) single-qubit Euler rotations Rz(alpha) Rx(beta) Rz(gamma).
3. 1-, 2-, and 3-stage FSim + 1Q universal SU(4) decomposition templates in NumPy & JAX.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional, Union

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = np


def fsim_matrix_np(theta: float, phi: float) -> np.ndarray:
    """Computes the 4x4 FSim(theta, phi) unitary matrix."""
    c = np.cos(theta)
    s = np.sin(theta)
    phase = np.exp(-1.0j * phi)

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -1.0j * s, 0.0],
            [0.0, -1.0j * s, c, 0.0],
            [0.0, 0.0, 0.0, phase],
        ],
        dtype=np.complex128,
    )


def single_qubit_zxz_np(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Computes the 2x2 unitary Rz(alpha) Rx(beta) Rz(gamma)."""
    # Rz(a) = diag(exp(-i a/2), exp(i a/2))
    # Rx(b) = [[cos(b/2), -i sin(b/2)], [-i sin(b/2), cos(b/2)]]
    cos_b = np.cos(0.5 * beta)
    sin_b = np.sin(0.5 * beta)

    e_a_m = np.exp(-0.5j * alpha)
    e_a_p = np.exp(0.5j * alpha)
    e_g_m = np.exp(-0.5j * gamma)
    e_g_p = np.exp(0.5j * gamma)

    m = np.array(
        [
            [e_a_m * cos_b * e_g_m, -1.0j * e_a_m * sin_b * e_g_p],
            [-1.0j * e_a_p * sin_b * e_g_m, e_a_p * cos_b * e_g_p],
        ],
        dtype=np.complex128,
    )
    return m


def fsim_matrix_jax(theta: "jnp.ndarray", phi: "jnp.ndarray") -> "jnp.ndarray":
    """JAX-differentiable FSim(theta, phi) matrix."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    phase = jnp.exp(-1.0j * phi)

    row0 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    row1 = jnp.array([0.0 + 0.0j, c + 0.0j, -1.0j * s, 0.0 + 0.0j])
    row2 = jnp.array([0.0 + 0.0j, -1.0j * s, c + 0.0j, 0.0 + 0.0j])
    row3 = jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, phase])

    return jnp.stack([row0, row1, row2, row3])


def single_qubit_zxz_jax(alpha: "jnp.ndarray", beta: "jnp.ndarray", gamma: "jnp.ndarray") -> "jnp.ndarray":
    """JAX-differentiable single-qubit Euler rotation matrix."""
    cos_b = jnp.cos(0.5 * beta)
    sin_b = jnp.sin(0.5 * beta)

    e_a_m = jnp.exp(-0.5j * alpha)
    e_a_p = jnp.exp(0.5j * alpha)
    e_g_m = jnp.exp(-0.5j * gamma)
    e_g_p = jnp.exp(0.5j * gamma)

    row0 = jnp.array([e_a_m * cos_b * e_g_m, -1.0j * e_a_m * sin_b * e_g_p])
    row1 = jnp.array([-1.0j * e_a_p * sin_b * e_g_m, e_a_p * cos_b * e_g_p])
    return jnp.stack([row0, row1])


class FSimCircuitTemplate:
    """
    Template for synthesizing 2-qubit unitaries using n_stages of FSim gates.
    """

    def __init__(self, n_stages: int = 2):
        if n_stages not in [1, 2, 3]:
            raise ValueError(f"n_stages must be 1, 2, or 3, got {n_stages}")
        self.n_stages = n_stages

    def num_params(self) -> int:
        # Each stage has 1 FSim (2 params) + 2 single-qubit gates (6 params)
        # Plus 1 initial layer of 2 single-qubit gates (6 params)
        # Total params = (n_stages + 1) * 6 + n_stages * 2
        return (self.n_stages + 1) * 6 + self.n_stages * 2

    def evaluate_unitary_np(self, params: np.ndarray) -> np.ndarray:
        """Evaluates the full 4x4 unitary using NumPy."""
        idx = 0
        # Initial 1Q layer
        u0 = single_qubit_zxz_np(params[idx], params[idx + 1], params[idx + 2])
        u1 = single_qubit_zxz_np(params[idx + 3], params[idx + 4], params[idx + 5])
        idx += 6
        current_U = np.kron(u0, u1)

        for s in range(self.n_stages):
            # FSim layer
            theta_s = params[idx]
            phi_s = params[idx + 1]
            idx += 2
            fsim_mat = fsim_matrix_np(theta_s, phi_s)
            current_U = fsim_mat @ current_U

            # 1Q layer
            u0 = single_qubit_zxz_np(params[idx], params[idx + 1], params[idx + 2])
            u1 = single_qubit_zxz_np(params[idx + 3], params[idx + 4], params[idx + 5])
            idx += 6
            layer_1q = np.kron(u0, u1)
            current_U = layer_1q @ current_U

        return current_U

    def evaluate_unitary_jax(self, params: "jnp.ndarray") -> "jnp.ndarray":
        """Evaluates the full 4x4 unitary using JAX."""
        idx = 0
        u0 = single_qubit_zxz_jax(params[idx], params[idx + 1], params[idx + 2])
        u1 = single_qubit_zxz_jax(params[idx + 3], params[idx + 4], params[idx + 5])
        idx += 6
        current_U = jnp.kron(u0, u1)

        for s in range(self.n_stages):
            theta_s = params[idx]
            phi_s = params[idx + 1]
            idx += 2
            fsim_mat = fsim_matrix_jax(theta_s, phi_s)
            current_U = fsim_mat @ current_U

            u0 = single_qubit_zxz_jax(params[idx], params[idx + 1], params[idx + 2])
            u1 = single_qubit_zxz_jax(params[idx + 3], params[idx + 4], params[idx + 5])
            idx += 6
            layer_1q = jnp.kron(u0, u1)
            current_U = layer_1q @ current_U

        return current_U
