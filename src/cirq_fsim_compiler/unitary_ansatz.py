"""
Parameterised circuit templates for FSim-based two-qubit synthesis.

    U(p) = L_n · FSim(θ_n, φ_n) · L_{n−1} ⋯ FSim(θ_1, φ_1) · L_0,
    L_k  = R(α_k, β_k, γ_k) ⊗ R(α'_k, β'_k, γ'_k),   R = Rz(α) Rx(β) Rz(γ)

with ``n_stages`` ∈ {1, 2, 3}. Angles may be free (``fixed_fsim=None``) or
frozen to calibrated native values (``fixed_fsim=[(θ, φ), …]``), in which case
only the single-qubit layers are optimised — the situation on real hardware,
where the coupler exposes one native gate (e.g. Sycamore's FSim(π/2, π/6)).

Both NumPy and JAX evaluations are provided and produce identical matrices.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False
    jax = None
    jnp = np


# --------------------------------------------------------------------------- #
# Elementary matrices
# --------------------------------------------------------------------------- #
def fsim_matrix_np(theta: float, phi: float) -> np.ndarray:
    """cirq.FSimGate(θ, φ): [[1,0,0,0],[0,c,−is,0],[0,−is,c,0],[0,0,0,e^{−iφ}]]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]],
        dtype=np.complex128,
    )


def single_qubit_zxz_np(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Rz(α) Rx(β) Rz(γ) with Rz(a) = diag(e^{−ia/2}, e^{ia/2}), Rx(b) = exp(−ibX/2)."""
    cb, sb = np.cos(0.5 * beta), np.sin(0.5 * beta)
    e_am, e_ap = np.exp(-0.5j * alpha), np.exp(0.5j * alpha)
    e_gm, e_gp = np.exp(-0.5j * gamma), np.exp(0.5j * gamma)
    return np.array(
        [[e_am * cb * e_gm, -1j * e_am * sb * e_gp], [-1j * e_ap * sb * e_gm, e_ap * cb * e_gp]],
        dtype=np.complex128,
    )


def euler_zxz_from_unitary(u: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`single_qubit_zxz_np` up to global phase."""
    u = np.asarray(u, dtype=np.complex128)
    det = np.linalg.det(u)
    v = u / np.sqrt(det)                       # SU(2)
    beta = 2.0 * np.arctan2(abs(v[1, 0]), abs(v[0, 0]))
    cb, sb = np.cos(0.5 * beta), np.sin(0.5 * beta)
    if cb > 1e-9 and sb > 1e-9:
        sum_ag = -2.0 * np.angle(v[0, 0])
        diff_ag = -2.0 * np.angle(1j * v[0, 1])
    elif sb <= 1e-9:                            # diagonal: split the phase evenly
        sum_ag, diff_ag = -2.0 * np.angle(v[0, 0]), 0.0
    else:                                       # anti-diagonal
        sum_ag, diff_ag = 0.0, -2.0 * np.angle(1j * v[0, 1])
    return float(0.5 * (sum_ag + diff_ag)), float(beta), float(0.5 * (sum_ag - diff_ag))


def haar_random_unitary(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Haar-random U(dim) via QR of a complex Ginibre matrix with phase correction."""
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


if HAS_JAX:

    def fsim_matrix_jax(theta, phi):
        c, s = jnp.cos(theta), jnp.sin(theta)
        z = jnp.zeros((), dtype=jnp.complex128)
        one = jnp.ones((), dtype=jnp.complex128)
        return jnp.array(
            [[one, z, z, z], [z, c + 0j, -1j * s, z], [z, -1j * s, c + 0j, z], [z, z, z, jnp.exp(-1j * phi)]]
        )

    def single_qubit_zxz_jax(alpha, beta, gamma):
        cb, sb = jnp.cos(0.5 * beta), jnp.sin(0.5 * beta)
        e_am, e_ap = jnp.exp(-0.5j * alpha), jnp.exp(0.5j * alpha)
        e_gm, e_gp = jnp.exp(-0.5j * gamma), jnp.exp(0.5j * gamma)
        return jnp.array([[e_am * cb * e_gm, -1j * e_am * sb * e_gp], [-1j * e_ap * sb * e_gm, e_ap * cb * e_gp]])


# --------------------------------------------------------------------------- #
# Template
# --------------------------------------------------------------------------- #
class FSimCircuitTemplate:
    """
    Parameters
    ----------
    n_stages : number of FSim interactions (1–3)
    fixed_fsim : optional list of ``n_stages`` (θ, φ) pairs to freeze
    """

    def __init__(self, n_stages: int = 2, fixed_fsim: Optional[Sequence[Tuple[float, float]]] = None):
        if n_stages not in (1, 2, 3):
            raise ValueError(f"n_stages must be 1, 2, or 3, got {n_stages}")
        self.n_stages = int(n_stages)
        if fixed_fsim is not None:
            fixed_fsim = [(float(t), float(p)) for t, p in fixed_fsim]
            if len(fixed_fsim) != self.n_stages:
                raise ValueError(f"fixed_fsim needs {self.n_stages} (θ, φ) pairs, got {len(fixed_fsim)}")
        self.fixed_fsim = fixed_fsim

    @property
    def is_fixed(self) -> bool:
        return self.fixed_fsim is not None

    def num_params(self) -> int:
        n_single = (self.n_stages + 1) * 6
        return n_single + (0 if self.is_fixed else 2 * self.n_stages)

    def _layout(self, params):
        """Yield (('sq', a0,b0,g0,a1,b1,g1) | ('fsim', θ, φ)) in circuit order."""
        idx = 0
        sq = params[idx:idx + 6]; idx += 6
        yield ("sq", sq)
        for s in range(self.n_stages):
            if self.is_fixed:
                yield ("fsim", self.fixed_fsim[s])
            else:
                yield ("fsim", (params[idx], params[idx + 1])); idx += 2
            sq = params[idx:idx + 6]; idx += 6
            yield ("sq", sq)

    def evaluate_unitary_np(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        U = np.eye(4, dtype=np.complex128)
        for kind, val in self._layout(params):
            if kind == "sq":
                layer = np.kron(single_qubit_zxz_np(val[0], val[1], val[2]), single_qubit_zxz_np(val[3], val[4], val[5]))
            else:
                layer = fsim_matrix_np(val[0], val[1])
            U = layer @ U
        return U

    def evaluate_unitary_jax(self, params):
        U = jnp.eye(4, dtype=jnp.complex128)
        for kind, val in self._layout(params):
            if kind == "sq":
                layer = jnp.kron(single_qubit_zxz_jax(val[0], val[1], val[2]), single_qubit_zxz_jax(val[3], val[4], val[5]))
            else:
                layer = fsim_matrix_jax(val[0], val[1])
            U = layer @ U
        return U

    def extract_angles(self, params: np.ndarray) -> Tuple[List[Tuple[float, float]], List[List[Tuple[float, float, float]]]]:
        """(fsim_angles, single_qubit_angles) with fixed angles substituted where frozen."""
        fsim, sq = [], []
        for kind, val in self._layout(np.asarray(params, dtype=float)):
            if kind == "sq":
                sq.append([(float(val[0]), float(val[1]), float(val[2])), (float(val[3]), float(val[4]), float(val[5]))])
            else:
                fsim.append((float(val[0]), float(val[1])))
        return fsim, sq
