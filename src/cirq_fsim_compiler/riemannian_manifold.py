"""
Riemannian gradient descent on the product manifold U(2)^{2(n+1)} × T^{2n}.

The single-qubit dressings V_j are kept *as unitaries*. At each step the loss
is differentiated with respect to Lie-algebra coordinates a ∈ ℝ⁴ of a
perturbation exp(A(a)) V_j, A(a) = i(a₀ 1 + a₁X + a₂Y + a₃Z); the resulting
Riemannian gradient is applied with the Cayley retraction

    V ← (1 − ½η A)⁻¹ (1 + ½η A) V,

which maps anti-Hermitian A to an exact unitary, so every iterate is unitary
to machine precision. FSim angles (when free) take Euclidean steps. Step size
is adapted by an Armijo backtracking rule. This is the manifold counterpart of
the Euler-angle chart used by the L-BFGS solver; both minimise
1 − |Tr(U_t† U)|²/16 and agree on the optimum.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from .unitary_ansatz import fsim_matrix_np, euler_zxz_from_unitary, haar_random_unitary

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False
    jax = None
    jnp = np

_PAULI = np.array([np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)


def lie_element(a: np.ndarray) -> np.ndarray:
    """A(a) = i Σ_k a_k σ_k (anti-Hermitian)."""
    return 1j * np.tensordot(np.asarray(a, dtype=float), _PAULI, axes=(0, 0))


def cayley_retract(V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """(1 − A/2)⁻¹ (1 + A/2) V for anti-Hermitian A — exactly unitary."""
    I = np.eye(V.shape[0])
    return np.linalg.solve(I - 0.5 * A, (I + 0.5 * A) @ V)


def _fsim_jax(theta, phi):
    c, s = jnp.cos(theta), jnp.sin(theta)
    z = jnp.zeros((), dtype=jnp.complex128); one = jnp.ones((), dtype=jnp.complex128)
    return jnp.array([[one, z, z, z], [z, c + 0j, -1j * s, z], [z, -1j * s, c + 0j, z], [z, z, z, jnp.exp(-1j * phi)]])


class RiemannianFSimSolver:
    """
    Parameters
    ----------
    n_stages : FSim stages (1–3)
    fixed_fsim : freeze the FSim angles (calibrated mode) or None for free angles
    step : initial step size
    max_iter : iterations per restart
    """

    def __init__(self, n_stages: int, fixed_fsim: Optional[Sequence[Tuple[float, float]]] = None,
                 step: float = 0.3, max_iter: int = 600, tol: float = 1e-14, seed: int = 0):
        if n_stages not in (1, 2, 3):
            raise ValueError("n_stages must be 1, 2 or 3")
        self.n_stages = int(n_stages)
        self.fixed = None if fixed_fsim is None else [(float(t), float(p)) for t, p in fixed_fsim]
        self.step = float(step)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)
        self.n_blocks = 2 * (self.n_stages + 1)
        self._grad = self._build_jax() if HAS_JAX else None

    # ----------------------------------------------------------------- #
    def unitary(self, blocks: List[np.ndarray], angles: np.ndarray) -> np.ndarray:
        U = np.kron(blocks[0], blocks[1])
        for s in range(self.n_stages):
            th, ph = self._angles(angles, s)
            U = fsim_matrix_np(th, ph) @ U
            U = np.kron(blocks[2 * s + 2], blocks[2 * s + 3]) @ U
        return U

    def _angles(self, angles, s):
        if self.fixed is not None:
            return self.fixed[s]
        return angles[2 * s], angles[2 * s + 1]

    @staticmethod
    def loss(U: np.ndarray, target: np.ndarray) -> float:
        return float(1.0 - abs(np.trace(target.conj().T @ U)) ** 2 / 16.0)

    # ----------------------------------------------------------------- #
    def _build_jax(self):
        n_stages, n_blocks, fixed = self.n_stages, self.n_blocks, self.fixed
        paulis = jnp.asarray(_PAULI)

        def loss_at(x, blocks, angles, target):
            # x = [a_0 (4), ..., a_{n_blocks-1} (4), δθ_1, δφ_1, ...]
            Vs = []
            for j in range(n_blocks):
                a = x[4 * j:4 * j + 4]
                A = 1j * jnp.tensordot(a, paulis, axes=(0, 0))
                Vs.append(jax.scipy.linalg.expm(A) @ blocks[j])
            U = jnp.kron(Vs[0], Vs[1])
            for s in range(n_stages):
                if fixed is not None:
                    th, ph = fixed[s]
                else:
                    th = angles[2 * s] + x[4 * n_blocks + 2 * s]
                    ph = angles[2 * s + 1] + x[4 * n_blocks + 2 * s + 1]
                U = _fsim_jax(th, ph) @ U
                U = jnp.kron(Vs[2 * s + 2], Vs[2 * s + 3]) @ U
            ov = jnp.trace(jnp.conj(target).T @ U)
            return 1.0 - jnp.real(ov * jnp.conj(ov)) / 16.0

        return jax.jit(jax.grad(loss_at))

    def gradient(self, blocks, angles, target) -> np.ndarray:
        """Riemannian gradient in Lie-algebra coordinates (+ Euclidean angle gradient)."""
        n_free = 0 if self.fixed is not None else 2 * self.n_stages
        n_x = 4 * self.n_blocks + n_free
        if self._grad is not None:
            g = self._grad(jnp.zeros(n_x), [jnp.asarray(b) for b in blocks], jnp.asarray(angles, dtype=float), jnp.asarray(target))
            return np.asarray(g, dtype=float)
        # finite differences
        g = np.zeros(n_x); eps = 1e-6
        for k in range(n_x):
            for sign in (+1, -1):
                x = np.zeros(n_x); x[k] = sign * eps
                blk = [cayley_retract(blocks[j], lie_element(x[4 * j:4 * j + 4])) for j in range(self.n_blocks)]
                ang = np.array(angles, dtype=float)
                if n_free:
                    ang = ang + x[4 * self.n_blocks:]
                g[k] += sign * self.loss(self.unitary(blk, ang), target) / (2 * eps)
        return g

    def _apply(self, blocks, angles, direction, eta):
        new_blocks = [cayley_retract(blocks[j], lie_element(eta * direction[4 * j:4 * j + 4])) for j in range(self.n_blocks)]
        new_angles = np.array(angles, dtype=float)
        if self.fixed is None:
            new_angles = new_angles + eta * direction[4 * self.n_blocks:]
        return new_blocks, new_angles

    # ----------------------------------------------------------------- #
    def run(self, target: np.ndarray, blocks: List[np.ndarray], angles: np.ndarray):
        eta = self.step
        cur = self.loss(self.unitary(blocks, angles), target)
        history = [cur]
        for _ in range(self.max_iter):
            g = self.gradient(blocks, angles, target)
            gnorm2 = float(g @ g)
            if gnorm2 < 1e-30:
                break
            d = -g
            accepted = False
            for _ls in range(30):
                nb, na = self._apply(blocks, angles, d, eta)
                new = self.loss(self.unitary(nb, na), target)
                if new <= cur - 1e-4 * eta * gnorm2:
                    accepted = True
                    break
                eta *= 0.5
            if not accepted:
                break
            blocks, angles, cur = nb, na, new
            history.append(cur)
            eta = min(eta * 1.3, 2.0)
            if cur < self.tol:
                break
        return blocks, angles, cur, history

    def solve(self, target: np.ndarray, n_restarts: int = 4, target_infidelity: float = 1e-8):
        """Multi-start solve; returns a :class:`DecompositionResult`."""
        from .riemannian_optimizer import DecompositionResult, SEED_ANGLES

        target = np.asarray(target, dtype=np.complex128)
        best = None
        for r in range(int(n_restarts)):
            blocks = [haar_random_unitary(2, seed=int(self.rng.integers(1 << 30))) for _ in range(self.n_blocks)]
            if self.fixed is None:
                base = SEED_ANGLES[r % len(SEED_ANGLES)] if r < len(SEED_ANGLES) else tuple(self.rng.uniform(-np.pi, np.pi, 2))
                angles = np.array(list(base) * self.n_stages, dtype=float)
            else:
                angles = np.zeros(0)
            blocks, angles, val, hist = self.run(target, blocks, angles)
            if best is None or val < best[2]:
                best = (blocks, angles, val, hist, r + 1)
            if val < target_infidelity:
                break
        blocks, angles, val, hist, used = best
        U = self.unitary(blocks, angles)
        fsim_angles = [tuple(map(float, self._angles(angles, s))) for s in range(self.n_stages)]
        sq = []
        for k in range(self.n_stages + 1):
            sq.append([euler_zxz_from_unitary(blocks[2 * k]), euler_zxz_from_unitary(blocks[2 * k + 1])])
        params = np.concatenate([np.array(sq[0]).ravel()] + [np.concatenate([np.asarray(fsim_angles[s]) if self.fixed is None else np.zeros(0), np.array(sq[s + 1]).ravel()]) for s in range(self.n_stages)])
        fid = 1.0 - val
        return DecompositionResult(
            n_stages=self.n_stages, infidelity=float(max(0.0, val)), fidelity=float(min(1.0, fid)), optimal_params=params,
            fsim_angles=fsim_angles, single_qubit_angles=sq, synthesized_unitary=U, target_unitary=target,
            is_success=bool(val <= target_infidelity * 10), native=self.fixed is not None, method="riemannian",
            loss_history=hist, n_restarts_used=used,
        )
