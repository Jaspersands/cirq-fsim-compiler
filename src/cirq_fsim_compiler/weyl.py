"""
Weyl-chamber analysis of two-qubit gates.

Every U in U(4) can be written

    U = g · (A1 ⊗ A0) · exp(i(x XX + y YY + z ZZ)) · (B1 ⊗ B0)

(``cirq.kak_decomposition``). The canonical interaction coefficients, with
π/4 ≥ x ≥ y ≥ |z| and z ≥ 0 when x = π/4, label the local-equivalence class:
two gates differ only by single-qubit gates exactly when their coordinates
agree. This module uses them for four things.

1. Closed-form minimum native-gate counts where they are known:

   * CZ/CNOT-class gates: 0 if local, 1 for (π/4, 0, 0), 2 if z = 0, else 3
     (Vidal and Dawson 2004; Shende, Markov and Bullock 2004).
   * √iSWAP: 0 if local, 1 for (π/8, π/8, 0), 2 if x − y ≥ |z|, else 3
     (Huang et al., PRL 130, 070601 (2023), arXiv:2105.06074; the two-gate
     region holds 79% of the Haar measure).
   * Free-angle FSim: 0 if local, 1 if two coordinates have equal magnitude
     (x = y or y = |z|), else 2. FSim(θ, φ) has interaction vector
     (−θ/2, −θ/2, −φ/4) up to single-qubit Z rotations, which gives the
     one-gate set. Two gates always suffice: conjugating the second FSim by
     H⊗H swaps XX and ZZ, and since XX, YY and ZZ commute the product is
     exp(i((a + c′) XX + (a + a′) YY + (c + a′) ZZ)), which reaches any
     (x, y, z) by choosing a, c, a′, c′.

2. The best process fidelity that zero or one copy of a native gate can reach,
   in closed form (:func:`class_fidelity`). The tests check it against
   numerical optimisation.

3. A prediction of the stage count before numerical synthesis
   (:func:`predict_stages`). The synthesiser uses it to skip stage counts
   that cannot succeed. For a native gate outside the families above, the
   prediction is only a lower bound. It uses the closed forms for zero and
   one gate, plus two rules for controlled-phase-type natives: FSim(0, φ),
   and FSim(π/2, φ), which is SWAP times a controlled phase of strength s
   (canonical coordinates (s, 0, 0)). Sycamore's FSim(π/2, π/6) is SWAP
   times a controlled phase with s = 5π/24.

   * Two such gates reach only the face z = 0. Proof: write each gate as
     SWAP·D with D diagonal (a controlled phase times Z rotations). The key
     step is that SWAP·K·SWAP is local for every local K, since it only
     exchanges the tensor factors. Then SWAP·D·K·SWAP·D′ =
     (SWAP·D·SWAP)·(SWAP·K·SWAP)·D′, and SWAP·D·SWAP is again diagonal, so
     two gates are two controlled phases with a local layer between. For
     those, tr γ(U) is real with γ(U) = U (Y⊗Y) Uᵀ (Y⊗Y). That is the
     Shende–Markov–Bullock criterion for two CNOTs, so z = 0.
   * Sum of strengths. A product of controlled phases of strengths s_1, …,
     s_n, with any single-qubit gates between them, has x + y + |z| ≤ Σ s_i.
     Proof sketch: in the magic basis m(U) = U_Bᵀ U_B, and for a local
     product the spectrum of m is that of a product of unitaries conjugate
     to m(e^{i s_i ZZ}), whose eigenphases are ±2 s_i. By Thompson's theorem
     (R. C. Thompson, Linear Multilinear Algebra 19, 187 (1986)),
     e^{iA} e^{iB} = e^{i(VAV† + WBW†)}, so the product is e^{iL} with
     ‖L‖ ≤ 2 Σ s_i and tr L = 0. The eigenphases of m for the canonical
     point are 2(x − y + z), 2(−x + y + z), 2(x + y − z), 2(−x − y − z), and
     the smallest largest-magnitude choice among equivalent points is
     2(x + y + |z|). SWAP times such a product (an odd number of
     FSim(π/2, φ)) maps the class (x, y, z) to one with sum 3π/4 − (x + y + |z|),
     so n FSim(π/2, φ) with n odd need x + y + |z| ≥ 3π/4 − n·s.
     For Sycamore: two gates reach only z = 0 with x + y ≤ 5π/12, which
     excludes iSWAP (x + y = π/2), and three gates need x + y + |z| ≥ π/8.
     The tests check both bounds against random single-qubit layers.

4. A numerical minimum for an arbitrary fixed FSim(θ, φ)
   (:func:`numerical_min_count`): multi-start synthesis at increasing stage
   counts with a strict tolerance. This is numerical evidence, not a proof.
   A failure means many restarts all failed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cirq

    HAS_CIRQ = True
except ImportError:  # pragma: no cover
    HAS_CIRQ = False
    cirq = None

ArrayLike = Union[np.ndarray, Sequence[float]]

QUARTER_PI = np.pi / 4
EIGHTH_PI = np.pi / 8

#: canonical coordinates of the named gate classes
CNOT_CLASS = np.array([QUARTER_PI, 0.0, 0.0])
SQRT_ISWAP_CLASS = np.array([EIGHTH_PI, EIGHTH_PI, 0.0])
ISWAP_CLASS = np.array([QUARTER_PI, QUARTER_PI, 0.0])
SWAP_CLASS = np.array([QUARTER_PI, QUARTER_PI, QUARTER_PI])


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def weyl_coordinates(u: np.ndarray) -> np.ndarray:
    """Canonical (x, y, z) of a 4×4 unitary from ``cirq.kak_decomposition``."""
    if not HAS_CIRQ:  # pragma: no cover
        raise RuntimeError("Cirq required.")
    u = np.asarray(u, dtype=np.complex128)
    if u.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got {u.shape}")
    return np.array(cirq.kak_decomposition(u).interaction_coefficients, dtype=float)


def canonicalize(x: float, y: float, z: float) -> np.ndarray:
    """Map any interaction vector into the Weyl chamber (Cirq's convention)."""
    return np.array(cirq.kak_canonicalize_vector(float(x), float(y), float(z)).interaction_coefficients, dtype=float)


def fsim_weyl_coordinates(theta: float, phi: float) -> np.ndarray:
    """Canonical coordinates of FSim(θ, φ): the class of (−θ/2, −θ/2, −φ/4)."""
    return canonicalize(-0.5 * theta, -0.5 * theta, -0.25 * phi)


def _coords(u_or_coords: ArrayLike) -> np.ndarray:
    """Canonical coordinates of a 4×4 unitary, or of any interaction 3-vector."""
    a = np.asarray(u_or_coords)
    if a.shape == (3,):
        return canonicalize(*np.real(a).astype(float))
    return weyl_coordinates(a)


def interaction_unitary(x: float, y: float, z: float) -> np.ndarray:
    """exp(i(x XX + y YY + z ZZ)), the representative of the class (x, y, z)."""
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)
    h = sum(c * np.kron(p, p) for c, p in zip((x, y, z), pauli))
    w, v = np.linalg.eigh(h)
    return (v * np.exp(1j * w)) @ v.conj().T


def _as_unitary(u_or_coords: ArrayLike) -> np.ndarray:
    a = np.asarray(u_or_coords)
    if a.shape == (3,):
        return interaction_unitary(*np.real(a).astype(float))
    return np.asarray(a, dtype=np.complex128)


# --------------------------------------------------------------------------- #
# Closed-form class fidelity
# --------------------------------------------------------------------------- #
_PERMS = list(itertools.permutations(range(3)))
_EVEN_SIGNS = np.array([s for s in itertools.product((1.0, -1.0), repeat=3) if np.prod(s) > 0])
# |Tr exp(i c·(XX, YY, ZZ))|² is π-periodic in each coordinate, so π/2 shifts mod π suffice.
_SHIFTS = np.array(list(itertools.product((0.0, 1.0), repeat=3))) * (np.pi / 2)


def _weyl_images(b: np.ndarray) -> np.ndarray:
    """Points equivalent to ``b`` under permutations, paired sign flips and π/2 shifts."""
    signed = np.array([s * b[list(p)] for p in _PERMS for s in _EVEN_SIGNS])
    return (signed[:, None, :] + _SHIFTS[None, :, :]).reshape(-1, 3)


def class_fidelity(target: ArrayLike, native: ArrayLike) -> float:
    """
    Best process fidelity max |Tr(U† K1 V K2)|²/16 over single-qubit layers K1, K2.

    ``target`` and ``native`` are 4×4 unitaries or canonical coordinates. The
    value is max over Weyl images c of cos²c_x cos²c_y cos²c_z + sin²c_x sin²c_y sin²c_z,
    with c the difference of the two coordinate vectors, since
    Tr exp(i(c_x XX + c_y YY + c_z ZZ)) = 4(cos c_x cos c_y cos c_z + i sin c_x sin c_y sin c_z).
    With ``native = (0, 0, 0)`` it is the best fidelity of a product of
    single-qubit gates. The tests compare it with numerical optimisation.
    """
    a, b = _coords(target), _coords(native)
    c = _weyl_images(b) - a
    f = np.prod(np.cos(c), axis=1) ** 2 + np.prod(np.sin(c), axis=1) ** 2
    return float(min(1.0, f.max()))


# --------------------------------------------------------------------------- #
# Analytic minimum counts
# --------------------------------------------------------------------------- #
def is_local(u_or_coords: ArrayLike, atol: float = 1e-8) -> bool:
    return bool(np.all(np.abs(_coords(u_or_coords)) <= atol))


def min_cz_count(u_or_coords: ArrayLike, atol: float = 1e-8) -> int:
    """Minimum number of CZ (or CNOT) gates: 0, 1, 2 or 3."""
    a = _coords(u_or_coords)
    x, y, z = a
    if np.all(np.abs(a) <= atol):
        return 0
    if np.all(np.abs(a - CNOT_CLASS) <= atol):
        return 1
    if abs(z) <= atol:
        return 2
    return 3


def min_sqrt_iswap_count(u_or_coords: ArrayLike, atol: float = 1e-8) -> int:
    """Minimum number of √iSWAP gates: 0, 1, 2 (x − y ≥ |z|) or 3."""
    a = _coords(u_or_coords)
    x, y, z = a
    if np.all(np.abs(a) <= atol):
        return 0
    if np.all(np.abs(a - SQRT_ISWAP_CLASS) <= atol):
        return 1
    if x - y + atol >= abs(z):
        return 2
    return 3


def min_free_fsim_count(u_or_coords: ArrayLike, atol: float = 1e-8) -> int:
    """Minimum number of FSim(θ, φ) gates with free angles: 0, 1 (x = y or y = |z|) or 2."""
    a = _coords(u_or_coords)
    x, y, z = a
    if np.all(np.abs(a) <= atol):
        return 0
    if abs(x - y) <= atol or abs(y - abs(z)) <= atol:
        return 1
    return 2


def native_gate_family(theta: float, phi: float, atol: float = 1e-9) -> str:
    """
    Classify a native FSim(θ, φ) by its local-equivalence class.

    ``"cz"`` (CNOT class), ``"sqrt_iswap"``, ``"cphase"`` (a partial
    controlled phase, coordinates (c, 0, 0)), ``"swap_cphase"`` (FSim(π/2, φ):
    SWAP times a controlled phase, coordinates (π/4, π/4, c), which includes
    Sycamore, iSWAP and SWAP), ``"identity"`` or ``"generic"``.
    """
    c = fsim_weyl_coordinates(theta, phi)
    if np.all(np.abs(c) <= atol):
        return "identity"
    if np.all(np.abs(c - CNOT_CLASS) <= atol):
        return "cz"
    if np.all(np.abs(c - SQRT_ISWAP_CLASS) <= atol):
        return "sqrt_iswap"
    if abs(c[1]) <= atol and abs(c[2]) <= atol:
        return "cphase"
    if abs(c[0] - QUARTER_PI) <= atol and abs(c[1] - QUARTER_PI) <= atol:
        return "swap_cphase"
    return "generic"


def cphase_strength(theta: float, phi: float) -> Optional[float]:
    """
    Strength s of the controlled phase inside FSim(0, φ) (the class (s, 0, 0))
    or FSim(π/2, φ) (SWAP times the class (s, 0, 0), coordinates
    (π/4, π/4, π/4 − s)). ``None`` for other natives.
    """
    family = native_gate_family(theta, phi)
    c = fsim_weyl_coordinates(theta, phi)
    if family in ("cz", "cphase"):
        return float(c[0])
    if family == "swap_cphase":
        return float(QUARTER_PI - abs(c[2]))
    return None


def cphase_count_excluded(u_or_coords: ArrayLike, n: int, theta: float, phi: float, tol: float = 1e-8) -> bool:
    """
    True when n copies of the controlled-phase-type native FSim(0, φ) or
    FSim(π/2, φ) provably cannot make the target (module docstring, item 3).
    False means only that these rules do not exclude n.
    """
    s = cphase_strength(theta, phi)
    if s is None or n < 1:
        return False
    x, y, z = _coords(u_or_coords)
    total = x + y + abs(z)
    if n == 2 and abs(z) > tol:
        return True
    swap_odd = native_gate_family(theta, phi) == "swap_cphase" and n % 2 == 1
    if swap_odd:
        return total < 3 * QUARTER_PI - n * s - tol
    return total > n * s + tol


def analytic_min_count(u_or_coords: ArrayLike, fsim_angles: Optional[Tuple[float, float]] = None,
                       atol: float = 1e-8) -> Optional[int]:
    """
    Closed-form minimum for free FSim (``fsim_angles=None``), CZ-class and
    √iSWAP-class native gates, and for targets that need zero or one native
    gate. ``None`` when no closed form is known; use :func:`numerical_min_count`.
    """
    pred = predict_stages(u_or_coords, fsim_angles=fsim_angles, tol=atol)
    return pred.exact


# --------------------------------------------------------------------------- #
# Stage prediction
# --------------------------------------------------------------------------- #
@dataclass
class StagePrediction:
    """What the Weyl coordinates say about the number of native gates."""

    lower: int                  # no decomposition with fewer native gates exists
    exact: Optional[int]        # the minimum, when it is known in closed form
    family: str                 # "free" or the native gate family
    coordinates: np.ndarray


def predict_stages(u_or_coords: ArrayLike, fsim_angles: Optional[Tuple[float, float]] = None,
                   tol: float = 1e-8) -> StagePrediction:
    """
    Lower bound and, where known, exact minimum number of native gates.

    ``fsim_angles=None`` means free-angle FSim; otherwise the fixed native
    (θ, φ). ``tol`` is a tolerance on the coordinates: a target within
    ``tol`` of a region counts as inside it. The zero- and one-gate tests use
    the closed-form infidelity instead and accept up to ``tol²``. Synthesis
    to a process infidelity ε uses tol = √ε, because the infidelity grows as
    the squared coordinate distance.
    """
    a = _coords(u_or_coords)
    inf_tol = max(tol * tol, 1e-12)
    if 1.0 - class_fidelity(a, np.zeros(3)) <= inf_tol:
        family = "free" if fsim_angles is None else native_gate_family(*fsim_angles)
        return StagePrediction(lower=0, exact=0, family=family, coordinates=a)
    if fsim_angles is None:
        n = max(1, min_free_fsim_count(a, tol))
        return StagePrediction(lower=n, exact=n, family="free", coordinates=a)
    theta, phi = fsim_angles
    family = native_gate_family(theta, phi)
    native = fsim_weyl_coordinates(theta, phi)
    one = 1.0 - class_fidelity(a, native) <= inf_tol
    if family in ("cz", "sqrt_iswap"):
        count = min_cz_count if family == "cz" else min_sqrt_iswap_count
        n = 1 if one else max(2, count(a, tol))
        return StagePrediction(lower=n, exact=n, family=family, coordinates=a)
    if family == "identity":
        return StagePrediction(lower=1, exact=None, family=family, coordinates=a)
    if one:
        return StagePrediction(lower=1, exact=1, family=family, coordinates=a)
    lower = 2
    if family in ("cphase", "swap_cphase"):
        while lower < 8 and cphase_count_excluded(a, lower, theta, phi, tol):
            lower += 1
    return StagePrediction(lower=lower, exact=None, family=family, coordinates=a)


# --------------------------------------------------------------------------- #
# Numerical minimum for an arbitrary fixed FSim
# --------------------------------------------------------------------------- #
def numerical_min_count(u: ArrayLike, theta: float, phi: float, max_stages: int = 4, n_restarts: int = 16,
                        max_iter: int = 500, tol: float = 1e-9, seed: int = 0, use_weyl: bool = True) -> Optional[int]:
    """
    Smallest n ≤ ``max_stages`` for which multi-start synthesis with the fixed
    FSim(θ, φ) reaches process infidelity ≤ ``tol``, or ``None``.

    This is numerical. With ``use_weyl=True``, n = 0 and n = 1 are decided by
    the closed forms, and n = 2 is skipped when the two-gate rule for
    controlled-phase-type gates excludes it. All other stage counts are
    decided by ``n_restarts`` L-BFGS runs, and a count is rejected only when
    every run ends above ``tol``.
    """
    from .riemannian_optimizer import DifferentiableFSimSynthesizer
    from .unitary_ansatz import check_max_stages

    max_stages = check_max_stages(max_stages)
    u = _as_unitary(u)
    start = 0
    if use_weyl:
        pred = predict_stages(u, fsim_angles=(theta, phi), tol=np.sqrt(tol))
        if pred.exact is not None and pred.exact <= 1:
            return pred.exact
        start = pred.lower
    syn = DifferentiableFSimSynthesizer(target_infidelity=tol, seed=seed, weyl_pruning=False)
    for n in range(start, int(max_stages) + 1):
        fixed = [(float(theta), float(phi))] * n
        res = syn._optimize_stage(u, n, n_restarts=n_restarts, max_iter=max_iter, fixed=fixed)
        if res.infidelity <= tol:
            return n
    return None


def minimum_count(u: ArrayLike, fsim_angles: Optional[Tuple[float, float]] = None, max_stages: int = 4,
                  tol: float = 1e-9, **kwargs) -> Tuple[Optional[int], str]:
    """
    Minimum native-gate count and how it is known: ``"closed form"``,
    ``"proven"`` (the search met a proven lower bound), ``"numerical"``
    (the search found it above every proven bound, so a smaller count is
    excluded only numerically) or ``"not found"``. ``u`` may be a 4×4 unitary
    or Weyl coordinates.
    """
    pred = predict_stages(u, fsim_angles=fsim_angles, tol=np.sqrt(tol))
    if pred.exact is not None:
        return pred.exact, "closed form"
    n = numerical_min_count(u, *fsim_angles, max_stages=max_stages, tol=tol, **kwargs)
    if n is None:
        return None, "not found"
    return n, ("proven" if n == pred.lower else "numerical")
