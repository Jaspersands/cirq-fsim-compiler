"""
Exact n-gate regions of a fixed two-qubit gate, from the monodromy polytope.

Peterson, Crooks and Smith (Quantum 4, 247 (2020), arXiv:1904.10541) decide
when a class C can be written A · k · B with fixed classes A, B and some
product k of single-qubit gates. This module chains their condition to find
the region that n copies of one native gate reach, with arbitrary
single-qubit gates between them, and from it the minimum count of any fixed
FSim(θ, φ).

Monodromy coordinates. For U in SU(4) let Γ(U) = U (Y⊗Y) Uᵀ (Y⊗Y). Its
eigenvalues are e^{2πi a_j}. Sorted, shifted so that they sum to zero, and
reflected into the half-alcove A_C2 (``rho`` below), the first three
logarithms (a1, a2, a3) label the class, as the Weyl coordinates do. On each
half of the Weyl chamber they are linear in (x, y, z): for z ≥ 0,
a = (x + y − z, x − y + z, −x + y + z)/π, and a class with z < 0 is first
written as (π/2 − x, y, −z). On the face z = 0 the points (x, y, 0) and
(π/2 − x, y, 0) are the same class and both lie in A_C2; ρ maps one to the
other, so the union over ρ below covers both.

The polytope. γ is the class of A k B for some local k exactly when
(α, β, γ) or (α, β, ρ(γ)) satisfies the inequalities

    d − Σ_i α_{k+i−a_i} − Σ_i β_{k+i−b_i} + Σ_i γ_{k+i−c_i} ≥ 0,

one for every nonzero quantum Littlewood–Richardson coefficient N_{ab}^{c,d}
of the Grassmannians Gr(r, 4), with α_4 = −α_1 − α_2 − α_3 (Theorem 23 of the
paper). ρ(γ) = (γ3 + 1/2, γ4 + 1/2, γ1 − 1/2) is the shift that a factor of i
makes in SU(4); a class in PU(4) has both lifts. The table of coefficients
(44 entries, 72 inequalities with the a ↔ b swaps) is copied from Peterson's
``monodromy`` package (``monodromy/static/qlr_table.py``, Apache-2.0), as is
the way an entry becomes an inequality. That package also does exact polytope
arithmetic, but it needs the ``lrs`` program (GPLv2), which is not on PyPI, so
this module does not use it.

n gates. Write c_1 = β for the native gate and c_n = γ for the target. γ is
reachable with n gates exactly when there are classes c_2, ..., c_{n−1} in
A_C2 with every (c_{j−1}, β, c_j) in the polytope. Each step lies in one of
two convex pieces, so for each of the 2^{n−1} choices the question is a
linear feasibility problem in the 3(n − 2) coordinates of the intermediate
classes. :func:`reach_margin` solves it with scipy's HiGHS, maximising the
smallest slack over all inequalities; for n = 2 there is nothing to solve.
The answer is exact up to floating point: a target within about ``tol`` of a
region's boundary may land on either side, as with the closed forms in
:mod:`weyl`.

Controlled-phase-type gates in closed form. For n controlled phases of equal
strength s (CNOT has strength π/4), with arbitrary single-qubit gates between
them, the region is

    x + y + |z| ≤ n s,    −x + y + |z| ≤ (n − 2) s,    |z| ≤ (n − 2) s

for n ≥ 2 (Peterson, Bishop and Javadi-Abhari, Quantum 6, 696 (2022), who
derive it from the monodromy polytope; it is the polytope behind Qiskit's
``XXDecomposer``). FSim(0, φ) is a controlled phase of strength |φ|/4.
FSim(π/2, φ) is SWAP times one of strength π/4 − |φ|/4; an odd number of them
leaves one SWAP, and SWAP† times the class (x, y, z) has (x, y, |z|) equal to
(π/4 − |z|, π/4 − y, π/4 − x). :func:`xx_region_contains` evaluates these
inequalities. The tests check them against the linear programs.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linprog

ArrayLike = Union[np.ndarray, Sequence[float]]

QUARTER_PI = np.pi / 4

#: Quantum Littlewood–Richardson coefficients of Gr(r, 4), as [r, k, a, b, c, d] with k = 4 − r: the relation
#: N_{ab}^{c,d} = 1. Only entries with a ≤ b are listed. Copied from monodromy/static/qlr_table.py (E. Peterson,
#: Apache-2.0).
QLR_TABLE = [
    [1, 3, [0], [0], [0], 0], [1, 3, [0], [1], [1], 0], [1, 3, [0], [2], [2], 0], [1, 3, [0], [3], [3], 0],
    [1, 3, [1], [1], [2], 0], [1, 3, [1], [2], [3], 0], [1, 3, [1], [3], [0], 1], [1, 3, [2], [2], [0], 1],
    [1, 3, [2], [3], [1], 1], [1, 3, [3], [3], [2], 1],
    [2, 2, [0, 0], [0, 0], [0, 0], 0], [2, 2, [0, 0], [1, 0], [1, 0], 0], [2, 2, [0, 0], [1, 1], [1, 1], 0],
    [2, 2, [0, 0], [2, 0], [2, 0], 0], [2, 2, [0, 0], [2, 1], [2, 1], 0], [2, 2, [0, 0], [2, 2], [2, 2], 0],
    [2, 2, [1, 0], [1, 0], [1, 1], 0], [2, 2, [1, 0], [1, 0], [2, 0], 0], [2, 2, [1, 0], [1, 1], [2, 1], 0],
    [2, 2, [1, 0], [2, 0], [2, 1], 0], [2, 2, [1, 0], [2, 1], [2, 2], 0], [2, 2, [1, 0], [2, 1], [0, 0], 1],
    [2, 2, [1, 0], [2, 2], [1, 0], 1], [2, 2, [1, 1], [1, 1], [2, 2], 0], [2, 2, [1, 1], [2, 0], [0, 0], 1],
    [2, 2, [1, 1], [2, 1], [1, 0], 1], [2, 2, [1, 1], [2, 2], [2, 0], 1], [2, 2, [2, 0], [2, 0], [2, 2], 0],
    [2, 2, [2, 0], [2, 1], [1, 0], 1], [2, 2, [2, 0], [2, 2], [1, 1], 1], [2, 2, [2, 1], [2, 1], [2, 0], 1],
    [2, 2, [2, 1], [2, 1], [1, 1], 1], [2, 2, [2, 1], [2, 2], [2, 1], 1], [2, 2, [2, 2], [2, 2], [0, 0], 2],
    [3, 1, [0, 0, 0], [0, 0, 0], [0, 0, 0], 0], [3, 1, [0, 0, 0], [1, 0, 0], [1, 0, 0], 0],
    [3, 1, [0, 0, 0], [1, 1, 0], [1, 1, 0], 0], [3, 1, [0, 0, 0], [1, 1, 1], [1, 1, 1], 0],
    [3, 1, [1, 0, 0], [1, 0, 0], [1, 1, 0], 0], [3, 1, [1, 0, 0], [1, 1, 0], [1, 1, 1], 0],
    [3, 1, [1, 0, 0], [1, 1, 1], [0, 0, 0], 1], [3, 1, [1, 1, 0], [1, 1, 0], [0, 0, 0], 1],
    [3, 1, [1, 1, 0], [1, 1, 1], [1, 0, 0], 1], [3, 1, [1, 1, 1], [1, 1, 1], [1, 1, 0], 1],
]


def _qlr_row(k: int, a, b, c, d) -> list:
    """The inequality of one coefficient, as [d, α1..α3, β1..β3, γ1..γ3]: row · (1, α, β, γ) ≥ 0."""
    row = [d] + [0] * 12
    for i, ai in enumerate(a):
        row[k + i + 1 - ai] -= 1
    for i, bi in enumerate(b):
        row[4 + k + i + 1 - bi] -= 1
    for i, ci in enumerate(c):
        row[8 + k + i + 1 - ci] += 1
    # the fourth coordinate of each point is minus the sum of the first three
    return [row[0], *[v - row[4] for v in row[1:4]], *[v - row[8] for v in row[5:8]], *[v - row[12] for v in row[9:12]]]


def qlr_inequalities() -> np.ndarray:
    """The 72 inequalities of the monodromy polytope, without the alcove constraints."""
    rows = []
    for _, k, a, b, c, d in QLR_TABLE:
        rows.append(_qlr_row(k, a, b, c, d))
        if a != b:
            rows.append(_qlr_row(k, b, a, c, d))
    return np.array(rows, dtype=float)


_QLR = qlr_inequalities()
#: the alcove A_SU(4) in monodromy coordinates: a1 ≥ a2 ≥ a3 ≥ a4 ≥ a1 − 1, rows · (1, a) ≥ 0
_ALCOVE = np.array([[0, 1, -1, 0], [0, 0, 1, -1], [0, 1, 1, 2], [1, -2, -1, -1]], dtype=float)
#: the half-alcove A_C2 (one representative per class in PU(4)): adds a2 + a3 ≥ 0
_ALCOVE_C2 = np.array([[1, -2, -1, -1], [0, 0, 1, 1], [0, 1, -1, 0], [0, 0, 1, -1]], dtype=float)
#: ρ(a) = M a + w, the shift by 1/2 that a factor of i makes in SU(4)
_RHO_M = np.array([[0, 0, 1], [-1, -1, -1], [1, 0, 0]], dtype=float)
_RHO_W = np.array([0.5, 0.5, -0.5])


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def monodromy_from_weyl(x: float, y: float, z: float) -> np.ndarray:
    """Monodromy coordinates (a1, a2, a3) of the canonical class (x, y, z) (Cirq's chamber)."""
    if z < 0:
        x, z = np.pi / 2 - x, -z
    return np.array([x + y - z, x - y + z, -x + y + z]) / np.pi


def weyl_from_monodromy(a1: float, a2: float, a3: float) -> np.ndarray:
    """The inverse of :func:`monodromy_from_weyl`, for a point of A_C2."""
    x, y, z = np.pi * np.array([a1 + a2, a1 + a3, a2 + a3]) / 2
    if x > QUARTER_PI:
        x, z = np.pi / 2 - x, -z
    return np.array([x, y, z])


def monodromy_from_unitary(u: np.ndarray) -> np.ndarray:
    """Monodromy coordinates of a 4×4 unitary, from the spectrum of Γ(U) (the definition, without Cirq)."""
    u = np.asarray(u, dtype=np.complex128)
    u = u * np.linalg.det(u) ** -0.25
    yy = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex128)
    gamma = u @ yy @ u.T @ yy
    a = np.sort(np.mod(np.angle(np.linalg.eigvals(gamma)) / (2 * np.pi), 1.0))[::-1]
    for _ in range(int(np.rint(a.sum()))):        # the logarithms sum to an integer; bring it to zero
        a = np.r_[a[1:], a[0] - 1.0]
    # reflect into A_C2. On its face a2 + a3 = 0 (the Weyl face z = 0) both (x, y, 0) and (π/2 − x, y, 0) lie in
    # A_C2; take the one with x ≤ π/4, as monodromy_from_weyl does.
    face = abs(a[1] + a[2]) <= 1e-9
    if (a[1] + a[2] < 0 and not face) or (face and a[0] + a[1] > 0.5 + 1e-12):
        a = np.array([a[2] + 0.5, a[3] + 0.5, a[0] - 0.5, a[1] - 0.5])
    return a[:3]


def monodromy_coordinates(u_or_coords: ArrayLike) -> np.ndarray:
    """Monodromy coordinates of a 4×4 unitary or of any interaction 3-vector."""
    from .weyl import _coords

    return monodromy_from_weyl(*_coords(u_or_coords))


# --------------------------------------------------------------------------- #
# Linear programs for n gates
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def _programs(native: Tuple[float, float, float], n: int):
    """
    For each choice of convex piece at each of the n − 1 steps, (G, h0, H) with
    the constraints G v + h0 + H γ ≥ s on the intermediate coordinates v and the target γ.
    """
    beta = np.array(native)
    nv = 3 * (n - 2)
    programs = []
    for choice in itertools.product((0, 1), repeat=n - 1):
        G, h0, H = [], [], []
        for j in range(n - 2):                       # intermediate classes lie in A_C2
            g = np.zeros((4, nv))
            g[:, 3 * j:3 * j + 3] = _ALCOVE_C2[:, 1:]
            G.append(g); h0.append(_ALCOVE_C2[:, 0]); H.append(np.zeros((4, 3)))
        for step in range(n - 1):
            # left point: the native at the first step, else intermediate step − 1; right point: the target at the
            # last step, else intermediate step. Each is (variable part, constant part, target part).
            la, lc = np.zeros((3, nv)), np.zeros(3)
            if step == 0:
                lc = beta
            else:
                la[:, 3 * (step - 1):3 * step] = np.eye(3)
            ra, rc, rt = np.zeros((3, nv)), np.zeros(3), np.zeros((3, 3))
            if step == n - 2:
                rt = np.eye(3)
            else:
                ra[:, 3 * step:3 * step + 3] = np.eye(3)
            if choice[step]:
                ra, rc, rt = _RHO_M @ ra, _RHO_M @ rc + _RHO_W, _RHO_M @ rt
            qa, qb, qc = _QLR[:, 1:4], _QLR[:, 4:7], _QLR[:, 7:10]
            G.append(qa @ la + qc @ ra)
            h0.append(_QLR[:, 0] + qa @ lc + qb @ beta + qc @ rc)
            H.append(qc @ rt)
            # the right point, after the shift, lies in the SU(4) alcove (implied by A_C2, kept as in the paper)
            G.append(_ALCOVE[:, 1:] @ ra)
            h0.append(_ALCOVE[:, 0] + _ALCOVE[:, 1:] @ rc)
            H.append(_ALCOVE[:, 1:] @ rt)
        programs.append((np.vstack(G), np.concatenate(h0), np.vstack(H)))
    return programs


def _native_key(native: ArrayLike) -> Tuple[float, float, float]:
    return tuple(float(v) for v in monodromy_coordinates(native))


def reach_margin(target: ArrayLike, native: ArrayLike, n: int) -> float:
    """
    Largest smallest slack, over the 2^{n−1} linear programs, of the
    inequalities for "``target`` is made by n copies of ``native``" (n ≥ 2).
    The target is in the n-gate region when this is ≥ 0 (in practice ≥ −tol).
    ``target`` and ``native`` are 4×4 unitaries or Weyl coordinates.
    """
    if n < 2:
        raise ValueError("reach_margin needs n >= 2; zero and one gate are single points")
    gamma = monodromy_coordinates(target)
    nv = 3 * (n - 2)
    best = -np.inf
    for G, h0, H in _programs(_native_key(native), n):
        h = h0 + H @ gamma
        if nv == 0:
            best = max(best, float(h.min()))
        else:
            # maximise s subject to G v + h ≥ s
            res = linprog(np.r_[np.zeros(nv), -1.0], A_ub=np.c_[-G, np.ones(len(h))], b_ub=h,
                          bounds=[(-1.0, 1.0)] * nv + [(None, 1.0)], method="highs")
            if res.status == 0:
                best = max(best, float(res.x[-1]))
        if best >= 0.0:
            break
    return best


def reachable(target: ArrayLike, native: ArrayLike, n: int, tol: float = 1e-9) -> bool:
    """True when n copies of ``native``, with single-qubit gates between them, can make ``target``."""
    t, b = monodromy_coordinates(target), monodromy_coordinates(native)
    if n == 0:
        return bool(np.max(np.abs(t)) <= tol)
    if n == 1:
        return bool(np.max(np.abs(t - b)) <= tol)
    return reach_margin(target, native, n) >= -tol


def min_count(target: ArrayLike, native: ArrayLike, max_count: int = 6, start: int = 0,
              tol: float = 1e-9) -> Optional[int]:
    """Smallest n in [start, max_count] with ``target`` in the n-gate region of ``native``, or ``None``."""
    for n in range(int(start), int(max_count) + 1):
        if reachable(target, native, n, tol):
            return n
    return None


# --------------------------------------------------------------------------- #
# Controlled-phase-type gates in closed form
# --------------------------------------------------------------------------- #
def xx_region_contains(coords: ArrayLike, n: int, strength: float, swap: bool = False, tol: float = 1e-9) -> bool:
    """
    True when the class ``coords`` is made by n controlled phases of strength
    ``strength`` with arbitrary single-qubit gates between them; with
    ``swap=True``, by SWAP times such a product (an odd number of FSim(π/2, φ)).
    """
    from .weyl import _coords

    x, y, z = _coords(coords)
    if swap:
        x, y, z = QUARTER_PI - abs(z), QUARTER_PI - y, QUARTER_PI - x
    z = abs(z)
    total = n * strength
    largest, second = (strength if n >= 1 else 0.0), (strength if n >= 2 else 0.0)
    return bool(x + y + z <= total + tol and -x + y + z <= total - 2 * largest + tol
                and z <= total - largest - second + tol)
