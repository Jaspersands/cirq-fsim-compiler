"""
Tests for the exact n-gate regions from the monodromy polytope: coordinates,
the closed forms they must reproduce (CZ, √iSWAP, Sycamore), the closed-form
controlled-phase regions against the linear programs, generic natives against
multi-start synthesis, and the Haar measure of the Sycamore four-gate region.
"""

import numpy as np
import pytest
from scipy import integrate

cirq = pytest.importorskip("cirq")

from cirq_fsim_compiler import monodromy, weyl  # noqa: E402
from cirq_fsim_compiler.calibration_map import CouplerCalibration  # noqa: E402
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer  # noqa: E402
from cirq_fsim_compiler.unitary_ansatz import fsim_matrix_np, haar_random_unitary  # noqa: E402

P4, P8 = np.pi / 4, np.pi / 8
SYC = CouplerCalibration.sycamore().angles
DRIFTED = (SYC[0] + 0.02, SYC[1] - 0.01)
#: the calibrated gate of the pair (3,3)-(3,4) in the released supremacy circuits, as quoted by Kalai, Rinott and
#: Shoham (arXiv:2210.12753)
PAIR_33_34 = (1.2947043217999283, 0.4859467238431821)


def local(seed):
    return np.kron(haar_random_unitary(2, seed=seed), haar_random_unitary(2, seed=seed + 1))


def chamber_points(m, seed, face=0):
    """Random canonical classes; ``face`` more on the face z = 0."""
    rng = np.random.default_rng(seed)
    pts = [weyl.canonicalize(*rng.uniform(-np.pi / 2, np.pi / 2, 3)) for _ in range(m)]
    pts += [weyl.canonicalize(*rng.uniform(-np.pi / 2, np.pi / 2, 2), 0.0) for _ in range(face)]
    return pts


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gate,expected", [
    (cirq.CNOT, (0.25, 0.25, -0.25)), (cirq.ISWAP, (0.5, 0, 0)), (cirq.SWAP, (0.25, 0.25, 0.25)),
    (cirq.IdentityGate(2), (0, 0, 0)), (cirq.SQRT_ISWAP, (0.25, 0, 0)),
], ids=["CNOT", "iSWAP", "SWAP", "I", "√iSWAP"])
def test_monodromy_coordinates_of_known_gates(gate, expected):
    u = cirq.unitary(gate)
    assert np.allclose(monodromy.monodromy_coordinates(u), expected, atol=1e-12)
    assert np.allclose(monodromy.monodromy_from_unitary(local(5) @ u @ local(9)), expected, atol=1e-9)


def test_linear_formula_matches_the_spectrum_definition():
    # Weyl coordinates through the linear formula against the eigenvalues of Γ(U), with no Cirq in between
    mats = [haar_random_unitary(4, seed=s) for s in range(40)]
    mats += [weyl.interaction_unitary(*c) for c in [(P4, 0.3, 0.3), (P4, 0.3, 0.0), (0.5, 0.5, -0.2), (0.6, 0.3, -0.3),
                                                    (0.6, 0.3, 0.3), (P4, P4, 0.1), (0.4, 0.0, 0.0)]]
    for u in mats:
        a = monodromy.monodromy_coordinates(u)
        assert np.allclose(a, monodromy.monodromy_from_unitary(u), atol=1e-9)
        assert np.allclose(monodromy.weyl_from_monodromy(*a), weyl.weyl_coordinates(u), atol=1e-12)
    assert len(monodromy.qlr_inequalities()) == 72


# --------------------------------------------------------------------------- #
# the regions reproduce the known closed forms
# --------------------------------------------------------------------------- #
def test_regions_reproduce_the_cz_and_sqrt_iswap_rules():
    cz, sq = cirq.unitary(cirq.CZ), cirq.unitary(cirq.SQRT_ISWAP)
    for c in chamber_points(80, seed=1, face=20):
        x, y, z = c
        assert monodromy.reachable(c, cz, 2) == (weyl.min_cz_count(c) <= 2)              # z = 0
        assert monodromy.reachable(c, sq, 2) == (x - y >= abs(z) - 1e-9)                # Huang et al. 2023
        assert monodromy.reachable(c, cz, 3) and monodromy.reachable(c, sq, 3)


def test_sycamore_regions_from_the_linear_programs():
    # two gates: the face z = 0 with x + y ≤ 5π/12; three: x + y + |z| ≥ π/8; four: everything
    syc = fsim_matrix_np(*SYC)
    for c in chamber_points(80, seed=2, face=20):
        x, y, z = c
        assert monodromy.reachable(c, syc, 2) == (abs(z) < 1e-9 and x + y <= 5 * np.pi / 12)
        assert monodromy.reachable(c, syc, 3) == (x + y + abs(z) >= P8)
        assert monodromy.reachable(c, syc, 4)
    edge = (np.pi / 24,) * 3                                       # on the plane x + y + |z| = π/8
    assert monodromy.reachable(edge, syc, 3)
    assert monodromy.reach_margin((np.pi / 24, np.pi / 24, np.pi / 24 - 1e-3), syc, 3) < -1e-4


@pytest.mark.parametrize("native", [(0.0, 1.0), (0.0, -2.2), (np.pi / 2, np.pi / 6), (np.pi / 2, 2.0), (np.pi / 2, -0.7)],
                         ids=["CPhase(1)", "CPhase(-2.2)", "Sycamore", "SWAP·CPhase(2)", "SWAP·CPhase(-0.7)"])
def test_closed_form_controlled_phase_regions_match_the_linear_programs(native):
    s = weyl.cphase_strength(*native)
    swap = weyl.native_gate_family(*native) == "swap_cphase"
    u = fsim_matrix_np(*native)
    for n in (2, 3, 4):
        for c in chamber_points(30, seed=10 + n, face=6):
            closed = monodromy.xx_region_contains(c, n, s, swap=swap and n % 2 == 1)
            assert closed == monodromy.reachable(c, u, n, tol=1e-8), (n, c)


# --------------------------------------------------------------------------- #
# Sycamore: which classes need four gates
# --------------------------------------------------------------------------- #
def test_sycamore_minimum_counts_from_the_regions():
    for coords, n in [((0.1, 0.05, 0.02), 4), ((np.pi / 24,) * 3, 3), ((0.2, 0.15, 0.05), 3), ((0.2, 0.14, 0.05), 4),
                      ((0.2, 0.15, 0.0), 2), ((P4, P4, 0.0), 3), ((0.3, 0.1, -0.02), 3)]:
        assert weyl.minimum_count(coords, SYC) == (n, "proven"), coords
        assert weyl.n_gate_region_contains(coords, 4, *SYC)


@pytest.mark.slow
def test_three_sycamore_gates_reach_classes_just_past_the_bound():
    # classes with z ≠ 0 and x + y + |z| a little above π/8, in all parts of that face of the region
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    rng = np.random.default_rng(4)
    cal = CouplerCalibration.sycamore()
    for k in range(4):
        w = rng.dirichlet([1, 1, 1])
        x, y, z = np.sort(w)[::-1] * (P8 + 0.02)
        z = z if k % 2 else -z
        u = local(70 + 2 * k) @ weyl.interaction_unitary(x, y, z) @ local(71 + 2 * k)
        assert weyl.minimum_count(u, SYC) == (3, "proven")
        r = syn.decompose(u, calibration=cal)
        assert r.n_stages == 3 and r.infidelity < 1e-9, (x, y, z)


def test_haar_measure_of_the_sycamore_four_gate_region():
    opts = dict(epsabs=1e-14, epsrel=1e-8)
    total = integrate.tplquad(lambda z, y, x: weyl.haar_density(x, y, z), 0, P4, 0, lambda x: x,
                              lambda x, y: -y, lambda x, y: y, **opts)[0]
    assert total == pytest.approx(0.0081812, rel=1e-4)
    # the √iSWAP two-gate region x − y ≥ |z| (79%, Huang et al. 2023) checks the density
    sq2 = integrate.tplquad(lambda z, y, x: weyl.haar_density(x, y, z), 0, P4, 0, lambda x: x,
                            lambda x, y: -min(x - y, y), lambda x, y: min(x - y, y), **opts)[0]
    assert sq2 / total == pytest.approx(0.7901, abs=1e-4)
    # x + y + |z| < π/8, both signs of z
    four = 2 * integrate.tplquad(lambda x, y, z: weyl.haar_density(x, y, z), 0, np.pi / 24, lambda z: z,
                                 lambda z: (P8 - z) / 2, lambda z, y: y, lambda z, y: P8 - y - z, **opts)[0]
    assert four / total == pytest.approx(1.3746e-4, rel=1e-3)
    # Haar sampling agrees with the density on a coarser region (x + y + |z| < 1.0 holds 21.35%)
    coords = np.array([monodromy.weyl_from_monodromy(*monodromy.monodromy_from_unitary(haar_random_unitary(4, seed=3000 + s)))
                       for s in range(3000)])
    share = np.mean(coords[:, 0] + coords[:, 1] + np.abs(coords[:, 2]) < 1.0)
    assert abs(share - 0.2135) < 4 * np.sqrt(0.2135 * 0.7865 / 3000)
    assert abs(np.mean(coords[:, 0] - coords[:, 1] >= np.abs(coords[:, 2])) - 0.7901) < 4 * np.sqrt(0.79 * 0.21 / 3000)


# --------------------------------------------------------------------------- #
# generic natives: drifted couplers
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("native", [DRIFTED, PAIR_33_34, (1.0, 0.4), (0.6, -1.1)],
                         ids=["drifted Sycamore", "pair (3,3)-(3,4)", "FSim(1, 0.4)", "FSim(0.6, -1.1)"])
def test_generic_regions_match_multi_start_synthesis(native):
    # the count from the polytope is reached by synthesis, and one gate fewer is not
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0, weyl_pruning=False)
    for s in range(4):
        u = haar_random_unitary(4, seed=900 + s)
        n, kind = weyl.minimum_count(u, native, max_stages=6)
        assert kind == "proven" and n >= 2
        assert syn._optimize_stage(u, n, n_restarts=12, max_iter=500, fixed=[native] * n).infidelity < 1e-8
        assert syn._optimize_stage(u, n - 1, n_restarts=12, max_iter=500, fixed=[native] * (n - 1)).infidelity > 1e-6


def test_mirror_image_native_gives_a_different_region():
    # FSim(θ, φ) and FSim(θ, −φ) are mirror images (z → −z); the regions differ, and synthesis agrees with the
    # region of the gate actually used, which checks the orientation of the coordinates
    native, mirror = (1.0, 0.4), (1.0, -0.4)
    found = []
    for s in range(200):
        u = haar_random_unitary(4, seed=s)
        a, b = weyl.n_gate_region_contains(u, 2, *native), weyl.n_gate_region_contains(u, 2, *mirror)
        if a != b:
            found.append((u, a))
        if len(found) == 2:
            break
    assert len(found) == 2
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0, weyl_pruning=False)
    for u, two in found:
        res = syn._optimize_stage(u, 2, n_restarts=12, max_iter=500, fixed=[native] * 2)
        assert (res.infidelity < 1e-8) == two


def test_drifted_coupler_counts_are_proven_and_agree_with_the_search():
    for s, n in [(5, 3), (907, 2)]:
        u = haar_random_unitary(4, seed=s)
        assert weyl.minimum_count(u, DRIFTED) == (n, "proven")
        assert weyl.minimum_count(u, DRIFTED, method="search", n_restarts=8, max_iter=300)[0] == n
    # the README's examples: a class that needs four Sycamore gates needs two here, and a weaker one needs four
    assert weyl.minimum_count((0.1, 0.05, 0.02), DRIFTED) == (2, "proven")
    assert weyl.minimum_count((0.01, 0.005, 0.002), DRIFTED) == (4, "proven")
    # a weak native cannot make SWAP in three applications, and the polytope says so without a search
    assert weyl.minimum_count(cirq.unitary(cirq.SWAP), (0.1, 0.05), max_stages=3) == (None, "not found")


@pytest.mark.parametrize("native", [DRIFTED, PAIR_33_34, (np.pi / 2 - 0.2, 0.6)], ids=["drifted", "pair", "FSim(π/2-0.2, 0.6)"])
def test_two_gate_controlled_phases_agree_with_cirq(native):
    # cirq.decompose_cphase_into_two_fsim succeeds exactly when the two-gate region contains the controlled phase
    q = cirq.LineQubit.range(2)
    for k in range(1, 9):
        cph = cirq.CZPowGate(exponent=k / 8)
        try:
            cirq.decompose_cphase_into_two_fsim(cph, fsim_gate=cirq.FSimGate(*native), qubits=q)
            two = True
        except ValueError:
            two = False
        assert weyl.n_gate_region_contains(cirq.unitary(cph), 2, *native) == two, k
        assert weyl.minimum_count(cirq.unitary(cph), native) == (2 if two else 3, "proven")


def test_synthesis_starts_at_the_proven_count_on_a_drifted_coupler():
    cal = CouplerCalibration(theta_cal=DRIFTED[0], phi_cal=DRIFTED[1])
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    r = syn.decompose(haar_random_unitary(4, seed=5), calibration=cal)
    assert r.predicted_stages == r.n_stages == 3 and r.infidelity < 1e-9
