"""
Tests for the Weyl-chamber analysis: coordinates of known gates, closed-form
minimum gate counts (checked against Cirq and against numerical synthesis),
the closed-form class fidelity, and the stage prediction.
"""

import numpy as np
import pytest
import scipy.linalg

cirq = pytest.importorskip("cirq")

from cirq_fsim_compiler import weyl  # noqa: E402
from cirq_fsim_compiler.calibration_map import CouplerCalibration  # noqa: E402
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer  # noqa: E402
from cirq_fsim_compiler.unitary_ansatz import fsim_matrix_np, haar_random_unitary  # noqa: E402

P4, P8 = np.pi / 4, np.pi / 8
XX = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
YY = np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])
ZZ = np.diag([1.0, -1.0, -1.0, 1.0])


def interaction(x, y, z):
    return scipy.linalg.expm(1j * (x * XX + y * YY + z * ZZ))


def local(seed):
    return np.kron(haar_random_unitary(2, seed=seed), haar_random_unitary(2, seed=seed + 1))


KNOWN = [
    ("I", np.eye(4), (0, 0, 0)),
    ("CNOT", cirq.unitary(cirq.CNOT), (P4, 0, 0)),
    ("CZ", cirq.unitary(cirq.CZ), (P4, 0, 0)),
    ("iSWAP", cirq.unitary(cirq.ISWAP), (P4, P4, 0)),
    ("√iSWAP", cirq.unitary(cirq.SQRT_ISWAP), (P8, P8, 0)),
    ("SWAP", cirq.unitary(cirq.SWAP), (P4, P4, P4)),
    ("Sycamore", fsim_matrix_np(np.pi / 2, np.pi / 6), (P4, P4, np.pi / 24)),
    ("CPhase(π/2)", cirq.unitary(cirq.CZPowGate(exponent=0.5)), (P8, 0, 0)),
    ("B", interaction(P4, P8, 0), (P4, P8, 0)),
]


@pytest.mark.parametrize("name,u,expected", KNOWN, ids=[k[0] for k in KNOWN])
def test_weyl_coordinates_of_known_gates(name, u, expected):
    c = weyl.weyl_coordinates(u)
    assert np.allclose(c, expected, atol=1e-12), (name, c)
    # unchanged by single-qubit gates on either side
    assert np.allclose(weyl.weyl_coordinates(local(3) @ u @ local(7)), expected, atol=1e-9)


def test_fsim_coordinates_formula_matches_kak():
    rng = np.random.default_rng(0)
    for theta, phi in rng.uniform(-np.pi, np.pi, size=(50, 2)):
        assert np.allclose(weyl.fsim_weyl_coordinates(theta, phi), weyl.weyl_coordinates(fsim_matrix_np(theta, phi)), atol=1e-12)


# --------------------------------------------------------------------------- #
# closed-form counts
# --------------------------------------------------------------------------- #
COUNTS = {   # name: (CZ, √iSWAP, free FSim)
    "I": (0, 0, 0), "CNOT": (1, 2, 1), "CZ": (1, 2, 1), "iSWAP": (2, 2, 1), "√iSWAP": (2, 1, 1),
    "SWAP": (3, 3, 1), "Sycamore": (3, 3, 1), "CPhase(π/2)": (2, 2, 1), "B": (2, 2, 2),
}


@pytest.mark.parametrize("name,u,_", KNOWN, ids=[k[0] for k in KNOWN])
def test_analytic_minimum_counts_of_known_gates(name, u, _):
    cz, sq, free = COUNTS[name]
    assert weyl.min_cz_count(u) == cz
    assert weyl.min_sqrt_iswap_count(u) == sq
    assert weyl.min_free_fsim_count(u) == free
    assert weyl.analytic_min_count(u, fsim_angles=CouplerCalibration.cz().angles) == cz
    assert weyl.analytic_min_count(u, fsim_angles=CouplerCalibration.sqrt_iswap().angles) == sq
    assert weyl.analytic_min_count(u, fsim_angles=None) == free


def test_cz_and_sqrt_iswap_counts_agree_with_cirq():
    q0, q1 = cirq.LineQubit.range(2)
    mats = [u for _, u, _ in KNOWN] + [haar_random_unitary(4, seed=s) for s in range(30)]
    mats += [cirq.unitary(cirq.CZPowGate(exponent=t)) @ local(5) @ cirq.unitary(cirq.ISWAP ** 0.3) for t in (0.2, 0.7)]
    for u in mats:
        assert weyl.min_cz_count(u) == cirq.num_cnots_required(u)
        n_sq = sum(1 for op in cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u) if len(op.qubits) == 2)
        assert weyl.min_sqrt_iswap_count(u) == n_sq


def test_free_fsim_one_gate_set_is_the_two_faces():
    # x = y and y = |z| are one FSim; the B gate (π/4, π/8, 0) is not
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0, weyl_pruning=False)
    for coords, n in [((0.5, 0.5, -0.2), 1), ((0.6, 0.3, -0.3), 1), ((P4, P8, 0.0), 2), ((0.7, 0.4, 0.1), 2)]:
        u = local(11) @ interaction(*coords) @ local(13)
        assert weyl.min_free_fsim_count(u) == n
        one = syn._optimize_stage(u, 1, n_restarts=8, max_iter=300)
        assert (one.infidelity < 1e-9) == (n == 1), coords
        assert syn._optimize_stage(u, 2, n_restarts=8, max_iter=300).infidelity < 1e-9


# --------------------------------------------------------------------------- #
# closed-form class fidelity
# --------------------------------------------------------------------------- #
def test_class_fidelity_matches_numerical_optimum():
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-15, seed=0)
    rng = np.random.default_rng(1)
    for s in range(6):
        u = haar_random_unitary(4, seed=100 + s)
        theta, phi = rng.uniform(-np.pi, np.pi, 2)
        one = syn._optimize_stage(u, 1, n_restarts=8, max_iter=500, fixed=[(theta, phi)])
        assert abs(one.fidelity - weyl.class_fidelity(u, fsim_matrix_np(theta, phi))) < 1e-9
        zero = syn._optimize_stage(u, 0, n_restarts=8, max_iter=500, fixed=[])
        assert abs(zero.fidelity - weyl.class_fidelity(u, np.zeros(3))) < 1e-9
    assert weyl.class_fidelity(cirq.unitary(cirq.CNOT), cirq.unitary(cirq.CZ)) == pytest.approx(1.0, abs=1e-14)


def test_two_controlled_phase_type_gates_reach_only_z_zero():
    # FSim(0, φ) and FSim(π/2, φ) (SWAP times a controlled phase): two of them give z = 0
    rng = np.random.default_rng(2)
    for k in range(300):
        theta = (0.0, np.pi / 2)[k % 2]
        p1, p2 = rng.uniform(-np.pi, np.pi, 2)
        u = fsim_matrix_np(theta, p1) @ local(1000 + 2 * k) @ fsim_matrix_np(theta, p2)
        assert abs(weyl.weyl_coordinates(u)[2]) < 1e-9
    generic = fsim_matrix_np(1.0, 0.4) @ local(3) @ fsim_matrix_np(1.0, 0.4)
    assert abs(weyl.weyl_coordinates(generic)[2]) > 1e-2


def test_native_gate_families():
    fam = weyl.native_gate_family
    assert fam(0.0, np.pi) == fam(0.0, -np.pi) == "cz"
    assert fam(np.pi / 4, 0.0) == fam(-np.pi / 4, 0.0) == "sqrt_iswap"
    assert fam(np.pi / 2, np.pi / 6) == "swap_cphase"
    assert fam(0.0, 1.0) == "cphase"
    assert fam(1.0, 0.4) == "generic"
    assert fam(0.0, 0.0) == "identity"


def test_stage_prediction():
    syc = CouplerCalibration.sycamore().angles
    haar = haar_random_unitary(4, seed=5)
    p = weyl.predict_stages(haar, fsim_angles=syc)
    assert p.lower == 3 and p.exact is None                      # n = 1 and n = 2 are excluded exactly
    assert weyl.predict_stages(cirq.unitary(cirq.CZ), fsim_angles=syc).lower == 2
    assert weyl.predict_stages(fsim_matrix_np(*syc), fsim_angles=syc).exact == 1
    assert weyl.predict_stages(local(9), fsim_angles=syc).exact == 0
    drifted = (syc[0] + 0.02, syc[1] - 0.01)                     # a generic gate: only a lower bound
    assert weyl.predict_stages(haar, fsim_angles=drifted).lower == 2
    assert weyl.predict_stages(haar, fsim_angles=None).exact == 2


def test_numerical_minimum_for_sycamore():
    syc = CouplerCalibration.sycamore().angles
    assert weyl.numerical_min_count(cirq.unitary(cirq.CZ), *syc) == 2
    assert weyl.numerical_min_count(cirq.unitary(cirq.ISWAP), *syc) == 3
    assert weyl.numerical_min_count(fsim_matrix_np(*syc), *syc) == 1
    # a weak native gate cannot make SWAP in three applications
    assert weyl.numerical_min_count(cirq.unitary(cirq.SWAP), 0.1, 0.05, max_stages=3, n_restarts=4, max_iter=200) is None


@pytest.mark.slow
def test_sqrt_iswap_two_gate_region_matches_numerical_synthesis():
    # Huang et al. 2023: two √iSWAP iff x − y ≥ |z|; check against multi-start synthesis
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0, weyl_pruning=False)
    sq = CouplerCalibration.sqrt_iswap().angles
    hits = 0
    for s in range(40):
        u = haar_random_unitary(4, seed=500 + s)
        two = syn._optimize_stage(u, 2, n_restarts=8, max_iter=300, fixed=[sq, sq]).infidelity < 1e-8
        assert two == (weyl.min_sqrt_iswap_count(u) == 2)
        hits += two
    assert 20 <= hits <= 39                                       # about 79% of the Haar measure


def test_two_free_fsims_cover_the_weyl_chamber():
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0)
    n = 4
    for i in range(n + 1):
        for j in range(i + 1):
            for k in range(-j, j + 1):
                x, y, z = P4 * i / n, P4 * j / n, P4 * k / n
                if i == n and k < 0:
                    continue
                res = syn._optimize_stage(interaction(x, y, z), 2, n_restarts=8, max_iter=300)
                assert res.infidelity < 1e-9, (x, y, z)


# --------------------------------------------------------------------------- #
# coordinate input, the x − y = |z| boundary and the sum-of-strengths bound
# --------------------------------------------------------------------------- #
def test_count_functions_canonicalise_coordinate_input():
    assert weyl.min_cz_count((0.1, 0, 0.3)) == 2                   # the class of (0.3, 0.1, 0)
    for f in (weyl.min_cz_count, weyl.min_sqrt_iswap_count, weyl.min_free_fsim_count):
        for c in [(0.1, 0, 0.3), (-0.2, 0.5, 0.1), (0.7, -0.3, 0.6)]:
            assert f(c) == f(interaction(*c)), (f.__name__, c)
    assert weyl.minimum_count((0.1, 0, 0.3), CouplerCalibration.sycamore().angles) == (2, "proven")


@pytest.mark.parametrize("coords", [(0.5, 0.3, 0.2), (0.6, 0.35, -0.25), (P4, P8, P8), (0.3, 0.15, -0.15)])
def test_sqrt_iswap_boundary_x_minus_y_equals_abs_z(coords):
    # on the plane x − y = |z| two √iSWAP still suffice (Huang et al. 2023)
    u = local(41) @ interaction(*coords) @ local(43)
    assert weyl.min_sqrt_iswap_count(u) == 2 and weyl.min_sqrt_iswap_count(coords) == 2
    sq = CouplerCalibration.sqrt_iswap().angles
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0)
    assert syn._optimize_stage(u, 2, n_restarts=8, max_iter=300, fixed=[sq, sq]).infidelity < 1e-10
    x, y, z = coords
    beyond = (x, y, np.sign(z or 1) * (abs(z) + 1e-3))              # just past the plane: three
    assert weyl.min_sqrt_iswap_count(beyond) == 3


def test_controlled_phase_strengths():
    syc = CouplerCalibration.sycamore().angles
    assert weyl.cphase_strength(*syc) == pytest.approx(5 * np.pi / 24)
    assert weyl.cphase_strength(0.0, np.pi) == pytest.approx(P4)        # CZ
    assert weyl.cphase_strength(0.0, 1.0) == pytest.approx(0.25)        # FSim(0, φ): φ/4
    assert weyl.cphase_strength(np.pi / 2, np.pi) == pytest.approx(0.0)     # FSim(π/2, π) is SWAP
    assert weyl.cphase_strength(np.pi / 2, 0.0) == pytest.approx(P4)        # SWAP · CZ, the iSWAP class
    assert weyl.cphase_strength(1.0, 0.4) is None
    # iSWAP: two Sycamores reach only x + y ≤ 5π/12, so three are proven necessary
    assert weyl.cphase_count_excluded(cirq.unitary(cirq.ISWAP), 2, *syc)
    assert weyl.minimum_count(cirq.unitary(cirq.ISWAP), syc) == (3, "proven")


@pytest.mark.parametrize("native", [(np.pi / 2, np.pi / 6), (0.0, 1.0), (np.pi / 2, 2.0)], ids=["Sycamore", "CPhase", "SWAP·CPhase"])
def test_sum_of_strengths_bound_holds_on_random_products(native):
    rng = np.random.default_rng(3)
    g = fsim_matrix_np(*native)
    for n in (2, 3, 4):
        for k in range(60):
            u = local(int(rng.integers(1 << 30)))
            for _ in range(n):
                u = local(int(rng.integers(1 << 30))) @ g @ u
            assert not weyl.cphase_count_excluded(u, n, *native, tol=1e-7), (n, weyl.weyl_coordinates(u))


def test_weak_target_needs_four_sycamore_gates():
    # exp(i(0.1 XX + 0.05 YY + 0.02 ZZ)): z ≠ 0 rules out two, x + y + |z| < π/8 rules out three
    syc = CouplerCalibration.sycamore()
    u = local(51) @ interaction(0.1, 0.05, 0.02) @ local(53)
    assert weyl.predict_stages(u, fsim_angles=syc.angles).lower == 4
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    three = syn.best_fit(u, 3, syc, n_restarts=8, max_iter=300)
    assert three.infidelity > 1e-3
    r = syn.decompose(u, calibration=syc)
    assert r.n_stages == 4 and r.infidelity < 1e-9
    assert weyl.minimum_count(u, syc.angles) == (4, "proven")


def test_three_sycamore_bound_is_attained():
    # three controlled phases of strength 5π/24 along XX, YY and ZZ give (5π/24)(1, 1, 1); the leftover
    # SWAP maps that to (π/24, π/24, π/24), where x + y + |z| = π/8 exactly
    syc = CouplerCalibration.sycamore()
    c = (np.pi / 24,) * 3
    assert not weyl.cphase_count_excluded(c, 3, *syc.angles)
    assert weyl.cphase_count_excluded((np.pi / 24, np.pi / 24, np.pi / 24 - 1e-3), 3, *syc.angles)
    u = local(61) @ interaction(*c) @ local(63)
    r = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0).decompose(u, calibration=syc)
    assert r.n_stages == 3 and r.infidelity < 1e-9


def test_zero_gate_label_uses_the_closed_form_infidelity():
    syc = CouplerCalibration.sycamore().angles
    small = cirq.unitary(cirq.CZPowGate(exponent=0.003))          # 0-gate infidelity 5.6e-6
    assert weyl.predict_stages(small, fsim_angles=syc, tol=np.sqrt(1e-6)).exact is None
    assert weyl.minimum_count(small, syc, tol=1e-6) == (2, "proven")
    assert weyl.predict_stages(small, fsim_angles=syc, tol=np.sqrt(1e-5)).exact == 0
    a = np.sqrt(2e-9 / 3)                                          # every coordinate below √ε, infidelity 2ε
    assert weyl.predict_stages((a, a, a), fsim_angles=syc, tol=np.sqrt(1e-9)).exact is None
