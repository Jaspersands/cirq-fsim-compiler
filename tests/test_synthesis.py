"""
Tests for the FSim circuit template and the differentiable synthesiser,
including calibrated (fixed native angle) mode and the NumPy fallback.
"""

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

from cirq_fsim_compiler.unitary_ansatz import (
    fsim_matrix_np,
    single_qubit_zxz_np,
    FSimCircuitTemplate,
    haar_random_unitary,
    euler_zxz_from_unitary,
    MAX_STAGES,
)
from cirq_fsim_compiler.calibration_map import CouplerCalibration
from cirq_fsim_compiler import riemannian_optimizer as ro
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer, synthesize_unitary_to_fsim

SQRT_ISWAP_ANGLES = (np.pi / 4, 0.0)
SYCAMORE_ANGLES = (np.pi / 2, np.pi / 6)


def fid(u, v):
    return abs(np.trace(u.conj().T @ v)) ** 2 / 16


def test_fsim_matrix_matches_cirq():
    for theta, phi in [(0.45, 0.22), SYCAMORE_ANGLES, (np.pi / 4, 0.0)]:
        assert np.allclose(fsim_matrix_np(theta, phi), cirq.unitary(cirq.FSimGate(theta, phi)))


def test_single_qubit_euler_and_inverse():
    rng = np.random.default_rng(0)
    for _ in range(30):
        a, b, g = rng.uniform(-np.pi, np.pi, 3)
        u = single_qubit_zxz_np(a, b, g)
        assert np.allclose(u.conj().T @ u, np.eye(2), atol=1e-12)
        a2, b2, g2 = euler_zxz_from_unitary(u)
        u2 = single_qubit_zxz_np(a2, b2, g2)
        assert abs(np.trace(u.conj().T @ u2)) ** 2 / 4 > 1 - 1e-10


def test_euler_from_unitary_handles_diagonal_and_antidiagonal():
    for u in (np.eye(2), np.diag([1, 1j]), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])):
        a, b, g = euler_zxz_from_unitary(np.asarray(u, dtype=complex))
        assert abs(np.trace(np.asarray(u).conj().T @ single_qubit_zxz_np(a, b, g))) ** 2 / 4 > 1 - 1e-10


def test_template_param_counts_free_and_fixed():
    assert FSimCircuitTemplate(n_stages=2).num_params() == 3 * 6 + 2 * 2
    fixed = FSimCircuitTemplate(n_stages=2, fixed_fsim=[SYCAMORE_ANGLES] * 2)
    assert fixed.num_params() == 3 * 6
    u = fixed.evaluate_unitary_np(np.zeros(fixed.num_params()))
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-12)
    angles, sq = fixed.extract_angles(np.zeros(fixed.num_params()))
    assert angles == [SYCAMORE_ANGLES, SYCAMORE_ANGLES]
    assert len(sq) == 3 and len(sq[0]) == 2
    assert FSimCircuitTemplate(n_stages=4, fixed_fsim=[SYCAMORE_ANGLES] * 4).num_params() == 5 * 6
    with pytest.raises(ValueError):
        FSimCircuitTemplate(n_stages=MAX_STAGES + 1)
    with pytest.raises(ValueError):
        FSimCircuitTemplate(n_stages=2, fixed_fsim=[SYCAMORE_ANGLES])


def test_haar_random_unitary_is_unitary_and_seeded():
    u1 = haar_random_unitary(4, seed=3)
    u2 = haar_random_unitary(4, seed=3)
    assert np.allclose(u1, u2)
    assert np.allclose(u1.conj().T @ u1, np.eye(4), atol=1e-12)


def test_free_angle_synthesis_standard_gates():
    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-8, seed=1)
    for name, u, stages in [("iSWAP", cirq.unitary(cirq.ISWAP), 1), ("CZ", cirq.unitary(cirq.CZ), 1),
                            ("CNOT", cirq.unitary(cirq.CNOT), 1), ("SWAP", cirq.unitary(cirq.SWAP), 3)]:
        res = synth.decompose(u, max_stages=3, n_restarts=4)
        assert res.infidelity < 1e-7, name
        assert res.n_stages <= stages, name
        assert fid(res.synthesized_unitary, u) > 1 - 1e-7
        assert res.native is False


def test_calibrated_sycamore_native_stage_counts():
    # With the fixed Sycamore gate FSim(π/2, π/6): CZ needs 2, SWAP needs 3, and
    # the native gate itself needs 1.
    cal = CouplerCalibration(theta_cal=SYCAMORE_ANGLES[0], phi_cal=SYCAMORE_ANGLES[1])
    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-8, seed=2)
    res_cz = synth.decompose(cirq.unitary(cirq.CZ), max_stages=3, n_restarts=6, calibration=cal)
    assert res_cz.native is True and res_cz.n_stages == 2 and res_cz.infidelity < 1e-7
    assert all(np.allclose(a, SYCAMORE_ANGLES) for a in res_cz.fsim_angles)
    res_swap = synth.decompose(cirq.unitary(cirq.SWAP), max_stages=3, n_restarts=6, calibration=cal)
    assert res_swap.n_stages == 3 and res_swap.infidelity < 1e-6
    res_native = synth.decompose(fsim_matrix_np(*SYCAMORE_ANGLES), max_stages=3, n_restarts=2, calibration=cal)
    assert res_native.n_stages == 1 and res_native.infidelity < 1e-9


def test_calibrated_sqrt_iswap_cz_needs_two_stages_and_iswap_two():
    cal = CouplerCalibration(theta_cal=np.pi / 4, phi_cal=0.0)
    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-8, seed=5)
    one = synth._optimize_stage(cirq.unitary(cirq.ISWAP), 1, n_restarts=6, max_iter=200, fixed=[cal.angles])
    assert one.infidelity > 1e-2                      # iSWAP is not reachable with a single √iSWAP
    two = synth.decompose(cirq.unitary(cirq.ISWAP), max_stages=3, n_restarts=6, calibration=cal)
    assert two.n_stages == 2 and two.infidelity < 1e-7


def test_random_u4_free_angles_two_stages_suffice_and_calibrated_needs_three():
    # Two free FSim(θ, φ) gates carry four interaction parameters, enough for the
    # three KAK coordinates of a generic SU(4); a fixed native gate needs three.
    u = haar_random_unitary(4, seed=11)
    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    res = synth.decompose(u, max_stages=3, n_restarts=6, max_iter=400)
    assert res.n_stages <= 3
    assert res.infidelity < 1e-8
    assert len(res.loss_history) > 0
    one_stage = synth._optimize_stage(u, 1, n_restarts=4, max_iter=300)
    assert one_stage.infidelity > 1e-3                 # generic U(4) is not one FSim
    cal = CouplerCalibration.sycamore()
    res_cal = synth.decompose(u, max_stages=3, n_restarts=8, max_iter=400, calibration=cal)
    assert res_cal.native and res_cal.n_stages == 3 and res_cal.infidelity < 1e-6


def test_numpy_fallback_gradient_matches_jax(monkeypatch):
    if not ro.HAS_JAX:
        pytest.skip("JAX not installed")
    u = cirq.unitary(cirq.CNOT)
    synth = DifferentiableFSimSynthesizer(seed=0)
    template, loss_jit, grad_jit = synth._compiled_templates[2]
    p = np.random.default_rng(4).uniform(-1, 1, template.num_params())
    import jax.numpy as jnp
    g_jax = np.asarray(grad_jit(jnp.asarray(p), jnp.asarray(u)))
    monkeypatch.setattr(ro, "HAS_JAX", False)
    synth_np = DifferentiableFSimSynthesizer(seed=0)
    val_np, g_np = synth_np._cost_and_grad_numpy(template, p, u)
    assert np.allclose(g_np, g_jax, rtol=1e-4, atol=1e-6)
    assert np.isclose(val_np, float(loss_jit(jnp.asarray(p), jnp.asarray(u))), atol=1e-10)


def test_invalid_matrix_dimensions():
    with pytest.raises(ValueError):
        synthesize_unitary_to_fsim(np.eye(3))


def test_controlled_phase_classes_use_the_analytic_two_gate_route():
    import scipy.linalg
    xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    loc = np.kron(haar_random_unitary(2, seed=1), haar_random_unitary(2, seed=2))
    cal = CouplerCalibration.sycamore()
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    # a controlled-phase class at Weyl (0.189, 0, 0), conjugated by a random local gate; the optimiser alone
    # converges slowly here, and the analytic route makes it exactly
    for u in (cirq.unitary(cirq.CNOT), cirq.unitary(cirq.CZPowGate(exponent=0.3)),
              loc @ scipy.linalg.expm(0.189j * xx) @ loc.conj().T):
        r = syn.decompose(u, calibration=cal)
        assert r.n_stages == 2 and r.infidelity < 1e-12 and "decompose_cphase_into_two_fsim" in r.method
        assert all(np.allclose(a, SYCAMORE_ANGLES) for a in r.fsim_angles)
        assert fid(r.synthesized_unitary, u) > 1 - 1e-12


def test_failed_exact_synthesis_warns_or_raises():
    weak = CouplerCalibration(theta_cal=0.1, phi_cal=0.05)        # two weak gates cannot make SWAP
    u = cirq.unitary(cirq.SWAP)
    with pytest.warns(ro.SynthesisWarning, match="not equivalent"):
        r = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0).decompose(
            u, calibration=weak, max_stages=2, n_restarts=2, max_iter=100)
    assert r.infidelity > 1e-3
    with pytest.raises(ro.SynthesisError):
        DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0, on_failure="raise").decompose(
            u, calibration=weak, max_stages=2, n_restarts=2, max_iter=100)
    with pytest.raises(ValueError):
        DifferentiableFSimSynthesizer(on_failure="sometimes")


def test_noise_aware_mode_does_not_warn_when_the_exact_candidate_misses():
    import warnings
    weak = CouplerCalibration(theta_cal=0.1, phi_cal=0.05)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0).decompose(
            cirq.unitary(cirq.SWAP), calibration=weak, max_stages=2, n_restarts=2, max_iter=100,
            approximation="noise_aware", gate_error=0.01)
    assert not any(issubclass(w.category, ro.SynthesisWarning) for w in caught)
    assert r.expected_infidelity == pytest.approx(r.infidelity + r.n_stages * 0.01, abs=1e-12)


def test_max_stages_is_validated():
    from cirq_fsim_compiler.cirq_transformer import FSimDecomposerTransformer
    with pytest.raises(ValueError, match="max_stages must be an integer from 0 to 6"):
        DifferentiableFSimSynthesizer().decompose(haar_random_unitary(4, seed=1), max_stages=7)
    with pytest.raises(ValueError, match="max_stages"):
        FSimDecomposerTransformer(max_stages=10)
