"""
Tests for the CZ calibration, Weyl-pruned stage search, zero-stage
decompositions and noise-aware approximate synthesis (single blocks and
whole circuits).
"""

import numpy as np
import pytest
import scipy.linalg

cirq = pytest.importorskip("cirq")

from cirq_fsim_compiler import weyl  # noqa: E402
from cirq_fsim_compiler.calibration_map import (  # noqa: E402
    CouplerCalibration, SycamoreCalibrationMap, WILLOW_CZ_ERROR, SYCAMORE_GATE_ERROR, SQRT_ISWAP_GATE_ERROR)
from cirq_fsim_compiler.cirq_transformer import FSimDecomposerTransformer, compile_to_fsim  # noqa: E402
from cirq_fsim_compiler.compare import qft_circuit  # noqa: E402
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer  # noqa: E402
from cirq_fsim_compiler.toffoli_decomposer import toffoli_circuit  # noqa: E402
from cirq_fsim_compiler.unitary_ansatz import fsim_matrix_np, haar_random_unitary  # noqa: E402

XX = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
YY = np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])
ZZ = np.diag([1.0, -1.0, -1.0, 1.0])


def fid(u, v):
    return abs(np.trace(u.conj().T @ v)) ** 2 / u.shape[0] ** 2


def local(seed):
    return np.kron(haar_random_unitary(2, seed=seed), haar_random_unitary(2, seed=seed + 1))


# --------------------------------------------------------------------------- #
# CZ calibration
# --------------------------------------------------------------------------- #
def test_cz_calibration_is_cirq_cz():
    cal = CouplerCalibration.cz()
    u = fsim_matrix_np(*cal.angles)
    assert np.allclose(u, cirq.unitary(cirq.CZ), atol=1e-15)            # equal, not only locally equivalent
    assert np.allclose(cirq.unitary(cirq.FSimGate(*cal.angles)), cirq.unitary(cirq.CZ), atol=1e-15)
    assert np.allclose(weyl.weyl_coordinates(u), weyl.weyl_coordinates(cirq.unitary(cirq.CZ)))
    assert weyl.native_gate_family(*cal.angles) == "cz"
    assert cal.gate_error == WILLOW_CZ_ERROR
    assert CouplerCalibration(**cal.to_dict()) == cal


def test_cz_calibrated_synthesis_counts_and_angles():
    cal = CouplerCalibration.cz()
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=3)
    for gate, n in [(cirq.CNOT, 1), (cirq.CZ, 1), (cirq.ISWAP, 2), (cirq.SQRT_ISWAP, 2), (cirq.SWAP, 3)]:
        u = cirq.unitary(gate)
        r = syn.decompose(u, calibration=cal)
        assert r.n_stages == n and r.infidelity < 1e-8, gate
        assert all(np.allclose(a, (0.0, np.pi)) for a in r.fsim_angles)
    r = syn.decompose(haar_random_unitary(4, seed=8), calibration=cal)
    assert r.n_stages == 3 and r.infidelity < 1e-8


def test_cz_transformer_output_uses_only_cz():
    q = cirq.LineQubit.range(3)
    c = toffoli_circuit(*q)
    out = compile_to_fsim(c, calibration=CouplerCalibration.cz(), target_infidelity=1e-9)
    two = [op for op in out.all_operations() if len(op.qubits) == 2]
    assert len(two) == 6
    assert all(np.allclose(cirq.unitary(op.gate), cirq.unitary(cirq.CZ)) for op in two)
    assert fid(cirq.unitary(out), cirq.unitary(c)) > 1 - 1e-7
    # the same through a calibration map with the CZ gate as nominal and no drift
    cm = SycamoreCalibrationMap(seed=0, theta_drift_std=0.0, phi_drift_std=0.0, nominal=CouplerCalibration.cz())
    assert cm.get_coupler(0, 1).gate_error == WILLOW_CZ_ERROR
    out2 = compile_to_fsim(c, calibration_map=cm, target_infidelity=1e-9)
    assert sum(len(op.qubits) == 2 for op in out2.all_operations()) == 6


# --------------------------------------------------------------------------- #
# stage search
# --------------------------------------------------------------------------- #
def _count_calls(monkeypatch, syn):
    calls = []
    orig = syn._optimize_stage

    def wrapped(target, n, *a, **k):
        calls.append(n)
        return orig(target, n, *a, **k)
    monkeypatch.setattr(syn, "_optimize_stage", wrapped)
    return calls


def test_weyl_pruning_skips_stage_counts_that_cannot_succeed(monkeypatch):
    u = haar_random_unitary(4, seed=4)
    for cal, pruned_calls in [(CouplerCalibration.cz(), [3]), (CouplerCalibration.sycamore(), [3]), (None, [2])]:
        syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
        calls = _count_calls(monkeypatch, syn)
        r = syn.decompose(u, calibration=cal)
        assert calls == pruned_calls and r.infidelity < 1e-8
        old = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0, weyl_pruning=False)
        old_calls = _count_calls(monkeypatch, old)
        r_old = old.decompose(u, calibration=cal)
        assert old_calls[0] == 1 and len(old_calls) > len(calls)
        assert r_old.n_stages == r.n_stages


def test_local_targets_take_zero_native_gates():
    u = local(21)
    for cal in (None, CouplerCalibration.sycamore(), CouplerCalibration.cz()):
        r = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0).decompose(u, calibration=cal)
        assert r.n_stages == 0 and r.fsim_angles == [] and r.infidelity < 1e-10
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(q0, q1), cirq.Z(q0), cirq.CNOT(q0, q1))      # a local block
    out = compile_to_fsim(c, calibration=CouplerCalibration.sycamore())
    assert not any(len(op.qubits) == 2 for op in out.all_operations())
    assert fid(cirq.unitary(out), cirq.unitary(c)) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# noise-aware synthesis
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_noise_aware_never_worse_than_exact():
    # a check of the construction (the exact result is always a candidate), not evidence of fit quality
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    for s in range(3):
        u = haar_random_unitary(4, seed=40 + s)
        for cal in (None, CouplerCalibration.sycamore(), CouplerCalibration.sqrt_iswap(), CouplerCalibration.cz()):
            exact = syn.decompose(u, calibration=cal)
            for eps in (1e-4, 2e-2):
                r = syn.decompose(u, calibration=cal, approximation="noise_aware", gate_error=eps)
                assert r.n_stages <= exact.n_stages and r.exact_n_stages == exact.n_stages
                assert r.expected_infidelity <= exact.infidelity + exact.n_stages * eps + 1e-12
                assert r.expected_infidelity == pytest.approx(r.infidelity + r.n_stages * eps, abs=1e-12)
                assert fid(r.synthesized_unitary, u) == pytest.approx(1 - r.infidelity, abs=1e-9)


def test_noise_aware_drops_a_gate_when_it_pays():
    # (π/4, 0.3, 0.02) needs three CZ; two CZ reach it to infidelity ≈ sin²(0.02) = 4e-4
    u = local(31) @ scipy.linalg.expm(1j * (np.pi / 4 * XX + 0.3 * YY + 0.02 * ZZ)) @ local(33)
    cal = CouplerCalibration.cz()
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    r = syn.decompose(u, calibration=cal, approximation="noise_aware")       # uses cal.gate_error = 0.41%
    assert r.gate_error == WILLOW_CZ_ERROR
    assert r.exact_n_stages == 3 and r.n_stages == 2 and r.approximate
    assert r.infidelity == pytest.approx(np.sin(0.02) ** 2, rel=1e-5)
    assert syn.decompose(u, calibration=cal, approximation="noise_aware", gate_error=1e-5).n_stages == 3
    # a small controlled phase is cheaper to drop entirely
    small = cirq.unitary(cirq.CZPowGate(exponent=0.02))
    r0 = syn.decompose(small, calibration=CouplerCalibration.sycamore(), approximation="noise_aware")
    assert r0.n_stages == 0 and r0.gate_error == SYCAMORE_GATE_ERROR
    assert r0.infidelity == pytest.approx(1 - weyl.class_fidelity(small, np.zeros(3)), abs=1e-9)


def test_noise_aware_requires_an_error_rate():
    with pytest.raises(ValueError):
        DifferentiableFSimSynthesizer().decompose(haar_random_unitary(4, seed=1), approximation="noise_aware")
    with pytest.raises(ValueError):
        DifferentiableFSimSynthesizer(approximation="approximately")


def test_noise_aware_circuit_within_bound_and_exact_circuit_equivalent():
    q = cirq.LineQubit.range(4)
    c = qft_circuit(q)
    ideal = cirq.unitary(c)
    eps = SQRT_ISWAP_GATE_ERROR
    exact_tr = FSimDecomposerTransformer(target_infidelity=1e-9, calibration=CouplerCalibration.sqrt_iswap())
    exact = exact_tr.optimize_circuit(c)
    assert fid(cirq.unitary(exact), ideal) > 1 - 1e-8
    assert exact_tr.stats["approximated_blocks"] == 0
    tr = FSimDecomposerTransformer(target_infidelity=1e-9, calibration=CouplerCalibration.sqrt_iswap(),
                                   approximation="noise_aware", gate_error=eps)
    approx = tr.optimize_circuit(c)
    n_exact = sum(len(op.qubits) == 2 for op in exact.all_operations())
    n_approx = sum(len(op.qubits) == 2 for op in approx.all_operations())
    assert tr.stats["approximated_blocks"] >= 1 and n_approx < n_exact     # drops the π/8 controlled phase
    infid = 1 - fid(cirq.unitary(approx), ideal)
    assert 0 < infid <= tr.infidelity_bound() + 1e-9
    # expected circuit fidelity is higher with the approximation
    assert (1 - infid) * (1 - eps) ** n_approx > (1 - eps) ** n_exact


def test_noise_aware_uses_per_edge_errors_from_a_calibration_map():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CZPowGate(exponent=0.05).on(q0, q1))
    cm = SycamoreCalibrationMap(seed=0, theta_drift_std=0.0, phi_drift_std=0.0, nominal=CouplerCalibration.cz())
    cm.set_coupler(0, 1, CouplerCalibration(theta_cal=0.0, phi_cal=np.pi, gate_error=1e-5))
    keep = FSimDecomposerTransformer(calibration_map=cm, approximation="noise_aware").optimize_circuit(c)
    assert sum(len(op.qubits) == 2 for op in keep.all_operations()) == 2
    cm.set_coupler(0, 1, CouplerCalibration(theta_cal=0.0, phi_cal=np.pi, gate_error=0.01))
    drop = FSimDecomposerTransformer(calibration_map=cm, approximation="noise_aware").optimize_circuit(c)
    assert sum(len(op.qubits) == 2 for op in drop.all_operations()) == 0


def test_published_errors_are_process_infidelities():
    # average gate infidelities (Willow RB, Huang iFRB) are scaled by (d + 1)/d = 5/4; the
    # Sycamore Pauli error is already a process infidelity
    assert WILLOW_CZ_ERROR == pytest.approx(1.25 * 0.0033) and SQRT_ISWAP_GATE_ERROR == pytest.approx(1.25 * 0.0069)
    assert SYCAMORE_GATE_ERROR == 0.0062


def test_free_one_gate_fit_uses_the_closed_form():
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=0)
    for s in range(3):
        u = haar_random_unitary(4, seed=70 + s)
        theta, phi, f = syn.best_single_free_fsim(u)
        r = syn.best_fit(u, 1, None)
        assert r.infidelity == pytest.approx(1 - f, abs=1e-10) and not r.native
        assert r.infidelity == pytest.approx(syn.closed_form_best_infidelity(u, 1, None), abs=1e-12)
        # no FSim angle pair on a coarse grid does better than the closed-form optimum
        grid = np.linspace(-np.pi, np.pi, 9)
        best_grid = max(weyl.class_fidelity(u, fsim_matrix_np(t, p)) for t in grid for p in grid)
        assert best_grid <= f + 1e-12
