"""
Adversarial and Stress Test Suite for cirq_fsim_compiler.
"""

import pytest
import numpy as np
import cirq
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer, synthesize_unitary_to_fsim


def test_adversarial_identity_decomposition():
    """Verify exact synthesis of the 4x4 Identity unitary."""
    eye4 = np.eye(4, dtype=np.complex128)
    res = synthesize_unitary_to_fsim(eye4, max_stages=1, target_infidelity=1e-4)
    assert res.infidelity < 1e-3
    assert res.fidelity > 0.999


def test_adversarial_global_phase_robustness():
    """Check that global phase e^{i phi} does not degrade synthesis infidelity."""
    cnot = cirq.unitary(cirq.CNOT)
    phase_factor = np.exp(1.0j * np.pi / 3.0)
    cnot_phased = cnot * phase_factor

    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-4)
    res = synth.decompose(cnot_phased, max_stages=3, n_restarts=3)
    assert res.infidelity < 1e-3
    assert res.fidelity > 0.999


def test_adversarial_tensor_product_single_qubit():
    """Check decomposition of pure single-qubit tensor product U1 (x) U2."""
    q0 = cirq.unitary(cirq.X)
    q1 = cirq.unitary(cirq.H)
    u_tensor = np.kron(q0, q1)

    res = synthesize_unitary_to_fsim(u_tensor, max_stages=1, target_infidelity=1e-4)
    assert res.infidelity < 1e-3
    assert res.fidelity > 0.999


def test_adversarial_invalid_matrix_dimensions():
    """Check that invalid matrix dimensions raise ValueError."""
    bad_mat = np.eye(3)
    with pytest.raises(ValueError):
        synthesize_unitary_to_fsim(bad_mat)
