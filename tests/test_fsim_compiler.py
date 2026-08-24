"""
Unit tests for Cirq FSim Differentiable Compiler.
"""

import pytest
import numpy as np
import cirq
from cirq_fsim_compiler.unitary_ansatz import (
    fsim_matrix_np,
    single_qubit_zxz_np,
    FSimCircuitTemplate,
)
from cirq_fsim_compiler.riemannian_optimizer import (
    DifferentiableFSimSynthesizer,
    synthesize_unitary_to_fsim,
)
from cirq_fsim_compiler.cirq_transformer import (
    FSimDecomposerTransformer,
    compile_circuit_to_sycamore_fsim,
)


def test_fsim_matrix_unitarity():
    mat = fsim_matrix_np(theta=0.45, phi=0.22)
    assert mat.shape == (4, 4)
    assert np.allclose(mat.conj().T @ mat, np.eye(4), atol=1e-8)


def test_single_qubit_euler_unitarity():
    u = single_qubit_zxz_np(alpha=0.3, beta=0.8, gamma=-0.5)
    assert u.shape == (2, 2)
    assert np.allclose(u.conj().T @ u, np.eye(2), atol=1e-8)


def test_circuit_template_evaluation():
    template = FSimCircuitTemplate(n_stages=2)
    p = np.zeros(template.num_params())
    u = template.evaluate_unitary_np(p)
    assert u.shape == (4, 4)
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-8)


def test_synthesize_iswap():
    # iSWAP is native 1-stage FSim(pi/2, 0)
    u_iswap = cirq.unitary(cirq.ISWAP)
    res = synthesize_unitary_to_fsim(u_iswap, max_stages=2, target_infidelity=1e-4)
    assert res.infidelity < 1e-3
    assert res.fidelity > 0.999


def test_synthesize_cnot():
    u_cnot = cirq.unitary(cirq.CNOT)
    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-4)
    res = synth.decompose(u_cnot, max_stages=3, n_restarts=3)
    assert res.infidelity < 1e-3
    assert res.fidelity > 0.999


def test_cirq_transformer_pipeline():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit([cirq.CNOT(q0, q1)])
    compiled = compile_circuit_to_sycamore_fsim(c)

    # Make sure CNOT was replaced by FSim
    has_fsim = any(isinstance(op.gate, cirq.FSimGate) for op in compiled.all_operations())
    assert has_fsim, "Compiled circuit must contain native FSim gates"

    # Verify unitary equivalence (up to global phase)
    u_orig = cirq.unitary(c)
    u_comp = cirq.unitary(compiled)
    fid = float(np.abs(np.trace(u_orig.conj().T @ u_comp)) ** 2 / 16.0)
    assert fid > 0.99
