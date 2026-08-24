"""
Tests for Project 3 Extensions: Calibration maps, Toffoli/Fredkin synthesis, OpenQASM3 export, and Batch compilation.
"""

import pytest
import numpy as np
import cirq
from cirq_fsim_compiler.calibration_map import SycamoreCalibrationMap, CouplerCalibration
from cirq_fsim_compiler.toffoli_decomposer import decompose_toffoli_to_sycamore, decompose_fredkin_to_sycamore
from cirq_fsim_compiler.openqasm3_io import export_to_openqasm3
from cirq_fsim_compiler.batch_compiler import BatchFSimCompiler


def test_sycamore_calibration_map():
    cal_map = SycamoreCalibrationMap(seed=42)
    coupler = cal_map.get_coupler(0, 1)
    assert abs(coupler.theta_cal - np.pi / 2.0) < 0.1
    assert abs(coupler.phi_cal - np.pi / 6.0) < 0.1


def test_decompose_toffoli_to_sycamore():
    q0, q1, q2 = cirq.LineQubit.range(3)
    compiled = decompose_toffoli_to_sycamore(q0, q1, q2, target_infidelity=1e-3)
    assert len(compiled) > 0

    # Ensure native FSim gates are present in the circuit
    has_fsim = any(isinstance(op.gate, cirq.FSimGate) for op in compiled.all_operations())
    assert has_fsim


def test_openqasm3_export(tmp_path):
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit([
        cirq.H(q0),
        cirq.FSimGate(np.pi / 2, np.pi / 6).on(q0, q1),
    ])
    qasm_file = str(tmp_path / "circuit.qasm")
    qasm_str = export_to_openqasm3(c, file_path=qasm_file)
    assert "OPENQASM 3.0;" in qasm_str
    assert "fsim(" in qasm_str


def test_batch_fsim_compiler():
    batch = [
        cirq.unitary(cirq.ISWAP),
        cirq.unitary(cirq.CZ),
    ]
    compiler = BatchFSimCompiler(target_infidelity=1e-4)
    results = compiler.compile_batch(batch)
    assert len(results) == 2

    stats = compiler.compute_summary_statistics(results)
    assert stats["total_unitaries"] == 2
    assert stats["success_rate"] == 1.0
