"""
Tests for the Cirq transformer, three-qubit decompositions, OpenQASM 3 I/O,
batch compilation and the CLI.
"""

import json

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

from cirq_fsim_compiler import __version__
from cirq_fsim_compiler.calibration_map import SycamoreCalibrationMap, CouplerCalibration
from cirq_fsim_compiler.cirq_transformer import (
    FSimDecomposerTransformer,
    compile_circuit_to_sycamore_fsim,
    compile_to_fsim,
    convert_decomposition_to_cirq_ops,
)
from cirq_fsim_compiler.toffoli_decomposer import (
    decompose_toffoli_to_sycamore,
    decompose_fredkin_to_sycamore,
    decompose_ccz_to_sycamore,
    decompose_qft3_to_sycamore,
    qft_unitary,
)
from cirq_fsim_compiler.openqasm3_io import export_to_openqasm3, parse_openqasm3
from cirq_fsim_compiler.batch_compiler import BatchFSimCompiler
from cirq_fsim_compiler.riemannian_optimizer import synthesize_unitary_to_fsim
from cirq_fsim_compiler.cli import main as cli_main


def fid(u, v):
    d = u.shape[0]
    return abs(np.trace(u.conj().T @ v)) ** 2 / d**2


def no_three_qubit_ops(circuit):
    return all(len(op.qubits) <= 2 for op in circuit.all_operations())


def has_fsim(circuit):
    return any(isinstance(op.gate, (cirq.FSimGate, cirq.PhasedFSimGate)) for op in circuit.all_operations())


def test_version():
    assert __version__ == "0.3.0"


def test_decomposition_to_cirq_ops_reproduces_unitary():
    q0, q1 = cirq.LineQubit.range(2)
    res = synthesize_unitary_to_fsim(cirq.unitary(cirq.CNOT), target_infidelity=1e-9)
    ops = convert_decomposition_to_cirq_ops(res, q0, q1)
    u = cirq.unitary(cirq.Circuit(ops))
    assert fid(u, cirq.unitary(cirq.CNOT)) > 1 - 1e-8


def test_transformer_compiles_two_qubit_gates_and_keeps_native():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.FSimGate(0.3, 0.1).on(q1, q2), cirq.ISWAP(q0, q2)])
    compiled = compile_circuit_to_sycamore_fsim(c, target_infidelity=1e-8)
    assert has_fsim(compiled)
    assert fid(cirq.unitary(compiled), cirq.unitary(c)) > 1 - 1e-7
    # the pre-existing native FSim gate passes through untouched
    assert any(isinstance(op.gate, cirq.FSimGate) and abs(op.gate.theta - 0.3) < 1e-12 for op in compiled.all_operations())


def test_transformer_decomposes_three_qubit_gates():
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.CCNOT(*q))
    compiled = compile_circuit_to_sycamore_fsim(c, target_infidelity=1e-8)
    assert no_three_qubit_ops(compiled)
    assert fid(cirq.unitary(compiled), cirq.unitary(c)) > 1 - 1e-6


def test_transformer_is_a_cirq_transformer_with_context():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CZ(q0, q1))
    out = compile_to_fsim(c, context=cirq.TransformerContext())
    assert has_fsim(out)


def test_transformer_uses_calibration_map():
    q0, q1 = cirq.LineQubit.range(2)
    cm = SycamoreCalibrationMap(seed=1)
    cm.set_coupler(0, 1, CouplerCalibration(theta_cal=1.5, phi_cal=0.4))
    c = cirq.Circuit(cirq.CZ(q0, q1))
    tr = FSimDecomposerTransformer(target_infidelity=1e-8, calibration_map=cm)
    compiled = tr(c)
    fsims = [op.gate for op in compiled.all_operations() if isinstance(op.gate, cirq.FSimGate)]
    assert fsims and all(abs(g.theta - 1.5) < 1e-12 and abs(g.phi - 0.4) < 1e-12 for g in fsims)
    assert fid(cirq.unitary(compiled), cirq.unitary(c)) > 1 - 1e-6


@pytest.mark.parametrize(
    "fn,reference",
    [
        (decompose_toffoli_to_sycamore, lambda q: cirq.unitary(cirq.CCNOT(*q))),
        (decompose_fredkin_to_sycamore, lambda q: cirq.unitary(cirq.CSWAP(*q))),
        (decompose_ccz_to_sycamore, lambda q: cirq.unitary(cirq.CCZ(*q))),
        (decompose_qft3_to_sycamore, lambda q: qft_unitary(3)),
    ],
)
def test_three_qubit_decompositions(fn, reference):
    q = cirq.LineQubit.range(3)
    c = fn(*q, target_infidelity=1e-8)
    assert no_three_qubit_ops(c)
    assert has_fsim(c)
    assert fid(cirq.unitary(c), reference(q)) > 1 - 1e-5


def test_qft_unitary_is_the_dft():
    u = qft_unitary(3)
    n = 8
    w = np.exp(2j * np.pi / n)
    ref = np.array([[w ** (j * k) for k in range(n)] for j in range(n)]) / np.sqrt(n)
    assert np.allclose(u, ref)


def test_openqasm3_export_and_roundtrip(tmp_path):
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])
    compiled = compile_circuit_to_sycamore_fsim(c, target_infidelity=1e-9)
    qasm = export_to_openqasm3(compiled, file_path=str(tmp_path / "c.qasm"))
    assert qasm.startswith("OPENQASM 3.0;")
    assert "gate fsim(theta, phi) a, b {" in qasm and "cp(-phi) a, b;" in qasm
    assert "defcal" in qasm
    back = parse_openqasm3(qasm)
    assert fid(cirq.unitary(back), cirq.unitary(compiled)) > 1 - 1e-9
    assert fid(cirq.unitary(back), cirq.unitary(c)) > 1 - 1e-8


def test_openqasm3_fsim_gate_definition_is_correct():
    # The stdgates body of `gate fsim` must equal cirq.FSimGate for arbitrary angles.
    q0, q1 = cirq.LineQubit.range(2)
    theta, phi = 0.73, -1.1
    text = export_to_openqasm3(cirq.Circuit(cirq.FSimGate(theta, phi).on(q0, q1)), expand_fsim=True)
    assert "fsim(" not in text.split("qubit[2] q;")[1]   # expanded into std gates
    back = parse_openqasm3(text)
    assert fid(cirq.unitary(back), cirq.unitary(cirq.FSimGate(theta, phi))) > 1 - 1e-9


def test_openqasm3_handles_common_gates_and_rejects_unknown():
    q = cirq.LineQubit.range(11)
    c = cirq.Circuit([cirq.X(q[10]), cirq.Y(q[2]), cirq.Z(q[0]), cirq.S(q[1]), cirq.T(q[1]), cirq.rx(0.2)(q[3]),
                      cirq.ry(0.3)(q[4]), cirq.rz(0.4)(q[5]), cirq.CZ(q[2], q[10]), cirq.CNOT(q[0], q[1]),
                      cirq.PhasedXZGate(x_exponent=0.3, z_exponent=0.2, axis_phase_exponent=0.1).on(q[6])])
    text = export_to_openqasm3(c)
    lines = text.splitlines()
    assert "qubit[11] q;" in lines                            # LineQubit indices are preserved
    assert any(l.startswith("x q[10];") for l in lines)      # q(10) stays q[10], not sorted as a string
    back = parse_openqasm3(text)
    assert fid(cirq.unitary(back), cirq.unitary(c)) > 1 - 1e-9
    with pytest.raises(NotImplementedError):
        export_to_openqasm3(cirq.Circuit(cirq.MatrixGate(np.eye(4)).on(q[0], q[1])))


def test_batch_compiler_stats_and_workers():
    batch = [cirq.unitary(cirq.ISWAP), cirq.unitary(cirq.CZ), cirq.unitary(cirq.SWAP)]
    comp = BatchFSimCompiler(target_infidelity=1e-7, n_workers=2)
    results = comp.compile_batch(batch)
    stats = comp.compute_summary_statistics(results)
    assert stats["total_unitaries"] == 3 and stats["success_rate"] == 1.0
    assert sum(stats["stage_histogram"].values()) == 3
    assert stats["total_fsim_gates"] == 3       # each is a single free FSim: SWAP = FSim(π/2, π) up to local Z
    assert stats["max_infidelity"] < 1e-6
    native = BatchFSimCompiler(target_infidelity=1e-7, calibration=CouplerCalibration.sycamore()).compile_batch(batch)
    nstats = BatchFSimCompiler.compute_summary_statistics(native)
    assert nstats["n_native"] == 3 and nstats["stage_histogram"][3] >= 1   # SWAP needs 3 native Sycamore gates


def test_cli_synthesize_and_toffoli(capsys, tmp_path):
    assert cli_main(["synthesize", "--gate", "cz", "--native", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["native"] is True and out["n_stages"] == 2 and out["infidelity"] < 1e-6
    assert cli_main(["synthesize", "--gate", "cnot", "--method", "riemannian", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "riemannian" and out["infidelity"] < 1e-6
    qasm_path = tmp_path / "t.qasm"
    assert cli_main(["toffoli", "--qasm-out", str(qasm_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fidelity"] > 1 - 1e-5 and qasm_path.exists()
