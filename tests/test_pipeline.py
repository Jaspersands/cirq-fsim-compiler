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
    assert __version__ == "0.5.0"


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


@pytest.mark.slow
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


def test_input_fsim_is_kept_only_when_it_matches_the_edge_calibration():
    q0, q1 = cirq.LineQubit.range(2)
    syc_gate = cirq.FSimGate(np.pi / 2, np.pi / 6)
    for consolidate in (False, True):
        # a matching gate passes through
        out = compile_to_fsim(cirq.Circuit(syc_gate.on(q0, q1)), calibration=CouplerCalibration.sycamore(),
                              consolidate=consolidate)
        assert list(out.all_operations()) == [syc_gate.on(q0, q1)]
        # mismatched FSim and PhasedFSim gates are resynthesised on the calibrated gate
        for gate, cal, native in [(cirq.FSimGate(0.5, 0.2), CouplerCalibration.cz(), cirq.CZ),
                                  (syc_gate, CouplerCalibration.sqrt_iswap(), cirq.FSimGate(np.pi / 4, 0.0)),
                                  (cirq.PhasedFSimGate(0.4, 0.1, 0.2, 0.3, 0.5), CouplerCalibration.cz(), cirq.CZ)]:
            c = cirq.Circuit(gate.on(q0, q1))
            out = compile_to_fsim(c, calibration=cal, target_infidelity=1e-9, consolidate=consolidate)
            two = [op for op in out.all_operations() if len(op.qubits) == 2]
            assert two and all(np.allclose(cirq.unitary(op.gate), cirq.unitary(native)) for op in two), (gate, consolidate)
            assert fid(cirq.unitary(out), cirq.unitary(c)) > 1 - 1e-8


def test_mid_circuit_measurement_is_rejected():
    q0, q1 = cirq.LineQubit.range(2)
    mid = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key="m"), cirq.CNOT(q0, q1))
    with pytest.raises(ValueError, match="measurement"):
        compile_to_fsim(mid)
    end = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m"))
    out = compile_to_fsim(end, calibration=CouplerCalibration.cz())
    assert out.are_all_measurements_terminal() and cirq.is_measurement(list(out.all_operations())[-1])


@pytest.mark.slow
def test_every_suite_circuit_compiles_equivalently_in_exact_mode():
    from cirq_fsim_compiler.compare import GATE_SETS, calibration_for, circuit_suite
    for name, c in circuit_suite().items():
        qs = sorted(c.all_qubits())
        ideal = cirq.unitary(c)
        for gs in GATE_SETS:
            tr = FSimDecomposerTransformer(target_infidelity=1e-9, calibration=calibration_for(gs), on_failure="raise")
            out = tr.optimize_circuit(c)
            assert fid(out.unitary(qubit_order=qs), ideal) > 1 - 1e-8, (name, gs)


def test_classically_controlled_and_non_unitary_operations_are_rejected():
    q0, q1 = cirq.LineQubit.range(2)
    for op in (cirq.X(q1).with_classical_controls("a"), cirq.CNOT(q0, q1).with_classical_controls("a")):
        c = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key="a"), op)
        with pytest.raises(ValueError, match="classically controlled"):
            compile_to_fsim(c)
    with pytest.raises(ValueError, match="not a unitary"):
        compile_to_fsim(cirq.Circuit(cirq.H(q0), cirq.depolarize(0.1).on(q0), cirq.CNOT(q0, q1)))


def test_mid_circuit_reset_is_kept_in_place():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.reset(q0), cirq.H(q0), cirq.CNOT(q0, q1))
    out = compile_to_fsim(c, calibration=CouplerCalibration.cz(), target_infidelity=1e-10)
    sim = cirq.DensityMatrixSimulator()
    rho = lambda circ: sim.simulate(circ, qubit_order=[q0, q1]).final_density_matrix
    assert np.allclose(rho(out), rho(c), atol=1e-5)


def test_calibrations_back_to_back_use_the_right_native_gate():
    q0, q1 = cirq.LineQubit.range(2)
    from cirq_fsim_compiler.unitary_ansatz import fsim_matrix_np, haar_random_unitary
    target = cirq.MatrixGate(haar_random_unitary(4, seed=12)).on(q0, q1)
    for cal in (CouplerCalibration.sycamore(), CouplerCalibration(theta_cal=1.5, phi_cal=0.6),
                CouplerCalibration.sqrt_iswap(), CouplerCalibration.sycamore()):
        out = compile_to_fsim(cirq.Circuit(target), calibration=cal, target_infidelity=1e-9)
        two = [op for op in out.all_operations() if len(op.qubits) == 2]
        assert two and all(np.allclose(cirq.unitary(op.gate), fsim_matrix_np(*cal.angles)) for op in two), cal.angles
        assert fid(cirq.unitary(out), cirq.unitary(target)) > 1 - 1e-8
