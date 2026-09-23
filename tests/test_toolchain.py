"""
Regression tests for the v0.3 adversarial audit: a real OpenQASM 3 parser (no
eval), measurements, block consolidation, routing onto a coupling graph and the
batch executors.
"""

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")
nx = pytest.importorskip("networkx")

from cirq_fsim_compiler.batch_compiler import BatchFSimCompiler  # noqa: E402
from cirq_fsim_compiler.calibration_map import CouplerCalibration, SycamoreCalibrationMap  # noqa: E402
from cirq_fsim_compiler.cirq_transformer import compile_circuit_to_sycamore_fsim  # noqa: E402
from cirq_fsim_compiler.openqasm3_io import QasmParseError, export_to_openqasm3, parse_openqasm3  # noqa: E402
from cirq_fsim_compiler.toffoli_decomposer import fredkin_circuit, toffoli_circuit  # noqa: E402


def fid(u, v):
    return abs(np.trace(u.conj().T @ v)) ** 2 / u.shape[0] ** 2


# --------------------------------------------------------------------------- #
# OpenQASM 3
# --------------------------------------------------------------------------- #
PROGRAM = """OPENQASM 3.0;
include "stdgates.inc";
/* user gates: nested braces, a gate calling a gate, parameter expressions */
gate myrot(a, b) x, y { rz(a/2) x; { cx x, y; } ry(-b**2 + pi/4) y; }
gate wrap(t) u, v { myrot(t, 2*t) v, u; h u; }
qubit[2] data;
qubit anc;
bit[2] c;
h data;
wrap(0.3) data[0], anc;
cp(pi/3) data[1], anc;
U(0.1, 0.2, sin(0.3)) anc;
barrier data;
c[0] = measure data[0];
measure data[1] -> c[1];
"""


def _reference():
    d0, d1, anc = cirq.LineQubit.range(3)
    a, b = 0.3, 0.6
    t, p, lam = 0.1, 0.2, np.sin(0.3)
    u3 = np.array([[np.cos(t / 2), -np.exp(1j * lam) * np.sin(t / 2)],
                   [np.exp(1j * p) * np.sin(t / 2), np.exp(1j * (p + lam)) * np.cos(t / 2)]])
    return cirq.Circuit([
        cirq.H(d0), cirq.H(d1),
        cirq.rz(a / 2)(anc), cirq.CNOT(anc, d0), cirq.ry(-(b**2) + np.pi / 4)(d0), cirq.H(d0),
        cirq.CZPowGate(exponent=1 / 3).on(d1, anc), cirq.MatrixGate(u3).on(anc),
    ])


def test_parser_handles_registers_user_gates_expressions_and_measurements():
    c = parse_openqasm3(PROGRAM)
    body = cirq.Circuit(op for op in c.all_operations() if not cirq.is_measurement(op))
    assert fid(body.unitary(qubit_order=cirq.LineQubit.range(3)), _reference().unitary(qubit_order=cirq.LineQubit.range(3))) > 1 - 1e-12
    keys = sorted(str(cirq.measurement_key_name(op)) for op in c.all_operations() if cirq.is_measurement(op))
    assert keys == ["c[0]", "c[1]"]


def test_power_binds_tighter_than_unary_minus():
    c = parse_openqasm3("qubit q; rz(-2**2) q; rx(2**-1) q;")
    ops = list(c.all_operations())
    assert np.isclose(cirq.unitary(ops[0])[1, 1] / cirq.unitary(ops[0])[0, 0], np.exp(-4j))
    assert fid(cirq.unitary(ops[1]), cirq.unitary(cirq.rx(0.5))) > 1 - 1e-12


@pytest.mark.parametrize("src,msg", [
    ('qubit[1] q; rx(__import__("os").system("true")) q[0];', "unexpected character|cannot evaluate"),
    ("qubit[1] q; rx(open) q[0];", "cannot evaluate"),
    ("qubit[2] q; foo q[0];", "unknown gate"),
    ("qubit[2] q; h q[5];", "out of range"),
    ("qubit[2] q; cx q[0], q[0];", "repeated qubit"),
    ("qubit[2] q; qubit[3] r; cx q, r;", "different sizes"),
    ("qubit[2] q; bit[2] c; if (c == 1) x q[0];", "outside the supported"),
    ("qubit[2] q; gate g a { h a; ", "unterminated"),
])
def test_parser_rejects_bad_input_with_line_numbers(src, msg):
    with pytest.raises(QasmParseError, match=msg):
        parse_openqasm3(src)


def test_openqasm2_style_registers_and_broadcast():
    c = parse_openqasm3("OPENQASM 2.0; qreg a[2]; qreg b[2]; creg m[2]; cx a, b; measure b[0] -> m[0];")
    two = [op for op in c.all_operations() if op.gate == cirq.CNOT]
    assert [(o.qubits[0].x, o.qubits[1].x) for o in two] == [(0, 2), (1, 3)]


def test_export_and_reimport_measurements_and_native_fsim():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q[0]), cirq.FSimGate(0.5, 0.2).on(*q), cirq.measure(*q, key="result"))
    text = export_to_openqasm3(c)
    assert "bit[2] result;" in text and "result[1] = measure q[1];" in text
    back = parse_openqasm3(text)
    assert any(isinstance(op.gate, cirq.FSimGate) for op in back.all_operations())   # fsim stays native
    inlined = parse_openqasm3(text, native_gates=())                                   # …or uses the body
    assert not any(isinstance(op.gate, cirq.FSimGate) for op in inlined.all_operations())
    strip = lambda circ: cirq.Circuit(op for op in circ.all_operations() if not cirq.is_measurement(op))
    assert fid(cirq.unitary(strip(inlined)), cirq.unitary(strip(c))) > 1 - 1e-9


# --------------------------------------------------------------------------- #
# consolidation and routing
# --------------------------------------------------------------------------- #
def _count_fsim(c):
    return sum(isinstance(op.gate, cirq.FSimGate) for op in c.all_operations())


def _nominal_sycamore():
    return SycamoreCalibrationMap(seed=0, theta_drift_std=0.0, phi_drift_std=0.0)


def test_consolidation_reduces_sycamore_counts():
    q = cirq.LineQubit.range(3)
    for build, reference in ((toffoli_circuit, cirq.CCNOT), (fredkin_circuit, cirq.CSWAP)):
        c = build(*q)
        merged = compile_circuit_to_sycamore_fsim(c, calibration_map=_nominal_sycamore(), target_infidelity=1e-8)
        naive = compile_circuit_to_sycamore_fsim(c, calibration_map=_nominal_sycamore(), target_infidelity=1e-8, consolidate=False)
        assert _count_fsim(merged) < _count_fsim(naive)
        assert fid(cirq.unitary(merged), cirq.unitary(reference(*q))) > 1 - 1e-6
    tof = compile_circuit_to_sycamore_fsim(toffoli_circuit(*q), calibration_map=_nominal_sycamore(), target_infidelity=1e-8)
    assert _count_fsim(tof) == 10                                  # 5 consolidated blocks × 2 Sycamores


def test_routing_puts_every_two_qubit_gate_on_a_coupler():
    q = cirq.LineQubit.range(3)
    line = nx.Graph([(q[0], q[1]), (q[1], q[2])])
    from cirq_fsim_compiler.cirq_transformer import FSimDecomposerTransformer

    c = toffoli_circuit(*q) + cirq.Circuit(cirq.measure(*q, key="m"))
    tr = FSimDecomposerTransformer(calibration_map=_nominal_sycamore(), target_infidelity=1e-8, device_graph=line)
    routed = tr.optimize_circuit(c)
    start = tr.last_routing["initial_map"]
    for op in routed.all_operations():
        if len(op.qubits) == 2:
            assert line.has_edge(*op.qubits)
    assert any(cirq.is_measurement(op) for op in routed.all_operations())
    # a line needs extra gates relative to all-to-all
    assert _count_fsim(routed) > 10
    # the routed program is the Toffoli up to the final qubit permutation: check on basis states
    sim = cirq.Simulator()
    for bits in range(8):
        init = [(bits >> (2 - k)) & 1 for k in range(3)]
        prep = cirq.Circuit(cirq.X(start[q[k]]) for k in range(3) if init[k])   # prepare on the physical start qubits
        out = sim.run(prep + routed).measurements["m"][0].tolist()
        expected = init[:2] + [init[2] ^ (init[0] & init[1])]
        assert out == expected


# --------------------------------------------------------------------------- #
# batch executors
# --------------------------------------------------------------------------- #
def test_batch_executors_agree():
    batch = [cirq.unitary(cirq.CZ), cirq.unitary(cirq.SWAP), cirq.unitary(cirq.ISWAP ** 0.5), cirq.unitary(cirq.CNOT)]
    cal = CouplerCalibration.sycamore()
    runs = {ex: BatchFSimCompiler(target_infidelity=1e-7, n_workers=2, executor=ex, calibration=cal).compile_batch(batch)
            for ex in ("serial", "thread", "process")}
    stages = {ex: [r.n_stages for r in res] for ex, res in runs.items()}
    assert stages["serial"] == stages["thread"] == stages["process"] == [2, 3, 2, 2]
    assert all(r.infidelity < 1e-6 for res in runs.values() for r in res)
    with pytest.raises(ValueError):
        BatchFSimCompiler(executor="gpu")
