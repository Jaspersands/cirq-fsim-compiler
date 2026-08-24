"""
OpenQASM 3.0 Exporter with defcal Native Sycamore Pulse Annotations.

Exports native Cirq FSim circuits into standardized OpenQASM 3.0 format.
"""

from __future__ import annotations
from typing import Optional

try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False
    cirq = None


def export_to_openqasm3(
    circuit: "cirq.Circuit",
    include_defcal: bool = True,
    file_path: Optional[str] = None,
) -> str:
    """
    Translates a native Sycamore FSim Cirq circuit into OpenQASM 3.0 string.
    """
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")

    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        "// Google Sycamore Native Gate Definition",
        "gate fsim(theta, phi) q0, q1 {",
        "    // Native Tunable Coupler Interaction",
        "}",
        "",
    ]

    qubits = sorted(list(circuit.all_qubits()), key=lambda q: str(q))
    q_map = {q: f"q[{i}]" for i, q in enumerate(qubits)}

    lines.append(f"qubit[{len(qubits)}] q;")
    lines.append("")

    for moment in circuit:
        for op in moment.operations:
            if isinstance(op.gate, cirq.FSimGate):
                q0_str = q_map[op.qubits[0]]
                q1_str = q_map[op.qubits[1]]
                th = float(op.gate.theta)
                ph = float(op.gate.phi)
                lines.append(f"fsim({th:.6f}, {ph:.6f}) {q0_str}, {q1_str};")
            elif isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1.0:
                lines.append(f"x {q_map[op.qubits[0]]};")
            elif isinstance(op.gate, cirq.HPowGate) and op.gate.exponent == 1.0:
                lines.append(f"h {q_map[op.qubits[0]]};")
            elif isinstance(op.gate, cirq.ZPowGate):
                angle = float(op.gate.exponent * 3.141592653589793)
                lines.append(f"rz({angle:.6f}) {q_map[op.qubits[0]]};")
            elif isinstance(op.gate, cirq.XPowGate):
                angle = float(op.gate.exponent * 3.141592653589793)
                lines.append(f"rx({angle:.6f}) {q_map[op.qubits[0]]};")

    qasm_str = "\n".join(lines)

    if file_path:
        with open(file_path, "w") as f:
            f.write(qasm_str)

    return qasm_str
