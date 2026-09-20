"""
OpenQASM 3.0 export / import for FSim circuits.

The exported program defines ``gate fsim(theta, phi) a, b`` in terms of
``stdgates.inc`` (so any OpenQASM 3 consumer can simulate it):

    FSim(θ, φ) = exp(−iθ/2 (XX + YY)) · exp(−iφ |11⟩⟨11|)
               = [H⊗H · e^{−iθ ZZ/2} · H⊗H] · [(SH)⊗(SH) · e^{−iθ ZZ/2} · (SH)†⊗(SH)†] · cp(−φ)

with e^{−iθ ZZ/2} = cx a,b; rz(θ) b; cx a,b. An optional ``defcal`` block
marks the gate as a native pulse-level primitive for OpenPulse-aware
compilers. :func:`parse_openqasm3` reads programs written with this subset
back into Cirq (used for round-trip tests).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

try:
    import cirq

    HAS_CIRQ = True
except ImportError:  # pragma: no cover
    HAS_CIRQ = False
    cirq = None

FSIM_GATE_DEF = """// Native tunable-coupler interaction, expressed with stdgates so that any
// OpenQASM 3 toolchain can simulate it. FSim(theta, phi) = exp(-i theta/2 (XX+YY)) cp(-phi).
gate fsim(theta, phi) a, b {
  h a; h b; cx a, b; rz(theta) b; cx a, b; h a; h b;
  sdg a; sdg b; h a; h b; cx a, b; rz(theta) b; cx a, b; h a; h b; s a; s b;
  cp(-phi) a, b;
}"""

FSIM_DEFCAL = """// Pulse-level binding (OpenPulse). Replace the body with the coupler's calibrated
// flux/microwave sequence on the target device.
defcal fsim(theta, phi) $0, $1 {
  // play(frame: coupler_frame, waveform: fsim_pulse(theta, phi));
}"""

_PI = np.pi


def _fmt(x: float) -> str:
    return f"{float(x):.10g}"


def _op_to_qasm(op: "cirq.Operation", qmap: Dict["cirq.Qid", str], expand_fsim: bool) -> List[str]:
    g = op.gate
    qs = [qmap[q] for q in op.qubits]
    if isinstance(g, cirq.IdentityGate):
        return []
    if isinstance(g, cirq.MeasurementGate):
        raise NotImplementedError("measurements are not exported; strip them before export")
    if isinstance(g, cirq.HPowGate) and abs(g.exponent - 1) < 1e-12:
        return [f"h {qs[0]};"]
    if isinstance(g, cirq.XPowGate):
        return [f"x {qs[0]};"] if abs(g.exponent - 1) < 1e-12 else [f"rx({_fmt(_PI * g.exponent)}) {qs[0]};"]
    if isinstance(g, cirq.YPowGate):
        return [f"y {qs[0]};"] if abs(g.exponent - 1) < 1e-12 else [f"ry({_fmt(_PI * g.exponent)}) {qs[0]};"]
    if isinstance(g, cirq.ZPowGate):
        e = g.exponent % 2
        for val, name in ((1, "z"), (0.5, "s"), (1.5, "sdg"), (0.25, "t"), (1.75, "tdg")):
            if abs(e - val) < 1e-12:
                return [f"{name} {qs[0]};"]
        return [f"rz({_fmt(_PI * g.exponent)}) {qs[0]};"]
    if isinstance(g, cirq.PhasedXZGate):
        a, x, z = g.axis_phase_exponent, g.x_exponent, g.z_exponent
        return [f"rz({_fmt(-_PI * a)}) {qs[0]};", f"rx({_fmt(_PI * x)}) {qs[0]};", f"rz({_fmt(_PI * (a + z))}) {qs[0]};"]
    if isinstance(g, cirq.CZPowGate):
        return [f"cz {qs[0]}, {qs[1]};"] if abs(g.exponent - 1) < 1e-12 else [f"cp({_fmt(_PI * g.exponent)}) {qs[0]}, {qs[1]};"]
    if isinstance(g, cirq.CXPowGate) and abs(g.exponent - 1) < 1e-12:
        return [f"cx {qs[0]}, {qs[1]};"]
    if isinstance(g, cirq.SwapPowGate) and abs(g.exponent - 1) < 1e-12:
        return [f"swap {qs[0]}, {qs[1]};"]
    if isinstance(g, cirq.PhasedFSimGate):
        if any(abs(v) > 1e-12 for v in (g.zeta, g.chi, g.gamma)):
            raise NotImplementedError("PhasedFSimGate with non-zero zeta/chi/gamma is not exported")
        g = cirq.FSimGate(theta=g.theta, phi=g.phi)
    if isinstance(g, cirq.FSimGate):
        th, ph = float(g.theta), float(g.phi)
        if not expand_fsim:
            return [f"fsim({_fmt(th)}, {_fmt(ph)}) {qs[0]}, {qs[1]};"]
        a, b = qs
        return [
            f"h {a}; h {b}; cx {a}, {b}; rz({_fmt(th)}) {b}; cx {a}, {b}; h {a}; h {b};",
            f"sdg {a}; sdg {b}; h {a}; h {b}; cx {a}, {b}; rz({_fmt(th)}) {b}; cx {a}, {b}; h {a}; h {b}; s {a}; s {b};",
            f"cp({_fmt(-ph)}) {a}, {b};",
        ]
    raise NotImplementedError(f"gate {g!r} has no OpenQASM 3 export; compile it to FSim first")


def export_to_openqasm3(
    circuit: "cirq.Circuit",
    include_defcal: bool = True,
    file_path: Optional[str] = None,
    expand_fsim: bool = False,
) -> str:
    """Export a compiled circuit. ``expand_fsim=True`` inlines FSim into std gates."""
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")
    qubits = sorted(circuit.all_qubits())
    if qubits and all(isinstance(q, cirq.LineQubit) and q.x >= 0 for q in qubits):
        # keep LineQubit indices (q[x]) so the program maps onto the same physical lines
        qmap = {q: f"q[{q.x}]" for q in qubits}
        n_declared = max(q.x for q in qubits) + 1
    else:
        qmap = {q: f"q[{i}]" for i, q in enumerate(qubits)}
        n_declared = len(qubits)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]
    if not expand_fsim:
        lines += [FSIM_GATE_DEF, ""]
        if include_defcal:
            lines += [FSIM_DEFCAL, ""]
    lines += [f"qubit[{n_declared}] q;", ""]
    for op in circuit.all_operations():
        lines += _op_to_qasm(op, qmap, expand_fsim)
    text = "\n".join(lines) + "\n"
    if file_path:
        with open(file_path, "w") as f:
            f.write(text)
    return text


# --------------------------------------------------------------------------- #
# Import (subset used by the exporter)
# --------------------------------------------------------------------------- #
_STMT = re.compile(r"^\s*([a-z]+)(?:\(([^)]*)\))?\s+(.*?);\s*$")


def parse_openqasm3(text: str) -> "cirq.Circuit":
    """Parse the OpenQASM 3 subset emitted by :func:`export_to_openqasm3` into a Cirq circuit."""
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")
    # strip comments and gate/defcal bodies
    src = re.sub(r"//[^\n]*", "", text)
    src = re.sub(r"gate\s+\w+[^{]*\{[^}]*\}", "", src)
    src = re.sub(r"defcal\s+\w+[^{]*\{[^}]*\}", "", src)
    n = int(re.search(r"qubit\[(\d+)\]\s+q;", src).group(1))
    q = cirq.LineQubit.range(n)
    ops: List["cirq.Operation"] = []

    def qb(token: str):
        return q[int(re.match(r"q\[(\d+)\]", token.strip()).group(1))]

    one = {"x": cirq.X, "y": cirq.Y, "z": cirq.Z, "h": cirq.H, "s": cirq.S, "sdg": cirq.S ** -1, "t": cirq.T, "tdg": cirq.T ** -1}
    for line in src.splitlines():
        for stmt in [s for s in line.split(";") if s.strip()]:
            m = _STMT.match(stmt + ";")
            if not m:
                continue
            name, args, targets = m.group(1), m.group(2), [t for t in m.group(3).split(",")]
            if name in ("OPENQASM", "include", "qubit"):
                continue
            vals = [float(eval(a, {"pi": np.pi, "__builtins__": {}})) for a in args.split(",")] if args else []
            if name in one:
                ops.append(one[name].on(qb(targets[0])))
            elif name == "rx":
                ops.append(cirq.rx(vals[0]).on(qb(targets[0])))
            elif name == "ry":
                ops.append(cirq.ry(vals[0]).on(qb(targets[0])))
            elif name == "rz":
                ops.append(cirq.rz(vals[0]).on(qb(targets[0])))
            elif name == "cx":
                ops.append(cirq.CNOT(qb(targets[0]), qb(targets[1])))
            elif name == "cz":
                ops.append(cirq.CZ(qb(targets[0]), qb(targets[1])))
            elif name == "cp":
                ops.append(cirq.CZPowGate(exponent=vals[0] / np.pi).on(qb(targets[0]), qb(targets[1])))
            elif name == "swap":
                ops.append(cirq.SWAP(qb(targets[0]), qb(targets[1])))
            elif name == "fsim":
                ops.append(cirq.FSimGate(theta=vals[0], phi=vals[1]).on(qb(targets[0]), qb(targets[1])))
            else:
                raise NotImplementedError(f"unsupported OpenQASM statement: {stmt.strip()}")
    return cirq.Circuit(ops)
