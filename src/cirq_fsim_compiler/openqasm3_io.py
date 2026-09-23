"""
OpenQASM 3.0 export / import for FSim circuits.

Export
------
The exported program defines ``gate fsim(theta, phi) a, b`` in terms of
``stdgates.inc`` (so any OpenQASM 3 consumer can simulate it):

    FSim(θ, φ) = exp(−iθ/2 (XX + YY)) · exp(−iφ |11⟩⟨11|)
               = [H⊗H · e^{−iθ ZZ/2} · H⊗H] · [(SH)⊗(SH) · e^{−iθ ZZ/2} · (SH)†⊗(SH)†] · cp(−φ)

with e^{−iθ ZZ/2} = cx a,b; rz(θ) b; cx a,b. An optional ``defcal`` block
marks the gate as a native pulse-level primitive for OpenPulse-aware
compilers. Measurements become one ``bit[k]`` register per measurement key.

Import
------
:func:`parse_openqasm3` is a small recursive-descent parser for the part of
OpenQASM 3 that gate-level circuits use:

* ``qubit[n] name;`` / ``qubit name;`` / ``bit[n] name;`` (and OpenQASM 2
  ``qreg`` / ``creg``), any register names, several registers;
* the gates of ``stdgates.inc`` (``u``, ``p``, ``rx…rz``, ``sx``, ``cx``, ``cy``,
  ``cz``, ``cp``, ``crx…crz``, ``ch``, ``swap``, ``ccx``, ``cswap`` …) plus ``fsim``;
* user ``gate`` definitions, inlined with parameter substitution (bodies may
  contain nested braces and calls to other user gates);
* register broadcasting (``h q;``), ``measure`` in both syntaxes, ``reset``,
  ``barrier`` (ignored), ``defcal`` / ``cal`` blocks (skipped);
* parameter expressions with + − * / ** ^, unary minus, parentheses,
  ``pi``/``π``, ``tau``, ``euler`` and sin/cos/tan/exp/ln/sqrt/arcsin/arccos/arctan.

Expressions are evaluated by the parser itself (never ``eval``); anything
outside this grammar raises :class:`QasmParseError` with the line number.
Classical control flow (``if``, ``for``, ``while``) is out of scope.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _op_to_qasm(op: "cirq.Operation", qmap: Dict["cirq.Qid", str], expand_fsim: bool,
                bits: Optional[Dict["cirq.Qid", str]] = None) -> List[str]:
    g = op.gate
    qs = [qmap[q] for q in op.qubits]
    if isinstance(g, cirq.IdentityGate):
        return []
    if isinstance(g, cirq.MeasurementGate):
        return [f"{bits[q]} = measure {qmap[q]};" for q in op.qubits]
    if isinstance(g, cirq.ResetChannel):
        return [f"reset {qs[0]};"]
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


def _bit_register_name(key: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", key)
    return name if re.match(r"[A-Za-z_]", name) else f"m_{name}"


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
    lines.append(f"qubit[{n_declared}] q;")
    # one bit register per measurement key, one bit per measured qubit
    registers: Dict[str, int] = {}
    op_bits: Dict[int, Dict["cirq.Qid", str]] = {}
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.MeasurementGate):
            name = _bit_register_name(str(cirq.measurement_key_name(op)))
            registers[name] = max(registers.get(name, 0), len(op.qubits))
            op_bits[id(op)] = {q: f"{name}[{k}]" for k, q in enumerate(op.qubits)}
    lines += [f"bit[{size}] {name};" for name, size in registers.items()]
    lines.append("")
    for op in circuit.all_operations():
        lines += _op_to_qasm(op, qmap, expand_fsim, op_bits.get(id(op)))
    text = "\n".join(lines) + "\n"
    if file_path:
        with open(file_path, "w") as f:
            f.write(text)
    return text


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
class QasmParseError(ValueError):
    """Raised for OpenQASM input outside the supported grammar (message includes the line)."""


_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<num>(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?)
  | (?P<str>"[^"]*")
  | (?P<id>[A-Za-z_πτ][A-Za-z_0-9]*|\$\d+)
  | (?P<op>->|\*\*|==|[-+*/^(){}\[\],;=@:])
""", re.VERBOSE | re.DOTALL)

_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "ln": np.log, "sqrt": np.sqrt,
          "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan}
_CONSTS = {"pi": np.pi, "π": np.pi, "tau": 2 * np.pi, "τ": 2 * np.pi, "euler": np.e}

Token = Tuple[str, str, int]


def _tokenize(text: str) -> List[Token]:
    out: List[Token] = []
    pos, line = 0, 1
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise QasmParseError(f"line {line}: unexpected character {text[pos]!r}")
        kind = m.lastgroup
        val = m.group(kind)
        if kind not in ("ws", "comment"):
            out.append((kind, val, line))
        line += val.count("\n")
        pos = m.end()
    out.append(("eof", "", line))
    return out


def _u3(t: float, p: float, lam: float) -> "cirq.Gate":
    return cirq.MatrixGate(np.array([[np.cos(t / 2), -np.exp(1j * lam) * np.sin(t / 2)],
                                     [np.exp(1j * p) * np.sin(t / 2), np.exp(1j * (p + lam)) * np.cos(t / 2)]]))


def _std_gates() -> Dict[str, Tuple[int, int, Callable[[List[float]], "cirq.Gate"]]]:
    """name → (number of parameters, number of qubits, factory)."""
    ctrl = lambda g: g.controlled(1)
    phase = lambda a: cirq.ZPowGate(exponent=a[0] / np.pi)
    cphase = lambda a: cirq.CZPowGate(exponent=a[0] / np.pi)
    u = lambda a: _u3(*a)
    return {
        "id": (0, 1, lambda a: cirq.I), "x": (0, 1, lambda a: cirq.X), "y": (0, 1, lambda a: cirq.Y),
        "z": (0, 1, lambda a: cirq.Z), "h": (0, 1, lambda a: cirq.H), "s": (0, 1, lambda a: cirq.S),
        "sdg": (0, 1, lambda a: cirq.S**-1), "t": (0, 1, lambda a: cirq.T), "tdg": (0, 1, lambda a: cirq.T**-1),
        "sx": (0, 1, lambda a: cirq.XPowGate(exponent=0.5)),
        "rx": (1, 1, lambda a: cirq.rx(a[0])), "ry": (1, 1, lambda a: cirq.ry(a[0])), "rz": (1, 1, lambda a: cirq.rz(a[0])),
        "p": (1, 1, phase), "phase": (1, 1, phase), "u": (3, 1, u), "U": (3, 1, u), "u3": (3, 1, u),
        "cx": (0, 2, lambda a: cirq.CNOT), "CX": (0, 2, lambda a: cirq.CNOT), "cy": (0, 2, lambda a: ctrl(cirq.Y)),
        "cz": (0, 2, lambda a: cirq.CZ), "ch": (0, 2, lambda a: ctrl(cirq.H)), "cp": (1, 2, cphase), "cphase": (1, 2, cphase),
        "crx": (1, 2, lambda a: ctrl(cirq.rx(a[0]))), "cry": (1, 2, lambda a: ctrl(cirq.ry(a[0]))),
        "crz": (1, 2, lambda a: ctrl(cirq.rz(a[0]))),
        "swap": (0, 2, lambda a: cirq.SWAP), "iswap": (0, 2, lambda a: cirq.ISWAP),
        "ccx": (0, 3, lambda a: cirq.CCX), "cswap": (0, 3, lambda a: cirq.CSWAP),
        "fsim": (2, 2, lambda a: cirq.FSimGate(theta=a[0], phi=a[1])),
    }


class _Parser:
    def __init__(self, tokens: List[Token], qubit_factory: Callable[[int], "cirq.Qid"], native_gates: Tuple[str, ...] = ("fsim",)):
        self.native = set(native_gates)
        self.toks = tokens
        self.i = 0
        self.std = _std_gates()
        self.user: Dict[str, Tuple[List[str], List[str], List[Token]]] = {}
        self.qregs: Dict[str, List["cirq.Qid"]] = {}
        self.cregs: Dict[str, int] = {}
        self.ops: List["cirq.Operation"] = []
        self.qubit_factory = qubit_factory
        self.n_qubits = 0

    # -- tokens ------------------------------------------------------------ #
    def peek(self) -> Token:
        return self.toks[self.i]

    def next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, val: str) -> Token:
        t = self.next()
        if t[1] != val:
            raise QasmParseError(f"line {t[2]}: expected {val!r}, found {t[1]!r}")
        return t

    def skip_statement(self) -> None:
        while self.next()[1] != ";":
            if self.peek()[0] == "eof":
                raise QasmParseError(f"line {self.peek()[2]}: missing ';'")

    def skip_block(self) -> List[Token]:
        """Consume a balanced { … } block (nested braces allowed) and return its inner tokens."""
        self.expect("{")
        start, depth = self.i, 1
        while depth:
            t = self.next()
            if t[0] == "eof":
                raise QasmParseError(f"line {t[2]}: unterminated block")
            depth += (t[1] == "{") - (t[1] == "}")
        return self.toks[start: self.i - 1]

    # -- expressions ------------------------------------------------------- #
    def expr(self, env: Dict[str, float]) -> float:
        v = self.term(env)
        while self.peek()[1] in ("+", "-"):
            op = self.next()[1]
            r = self.term(env)
            v = v + r if op == "+" else v - r
        return v

    def term(self, env: Dict[str, float]) -> float:
        v = self.unary(env)
        while self.peek()[1] in ("*", "/"):
            op = self.next()[1]
            r = self.unary(env)
            v = v * r if op == "*" else v / r
        return v

    def unary(self, env: Dict[str, float]) -> float:
        # unary minus binds looser than ** (−b**2 = −(b**2)), as in OpenQASM 3 and Python
        if self.peek()[1] in ("-", "+"):
            sign = -1.0 if self.next()[1] == "-" else 1.0
            return sign * self.unary(env)
        return self.power(env)

    def power(self, env: Dict[str, float]) -> float:
        v = self.atom(env)
        if self.peek()[1] in ("**", "^"):
            self.next()
            return v ** self.unary(env)                       # right associative, 2**-1 allowed
        return v

    def atom(self, env: Dict[str, float]) -> float:
        kind, val, line = self.next()
        if kind == "num":
            return float(val)
        if val == "(":
            v = self.expr(env)
            self.expect(")")
            return v
        if kind == "id":
            if val in env:
                return env[val]
            if val in _CONSTS:
                return _CONSTS[val]
            if val in _FUNCS and self.peek()[1] == "(":
                self.next()
                v = self.expr(env)
                self.expect(")")
                return float(_FUNCS[val](v))
        raise QasmParseError(f"line {line}: cannot evaluate {val!r} in an expression")

    def int_expr(self) -> int:
        line = self.peek()[2]
        v = self.expr({})
        if abs(v - round(v)) > 1e-9:
            raise QasmParseError(f"line {line}: expected an integer, got {v}")
        return int(round(v))

    # -- operands ---------------------------------------------------------- #
    def qubit_operand(self) -> List["cirq.Qid"]:
        kind, name, line = self.next()
        if kind != "id":
            raise QasmParseError(f"line {line}: expected a qubit, found {name!r}")
        if name.startswith("$"):
            return [self.qubit_factory(int(name[1:]))]
        if name not in self.qregs:
            raise QasmParseError(f"line {line}: unknown qubit register {name!r}")
        reg = self.qregs[name]
        if self.peek()[1] == "[":
            self.next()
            idx = self.int_expr()
            self.expect("]")
            if not 0 <= idx < len(reg):
                raise QasmParseError(f"line {line}: index {idx} out of range for {name}[{len(reg)}]")
            return [reg[idx]]
        return list(reg)

    def bit_operand(self) -> Tuple[str, Optional[int]]:
        kind, name, line = self.next()
        if name not in self.cregs:
            raise QasmParseError(f"line {line}: unknown bit register {name!r}")
        if self.peek()[1] == "[":
            self.next()
            idx = self.int_expr()
            self.expect("]")
            if not 0 <= idx < self.cregs[name]:
                raise QasmParseError(f"line {line}: index {idx} out of range for {name}[{self.cregs[name]}]")
            return name, idx
        return name, None

    # -- statements -------------------------------------------------------- #
    def parse(self) -> "cirq.Circuit":
        while self.peek()[0] != "eof":
            self.statement()
        return cirq.Circuit(self.ops)

    def declare(self, kind: str, name: str, size: int, line: int) -> None:
        if name in self.qregs or name in self.cregs:
            raise QasmParseError(f"line {line}: {name!r} declared twice")
        if kind == "qubit":
            self.qregs[name] = [self.qubit_factory(self.n_qubits + k) for k in range(size)]
            self.n_qubits += size
        else:
            self.cregs[name] = size

    def statement(self) -> None:
        kind, val, line = self.peek()
        if val in ("OPENQASM", "include", "barrier"):
            self.skip_statement()
        elif val in ("qubit", "bit"):
            self.next()
            size = 1
            if self.peek()[1] == "[":
                self.next()
                size = self.int_expr()
                self.expect("]")
            name = self.next()[1]
            if self.peek()[1] == "=":
                raise QasmParseError(f"line {line}: initialised declarations are not supported")
            self.expect(";")
            self.declare(val, name, size, line)
        elif val in ("qreg", "creg"):
            self.next()
            name = self.next()[1]
            self.expect("[")
            size = self.int_expr()
            self.expect("]")
            self.expect(";")
            self.declare("qubit" if val == "qreg" else "bit", name, size, line)
        elif val == "gate":
            self.gate_definition()
        elif val in ("defcal", "cal", "defcalgrammar"):
            self.next()
            while self.peek()[1] not in ("{", ";"):
                self.next()
            if self.peek()[1] == "{":
                self.skip_block()
            else:
                self.next()
        elif val == "reset":
            self.next()
            for q in self.qubit_operand():
                self.ops.append(cirq.ResetChannel().on(q))
            self.expect(";")
        elif val == "measure":                                   # measure q[0] -> c[0];
            self.next()
            qs = self.qubit_operand()
            self.expect("->")
            reg, idx = self.bit_operand()
            self.expect(";")
            self.measure(qs, reg, idx, line)
        elif kind == "id" and val in self.cregs:                 # c[0] = measure q[0];
            reg, idx = self.bit_operand()
            self.expect("=")
            self.expect("measure")
            qs = self.qubit_operand()
            self.expect(";")
            self.measure(qs, reg, idx, line)
        elif val in ("if", "for", "while", "def", "input", "output", "ctrl", "inv", "pow", "negctrl"):
            raise QasmParseError(f"line {line}: {val!r} is outside the supported gate-level subset")
        else:
            self.gate_call({}, None)

    def measure(self, qs: List["cirq.Qid"], reg: str, idx: Optional[int], line: int) -> None:
        idxs = [idx] if idx is not None else list(range(self.cregs[reg]))
        if len(idxs) != len(qs):
            raise QasmParseError(f"line {line}: measuring {len(qs)} qubits into {len(idxs)} bits")
        for q, k in zip(qs, idxs):
            self.ops.append(cirq.measure(q, key=f"{reg}[{k}]"))

    def gate_definition(self) -> None:
        self.expect("gate")
        name = self.next()[1]
        native = name in self.native and name in self.std    # keep e.g. fsim as a native Cirq gate
        if not native:
            self.std.pop(name, None)             # otherwise the program's own definition wins
        params: List[str] = []
        if self.peek()[1] == "(":
            self.next()
            while self.peek()[1] != ")":
                params.append(self.next()[1])
                if self.peek()[1] == ",":
                    self.next()
            self.expect(")")
        args: List[str] = []
        while self.peek()[1] != "{":
            t = self.next()
            if t[0] == "eof":
                raise QasmParseError(f"line {t[2]}: gate {name!r} has no body")
            if t[1] != ",":
                args.append(t[1])
        body = self.skip_block()
        if not native:
            self.user[name] = (params, args, body)

    def gate_call(self, env: Dict[str, float], local: Optional[Dict[str, "cirq.Qid"]]) -> None:
        kind, name, line = self.next()
        if name in ("{", "}"):                                   # bare braces inside a gate body
            return
        if kind != "id":
            raise QasmParseError(f"line {line}: unexpected {name!r}")
        vals: List[float] = []
        if self.peek()[1] == "(":
            self.next()
            while self.peek()[1] != ")":
                vals.append(self.expr(env))
                if self.peek()[1] == ",":
                    self.next()
            self.expect(")")
        operands: List[List["cirq.Qid"]] = []
        while True:
            if local is not None:
                t = self.next()
                if t[1] not in local:
                    raise QasmParseError(f"line {t[2]}: unknown gate argument {t[1]!r}")
                operands.append([local[t[1]]])
            else:
                operands.append(self.qubit_operand())
            if self.peek()[1] != ",":
                break
            self.next()
        self.expect(";")
        width = max(len(o) for o in operands)                    # register broadcasting
        if any(len(o) not in (1, width) for o in operands):
            raise QasmParseError(f"line {line}: registers of different sizes in one gate call")
        for k in range(width):
            self.apply(name, vals, [o[k] if len(o) > 1 else o[0] for o in operands], line)

    def apply(self, name: str, vals: List[float], qs: List["cirq.Qid"], line: int) -> None:
        if len(set(qs)) != len(qs):
            raise QasmParseError(f"line {line}: repeated qubit in {name}")
        if name in self.user:
            params, args, body = self.user[name]
            if len(params) != len(vals) or len(args) != len(qs):
                raise QasmParseError(f"line {line}: {name} expects {len(params)} parameters and {len(args)} qubits")
            sub = _Parser(body + [("eof", "", line)], self.qubit_factory, tuple(self.native))
            sub.std, sub.user, sub.qregs, sub.cregs, sub.ops = self.std, self.user, self.qregs, self.cregs, self.ops
            env, local = dict(zip(params, vals)), dict(zip(args, qs))
            while sub.peek()[0] != "eof":
                sub.gate_call(env, local)
            return
        if name not in self.std:
            raise QasmParseError(f"line {line}: unknown gate {name!r}")
        n_par, n_q, make = self.std[name]
        if len(vals) != n_par or len(qs) != n_q:
            raise QasmParseError(f"line {line}: {name} expects {n_par} parameters and {n_q} qubits")
        gate = make(vals)
        if gate is not cirq.I:
            self.ops.append(gate.on(*qs))


def parse_openqasm3(text: str, qubit_factory: Optional[Callable[[int], "cirq.Qid"]] = None,
                    native_gates: Tuple[str, ...] = ("fsim",)) -> "cirq.Circuit":
    """
    Parse OpenQASM 3 (grammar in the module docstring) into a Cirq circuit.
    Qubits of all registers are numbered in declaration order and created with
    ``qubit_factory(k)`` (default ``cirq.LineQubit(k)``). Measurement keys are
    ``"<register>[<index>]"``. Gates named in ``native_gates`` become native Cirq
    gates even when the program defines them (the exporter's ``gate fsim`` body is
    a simulation fallback); pass ``native_gates=()`` to inline every definition.
    """
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")
    return _Parser(_tokenize(text), qubit_factory or cirq.LineQubit, tuple(native_gates)).parse()
