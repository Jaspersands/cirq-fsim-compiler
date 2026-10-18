"""
Command-line interface: ``cirq-fsim <command>``.

synthesize   compile a standard 2-qubit gate (or a random U(4)) into FSim stages
toffoli | fredkin | ccz | qft3   compile a 3-qubit gate, optionally export OpenQASM 3
qasm         parse an OpenQASM 3 file written by this tool and re-export it
benchmark    run the benchmark suite
compare      compare with Cirq's built-in decompositions (and noise-aware synthesis)
weyl         Weyl coordinates and minimum native-gate counts of a gate
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from .riemannian_optimizer import synthesize_unitary_to_fsim
from .calibration_map import CouplerCalibration
from .unitary_ansatz import haar_random_unitary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cirq-fsim", description="FSim compiler: U(4) synthesis into free or calibrated FSim gates, Weyl-chamber counts, comparison with Cirq, OpenQASM 3")
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("synthesize", help="Synthesise a 2-qubit gate")
    s.add_argument("--gate", choices=["cnot", "iswap", "sqrt_iswap", "swap", "cz", "random"], default="cnot")
    s.add_argument("--max-stages", type=int, default=4)
    s.add_argument("--native", action="store_true", help="fix FSim to the Sycamore native gate FSim(π/2, π/6)")
    s.add_argument("--target", choices=["sycamore", "sqrt_iswap", "cz"], default=None,
                   help="fix FSim to one of the calibrated gate families")
    s.add_argument("--noise-aware", type=float, default=None, metavar="EPS",
                   help="noise-aware approximation with per-gate process infidelity EPS")
    s.add_argument("--calibrated", type=float, nargs=2, metavar=("THETA", "PHI"), help="fix FSim to these calibrated angles")
    s.add_argument("--method", choices=["euclidean", "riemannian"], default="euclidean")
    s.add_argument("--target-infidelity", type=float, default=1e-8)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--json", action="store_true")

    for name in ("toffoli", "fredkin", "ccz", "qft3"):
        p = sub.add_parser(name, help=f"Compile {name} to FSim + 1Q gates")
        p.add_argument("--qasm-out", type=str, default=None)
        p.add_argument("--native", action="store_true", help="use the Sycamore native gate on every edge")
        p.add_argument("--target-infidelity", type=float, default=1e-6)
        p.add_argument("--json", action="store_true")

    q = sub.add_parser("qasm", help="Round-trip an OpenQASM 3 file")
    q.add_argument("--in", dest="inp", required=True)
    q.add_argument("--out", dest="out", default=None)
    q.add_argument("--expand-fsim", action="store_true")
    q.add_argument("--json", action="store_true")

    b = sub.add_parser("benchmark", help="Run the benchmark suite")
    b.add_argument("--json", action="store_true")
    b.add_argument("--quick", action="store_true")

    c = sub.add_parser("compare", help="Compare with Cirq's built-in decompositions")
    c.add_argument("--quick", action="store_true", help="10 Haar targets and three circuits")
    c.add_argument("--n-haar", type=int, default=200)
    c.add_argument("--no-circuits", action="store_true")
    c.add_argument("--json", action="store_true", help="print the summaries as JSON")
    c.add_argument("--out", type=str, default=None, help="also write the full results to this JSON file")

    w = sub.add_parser("weyl", help="Weyl coordinates and minimum native-gate counts")
    w.add_argument("--gate", choices=["cnot", "iswap", "sqrt_iswap", "swap", "cz", "random"], default="cnot")
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--json", action="store_true")
    return parser


def _emit(payload: Dict[str, Any], as_json: bool, lines: List[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        print("\n".join(lines))


def cmd_synthesize(args) -> int:
    import cirq
    gates = {"cnot": cirq.CNOT, "iswap": cirq.ISWAP, "sqrt_iswap": cirq.SQRT_ISWAP, "swap": cirq.SWAP, "cz": cirq.CZ}
    u = haar_random_unitary(4, seed=args.seed) if args.gate == "random" else cirq.unitary(gates[args.gate])
    cal = None
    if args.native:
        cal = CouplerCalibration.sycamore()
    elif args.target:
        cal = getattr(CouplerCalibration, args.target)()
    elif args.calibrated:
        cal = CouplerCalibration(theta_cal=args.calibrated[0], phi_cal=args.calibrated[1])
    approx = "noise_aware" if args.noise_aware is not None else "exact"
    res = synthesize_unitary_to_fsim(u, max_stages=args.max_stages, target_infidelity=args.target_infidelity,
                                     calibration=cal, method=args.method, seed=args.seed,
                                     approximation=approx, gate_error=args.noise_aware)
    payload = {"gate": args.gate, "method": res.method, "native": res.native, "n_stages": res.n_stages,
               "fidelity": res.fidelity, "infidelity": res.infidelity, "fsim_angles": res.fsim_angles,
               "single_qubit_angles": res.single_qubit_angles, "restarts": res.n_restarts_used, "iterations": len(res.loss_history),
               "exact_n_stages": res.exact_n_stages, "approximate": res.approximate, "expected_infidelity": res.expected_infidelity}
    lines = [
        f"[*] {args.gate} → FSim ({args.method}{', native FSim(' + ', '.join(f'{a:.4f}' for a in cal.angles) + ')' if cal else ', free angles'})",
        f"[+] stages = {res.n_stages}  infidelity = {res.infidelity:.2e}  restarts = {res.n_restarts_used}",
        "[+] FSim angles: " + ", ".join(f"(θ={t:.4f}, φ={p:.4f})" for t, p in res.fsim_angles),
    ]
    if res.expected_infidelity is not None:
        lines.append(f"[+] noise-aware at ε = {res.gate_error:.2e}: exact needs {res.exact_n_stages}, chose {res.n_stages}, "
                     f"expected infidelity {res.expected_infidelity:.2e}")
    _emit(payload, args.json, lines)
    return 0


def cmd_three_qubit(args) -> int:
    import cirq
    from . import toffoli_decomposer as td
    from .openqasm3_io import export_to_openqasm3
    from .calibration_map import SycamoreCalibrationMap
    q = cirq.LineQubit.range(3)
    fn = {"toffoli": (td.decompose_toffoli_to_sycamore, lambda: cirq.unitary(cirq.CCNOT(*q))),
          "fredkin": (td.decompose_fredkin_to_sycamore, lambda: cirq.unitary(cirq.CSWAP(*q))),
          "ccz": (td.decompose_ccz_to_sycamore, lambda: cirq.unitary(cirq.CCZ(*q))),
          "qft3": (td.decompose_qft3_to_sycamore, lambda: td.qft_unitary(3))}[args.command]
    cm = SycamoreCalibrationMap(seed=0, theta_drift_std=0.0, phi_drift_std=0.0) if args.native else None
    c = fn[0](*q, target_infidelity=args.target_infidelity, calibration_map=cm)
    u, ref = cirq.unitary(c), fn[1]()
    fidelity = float(abs(np.trace(ref.conj().T @ u)) ** 2 / 64)
    n_fsim = sum(1 for op in c.all_operations() if isinstance(op.gate, cirq.FSimGate))
    payload = {"gate": args.command, "fidelity": fidelity, "fsim_count": n_fsim, "moments": len(c), "native": bool(cm), "circuit": str(c)}
    lines = [f"[*] {args.command} → FSim + 1Q ({'native Sycamore gate' if cm else 'free angles'})",
             f"[+] fidelity = {fidelity:.8f}  FSim gates = {n_fsim}  moments = {len(c)}", str(c)]
    if args.qasm_out:
        export_to_openqasm3(c, file_path=args.qasm_out)
        payload["qasm_out"] = args.qasm_out
        lines.append(f"[+] wrote OpenQASM 3 to {args.qasm_out}")
    _emit(payload, args.json, lines)
    return 0


def cmd_qasm(args) -> int:
    import cirq
    from .openqasm3_io import export_to_openqasm3, parse_openqasm3
    text = open(args.inp).read()
    c = parse_openqasm3(text)
    out = export_to_openqasm3(c, file_path=args.out, expand_fsim=args.expand_fsim)
    payload = {"input": args.inp, "output": args.out, "n_qubits": len(c.all_qubits()), "n_ops": sum(1 for _ in c.all_operations())}
    _emit(payload, args.json, [f"[+] parsed {payload['n_ops']} operations on {payload['n_qubits']} qubits", out if not args.out else f"[+] wrote {args.out}"])
    return 0


def cmd_compare(args) -> int:
    from . import compare
    res = compare.run_comparison(n_haar=args.n_haar, circuits=not args.no_circuits, quick=args.quick,
                                 progress=None if args.json else (lambda m: print(f"[*] {m}", flush=True)))
    if args.out:
        compare.dump_json(res, args.out)
    payload = {"two_qubit_summary": res["two_qubit_summary"], "noise_aware": res["noise"]["at_cited"],
               "circuit_summary": res.get("circuit_summary")}
    _emit(payload, args.json, [res["markdown"]] + ([f"[+] wrote {args.out}"] if args.out else []))
    return 0


def cmd_weyl(args) -> int:
    import cirq
    from . import weyl
    from .calibration_map import CouplerCalibration
    gates = {"cnot": cirq.CNOT, "iswap": cirq.ISWAP, "sqrt_iswap": cirq.SQRT_ISWAP, "swap": cirq.SWAP, "cz": cirq.CZ}
    u = haar_random_unitary(4, seed=args.seed) if args.gate == "random" else cirq.unitary(gates[args.gate])
    syc = CouplerCalibration.sycamore()
    payload = {
        "gate": args.gate,
        "weyl_coordinates": weyl.weyl_coordinates(u).tolist(),
        "min_cz": weyl.min_cz_count(u),
        "min_sqrt_iswap": weyl.min_sqrt_iswap_count(u),
        "min_free_fsim": weyl.min_free_fsim_count(u),
    }
    payload["min_sycamore"], payload["min_sycamore_kind"] = weyl.minimum_count(u, syc.angles)
    x, y, z = payload["weyl_coordinates"]
    lines = [f"[*] {args.gate}: Weyl coordinates (x, y, z) = ({x:.6f}, {y:.6f}, {z:.6f})",
             f"[+] minimum gates: CZ {payload['min_cz']}, √iSWAP {payload['min_sqrt_iswap']}, "
             f"free FSim {payload['min_free_fsim']} (closed forms); "
             f"Sycamore {payload['min_sycamore']} ({payload['min_sycamore_kind']})"]
    _emit(payload, args.json, lines)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "benchmark":
        from .benchmark import run_compiler_benchmark
        return run_compiler_benchmark(as_json=args.json, quick=args.quick)
    if args.command == "synthesize":
        return cmd_synthesize(args)
    if args.command in ("toffoli", "fredkin", "ccz", "qft3"):
        return cmd_three_qubit(args)
    if args.command == "qasm":
        return cmd_qasm(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "weyl":
        return cmd_weyl(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
