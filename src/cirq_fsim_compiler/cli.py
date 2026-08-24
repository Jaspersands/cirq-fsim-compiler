"""
Command-Line Interface for cirq_fsim_compiler.
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import cirq
from .riemannian_optimizer import synthesize_unitary_to_fsim
from .cirq_transformer import compile_circuit_to_sycamore_fsim
from .toffoli_decomposer import decompose_toffoli_to_sycamore
from .openqasm3_io import export_to_openqasm3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cirq-fsim",
        description="Native Google Sycamore FSim Compiler CLI (Riemannian U(4) / Toffoli / OpenQASM 3.0)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: synthesize
    syn_parser = subparsers.add_parser("synthesize", help="Synthesize standard 2Q gate to native FSim")
    syn_parser.add_argument(
        "--gate",
        type=str,
        choices=["cnot", "iswap", "sqrt_iswap", "swap", "cz"],
        default="cnot",
        help="Target 2-qubit gate",
    )
    syn_parser.add_argument("--max-stages", type=int, default=3, help="Maximum FSim interaction stages")

    # Command: toffoli
    tof_parser = subparsers.add_parser("toffoli", help="Decompose 3-qubit Toffoli into native Sycamore circuit")
    tof_parser.add_argument("--qasm-out", type=str, default=None, help="Output OpenQASM 3.0 file path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "synthesize":
        gate_map = {
            "cnot": cirq.unitary(cirq.CNOT),
            "iswap": cirq.unitary(cirq.ISWAP),
            "sqrt_iswap": cirq.unitary(cirq.SQRT_ISWAP),
            "swap": cirq.unitary(cirq.SWAP),
            "cz": cirq.unitary(cirq.CZ),
        }
        target_u = gate_map[args.gate]
        print(f"[*] Synthesizing {args.gate.upper()} onto Sycamore FSim via JAX Riemannian optimization...")
        res = synthesize_unitary_to_fsim(target_u, max_stages=args.max_stages)
        print(f"[+] Synthesis Complete! Stages: {res.n_stages}, Fidelity: {res.fidelity * 100:.6f}% (Infidelity: {res.infidelity:.2e})")
        return 0

    elif args.command == "toffoli":
        print("[*] Decomposing 3-Qubit Toffoli into native Sycamore FSim circuit...")
        q0, q1, q2 = cirq.LineQubit.range(3)
        c = decompose_toffoli_to_sycamore(q0, q1, q2)
        print(f"[+] Transpiled Circuit:\n{c}")
        if args.qasm_out:
            export_to_openqasm3(c, file_path=args.qasm_out)
            print(f"[+] Exported OpenQASM 3.0 to {args.qasm_out}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
