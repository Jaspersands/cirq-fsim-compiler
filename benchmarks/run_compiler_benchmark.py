"""
Benchmark suite comparing Differentiable FSim compilation vs KAK on standard quantum gates.
"""

import time
import numpy as np
import cirq
from cirq_fsim_compiler.unitary_ansatz import fsim_matrix_np
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer
from cirq_fsim_compiler.cirq_transformer import compile_circuit_to_sycamore_fsim


def run_compiler_benchmark():
    print("=" * 65)
    print("NATIVE CIRQ FSIM COMPILER BENCHMARK (DIFFERENTIABLE U(4) SYNTHESIS)")
    print("=" * 65)

    synth = DifferentiableFSimSynthesizer(target_infidelity=1e-5)

    test_gates = {
        "CNOT": cirq.unitary(cirq.CNOT),
        "CZ": cirq.unitary(cirq.CZ),
        "iSWAP": cirq.unitary(cirq.ISWAP),
        "SWAP": cirq.unitary(cirq.SWAP),
        "Givens (theta=0.6)": cirq.unitary(cirq.FSimGate(0.6, 0.0)),
    }

    print("\n1. Synthesizing Standard 2-Qubit Gates:")
    print("   ---------------------------------------------------------------")
    print("   Gate Name           | Stages | Infidelity (1-F) | Time (ms)")
    print("   ---------------------------------------------------------------")

    for name, u_mat in test_gates.items():
        t0 = time.time()
        res = synth.decompose(u_mat, max_stages=3, n_restarts=3)
        t_ms = (time.time() - t0) * 1000.0
        print(f"   {name:<20} |   {res.n_stages}    |    {res.infidelity:10.2e}    | {t_ms:7.1f} ms")

    print("\n2. Compiling Multi-Qubit Cirq Circuit:")
    q = cirq.LineQubit.range(3)
    orig_circuit = cirq.Circuit([
        cirq.H(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[1], q[2]),
        cirq.ISWAP(q[0], q[2]),
    ])
    print("   Original Circuit:")
    print(orig_circuit)

    compiled = compile_circuit_to_sycamore_fsim(orig_circuit)
    print("\n   Compiled Native Sycamore FSim Circuit:")
    print(compiled)
    print("=" * 65)


if __name__ == "__main__":
    run_compiler_benchmark()
