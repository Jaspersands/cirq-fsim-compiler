"""
Cirq Transformer Plugin for Compiling Circuits to Native Google Sycamore FSim Architectures.

Provides:
1. FSimDecomposerTransformer: Drops into Cirq compilation pipelines.
2. Direct OpenQASM / Cirq circuit transpilation.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, List, Optional, Callable, Dict, Any

try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False
    cirq = None

from .riemannian_optimizer import DifferentiableFSimSynthesizer, DecompositionResult


def convert_decomposition_to_cirq_ops(
    decomp: DecompositionResult,
    q0: "cirq.Qid",
    q1: "cirq.Qid",
) -> List["cirq.Operation"]:
    """
    Translates a DecompositionResult into a sequence of Cirq native operations.
    """
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")

    ops = []

    def apply_1q_euler(q: "cirq.Qid", a: float, b: float, g: float):
        # Rz(a) Rx(b) Rz(g)
        if abs(g) > 1e-6:
            ops.append(cirq.rz(g).on(q))
        if abs(b) > 1e-6:
            ops.append(cirq.rx(b).on(q))
        if abs(a) > 1e-6:
            ops.append(cirq.rz(a).on(q))

    # Initial 1Q layer
    sq0 = decomp.single_qubit_angles[0]
    apply_1q_euler(q0, sq0[0][0], sq0[0][1], sq0[0][2])
    apply_1q_euler(q1, sq0[1][0], sq0[1][1], sq0[1][2])

    for s in range(decomp.n_stages):
        # Native FSim gate
        theta, phi = decomp.fsim_angles[s]
        ops.append(cirq.FSimGate(theta=theta, phi=phi).on(q0, q1))

        # Interleaved/Final 1Q layer
        sq_layer = decomp.single_qubit_angles[s + 1]
        apply_1q_euler(q0, sq_layer[0][0], sq_layer[0][1], sq_layer[0][2])
        apply_1q_euler(q1, sq_layer[1][0], sq_layer[1][1], sq_layer[1][2])

    return ops


if HAS_CIRQ:
    class FSimDecomposerTransformer:
        """
        Cirq transformer that replaces non-native two-qubit gates
        with differentiably optimized FSim sequences.
        """

        def __init__(self, target_infidelity: float = 1e-6, max_stages: int = 3):
            self.synthesizer = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity)
            self.max_stages = max_stages

        def __call__(self, circuit: "cirq.Circuit", context: Optional[cirq.TransformerContext] = None) -> "cirq.Circuit":
            return self.optimize_circuit(circuit)

        def optimize_circuit(self, circuit: "cirq.Circuit") -> "cirq.Circuit":
            compiled = cirq.Circuit()

            for moment in circuit:
                for op in moment.operations:
                    if len(op.qubits) == 2:
                        # Check if already native FSim
                        if isinstance(op.gate, (cirq.FSimGate, cirq.PhasedFSimGate)):
                            compiled.append(op)
                        else:
                            # Extract 4x4 matrix
                            mat = cirq.unitary(op)
                            q0, q1 = op.qubits[0], op.qubits[1]
                            decomp = self.synthesizer.decompose(mat, max_stages=self.max_stages)
                            native_ops = convert_decomposition_to_cirq_ops(decomp, q0, q1)
                            compiled.append(native_ops)
                    else:
                        compiled.append(op)

            return compiled


def compile_circuit_to_sycamore_fsim(
    circuit: "cirq.Circuit", target_infidelity: float = 1e-6
) -> "cirq.Circuit":
    """Compiles a Cirq circuit to native Sycamore FSim gates."""
    transformer = FSimDecomposerTransformer(target_infidelity=target_infidelity)
    return transformer.optimize_circuit(circuit)
