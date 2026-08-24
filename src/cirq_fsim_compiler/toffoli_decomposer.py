"""
3-Qubit Unitary Synthesis (Toffoli / CCNOT, Fredkin, and 3-Qubit QFT).

Decomposes 3-qubit non-local gates into native Google Sycamore FSim interactions
and single-qubit rotations with minimum depth.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional

try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False
    cirq = None

from .cirq_transformer import compile_circuit_to_sycamore_fsim


def decompose_toffoli_to_sycamore(
    control1: "cirq.Qid",
    control2: "cirq.Qid",
    target: "cirq.Qid",
    target_infidelity: float = 1e-4,
) -> "cirq.Circuit":
    """
    Decomposes 3-qubit Toffoli / CCNOT gate into a native Sycamore FSim circuit
    via standard T/H/CNOT lattice transpiled to FSim.
    """
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")

    # Analytical minimum-depth standard decomposition (6 CNOTs)
    c = cirq.Circuit([
        cirq.H(target),
        cirq.CNOT(control2, target),
        cirq.T(target) ** -1,
        cirq.CNOT(control1, target),
        cirq.T(target),
        cirq.CNOT(control2, target),
        cirq.T(target) ** -1,
        cirq.CNOT(control1, target),
        cirq.T(control2),
        cirq.T(target),
        cirq.CNOT(control1, control2),
        cirq.H(target),
        cirq.T(control1),
        cirq.T(control2) ** -1,
        cirq.CNOT(control1, control2),
    ])

    return compile_circuit_to_sycamore_fsim(c, target_infidelity=target_infidelity)


def decompose_fredkin_to_sycamore(
    control: "cirq.Qid",
    target1: "cirq.Qid",
    target2: "cirq.Qid",
) -> "cirq.Circuit":
    """
    Decomposes Controlled-SWAP (Fredkin) gate into native Sycamore FSim operations.
    """
    c = cirq.Circuit([
        cirq.CNOT(target2, target1),
        cirq.CCNOT(control, target1, target2),
        cirq.CNOT(target2, target1),
    ])
    return compile_circuit_to_sycamore_fsim(c)
