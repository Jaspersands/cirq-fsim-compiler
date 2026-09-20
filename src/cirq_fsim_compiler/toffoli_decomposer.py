"""
Three-qubit gates compiled to FSim + single-qubit operations.

Each function builds the standard textbook circuit (CNOT / H / T for Toffoli,
CNOT·Toffoli·CNOT for Fredkin, H·Toffoli·H for CCZ, H + controlled phases +
SWAP for the 3-qubit QFT), then runs it through the FSim compiler so that the
result contains only ≤2-qubit operations with FSim interactions.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .cirq_transformer import compile_circuit_to_sycamore_fsim
from .calibration_map import SycamoreCalibrationMap

try:
    import cirq

    HAS_CIRQ = True
except ImportError:  # pragma: no cover
    HAS_CIRQ = False
    cirq = None


def toffoli_circuit(c1: "cirq.Qid", c2: "cirq.Qid", t: "cirq.Qid") -> "cirq.Circuit":
    """Six-CNOT Toffoli (Nielsen & Chuang Fig. 4.9)."""
    return cirq.Circuit([
        cirq.H(t), cirq.CNOT(c2, t), cirq.T(t) ** -1, cirq.CNOT(c1, t), cirq.T(t), cirq.CNOT(c2, t),
        cirq.T(t) ** -1, cirq.CNOT(c1, t), cirq.T(c2), cirq.T(t), cirq.CNOT(c1, c2), cirq.H(t),
        cirq.T(c1), cirq.T(c2) ** -1, cirq.CNOT(c1, c2),
    ])


def fredkin_circuit(c: "cirq.Qid", t1: "cirq.Qid", t2: "cirq.Qid") -> "cirq.Circuit":
    return cirq.Circuit([cirq.CNOT(t2, t1)]) + toffoli_circuit(c, t1, t2) + cirq.Circuit([cirq.CNOT(t2, t1)])


def ccz_circuit(a: "cirq.Qid", b: "cirq.Qid", c: "cirq.Qid") -> "cirq.Circuit":
    return cirq.Circuit([cirq.H(c)]) + toffoli_circuit(a, b, c) + cirq.Circuit([cirq.H(c)])


def qft3_circuit(q0: "cirq.Qid", q1: "cirq.Qid", q2: "cirq.Qid") -> "cirq.Circuit":
    """QFT on three qubits (q0 most significant), including the final reversal SWAP."""
    return cirq.Circuit([
        cirq.H(q0), cirq.CZPowGate(exponent=0.5).on(q1, q0), cirq.CZPowGate(exponent=0.25).on(q2, q0),
        cirq.H(q1), cirq.CZPowGate(exponent=0.5).on(q2, q1),
        cirq.H(q2),
        cirq.SWAP(q0, q2),
    ])


def qft_unitary(n_qubits: int) -> np.ndarray:
    """Discrete Fourier transform matrix F_{jk} = ω^{jk}/√N, ω = e^{2πi/N}."""
    N = 2**n_qubits
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.exp(2j * np.pi * j * k / N) / np.sqrt(N)


def _compile(circuit, target_infidelity, calibration_map):
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")
    return compile_circuit_to_sycamore_fsim(circuit, target_infidelity=target_infidelity, calibration_map=calibration_map)


def decompose_toffoli_to_sycamore(control1, control2, target, target_infidelity: float = 1e-6,
                                  calibration_map: Optional[SycamoreCalibrationMap] = None) -> "cirq.Circuit":
    return _compile(toffoli_circuit(control1, control2, target), target_infidelity, calibration_map)


def decompose_fredkin_to_sycamore(control, target1, target2, target_infidelity: float = 1e-6,
                                  calibration_map: Optional[SycamoreCalibrationMap] = None) -> "cirq.Circuit":
    return _compile(fredkin_circuit(control, target1, target2), target_infidelity, calibration_map)


def decompose_ccz_to_sycamore(a, b, c, target_infidelity: float = 1e-6,
                              calibration_map: Optional[SycamoreCalibrationMap] = None) -> "cirq.Circuit":
    return _compile(ccz_circuit(a, b, c), target_infidelity, calibration_map)


def decompose_qft3_to_sycamore(q0, q1, q2, target_infidelity: float = 1e-6,
                               calibration_map: Optional[SycamoreCalibrationMap] = None) -> "cirq.Circuit":
    return _compile(qft3_circuit(q0, q1, q2), target_infidelity, calibration_map)
