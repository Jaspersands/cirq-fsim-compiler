"""
Cirq transformer that compiles any circuit to FSim + single-qubit gates.

* Two-qubit gates that are not already ``FSimGate``/``PhasedFSimGate`` are
  synthesised with :class:`DifferentiableFSimSynthesizer` (free angles or the
  coupler's calibrated native gate when a ``SycamoreCalibrationMap`` is given).
* Gates on three or more qubits are first expanded with ``cirq.decompose`` to
  one- and two-qubit operations and then compiled.
* With ``consolidate=True`` (default) every maximal run of operations on the
  same qubit pair — including the single-qubit gates between them — is merged
  into one 4×4 unitary before synthesis, so a block such as CNOT·(T⊗T†)·CNOT
  costs what its KAK class costs (≤ 3 native gates) instead of 2 per CNOT.
* ``device_graph`` (a networkx graph of physical qubits) routes the circuit
  first with Cirq's ``RouteCQC``, inserting SWAPs so that every two-qubit
  operation acts on a physical coupler; consolidation then folds each SWAP into
  its neighbouring gates where possible.
* Identical two-qubit unitaries are synthesised once and cached.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .riemannian_optimizer import DifferentiableFSimSynthesizer, DecompositionResult
from .calibration_map import SycamoreCalibrationMap, CouplerCalibration

try:
    import cirq

    HAS_CIRQ = True
except ImportError:  # pragma: no cover
    HAS_CIRQ = False
    cirq = None


def convert_decomposition_to_cirq_ops(decomp: DecompositionResult, q0: "cirq.Qid", q1: "cirq.Qid") -> List["cirq.Operation"]:
    """Translate a :class:`DecompositionResult` into Cirq operations (Rz·Rx·Rz layers and FSim gates)."""
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")
    ops: List["cirq.Operation"] = []

    def euler(q, a, b, g):
        # matrix is Rz(a) Rx(b) Rz(g): Rz(g) acts first
        if abs(g) > 1e-12:
            ops.append(cirq.rz(g).on(q))
        if abs(b) > 1e-12:
            ops.append(cirq.rx(b).on(q))
        if abs(a) > 1e-12:
            ops.append(cirq.rz(a).on(q))

    sq0 = decomp.single_qubit_angles[0]
    euler(q0, *sq0[0]); euler(q1, *sq0[1])
    for s in range(decomp.n_stages):
        theta, phi = decomp.fsim_angles[s]
        ops.append(cirq.FSimGate(theta=theta, phi=phi).on(q0, q1))
        sq = decomp.single_qubit_angles[s + 1]
        euler(q0, *sq[0]); euler(q1, *sq[1])
    return ops


def _qubit_index(q: "cirq.Qid") -> int:
    if isinstance(q, cirq.GridQubit):
        return q.row * 1000 + q.col
    if isinstance(q, cirq.LineQubit):
        return q.x
    return hash(q) % 100000


class FSimDecomposerTransformer:
    """
    Parameters
    ----------
    target_infidelity, max_stages : synthesis accuracy and depth budget
    calibration_map : if given, every edge uses its calibrated native gate
    method : ``"euclidean"`` or ``"riemannian"``
    """

    def __init__(
        self,
        target_infidelity: float = 1e-6,
        max_stages: int = 3,
        calibration_map: Optional[SycamoreCalibrationMap] = None,
        method: str = "euclidean",
        seed: int = 42,
        n_restarts: int = 4,
        consolidate: bool = True,
        device_graph=None,
    ):
        self.synthesizer = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, seed=seed, method=method)
        self.max_stages = int(max_stages)
        self.calibration_map = calibration_map
        self.n_restarts = int(n_restarts)
        self.consolidate = bool(consolidate)
        self.device_graph = device_graph
        self.last_routing = None
        self._cache: Dict[Tuple[bytes, Optional[Tuple[float, float]]], DecompositionResult] = {}
        self.stats = {"synthesised": 0, "cache_hits": 0, "native_kept": 0, "decomposed_multi_qubit": 0}

    def __call__(self, circuit: "cirq.Circuit", *, context: Optional["cirq.TransformerContext"] = None) -> "cirq.Circuit":
        return self.optimize_circuit(circuit)

    def _calibration_for(self, q0, q1) -> Optional[CouplerCalibration]:
        if self.calibration_map is None:
            return None
        return self.calibration_map.get_coupler(_qubit_index(q0), _qubit_index(q1))

    def _synthesise(self, mat: np.ndarray, cal: Optional[CouplerCalibration]) -> DecompositionResult:
        key = (np.round(mat, 10).tobytes(), None if cal is None else cal.angles)
        if key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]
        res = self.synthesizer.decompose(mat, max_stages=self.max_stages, n_restarts=self.n_restarts, calibration=cal)
        self.stats["synthesised"] += 1
        self._cache[key] = res
        return res

    def _compile_op(self, op: "cirq.Operation", out: List["cirq.Operation"]) -> None:
        n = len(op.qubits)
        if n <= 1:
            out.append(op)
            return
        if n == 2:
            if isinstance(op.gate, (cirq.FSimGate, cirq.PhasedFSimGate)):
                self.stats["native_kept"] += 1
                out.append(op)
                return
            q0, q1 = op.qubits
            res = self._synthesise(cirq.unitary(op), self._calibration_for(q0, q1))
            out.extend(convert_decomposition_to_cirq_ops(res, q0, q1))
            return
        # three or more qubits: expand first
        sub = cirq.decompose(op, keep=lambda o: len(o.qubits) <= 2)
        if any(len(o.qubits) > 2 for o in sub):
            raise NotImplementedError(f"cannot decompose {op} into ≤2-qubit operations")
        self.stats["decomposed_multi_qubit"] += 1
        for o in sub:
            self._compile_op(o, out)

    def _expand_multi_qubit(self, circuit: "cirq.Circuit") -> "cirq.Circuit":
        ops: List["cirq.Operation"] = []
        for op in circuit.all_operations():
            if len(op.qubits) > 2:
                sub = cirq.decompose(op, keep=lambda o: len(o.qubits) <= 2)
                if any(len(o.qubits) > 2 for o in sub):
                    raise NotImplementedError(f"cannot decompose {op} into ≤2-qubit operations")
                self.stats["decomposed_multi_qubit"] += 1
                ops.extend(sub)
            else:
                ops.append(op)
        return cirq.Circuit(ops)

    def route(self, circuit: "cirq.Circuit") -> "cirq.Circuit":
        """Map onto ``device_graph`` with SWAP insertion (Cirq RouteCQC); records the qubit maps."""
        router = cirq.RouteCQC(self.device_graph)
        routed, initial, swaps = router.route_circuit(circuit)
        # logical qubit q starts on initial[q] and ends on swaps[initial[q]]
        final = {q: swaps.get(p, p) for q, p in initial.items()}
        self.last_routing = {"initial_map": initial, "final_map": final,
                             "swaps": sum(1 for op in routed.all_operations() if op.gate == cirq.SWAP)}
        return routed

    def optimize_circuit(self, circuit: "cirq.Circuit") -> "cirq.Circuit":
        measurements = [op for op in circuit.all_operations() if cirq.is_measurement(op)]
        body = cirq.Circuit(op for op in circuit.all_operations() if not cirq.is_measurement(op))
        body = self._expand_multi_qubit(body)
        if self.device_graph is not None:
            body = self.route(body)
            if measurements:
                final = self.last_routing["final_map"]
                measurements = [m.transform_qubits(lambda q: final[q]) for m in measurements]
        if self.consolidate:
            body = cirq.merge_k_qubit_unitaries(body, k=2, rewriter=lambda op: op)
        ops: List["cirq.Operation"] = []
        for op in body.all_operations():
            if isinstance(op, cirq.CircuitOperation):
                inner = list(op.mapped_circuit().all_operations())
                if len(op.qubits) == 2 and len(inner) == 1 and isinstance(inner[0].gate, (cirq.FSimGate, cirq.PhasedFSimGate)):
                    self.stats["native_kept"] += 1
                    ops.append(inner[0])
                    continue
                if len(op.qubits) == 1:
                    ops.append(cirq.MatrixGate(cirq.unitary(op)).on(*op.qubits))
                    continue
                if len(op.qubits) == 2:
                    q0, q1 = op.qubits
                    res = self._synthesise(cirq.unitary(op), self._calibration_for(q0, q1))
                    ops.extend(convert_decomposition_to_cirq_ops(res, q0, q1))
                    continue
            self._compile_op(op, ops)
        return cirq.Circuit(ops + measurements)


if HAS_CIRQ:

    @cirq.transformer
    def compile_to_fsim(
        circuit: "cirq.AbstractCircuit",
        *,
        context: Optional["cirq.TransformerContext"] = None,
        target_infidelity: float = 1e-6,
        max_stages: int = 3,
        calibration_map: Optional[SycamoreCalibrationMap] = None,
        method: str = "euclidean",
        consolidate: bool = True,
        device_graph=None,
    ) -> "cirq.Circuit":
        """``cirq.transformer``-compatible entry point."""
        tr = FSimDecomposerTransformer(target_infidelity=target_infidelity, max_stages=max_stages,
                                       calibration_map=calibration_map, method=method,
                                       consolidate=consolidate, device_graph=device_graph)
        return tr.optimize_circuit(cirq.Circuit(circuit))

else:  # pragma: no cover

    def compile_to_fsim(*args, **kwargs):
        raise RuntimeError("Cirq required.")


def compile_circuit_to_sycamore_fsim(
    circuit: "cirq.Circuit",
    target_infidelity: float = 1e-6,
    calibration_map: Optional[SycamoreCalibrationMap] = None,
    method: str = "euclidean",
    max_stages: int = 3,
    consolidate: bool = True,
    device_graph=None,
) -> "cirq.Circuit":
    """Compile ``circuit`` to FSim + single-qubit gates (optionally routed onto ``device_graph``)."""
    tr = FSimDecomposerTransformer(target_infidelity=target_infidelity, max_stages=max_stages,
                                   calibration_map=calibration_map, method=method,
                                   consolidate=consolidate, device_graph=device_graph)
    return tr.optimize_circuit(circuit)
