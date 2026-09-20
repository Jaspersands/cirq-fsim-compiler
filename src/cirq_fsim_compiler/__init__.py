"""
cirq_fsim_compiler
==================
Differentiable and Riemannian synthesis of two-qubit unitaries into FSim
interactions — with free angles or a coupler's calibrated native gate — a Cirq
transformer for whole circuits (three-qubit gates included), and correct
OpenQASM 3 export/import.
"""

from .unitary_ansatz import (
    fsim_matrix_np,
    single_qubit_zxz_np,
    euler_zxz_from_unitary,
    haar_random_unitary,
    FSimCircuitTemplate,
)
from .riemannian_optimizer import (
    DifferentiableFSimSynthesizer,
    DecompositionResult,
    synthesize_unitary_to_fsim,
    process_fidelity,
    SEED_ANGLES,
)
from .riemannian_manifold import RiemannianFSimSolver, cayley_retract, lie_element
from .cirq_transformer import (
    FSimDecomposerTransformer,
    compile_circuit_to_sycamore_fsim,
    compile_to_fsim,
    convert_decomposition_to_cirq_ops,
)
from .calibration_map import SycamoreCalibrationMap, CouplerCalibration, SYCAMORE_THETA, SYCAMORE_PHI
from .toffoli_decomposer import (
    decompose_toffoli_to_sycamore,
    decompose_fredkin_to_sycamore,
    decompose_ccz_to_sycamore,
    decompose_qft3_to_sycamore,
    toffoli_circuit,
    qft_unitary,
)
from .openqasm3_io import export_to_openqasm3, parse_openqasm3
from .batch_compiler import BatchFSimCompiler

__version__ = "0.3.0"
__all__ = [
    "fsim_matrix_np", "single_qubit_zxz_np", "euler_zxz_from_unitary", "haar_random_unitary", "FSimCircuitTemplate",
    "DifferentiableFSimSynthesizer", "DecompositionResult", "synthesize_unitary_to_fsim", "process_fidelity", "SEED_ANGLES",
    "RiemannianFSimSolver", "cayley_retract", "lie_element",
    "FSimDecomposerTransformer", "compile_circuit_to_sycamore_fsim", "compile_to_fsim", "convert_decomposition_to_cirq_ops",
    "SycamoreCalibrationMap", "CouplerCalibration", "SYCAMORE_THETA", "SYCAMORE_PHI",
    "decompose_toffoli_to_sycamore", "decompose_fredkin_to_sycamore", "decompose_ccz_to_sycamore", "decompose_qft3_to_sycamore",
    "toffoli_circuit", "qft_unitary",
    "export_to_openqasm3", "parse_openqasm3",
    "BatchFSimCompiler",
]
