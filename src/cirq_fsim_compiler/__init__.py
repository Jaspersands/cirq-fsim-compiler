"""
cirq_fsim_compiler
==================
Native Google Sycamore FSim Compiler via Differentiable Unitary Decomposition in PennyLane & JAX.
"""

from .unitary_ansatz import (
    fsim_matrix_np,
    single_qubit_zxz_np,
    FSimCircuitTemplate,
)
from .riemannian_optimizer import (
    DifferentiableFSimSynthesizer,
    DecompositionResult,
    synthesize_unitary_to_fsim,
)
from .cirq_transformer import (
    FSimDecomposerTransformer,
    compile_circuit_to_sycamore_fsim,
)
from .calibration_map import SycamoreCalibrationMap, CouplerCalibration
from .toffoli_decomposer import decompose_toffoli_to_sycamore, decompose_fredkin_to_sycamore
from .openqasm3_io import export_to_openqasm3
from .batch_compiler import BatchFSimCompiler

__version__ = "0.2.0"
__all__ = [
    "fsim_matrix_np",
    "single_qubit_zxz_np",
    "FSimCircuitTemplate",
    "DifferentiableFSimSynthesizer",
    "DecompositionResult",
    "synthesize_unitary_to_fsim",
    "FSimDecomposerTransformer",
    "compile_circuit_to_sycamore_fsim",
    "SycamoreCalibrationMap",
    "CouplerCalibration",
    "decompose_toffoli_to_sycamore",
    "decompose_fredkin_to_sycamore",
    "export_to_openqasm3",
    "BatchFSimCompiler",
]
