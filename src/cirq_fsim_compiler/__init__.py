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

__version__ = "0.1.0"
__all__ = [
    "fsim_matrix_np",
    "single_qubit_zxz_np",
    "FSimCircuitTemplate",
    "DifferentiableFSimSynthesizer",
    "DecompositionResult",
    "synthesize_unitary_to_fsim",
    "FSimDecomposerTransformer",
    "compile_circuit_to_sycamore_fsim",
]
