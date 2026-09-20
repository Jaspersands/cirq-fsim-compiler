"""
Batch synthesis of many 4×4 unitaries with optional thread parallelism.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .riemannian_optimizer import DifferentiableFSimSynthesizer, DecompositionResult
from .calibration_map import CouplerCalibration


class BatchFSimCompiler:
    def __init__(self, target_infidelity: float = 1e-5, max_stages: int = 3, n_workers: int = 1,
                 calibration: Optional[CouplerCalibration] = None, method: str = "euclidean", seed: int = 42):
        self.synthesizer = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, seed=seed, method=method)
        self.max_stages = int(max_stages)
        self.n_workers = max(1, int(n_workers))
        self.calibration = calibration

    def compile_batch(self, unitaries: Sequence[np.ndarray]) -> List[DecompositionResult]:
        fn = lambda u: self.synthesizer.decompose(np.asarray(u), max_stages=self.max_stages, calibration=self.calibration)
        if self.n_workers == 1:
            return [fn(u) for u in unitaries]
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            return list(pool.map(fn, unitaries))

    @staticmethod
    def compute_summary_statistics(results: Sequence[DecompositionResult]) -> Dict[str, Any]:
        stages = [r.n_stages for r in results]
        return {
            "total_unitaries": len(results),
            "total_fsim_gates": int(sum(stages)),
            "stage_histogram": {k: stages.count(k) for k in (1, 2, 3)},
            "mean_infidelity": float(np.mean([r.infidelity for r in results])) if results else 0.0,
            "max_infidelity": float(np.max([r.infidelity for r in results])) if results else 0.0,
            "success_rate": float(np.mean([1.0 if r.is_success else 0.0 for r in results])) if results else 0.0,
            "n_native": int(sum(1 for r in results if r.native)),
        }
