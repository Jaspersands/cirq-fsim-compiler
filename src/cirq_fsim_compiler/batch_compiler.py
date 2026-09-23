"""
Batch synthesis of many 4×4 unitaries.

Synthesis is CPU-bound Python (SciPy optimiser loops, JAX dispatch, Cayley
updates), so threads mostly contend for the GIL. ``executor="process"`` (the
default when ``n_workers > 1``) runs a pool of worker processes, each with its
own synthesiser built once by the pool initialiser; ``executor="thread"`` is
kept for environments where processes are unavailable. Results are
independent of the executor: every unitary gets its own deterministic seed
derived from its position in the batch.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .riemannian_optimizer import DifferentiableFSimSynthesizer, DecompositionResult
from .calibration_map import CouplerCalibration

_WORKER: Dict[str, Any] = {}
_SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                      "VECLIB_MAXIMUM_THREADS": "1",
                      "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}


@contextmanager
def _single_threaded_children():
    """Workers inherit the environment at spawn: give each one BLAS/XLA thread so they do not oversubscribe."""
    saved = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
    os.environ.update(_SINGLE_THREAD_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _init_worker(target_infidelity: float, method: str) -> None:
    _WORKER["syn"] = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, method=method)


def _synthesise_one(job: Tuple[np.ndarray, int, Optional[CouplerCalibration], int]) -> DecompositionResult:
    u, max_stages, calibration, seed = job
    syn = _WORKER["syn"]
    syn.seed = seed
    return syn.decompose(np.asarray(u), max_stages=max_stages, calibration=calibration)


class BatchFSimCompiler:
    def __init__(self, target_infidelity: float = 1e-5, max_stages: int = 3, n_workers: int = 1,
                 calibration: Optional[CouplerCalibration] = None, method: str = "euclidean", seed: int = 42,
                 executor: Optional[str] = None):
        self.target_infidelity = float(target_infidelity)
        self.method = method
        self.max_stages = int(max_stages)
        self.n_workers = max(1, int(n_workers))
        self.calibration = calibration
        self.seed = int(seed)
        self.executor = executor or ("process" if self.n_workers > 1 else "serial")
        if self.executor not in ("serial", "thread", "process"):
            raise ValueError("executor must be 'serial', 'thread' or 'process'")
        self.synthesizer = DifferentiableFSimSynthesizer(target_infidelity=target_infidelity, seed=seed, method=method)

    def _jobs(self, unitaries: Sequence[np.ndarray]):
        return [(np.asarray(u), self.max_stages, self.calibration, self.seed + k) for k, u in enumerate(unitaries)]

    def compile_batch(self, unitaries: Sequence[np.ndarray]) -> List[DecompositionResult]:
        jobs = self._jobs(unitaries)
        if self.executor == "process" and len(jobs) > 1:
            ctx = mp.get_context("spawn")                  # JAX is not fork-safe
            with _single_threaded_children(), ProcessPoolExecutor(
                    max_workers=self.n_workers, mp_context=ctx, initializer=_init_worker,
                    initargs=(self.target_infidelity, self.method)) as pool:
                chunk = max(1, len(jobs) // (4 * self.n_workers))
                return list(pool.map(_synthesise_one, jobs, chunksize=chunk))
        _WORKER.setdefault("syn", self.synthesizer)
        if self.executor == "thread" and len(jobs) > 1:
            local = threading.local()                      # one synthesiser per thread (decompose() sets the seed)

            def run(job):
                if not hasattr(local, "syn"):
                    local.syn = DifferentiableFSimSynthesizer(target_infidelity=self.target_infidelity, method=self.method)
                local.syn.seed = job[3]
                return local.syn.decompose(job[0], max_stages=job[1], calibration=job[2])
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                return list(pool.map(run, jobs))
        out = []
        for u, stages, cal, seed in jobs:
            self.synthesizer.seed = seed
            out.append(self.synthesizer.decompose(u, max_stages=stages, calibration=cal))
        return out

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
