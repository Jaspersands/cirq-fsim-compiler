"""python benchmarks/run_compiler_benchmark.py [--json] [--quick]  (== cirq-fsim benchmark)"""

import sys

from cirq_fsim_compiler.benchmark import run_compiler_benchmark

if __name__ == "__main__":
    sys.exit(run_compiler_benchmark(as_json="--json" in sys.argv, quick="--quick" in sys.argv))
