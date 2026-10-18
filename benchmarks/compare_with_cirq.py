"""
Compare cirq_fsim_compiler with Cirq's built-in decompositions.

    python benchmarks/compare_with_cirq.py            # full run (about six minutes on a laptop)
    python benchmarks/compare_with_cirq.py --quick    # 10 Haar targets, three circuits
    python benchmarks/compare_with_cirq.py --n-haar 50 --no-circuits

Writes
    benchmarks/results/compare_with_cirq.json       compact results: summaries, tables, timing,
                                                    noise-aware study (committed; make_figures reads it)
    benchmarks/results/compare_with_cirq.md         the README tables
    benchmarks/results/full/compare_with_cirq.json  every row (not committed)

A --quick run writes only under benchmarks/results/full/. The same run is
available as ``cirq-fsim compare``.
"""

import argparse
import os
import sys
import time

from cirq_fsim_compiler.compare import compact_results, dump_json, run_comparison

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-haar", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-circuits", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args(argv)

    t0 = time.time()
    res = run_comparison(n_haar=args.n_haar, seed=args.seed, circuits=not args.no_circuits, quick=args.quick,
                         progress=lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True))
    full_dir = os.path.join(args.out, "full")
    os.makedirs(full_dir, exist_ok=True)
    stem = "compare_with_cirq_quick" if args.quick else "compare_with_cirq"
    compact_dir = full_dir if args.quick else args.out
    dump_json(res, os.path.join(full_dir, stem + ".json"))
    dump_json(compact_results(res), os.path.join(compact_dir, stem + (".compact" if args.quick else "") + ".json"))
    with open(os.path.join(compact_dir, stem + ".md"), "w") as f:
        f.write(res["markdown"] + "\n")
    print(res["markdown"])
    print(f"\nwrote {compact_dir}/{stem}.* and {full_dir}/{stem}.json in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
