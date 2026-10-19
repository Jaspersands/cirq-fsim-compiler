"""
Exact gate-count regions for Sycamore and for drifted couplers, from the monodromy polytope.

    python benchmarks/fsim_regions.py                  # about ten minutes on a laptop
    python benchmarks/fsim_regions.py --n-haar 300     # quicker, noisier

Writes benchmarks/results/fsim_regions.json:

    sycamore     Haar measure of the four-gate region x + y + |z| < π/8 (quadrature of the Haar density), and
                 the Haar share of x + y + |z| < s on a grid of s
    drift        for FSim(π/2 ± δ, π/6), the share of Haar-random targets whose minimum count is 2, 3, 4, ...
                 (FSim(π/2 + δ, φ) and FSim(π/2 − δ, φ) are the same class, so only δ ≥ 0 is computed)
    pair_33_34   the same for FSim(1.2947, 0.4859), the calibrated gate of the pair (3,3)-(3,4) in the released
                 supremacy circuits, as quoted by Kalai, Rinott and Shoham (arXiv:2210.12753)
    map          the same pooled over couplers drawn with SycamoreCalibrationMap's default drift
    check        polytope counts against multi-start synthesis on a subset of targets
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import integrate

from cirq_fsim_compiler import weyl
from cirq_fsim_compiler.calibration_map import SycamoreCalibrationMap
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer
from cirq_fsim_compiler.unitary_ansatz import haar_random_unitary

HERE = os.path.dirname(os.path.abspath(__file__))
P4, P8 = np.pi / 4, np.pi / 8
SYC = (np.pi / 2, np.pi / 6)
PAIR_33_34 = (1.2947043217999283, 0.4859467238431821)
DELTAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]


def sycamore_measures():
    opts = dict(epsabs=1e-15, epsrel=1e-10)
    total = integrate.tplquad(lambda z, y, x: weyl.haar_density(x, y, z), 0, P4, 0, lambda x: x,
                              lambda x, y: -y, lambda x, y: y, **opts)[0]
    out = {"four_gate_share": below(P8, total), "density_total": total}
    out["three_gate_share"] = 1 - out["four_gate_share"]
    grid = np.geomspace(0.1, 3 * P4, 32)
    out["cdf"] = {"s": grid.tolist(), "share_below": [below(t, total) for t in grid]}
    return out


def below(t, total):
    """Haar share of the classes with x + y + |z| < t (both signs of z)."""
    opts = dict(epsabs=1e-15, epsrel=1e-10)
    return 2 * integrate.tplquad(lambda x, y, z: weyl.haar_density(x, y, z), 0, min(t / 3, P4), lambda z: z,
                                 lambda z: min((t - z) / 2, P4), lambda z, y: y, lambda z, y: min(t - y - z, P4),
                                 **opts)[0] / total


def shares(coords, native, max_count):
    counts = [weyl.minimum_count(c, native, max_stages=max_count)[0] for c in coords]
    out = {}
    for n in sorted(set(c for c in counts if c is not None)):
        out[str(n)] = counts.count(n) / len(counts)
    out["none"] = counts.count(None) / len(counts)
    return out, counts


def synthesis_check(targets, counts, native, k, syn):
    """For the first k targets: synthesis reaches the polytope count n (below 1e-8), and one gate fewer stays above 1e-6."""
    rows = []
    for u, n in list(zip(targets, counts))[:k]:
        at_n = syn._optimize_stage(u, n, n_restarts=12, max_iter=500, fixed=[native] * n).infidelity
        below = syn._optimize_stage(u, n - 1, n_restarts=12, max_iter=500, fixed=[native] * (n - 1)).infidelity
        rows.append({"n": n, "infidelity_n": at_n, "infidelity_n_minus_1": below})
    agree = sum(r["infidelity_n"] < 1e-8 and r["infidelity_n_minus_1"] > 1e-6 for r in rows)
    return {"targets": k, "agree": int(agree), "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-haar", type=int, default=2000)
    ap.add_argument("--n-check", type=int, default=10)
    ap.add_argument("--n-couplers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "fsim_regions.json"))
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    targets = [haar_random_unitary(4, seed=args.seed + i) for i in range(args.n_haar)]
    coords = [weyl.weyl_coordinates(u) for u in targets]
    syn = DifferentiableFSimSynthesizer(target_infidelity=1e-10, seed=0, weyl_pruning=False)
    res = {"meta": {"n_haar": args.n_haar, "seed": args.seed, "n_check": args.n_check}}
    res["sycamore"] = sycamore_measures()
    log(f"Sycamore four-gate share {res['sycamore']['four_gate_share']:.4e}")

    res["drift"] = []
    for d in DELTAS:
        native = (np.pi / 2 + d, np.pi / 6)
        sh, counts = shares(coords, native, 6)
        row = {"delta_theta": d, "native": native, "shares": sh,
               "check": synthesis_check(targets, counts, native, args.n_check, syn)}
        res["drift"].append(row)
        log(f"δθ = {d}: {sh}, synthesis agrees on {row['check']['agree']}/{row['check']['targets']}")

    sh, counts = shares(coords, PAIR_33_34, 6)
    res["pair_33_34"] = {"native": PAIR_33_34, "shares": sh,
                         "check": synthesis_check(targets, counts, PAIR_33_34, args.n_check, syn)}
    log(f"pair (3,3)-(3,4): {sh}, synthesis agrees on {res['pair_33_34']['check']['agree']}/{args.n_check}")

    cmap = SycamoreCalibrationMap(seed=1)
    per = max(1, args.n_haar // args.n_couplers)
    pooled, natives = [], []
    for e in range(args.n_couplers):
        native = cmap.get_coupler(e, e + 1).angles
        natives.append(native)
        pooled += [weyl.minimum_count(c, native, max_stages=6)[0] for c in coords[e * per:(e + 1) * per]]
    res["map"] = {"theta_drift_std": cmap.theta_drift_std, "phi_drift_std": cmap.phi_drift_std, "natives": natives,
                  "targets_per_coupler": per,
                  "shares": {str(n): pooled.count(n) / len(pooled) for n in sorted(set(pooled), key=str)}}
    log(f"calibration map, {args.n_couplers} couplers: {res['map']['shares']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    log(f"wrote {os.path.relpath(args.out, os.path.dirname(HERE))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
