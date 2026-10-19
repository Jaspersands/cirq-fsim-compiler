"""
Draw the README figures from the compact results, benchmarks/results/compare_with_cirq.json.

    python benchmarks/compare_with_cirq.py     # writes the JSON (slow)
    python benchmarks/make_figures.py          # writes figures/*.png (seconds)

Figures:
    noise_aware.png        fraction of Haar-random U(4) where fewer native gates are better, against ε,
                           and the expected infidelity saved
    weyl_choice.png        where those unitaries sit in the Weyl chamber (CZ and √iSWAP)
    circuits_sycamore.png  two-qubit counts for the circuit suite on Sycamore
    timing.png             wall-clock time per two-qubit decomposition
    fsim_regions.png       the Sycamore four-gate region's Haar share, and two-gate shares on drifted couplers
                           (reads benchmarks/results/fsim_regions.json from benchmarks/fsim_regions.py)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ORDER = ["free", "sycamore", "sqrt_iswap", "cz"]
LABEL = {"free": "free FSim", "sycamore": "Sycamore", "sqrt_iswap": "√iSWAP", "cz": "CZ"}
COLOR = {"free": "#2f6aa8", "sycamore": "#c8692a", "sqrt_iswap": "#7b5bb0", "cz": "#2f8a5a"}
MARKER = {"free": "o", "sycamore": "s", "sqrt_iswap": "^", "cz": "D"}
OURS, OURS_LIGHT, CIRQ, CIRQ_DARK = "#2f6aa8", "#9dbbdc", "#9a9a9a", "#555555"
#: where each published ε is drawn: (gate set, label, which side of the line the label goes)
CITED = [("cz", "Willow CZ", "right"), ("sycamore", "Sycamore", "right"), ("sqrt_iswap", "SQiSW", "left")]


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.7,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e8e8e8", "grid.linewidth": 0.7,
        "xtick.color": "#555555", "ytick.color": "#555555", "axes.labelcolor": "#333333", "text.color": "#222222",
        "legend.frameon": False, "lines.linewidth": 1.6, "axes.titlelocation": "left", "axes.titlepad": 8,
    })


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)} ({os.path.getsize(path) / 1024:.0f} KB)")


def costs(profile, eps):
    c = [inf + n * eps for n, inf in enumerate(profile)]
    return int(np.argmin(c)), min(c), c[-1]


def fig_noise_aware(d, out):
    noise = d["noise"]
    eps = np.array(noise["eps_grid"])
    cited = [(d["meta"]["gate_errors"][gs], lab, side) for gs, lab, side in CITED]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})
    for gs in ORDER:
        sw = noise["sweep"][gs]
        a.plot(eps, 100 * np.array(sw["fraction_fewer"]), color=COLOR[gs], marker=MARKER[gs], ms=3.2, markevery=3, label=LABEL[gs])
        rel = []
        for e in eps:
            picks = [costs(p, e) for p in d["profiles"][gs]]
            rel.append(100 * np.sum([ce - c for _, c, ce in picks]) / np.sum([ce for _, _, ce in picks]))
        b.plot(eps, rel, color=COLOR[gs], marker=MARKER[gs], ms=3.2, markevery=3, label=LABEL[gs])
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlabel("two-qubit gate error ε (process infidelity)")
        for e, _, _ in cited:
            ax.axvline(e, color="#bbbbbb", lw=0.7, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(eps[0], eps[-1])
    a.set_ylim(0, 100)
    a.set_ylabel("Haar-random U(4), fewer gates better (%)")
    a.set_title("How often fewer gates are better (lower bound)")
    b.set_ylabel("expected infidelity saved (% of exact)")
    b.set_title("How much it saves")
    b.set_ylim(bottom=0)
    for ax in (a, b):
        top = ax.get_ylim()[1]
        for e, lab, side in cited:
            x = e * (0.95 if side == "right" else 1.05)
            ax.text(x, top * 0.98, f"{lab} {100 * e:.2f}%", rotation=90, ha=side, va="top", fontsize=6.0, color="#777777")
    a.legend(loc="upper left", handlelength=1.8)
    save(fig, os.path.join(out, "noise_aware.png"))


def fig_weyl(d, out):
    profiles = d["profiles"]
    w = np.array(d["haar_weyl"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.3})
    # CZ: two gates reach z = 0; the approximation wins when sin²z < ε
    e_cz = d["meta"]["gate_errors"]["cz"]
    pick = np.array([costs(p, e_cz)[0] < len(p) - 1 for p in profiles["cz"]])
    a.scatter(w[~pick, 0], w[~pick, 2], s=10, facecolors="none", edgecolors="#b0b0b0", linewidths=0.7, label="exact: 3 CZ")
    a.scatter(w[pick, 0], w[pick, 2], s=10, color=COLOR["cz"], label="noise-aware: 2 CZ")
    zc = np.arcsin(np.sqrt(e_cz))
    for sgn in (1, -1):
        a.axhline(sgn * zc, color=COLOR["cz"], lw=0.8, ls=(0, (3, 2)), label="|z| = arcsin √ε" if sgn > 0 else None)
    a.set_xlabel("Weyl coordinate x"); a.set_ylabel("Weyl coordinate z")
    a.set_title(f"CZ at ε = {100 * e_cz:.2f}%")
    a.legend(loc="upper left", handletextpad=0.3)
    a.set_ylim(top=a.get_ylim()[1] * 1.35)
    # √iSWAP: two gates reach x − y ≥ |z|
    e_sq = d["meta"]["gate_errors"]["sqrt_iswap"]
    profs = profiles["sqrt_iswap"]
    n_exact = np.array([len(p) - 1 for p in profs])
    pick = np.array([costs(p, e_sq)[0] < len(p) - 1 for p in profs])
    xm, zm = w[:, 0] - w[:, 1], np.abs(w[:, 2])
    two = (n_exact == 2) & ~pick
    three = (n_exact == 3) & ~pick
    b.scatter(xm[two], zm[two], s=10, facecolors="none", edgecolors="#b0b0b0", linewidths=0.7, label="exact: 2 √iSWAP")
    b.scatter(xm[three], zm[three], s=10, color="#666666", marker="x", linewidths=0.7, label="exact: 3 √iSWAP")
    b.scatter(xm[pick], zm[pick], s=10, color=COLOR["sqrt_iswap"], label="noise-aware: fewer")
    lim = max(xm.max(), zm.max()) * 1.05
    b.plot([0, lim], [0, lim], color=COLOR["sqrt_iswap"], lw=0.8, ls=(0, (3, 2)), label="|z| = x − y")
    b.set_xlabel("x − y"); b.set_ylabel("|z|")
    b.set_xlim(0, xm.max() * 1.05); b.set_ylim(0, zm.max() * 1.6)
    b.set_title(f"√iSWAP at ε = {100 * e_sq:.2f}%")
    b.legend(loc="upper right", handletextpad=0.3)
    save(fig, os.path.join(out, "weyl_choice.png"))


def fig_circuits(d, out):
    rows = [r for r in d.get("circuits", []) if r["gate_set"] == "sycamore"]
    if not rows:
        return
    names = [r["circuit"] for r in rows][::-1]
    by = {r["circuit"]: r for r in rows}
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(names) + 1.0))
    ax.grid(False)
    ax.grid(True, axis="x", color="#e8e8e8", lw=0.7)
    series = []
    if "google_n" in rows[0]:
        series.append(("google_n", "cirq_google SycamoreTargetGateset", CIRQ_DARK, "^", "full"))
    series.append(("cirq_n", "cirq-core routines per merged block", CIRQ, "s", "full"))
    series += [("ours_n", "this package, exact", OURS, "o", "full"), ("na_n", "this package, noise-aware", OURS, "o", "none")]
    offs = np.linspace(-0.24, 0.24, len(series))
    for (key, lab, col, mk, fill), dy in zip(series, offs):
        xs = [by[n][key] for n in names]
        ax.plot(xs, y + dy, ls="none", marker=mk, ms=4.2, color=col, mfc=col if fill == "full" else "white", mew=1.0, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sycamore FSim(π/2, π/6) gates")
    ax.set_xlim(left=0)
    ax.set_title("Circuit suite compiled to the Sycamore gate (all-to-all connectivity)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12 - 0.3 / len(names)), ncol=2, columnspacing=1.2, handletextpad=0.3)
    save(fig, os.path.join(out, "circuits_sycamore.png"))


def fig_timing(d, out):
    timing = d["timing"]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    pos, data, cols, ticks, x = [], [], [], [], 0
    for gs in ORDER:
        t = timing[gs]
        groups = [(t["ours_ms"], OURS), (t.get("unpruned_ms", []), OURS_LIGHT), (t["cirq_ms"], CIRQ)]
        if gs == "sycamore" and t.get("google_ms"):
            groups.append((t["google_ms"], CIRQ_DARK))
        start = x
        for vals, col in groups:
            if vals:
                pos.append(x); data.append(vals); cols.append(col); x += 1
        ticks.append(0.5 * (start + x - 1))
        x += 1
    bp = ax.boxplot(data, positions=pos, widths=0.7, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops={"color": "#222222", "lw": 1.0}, whiskerprops={"color": "#888888", "lw": 0.8},
                    capprops={"color": "#888888", "lw": 0.8}, boxprops={"lw": 0.0})
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([LABEL[g] for g in ORDER])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("ms per decomposition")
    ax.set_title("Wall-clock time per two-qubit decomposition, best of three calls (5–95% whiskers)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=OURS, label="this package"), Patch(color=OURS_LIGHT, label="without Weyl pruning"),
                       Patch(color=CIRQ, label="Cirq core"), Patch(color=CIRQ_DARK, label="cirq_google")],
              loc="upper left", ncol=4, handlelength=1.0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 4)
    save(fig, os.path.join(out, "timing.png"))


def fig_regions(r, out):
    syc = r["sycamore"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})
    s, share = np.array(syc["cdf"]["s"]), np.array(syc["cdf"]["share_below"])
    a.plot(s, share, color=COLOR["sycamore"])
    a.set_yscale("log"); a.set_xscale("log")
    a.axvline(np.pi / 8, color="#bbbbbb", lw=0.7, ls=(0, (2, 2)), zorder=0)
    a.plot([np.pi / 8], [syc["four_gate_share"]], ls="none", marker="o", ms=4, color=COLOR["sycamore"])
    a.annotate(f"x + y + |z| < π/8\n{1e4 * syc['four_gate_share']:.2f} × 10⁻⁴, four gates",
               (np.pi / 8, syc["four_gate_share"]),
               xytext=(8, -2), textcoords="offset points", fontsize=6.8, color="#444444", va="top")
    a.set_xlabel("s"); a.set_ylabel("Haar share with x + y + |z| < s")
    a.set_title("Sycamore: the four-gate region")
    a.set_xlim(s[0], s[-1])
    drift = r["drift"]
    d = np.array([row["delta_theta"] for row in drift])
    for n, mk in [("2", "o"), ("3", "s")]:
        b.plot(d, [100 * row["shares"].get(n, 0.0) for row in drift], color=COLOR["sycamore"] if n == "2" else "#9a9a9a",
               marker=mk, ms=3.6)
    pair = r["pair_33_34"]
    dp = abs(pair["native"][0] - np.pi / 2)
    b.plot([dp], [100 * pair["shares"].get("2", 0.0)], ls="none", marker="D", ms=4.2, mfc="white", mew=1.0,
           color=COLOR["sycamore"])
    b.set_xlabel("|θ − π/2| (rad); φ = π/6 except the pair"); b.set_ylabel("Haar-random targets (%)")
    b.set_title("Minimum count on a drifted coupler")
    b.set_ylim(0, 100); b.set_xlim(left=0)
    lab = dict(fontsize=7.2, color="#444444")
    b.text(0.215, 66, "two gates", va="center", **lab)
    b.text(0.215, 31, "three gates", va="center", **lab)
    b.text(dp - 0.008, 95, "pair (3,3)–(3,4)", ha="right", va="center", **lab)
    save(fig, os.path.join(out, "fsim_regions.png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results", "compare_with_cirq.json"))
    ap.add_argument("--regions", default=os.path.join(HERE, "results", "fsim_regions.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    args = ap.parse_args(argv)
    with open(args.results) as f:
        d = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    style()
    fig_noise_aware(d, args.out)
    fig_weyl(d, args.out)
    fig_circuits(d, args.out)
    fig_timing(d, args.out)
    if os.path.exists(args.regions):
        with open(args.regions) as f:
            fig_regions(json.load(f), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
