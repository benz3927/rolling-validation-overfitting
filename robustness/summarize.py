"""
Summarize the main run and the robustness runs side by side.
Writes robustness/summary.md and figures/robustness.png.

Usage (from the repo root): python3 robustness/summarize.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("main: s=1, cap 400", "results.json"),
    ("cap 1200", "robustness/cap1200_power.json"),
    ("cap 1200, log-adjusted truth", "robustness/cap1200_powerlog.json"),
    ("s=2, cap 400", "robustness/s2_power.json"),
]


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def slope(x, y, k=5):
    """Log-log slope over the last k checkpoints; nan if the quantity is not positive there
    (e.g. a signal gap that changes sign), since a power-law slope is then undefined."""
    y = np.asarray(y[-k:])
    if np.any(y <= 0):
        return float("nan")
    return float(np.polyfit(np.log(x[-k:]), np.log(y), 1)[0])


def analyze(path):
    R = json.load(open(path))
    rec = np.array(R["rec"])
    ns = np.array(R["checkpoints"])
    names = R["candidate_names"]
    T = {t: k for k, t in enumerate(R["settings"]["terms"])}
    opt = names.index("opt")
    reps = rec.shape[0]
    out = {"settings": R["settings"], "checks": R["checks"], "n": ns, "reps": reps, "cmp": {}}
    for other in names:
        if other == "opt":
            continue
        res = {}
        for xk, xi in enumerate(R["settings"]["xis"]):
            d = {t: rec[:, :, names.index(other), xk, T[t]] - rec[:, :, opt, xk, T[t]] for t in T}
            F = d["A"] + d["B"] - d["C"]
            p = (d["rv"] > 0).mean(0)
            lo, hi = wilson(p, reps)
            sep = sum(d[t].var(0, ddof=1) for t in ("A", "B", "C"))
            sig = d["signal"].mean(0)
            sd = F.std(0, ddof=1)
            res[int(xi)] = {
                "p": p, "lo": lo, "hi": hi,
                "ratio": F.var(0, ddof=1) / sep,
                "corr_B_mC": np.array([np.corrcoef(d["B"][:, k], -d["C"][:, k])[0, 1] for k in range(len(ns))]),
                "snr": sig / sd,
                "slope_signal": slope(ns, sig),
                "slope_sd": slope(ns, sd),
                "slope_sdB": slope(ns, d["B"].std(0, ddof=1)),
                "slope_sdC": slope(ns, d["C"].std(0, ddof=1)),
            }
        out["cmp"][other] = (R["settings"]["candidates"][other], res)
    return out


results = [(label, analyze(path)) for label, path in RUNS if os.path.exists(path)]

lines = ["# Robustness summary", "",
         "Each row compares the rate-optimal sequence with one alternative, using weight xi = 1 unless noted.",
         "P(correct) is the share of trajectories in which wRV picks the optimal sequence (95% Wilson interval).",
         "Slopes are log-log slopes over the last five checkpoints (n from about 16,000 to 100,000).", ""]
for label, A in results:
    st = A["settings"]
    lines.append(f"## {label}")
    lines.append(f"truth = {st.get('truth', 'power')}, s = {st['s']}, J_CAP = {st['J_CAP']}, "
                 f"reps = {A['reps']}, opt exponent = {st['candidates']['opt']:.3f}")
    ck = A["checks"]
    lines.append(f"checks: orthonormality {ck['basis_orthonormality_max_error']:.1e}, "
                 f"leakage {ck['no_leakage_max_abs_diff']:.1e}, decomposition {ck['decomposition_max_rel_error']:.1e}, "
                 f"centering {ck['centering_ratio_min']:.3f} to {ck['centering_ratio_max']:.3f}")
    lines.append("")
    lines.append("| vs | a | P(correct) n=1e3 | P(correct) n=1e5 | P at n=1e5, xi=0 / 2 | Var ratio | corr(dB,-dC) | slope signal | slope sd(sum) | sd dB / dC | SNR 1e3 -> 1e5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for other, (a, res) in A["cmp"].items():
        r = res[1]
        lines.append(
            f"| {other} | {a:.2f} | {r['p'][0]:.3f} ({r['lo'][0]:.2f}-{r['hi'][0]:.2f}) | "
            f"{r['p'][-1]:.3f} ({r['lo'][-1]:.2f}-{r['hi'][-1]:.2f}) | {res[0]['p'][-1]:.3f} / {res[2]['p'][-1]:.3f} | "
            f"{r['ratio'].min():.2f}-{r['ratio'].max():.2f} | {r['corr_B_mC'].mean():+.2f} | "
            f"{r['slope_signal']:.2f} | {r['slope_sd']:.2f} | {r['slope_sdB']:.2f} / {r['slope_sdC']:.2f} | "
            f"{r['snr'][0]:.2f} -> {r['snr'][-1]:.2f} |")
    lines.append("")
open("robustness/summary.md", "w").write("\n".join(lines))
print("\n".join(lines))

# Figure: the mildest overfitter in each setting
INK, MUTED, GRIDC = "#1f2328", "#6e7781", "#d0d7de"
COLS = ["#1f2328", "#0969da", "#bf3989", "#1a7f37"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.6, "legend.frameon": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
for (label, A), col in zip(results, COLS):
    overs = {o: v for o, v in A["cmp"].items() if v[0] > A["settings"]["candidates"]["opt"]}
    o = min(overs, key=lambda k: overs[k][0])
    r = overs[o][1][1]
    lab = f"{label} (opt vs {o})"
    axes[0].plot(A["n"], r["p"], "-o", ms=3, lw=1.8, color=col, label=lab)
    axes[0].fill_between(A["n"], r["lo"], r["hi"], color=col, alpha=0.12, lw=0)
    axes[1].plot(A["n"], r["snr"], "-o", ms=3, lw=1.8, color=col, label=lab)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("sample size n")
axes[1].axhline(0, color=MUTED, lw=1, ls=":")
axes[0].axhline(0.5, color=MUTED, lw=1, ls=":")
axes[0].set_ylabel("P(wRV picks opt)")
axes[1].set_ylabel("mean signal gap / sd of fluctuation")
axes[0].set_title("Selection accuracy, mildest overfitter, ξ = 1", loc="left", color=INK, fontsize=10)
axes[1].set_title("Signal-to-noise ratio (below 0: the overfitter has lower risk)", loc="left", color=INK, fontsize=10)
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/robustness.png", dpi=180)
print("wrote figures/robustness.png")
