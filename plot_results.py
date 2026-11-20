"""
Summarize results.json from wrv_sim.py: selection probabilities, the cancellation
ratio, correlations between the three fluctuation terms, and log-log growth slopes.
Writes figures to figures/ and numbers to summary.json.

Usage: python3 plot_results.py results.json
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
R = json.load(open(path))
rec = np.array(R["rec"])                  # (reps, ncp, ncand, nxi, 6 terms)
ns = np.array(R["checkpoints"])
names = R["candidate_names"]
xis = R["settings"]["xis"]
T = {t: k for k, t in enumerate(R["settings"]["terms"])}
reps = rec.shape[0]
opt = names.index("opt")
os.makedirs("figures", exist_ok=True)

INK, MUTED, GRIDC = "#1f2328", "#6e7781", "#d0d7de"
COL = {"over40": "#0969da", "over50": "#8250df", "under20": "#bf8700",
       "A": "#0969da", "B": "#bf3989", "C": "#1a7f37", "sum": INK, "signal": "#cf222e"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC,
                     "grid.linewidth": 0.6, "legend.frameon": False})


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def diff(other, xi_k, term):
    return rec[:, :, names.index(other), xi_k, T[term]] - rec[:, :, opt, xi_k, T[term]]


def slope(x, y, k=5):
    """Least-squares slope of log y on log n over the last k checkpoints."""
    return float(np.polyfit(np.log(x[-k:]), np.log(y[-k:]), 1)[0])


summary = {"reps": reps, "n": ns.tolist(), "checks": R["checks"], "comparisons": {}}
for other in ("over40", "over50", "under20"):
    for xk, xi in enumerate(xis):
        dRV = diff(other, xk, "rv")
        dA, dB, dC = diff(other, xk, "A"), diff(other, xk, "B"), diff(other, xk, "C")
        dS = diff(other, xk, "signal")
        F = dA + dB - dC                          # total fluctuation of the score gap
        p = (dRV > 0).mean(0)
        lo, hi = wilson(p, reps)
        sep = dA.var(0, ddof=1) + dB.var(0, ddof=1) + dC.var(0, ddof=1)
        ratio = F.var(0, ddof=1) / sep
        corr = [np.corrcoef(np.stack([dA[:, k], dB[:, k], -dC[:, k]]))
                for k in range(len(ns))]
        sig = dS.mean(0)
        sd = {t: v.std(0, ddof=1) for t, v in (("A", dA), ("B", dB), ("C", dC), ("sum", F))}
        summary["comparisons"][f"opt_vs_{other}_xi{int(xi)}"] = {
            "p_correct": p.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
            "cancellation_ratio": ratio.tolist(),
            "corr_AB": [float(c[0, 1]) for c in corr],
            "corr_A_minusC": [float(c[0, 2]) for c in corr],
            "corr_B_minusC": [float(c[1, 2]) for c in corr],
            "signal_mean": sig.tolist(),
            "sd": {k: v.tolist() for k, v in sd.items()},
            "snr": (sig / sd["sum"]).tolist(),
            "slopes_last5": {"signal": slope(ns, sig),
                             **{f"sd_{k}": slope(ns, v) for k, v in sd.items()}},
        }

json.dump(summary, open("summary.json", "w"), indent=1)


def key(o, xi):
    return summary["comparisons"][f"opt_vs_{o}_xi{xi}"]


# Figure 1: selection probability (xi = 1)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for o in ("over40", "over50", "under20"):
    c = key(o, 1)
    ax.plot(ns, c["p_correct"], "-o", ms=3.5, lw=1.8, color=COL[o], label=f"opt vs {o}")
    ax.fill_between(ns, c["ci_lo"], c["ci_hi"], color=COL[o], alpha=0.15, lw=0)
ax.set_xscale("log")
ax.set_xlabel("sample size n")
ax.set_ylabel("P(wRV picks the optimal sequence)")
ax.set_title("wRV selection accuracy, ξ = 1", loc="left", color=INK)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("figures/selection_probability.png", dpi=180)

# Figure 2: signal vs fluctuation scale (opt vs over40, xi = 1)
c = key("over40", 1)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(ns, c["signal_mean"], "-", lw=2.4, color=COL["signal"],
        label=f"signal gap (slope {c['slopes_last5']['signal']:.2f})")
for t, lab in (("A", "sd ΔA"), ("B", "sd ΔB"), ("C", "sd ΔC"), ("sum", "sd of ΔA+ΔB−ΔC")):
    ax.plot(ns, c["sd"][t], "--" if t != "sum" else "-", lw=1.8 if t != "sum" else 2.2,
            color=COL[t], label=f"{lab} (slope {c['slopes_last5']['sd_' + t]:.2f})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("sample size n")
ax.set_ylabel("size (log scale)")
ax.set_title("Signal vs. noise in the score gap: opt vs over40, ξ = 1", loc="left", color=INK)
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig("figures/signal_vs_fluctuation.png", dpi=180)

# Figure 3: cancellation ratio and correlations (opt vs over40, xi = 1)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax = axes[0]
for o in ("over40", "over50"):
    ax.plot(ns, key(o, 1)["cancellation_ratio"], "-o", ms=3.5, lw=1.8, color=COL[o], label=f"opt vs {o}")
ax.axhline(1, color=MUTED, lw=1, ls=":")
ax.set_xscale("log")
ax.set_ylim(0, 1.2)
ax.set_xlabel("sample size n")
ax.set_ylabel("Var(sum) / sum of Vars")
ax.set_title("Cancellation ratio (1 = no cancellation)", loc="left", color=INK, fontsize=10)
ax.legend(loc="lower right")
ax = axes[1]
for lab, k, col in (("corr(ΔA, ΔB)", "corr_AB", COL["A"]),
                    ("corr(ΔA, −ΔC)", "corr_A_minusC", COL["C"]),
                    ("corr(ΔB, −ΔC)", "corr_B_minusC", COL["B"])):
    ax.plot(ns, c[k], "-o", ms=3.5, lw=1.8, color=col, label=lab)
ax.axhline(0, color=MUTED, lw=1, ls=":")
ax.set_xscale("log")
ax.set_ylim(-1, 1)
ax.set_xlabel("sample size n")
ax.set_title("Correlations between the terms (opt vs over40)", loc="left", color=INK, fontsize=10)
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig("figures/cancellation.png", dpi=180)

# console report
for o in ("over40", "over50", "under20"):
    for xi in (0, 1, 2):
        c = key(o, xi)
        print(f"opt vs {o:7s} xi={xi}: P(correct) {c['p_correct'][0]:.3f} -> {c['p_correct'][-1]:.3f} | "
              f"ratio {min(c['cancellation_ratio']):.2f}-{max(c['cancellation_ratio']):.2f} | "
              f"slopes {json.dumps({k: round(v, 2) for k, v in c['slopes_last5'].items()})}")
c = key("over40", 1)
print("over40 xi=1 corr B,-C:", [round(v, 2) for v in c["corr_B_minusC"]])
print("over40 xi=1 corr A,B:", [round(v, 2) for v in c["corr_AB"]])
print("over40 xi=1 corr A,-C:", [round(v, 2) for v in c["corr_A_minusC"]])
print("over40 xi=1 snr:", [round(v, 2) for v in c["snr"]])
print("checks:", R["checks"])
