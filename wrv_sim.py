"""
Weighted rolling validation (wRV) at the overfitting boundary.

Simulation for Open Problem 07 (Regression Analysis, Rutgers, Fall 2026).
Source paper: Zhang & Lei (2025), "Online estimation with rolling validation",
Annals of Statistics 53(6); arXiv:2310.12140v4.

Model
    X ~ Uniform[0, 1],  Y = f0(X) + eps,  eps = +/- sigma with equal probability.
    Trig basis: phi_0 = 1, phi_{2k-1} = sqrt(2) cos(2 pi k x), phi_{2k} = sqrt(2) sin(2 pi k x).
    The basis is orthonormal under Uniform[0,1], so the projection estimator
    trained on the first m points uses theta_hat_j = (1/m) sum_t Y_t phi_j(X_t).

Candidates (sequences of model sizes): J(m) = min(ceil(m^a), J_CAP).
    opt     a = 1/(2s+1)  (rate optimal; 1/3 when s = 1)
    over40  a = 0.40      (overfitting but consistent: the open boundary case)
    over50  a = 0.50      (overfitting, also on the boundary)
    under20 a = 0.20      (underfitting control, covered by the paper's theory)

wRV score (paper's indexing): RV_n = sum_{i} i^xi (fhat_{i-1}(X_i) - Y_i)^2.
The model scored on point i is trained only on points 1..i-1.

Decomposition (paper eq. 15). Write D_i = fhat_{i-1} - f0, w_i = i^xi,
r_i = E[D_i(X)^2 | first i-1 points] (conditional risk), m_i = E[r_i]. Then exactly
    RV_n = sum w eps^2              (noise, identical for all candidates)
         + sum w m                  (signal: expected risk)
         + A = sum w (D^2 - r)      (martingale term)
         + B = sum w (r - m)        (non-martingale conditional-risk term)
         - C = 2 sum w eps D        (noise x estimation-error martingale)
Because f0 lies in the span of the first J_CAP basis functions, r_i and m_i are
computed exactly with Parseval's identity (no Monte Carlo calibration bank needed).

Usage
    python3 wrv_sim.py --reps 200 --nmax 100000 --out results.json
"""
import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np

# ---------------------------------------------------------------- settings
# Defaults reproduce the main run. Robustness runs override them with environment
# variables, e.g.  WRV_JCAP=1200 WRV_TRUTH=powerlog python3 wrv_sim.py ...
import os

S = float(os.environ.get("WRV_S", 1.0))          # smoothness used for the truth and the optimal rate
SIGMA = 1.0        # noise level (bounded noise +/- SIGMA)
J_CAP = int(os.environ.get("WRV_JCAP", 400))     # basis functions available (truth lives here too)
C_TRUTH = float(os.environ.get("WRV_CTRUTH", 0.65))  # scale of the truth's coefficients
TRUTH = os.environ.get("WRV_TRUTH", "power")     # "power": k^-(s+1/2); "powerlog": k^-(s+1/2) / (1 + log k)
BURN = 100         # start scoring once the model has >= BURN training points
XIS = (0.0, 1.0, 2.0)
BLOCK = 2500
GRID = int(os.environ.get("WRV_GRID", max(8192, 8 * J_CAP)))  # quadrature grid for exact integrals

_default_cands = {"opt": 1.0 / (2 * S + 1), "over40": 0.40, "over50": 0.50, "under20": 0.20}
CANDIDATES = json.loads(os.environ["WRV_CANDS"]) if "WRV_CANDS" in os.environ else _default_cands
CANDIDATES["opt"] = 1.0 / (2 * S + 1)
TERMS = ("rv", "noise", "signal", "A", "B", "C")


def matmul(a, b):
    """a @ b with a hard finiteness check.
    On macOS, NumPy linked to Apple's Accelerate library can raise spurious floating-point
    warnings inside matrix products even when the result is correct. We silence those flags
    only for the product itself, then stop if any entry is actually NaN or infinite."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        out = np.asarray(a) @ np.asarray(b)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite value in matrix product")
    return out


# ---------------------------------------------------------------- basis and truth
def basis(x):
    """Return the (len(x), J_CAP) matrix of trig basis functions."""
    x = np.asarray(x, dtype=float)
    out = np.empty((x.size, J_CAP))
    out[:, 0] = 1.0
    k = np.arange(1, (J_CAP - 1) // 2 + 2)
    ang = 2 * np.pi * np.outer(x, k)
    c = math.sqrt(2) * np.cos(ang)
    s = math.sqrt(2) * np.sin(ang)
    out[:, 1::2] = c[:, : out[:, 1::2].shape[1]]
    out[:, 2::2] = s[:, : out[:, 2::2].shape[1]]
    return out


def truth_coefs():
    """theta_0 = 0; both coefficients at frequency k equal C_TRUTH * k^-(s + 1/2)."""
    th = np.zeros(J_CAP)
    j = np.arange(1, J_CAP)
    k = (j + 1) // 2
    th[1:] = C_TRUTH * k ** (-(S + 0.5))
    if TRUTH == "powerlog":   # finite Sobolev-s norm even without the cap
        th[1:] /= 1.0 + np.log(k)
    elif TRUTH != "power":
        raise ValueError(f"unknown WRV_TRUTH {TRUTH}")
    return th


THETA = truth_coefs()
GRID_X = (np.arange(GRID) + 0.5) / GRID
PHI_GRID = basis(GRID_X)
F0_GRID = matmul(PHI_GRID, THETA)
# V_j = Var(Y phi_j(X)) = E[f0^2 phi_j^2] + sigma^2 - theta_j^2  (E phi_j^2 = 1)
V = (F0_GRID[:, None] ** 2 * PHI_GRID ** 2).mean(0) + SIGMA ** 2 - THETA ** 2
CUM_V = np.concatenate([[0.0], np.cumsum(V)])                  # CUM_V[J] = sum_{j<J} V_j
TAIL = np.concatenate([np.cumsum((THETA ** 2)[::-1])[::-1], [0.0]])  # TAIL[J] = sum_{j>=J} theta_j^2


def sizes(m, a):
    """Model size after m training points: the smallest integer >= m^a, capped at J_CAP.
    Floating point can put an exact integer power just above the integer
    (e.g. 32768**0.2 = 8.000000000000002), and a plain ceil would then add one term.
    Values within 1e-9 (relative) of an integer are treated as that integer."""
    x = np.asarray(m, dtype=float) ** a
    k = np.rint(x)
    J = np.where(np.abs(x - k) <= 1e-9 * np.maximum(1.0, x), k, np.ceil(x)).astype(int)
    return np.minimum(J, J_CAP)


def exact_risk(m, a):
    """m_i: expected integrated squared error after m training points."""
    J = sizes(m, a)
    return CUM_V[J] / m + TAIL[J]


# ---------------------------------------------------------------- one trajectory
def run_trajectory(args):
    seed, nmax, checkpoints, keep_points = args
    rng = np.random.default_rng(seed)
    X = rng.random(nmax)
    eps = SIGMA * rng.choice([-1.0, 1.0], size=nmax)

    cands = list(CANDIDATES.items())
    nc, nx = len(cands), len(XIS)
    totals = np.zeros((nc, nx, len(TERMS)))
    rec = np.zeros((len(checkpoints), nc, nx, len(TERMS)))
    risk_at = np.zeros((len(checkpoints), nc, 2))  # (r_i, m_i) at checkpoint times
    cp_index = {c: k for k, c in enumerate(checkpoints)}
    S_prev = np.zeros(J_CAP)
    kept = {name: np.full(nmax + 1, np.nan) for name, _ in cands} if keep_points else None
    Ycache = np.empty(nmax) if keep_points else None

    for start in range(0, nmax, BLOCK):
        stop = min(start + BLOCK, nmax)
        i = np.arange(start + 1, stop + 1)          # 1-based time of the scored point
        Phi = basis(X[start:stop])
        f0 = matmul(Phi, THETA)
        Y = f0 + eps[start:stop]
        if keep_points:
            Ycache[start:stop] = Y
        P = Y[:, None] * Phi
        csum = np.cumsum(P, axis=0)
        S_excl = S_prev + csum - P                  # sum over points 1..i-1 only
        S_prev = S_prev + csum[-1]
        mtrain = i - 1
        ok = mtrain >= BURN
        if not ok.any():
            continue
        i_ok, m_ok = i[ok], mtrain[ok]
        th_hat = S_excl[ok] / m_ok[:, None]
        Phi_ok, f0_ok, e_ok, Y_ok = Phi[ok], f0[ok], eps[start:stop][ok], Y[ok]
        dcoef = th_hat - THETA
        cum_d2 = np.concatenate([np.zeros((dcoef.shape[0], 1)), np.cumsum(dcoef ** 2, axis=1)], axis=1)
        cum_pred = np.concatenate([np.zeros((dcoef.shape[0], 1)), np.cumsum(th_hat * Phi_ok, axis=1)], axis=1)
        rows = np.arange(dcoef.shape[0])
        W = np.stack([i_ok.astype(float) ** xi for xi in XIS])   # (nx, b)

        for c, (name, a) in enumerate(cands):
            J = sizes(m_ok, a)
            fhat = cum_pred[rows, J]
            D = fhat - f0_ok
            r = cum_d2[rows, J] + TAIL[J]
            mrisk = CUM_V[J] / m_ok + TAIL[J]
            per = np.stack([
                (fhat - Y_ok) ** 2,        # rv
                e_ok ** 2,                 # noise
                mrisk,                     # signal
                D ** 2 - r,                # A
                r - mrisk,                 # B
                2 * e_ok * D,              # C
            ])                              # (6, b)
            contrib = W[:, None, :] * per[None, :, :]            # (nx, 6, b)
            run = totals[c][:, :, None] + np.cumsum(contrib, axis=2)
            totals[c] = run[:, :, -1]
            for pos, t in enumerate(i_ok):
                k = cp_index.get(int(t))
                if k is not None:
                    rec[k, c] = run[:, :, pos]
                    risk_at[k, c] = (r[pos], mrisk[pos])
            if keep_points:
                kept[name][i_ok] = fhat

    out = {"rec": rec, "risk_at": risk_at}
    if keep_points:
        out["kept"] = kept
        out["X"], out["Y"] = X, Ycache
    return out


# ---------------------------------------------------------------- checks
def check_orthonormality():
    G = matmul(PHI_GRID.T, PHI_GRID) / GRID
    return float(np.abs(G - np.eye(J_CAP)).max())


def check_no_leakage(traj, n_checks=5, seed=0):
    """Refit from scratch at random times; predictions must match the running-sum version."""
    rng = np.random.default_rng(seed)
    X, Y = traj["X"], traj["Y"]
    n = X.size
    worst = 0.0
    for t in rng.integers(BURN + 2, n, size=n_checks):
        m = t - 1                                   # train on points 1..t-1 (0-based :m)
        th = (Y[:m, None] * basis(X[:m])).mean(0)
        phi_t = basis(X[t - 1 : t])[0]
        for name, a in CANDIDATES.items():
            J = int(sizes(m, a))
            pred = float(matmul(th[:J], phi_t[:J]))
            worst = max(worst, abs(pred - traj["kept"][name][t]))
    return worst


def check_decomposition(rec):
    rv, noise, signal, A, B, C = (rec[..., k] for k in range(6))
    recomposed = noise + signal + A + B - C
    return float(np.max(np.abs(recomposed - rv) / np.abs(rv)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--nmax", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    checkpoints = sorted({int(round(v)) for v in np.logspace(3, math.log10(args.nmax), 11)})
    t0 = time.time()

    # checks on a small trajectory first
    small = run_trajectory((args.seed - 1, 5000, [5000], True))
    checks = {
        "basis_orthonormality_max_error": check_orthonormality(),
        "no_leakage_max_abs_diff": check_no_leakage(small),
    }
    print("checks:", checks, flush=True)

    jobs = [(args.seed + r, args.nmax, checkpoints, False) for r in range(args.reps)]
    with Pool(args.workers) as pool:
        outs = pool.map(run_trajectory, jobs)
    rec = np.stack([o["rec"] for o in outs])            # (reps, ncp, nc, nx, 6)
    risk = np.stack([o["risk_at"] for o in outs])       # (reps, ncp, nc, 2)

    checks["decomposition_max_rel_error"] = check_decomposition(rec)
    ratio = risk[..., 0].mean(0) / risk[..., 1].mean(0)  # MC mean of r_i / exact m_i
    checks["centering_ratio_min"] = float(ratio.min())
    checks["centering_ratio_max"] = float(ratio.max())
    print("checks:", checks, flush=True)

    result = {
        "settings": {
            "s": S, "sigma": SIGMA, "J_CAP": J_CAP, "C_TRUTH": C_TRUTH, "burn_in": BURN,
            "truth": TRUTH,
            "xis": list(XIS), "candidates": CANDIDATES, "reps": args.reps,
            "nmax": args.nmax, "seed": args.seed, "terms": list(TERMS),
            "noise": "Rademacher +/- sigma", "basis": "trig, orthonormal on [0,1]",
        },
        "checkpoints": checkpoints,
        "candidate_names": list(CANDIDATES),
        "checks": checks,
        "rec": rec.tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"done in {result['seconds']} s -> {args.out}")


if __name__ == "__main__":
    main()
