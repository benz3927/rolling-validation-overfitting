"""
Exact variance of the paired B term, Delta B_n = B_n(h) - B_n(g), for nested projection
estimators, compared with the saved simulation (results.json).

Derivation (see the blog, section "The B term, exactly").
For m training points write delta_{j,m} = theta_hat_{j,m} - theta_j = S_{j,m} / m, where
S_{j,m} = sum_{t<=m} Z_{j,t} and Z_{j,t} = Y_t phi_j(X_t) - theta_j are iid, mean zero.
Since r_m - E r_m = sum_{j<J(m)} (delta_{j,m}^2 - V_j/m), the tail terms cancel and
    Delta B_n = sum_{i} w_i sum_{j in band(m)} (delta_{j,m}^2 - V_j/m),   m = i - 1,
where band(m) = [J_g(m), J_h(m)): only coefficients used by h but not by g remain.

For m <= m', the part of S_{l,m'} after time m is independent of everything up to m and has
mean zero, so Cov(S_{j,m}^2, S_{l,m'}^2) = Cov(S_{j,m}^2, S_{l,m}^2). For an iid sum,
    Cov(S_{j,m}^2, S_{l,m}^2) = m (Q_jl - V_j V_l) + 2 m (m - 1) Sigma_jl^2,
with Sigma_jl = E[Z_j Z_l], V_j = Sigma_jj, Q_jl = E[Z_j^2 Z_l^2]. Hence
    Cov(delta_{j,m}^2, delta_{l,m'}^2) = a_jl(m) / m'^2,
    a_jl(m) = ((Q_jl - V_j V_l) + 2 (m - 1) Sigma_jl^2) / m,     m = min, m' = max.
Each coefficient j sits in the band for one interval of m, and the kernel factorizes into
a function of the earlier time times a function of the later time, so the double sum over
time pairs reduces to prefix sums. Everything below is exact for the simulated model.

Usage: python3 check_B_variance.py   (reads results.json written with the default settings)
"""
import json
from pathlib import Path

import numpy as np
import wrv_sim as w


def moments(jmax):
    """Sigma, Q for Z_j = Y phi_j(X) - theta_j, exact over X (grid) and eps = +/- sigma."""
    phi = w.PHI_GRID[:, :jmax]
    f0 = w.F0_GRID
    th = w.THETA[:jmax]
    Zs = [(f0 + e)[:, None] * phi - th for e in (w.SIGMA, -w.SIGMA)]
    Z = np.vstack(Zs)
    N = Z.shape[0]
    Sigma = w.matmul(Z.T, Z) / N
    Z2 = Z ** 2
    Q = w.matmul(Z2.T, Z2) / N
    return Sigma, Q


def band_intervals(a_g, a_h, m_lo, m_hi):
    """For each coefficient j, the interval [L_j, U_j] of m in [m_lo, m_hi] with J_g(m) <= j < J_h(m)."""
    m = np.arange(m_lo, m_hi + 1)
    Jg, Jh = w.sizes(m, a_g), w.sizes(m, a_h)
    jmax = int(Jh.max())
    L = np.full(jmax, m_hi + 1)
    U = np.full(jmax, m_lo - 1)
    for j in range(jmax):
        inb = (Jg <= j) & (j < Jh)
        if inb.any():
            idx = np.nonzero(inb)[0]
            L[j], U[j] = m[idx[0]], m[idx[-1]]
    return L, U, jmax


def var_delta_B(n, xi, a_g, a_h):
    m_lo, m_hi = w.BURN, n - 1                  # scored times i = BURN+1..n use m = i-1 points
    L, U, jmax = band_intervals(a_g, a_h, m_lo, m_hi)
    Sigma, Q = moments(jmax)
    V = np.diag(Sigma)
    c1 = Q - np.outer(V, V)
    c2 = 2 * Sigma ** 2

    mm = np.arange(0, m_hi + 1, dtype=float)
    wgt = np.where(mm >= m_lo, (mm + 1) ** xi, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(mm > 0, 1 / mm, 0.0)
    b = inv ** 2
    pre = lambda v: np.concatenate([[0.0], np.cumsum(v)])   # pre[x+1] = sum_{m<=x}
    A1, A2 = pre(wgt * inv), pre(wgt * (mm - 1) * inv)
    Bp = pre(wgt * b)
    G1, G2 = pre(wgt * b * A1[1:]), pre(wgt * b * A2[1:])
    D1, D2 = pre(wgt ** 2 * b * inv), pre(wgt ** 2 * b * (mm - 1) * inv)

    def rng(P, lo, hi):                          # sum over m in [lo, hi], 0 if empty
        lo = np.clip(lo, 0, m_hi + 1)
        hi = np.clip(hi, -1, m_hi)
        return np.where(hi >= lo, P[np.maximum(hi, -1) + 1] - P[lo], 0.0)

    Lj, Uj = L[:, None], U[:, None]
    Ll, Ul = L[None, :], U[None, :]

    def T(Lj, Uj, Ll, Ul, c1, c2):
        # sum over m in [Lj,Uj] (time of j), m' in [Ll,Ul] (time of l), m <= m'
        lo1, hi1 = np.maximum(Ll, Lj), np.minimum(Ul, Uj)
        Abase1, Abase2 = A1[np.clip(Lj, 0, m_hi + 1)], A2[np.clip(Lj, 0, m_hi + 1)]
        part1 = c1 * (rng(G1, lo1, hi1) - Abase1 * rng(Bp, lo1, hi1)) + \
                c2 * (rng(G2, lo1, hi1) - Abase2 * rng(Bp, lo1, hi1))
        lo2 = np.maximum(np.maximum(Ll, Lj), Uj + 1)
        span1 = rng(A1, Lj, Uj)
        span2 = rng(A2, Lj, Uj)
        part2 = (c1 * span1 + c2 * span2) * rng(Bp, lo2, Ul)
        return part1 + part2

    tot = T(Lj, Uj, Ll, Ul, c1, c2) + T(Ll, Ul, Lj, Uj, c1, c2)
    lo_d, hi_d = np.maximum(Lj, Ll), np.minimum(Uj, Ul)
    tot -= c1 * rng(D1, lo_d, hi_d) + c2 * rng(D2, lo_d, hi_d)
    return float(tot.sum())


def main():
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("results.json"))
    if src.endswith(".gz"):
        import gzip
        with gzip.open(src, "rt") as fh:
            data = json.load(fh)
    else:
        data = json.loads(Path(src).read_text())
    st = data["settings"]
    for key, val in [("J_CAP", w.J_CAP), ("s", w.S), ("C_TRUTH", w.C_TRUTH), ("burn_in", w.BURN)]:
        if st[key] != val:
            raise ValueError(f"saved settings differ from current code: {key}")
    rec = np.asarray(data["rec"])
    names = data["candidate_names"]
    ns = data["checkpoints"]
    kB = st["terms"].index("B")
    out = {}
    print("comparison  xi       n     exact_sd(dB)   simulated_sd   sim/exact")
    for other in ("over40", "over50"):
        for xk, xi in enumerate(st["xis"]):
            d = rec[:, :, names.index(other), xk, kB] - rec[:, :, names.index("opt"), xk, kB]
            sim = d.std(0, ddof=1)
            ex = np.array([np.sqrt(var_delta_B(n, xi, st["candidates"]["opt"], st["candidates"][other])) for n in ns])
            out[f"{other}_xi{int(xi)}"] = {"n": ns, "exact_sd": ex.tolist(), "sim_sd": sim.tolist()}
            for k in range(len(ns)):
                if k in (0, 5, len(ns) - 1):
                    print(f"opt-{other:6s} {xi:3g} {ns[k]:7d}  {ex[k]:13.5g}  {sim[k]:13.5g}   {sim[k]/ex[k]:.3f}")
    Path("B_variance_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
