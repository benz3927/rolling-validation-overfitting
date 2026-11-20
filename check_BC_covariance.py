"""
Exact covariance of the paired B and C terms, Cov(Delta B_n, Delta C_n), compared with
the simulation. This covariance is what produces the partial cancellation.

Delta C_n = 2 sum_i w_i eps_i H_{i-1}(X_i),  H = fhat_h - fhat_g = sum_{l in band(i-1)} theta_hat_l phi_l.
Delta B_n = sum_k w_k sum_{j in band(k-1)} (delta_{j,k-1}^2 - V_j/(k-1)).

Point i enters the training data of every later fit (k - 1 >= i). Expanding delta^2 and
using independence of past, present and future points gives, for k - 1 >= i,
  Cov(eps_i H_{i-1}(X_i), delta_{j,k-1}^2)
      = [ 2 sigma^2 V_j 1{j in band(i-1)} + 2 sigma^2 sum_{l in band(i-1)} theta_l M_lj ] / (k-1)^2,
  M_lj = E[phi_l phi_j (f0 phi_j - theta_j)],
and zero for k - 1 < i. The first part is the main one: the noise eps_i that the overfitting
model's extra terms happen to fit (making Delta C large, which lowers its score) later enters
those same coefficients' estimates (making Delta B large, which raises its score).

Usage: python3 check_BC_covariance.py
"""
import json
from pathlib import Path

import numpy as np
import wrv_sim as w
from check_B_variance import band_intervals, var_delta_B


def var_delta_C(n, xi, a_g, a_h):
    i = np.arange(w.BURN + 1, n + 1)
    m = i - 1
    jg, jh = w.sizes(m, a_g), w.sizes(m, a_h)
    band = (w.CUM_V[jh] - w.CUM_V[jg]) / m + w.TAIL[jg] - w.TAIL[jh]
    return float(4 * w.SIGMA ** 2 * np.sum(i.astype(float) ** (2 * xi) * band))


def cov_delta_BC(n, xi, a_g, a_h):
    m_lo, m_hi = w.BURN, n - 1
    L, U, jmax = band_intervals(a_g, a_h, m_lo, m_hi)
    phi = w.PHI_GRID[:, :jmax]
    th = w.THETA[:jmax]
    V = w.V[:jmax]
    M = w.matmul(phi.T, phi * (w.F0_GRID[:, None] * phi - th)) / phi.shape[0]   # M[l, j]

    mm = np.arange(0, m_hi + 2, dtype=float)
    wq = (mm + 1) ** xi
    g = np.where(mm >= m_lo, wq, 0.0)                     # weight of the C time, p = i - 1
    with np.errstate(divide="ignore"):
        h = np.where(mm >= m_lo, wq / np.maximum(mm, 1) ** 2, 0.0)   # weight/q^2 of the B time, q = k - 1
    pre = lambda v: np.concatenate([[0.0], np.cumsum(v)])
    Gp = pre(g)
    Hq = pre(h)
    HG = pre(h * np.concatenate([[0.0], Gp[1:-1]]))       # sum_q h(q) * Gp-sum over p <= q-1

    def rng(P, lo, hi):
        lo = np.clip(lo, 0, len(P) - 1)
        hi = np.clip(hi, -1, len(P) - 2)
        return np.where(hi >= lo, P[np.maximum(hi, -1) + 1] - P[lo], 0.0)

    def pairsum(Lp, Up, Lq, Uq):
        """sum over p in [Lp,Up], q in [Lq,Uq], q >= p + 1 of g(p) h(q)."""
        # q in [max(Lq, Lp+1), min(Uq, Up+1)]: inner = Gp(q-1) - Gp(Lp-1)
        lo1, hi1 = np.maximum(Lq, Lp + 1), np.minimum(Uq, Up + 1)
        base = Gp[np.clip(Lp, 0, len(Gp) - 1)]
        part1 = rng(HG, lo1, hi1) - base * rng(Hq, lo1, hi1)
        lo2 = np.maximum(Lq, Up + 2)
        part2 = rng(Gp, Lp, Up) * rng(Hq, lo2, Uq)
        return part1 + part2

    diag = pairsum(L, U, L, U)
    main = 4 * w.SIGMA ** 2 * float(np.sum(V * diag))
    cross = pairsum(L[:, None], U[:, None], L[None, :], U[None, :])   # p uses l (rows), q uses j (cols)
    extra = 4 * w.SIGMA ** 2 * float(np.sum(th[:, None] * M * cross))
    return main + extra, main


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
    rec = np.asarray(data["rec"])
    names = data["candidate_names"]
    ns = data["checkpoints"]
    kB, kC = st["terms"].index("B"), st["terms"].index("C")
    out = {}
    print("comparison  xi       n   exact corr(dB,-dC)  simulated   | exact Cov / simulated Cov | main-term share")
    for other in ("over40", "over50"):
        a_g, a_h = st["candidates"]["opt"], st["candidates"][other]
        for xk, xi in enumerate(st["xis"]):
            dB = rec[:, :, names.index(other), xk, kB] - rec[:, :, names.index("opt"), xk, kB]
            dC = rec[:, :, names.index(other), xk, kC] - rec[:, :, names.index("opt"), xk, kC]
            rows = []
            for k, n in enumerate(ns):
                cov, main_part = cov_delta_BC(n, xi, a_g, a_h)
                vb, vc = var_delta_B(n, xi, a_g, a_h), var_delta_C(n, xi, a_g, a_h)
                corr = -cov / np.sqrt(vb * vc)
                sim_cov = np.cov(dB[:, k], dC[:, k])[0, 1]
                sim_corr = np.corrcoef(dB[:, k], -dC[:, k])[0, 1]
                rows.append((n, corr, sim_corr, cov / sim_cov, main_part / cov, vb, vc, cov))
                if k in (0, 5, len(ns) - 1):
                    print(f"opt-{other:6s} {xi:3g} {n:7d}   {corr:+.3f}            {sim_corr:+.3f}      "
                          f"{cov/sim_cov:6.3f}                    {main_part/cov:.3f}")
            out[f"{other}_xi{int(xi)}"] = [dict(zip(
                ["n", "exact_corr_B_minusC", "sim_corr", "cov_exact_over_sim", "main_share", "varB", "varC", "covBC"],
                map(float, r))) for r in rows]
    Path("BC_covariance_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
