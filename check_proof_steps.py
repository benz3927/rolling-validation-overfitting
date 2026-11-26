"""Numeric tests of each inequality in theory/consistency_proof.pdf, in the finite simulation model.
These are spot checks in one model (s=1, trig basis, noise +/-1). They cannot prove the theorem."""
import numpy as np, wrv_sim as w
from check_B_variance import moments, var_delta_B

K = np.sqrt(2.0); M = np.abs(w.F0_GRID).max() + w.SIGMA; sig2 = w.SIGMA ** 2
Sg, Q = moments(w.J_CAP); V = np.diag(Sg); TH2 = w.THETA ** 2
ok = lambda b: "PASS" if b else "FAIL"
print("model: s=%g, J_CAP=%d, K=%.3f, M=%.3f" % (w.S, w.J_CAP, K, M))

print("\nF1  sigma^2 <= V_j <= M^2:  min V=%.4f  max V=%.4f  M^2=%.3f  %s" % (V.min(), V.max(), M*M, ok(V.min() >= sig2 - 1e-9 and V.max() <= M*M)))
r = (Sg ** 2).sum(1).max()
print("F2  max_j sum_l Sigma_jl^2 = %.3f <= 4M^4 = %.3f  %s" % (r, 4*M**4, ok(r <= 4*M**4)))
rng = np.random.default_rng(0); worst = 0
for _ in range(2000):
    J = rng.integers(1, 200); a = rng.integers(0, w.J_CAP - J); c = rng.standard_normal(J)
    u = w.PHI_GRID[:, a:a+J] @ c; worst = max(worst, (u**4).mean() / (K*K*J*(c@c)**2))
print("F3  E u^4 / (K^2 |I| (sum c^2)^2), worst of 2000 random u: %.3f <= 1  %s" % (worst, ok(worst <= 1)))
Js = np.arange(1, 300); b = np.array([np.abs(w.PHI_GRID[:, J:] @ w.THETA[J:]).max() for J in Js])
print("F3  sup|b_J| / J^(1/2-s): max %.3f (bounded)   ||b_J||^2 / J^(-2s): max %.3f" % ((b / Js**(0.5-w.S)).max(), (w.TAIL[Js] / Js**(-2*w.S)).max()))

def bands(beta, alpha, mmax):
    ms = np.arange(w.BURN, mmax + 1); Jg = w.sizes(ms, beta).astype(int); Jh = w.sizes(ms, alpha).astype(int)
    return ms, Jg, Jh

for beta, alpha, mmax in ((1/3, 0.4, 10**6), (1/3, 0.7, 5000)):
    ms, Jg, Jh = bands(beta, alpha, mmax); assert Jh.max() <= w.J_CAP
    ED = np.empty(len(ms)); VD = np.empty(len(ms)); bound = np.empty(len(ms)); cache = {}
    cV = np.concatenate([[0], np.cumsum(V)]); cT = np.concatenate([[0], np.cumsum(TH2)])
    for k, (m, a, c) in enumerate(zip(ms, Jg, Jh)):
        if (a, c) not in cache:
            cache[(a, c)] = ((Q[a:c, a:c] - np.outer(V[a:c], V[a:c])).sum(), (Sg[a:c, a:c] ** 2).sum(), np.abs(Q[a:c, a:c] - np.outer(V[a:c], V[a:c])).max())
        q, s2, qmax = cache[(a, c)]; n = c - a
        ED[k] = (cV[c] - cV[a]) / m - (cT[c] - cT[a])
        VD[k] = q / m**3 + 2*(m-1)*s2 / m**3
        bound[k] = (2*K*M)**4 * n*n / m**3 + 8*M**4 * n / m**2
    print("\n--- beta=%.3f alpha=%.1f, m = %d..%d ---" % (beta, alpha, ms[0], ms[-1]))
    r1 = ED / (sig2/4 * ms**(alpha-1.0))
    bad = ms[r1 < 1]; mstar = bad.max() + 1 if len(bad) else ms[0]
    print("Step 1  E D_m >= sigma^2/4 m^(alpha-1) for every m >= m_* = %d (the proof claims it only for large m); min ratio after m_*: %.2f  %s" % (mstar, r1[ms >= mstar].min(), ok(mstar < ms[-1] // 10)))
    print("Step 2  Var(D_m) <= explicit bound for every m: max ratio %.4f  %s" % ((VD/bound).max(), ok((VD <= bound).all())))
    r2 = VD / ms**(alpha-2.0); print("Step 2  Var(D_m) / m^(alpha-2): max %.3f, at m_max %.3f (bounded, not growing)" % (r2.max(), r2[-1]))
    r3 = np.sqrt(VD) / ED; i0 = [0, len(ms)//100, len(ms)-1]
    print("        sd(D_m)/E D_m at m=%s: %s  (theory: falls like m^(-alpha/2))" % (ms[i0].tolist(), np.round(r3[i0], 3).tolist()))
    if alpha == 0.4:
        for xi in (0.0, 1.0, 2.0):
            for n in (10**4, 10**5):
                sel = ms < n; tri = ((ms[sel] + 1.0)**xi * np.sqrt(VD[sel])).sum(); sdB = np.sqrt(var_delta_B(n, xi, beta, alpha))
                print("Step 2  xi=%g n=%d: exact sd(B_n)=%.4g <= triangle bound %.4g (ratio %.3f)  %s" % (xi, n, sdB, tri, sdB/tri, ok(sdB <= tri)))

print("\n--- Step 3: E Delta_i^2 by simulation (beta=1/3, alpha=0.4) ---")
rng = np.random.default_rng(3); beta, alpha = 1/3, 0.4
for m, reps in ((500, 600), (5000, 300), (20000, 120)):
    Jg = int(w.sizes(m, beta)); Jh = int(w.sizes(m, alpha)); acc = np.zeros(4)
    for _ in range(reps):
        X = rng.random(m); B = w.basis(X); Y = B @ w.THETA + rng.choice([-1., 1.], m)
        th = (Y[:, None] * B[:, :Jh]).mean(0)
        Xt = rng.random(4000); Bt = w.basis(Xt); f0 = Bt @ w.THETA; eps = rng.choice([-1., 1.], 4000)
        fh = Bt[:, :Jh] @ th; fg = Bt[:, :Jg] @ th[:Jg]; H = fh - fg; E2 = (fh - f0) + (fg - f0)
        D = H * (E2 - 2*eps); acc += [(D**2).mean(), (H**4).mean(), (E2**4).mean(), (H**2).mean()]
    d2, h4, e4, h2 = acc / reps; rhs = 2*np.sqrt(h4*e4) + 32*M*M*h2
    print("m=%6d  E Delta^2=%.4g <= 2 sqrt(EH^4 E(e_h+e_g)^4) + 32 M^2 E||H||^2 = %.4g  %s ;  E Delta^2 / m^(alpha-1) = %.2f" % (m, d2, rhs, ok(d2 <= rhs), d2 / m**(alpha-1)))
