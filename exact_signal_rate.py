"""
Exact growth rate of the expected score gap (the "signal") between over40 and opt.
Uses the exact risk formula from wrv_sim.py, so no simulation is needed.
Prints the local log-log slope of sum_i i^xi (m_over40,i - m_opt,i) as n grows.
The uncapped power-law model predicts xi + 0.4. This fixed-cap experiment does not
have that asymptotic limit: eventually both candidate sequences coincide.

Usage: python3 exact_signal_rate.py
"""
import numpy as np
import wrv_sim as w

XI = 1.0
ns = np.unique(np.logspace(3, 6.4, 60).astype(int))
i = np.arange(w.BURN + 1, ns[-1] + 1)
gap = (w.exact_risk(i - 1, 0.4) - w.exact_risk(i - 1, 1 / 3)) * i ** XI
S = np.cumsum(gap)[ns - (w.BURN + 1)]
slopes = np.diff(np.log(S)) / np.diff(np.log(ns))
print("n, local slope of the signal gap (xi = 1, finite capped model; uncapped prediction 1.4)")
for n, s in zip(ns[1::6], slopes[::6]):
    print(f"{n:>9d}  {s:.3f}")
