# When Can Validation Catch Mild Overfitting?

Course project for Regression Analysis (16:960:563), Rutgers, Fall 2026. Open Problem 07.

This project studies weighted rolling validation (wRV) from Zhang & Lei (2025), *Online estimation with rolling validation*, Annals of Statistics 53(6), arXiv:2310.12140. The question is whether wRV picks the rate-optimal model over a mildly overfitting one. Their proof stops just short of covering that case.

The blog post is `index.html`, served with GitHub Pages.

## Files

- `wrv_sim.py` is the simulation. It contains the projection estimators, the wRV score, the exact three-term decomposition (paper eq. 15) and the built-in checks.
- `plot_results.py` takes `results.json` and produces the figures in `figures/` plus `summary.json`.
- `exact_signal_rate.py` gives local slopes of the expected score gap in the finite capped model. The uncapped prediction of 1.4 is not its asymptotic limit.
- `robustness/run_all.sh` and `robustness/summarize.py` rerun the study with a 1,200-term cutoff, a log-adjusted truth, and smoothness s = 2. They write `robustness/summary.md` and `figures/robustness.png`. Settings are passed to `wrv_sim.py` through `WRV_*` environment variables; the defaults reproduce the main run.
- `check_paired_variance.py` compares the exact variance of the paired C-term difference with the saved simulation.
- `results.json` and `summary.json` hold the raw and summarized output of the main run (400 trajectories, n up to 100,000).

## Reproduce

```
pip install numpy matplotlib
python3 wrv_sim.py --reps 400 --nmax 100000   # about 7 min on 2 cores
python3 plot_results.py results.json
python3 exact_signal_rate.py
python3 check_paired_variance.py
./robustness/run_all.sh      # about 25 min
python3 robustness/summarize.py
```

## Checks (main run)

| Check | Result |
|---|---|
| Basis orthonormality on the quadrature grid | max error 7e-15 |
| From-scratch refit at random times (no leakage) | max difference 3e-15 |
| noise + signal + A + B − C equals the direct wRV score | max relative error 5e-15 |
| Monte Carlo mean of the conditional risk vs. exact expected risk | ratio 0.97 to 1.02 |

## Interpretation and remaining work

These simulations support improved pairwise selection over the tested sample sizes, not a proof of asymptotic consistency. Every candidate is capped at 400 and eventually the sequences coincide. The observed cancellation is modest over this range; its asymptotic order remains open.

Before submission:

- Complete the personal AI-use reflections in the blog using your actual experience.
- Done (Nov 19): larger cutoff, log-adjusted truth, and s = 2. Over the tested range, cancellation was modest and roughly constant (variance ratio 0.53 to 0.81), and the noise grew more slowly than the signal in all settings. For s = 2, the expected cumulative wRV score favors the overfitting sequence until n ≈ 9,800 (ξ = 1), and its current risk is lower until about 32,800, which explains wRV's early choices there.
- Derive or bound the paired B-term variance; this remains the central theoretical gap.

The independent paired-C check gives simulated/exact standard-deviation ratios of approximately 0.96–1.08 at the first and last checkpoints across the three weights. These are Monte Carlo comparisons, not a formal acceptance threshold.

## Exact paired B variance

Run `python3 exact_b_variance.py` from the repository root with default settings. The blog derives the covariance of squared coefficient errors at two training times. The script sums it using fourth moments integrated over the finite trigonometric model and both noise signs. At n=100,000, simulated/exact standard-deviation ratios are 0.979, 0.954, and 0.954 for weights 0, 1, and 2. This finite-sample formula does not establish an asymptotic rate.

## Exact B-C covariance (where the cancellation comes from)

`check_BC_covariance.py` computes Cov(ΔB, ΔC) exactly for the finite model. `check_B_variance.py` is a second, independently written derivation of Var(ΔB); it agrees with `exact_b_variance.py` to every printed digit. Both accept a results file:

```
python3 check_B_variance.py robustness/validation_n3000.json.gz
python3 check_BC_covariance.py robustness/validation_n3000.json.gz
```

`robustness/validation_n3000.json.gz` holds 10,000 trajectories up to n = 3,000. Against it, the exact sd(ΔB) is within 2.2% and the exact corr(ΔB, −ΔC) is within 0.03 at every checkpoint.

## Model-size rounding

`sizes()` in `wrv_sim.py` returns the smallest integer ≥ m^a. Values within 1e-9 of an integer count as that integer, so exact powers such as 32768^0.2 = 8 are not rounded up to 9. All results were regenerated after this fix with `robustness/rerun_everything.sh` (about 40 minutes). Reported numbers changed by at most 0.005.
