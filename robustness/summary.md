# Robustness summary

Each row compares the rate-optimal sequence with one alternative, using weight xi = 1 unless noted.
P(correct) is the share of trajectories in which wRV picks the optimal sequence (95% Wilson interval).
Slopes are log-log slopes over the last five checkpoints (n from about 16,000 to 100,000).

## main: s=1, cap 400
truth = power, s = 1.0, J_CAP = 400, reps = 400, opt exponent = 0.333
checks: orthonormality 7.1e-15, leakage 3.1e-15, decomposition 4.4e-15, centering 0.968 to 1.020

| vs | a | P(correct) n=1e3 | P(correct) n=1e5 | P at n=1e5, xi=0 / 2 | Var ratio | corr(dB,-dC) | slope signal | slope sd(sum) | sd dB / dC | SNR 1e3 -> 1e5 |
|---|---|---|---|---|---|---|---|---|---|---|
| over40 | 0.40 | 0.600 (0.55-0.65) | 0.953 (0.93-0.97) | 0.980 / 0.915 | 0.65-0.79 | -0.29 | 1.60 | 1.28 | 1.24 / 1.27 | 0.25 -> 1.58 |
| over50 | 0.50 | 0.973 (0.95-0.98) | 1.000 (0.99-1.00) | 1.000 / 1.000 | 0.61-0.66 | -0.37 | 1.58 | 1.26 | 1.24 / 1.25 | 1.75 -> 8.82 |
| under20 | 0.20 | 1.000 (0.99-1.00) | 1.000 (0.99-1.00) | 1.000 / 1.000 | 0.90-0.98 | -0.11 | 1.68 | 1.29 | 1.18 / 1.29 | 3.20 -> 17.69 |

## cap 1200
truth = power, s = 1.0, J_CAP = 1200, reps = 200, opt exponent = 0.333
checks: orthonormality 4.1e-14, leakage 1.8e-15, decomposition 4.4e-15, centering 0.955 to 1.019

| vs | a | P(correct) n=1e3 | P(correct) n=1e5 | P at n=1e5, xi=0 / 2 | Var ratio | corr(dB,-dC) | slope signal | slope sd(sum) | sd dB / dC | SNR 1e3 -> 1e5 |
|---|---|---|---|---|---|---|---|---|---|---|
| over40 | 0.40 | 0.650 (0.58-0.71) | 0.955 (0.92-0.98) | 0.975 / 0.915 | 0.65-0.79 | -0.29 | 1.60 | 1.31 | 1.27 / 1.30 | 0.27 -> 1.56 |
| over50 | 0.50 | 0.965 (0.93-0.98) | 1.000 (0.98-1.00) | 1.000 / 1.000 | 0.58-0.76 | -0.35 | 1.58 | 1.26 | 1.25 / 1.27 | 1.80 -> 8.68 |
| under20 | 0.20 | 1.000 (0.98-1.00) | 1.000 (0.98-1.00) | 1.000 / 1.000 | 0.87-1.00 | -0.12 | 1.68 | 1.30 | 1.16 / 1.30 | 3.06 -> 17.45 |

## cap 1200, log-adjusted truth
truth = powerlog, s = 1.0, J_CAP = 1200, reps = 200, opt exponent = 0.333
checks: orthonormality 4.1e-14, leakage 3.1e-15, decomposition 4.9e-15, centering 0.946 to 1.018

| vs | a | P(correct) n=1e3 | P(correct) n=1e5 | P at n=1e5, xi=0 / 2 | Var ratio | corr(dB,-dC) | slope signal | slope sd(sum) | sd dB / dC | SNR 1e3 -> 1e5 |
|---|---|---|---|---|---|---|---|---|---|---|
| over40 | 0.40 | 0.945 (0.90-0.97) | 1.000 (0.98-1.00) | 1.000 / 1.000 | 0.57-0.69 | -0.38 | 1.48 | 1.30 | 1.29 / 1.27 | 1.51 -> 4.70 |
| over50 | 0.50 | 1.000 (0.98-1.00) | 1.000 (0.98-1.00) | 1.000 / 1.000 | 0.53-0.71 | -0.41 | 1.54 | 1.26 | 1.26 / 1.27 | 3.04 -> 11.58 |
| under20 | 0.20 | 0.815 (0.76-0.86) | 1.000 (0.98-1.00) | 1.000 / 1.000 | 0.74-0.87 | -0.24 | 1.63 | 1.22 | 1.11 / 1.21 | 0.79 -> 3.13 |

## s=2, cap 400
truth = power, s = 2.0, J_CAP = 400, reps = 200, opt exponent = 0.200
checks: orthonormality 7.1e-15, leakage 2.7e-15, decomposition 4.6e-15, centering 0.918 to 1.017

| vs | a | P(correct) n=1e3 | P(correct) n=1e5 | P at n=1e5, xi=0 / 2 | Var ratio | corr(dB,-dC) | slope signal | slope sd(sum) | sd dB / dC | SNR 1e3 -> 1e5 |
|---|---|---|---|---|---|---|---|---|---|---|
| over30 | 0.30 | 0.265 (0.21-0.33) | 0.740 (0.68-0.80) | 0.585 / 0.685 | 0.71-0.81 | -0.29 | 1.91 | 1.11 | 1.07 / 1.13 | -0.53 -> 0.63 |
| over40 | 0.40 | 0.735 (0.67-0.79) | 1.000 (0.98-1.00) | 1.000 / 1.000 | 0.58-0.73 | -0.40 | 1.49 | 1.20 | 1.22 / 1.16 | 0.70 -> 5.19 |
| under10 | 0.10 | 1.000 (0.98-1.00) | 1.000 (0.98-1.00) | 1.000 / 1.000 | 0.96-1.02 | -0.05 | 1.85 | 1.41 | 1.17 / 1.41 | 8.48 -> 20.19 |
