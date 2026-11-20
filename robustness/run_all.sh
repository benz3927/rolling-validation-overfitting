#!/bin/bash
# Robustness runs. Each writes robustness/<name>.json. Run from the repo root.
set -e
WRV_JCAP=1200 WRV_TRUTH=power python3 wrv_sim.py --reps 200 --nmax 100000 --out robustness/cap1200_power.json
WRV_JCAP=1200 WRV_TRUTH=powerlog python3 wrv_sim.py --reps 200 --nmax 100000 --out robustness/cap1200_powerlog.json
WRV_S=2 WRV_JCAP=400 WRV_CANDS='{"over30":0.30,"over40":0.40,"under10":0.10}' python3 wrv_sim.py --reps 200 --nmax 100000 --out robustness/s2_power.json
