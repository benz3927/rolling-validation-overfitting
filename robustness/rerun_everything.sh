#!/bin/bash
# Full rerun after the model-size rounding fix (about 40 minutes on 2 cores).
set -e
python3 wrv_sim.py --reps 400 --nmax 100000 --out results.json
python3 plot_results.py results.json
./robustness/run_all.sh
python3 wrv_sim.py --reps 10000 --nmax 3000 --out robustness/validation_n3000.json
python3 robustness/reduce_validation.py
python3 robustness/summarize.py
