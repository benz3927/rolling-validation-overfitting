"""Compare exact Var(C_over40 - C_opt) with saved Monte Carlo output.

Nested estimators share coefficient averages. Their difference contains only
coefficients between their model sizes; E[theta_hat_j^2] = V_j/m + theta_j^2.
Independent, mean-zero noise makes the weighted increments orthogonal in time.
"""
import json
from pathlib import Path
import numpy as np
import wrv_sim as w


def main():
    data = json.loads(Path(__file__).with_name('results.json').read_text())
    settings = data['settings']
    for key, expected in [('J_CAP', w.J_CAP), ('s', w.S), ('sigma', w.SIGMA),
                          ('C_TRUTH', w.C_TRUTH), ('burn_in', w.BURN)]:
        if settings[key] != expected:
            raise ValueError(f'Saved settings differ from current code: {key}')
    ns = np.array(data['checkpoints'])
    i = np.arange(w.BURN + 1, ns[-1] + 1)
    m = i - 1
    jg = w.sizes(m, settings['candidates']['opt'])
    jh = w.sizes(m, settings['candidates']['over40'])
    band = (w.CUM_V[jh] - w.CUM_V[jg]) / m + w.TAIL[jg] - w.TAIL[jh]
    rec = np.asarray(data['rec'])
    names = data['candidate_names']
    c = settings['terms'].index('C')
    print('xi  n        exact_sd     simulated_sd  simulated/exact')
    for xk, xi in enumerate(settings['xis']):
        exact = np.sqrt((4 * w.SIGMA**2 * np.cumsum(i.astype(float)**(2*xi) * band))[ns - w.BURN - 1])
        delta = rec[:, :, names.index('over40'), xk, c] - rec[:, :, names.index('opt'), xk, c]
        empirical = delta.std(axis=0, ddof=1)
        for k in (0, len(ns)-1):
            print(f'{xi:g} {ns[k]:7d} {exact[k]:12.5g} {empirical[k]:12.5g} {empirical[k]/exact[k]:.3f}')


if __name__ == '__main__':
    main()
