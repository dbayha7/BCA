"""EXPLORATORY diagnosis: BCA capped pull minus plain TD3+BC along training (page lc-data, cotrain curves, 5 seeds).
Reads lc_data.json (extracted verbatim from the page's lc-data script). No training."""
import json, numpy as np
d = json.load(open('lc_data.json'))
co = d['cotrain']; steps = co['steps']
print('steps', steps)
for q in d['qualities']:
    for law in ('none', 'strong'):
        h = np.array(co['arms']['host_gap'][q]['pts'][law], float)
        for arm in ('capped_unweighted', 'capped_exact_strong'):
            b = np.array(co['arms'][arm][q]['pts'][law], float)
            diff = b - h
            m = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(diff.shape[0])
            pick = [0, 1, 2, 3, 5, 9, 14, 19]
            print(f"{q:14s} {law:6s} {arm:20s} " + " ".join(f"{steps[i]//1000}k:{m[i]:+5.1f}" for i in pick),
                  f"| final per seed {np.round(diff[:, -1], 1).tolist()}  last-5-avg {diff[:, -5:].mean():+.1f}")
    print()
