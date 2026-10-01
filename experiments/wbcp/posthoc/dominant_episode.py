# Exact script (unchanged below this block) that wrote runs/wbcp_dependence/hopper-u100000/posthoc_dominant_episode.json.
# Run from the BCA root. Random seed 11. Written after the Change 5 results (post hoc).
# Note: tilt_feature is called with its default knn_seed=0, while the benchmark runs used
# --knn-seed 20260930, so the density feature differs slightly from the benchmark's.
"""Post-hoc check: are density-tilt failures at K=10 driven by banks dominated by one episode?

Per bank (oracle weights, density tilt, normalized score): WBCP threshold, failure, and the
largest share of the bank's total weight held by a single episode. Compares failure rates
across quartiles of that share, for iid, K=5 and K=10 banks.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from calibration.wbcp import calibrate  # noqa: E402
from experiments.wbcp import d4rl_benchmark as bench  # noqa: E402

pool = bench.load_pool("runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000", "heldout")
groups = bench.episode_groups(pool.episode, pool.timestep)
z = bench.standardize(bench.tilt_feature("density", pool.obs, pool.action, pool.policy_action))
scores = pool.scores["normalized"]
out = {}
for gamma in (0.5, 1.0):
    a = np.exp(gamma * (z - z.max()))
    w = a / a.mean()
    wbar = float(np.mean(a * a) / np.mean(a) ** 2)
    lam = bench.ExactRisk(scores, a).lambda_star(0.1)
    for design in ("iid", 5, 10):
        rng = np.random.default_rng(11)
        fails, shares = [], []
        for b in range(2000):
            rows = (bench.draw_calibration(rng, 1103, pool.size) if design == "iid"
                    else bench.draw_calibration(rng, 1103, pool.size, groups, per_episode=design))
            threshold = calibrate(scores[rows], rng, w[rows], wbar, alpha=0.1, beta=0.95, draws=1000).threshold
            fails.append(threshold < lam)
            per_episode = np.bincount(np.unique(pool.episode[rows], return_inverse=True)[1], weights=w[rows])
            shares.append(per_episode.max() / per_episode.sum())
        fails, shares = np.array(fails), np.array(shares)
        edges = np.quantile(shares, [0.25, 0.5, 0.75])
        quart = np.digitize(shares, edges)
        by_q = [float(fails[quart == q].mean()) for q in range(4)]
        out[f"gamma={gamma} {design}"] = dict(fail=float(fails.mean()), median_top_share=float(np.median(shares)),
                                              p90_top_share=float(np.quantile(shares, 0.9)), fail_by_share_quartile=by_q)
        print(f"density g={gamma} {str(design):>3}: fail {100 * fails.mean():5.1f}%  top-episode weight share "
              f"median {np.median(shares):.3f} p90 {np.quantile(shares, 0.9):.3f}  | fail by share quartile "
              + " ".join(f"{100 * v:5.1f}%" for v in by_q), flush=True)
json.dump(out, open("runs/wbcp_dependence/hopper-u100000/posthoc_dominant_episode.json", "x"), indent=1)
