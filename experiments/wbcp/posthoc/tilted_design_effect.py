# Exact script (unchanged below this block) that wrote runs/wbcp_dependence/hopper-u100000/posthoc_tilted_design_effect.json.
# Run from the BCA root. Random seed 7. Written after the Change 5 results (post hoc).
# Note: tilt_feature is called with its default knn_seed=0, while the benchmark runs used
# --knn-seed 20260930, so the density feature differs slightly from the benchmark's.
"""Post-hoc check: does dependence interact with the tilt? (scratch; not pre-registered)

For each tilt, compute (a) the intra-episode correlation of the tilt feature z, and (b) the
Monte Carlo design effect of the quantity WBCP relies on under shift, the self-normalized
weighted miss rate sum w I / sum w with oracle weights w and I = 1{score > lambda*_tilt},
under iid rows and thinned banks K = 5, 10. Predicted failure = 1 - Phi(z0 / sqrt(D)), with
z0 calibrated to the observed iid oracle failure for that tilt (the iid baseline is not 5%
under strong tilts).
"""
import json
import math
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from experiments.wbcp import d4rl_benchmark as bench  # noqa: E402
from experiments.wbcp.dependence import intraclass_correlation  # noqa: E402

FROZEN = "runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000"
O = "runs/wbcp_dependence/hopper-u100000"
pool = bench.load_pool(FROZEN, "heldout")
groups = bench.episode_groups(pool.episode, pool.timestep)
main = {(b["tilt"], b["gamma"], b["n"], b["score"]): b for b in json.load(open("runs/wbcp_bench/main.json"))["blocks"]}
e3 = {k: {(b["tilt"], b["gamma"], b["score"]): b for b in json.load(open(f"{O}/e3_k{k}.json"))["blocks"]} for k in (5, 10)}
features = {name: bench.standardize(bench.tilt_feature(name, pool.obs, pool.action, pool.policy_action))
            for name in ("policy", "density", "state")}
rng = np.random.default_rng(7)
banks, n = 6000, 1103
out = {}
for name, z in features.items():
    rho_z = intraclass_correlation(z, pool.episode)
    for gamma in (0.5, 1.0):
        a = np.exp(gamma * (z - z.max()))
        w = a / a.mean()
        for score in ("normalized",):
            s = pool.scores[score]
            lam = bench.ExactRisk(s, a).lambda_star(0.1)
            miss = (s > lam).astype(float)
            rates = {}
            for design in ("iid", 5, 10):
                values = np.empty(banks)
                for b in range(banks):
                    rows = (bench.draw_calibration(rng, n, pool.size) if design == "iid"
                            else bench.draw_calibration(rng, n, pool.size, groups, per_episode=design))
                    values[b] = np.dot(w[rows], miss[rows]) / w[rows].sum()
                rates[design] = values.var(ddof=1)
            iid_fail = main[(name, gamma, n, score)]["arms"]["WBCP (oracle w)"]["fail"]
            z0 = stats.norm.isf(iid_fail)
            row = dict(rho_feature=rho_z, rho_miss=intraclass_correlation(miss, pool.episode),
                       rho_weighted_miss=intraclass_correlation(w * miss, pool.episode), iid_oracle_fail=iid_fail)
            for k in (5, 10):
                d = rates[k] / rates["iid"]
                row[f"D_K{k}"] = d
                row[f"predicted_K{k}"] = float(stats.norm.sf(z0 / math.sqrt(d)))
                row[f"observed_K{k}"] = e3[k][(name, gamma, score)]["arms"]["WBCP (oracle w)"]["fail"]
            out[f"{name} gamma={gamma}"] = row
            print(f"{name:8} g={gamma}: rho(z)={rho_z:.3f} rho(miss)={row['rho_miss']:.4f} rho(w*miss)={row['rho_weighted_miss']:.4f} | "
                  f"iid {100 * iid_fail:.1f}% | K5: D={row['D_K5']:.2f} pred {100 * row['predicted_K5']:.1f}% obs {100 * row['observed_K5']:.1f}% | "
                  f"K10: D={row['D_K10']:.2f} pred {100 * row['predicted_K10']:.1f}% obs {100 * row['observed_K10']:.1f}%", flush=True)
json.dump(out, open(f"{O}/posthoc_tilted_design_effect.json", "x"), indent=1)
