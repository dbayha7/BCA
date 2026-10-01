# Post hoc (written after the Change 10 shift results). Run from the BCA root.
"""Intra-episode correlation of the weighted miss term under each tilt, for one frozen pool.

Under a tilt the variance of WBCP's self-normalized miss rate at lambda* depends on
u = w (I - R), w the oracle ratio, I = 1{score > lambda*_tilt}, R = alpha. Its intra-class
correlation rho_w replaces rho(I) in the design effect 1 + (K - 1) rho_w of a K-row bank.

python experiments/wbcp/posthoc/tilted_rho.py --frozen F --k 23 --output out.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from experiments.wbcp import d4rl_benchmark as bench  # noqa: E402
from experiments.wbcp.dependence import intraclass_correlation  # noqa: E402

CASES = [("policy", 0.0), ("policy", 0.5), ("policy", 1.0), ("density", 0.5), ("density", 1.0), ("state", 0.5)]

parser = argparse.ArgumentParser()
parser.add_argument("--frozen", required=True)
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rows = []
for name, gamma in CASES:
    setup = bench.build_setup(bench.parse_args(["--frozen", args.frozen, "--tilt", name, "--gamma", str(gamma),
                                                "--score", "normalized"]))
    tilt, pool = setup.tilts[0], setup.pool
    lam = setup.risks[(0, "normalized")].lambda_star(0.1)
    w = tilt.mass / tilt.mass.mean()
    miss = (pool.scores["normalized"] > lam).astype(float)
    rho_i = intraclass_correlation(miss, pool.episode)
    rho_w = intraclass_correlation(w * (miss - 0.1), pool.episode)
    design = 1 + (args.k - 1) * rho_w
    rows.append(dict(tilt=name, gamma=gamma, lambda_star=float(lam), rho_miss=rho_i, rho_weighted=rho_w,
                     design_effect=design, predicted_failure=float(stats.norm.sf(1.6449 / math.sqrt(max(design, 1e-9))))))
    print(f"{name:8s} gamma={gamma:<4} lambda*={lam:.3f} rho(I)={rho_i:.4f} rho(w(I-R))={rho_w:.4f} "
          f"D(K={args.k})={design:.2f} -> {100 * rows[-1]['predicted_failure']:.1f}%")
with open(args.output, "x") as handle:
    json.dump(dict(frozen=args.frozen, k=args.k, rows=rows), handle, indent=1)
