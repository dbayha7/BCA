"""Theorem 4's concentration slack eta_n for the benchmark's tilts (Lou and Luo, Appendix B).

With oracle weights under pure covariate shift, Theorem 4 bounds the deployed risk by
alpha + B_l eta_n(delta, m) with probability 1 - delta. Lemma 9 (Hoeffding):
eps = B sqrt(log((4m + 6) / delta) / (2n)); Remark 3 (Bernstein, per-evaluation):
eps = sqrt(2 log((4m + 6) / delta) / n_eff_pop) + 2 B log((4m + 6) / delta) / (3n), with
n_eff_pop = n / E_cal[w^2]; then eta = (2 eps + 1/m) / (1 - eps), m = ceil(sqrt(n)).
B = ess sup w / E_cal[w], here the maximum over the pool. Miscoverage loss: B_l = 1.

python experiments/wbcp/certificate_slack.py --frozen F --output out.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.wbcp import d4rl_benchmark as bench  # noqa: E402

CASES = [("policy", 0.0), ("policy", 1.0), ("density", 0.5), ("density", 1.0), ("state", 0.5), ("state", 1.0)]


def slack(eps, m):
    return (2 * eps + 1 / m) / (1 - eps) if eps < 1 else math.inf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[1103, 8824])
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        parser.error("--output already exists")
    report = []
    for name, gamma in CASES:
        setup = bench.build_setup(bench.parse_args(["--frozen", args.frozen, "--tilt", name, "--gamma", str(gamma)]))
        a = setup.tilts[0].mass
        w = a / a.mean()  # E_cal[w] = 1 over the pool
        big_b, second = float(w.max()), float(np.mean(w ** 2))
        for n in args.n:
            m = math.ceil(math.sqrt(n))
            log_term = math.log((4 * m + 6) / args.delta)
            hoeffding = big_b * math.sqrt(log_term / (2 * n))
            bernstein = math.sqrt(2 * log_term * second / n) + 2 * big_b * log_term / (3 * n)
            row = dict(tilt=name, gamma=gamma, n=n, B=big_b, E_w2=second, n_eff_pop=n / second,
                       eta_hoeffding=slack(hoeffding, m), eta_bernstein=slack(bernstein, m))
            report.append(row)
            print(f"{name:8s} gamma={gamma:<4} n={n:<5d} B={big_b:8.2f} E[w^2]={second:6.2f}  "
                  f"eta Hoeffding={row['eta_hoeffding']:.3f}  eta Bernstein={row['eta_bernstein']:.3f}")
    with open(args.output, "x") as handle:
        json.dump(dict(settings=vars(args), rows=report), handle, indent=1, allow_nan=True)


if __name__ == "__main__":
    main()
