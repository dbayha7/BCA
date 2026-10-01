"""Per-bank view of oracle-weight WBCP at the true threshold: why failures exceed 1 - beta.

For each iid bank of the D4RL benchmark (same random streams as d4rl_benchmark.py, so each
bank and its WBCP threshold are the benchmark's own), record at the population threshold
lambda* of the tilted test law:
  r_hat  the bank's self-normalized weighted miss rate sum w I / sum w, I = 1{score > lambda*}
  mean, sd   the WBCP posterior mean and SD of the risk at lambda* (Algorithm 1's masses,
             w_i E_i and the test atom wbar E_{n+1} at loss one; --draws exponential draws)
  u      the posterior probability that the risk at lambda* is at most alpha
  fail   whether the benchmark's oracle-WBCP threshold has realized risk above alpha
Under uniform weights the posterior is frequentist-exact, so Pr(u >= beta) = 1 - beta. The
summary compares the across-bank SD of r_hat with the average posterior SD, and the
posterior SD of failing banks with the across-bank SD.

OPENBLAS_NUM_THREADS=1 python experiments/wbcp/weighted_mechanism.py --frozen F --tilt density --gamma 1 --output out.json
"""

import argparse
import json
import math
import multiprocessing
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from calibration.wbcp import calibrate  # noqa: E402
from experiments.wbcp import d4rl_benchmark as bench  # noqa: E402

_STATE = None  # inherited by forked workers


def bank(index):
    setup, args, lam, risk = _STATE
    pool, tilt = setup.pool, setup.tilts[0]
    rows = bench.draw_calibration(bench.stream(args.seed, "cal", args.n, index), args.n, pool.size)
    scores = pool.scores["normalized"][rows]
    weights, wbar = bench.common_scale(tilt.log_mass[rows], math.log(tilt.square_total / tilt.total))
    result = calibrate(scores, bench.stream(args.seed, "oracle", args.n, index, tilt.name, tilt.gamma, "normalized"),
                       weights, max(wbar, np.finfo(np.float64).tiny), alpha=args.alpha, beta=args.beta, draws=1000)
    miss = scores > lam
    draws = np.random.default_rng([args.seed, index]).exponential(size=(args.draws, rows.size + 1))
    atom = wbar * draws[:, -1]
    posterior = (draws[:, :-1] @ (weights * miss) + atom) / (draws[:, :-1] @ weights + atom)
    return (float(weights @ miss / weights.sum()), float(posterior.mean()), float(posterior.std()),
            float(np.mean(posterior <= args.alpha)), bool(risk(np.array([result.threshold]))[0] > args.alpha),
            bool(result.lambda_hat < lam))


def main(argv=None):
    global _STATE
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--tilt", required=True, choices=("policy", "density", "state"))
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--shuffle-tilt", type=int, default=None)
    parser.add_argument("--n", type=int, default=1103)
    parser.add_argument("--banks", type=int, default=4000)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026093001)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        parser.error("--output already exists")
    bench_argv = ["--frozen", args.frozen, "--tilt", args.tilt, "--gamma", str(args.gamma), "--score", "normalized"]
    if args.shuffle_tilt is not None:
        bench_argv += ["--shuffle-tilt", str(args.shuffle_tilt)]
    setup = bench.build_setup(bench.parse_args(bench_argv))
    risk = setup.risks[(0, "normalized")]
    lam = risk.lambda_star(args.alpha)
    _STATE = (setup, args, lam, risk)
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as workers:
            rows = workers.map(bank, range(args.banks), chunksize=16)
    else:
        rows = [bank(i) for i in range(args.banks)]
    r_hat, mean, sd, u, fail, clamp_below = (np.array(column) for column in zip(*rows))
    spread = float(r_hat.std(ddof=1))
    implied = (u >= math.ceil(args.beta * 1000) / 1000) & clamp_below  # failure read off the posterior at lambda*
    summary = dict(
        settings=vars(args), lambda_star=float(lam), exact_risk_at_lambda_star=float(risk(np.array([lam]))[0]),
        fail=float(fail.mean()), fail_ci=list(stats.binomtest(int(fail.sum()), fail.size).proportion_ci(method="exact")),
        fail_from_posterior_at_lambda_star=float(implied.mean()), agreement=float(np.mean(implied == fail)),
        r_hat_mean=float(r_hat.mean()), r_hat_sd_across_banks=spread, r_hat_skewness=float(stats.skew(r_hat)),
        posterior_sd_mean=float(sd.mean()), sd_ratio_across_over_posterior=spread / float(sd.mean()),
        corr_r_hat_posterior_sd=float(np.corrcoef(r_hat, sd)[0, 1]),
        failing_median_posterior_sd_over_across=float(np.median(sd[fail]) / spread) if fail.any() else None,
        passing_median_posterior_sd_over_across=float(np.median(sd[~fail]) / spread),
        failing_median_r_hat=float(np.median(r_hat[fail])) if fail.any() else None,
        u_ge_beta=float(np.mean(u >= args.beta)),
    )
    with open(args.output, "x") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=1))


if __name__ == "__main__":
    main()
