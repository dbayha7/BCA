"""Within-episode dependence of a frozen score pool, and predicted bank failure rates.

Reads a frozen artifact (freeze_scores.py) and, for the uniform test law, measures how
strongly rows of one episode depend on each other in the quantity calibration relies on:
the exceedance indicator I = 1{score > lambda*}, lambda* the shortest valid threshold.

WBCP's posterior treats calibration rows as independent. When a bank holds m correlated
rows per episode, the empirical miss rate varies more than the posterior assumes, by the
design effect D = 1 + (m - 1) rho (rho = intra-episode correlation of I). The beta-credible
threshold then sits about sqrt(D) times too close to the true quantile, so, in a normal
approximation, a bank fails with probability 1 - Phi(z_beta / sqrt(D)) instead of 1 - beta.
D is estimated two ways: from the ANOVA intra-class correlation, and by Monte Carlo over
the benchmark's own bank sampler (d4rl_benchmark.draw_calibration), which needs no model.

Designs: whole-episode banks (BCA's block reservation) of several sizes, and thinned banks
with K rows from each of ceil(n / K) episodes. For each it reports the predicted failure,
and the share of the full dataset a real bank would reserve (whole episodes must leave
training to keep calibration rows disjoint from training trajectories).

Written to be run BEFORE the benchmark sweeps, so the predictions are on record first:

python experiments/wbcp/dependence.py --frozen runs/wbcp_frozen/<name> --output runs/wbcp_dependence/<name>
"""

import argparse
import datetime
import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.wbcp import d4rl_benchmark as bench  # noqa: E402

LAGS = (1, 2, 5, 10, 25, 50, 100, 200)


def intraclass_correlation(values, groups):
    """One-way ANOVA estimator of the intra-class correlation with unequal group sizes."""
    values = np.asarray(values, np.float64)
    _, index, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    k, total = sizes.size, values.size
    means = np.bincount(index, values) / sizes
    grand = values.mean()
    between = np.sum(sizes * (means - grand) ** 2) / (k - 1)
    within = np.sum((values - means[index]) ** 2) / (total - k)
    size0 = (total - np.sum(sizes.astype(np.float64) ** 2) / total) / (k - 1)
    return float((between - within) / (between + (size0 - 1) * within))


def lag_correlations(values, episode, timestep, lags=LAGS):
    """Correlation of values at timesteps t and t + lag within the same episode."""
    order = np.lexsort((timestep, episode))
    v, e, t = values[order], episode[order], timestep[order]
    result = {}
    for lag in lags:
        same = (e[lag:] == e[:-lag]) & (t[lag:] - t[:-lag] == lag)
        result[lag] = float(np.corrcoef(v[:-lag][same], v[lag:][same])[0, 1]) if same.sum() > 2 else None
    return result


def monte_carlo_design_effect(indicator, groups, design, n, banks, seed):
    """Var(bank miss rate) / (p (1 - p) / mean bank size) under the benchmark's sampler."""
    rng = np.random.default_rng(seed)
    rates, sizes = np.empty(banks), np.empty(banks)
    for b in range(banks):
        if design == "iid":
            rows = bench.draw_calibration(rng, n, indicator.size)
        elif design == "blocks":
            rows = bench.draw_calibration(rng, n, indicator.size, groups)
        elif design.startswith("strat"):
            rows = bench.draw_calibration(rng, n, indicator.size, groups, per_episode=int(design[5:]),
                                          spacing="stratified")
        elif design.startswith("resv"):
            rows = bench.draw_calibration(rng, n, indicator.size, groups, per_episode=int(design[4:]),
                                          spacing="reservation")
        else:
            rows = bench.draw_calibration(rng, n, indicator.size, groups, per_episode=int(design))
        rates[b], sizes[b] = indicator[rows].mean(), rows.size
    p = indicator.mean()
    return float(rates.var(ddof=1) / (p * (1 - p) / sizes.mean())), float(sizes.mean())


def predicted_failure(design_effect, beta):
    return float(stats.norm.sf(stats.norm.isf(1 - beta) / math.sqrt(design_effect)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--output", required=True, help="new directory for predictions.json / predictions.md")
    parser.add_argument("--n", type=int, default=1103, help="bank size for the thinned designs (BCA's hopper bank)")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1103, 2300, 4600, 11500])
    parser.add_argument("--per-episode", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100])
    parser.add_argument("--stratified", type=int, nargs="*", default=[],
                        help="also predict banks with K rows spread one per K-th of each episode")
    parser.add_argument("--reservation", type=int, nargs="*", default=[],
                        help="also predict BCA's reservation sampler (distinct episodes, stratified rows)")
    parser.add_argument("--banks", type=int, default=20000, help="Monte Carlo banks per design")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        parser.error("--output already exists")

    pool = bench.load_pool(args.frozen, "heldout")
    groups = bench.episode_groups(pool.episode, pool.timestep)
    lengths = groups[2].astype(np.float64)
    meta_rows = pool.metadata.get("rows", {})
    total_rows = meta_rows.get("converted")
    total_episodes = meta_rows.get("episodes") or meta_rows.get("converted_components")
    mean_length_full = total_rows / total_episodes if total_rows and total_episodes else float(lengths.mean())
    size_biased_length = float(np.sum(lengths ** 2) / np.sum(lengths))

    report = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        frozen=os.path.abspath(args.frozen), frozen_npz_sha256=pool.metadata.get("npz_sha256"),
        settings={k: v for k, v in vars(args).items()},
        git=subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip(),
        pool=dict(rows=pool.size, episodes=int(lengths.size), mean_length=float(lengths.mean()),
                  size_biased_length=size_biased_length, min_length=int(lengths.min()),
                  max_length=int(lengths.max())),
        dataset=dict(rows=total_rows, episodes=total_episodes, mean_length=mean_length_full),
        model=("normal approximation: failure = 1 - Phi(z_beta / sqrt(D)); D = 1 + (m - 1) rho or Monte Carlo; "
               "ignores discreteness, the posterior's exact shape and Monte Carlo noise in lambda_hpd"),
        scores={},
    )
    designs = ([("blocks", n) for n in args.block_sizes] + [(str(k), args.n) for k in args.per_episode]
               + [(f"strat{k}", args.n) for k in args.stratified]
               + [(f"resv{k}", args.n) for k in args.reservation])
    for s, name in enumerate(("normalized", "raw")):
        scores = pool.scores[name]
        lam = bench.ExactRisk(scores, np.ones(scores.size)).lambda_star(args.alpha)
        indicator = (scores > lam).astype(np.float64)
        rho = intraclass_correlation(indicator, pool.episode)
        entry = dict(lambda_star=lam, exceedance_rate=float(indicator.mean()), icc=rho,
                     icc_score=intraclass_correlation(scores, pool.episode),
                     lag_correlation=lag_correlations(indicator, pool.episode, pool.timestep), designs=[])
        for d, (design, n) in enumerate(designs):
            mc, size = monte_carlo_design_effect(indicator, groups, design, n, args.banks, args.seed + 100 * s + d)
            stratified, bca = design.startswith("strat"), design.startswith("resv")
            k = (int(design[5:]) if stratified else int(design[4:]) if bca
                 else None if design == "blocks" else int(design))
            rows_per_episode = size_biased_length if design == "blocks" else float(k)
            # The ICC model averages over random pairs; stratified rows are farther apart than
            # random ones, so only the Monte Carlo design effect applies to them. The reservation
            # sampler also draws distinct episodes from this finite pool, so its Monte Carlo
            # value includes a finite-population reduction a real deployment would not get.
            model = None if stratified or bca else 1.0 + (rows_per_episode - 1.0) * rho
            episodes_needed = size / mean_length_full if design == "blocks" else math.ceil(n / k)
            reserved = episodes_needed * mean_length_full
            entry["designs"].append(dict(
                design=("whole episodes" if design == "blocks" else f"{k} per episode, stratified" if stratified
                        else f"{k} per episode, BCA reservation" if bca else f"{k} per episode"), n=n,
                mean_bank_size=size, rows_per_episode=rows_per_episode,
                design_effect_icc=model, design_effect_mc=mc,
                predicted_failure_icc=None if model is None else predicted_failure(model, args.beta),
                predicted_failure_mc=predicted_failure(mc, args.beta),
                episodes_reserved=episodes_needed, reserved_share_of_dataset=reserved / total_rows if total_rows else None,
            ))
        report["scores"][name] = entry

    os.makedirs(args.output)
    with open(os.path.join(args.output, "predictions.json"), "x") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)
    lines = [f"# Dependence predictions ({report['created_utc']})", "",
             f"Pool: {pool.size} rows, {int(lengths.size)} episodes, mean length {lengths.mean():.1f} "
             f"(size-biased {size_biased_length:.1f}). Dataset: {total_rows} rows, {total_episodes} episodes.", ""]
    for name, entry in report["scores"].items():
        lines += [f"## {name} score: lambda* = {entry['lambda_star']:.4f}, rho(I) = {entry['icc']:.4f}, "
                  f"rho(score) = {entry['icc_score']:.4f}", "",
                  "lag correlation of I: " + ", ".join(f"{k}: {v:.3f}" for k, v in entry["lag_correlation"].items()
                                                       if v is not None), "",
                  "| design | n | rows/episode | D (ICC) | D (MC) | predicted failure (ICC / MC) | episodes reserved | share of dataset |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in entry["designs"]:
            icc = "n/a" if row["design_effect_icc"] is None else f"{row['design_effect_icc']:.2f}"
            icc_fail = "n/a" if row["predicted_failure_icc"] is None else f"{100 * row['predicted_failure_icc']:.1f}%"
            lines.append(f"| {row['design']} | {row['n']} | {row['rows_per_episode']:.1f} | {icc} | "
                         f"{row['design_effect_mc']:.2f} | {icc_fail} / {100 * row['predicted_failure_mc']:.1f}% | "
                         f"{row['episodes_reserved']:.0f} | {100 * (row['reserved_share_of_dataset'] or 0):.1f}% |")
        lines.append("")
    with open(os.path.join(args.output, "predictions.md"), "x") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
