"""How many episodes each dataset can withhold within a cap, for every simulated seed.

For each cached D4RL dataset in the BCA configs and each cap, the largest number of
length-proportional episodes m whose withheld rows stay within the cap for every one of
--seeds reservation seeds (BCA's sampler, calibration/bank.py, one row per episode). A bank
of K rows per episode then has about K * m rows (DEPENDENCE.md, Change 12).

python experiments/wbcp/episode_budget.py --output runs/wbcp_dependence/all-datasets/episode_budget.json
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.wbcp.choose_rows_per_episode import episode_lengths, fits  # noqa: E402


def largest(lengths, cap, seeds):
    lo, hi = 1, len(lengths)
    if not fits(lengths, 1, 1, cap, seeds):
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(lengths, mid, 1, cap, seeds):
            lo = mid
        else:
            hi = mid - 1
    return lo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=os.path.expanduser("~/.d4rl/datasets"))
    parser.add_argument("--caps", type=float, nargs="+", default=[0.10, 0.25])
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        parser.error("--output already exists")
    config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "td3_bc.yaml")))
    report = {}
    for name, entry in config["datasets"].items():
        lengths = episode_lengths(os.path.join(args.data, entry["cache"]["filename"]))
        row = dict(episodes=int(lengths.size), rows=int(lengths.sum()), mean_length=float(lengths.mean()),
                   size_biased_length=float((lengths ** 2).sum() / lengths.sum()))
        for cap in args.caps:
            row[f"episodes_within_{cap:g}"] = largest(lengths, cap, args.seeds)
        report[name] = row
        print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(args.output, "x") as handle:
        json.dump(dict(seeds=args.seeds, caps=args.caps, datasets=report), handle, indent=1)


if __name__ == "__main__":
    main()
