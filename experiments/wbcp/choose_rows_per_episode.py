"""Rows per episode K for each BCA config: the smallest K >= 5 whose reservation fits the cap.

User rule (2026-09-30): K = 5 where it fits the config's withholding cap, otherwise raise K
until it does (DEPENDENCE.md, Change 9). "Fits" means the withheld rows stay within
max_fraction of the dataset for every one of --seeds simulated reservation seeds, with
BCA's own sampler (calibration/bank.py) on the dataset's D4RL episode boundaries. Also
reports the withheld share at the configured reservation seed and the bank's measured
dependence evidence for the score the host's BCA calibrates (calibration/bank.py,
dependence_evidence and DEPLOYED_SCORE).

python experiments/wbcp/choose_rows_per_episode.py --output runs/rows_per_episode.json
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from calibration.bank import DEPLOYED_SCORE, dependence_evidence, stratified_bank  # noqa: E402
from calibration.reference import qlearning_episode_ids  # noqa: E402

HOSTS = ("td3_bc", "rebrac", "cql", "iql")


def episode_lengths(path):
    with h5py.File(path, "r") as handle:
        ids = qlearning_episode_ids({k: handle[k][:] for k in ("terminals", "timeouts")})
    return np.bincount(ids - ids.min())[np.unique(ids - ids.min())]


def fits(lengths, target, k, cap, seeds):
    limit = cap * lengths.sum()
    for seed in range(seeds):
        try:
            episodes, _ = stratified_bank(lengths, target, k, np.random.default_rng(seed))
        except ValueError:
            return False
        if lengths[episodes].sum() > limit:
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=os.path.expanduser("~/.d4rl/datasets"))
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--minimum", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        parser.error("--output already exists")
    root = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    report, lengths_cache = {}, {}
    for host in HOSTS:
        config = yaml.safe_load(open(os.path.join(root, host + ".yaml")))
        cap, seed = config["reservation"]["max_fraction"], config["reservation"]["seed"]
        for name, entry in config["datasets"].items():
            filename = entry["cache"]["filename"]
            if filename not in lengths_cache:
                lengths_cache[filename] = episode_lengths(os.path.join(args.data, filename))
            lengths = lengths_cache[filename]
            target = entry["reservation"]["size"]
            k = args.minimum
            while not fits(lengths, target, k, cap, args.seeds):
                k += 1
                if k > max(target, lengths.max()):
                    raise RuntimeError(f"{host}/{name}: no K fits")
            episodes, offsets = stratified_bank(lengths, target, k, np.random.default_rng(seed))
            row = dict(target=target, cap=cap, rows_per_episode=k, episodes=len(episodes),
                       calibration_rows=int(sum(len(o) for o in offsets)),
                       withheld_share_at_seed=float(lengths[episodes].sum() / lengths.sum()),
                       dataset_episodes=int(lengths.size),
                       dependence_evidence=dependence_evidence(host, entry["environment"], k, target,
                                                               score=DEPLOYED_SCORE[host])["status"])
            report.setdefault(host, {})[name] = row
            print(f"{host:7s} {name:12s} n={target:<5d} cap={cap:.2f}  K={k:<4d} episodes={row['episodes']:<5d} "
                  f"bank={row['calibration_rows']:<5d} withheld={100 * row['withheld_share_at_seed']:5.1f}%  "
                  f"{row['dependence_evidence']}", flush=True)
    with open(args.output, "x") as handle:
        json.dump(dict(seeds=args.seeds, minimum=args.minimum, hosts=report), handle, indent=1)


if __name__ == "__main__":
    main()
