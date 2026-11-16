"""The saved trained policies behind the latest reported numbers. Read-only; trains nothing.

Honeypot and Bottleneck: experiments/gridworld_ood_v1/results/confirm_v5, the nine judged datasets
7501 to 7509, training seeds 70 to 72.
Published gridworld: experiments/gridworld_critical_v1/results/confirm_version_a, datasets 6501 to
6509, training seeds 20 to 22.

Each record: method, data_seed, train_seed, task, commands [100, moves] (the policy: one command per
cell), q1 [100, corner commands] (the first critic's value), width (the saved margin).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OOD = ROOT / "experiments" / "gridworld_ood_v1"
CRITICAL = ROOT / "experiments" / "gridworld_critical_v1"
INHOUSE_BLOCK = OOD / "results" / "confirm_v5"
INHOUSE_DATA = {"honeypot": OOD / "data" / "confirm5_h", "bottleneck": OOD / "data" / "confirm5_b"}
INHOUSE_DATASETS = tuple(range(7501, 7510))
PUBLISHED_BLOCK = CRITICAL / "results" / "confirm_version_a"
PUBLISHED_DATA = CRITICAL / "data" / "published3"
PUBLISHED_DATASETS = tuple(range(6501, 6510))

# The names used in the meeting deck, in the order figures draw them.
METHOD_NAMES = {"baseline": "Standard learner", "support_blind_width": "BCA policy guard",
                "support_shuffled": "Shuffled control"}


def _load(folder, stem, keep):
    runs = json.loads((folder / f"{stem}.json").read_text())["runs"]
    with np.load(folder / f"{stem}.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("commands", "q1", "width")}
    out = []
    for run in runs:
        if keep(run):
            i = run["index"]
            out.append({"method": run["method"], "data_seed": run["data_seed"], "train_seed": run["train_seed"],
                        "task": run["task"], "commands": arrays["commands"][i], "q1": arrays["q1"][i],
                        "width": arrays["width"][i]})
    return out


def inhouse(setting, arm="targeted"):
    """setting: "honeypot", or "bottleneck_r2_exit45" / "..._exit35" / "..._exit25"."""
    family = "honeypot" if setting == "honeypot" else "bottleneck_r2_one_hot"

    def keep(run):
        return run["task"] == setting and run["arm"] == arm and run["data_seed"] in INHOUSE_DATASETS

    return (_load(INHOUSE_BLOCK, f"stage1_{family}", keep)
            + _load(INHOUSE_BLOCK, f"stage2_{family}__support_blind_width", keep))


def published(recipe="optimal"):
    def keep(run):
        return run["recipe"] == recipe and run["data_seed"] in PUBLISHED_DATASETS

    stem = "methods_single_critical_noise10__"
    return _load(PUBLISHED_BLOCK, stem + "baseline", keep) + _load(PUBLISHED_BLOCK, stem + "support_blind_width", keep)


def inhouse_dataset(setting, seed, arm="targeted"):
    family = "honeypot" if setting == "honeypot" else "bottleneck"
    with np.load(INHOUSE_DATA[family] / f"{setting}_{arm}_seed{seed}.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def published_dataset(seed, recipe="optimal"):
    with np.load(PUBLISHED_DATA / f"single_critical_noise10_{recipe}_seed{seed}.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}
