"""EXPLORATORY, not registered in PROTOCOL.md. No training.

Question: how much of what the calibrated width adds to the support penalty (Amendment 4) can
be had by simply scaling commands after training?

Each support-only policy's commands are multiplied by one number, then clipped to [-1, 1],
until their average absolute preference equals that of the matched change-1 policy. The
result is evaluated exactly, like every other policy. The baseline gets the same treatment.

    python -I -B sharpen_after_training.py --label dev_v4

Writes results/<label>/exploratory_sharpening.json and prints a table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluation as ev  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

ARMS = ("complete", "targeted", "placebo")
ROWS = ("baseline", "support", "support_only", "support_only_then_sharpened", "baseline_then_sharpened")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    commands, setting_task = {}, {}
    for meta_path in sorted(folder.glob("stage*_*.json")):
        if meta_path.name.endswith("_datasets.json"):
            continue
        meta = json.loads(meta_path.read_text())
        with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as arrays:
            for run in meta["runs"]:
                key = (run["setting"], run["data_seed"], run["arm"], run["train_seed"], run["method"])
                commands[key] = arrays["commands"][run["index"]]
                setting_task[run["setting"]] = run["task"]
    out = {}
    for setting in sorted(setting_task, key=lambda t: (t != "honeypot", t)):
        task = checks.TASKS[setting_task[setting]]
        free = task.free_cells()
        runs = sorted({(k[1], k[3]) for k in commands if k[0] == setting})
        for arm in ARMS:
            rows = {name: [] for name in ROWS}
            factors = []
            for data_seed, train_seed in runs:
                get = lambda method: commands[(setting, data_seed, arm, train_seed, method)]  # noqa: E731
                target = float(np.abs(get("support")[free]).mean())
                sharpened_support_only, factor = ev.sharpen(get("support_only"), free, target)
                sharpened_baseline, _ = ev.sharpen(get("baseline"), free, target)
                factors.append(factor)
                for name, policy in (("baseline", get("baseline")), ("support", get("support")),
                                     ("support_only", get("support_only")),
                                     ("support_only_then_sharpened", sharpened_support_only),
                                     ("baseline_then_sharpened", sharpened_baseline)):
                    record, reached = ev.policy_record(task, policy), task.outcomes(policy)
                    rows[name].append({"data_seed": data_seed, "train_seed": train_seed,
                                       "primary_outcome": record["primary_outcome"],
                                       "mean_value_all_starts": record["mean_value_all_starts"],
                                       "bad_endings_all_starts": float((reached["hazard"] + reached["timeout"])[free].mean()),
                                       "stalled_start_cells": int((reached["timeout"][free] > 0.5).sum())})
            out[f"{setting}|{arm}"] = {"mean_scale_factor": float(np.mean(factors)), "runs": rows}
            print(f"{setting} | {arm} | mean scale factor {np.mean(factors):.2f}")
            for name, values in rows.items():
                primary = np.array([v["primary_outcome"] for v in values])
                print(f"    {name:28s} primary {100 * primary.mean():6.2f}% | at zero in {int((primary == 0).sum())} of {len(values)} runs"
                      f" | value {np.mean([v['mean_value_all_starts'] for v in values]):.2f}"
                      f" | bad endings {100 * np.mean([v['bad_endings_all_starts'] for v in values]):.2f}%"
                      f" | stalled cells {sum(v['stalled_start_cells'] for v in values)}")
    (folder / "exploratory_sharpening.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
