"""Generate datasets for both tasks and all three arms, check them, and write receipts.

No training and no learner code. Refuses to overwrite an existing label.

    python -I -B build_datasets.py --label dev --seeds 5101 5102 5103
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402
import environment as env  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCES = ("environment.py", "collect.py", "unseen_move_checks.py", "build_datasets.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(task):
    commands, value = env.best_policy(task)
    names = env.DIRECTION_NAMES
    entry = {"walls": sorted(task.walls), "hazards": sorted(task.hazards), "bump_reward": task.bump_reward,
             "goal": env.GOAL,
             "free_cells": len(task.free_cells()),
             "targeted_moves": {str(c): [names[d] for d in ds] for c, ds in collect.targeted_moves(task).items()},
             "placebo_moves": {str(c): [names[d] for d in ds] for c, ds in collect.placebo_moves(task).items()},
             "best_policy_value_from_A": float(value[env.A_START])}
    if task.low_exit is not None:
        per_direction = task.direction_values(value)[env.B]
        entry.update(low_exit_cell=task.low_exit, low_exit_reward=task.low_exit_reward,
                     value_of_crossing_at_B=float(per_direction[env.RIGHT]),
                     true_margin_at_B_crossing_minus_low_exit=float(per_direction[env.RIGHT] - per_direction[env.UP]))
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="dev")
    parser.add_argument("--seeds", type=int, nargs="+", default=[5101, 5102, 5103])
    parser.add_argument("--purpose", default="development: design shakedown only, never confirmation")
    parser.add_argument("--tasks", default="", help="only task settings whose name starts with this text")
    args = parser.parse_args()
    tasks = {name: task for name, task in checks.TASKS.items() if name.startswith(args.tasks)}
    if not tasks:
        sys.exit(f"no task setting starts with '{args.tasks}'")

    folder = HERE / "data" / args.label
    receipts = HERE / "receipts"
    manifest_path = receipts / f"{args.label}_manifest.json"
    report_path = receipts / f"{args.label}_unseen_move_checks.json"
    if folder.exists() or manifest_path.exists() or report_path.exists():
        sys.exit(f"label '{args.label}' already exists; choose another label or remove it deliberately")
    folder.mkdir(parents=True)
    receipts.mkdir(exist_ok=True)

    manifest = {"label": args.label, "purpose": args.purpose, "training_updates": 0,
                "python": platform.python_version(), "numpy": np.__version__,
                "source_sha256": {name: sha256(HERE / name) for name in SOURCES},
                "execution": {"beta": env.BETA, "cutoff": env.CUTOFF},
                "collection": {"rows_per_cell": collect.ROWS_PER_CELL, "uniform_rows": collect.UNIFORM_ROWS,
                               "demonstrator_noise_sd": collect.NOISE_SD},
                "minimum_executions_for_a_supported_move": checks.MIN_SUPPORT,
                "seeds": args.seeds,
                "tasks": {name: describe(task) for name, task in tasks.items()},
                "datasets": []}
    reports = []
    for name, task in tasks.items():
        for seed in args.seeds:
            complete = None
            for arm in collect.ARMS:
                data, facts = collect.generate(task, arm, seed)
                if arm == "complete":
                    complete = data
                report = checks.check_dataset(task, arm, data, None if arm == "complete" else complete)
                file_name = f"{name}_{arm}_seed{seed}.npz"
                np.savez_compressed(folder / file_name, **data)
                manifest["datasets"].append({"file": file_name, "task": name, "arm": arm, "seed": seed,
                                             "array_sha256": collect.digest(data),
                                             "file_sha256": sha256(folder / file_name),
                                             "learner_fields": list(collect.LEARNER_FIELDS),
                                             "checks_passed": report["passed"], **facts})
                reports.append({"file": file_name, "seed": seed, **report})
                print("PASS" if report["passed"] else "FAIL", file_name, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    report_path.write_text(json.dumps(reports, indent=2, allow_nan=False))
    failed = [r["file"] for r in reports if not r["passed"]]
    print(f"{len(reports)} datasets written to {folder}; {len(failed)} failed their checks")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
