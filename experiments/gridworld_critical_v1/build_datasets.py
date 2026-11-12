"""Generate, check and save a labeled set of datasets. A label is written once.

    python -I -B build_datasets.py --label dev --seeds 6101 6102 6103 --purpose "development"

Writes data/<label>/ (ignored by git), receipts/<label>_manifest.json and receipts/<label>_checks.json.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import critical_checks as checks  # noqa: E402
import critical_collect as C  # noqa: E402
import critical_env as E  # noqa: E402

SOURCES = ("critical_env.py", "critical_collect.py", "critical_checks.py", "build_datasets.py", "PUBLISHED_SETUP.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(checks.ORIGINAL_TASKS))
    args = parser.parse_args()
    folder, receipts = HERE / "data" / args.label, HERE / "receipts"
    manifest_path = receipts / f"{args.label}_manifest.json"
    if folder.exists() or manifest_path.exists():
        sys.exit(f"label {args.label!r} already exists; labels are written once")
    folder.mkdir(parents=True)
    receipts.mkdir(exist_ok=True)
    entries, reports = [], {}
    for task_name in args.tasks:
        task = checks.TASKS[task_name]
        for seed in args.seeds:
            complete = None
            for arm in C.ARMS:
                data = C.generate(task, arm, seed)
                if arm == "complete":
                    complete = data
                report = checks.check_dataset(task, arm, data, None if arm == "complete" else complete)
                name = f"{task_name}_{arm}_seed{seed}.npz"
                np.savez_compressed(folder / name, **data)
                reloaded = checks.load(folder / name)
                assert C.digest(reloaded) == C.digest(data), name
                entries.append({"file": name, "task": task_name, "noise": task.noise, "arm": arm, "seed": seed,
                                "rows": int(len(data["s"])), "array_sha256": C.digest(data),
                                "left_out": report["counts"]["left_out"], "checks_passed": report["passed"]})
                reports[name] = report
                print(f"{name:52s} {'pass' if report['passed'] else 'FAIL'}", flush=True)
    manifest = {"label": args.label, "purpose": args.purpose, "created_unix": time.time(), "seeds": args.seeds,
                "rows_per_cell": C.ROWS_PER_CELL, "uniform_rows": C.UNIFORM_ROWS, "noise_sd": C.NOISE_SD,
                "decision_cell": C.DECISION_CELL, "gamma": E.GAMMA, "beta": E.BETA, "cutoff": E.CUTOFF,
                "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES},
                "datasets": entries}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    (receipts / f"{args.label}_checks.json").write_text(json.dumps(reports, indent=1))
    failed = [e["file"] for e in entries if not e["checks_passed"]]
    print(f"{len(entries)} datasets, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
