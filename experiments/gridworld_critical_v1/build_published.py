"""Generate, check and save the datasets of PROTOCOL.md, Amendment 2. A label is written once.

    python -I -B build_published.py --label published --seeds 6301 6302 6303 6304 6305 --purpose "..."

Writes data/<label>/ (ignored by git), receipts/<label>_manifest.json and receipts/<label>_checks.json.
A seed whose three datasets do not all pass is dropped whole (gate W0); the manifest lists which.
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
import critical_env as E  # noqa: E402
import published_checks as checks  # noqa: E402
import published_collect as C  # noqa: E402

SOURCES = ("critical_env.py", "published_collect.py", "published_checks.py", "build_published.py", "PUBLISHED_SETUP.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    folder, receipts = HERE / "data" / args.label, HERE / "receipts"
    manifest_path = receipts / f"{args.label}_manifest.json"
    if folder.exists() or manifest_path.exists():
        sys.exit(f"label {args.label!r} already exists; labels are written once")
    folder.mkdir(parents=True)
    receipts.mkdir(exist_ok=True)
    task = E.published()
    entries, reports, seed_passed = [], {}, {}
    for seed in args.seeds:
        made = {recipe: C.generate(task, recipe, seed) for recipe in C.RECIPES}
        for recipe, data in made.items():
            report = checks.check_dataset(task, recipe, data, made["optimal"] if recipe == "mixed" else None)
            name = checks.file_name(task, recipe, seed)
            np.savez_compressed(folder / name, **data)
            assert C.digest(checks.load(folder / name)) == C.digest(data), name
            entries.append({"file": name, "task": task.name, "noise": task.noise, "recipe": recipe, "seed": seed,
                            "rows": int(len(data["s"])), "array_sha256": C.digest(data),
                            "checks_passed": report["passed"]})
            reports[name] = report
            print(f"{name:52s} {'pass' if report['passed'] else 'FAIL'}", flush=True)
        seed_passed[str(seed)] = all(e["checks_passed"] for e in entries if e["seed"] == seed)
    manifest = {"label": args.label, "purpose": args.purpose, "created_unix": time.time(), "seeds": args.seeds,
                "seeds_passing_every_check": [seed for seed in args.seeds if seed_passed[str(seed)]],
                "rows": C.ROWS, "random_share": C.RANDOM_SHARE, "recipes": list(C.RECIPES),
                "gamma": E.GAMMA, "horizon": E.HORIZON, "beta": E.BETA, "cutoff": E.CUTOFF,
                "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES},
                "datasets": entries}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    (receipts / f"{args.label}_checks.json").write_text(json.dumps(reports, indent=1))
    dropped = [seed for seed in args.seeds if not seed_passed[str(seed)]]
    print(f"{len(entries)} datasets; seeds dropped by the checks: {dropped or 'none'}")


if __name__ == "__main__":
    main()
