"""Run only the two saved-array analyses into a new local output directory.

No experiment module is imported. No model, simulator, calibration or training
command is dispatched. Requires the existing local NumPy/matplotlib environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EST = timezone(timedelta(hours=-5))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory beneath this analysis directory; never overwrites an earlier run.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(HERE) or out == HERE:
        raise ValueError("Output must be a new child of this analysis directory")
    scripts = (HERE / "dataset_comparison.py", HERE / "uncertainty_diagnostics.py")
    for path in scripts:
        if not path.is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {
        "status": "running_saved_array_analysis",
        "started_est": datetime.now(EST).isoformat(),
        "analysis_sources": {p.name: sha(p) for p in (Path(__file__), *scripts)},
        "python": sys.version,
        "jobs": [],
        "scientific_execution": {"model_calls": 0, "simulator_calls": 0,
                                 "optimizer_updates": 0, "calibrations": 0},
        "scope": "Exploratory arithmetic on already completed knee study; no Path A mutation or new experiment.",
    }
    try:
        for script, name in zip(scripts, ("dataset", "diagnostics")):
            command = [sys.executable, "-B", str(script), "--output-dir", str(out / name)]
            t0 = time.monotonic()
            with (out / (name + "_console.txt")).open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            receipt["jobs"].append({"analysis": name, "exit_code": result.returncode,
                                    "elapsed_seconds": time.monotonic() - t0})
            if result.returncode:
                raise RuntimeError(f"Saved-data {name} analysis failed; inspect its preserved console log")
        if any(sha(HERE / name) != digest for name, digest in receipt["analysis_sources"].items()):
            raise RuntimeError("Analysis source changed during execution")
        receipt["status"] = "completed_saved_array_analysis"
    except BaseException as exc:
        receipt["status"] = "failed_or_interrupted_saved_array_analysis"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["finished_est"] = datetime.now(EST).isoformat()
        receipt["outputs"] = {p.relative_to(out).as_posix(): sha(p)
                              for p in sorted(out.rglob("*")) if p.is_file()}
        (out / "RUN_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(out / "RUN_RECEIPT.json")}))


if __name__ == "__main__":
    main()
