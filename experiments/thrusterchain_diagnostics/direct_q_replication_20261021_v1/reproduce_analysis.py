"""Reproduce saved arithmetic only; no model imports, training or rollouts."""
import argparse
from pathlib import Path
import numpy as np
import replication as R


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = arrays(args.root / "prepared" / "inputs.npz")
    scores = arrays(args.root / "prepared" / "scores.npz")
    draws = arrays(args.root / "prepared" / "bootstrap_plan.npz")["state_indices"]
    raw = arrays(args.root / "results" / "raw_results.npz")
    truth = raw["true_policy_q"]
    for key in ("rollout_completed", "rollout_finite", "rollout_steps"):
        bank[key] = raw[key]
    actual, rows, boot = R.analyze(bank, truth, scores, draws)
    expected = R.read_json(args.root / "results" / "paired_results.json")
    for key in actual:
        R.require(actual[key] == expected[key], "saved analysis mismatch: " + key)
    R.require(rows == expected["state_records"], "saved per-state record mismatch")
    saved_boot = arrays(args.root / "results" / "bootstrap_statistics.npz")
    R.require(set(boot) == set(saved_boot), "bootstrap statistic keys differ")
    for key in boot:
        np.testing.assert_array_equal(boot[key], saved_boot[key])
    R.write_json(args.output / "reproduction_receipt.json", {
        "status": "passed", "simulator_calls": 0, "model_calls": 0,
        "metrics": len(actual["metrics"]), "states": len(rows),
        "bootstrap_arrays": len(boot),
        "source_sha256": R.digest(R.ROOT / "replication.py"),
        "expected_results_sha256": R.digest(args.root / "results" / "paired_results.json"),
    })
    print("Saved arithmetic and bootstrap arrays reproduced; zero model or simulator calls.")


if __name__ == "__main__":
    main()
