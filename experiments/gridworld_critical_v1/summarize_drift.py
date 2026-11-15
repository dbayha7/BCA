"""Summarize the read-outs of diagnose_version_c3.py (PROTOCOL.md, Amendment 9). NumPy only.

    python -I -B summarize_drift.py --label diag_version_c3 --against version_c3_published
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ("next_commands_flagged_with_noise", "next_commands_flagged_without_noise", "score_with_noise_minus_without",
         "charge_at_next_command_with_noise", "charge_at_next_command_without_noise", "comparison_commands_flagged",
         "score_at_flagged_comparison_commands", "error_at_recorded_commands", "error_at_next_command_without_noise",
         "error_at_next_command_with_noise", "true_value_at_recorded_commands", "unit", "cap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--against", default=None, help="an earlier run of the same version to compare policies with")
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    records = json.loads((folder / "drift.json").read_text())
    summary = {}
    for recipe in ("optimal", "other_starts", "mixed"):
        for step in sorted({r["step"] for r in records}):
            rows = [r for r in records if r["recipe"] == recipe and r["step"] == step]
            summary[f"{recipe} {step}"] = {name: float(np.mean([r[name] for r in rows if r[name] is not None]))
                                           for name in NAMES}
            summary[f"{recipe} {step}"]["runs"] = len(rows)
    out = {"summary": summary}
    if args.against:
        name = "methods_single_critical_noise10__version_c3.npz"
        with np.load(folder / name) as a, np.load(HERE / "results" / args.against / name) as b:
            out["largest_difference_from_the_earlier_run"] = float(np.abs(a["commands"] - b["commands"]).max())
    (folder / "drift_summary.json").write_text(json.dumps(out, indent=1))
    for key, row in summary.items():
        print(f"{key:20s} flagged next: noise {100 * row['next_commands_flagged_with_noise']:5.1f}% clean "
              f"{100 * row['next_commands_flagged_without_noise']:5.1f}% | score noise-clean "
              f"{row['score_with_noise_minus_without']:+7.3f} | charge noise {row['charge_at_next_command_with_noise']:.3f} "
              f"clean {row['charge_at_next_command_without_noise']:.3f} | compared flagged "
              f"{100 * row['comparison_commands_flagged']:5.1f}% at score {row['score_at_flagged_comparison_commands']:+7.2f} | "
              f"error: recorded {row['error_at_recorded_commands']:+7.3f} next clean "
              f"{row['error_at_next_command_without_noise']:+7.3f} next noisy {row['error_at_next_command_with_noise']:+7.3f} "
              f"(true {row['true_value_at_recorded_commands']:.2f}) | unit {row['unit']:.3f} cap {row['cap']:.2f}")
    if args.against:
        print("largest difference in any preference from", args.against, "=",
              out["largest_difference_from_the_earlier_run"])


if __name__ == "__main__":
    main()
