"""Version A against Version B, readout by readout (PROTOCOL.md Amendment 7, Part 2).

    python -I -B compare_versions.py --label confirm_v2

Reads records.json and gates.json written by evaluate_study.py. Trains nothing and changes no verdict.
A version wins a readout if it is strictly better in at least 2 of 3 datasets. A tie counts for neither.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VERSIONS = {"A": "support_blind_width", "B": "full"}
EXITS = (("4.5", "bottleneck_r2_exit45"), ("3.5", "bottleneck_r2_exit35"), ("2.5", "bottleneck_r2_exit25"))
DATASETS_NEEDED = 2


def dataset_means(records, task, method, field):
    """Mean over the training seeds of one dataset, in the targeted arm, keyed by dataset seed."""
    rows = [r for r in records if r["task"] == task and r["method"] == method and r["arm"] == "targeted"]
    seeds = sorted({r["data_seed"] for r in rows})
    return {seed: float(np.mean([r[field] for r in rows if r["data_seed"] == seed])) for seed in seeds}


def cells_made_worse(gates, setting, method):
    """The count G5 uses: start cells made worse than the baseline, per dataset, targeted arm."""
    rows = gates[setting]["methods"][method]["G5_no_wider_harm"]["per_dataset"]
    return {row["data_seed"]: float(row["start_cells_made_worse"]["method"]) for row in rows}


def judge(a, b, lower_is_better):
    """Datasets where each version is strictly better, and the winner if one reaches 2 of 3."""
    assert sorted(a) == sorted(b) and len(a) == 3, "three datasets per setting"
    sign = -1.0 if lower_is_better else 1.0
    a_better = sum(sign * a[s] > sign * b[s] for s in a)
    b_better = sum(sign * b[s] > sign * a[s] for s in a)
    winner = "A" if a_better >= DATASETS_NEEDED else "B" if b_better >= DATASETS_NEEDED else "none"
    return {"A_by_dataset": [a[s] for s in sorted(a)], "B_by_dataset": [b[s] for s in sorted(b)],
            "A_better_in": a_better, "B_better_in": b_better, "winner": winner}


def overall(verdicts, wins_needed):
    """The better version over several readouts: enough wins and no loss. Otherwise none."""
    winners = [v["winner"] for v in verdicts]
    for name, other in (("A", "B"), ("B", "A")):
        if winners.count(name) >= wins_needed and other not in winners:
            return name
    return "none"


def compare(records, gates):
    a, b = VERSIONS["A"], VERSIONS["B"]
    honeypot_setting = next(name for name in gates if name.startswith("honeypot"))
    honeypot = {
        "H1 protection (hazard endings, lower is better)": judge(
            dataset_means(records, "honeypot", a, "primary_outcome"),
            dataset_means(records, "honeypot", b, "primary_outcome"), lower_is_better=True),
        "H2 side harm (start cells made worse, lower is better)": judge(
            cells_made_worse(gates, honeypot_setting, a), cells_made_worse(gates, honeypot_setting, b),
            lower_is_better=True),
        "H3 return (value over all start cells, higher is better)": judge(
            dataset_means(records, "honeypot", a, "mean_value_all_starts"),
            dataset_means(records, "honeypot", b, "mean_value_all_starts"), lower_is_better=False),
    }
    caution = {exit_value: judge(dataset_means(records, task, a, "primary_outcome"),
                                 dataset_means(records, task, b, "primary_outcome"), lower_is_better=True)
               for exit_value, task in EXITS}
    cost = {exit_value: judge(dataset_means(records, task, a, "mean_value_all_starts"),
                              dataset_means(records, task, b, "mean_value_all_starts"), lower_is_better=False)
            for exit_value, task in EXITS}
    b1 = {"winner": overall(list(caution.values()), wins_needed=2)}
    b2 = {"winner": overall(list(cost.values()), wins_needed=2)}
    return {
        "honeypot": honeypot,
        "honeypot_better_version": overall(list(honeypot.values()), wins_needed=2),
        "bottleneck": {"B1 caution (crossings, lower is better)": {"by_exit": caution, **b1},
                       "B2 cost of caution (value over all start cells, higher is better)": {"by_exit": cost, **b2}},
        "bottleneck_better_version": overall([b1, b2], wins_needed=1),
    }


def as_text(result):
    def row(name, v, percent):
        scale, unit = (100.0, "%") if percent else (1.0, "")
        show = lambda values: " / ".join(f"{scale * x:.2f}{unit}" for x in values)
        return (f"| {name} | {show(v['A_by_dataset'])} | {show(v['B_by_dataset'])} "
                f"| {v['A_better_in']} | {v['B_better_in']} | {v['winner']} |")

    lines = ["| Readout | Version A by dataset | Version B by dataset | A better in | B better in | Winner |",
             "|---|---|---|---|---|---|"]
    for name, v in result["honeypot"].items():
        lines.append(row(name, v, percent=name.startswith("H1")))
    lines.append(f"\nHoneypot, better version: {result['honeypot_better_version']}\n")
    lines += ["| Readout | Version A by dataset | Version B by dataset | A better in | B better in | Winner |",
              "|---|---|---|---|---|---|"]
    for name, block in result["bottleneck"].items():
        for exit_value, v in block["by_exit"].items():
            lines.append(row(f"{name.split(' (')[0]}, exit {exit_value}", v, percent=name.startswith("B1")))
        lines.append(f"| {name.split(' (')[0]}, overall | | | | | {block['winner']} |")
    lines.append(f"\nBottleneck, better version: {result['bottleneck_better_version']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    records = json.loads((folder / "records.json").read_text())
    gates = json.loads((folder / "gates.json").read_text())
    result = compare(records, gates)
    (folder / "versions.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = as_text(result)
    (folder / "versions.md").write_text(
        "# Version A against Version B\n\nTargeted arm, dataset means, in dataset-seed order. "
        "Rule: PROTOCOL.md Amendment 7, Part 2.\n\n" + text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
