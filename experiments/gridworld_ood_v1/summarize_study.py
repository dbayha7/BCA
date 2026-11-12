"""Compact tables from an evaluated study: outcomes, gates, and what the width and support score did.

    python -I -B summarize_study.py --label dev_v3

Reads records.json and gates.json written by evaluate_study.py; writes summary.md beside them.
Every number is a plain mean over the runs named in the column or row. No learner code.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluation as ev  # noqa: E402
from evaluate_study import VARIANT_NAMES, family, method_name, table  # noqa: E402

ARMS = ("complete", "targeted", "placebo")
GATES = ("G2_width_responds", "G3_more_than_decisiveness", "G4_specific_to_the_gap", "G5_no_wider_harm")


def mean(records, getter):
    values = [getter(r) for r in records]
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def show(value, kind="number"):
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%" if kind == "percent" else f"{value:.2f}"


def summarize(folder):
    records = json.loads((folder / "records.json").read_text())
    gates = json.loads((folder / "gates.json").read_text())
    by = defaultdict(list)
    for r in records:
        by[(r["setting"], r["method"], r["arm"])].append(r)
    settings = sorted({k[0] for k in by}, key=lambda t: (t != "honeypot", t))
    n_data, n_train = len({r["data_seed"] for r in records}), len({r["train_seed"] for r in records})
    per_setting = len({(r["data_seed"], r["train_seed"]) for r in records if r["setting"] == settings[0]})
    doc = ["# Study summary\n", f"Every entry is a mean over the {per_setting} runs of its setting, method and data arm "
           f"({n_data} dataset seeds and {n_train} training seeds in this folder). Written by `summarize_study.py` "
           "from `records.json` and `gates.json`.\n"]
    for setting in settings:
        variants = [v for v in ev.CONTROL_OF if (setting, v, "complete") in by]
        honeypot = family(setting) == "honeypot"
        outcome = "H: episodes ending in hazard 34 (lower is better)" if honeypot else \
            "X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it)"
        doc.append(f"## {setting}\n")
        doc.append(f"Primary outcome, {outcome}.\n")
        methods = list(dict.fromkeys(
            ["baseline"] + [m for v in variants for m in (v, ev.CONTROL_OF[v], f"sharpened:{v}")]))
        doc.append(table(["Method", *ARMS], [[method_name(m)] + [
            show(mean(by[(setting, m, arm)], lambda r: r["primary_outcome"]), "percent") for arm in ARMS] for m in methods]))
        doc.append("Executed probability of the withheld move at the decision cells.\n")
        doc.append(table(["Method", *ARMS], [[method_name(m)] + [
            show(mean(by[(setting, m, arm)], lambda r: r["probability_readout"]), "percent") for arm in ARMS] for m in methods]))
        for title, name in (("Average true value of an episode, over all start cells (not a gate).", "mean_value_all_starts"),
                            ("Average true value of an episode starting at a decision cell (not a gate).",
                             "mean_value_decision_cells")):
            doc.append(title + "\n")
            doc.append(table(["Method", *ARMS], [[method_name(m)] + [
                show(mean(by[(setting, m, arm)], lambda r: r[name])) for arm in ARMS] for m in methods]))
        if setting in gates and gates[setting].get("methods"):
            g1 = gates[setting]["G1_gap_misleads_the_baseline"]
            doc.append(f"G1 (the gap misleads the baseline): {g1['datasets_passing']} of 3 datasets.\n")
            doc.append(table(["Method", "G2 width responds", "G3 beats both controls", "G4 specific to the gap",
                              "G5 no wider harm"],
                             [[VARIANT_NAMES[v]] + [f"{'PASS' if g[n]['passed'] else 'FAIL'} {g[n]['datasets_passing']}/3"
                                                    for n in GATES] for v, g in gates[setting]["methods"].items()]))
        doc.append("What the width and the support score did. \"Target\" is the command that executes the withheld "
                   "move with certainty; \"best\" is the truly best corner command at the same cell.\n")
        rows = []
        for m in ["baseline", *variants]:
            for arm in ARMS:
                recs = by[(setting, m, arm)]
                rows.append([method_name(m), arm,
                             show(mean(recs, lambda r: r["target_move_rows"]["share_scored_unsupported"]), "percent"),
                             show(mean(recs, lambda r: r["audit_bank"]["share_scored_unsupported"]), "percent"),
                             show(mean(recs, lambda r: r.get("decades_below_floor_at_target"))),
                             show(mean(recs, lambda r: r.get("width_at_target"))),
                             show(mean(recs, lambda r: r.get("penalty_at_target"))),
                             show(mean(recs, lambda r: r.get("penalty_at_best"))),
                             show(mean(recs, lambda r: r.get("critic_error_at_target"))),
                             show(mean(recs, lambda r: r["audit_bank"]["coverage"]), "percent"),
                             show(mean(recs, lambda r: float(r["reference_ready"])), "percent")])
        doc.append(table(["Method", "Data arm", "Withheld-move rows scored unsupported", "Fresh ordinary rows scored unsupported",
                          "Factors of ten below the floor, target", "Width used, target", "Penalty, target",
                          "Penalty, best", "Critic error, target", "Coverage on fresh rows", "Runs with a usable width"], rows))
    return "\n".join(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    text = summarize(folder)
    (folder / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
