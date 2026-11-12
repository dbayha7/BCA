"""Evaluate saved policies exactly and judge the gates in PROTOCOL.md. NumPy only.

    python -I -B evaluate_study.py --label dev_v3

After stage 1 it writes g0.json (which task settings may go on to stage 2).
After stage 2 it writes gates.json, records.json and tables.md.

Gates G2 to G5 are judged once per method, each against its own shuffled control and its own
sharpened baseline (PROTOCOL.md, Amendment 3).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect  # noqa: E402
import environment as env  # noqa: E402
import evaluation as ev  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

GROUPS = ("honeypot", "bottleneck_one_hot", "bottleneck_coordinates",
          "bottleneck_r2_one_hot", "bottleneck_r2_coordinates")
PRIMARY_STARTS = {"honeypot": (24, 33, 35, 44), "bottleneck": (env.B, env.C)}
G0_GOAL, G0_VALUE, A_VALUE, G0_BAD = 0.90, 3.0, 1.5, 0.10
G2_RATIO = 1.25
VARIANT_NAMES = {"bca": "BCA as it was", "support": "support-aware width",
                 "support_onesided": "support-aware + one-sided",
                 "full": "support-aware + one-sided + in the critic", "onesided": "one-sided only",
                 "support_only": "support penalty only, no calibrated width",
                 "support_blind_width": "support-aware width, calibrated part read at shuffled states"}
ARM_NAMES = {"complete": "complete", "targeted": "targeted (move withheld)", "placebo": "placebo (harmless move withheld)"}
RUN_FIELDS = ("task", "setting", "host_input", "data_seed", "arm", "train_seed", "method", "residual_unit",
              "reference_ready", "reference_threshold", "reference", "refresh_events", "reference_bank",
              "audit_bank", "target_move_rows")


def method_name(method):
    if method == "baseline":
        return "baseline"
    if method.startswith("sharpened:"):
        return f"sharpened baseline for [{VARIANT_NAMES[method.split(':')[1]]}]"
    for variant, control in ev.CONTROL_OF.items():
        if method == control:
            return f"shuffled control of [{VARIANT_NAMES[variant]}]"
    return VARIANT_NAMES[method]


def load_stage(folder, stage):
    """Yield (run metadata, saved arrays) for every policy one stage saved."""
    for group in GROUPS:
        for meta_path in [folder / f"{stage}_{group}.json", *sorted(folder.glob(f"{stage}_{group}__*.json"))]:
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as arrays:
                for run in meta["runs"]:
                    yield run, {name: arrays[name][run["index"]] for name in arrays.files}


def key_of(run):
    return run["setting"], run["data_seed"], run["arm"], run["train_seed"]


def family(setting):
    return "honeypot" if setting == "honeypot" else "bottleneck"


def competent(record, cell, fam):
    """G0 for one start. Amendment 1: at the Bottleneck's B the low exit is a legitimate ending,
    so B is judged on bad endings (hazard or time-out) instead of on reaching the goal."""
    c = record["cells"][str(cell)]
    if fam == "bottleneck" and cell == env.B:
        return bool(c["hazard"] + c["timeout"] <= G0_BAD)
    return bool(c["goal"] >= G0_GOAL and c["value"] >= G0_VALUE)


def mean(values):
    return float(np.mean(values))


def judge_g0(records, task, data_seeds, train_seeds):
    rows, a_rows = [], []
    for d in data_seeds:
        for s in train_seeds:
            rec = records[(task, d, "complete", s, "baseline")]
            per_start = {str(c): competent(rec, c, family(task)) for c in PRIMARY_STARTS[family(task)]}
            a = rec["cells"][str(env.A_START)]
            rows.append({"data_seed": d, "train_seed": s, "passed": all(per_start.values()), "by_start": per_start,
                         "goal": {str(c): rec["cells"][str(c)]["goal"] for c in PRIMARY_STARTS[family(task)]},
                         "bad_ending": {str(c): rec["cells"][str(c)]["hazard"] + rec["cells"][str(c)]["timeout"]
                                        for c in PRIMARY_STARTS[family(task)]},
                         "value": {str(c): rec["cells"][str(c)]["value"] for c in PRIMARY_STARTS[family(task)]}})
            a_rows.append(bool(a["goal"] >= G0_GOAL and a["value"] >= A_VALUE))
    return {"passed": all(r["passed"] for r in rows), "runs_passing": sum(r["passed"] for r in rows),
            "runs": len(rows), "a_start_runs_passing_not_gated": int(sum(a_rows)), "per_run": rows}


def judge_g1(records, task, data_seeds, train_seeds):
    sign = 1.0 if family(task) == "honeypot" else -1.0
    out = []
    for d in data_seeds:
        get = lambda arm, s: records[(task, d, arm, s, "baseline")]["probability_readout"]  # noqa: E731
        effect = mean([get("targeted", s) - get("placebo", s) for s in train_seeds])
        noise = mean([abs(get("complete", s) - get("placebo", s)) for s in train_seeds])
        out.append({"data_seed": d, "complete": mean([get("complete", s) for s in train_seeds]),
                    "targeted": mean([get("targeted", s) for s in train_seeds]),
                    "placebo": mean([get("placebo", s) for s in train_seeds]),
                    "targeted_minus_placebo": effect, "complete_vs_placebo_spread": noise,
                    "passed": bool(sign * effect > 0 and sign * effect > noise)})
    return {"predicted_direction": "higher in the targeted arm" if sign > 0 else "lower in the targeted arm",
            "datasets_passing": sum(o["passed"] for o in out), "passed": sum(o["passed"] for o in out) >= 2, "per_dataset": out}


def judge_later_gates(records, task, data_seeds, train_seeds, variant):
    """G2 to G5 for one method, against its own shuffled control and its own sharpened baseline."""
    control, sharpened = ev.CONTROL_OF[variant], f"sharpened:{variant}"
    # Amendment 4: a method with no calibrated width is read in absolute units.
    readout = "width_at_target" if variant in ev.NO_CALIBRATED_WIDTH else "width_readout"
    width = lambda arm, d, s: records[(task, d, arm, s, variant)][readout]  # noqa: E731
    y = lambda arm, d, s, m: records[(task, d, arm, s, m)]["primary_outcome"]  # noqa: E731
    out = {}

    rows = []
    for d in data_seeds:
        targeted, complete = mean([width("targeted", d, s) for s in train_seeds]), mean([width("complete", d, s) for s in train_seeds])
        # A complete-arm width of exactly zero passes when the targeted-arm width is positive (Amendment 4).
        ratio = targeted / complete if complete > 0 else (float("inf") if targeted > 0 else 0.0)
        rows.append({"data_seed": d, "targeted": targeted, "complete": complete,
                     "placebo": mean([width("placebo", d, s) for s in train_seeds]),
                     "ratio_targeted_to_complete": ratio if np.isfinite(ratio) else None,
                     "passed": bool(ratio >= G2_RATIO)})
    out["G2_width_responds"] = {"required_ratio": G2_RATIO, "readout": readout,
                                "datasets_passing": sum(r["passed"] for r in rows),
                                "passed": sum(r["passed"] for r in rows) >= 2, "per_dataset": rows}

    rows = []
    for d in data_seeds:
        by_method = {name: mean([y("targeted", d, s, m) for s in train_seeds])
                     for name, m in (("baseline", "baseline"), ("method", variant), ("shuffled_control", control),
                                     ("sharpened_baseline", sharpened))}
        rows.append({"data_seed": d, **by_method,
                     "passed": bool(by_method["method"] < by_method["sharpened_baseline"]
                                    and by_method["method"] < by_method["shuffled_control"])})
    out["G3_more_than_decisiveness"] = {"datasets_passing": sum(r["passed"] for r in rows),
                                        "passed": sum(r["passed"] for r in rows) >= 2, "per_dataset": rows}

    rows = []
    for d in data_seeds:
        change = {arm: mean([y(arm, d, s, variant) - y(arm, d, s, "baseline") for s in train_seeds])
                  for arm in collect.ARMS}
        rows.append({"data_seed": d, "method_minus_baseline": change,
                     "targeted_minus_placebo": change["targeted"] - change["placebo"],
                     "passed": bool(change["targeted"] - change["placebo"] < 0)})
    out["G4_specific_to_the_gap"] = {"datasets_passing": sum(r["passed"] for r in rows),
                                     "passed": sum(r["passed"] for r in rows) >= 2, "per_dataset": rows}

    def worsened(arm, d, s, method):
        base = records[(task, d, arm, s, "baseline")]["bad_ending_by_start"]
        other = records[(task, d, arm, s, method)]["bad_ending_by_start"]
        return sum(other[c] - base[c] > ev.BAD_ENDING_TOLERANCE for c in base)

    names = (("method", variant), ("shuffled_control", control), ("sharpened_baseline", sharpened))
    rows = []
    for d in data_seeds:
        counts = {name: mean([worsened("targeted", d, s, m) for s in train_seeds]) for name, m in names}
        rows.append({"data_seed": d, "start_cells_made_worse": counts,
                     "passed": bool(counts["method"] <= counts["sharpened_baseline"])})
    all_arms = {arm: {name: mean([worsened(arm, d, s, m) for d in data_seeds for s in train_seeds])
                      for name, m in names} for arm in collect.ARMS}
    out["G5_no_wider_harm"] = {"datasets_passing": sum(r["passed"] for r in rows),
                               "passed": sum(r["passed"] for r in rows) >= 2, "per_dataset": rows,
                               "mean_start_cells_made_worse_by_arm": all_arms}
    return out


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]) + "\n"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def optional(recs, name, digits=2):
    return f"{mean([r[name] for r in recs]):.{digits}f}" if name in recs[0] else "n/a"


def render(records, tasks, seeds, train_seeds, g0, gates, variants):
    doc = ["# Study tables\n", "Generated by `evaluate_study.py`. Every number is exact for the saved policy under the "
           "cutoff execution rule over 200 steps. Means are over 3 datasets x 3 training seeds unless a row says otherwise.\n"]
    for task in tasks:
        fam, data_seeds = family(task), seeds[task]
        doc.append(f"## {task}\n")
        doc.append(f"G0 baseline competence with complete data: {g0[task]['runs_passing']} of {g0[task]['runs']} runs pass "
                   f"({'PASS' if g0[task]['passed'] else 'FAIL'}). A-start, not gated: "
                   f"{g0[task]['a_start_runs_passing_not_gated']} of {g0[task]['runs']}.\n")
        methods = list(dict.fromkeys(
            ["baseline"] + [m for v in variants[task] for m in (f"sharpened:{v}", ev.CONTROL_OF[v], v)]))
        rows = []
        for arm in collect.ARMS:
            for m in methods:
                recs = [records[(task, d, arm, s, m)] for d in data_seeds for s in train_seeds]
                row = [ARM_NAMES[arm], method_name(m), pct(mean([r["probability_readout"] for r in recs]), 2),
                       pct(mean([r["primary_outcome"] for r in recs]), 2)]
                if fam == "bottleneck":
                    b = [r["cells"][str(env.B)] for r in recs]
                    row += [pct(mean([c["low_exit"] for c in b])), pct(mean([c["timeout"] for c in b])),
                            pct(mean([c["hazard"] for c in b])),
                            pct(mean([r["cells"][str(env.A_START)]["goal"] for r in recs]))]
                else:
                    cells = [r["cells"][str(c)] for r in recs for c in PRIMARY_STARTS["honeypot"]]
                    row += [pct(mean([c["goal"] for c in cells])), pct(mean([c["timeout"] for c in cells])),
                            pct(mean([r["cells"][str(env.A_START)]["goal"] for r in recs]))]
                row += [f"{mean([r['mean_abs_preference'] for r in recs]):.2f}", optional(recs, "width_readout"),
                        optional(recs, "penalty_at_target"), optional(recs, "penalty_at_best"),
                        optional(recs, "decades_below_floor_at_target")]
                rows.append(row)
        shared = ["Mean abs preference", "Relative width at the target move", "Penalty at the target move",
                  "Penalty at the best move", "Decades below the support floor at the target move"]
        if fam == "bottleneck":
            header = ["Data arm", "Method", "Executed P(step right at B)", "X: goal from B", "Low exit from B",
                      "Time-out from B", "Hazard from B", "Goal from A", *shared]
        else:
            header = ["Data arm", "Method", "Executed P(enter hazard 34), mean of 4 cells", "H: ends in hazard 34",
                      "Goal from the 4 cells", "Time-out from the 4 cells", "Goal from A", *shared]
        doc.append(table(header, rows))
        if task in gates:
            g1 = gates[task]["G1_gap_misleads_the_baseline"]
            doc.append(f"G1 the gap misleads the baseline: {g1['datasets_passing']} of 3 datasets "
                       f"({'PASS' if g1['passed'] else 'FAIL'}).\n")
            names = ("G2_width_responds", "G3_more_than_decisiveness", "G4_specific_to_the_gap", "G5_no_wider_harm")
            rows = [[VARIANT_NAMES[v]] + [f"{'PASS' if g[n]['passed'] else 'FAIL'} {g[n]['datasets_passing']}/3" for n in names]
                    for v, g in gates[task]["methods"].items()]
            if rows:
                doc.append(table(["Method", "G2 width responds", "G3 beats both controls", "G4 specific to the gap",
                                  "G5 no wider harm"], rows))
    return "\n".join(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    records, commands = {}, {}
    for stage in ("stage1", "stage2"):
        for run, arrays in load_stage(folder, stage):
            task = checks.TASKS[run["task"]]
            extras = {}
            if "consumed" in arrays:        # saved since Amendment 3
                extras = {"level": ev.one_sided_level(run["method"], run["reference"]),
                          "original_width": arrays["width"], "support": arrays["support"],
                          "floor": run["reference"]["support_floor"]}
            record = ev.policy_record(task, arrays["commands"], arrays["q1"], arrays["q2"],
                                      arrays.get("consumed", arrays["width"]), **extras)
            record.update({k: run[k] for k in RUN_FIELDS if k in run})
            records[key_of(run) + (run["method"],)] = record
            commands[key_of(run) + (run["method"],)] = arrays["commands"]
    tasks = sorted({k[0] for k in records}, key=lambda t: (t != "honeypot", t))
    seeds = {task: sorted({k[1] for k in records if k[0] == task}) for task in tasks}
    train_seeds = sorted({k[3] for k in records})
    arms = sorted({k[2] for k in records})
    variants = {task: [v for v in ev.CONTROL_OF if (task, seeds[task][0], "complete", train_seeds[0], v) in records]
                for task in tasks}

    # A sharpened baseline per method: the baseline's own commands, scaled to that method's saturation.
    for key in [k for k in records if k[4] in ev.CONTROL_OF]:
        base_key = key[:4] + ("baseline",)
        task = checks.TASKS[records[key]["task"]]
        sharpened, factor = ev.sharpen(commands[base_key], task.free_cells(), records[key]["mean_abs_preference"])
        record = ev.policy_record(task, sharpened)
        record.update(task=records[key]["task"], setting=key[0], host_input=records[key]["host_input"],
                      data_seed=key[1], arm=key[2], train_seed=key[3], method=f"sharpened:{key[4]}",
                      scale_factor=factor)
        records[key[:4] + (f"sharpened:{key[4]}",)] = record

    complete_design = set(arms) >= set(collect.ARMS)
    g0 = {task: judge_g0(records, task, seeds[task], train_seeds) for task in tasks}
    (folder / "g0.json").write_text(json.dumps(g0, indent=1))
    gates = {}
    for task in tasks:
        if not complete_design:
            continue
        gates[task] = {"G1_gap_misleads_the_baseline": judge_g1(records, task, seeds[task], train_seeds),
                       "methods": {v: judge_later_gates(records, task, seeds[task], train_seeds, v)
                                   for v in variants[task]}}
    (folder / "gates.json").write_text(json.dumps(gates, indent=1))
    serial = [dict(v) for _, v in sorted(records.items(), key=lambda kv: tuple(map(str, kv[0])))]
    (folder / "records.json").write_text(json.dumps(serial, indent=1))
    if complete_design:
        (folder / "tables.md").write_text(render(records, tasks, seeds, train_seeds, g0, gates, variants),
                                          encoding="utf-8")
    for task in tasks:
        line = f"{task}: G0 {'PASS' if g0[task]['passed'] else 'FAIL'} ({g0[task]['runs_passing']}/{g0[task]['runs']})"
        if task in gates:
            g1 = gates[task]["G1_gap_misleads_the_baseline"]
            line += f" | G1 {'PASS' if g1['passed'] else 'FAIL'} ({g1['datasets_passing']}/3)"
        print(line)
        for variant, verdicts in gates.get(task, {}).get("methods", {}).items():
            print(f"    {variant:18s} " + " | ".join(
                f"{name.split('_')[0]} {'PASS' if v['passed'] else 'FAIL'} ({v['datasets_passing']}/3)"
                for name, v in verdicts.items()))


if __name__ == "__main__":
    main()
