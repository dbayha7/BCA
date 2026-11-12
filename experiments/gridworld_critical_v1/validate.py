"""Evaluate the saved baselines exactly and judge the validation gates of PROTOCOL.md. NumPy only.

    python -I -B validate.py --label dev_validation

Writes validation.json, records.json and validation_tables.md into results/<label>/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import critical_checks as checks  # noqa: E402
import critical_collect as C  # noqa: E402
import critical_env as E  # noqa: E402

STARTS = {"start": E.START, "above": E.ABOVE, "critical": E.CRITICAL}
GOAL_MARGIN, VALUE_SHARE, ERROR_MARK = 0.10, 0.60, 0.05
FATAL = (E.LEFT, E.RIGHT)


def mean(values):
    return float(np.mean(values))


def best_possible(task):
    best, value = task.best_policy()
    out = task.outcomes(E.policy_from_moves(best))
    return {name: {"goal": float(out["goal"][cell]), "value": float(value[cell])} for name, cell in STARTS.items()}


def policy_record(task, commands, q1):
    """Everything the gates need about one policy."""
    free = task.free_cells()
    value = task.value(commands)
    out = task.outcomes(commands, crossing=(E.CRITICAL, E.BELOW))
    chance = task.command_probabilities(commands[E.CRITICAL])
    true_q = task.command_values(value)[E.CRITICAL]      # command a move once here, then follow this policy
    error = q1[E.CRITICAL] - true_q
    return {
        "starts": {name: {"goal": float(out["goal"][cell]), "lava": float(out["lava"][cell]),
                          "timeout": float(out["timeout"][cell]), "value": float(value[cell])}
                   for name, cell in STARTS.items()},
        "command_chance_at_critical": dict(zip(E.MOVE_NAMES, (float(x) for x in chance))),
        "chance_fatal": float(chance[list(FATAL)].sum()), "chance_down": float(chance[E.DOWN]),
        "critic_at_critical": dict(zip(E.MOVE_NAMES, (float(x) for x in q1[E.CRITICAL]))),
        "true_value_at_critical": dict(zip(E.MOVE_NAMES, (float(x) for x in true_q))),
        "critic_error_fatal": float(error[list(FATAL)].mean()), "critic_error_down": float(error[E.DOWN]),
        "flank_lava_from_critical": float(out["lava_by_cell"][E.LAVA_LEFT][E.CRITICAL]
                                          + out["lava_by_cell"][E.LAVA_RIGHT][E.CRITICAL]),
        "went_through_from_critical": float(out["crossed"][E.CRITICAL]),
        "stalled_start_cells": int((out["timeout"][free] > 0.5).sum()),
        "mean_abs_preference": float(np.abs(commands[free]).mean()),
        "mean_goal_all_starts": float(out["goal"][free].mean()),
    }


def competence(records, task_name, data_seeds, train_seeds, best):
    """V0: every complete-data run, from each of the three starts, is within 10 points of the best
    possible goal rate and has at least 60% of the best possible value."""
    rows = []
    for d in data_seeds:
        for s in train_seeds:
            r = records[(task_name, d, "complete", s)]
            by_start = {name: bool(r["starts"][name]["goal"] >= best[name]["goal"] - GOAL_MARGIN
                                   and r["starts"][name]["value"] >= VALUE_SHARE * best[name]["value"]) for name in STARTS}
            rows.append({"data_seed": d, "train_seed": s, "passed": all(by_start.values()), "by_start": by_start,
                         "goal": {name: r["starts"][name]["goal"] for name in STARTS},
                         "value": {name: r["starts"][name]["value"] for name in STARTS}})
    return {"passed": all(r["passed"] for r in rows), "runs_passing": sum(r["passed"] for r in rows),
            "runs": len(rows), "best_possible": best, "per_run": rows}


def judge(records, task_name, data_seeds, train_seeds, best):
    get = lambda d, arm, s: records[(task_name, d, arm, s)]  # noqa: E731
    gates = {"V0_baseline_competence": competence(records, task_name, data_seeds, train_seeds, best)}

    def estimation(arm, name, sign):
        rows = []
        for d in data_seeds:
            gap = mean([get(d, arm, s)[name] for s in train_seeds])
            full = mean([get(d, "complete", s)[name] for s in train_seeds])
            rows.append({"data_seed": d, "left_out_arm": gap, "complete_arm": full,
                         "placebo_arm": mean([get(d, "placebo", s)[name] for s in train_seeds]),
                         "passed": bool(sign * gap >= ERROR_MARK and sign * (gap - full) >= ERROR_MARK)})
        return {"datasets_passing": sum(r["passed"] for r in rows), "passed": sum(r["passed"] for r in rows) >= 2,
                "per_dataset": rows}

    def behavior(arm, name, sign):
        rows = []
        for d in data_seeds:
            gap = mean([get(d, arm, s)[name] for s in train_seeds])
            placebo = mean([get(d, "placebo", s)[name] for s in train_seeds])
            spread = mean([abs(get(d, "complete", s)[name] - get(d, "placebo", s)[name]) for s in train_seeds])
            rows.append({"data_seed": d, "left_out_arm": gap, "placebo_arm": placebo,
                         "complete_arm": mean([get(d, "complete", s)[name] for s in train_seeds]),
                         "complete_vs_placebo_spread": spread,
                         "passed": bool(sign * (gap - placebo) > 0 and sign * (gap - placebo) > spread)})
        return {"datasets_passing": sum(r["passed"] for r in rows), "passed": sum(r["passed"] for r in rows) >= 2,
                "per_dataset": rows}

    gates["V1_over_critic_overestimates"] = estimation("over", "critic_error_fatal", +1.0)
    gates["V1_under_critic_underestimates"] = estimation("under", "critic_error_down", -1.0)
    gates["V2_over_gap_misleads_behavior"] = behavior("over", "chance_fatal", +1.0)
    gates["V2_under_gap_misleads_behavior"] = behavior("under", "chance_down", -1.0)
    return gates


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]) + "\n"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def render(records, gates, tasks, data_seeds, train_seeds):
    doc = ["# Validation tables\n", "Generated by `validate.py`. Every number is exact for the saved policy, over 200 steps. "
           "Means are over 3 datasets x 3 training seeds unless a row says otherwise.\n"]
    for task_name in tasks:
        g = gates[task_name]
        best = g["V0_baseline_competence"]["best_possible"]
        doc.append(f"## {task_name}\n")
        doc.append("Best any policy can do: " + ", ".join(
            f"from {name} goal {pct(best[name]['goal'])}, value {best[name]['value']:.3f}" for name in STARTS) + ".\n")
        rows = []
        for arm in C.ARMS:
            recs = [records[(task_name, d, arm, s)] for d in data_seeds for s in train_seeds]
            rows.append([arm, pct(mean([r["starts"]["start"]["goal"] for r in recs])),
                         pct(mean([r["starts"]["critical"]["goal"] for r in recs])),
                         f"{mean([r['starts']['critical']['value'] for r in recs]):.3f}",
                         pct(mean([r["chance_fatal"] for r in recs]), 2), pct(mean([r["chance_down"] for r in recs]), 2),
                         f"{mean([r['critic_error_fatal'] for r in recs]):+.3f}",
                         f"{mean([r['critic_error_down'] for r in recs]):+.3f}",
                         pct(mean([r["flank_lava_from_critical"] for r in recs]), 2),
                         pct(mean([r["went_through_from_critical"] for r in recs])),
                         pct(mean([r["starts"]["critical"]["timeout"] for r in recs])),
                         sum(r["stalled_start_cells"] for r in recs)])
        doc.append(table(["Data arm", "Goal from start", "Goal from critical cell", "Value at critical cell",
                          "Commands a fatal move", "Commands down", "Critic error, fatal moves", "Critic error, down",
                          "Ends in flank lava", "Goes through", "Times out", "Stalled start cells (all runs)"], rows))
        v0 = g["V0_baseline_competence"]
        rows = [["V0 baseline competence", f"{v0['runs_passing']} of {v0['runs']} runs", "PASS" if v0["passed"] else "FAIL"]]
        for name in list(g)[1:]:
            rows.append([name.replace("_", " "), f"{g[name]['datasets_passing']} of 3 datasets", "PASS" if g[name]["passed"] else "FAIL"])
        doc.append(table(["Gate", "Passing", "Verdict"], rows))
    return "\n".join(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    records, meta = {}, {}
    for path in sorted(folder.glob("baselines_*.json")):
        saved = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            for run in saved["runs"]:
                task = checks.TASKS[run["task"]]
                key = (run["task"], run["data_seed"], run["arm"], run["train_seed"])
                records[key] = policy_record(task, arrays["commands"][run["index"]], arrays["q1"][run["index"]])
                meta[key] = {k: run[k] for k in ("task", "noise", "sharpness", "data_seed", "arm", "train_seed", "method",
                                                 "reference_ready", "audit_bank") if k in run}
    tasks = sorted({k[0] for k in records}, reverse=True)
    data_seeds, train_seeds = sorted({k[1] for k in records}), sorted({k[3] for k in records})
    arms = {k[2] for k in records}
    gates, pilot = {}, {}
    if arms >= set(C.ARMS) and len(data_seeds) >= 3:
        gates = {t: judge(records, t, data_seeds, train_seeds, best_possible(checks.TASKS[t])) for t in tasks}
        (folder / "validation_tables.md").write_text(render(records, gates, tasks, data_seeds, train_seeds), encoding="utf-8")
    (folder / "validation.json").write_text(json.dumps(gates, indent=1))
    (folder / "records.json").write_text(json.dumps(
        [{**meta[k], **records[k]} for k in sorted(records, key=lambda k: tuple(map(str, k)))], indent=1))
    for t in tasks:
        if t in gates:
            g = gates[t]
            print(t + ": " + " | ".join(
                f"{name.split('_')[0]}{'-' + name.split('_')[1] if name.startswith('V1') or name.startswith('V2') else ''} "
                f"{'PASS' if v['passed'] else 'FAIL'} ({v.get('runs_passing', v.get('datasets_passing'))}/{v.get('runs', 3)})"
                for name, v in g.items()))
        elif "complete" in arms:            # a pilot or a smoke run: competence only
            v0 = competence(records, t, data_seeds, train_seeds, best_possible(checks.TASKS[t]))
            pilot[t] = v0
            recs = [records[(t, d, "complete", s)] for d in data_seeds for s in train_seeds]
            print(f"{t}: V0 {v0['runs_passing']}/{v0['runs']} runs pass | goal from start "
                  f"{100 * mean([r['starts']['start']['goal'] for r in recs]):.1f}% (best {100 * v0['best_possible']['start']['goal']:.1f}%)"
                  f" | value from start {mean([r['starts']['start']['value'] for r in recs]):.3f} (best {v0['best_possible']['start']['value']:.3f})"
                  f" | commands a fatal move at the critical cell {100 * mean([r['chance_fatal'] for r in recs]):.2f}%"
                  f" | mean command strength {mean([r['mean_abs_preference'] for r in recs]):.2f}")
    if pilot:
        (folder / "competence.json").write_text(json.dumps(pilot, indent=1))


if __name__ == "__main__":
    main()
