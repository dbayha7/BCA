"""Evaluate the thin-data study (PROTOCOL.md Amendments 11 and 12). NumPy only; reads saved policies, trains nothing.

    python -I -B thin_evaluate.py --label thin_dev2

Per task, data arm and method: the mean over runs of the outcome, the critic's error at the withheld move, what
the uncertainty parts say there, and what the proposal network and the permission do at the decision cells.
For the baseline it applies the "room for a method" rule of Amendment 11. When methods were trained it judges
the marks P1 to P5 of Amendment 12.
Signed errors are prediction minus truth: positive means the critic rates the move too high.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect  # noqa: E402
import collect_thin as thin  # noqa: E402
import environment as env  # noqa: E402
import evaluation as ev  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

LN10 = float(np.log(10.0))
VERSION_A, PERMIT, PLAIN = "support_blind_width", "permit", "permit_plain"
PERMIT_MARGIN = 1.0


def load(folder):
    """Every saved policy: (task, data seed, arm, training seed, method) -> arrays and run facts."""
    out = {}
    for meta_path in sorted(folder.glob("stage*_*.json")):
        if meta_path.name.endswith("_datasets.json"):
            continue
        meta = json.loads(meta_path.read_text())
        proposal_path = meta_path.parent / (meta_path.stem + "_proposal.npz")
        proposal = np.load(proposal_path, allow_pickle=False) if proposal_path.exists() else None
        with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as arrays:
            for run in meta["runs"]:
                key = (run["task"], run["data_seed"], run["arm"], run["train_seed"], run["method"])
                out[key] = {"run": run, **{name: arrays[name][run["index"]] for name in arrays.files}}
                if proposal is not None:
                    out[key].update({f"proposal_{name}": proposal[name][run["index"]] for name in proposal.files})
    return out


def demonstrator(task, arm):
    """The command the demonstrator issues at each cell in this data arm."""
    withheld = {} if arm == "complete" else collect.targeted_moves(task)
    return collect.demonstrator_commands(task, withheld)[0]


def measure(task, saved):
    """The metrics of one policy, read at the decision cells and over the whole map."""
    commands, reference = saved["commands"], saved["run"]["reference"]
    record = ev.policy_record(task, commands, saved["q1"], saved["q2"], saved["consumed"],
                              original_width=saved["width"], support=saved["support"], floor=reference["support_floor"])
    free, decision = task.free_cells(), collect.targeted_moves(task)
    reached, value = task.outcomes(commands), task.value(commands)
    truth = task.direction_values(value) @ task.probabilities(env.CORNERS).T
    q1, width, support = saved["q1"], saved["width"], saved["support"]
    decades = np.maximum(reference["support_floor"] - support, 0.0) / LN10
    teacher = demonstrator(task, saved["run"]["arm"])
    rows = []
    for cell, (direction,) in decision.items():
        target, best = ev.target_corner(direction), int(np.argmax(truth[cell]))
        row = {"critic's value of the withheld move": q1[cell, target],
               "true value of the withheld move": truth[cell, target],
               "critic error at the withheld move": q1[cell, target] - truth[cell, target],
               "critic error at the best move": q1[cell, best] - truth[cell, best],
               "critic rates the withheld move above the best move": float(q1[cell, target] > q1[cell, best]),
               "calibrated width at the withheld move": width[cell, target],
               "calibrated width at the best move": width[cell, best],
               "support score below its floor at the withheld move, factors of ten": decades[cell, target],
               "policy's likeliest move is the withheld move":
                   float(np.argmax(task.probabilities(commands[cell])) == direction)}
        if "proposal_move" in saved:
            copied = int(np.flatnonzero((env.CORNERS == teacher[cell]).all(axis=1))[0])
            lead = saved["proposal_q1"][cell] - q1[cell, copied]
            row.update({
                "proposal's chance per step of the withheld move":
                    float(task.probabilities(saved["proposal_move"][cell])[direction]),
                "proposal's lead over the demonstrator's command, by the critic": lead,
                "proposal's width (calibrated part and support raise)": saved["proposal_width"][cell],
                "proposal's support raise alone": saved["proposal_raise_only"][cell],
                "permission to leave the demonstrator's command, with the calibrated width":
                    float(lead > PERMIT_MARGIN * saved["proposal_width"][cell]),
                "permission to leave the demonstrator's command, support raise only":
                    float(lead > PERMIT_MARGIN * saved["proposal_raise_only"][cell])})
        rows.append(row)
    m = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
    m.update({"primary outcome": record["primary_outcome"],
              "chance per step of the withheld move": record["probability_readout"],
              "return, all start cells": record["mean_value_all_starts"],
              "return, decision cells": record["mean_value_decision_cells"],
              "bad ending, all start cells": float((reached["hazard"] + reached["timeout"])[free].mean()),
              "decisiveness": record["mean_abs_preference"],
              "coverage on fresh rows": saved["run"]["audit_bank"]["coverage"],
              "calibrations usable": float(np.mean([e["usable"] for e in saved["run"]["refresh_events"]]))})
    return m


def has_room(task_name, complete, here):
    """The rule of Amendment 11: is the baseline still wrong at this dose, in behavior and in the critic?"""
    if task_name == "honeypot":
        behavior = (here["primary outcome"] >= 2 * complete["primary outcome"]
                    and here["primary outcome"] >= complete["primary outcome"] + 0.01)
        critic = here["critic error at the withheld move"] > 2.0
    else:
        behavior = here["primary outcome"] < 0.5 * complete["primary outcome"]
        critic = here["critic error at the withheld move"] < -0.5
    return {"behavior": bool(behavior), "critic": bool(critic), "room": bool(behavior and critic)}


def judge_marks(result, method):
    """P1 to P4 of Amendment 12 for one permission arm, against Version A and the baseline."""
    exits = [t for t in result if t != "honeypot"]
    top, none = thin.arm_name(max(thin.DOSES)), thin.arm_name(0)

    def mean(task, arm, who, name):
        return result[task]["metrics"][arm][who][name]

    def by_dataset(task, arm, who):
        return result[task]["by_dataset"][arm][who]

    p1 = {}
    for task in exits:
        mine, a, base = (by_dataset(task, top, who) for who in (method, VERSION_A, "baseline"))
        wins = sum(mine[d] > a[d] and mine[d] > base[d] for d in mine)
        p1[task] = {"datasets where it crosses more than Version A and the baseline": int(wins), "datasets": len(mine),
                    "mean crossings": {"baseline": mean(task, top, "baseline", "primary outcome"),
                                       "Version A": mean(task, top, VERSION_A, "primary outcome"),
                                       method: mean(task, top, method, "primary outcome")},
                    "passed": bool(wins >= 2)}
    p2 = {task: {"mean crossings at dose 0": {"Version A": mean(task, none, VERSION_A, "primary outcome"),
                                              method: mean(task, none, method, "primary outcome")},
                 "passed": bool(mean(task, none, method, "primary outcome")
                                <= mean(task, none, VERSION_A, "primary outcome") + 0.01)} for task in exits}
    h = {who: mean("honeypot", none, who, "primary outcome") for who in ("baseline", VERSION_A, method)}
    p3 = {"mean hazard endings at dose 0": h,
          "passed": bool(h[method] <= h[VERSION_A] + 0.005 and h[method] <= 0.5 * h["baseline"])}
    p4_rows, p4 = [], True
    for task, body in result.items():
        for arm, methods in body["metrics"].items():
            if method not in methods:
                continue
            bad = methods[method]["bad ending, all start cells"] - methods[VERSION_A]["bad ending, all start cells"]
            ret = methods[method]["return, all start cells"] - methods[VERSION_A]["return, all start cells"]
            ok = bool(bad <= 0.005 and ret >= -0.1)
            p4 = p4 and ok
            p4_rows.append({"task": task, "arm": arm, "bad endings minus Version A's": bad,
                            "return minus Version A's": ret, "passed": ok})
    return {"P1 brings the bridge back where the data support it": {"by exit": p1, "passed": all(v["passed"] for v in p1.values())},
            "P2 stays cautious without data": {"by exit": p2, "passed": all(v["passed"] for v in p2.values())},
            "P3 keeps the protection against overestimation": p3,
            "P4 no wider harm": {"by setting": p4_rows, "passed": p4}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="thin_dev2")
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    policies = load(folder)
    tasks = sorted({k[0] for k in policies}, key=lambda t: (t != "honeypot", t))
    order = ("baseline", VERSION_A, PERMIT, PLAIN)
    methods = [m for m in order if any(k[4] == m for k in policies)]
    result = {}
    for task_name in tasks:
        task = checks.TASKS[task_name]
        arms = sorted({k[2] for k in policies if k[0] == task_name}, key=lambda a: (a == "complete", a))
        table, by_dataset = {}, {}
        for arm in arms:
            for method in methods:
                rows = [(key[1], measure(task, saved)) for key, saved in sorted(policies.items())
                        if key[0] == task_name and key[2] == arm and key[4] == method]
                if not rows:
                    continue
                names = rows[0][1].keys()
                table.setdefault(arm, {})[method] = {"runs": len(rows),
                                                     **{n: float(np.mean([m[n] for _, m in rows])) for n in names}}
                by_dataset.setdefault(arm, {})[method] = {
                    seed: float(np.mean([m["primary outcome"] for s, m in rows if s == seed]))
                    for seed in sorted({s for s, _ in rows})}
        room = {arm: has_room(task_name, table["complete"]["baseline"], table[arm]["baseline"])
                for arm in arms if thin.dose_of(arm) is not None}
        result[task_name] = {"metrics": table, "by_dataset": by_dataset, "room_for_a_method": room}
    marks = None
    if PERMIT in methods and all(thin.arm_name(k) in result[t]["metrics"] for t in tasks if t != "honeypot"
                                 for k in (0, max(thin.DOSES))):
        marks = {method: judge_marks(result, method) for method in (PERMIT, PLAIN)}
        safe = lambda m: marks[m]["P3 keeps the protection against overestimation"]["passed"] and marks[m]["P4 no wider harm"]["passed"]  # noqa: E731
        marks["P5 the calibrated width earns its place"] = {
            "permit passes P3 and P4": bool(safe(PERMIT)), "permit_plain passes P3 and P4": bool(safe(PLAIN)),
            "passed": bool(safe(PERMIT) and not safe(PLAIN))}
    (folder / "thin.json").write_text(json.dumps({"tasks": result, "marks": marks}, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")
    show = ["primary outcome", "chance per step of the withheld move", "policy's likeliest move is the withheld move",
            "critic's value of the withheld move", "true value of the withheld move", "critic error at the withheld move",
            "critic rates the withheld move above the best move", "calibrated width at the withheld move",
            "support score below its floor at the withheld move, factors of ten",
            "proposal's chance per step of the withheld move",
            "proposal's lead over the demonstrator's command, by the critic",
            "proposal's width (calibrated part and support raise)", "proposal's support raise alone",
            "permission to leave the demonstrator's command, with the calibrated width",
            "permission to leave the demonstrator's command, support raise only",
            "return, all start cells", "return, decision cells", "bad ending, all start cells", "decisiveness",
            "coverage on fresh rows"]
    for task_name, body in result.items():
        for method in methods:
            arms = [arm for arm in body["metrics"] if method in body["metrics"][arm]]
            if not arms:
                continue
            print(f"\n## {task_name}, {method}, runs per arm {body['metrics'][arms[0]][method]['runs']}")
            print("| Metric | " + " | ".join(arms) + " |\n|---|" + "---:|" * len(arms))
            for name in show:
                if name in body["metrics"][arms[0]][method]:
                    print(f"| {name} | " + " | ".join(f"{body['metrics'][arm][method][name]:.3f}" for arm in arms) + " |")
        print(f"\nroom for a method, {task_name}: " + json.dumps(body["room_for_a_method"]))
    if marks:
        print("\n## Marks of Amendment 12")
        for method in (PERMIT, PLAIN):
            print(f"\n{method}: " + ", ".join(f"{name.split(' ')[0]} {'pass' if block['passed'] else 'FAIL'}"
                                              for name, block in marks[method].items()))
            print(json.dumps(marks[method], indent=1))
        print("\nP5: " + json.dumps(marks["P5 the calibrated width earns its place"]))


if __name__ == "__main__":
    main()
