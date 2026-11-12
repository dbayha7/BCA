"""Every metric we can read for Version A, from the saved policies of the fresh-data runs.

    python -I -B audit_version_a.py            (writes results/version_a_audit.json and prints tables)

Read-only. Trains nothing, fits nothing, and changes no file of any run. It reads, per policy, the command at
every cell and, at every cell and corner command, the two critics, the calibrated width and the support score.

This is a description of runs whose verdicts are already fixed (PROTOCOL.md Amendments 6, 7 and 9). Pooling
them here confirms nothing new.

Signed errors are prediction minus truth: positive means the critic rates a command too high.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402
import environment as env  # noqa: E402
import evaluation as ev  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
A, CONTROL = "support_blind_width", "support_shuffled"
SHARP = "sharpened"
HONEYPOT_RUNS = {"confirm_v1": "confirm_h", "confirm_v2": "confirm2_h", "confirm_v4": "confirm4_h"}
BOTTLENECK_RUNS = {"confirm_v1": "confirm_b", "confirm_v2": "confirm2_b"}
ARMS = ("targeted", "complete", "placebo")
METHODS = ("baseline", SHARP, A, CONTROL)
LN10 = float(np.log(10.0))
ONE_POINT = 0.01


def load(label, group):
    """Every saved policy of one group: (task, data seed, arm, training seed, method) -> arrays and run facts."""
    out = {}
    for stem in (f"stage1_{group}", f"stage2_{group}__{A}"):
        meta = json.loads((RESULTS / label / f"{stem}.json").read_text())
        with np.load(RESULTS / label / f"{stem}.npz", allow_pickle=False) as arrays:
            for run in meta["runs"]:
                key = (run["task"], run["data_seed"], run["arm"], run["train_seed"], run["method"])
                out[key] = {"run": run, **{name: arrays[name][run["index"]] for name in arrays.files}}
    used = {k[1] for k in out if k[4] == A}          # datasets that have Version A policies
    return {k: v for k, v in out.items() if k[1] in used}


def corner_truth(task, value):
    """True value of every corner command at every cell: issue it once, then follow the policy behind `value`."""
    return task.direction_values(value) @ task.probabilities(env.CORNERS).T        # [100 cells, 16 corners]


def measure(task, commands, optimum, data=None, saved=None):
    """All metrics of one policy. `saved` carries the critics, widths and support score when they exist."""
    free = task.free_cells()
    decision = collect.targeted_moves(task)
    honeypot = task.name == "honeypot"
    extras = {}
    if saved is not None:
        extras = dict(q1=saved["q1"], q2=saved["q2"], width=saved["consumed"], original_width=saved["width"],
                      support=saved["support"], floor=saved["run"]["reference"]["support_floor"])
    record = ev.policy_record(task, commands, **extras)
    reached, value = task.outcomes(commands), task.value(commands)
    probs = task.probabilities(commands[free])
    # A cell can have more than one best move (two ways toward the goal that are equally good).
    best_values = task.direction_values(optimum["value"])[free]
    is_best = best_values >= best_values.max(axis=1, keepdims=True) - 1e-9
    m = {
        "primary outcome": record["primary_outcome"],
        "chance per step of the withheld move at the decision cells": record["probability_readout"],
        "runs where the withheld move has zero chance at every decision cell":
            float(all(record["cells"][str(c)]["p_target_move"] == 0.0 for c in decision)),
        "return, all start cells": record["mean_value_all_starts"],
        "return, decision cells": record["mean_value_decision_cells"],
        "return, far start A": float(value[env.A_START]),
        "share of the best possible return, all start cells": record["mean_value_all_starts"] / optimum["mean_value"],
        "goal reached, all start cells": float(reached["goal"][free].mean()),
        "any hazard ending, all start cells": float(reached["hazard"][free].mean()),
        "time-out, all start cells": float(reached["timeout"][free].mean()),
        "bad ending (hazard or time-out), all start cells": float((reached["hazard"] + reached["timeout"])[free].mean()),
        "worst start cell, chance of a bad ending": float((reached["hazard"] + reached["timeout"])[free].max()),
        "start cells that never reach the goal": float((reached["goal"][free] == 0.0).sum()),
        "decisiveness (mean absolute preference)": record["mean_abs_preference"],
        "cells whose likeliest move is a best move": float(is_best[np.arange(len(free)), np.argmax(probs, axis=1)].mean()),
        "probability placed on best moves": float((probs * is_best).sum(axis=1).mean()),
    }
    if honeypot:
        m["withheld-hazard ending, all start cells"] = float(ev.unseen_hazard_probability(task, commands)[free].mean())
    if data is not None:
        m["cloning error (mean squared distance to recorded commands)"] = float(
            np.mean(np.square(commands[data["s"]] - data["a"])))
    per_cell = {"bad": (reached["hazard"] + reached["timeout"])[free], "value": value[free]}
    if saved is None:
        return m, per_cell

    q1, q2, width, support = saved["q1"], saved["q2"], saved["width"], saved["support"]
    reference = saved["run"]["reference"]
    floor, cap = reference["support_floor"], reference["width_cap"]
    decades = np.maximum(floor - support, 0.0) / LN10
    raised = width + decades * np.maximum(cap - width, 0.0)       # the support-aware width, at the true state
    truth = corner_truth(task, value)
    corner_probs = task.probabilities(env.CORNERS)
    rows = []
    for cell, (direction,) in decision.items():
        target, best = ev.target_corner(direction), int(np.argmax(truth[cell]))
        could = corner_probs[:, direction] > 0
        charge_w, charge_r = ev.KAPPA * width[cell], ev.KAPPA * raised[cell]
        rows.append({
            "critic error at the withheld move": q1[cell, target] - truth[cell, target],
            "critic error at the withheld move, lower of the two critics":
                min(q1[cell, target], q2[cell, target]) - truth[cell, target],
            "critic error at the best move": q1[cell, best] - truth[cell, best],
            "critic error, corner commands that could execute the withheld move": float((q1[cell] - truth[cell])[could].mean()),
            "critic error, corner commands that cannot": float((q1[cell] - truth[cell])[~could].mean()),
            "true value of the withheld move": truth[cell, target],
            "true value of the best move": truth[cell, best],
            "critic's value of the withheld move": q1[cell, target],
            "critic's value of the best move": q1[cell, best],
            "critic rates the withheld move above the best move": float(q1[cell, target] > q1[cell, best]),
            "still above after the calibrated width's penalty":
                float(q1[cell, target] - charge_w[target] > q1[cell, best] - charge_w[best]),
            "still above after the support-aware width's penalty":
                float(q1[cell, target] - charge_r[target] > q1[cell, best] - charge_r[best]),
            "twin critics disagree by, at the withheld move": abs(q1[cell, target] - q2[cell, target]),
            "twin critics disagree by, at the best move": abs(q1[cell, best] - q2[cell, best]),
            "calibrated width at the withheld move": width[cell, target],
            "calibrated width at the best move": width[cell, best],
            "support score below its floor at the withheld move, factors of ten": decades[cell, target],
            "support score below its floor at the best move, factors of ten": decades[cell, best],
            "support-aware width at the withheld move": raised[cell, target],
            "support-aware width at the best move": raised[cell, best],
            "penalty from the calibrated width at the withheld move": charge_w[target],
            "penalty from the support-aware width at the withheld move": charge_r[target],
            "penalty from the support-aware width at the best move": charge_r[best],
        })
    for name in rows[0]:
        m[name] = float(np.mean([row[name] for row in rows]))
    error = (q1 - truth)[free]
    m["critic error over all cells and corner commands, signed"] = float(error.mean())
    m["critic error over all cells and corner commands, absolute"] = float(np.abs(error).mean())
    m["calibrated width, mean over all cells and corner commands"] = float(width[free].mean())
    m["corner commands flagged unsupported, all cells"] = float((decades[free] > 0).mean())
    ordinary = [cell for cell in free if cell not in decision]
    m["corner commands flagged unsupported, cells other than the decision cells"] = float((decades[ordinary] > 0).mean())
    best_corner = np.argmax(truth, axis=1)
    m["cells whose best corner command is flagged unsupported"] = float(
        (decades[free, best_corner[free]] > 0).mean())
    m["twin critics disagree by, mean over all cells and corner commands"] = float(np.abs(q1 - q2)[free].mean())
    run = saved["run"]
    for bank, title in (("audit_bank", "fresh ordinary rows"), ("target_move_rows", "rows that could execute the withheld move")):
        m[f"coverage of the width this method uses, {title}"] = run[bank]["coverage"]
        m[f"mean absolute one-step error, {title}"] = run[bank]["mean_abs_residual"]
        m[f"mean width this method uses, {title}"] = run[bank]["mean_width"]
        m[f"flagged unsupported, {title}"] = run[bank]["share_scored_unsupported"]
    m["calibrations usable"] = float(np.mean([event["usable"] for event in run["refresh_events"]]))
    return m, per_cell


def audit_group(runs, group, tasks):
    """Per task, arm and method: the mean of every metric over runs, and paired comparisons with the baseline."""
    out = {}
    for task_name in tasks:
        task = checks.TASKS[task_name]
        commands, value = env.best_policy(task)
        free = task.free_cells()
        optimum = {"value": value, "mean_value": float(value[free].mean())}
        policies = {}
        for label, data_label in runs.items():
            for key, saved in load(label, group).items():
                if key[0] == task_name:
                    policies[(label,) + key[1:]] = (saved, data_label)
        table, paired = {}, {}
        datasets = sorted({(k[0], k[1]) for k in policies})
        for arm in ARMS:
            per_method = {method: [] for method in METHODS}
            cells = {method: {} for method in METHODS}
            for (label, seed, this_arm, train_seed, method), (saved, data_label) in sorted(policies.items()):
                if this_arm != arm or method not in (A, CONTROL, "baseline"):
                    continue
                with np.load(HERE / "data" / data_label / f"{task_name}_{arm}_seed{seed}.npz") as file:
                    data = {"s": file["s"], "a": file["a"]}
                m, per_cell = measure(task, saved["commands"], optimum, data, saved)
                run_key = (label, seed, train_seed)
                per_method[method].append((run_key, m))
                cells[method][run_key] = per_cell
                if method == A:     # its sharpened baseline: the baseline's commands scaled to Version A's decisiveness
                    base = policies[(label, seed, arm, train_seed, "baseline")][0]["commands"]
                    sharpened, _ = ev.sharpen(base, free, m["decisiveness (mean absolute preference)"])
                    ms, pc = measure(task, sharpened, optimum, data)
                    per_method[SHARP].append((run_key, ms))
                    cells[SHARP][run_key] = pc
            names = sorted({name for rows in per_method.values() for _, m in rows for name in m})
            table[arm] = {name: {method: (float(np.mean([m[name] for _, m in rows if name in m]))
                                          if any(name in m for _, m in rows) else None)
                                 for method, rows in per_method.items()} for name in names}
            table[arm]["runs"] = {method: len(rows) for method, rows in per_method.items()}
            # Paired with the baseline, run by run and dataset by dataset.
            paired[arm] = {}
            for method in (SHARP, A, CONTROL):
                worse = [float((cells[method][k]["bad"] > cells["baseline"][k]["bad"] + ONE_POINT).sum()) for k in cells[method]]
                better = [float((cells[method][k]["bad"] < cells["baseline"][k]["bad"] - ONE_POINT).sum()) for k in cells[method]]
                lower_value = [float((cells[method][k]["value"] < cells["baseline"][k]["value"] - 0.1).sum()) for k in cells[method]]
                higher_value = [float((cells[method][k]["value"] > cells["baseline"][k]["value"] + 0.1).sum()) for k in cells[method]]
                mine, base = dict(per_method[method]), dict(per_method["baseline"])
                by_dataset = {}
                for name in ("primary outcome", "return, all start cells", "bad ending (hazard or time-out), all start cells"):
                    diffs = [np.mean([mine[k][name] - base[k][name] for k in mine if k[:2] == d]) for d in datasets]
                    by_dataset[name] = {"lower than the baseline in": int(sum(x < 0 for x in diffs)),
                                        "equal": int(sum(x == 0 for x in diffs)),
                                        "higher than the baseline in": int(sum(x > 0 for x in diffs)),
                                        "datasets": len(diffs)}
                paired[arm][method] = {
                    "start cells made worse per run (bad endings up by more than 1 point)": float(np.mean(worse)),
                    "start cells made better per run (bad endings down by more than 1 point)": float(np.mean(better)),
                    "start cells with lower return per run (by more than 0.1)": float(np.mean(lower_value)),
                    "start cells with higher return per run (by more than 0.1)": float(np.mean(higher_value)),
                    "by dataset": by_dataset}
        out[task_name] = {"datasets": len(datasets), "free_cells": len(free), "best_possible_return": optimum["mean_value"],
                          "metrics": table, "paired_with_the_baseline": paired}
    return out


def mechanism():
    """Honeypot, targeted arm, one case per run and decision cell: what the baseline's critic believes there,
    and what the baseline, its sharpened copy and Version A then do."""
    task = checks.TASKS["honeypot"]
    free, decision = task.free_cells(), collect.targeted_moves(task)
    cases, leftovers, flagged_by_plus_ones = [], [], {k: [] for k in range(5)}
    plus_ones = (env.CORNERS > 0).sum(axis=1)
    ordinary = [cell for cell in free if cell not in decision]
    for label in HONEYPOT_RUNS:
        policies = load(label, "honeypot")
        for key, base in policies.items():
            if key[2] != "targeted" or key[4] != "baseline":
                continue
            version_a = policies[key[:4] + (A,)]
            truth = corner_truth(task, task.value(base["commands"]))
            scale = float(np.abs(version_a["commands"][free]).mean())
            sharpened, _ = ev.sharpen(base["commands"], free, scale)
            floor = base["run"]["reference"]["support_floor"]
            decades = np.maximum(floor - base["support"], 0.0) / LN10
            for k in range(5):
                flagged_by_plus_ones[k].append(float((decades[ordinary][:, plus_ones == k] > 0).mean()))
            for cell, (direction,) in decision.items():
                target, best = ev.target_corner(direction), int(np.argmax(truth[cell]))
                chance = {name: float(task.probabilities(commands[cell])[direction])
                          for name, commands in (("baseline", base["commands"]), ("sharpened", sharpened),
                                                 ("version_a", version_a["commands"]))}
                cases.append({"critic_prefers_withheld": bool(base["q1"][cell, target] > base["q1"][cell, best]),
                              "critic_margin_best_minus_withheld": float(base["q1"][cell, best] - base["q1"][cell, target]),
                              "baseline_likeliest_is_withheld": bool(np.argmax(task.probabilities(base["commands"][cell])) == direction),
                              **chance})
                if chance["version_a"] > 0:
                    a_floor = version_a["run"]["reference"]["support_floor"]
                    leftovers.append({"run": label, "data_seed": key[1], "train_seed": key[3], "cell": cell,
                                      "chance_per_step": chance["version_a"],
                                      "preferences": [round(float(x), 3) for x in version_a["commands"][cell]],
                                      "withheld_direction": env.DIRECTION_NAMES[direction],
                                      "factors_of_ten_below_floor_at_the_pure_withheld_command":
                                          float(max(a_floor - version_a["support"][cell, target], 0.0) / LN10),
                                      "baseline_chance_per_step": chance["baseline"]})

    def summarize(rows):
        if not rows:
            return {"cases": 0}
        return {"cases": len(rows),
                "baseline: mean chance per step of the withheld move": float(np.mean([r["baseline"] for r in rows])),
                "baseline: cases where the withheld move can happen": float(np.mean([r["baseline"] > 0 for r in rows])),
                "sharpened baseline: mean chance per step": float(np.mean([r["sharpened"] for r in rows])),
                "sharpened baseline: cases where it can happen": float(np.mean([r["sharpened"] > 0 for r in rows])),
                "Version A: mean chance per step": float(np.mean([r["version_a"] for r in rows])),
                "Version A: cases where it can happen": float(np.mean([r["version_a"] > 0 for r in rows])),
                "baseline's likeliest move is the withheld move": float(np.mean([r["baseline_likeliest_is_withheld"] for r in rows])),
                "critic's margin, best move minus withheld move": float(np.mean([r["critic_margin_best_minus_withheld"] for r in rows]))}

    return {"all cases": summarize(cases),
            "critic rates the withheld move above the best move": summarize([c for c in cases if c["critic_prefers_withheld"]]),
            "critic still rates the best move higher": summarize([c for c in cases if not c["critic_prefers_withheld"]]),
            "cases where Version A can still execute the withheld move": leftovers,
            "ordinary cells: share of corner commands flagged unsupported, by how many preferences are +1":
                {str(k): float(np.mean(v)) for k, v in flagged_by_plus_ones.items()}}


def environment_facts():
    """What the Honeypot is, and what its datasets hold, checked from the files themselves."""
    task = checks.TASKS["honeypot"]
    free, decision = task.free_cells(), collect.targeted_moves(task)
    commands, value = env.best_policy(task)
    per_direction = task.direction_values(value)
    reached = task.outcomes(commands)

    def path(start):
        cell, seen = start, [start]
        for _ in range(env.HORIZON):
            direction = int(np.argmax(task.probabilities(commands[cell])))
            cell, _, done = task.move(cell, direction)
            seen.append(cell)
            if done:
                break
        return seen

    through = [start for start in free if any(c in decision for c in path(start)[1:])]
    facts = {
        "free cells": len(free), "walls": sorted(task.walls), "hazards": sorted(task.hazards), "goal": env.GOAL,
        "rewards": {"step": env.STEP_REWARD, "goal": env.GOAL_REWARD, "hazard": env.HAZARD_REWARD},
        "discount": env.GAMMA, "horizon": env.HORIZON, "cutoff": env.CUTOFF,
        "best policy: mean return over start cells": float(value[free].mean()),
        "best policy: return from the far start A": float(value[env.A_START]),
        "best policy: goal reached from every start cell": bool(np.allclose(reached["goal"][free], 1.0)),
        "best policy: withheld-hazard endings from the decision cells": float(
            np.mean([ev.unseen_hazard_probability(task, commands)[c] for c in decision])),
        "start cells whose best path passes through a decision cell": len(through),
        "best path from A passes through a decision cell": bool(any(c in decision for c in path(env.A_START)[1:])),
        "decision cells": {str(cell): {"withheld move": env.DIRECTION_NAMES[d],
                                       "true value of each move under the best policy":
                                           {env.DIRECTION_NAMES[i]: round(float(per_direction[cell, i]), 3) for i in range(4)}}
                           for cell, (d,) in decision.items()},
    }
    rows = []
    for label, data_label in HONEYPOT_RUNS.items():
        for seed in sorted({k[1] for k in load(label, "honeypot")}):
            files = {arm: dict(np.load(HERE / "data" / data_label / f"honeypot_{arm}_seed{seed}.npz")) for arm in ARMS}
            targeted, complete = files["targeted"], files["complete"]
            at_decision = np.isin(targeted["s"], list(decision))
            entry = {"seed": seed,
                     "rows per cell, smallest and largest": [int(np.bincount(targeted["s"], minlength=100)[free].min()),
                                                             int(np.bincount(targeted["s"], minlength=100)[free].max())],
                     "targeted and complete identical outside the decision cells": bool(all(
                         np.array_equal(targeted[f][~at_decision], complete[f][~np.isin(complete["s"], list(decision))])
                         for f in ("s", "a", "r", "s2", "done"))),
                     "uniform commands redrawn at the decision cells": float(
                         (targeted["redraws"][at_decision & ~targeted["from_demonstrator"]] > 0).mean())}
            could_t = could_c = did_c = fewest_other = 0
            fewest_other = 10 ** 9
            for cell, (d,) in decision.items():
                for name, data in (("t", targeted), ("c", complete)):
                    here = data["s"] == cell
                    can = np.array([task.could_execute(a, d) for a in data["a"][here]])
                    if name == "t":
                        could_t += int(can.sum())
                        executed = np.bincount(data["direction"][here], minlength=4)
                        assert executed[d] == 0
                        fewest_other = min(fewest_other, int(min(executed[i] for i in range(4) if i != d)))
                    else:
                        could_c += int(can.sum())
                        did_c += int((data["direction"][here] == d).sum())
            entry.update({"targeted: commands that could execute the withheld move": could_t,
                          "targeted: fewest executions of any other move at a decision cell": fewest_other,
                          "complete: commands that could execute it, per decision cell": could_c / len(decision),
                          "complete: times it was executed, per decision cell": did_c / len(decision)})
            rows.append(entry)
    facts["datasets"] = {
        "count": len(rows),
        "rows per cell is 256 everywhere": all(r["rows per cell, smallest and largest"] == [256, 256] for r in rows),
        "targeted and complete identical outside the decision cells, in every dataset": all(
            r["targeted and complete identical outside the decision cells"] for r in rows),
        "targeted: commands that could execute the withheld move, total": sum(
            r["targeted: commands that could execute the withheld move"] for r in rows),
        "targeted: fewest executions of any other move at a decision cell, over all datasets": min(
            r["targeted: fewest executions of any other move at a decision cell"] for r in rows),
        "uniform commands redrawn at the decision cells, mean share": float(np.mean(
            [r["uniform commands redrawn at the decision cells"] for r in rows])),
        "complete: commands that could execute it, per decision cell, mean": float(np.mean(
            [r["complete: commands that could execute it, per decision cell"] for r in rows])),
        "complete: times it was executed, per decision cell, mean": float(np.mean(
            [r["complete: times it was executed, per decision cell"] for r in rows]))}
    return facts


def show(result, task_name, arm, names=None):
    block = result[task_name]["metrics"][arm]
    print(f"\n### {task_name}, {arm} arm, runs per method {block['runs']}")
    print("| Metric | Baseline | Sharpened baseline | Version A | Shuffled control |\n|---|---:|---:|---:|---:|")
    for name in (names or [n for n in block if n != "runs"]):
        row = block[name]
        cells = ["" if row[m] is None else f"{row[m]:.4f}" for m in METHODS]
        print(f"| {name} | " + " | ".join(cells) + " |")


def main():
    result = {"honeypot": audit_group(HONEYPOT_RUNS, "honeypot", ["honeypot"]),
              "bottleneck": audit_group(BOTTLENECK_RUNS, "bottleneck_r2_one_hot",
                                        ["bottleneck_r2_exit45", "bottleneck_r2_exit35", "bottleneck_r2_exit25"])}
    extra = {"mechanism": mechanism(), "environment": environment_facts()}
    (RESULTS / "version_a_audit.json").write_text(json.dumps({**result, **extra}, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps(extra, indent=1))
    for family, block in result.items():
        for task_name, body in block.items():
            print(f"\n## {task_name}: {body['datasets']} datasets, {body['free_cells']} free cells, "
                  f"best possible return {body['best_possible_return']:.3f}")
            for arm in ARMS:
                show(block, task_name, arm)
                print(f"\npaired with the baseline, {arm} arm:")
                print(json.dumps(body["paired_with_the_baseline"][arm], indent=1))


if __name__ == "__main__":
    main()
