"""Judge marks K1 to K3 of PROTOCOL.md, Amendment 4, from the saved policies. NumPy only.

    python -I -B judge_version_a.py --label version_a_published

Reads the Version A run and the baselines run it started from. Writes version_a.json and
version_a.md into results/<label>/. The score is the paper's: the chance of reaching the goal
from the start cell within 200 steps, as a share of the optimal policy's.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import critical_env as E  # noqa: E402
import published_checks as checks  # noqa: E402
import published_collect as C  # noqa: E402
import validate_published as V  # noqa: E402

VERSION_A, CONTROL, BASELINE = "support_blind_width", "support_shuffled", "baseline"
DECISIVE, SHARPENED = "decisive_baseline", "scaled_baseline"
NAMES = {BASELINE: "Baseline", VERSION_A: "Version A", CONTROL: "Shuffled control", DECISIVE: "Decisive baseline",
         SHARPENED: "Scaled baseline (no mark)"}
GAIN_MARK, SCORE_HARM, LAVA_HARM, DATASETS_NEEDED = 0.05, 0.02, 0.01, 2
GOOD_RECIPES, ROOM_RECIPE = ("optimal", "other_starts"), "mixed"
OUT_OF_RANGE = 1.5              # no score can truly exceed 1.0


def mean(values):
    return float(np.mean(values))


def sharpen(commands, cells, target_mean_abs):
    """Scale every command by one factor, then clip, to reach a target average |preference|
    (evaluation.sharpen of the earlier testbed)."""
    def level(factor):
        return float(np.abs(np.clip(factor * commands[cells], -1, 1)).mean())
    low, high = 0.0, 1.0
    while level(high) < target_mean_abs and high < 1e6:
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if level(mid) < target_mean_abs else (low, mid)
    return np.clip(0.5 * (low + high) * commands, -1, 1), 0.5 * (low + high)


def record(task, best, commands, q1=None, consumed=None):
    """Score and read-outs of one policy. `consumed` is the width the method charges, [cell, command]."""
    free = task.free_cells()
    out = V.policy_record(task, commands, np.zeros((E.SIZE * E.SIZE, E.N_MOVES)) if q1 is None else q1, best)
    rec = {"score": out["score"], "goal": out["goal"], "lava": out["lava"], "timeout": out["timeout"],
           "command_strength": float(np.abs(commands[free]).mean())}
    for name in checks.CRITICAL_CELLS:
        spot = checks.critical_spot(task, name)
        rec[f"{name}_fatal_command"] = out[name]["commands_a_fatal_move"]
        if q1 is not None:
            rec[f"{name}_critic_error_fatal"] = out[name]["critic_error_fatal"]
            rec[f"{name}_critic_rates_fatal_above"] = out[name]["critic_rates_a_fatal_move_above_the_way_through"]
        if consumed is not None:
            rec[f"{name}_width_fatal"] = float(consumed[spot["cell"]][list(spot["fatal"])].mean())
            rec[f"{name}_width_through"] = float(consumed[spot["cell"]][spot["through"]])
    if q1 is not None:
        rec["largest_critic_score"] = float(np.abs(q1[free]).max())
        rec["critic_out_of_range"] = bool(rec["largest_critic_score"] > OUT_OF_RANGE)
    return rec


def load(folder, stem, task, best, wanted_seeds=None):
    """{(recipe, data seed, training seed, method): (record, commands)} from one saved file pair."""
    out = {}
    path = folder / f"{stem}.json"
    saved = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        for run in saved["runs"]:
            if wanted_seeds is not None and run["data_seed"] not in wanted_seeds:
                continue
            i = run["index"]
            commands = arrays["commands"][i]
            rec = record(task, best, commands, arrays["q1"][i], arrays["consumed"][i])
            rec["reference_ready"] = bool(run["reference_ready"])
            out[(run["recipe"], run["data_seed"], run["train_seed"], run["method"])] = (rec, commands)
    return out


def judge(records, seeds, train_seeds):
    def by_dataset(recipe, method, name="score"):
        return {d: mean([records[(recipe, d, s, method)][name] for s in train_seeds]) for d in seeds}

    def overall(recipe, method, name="score"):
        return mean([records[(recipe, d, s, method)][name] for d in seeds for s in train_seeds])

    a, base = by_dataset(ROOM_RECIPE, VERSION_A), by_dataset(ROOM_RECIPE, BASELINE)
    gain = overall(ROOM_RECIPE, VERSION_A) - overall(ROOM_RECIPE, BASELINE)
    k1 = {"per_dataset": {str(d): {"Version A": a[d], "Baseline": base[d], "higher": bool(a[d] > base[d])} for d in seeds},
          "datasets_higher": int(sum(a[d] > base[d] for d in seeds)), "mean_gain": gain,
          "passed": bool(sum(a[d] > base[d] for d in seeds) >= DATASETS_NEEDED and gain >= GAIN_MARK)}
    k2 = {"passed": True}
    for control in (CONTROL, DECISIVE):
        other = by_dataset(ROOM_RECIPE, control)
        wins = int(sum(a[d] > other[d] for d in seeds))
        k2[NAMES[control]] = {"per_dataset": {str(d): {"Version A": a[d], NAMES[control]: other[d]} for d in seeds},
                              "datasets_higher": wins, "passed": bool(wins >= DATASETS_NEEDED)}
        k2["passed"] = bool(k2["passed"] and wins >= DATASETS_NEEDED)
    k3 = {"passed": True}
    for recipe in GOOD_RECIPES:
        score_change = overall(recipe, VERSION_A) - overall(recipe, BASELINE)
        lava_change = overall(recipe, VERSION_A, "lava") - overall(recipe, BASELINE, "lava")
        ok = bool(score_change >= -SCORE_HARM and lava_change <= LAVA_HARM)
        k3[recipe] = {"score_change": score_change, "lava_change": lava_change, "passed": ok}
        k3["passed"] = bool(k3["passed"] and ok)
    return {"K1_helps_where_there_is_room": k1, "K2_not_a_side_effect": k2, "K3_no_harm_where_the_baseline_is_good": k3,
            "version_a_helps_on_the_published_map": bool(k1["passed"] and k2["passed"] and k3["passed"])}


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    start = json.loads((folder / "START.json").read_text())
    task = E.published()
    best = V.best_possible(task)
    seeds = sorted(start["data_seeds"])
    loaded = load(HERE / "results" / start["baselines"], f"baselines_{task.name}", task, best, seeds)
    continued = load(folder, f"methods_{task.name}__{BASELINE}", task, best)
    loaded.update(load(folder, f"methods_{task.name}__{VERSION_A}", task, best))
    records = {key: rec for key, (rec, _) in loaded.items()}
    train_seeds = sorted({key[2] for key in records})
    # Two versions of the baseline that train nothing. Decisive: its top choice at every cell, commanded
    # with certainty (the control of mark K2). Scaled: its commands scaled to the paired Version A run's
    # average strength, as in the earlier testbed (reported only; see the correction in Amendment 4).
    for (recipe, d, s, method), (rec, _) in list(loaded.items()):
        if method == VERSION_A:
            base_commands = loaded[(recipe, d, s, BASELINE)][1]
            records[(recipe, d, s, DECISIVE)] = record(task, best, E.policy_from_moves(base_commands.argmax(axis=1)))
            scaled, factor = sharpen(base_commands, task.free_cells(), rec["command_strength"])
            records[(recipe, d, s, SHARPENED)] = dict(record(task, best, scaled), factor=float(factor))
    # Does the continued baseline reproduce the saved one?
    gaps = [float(np.abs(continued[key][1] - loaded[key][1]).max()) for key in continued]
    reproduction = {"runs": len(gaps), "largest_difference_in_any_preference": max(gaps),
                    "runs_identical": int(sum(g == 0.0 for g in gaps)),
                    "largest_difference_in_score": max(abs(continued[key][0]["score"] - records[key]["score"]) for key in continued)}
    marks = judge(records, seeds, train_seeds)
    methods = (BASELINE, DECISIVE, SHARPENED, CONTROL, VERSION_A)
    summary = {}
    for recipe in C.RECIPES:
        summary[recipe] = {}
        for method in methods:
            recs = [records[(recipe, d, s, method)] for d in seeds for s in train_seeds]
            row = {"runs": len(recs), "score": mean([r["score"] for r in recs]), "lowest": min(r["score"] for r in recs),
                   "lava": mean([r["lava"] for r in recs]), "timeout": mean([r["timeout"] for r in recs]),
                   "command_strength": mean([r["command_strength"] for r in recs]),
                   "per_dataset": {str(d): mean([records[(recipe, d, s, method)]["score"] for s in train_seeds]) for d in seeds}}
            for name in checks.CRITICAL_CELLS:
                row[f"{name}_fatal_command"] = mean([r[f"{name}_fatal_command"] for r in recs])
                if f"{name}_width_fatal" in recs[0]:
                    row[f"{name}_width_fatal"] = mean([r[f"{name}_width_fatal"] for r in recs])
                    row[f"{name}_width_through"] = mean([r[f"{name}_width_through"] for r in recs])
                if f"{name}_critic_rates_fatal_above" in recs[0]:
                    row[f"{name}_critic_rates_fatal_above"] = int(sum(r[f"{name}_critic_rates_fatal_above"] for r in recs))
            if "critic_out_of_range" in recs[0]:
                row["critic_out_of_range"] = int(sum(r["critic_out_of_range"] for r in recs))
                row["calibrations_usable"] = int(sum(r["reference_ready"] for r in recs))
            summary[recipe][method] = row
    result = {"baselines": start["baselines"], "alpha": start["alpha"], "data_seeds": seeds, "train_seeds": train_seeds,
              "devices": start["devices"], "marks": marks, "continued_baseline_reproduces_the_saved_one": reproduction,
              "summary": summary}
    (folder / "version_a.json").write_text(json.dumps(result, indent=1))
    (folder / "version_a_records.json").write_text(json.dumps(
        [dict(recipe=k[0], data_seed=k[1], train_seed=k[2], method=k[3], **v) for k, v in sorted(records.items())], indent=1))

    lines = ["# Version A on the published map\n",
             f"Generated by `judge_version_a.py`. Dataset seeds {seeds}, training seeds {train_seeds}, alpha {start['alpha']:g}. "
             "Every number is exact for the saved policy.\n", "## Scores\n"]
    for recipe in C.RECIPES:
        rows = [[NAMES[m], f"{summary[recipe][m]['score']:.3f}", f"{summary[recipe][m]['lowest']:.3f}",
                 ", ".join(f"{v:.3f}" for v in summary[recipe][m]["per_dataset"].values()),
                 pct(summary[recipe][m]["lava"]), pct(summary[recipe][m]["timeout"]),
                 f"{summary[recipe][m]['command_strength']:.2f}",
                 summary[recipe][m].get("critic_out_of_range", "-")] for m in methods]
        lines += [f"### `{recipe}`\n", V.table(["Policy", "Score, mean", "Lowest", "Per dataset", "Ends in lava", "Times out",
                                               "Command strength", "Runs whose critic leaves the possible range"], rows)]
    lines.append("## At the critical cells\n")
    for name in checks.CRITICAL_CELLS:
        rows = []
        for recipe in C.RECIPES:
            b, a = summary[recipe][BASELINE], summary[recipe][VERSION_A]
            rows.append([recipe, pct(b[f"{name}_fatal_command"], 2), pct(a[f"{name}_fatal_command"], 2),
                         f"{a[f'{name}_width_fatal']:.2f}", f"{a[f'{name}_width_through']:.2f}",
                         f"{b[f'{name}_critic_rates_fatal_above']} / {a[f'{name}_critic_rates_fatal_above']} of {a['runs']}"])
        lines += [f"### The {name.replace('_', ' ')} cell\n",
                  V.table(["Recipe", "Baseline commands a fatal move", "Version A commands a fatal move",
                           "Width Version A charges a fatal move", "Width it charges the way through",
                           "Runs where the critic rates a fatal move above the way through (baseline / Version A)"], rows)]
    k1, k2, k3 = (marks[k] for k in ("K1_helps_where_there_is_room", "K2_not_a_side_effect",
                                    "K3_no_harm_where_the_baseline_is_good"))
    verdict = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines += ["## Marks\n", V.table(["Mark", "Result", "Verdict"], [
        ["K1 helps where there is room", f"higher than the baseline in {k1['datasets_higher']} of {len(seeds)} datasets; "
                                         f"mean gain {k1['mean_gain']:+.3f} (mark +{GAIN_MARK})", verdict(k1["passed"])],
        ["K2 not a side effect", f"higher than the shuffled control in {k2[NAMES[CONTROL]]['datasets_higher']} of {len(seeds)}; "
                                 f"higher than the decisive baseline in {k2[NAMES[DECISIVE]]['datasets_higher']} of {len(seeds)}",
         verdict(k2["passed"])],
        ["K3 no harm where the baseline is good",
         "; ".join(f"{r}: score {k3[r]['score_change']:+.3f}, lava {100 * k3[r]['lava_change']:+.1f} points" for r in GOOD_RECIPES),
         verdict(k3["passed"])]]),
        f"**Version A helps on the published map: {'yes' if marks['version_a_helps_on_the_published_map'] else 'no'}.**\n",
        f"Continued baseline against the saved one: {reproduction['runs_identical']} of {reproduction['runs']} runs identical; "
        f"largest difference in any preference {reproduction['largest_difference_in_any_preference']:.2e}, "
        f"in score {reproduction['largest_difference_in_score']:.2e}.\n"]
    (folder / "version_a.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
