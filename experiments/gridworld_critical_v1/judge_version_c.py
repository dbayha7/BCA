"""Judge marks V1 to V4 of PROTOCOL.md, Amendment 6, from the saved policies. NumPy only.

    python -I -B judge_version_c.py --label version_c_published --version-a version_a_published

Reads the Version C run, the baselines it started from, and the Version A run on the same datasets.
Writes version_c.json, version_c_records.json and version_c.md into results/<label>/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import critical_env as E  # noqa: E402
import judge_version_a as J  # noqa: E402
import published_checks as checks  # noqa: E402
import published_collect as C  # noqa: E402
import validate_published as V  # noqa: E402

VERSION_C = "version_c"
NAMES = dict(J.NAMES, **{VERSION_C: "Version C"})
RANKING_MARK, DATASETS_NEEDED = 1, 2
BASELINE_GAIN = {"optimal": 0.02, "mixed": 0.05}
SCORE_SLACK, LAVA_SLACK, COLLAPSE = 0.02, 0.01, 0.50
STRONG, WEAK = ("optimal", "other_starts"), "mixed"


def judge(records, seeds, train_seeds):
    def runs(recipe, method):
        return [records[(recipe, d, s, method)] for d in seeds for s in train_seeds]

    def overall(recipe, method, name="score"):
        return float(np.mean([r[name] for r in runs(recipe, method)]))

    def by_dataset(recipe, method):
        return [float(np.mean([records[(recipe, d, s, method)]["score"] for s in train_seeds])) for d in seeds]

    def higher(recipe, other):
        return int(sum(a > b for a, b in zip(by_dataset(recipe, VERSION_C), by_dataset(recipe, other))))

    v1 = {"passed": True}
    for recipe in STRONG:
        counts = {NAMES[m]: int(sum(r["critical_critic_rates_fatal_above"] for r in runs(recipe, m)))
                  for m in (J.BASELINE, J.VERSION_A, VERSION_C)}
        ok = counts[NAMES[VERSION_C]] <= RANKING_MARK
        v1[recipe] = dict(counts, tested=counts[NAMES[VERSION_C]], runs=len(runs(recipe, VERSION_C)), passed=bool(ok))
        v1["passed"] = bool(v1["passed"] and ok)
    v2 = {"passed": True}
    for recipe, mark in BASELINE_GAIN.items():
        gain, wins = overall(recipe, VERSION_C) - overall(recipe, J.BASELINE), higher(recipe, J.BASELINE)
        ok = bool(gain >= mark and wins >= DATASETS_NEEDED)
        v2[recipe] = {"gain": gain, "mark": mark, "datasets_higher": wins, "passed": ok}
        v2["passed"] = bool(v2["passed"] and ok)
    v3 = {"passed": True}
    for recipe in STRONG:
        score = overall(recipe, VERSION_C) - overall(recipe, J.VERSION_A)
        lava = overall(recipe, VERSION_C, "lava") - overall(recipe, J.VERSION_A, "lava")
        ok = bool(score >= -SCORE_SLACK and lava <= LAVA_SLACK)
        v3[recipe] = {"score_change": score, "lava_change": lava, "passed": ok}
        v3["passed"] = bool(v3["passed"] and ok)
    gain, wins = overall(WEAK, VERSION_C) - overall(WEAK, J.VERSION_A), higher(WEAK, J.VERSION_A)
    v4 = {"gain": gain, "datasets_higher": wins, "passed": bool(gain > 0 and wins >= DATASETS_NEEDED)}
    marks = {"V1_the_q_values_are_corrected": v1, "V2_better_than_the_baseline": v2,
             "V3_at_least_as_good_as_version_a_where_it_is_strong": v3,
             "V4_better_than_version_a_where_it_is_weak": v4}
    return dict(marks, version_c_passes_its_first_test=bool(all(m["passed"] for m in marks.values())))


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--version-a", required=True, help="results label of the Version A run on the same datasets")
    parser.add_argument("--method", default="version_c", choices=("version_c", "version_c2", "version_c3", "version_c4"),
                        help="which run is judged; the marks are the same (Amendments 6, 7, 8 and 10)")
    args = parser.parse_args()
    global VERSION_C
    VERSION_C = args.method
    NAMES[VERSION_C] = {"version_c": "Version C", "version_c2": "Version C2", "version_c3": "Version C3",
                        "version_c4": "Version C4"}[args.method]
    folder = HERE / "results" / args.label
    start = json.loads((folder / "START.json").read_text())
    task = E.published()
    best = V.best_possible(task)
    seeds = sorted(start["data_seeds"])
    loaded = J.load(HERE / "results" / start["baselines"], f"baselines_{task.name}", task, best, seeds)
    loaded.update(J.load(HERE / "results" / args.version_a, f"methods_{task.name}__{J.VERSION_A}", task, best, seeds))
    loaded.update(J.load(folder, f"methods_{task.name}__{VERSION_C}", task, best))
    records = {key: rec for key, (rec, _) in loaded.items()}
    train_seeds = sorted({key[2] for key in records if key[3] == VERSION_C})
    for (recipe, d, s, method), (_, commands) in list(loaded.items()):
        if method == J.BASELINE:
            records[(recipe, d, s, J.DECISIVE)] = J.record(task, best, E.policy_from_moves(commands.argmax(axis=1)))
    # Version C's own charge at the critical cells (the saved "consumed" width is the shared learner's, not this).
    order = json.loads((folder / f"methods_{task.name}__{VERSION_C}.json").read_text())["runs"]
    with np.load(folder / f"charge_{task.name}__{VERSION_C}.npz") as saved:
        charge, flagged, ready = saved["charge"], saved["cannot_certify"], saved["iw_ready"]
    for run in order:
        rec = records[(run["recipe"], run["data_seed"], run["train_seed"], VERSION_C)]
        rec["table_ready"] = bool(ready[run["index"]])
        for name in checks.CRITICAL_CELLS:
            spot = checks.critical_spot(task, name)
            cell, fatal = spot["cell"], list(spot["fatal"])
            rec[f"{name}_charge_fatal"] = float(charge[run["index"]][cell][fatal].mean())
            rec[f"{name}_charge_through"] = float(charge[run["index"]][cell][spot["through"]])
            rec[f"{name}_fatal_cannot_be_certified"] = bool(flagged[run["index"]][cell][fatal].all())
    marks = judge(records, seeds, train_seeds)
    methods = (J.BASELINE, J.DECISIVE, J.CONTROL, J.VERSION_A, VERSION_C)
    summary = {}
    for recipe in C.RECIPES:
        summary[recipe] = {}
        for method in methods:
            recs = [records[(recipe, d, s, method)] for d in seeds for s in train_seeds]
            row = {"runs": len(recs), "score": float(np.mean([r["score"] for r in recs])),
                   "lowest": float(min(r["score"] for r in recs)), "lava": float(np.mean([r["lava"] for r in recs])),
                   "timeout": float(np.mean([r["timeout"] for r in recs])),
                   "runs_below_half": int(sum(r["score"] < COLLAPSE for r in recs)),
                   "per_dataset": [float(np.mean([records[(recipe, d, s, method)]["score"] for s in train_seeds]))
                                   for d in seeds]}
            for name in checks.CRITICAL_CELLS:
                row[f"{name}_fatal_command"] = float(np.mean([r[f"{name}_fatal_command"] for r in recs]))
                for extra in ("critic_rates_fatal_above", "fatal_cannot_be_certified"):
                    if f"{name}_{extra}" in recs[0]:
                        row[f"{name}_{extra}"] = int(sum(r[f"{name}_{extra}"] for r in recs))
                for extra in ("critic_error_fatal", "charge_fatal", "charge_through"):
                    if f"{name}_{extra}" in recs[0]:
                        row[f"{name}_{extra}"] = float(np.mean([r[f"{name}_{extra}"] for r in recs]))
            if "critic_out_of_range" in recs[0]:
                row["critic_out_of_range"] = int(sum(r["critic_out_of_range"] for r in recs))
                row["calibrations_usable"] = int(sum(r["reference_ready"] for r in recs))
            if "table_ready" in recs[0]:
                row["tables_ready"] = int(sum(r["table_ready"] for r in recs))
            summary[recipe][method] = row
    result = {"baselines": start["baselines"], "version_a": args.version_a, "alpha": start["alpha"], "data_seeds": seeds,
              "train_seeds": train_seeds, "devices": start["devices"], "marks": marks, "summary": summary}
    (folder / "version_c.json").write_text(json.dumps(result, indent=1))
    (folder / "version_c_records.json").write_text(json.dumps(
        [dict(recipe=k[0], data_seed=k[1], train_seed=k[2], method=k[3], **v) for k, v in sorted(records.items())], indent=1))

    lines = ["# Version C on the published map: first test\n",
             f"Generated by `judge_version_c.py`. Dataset seeds {seeds}, training seeds {train_seeds}, "
             f"alpha {start['alpha']:g}. Every number is exact for the saved policy.\n", "## Scores\n"]
    for recipe in C.RECIPES:
        rows = [[NAMES[m], f"{summary[recipe][m]['score']:.3f}", f"{summary[recipe][m]['lowest']:.3f}",
                 ", ".join(f"{v:.3f}" for v in summary[recipe][m]["per_dataset"]), summary[recipe][m]["runs_below_half"],
                 pct(summary[recipe][m]["lava"]), pct(summary[recipe][m]["timeout"]),
                 summary[recipe][m].get("critic_out_of_range", "-")] for m in methods]
        lines += [f"### `{recipe}`\n", V.table(["Policy", "Score, mean", "Lowest run", "Per dataset", "Runs below 0.50",
                                               "Ends in lava", "Times out", "Runs whose critic leaves the possible range"],
                                              rows)]
    lines.append("## The critic at the critical cells\n")
    for name in checks.CRITICAL_CELLS:
        rows = []
        for recipe in C.RECIPES:
            s = summary[recipe]
            rows.append([recipe] + [f"{s[m][f'{name}_critic_rates_fatal_above']} of {s[m]['runs']}"
                                    for m in (J.BASELINE, J.VERSION_A, VERSION_C)]
                        + [f"{s[m][f'{name}_critic_error_fatal']:+.2f}" for m in (J.BASELINE, J.VERSION_A, VERSION_C)]
                        + [pct(s[VERSION_C][f"{name}_fatal_command"], 2), f"{s[VERSION_C][f'{name}_charge_fatal']:.2f}",
                           f"{s[VERSION_C][f'{name}_charge_through']:.2f}",
                           f"{s[VERSION_C][f'{name}_fatal_cannot_be_certified']} of {s[VERSION_C]['runs']}"])
        lines += [f"### The {name.replace('_', ' ')} cell\n",
                  V.table(["Recipe", "Critic rates a fatal move above the way through: baseline", "Version A", "Version C",
                           "Critic's error at the fatal moves: baseline", "Version A", "Version C",
                           "Version C commands a fatal move", "Version C's charge, fatal move", "way through",
                           "Runs where the fatal moves cannot be certified"], rows)]
    v1, v2, v3, v4 = (marks[k] for k in list(marks)[:4])
    verdict = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines += ["## Marks\n", V.table(["Mark", "Result", "Verdict"], [
        ["V1 the Q values are corrected",
         "; ".join(f"{r}: {NAMES[VERSION_C]} {v1[r]['tested']} of {v1[r]['runs']} (baseline {v1[r]['Baseline']}, "
                   f"Version A {v1[r]['Version A']})" for r in STRONG), verdict(v1["passed"])],
        ["V2 better than the baseline",
         "; ".join(f"{r}: {v2[r]['gain']:+.3f} (mark +{v2[r]['mark']}), higher in {v2[r]['datasets_higher']} of "
                   f"{len(seeds)} datasets" for r in BASELINE_GAIN), verdict(v2["passed"])],
        ["V3 at least as good as Version A where it is strong",
         "; ".join(f"{r}: score {v3[r]['score_change']:+.3f}, lava {100 * v3[r]['lava_change']:+.1f} points"
                   for r in STRONG), verdict(v3["passed"])],
        ["V4 better than Version A where it is weak",
         f"mixed: {v4['gain']:+.3f}, higher in {v4['datasets_higher']} of {len(seeds)} datasets", verdict(v4["passed"])]]),
        f"**Version C passes its first test: {'yes' if marks['version_c_passes_its_first_test'] else 'no'}.**\n",
        "Tables ready: " + ", ".join(f"{r} {summary[r][VERSION_C]['tables_ready']} of {summary[r][VERSION_C]['runs']}"
                                     for r in C.RECIPES) + ".\n"]
    (folder / "version_c.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
