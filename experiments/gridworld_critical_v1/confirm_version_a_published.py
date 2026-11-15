"""Select the datasets and judge the claims of PROTOCOL.md, Amendment 5. NumPy only.

    python -I -B confirm_version_a_published.py select --baselines confirm_baselines
    python -I -B confirm_version_a_published.py judge --label confirm_version_a

`select` reads the baselines alone and writes selected_datasets.json: the first nine valid dataset
seeds. `judge` reads the Version A run and writes confirmation.json and confirmation.md.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import critical_env as E  # noqa: E402
import judge_version_a as J  # noqa: E402
import published_collect as C  # noqa: E402
import validate_published as V  # noqa: E402

DATASETS_WANTED, DATASETS_HIGHER, P_MARK = 9, 6, 0.05
GAIN_MARKS = {"mixed": 0.05, "optimal": 0.02}
SCORE_HARM, LAVA_HARM, COLLAPSE = 0.02, 0.01, 0.50
NAMES = J.NAMES


def sign_flip_p(differences):
    """Exact one-sided p: the share of all sign patterns whose mean is at least the observed mean
    (confirm_version_a.sign_flip_p of the earlier testbed)."""
    observed = sum(differences)
    patterns = list(itertools.product((1.0, -1.0), repeat=len(differences)))
    as_large = sum(sum(s * d for s, d in zip(signs, differences)) >= observed - 1e-12 for signs in patterns)
    return as_large / len(patterns)


def select(valid_seeds, wanted=DATASETS_WANTED):
    """The first `wanted` valid seeds in increasing order, or none if there are too few."""
    ordered = sorted(valid_seeds)
    enough = len(ordered) >= wanted
    return {"valid": ordered, "wanted": wanted, "enough": enough, "selected": ordered[:wanted] if enough else []}


def comparison(records, recipe, first, second, seeds, train_seeds):
    """Per-dataset differences of mean score, the count of datasets where `first` is higher, and the exact test."""
    def dataset_mean(method, d):
        return float(np.mean([records[(recipe, d, s, method)]["score"] for s in train_seeds]))
    differences = [dataset_mean(first, d) - dataset_mean(second, d) for d in seeds]
    higher, p = int(sum(x > 0 for x in differences)), sign_flip_p(differences)
    needed = DATASETS_HIGHER if len(seeds) == DATASETS_WANTED else int(np.ceil(len(seeds) * DATASETS_HIGHER / DATASETS_WANTED))
    return {"first": NAMES[first], "second": NAMES[second], "recipe": recipe, "differences": differences,
            "mean_difference": float(np.mean(differences)), "datasets_higher": higher, "datasets": len(seeds),
            "datasets_needed": needed, "p": p, "holds": bool(higher >= needed and p <= P_MARK)}


def judge(records, seeds, train_seeds):
    def overall(recipe, method, name="score"):
        return float(np.mean([records[(recipe, d, s, method)][name] for d in seeds for s in train_seeds]))

    def against(recipe, other):
        return comparison(records, recipe, J.VERSION_A, other, seeds, train_seeds)

    claims = {}
    for name, recipe in (("C1_helps_on_the_half_random_data", "mixed"), ("C2_helps_on_optimal_data", "optimal")):
        c = against(recipe, J.BASELINE)
        gain_ok = bool(c["mean_difference"] >= GAIN_MARKS[recipe])
        claims[name] = dict(c, gain_mark=GAIN_MARKS[recipe], gain_reached=gain_ok, passed=bool(c["holds"] and gain_ok))
    score_change = overall("other_starts", J.VERSION_A) - overall("other_starts", J.BASELINE)
    lava_change = overall("other_starts", J.VERSION_A, "lava") - overall("other_starts", J.BASELINE, "lava")
    claims["C3_no_harm_where_there_is_no_room"] = {
        "score_change": score_change, "lava_change": lava_change,
        "passed": bool(score_change >= -SCORE_HARM and lava_change <= LAVA_HARM)}
    d1 = {recipe: against(recipe, J.DECISIVE) for recipe in ("mixed", "optimal")}
    claims["D1_more_than_decisiveness"] = dict(d1, passed=all(c["holds"] for c in d1.values()))
    d2 = {recipe: against(recipe, J.CONTROL) for recipe in C.RECIPES}
    claims["D2_the_true_cell_matters"] = dict(d2, passed=all(c["holds"] for c in d2.values()))
    confirmed = all(claims[k]["passed"] for k in ("C1_helps_on_the_half_random_data", "C2_helps_on_optimal_data",
                                                 "C3_no_harm_where_there_is_no_room"))
    return dict(claims, version_a_is_confirmed_on_the_published_map=bool(confirmed))


def do_select(args):
    folder = HERE / "results" / args.baselines
    verdict = json.loads((folder / "published_validation.json").read_text())
    w1 = verdict["W1_baseline_is_competent_on_optimal_data"]
    choice = select([row["data_seed"] for row in w1["per_seed"] if row["valid"]])
    choice["candidates"] = verdict["W0_data_are_what_the_paper_describes"]["seeds_generated"]
    choice["dropped_by_the_checks"] = verdict["W0_data_are_what_the_paper_describes"]["seeds_dropped"]
    choice["not_valid"] = [row["data_seed"] for row in w1["per_seed"] if not row["valid"]]
    (folder / "selected_datasets.json").write_text(json.dumps(choice, indent=1))
    print(f"candidates {choice['candidates']}\nnot valid {choice['not_valid']}\nselected {choice['selected']}"
          f"\nenough: {choice['enough']}")
    sys.exit(0 if choice["enough"] else 4)


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def line(c):
    return (f"higher in {c['datasets_higher']} of {c['datasets']} datasets, mean difference {c['mean_difference']:+.3f}, "
            f"p = {c['p']:.4f}")


def do_judge(args):
    folder = HERE / "results" / args.label
    start = json.loads((folder / "START.json").read_text())
    task = E.published()
    best = V.best_possible(task)
    seeds = sorted(start["data_seeds"])
    selection = json.loads((HERE / "results" / start["baselines"] / "selected_datasets.json").read_text())
    assert seeds == selection["selected"], "the run was not trained on the selected datasets"
    loaded = J.load(HERE / "results" / start["baselines"], f"baselines_{task.name}", task, best, seeds)
    continued = J.load(folder, f"methods_{task.name}__{J.BASELINE}", task, best)
    loaded.update(J.load(folder, f"methods_{task.name}__{J.VERSION_A}", task, best))
    records = {key: rec for key, (rec, _) in loaded.items()}
    train_seeds = sorted({key[2] for key in records})
    for (recipe, d, s, method), (rec, _) in list(loaded.items()):
        if method == J.VERSION_A:
            base_commands = loaded[(recipe, d, s, J.BASELINE)][1]
            records[(recipe, d, s, J.DECISIVE)] = J.record(task, best, E.policy_from_moves(base_commands.argmax(axis=1)))
            scaled, _ = J.sharpen(base_commands, task.free_cells(), rec["command_strength"])
            records[(recipe, d, s, J.SHARPENED)] = J.record(task, best, scaled)
    gaps = [float(np.abs(continued[key][1] - loaded[key][1]).max()) for key in continued]
    reproduction = {"runs": len(gaps), "runs_identical": int(sum(g == 0.0 for g in gaps)),
                    "largest_difference_in_any_preference": max(gaps)}
    claims = judge(records, seeds, train_seeds)
    methods = (J.BASELINE, J.DECISIVE, J.SHARPENED, J.CONTROL, J.VERSION_A)
    summary = {}
    for recipe in C.RECIPES:
        summary[recipe] = {}
        for method in methods:
            recs = [records[(recipe, d, s, method)] for d in seeds for s in train_seeds]
            row = {"runs": len(recs), "score": float(np.mean([r["score"] for r in recs])),
                   "lowest": float(min(r["score"] for r in recs)), "lava": float(np.mean([r["lava"] for r in recs])),
                   "timeout": float(np.mean([r["timeout"] for r in recs])),
                   "runs_below_half": int(sum(r["score"] < COLLAPSE for r in recs)),
                   "command_strength": float(np.mean([r["command_strength"] for r in recs])),
                   "per_dataset": {str(d): float(np.mean([records[(recipe, d, s, method)]["score"] for s in train_seeds]))
                                   for d in seeds}}
            for name in ("critical", "second_critical"):
                row[f"{name}_fatal_command"] = float(np.mean([r[f"{name}_fatal_command"] for r in recs]))
                if f"{name}_width_fatal" in recs[0]:
                    row[f"{name}_width_fatal"] = float(np.mean([r[f"{name}_width_fatal"] for r in recs]))
                    row[f"{name}_width_through"] = float(np.mean([r[f"{name}_width_through"] for r in recs]))
            if "critic_out_of_range" in recs[0]:
                row["critic_out_of_range"] = int(sum(r["critic_out_of_range"] for r in recs))
                row["calibrations_usable"] = int(sum(r["reference_ready"] for r in recs))
            summary[recipe][method] = row
    result = {"baselines": start["baselines"], "alpha": start["alpha"], "selection": selection, "data_seeds": seeds,
              "train_seeds": train_seeds, "devices": start["devices"], "claims": claims,
              "continued_baseline_reproduces_the_saved_one": reproduction, "summary": summary}
    (folder / "confirmation.json").write_text(json.dumps(result, indent=1))
    (folder / "confirmation_records.json").write_text(json.dumps(
        [dict(recipe=k[0], data_seed=k[1], train_seed=k[2], method=k[3], **v) for k, v in sorted(records.items())], indent=1))

    verdict = lambda ok: "HOLDS" if ok else "DOES NOT HOLD"  # noqa: E731
    lines = ["# Confirmation of Version A on the published map\n",
             f"Generated by `confirm_version_a_published.py`. Datasets {seeds}; training seeds {train_seeds}; "
             f"alpha {start['alpha']:g}. Candidates not valid: {selection['not_valid'] or 'none'}. "
             "Every number is exact for the saved policy.\n", "## Scores\n"]
    for recipe in C.RECIPES:
        rows = [[NAMES[m], f"{summary[recipe][m]['score']:.3f}", f"{summary[recipe][m]['lowest']:.3f}",
                 summary[recipe][m]["runs_below_half"], pct(summary[recipe][m]["lava"]), pct(summary[recipe][m]["timeout"]),
                 summary[recipe][m].get("critic_out_of_range", "-")] for m in methods]
        lines += [f"### `{recipe}` ({summary[recipe][J.VERSION_A]['runs']} runs per policy)\n",
                  V.table(["Policy", "Score, mean", "Lowest run", "Runs below 0.50", "Ends in lava", "Times out",
                           "Runs whose critic leaves the possible range"], rows)]
    c1, c2, c3 = (claims[k] for k in ("C1_helps_on_the_half_random_data", "C2_helps_on_optimal_data",
                                    "C3_no_harm_where_there_is_no_room"))
    d1, d2 = claims["D1_more_than_decisiveness"], claims["D2_the_true_cell_matters"]
    lines += ["## Claims\n", V.table(["Claim", "Result", "Verdict"], [
        ["C1 helps on the half-random data", f"{line(c1)}; gain mark +{c1['gain_mark']}", verdict(c1["passed"])],
        ["C2 helps on optimal data", f"{line(c2)}; gain mark +{c2['gain_mark']}", verdict(c2["passed"])],
        ["C3 no harm where there is no room",
         f"other_starts: score {c3['score_change']:+.3f}, lava {100 * c3['lava_change']:+.1f} points", verdict(c3["passed"])],
        ["D1 more than decisiveness", "; ".join(f"{r}: {line(d1[r])}" for r in ("mixed", "optimal")), verdict(d1["passed"])],
        ["D2 the true cell matters", "; ".join(f"{r}: {line(d2[r])}" for r in C.RECIPES), verdict(d2["passed"])]]),
        f"**Version A is confirmed on the published map: "
        f"{'yes' if claims['version_a_is_confirmed_on_the_published_map'] else 'no'}.**\n",
        "## At the critical cells\n"]
    for name in ("critical", "second_critical"):
        rows = [[recipe, pct(summary[recipe][J.BASELINE][f"{name}_fatal_command"], 2),
                 pct(summary[recipe][J.VERSION_A][f"{name}_fatal_command"], 2),
                 f"{summary[recipe][J.VERSION_A][f'{name}_width_fatal']:.2f}",
                 f"{summary[recipe][J.VERSION_A][f'{name}_width_through']:.2f}"] for recipe in C.RECIPES]
        lines += [f"### The {name.replace('_', ' ')} cell\n",
                  V.table(["Recipe", "Baseline commands a fatal move", "Version A commands a fatal move",
                           "Charge for a fatal move", "Charge for the way through"], rows)]
    lines.append(f"Continued baseline against the saved one: {reproduction['runs_identical']} of {reproduction['runs']} "
                 f"runs identical; largest difference in any preference "
                 f"{reproduction['largest_difference_in_any_preference']:.2e}.\n")
    (folder / "confirmation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    chosen = commands.add_parser("select")
    chosen.add_argument("--baselines", required=True)
    judged = commands.add_parser("judge")
    judged.add_argument("--label", required=True)
    args = parser.parse_args()
    (do_select if args.command == "select" else do_judge)(args)


if __name__ == "__main__":
    main()
