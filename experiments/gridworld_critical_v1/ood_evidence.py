"""Does the published map present an out-of-distribution (OOD) action situation? Reads only; trains nothing.

    python -I -B ood_evidence.py --baselines published_validation_2

For every saved baseline (TD3+BC, alpha as recorded) it sorts each (cell, commanded move) pair at an open
cell by how many training rows command that move at that cell, and compares the critic with the truth:

    signed error = critic's value for the command - true value of commanding that move once at that cell
                   and then following the same saved policy            (positive = overestimation)

The truth is exact: the map is small enough to solve. A pair is "never commanded at a cell the data
visit" when the cell has VISITED_ROWS or more training rows and none of them commands that move. That is
the out-of-distribution action at an in-distribution state. Results are averaged over the training seeds
of a dataset first; the spread is then given across datasets, which are the independent units.
Writes results/ood_evidence_<baselines>.json and prints the tables.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import critical_env as E  # noqa: E402
import published_checks as checks  # noqa: E402
import published_collect as C  # noqa: E402

VISITED_ROWS = 16            # a cell with this many training rows counts as visited by the data
TOO_HIGH = 0.10              # the critic "overestimates" a pair when its error is above this
BUCKETS = ("commanded 1,000+ times", "commanded 16 to 999 times", "commanded 1 to 15 times",
           "never commanded, at a cell the data visit", "never commanded, at a cell the data barely visit")


def bucket_of(rows_for_pair, rows_at_cell):
    if rows_for_pair >= 1000:
        return 0
    if rows_for_pair >= 16:
        return 1
    if rows_for_pair >= 1:
        return 2
    return 3 if rows_at_cell >= VISITED_ROWS else 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", required=True, help="results label of a finished baselines run")
    args = parser.parse_args()
    folder = HERE / "results" / args.baselines
    start = json.loads((folder / "START.json").read_text())
    task = E.published()
    free = np.array(sorted(task.free_cells()))
    lava_moves = {(cell, move) for cell in free for move in range(E.N_MOVES)
                  if (cell // E.SIZE + E.DELTA[move][0], cell % E.SIZE + E.DELTA[move][1]) in
                  {(l // E.SIZE, l % E.SIZE) for l in task.lava}}
    saved = json.loads((folder / f"baselines_{task.name}.json").read_text())["runs"]
    arrays = np.load(folder / f"baselines_{task.name}.npz", allow_pickle=False)
    per_dataset, fatal = {}, {}
    for run in saved:
        recipe, seed, i = run["recipe"], run["data_seed"], run["index"]
        data = checks.load(HERE / "data" / start["data_label"] / checks.file_name(task, recipe, seed))
        counts = np.zeros((E.SIZE * E.SIZE, E.N_MOVES), int)
        np.add.at(counts, (data["s"], data["command"].astype(int)), 1)
        commands, q1 = arrays["commands"][i], arrays["q1"][i]
        truth = task.command_values(task.value(commands))
        error = q1 - truth
        rows = {b: [] for b in range(len(BUCKETS))}
        for cell in free:
            at_cell = counts[cell].sum()
            for move in range(E.N_MOVES):
                rows[bucket_of(counts[cell, move], at_cell)].append((error[cell, move], truth[cell, move], q1[cell, move]))
        entry = per_dataset.setdefault((recipe, seed), {b: [] for b in range(len(BUCKETS))})
        for b, values in rows.items():
            v = np.array(values) if values else np.zeros((0, 3))
            entry[b].append({"pairs": len(values),
                             "mean_error": float(v[:, 0].mean()) if len(values) else None,
                             "share_too_high": float((v[:, 0] > TOO_HIGH).mean()) if len(values) else None,
                             "mean_truth": float(v[:, 1].mean()) if len(values) else None,
                             "mean_critic": float(v[:, 2].mean()) if len(values) else None})
        f = fatal.setdefault((recipe, seed), [])
        pairs = sorted(lava_moves)
        f.append({"pairs": len(pairs),
                  "never_commanded": int(sum(counts[c, m] == 0 for c, m in pairs)),
                  "rows_commanding_one": int(sum(counts[c, m] for c, m in pairs)),
                  "mean_error_never_commanded": float(np.mean([error[c, m] for c, m in pairs if counts[c, m] == 0]))
                  if any(counts[c, m] == 0 for c, m in pairs) else None,
                  "mean_truth": float(np.mean([truth[c, m] for c, m in pairs]))})

    def across(values):
        values = [v for v in values if v is not None]
        return None if not values else {"mean": float(np.mean(values)), "lowest": float(min(values)),
                                        "highest": float(max(values)), "datasets": len(values)}

    result = {"baselines": args.baselines, "alpha": start.get("alpha"), "visited_rows": VISITED_ROWS,
              "too_high": TOO_HIGH, "recipes": {}}
    for recipe in C.RECIPES:
        seeds = sorted(s for r, s in per_dataset if r == recipe)
        table = {}
        for b, name in enumerate(BUCKETS):
            def per_seed(key):
                out = []
                for s in seeds:
                    values = [run[key] for run in per_dataset[(recipe, s)][b] if run[key] is not None]
                    out.append(float(np.mean(values)) if values else None)
                return out
            table[name] = {key: across(per_seed(key)) for key in ("pairs", "mean_error", "share_too_high",
                                                                 "mean_truth", "mean_critic")}
        moves = {key: across([float(np.mean([run[key] for run in fatal[(recipe, s)] if run[key] is not None]))
                              if any(run[key] is not None for run in fatal[(recipe, s)]) else None for s in seeds])
                 for key in ("pairs", "never_commanded", "rows_commanding_one", "mean_error_never_commanded", "mean_truth")}
        result["recipes"][recipe] = {"datasets": seeds, "by_support": table, "moves_into_lava": moves}
        print(f"\n## {recipe}  (datasets {seeds}, three training seeds each)\n")
        print("| Support in the training data | Pairs | Critic's signed error, mean (range across datasets) | "
              "Share too high by more than 0.10 | True value, mean | Critic's value, mean |")
        print("|---|---:|---|---:|---:|---:|")
        for name in BUCKETS:
            t = table[name]
            if t["pairs"] is None or t["pairs"]["mean"] == 0 or t["mean_error"] is None:
                print(f"| {name} | 0 | | | | |")
                continue
            e = t["mean_error"]
            print(f"| {name} | {t['pairs']['mean']:.0f} | {e['mean']:+.3f} ({e['lowest']:+.3f} to {e['highest']:+.3f}) | "
                  f"{100 * t['share_too_high']['mean']:.0f}% | {t['mean_truth']['mean']:.3f} | {t['mean_critic']['mean']:.3f} |")
        m = moves
        print(f"\nMoves into lava: {m['pairs']['mean']:.0f} (cell, move) pairs. Never commanded in the training data: "
              f"{m['never_commanded']['mean']:.1f} of them. Training rows that command one: {m['rows_commanding_one']['mean']:.0f}. "
              + (f"Critic's error at the never-commanded ones: {m['mean_error_never_commanded']['mean']:+.3f} "
                 f"(true value {m['mean_truth']['mean']:.3f})." if m["mean_error_never_commanded"] else ""))
    (HERE / "results" / f"ood_evidence_{args.baselines}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
