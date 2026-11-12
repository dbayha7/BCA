"""Acceptance checks: is the withheld move truly absent from a dataset?

A move at a cell counts as out of distribution only if all three hold:
  1. no command in the data at that cell could have executed it,
  2. it was never executed there, so its consequence never appears,
  3. other moves at that same cell are still well supported.

A dataset that fails is rejected. It is never repaired by deleting rows.

Run on saved datasets:
    python -I -B unseen_move_checks.py --label dev
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402
import environment as env  # noqa: E402

MIN_SUPPORT = 16    # a move counts as supported at a cell if executed in at least 16 of its 256 rows
HERE = Path(__file__).resolve().parent
TASKS = {task.name: task for task in [env.honeypot()]
         + [env.bottleneck(r) for r in env.LOW_EXIT_REWARDS]          # revision 1, kept for its records
         + [env.bottleneck_r2(r) for r in env.LOW_EXIT_REWARDS]}      # revision 2, the fixed layout


def destination_count(data, cell, destination):
    """Rows that start at `cell` and end at `destination`. Needs only the learner's fields,
    so it also works on archived datasets that did not record the executed direction."""
    return int(((np.asarray(data["s"]) == cell) & (np.asarray(data["s2"]) == destination)).sum())


def _result(name, passed, **detail):
    return {"check": name, "passed": bool(passed), **detail}


def check_dataset(task, arm, data, complete=None):
    """All checks for one dataset. `complete` is the complete-data arm of the same task and seed."""
    s, s2, direction = data["s"], data["s2"], data["direction"]
    probabilities = task.probabilities(data["a"])          # raises on any invalid command
    targeted = collect.targeted_moves(task)
    withheld = collect.withheld_moves(task, arm)
    names = env.DIRECTION_NAMES
    results = []

    counts = np.bincount(s, minlength=100)
    free = task.free_cells()
    results.append(_result(
        "every_free_cell_has_the_full_row_count",
        all(counts[c] == collect.ROWS_PER_CELL for c in free) and counts.sum() == len(free) * collect.ROWS_PER_CELL,
        rows=int(counts.sum()), cells=len(free)))

    wrong = 0
    for i in range(len(s)):
        nxt, reward, done = task.move(int(s[i]), int(direction[i]))
        if (nxt != s2[i] or not np.isclose(reward, float(data["r"][i, 0]), atol=1e-6)
                or done != bool(data["done"][i, 0]) or probabilities[i, direction[i]] <= 0):
            wrong += 1
    results.append(_result("rows_follow_the_true_environment", wrong == 0, rows_that_do_not=wrong))

    executed = {cell: np.bincount(direction[s == cell], minlength=4) for cell in sorted(set(targeted) | set(withheld))}
    for cell, moves in withheld.items():
        here = s == cell
        for d in moves:
            label = f"cell_{cell}_{names[d]}"
            could = int((probabilities[here, d] > 0).sum())
            results.append(_result("no_command_could_execute_the_withheld_move", could == 0,
                                   move=label, commands_that_could=could))
            results.append(_result("withheld_move_never_executed", executed[cell][d] == 0,
                                   move=label, times_executed=int(executed[cell][d])))
            destination = task.move(cell, d)[0]
            results.append(_result("withheld_destination_never_reached_from_the_cell",
                                   destination_count(data, cell, destination) == 0,
                                   move=label, destination=int(destination),
                                   rows=destination_count(data, cell, destination)))
        supported = [names[d] for d in range(4) if d not in moves and executed[cell][d] >= MIN_SUPPORT]
        results.append(_result("other_moves_at_the_cell_still_supported", len(supported) >= 2,
                               cell=int(cell), supported_moves=supported,
                               times_each_direction_was_executed=dict(zip(names, executed[cell].tolist()))))

    if arm == "targeted" and task.name == "honeypot":
        entries = int((s2 == env.UNSEEN_HAZARD).sum())
        results.append(_result("unseen_hazard_never_entered_from_anywhere", entries == 0,
                               hazard=env.UNSEEN_HAZARD, entries=entries))

    if arm in ("complete", "placebo"):
        for cell, moves in targeted.items():
            for d in moves:
                results.append(_result("target_move_is_present_in_this_arm", executed[cell][d] >= MIN_SUPPORT,
                                       move=f"cell_{cell}_{names[d]}", times_executed=int(executed[cell][d])))

    decision_cells = sorted(withheld)
    redrawn_elsewhere = int(((data["redraws"] > 0) & ~np.isin(s, decision_cells)).sum())
    results.append(_result("commands_redrawn_only_at_decision_cells", redrawn_elsewhere == 0,
                           rows_redrawn_elsewhere=redrawn_elsewhere,
                           rows_redrawn=int((data["redraws"] > 0).sum())))

    if arm != "complete":
        if complete is None:
            results.append(_result("identical_to_the_complete_arm_outside_decision_cells", False,
                                   reason="complete arm not supplied"))
        else:
            outside = ~np.isin(s, decision_cells)
            same = len(s) == len(complete["s"]) and all(
                np.array_equal(data[key][outside], complete[key][outside]) for key in data)
            results.append(_result("identical_to_the_complete_arm_outside_decision_cells", same,
                                   rows_compared=int(outside.sum()),
                                   rows_at_decision_cells=int((~outside).sum())))

    return {"task": task.name, "arm": arm, "passed": all(r["passed"] for r in results), "checks": results,
            "times_each_direction_was_executed_at_decision_cells":
                {str(cell): dict(zip(names, counts_.tolist())) for cell, counts_ in executed.items()}}


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="dev")
    args = parser.parse_args()
    manifest = json.loads((HERE / "receipts" / f"{args.label}_manifest.json").read_text())
    folder = HERE / "data" / args.label
    failures = 0
    for entry in manifest["datasets"]:
        task = TASKS[entry["task"]]
        data = load(folder / entry["file"])
        if collect.digest(data) != entry["array_sha256"]:
            print("CONTENT CHANGED", entry["file"])
            failures += 1
            continue
        complete = None
        if entry["arm"] != "complete":
            complete = load(folder / f"{entry['task']}_complete_seed{entry['seed']}.npz")
        report = check_dataset(task, entry["arm"], data, complete)
        failures += not report["passed"]
        print("PASS" if report["passed"] else "FAIL", entry["file"])
    print(f"{len(manifest['datasets'])} datasets checked, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
