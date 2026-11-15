"""Exploratory, not registered, trains nothing: how well does plain copying of the data do?

    python -I -B cloning_check.py --label published

For each saved dataset, the copying policy commands at every cell the move the data command most
often there, with certainty, and a random move at a cell with no rows. This is the tabular
behavior cloning of the paper, evaluated exactly. It separates "the data are not enough" from
"the learner's critic misleads it". Writes results/<label>_cloning_check.json.
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


def copying_policy(data):
    """One preference vector per cell: the corner of the most commanded move, zeros where no row is."""
    counts = np.zeros((E.SIZE * E.SIZE, E.N_MOVES), int)
    np.add.at(counts, (data["s"].astype(int), data["command"].astype(int)), 1)
    policy = E.CORNERS[counts.argmax(axis=1)].copy()
    policy[counts.sum(axis=1) == 0] = 0.0          # equal preferences: one of the five moves at random
    return policy, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / "receipts" / f"{args.label}_manifest.json").read_text())
    task = E.published()
    moves, _ = task.best_policy()
    best_goal = float(task.outcomes(E.policy_from_moves(moves))["goal"][task.start])
    free = task.free_cells()
    rows = []
    for entry in manifest["datasets"]:
        data = checks.load(HERE / "data" / args.label / entry["file"])
        assert C.digest(data) == entry["array_sha256"], entry["file"]
        policy, counts = copying_policy(data)
        out = task.outcomes(policy)
        seen = free[counts[free].sum(axis=1) > 0]
        rows.append({"recipe": entry["recipe"], "seed": entry["seed"],
                     "score": float(out["goal"][task.start] / best_goal),
                     "goal": float(out["goal"][task.start]), "lava": float(out["lava"][task.start]),
                     "timeout": float(out["timeout"][task.start]),
                     "cells_with_rows": int(len(seen)),
                     "cells_with_rows_where_the_copied_move_is_optimal": int((counts[seen].argmax(axis=1) == moves[seen]).sum())})
    (HERE / "results" / f"{args.label}_cloning_check.json").write_text(json.dumps(
        {"note": "exploratory, not registered, no training", "optimal_goal_rate": best_goal, "datasets": rows}, indent=1))
    for recipe in C.RECIPES:
        mine = [r for r in rows if r["recipe"] == recipe]
        print(f"{recipe:13s} copying the data: score mean {np.mean([r['score'] for r in mine]):.3f} "
              f"lowest {min(r['score'] for r in mine):.3f} | lava {100 * np.mean([r['lava'] for r in mine]):.1f}% "
              f"| timeout {100 * np.mean([r['timeout'] for r in mine]):.1f}% | cells with rows "
              f"{np.mean([r['cells_with_rows'] for r in mine]):.1f} of {len(free)}, copied move optimal at "
              f"{np.mean([r['cells_with_rows_where_the_copied_move_is_optimal'] for r in mine]):.1f}")


if __name__ == "__main__":
    main()
