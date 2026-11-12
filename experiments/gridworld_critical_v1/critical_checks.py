"""Acceptance checks for a dataset. A dataset that fails is rejected, never repaired.

    python -I -B critical_checks.py --label dev
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import critical_collect as C  # noqa: E402
import critical_env as E  # noqa: E402

MIN_SUPPORT = 16
SHARPNESS_VALUES = (2.0, 3.0, 4.0, 6.0)      # 2 is the original; the others are Amendment 1's candidates
TASKS = {task.name: task for task in (E.published(noise, beta) for beta in SHARPNESS_VALUES
                                      for noise in (E.PUBLISHED_NOISE, 0.0))}
ORIGINAL_TASKS = ("single_critical_noise10", "single_critical_noise00")


# These two count how often moves appear at the critical cell. They protect training data. They do
# not apply to held-out rows, which are only used to calibrate and to report (PROTOCOL.md, defect 1).
SUPPORT_COUNT_CHECKS = ("at_least_two_other_moves_are_well_supported", "moves_not_left_out_here_are_in_the_data")


def acceptable_as_held_out(report):
    """Held-out rows must pass every check except the two support counts."""
    return all(ok for name, ok in report["checks"].items() if name not in SUPPORT_COUNT_CHECKS)


def accident_limit(task, rows):
    """Most times a left-out move may happen by accident at one cell: six standard deviations above
    what the published noise gives. Zero without noise."""
    if task.noise == 0:
        return 0
    p = task.noise / len(task.legal_moves(C.DECISION_CELL))
    return math.ceil(rows * p + 6.0 * math.sqrt(rows * p * (1.0 - p)))


def check_dataset(task, arm, data, complete=None):
    """Returns {"passed", "checks": {name: bool}, "counts": {...}}. `complete` is the complete-data
    arm with the same seed, needed to check that the arms differ only at the critical cell."""
    gap = C.withheld(task, arm)
    cell = C.DECISION_CELL
    here = data["s"] == cell
    free = task.free_cells()
    checks, counts = {}, {}

    per_cell = np.bincount(data["s"], minlength=100)
    checks["every_free_cell_has_256_rows_and_no_other_cell_has_any"] = bool(
        np.all(per_cell[free] == C.ROWS_PER_CELL) and per_cell.sum() == C.ROWS_PER_CELL * len(free))

    chance = task.command_probabilities(data["a"])
    follows = True
    for i in range(len(data["s"])):
        s, command, executed = int(data["s"][i]), int(data["command"][i]), int(data["executed"][i])
        if s not in task._legal or chance[i, command] <= 0 or task.execution(s)[command, executed] <= 0:
            follows = False
            break
        nxt, reward, done = task.outcome(s, executed)
        if (nxt, reward, float(done)) != (int(data["s2"][i]), float(data["r"][i]), float(data["done"][i])):
            follows = False
            break
    checks["every_row_follows_the_true_environment"] = follows

    blocked = gap.get(cell, ())
    could = {E.MOVE_NAMES[m]: int((chance[here, m] > 0).sum()) for m in range(E.N_MOVES)}
    executed = {E.MOVE_NAMES[m]: int((data["executed"][here] == m).sum()) for m in range(E.N_MOVES)}
    counts.update(rows_at_the_critical_cell=int(here.sum()), rows_that_could_command=could, times_executed=executed,
                  left_out=[E.MOVE_NAMES[m] for m in blocked],
                  mean_redraws_at_the_critical_cell=float(data["redraws"][here].mean()))
    checks["no_row_could_command_a_left_out_move"] = all(could[E.MOVE_NAMES[m]] == 0 for m in blocked)
    limit = accident_limit(task, int(here.sum()))
    counts["accident_limit"] = limit
    checks["left_out_moves_happen_no_more_than_the_noise_allows"] = all(executed[E.MOVE_NAMES[m]] <= limit for m in blocked)
    others = [m for m in range(E.N_MOVES) if m not in blocked]
    checks["at_least_two_other_moves_are_well_supported"] = sum(executed[E.MOVE_NAMES[m]] >= MIN_SUPPORT for m in others) >= 2
    present = [m for test in C.TARGETS.values() for m in test if m not in blocked]
    checks["moves_not_left_out_here_are_in_the_data"] = all(
        could[E.MOVE_NAMES[m]] >= MIN_SUPPORT and executed[E.MOVE_NAMES[m]] >= MIN_SUPPORT for m in present)
    checks["commands_are_redrawn_only_at_the_critical_cell"] = bool(np.all(data["redraws"][~here] == 0))
    if arm == "complete":
        checks["nothing_is_redrawn_in_the_complete_arm"] = bool(np.all(data["redraws"] == 0))
    elif complete is not None:
        elsewhere, base = ~here, complete["s"] != cell
        checks["identical_to_the_complete_arm_away_from_the_critical_cell"] = all(
            np.array_equal(data[name][elsewhere], complete[name][base]) for name in C.FIELDS if name != "row_id")
    return {"passed": all(checks.values()), "checks": checks, "counts": counts}


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in C.FIELDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / "receipts" / f"{args.label}_manifest.json").read_text())
    failures = 0
    for entry in manifest["datasets"]:
        task = TASKS[entry["task"]]
        data = load(HERE / "data" / args.label / entry["file"])
        ok = C.digest(data) == entry["array_sha256"]
        complete = None
        if entry["arm"] != "complete":
            complete = load(HERE / "data" / args.label / f"{entry['task']}_complete_seed{entry['seed']}.npz")
        report = check_dataset(task, entry["arm"], data, complete)
        failures += not (ok and report["passed"])
        print(f"{entry['file']:52s} hash {'ok' if ok else 'MISMATCH'}  checks {'pass' if report['passed'] else 'FAIL'}")
    print("all pass" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
