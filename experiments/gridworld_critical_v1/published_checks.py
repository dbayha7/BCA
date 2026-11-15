"""Acceptance checks for a dataset built by published_collect.py (PROTOCOL.md, Amendment 2, gate W0).
A dataset that fails is rejected, never repaired.

    python -I -B published_checks.py --label published
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import critical_env as E  # noqa: E402
import published_collect as C  # noqa: E402

SIGMAS = 6.0        # a count may sit this many standard deviations from what chance gives
CRITICAL_CELLS = {"critical": E.CRITICAL, "second_critical": E.SECOND_CRITICAL}
NESTED = ("s", "a", "r", "s2", "done", "command", "executed", "episode", "step")


def critical_spot(task, name):
    """A critical cell, the moves there that land in lava, and the move the optimal policy commands."""
    cell = CRITICAL_CELLS[name]
    fatal = tuple(m for m in task.legal_moves(cell) if task.destination(cell, m) in task.lava)
    return {"cell": cell, "fatal": fatal, "through": int(C.optimal_moves(task)[cell])}


def within(count, expected, variance):
    return bool(abs(count - expected) <= SIGMAS * math.sqrt(variance) + 1e-9)


def by_move(values):
    return {E.MOVE_NAMES[m]: int((values == m).sum()) for m in range(E.N_MOVES)}


def check_dataset(task, recipe, data, optimal=None):
    """Returns {"passed", "checks": {name: bool}, "counts": {...}}. `optimal` is the `optimal`
    recipe with the same seed; `mixed` needs it to show that its optimal half is the same rows."""
    checks, counts = {}, {}
    n = len(data["s"])
    s, s2, command, executed = (data[name].astype(int) for name in ("s", "s2", "command", "executed"))
    episode, step, source = (data[name].astype(int) for name in ("episode", "step", "source"))
    done, cut = data["done"] == 1.0, data["cut"].astype(bool)
    free = [int(cell) for cell in task.free_cells()]
    layout = C.parts(recipe)

    checks["the_dataset_has_20000_rows_in_every_field"] = bool(
        n == C.ROWS and all(len(data[name]) == n for name in C.FIELDS) and np.array_equal(data["row_id"], np.arange(n)))

    moves_ok = bool(np.all((command >= 0) & (command < E.N_MOVES) & (executed >= 0) & (executed < E.N_MOVES)))
    certain = False
    if moves_ok and data["a"].shape == (n, E.N_MOVES):
        chance = task.command_probabilities(data["a"])
        certain = bool(np.array_equal(data["a"], E.CORNERS[command].astype(np.float32))
                       and np.all(chance[np.arange(n), command] == 1.0))
    checks["every_recorded_action_commands_its_move_with_certainty"] = certain

    # Every row is a real step of the published game, and `done` is set by the goal and lava only.
    execution = {cell: task.execution(cell) for cell in free}
    follows, chance_off, off = moves_ok, np.zeros(n), np.zeros(n, bool)
    for i in range(n if moves_ok else 0):
        cell = int(s[i])
        if cell not in execution or execution[cell][command[i], executed[i]] <= 0:
            follows = False
            break
        nxt, reward, ended = task.outcome(cell, int(executed[i]))
        if (nxt, reward, bool(ended)) != (int(s2[i]), float(data["r"][i]), bool(done[i])):
            follows = False
            break
        legal = task.legal_moves(cell)
        intended = int(command[i]) if int(command[i]) in legal else E.STAY
        off[i] = executed[i] != intended
        chance_off[i] = task.noise * (1.0 - 1.0 / len(legal))
    checks["every_row_follows_the_published_game"] = follows
    counts["rows_where_noise_changed_the_move"] = int(off.sum())
    counts["rows_where_noise_was_expected_to"] = float(chance_off.sum())
    checks["the_noise_is_the_published_ten_percent"] = follows and within(
        off.sum(), chance_off.sum(), (chance_off * (1.0 - chance_off)).sum())

    # Episodes: numbered in order, each row starting where the last one ended.
    new = np.r_[True, episode[1:] != episode[:-1]]
    last = np.r_[new[1:], True]
    inside = ~new[1:]
    checks["episodes_are_numbered_in_order_and_steps_count_from_zero"] = bool(
        episode[0] == 0 and np.all(np.isin(np.diff(episode), (0, 1))) and np.all(step[new] == 0)
        and np.all(step[1:][inside] == step[:-1][inside] + 1) and np.all(step < E.HORIZON))
    checks["each_row_starts_where_the_last_one_ended"] = bool(
        np.all(s[1:][inside] == s2[:-1][inside]) and np.all(source[1:][inside] == source[:-1][inside]))
    checks["an_episode_ends_at_the_goal_in_lava_or_at_a_cut_and_nowhere_else"] = bool(
        np.array_equal(done | cut, last) and not np.any(done & cut))
    block_ends = np.cumsum([rows for _, _, rows in layout]) - 1
    checks["a_cut_is_at_200_steps_or_where_a_block_is_full"] = bool(
        np.all((step[cut] == E.HORIZON - 1) | np.isin(np.flatnonzero(cut), block_ends)))

    # Who commanded what.
    expected_source = np.concatenate([np.full(rows, kind) for kind, _, rows in layout])
    checks["the_blocks_are_the_recipe's"] = bool(n == len(expected_source) and np.array_equal(source, expected_source))
    from_optimal = source == C.OPTIMAL
    best = C.optimal_moves(task)
    checks["optimal_rows_command_the_optimal_move"] = bool(np.array_equal(command[from_optimal], best[s[from_optimal]]))
    random_rows = int((~from_optimal).sum())
    counts["random_rows"] = random_rows
    counts["random_commands"] = by_move(command[~from_optimal])
    checks["random_rows_command_the_five_moves_evenly"] = all(
        within(count, random_rows / E.N_MOVES, random_rows * 0.2 * 0.8) for count in counts["random_commands"].values())

    # Where episodes start.
    starts = s[new]
    anywhere = np.concatenate([np.full(rows, far) for _, far, rows in layout])[new] if n == len(expected_source) else None
    if anywhere is None:
        checks["episodes_start_where_the_recipe_says"] = False
    else:
        ok = bool(np.all(starts[~anywhere] == task.start))
        spread = np.bincount(starts[anywhere], minlength=E.SIZE * E.SIZE)[free]
        if anywhere.any():
            episodes, share = int(anywhere.sum()), 1.0 / len(free)
            ok = ok and bool(np.all(np.isin(starts[anywhere], free)) and spread.min() >= 1 and all(
                within(count, episodes * share, episodes * share * (1.0 - share)) for count in spread))
            counts["fewest_and_most_episodes_started_at_one_cell"] = [int(spread.min()), int(spread.max())]
        checks["episodes_start_where_the_recipe_says"] = ok

    if recipe == "mixed" and optimal is not None:
        rows = layout[0][2]
        checks["the_optimal_half_is_the_first_half_of_the_optimal_recipe"] = all(
            np.array_equal(data[name][:rows], optimal[name][:rows]) for name in NESTED)

    visited = np.bincount(s, minlength=E.SIZE * E.SIZE)
    lengths = step[last] + 1
    counts.update(
        episodes=int(new.sum()), mean_episode_length=float(lengths.mean()),
        episodes_ending_in_the_goal=int((done & (s2 == task.goal)).sum()),
        episodes_ending_in_lava=int((done & np.isin(s2, sorted(task.lava))).sum()),
        episodes_cut_at_200_steps=int((cut & (step == E.HORIZON - 1)).sum()),
        episodes_cut_where_a_block_is_full=int((cut & (step != E.HORIZON - 1)).sum()),
        open_cells=len(free), open_cells_with_no_rows=int((visited[free] == 0).sum()),
        open_cells_with_fewer_than_16_rows=int((visited[free] < 16).sum()))
    for name, cell in CRITICAL_CELLS.items():
        here = s == cell
        counts[f"rows_at_the_{name}_cell"] = int(here.sum())
        counts[f"commanded_at_the_{name}_cell"] = by_move(command[here])
        counts[f"executed_at_the_{name}_cell"] = by_move(executed[here])
    return {"passed": all(checks.values()), "checks": checks, "counts": counts}


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in C.FIELDS}


def file_name(task, recipe, seed):
    return f"{task.name}_{recipe}_seed{seed}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / "receipts" / f"{args.label}_manifest.json").read_text())
    task, failures = E.published(), 0
    for entry in manifest["datasets"]:
        folder = HERE / "data" / args.label
        data = load(folder / entry["file"])
        same = C.digest(data) == entry["array_sha256"]
        optimal = load(folder / file_name(task, "optimal", entry["seed"])) if entry["recipe"] == "mixed" else None
        report = check_dataset(task, entry["recipe"], data, optimal)
        failures += not (same and report["passed"] == entry["checks_passed"])
        print(f"{entry['file']:52s} hash {'ok' if same else 'MISMATCH'}  checks {'pass' if report['passed'] else 'FAIL'}")
    print("all as recorded" if failures == 0 else f"{failures} DIFFER FROM THE MANIFEST")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
