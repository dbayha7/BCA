"""Thin data: the withheld move is in the data a few times instead of never (PROTOCOL.md Amendment 11).

A thin dataset is the targeted dataset of the same task and seed with one change. At each decision cell,
the first `k` uniformly random commands that could execute the withheld move are kept instead of redrawn.
Every later one is redrawn, as in the targeted arm. The demonstrator's rows are unchanged: the demonstrator
still never attempts the move.

No outcome decides what is kept. Each command is accepted or redrawn before its outcome is sampled.
Dose 0 is the targeted dataset exactly. collect.py is not modified; this module reuses its streams and rules.
"""
import numpy as np

import collect
import environment as env
import unseen_move_checks as checks

DOSES = (0, 2, 8, 32)


def arm_name(k):
    return f"thin{int(k):03d}"


def dose_of(arm):
    """The dose of a thin arm name, or None for any other arm."""
    return int(arm[4:]) if arm.startswith("thin") and arm[4:].isdigit() else None


def generate(task, seed, k):
    """One thin dataset. Returns (arrays, facts about how it was made)."""
    withheld = collect.targeted_moves(task)
    teacher, replaced = collect.demonstrator_commands(task, withheld)
    base, _ = collect.generate(task, "targeted", seed)
    data = {key: value.copy() for key, value in base.items()}
    data["kept"] = np.zeros(len(base["s"]), bool)       # rows whose command could execute the withheld move
    kept_per_cell, executed_per_cell = {}, {}
    for cell, blocked in withheld.items():
        draw = np.random.default_rng(np.random.SeedSequence([seed, cell, 0]))
        outcome_draws = np.random.default_rng(np.random.SeedSequence([seed, cell, 1])).random(collect.ROWS_PER_CELL)
        rows = np.flatnonzero(base["s"] == cell)
        assert len(rows) == collect.ROWS_PER_CELL and np.all(np.diff(rows) == 1), cell
        kept = executed = 0
        for i, row in enumerate(rows):
            from_demonstrator = i >= collect.UNIFORM_ROWS
            redraws, is_kept = 0, False
            while True:
                if from_demonstrator:
                    command = np.clip(teacher[cell] + draw.normal(0, collect.NOISE_SD, 4), -1, 1).astype(np.float32)
                else:
                    command = draw.uniform(-1, 1, 4).astype(np.float32)
                if not any(task.could_execute(command, d) for d in blocked):
                    break
                if not from_demonstrator and kept < k:
                    kept, is_kept = kept + 1, True
                    break
                redraws += 1
                if redraws > collect.MAX_REDRAWS:
                    raise RuntimeError(f"no acceptable command at cell {cell}")
            direction = task.sample_direction(command, outcome_draws[i])
            nxt, reward, done = task.move(cell, direction)
            executed += int(direction in blocked)
            data["a"][row], data["r"][row, 0], data["s2"][row], data["done"][row, 0] = command, reward, nxt, done
            data["direction"][row], data["redraws"][row], data["kept"][row] = direction, redraws, is_kept
            assert data["s"][row] == cell and bool(data["from_demonstrator"][row]) == from_demonstrator
        kept_per_cell[str(cell)], executed_per_cell[str(cell)] = kept, executed
    facts = {"dose": int(k),
             "withheld_moves": {str(c): [env.DIRECTION_NAMES[d] for d in ds] for c, ds in withheld.items()},
             "demonstrator_commands_replaced": {str(c): v for c, v in replaced.items()},
             "rows": int(len(data["s"])),
             "commands_kept_that_could_execute_the_move": kept_per_cell,
             "times_the_move_was_executed": executed_per_cell}
    return data, facts


def check(task, k, data, targeted):
    """Acceptance checks for one thin dataset. `targeted` is the targeted arm of the same task and seed."""
    s, direction = data["s"], data["direction"]
    probabilities = task.probabilities(data["a"])
    withheld = collect.targeted_moves(task)
    names = env.DIRECTION_NAMES
    free = task.free_cells()
    results = []

    def add(name, passed, **detail):
        results.append({"check": name, "passed": bool(passed), **detail})

    counts = np.bincount(s, minlength=100)
    add("every_free_cell_has_the_full_row_count",
        all(counts[c] == collect.ROWS_PER_CELL for c in free) and counts.sum() == len(free) * collect.ROWS_PER_CELL)
    wrong = 0
    for i in range(len(s)):
        nxt, reward, done = task.move(int(s[i]), int(direction[i]))
        if (nxt != data["s2"][i] or not np.isclose(reward, float(data["r"][i, 0]), atol=1e-6)
                or done != bool(data["done"][i, 0]) or probabilities[i, direction[i]] <= 0):
            wrong += 1
    add("rows_follow_the_true_environment", wrong == 0, rows_that_do_not=wrong)

    outside = ~np.isin(s, sorted(withheld))
    shared = [key for key in targeted if key in data]
    add("identical_to_the_targeted_arm_outside_decision_cells",
        len(s) == len(targeted["s"]) and all(np.array_equal(data[key][outside], targeted[key][outside]) for key in shared),
        rows_compared=int(outside.sum()))
    if k == 0:
        add("dose_zero_is_the_targeted_arm_exactly",
            all(np.array_equal(data[key], targeted[key]) for key in shared) and not data["kept"].any())
    executed = {}
    for cell, moves in withheld.items():
        here = s == cell
        could = np.zeros(int(here.sum()), bool)
        for d in moves:
            could |= probabilities[here, d] > 0
        kept_here = data["kept"][here]
        add("commands_that_could_execute_the_move_are_exactly_the_kept_rows", np.array_equal(could, kept_here),
            cell=int(cell), commands_that_could=int(could.sum()))
        add("the_dose_is_met", int(kept_here.sum()) == k, cell=int(cell), kept=int(kept_here.sum()), dose=int(k))
        add("kept_rows_are_uniform_rows", not (kept_here & data["from_demonstrator"][here]).any(), cell=int(cell))
        executed[cell] = np.bincount(direction[here], minlength=4)
        supported = [names[d] for d in range(4) if d not in moves and executed[cell][d] >= checks.MIN_SUPPORT]
        add("other_moves_at_the_cell_still_supported", len(supported) >= 2, cell=int(cell), supported_moves=supported)
    return {"task": task.name, "arm": arm_name(k), "passed": all(r["passed"] for r in results), "checks": results,
            "times_each_direction_was_executed_at_decision_cells":
                {str(cell): dict(zip(names, c.tolist())) for cell, c in executed.items()}}
