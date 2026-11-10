"""Data collection with a gap built into the demonstrator.

Every free cell gets 256 single-step rows, as in the archived Honeypot collector:
128 uniform commands, then 128 copies of the best command with small noise.

Three data arms:
  complete  no gap.
  targeted  at each decision cell, a command that could execute the target move is
            never issued. It is redrawn BEFORE its outcome is sampled, so no outcome
            is ever used to decide what stays, and the cell keeps all 256 rows.
  placebo   the same gap around a harmless move at the same cells.

Each cell has its own random streams. Cells without a gap therefore come out
byte-identical in all three arms; unseen_move_checks.py verifies that.
"""
import hashlib

import numpy as np

import environment as env

ROWS_PER_CELL = 256
UNIFORM_ROWS = 128
NOISE_SD = 0.1
ARMS = ("complete", "targeted", "placebo")
MAX_REDRAWS = 100_000
LEARNER_FIELDS = ("s", "a", "r", "s2", "done")


def targeted_moves(task):
    """Cell -> directions withheld in the targeted arm."""
    if task.name == "honeypot":
        return {cell: (d,) for cell in task.free_cells() for d in range(4)
                if task.move(cell, d)[0] == env.UNSEEN_HAZARD}
    return {env.B: (env.RIGHT,)}


def placebo_moves(task):
    """Same cells, a move nobody needs: the lowest-value ordinary step that is not the target."""
    _, value = env.best_policy(task)
    per_direction = task.direction_values(value)
    moves = {}
    for cell, target in targeted_moves(task).items():
        ordinary = [d for d in range(4) if d not in target and task.role(cell, d) == "open"]
        if not ordinary:
            raise ValueError(f"cell {cell} has no ordinary step to use as a placebo")
        moves[cell] = (min(ordinary, key=lambda d: per_direction[cell, d]),)
    return moves


def withheld_moves(task, arm):
    if arm == "complete":
        return {}
    if arm == "targeted":
        return targeted_moves(task)
    if arm == "placebo":
        return placebo_moves(task)
    raise ValueError(arm)


def demonstrator_commands(task, withheld):
    """Best command per cell. Where that command could execute a withheld move, use the
    best command of a demonstrator who never attempts it. Every other cell is unchanged."""
    commands, _ = env.best_policy(task)
    replaced = {}
    if withheld:
        constrained, _ = env.best_policy(task, forbidden=withheld)
        for cell, directions in withheld.items():
            if any(task.could_execute(commands[cell], d) for d in directions):
                replaced[cell] = {"from": commands[cell].tolist(), "to": constrained[cell].tolist()}
                commands[cell] = constrained[cell]
    return commands, replaced


def generate(task, arm, seed):
    """One dataset. Returns (arrays, facts about how it was made)."""
    withheld = withheld_moves(task, arm)
    teacher, replaced = demonstrator_commands(task, withheld)
    rows = []
    for cell in task.free_cells():
        draw = np.random.default_rng(np.random.SeedSequence([seed, cell, 0]))
        outcome_draws = np.random.default_rng(np.random.SeedSequence([seed, cell, 1])).random(ROWS_PER_CELL)
        blocked = withheld.get(cell, ())
        for i in range(ROWS_PER_CELL):
            from_demonstrator = i >= UNIFORM_ROWS
            redraws = 0
            while True:
                if from_demonstrator:
                    command = np.clip(teacher[cell] + draw.normal(0, NOISE_SD, 4), -1, 1).astype(np.float32)
                else:
                    command = draw.uniform(-1, 1, 4).astype(np.float32)
                if not any(task.could_execute(command, d) for d in blocked):
                    break
                redraws += 1
                if redraws > MAX_REDRAWS:
                    raise RuntimeError(f"no acceptable command at cell {cell}")
            direction = task.sample_direction(command, outcome_draws[i])
            nxt, reward, done = task.move(cell, direction)
            rows.append((cell, command, reward, nxt, done, direction, from_demonstrator, redraws))
    data = {"s": np.array([r[0] for r in rows], np.int64),
            "a": np.array([r[1] for r in rows], np.float32).reshape(-1, 4),
            "r": np.array([r[2] for r in rows], np.float32).reshape(-1, 1),
            "s2": np.array([r[3] for r in rows], np.int64),
            "done": np.array([r[4] for r in rows], np.float32).reshape(-1, 1),
            "direction": np.array([r[5] for r in rows], np.int64),
            "from_demonstrator": np.array([r[6] for r in rows], bool),
            "redraws": np.array([r[7] for r in rows], np.int64),
            "row_id": np.arange(len(rows), dtype=np.int64)}
    facts = {"withheld_moves": {str(c): [env.DIRECTION_NAMES[d] for d in ds] for c, ds in withheld.items()},
             "demonstrator_commands_replaced": {str(c): v for c, v in replaced.items()},
             "rows": int(len(rows)),
             "commands_redrawn": int(data["redraws"].sum()),
             "rows_with_a_redraw": int((data["redraws"] > 0).sum())}
    return data, facts


def digest(data):
    """Content hash of the arrays (independent of how the file was written)."""
    h = hashlib.sha256()
    for key in sorted(data):
        value = np.ascontiguousarray(data[key])
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()
