"""Datasets for the two tests: every cell gets the same number of rows, and one data arm leaves
a move out at the critical cell. This layer is ours; the published setup has no left-out move.

Data arms:
  complete   nothing is left out
  over       at the critical cell, "left" and "right" (both land in lava) are never commanded
  under      at the critical cell, "down" (the way through) is never commanded
  placebo    at the critical cell, the least useful harmless move is never commanded

A command is "never commanded" by redrawing the preference vector before anything happens, so no
row is deleted afterward and nothing is filtered on its outcome. With the published noise the
left-out move can still happen by accident; the checks count how often.
"""
import hashlib

import numpy as np

import critical_env as E

ROWS_PER_CELL, UNIFORM_ROWS, NOISE_SD = 256, 128, 0.1
ARMS = ("complete", "over", "under", "placebo")
DECISION_CELL = E.CRITICAL
TARGETS = {"over": (E.LEFT, E.RIGHT), "under": (E.DOWN,)}
FIELDS = ("s", "a", "r", "s2", "done", "command", "executed", "from_demonstrator", "redraws", "row_id")


def placebo_move(task):
    """The least useful move at the critical cell that is neither fatal nor the way through."""
    _, value = task.best_policy()
    q = task.command_values(value)[DECISION_CELL]
    candidates = [m for m in range(E.N_MOVES) if m not in TARGETS["over"] + TARGETS["under"]]
    return min(candidates, key=lambda m: (q[m], m))


def withheld(task, arm):
    """{cell: moves never commanded there} for a data arm."""
    if arm == "complete":
        return {}
    if arm == "placebo":
        return {DECISION_CELL: (placebo_move(task),)}
    return {DECISION_CELL: TARGETS[arm]}


def demonstrator_moves(task, arm):
    """The best move at every cell. It changes only at the critical cell, and only when the best
    move there is the one left out; it is then the best move of someone who may not command it."""
    best, _ = task.best_policy()
    gap = withheld(task, arm)
    for cell, moves in gap.items():
        if best[cell] in moves:
            best = best.copy()
            best[cell] = task.best_policy(forbidden=gap)[0][cell]
    return best


def generate(task, arm, seed):
    """256 single-step rows at every free cell: 128 uniform preference vectors, then 128 copies of
    the demonstrator's command with noise of standard deviation 0.1. Each cell has its own random
    streams, so the arms are identical everywhere except at the critical cell."""
    gap, moves = withheld(task, arm), demonstrator_moves(task, arm)
    rows = {name: [] for name in FIELDS}
    for cell in task.free_cells():
        cell = int(cell)
        streams = [np.random.default_rng(np.random.SeedSequence([seed, cell, k])) for k in (0, 1)]
        blocked, execution = gap.get(cell, ()), task.execution(cell)
        for i in range(ROWS_PER_CELL):
            from_demonstrator = i >= UNIFORM_ROWS
            rng, redraws = streams[int(from_demonstrator)], 0
            while True:
                if from_demonstrator:
                    a = np.clip(E.CORNERS[moves[cell]] + rng.normal(0.0, NOISE_SD, E.N_MOVES), -1.0, 1.0)
                else:
                    a = rng.uniform(-1.0, 1.0, E.N_MOVES)
                a = a.astype(np.float32)                 # what is stored is what decides
                chance = task.command_probabilities(a)
                if not any(chance[m] > 0 for m in blocked):
                    break
                redraws += 1
            command = int(rng.choice(E.N_MOVES, p=chance))
            executed = int(rng.choice(E.N_MOVES, p=execution[command]))
            nxt, reward, done = task.outcome(cell, executed)
            for name, value in zip(FIELDS, (cell, a, reward, nxt, done, command, executed, from_demonstrator,
                                            redraws, len(rows["s"]))):
                rows[name].append(value)
    return {"s": np.asarray(rows["s"], np.int32), "a": np.asarray(rows["a"], np.float32),
            "r": np.asarray(rows["r"], np.float32), "s2": np.asarray(rows["s2"], np.int32),
            "done": np.asarray(rows["done"], np.float32), "command": np.asarray(rows["command"], np.int8),
            "executed": np.asarray(rows["executed"], np.int8),
            "from_demonstrator": np.asarray(rows["from_demonstrator"], bool),
            "redraws": np.asarray(rows["redraws"], np.int32), "row_id": np.asarray(rows["row_id"], np.int32)}


def digest(data):
    h = hashlib.sha256()
    for name in FIELDS:
        array = np.ascontiguousarray(data[name])
        h.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return h.hexdigest()
