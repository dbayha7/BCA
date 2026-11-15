"""Datasets built the way the paper describes them (PROTOCOL.md, Amendment 2). No move is left out.

Recipes:
  optimal        episodes of the optimal policy from the start cell
  other_starts   the same, each episode starting at a cell drawn evenly from the open cells
  mixed          half the rows from the optimal policy, half from a policy that commands one of
                 the five moves at random; both from the start cell

A row records the commanded move as the preference vector that commands it with certainty. The
published 10% noise then decides which move is executed. An episode ends in the goal or in lava
(`done`). It is cut after 200 steps or where its block of the dataset is full (`cut`); a cut row
keeps its real next cell and is not an ending.

With the same seed, the optimal half of `mixed` is the first half of `optimal`.
"""
import hashlib

import numpy as np

import critical_env as E

ROWS, RANDOM_SHARE = 20_000, 0.5
RECIPES = ("optimal", "other_starts", "mixed")
OPTIMAL, RANDOM = 0, 1                      # which policy commanded a row
FIELDS = ("s", "a", "r", "s2", "done", "command", "executed", "source", "episode", "step", "cut", "row_id")
TYPES = {"s": np.int32, "a": np.float32, "r": np.float32, "s2": np.int32, "done": np.float32, "command": np.int8,
         "executed": np.int8, "source": np.int8, "episode": np.int32, "step": np.int32, "cut": np.bool_,
         "row_id": np.int32}
_ROLLED = ("s", "r", "s2", "done", "command", "executed", "source", "episode", "step", "cut")
_BEST = {}


def optimal_moves(task):
    """The move the optimal policy commands at every cell. Ties go to the first of stay, up, down,
    left, right, so the policy is one fixed route."""
    key = (task.name, task.noise, task.grid)
    if key not in _BEST:
        _BEST[key] = task.best_policy()[0]
    return _BEST[key].copy()


def parts(recipe):
    """The blocks of a recipe, in the order their rows are stored: (source, starts anywhere, rows)."""
    random_rows = round(ROWS * RANDOM_SHARE)
    return {"optimal": ((OPTIMAL, False, ROWS),),
            "other_starts": ((OPTIMAL, True, ROWS),),
            "mixed": ((OPTIMAL, False, ROWS - random_rows), (RANDOM, False, random_rows))}[recipe]


def roll(task, seed, source, anywhere, rows):
    """`rows` consecutive rows of episodes from one random stream. The stream depends only on the
    seed, the policy and the kind of start, never on how many rows are asked for."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, source, int(anywhere)]))
    moves, free = optimal_moves(task), task.free_cells()
    execution = {int(cell): task.execution(cell) for cell in free}
    block = {name: [] for name in _ROLLED}
    episode = 0
    while len(block["s"]) < rows:
        cell = int(rng.choice(free)) if anywhere else task.start
        for step in range(E.HORIZON):
            command = int(moves[cell]) if source == OPTIMAL else int(rng.integers(E.N_MOVES))
            executed = int(rng.choice(E.N_MOVES, p=execution[cell][command]))
            nxt, reward, done = task.outcome(cell, executed)
            cut = not done and (step == E.HORIZON - 1 or len(block["s"]) == rows - 1)
            for name, value in zip(_ROLLED, (cell, reward, nxt, float(done), command, executed, source, episode,
                                             step, cut)):
                block[name].append(value)
            if done or cut:
                break
            cell = nxt
        episode += 1
    return block


def generate(task, recipe, seed):
    blocks, first_episode = [], 0
    for source, anywhere, rows in parts(recipe):
        block = roll(task, seed, source, anywhere, rows)
        block["episode"] = [number + first_episode for number in block["episode"]]
        first_episode = block["episode"][-1] + 1
        blocks.append(block)
    data = {name: np.concatenate([np.asarray(block[name]) for block in blocks]) for name in _ROLLED}
    data["a"] = E.CORNERS[data["command"].astype(int)]
    data["row_id"] = np.arange(len(data["s"]))
    return {name: np.asarray(data[name], TYPES[name]) for name in FIELDS}


def digest(data):
    h = hashlib.sha256()
    for name in FIELDS:
        array = np.ascontiguousarray(data[name])
        h.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return h.hexdigest()
