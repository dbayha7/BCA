"""Play episodes move by move in the Honeypot, the Bottleneck and the published gridworld.

NumPy only. No learner code is imported, and no environment file is changed.

Two ways to play, which EPISODE_CHECK.md (mark E1) requires to agree exactly on the same draws:

    play_one   one episode, one step at a time, calling the environment's own functions.
    play       many episodes at once from tables built once from those same functions.

A "world" wraps one task and knows how to make one move:

    InHouseWorld    Honeypot or Bottleneck. One draw per move: the environment's `sample_direction`
                    picks the executed direction, its `move` gives the result.
    PublishedWorld  the published gridworld. Three draws per move, following the rule as the
                    environment file states it: draw the commanded move; with chance `noise` draw
                    the executed move evenly from the legal moves, otherwise execute the commanded
                    move or stay if it is not legal. This path never touches the execution table
                    that the exact `chain` is built from.

Episodes end at the first terminal cell or after `horizon` moves (200 in both environments).
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _folder in ("gridworld_ood_v1", "gridworld_critical_v1"):
    _path = str(ROOT / "experiments" / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)
import critical_env as cenv  # noqa: E402  (published gridworld)
import environment as env  # noqa: E402  (Honeypot and Bottleneck)

CELLS = 100
GOAL_END, BAD_END, LOW_EXIT_END, TIMEOUT = 0, 1, 2, 3
ENDING_NAMES = ("goal", "bad", "low_exit", "timeout")      # "bad" = a hazard, or lava


def pick(chances, u):
    """Index chosen by one uniform draw u: the first whose running total exceeds u.

    Same rule as environment.Task.sample_direction, including its guard for rounding at the very
    top of the range, written for any number of options."""
    index = int(np.searchsorted(np.cumsum(chances), u, side="right"))
    if index > len(chances) - 1 or chances[index] == 0:
        index = int(np.flatnonzero(chances)[-1])
    return index


def _pick_batch(chances, cum, last, cells, u):
    """The same choice for many episodes: `chances`, `cum`, `last` are per-cell tables."""
    top = chances.shape[1] - 1
    index = (cum[cells] <= u[:, None]).sum(axis=1)
    safe = np.minimum(index, top)
    fallback = (index > top) | (chances[cells, safe] == 0)
    return np.where(fallback, last[cells], safe)


def _tables(chances):
    """Running totals and the last option with any chance, per cell."""
    width = chances.shape[1]
    return np.cumsum(chances, axis=1), width - 1 - np.argmax(chances[:, ::-1] > 0, axis=1)


class InHouseWorld:
    """Honeypot or Bottleneck (experiments/gridworld_ood_v1/environment.py)."""
    draws = 1

    def __init__(self, task):
        self.task, self.gamma, self.horizon = task, env.GAMMA, env.HORIZON
        self.free = np.array(task.free_cells())
        self.next = np.zeros((CELLS, 4), int)
        self.reward = np.zeros((CELLS, 4))
        self.done = np.zeros((CELLS, 4), bool)
        for cell in self.free:
            for direction in range(4):
                self.next[cell, direction], self.reward[cell, direction], self.done[cell, direction] = \
                    task.move(int(cell), direction)
        self.ending_codes = np.full(CELLS, -1)
        self.ending_codes[env.GOAL] = GOAL_END
        self.ending_codes[list(task.hazards)] = BAD_END
        if task.low_exit is not None:
            self.ending_codes[task.low_exit] = LOW_EXIT_END

    def step_one(self, policy, cell, u):
        """One move with the environment's own functions and nothing else."""
        direction = self.task.sample_direction(policy[cell], float(u[0]))
        return self.task.move(cell, direction)

    def prepare(self, policy):
        chances = self.task.probabilities(policy)
        return (chances, *_tables(chances))

    def step_batch(self, prepared, cells, u):
        direction = _pick_batch(*prepared, cells, u[:, 0])
        return self.next[cells, direction], self.reward[cells, direction], self.done[cells, direction]

    def allowed(self, policy):
        """[from, to]: True where the one-move rule gives that step a chance above zero under this policy."""
        chances = self.task.probabilities(policy)
        out = np.zeros((CELLS, CELLS), bool)
        for cell in self.free:
            for direction in np.flatnonzero(chances[cell] > 0):
                out[cell, self.task.move(int(cell), int(direction))[0]] = True
        return out


class PublishedWorld:
    """The published gridworld (experiments/gridworld_critical_v1/critical_env.py)."""
    draws = 3

    def __init__(self, task):
        self.task, self.gamma, self.horizon = task, cenv.GAMMA, cenv.HORIZON
        self.free = np.array(task.free_cells())
        moves = cenv.N_MOVES
        self.legal = np.zeros((CELLS, moves), bool)
        self.legal_list = np.zeros((CELLS, moves), int)
        self.legal_count = np.ones(CELLS, int)
        self.next = np.zeros((CELLS, moves), int)
        self.reward = np.zeros((CELLS, moves))
        self.done = np.zeros((CELLS, moves), bool)
        for cell in self.free:
            legal = task.legal_moves(int(cell))
            self.legal[cell, list(legal)] = True
            self.legal_list[cell, :len(legal)] = legal
            self.legal_count[cell] = len(legal)
            for move in legal:
                self.next[cell, move], self.reward[cell, move], self.done[cell, move] = task.outcome(int(cell), move)
        self.ending_codes = np.full(CELLS, -1)
        self.ending_codes[task.goal] = GOAL_END
        self.ending_codes[list(task.lava)] = BAD_END

    def step_one(self, policy, cell, u):
        """One move, following the rule in the environment file's own words."""
        command = pick(self.task.command_probabilities(policy[cell]), float(u[0]))
        legal = self.task.legal_moves(cell)
        if u[1] < self.task.noise:                                   # the noise: any legal move, evenly
            move = legal[min(int(u[2] * len(legal)), len(legal) - 1)]
        else:                                                        # the commanded move, or stay
            move = command if command in legal else cenv.STAY
        return self.task.outcome(cell, move)

    def prepare(self, policy):
        chances = self.task.command_probabilities(policy)
        return (chances, *_tables(chances))

    def step_batch(self, prepared, cells, u):
        command = _pick_batch(*prepared, cells, u[:, 0])
        count = self.legal_count[cells]
        slipped = self.legal_list[cells, np.minimum((u[:, 2] * count).astype(int), count - 1)]
        intended = np.where(self.legal[cells, command], command, cenv.STAY)
        move = np.where(u[:, 1] < self.task.noise, slipped, intended)
        return self.next[cells, move], self.reward[cells, move], self.done[cells, move]

    def allowed(self, policy):
        out = np.zeros((CELLS, CELLS), bool)
        for cell in self.free:
            for move in np.flatnonzero(self.task.move_probabilities(int(cell), policy[cell]) > 0):
                out[cell, self.task.destination(int(cell), int(move))] = True
        return out


def play_one(world, policy, start, draws):
    """One episode, one step at a time. `draws` is [horizon, world.draws] uniform numbers.

    Returns (cells visited including the start, reward of each move, ending code)."""
    cell, path, rewards = int(start), [int(start)], []
    for t in range(world.horizon):
        nxt, reward, done = world.step_one(policy, cell, draws[t])
        cell = int(nxt)
        path.append(cell)
        rewards.append(float(reward))
        if done:
            return path, rewards, int(world.ending_codes[cell])
    return path, rewards, TIMEOUT


def play(world, policy, start, n, rng=None, draws=None, keep_paths=False):
    """Play n episodes from `start`.

    Pass `rng` for fresh draws, or `draws` of shape [horizon, n, world.draws] to fix them (episode
    i then uses draws[:, i], whatever the other episodes do).

    Returns a dict: ending [n] (codes above), last_cell [n] (the terminal cell entered, or the cell
    at time-out), length [n] (moves made), ret [n] (discounted return), transitions [100, 100]
    (count of every step from one cell to the next), and with keep_paths also paths [n, horizon + 1]
    (cells visited, -1 after the end) and rewards [n, horizon]."""
    prepared = world.prepare(policy)
    cells = np.full(n, int(start))
    alive = np.arange(n)
    ending = np.full(n, TIMEOUT)
    length = np.zeros(n, int)
    ret = np.zeros(n)
    transitions = np.zeros(CELLS * CELLS, np.int64)
    if keep_paths:
        paths = np.full((n, world.horizon + 1), -1, np.int16)
        paths[:, 0] = start
        rewards = np.zeros((n, world.horizon))
    discount = 1.0
    for t in range(world.horizon):
        if alive.size == 0:
            break
        u = draws[t][alive] if draws is not None else rng.random((alive.size, world.draws))
        here = cells[alive]
        nxt, reward, done = world.step_batch(prepared, here, u)
        transitions += np.bincount(here * CELLS + nxt, minlength=CELLS * CELLS)
        ret[alive] += discount * reward
        length[alive] = t + 1
        cells[alive] = nxt
        if keep_paths:
            paths[alive, t + 1] = nxt
            rewards[alive, t] = reward
        ending[alive[done]] = world.ending_codes[nxt[done]]
        alive = alive[~done]
        discount *= world.gamma
    out = {"ending": ending, "last_cell": cells, "length": length, "ret": ret,
           "transitions": transitions.reshape(CELLS, CELLS)}
    if keep_paths:
        out["paths"], out["rewards"] = paths, rewards
    return out
