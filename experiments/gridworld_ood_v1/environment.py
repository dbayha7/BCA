"""GridWorld out-of-distribution-action testbed, revision 1: the environment.

Grid, rewards, hazards, goal, discount and the four-preference command are the same
as the archived adapted GridWorld
(analysis/gridworld_stress_20261107/archive/td3_gridworld/environment.py).

Three things are different. README.md gives the reason for each.

1. Execution cutoff. Movement probabilities are still softmax(2 * command), but a
   direction that receives less than CUTOFF is never executed and the rest are
   renormalized. Under the archived rule every direction keeps at least 0.61% per
   step, so no move at a visited cell can ever be absent from the data.
2. Bottleneck opening on row 5 instead of row 4, so neither side of the opening
   sits next to a hazard.
3. Bottleneck low-yield exit: cell 44, directly above B, ends the episode with a
   small reward. It gives a learner that will not cross something else to choose.

Setting cutoff=0 and opening_row=4 with no low exit reproduces the archived
environment exactly; test_environment.py checks that against the archived file.

Cell id is 10*y + x, x rightward, y downward. Direction order is UP, RIGHT, DOWN,
LEFT, which is also the order of the four command preferences.
"""
import itertools
from dataclasses import dataclass

import numpy as np

SIZE = 10
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTION_NAMES = ("up", "right", "down", "left")
_DELTA = ((0, -1), (1, 0), (0, 1), (-1, 0))

SHARED_WALLS = frozenset({16, 26, 36, 62, 63, 64})
HAZARDS = frozenset({21, 34, 58, 86})
GOAL = 9
A_START = 90
STEP_REWARD, GOAL_REWARD, HAZARD_REWARD = -0.1, 10.0, -5.0
GAMMA = 0.95
HORIZON = 200
BETA = 2.0                 # execution sharpness, unchanged from the archive
CUTOFF = 0.05              # new: a direction given less than this is never executed
COMMAND_TOLERANCE = 1e-7

# Honeypot: the hazard whose entry is withheld from the targeted data.
UNSEEN_HAZARD = 34
# Bottleneck: opening row, the cell before it (B), the opening (GATE), the cell after (C).
OPENING_ROW = 5
B, GATE, C = 54, 55, 56
LOW_EXIT = 44
LOW_EXIT_REWARDS = (4.5, 3.5, 2.5)   # how tempting the low exit is; the goal pays 10

CORNERS = np.array(list(itertools.product([-1.0, 1.0], repeat=4)))


@dataclass(frozen=True)
class Task:
    name: str
    walls: frozenset
    low_exit: int | None = None
    low_exit_reward: float | None = None
    beta: float = BETA
    cutoff: float = CUTOFF
    hazards: frozenset = HAZARDS
    bump_reward: float = STEP_REWARD      # reward for a move that is blocked by a wall or the grid edge

    # ---- layout -------------------------------------------------------------------
    @property
    def terminals(self):
        extra = frozenset() if self.low_exit is None else frozenset({self.low_exit})
        return self.hazards | {GOAL} | extra

    def free_cells(self):
        """Cells an agent can stand on and act from."""
        closed = self.walls | self.terminals
        return [cell for cell in range(SIZE * SIZE) if cell not in closed]

    def move(self, cell, direction):
        """One executed direction: (next cell, reward, terminal)."""
        if cell in self.terminals:
            return cell, 0.0, True
        x, y = cell % SIZE, cell // SIZE
        dx, dy = _DELTA[direction]
        nxt = min(SIZE - 1, max(0, y + dy)) * SIZE + min(SIZE - 1, max(0, x + dx))
        if nxt in self.walls:
            nxt = cell
        if nxt == GOAL:
            return nxt, GOAL_REWARD, True
        if nxt in self.hazards:
            return nxt, HAZARD_REWARD, True
        if nxt == self.low_exit:
            return nxt, float(self.low_exit_reward), True
        if nxt == cell:
            return nxt, self.bump_reward, False
        return nxt, STEP_REWARD, False

    def role(self, cell, direction):
        """What executing this direction from this cell does."""
        nxt, _, _ = self.move(cell, direction)
        if nxt == GOAL:
            return "goal"
        if nxt in self.hazards:
            return "hazard"
        if nxt == self.low_exit:
            return "low_exit"
        if nxt == cell:
            return "blocked"   # wall or grid edge: stay put and pay the step cost
        return "open"

    # ---- execution ------------------------------------------------------------------
    def probabilities(self, command):
        """Movement probabilities for one command, or for a batch of commands."""
        a = np.asarray(command, dtype=float)
        if a.shape[-1:] != (4,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1 + COMMAND_TOLERANCE):
            raise ValueError("a command is four finite preferences, each in [-1, 1]")
        logits = self.beta * a
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = weights / weights.sum(axis=-1, keepdims=True)
        if self.cutoff > 0:
            probs = np.where(probs >= self.cutoff, probs, 0.0)
            probs = probs / probs.sum(axis=-1, keepdims=True)
        return probs

    def could_execute(self, command, direction):
        """True if this command gives this direction any chance of being executed."""
        return bool(self.probabilities(command)[direction] > 0)

    def sample_direction(self, command, u):
        """Pick the executed direction from one uniform draw u in [0, 1)."""
        probs = self.probabilities(command)
        direction = int(np.searchsorted(np.cumsum(probs), u, side="right"))
        if direction > 3 or probs[direction] == 0:      # rounding at the very top of the range
            direction = int(np.flatnonzero(probs)[-1])
        return direction

    # ---- exact evaluation of one command per cell ---------------------------------------
    def chain(self, commands):
        """Markov chain induced by a table of commands, one per cell."""
        n = SIZE * SIZE
        P, reward = np.zeros((n, n)), np.zeros(n)
        entering = {"goal": np.zeros(n), "hazard": np.zeros(n), "low_exit": np.zeros(n)}
        for cell in self.free_cells():
            for direction, prob in enumerate(self.probabilities(commands[cell])):
                if prob == 0:
                    continue
                nxt, r, done = self.move(cell, direction)
                reward[cell] += prob * r
                if done:
                    entering[self.role(cell, direction)][cell] += prob
                else:
                    P[cell, nxt] += prob
        return P, reward, entering

    def outcomes(self, commands, horizon=HORIZON):
        """Exact probability, per start cell, of each way an episode ends within `horizon` steps."""
        P, _, entering = self.chain(commands)
        reached = {name: np.zeros(SIZE * SIZE) for name in entering}
        for _ in range(horizon):
            reached = {name: entering[name] + P @ reached[name] for name in entering}
        reached["timeout"] = 1.0 - sum(reached.values())
        return reached

    def value(self, commands, gamma=GAMMA):
        """Exact discounted value of following `commands` forever."""
        P, reward, _ = self.chain(commands)
        return np.linalg.solve(np.eye(SIZE * SIZE) - gamma * P, reward)

    def direction_values(self, value, gamma=GAMMA):
        """Value of each executed direction at each cell, given a continuation value."""
        out = np.zeros((SIZE * SIZE, 4))
        for cell in self.free_cells():
            for direction in range(4):
                nxt, r, done = self.move(cell, direction)
                out[cell, direction] = r + (0.0 if done else gamma * value[nxt])
        return out

    def command_value(self, cell, command, value, gamma=GAMMA):
        """Value of issuing `command` once at `cell`, then following the policy behind `value`."""
        return float(self.probabilities(command) @ self.direction_values(value, gamma)[cell])


def best_policy(task, forbidden=None, gamma=GAMMA, sweeps=5000, tol=1e-12):
    """Best corner command per cell, and its exact value.

    `forbidden` maps a cell to directions that must have no chance of being executed
    there. It describes a demonstrator who never attempts those moves.
    """
    forbidden = forbidden or {}
    corner_probs = task.probabilities(CORNERS)
    allowed = {cell: [i for i in range(len(CORNERS))
                      if not any(corner_probs[i, d] > 0 for d in forbidden.get(cell, ()))]
               for cell in task.free_cells()}
    value = np.zeros(SIZE * SIZE)
    commands = np.zeros((SIZE * SIZE, 4))
    for _ in range(sweeps):
        previous = value.copy()
        per_direction = task.direction_values(previous, gamma)
        for cell, options in allowed.items():
            scores = corner_probs[options] @ per_direction[cell]
            best = int(np.argmax(scores))
            value[cell], commands[cell] = scores[best], CORNERS[options[best]]
        if np.max(np.abs(value - previous)) < tol:
            break
    return commands, task.value(commands, gamma)


# ---- the two tasks ----------------------------------------------------------------------
def honeypot(cutoff=CUTOFF):
    return Task("honeypot", SHARED_WALLS, cutoff=cutoff)


def bottleneck(low_exit_reward=LOW_EXIT_REWARDS[1], cutoff=CUTOFF, opening_row=OPENING_ROW, low_exit=LOW_EXIT,
               hazards=HAZARDS, bump_reward=STEP_REWARD, prefix="bottleneck"):
    """Dividing wall in column 5 with one opening. Pass low_exit=None for no low exit.

    The defaults are revision 1, kept so its datasets and receipts stay checkable.
    """
    walls = SHARED_WALLS | {10 * y + 5 for y in range(SIZE) if y != opening_row}
    if low_exit is None:
        return Task(f"{prefix}_row{opening_row}", frozenset(walls), cutoff=cutoff, hazards=frozenset(hazards),
                    bump_reward=bump_reward)
    name = f"{prefix}_exit{int(round(10 * low_exit_reward)):02d}"
    return Task(name, frozenset(walls), low_exit, float(low_exit_reward), cutoff=cutoff,
                hazards=frozenset(hazards), bump_reward=bump_reward)


# Bottleneck revision 2. Two faults found with baseline-only runs are fixed (see PROTOCOL.md, Amendment 2):
#   hazard 58 sat on the opening's row, two cells past C, so policies that kept stepping right walked into it;
#   a move blocked by a wall cost the same as a step, so stalling against a wall looked almost as good as
#   the best move, and under the cutoff rule a chosen stall never ends.
BOTTLENECK_R2_HAZARDS = HAZARDS - {58}
BOTTLENECK_R2_BUMP_REWARD = -1.0


def bottleneck_r2(low_exit_reward=LOW_EXIT_REWARDS[1], cutoff=CUTOFF):
    return bottleneck(low_exit_reward, cutoff=cutoff, hazards=BOTTLENECK_R2_HAZARDS,
                      bump_reward=BOTTLENECK_R2_BUMP_REWARD, prefix="bottleneck_r2")


# ---- what a learner is given ---------------------------------------------------------------
def one_hot(cells):
    """The archived Honeypot host's input: float32 one-hot of length 100."""
    return np.eye(SIZE * SIZE, dtype=np.float32)[np.asarray(cells)]


def coordinates(cells):
    """The archived Bottleneck host's input: float32 [2x/9 - 1, 2y/9 - 1]."""
    cells = np.asarray(cells)
    return np.stack([2 * (cells % SIZE) / 9 - 1, 2 * (cells // SIZE) / 9 - 1], axis=-1).astype(np.float32)
