"""The "Single Critical" gridworld of Kumar et al. (ICLR 2022), with continuous preferences on top.

Two layers, kept apart on purpose.

Layer 1, published (PUBLISHED_SETUP.md): the 10 x 10 map, five moves (stay, up, down, left,
right), the noise rule, reward 1 for reaching the goal, lava ends the run with nothing.

Layer 2, ours: the policy outputs five preferences in [-1, 1]. softmax(2 x preferences) is the
chance of commanding each move, and a move given under 5% is never commanded. The published
noise is then applied to the commanded move.

NumPy only. Cells are numbered 10 * row + column, row 0 at the top.
"""
import numpy as np

SIZE = 10
# Read from Figure 7 of the paper by transcribe_figure.py. S start, G goal, L lava, # wall.
SINGLE_CRITICAL = (
    "S......###",
    ".###......",
    ".....#...#",
    "#..#...#..",
    "#.....##..",
    "#........#",
    "#..###L.L#",
    "...L......",
    "#.......##",
    "...L.....G",
)
# Numbered as in the code the paper cites (gridcraft): no-op first.
MOVE_NAMES = ("stay", "up", "down", "left", "right")
STAY, UP, DOWN, LEFT, RIGHT = range(5)
DELTA = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))          # (row change, column change)
N_MOVES = 5

PUBLISHED_NOISE = 0.10      # the paper: "a 10% chance that the agent travels in a different direction"
GOAL_REWARD = 1.0           # the paper: "a reward of 1 upon reaching the goal state"
GAMMA = 0.95                # not stated in the paper; our choice
HORIZON = 200               # not stated in the paper; our choice
BETA, CUTOFF = 2.0, 0.05    # layer 2, ours

# Corner m is the preference vector that commands move m with certainty.
CORNERS = np.where(np.eye(N_MOVES, dtype=bool), 1.0, -1.0)

START, GOAL = 0, 99
CRITICAL = 67               # row 6, column 7: lava on its left and its right
LAVA_LEFT, LAVA_RIGHT = 66, 68
ABOVE, BELOW = 57, 77       # the cells before and after the critical cell on the route
SECOND_CRITICAL = 83        # row 8, column 3: lava above and below; on the other route


class Task:
    """One setting of the environment. `noise` is 0.10 as published, or 0.0 for the no-noise variant."""

    def __init__(self, name, noise, grid=SINGLE_CRITICAL, beta=BETA, cutoff=CUTOFF):
        assert len(grid) == SIZE and all(len(row) == SIZE for row in grid)
        self.name, self.noise, self.grid, self.beta, self.cutoff = name, float(noise), tuple(grid), beta, cutoff
        flat = "".join(grid)
        self.walls = frozenset(i for i, ch in enumerate(flat) if ch == "#")
        self.lava = frozenset(i for i, ch in enumerate(flat) if ch == "L")
        self.goal = flat.index("G")
        self.start = flat.index("S")
        self.terminals = self.lava | {self.goal}
        self._free = np.array([i for i in range(SIZE * SIZE) if i not in self.walls | self.terminals])
        self._legal = {cell: self._find_legal(cell) for cell in range(SIZE * SIZE) if cell not in self.walls}
        self._execution = {cell: self._build_execution(cell) for cell in self._free}

    # ---- layer 1: the published game ------------------------------------------------------------
    def free_cells(self):
        """Cells an episode can be in before it ends: not wall, not lava, not the goal."""
        return self._free.copy()

    def _find_legal(self, cell):
        row, col = divmod(cell, SIZE)
        legal = []
        for move, (dr, dc) in enumerate(DELTA):
            r, c = row + dr, col + dc
            if 0 <= r < SIZE and 0 <= c < SIZE and SIZE * r + c not in self.walls:
                legal.append(move)
        return tuple(legal)

    def legal_moves(self, cell):
        """Moves that do not run into a wall or off the grid. Staying is always legal."""
        return self._legal[cell]

    def destination(self, cell, move):
        dr, dc = DELTA[move]
        return cell + SIZE * dr + dc

    def _build_execution(self, cell):
        """E[commanded, executed]. With probability `noise` the executed move is drawn evenly from
        the legal moves, the commanded one included. Otherwise the commanded move is executed, or
        the agent stays if that move is not legal. (TransitionModel.get_aprobs in the cited code.)"""
        legal = list(self._legal[cell])
        matrix = np.zeros((N_MOVES, N_MOVES))
        matrix[:, legal] = self.noise / len(legal)
        for command in range(N_MOVES):
            matrix[command, command if command in legal else STAY] += 1.0 - self.noise
        return matrix

    def execution(self, cell):
        return self._execution[cell].copy()

    def outcome(self, cell, move):
        """(next cell, reward, done) for an executed legal move from a free cell."""
        nxt = self.destination(cell, move)
        return nxt, (GOAL_REWARD if nxt == self.goal else 0.0), nxt in self.terminals

    # ---- layer 2: continuous preferences ----------------------------------------------------------
    def command_probabilities(self, preferences):
        """Chance of commanding each move. Rejects invalid preferences; never clips them."""
        p = np.asarray(preferences, float)
        if p.shape[-1] != N_MOVES or not np.isfinite(p).all() or np.any(np.abs(p) > 1 + 1e-7):
            raise ValueError("preferences must be finite, of length 5, in [-1, 1]")
        logits = self.beta * p
        soft = np.exp(logits - logits.max(axis=-1, keepdims=True))
        soft = soft / soft.sum(axis=-1, keepdims=True)
        kept = np.where(soft >= self.cutoff, soft, 0.0)
        return kept / kept.sum(axis=-1, keepdims=True)

    def could_command(self, preferences, move):
        return self.command_probabilities(preferences)[..., move] > 0

    def move_probabilities(self, cell, preferences):
        """Chance of each move being executed at a cell: the command chance, then the noise."""
        return self.command_probabilities(preferences) @ self._execution[cell]

    # ---- exact evaluation of a policy (one preference vector per cell) ---------------------------------
    def chain(self, policy):
        """P[from, to] over all 100 cells for one step. Terminal cells and walls have empty rows."""
        P = np.zeros((SIZE * SIZE, SIZE * SIZE))
        for cell in self._free:
            for move, prob in enumerate(self.move_probabilities(cell, policy[cell])):
                if prob > 0:
                    P[cell, self.destination(cell, move)] += prob
        return P

    def outcomes(self, policy, horizon=HORIZON, crossing=None):
        """Per start cell, the chance an episode has ended in the goal, in each lava cell, or is
        still running after `horizon` steps. If `crossing` is (from, to), also the chance that this
        one step is ever taken."""
        P = self.chain(policy)
        inner = P.copy()
        inner[:, list(self.terminals)] = 0.0
        ended = {cell: P[:, cell].copy() for cell in self.terminals}
        reached = {cell: np.zeros(SIZE * SIZE) for cell in self.terminals}
        alive = np.eye(SIZE * SIZE)
        for _ in range(horizon):
            for cell in self.terminals:
                reached[cell] += alive @ ended[cell]
            alive = alive @ inner
        out = {"goal": reached[self.goal],
               "lava": sum(reached[cell] for cell in self.lava),
               "lava_by_cell": {int(cell): reached[cell] for cell in sorted(self.lava)},
               "timeout": alive.sum(axis=1)}
        if crossing is not None:
            src, dst = crossing
            step = np.zeros(SIZE * SIZE)
            step[src] = P[src, dst]
            blocked = inner.copy()
            blocked[src, dst] = 0.0
            crossed, alive = np.zeros(SIZE * SIZE), np.eye(SIZE * SIZE)
            for _ in range(horizon):
                crossed += alive @ step
                alive = alive @ blocked
            out["crossed"] = crossed
        return out

    def value(self, policy, gamma=GAMMA):
        """Discounted value of following the policy from each cell. Zero at terminals and walls."""
        P = self.chain(policy)
        inner = P.copy()
        inner[:, list(self.terminals)] = 0.0
        return np.linalg.solve(np.eye(SIZE * SIZE) - gamma * inner, P[:, self.goal] * GOAL_REWARD)

    def move_values(self, value, gamma=GAMMA):
        """[cell, executed move]: reward plus discounted value of where the move lands."""
        out = np.zeros((SIZE * SIZE, N_MOVES))
        for cell in self._free:
            for move in self._legal[cell]:
                nxt, reward, done = self.outcome(cell, move)
                out[cell, move] = reward + (0.0 if done else gamma * value[nxt])
        return out

    def command_values(self, value, gamma=GAMMA):
        """[cell, commanded move]: true value of commanding a move once (noise included), then
        following the policy whose value is given."""
        per_move = self.move_values(value, gamma)
        out = np.zeros((SIZE * SIZE, N_MOVES))
        for cell in self._free:
            out[cell] = self._execution[cell] @ per_move[cell]
        return out

    def best_policy(self, forbidden=None, gamma=GAMMA):
        """Best commanded move per cell, by value iteration. `forbidden` maps a cell to the moves
        that may not be commanded there. Returns (best move per cell, optimal value per cell).
        Ties go to the first move in the order stay, up, down, left, right."""
        forbidden = forbidden or {}
        value = np.zeros(SIZE * SIZE)
        for _ in range(2000):
            q = self.command_values(value, gamma)
            for cell, moves in forbidden.items():
                q[cell, list(moves)] = -np.inf
            new = np.zeros(SIZE * SIZE)
            new[self._free] = q[self._free].max(axis=1)
            if np.max(np.abs(new - value)) < 1e-13:
                value = new
                break
            value = new
        q = self.command_values(value, gamma)
        for cell, moves in forbidden.items():
            q[cell, list(moves)] = -np.inf
        best = np.zeros(SIZE * SIZE, int)
        for cell in self._free:
            best[cell] = int(np.flatnonzero(q[cell] >= q[cell].max() - 1e-12)[0])
        return best, value


def published(noise=PUBLISHED_NOISE, beta=BETA):
    """The published map and rules. noise=0.10 is the paper's; noise=0.0 is the no-noise variant.
    `beta` is the sharpness of our preference-to-move rule (PROTOCOL.md, Amendment 1); it does not
    touch the published game. The name carries it whenever it is not the original 2."""
    suffix = "" if beta == BETA else f"_sharp{beta:g}"
    return Task(f"single_critical_noise{round(100 * noise):02d}{suffix}", noise, beta=beta)


def one_hot(cells):
    return np.eye(SIZE * SIZE, dtype=np.float32)[np.asarray(cells)]


def policy_from_moves(moves):
    """A preference vector per cell that commands the given move with certainty."""
    return CORNERS[np.asarray(moves)]
