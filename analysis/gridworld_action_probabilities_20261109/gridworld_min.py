"""Minimal, independent re-statement of the adapted GridWorld mechanics.

Written from the handoff description so this analysis never imports code from
the preserved evidence folder. measure_action_probabilities.py checks these
mechanics against the saved exact outcomes before reporting any number.

Cell id is 10*y + x, x rightward, y downward. Direction order is
UP, RIGHT, DOWN, LEFT, the order of the actor's four preferences.
"""
import itertools

import numpy as np

SIZE = 10
SHARED_WALLS = frozenset({16, 26, 36, 62, 63, 64})
HAZARDS = frozenset({21, 34, 58, 86})
GOAL = 9
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTION_NAMES = ("up", "right", "down", "left")
_DELTA = ((0, -1), (1, 0), (0, 1), (-1, 0))
STEP_REWARD, GOAL_REWARD, HAZARD_REWARD = -0.1, 10.0, -5.0


def walls(scenario):
    """"honeypot", "bottleneck" (opening on row 4, as archived), or "bottleneck_rowN":
    the same dividing wall with its single opening moved to row N. The rowN layouts
    exist only to compare candidate designs; no archived policy was trained on them."""
    if scenario == "honeypot":
        return SHARED_WALLS
    if scenario == "bottleneck":
        return SHARED_WALLS | {10 * y + 5 for y in range(SIZE) if y != 4}
    if scenario.startswith("bottleneck_row"):
        opening = int(scenario[len("bottleneck_row"):])
        return SHARED_WALLS | {10 * y + 5 for y in range(SIZE) if y != opening}
    raise ValueError(scenario)


def free_cells(scenario):
    closed = walls(scenario) | HAZARDS | {GOAL}
    return [cell for cell in range(SIZE * SIZE) if cell not in closed]


def move(scenario, cell, direction):
    """One executed direction: (next cell, reward, terminal)."""
    if cell in HAZARDS or cell == GOAL:
        return cell, 0.0, True
    x, y = cell % SIZE, cell // SIZE
    dx, dy = _DELTA[direction]
    nxt = min(SIZE - 1, max(0, y + dy)) * SIZE + min(SIZE - 1, max(0, x + dx))
    if nxt in walls(scenario):
        nxt = cell
    if nxt == GOAL:
        return nxt, GOAL_REWARD, True
    if nxt in HAZARDS:
        return nxt, HAZARD_REWARD, True
    return nxt, STEP_REWARD, False


def direction_role(scenario, cell, direction):
    """What executing this direction from this cell does."""
    nxt, _, _ = move(scenario, cell, direction)
    if nxt == GOAL:
        return "goal"
    if nxt in HAZARDS:
        return "hazard"
    if nxt == cell:
        return "blocked"  # wall or grid edge: the agent stays put and pays the step cost
    return "open"


def movement_probabilities(command, beta=2.0):
    """softmax(beta * command). Larger beta sharpens; beta 2 is the original setting."""
    logits = beta * np.asarray(command, dtype=float)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def probability_bounds(beta=2.0):
    """Smallest and largest probability any one direction can receive from a command in [-1,1]^4."""
    high = float(np.exp(2 * beta))
    return 1.0 / (1.0 + 3.0 * high), high / (high + 3.0)


def transition_model(scenario, commands, beta=2.0):
    """Markov chain induced by one command per cell.

    Returns P (non-terminal moves only), expected reward, and the per-step
    probabilities of entering the goal and of entering a hazard.
    """
    n = SIZE * SIZE
    P, reward = np.zeros((n, n)), np.zeros(n)
    goal_in, hazard_in = np.zeros(n), np.zeros(n)
    for cell in free_cells(scenario):
        for direction, prob in enumerate(movement_probabilities(commands[cell], beta)):
            nxt, r, done = move(scenario, cell, direction)
            reward[cell] += prob * r
            if nxt == GOAL:
                goal_in[cell] += prob
            elif nxt in HAZARDS:
                hazard_in[cell] += prob
            if not done:
                P[cell, nxt] += prob
    return P, reward, goal_in, hazard_in


def finite_outcomes(scenario, commands, beta=2.0, horizon=200):
    """Exact probability of reaching the goal / a hazard within `horizon` steps, per start cell."""
    P, _, goal_in, hazard_in = transition_model(scenario, commands, beta)
    goal, hazard = np.zeros(SIZE * SIZE), np.zeros(SIZE * SIZE)
    for _ in range(horizon):
        goal, hazard = goal_in + P @ goal, hazard_in + P @ hazard
    return goal, hazard


def discounted_value(scenario, commands, beta=2.0, gamma=0.95):
    """Exact infinite-horizon discounted value of following `commands` forever."""
    P, reward, _, _ = transition_model(scenario, commands, beta)
    return np.linalg.solve(np.eye(SIZE * SIZE) - gamma * P, reward)


def direction_values(scenario, value, gamma=0.95):
    """Value of each executed direction at each cell, given a continuation value."""
    out = np.zeros((SIZE * SIZE, 4))
    for cell in free_cells(scenario):
        for direction in range(4):
            nxt, r, done = move(scenario, cell, direction)
            out[cell, direction] = r + (0.0 if done else gamma * value[nxt])
    return out


def command_value(scenario, cell, command, value, beta=2.0, gamma=0.95):
    """Q of taking `command` once at `cell`, then following the policy whose value is `value`."""
    probs = movement_probabilities(command, beta)
    return float(probs @ direction_values(scenario, value, gamma)[cell])


CORNERS = np.array(list(itertools.product([-1.0, 1.0], repeat=4)))


def oracle(scenario, beta=2.0, gamma=0.95, sweeps=2000, tol=1e-12):
    """Best achievable bounded command per cell (the optimum of a linear-fractional
    objective over a box sits at a corner), with its exact value."""
    corner_probs = movement_probabilities(CORNERS, beta)
    value = np.zeros(SIZE * SIZE)
    commands = np.zeros((SIZE * SIZE, 4))
    for _ in range(sweeps):
        previous = value.copy()
        per_direction = direction_values(scenario, previous, gamma)
        for cell in free_cells(scenario):
            scores = corner_probs @ per_direction[cell]
            best = int(np.argmax(scores))
            value[cell], commands[cell] = scores[best], CORNERS[best]
        if np.max(np.abs(value - previous)) < tol:
            break
    return commands, discounted_value(scenario, commands, beta, gamma)
