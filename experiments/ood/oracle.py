"""Exact toy dynamics for harness tests; no learned policy or BCA result."""

import itertools
import math


class Oracle:
    """Move x by a in [-1,1]; reward = -(x_next-1)^2 - 0.1*a^2."""

    def __init__(self, x=0.0, limit=5):
        self.x, self.elapsed, self.limit = float(x), 0, limit
        self.terminated = self.truncated = False

    @staticmethod
    def in_support(x, action):
        # Synthetic behavior support is stipulated, so novelty has ground truth.
        return -1 <= x <= 1 and -1 <= action <= 0

    def capture(self):
        return dict(x=self.x, elapsed=self.elapsed, limit=self.limit,
                    terminated=self.terminated, truncated=self.truncated)

    def restore(self, state):
        if set(state) != {"x", "elapsed", "limit", "terminated", "truncated"}:
            raise ValueError("Incomplete oracle state.")
        if not math.isfinite(state["x"]) or not 0 <= state["elapsed"] <= state["limit"]:
            raise ValueError("Invalid oracle state.")
        self.__dict__.update(state)

    def step(self, action):
        if self.terminated or self.truncated or not math.isfinite(action) or abs(action) > 1:
            raise ValueError("Invalid action or episode already ended.")
        self.x += action
        self.elapsed += 1
        self.terminated = abs(self.x) >= 2
        self.truncated = self.elapsed >= self.limit and not self.terminated
        reward = -(self.x - 1)**2 - .1 * action**2
        return self.x, reward, self.terminated, self.truncated


def enumerate_returns(x, first_actions, *, continuation, horizon):
    """Directly enumerate the fixed finite action sequences, without the simulator."""
    if type(horizon) is not int or not 1 <= horizon <= 5:
        raise ValueError("Oracle horizon must be in [1,5].")
    returns = []
    for first in first_actions:
        position, total = x, 0.0
        for action in itertools.chain([first], itertools.repeat(continuation, horizon-1)):
            if not math.isfinite(action) or abs(action) > 1:
                raise ValueError("Oracle action outside bounds.")
            position += action
            total += -(position - 1)**2 - .1 * action**2
            if abs(position) >= 2:
                break
        returns.append(total)
    return returns
