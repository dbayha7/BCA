"""Explicit evaluation banks and Gymnasium-interface policy rollouts.

These tools do not make a previously inspected bank a holdout. Freeze the bank
identities and selection protocol before looking at final results. Policies must
consume the public float32 observation; a factory creates one policy per episode.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from experiments.shift import thruster_chain as tc


@dataclass(frozen=True)
class BankSpec:
    """Independent named random stream; bank identities must be frozen in advance."""

    seed: int
    role: str
    size: int

    def __post_init__(self):
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.role not in ("development", "selection", "final"):
            raise ValueError("role must be development, selection, or final")
        if not isinstance(self.size, int) or self.size < 1:
            raise ValueError("size must be a positive integer")


def starts(spec, situation):
    """Generate per-episode streams; changing size preserves existing starts.

    Situation is deliberately absent from the RNG key, providing common random
    numbers across start laws. Roles use distinct seed namespaces. Stream
    separation is reproducibility, not secrecy or proof of a clean holdout.
    """
    if situation not in tc.START_LAWS:
        raise ValueError(f"unknown situation: {situation}")
    role = {"development": 0, "selection": 1, "final": 2}[spec.role]
    return np.stack([
        tc.draw_starts(1, np.random.default_rng([spec.seed, role, i]), tc.START_LAWS[situation])[0]
        for i in range(spec.size)
    ])


def evaluate(policy_factory, spec, situation, world="stall"):
    """Return one undiscounted and discounted return per start via Env.step.

    policy_factory() creates an episode-local callable taking an observation.
    Actions and observations therefore follow the public float32 interface,
    including clipping, validation, and the episode truncation rule.
    """
    if world not in tc.WORLDS:
        raise ValueError(f"unknown world: {world}")
    initial = starts(spec, situation)
    env = tc.ThrusterChainEnv(stall=world == "stall", start=situation)
    returns, discounted, steps = [], [], []
    try:
        for state in initial:
            policy = policy_factory()
            obs, _ = env.reset(options={"state": state})
            total = total_discounted = 0.0
            t = 0
            while True:
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                if not math.isfinite(reward):
                    raise ValueError("nonfinite reward")
                total += reward
                total_discounted += env.physics.gamma ** t * reward
                t += 1
                if terminated or truncated:
                    break
            returns.append(total)
            discounted.append(total_discounted)
            steps.append(t)
    finally:
        env.close()
    return {
        "bank": {"seed": spec.seed, "role": spec.role, "size": spec.size,
                 "situation": situation, "world": world,
                 "starts_sha256": hashlib.sha256(initial.astype("<f8").tobytes()).hexdigest()},
        "returns": np.asarray(returns),
        "discounted_returns": np.asarray(discounted),
        "steps": np.asarray(steps),
    }


def paired_difference(first, second):
    """Per-start return differences for two results on precisely the same bank.

    The episode SE is conditional on the evaluated policies and bank sampling
    scheme; it is not uncertainty over independently trained policies.
    """
    if first["bank"] != second["bank"]:
        raise ValueError("paired evaluation requires identical bank identities")
    a, b = np.asarray(first["returns"]), np.asarray(second["returns"])
    if a.shape != b.shape or a.ndim != 1 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("returns must be finite, matching vectors")
    delta = a - b
    return {"differences": delta, "mean": float(delta.mean()),
            "episode_se": float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else None}
