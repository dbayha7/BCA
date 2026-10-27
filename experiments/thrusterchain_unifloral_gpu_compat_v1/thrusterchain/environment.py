"""Strict, time-aware Gymnasium adapter around the unchanged ThrusterChain plant."""
from collections.abc import Mapping
import math

import numpy as np
from gymnasium import spaces

from .canonical import load_canonical, provenance

canonical = load_canonical()
Physics = canonical.Physics
STALL, NO_STALL = canonical.STALL, canonical.NO_STALL
DS, DX, DA = canonical.DS, canonical.DX, canonical.DA
KNEE = canonical.KNEE
START_LAWS = canonical.START_LAWS
dynamics = canonical.dynamics
dynamics_jax = canonical.dynamics_jax
matrices = canonical.matrices
thrust = canonical.thrust
thrust_jax = canonical.thrust_jax
draw_starts = canonical.draw_starts
bank_starts = canonical.bank_starts
in_start_set = canonical.in_start_set


def _finite_real(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array of exact shape {shape}")
    out = raw.astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite numbers")
    return out


def canonical_config():
    """Return a fresh JSON-compatible description of the canonical default task."""
    return {
        "physics": STALL.as_dict(), "state_dim": DX, "action_dim": DA,
        "observation_dim": DX + 1, "observation_mode": "time_aware",
        "task_semantics": "finite_horizon", "objective_gamma": STALL.gamma,
        "horizon": STALL.length, "terminal_reward": 0.0,
        "terminal_bootstrap": False, "matrix_sha256": provenance()["matrix_sha256"],
    }


class ThrusterChainEnv(canonical.ThrusterChainEnv):
    """Canonical physical dynamics with explicit finite-task or legacy limit semantics.

    Default observation is x[13] followed by (100-t)/100, float32. The finite
    task terminates at t=100; its continuation value is zero. Legacy replay
    mode ``task_semantics='continuing_time_limit'`` preserves upstream truncation.
    It uses stationary physical observations, either the upstream 13-vector or
    a 14-vector with a constant zero padding feature for benchmark adapters.
    """

    def __init__(self, stall=True, start="recorded", render_mode=None, stall_slope=1.5,
                 observation_mode="time_aware", task_semantics="finite_horizon"):
        if not isinstance(stall, (bool, np.bool_)):
            raise ValueError("stall must be a boolean")
        if not isinstance(stall_slope, (int, float, np.integer, np.floating)) or isinstance(stall_slope, bool):
            raise ValueError("stall_slope must be numeric")
        if not math.isfinite(float(stall_slope)) or float(stall_slope) not in (1.5, 3.0):
            raise ValueError("supported canonical stall slopes are 1.5 and the documented retune 3.0")
        if observation_mode not in ("time_aware", "physical", "physical_padded"):
            raise ValueError("unknown observation_mode")
        if task_semantics not in ("finite_horizon", "continuing_time_limit"):
            raise ValueError("unknown task_semantics")
        if task_semantics == "finite_horizon" and observation_mode != "time_aware":
            raise ValueError("finite_horizon requires time_aware observation")
        if task_semantics == "continuing_time_limit" and observation_mode == "time_aware":
            raise ValueError("continuing_time_limit requires stationary physical observations")
        self.observation_mode = observation_mode
        self.task_semantics = task_semantics
        super().__init__(stall=stall, start=start, render_mode=render_mode, stall_slope=stall_slope)
        if observation_mode == "time_aware":
            self.observation_space = spaces.Box(
                np.r_[canonical.OBS_LOW, np.float32(0)].astype(np.float32),
                np.r_[canonical.OBS_HIGH, np.float32(1)].astype(np.float32), dtype=np.float32)
        elif observation_mode == "physical_padded":
            self.observation_space = spaces.Box(
                np.r_[canonical.OBS_LOW, np.float32(0)].astype(np.float32),
                np.r_[canonical.OBS_HIGH, np.float32(0)].astype(np.float32), dtype=np.float32)

    @property
    def state(self):
        """A defensive copy of the exact float64 physical state, or None."""
        return None if self._x is None else self._x.copy()

    @property
    def t(self):
        return self._t

    @property
    def config(self):
        return {**canonical_config(), "physics": self.physics.as_dict(), "start": self.start,
                "observation_dim": int(self.observation_space.shape[0]),
                "observation_mode": self.observation_mode, "task_semantics": self.task_semantics,
                "terminal_bootstrap": self.task_semantics == "continuing_time_limit"}

    def _obs(self):
        if self.observation_mode == "physical":
            return self._x.astype(np.float32)
        if self.observation_mode == "physical_padded":
            return np.r_[self._x, 0.0].astype(np.float32)
        return np.r_[self._x, (self.physics.length - self._t) / self.physics.length].astype(np.float32)

    def _augment_info(self, info, canonical_terminated=False, canonical_truncated=False):
        info = dict(info)
        finite_end = self.task_semantics == "finite_horizon" and self._t == self.physics.length
        info.update(remaining_steps=self.physics.length - self._t,
                    canonical_terminated=bool(canonical_terminated), canonical_truncated=bool(canonical_truncated),
                    task_semantics=self.task_semantics, bootstrap_mask=0.0 if finite_end else 1.0,
                    continuation_discount=0.0 if finite_end else self.physics.gamma)
        return info

    def reset(self, *, seed=None, options=None):
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("reset options must be a mapping")
        options = dict(options or {})
        if "state" in options:
            options["state"] = _finite_real(options["state"], (DX,), "reset state")
        if "start" in options:
            override = options["start"]
            if not isinstance(override, Mapping):
                raise ValueError("options['start'] must be a mapping")
            override = dict(override)
            for key in ("current", "wave"):
                if key in override:
                    override[key] = tuple(_finite_real(override[key], (2,), f"{key} range"))
            if "kick" in override:
                kick = _finite_real(override["kick"], (), "kick")
                override["kick"] = float(kick)
            options["start"] = override
        observation, info = super().reset(seed=seed, options=options)
        if not self.observation_space.contains(observation):
            raise RuntimeError("canonical reset produced an out-of-space observation")
        return observation, self._augment_info(info)

    def step(self, action):
        # Validate before the upstream flattening so malformed matrices are rejected.
        action = _finite_real(action, (DA,), "action")
        with np.errstate(over="ignore", invalid="ignore"):
            deployed = action.astype(np.float32)
        if not np.all(np.isfinite(deployed)):
            raise ValueError("action must remain finite at the float32 deployment boundary")
        observation, reward, raw_terminated, raw_truncated, info = super().step(deployed)
        if not self.observation_space.contains(observation) or not math.isfinite(reward):
            raise RuntimeError("canonical transition produced an invalid observation or reward")
        if self.task_semantics == "finite_horizon":
            terminated, truncated = bool(raw_terminated or raw_truncated), False
        else:
            terminated, truncated = bool(raw_terminated), bool(raw_truncated)
        return observation, reward, terminated, truncated, self._augment_info(info, raw_terminated, raw_truncated)


__all__ = ["ThrusterChainEnv", "canonical_config", "canonical", "Physics", "STALL", "NO_STALL",
           "DS", "DX", "DA", "KNEE", "START_LAWS", "dynamics", "dynamics_jax", "matrices",
           "thrust", "thrust_jax", "draw_starts", "bank_starts", "in_start_set"]
