"""Synthetic test fixtures. Test-only entropies (never the sealed seeds); no simulator dynamics.

Load is action-independent (the current is constant and the wave turns 0.27 rad per step), so a dataset's loads,
recorder commands and masks can be built from the reset law and recorder alone, as the sealed Monte Carlo did.
"""
import numpy as np

import recorder

TEST_ENTROPY = 424242
TEST_SEEDS = {"random_control": 11, "calibration_original": 12, "calibration_replacement": 13,
              "gradient_probes": 14, "scale_bank": 15}


def synthetic_arrays(episodes=1280, entropy=TEST_ENTROPY):
    """Dataset-shaped arrays with exact loads and recorder commands; other state coordinates are zero."""
    rng = np.random.Generator(np.random.PCG64(entropy))
    codes, _ = recorder.split_assignment(episodes, entropy)
    component = np.asarray([recorder.mixture_component(entropy, e) for e in range(episodes)], np.uint8)
    current = np.where(component == 1, rng.uniform(0.9, 1.1, episodes), rng.uniform(0.1, 0.7, episodes))
    amp, phase = rng.uniform(0.1, 0.4, episodes), rng.uniform(0, 2 * np.pi, episodes)
    t = np.arange(101)
    states = np.zeros((episodes, 101, 13))
    states[:, :, 10] = current[:, None]
    states[:, :, 11] = amp[:, None] * np.cos(phase[:, None] + 0.27 * t)
    states[:, :, 12] = amp[:, None] * np.sin(phase[:, None] + 0.27 * t)
    obs = np.concatenate([states, np.broadcast_to(((100 - t) / 100), (episodes, 101))[..., None]], -1).astype(np.float32)
    requested = np.zeros((episodes, 100, 4))
    for e in range(episodes):
        draws = recorder.episode_draws(entropy, e)
        for s in range(100):
            requested[e, s] = recorder.requested_command(obs[e, s], draws[s])[0]
    terminated = np.zeros((episodes, 100), bool)
    terminated[:, -1] = True
    return {"observations": obs, "states": states, "actions": requested.astype(np.float32),
            "requested_actions": requested, "rewards": -np.ones((episodes, 100)), "terminated": terminated,
            "split_codes": codes, "group_ids": np.arange(episodes, dtype=np.int64),
            "trajectory_ids": np.arange(episodes, dtype=np.int64), "reset_component": component}


def toy_actor(gain=0.3, bias=(-0.6, 0.3, -0.2, 0.5)):
    """A deterministic float32 'actor' for evaluator tests: a fixed linear map of the observation, squashed."""
    weights = np.random.Generator(np.random.PCG64(7)).normal(0, gain, (14, 4)).astype(np.float32)
    b = np.asarray(bias, np.float32)

    def forward(obs):
        obs = np.asarray(obs, np.float32)
        return np.tanh(obs @ weights + b).astype(np.float32)
    return forward


class SyntheticDataset:
    """A smoke-provenance stand-in with the attributes the planners read (arrays, metadata, digest)."""

    def __init__(self, arrays, world, entropy=TEST_ENTROPY):
        self.arrays = arrays
        self.metadata = {"world": world, "physics": __import__("worlds").WORLD_DICTS[world], "provenance": "smoke",
                         "seed": entropy}
        self.digest = "synthetic-" + world + "-" + str(entropy)
