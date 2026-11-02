"""The knee_mix15 reset law and the knee_edge_mixture_v1 recorder (protocol Part 3.2-3.3).

Every draw comes from a named stream addressed by (entropy, spawn_key), so consuming one stream never advances
another and the stall and no-stall twins share every reset seed and recorder draw.
- reset (1801, episode): the Gymnasium reset seed (the environment then draws kicks, current, wave, phase in its
  canonical order);
- mixture (1804, episode): one uniform per episode choosing the current range;
- actions (1803, episode): five uniforms per step (u, v, w1, w2, w3), always all five, on both branches;
- split (1802,): the permutation of complete episodes into the four roles.
"""
import math

import numpy as np

from core import require
from knee_region import load_band, load_g

STREAM_KEYS = {"reset": 1801, "split": 1802, "actions": 1803, "mixture": 1804}
DEEP_PROBABILITY = 0.15
CURRENT_RANGES = {0: (0.1, 0.7), 1: (0.9, 1.1)}      # mixture component 0: recorded range; 1: deep-load range
WINDOW = (-0.9, -0.7)
WINDOW_WIDTH = 0.2                                   # the literal 0.2 of the rule; WINDOW[1] - WINDOW[0] is 0.20000000000000007
WINDOW_PROBABILITY = 0.98
SIDE_RANGE = 0.7                                     # a1..a3 = -0.7 + 1.4 w, never past the knee
LOG_SIDE = 3 * math.log(1 / (2 * SIDE_RANGE))
BEHAVIOR = {
    "name": "knee_edge_mixture_v1", "uniforms_per_step": ["u", "v", "w1", "w2", "w3"],
    "rule": "if load_band(g) >= 1 and u < 0.98: a0 = -0.9 + 0.2 v; else a0 = -1 + 2 v; a_k = -0.7 + 1.4 w_k",
    "load": "g = float64(obs10) + float64(obs11), at the observation the command is given",
    "density_given_load": {"g >= 0.4": "p(a0) = 4.9 * 1[a0 in [-0.9, -0.7]] + 0.01", "g < 0.4": "p(a0) = 0.5",
                           "a1..a3": "uniform on [-0.7, 0.7] each, density (1/1.4)^3"},
    "state_independent": False,
    "execution_transform": "float64 request -> float32 command; bounded, so clipping is inactive",
}
RESET_LAW = {"name": "knee_mix15", "deep_probability": DEEP_PROBABILITY,
             "current_ranges": {"0": list(CURRENT_RANGES[0]), "1": list(CURRENT_RANGES[1])},
             "wave": [0.1, 0.4], "kick": 0.5,
             "draw_order": "mixture uniform from its own stream; then the environment's canonical draw_starts order: "
                           "kick normals, current, wave amplitude, phase"}


def seed_sequence(entropy, name, episode=None):
    require(name in STREAM_KEYS, "Unknown stream: " + str(name))
    require(type(entropy) is int and entropy >= 0, "Entropy must be a nonnegative int")
    if name == "split":
        require(episode is None, "The split stream has no episode child")
        key = (STREAM_KEYS[name],)
    else:
        require(type(episode) is int and episode >= 0, "Episode streams need a nonnegative int episode")
        key = (STREAM_KEYS[name], episode)
    return np.random.SeedSequence(entropy, spawn_key=key, pool_size=4)


def generator(entropy, name, episode=None):
    return np.random.Generator(np.random.PCG64(seed_sequence(entropy, name, episode)))


def reset_seed(entropy, episode):
    word = seed_sequence(entropy, "reset", episode).generate_state(1, dtype=np.uint64)[0]
    return int(word) & ((1 << 63) - 1)


def mixture_component(entropy, episode):
    return int(generator(entropy, "mixture", episode).random() < DEEP_PROBABILITY)


def reset_options(component):
    return {"start": {"current": CURRENT_RANGES[int(component)]}}


def split_assignment(episodes, entropy):
    """768 / 256 / 128 / 128 at 1,280 episodes: codes 0 fitting, 1 tuning, 2 selection, 3 final calibration."""
    n_tune, n_select, n_cal = max(1, episodes // 5), max(1, episodes // 10), max(1, episodes // 10)
    counts = [episodes - n_tune - n_select - n_cal, n_tune, n_select, n_cal]
    codes = np.repeat(np.arange(4, dtype=np.uint8), counts)
    return codes[generator(entropy, "split").permutation(episodes)], counts


def requested_command(observation, uniforms):
    """(requested float64 (4,), component, log density) from one observation and the step's five uniforms."""
    u, v, w1, w2, w3 = (float(x) for x in uniforms)
    band = int(load_band(load_g(np.asarray(observation, np.float32)[None]))[0])
    window = band >= 1 and u < WINDOW_PROBABILITY
    a0 = -0.9 + 0.2 * v if window else -1.0 + 2.0 * v
    requested = np.array([a0, -SIDE_RANGE + 2 * SIDE_RANGE * w1, -SIDE_RANGE + 2 * SIDE_RANGE * w2,
                          -SIDE_RANGE + 2 * SIDE_RANGE * w3], np.float64)
    return requested, int(window), float(log_density(requested[None], np.asarray([band]))[0])


def log_density(requested, bands):
    """Closed-form log density of requested commands given each row's load band (state-dependent through load)."""
    requested = np.asarray(requested, np.float64)
    bands = np.asarray(bands)
    require(requested.shape[-1:] == (4,) and requested.shape[:-1] == bands.shape, "Shape mismatch")
    a0, side = requested[..., 0], requested[..., 1:]
    in_window = (a0 >= WINDOW[0]) & (a0 <= WINDOW[1])
    high = WINDOW_PROBABILITY / WINDOW_WIDTH * in_window + (1 - WINDOW_PROBABILITY) / 2.0
    p0 = np.where(bands >= 1, high, 0.5)
    support = (np.abs(a0) <= 1.0) & np.all(np.abs(side) <= SIDE_RANGE, axis=-1)
    with np.errstate(divide="ignore"):
        return np.where(support, np.log(p0) + LOG_SIDE, -np.inf)


def episode_draws(entropy, episode, horizon=100):
    """The episode's full (horizon, 5) uniform block, drawn in step order (used by collection and lineage replay)."""
    return generator(entropy, "actions", episode).random((horizon, 5))
