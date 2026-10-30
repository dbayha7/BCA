"""Arm masks, common calibration rows and the normalization source (protocol Part 3.7 and 3.9).

CUTOFF drops every fitting row in W. RANDOM keeps, in each stratum (episode x ten-step bin x load band), exactly
CUTOFF's count, chosen uniformly from all the stratum's rows. That is the sampling procedure of
conditional_ood_gate_central_quartile_v1/geometry.py control_masks (stable lexicographic grouping, one PCG64 stream,
rng.choice without replacement per stratum), with the knee predicate and three load bands instead of a two-level flag.
Calibration copies geometry.py calibration_plan (lines 311-355) with W as the ineligible set.
"""
import hashlib

import numpy as np

from core import require
from knee_region import load_band, load_g, past_knee_a0

TIME_BIN = 10
HORIZON = 100
CALIBRATION_EPISODES = 128


def _digest(rows):
    return hashlib.sha256(np.asarray(rows, "<i8").tobytes()).hexdigest()


def fitting_rows(arrays):
    """(obs, actions, times, episode ids, bands, past-knee mask) of every fitting (code 0) row, in episode order."""
    episodes = np.flatnonzero(np.asarray(arrays["split_codes"]) == 0)
    obs = np.asarray(arrays["observations"])[episodes, :-1].reshape(-1, 14)
    actions = np.asarray(arrays["actions"])[episodes].reshape(-1, 4)
    times = np.tile(np.arange(HORIZON, dtype=np.int64), len(episodes))
    ids = np.repeat(np.asarray(arrays["group_ids"])[episodes], HORIZON)
    return {"episodes": episodes, "obs": obs, "actions": actions, "times": times, "episode_ids": ids,
            "bands": load_band(load_g(obs)), "past_knee": past_knee_a0(actions)[0]}


def arm_masks(obs, actions, times, episode_ids, *, seed, bands=None, time_bin=TIME_BIN):
    """(masks {"cutoff", "random"}, receipt). `bands` may be overridden only by known-fault tests."""
    obs, actions = np.asarray(obs), np.asarray(actions)
    times, episode_ids = np.asarray(times, np.int64), np.asarray(episode_ids, np.int64)
    n = len(obs)
    require(actions.shape == (n, 4) and times.shape == (n,) and episode_ids.shape == (n,), "Row arrays disagree")
    withheld = past_knee_a0(actions)[0]
    bands = load_band(load_g(obs)) if bands is None else np.asarray(bands, np.int64)
    strata = np.stack((episode_ids, times // time_bin, bands.astype(np.int64)), axis=1)
    cutoff = ~withheld
    random = np.zeros(n, bool)
    rng = np.random.Generator(np.random.PCG64(seed))
    order = np.lexsort((np.arange(n), strata[:, 2], strata[:, 1], strata[:, 0]))
    ordered = strata[order]
    bounds = np.concatenate(([0], np.flatnonzero(np.any(np.diff(ordered, axis=0), axis=1)) + 1, [n]))
    receipt = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        rows = order[start:end]
        key = strata[rows[0]]
        count = int(np.sum(cutoff[rows]))
        random[rng.choice(rows, count, replace=False)] = True
        receipt.append({"episode_id": int(key[0]), "time_bin": int(key[1]), "load_band": int(key[2]),
                        "rows": len(rows), "cutoff_retained": count, "random_retained": int(np.sum(random[rows]))})
    masks = {"cutoff": cutoff, "random": random}
    return masks, {"seed": seed, "time_bin_width": time_bin, "strata": receipt,
                   "retained_row_digests": {k: _digest(np.flatnonzero(v)) for k, v in masks.items()},
                   "cutoff_retained_w_rows": int(np.sum(cutoff & withheld)),
                   "random_retained_w_rows": int(np.sum(random & withheld))}


def three_band_counts_equal(masks, obs, actions, times, episode_ids, time_bin=TIME_BIN):
    """4.7(i), recomputed independently of the builder's receipt: RANDOM keeps CUTOFF's count in every
    (episode x ten-step bin x three-band load) stratum, and CUTOFF keeps exactly the non-W rows."""
    bands = load_band(load_g(np.asarray(obs))).astype(np.int64)
    key = (np.asarray(episode_ids, np.int64) * 1000 + np.asarray(times, np.int64) // time_bin) * 3 + bands
    _, inverse = np.unique(key, return_inverse=True)
    withheld = past_knee_a0(np.asarray(actions))[0]
    cutoff_ok = np.array_equal(masks["cutoff"], ~withheld)
    counts_ok = np.array_equal(np.bincount(inverse, weights=masks["cutoff"]), np.bincount(inverse, weights=masks["random"]))
    return bool(cutoff_ok and counts_ok)


def calibration_rows(arrays, *, seed, replacement_seed):
    """One common row per episode from the first 128 tuning (code 1) episodes by group id; W is ineligible."""
    tuning = np.flatnonzero(np.asarray(arrays["split_codes"]) == 1)
    tuning = tuning[np.argsort(np.asarray(arrays["group_ids"])[tuning], kind="stable")][:CALIBRATION_EPISODES]
    require(len(tuning) == CALIBRATION_EPISODES, "Fewer than 128 tuning episodes")
    actions = np.asarray(arrays["actions"])[tuning].reshape(-1, 4)
    eligible = ~past_knee_a0(actions)[0]
    original_times = np.random.Generator(np.random.PCG64(seed)).integers(0, HORIZON, size=len(tuning))
    replacement = np.random.Generator(np.random.PCG64(replacement_seed))
    selected, replaced = [], []
    for ordinal in range(len(tuning)):
        rows = np.arange(ordinal * HORIZON, (ordinal + 1) * HORIZON, dtype=np.int64)
        original = int(ordinal * HORIZON + original_times[ordinal])
        candidates = rows[eligible[rows]]
        require(len(candidates) > 0, "Calibration episode with no eligible row fails closed")
        if eligible[original]:
            selected.append(original)
        else:
            selected.append(int(candidates[replacement.integers(len(candidates))]))
            replaced.append(int(np.asarray(arrays["group_ids"])[tuning[ordinal]]))
    selected = np.asarray(selected, np.int64)
    require(not np.any(past_knee_a0(actions[selected])[0]), "Calibration row in W")
    return {"episodes": tuning, "local_rows": selected, "replacements": len(replaced),
            "replacement_group_ids": replaced, "selected_row_digest": _digest(selected),
            "seed": seed, "replacement_seed": replacement_seed,
            "rule": "geometry.py calibration_plan with W ineligible: original uniform time per episode from "
                    "PCG64(seed); if in W, a uniform eligible row of the same episode from PCG64(replacement_seed)"}


def normalization_rows(masks):
    """The retained intersection CUTOFF and RANDOM within one world (Part 3.8)."""
    rows = np.flatnonzero(masks["cutoff"] & masks["random"])
    require(len(rows) > 0, "Empty normalization intersection")
    return rows


def l14_density(masks, actions, bands):
    """L1.4: retained rows in the band W- with a0 >= float32(-0.9), over (high-load rows x 0.1), per load group."""
    a0 = np.asarray(actions)[:, 0].astype(np.float64)
    in_band = (a0 >= float(np.float32(-0.9))) & (a0 < -float(np.float32(0.8)))
    out = {}
    for arm, keep in masks.items():
        out[arm] = {}
        for name, group in (("pooled_high", bands >= 1), ("mid", bands == 1), ("deep", bands == 2)):
            out[arm][name] = float(np.sum(keep & in_band & group) / (np.sum(group) * 0.1))
    return out
