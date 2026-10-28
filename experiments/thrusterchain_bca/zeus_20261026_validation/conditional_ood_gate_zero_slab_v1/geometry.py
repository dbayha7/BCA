"""Pure, array-in/receipt-out conditional-support diagnostics.

No file reads, simulator imports, learner construction or data mutation occurs.
The caller authenticates episode roles and freezes every setting before loading
new gate data. A mask identifies constructed support removal, not action quality.
"""
import hashlib

import numpy as np

KNEE = 0.800000011920929          # stall knee; not the band center in this gate
CENTER = 0.0                      # zero thrust on the first thruster
HALF_WIDTH = 0.1
LOAD_CUTOFF = 0.4
LOAD_DEPTH_EDGES = (0.4, 0.5, 0.6)
ROW_CAP = 64
REFERENCE_QUANTILE = 0.95
MIN_REFERENCE_ROWS = 2
TIME_BIN = 10
STD_ZERO_THRESHOLD = 1e-12
CONTROL_SEED = 2026102901
CALIBRATION_SEED = 2026102362
CALIBRATION_REPLACEMENT_SEED = 2026102902
CONDITIONS = ("full", "hole", "random")

# Both alternative bands must stay strictly below the stall knee.
assert abs(CENTER) + 3 * HALF_WIDTH < KNEE


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _observations(value):
    obs = np.asarray(value)
    _require(obs.ndim == 2 and obs.shape[1] == 14,
             "Expected rows of 14 time-aware observations")
    _require(np.all(np.isfinite(obs)), "Nonfinite observation")
    executed = obs.astype(np.float32).astype(np.float64)
    _require(np.all(np.isfinite(executed)), "Observation conversion overflow")
    return executed


def _actions(value, n):
    actions = np.asarray(value)
    _require(actions.shape == (n, 4), "Expected aligned four-coordinate actions")
    _require(np.all(np.isfinite(actions)) and np.all(np.abs(actions) <= 1),
             "Actions must be finite and within [-1,1]")
    return actions.astype(np.float32).astype(np.float64)


def _identities(value, n, name, unique=False):
    ids = np.asarray(value)
    _require(ids.shape == (n,) and ids.dtype.kind in "iu", "Invalid " + name)
    if unique:
        _require(len(np.unique(ids)) == n, "Duplicate " + name)
    return ids.astype(np.int64)


def _times(value, n):
    times = _identities(value, n, "times")
    _require(np.all((times >= 0) & (times < 100)), "Time outside [0,99]")
    return times


def _masks(value, n):
    _require(set(value) == set(CONDITIONS), "Expected full, hole and random masks")
    result = {}
    for name in CONDITIONS:
        mask = np.asarray(value[name])
        _require(mask.shape == (n,) and mask.dtype.kind == "b", "Invalid keep mask")
        result[name] = mask
    _require(np.all(result["full"]), "Full condition must retain every source row")
    return result


def _digest(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def region_masks(observations, actions):
    """Use executed float32 inputs, with both load summands promoted separately."""
    obs = _observations(observations)
    action = _actions(actions, len(obs))
    load = obs[:, 10] + obs[:, 11]
    high = load >= LOAD_CUTOFF
    offset = action[:, 0] - CENTER
    in_band = np.abs(offset) <= HALF_WIDTH
    # Adjacent bands have equal width 2h and exclude the inside-band boundary.
    left = (offset >= -3 * HALF_WIDTH) & (offset < -HALF_WIDTH)
    right = (offset > HALF_WIDTH) & (offset <= 3 * HALF_WIDTH)
    return {"load": load, "high": high, "in_band": in_band,
            "hole": high & in_band, "left_alternative": left,
            "right_alternative": right, "load_depth": load - LOAD_CUTOFF}


def control_masks(observations, actions, times, *, seed=CONTROL_SEED,
                  time_bin=TIME_BIN, episode_ids=None, match_within_episode=True):
    """Exact keep-count matching without matching rewards or future values.

    Random-removal rows are sampled uniformly without replacement in frozen
    state/time strata and may retain hole labels. No scientific labels are used.
    """
    obs = _observations(observations)
    regions = region_masks(obs, actions)
    times = _times(times, len(obs))
    _require(type(time_bin) is int and time_bin > 0, "Invalid time-bin width")
    if match_within_episode:
        episodes = _identities(episode_ids, len(obs), "episode IDs")
    else:
        episodes = np.zeros(len(obs), dtype=np.int64)
    strata = np.stack((episodes, times // time_bin, regions["high"].astype(np.int64)), axis=1)
    hole_keep = ~regions["hole"]
    random_keep = np.zeros(len(obs), dtype=bool)
    rng = np.random.Generator(np.random.PCG64(seed))
    receipt = []
    # Stable lexicographic grouping avoids scanning all rows once per stratum.
    order = np.lexsort((np.arange(len(obs)), strata[:, 2], strata[:, 1], strata[:, 0]))
    sorted_strata = strata[order]
    boundaries = np.concatenate(([0], np.flatnonzero(np.any(np.diff(sorted_strata, axis=0), axis=1)) + 1, [len(order)]))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:end]
        if len(rows) == 0:
            continue
        key = strata[rows[0]]
        count = int(np.sum(hole_keep[rows]))
        chosen = rng.choice(rows, count, replace=False)
        random_keep[chosen] = True
        receipt.append({"episode_id": int(key[0]) if match_within_episode else None,
                        "time_bin": int(key[1]), "high_load": bool(key[2]),
                        "original_rows": len(rows), "hole_retained": count,
                        "random_retained": int(np.sum(random_keep[rows]))})
    masks = {"full": np.ones(len(obs), dtype=bool), "hole": hole_keep, "random": random_keep}
    return masks, {"seed": seed, "time_bin_width": time_bin,
                   "match_within_episode": match_within_episode, "strata": receipt,
                   "retained_row_digests": {key: _digest(np.flatnonzero(mask)) for key, mask in masks.items()},
                   "random_retained_hole_labels": int(np.sum(random_keep & regions["hole"])),
                   "hole_retained_forbidden_labels": int(np.sum(hole_keep & regions["hole"]))}


def fit_support_metric(training_observations, *, zero_threshold=STD_ZERO_THRESHOLD):
    """Audit-only population diagonal scale, fitted on original train rows."""
    obs = _observations(training_observations)[:, :13]
    _require(len(obs) > 0 and np.isfinite(zero_threshold) and zero_threshold > 0, "Invalid metric fit")
    mean = np.mean(obs, axis=0, dtype=np.float64)
    raw_std = np.std(obs, axis=0, dtype=np.float64, ddof=0)
    scale = np.where(raw_std < zero_threshold, 1.0, raw_std)
    return {"mean": mean, "scale": scale, "raw_std": raw_std,
            "standard_deviation_zero_threshold": float(zero_threshold),
            "zero_standard_deviation_replacement": 1.0, "coordinates": 13,
            "fit_rows": len(obs), "fit_source": "original training observations only"}


def reference_cap_distances(z, episodes, times, high, query_times, *, row_cap=ROW_CAP,
                            time_bin=TIME_BIN, quantile=REFERENCE_QUANTILE,
                            min_reference_rows=MIN_REFERENCE_ROWS):
    """Training-only locality thresholds; no anchor, label, value or BCA output is used.

    For each queried time t and load side, every original training row at exactly time
    t on that side is a reference row. Its pool is the training rows in the same
    ten-step bin and load side from other episodes (its own episode is excluded, as an
    anchor's is). The statistic is the RMS distance to the row_cap-th nearest pool row,
    computed with the same formula as for anchors. The threshold is the declared
    quantile (NumPy default linear interpolation) over reference rows. Fewer than
    min_reference_rows reference rows with a full pool gives no threshold.
    """
    _require(0 < quantile < 1 and type(row_cap) is int and row_cap > 0
             and type(min_reference_rows) is int and min_reference_rows >= 1, "Invalid reference settings")
    thresholds = {}
    for t in sorted({int(x) for x in np.asarray(query_times).tolist()}):
        for side in (False, True):
            refs = np.flatnonzero((times == t) & (high == side))
            pool_mask = (times // time_bin == t // time_bin) & (high == side)
            stats = []
            for r in refs:
                pool = np.flatnonzero(pool_mask & (episodes != episodes[r]))
                if len(pool) < row_cap:
                    continue
                d = np.sqrt(np.mean((z[pool] - z[r]) ** 2, axis=1))
                stats.append(float(np.partition(d, row_cap - 1)[row_cap - 1]))
            entry = {"time": t, "high_load": bool(side), "reference_rows": len(refs),
                     "reference_rows_with_full_pool": len(stats), "quantile": quantile}
            if len(stats) >= min_reference_rows:
                stats = np.asarray(stats)
                entry.update(threshold=float(np.quantile(stats, quantile)),
                             median_reference_cap_distance=float(np.median(stats)),
                             min_reference_cap_distance=float(stats.min()),
                             max_reference_cap_distance=float(stats.max()))
            else:
                entry.update(threshold=None)
            thresholds[(t, bool(side))] = entry
    return thresholds


def support_report(training_observations, training_actions, training_episode_ids,
                   training_times, anchor_observations, anchor_episode_ids,
                   anchor_times, keep_masks, metric, *, source_row_ids=None,
                   row_cap=ROW_CAP, time_bin=TIME_BIN, reference_quantile=REFERENCE_QUANTILE):
    """Fix original neighbors first, then apply retention without refilling.

    Neighbors are the row_cap nearest original training rows in the anchor's ten-step
    bin and load side; there is no absolute radius. Entire anchor episodes are excluded.
    Locality passes when the distance to the row_cap-th neighbor is at most the
    training-only reference quantile for the anchor's exact time and side. Cross-load
    diagnostics rank both sides together and are never added to the primary count.
    Other action coordinates are unconstrained: this is marginal-a0 evidence, not
    evidence for full four-dimensional action support.
    """
    train = _observations(training_observations)
    anchors = _observations(anchor_observations)
    n = len(train)
    episodes = _identities(training_episode_ids, n, "training episode IDs")
    anchor_ids = _identities(anchor_episode_ids, len(anchors), "anchor episode IDs")
    times = _times(training_times, n)
    atimes = _times(anchor_times, len(anchors))
    rows = np.arange(n, dtype=np.int64) if source_row_ids is None else _identities(source_row_ids, n, "source row IDs", True)
    masks = _masks(keep_masks, n)
    region = region_masks(train, training_actions)
    mean, scale = np.asarray(metric["mean"]), np.asarray(metric["scale"])
    _require(mean.shape == (13,) and scale.shape == (13,) and np.all(np.isfinite(mean))
             and np.all(np.isfinite(scale)) and np.all(scale > 0), "Invalid support metric")
    _require(0 < reference_quantile < 1 and type(row_cap) is int and row_cap > 0
             and type(time_bin) is int and time_bin > 0, "Invalid neighborhood settings")
    z = (train[:, :13] - mean) / scale
    az = (anchors[:, :13] - mean) / scale
    thresholds = reference_cap_distances(z, episodes, times, region["high"], atimes,
                                         row_cap=row_cap, time_bin=time_bin, quantile=reference_quantile)
    output = []

    def counts(chosen):
        adjacent = region["left_alternative"][chosen] | region["right_alternative"][chosen]
        return {"rows": len(chosen), "episodes": len(np.unique(episodes[chosen])),
                "in_band_rows": int(np.sum(region["in_band"][chosen])),
                "in_band_episodes": len(np.unique(episodes[chosen[region["in_band"][chosen]]])),
                "hole_rows": int(np.sum(region["hole"][chosen])),
                "left_alternative_rows": int(np.sum(region["left_alternative"][chosen])),
                "left_alternative_episodes": len(np.unique(episodes[chosen[region["left_alternative"][chosen]]])),
                "right_alternative_rows": int(np.sum(region["right_alternative"][chosen])),
                "right_alternative_episodes": len(np.unique(episodes[chosen[region["right_alternative"][chosen]]])),
                "adjacent_alternative_rows": int(np.sum(adjacent)),
                "adjacent_alternative_episodes": len(np.unique(episodes[chosen[adjacent]]))}

    for i, anchor in enumerate(anchors):
        load = float(anchor[10] + anchor[11])
        high = load >= LOAD_CUTOFF
        distances = np.sqrt(np.mean((z - az[i]) ** 2, axis=1))
        base = (episodes != anchor_ids[i]) & (times // time_bin == atimes[i] // time_bin)
        primary = base & (region["high"] == high)

        def ranked(mask):
            candidates = np.flatnonzero(mask)
            return candidates[np.lexsort((rows[candidates], distances[candidates]))]

        ranked_primary = ranked(primary)
        selected = ranked_primary[:row_cap]
        cap_distance = float(distances[ranked_primary[row_cap - 1]]) if len(ranked_primary) >= row_cap else None
        reference = thresholds.get((int(atimes[i]), bool(high)), {"threshold": None})
        threshold = reference["threshold"]
        locality = cap_distance is not None and threshold is not None and cap_distance <= threshold
        unrestricted = ranked(base)[:row_cap]
        cross = unrestricted[region["high"][unrestricted] != high]
        output.append({"anchor_index": i, "anchor_episode_id": int(anchor_ids[i]),
                       "time": int(atimes[i]), "high_load": bool(high),
                       "load": load, "load_depth": load - LOAD_CUTOFF,
                       "same_side_pool_rows": len(ranked_primary),
                       "distance_to_cap_neighbor": cap_distance,
                       "reference_cap_distance_threshold": threshold,
                       "locality_passed": bool(locality),
                       "neighbor_source_row_ids": rows[selected].tolist(),
                       "neighbor_distances": distances[selected].tolist(),
                       "counts": {name: counts(selected[mask[selected]]) for name, mask in masks.items()},
                       "unrestricted_load_counts": counts(unrestricted),
                       "cross_load_counts": counts(cross)})
    return {"method": "row_cap nearest same-bin same-side neighbors; locality by training-only reference quantile; no absolute radius",
            "row_cap": row_cap, "time_bin_width": time_bin, "reference_quantile": reference_quantile,
            "reference_thresholds": [dict(v) for _, v in sorted(thresholds.items())],
            "distance": "sqrt(mean(((obs[:13]-anchor[:13])/original_train_scale)**2))",
            "whole_anchor_episode_excluded": True,
            "neighbors_fixed_before_retention": True, "other_action_coordinates_matched": False,
            "anchors": output}


def calibration_plan(observations, actions, episode_ids, *,
                     seed=CALIBRATION_SEED, replacement_seed=CALIBRATION_REPLACEMENT_SEED,
                     horizon=100):
    """Select one common eligible row per supplied calibration episode.

    Caller must supply exactly the authorized first128 sorted tuning groups in
    production. This generic pure function permits small synthetic fixtures.
    It does not estimate residuals, fit a calibrator or claim hole coverage.
    """
    obs = _observations(observations)
    episodes = np.asarray(episode_ids)
    _require(episodes.ndim == 1 and episodes.dtype.kind in "iu" and len(np.unique(episodes)) == len(episodes), "Unique calibration episode IDs required")
    _require(type(horizon) is int and horizon > 0 and len(obs) == len(episodes) * horizon,
             "Calibration rows must be complete fixed-horizon episodes")
    hole = region_masks(obs, actions)["hole"]
    eligible_mask = ~hole
    original_rng = np.random.Generator(np.random.PCG64(seed))
    original_times = original_rng.integers(0, horizon, size=len(episodes))
    eligible_rng = np.random.Generator(np.random.PCG64(replacement_seed))
    selected, original, empty, replaced = [], [], [], []
    for ordinal, index in enumerate(np.argsort(episodes, kind="stable")):
        source_rows = np.arange(index * horizon, (index + 1) * horizon, dtype=np.int64)
        original_row = int(index * horizon + original_times[ordinal])
        original.append(original_row)
        eligible = source_rows[eligible_mask[source_rows]]
        if len(eligible) == 0:
            empty.append(int(episodes[index]))
        elif eligible_mask[original_row]:
            selected.append(original_row)
        else:
            selected.append(int(eligible[eligible_rng.integers(len(eligible))]))
            replaced.append(int(episodes[index]))
    original = np.asarray(original, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.int64)
    return {"status": "failed_empty_eligible_episode" if empty else "eligible_plan_only",
            "selection_rule": "preserve original selected row if outside hole; otherwise uniform eligible row in same episode using separate fixed PCG64 seed; identical rows across all arms",
            "seed": seed, "replacement_seed": replacement_seed,
            "selected_rows": selected.tolist(),
            "selected_row_digest": _digest(selected), "original_selected_rows": original.tolist(),
            "original_rows_inside_hole": int(np.sum(hole[original])),
            "replacement_episode_ids": replaced,
            "empty_eligible_episode_ids": empty,
            "calibration_rows_planned": len(selected), "calibration_updates": 0,
            "hole_rows_used": int(np.sum(hole[selected])),
            "calibration_scope": "retained support only; no hole-conditional guarantee"}
