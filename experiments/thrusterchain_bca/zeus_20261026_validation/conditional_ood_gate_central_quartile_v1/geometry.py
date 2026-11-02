"""Pure, array-in/receipt-out conditional-support diagnostics.

No file reads, simulator imports, learner construction or data mutation occurs.
The caller authenticates episode roles and freezes every setting before loading
new gate data. A mask identifies constructed support removal, not action quality.
"""
import hashlib
import time

import numpy as np

CENTER = 0.0
HALF_WIDTH = 0.25
LOAD_CUTOFF = 0.4
ROW_CAP = 64
TIME_BIN = 10
REFERENCE_PER_STRATUM = 32
REFERENCE_QUANTILE = 0.95
MAX_REFERENCE_QUERIES = 640
MAX_ANCHOR_QUERIES = 64
MAX_DISTANCE_PAIRS = 76800 * (MAX_REFERENCE_QUERIES + MAX_ANCHOR_QUERIES)
STD_ZERO_THRESHOLD = 1e-12
CONTROL_SEED = 2026102811
CALIBRATION_SEED = 2026102362
CALIBRATION_REPLACEMENT_SEED = 2026102812
CONDITIONS = ("full", "hole", "random")


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


def support_report(training_observations, training_actions, training_episode_ids,
                   training_times, anchor_observations, anchor_episode_ids,
                   anchor_times, keep_masks, metric, *, source_row_ids=None,
                   row_cap=ROW_CAP, time_bin=TIME_BIN,
                   reference_per_stratum=REFERENCE_PER_STRATUM,
                   reference_quantile=REFERENCE_QUANTILE,
                   deadline=None, progress=None, ledger=None):
    """Training-relative state typicalness and marginal-a0 occupancy.

    No absolute radius is declared. In each time/load stratum, a fixed hash
    ordering selects one row per distinct training episode, up to exactly 32.
    The higher 95 percent quantile of these rows' leave-whole-episode-out 64NN
    distances defines a training-only reference. Missing reference coverage
    fails the stratum without pooling, backfilling or changing the quantile.
    Query neighbors are fixed before retention and never refilled. This does
    not prove dense full-state coverage or joint four-dimensional action
    support, and it does not use actions to choose neighbors or thresholds.

    Smaller row/reference counts are exposed only for synthetic fixtures;
    production callers bind the defaults to the frozen contract.
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
    _require(type(row_cap) is int and 0 < row_cap <= ROW_CAP
             and type(time_bin) is int and 0 < time_bin <= 100
             and 100 % time_bin == 0
             and type(reference_per_stratum) is int and 0 < reference_per_stratum <= REFERENCE_PER_STRATUM
             and np.isfinite(reference_quantile) and 0 <= reference_quantile <= 1,
             "Invalid neighborhood settings")
    _require(0 < n <= 76800 and len(anchors) <= MAX_ANCHOR_QUERIES,
             "Support row/anchor cap exceeded")
    _require(2 * (100 // time_bin) * reference_per_stratum <= MAX_REFERENCE_QUERIES,
             "Support reference cap exceeded")
    z = (train[:, :13] - mean) / scale
    az = (anchors[:, :13] - mean) / scale
    output = []
    ledger = {} if ledger is None else ledger
    for kind in ('reference_queries', 'anchor_queries', 'distance_pairs'):
        for state in ('attempted', 'completed'):
            ledger.setdefault('support_' + kind + '_' + state, 0)

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Support deadline reached before distance dispatch')

    def ranked(query, excluded_episode, candidates, kind):
        check_deadline()
        candidates = candidates[episodes[candidates] != excluded_episode]
        count_key = 'support_' + kind + '_queries_attempted'
        count_cap = MAX_REFERENCE_QUERIES if kind == 'reference' else MAX_ANCHOR_QUERIES
        _require(ledger[count_key] + 1 <= count_cap, 'Support query cap exceeded')
        _require(ledger['support_distance_pairs_attempted'] + len(candidates) <= MAX_DISTANCE_PAIRS,
                 'Support distance-pair cap exceeded')
        ledger[count_key] += 1
        ledger['support_distance_pairs_attempted'] += len(candidates)
        if progress is not None:
            progress(dict(ledger))
        distances = np.sqrt(np.mean((z[candidates] - query) ** 2, axis=1))
        _require(np.all(np.isfinite(distances)), 'Nonfinite support distance')
        order = np.lexsort((rows[candidates], distances))
        ledger['support_' + kind + '_queries_completed'] += 1
        ledger['support_distance_pairs_completed'] += len(candidates)
        return candidates[order], distances[order]

    pools, references = {}, {}
    for time_index in range(100 // time_bin):
        for high in (False, True):
            check_deadline()
            key = (time_index, high)
            pool = np.flatnonzero((times // time_bin == time_index) & (region['high'] == high))
            pools[key] = pool
            # The ordering depends on source IDs alone, never state distances,
            # actions, retained rows, anchors, critic values or outcomes.
            ordered = sorted(pool.tolist(), key=lambda j: (
                hashlib.sha256(('central-quartile-reference-v1:' + str(int(rows[j]))).encode('ascii')).digest(),
                int(rows[j])))
            selected, seen_episodes = [], set()
            for j in ordered:
                if int(episodes[j]) not in seen_episodes:
                    selected.append(j)
                    seen_episodes.add(int(episodes[j]))
                if len(selected) == reference_per_stratum:
                    break
            distances = []
            for j in selected:
                _, ordered_distances = ranked(z[j], episodes[j], pool, 'reference')
                distances.append(float(ordered_distances[row_cap-1]) if len(ordered_distances) >= row_cap else None)
            available = len(selected) == reference_per_stratum and all(v is not None for v in distances)
            threshold = float(np.quantile(distances, reference_quantile, method='higher')) if available else None
            references[key] = {
                'time_bin': time_index, 'high_load': high, 'available': available,
                'source_rows_in_stratum': len(pool),
                'distinct_episodes_in_stratum': len(np.unique(episodes[pool])),
                'reference_source_row_ids': rows[selected].tolist(),
                'reference_episode_ids': episodes[selected].tolist(),
                'reference_kth_distances': distances,
                'reference_rows': len(selected), 'threshold': threshold,
                'failure': None if available else 'insufficient_distinct_reference_episodes_or_neighbor_rows'}

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
        key = (int(atimes[i] // time_bin), bool(high))
        reference = references[key]
        ranked_primary, distances = ranked(az[i], anchor_ids[i], pools[key], 'anchor')
        selected = ranked_primary[:row_cap]
        kth = float(distances[row_cap-1]) if len(distances) >= row_cap else None
        locality_pass = reference['available'] and kth is not None and kth <= reference['threshold']
        output.append({"anchor_index": i, "anchor_episode_id": int(anchor_ids[i]),
                       "time": int(atimes[i]), "high_load": bool(high),
                       "load": load, "load_depth": load - LOAD_CUTOFF,
                       "same_side_available_before_cap": len(ranked_primary),
                       "distance_to_cap_neighbor": kth,
                       "locality_reference_threshold": reference['threshold'],
                       "locality_reference_available": reference['available'],
                       "locality_pass": bool(locality_pass),
                       "neighbor_source_row_ids": rows[selected].tolist(),
                       "neighbor_distances": distances[:row_cap].tolist(),
                       "counts": {name: counts(selected[mask[selected]]) for name, mask in masks.items()}})
    return {"fixed_absolute_radius": None, "row_cap": row_cap, "time_bin_width": time_bin,
            "distance": "sqrt(mean(((obs[:13]-anchor[:13])/original_train_scale)**2))",
            "reference_per_stratum": reference_per_stratum,
            "reference_quantile": reference_quantile, "quantile_method": "higher",
            "reference_selection": "SHA256('central-quartile-reference-v1:'+source_row_id), distinct episode first; independent of actions and outcomes",
            "reference_strata": list(references.values()),
            "all_reference_strata_available": all(r['available'] for r in references.values()),
            "whole_anchor_episode_excluded": True,
            "neighbors_fixed_before_retention": True, "other_action_coordinates_matched": False,
            "scope": "Training-relative state typicalness and marginal a0 occupancy, not absolute state density or joint 4D action support",
            "distance_accounting": {k: v for k, v in ledger.items() if k.startswith('support_')},
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
