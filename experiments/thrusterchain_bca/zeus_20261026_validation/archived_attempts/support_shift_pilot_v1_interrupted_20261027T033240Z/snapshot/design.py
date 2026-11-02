"""Proposal-only data planning/math. No learner, environment or launcher imports."""
import numpy as np

FACTOR = 19.29619572569095
CUTOFF = 0.5
SEEDS = {"broad_rows": 2026103051, "gradient_probes": 2026103052,
         "scale_bank": 2026103053, "calibration_rows": 2026103054}


def actions_checked(actions):
    value = np.asarray(actions)
    if value.ndim not in (2, 3) or value.shape[-1] != 4:
        raise ValueError("Expected rows or episodes of four-coordinate actions")
    if not np.all(np.isfinite(value)) or np.any(np.abs(value) > 1):
        raise ValueError("Actions must be finite and in original [-1,1] coordinates")
    return value


def support_mask(actions):
    return actions_checked(actions)[..., 0] <= CUTOFF


def training_rows(actions, episode_ids):
    """Return intact flattened row indices. Input must be train-role episodes only."""
    actions = actions_checked(actions)
    ids = np.asarray(episode_ids)
    if actions.ndim != 3 or ids.shape != (len(actions),) or len(np.unique(ids)) != len(ids):
        raise ValueError("One unique source episode ID per training episode required")
    mask = support_mask(actions)
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Required episode has no eligible action; never backfill")
    rng = np.random.Generator(np.random.PCG64(SEEDS["broad_rows"]))
    horizon = actions.shape[1]
    broad = []
    for index in np.argsort(ids, kind="stable"):
        rows = rng.choice(horizon, int(counts[index]), replace=False)
        broad.extend((index * horizon + rows).tolist())
    return {"restricted": np.flatnonzero(mask.reshape(-1)),
            "broad_count_matched": np.sort(np.asarray(broad, dtype=np.int64))}


def one_per_episode(eligible_rows, episode_ids, horizon, seed):
    ids = np.asarray(episode_ids)
    rows = np.asarray(eligible_rows, dtype=np.int64)
    if (rows.ndim != 1 or len(np.unique(rows)) != len(rows)
            or len(np.unique(ids)) != len(ids) or horizon <= 0
            or np.any(rows < 0) or np.any(rows >= len(ids) * horizon)):
        raise ValueError("Invalid eligible source rows/episode identities")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for index in np.argsort(ids, kind="stable"):
        candidates = np.sort(rows[rows // horizon == index])
        if len(candidates) == 0:
            raise ValueError("Required episode has no eligible row; never backfill")
        result.append(int(candidates[rng.integers(len(candidates))]))
    return np.asarray(result, dtype=np.int64)


def select_intact_transitions(transitions, rows):
    """Select whole existing tuples; never replace an action or successor."""
    required = {"obs", "action", "reward", "next_obs", "done"}
    if set(transitions) != required:
        raise ValueError("Complete one-step transition fields are required")
    sizes = {len(np.asarray(value)) for value in transitions.values()}
    rows = np.asarray(rows, dtype=np.int64)
    if len(sizes) != 1 or rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise ValueError("Transition rows misaligned or duplicated")
    size = next(iter(sizes))
    if np.any(rows < 0) or np.any(rows >= size):
        raise ValueError("Out-of-range source transition")
    return {key: np.asarray(value)[rows].copy() for key, value in transitions.items()}


def distance_action_gradient(actions, anchor_actions):
    actions, anchors = actions_checked(actions), actions_checked(anchor_actions)
    if actions.shape != anchors.shape or actions.ndim != 2:
        raise ValueError("Action and frozen anchor rows must align")
    return (2 * FACTOR / 4) * (actions.astype(np.float64) - anchors.astype(np.float64))


def match_coefficient(unscaled_bca_action_gradients, distance_action_gradients):
    bca = np.asarray(unscaled_bca_action_gradients, dtype=np.float64)
    distance = np.asarray(distance_action_gradients, dtype=np.float64)
    if bca.ndim != 2 or bca.shape[-1] != 4 or bca.shape != distance.shape or len(bca) == 0:
        raise ValueError("Aligned per-row action gradients required")
    if not np.all(np.isfinite(bca)) or not np.all(np.isfinite(distance)):
        raise ValueError("Nonfinite gradients; stop without a fallback coefficient")
    gb = float(np.sqrt(np.mean(np.sum((3.0 * bca) ** 2, axis=-1))))
    gd = float(np.sqrt(np.mean(np.sum(distance ** 2, axis=-1))))
    if not np.isfinite(gb + gd) or gb <= 1e-6 or gd <= 1e-6:
        raise ValueError("Degenerate matching norms; stop without rematching")
    coefficient = gb / gd
    if not np.isfinite(coefficient) or coefficient <= 0 or coefficient > np.finfo(np.float32).max:
        raise ValueError("Coefficient not representable; no clipping allowed")
    return {"bca_gradient_rms": gb, "distance_gradient_rms": gd,
            "distance_coefficient": coefficient,
            "scope": "One frozen-reference match on logged training actions, not on-policy or trajectory gradient matching"}


def maximum_counts():
    cells, branches = 4, 12
    return {"warmup_states": cells, "final_policies": branches, "untrained_reference_policies": cells,
            "critic_updates": cells * 10000 + branches * 2000,
            "actor_updates": cells * 5000 + branches * 1000,
            "auxiliary_updates": cells * 10000 + branches * 2000,
            "calibrations": cells + branches, "posterior_draws": 1000 * (cells + branches),
            "evaluation_episodes": (branches + cells) * 32, "evaluation_transitions": (branches + cells) * 32 * 100,
            "replay_rows_drawn_with_replacement": (cells * 5000 + branches * 1000) * 256,
            "gradient_probe_rows": cells * 768, "global_seconds": 9000}
