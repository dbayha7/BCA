"""BCA corl training reward range.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

import hashlib
import numpy as np


class RewardRangeError(ValueError):

    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


def normalize_training_rewards(
    rewards, terminals, training_indices, *, max_episode_steps, reward_scale, reward_bias
):
    r, done, idx = (
        np.asarray(rewards, np.float64),
        np.asarray(terminals),
        np.asarray(training_indices),
    )
    if (
        r.ndim != 1
        or not len(r)
        or done.shape != r.shape
        or (not np.all(np.isin(done, (0, 1))))
        or (not np.all(np.isfinite(r)))
    ):
        raise ValueError("finite aligned rewards and binary terminal vectors required")
    if (
        isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
        or max_episode_steps <= 0
    ):
        raise ValueError("max_episode_steps must be a positive integer")
    if (
        idx.ndim != 1
        or not len(idx)
        or (not np.issubdtype(idx.dtype, np.integer))
        or np.any(idx < 0)
        or np.any(idx >= len(r))
        or (len(np.unique(idx)) != len(idx))
    ):
        raise ValueError("training_indices must identify unique nonempty in-range rows")
    if (
        isinstance(reward_scale, bool)
        or isinstance(reward_bias, bool)
        or (not np.all(np.isfinite([reward_scale, reward_bias])))
    ):
        raise ValueError("finite native affine scale/bias required")
    training = np.zeros(len(r), bool)
    training[idx] = True
    segments, selected, excluded, fit_rows, fit_returns = ([], [], [], [], [])
    start, length = (0, 0)
    for i, terminal in enumerate(done):
        length += 1
        if terminal or length == max_episode_steps:
            stop = i + 1
            segment_id = len(segments)
            eligible = bool(np.all(training[start:stop]))
            segments.append(
                {
                    "id": segment_id,
                    "start": start,
                    "stop": stop,
                    "length": length,
                    "terminal_boundary": bool(terminal),
                    "maximum_length_boundary": length == max_episode_steps,
                    "training_rows": int(training[start:stop].sum()),
                    "heldout_rows": int((~training[start:stop]).sum()),
                    "selected": eligible,
                }
            )
            if eligible:
                value = 0.0
                for reward in r[start:stop]:
                    value += float(reward)
                fit_returns.append(value)
                selected.append(segment_id)
                fit_rows.extend(range(start, stop))
            else:
                excluded.append(segment_id)
            start, length = (stop, 0)
    membership = np.asarray(fit_rows, dtype="<i8")
    metadata = {
        "mode": "training_only_original_native_segments_v1",
        "source_rule": "corl_common.return_reward_range: terminal or max_episode_steps in original converted order",
        "max_episode_steps": max_episode_steps,
        "converted_rows": len(r),
        "training_rows": int(training.sum()),
        "heldout_rows": int((~training).sum()),
        "segments": segments,
        "selected_segment_ids": selected,
        "excluded_segment_ids": excluded,
        "fit_converted_indices": fit_rows,
        "fit_membership_sha256": hashlib.sha256(membership.tobytes()).hexdigest(),
        "selected_segment_returns": fit_returns,
        "ignored_trailing_segment": {
            "start": start,
            "stop": len(r),
            "length": length,
            "training_rows": int(training[start:].sum()),
            "heldout_rows": int((~training[start:]).sum()),
        },
        "fit_min_return": None,
        "fit_max_return": None,
        "return_range": None,
        "normalization_multiplier": None,
        "reward_scale": float(reward_scale),
        "reward_bias": float(reward_bias),
        "limitations": [
            "Training-only extension, not full-data native equivalence for reserved arms",
            "Native segments can cross D4RL timeout/discontinuity blocks; touching heldout excludes the whole segment",
            "Partial fragments and trailing incomplete rows do not fit the range",
            "Heldout rewards receive the same fitted transform but never select its extrema",
        ],
    }
    if not fit_returns:
        raise RewardRangeError(
            "no complete original native segment is entirely in training", metadata
        )
    if not np.all(np.isfinite(fit_returns)):
        raise RewardRangeError("nonfinite completed training return", metadata)
    lo, hi = (min(fit_returns), max(fit_returns))
    span = hi - lo
    metadata.update(fit_min_return=lo, fit_max_return=hi, return_range=span)
    if not np.isfinite(span) or span <= 0:
        raise RewardRangeError("training return range must be positive and finite", metadata)
    multiplier = max_episode_steps / span
    metadata["normalization_multiplier"] = multiplier
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise RewardRangeError(
            "training normalization multiplier must be positive and finite", metadata
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = r / span
        transformed = transformed * max_episode_steps
        if reward_scale != 1.0 or reward_bias != 0.0:
            transformed = transformed * reward_scale + reward_bias
    if not np.all(np.isfinite(transformed)):
        raise RewardRangeError(
            "nonfinite reward after training-fitted native transform", metadata
        )
    return (transformed, metadata)
