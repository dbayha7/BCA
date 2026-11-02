"""Fixed training-only control math and explicit budget; no scientific dispatch."""
import numpy as np

FACTOR = 19.29619572569095


def distance_action_gradient(actions, anchor_actions):
    a, b = np.asarray(actions, np.float64), np.asarray(anchor_actions, np.float64)
    if a.ndim != 2 or a.shape[-1] != 4 or a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Aligned finite four-coordinate actions required")
    if np.any(np.abs(a) > 1) or np.any(np.abs(b) > 1):
        raise ValueError("Original action bounds required")
    return (2 * FACTOR / 4) * (a - b)


def match_coefficient(unscaled_bca_action_gradients, distance_action_gradients):
    bca, distance = np.asarray(unscaled_bca_action_gradients, np.float64), np.asarray(distance_action_gradients, np.float64)
    if bca.ndim != 2 or bca.shape[-1] != 4 or bca.shape != distance.shape or not len(bca):
        raise ValueError("Aligned nonempty action gradients required")
    if not np.all(np.isfinite(bca)) or not np.all(np.isfinite(distance)):
        raise ValueError("Nonfinite gradients; no fallback or rematch")
    gb = float(np.sqrt(np.mean(np.sum((3 * bca)**2, axis=-1))))
    gd = float(np.sqrt(np.mean(np.sum(distance**2, axis=-1))))
    if not np.isfinite(gb + gd) or gb <= 1e-6 or gd <= 1e-6:
        raise ValueError("Degenerate training-only matching norm; stop")
    coefficient = gb / gd
    if not np.isfinite(coefficient) or not 0 < coefficient <= np.finfo(np.float32).max:
        raise ValueError("Coefficient cannot be represented; no clipping")
    return dict(bca_gradient_rms=gb, distance_gradient_rms=gd, distance_coefficient=coefficient,
                scope="RMS action-gradient match on logged training actions only; not on-policy matching",
                live_fork_distance_gradient_is_zero=True)


def maximum_counts():
    cells, branches = 3 * 3, 3 * 3 * 3
    return dict(warmup_states=cells, branches=branches, training_jobs=cells + branches,
        critic_updates=(cells + branches) * 10000,
        actor_updates=(cells + branches) * 5000,
        auxiliary_updates=(cells + branches) * 10000,
        calibrations=cells + branches, posterior_draws=1000 * (cells + branches),
        replay_row_draws=(cells + branches) * 5000 * 256,
        critic_row_uses=(cells + branches) * 10000 * 256,
        auxiliary_row_uses=(cells + branches) * 10000 * 256,
        gradient_probe_rows=cells * 768,
        free_policy_snapshots=72, q_policy_snapshots=63,
        evaluation_episodes=72 * 2 * 32, evaluation_free_steps=72 * 2 * 32 * 100,
        q_values=63 * 2 * 32 * 6, q_continuation_steps=63 * 2 * 6 * 8 * (100+75+50+25),
        evaluation_prefix_steps=2 * 8 * (0+25+50+75), parity_steps=104,
        simulator_transition_cap=1975304, total_timeout_seconds=43200)
