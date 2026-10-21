"""Method-neutral, exact finite-model diagnostics; this module does no training.

Policies have shape [H,S,A]. Refusal is a probability with shape [H,S]; a
refused decision executes an explicitly supplied fallback policy. Optional
estimates always refer to the proposed policy (``value_target='policy'``) or
the optimal policy (``'optimal'``), never implicitly to the deployed mixture.
Interval coverage is weighted coverage across queries in one run. Repeated,
independent datasets are needed to assess frequentist calibration.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MethodOutput:
    policy: np.ndarray
    q_values: Optional[np.ndarray] = None
    v_values: Optional[np.ndarray] = None
    q_intervals: Optional[np.ndarray] = None
    v_intervals: Optional[np.ndarray] = None
    refusal: Optional[np.ndarray] = None
    value_target: str = "policy"


def _array(value, shape, name, *, nonnegative=False):
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _policy(value, shape, name):
    result = _array(value, shape, name, nonnegative=True)
    if not np.allclose(result.sum(axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError(f"{name} action probabilities must sum to one")
    return result


def _weights(value, shape):
    return np.ones(shape) if value is None else _array(value, shape, "weights", nonnegative=True)


def _mean(value, weights):
    scale = float(np.max(weights)) if weights.size else 0.0
    if scale == 0:
        return None
    normalized = weights / scale
    normalized /= normalized.sum()
    return float(np.sum(value * normalized))


def prediction_metrics(estimate, target, weights=None):
    """Weighted point error; return None for an empty evaluation stratum."""
    target = np.asarray(target, dtype=float)
    target = _array(target, target.shape, "target")
    estimate = _array(estimate, target.shape, "estimate")
    weights = _weights(weights, target.shape)
    if not weights.any():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        error = estimate - target
    if not np.isfinite(error).all():
        raise ValueError("prediction differences must be finite")
    active = weights > 0
    scale = float(np.max(np.abs(error)[active]))
    return {
        "bias": _mean(error, weights),
        "mae": _mean(np.abs(error), weights),
        "rmse": 0.0 if scale == 0 else float(scale * np.sqrt(_mean((error[active] / scale) ** 2, weights[active]))),
        "max_absolute_error": scale,
    }


def interval_metrics(intervals, target, weights=None, *, level=0.95):
    """Closed-interval query coverage and width; finite endpoints required."""
    if not 0 < level < 1:
        raise ValueError("interval level must lie strictly between zero and one")
    target = np.asarray(target, dtype=float)
    target = _array(target, target.shape, "target")
    bounds = _array(intervals, target.shape + (2,), "intervals")
    if np.any(bounds[..., 0] > bounds[..., 1]):
        raise ValueError("interval lower endpoints must not exceed upper endpoints")
    with np.errstate(over="ignore", invalid="ignore"):
        width = bounds[..., 1] - bounds[..., 0]
    if not np.isfinite(width).all():
        raise ValueError("interval widths exceed the finite floating-point range")
    weights = _weights(weights, target.shape)
    if not weights.any():
        return None
    covered = (bounds[..., 0] <= target) & (target <= bounds[..., 1])
    coverage = _mean(covered, weights)
    return {
        "query_coverage": coverage,
        "nominal_level": float(level),
        "coverage_minus_nominal": coverage - float(level),
        "mean_width": _mean(width, weights),
        "lower_miss_rate": _mean(target < bounds[..., 0], weights),
        "upper_miss_rate": _mean(target > bounds[..., 1], weights),
    }


def ranking_metrics(estimate, target, state_weights=None):
    """Ranking error; truth ties excluded, predicted pair ties count one half."""
    target = np.asarray(target, dtype=float)
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("rankings require an action axis")
    target = _array(target, target.shape, "target")
    estimate = _array(estimate, target.shape, "estimate")
    weights = _weights(state_weights, target.shape[:-1])
    if not weights.any():
        return None
    weights = weights / weights.max()
    weights /= weights.sum()
    predicted_best = estimate == np.max(estimate, axis=-1, keepdims=True)
    greedy = predicted_best / predicted_best.sum(axis=-1, keepdims=True)
    truth_best = target == np.max(target, axis=-1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        regret = np.max(target, axis=-1) - np.sum(greedy * target, axis=-1)
    if not np.isfinite(regret).all():
        raise ValueError("ranking regrets exceed the finite floating-point range")
    pair_error = 0.0
    pair_mass = 0.0
    for a in range(target.shape[-1]):
        for b in range(a):
            truth_greater = target[..., a] > target[..., b]
            truth_less = target[..., a] < target[..., b]
            eligible = truth_greater | truth_less
            reversed_pair = ((truth_greater & (estimate[..., a] < estimate[..., b]))
                             | (truth_less & (estimate[..., a] > estimate[..., b])))
            predicted_tie = estimate[..., a] == estimate[..., b]
            pair_error += float(np.sum(weights * eligible * (reversed_pair + 0.5 * predicted_tie)))
            pair_mass += float(np.sum(weights * eligible))
    return {
        "greedy_action_regret": _mean(regret, weights),
        "greedy_nonoptimal_action_probability": _mean(np.sum(greedy * ~truth_best, axis=-1), weights),
        "pairwise_disagreement": None if pair_mass == 0 else pair_error / pair_mass,
    }


def _occupancy(P, policy, initial):
    horizon, states, _ = policy.shape
    state = np.zeros((horizon + 1, states))
    state[0] = initial
    sa = np.zeros_like(policy)
    for t in range(horizon):
        sa[t] = state[t, :, None] * policy[t]
        state[t + 1] = np.einsum("sa,san->n", sa[t], P)
    return state, sa


def _state_event_probability(P, policy, initial, mask):
    """Probability of at least one flagged state, including the initial state."""
    survivor = initial * ~mask[0]
    for t in range(policy.shape[0]):
        survivor = np.einsum("s,sa,san->n", survivor, policy[t], P)
        survivor *= ~mask[t + 1]
    return float(np.clip(1 - survivor.sum(), 0, 1))


def _action_event_probability(P, policy, initial, mask):
    survivor = initial.copy()
    for t in range(policy.shape[0]):
        survivor = np.einsum("s,sa,sa,san->n", survivor, policy[t], ~mask[t], P)
    return float(np.clip(1 - survivor.sum(), 0, 1))


def _mask(value, shape, trailing_shape, name):
    result = np.asarray(value)
    if result.shape == trailing_shape:
        result = np.broadcast_to(result, shape)
    if result.shape != shape or not np.all(np.isin(result, [False, True])):
        raise ValueError(f"{name} must be a boolean mask with shape {shape} or {trailing_shape}")
    return result.astype(bool)


def _value_reference(value, reference, state_weights, name):
    """Allow a V table with or without its terminal row, weighted zero at H."""
    shape = np.asarray(value).shape
    if shape == reference.shape:
        return reference, np.concatenate([state_weights, np.zeros_like(state_weights[:1])])
    if shape == reference[:-1].shape:
        return reference[:-1], state_weights
    raise ValueError(f"{name} must have shape {reference.shape} or {reference[:-1].shape}")


def evaluate_method(
    mdp, output, *, initial_distribution=None, fallback_policy=None,
    baseline_policy=None, logging_mass=None, retained_counts=None,
    critical_mask=None, failure_mask=None, query_weights=None, interval_level=0.95,
):
    """Return a JSON-ready report using the finite companion's exact oracle.

    ``logging_mass`` is population mass [H,S,A], not empirical counts. Its
    normalized distribution is compared with normalized deployed occupancy.
    ``retained_counts`` is a separate integer [H,S,A] table. Default prediction
    queries are uniform over the full finite time/state/action grid, including
    unreachable states; pass ``query_weights`` to choose another distribution.
    ``failure_mask`` flags states [S] or [H+1,S]. Failure means any visit.
    Local decision metrics use Q of the fixed baseline as continuation and
    average over deployed state occupancy. They are not trajectory causal losses.
    """
    from .finite import evaluate_policy, solve_optimal

    H, S, A = mdp.horizon, mdp.n_states, mdp.n_actions
    shape = (H, S, A)
    proposed = _policy(output.policy, shape, "policy")
    if output.value_target not in ("policy", "optimal"):
        raise ValueError("value_target must be 'policy' or 'optimal'")
    initial = _array(
        mdp.initial_distribution() if initial_distribution is None else initial_distribution,
        (S,), "initial_distribution", nonnegative=True,
    )
    if not np.isclose(initial.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("initial_distribution must sum to one")
    refusal = np.zeros((H, S)) if output.refusal is None else _array(output.refusal, (H, S), "refusal", nonnegative=True)
    if np.any(refusal > 1):
        raise ValueError("refusal probabilities must be at most one")
    if np.any(refusal > 0) and fallback_policy is None:
        raise ValueError("nonzero refusal requires an explicit fallback_policy")
    fallback = proposed if fallback_policy is None else _policy(fallback_policy, shape, "fallback_policy")
    deployed = proposed * (1 - refusal[..., None]) + fallback * refusal[..., None]
    baseline = None if baseline_policy is None else _policy(baseline_policy, shape, "baseline_policy")
    query = _weights(query_weights, shape).copy()
    if not query.any():
        raise ValueError("query_weights must have positive total mass")
    query /= query.max()
    query /= query.sum()
    state_query = query.sum(axis=-1)
    proposed_dp = evaluate_policy(mdp, proposed)
    deployed_dp = evaluate_policy(mdp, deployed)
    optimal_dp = solve_optimal(mdp)
    baseline_dp = None if baseline is None else evaluate_policy(mdp, baseline)
    truth = proposed_dp if output.value_target == "policy" else optimal_dp
    state, sa = _occupancy(mdp.P, deployed, initial)
    decision_weights = state[:-1] / H
    J_proposed = float(initial @ proposed_dp.V[0])
    J_deployed = float(initial @ deployed_dp.V[0])
    J_optimal = float(initial @ optimal_dp.V[0])
    J_baseline = None if baseline_dp is None else float(initial @ baseline_dp.V[0])
    result = {
        "value_target": output.value_target,
        "query_distribution": "uniform_full_grid" if query_weights is None else "provided_weights",
        "returns": {
            "proposed": J_proposed, "deployed": J_deployed, "optimal": J_optimal,
            "baseline": J_baseline, "regret": max(0.0, J_optimal - J_deployed),
            "harm_vs_baseline": None if J_baseline is None else max(0.0, J_baseline - J_deployed),
            "gain_vs_baseline": None if J_baseline is None else J_deployed - J_baseline,
        },
        "failure": None, "support": None,
        "errors": {"q": None, "v": None, "q_supported": None, "q_unsupported": None},
        "calibration": {"q": None, "v": None}, "ranking": None,
        "refusal": {
            "decision_refusal_rate": _mean(refusal, decision_weights),
            "query_acceptance_rate": _mean(1 - refusal, state_query),
            "accepted_q_error": None,
            "fallback_return": None if fallback_policy is None else float(initial @ evaluate_policy(mdp, fallback).V[0]),
        },
        "decisions": None,
    }
    if failure_mask is not None:
        failed = _mask(failure_mask, (H + 1, S), (S,), "failure_mask")
        result["failure"] = {
            "any_visit_probability": _state_event_probability(mdp.P, deployed, initial, failed),
            "expected_fraction_of_state_times": float(np.sum(state * failed) / (H + 1)),
        }
    if output.q_values is not None:
        result["errors"]["q"] = prediction_metrics(output.q_values, truth.Q, query)
        result["ranking"] = ranking_metrics(output.q_values, truth.Q, state_query)
        result["refusal"]["accepted_q_error"] = prediction_metrics(output.q_values, truth.Q, query * (1 - refusal[..., None]))
    if output.v_values is not None:
        reference, weights = _value_reference(output.v_values, truth.V, state_query, "v_values")
        result["errors"]["v"] = prediction_metrics(output.v_values, reference, weights)
    if output.q_intervals is not None:
        result["calibration"]["q"] = interval_metrics(output.q_intervals, truth.Q, query, level=interval_level)
    if output.v_intervals is not None:
        bounds = np.asarray(output.v_intervals)
        if bounds.ndim < 1 or bounds.shape[-1] != 2:
            raise ValueError("v_intervals must end with a lower/upper endpoint axis")
        reference, weights = _value_reference(bounds[..., 0], truth.V, state_query, "v_intervals")
        result["calibration"]["v"] = interval_metrics(bounds, reference, weights, level=interval_level)
    if logging_mass is not None or retained_counts is not None:
        support = {"population": None, "counts": None}
        critical_value = getattr(mdp, "critical_mask", None) if critical_mask is None else critical_mask
        if callable(critical_value):
            critical_value = critical_value()
        critical = None if critical_value is None else _mask(critical_value, shape, (S, A), "critical_mask")
        if logging_mass is not None:
            mass = _array(logging_mass, shape, "logging_mass", nonnegative=True)
            present = mass > 0
            has_mass = bool(present.any())
            mu = mass / mass.max() if has_mass else mass.copy()
            if has_mass:
                mu /= mu.sum()
            target_mass = sa / H
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                ratio = np.divide(target_mass, mu, out=np.zeros_like(mu), where=present)
            if not np.isfinite(ratio).all():
                raise ValueError("support ratios exceed the finite floating-point range")
            support["population"] = {
                "unsupported_deployed_mass": float(np.sum(target_mass * ~present)),
                "unsupported_action_hit_probability": _action_event_probability(mdp.P, deployed, initial, ~present),
                "occupancy_total_variation": None if not has_mass else float(0.5 * np.abs(target_mass - mu).sum()),
                "max_ratio_on_supported_entries": None if not present.any() else float(ratio[present].max()),
                "critical_unsupported_deployed_mass": None if critical is None else float(np.sum(target_mass * critical * ~present)),
            }
            if output.q_values is not None:
                result["errors"]["q_supported"] = prediction_metrics(output.q_values, truth.Q, query * present)
                result["errors"]["q_unsupported"] = prediction_metrics(output.q_values, truth.Q, query * ~present)
        if retained_counts is not None:
            counts = _array(retained_counts, shape, "retained_counts", nonnegative=True)
            if np.any(counts != np.floor(counts)):
                raise ValueError("retained_counts must contain integers")
            support["counts"] = {
                "retained_transitions": int(counts.sum()),
                "observed_time_state_action_entries": int(np.count_nonzero(counts)),
                "critical_retained_transitions": None if critical is None else int(np.sum(counts * critical)),
                "critical_observed_entries": None if critical is None else int(np.count_nonzero(counts * critical)),
                "critical_unobserved_entries": None if critical is None else int(np.count_nonzero(critical & (counts == 0))),
            }
        result["support"] = support
    if baseline_dp is not None:
        score_proposed = np.sum(proposed * baseline_dp.Q, axis=-1)
        score_deployed = np.sum(deployed * baseline_dp.Q, axis=-1)
        score_fallback = np.sum(fallback * baseline_dp.Q, axis=-1)
        score_baseline = np.sum(baseline * baseline_dp.Q, axis=-1)
        result["decisions"] = {
            "baseline_continuation_mean_disadvantage": _mean(np.maximum(score_baseline - score_deployed, 0), decision_weights),
            "baseline_continuation_harmful_policy_distribution_rate": _mean(score_deployed < score_baseline - 1e-12, decision_weights),
            "baseline_continuation_harmful_action_probability": _mean(np.sum(deployed * (baseline_dp.Q < score_baseline[..., None] - 1e-12), axis=-1), decision_weights),
            "refused_one_step_opportunity_vs_fallback": _mean(refusal * np.maximum(score_proposed - score_fallback, 0), decision_weights),
        }
    return result
