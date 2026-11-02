"""Saved-development mechanism descriptions; no selection fitting or inference.

States are the units of aggregation. Candidate pairs are dependent comparisons
within a state, never extra independent observations. No file/model imports.
"""
import math

OUTCOMES = ("signed_error", "positive_optimism", "absolute_error", "candidate_regret")


def _finite(values):
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Expected finite numeric observations")
        result.append(float(value))
    if not result:
        raise ValueError("Empty state vector")
    return result


def _mean(values):
    return None if not values else math.fsum(value / len(values) for value in values)


def centered(values):
    values = _finite(values)
    mean = _mean(values)
    result = [value - mean for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Centered values overflowed")
    return result


def _unit_centered(values):
    values = _finite(values)
    if len(values) < 2:
        return None, "fewer_than_two_candidates"
    if all(value == values[0] for value in values):
        return None, "constant_vector"
    residual = centered(values)
    maximum = max(abs(value) for value in residual)
    if maximum == 0:
        return None, "zero_centered_variance"
    scaled = [value / maximum for value in residual]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return [value / norm for value in scaled], None


def _undefined(reason, n):
    return {"defined": False, "value": None, "reason": reason, "candidate_count": n}


def pearson(x, y):
    x, y = _finite(x), _finite(y)
    if len(x) != len(y):
        raise ValueError("Correlation vectors must align")
    a, reason_a = _unit_centered(x)
    b, reason_b = _unit_centered(y)
    if a is None or b is None:
        return _undefined("predictor_" + reason_a if a is None else "outcome_" + reason_b, len(x))
    value = max(-1., min(1., math.fsum(first * second for first, second in zip(a, b))))
    return {"defined": True, "value": value, "reason": None, "candidate_count": len(x)}


def average_ranks(values):
    values = _finite(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(x, y):
    return pearson(average_ranks(x), average_ranks(y))


def partial_pearson(x, y, distance, relative_floor):
    """Residual correlation after intercept and one linear distance projection.

    Centered input vectors are unit-normalized first. The residual norm floor
    is therefore relative to each original centered norm, not a unitful floor.
    A constant distance covariate is explicitly undefined by this protocol.
    """
    x, y, distance = _finite(x), _finite(y), _finite(distance)
    if len(x) != len(y) or len(x) != len(distance) or not 0 < relative_floor < 1:
        raise ValueError("Aligned vectors and a relative floor in (0,1) required")
    units = []
    for name, values in (("predictor", x), ("outcome", y), ("distance", distance)):
        unit, reason = _unit_centered(values)
        if unit is None:
            return _undefined(name + "_" + reason, len(x))
        units.append(unit)
    a, b, d = units
    residuals, norms = [], []
    for name, unit in (("predictor", a), ("outcome", b)):
        coefficient = math.fsum(value * covariate for value, covariate in zip(unit, d))
        residual = [value - coefficient * covariate for value, covariate in zip(unit, d)]
        norm = math.sqrt(math.fsum(value * value for value in residual))
        if norm <= relative_floor:
            result = _undefined(name + "_residual_degenerate", len(x))
            result["residual_relative_norm"] = norm
            result["relative_floor"] = relative_floor
            return result
        residuals.append(residual)
        norms.append(norm)
    value = math.fsum((first / norms[0]) * (second / norms[1]) for first, second in zip(*residuals))
    return {"defined": True, "value": max(-1., min(1., value)), "reason": None, "candidate_count": len(x),
            "predictor_residual_relative_norm": norms[0], "outcome_residual_relative_norm": norms[1],
            "relative_floor": relative_floor}


def tolerance(first, second, atol, rtol):
    return atol + rtol * max(abs(first), abs(second))


def _direction(first, second, atol, rtol):
    difference = first - second
    if abs(difference) <= tolerance(first, second, atol, rtol):
        return 0
    return 1 if first > second else -1


def pairwise_concordance(signal, error, atol, rtol):
    signal, error = _finite(signal), _finite(error)
    if len(signal) != len(error):
        raise ValueError("Pairwise vectors must align")
    correct = wrong = signal_ties = error_ties = both_ties = 0
    for i in range(len(signal)):
        for j in range(i + 1, len(signal)):
            sign_signal = _direction(signal[i], signal[j], atol, rtol)
            sign_error = _direction(error[i], error[j], atol, rtol)
            if sign_signal == 0 and sign_error == 0:
                both_ties += 1
            elif sign_signal == 0:
                signal_ties += 1
            elif sign_error == 0:
                error_ties += 1
            elif sign_signal == sign_error:
                correct += 1
            else:
                wrong += 1
    total = len(signal) * (len(signal) - 1) // 2
    comparable = correct + wrong
    tied = signal_ties + error_ties + both_ties
    return {"total_pairs": total, "comparable_pairs": comparable, "correct_pairs": correct,
            "wrong_pairs": wrong, "signal_only_tied_pairs": signal_ties, "error_only_tied_pairs": error_ties,
            "both_tied_pairs": both_ties, "tied_pairs": tied,
            "correct_rate_among_comparable": None if not comparable else correct / comparable,
            "correct_rate_defined": bool(comparable), "tie_fraction": None if not total else tied / total,
            "reason": None if comparable else "no_comparable_pairs",
            "independence_unit": "state, not candidate pair"}


def _first_near_maximum(values, atol, rtol):
    maximum = max(values)
    close = [i for i, value in enumerate(values) if abs(value - maximum) <= atol + rtol * abs(maximum)]
    return close[0], close


def decision_direction(q1, signal, distance, values, raw_q_index, atol, rtol):
    q1, signal, distance, values = map(_finite, (q1, signal, distance, values))
    if len({len(q1), len(signal), len(distance), len(values)}) != 1:
        raise ValueError("Decision vectors must align")
    expected_raw, raw_near = _first_near_maximum(q1, atol, rtol)
    if isinstance(raw_q_index, bool) or not isinstance(raw_q_index, int) or raw_q_index != expected_raw:
        raise ValueError("Saved raw-Q choice differs from the reviewed first-near-maximum rule")
    best_index, best_near = _first_near_maximum(values, atol, rtol)
    maximum = max(values)
    regret = maximum - values[raw_q_index]
    regret_band = tolerance(maximum, values[raw_q_index], atol, rtol)
    eligible = regret > regret_band
    u_sign = _direction(signal[raw_q_index], signal[best_index], atol, rtol)
    d_sign = _direction(distance[raw_q_index], distance[best_index], atol, rtol)
    joint = ("both_support" if u_sign == d_sign == 1 else "both_oppose" if u_sign == d_sign == -1
             else "conflicting" if u_sign * d_sign == -1 else "one_or_both_tied")
    label = lambda sign: "supports_better_candidate" if sign > 0 else "opposes_better_candidate" if sign < 0 else "tied"
    return {"raw_q_index": raw_q_index, "best_value_representative_index": best_index,
            "best_value_near_indices": best_near, "raw_q_near_indices": raw_near,
            "best_value_ambiguous": len(best_near) > 1, "raw_q_score_ambiguous": len(raw_near) > 1,
            "exact_candidate_bank_regret": regret, "regret_numerical_band": regret_band,
            "regret_above_band": eligible, "delta_u": signal[raw_q_index] - signal[best_index],
            "delta_actor_distance": distance[raw_q_index] - distance[best_index],
            "u_direction": label(u_sign), "distance_direction": label(d_sign), "joint_direction": joint,
            "eligible_unambiguous_best": eligible and len(best_near) == 1,
            "scope": "Penalty contrast toward a better candidate; not a policy improvement or causal claim"}


def partial_pearson_two_controls(x, y, distance, q1, relative_floor, rank_floor):
    """Intercept plus full-rank centered distance/Q1 projection, no fitted selector."""
    vectors = [_finite(v) for v in (x, y, distance, q1)]
    if len({len(v) for v in vectors}) != 1 or not 0 < rank_floor < 1 or not 0 < relative_floor < 1:
        raise ValueError("Aligned vectors and valid relative floors required")
    n = len(vectors[0])
    units = []
    for name, vector in zip(("predictor", "outcome", "distance", "q1"), vectors):
        unit, reason = _unit_centered(vector)
        if unit is None:
            return _undefined(name + "_" + reason, n)
        units.append(unit)
    a, b, d, q = units
    projection = math.fsum(first * second for first, second in zip(q, d))
    q_residual = [first - projection * second for first, second in zip(q, d)]
    # Reorthogonalize the second column before its rank/norm test, avoiding
    # loss of orthogonality for nearly collinear controls.
    correction = math.fsum(first * second for first, second in zip(q_residual, d))
    q_residual = [first - correction * second for first, second in zip(q_residual, d)]
    q_norm = math.sqrt(math.fsum(value * value for value in q_residual))
    if q_norm <= rank_floor:
        result = _undefined("control_rank_deficient", n)
        result.update(control_rank_residual_relative_norm=q_norm, rank_relative_floor=rank_floor)
        return result
    basis = [d, [value / q_norm for value in q_residual]]
    residuals, norms = [], []
    for name, unit in (("predictor", a), ("outcome", b)):
        coefficients = [math.fsum(value * base for value, base in zip(unit, column)) for column in basis]
        residual = [value - math.fsum(coef * column[i] for coef, column in zip(coefficients, basis))
                    for i, value in enumerate(unit)]
        norm = math.sqrt(math.fsum(value * value for value in residual))
        if norm <= relative_floor:
            return {**_undefined(name + "_residual_degenerate", n), "residual_relative_norm": norm,
                    "relative_floor": relative_floor, "control_rank": 2}
        residuals.append(residual)
        norms.append(norm)
    value = math.fsum((first / norms[0]) * (second / norms[1]) for first, second in zip(*residuals))
    return {"defined": True, "value": max(-1., min(1., value)), "reason": None, "candidate_count": n,
            "predictor_residual_relative_norm": norms[0], "outcome_residual_relative_norm": norms[1],
            "control_rank": 2, "control_rank_residual_relative_norm": q_norm,
            "relative_floor": relative_floor, "rank_relative_floor": rank_floor}


def _rms(values):
    values = _finite(values)
    maximum = max(abs(value) for value in values)
    return 0. if maximum == 0 else maximum * math.sqrt(math.fsum((value / maximum) ** 2 for value in values) / len(values))


def _contrast(values):
    raw, contrast = _rms(values), _rms(centered(values))
    return {"raw_rms": raw, "centered_rms": contrast,
            "centered_to_raw_rms": None if raw == 0 else contrast / raw,
            "ratio_defined": raw != 0, "reason": "zero_raw_rms" if raw == 0 else None}


def _u_remaining(u, distance):
    a, reason_a = _unit_centered(u)
    d, reason_d = _unit_centered(distance)
    if a is None or d is None:
        return {"defined": False, "value": None,
                "reason": "u_" + reason_a if a is None else "distance_" + reason_d}
    coefficient = math.fsum(first * second for first, second in zip(a, d))
    norm = math.sqrt(math.fsum((first - coefficient * second) ** 2 for first, second in zip(a, d)))
    return {"defined": True, "value": norm, "reason": None}


def _state(query, batch_name, protocol):
    q1, u, distance, values = [_finite(query[key]) for key in ("q1", "u", "actor_distance", "values")]
    if len({len(q1), len(u), len(distance), len(values)}) != 1 or not 1 <= len(q1) <= protocol["max_candidates_per_state"]:
        raise ValueError("Misaligned or oversized candidate bank")
    if min(u + distance) < 0:
        raise ValueError("Uncertainty and squared actor distance must be nonnegative")
    error = _finite([q - value for q, value in zip(q1, values)])
    best_value = max(values)
    outcomes = {"signed_error": error, "positive_optimism": [max(x, 0.) for x in error],
                "absolute_error": [abs(x) for x in error], "candidate_regret": [best_value - value for value in values]}
    signals = {"u": u, "actor_distance": distance}
    relationships = {signal: {name: {"pearson": pearson(vector, outcome), "spearman": spearman(vector, outcome)}
                               for name, outcome in outcomes.items()} for signal, vector in signals.items()}
    atol, rtol = protocol["tie_atol"], protocol["tie_rtol"]
    direction = decision_direction(q1, u, distance, values, query["raw_q_index"], atol, rtol)
    bca_scores = _finite([q - protocol["kappa_penalty_bca"] * value for q, value in zip(q1, u)])
    control_scores = _finite([q - protocol["kappa_penalty_actor_distance"] * protocol["actor_distance_factor"] * value
                              for q, value in zip(q1, distance)])
    bca_index, bca_near = _first_near_maximum(bca_scores, atol, rtol)
    control_index, control_near = _first_near_maximum(control_scores, atol, rtol)
    delta = values[bca_index] - values[control_index]
    event_direction = _direction(values[bca_index], values[control_index], atol, rtol)
    raw_value = values[query["raw_q_index"]]
    return {"batch": batch_name, "group_id": query["group_id"], "time": query["time"],
            "episode_seed": query["episode_seed"], "candidate_count": len(q1),
            "relationships": relationships,
            "partial_given_distance": partial_pearson(u, error, distance, protocol["residual_relative_l2_floor"]),
            "partial_given_distance_and_q1": partial_pearson_two_controls(u, error, distance, q1,
                protocol["residual_relative_l2_floor"], protocol["control_rank_relative_floor"]),
            "centered": {**{name: centered(vector) for name, vector in signals.items()},
                         **{name: centered(vector) for name, vector in outcomes.items()},
                         "q1": centered(q1)},
            "outcomes": outcomes,
            "magnitude": {"u": _contrast(u), "q1": _contrast(q1), "u_after_linear_distance_relative_norm": _u_remaining(u, distance)},
            "pairwise": {name: pairwise_concordance(vector, error, atol, rtol) for name, vector in signals.items()},
            "decision_direction": direction, "value_range": max(values) - min(values),
            "fixed_selector_event": {"bca_index": bca_index, "control_index": control_index,
                "different_indices": bca_index != control_index, "bca_score_ambiguous": len(bca_near) > 1,
                "control_score_ambiguous": len(control_near) > 1,
                "bca_minus_control_value": delta, "bca_minus_raw_q_value": values[bca_index] - raw_value,
                "control_minus_raw_q_value": values[control_index] - raw_value,
                "return_direction": "win" if event_direction > 0 else "loss" if event_direction < 0 else "near_equal"}}


def _defined_summary(entries):
    values = [entry["value"] for entry in entries if entry["defined"]]
    reasons = {}
    for entry in entries:
        if not entry["defined"]:
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    return {"state_count": len(entries), "defined_states": len(values), "undefined_states": len(entries) - len(values),
            "mean_over_defined_states": _mean(values), "undefined_reasons": reasons,
            "weighting": "One equal weight per defined state; undefined is not zero"}


def _directions_summary(directions):
    result = {"state_count": len(directions)}
    for signal in ("u_direction", "distance_direction"):
        result[signal] = {label: sum(row[signal] == label for row in directions)
                         for label in ("supports_better_candidate", "opposes_better_candidate", "tied")}
    result["joint"] = {label: sum(row["joint_direction"] == label for row in directions)
                       for label in ("both_support", "both_oppose", "conflicting", "one_or_both_tied")}
    result["u_supports_distance_opposes"] = sum(row["u_direction"] == "supports_better_candidate" and
                                               row["distance_direction"] == "opposes_better_candidate" for row in directions)
    result["distance_supports_u_opposes"] = sum(row["distance_direction"] == "supports_better_candidate" and
                                               row["u_direction"] == "opposes_better_candidate" for row in directions)
    return result


def summarize(states):
    relationships = {signal: {outcome: {statistic: _defined_summary([row["relationships"][signal][outcome][statistic] for row in states])
                                       for statistic in ("pearson", "spearman")}
                               for outcome in OUTCOMES} for signal in ("u", "actor_distance")}
    partial = {key: _defined_summary([row[key] for row in states])
               for key in ("partial_given_distance", "partial_given_distance_and_q1")}
    common = {}
    for key in partial:
        available = [row for row in states if row[key]["defined"] and row["relationships"]["u"]["signed_error"]["pearson"]["defined"]]
        marginal = [row["relationships"]["u"]["signed_error"]["pearson"]["value"] for row in available]
        conditional = [row[key]["value"] for row in available]
        common[key] = {"defined_states": len(available), "marginal_mean": _mean(marginal),
                       "partial_mean": _mean(conditional),
                       "mean_partial_minus_marginal": _mean([a-b for a,b in zip(conditional, marginal)])}
    pairwise = {}
    for signal in ("u", "actor_distance"):
        pairs = [row["pairwise"][signal] for row in states]
        rates = [row["correct_rate_among_comparable"] for row in pairs if row["correct_rate_defined"]]
        pairwise[signal] = {"defined_states": len(rates), "undefined_states": len(states)-len(rates),
                            "equal_state_mean_correct_rate_among_comparable": _mean(rates),
                            "equal_state_mean_tie_fraction": _mean([row["tie_fraction"] for row in pairs if row["tie_fraction"] is not None]),
                            "descriptive_pair_totals": {key: sum(row[key] for row in pairs) for key in
                                ("total_pairs", "comparable_pairs", "correct_pairs", "wrong_pairs", "tied_pairs", "signal_only_tied_pairs", "error_only_tied_pairs", "both_tied_pairs")},
                            "pair_counts_are_independent_sample_size": False}
    common_pairs = [row for row in states if row["pairwise"]["u"]["correct_rate_defined"]
                    and row["pairwise"]["actor_distance"]["correct_rate_defined"]]
    pairwise["common_defined_u_minus_distance"] = {
        "defined_states": len(common_pairs),
        "equal_state_mean_accuracy_difference": _mean([row["pairwise"]["u"]["correct_rate_among_comparable"]
            - row["pairwise"]["actor_distance"]["correct_rate_among_comparable"] for row in common_pairs]),
        "scope": "Paired state-level descriptive difference, no significance test"}
    direction = [row["decision_direction"] for row in states]
    eligible = [row for row in direction if row["regret_above_band"]]
    unique = [row for row in eligible if row["eligible_unambiguous_best"]]
    events = [row["fixed_selector_event"] for row in states]
    different = [row for row in events if row["different_indices"]]
    frequency = None if not states else len(different) / len(states)
    conditional_gain = _mean([row["bca_minus_control_value"] for row in different])
    unconditional = _mean([row["bca_minus_control_value"] for row in events])
    product = None if not states else 0. if not different else frequency * conditional_gain
    error = None if not states else abs(unconditional-product)
    identity_tolerance = None if not states else 16 * math.ulp(max(1., abs(unconditional), abs(product)))
    if error is not None and error > identity_tolerance:
        raise ValueError("Fixed-selector event decomposition identity failed")
    return {"state_count": len(states), "relationships": relationships, **partial,
            "common_defined_subsets": common, "pairwise": pairwise,
            "decision_direction": {"positive_regret_states": len(eligible), "zero_or_near_zero_regret_states": len(states)-len(eligible),
                "positive_regret_frequency": None if not states else len(eligible)/len(states),
                "ambiguous_best_states": sum(row["best_value_ambiguous"] for row in direction),
                "positive_regret_ambiguous_best_states": len(eligible)-len(unique),
                "raw_q_score_ambiguous_states": sum(row["raw_q_score_ambiguous"] for row in direction),
                "all_positive_regret_representatives": _directions_summary(eligible),
                "unambiguous_best_positive_regret": _directions_summary(unique),
                "ambiguous_best_positive_regret": _directions_summary([row for row in eligible if row["best_value_ambiguous"]])},
            "magnitude": {"u_centered_to_raw_rms": _defined_summary([{"defined": row["magnitude"]["u"]["ratio_defined"],
                            "value": row["magnitude"]["u"]["centered_to_raw_rms"], "reason": row["magnitude"]["u"]["reason"]} for row in states]),
                          "q1_centered_to_raw_rms": _defined_summary([{"defined": row["magnitude"]["q1"]["ratio_defined"],
                            "value": row["magnitude"]["q1"]["centered_to_raw_rms"], "reason": row["magnitude"]["q1"]["reason"]} for row in states]),
                          "u_after_linear_distance_relative_norm": _defined_summary([row["magnitude"]["u_after_linear_distance_relative_norm"] for row in states])},
            "candidate_bank_informativeness": {"mean_value_range": _mean([row["value_range"] for row in states]),
                "mean_raw_q_regret": _mean([row["exact_candidate_bank_regret"] for row in direction])},
            "fixed_selector_event": {"different_index_states": len(different), "disagreement_frequency": frequency,
                "conditional_mean_gain_when_different": conditional_gain, "unconditional_mean_gain": unconditional,
                "frequency_times_conditional_gain": product, "identity_absolute_error": error,
                "identity_roundoff_tolerance": identity_tolerance, "identity_verified": error is None or error <= identity_tolerance,
                "conditional_win_loss_near_equal": {label: sum(row["return_direction"] == label for row in different) for label in ("win", "loss", "near_equal")},
                "mean_bca_minus_raw_q": _mean([row["bca_minus_raw_q_value"] for row in events]),
                "mean_control_minus_raw_q": _mean([row["control_minus_raw_q_value"] for row in events]),
                "score_ambiguous_states": sum(row["bca_score_ambiguous"] or row["control_score_ambiguous"] for row in events)}}


def precision_planning(states, protocol):
    by_time = []
    for time in protocol["query_times"]:
        values = [row["fixed_selector_event"]["bca_minus_control_value"] for row in states if row["time"] == time]
        if len(values) < 2:
            raise ValueError("Precision planning requires at least two development episodes per time")
        variance = 0. if all(value == values[0] for value in values) else math.fsum((value - _mean(values)) ** 2 for value in values)/(len(values)-1)
        by_time.append({"time": time, "episodes": len(values), "sample_variance": variance})
    variance = _mean([row["sample_variance"] for row in by_time])
    estimates = []
    for target in protocol["precision_planning"]["effect_targets"]:
        if target <= 0:
            raise ValueError("Planning targets must be positive")
        estimable = variance > 0
        estimates.append({"illustrative_effect_or_half_width": target, "estimable": estimable,
                          "reason": None if estimable else "zero_observed_variance_not_estimable",
                          "normal_approximation_80pct_power_total_episodes": None if not estimable else 4*math.ceil(((1.96+.84)**2*variance/target**2)/4),
                          "normal_approximation_95pct_ci_half_width_total_episodes": None if not estimable else 4*math.ceil((1.96**2*variance/target**2)/4)})
    return {"within_time": by_time, "mean_within_time_sample_variance": variance, "estimates": estimates,
            "practical_effect_threshold_selected": False, "sample_size_recommendation": False,
            "additional_sampling_authorized": False, "power_guarantee": False,
            "scope": protocol["precision_planning"]["assumptions"]}


def analyze(batches, protocol):
    if (protocol.get("phase") != "exploratory_saved_development_only" or len(batches) != 2
            or protocol["residual_relative_l2_floor"] != 1e-12 or protocol["control_rank_relative_floor"] != 1e-12
            or protocol["tie_atol"] != 1e-6 or protocol["tie_rtol"] != 1e-6
            or protocol["kappa_penalty_bca"] != 3. or protocol["kappa_penalty_actor_distance"] != .1
            or protocol["actor_distance_factor"] != 19.29619572569095
            or protocol["precision_planning"]["effect_targets"] != [.05, .1, .2]):
        raise ValueError("The frozen development-only definitions changed")
    if [batch["name"] for batch in batches] != protocol["batch_names"]:
        raise ValueError("Development batch names/order changed")
    states = []
    for batch in batches:
        if len(batch["queries"]) != 64:
            raise ValueError("Each development batch must contain 64 states")
        if any(sum(query["time"] == time for query in batch["queries"]) != 16 for time in protocol["query_times"]):
            raise ValueError("Each batch must retain its 16-per-time design")
        states.extend(_state(query, batch["name"], protocol) for query in batch["queries"])
    if sorted(row["group_id"] for row in states) != list(range(128)) or len({row["episode_seed"] for row in states}) != 128:
        raise ValueError("Expected 128 distinct episode groups and seeds")
    return {"status": "pass", "phase": protocol["phase"], "claim_scope": "exploratory_saved_development_mechanism_only",
            "state_records": states, "overall": summarize(states),
            "by_batch": {name: summarize([row for row in states if row["batch"] == name]) for name in protocol["batch_names"]},
            "by_time": {str(time): summarize([row for row in states if row["time"] == time]) for time in protocol["query_times"]},
            "by_batch_time": {name: {str(time): summarize([row for row in states if row["batch"] == name and row["time"] == time])
                                      for time in protocol["query_times"]} for name in protocol["batch_names"]},
            "precision_planning": precision_planning(states, protocol),
            "new_model_fitted": False, "method_or_strength_selected": False, "bootstrap_performed": False,
            "full_training_hold": True, "automatic_training_approval": False,
            "confirmation_files_read": 0, "new_simulator_calls": 0, "new_gpu_work": 0,
            "limitations": ["Exploratory saved-development audit after known aggregate conclusions, not independent confirmation.",
                "Partial correlations remove only the specified linear control span; they are not causal or information-complete.",
                "Different undefined-state subsets can change averages; common-defined summaries are included.",
                "Candidates and pairs within a state are dependent. Pair totals are descriptive, not independent sample sizes.",
                "The raw-Q versus better-value pair's error ordering is partly imposed by its construction; positive DeltaU only reduces that pair's relative penalized score gap.",
                "Near-best representative choice can reverse penalty directions; ambiguous cases are separated.",
                "Endpoint-relative numerical tie bands are offset-sensitive and their absolute floor prevents exact unit invariance.",
                "Q1 and current frozen-policy continuation may have different target-policy estimands.",
                "Finite candidate banks and one trained checkpoint do not establish trained-policy or cross-seed efficacy.",
                "Planning estimates use observed development variance and normal approximations, not a practical target choice, power guarantee or permission to extend confirmation."]}
