"""Pure saved-array summaries; no learner, dataset loader or simulator imports.

The caller must bind Qpi to the same checkpoint's unmodified deployed continuation,
with identical state/time/complete-action rows. The first three columns are the
common action bank; the last is that model's actor action. Only common columns
enter action-bank ranking comparisons. These summaries make no significance claim.
"""
import math
import statistics


ERROR_TOLERANCE = 0.15
TIE_TOLERANCE = 1e-10
KAPPA = 3.0


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric, not boolean")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be finite numeric data") from exc
    if not math.isfinite(out):
        raise ValueError(name + " must be finite")
    return out


def _vector(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + " must be a nonempty sequence")
    return [_finite(x, name) for x in values]


def _matrix(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + " must be a nonempty matrix")
    rows = [_vector(row, name) for row in values]
    if len({len(row) for row in rows}) != 1:
        raise ValueError(name + " is ragged")
    return rows


def _ids(values, count, name):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(name + " count differs")
    if any(isinstance(x, bool) or not isinstance(x, (str, int)) for x in values):
        raise ValueError(name + " requires string or integer identifiers")
    # Keep types distinct: integer 1 and string "1" are different identifiers.
    tagged = [(type(x).__name__, x) for x in values]
    if len(set(tagged)) != count:
        raise ValueError(name + " contains duplicate identifiers")
    return tagged


def paired_returns(native, bca, native_episode_ids, bca_episode_ids, *, expected_count=128):
    """Pair by explicit episode identity; never silently pair rows by position alone."""
    native = _vector(native, "native returns")
    bca = _vector(bca, "BCA returns")
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    if len(native) != expected_count or len(bca) != expected_count:
        raise ValueError("Incomplete or unexpected return denominator")
    left = _ids(native_episode_ids, expected_count, "native episode ids")
    right = _ids(bca_episode_ids, expected_count, "BCA episode ids")
    if left != right:
        raise ValueError("Episode identities/order do not match")
    delta = [b - n for n, b in zip(native, bca)]
    if not all(math.isfinite(x) for x in delta):
        raise ValueError("Return difference overflow")
    return {"episodes": expected_count, "native_mean": statistics.mean(native),
            "bca_mean": statistics.mean(bca), "mean_difference": statistics.mean(delta),
            "median_difference": statistics.median(delta), "differences": delta,
            "positive_episodes": sum(x > 0 for x in delta),
            "negative_episodes": sum(x < 0 for x in delta),
            "zero_episodes": sum(x == 0 for x in delta)}


def _errors(errors, tolerance):
    over = [x for x in errors if x > tolerance]
    under = [x for x in errors if x < -tolerance]
    return {"count": len(errors), "mean_signed_error": statistics.mean(errors),
            "mae": statistics.mean(abs(x) for x in errors),
            "over_count": len(over), "under_count": len(under),
            "near_zero_count": len(errors) - len(over) - len(under),
            "over_mean_magnitude": statistics.mean(over) if over else None,
            "under_mean_magnitude": statistics.mean(-x for x in under) if under else None,
            "signed_errors": errors}


def _adjustments(raw, adjusted, tolerance, tie_tolerance):
    improvement = [abs(a) - abs(b) for a, b in zip(raw, adjusted)]
    return {"count": len(raw),
            "absolute_error_improved": sum(x > tie_tolerance for x in improvement),
            "absolute_error_worsened": sum(x < -tie_tolerance for x in improvement),
            "absolute_error_tied": sum(abs(x) <= tie_tolerance for x in improvement),
            "mean_absolute_error_reduction": statistics.mean(improvement),
            "over_to_under": sum(a > tolerance and b < -tolerance for a, b in zip(raw, adjusted)),
            "under_to_over": sum(a < -tolerance and b > tolerance for a, b in zip(raw, adjusted)),
            "positive_to_negative": sum(a > 0 and b < 0 for a, b in zip(raw, adjusted)),
            "negative_to_positive": sum(a < 0 and b > 0 for a, b in zip(raw, adjusted))}


def _pick(row, tie_tolerance):
    best = max(row)
    # Lowest declared action index among scores within the fixed absolute tolerance.
    return next(i for i, x in enumerate(row) if best - x <= tie_tolerance)


def score_summary(q1, qpi, width, *, kappa=KAPPA, error_tolerance=ERROR_TOLERANCE,
                  tie_tolerance=TIE_TOLERANCE, common_action_count=3):
    """Compare raw Q1 and Q1-kappa*U against one fixed, same-policy Qpi panel.

    Input matrices have common_action_count common actions plus one actor-action
    column. Regret is within the fixed common bank, never global optimal-policy
    regret. The actor-action column is excluded from the ranking denominator.
    """
    q1, qpi, width = (_matrix(v, n) for v, n in ((q1, "Q1"), (qpi, "Qpi"), (width, "width")))
    shape = (len(q1), len(q1[0]))
    if any((len(v), len(v[0])) != shape for v in (qpi, width)):
        raise ValueError("Q1/Qpi/width matrix shapes differ")
    if type(common_action_count) is not int or common_action_count <= 0 or shape[1] != common_action_count + 1:
        raise ValueError("Expected common actions plus exactly one actor-action column")
    kappa, error_tolerance, tie_tolerance = (
        _finite(x, name) for x, name in ((kappa, "kappa"), (error_tolerance, "error tolerance"),
                                       (tie_tolerance, "tie tolerance")))
    if min(kappa, error_tolerance, tie_tolerance) < 0 or any(x < 0 for row in width for x in row):
        raise ValueError("Penalty, tolerances and widths must be nonnegative")
    adjusted = [[a - kappa * u for a, u in zip(row, ur)] for row, ur in zip(q1, width)]
    raw_error = [[a - y for a, y in zip(row, truth)] for row, truth in zip(q1, qpi)]
    adjusted_error = [[a - y for a, y in zip(row, truth)] for row, truth in zip(adjusted, qpi)]
    if not all(math.isfinite(x) for matrix in (adjusted, raw_error, adjusted_error) for row in matrix for x in row):
        raise ValueError("Score or error arithmetic overflow")
    result = {"anchors": shape[0], "common_actions": common_action_count,
              "kappa": kappa, "error_tolerance": error_tolerance, "tie_tolerance": tie_tolerance,
              "reference": "same-checkpoint unmodified own-policy Qpi; caller authenticates alignment"}
    for name, indices in (("all", range(shape[1])), ("common", range(common_action_count)),
                          ("actor", (common_action_count,))):
        a = [row[i] for row in raw_error for i in indices]
        b = [row[i] for row in adjusted_error for i in indices]
        result[name] = {"raw": _errors(a, error_tolerance), "adjusted": _errors(b, error_tolerance),
                        "adjustment": _adjustments(a, b, error_tolerance, tie_tolerance)}
    raw_indices, adjusted_indices, raw_regret, adjusted_regret, gains = [], [], [], [], []
    raw_correct, adjusted_correct = 0, 0
    for raw, adj, truth in zip(q1, adjusted, qpi):
        raw, adj, truth = raw[:common_action_count], adj[:common_action_count], truth[:common_action_count]
        i, j = _pick(raw, tie_tolerance), _pick(adj, tie_tolerance)
        optimum = max(truth)
        raw_indices.append(i)
        adjusted_indices.append(j)
        raw_regret.append(optimum - truth[i])
        adjusted_regret.append(optimum - truth[j])
        gains.append(truth[j] - truth[i])
        raw_correct += optimum - truth[i] <= tie_tolerance
        adjusted_correct += optimum - truth[j] <= tie_tolerance
    if not all(math.isfinite(x) for array in (raw_regret, adjusted_regret, gains) for x in array):
        raise ValueError("Ranking arithmetic overflow")
    result["common_bank"] = {
        "anchors": shape[0], "raw_indices": raw_indices, "adjusted_indices": adjusted_indices,
        "raw_regret": raw_regret, "adjusted_regret": adjusted_regret,
        "raw_mean_regret": statistics.mean(raw_regret), "adjusted_mean_regret": statistics.mean(adjusted_regret),
        "raw_optimal_count": raw_correct, "adjusted_optimal_count": adjusted_correct,
        "selected_value_gains": gains, "mean_selected_value_gain": statistics.mean(gains),
        "switches": sum(i != j for i, j in zip(raw_indices, adjusted_indices)),
        "beneficial_switches": sum(x > tie_tolerance for x in gains),
        "harmful_switches": sum(x < -tie_tolerance for x in gains),
        "value_tied_switches": sum(i != j and abs(g) <= tie_tolerance
                                   for i, j, g in zip(raw_indices, adjusted_indices, gains))}
    return result


def across_seeds(per_seed, *, expected_seeds):
    """Equal-weight seed effects, preserving all three independent training units."""
    expected = list(expected_seeds)
    _ids(expected, len(expected), "expected seeds")
    if len(expected) != 3 or set(per_seed) != set(expected):
        raise ValueError("Exactly the three prespecified training seeds are required")
    effects = [_finite(per_seed[s]["mean_difference"], "seed effect") for s in expected]
    return {"independent_training_seeds": 3, "seed_order": expected, "per_seed_mean_differences": effects,
            "mean_seed_difference": statistics.mean(effects), "median_seed_difference": statistics.median(effects),
            "minimum_seed_difference": min(effects), "maximum_seed_difference": max(effects),
            "positive_seeds": sum(x > 0 for x in effects), "negative_seeds": sum(x < 0 for x in effects),
            "note": "Exploratory per-seed effects; no significance or confirmation decision."}
