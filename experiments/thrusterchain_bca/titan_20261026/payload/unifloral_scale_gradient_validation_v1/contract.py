"""Standard-library diagnostics only; no approved production normalizer."""
import math


def _rows(rows):
    values = []
    for row in rows:
        row = list(row)
        if not row:
            raise ValueError("Each fitting state needs at least one action")
        if any(isinstance(x, bool) for x in row):
            raise ValueError("Boolean values are not numeric action scores")
        row = [float(x) for x in row]
        if not all(math.isfinite(x) for x in row):
            raise ValueError("Nonfinite scores invalidate the diagnostic")
        values.append(row)
    if not values:
        raise ValueError("Empty fitting bank")
    return values


def _mean(row):
    maximum = max(abs(x) for x in row)
    return 0.0 if maximum == 0 else maximum * (math.fsum(x / maximum for x in row) / len(row))


def _rms(rows):
    maximum = max(abs(x) for row in rows for x in row)
    if maximum == 0:
        return 0.0
    mean_square = math.fsum(math.fsum((x / maximum) ** 2 for x in row) / len(row)
                            for row in rows) / len(rows)
    result = maximum * math.sqrt(mean_square)
    if not math.isfinite(result):
        raise ValueError("RMS overflow invalidates the diagnostic")
    return result


def contrast_statistics(rows):
    """Equal weight per state, then equal weight among its feasible candidates."""
    rows = _rows(rows)
    centered = [[x - _mean(row) for x in row] for row in rows]
    if not all(math.isfinite(x) for row in centered for x in row):
        raise ValueError("Centering overflow invalidates the diagnostic")
    raw, contrasts = _rms(rows), _rms(centered)
    relative = None if raw == 0 else contrasts / raw
    return {"states": len(rows), "candidate_counts": [len(row) for row in rows],
            "raw_rms": raw, "centered_rms": contrasts,
            "relative_contrast": relative,
            "centered_fraction_squared": None if relative is None else relative ** 2}


def candidate_scale(q_rows, u_rows, relative_floor=1e-6):
    """Hypothetical detached C_Q/C_U. Unavailable means no replacement factor."""
    if not math.isfinite(relative_floor) or not 0 < relative_floor < 1:
        raise ValueError("Relative contrast floor must be finite and between zero and one")
    q, u = contrast_statistics(q_rows), contrast_statistics(u_rows)
    if q["candidate_counts"] != u["candidate_counts"]:
        raise ValueError("Q and U must refer to identical candidate banks")
    result = {"status": "unavailable", "reason": None, "factor": None, "q": q, "u": u,
              "relative_floor": relative_floor,
              "role": "Candidate diagnostic only; any JAX use must stop gradients through the factor"}
    for label, item in (("q", q), ("u", u)):
        if item["raw_rms"] == 0 or item["centered_rms"] == 0:
            result["reason"] = label + " has zero raw or centered scale"
            return result
        if item["relative_contrast"] <= relative_floor:
            result["reason"] = label + " has near-constant action scores under the declared relative rule"
            return result
    factor = q["centered_rms"] / u["centered_rms"]
    if not math.isfinite(factor) or factor <= 0:
        result["reason"] = "Candidate scale factor is not finite and positive"
        return result
    result.update(status="available", factor=factor)
    return result


def native_epsilon_response(mean_abs_q, reward_scale, alpha=2.5, epsilon=1e-7):
    """Analytic native coefficient behavior; fixed epsilon remains unchanged."""
    m, s, alpha, epsilon = map(float, (mean_abs_q, reward_scale, alpha, epsilon))
    if not all(math.isfinite(x) for x in (m, s, alpha, epsilon)) or m < 0 or min(s, alpha, epsilon) <= 0:
        raise ValueError("Invalid native coefficient inputs")
    denominator = s * m + epsilon
    if not math.isfinite(denominator):
        raise ValueError("Scaled coefficient denominator overflow")
    native = alpha / (m + epsilon)
    scaled = alpha / denominator
    return {"q_scale_coefficient": native, "scaled_q_scale_coefficient": scaled,
            "value_gradient_multiplier": s * (m + epsilon) / denominator,
            "co_scaled_epsilon_gradient_multiplier": 1.0,
            "zero_q_native_coefficient": alpha / epsilon,
            "scope": "Local frozen-parameter unit algebra; not optimizer/training invariance"}
