"""Standard-library rules for exhaustive fitting blocks and descriptive summaries."""
import math
import statistics


def block_plan(total_rows, block_size, starts):
    if type(total_rows) is not int or type(block_size) is not int or min(total_rows, block_size) <= 0:
        raise ValueError("Row and block counts must be positive integers")
    if total_rows % block_size:
        raise ValueError("The fixed plan requires complete equal-size blocks")
    starts = list(starts)
    expected = list(range(0, total_rows, block_size))
    if any(type(x) is not int for x in starts) or starts != expected:
        raise ValueError("Blocks must cover each fitting row once in existing order")
    return [(start, start + block_size) for start in starts]


def normalized_weights(counts):
    counts = list(counts)
    if not counts or any(type(x) is not int or x <= 0 for x in counts):
        raise ValueError("Counts must be nonempty positive integers")
    total = sum(counts)
    return [count / total for count in counts]


def weighted_mean(values, counts):
    values, counts = list(values), list(counts)
    weights = normalized_weights(counts)
    if len(values) != len(weights) or any(isinstance(x, bool) or not math.isfinite(x) for x in values):
        raise ValueError("Finite values must match block counts")
    result = math.fsum(float(x) * weight for x, weight in zip(values, weights))
    if not math.isfinite(result):
        raise ValueError("Weighted summary overflowed")
    return result


def ratio_summary(values):
    values = list(values)
    defined = [x for x in values if x is not None]
    if any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in defined):
        raise ValueError("Ratios must be finite nonnegative values or None")
    return {"count": len(values), "defined": len(defined), "unavailable": len(values) - len(defined),
            "min": min(defined) if defined else None, "max": max(defined) if defined else None,
            "median": statistics.median(defined) if defined else None,
            "mean": math.fsum(defined) / len(defined) if defined else None,
            "interpretation": "Descriptive fitting-block summary, not an uncertainty interval or global gradient ratio"}
