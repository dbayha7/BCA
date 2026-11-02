"""Recompute control summaries and validate keyed raw CSV exports.

Consistency checks detect contradictory artifacts; jointly fabricated records
are outside this check's scope. They are not authentication or a security gate.
"""
import csv
from pathlib import Path

import numpy as np

FIELDS = ("method", "family", "evaluation_seed", "return_undiscounted",
          "return_discounted", "stall_fraction")
RTOL, ATOL = 1e-12, 1e-10


def _equal(actual, expected, label):
    if expected is None:
        if actual is not None:
            raise AssertionError("control summary mismatch: " + label)
        return
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isfinite(actual):
        raise AssertionError("nonfinite or invalid control value: " + label)
    if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL):
        raise AssertionError("control summary mismatch: " + label)


def control_rows(results):
    """Derive rows from raw arrays, checking every reported cell summary/count."""
    expected_n = results["evaluation_episodes_per_family"]
    if isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n < 1:
        raise AssertionError("invalid expected episode count")
    rows = []
    for method, record in results["baselines"].items():
        for family, cell in record["families"].items():
            prefix = method + "/" + family
            seeds = cell["episode_seeds"]
            if (len(seeds) != expected_n or len(set(seeds)) != expected_n or
                any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds)):
                raise AssertionError("invalid/duplicate episode seed keys: " + prefix)
            if isinstance(cell["episodes"], bool) or not isinstance(cell["episodes"], int) or cell["episodes"] != expected_n:
                raise AssertionError("control episode count mismatch: " + prefix)
            values = {}
            for name in ("returns", "discounted_returns", "stall_fractions"):
                if any(isinstance(value, (bool, np.bool_)) for value in cell[name]):
                    raise AssertionError("boolean raw control value: " + prefix + "/" + name)
                raw = np.asarray(cell[name])
                if raw.dtype.kind not in "fiu" or raw.shape != (expected_n,) or not np.isfinite(raw).all():
                    raise AssertionError("invalid raw control array: " + prefix + "/" + name)
                values[name] = raw.astype(float)
            if np.any((values["stall_fractions"] < 0) | (values["stall_fractions"] > 1)):
                raise AssertionError("invalid stall fraction: " + prefix)
            _equal(cell["mean_return"], float(np.mean(values["returns"])), prefix + "/mean_return")
            _equal(cell["mean_discounted_return"], float(np.mean(values["discounted_returns"])), prefix + "/mean_discounted_return")
            _equal(cell["sample_sd_return"], float(np.std(values["returns"], ddof=1)) if expected_n > 1 else None,
                   prefix + "/sample_sd_return")
            for i, seed in enumerate(seeds):
                rows.append(dict(method=method, family=family, evaluation_seed=seed,
                    return_undiscounted=float(values["returns"][i]),
                    return_discounted=float(values["discounted_returns"][i]),
                    stall_fraction=float(values["stall_fractions"][i])))
        for field, count in (("environment_episodes", expected_n * len(record["families"])),
                             ("environment_steps", 100 * expected_n * len(record["families"]))):
            if method == "behavior_cloning" and field not in record:
                raise AssertionError("missing control total count: " + field)
            if field in record and (isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] != count):
                raise AssertionError("control total count mismatch: " + method + "/" + field)
    return rows


def write_control_csv(results, path):
    rows = control_rows(results)
    with Path(path).open("x", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def validate_control_records(results, path):
    expected = {(r["method"], r["family"], r["evaluation_seed"]): r for r in control_rows(results)}
    observed = {}
    try:
        with Path(path).open(encoding="utf8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise AssertionError("control CSV schema mismatch")
            for row in reader:
                try:
                    if set(row) != set(FIELDS) or any(value is None for value in row.values()):
                        raise ValueError("wrong number of CSV fields")
                    seed = int(row["evaluation_seed"])
                    if str(seed) != row["evaluation_seed"]:
                        raise ValueError("noncanonical integer")
                    key = (row["method"], row["family"], seed)
                    if key in observed:
                        raise AssertionError("duplicate control CSV key")
                    observed[key] = {field: float(row[field]) for field in FIELDS[3:]}
                except (ValueError, TypeError, KeyError) as exc:
                    raise AssertionError("invalid control CSV row") from exc
    except csv.Error as exc:
        raise ValueError("invalid control CSV syntax") from exc
    if set(observed) != set(expected):
        raise AssertionError("control CSV row keys/count mismatch")
    for key in expected:
        for field in FIELDS[3:]:
            _equal(observed[key][field], expected[key][field], "CSV/" + str(key) + "/" + field)
    return dict(rows=len(expected), summary_cells=sum(len(x["families"]) for x in results["baselines"].values()),
                join_keys=list(FIELDS[:3]), absolute_tolerance=ATOL, relative_tolerance=RTOL,
                scope="raw arrays, summaries, counts and CSV consistency; not proof of authentic collection")
