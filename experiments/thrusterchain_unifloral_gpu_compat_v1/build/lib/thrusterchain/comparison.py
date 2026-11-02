"""Validate declared evaluation partitions and aggregate declared fitted runs.

Seed separation checks declarations, not whether a human previously inspected
the results. This module does not choose policies, run an environment, or train.
"""
from collections import defaultdict
import math
import numbers
import re

import numpy as np

_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _seed(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _digest(value, name):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_partitions(dataset_seeds, selection_seeds, final_seeds):
    """Reject duplicates within roles and shared episode seeds between roles."""
    groups = {}
    for role, seeds in (("dataset", dataset_seeds), ("selection", selection_seeds),
                        ("final", final_seeds)):
        values = [_seed(x, role + " seed") for x in seeds]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {role} seed")
        groups[role] = set(values)
    if not groups["dataset"] or not groups["final"]:
        raise ValueError("dataset and final seed sets must be nonempty")
    for left, right in (("dataset", "selection"), ("dataset", "final"), ("selection", "final")):
        if groups[left] & groups[right]:
            raise ValueError(f"{left} and {right} episode seeds overlap")
    return {name: sorted(values) for name, values in groups.items()}


def summarize_final(records, partitions, *, expected, selection_mode, selection_interactions):
    """Summarize final discounted returns at the declared training-seed level.

    Each record identifies algorithm, training seed, dataset and frozen policy,
    scenario family, evaluation episode seed, and discounted return. A family
    may share evaluation seeds with other families or algorithms. Returns are
    averaged within each trained policy before calculating across-run statistics.
    This deliberately reports no confidence interval for a single run and makes
    no claim that an evaluation split was unobserved before selection.
    """
    if not isinstance(partitions, dict) or set(partitions) != {"dataset", "selection", "final"}:
        raise ValueError("partitions must specify dataset, selection, and final")
    parts = validate_partitions(partitions["dataset"], partitions["selection"], partitions["final"])
    if selection_mode not in ("strict_offline", "simulator_assisted"):
        raise ValueError("unknown selection mode")
    interactions = _seed(selection_interactions, "selection interactions")
    if selection_mode == "strict_offline" and (interactions or parts["selection"]):
        raise ValueError("strict offline selection cannot use simulator selection episodes")
    if selection_mode == "simulator_assisted" and (not interactions or not parts["selection"]):
        raise ValueError("simulator-assisted selection requires declared seeds and interactions")
    # Compare to the declared matrix. This does not prove it was frozen before
    # inspecting outcomes; evidence of precommitment must be tracked separately.
    planned, planned_runs = set(), defaultdict(set)
    for plan in expected:
        if not isinstance(plan, dict):
            raise ValueError("each expected run must be a mapping")
        algorithm, family = plan.get("algorithm"), plan.get("family")
        if not isinstance(algorithm, str) or not algorithm.strip() or not isinstance(family, str) or not family.strip():
            raise ValueError("expected algorithm and family must be nonempty strings")
        train_seed = _seed(plan.get("training_seed"), "expected training seed")
        episode_seeds = [_seed(seed, "expected episode seed") for seed in plan.get("episode_seeds", [])]
        if not episode_seeds or len(episode_seeds) != len(set(episode_seeds)):
            raise ValueError("expected episode seeds must be nonempty and unique")
        if train_seed in planned_runs[(algorithm, family)]:
            raise ValueError("duplicate expected training run")
        planned_runs[(algorithm, family)].add(train_seed)
        for episode_seed in episode_seeds:
            if episode_seed not in parts["final"]:
                raise ValueError("expected episode is outside the final partition")
            planned.add((algorithm, train_seed, family, episode_seed))
    if not planned:
        raise ValueError("a nonempty expected evaluation matrix is required")
    buckets, identity, seen = defaultdict(list), {}, set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each record must be a mapping")
        algorithm, family = row.get("algorithm"), row.get("family")
        if not isinstance(algorithm, str) or not algorithm.strip() or not isinstance(family, str) or not family.strip():
            raise ValueError("algorithm and family must be nonempty strings")
        train_seed = _seed(row.get("training_seed"), "training seed")
        episode_seed = _seed(row.get("episode_seed"), "episode seed")
        if episode_seed not in parts["final"]:
            raise ValueError("evaluation record is not in the declared final partition")
        dataset = _digest(row.get("dataset_sha256"), "dataset_sha256")
        policy = _digest(row.get("policy_sha256"), "policy_sha256")
        run = (algorithm, train_seed)
        if run in identity and identity[run] != (dataset, policy):
            raise ValueError("one training run must have one dataset and frozen policy identity")
        identity[run] = (dataset, policy)
        key = (algorithm, train_seed, family, episode_seed)
        if key not in planned:
            raise ValueError("evaluation record was not in the declared expected matrix")
        if key in seen:
            raise ValueError("duplicate policy/family/episode record")
        seen.add(key)
        value = row.get("discounted_return")
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError("discounted return must be finite and numeric")
        buckets[(algorithm, family, train_seed)].append(float(value))
    grouped = defaultdict(list)
    for (algorithm, family, train_seed), values in sorted(buckets.items()):
        grouped[(algorithm, family)].append({"training_seed": train_seed,
            "episodes": len(values), "mean_discounted_return": float(np.mean(values))})
    summaries = []
    missing = planned - seen
    for algorithm, family in sorted(planned_runs):
        runs = grouped[(algorithm, family)]
        group_missing = sorted(key for key in missing if key[0] == algorithm and key[2] == family)
        means = np.asarray([run["mean_discounted_return"] for run in runs])
        summaries.append({"algorithm": algorithm, "family": family,
            "status": "incomplete" if group_missing else "complete",
            "expected_training_runs": len(planned_runs[(algorithm, family)]),
            "observed_training_runs": len(runs), "per_training_run": runs,
            "missing_episodes": [{"training_seed": k[1], "episode_seed": k[3]} for k in group_missing],
            "mean_of_run_means": None if group_missing else float(np.mean(means)),
            "sample_sd_across_training_runs": float(np.std(means, ddof=1)) if not group_missing and len(means) > 1 else None})
    return {"schema": "thrusterchain.comparison.v1", "partitions": parts,
        "selection_mode": selection_mode, "selection_interactions": interactions,
        "final_episodes": len(seen), "expected_final_episodes": len(planned),
        "complete": not missing, "summaries": summaries,
        "holdout_exposure": "not established by seed separation",
        "matrix_precommitment": "not established by this calculation",
        "uncertainty_unit": "declared training run; independence unverified; no interval estimated"}
