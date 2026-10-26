"""Thinned calibration bank: few rows from each of many length-weighted episodes.

Rows of one episode are correlated, while WBCP's posterior treats calibration rows as
independent. A bank of whole episodes is therefore overconfident; on hopper-medium about
28% of such banks fail with no shift, against at most 5% nominal (experiments/wbcp/
DEPENDENCE.md). Taking K rows from each of ceil(n / K) episodes keeps the design effect
1 + (K - 1) rho near one. Every row of a reserved episode must still leave training.

Whether a configured bank stays within the 5% budget is measured, not assumed:
dependence_evidence() looks the bank up in calibration/dependence_evidence.json, which
experiments/wbcp/dependence_evidence.py writes from benchmark runs. A bank is keyed by
host, dataset, K, n, sampler and the score it calibrates; DEPLOYED_SCORE names the score
each host's BCA calibrates today, so evidence for another score of the same host and
dataset (a Q1 or max bank) never stands in for the deployed one.

NumPy only, so the D4RL benchmark runs the exact sampler BCA uses.
"""

import copy
import json
import os
from functools import lru_cache

import numpy as np

EVIDENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dependence_evidence.json")
EVIDENCE_SCHEMA = "bca-dependence-evidence-v1"
DEPENDENCE_BUDGET = 0.05  # 1 - credibility: the share of banks a beta = 0.95 threshold may fail without shift
CONSISTENT = "consistent with the 5% budget"
EXCEEDS = "exceeds the 5% budget"
NOT_VALIDATED = "not validated for this host/dataset/K"
# The score each host's BCA calibrates, as the registry labels it: the host's native target
# minus the min over its two critic heads, divided by the frozen scale. TD3+BC, ReBRAC and
# CQL take the min over the online heads (algorithms/{td3_bc,rebrac,cql}_bca.py); IQL over
# the Polyak target heads its actor advantage reads (calibration/iql_reference.refresh).
MIN_Q = "min(Q1,Q2) residual / sigma, normalized"
TARGET_MIN_Q = "min(target Q1, target Q2) residual / sigma, normalized"
DEPLOYED_SCORE = dict(td3_bc=MIN_Q, rebrac=MIN_Q, cql=MIN_Q, iql=TARGET_MIN_Q)
# Sampler revision that experiments/wbcp/d4rl_benchmark.py writes into each result as
# `bank_trim`: both its per-episode sampler and stratified_bank remove the surplus over n
# uniformly at random. A result without it was drawn before the trim.
REMAINDER_TRIM = "surplus over n removed uniformly at random (2026-10-01)"


def stratified_bank(lengths, target_size, rows_per_episode, rng):
    """Reserved episodes and the calibration offsets inside each.

    m = ceil(target / K) distinct episodes are chosen with inclusion probability exactly
    m L / N: the episodes are laid end to end in a random order and the episode under each
    of m equally spaced points is reserved (systematic sampling). One uniform row is then
    taken from each of K equal segments of every reserved episode (every row when L <= K).
    Segment edges fall on whole rows, so segments differ in size by one; rotating the
    segment grid by a uniform offset (a segment may wrap past the episode's end) makes each
    row of an episode with L > K drawn with probability exactly K m / N. The K rows are
    distinct and sit about L / K steps apart.

    When more rows are drawn than the target (K m > n whenever K does not divide n), one
    further draw removes the surplus uniformly at random across all drawn rows, so the bank
    holds exactly n rows. The reserved episodes stay as drawn: an episode left with fewer
    rows, or none, still leaves training. Rows of episodes with L > K are calibrated with
    probability n / N when every reserved episode has L > K, which holds in every draw when
    every episode is longer than K: each draw then yields K m rows and the trim keeps each
    with probability n / (K m). Short episodes are the exception. An episode of L <= K rows
    enters whole, its rows drawn with probability m L / N <= K m / N, and a draw that holds
    one can yield fewer than K m rows, so the trim removes fewer rows (none once the draw
    has n rows or fewer) and inclusion is no longer exactly n / N. Without a surplus the
    trim draws nothing, so the bank is the untrimmed one bit for bit.
    """
    lengths = np.asarray(lengths, np.int64)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths < 1):
        raise ValueError("episode lengths must be a nonempty vector of positive integers")
    k = rows_per_episode
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("rows_per_episode must be a positive integer")
    if isinstance(target_size, bool) or not isinstance(target_size, (int, np.integer)) or target_size < 1:
        raise ValueError("target_size must be a positive integer")
    total, draws = int(lengths.sum()), -(-int(target_size) // int(k))
    if draws * int(lengths.max()) > total:
        raise ValueError(
            f"{draws} length-proportional episodes are not possible here: the longest episode "
            f"({int(lengths.max())} rows) exceeds the spacing {total / draws:.1f}; raise "
            "rows_per_episode or lower the target"
        )
    order = rng.permutation(lengths.size)
    points = (rng.random() + np.arange(draws)) * (total / draws)
    hits = np.searchsorted(np.cumsum(lengths[order]), points, side="right")
    episodes = order[np.minimum(hits, lengths.size - 1)]  # guards a point rounded up to N
    offsets = []
    for length in lengths[episodes]:
        if length <= k:
            offsets.append(np.arange(length))
        else:
            edges = np.arange(k + 1) * length // k
            picks = edges[:-1] + (rng.random(k) * np.diff(edges)).astype(np.int64)
            offsets.append(np.sort((picks + rng.integers(length)) % length))
    sizes = np.array([rows.size for rows in offsets])
    surplus = int(sizes.sum()) - int(target_size)
    if surplus > 0:  # one uniform draw over all drawn rows; none at all without a surplus
        keep = np.ones(int(sizes.sum()), bool)
        keep[rng.choice(keep.size, surplus, replace=False)] = False
        offsets = [rows[kept] for rows, kept in zip(offsets, np.split(keep, np.cumsum(sizes)[:-1]))]
    return episodes, offsets


def evidence_status(ci):
    """The budget verdict for one measured design, from the 95% interval of its failure rate:
    consistent when the lower bound is at most the 5% budget, exceeding it otherwise."""
    low, high = (float(value) for value in ci)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("a failure-rate interval needs 0 <= low <= high <= 1")
    return CONSISTENT if low <= DEPENDENCE_BUDGET else EXCEEDS


@lru_cache(maxsize=None)
def _registry(path):
    with open(path, encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("schema") != EVIDENCE_SCHEMA or not isinstance(registry.get("entries"), list):
        raise ValueError(f"{path} is not a {EVIDENCE_SCHEMA} registry")
    return registry


def load_evidence(path=EVIDENCE_PATH):
    """The registry's entries, as copies (the parsed file is cached per path)."""
    return copy.deepcopy(_registry(os.path.abspath(path))["entries"])


def dependence_evidence(host, dataset, rows_per_episode, size, *, score, sampler="reservation", entries=None):
    """Measured within-episode dependence evidence for one configured calibration bank.

    Finds the benchmark run on this host's frozen pool of this dataset (D4RL name) with K rows
    per episode and bank size n, drawn by `sampler` (BCA's own, stratified_bank, by default)
    and calibrating `score` (for a deployed bank, DEPLOYED_SCORE[host]; required, so a run of
    another score is never matched by omission). Returns the entry's fields plus `status`
    (evidence_status of its interval), or the query with status NOT_VALIDATED when no run
    matches exactly: evidence does not carry over to another host, dataset, K, n, sampler or
    score. `entries` replaces the registry file. Two entries for one design are an error (the
    generator refuses to write them).
    """
    entries = load_evidence() if entries is None else entries
    query = dict(host=host, dataset=dataset, rows_per_episode=int(rows_per_episode), size=int(size), sampler=sampler,
                 score=score)
    found = [entry for entry in entries if all(entry.get(name) == value for name, value in query.items())]
    if len(found) > 1:
        raise ValueError(f"the dependence evidence registry holds {len(found)} entries for one design {query}")
    if not found:
        return dict(status=NOT_VALIDATED, **query)
    return dict(status=evidence_status(found[0]["ci95"]), **copy.deepcopy(found[0]))
