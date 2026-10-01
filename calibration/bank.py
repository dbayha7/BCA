"""Thinned calibration bank: few rows from each of many length-weighted episodes.

Rows of one episode are correlated, while WBCP's posterior treats calibration rows as
independent. A bank of whole episodes is therefore overconfident; on hopper-medium about
28% of such banks fail with no shift, against at most 5% nominal (experiments/wbcp/
DEPENDENCE.md). Taking K rows from each of ceil(n / K) episodes keeps the design effect
1 + (K - 1) rho near one. Every row of a reserved episode must still leave training.

NumPy only, so the D4RL benchmark runs the exact sampler BCA uses.
"""

import numpy as np

# The range the hopper-medium benchmark validated without shift (DEPENDENCE.md, Changes
# 4 and 6): at most 10 rows from each of at least 100 episodes. Under the density tilt
# only K = 5 held (Changes 7 and 8).
VALIDATED_ROWS_PER_EPISODE = 10
VALIDATED_MIN_EPISODES = 100


def stratified_bank(lengths, target_size, rows_per_episode, rng):
    """Reserved episodes and the calibration offsets inside each.

    m = ceil(target / K) distinct episodes are chosen with inclusion probability exactly
    m L / N: the episodes are laid end to end in a random order and the episode under each
    of m equally spaced points is reserved (systematic sampling). One uniform row is then
    taken from each of K equal segments of every reserved episode (every row when L <= K).
    Segment edges fall on whole rows, so segments differ in size by one; rotating the
    segment grid by a uniform offset (a segment may wrap past the episode's end) makes each
    row of an episode with L > K calibrated with probability exactly K m / N. The K rows
    are distinct and sit about L / K steps apart.
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
    return episodes, offsets


def dependence_validated(rows_per_episode, episodes):
    return bool(rows_per_episode <= VALIDATED_ROWS_PER_EPISODE and episodes >= VALIDATED_MIN_EPISODES)
