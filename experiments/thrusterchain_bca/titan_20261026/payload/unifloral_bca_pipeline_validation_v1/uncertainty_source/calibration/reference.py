"""Thinned episode reservation and the frozen scale/threshold reference every host consumes."""

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from calibration import bank, wbcp


@dataclass(frozen=True)
class WBCPConfig:
    alpha: float = 0.1  # target miscoverage; also the scale fit's coverage target 1-alpha
    credibility: float = 0.95  # posterior credibility beta of the selected threshold, Eq. (7)
    draws: int = 1000  # posterior draws M, Algorithm 1

    def __post_init__(self):
        for name in ("alpha", "credibility"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 < value < 1.0:
                raise ValueError(name + " must be strictly between zero and one")
        if isinstance(self.draws, bool) or not isinstance(self.draws, Integral) or self.draws < 1:
            raise ValueError("draws must be a positive integer")


class FrozenReference(NamedTuple):
    cal_params: Any  # scale-network parameters frozen at the refresh
    residual_scale: jax.Array  # residual unit frozen with them
    threshold: jax.Array  # lambda_dep = max(lambda_hat, lambda_hpd); +inf when nothing is certifiable
    lambda_hat: jax.Array  # weighted empirical selection, Eq. (1)
    lambda_hpd: jax.Array  # beta-credible posterior selection, Eq. (7)
    n_eff: jax.Array  # Kish effective sample size of the calibration weights
    ready: jax.Array


def qlearning_episode_ids(raw):
    if "timeouts" not in raw:
        return None
    ends = np.asarray(raw["terminals"], bool) | np.asarray(raw["timeouts"], bool)
    ids = np.concatenate([[0], np.cumsum(ends[:-1], dtype=np.int64)])
    return ids[:-1][~np.asarray(raw["timeouts"][:-1], bool)]


def reserve_calibration(
    obs, next_obs, done, target_size, seed, rows_per_episode, max_fraction=0.25,
    episode_ids=None,
):
    """Withhold length-weighted episodes from training and calibrate on K rows of each.

    Returns (training, withheld, calibration, metadata) row indices: training and withheld
    partition the data, calibration is the thinned subset of withheld (calibration/bank.py),
    target_size rows unless episodes shorter than K leave fewer. Its dependence evidence is
    looked up per host and dataset by runtime/config.resolve (bank.dependence_evidence).
    rows_per_episode=None withholds whole episodes in seeded-permutation order until the
    target is covered and returns all their rows as calibration. That is only for splitting
    a dataset into populations (experiments/wbcp/freeze_scores.py): as a calibration bank,
    whole episodes fail about 28% of the time on hopper-medium (DEPENDENCE.md).
    """
    obs, next_obs, done = (np.asarray(obs), np.asarray(next_obs), np.asarray(done))
    n = len(obs)
    if obs.ndim != 2 or next_obs.shape != obs.shape or done.shape != (n,):
        raise ValueError("expected aligned obs/next_obs matrices and done vector")
    if not 0 < target_size < n or not 0 < max_fraction < 1:
        raise ValueError("calibration target must leave a nonempty training pool")
    rule = "terminal or exact next_obs/obs discontinuity"
    if episode_ids is not None:
        ids = np.asarray(episode_ids)
        if ids.shape != (n,) or np.any(ids[1:] < ids[:-1]):
            raise ValueError("episode IDs must be aligned and nondecreasing")
        ends = np.concatenate([ids[1:] != ids[:-1], [True]])
        rule = "raw terminal/timeout IDs mapped through default D4RL filtering"
    else:
        ends = (done != 0).copy()
        ends[:-1] |= np.any(next_obs[:-1] != obs[1:], axis=1)
    ends[-1] = True
    boundaries = np.concatenate([[0], np.flatnonzero(ends) + 1])
    lengths = np.diff(boundaries)
    if rows_per_episode is None:
        order = np.random.default_rng(seed).permutation(len(lengths))
        chosen = order[: int(np.searchsorted(np.cumsum(lengths[order]), target_size)) + 1]
        offsets = [np.arange(lengths[i]) for i in chosen]
    else:
        chosen, offsets = bank.stratified_bank(
            lengths, target_size, rows_per_episode, np.random.default_rng(seed)
        )
    withheld = np.sort(
        np.concatenate([np.arange(boundaries[i], boundaries[i + 1]) for i in chosen])
    )
    if len(withheld) > max_fraction * n:
        raise ValueError("withheld episodes exceed the reserved fraction limit; raise rows_per_episode")
    cal = np.sort(np.concatenate([boundaries[i] + o for i, o in zip(chosen, offsets)]))
    train = np.flatnonzero(~np.isin(np.arange(n), withheld))
    digest = lambda rows: hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
    return (
        train.astype(np.int32),
        withheld.astype(np.int32),
        cal.astype(np.int32),
        {
            "boundary_rule": rule,
            "design": ("whole episodes (population split, not a calibration design)"
                       if rows_per_episode is None
                       else "length-proportional episodes, one row per K equal segments, "
                            "surplus over the target removed at random"),
            "target_size": int(target_size),
            "rows_per_episode": None if rows_per_episode is None else int(rows_per_episode),
            "calibration_size": len(cal),
            "withheld_size": len(withheld),
            "withheld_fraction": len(withheld) / n,
            "training_size": len(train),
            "reserved_blocks": len(chosen),
            "total_blocks": len(lengths),
            "seed": int(seed),
            "calibration_indices_sha256": digest(cal),
            "withheld_indices_sha256": digest(withheld),
        },
    )


def certifiable(n, config, tolerance=1e-6):
    """Whether a uniform-weight bank of n held-out scores certifies a finite threshold.

    A posterior draw crosses iff the test atom's mass, Beta(1, n), is at most alpha,
    so the crossing draws are Binomial(M, 1 - (1 - alpha)^n) and lambda_hpd needs
    ceil(credibility * M) of them. The bank qualifies when failing that has
    probability at most `tolerance` (n >= 36 at the declared 0.1 / 0.95 / 1000).
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        return False
    need = math.ceil(Fraction(str(config.credibility)) * config.draws)
    crossing = -math.expm1(n * math.log1p(-config.alpha))
    return float(stats.binom.cdf(need - 1, config.draws, crossing)) <= tolerance


def initial_reference(cal_params):
    """Not ready: hosts keep their native weighting until the first refresh."""
    inf = jnp.asarray(jnp.inf, jnp.float32)
    return FrozenReference(
        cal_params, jnp.asarray(1.0, jnp.float32), inf, inf, inf,
        jnp.asarray(0.0, jnp.float32), jnp.asarray(False),
    )


def positive_scale(predictions, residual_scale):
    return jnp.maximum(predictions, 1e-06) * residual_scale


def freeze_reference(cal_params, residual_scale, predictions, residuals, key, config):
    """Score held-out residuals with the frozen scale and select the WBCP threshold.

    Scores are |y - q| / (max(eta, 1e-6) u) with unit weights, the exchangeable case
    (BQ-CP with the test atom). Runs eagerly on the host at a refresh. Returns the
    reference, whether its inputs were valid, and scalar diagnostics; an invalid
    refresh returns valid=False and leaves the caller to keep its previous state.
    """
    residual_scale = jnp.asarray(residual_scale)
    predictions, residuals = jnp.asarray(predictions), jnp.asarray(residuals)
    if residual_scale.ndim != 0 or predictions.ndim != 1 or predictions.shape != residuals.shape:
        raise ValueError("expected aligned prediction/residual vectors and a scalar residual unit")
    scale = np.asarray(positive_scale(predictions, residual_scale), np.float64)
    residuals = np.asarray(residuals, np.float64)
    valid = bool(
        tree_valid(cal_params)
        and np.isfinite(float(residual_scale)) and float(residual_scale) > 0
        and predictions.size > 0
        and np.all(np.isfinite(scale) & (scale > 0))
        and np.all(np.isfinite(residuals))
    )
    if not valid:
        return initial_reference(cal_params), False, dict(certified=False)
    scores = np.abs(residuals) / scale
    result = wbcp.calibrate(
        scores, np.random.default_rng(np.asarray(key, np.uint32).ravel()),
        alpha=config.alpha, beta=config.credibility, draws=config.draws,
    )
    with np.errstate(over="ignore"):
        stored = np.float32(result.threshold)
    if result.certified and not np.isfinite(stored):  # a finite threshold must stay finite when stored
        return initial_reference(cal_params), False, dict(certified=False)
    as32 = lambda value: jnp.asarray(value, jnp.float32)
    reference = FrozenReference(
        cal_params, residual_scale.astype(jnp.float32), as32(result.threshold),
        as32(result.lambda_hat), as32(result.lambda_hpd), as32(result.n_eff), jnp.asarray(True),
    )
    diagnostics = dict(
        certified=result.certified, threshold=result.threshold, lambda_hat=result.lambda_hat,
        lambda_hpd=result.lambda_hpd, sigma_post=result.sigma_post, n_eff=result.n_eff,
        clamp_binds=result.lambda_hat > result.lambda_hpd, scores=int(scores.size),
    )
    return reference, True, diagnostics


def tree_valid(tree):
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


def reference_valid(reference):
    """Traceable storage check shared by the hosts' jitted updates."""
    fields = (reference.residual_scale, reference.threshold, reference.lambda_hat,
              reference.lambda_hpd, reference.n_eff, reference.ready)
    if any(jnp.shape(value) != () for value in fields) or reference.ready.dtype != jnp.bool_:
        raise ValueError("the frozen reference stores scalar thresholds and one residual unit")
    valid = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(reference.cal_params):
        valid = valid & jnp.all(jnp.isfinite(leaf))
    for value in (reference.threshold, reference.lambda_hat, reference.lambda_hpd):
        valid = valid & ~jnp.isnan(value) & (value >= 0)
    return (
        valid
        & jnp.isfinite(reference.residual_scale) & (reference.residual_scale > 0)
        & jnp.isfinite(reference.n_eff) & (reference.n_eff >= 0)
        & (~reference.ready
           | (reference.threshold == jnp.maximum(reference.lambda_hat, reference.lambda_hpd)))
    )
