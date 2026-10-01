"""Weighted Bayesian conformal prediction (WBCP) for the miscoverage loss.

Lou and Luo, *Weighted Bayesian Conformal Prediction*, arXiv:2604.06464v3,
Algorithm 1. Held-out nonconformity scores receive the normalized weighted
Bayesian bootstrap masses w_i E_i / sum_j w_j E_j: the exact posterior of a
DP(tau -> 0) prior on the calibration law, pushed through the likelihood ratio
(Eq. 3). One extra atom with mass wbar E_{n+1} carries the unseen test point at
the worst-case loss (Eq. 6). The threshold is the smallest score at which the
posterior probability that miscoverage stays below alpha reaches beta (Eq. 7),
clamped from below by the weighted empirical quantile (Eq. 1). Uniform weights
recover BQ-CP (Snell and Griffiths, 2025) exactly.

Weights and the test mass are w(x) = dP_test/dP_cal at the calibration points
and wbar = E_test[w]. They must share one scale; a common factor cancels.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

_CHUNK_ELEMENTS = 4_000_000  # bounds the draws-by-scores matrix held at once


@dataclass(frozen=True)
class Calibration:
    threshold: float  # lambda_dep = max(lambda_hat, lambda_hpd); inf: nothing certifiable
    lambda_hat: float  # weighted empirical selection, Eq. (1)
    lambda_hpd: float  # beta-credible posterior selection, Eq. (7)
    crossings: np.ndarray  # sorted per-draw alpha-crossings lambda^(m); the threshold posterior
    n_eff: float  # Kish effective sample size of the calibration weights, Eq. (5)

    @property
    def certified(self):
        return math.isfinite(self.threshold)

    @property
    def sigma_post(self):
        return float(np.std(self.crossings)) if np.all(np.isfinite(self.crossings)) else math.inf


def crossings(sorted_scores, sorted_weights, test_mass, exponentials, alpha):
    """Algorithm 1, lines 3-4: each draw's alpha-crossing on the score grid.

    Row m of `exponentials` holds E_1..E_n, aligned with the sorted scores, then
    E_{n+1} for the test atom. L+(lambda) <= alpha exactly when the posterior mass
    of scores <= lambda reaches 1-alpha of the total, the test atom included; a
    draw whose test atom alone exceeds alpha never crosses (+inf).
    """
    mass = exponentials[:, :-1] * sorted_weights
    total = mass.sum(axis=1) + exponentials[:, -1] * test_mass
    covered = np.cumsum(mass, axis=1) >= (1.0 - alpha) * total[:, None]
    return np.where(covered.any(axis=1), sorted_scores[covered.argmax(axis=1)], np.inf)


def calibrate(scores, rng, weights=None, test_mass=None, *, alpha=0.1, beta=0.95, draws=1000):
    """Run Algorithm 1 and return the deployed threshold with its posterior.

    `rng` is a numpy Generator; no randomness is drawn implicitly. Omitting
    `weights` is the exchangeable case (all weights and the test mass equal 1).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a nonempty finite vector")
    if weights is None:
        if test_mass is not None:
            raise ValueError("test_mass requires explicit calibration weights")
        weights, test_mass = np.ones_like(scores), 1.0
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != scores.shape or not np.all(np.isfinite(weights) & (weights >= 0)):
            raise ValueError("weights must be finite, nonnegative and aligned with scores")
        if not weights.max() > 0:
            raise ValueError("calibration weights need positive total mass")
        if test_mass is None or not (math.isfinite(test_mass) and test_mass > 0):
            raise ValueError("weighted calibration requires a finite positive test_mass")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must lie strictly between zero and one")
    if not isinstance(rng, np.random.Generator) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("rng must be a numpy Generator and draws a positive integer")

    # Only weight ratios matter; rescaling to max 1 keeps every running sum finite.
    scale = weights.max()
    weights, test_mass = weights / scale, test_mass / scale
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_weights = scores[order], weights[order]

    # Eq. (1): smallest score whose weighted empirical miscoverage is at most alpha.
    # The float index is settled in exact rationals, since R_w = alpha counts.
    cumulative = np.cumsum(sorted_weights)
    k = int(np.argmax(cumulative >= (1.0 - alpha) * cumulative[-1]))
    exact = [Fraction(w) for w in sorted_weights]
    target = (1 - Fraction(str(alpha))) * sum(exact)
    covered = sum(exact[:k + 1])
    while k > 0 and covered - exact[k] >= target:
        covered -= exact[k]
        k -= 1
    while covered < target:
        k += 1
        covered += exact[k]
    lambda_hat = float(sorted_scores[k])

    n = scores.size
    chunk = max(1, _CHUNK_ELEMENTS // (n + 1))
    posterior = np.concatenate([
        crossings(sorted_scores, sorted_weights, test_mass,
                  rng.standard_exponential((min(chunk, draws - start), n + 1)), alpha)
        for start in range(0, draws, chunk)
    ])
    posterior.sort()

    # Eq. (7): Pr(L+ <= alpha) at lambda is the fraction of draws crossing at or
    # below lambda, so the smallest beta-credible grid point is an order statistic.
    lambda_hpd = float(posterior[math.ceil(Fraction(str(beta)) * int(draws)) - 1])
    n_eff = float(weights.sum() ** 2 / np.dot(weights, weights))  # weights rescaled above
    return Calibration(max(lambda_hat, lambda_hpd), lambda_hat, lambda_hpd, posterior, n_eff)
