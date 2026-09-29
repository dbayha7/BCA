"""CPU reference for weighted split conformal; not imported by training workers.

Inputs are explicit target/source density ratios, up to ONE common positive
constant for calibration and query points. This module checks arithmetic, not
whether a supplied ratio is statistically correct. Tibshirani et al. (2019),
equations (5)-(7), supply the covariate-shift contract.

The conformal calculation includes the query atom at infinity. The additional
Bayesian component tilts observed-support exponential bootstrap masses, retains
the existing BCA construction, and makes no posterior-risk confidence claim.
"""
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import itertools
import math

import numpy as np


def _vector(value, name, *, nonnegative=False, positive=False):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(name + ' must be a nonempty finite vector')
    if (nonnegative and np.any(result < 0)) or (positive and np.any(result <= 0)):
        raise ValueError(name + ' has invalid sign or zero scale')
    return result


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _probability(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(name + ' must be strictly between zero and one')
    return Fraction(str(value))


def density_ratios(source_density, target_density):
    """Compute Q/P at supplied points; no fitted-density correctness is implied.

    P=0 is refused, including 0/0. Target-only support cannot be repaired by an
    epsilon denominator. Representability errors are refused, never clipped.
    """
    source = _vector(source_density, 'source density', positive=True)
    target = _vector(target_density, 'target density', nonnegative=True)
    if source.shape != target.shape:
        raise ValueError('source and target densities must align')
    with np.errstate(over='ignore', under='ignore'):
        ratio = target / source
    if np.any(~np.isfinite(ratio)) or np.any((target > 0) & (ratio == 0)):
        raise FloatingPointError('density ratio is not representable; no clipping applied')
    return ratio


def normalized_scores(target, center, positive_scale):
    """Score a frozen prediction: |Y - center(X)| / scale(X)."""
    target = _vector(target, 'target')
    center = _vector(center, 'center')
    scale = _vector(positive_scale, 'positive scale', positive=True)
    if target.shape != center.shape or target.shape != scale.shape:
        raise ValueError('targets, centers and scales must align')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        error = np.abs(target - center)
        scores = error / scale
    if np.any(~np.isfinite(scores)) or np.any((error > 0) & (scores == 0)):
        raise FloatingPointError('normalized score is not representable')
    return scores


@dataclass(frozen=True)
class Radii:
    conformal: np.ndarray
    bayesian: np.ndarray
    full: np.ndarray
    query_mass: np.ndarray


@dataclass(frozen=True)
class Intervals:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    radii: Radii


@dataclass(frozen=True)
class WeightedReference:
    scores: np.ndarray
    ratios: np.ndarray
    sorted_scores: np.ndarray
    cumulative_mass: tuple
    total_mass: Fraction
    alpha: Fraction
    bayesian: float
    bootstrap_quantiles: np.ndarray
    ess: float

    def radii(self, query_ratios):
        """A query-specific conformal floor; infinity is a valid result.

        Compare cumulative calibration mass with
        (1-alpha)*(total calibration mass + query mass).
        Exact rational comparisons avoid rounding a tiny query mass to zero at
        a boundary. They are exact for the represented float64 ratios, not for
        an unknown true population density ratio.
        """
        queries = _vector(query_ratios, 'query ratios', nonnegative=True)
        floor, query_mass = [], []
        for query in queries:
            query = Fraction.from_float(float(query))
            cutoff = (1 - self.alpha) * (self.total_mass + query)
            index = bisect_left(self.cumulative_mass, cutoff)
            floor.append(float(self.sorted_scores[index])
                         if index < len(self.sorted_scores) else math.inf)
            query_mass.append(float(query / (self.total_mass + query)))
        floor = np.asarray(floor)
        bayes = np.full(floor.shape, self.bayesian)
        return Radii(*map(_readonly, (floor, bayes, np.maximum(floor, bayes), query_mass)))

    def intervals(self, centers, positive_scales, query_ratios):
        """Construct intervals for the named response Y, not true policy value."""
        centers = _vector(centers, 'query centers')
        scales = _vector(positive_scales, 'query scales', positive=True)
        result = self.radii(query_ratios)
        if centers.shape != scales.shape or centers.shape != result.full.shape:
            raise ValueError('query centers, scales and ratios must align')
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            width = scales * result.full
            lower, upper = centers - width, centers + width
        finite_radius = np.isfinite(result.full)
        invalid = finite_radius & (~np.isfinite(width) | ~np.isfinite(lower)
                                   | ~np.isfinite(upper)
                                   | ((result.full > 0) & (width == 0)))
        if np.any(invalid):
            raise FloatingPointError('interval arithmetic overflow/underflow; not a statistical infinity')
        return Intervals(_readonly(lower), _readonly(upper), _readonly(width), result)


def build_reference(scores, calibration_ratios, bootstrap_draws, *,
                    alpha=0.1, credibility=0.95):
    """Freeze an explicit weighted calibration bank and bootstrap realization.

    Draws must be provided by the caller (normally iid Exponential(1)); no RNG
    is consumed implicitly. The reference does not fit ratios, choose a target
    population, cap/temper weights, or attest independence of calibration data.
    """
    scores = _vector(scores, 'scores', nonnegative=True)
    ratios = _vector(calibration_ratios, 'calibration ratios', nonnegative=True)
    alpha = _probability(alpha, 'alpha')
    credibility = _probability(credibility, 'credibility')
    if scores.shape != ratios.shape or not np.any(ratios > 0):
        raise ValueError('aligned ratios with positive calibration mass required')
    draws = np.array(bootstrap_draws, dtype=np.float64, copy=True)
    if (draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] != scores.size
            or not np.all(np.isfinite(draws)) or np.any(draws < 0)):
        raise ValueError('bootstrap draws must be a finite nonnegative M by n matrix, M>=2')
    support = (draws > 0) & (ratios[None, :] > 0)
    if not np.all(np.any(support, axis=1)):
        raise ValueError('every bootstrap draw requires positive supported mass')
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    exact_mass = tuple(Fraction.from_float(float(w)) for w in ratios[order])
    cumulative = tuple(itertools.accumulate(exact_mass))

    # Tilting in log space avoids overflowing raw draws*ratios. This is the
    # observed-support Bayesian component, not the conformal query atom.
    with np.errstate(divide='ignore'):
        log_mass = np.log(draws) + np.log(ratios)[None, :]
    mass = np.exp(log_mass - np.max(log_mass, axis=1, keepdims=True))
    if np.any(support & (mass == 0)):
        raise FloatingPointError('positive bootstrap mass underflowed; no silent truncation')
    cumulative_draws = np.cumsum(mass[:, order], axis=1)
    cutoffs = float(1 - alpha) * cumulative_draws[:, -1]
    indices = np.sum(cumulative_draws < cutoffs[:, None], axis=1)
    quantiles = sorted_scores[indices]
    rank = math.ceil(credibility * len(quantiles))
    bayesian = float(np.sort(quantiles)[rank - 1])
    normalized = ratios / np.max(ratios)
    ess = float(np.sum(normalized) ** 2 / np.dot(normalized, normalized))
    return WeightedReference(_readonly(scores), _readonly(ratios), _readonly(sorted_scores),
                             cumulative, cumulative[-1], alpha, bayesian,
                             _readonly(quantiles), ess)
