"""Exact-query adapter around the actual native WBCP procedure (NumPy only).

This module does not claim a frequentist coverage theorem. ``load_native_wbcp``
executes the supplied, unmodified calibration/wbcp.py. Each distinct query mass
gets a native calibration call with the same exponential random stream. There is
no query grid approximation, ratio clipping, zero-weight deletion, or training.
"""
from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


@dataclass(frozen=True)
class QueryCalibration:
    threshold: np.ndarray
    lambda_hat: np.ndarray
    lambda_hpd: np.ndarray
    refused: np.ndarray
    reason: np.ndarray
    query_mass: np.ndarray
    n_eff: float
    positive_weights: int
    bank_size: int
    weight_scale: float
    unique_results: tuple
    settings: dict

    def summary(self):
        return json_ready(dict(bank_size=self.bank_size, positive_weights=self.positive_weights,
            n_eff=self.n_eff, weight_scale=self.weight_scale,
            unique_native_calls=len(self.unique_results), unique_results=self.unique_results,
            settings=self.settings, refusal_counts={str(k): int(v) for k,v in
                zip(*np.unique(self.reason, return_counts=True))}))


@dataclass(frozen=True)
class QueryEvaluation:
    half_width: np.ndarray
    dose: np.ndarray
    covered: np.ndarray
    usable: np.ndarray
    refused: np.ndarray
    summary: dict

    @property
    def accepted(self):
        """Finite-band usability, separate from the covered-and-accepted mask."""
        return ~self.refused

    @property
    def usable_coverage_mask(self):
        return self.usable


def json_ready(value):
    """Retain infinity as an explicit string; undefined metrics use None."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return 'inf' if np.isposinf(value) else '-inf' if np.isneginf(value) else float(value)
    return value


def load_native_wbcp(path):
    """Load just the actual NumPy source, without package imports or bytecode writes."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'calibration' / 'wbcp.py'
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    name = '_action_only_native_wbcp_' + digest
    if name not in sys.modules:
        module = ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        try:
            exec(compile(source, str(path), 'exec'), module.__dict__)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def _vector(value, name, *, nonnegative=False):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(name + ' must be a nonempty finite vector')
    if nonnegative and np.any(array < 0):
        raise ValueError(name + ' must be nonnegative')
    return array


def _seed(seed):
    values = [seed] if isinstance(seed, (int, np.integer)) else list(seed) if isinstance(seed, (list, tuple)) else []
    if not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in values):
        raise ValueError('posterior_seed must be an explicit nonnegative integer or sequence of integers')
    return [int(v) for v in values]


class _ReplayExponentials(np.random.Generator):
    """Satisfy the native Generator contract while replaying caller-owned draws."""
    def __init__(self, values):
        super().__init__(np.random.PCG64(0))
        self.values = values
        self.position = 0

    def standard_exponential(self, size=None, dtype=np.float64, method='zig', out=None):
        if out is not None or np.dtype(dtype) != np.dtype(np.float64) or len(size) != 2 or size[1] != self.values.shape[1]:
            raise ValueError('native exponential request does not match the supplied draw matrix')
        stop = self.position + size[0]
        if stop > len(self.values):
            raise ValueError('native requested more exponential draws than supplied')
        answer = self.values[self.position:stop]
        self.position = stop
        return answer


def calibrate_queries(scores, calibration_weights, query_masses, *, posterior_seed=None,
                      native, alpha=.1, beta=.95, draws=1000, query_supported=None,
                      exponential_draws=None):
    """Run native WBCP exactly once per distinct supported finite query mass.

    Native normalization divides BOTH weights and test mass by max(bank weights).
    Every call restarts the same Generator, preserving all n zero-weight slots;
    thus n+1 exponential draws align across query masses and weighting methods.

    All-zero bank -> ``no_positive_calibration_weight``. Explicit unsupported
    mask or +inf query ratio -> ``unsupported_query``. These are different from
    ``posterior_infinite`` with a nonempty positive-weight bank. All return R=inf.
    Query mass zero is not a target query and is rejected rather than floored.
    """
    scores = _vector(scores, 'scores', nonnegative=True)
    weights = _vector(calibration_weights, 'calibration_weights', nonnegative=True)
    if weights.shape != scores.shape:
        raise ValueError('calibration_weights must align with scores')
    mass = np.asarray(query_masses, dtype=np.float64)
    if mass.ndim != 1 or not mass.size or np.any(np.isnan(mass) | (mass <= 0)):
        raise ValueError('query_masses must be a nonempty positive vector (+inf means unsupported)')
    if query_supported is None:
        supported = np.ones(mass.shape, dtype=bool)
    else:
        supported = np.asarray(query_supported)
        if supported.shape != mass.shape or supported.dtype != np.bool_:
            raise ValueError('query_supported must be a matching boolean vector')
        supported = supported.copy()
    if not (np.isfinite(alpha) and np.isfinite(beta) and 0 < alpha < 1 and 0 < beta < 1):
        raise ValueError('alpha and beta must lie strictly between zero and one')
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError('draws must be a positive integer')
    seed = None
    exponentials = None
    if exponential_draws is not None:
        if posterior_seed is not None:
            raise ValueError('provide posterior_seed OR exponential_draws, not both')
        exponentials = np.asarray(exponential_draws, dtype=np.float64)
        if exponentials.shape != (draws,len(scores)+1) or not np.isfinite(exponentials).all() or np.any(exponentials < 0):
            raise ValueError('exponential_draws must be a finite nonnegative [draws,n_scores+1] matrix')
    else:
        seed = _seed(posterior_seed)
    supported &= np.isfinite(mass)
    threshold = np.full(mass.shape, np.inf)
    empirical = np.full(mass.shape, np.nan)
    posterior = np.full(mass.shape, np.nan)
    reason = np.full(mass.shape, 'unsupported_query', dtype='<U40')
    scale = float(weights.max())
    n_positive = int(np.count_nonzero(weights))
    table = []
    ess = 0.
    if n_positive:
        stable_weights = weights / scale
        ess = float(stable_weights.sum() ** 2 / np.dot(stable_weights, stable_weights))
        with np.errstate(over='ignore', under='ignore'):
            normalized_mass = mass[supported] / scale
        if np.any(~np.isfinite(normalized_mass) | (normalized_mass <= 0)):
            raise ValueError('query mass / maximum bank weight is not representable; no numerical support repair is applied')
        sorted_weights = stable_weights[np.argsort(scores, kind='stable')]
        if exponentials is not None and np.any((exponentials[:,:-1] @ sorted_weights == 0) & (exponentials[:,-1] == 0)):
            raise ValueError('supplied exponential draw has zero posterior mass')
        # No rounding: exact floating-point unique values are the actual queries.
        for test_mass in np.unique(mass[supported]):
            rng = _ReplayExponentials(exponentials) if exponentials is not None else np.random.default_rng(seed)
            cal = native.calibrate(scores, rng, weights,
                                   float(test_mass), alpha=float(alpha), beta=float(beta), draws=int(draws))
            selected = supported & (mass == test_mass)
            threshold[selected] = cal.threshold
            empirical[selected] = cal.lambda_hat
            posterior[selected] = cal.lambda_hpd
            reason[selected] = 'accepted' if math.isfinite(cal.threshold) else 'posterior_infinite'
            table.append(dict(query_mass=float(test_mass), normalized_query_mass=float(test_mass/scale),
                threshold=float(cal.threshold), lambda_hat=float(cal.lambda_hat),
                lambda_hpd=float(cal.lambda_hpd), n_eff=float(cal.n_eff),
                finite_posterior_fraction=float(np.isfinite(cal.crossings).mean()),
                crossings_sha256=hashlib.sha256(np.asarray(cal.crossings, dtype='<f8').tobytes()).hexdigest()))
    else:
        reason[supported] = 'no_positive_calibration_weight'
    return QueryCalibration(threshold, empirical, posterior, ~np.isfinite(threshold), reason,
        mass.copy(), ess, n_positive, len(scores), scale, tuple(table),
        dict(alpha=float(alpha), beta=float(beta), draws=int(draws), posterior_seed=seed,
             exponential_draws_sha256=hashlib.sha256(np.asarray(exponentials,dtype='<f8').tobytes()).hexdigest() if exponentials is not None else None,
             query_rule='per-query exact mass', approximation='none', zero_weight_rows='retained',
             threshold_arithmetic='native WBCP float64; empirical boundary correction uses exact fractions',
             fallback='neutral dose 1; no score or query is removed'))


def calibrate(scores, weights, test_masses, exponential_draws, *, native, alpha=.1,
              credibility=.95, query_supported=None):
    """Driver API: all randomness belongs to the caller, actual native code owns R.

    Exponentials have columns aligned with stable SCORE order, not original row
    order, followed by the unseen query atom. Native always performs that same
    stable score sort; zero-weight rows are retained in both arms.
    """
    exponentials = np.asarray(exponential_draws, dtype=np.float64)
    if exponentials.ndim != 2:
        raise ValueError('exponential_draws must be a matrix')
    return calibrate_queries(scores,weights,test_masses,native=native,alpha=alpha,beta=credibility,
        draws=len(exponentials),query_supported=query_supported,exponential_draws=exponentials)


def calibrate_pair(scores, calibration_weights, query_masses, *, posterior_seed,
                   native, alpha=.1, beta=.95, draws=1000, query_supported=None):
    """Same source scores/stream: baseline ones/1 versus supplied exact ratios."""
    mass = np.asarray(query_masses, dtype=np.float64)
    if query_supported is None:
        query_supported = np.isfinite(mass)
    else:
        query_supported = np.asarray(query_supported)
        if query_supported.dtype != np.bool_ or query_supported.shape != mass.shape:
            raise ValueError('query_supported must be a matching boolean vector')
        query_supported = query_supported & np.isfinite(mass)
    common = dict(posterior_seed=posterior_seed, native=native, alpha=alpha, beta=beta,
                  draws=draws, query_supported=query_supported)
    plain = calibrate_queries(scores, np.ones_like(np.asarray(scores, dtype=np.float64)),
                              np.ones_like(mass), **common)
    weighted = calibrate_queries(scores, calibration_weights, mass, **common)
    return {'unweighted': plain, 'exact_iw': weighted}


def _ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    high = np.cumsum(counts)
    return (high - (counts - 1)/2)[inverse]


def _spearman(x, y):
    if len(x) < 2:
        return None
    x, y = _ranks(x), _ranks(y)
    x, y = x-x.mean(), y-y.mean()
    norm = math.sqrt(float(np.dot(x,x) * np.dot(y,y)))
    return float(np.dot(x,y)/norm) if norm else None


def evaluate_queries(result, scores, scales, *, unit, blend=.5,
                     absolute_error=None, action_harm=None):
    """Count every query, including refusals; scale is sigma, not raw eta.

    coverage_including_infinite counts mathematically vacuous infinite intervals.
    usable_coverage requires a finite accepted interval, with ALL queries as its
    denominator. finite_conditional_coverage uses only accepted queries and may
    be None. These descriptive metrics do not assert calibrated coverage.
    """
    scores = _vector(scores, 'evaluation scores', nonnegative=True)
    scales = _vector(scales, 'scales', nonnegative=True)
    if scores.shape != scales.shape or scores.shape != result.threshold.shape or np.any(scales <= 0):
        raise ValueError('scores, strictly positive scales, and query thresholds must align')
    if not (np.isfinite(unit) and unit > 0 and np.isfinite(blend) and 0 <= blend <= 1):
        raise ValueError('unit must be finite positive and blend must be in [0,1]')
    accepted = ~result.refused
    width = np.full(scores.shape, np.inf)
    with np.errstate(over='ignore'):
        width[accepted] = result.threshold[accepted] * scales[accepted]
    if not np.isfinite(width[accepted]).all():
        raise ValueError('finite threshold times scale overflowed; refusing to mislabel numerical failure as posterior refusal')
    dose = np.ones(scores.shape)
    # Stable U/(U+unit), including U=0; all refused rows retain neutral dose 1.
    positive = accepted & (width > 0)
    big = positive & (width >= unit)
    dose[big] += blend / (1 + unit/width[big])
    small = positive & ~big
    ratio = width[small]/unit
    dose[small] += blend * ratio/(1+ratio)
    covered = scores <= result.threshold
    usable = covered & accepted
    finite_width = width[accepted]
    structural = result.reason == 'unsupported_query'
    summary = dict(n_queries=len(scores), accepted_queries=int(accepted.sum()),
        refused_queries=int(result.refused.sum()), fallback_queries=int(result.refused.sum()),
        coverage_including_infinite_count=int(covered.sum()), usable_coverage_count=int(usable.sum()),
        finite_conditional_coverage_denominator=int(accepted.sum()),
        structural_support_failure_count=int(structural.sum()),
        importance_law_valid_on_queries=not bool(structural.any()),
        coverage_interpretation='descriptive interval containment only; source-support failure invalidates an importance-law claim' if structural.any() else 'descriptive; no coverage theorem asserted',
        width_dose_arithmetic='float64 in both arms; not bitwise deployed FrozenReference float32 arithmetic',
        refusal_rate=float(result.refused.mean()), coverage_including_infinite=float(covered.mean()),
        usable_coverage=float(usable.mean()),
        finite_conditional_coverage=float(covered[accepted].mean()) if accepted.any() else None,
        mean_half_width_including_infinite=float(width.mean()),
        mean_finite_half_width=float(finite_width.mean()) if finite_width.size else None,
        median_finite_half_width=float(np.median(finite_width)) if finite_width.size else None,
        mean_dose_including_refusal=float(dose.mean()),
        mean_accepted_dose=float(dose[accepted].mean()) if accepted.any() else None,
        fallback_dose=1., refusal_counts={str(k):int(v) for k,v in zip(*np.unique(result.reason, return_counts=True))})
    for name, values in [('absolute_error', absolute_error), ('action_harm', action_harm)]:
        if values is not None:
            values = _vector(values, name, nonnegative=True)
            if values.shape != scores.shape:
                raise ValueError(name + ' must align with queries')
            summary[name+'_dose_spearman_all'] = _spearman(dose, values)
            summary[name+'_width_spearman_accepted'] = _spearman(width[accepted], values[accepted])
            summary[name+'_accepted_count'] = int(accepted.sum())
    return QueryEvaluation(width, dose, covered, usable, result.refused.copy(), json_ready(summary))


def paired_metrics(unweighted, weighted):
    """Descriptive differences on identical query rows; not an independence claim."""
    if unweighted.dose.shape != weighted.dose.shape:
        raise ValueError('paired evaluations must have matching row shapes')
    joint = ~unweighted.refused & ~weighted.refused
    return json_ready(dict(n_queries=len(joint),
        delta_coverage_including_infinite=float((weighted.covered.astype(float)-unweighted.covered).mean()),
        delta_usable_coverage=float((weighted.usable.astype(float)-unweighted.usable).mean()),
        delta_refusal_rate=float((weighted.refused.astype(float)-unweighted.refused).mean()),
        delta_mean_dose=float(np.mean(weighted.dose-unweighted.dose)),
        jointly_accepted_fraction=float(joint.mean()),
        delta_mean_half_width_jointly_accepted=float(np.mean(weighted.half_width[joint]-unweighted.half_width[joint])) if joint.any() else None,
        only_weighted_refused=int(np.sum(weighted.refused & ~unweighted.refused)),
        only_unweighted_refused=int(np.sum(unweighted.refused & ~weighted.refused)),
        both_refused=int(np.sum(weighted.refused & unweighted.refused))))


def advisor_source_map(native):
    """JSON-ready stage labels and actual native source ranges for the report."""
    refs = {}
    for name in ('crossings', 'calibrate'):
        function = getattr(native, name)
        lines, first = inspect.getsourcelines(function)
        path = Path(inspect.getsourcefile(function)).resolve()
        refs[name] = dict(path=str(path), first_line=first, last_line=first+len(lines)-1,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(native=refs, stages=[
        dict(id='C1', purpose='Preserve support and the actual bank size',
             pseudocode='Keep zero-weight rows. Refuse unsupported queries and all-zero banks explicitly.'),
        dict(id='C2', purpose='Preserve relative query mass', source='calibrate',
             pseudocode='Native: divide bank weights and query mass by the SAME maximum bank weight.'),
        dict(id='C3', purpose='Empirical weighted tail-risk floor', source='calibrate',
             pseudocode='Native: select smallest score with weighted empirical tail <= alpha using rational boundary correction.'),
        dict(id='C4', purpose='Posterior query atom', source='crossings',
             pseudocode='Native: use w_i*E_i and t*E_query; choose first score reaching (1-alpha) total mass, else infinity.'),
        dict(id='C5', purpose='Credibility selection', source='calibrate',
             pseudocode='Native: R=max(empirical floor, ceil(beta*M)-th sorted posterior crossing).'),
        dict(id='C6', purpose='Exact per-query reuse',
             pseudocode='For each EXACT unique finite query mass, call native with the SAME seeded Generator or replayed exponential matrix. No grid rounding.'),
        dict(id='C7', purpose='Matched fallback and visible refusal',
             pseudocode='Half-width=R*sigma; finite dose=1+blend*U/(U+unit); refused dose=1. Count all queries.'),
        dict(id='C8', purpose='Separate useful coverage from vacuous intervals',
             pseudocode='Report coverage including infinity, finite conditional coverage, usable coverage, refusal, and paired changes separately.')])
