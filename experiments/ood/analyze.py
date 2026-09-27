"""Offline OOD calculations. No model queries, fitting, environment or file writes.

Returns measure a fixed continuation, widths measure frozen residual bands, and
support distance is only a proxy. None is interchangeable with true Q error.
"""

from decimal import Decimal, ROUND_CEILING
import math
import numpy as np
from scipy.stats import beta, binom, rankdata


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vector(values, *, infinity=False):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and not np.isnan(a).any(), 'Expected a vector without NaN.')
    require(infinity or np.isfinite(a).all(), 'Nonfinite input.')
    return a


def harm(reference_return, candidate_returns):
    """Positive loss means the candidate collected less reward than the reference."""
    require(np.ndim(reference_return) == 0 and np.isfinite(reference_return), 'Invalid reference return.')
    return float(reference_return) - vector(candidate_returns)


def _rank_inputs(labels, scores):
    y, s = vector(labels), vector(scores, infinity=True)
    require(y.shape == s.shape and np.isin(y, [0, 1]).all(), 'Ranking needs matched binary labels.')
    return y.astype(bool), s


def auc(labels, scores):
    """P(score_harmful > score_nonharmful) plus half credit for ties."""
    y, s = _rank_inputs(labels, scores)
    positive, negative = s[y], s[~y]
    if not len(positive) or not len(negative):
        return None
    return float(((positive[:, None] > negative).astype(float)
                  + .5*(positive[:, None] == negative)).mean())


def average_precision(labels, scores):
    """Stepwise AP; tied scores enter together, never sorted by their labels."""
    y, s = _rank_inputs(labels, scores)
    if not y.any() or y.all():
        return None
    total = 0.
    for cutoff in np.unique(s)[::-1]:
        group, selected = s == cutoff, s >= cutoff
        total += y[group].sum() / y.sum() * y[selected].mean()
    return float(total)


def loss_rank(losses, scores):
    a, b = rankdata(vector(losses)), rankdata(vector(scores, infinity=True))
    require(a.shape == b.shape, 'Loss/score shapes differ.')
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def risk_retention(losses, scores, threshold=1.):
    """Keep complete low-score tie groups; no label-informed tie breaking."""
    losses, scores = vector(losses), vector(scores, infinity=True)
    require(losses.shape == scores.shape, 'Loss/score shapes differ.')
    return [dict(cutoff=float(c), retained_count=int((scores <= c).sum()),
                 retained_fraction=float((scores <= c).mean()),
                 mean_harm=float(losses[scores <= c].mean()),
                 harmful_fraction=float((losses[scores <= c] > threshold).mean()))
            for c in np.unique(scores)]


def state_metrics(panel, threshold=1.):
    """One state/continuation; slot 0 is the reference, even if aliased later.

Aliases must have identical outcomes and scores. A mismatch is a provenance
failure, not permission to average or select a convenient duplicate.
"""
    actions = np.asarray(panel['actions'], dtype=np.float64)
    returns = vector(panel['returns'])
    require(actions.ndim == 2 and len(actions) == len(returns) and len(actions) > 0
            and np.isfinite(actions).all(), 'Invalid applied-action panel.')
    scores = {k: vector(panel[k], infinity=(k == 'width')) for k in ['width', 'support', 'random']}
    require(all(len(s) == len(actions) for s in scores.values()), 'Panel lengths disagree.')
    require((scores['width'] >= 0).all() and (scores['support'] >= 0).all(), 'Negative width/distance.')
    groups = []
    for i, action in enumerate(actions):
        match = next((g for g in groups if np.array_equal(actions[g[0]], action)), None)
        if match is None:
            groups.append([i])
        else:
            require(returns[i] == returns[match[0]] and all(s[i] == s[match[0]] for s in scores.values()),
                    'Conflicting duplicate applied action; preserve and investigate.')
            match.append(i)
    indices = [g[0] for g in groups if 0 not in g]
    losses = harm(returns[0], returns[indices])
    y = losses > threshold
    result = dict(aliases=groups, alternative_indices=indices, alternatives=len(indices),
                  harmful=int(y.sum()), losses=losses.tolist(), labels=y.tolist())
    scores = {k: s[indices] for k, s in scores.items()}
    scores['constant'] = np.ones(len(indices))
    for name, score in scores.items():
        result[name+'_auc'] = auc(y, score)
        result[name+'_ap'] = average_precision(y, score)
    result['loss_rank'] = loss_rank(losses, scores['width'])
    result['risk_retention'] = risk_retention(losses, scores['width'], threshold)
    result['scores'] = {k: v.tolist() for k, v in scores.items()}
    return result


def _scores(scores):
    x = vector(scores)
    require(len(x) > 0 and (x >= 0).all(), 'Calibration scores must be nonnegative and nonempty.')
    return np.sort(x)


def _ceil_probability(n, p):
    return int((Decimal(n)*p).to_integral_value(rounding=ROUND_CEILING))


def conformal_radius(scores, alpha=.1):
    """Unweighted finite-sample rank ceil((n+1)*(1-alpha)); infinity is retained."""
    x = _scores(scores)
    require(0 < alpha < 1, 'Invalid miscoverage level.')
    k = _ceil_probability(len(x)+1, Decimal(1)-Decimal(str(alpha)))
    return float(x[k-1]) if k <= len(x) else math.inf


def tolerance_radius(scores, probability=.9, credibility=.95):
    """IID order-statistic risk reference, distinct from marginal conformal rank."""
    x = _scores(scores)
    require(0 < probability < 1 and 0 < credibility < 1, 'Invalid probability.')
    choices = np.flatnonzero(binom.cdf(np.arange(len(x)), len(x), probability) >= credibility)
    return float(x[choices[0]]) if len(choices) else math.inf


def radius_reference(scores, bootstrap_weights, alpha=.1, credibility=.95):
    """Independent NumPy audit of explicit unweighted Bayesian-bootstrap draws.

Uses saved draws; does not replace or change the production JAX calibrator.
No weighted-shift guarantee is introduced by this reference calculation.
"""
    raw = vector(scores)
    _scores(raw)
    require(0 < alpha < 1 and 0 < credibility < 1, 'Invalid probability.')
    w = np.asarray(bootstrap_weights, dtype=float)
    require(w.ndim == 2 and w.shape[1] == len(raw) and len(w) >= 2
            and np.isfinite(w).all() and (w >= 0).all() and (w.sum(1) > 0).all(), 'Invalid saved draws.')
    order = np.argsort(raw)
    cdf = np.cumsum(w[:, order] / w.sum(1, keepdims=True), axis=1)
    at = np.minimum((cdf < 1-alpha).sum(1), len(raw)-1)
    qs = raw[order][at]
    k = _ceil_probability(len(qs), Decimal(str(credibility)))
    bayes = float(np.sort(qs)[k-1])
    floor = conformal_radius(raw, alpha)
    return dict(bayesian=bayes, conformal=floor, radius=max(bayes, floor), posterior_quantiles=qs)


def coverage(errors, widths, confidence=.95):
    """Conditional test-bank miscoverage and Clopper-Pearson episode interval."""
    errors, widths = vector(errors), vector(widths, infinity=True)
    require(len(errors) > 0 and errors.shape == widths.shape and (errors >= 0).all()
            and (widths >= 0).all() and 0 < confidence < 1, 'Invalid coverage inputs.')
    n, k = len(errors), int((errors > widths).sum())
    tail = (1-confidence)/2
    lo = 0. if k == 0 else float(beta.ppf(tail, k, n-k+1))
    hi = 1. if k == n else float(beta.ppf(1-tail, k+1, n-k))
    return dict(samples=n, failures=k, miscoverage=k/n, coverage=1-k/n,
                miscoverage_interval=[lo, hi], infinite_bounds=int(np.isinf(widths).sum()),
                mean_width=float(widths.mean()))


def cluster_indices(banks, *, draws, seed, crossed=False):
    """Resample seeds, then WHOLE paired reset blocks; no action-level resampling."""
    require(type(draws) is int and draws > 0 and type(seed) is int and seed >= 0, 'Invalid bootstrap declaration.')
    seeds = sorted(banks)
    require(seeds and all(banks[s] and len(set(banks[s])) == len(banks[s]) for s in seeds), 'Invalid block bank.')
    sets = [set(banks[s]) for s in seeds]
    if crossed:
        require(all(s == sets[0] for s in sets), 'Partially crossed resets need a separate declared design.')
    else:
        require(all(not a & b for i,a in enumerate(sets) for b in sets[i+1:]),
                'Shared reset IDs require crossed resampling.')
    rng, result = np.random.default_rng(seed), []
    for _ in range(draws):
        selected = rng.choice(seeds, len(seeds))
        common = rng.choice(sorted(sets[0]), len(sets[0])).tolist() if crossed else None
        result.append([(int(s), common[:] if crossed else rng.choice(banks[int(s)], len(banks[int(s)])).tolist())
                       for s in selected])
    return result


def paired_ranking_bootstrap(records, *, expected_strata, draws=10000, seed,
                             crossed=False, confidence=.95, family_size=8, expected_seeds=None):
    """Approximate seed/reset intervals for two paired within-state AUC contrasts.

Each row is ONE state/continuation; all its scores travel together. Strata are
collector/capture identities. Call once per declared continuation. Missing strata
or any unestimable draw withhold intervals rather than silently dropping draws.
"""
    require(records and expected_strata and len(set(expected_strata)) == len(expected_strata), 'Empty/duplicate design.')
    require(0 < confidence < 1 and type(family_size) is int and family_size >= 1, 'Invalid interval family.')
    seeds = sorted(set(r['seed'] for r in records))
    require(all(type(s) is int for s in seeds), 'Integer seed identity required.')
    require(expected_seeds is None or set(seeds) == set(expected_seeds), 'Missing/extra declared training seed.')
    banks = {s: sorted(set(r['block'] for r in records if r['seed'] == s)) for s in seeds}
    groups = {(s,b): [r for r in records if r['seed'] == s and r['block'] == b] for s in seeds for b in banks[s]}
    require(all(r['stratum'] in expected_strata for r in records), 'Unknown stratum.')
    # Per-block sums/counts preserve unequal state counts and missing classes.
    tables = {}
    for s in seeds:
        totals = np.zeros((len(banks[s]), len(expected_strata), 3))
        for bi,b in enumerate(banks[s]):
            for r in groups[s,b]:
                w, d = r['width_auc'], r['support_auc']
                require((w is None) == (d is None), 'Scores must use the same valid-state mask.')
                if w is not None:
                    require(np.isfinite([w,d]).all() and 0 <= w <= 1 and 0 <= d <= 1, 'Invalid AUROC.')
                    j = expected_strata.index(r['stratum'])
                    totals[bi,j] += [w-.5, w-d, 1]
        tables[s] = totals

    def estimate(s, selected):
        sums = tables[s][selected].sum(0)
        good = sums[:,2] > 0
        if not good.all():
            return None
        return (sums[:,:2]/sums[:,2,None]).mean(0)

    original = [estimate(s, list(range(len(banks[s])))) for s in seeds]
    complete = all(v is not None for v in original)
    point = np.mean(original, axis=0) if complete else [None, None]
    plans = cluster_indices(banks, draws=draws, seed=seed, crossed=crossed)
    index = {s:{b:i for i,b in enumerate(banks[s])} for s in seeds}
    samples=[]
    for plan in plans:
        values=[estimate(s,[index[s][b] for b in blocks]) for s,blocks in plan]
        if all(v is not None for v in values):
            samples.append(np.mean(values,axis=0))
    available = complete and len(seeds) >= 2 and len(samples) == draws
    result = dict(seed_count=len(seeds), block_counts={s:len(banks[s]) for s in seeds},
                  valid_draws=len(samples), requested_draws=draws, inference_available=available,
                  complete_strata=complete, approximate=True, crossed=crossed,
                  seed_estimates={s:None if v is None else v.tolist() for s,v in zip(seeds,original)})
    for i,name in enumerate(['width_minus_chance','width_minus_support']):
        tail = (1-confidence)/2
        result[name] = dict(estimate=None if point[i] is None else float(point[i]),
            interval=np.quantile(np.asarray(samples)[:,i],[tail,1-tail]).tolist() if available else None,
            family_interval=np.quantile(np.asarray(samples)[:,i],[tail/family_size,1-tail/family_size]).tolist()
            if available else None)
    return result


def actor_update_mean(values, update_mask):
    values, mask = vector(values), np.asarray(update_mask)
    require(mask.dtype == bool and mask.shape == values.shape and mask.any(), 'Explicit actor-update mask required.')
    return float(values[mask].mean())


def ranking_summary(records, expected_strata, minimum_valid=30, minimum_harmful=30):
    """One seed and continuation; expose available-strata and complete estimands.

Reset-time copies may appear in both collector strata, preserving the declared
balanced estimand, but count once in unique-state denominators and pooled AUC.
"""
    require(records and len({(r['seed'], r['continuation']) for r in records}) == 1,
            'Summarize exactly one training seed and continuation.')
    seen, unique = set(), {}
    for r in records:
        require(r['stratum'] in expected_strata, 'Unknown stratum.')
        key = (r['block'], r['state_id'])
        require((key, r['stratum']) not in seen, 'Duplicate row within a stratum.')
        seen.add((key, r['stratum']))
        if key in unique:
            require(all(r[k] == unique[key][k] for k in ['labels','losses','scores']),
                    'Shared physical state has conflicting scores/outcomes.')
        else:
            unique[key] = r
    strata = {}
    for s in expected_strata:
        group = [r for r in records if r['stratum'] == s]
        valid = [r for r in group if r['width_auc'] is not None]
        strata[s] = dict(captured=len(group), valid=len(valid),
            width_auc=float(np.mean([r['width_auc'] for r in valid])) if valid else None,
            support_auc=float(np.mean([r['support_auc'] for r in valid])) if valid else None)
    available = [v for v in strata.values() if v['valid']]
    valid_count = sum(r['width_auc'] is not None for r in unique.values())
    harmful = sum(r['harmful'] for r in unique.values())
    labels = [x for r in unique.values() for x in r['labels']]
    scores = [x for r in unique.values() for x in r['scores']['width']]
    value = float(np.mean([r['width_auc'] for r in available])) if available else None
    return dict(strata=strata, unique_captured_states=len(unique), unique_valid_states=valid_count,
        harmful_alternatives=harmful, alternatives=len(labels),
        missing_strata=[s for s,v in strata.items() if not v['valid']],
        within_available_strata=value, within_all_strata=value if len(available)==len(expected_strata) else None,
        pooled_auc=auc(labels,scores), sparse=valid_count < minimum_valid or harmful < minimum_harmful,
        broad_claim_allowed=valid_count >= minimum_valid and harmful >= minimum_harmful
                            and len(available)==len(expected_strata))


def policy_summary(steps, host_curves, bca_curves, host_final, bca_final):
    require(np.array_equal(steps, np.arange(5000, 1000001, 5000)), 'Full common 5k-to-1M schedule required.')
    h,b,hf,bf = [np.asarray(x,dtype=float) for x in [host_curves,bca_curves,host_final,bca_final]]
    require(h.ndim == b.ndim == hf.ndim == bf.ndim == 2 and h.shape == b.shape and h.shape[1] == 200
            and hf.shape == bf.shape == (h.shape[0],20) and h.shape[0] > 0
            and all(np.isfinite(x).all() for x in [h,b,hf,bf]), 'Unpaired or incomplete policy banks.')
    return dict(host_curve=h.mean(1), bca_curve=b.mean(1), curve_difference=(b-h).mean(1),
                host_final=hf.mean(1), bca_final=bf.mean(1), final_difference=(bf-hf).mean(1))
