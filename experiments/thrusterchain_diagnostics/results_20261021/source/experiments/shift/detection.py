"""Statistics and the detection measures of the ThrusterChain shift test (pure NumPy and SciPy; nothing is trained).

Statistics used by every stage:
  t_lower / t_upper    one-sided 95% bound of a mean over seeds (Student t with n - 1 degrees of freedom; t(4) = 2.132
                       for 5 seeds); half_width is the two-sided 95% half-width.
  p_greater            one-sided t-test p-value of "mean > null".
  holm                 Holm's step-down adjustment of a family of p-values.
  block_bootstrap      percentile values of a statistic over resampled blocks (episodes), as in the D4RL protocol.
  ratio_of_medians     the error ratio of the harm checks: median |x| / median |y|; pooled over seeds as the geometric
                       mean of the per-seed ratios (the D4RL protocol's rule).
  auc, spearman, partial_spearman (any number of controls: the correlation of rank residuals).

Detection (BCA's doses at its final snapshot against the exact over-preference D_i of its own critic and actor):
  D1 ranking         AUC of the dose for the top 10% of rows by D_i against the rest (ties of D_i broken by a fixed
                     seeded order).
  D2 shift-specific  C = mean dose on full-throttle-region rows / mean dose on the other rows; the gapped run's C over the
                     full-log run's C of the same seed.
  D3 beyond target noise  the partial Spearman correlation of dose with D_i, controlling for the noise map and
                     full-throttle-region membership.
  placement efficiency  of a dose multiset placed on rows: 0 for even doses (or a random placement on average), 1 for
                     the best arrangement of the same doses by D_i, -1 for the worst.
  comparators        the same dose multiset placed by another signal (largest dose on the largest signal): the noise
                     map, critic disagreement |Q1 - Q2|, data thinness (distance to the 10th nearest training row),
                     critic size |Q1| and the raw TD residual; each scored like BCA (AUC and placement efficiency).
  state maps         rows grouped into ten equal-count groups by m(s) (the removed share at the row's state): where
                     data are missing, where the critic is wrong and where caution goes.
"""

import math

import numpy as np
from scipy import stats

TOP_SHARE = 0.10


# ------------------------------------------------------------------------------------------------ basic statistics

def _finite(values):
    return np.asarray([v for v in values if v is not None and np.isfinite(v)], np.float64)


def mean(values):
    v = _finite(values)
    return float(v.mean()) if len(v) else None


def t_lower(values, level=0.95):
    """One-sided lower bound of the mean: mean - t_{level, n-1} sd / sqrt(n) (None for fewer than 2 values)."""
    v = _finite(values)
    if len(v) < 2:
        return None
    return float(v.mean() - stats.t.ppf(level, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def t_upper(values, level=0.95):
    v = _finite(values)
    if len(v) < 2:
        return None
    return float(v.mean() + stats.t.ppf(level, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def half_width(values, level=0.95):
    """Two-sided half-width of the t interval of the mean."""
    v = _finite(values)
    if len(v) < 2:
        return None
    return float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def p_greater(values, null=0.0):
    """One-sided t-test p-value of mean > null (1.0 when undefined)."""
    v = _finite(values)
    if len(v) < 2:
        return 1.0
    sd = v.std(ddof=1)
    if sd == 0:
        return 0.0 if v.mean() > null else 1.0
    t = (v.mean() - null) / (sd / math.sqrt(len(v)))
    return float(stats.t.sf(t, len(v) - 1))


def summary(values, null=0.0):
    """n, mean, sd, one-sided bounds, half-width, p of mean > null, count positive."""
    v = _finite(values)
    return dict(n=int(len(v)), values=v.tolist(), mean=mean(v), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                lower=t_lower(v), upper=t_upper(v), half_width=half_width(v), p=p_greater(v, null),
                positive=int((v > null).sum()))


def holm(pvalues):
    """Holm-adjusted p-values for {name: p} (a missing p counts as 1)."""
    keys = sorted(pvalues, key=lambda k: 1.0 if pvalues[k] is None else pvalues[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        p = 1.0 if pvalues[k] is None else float(pvalues[k])
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def need(share, n):
    """'In at least 4 of 5 seeds' as a share: ceil(share x n) seeds."""
    return int(math.ceil(share * n - 1e-9))


def block_bootstrap(blocks, statistic, replicates, seed):
    """statistic(index array) over block-resampled index sets (blocks drawn with replacement); None values dropped."""
    _, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    members = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    g = np.random.default_rng(seed)
    values = []
    for _ in range(int(replicates)):
        pick = g.integers(0, len(members), len(members))
        v = statistic(np.concatenate([members[i] for i in pick]))
        if v is not None and np.isfinite(v):
            values.append(v)
    return np.asarray(values, np.float64)


def interval(values, level=0.95):
    v = _finite(values)
    if not len(v):
        return None, None
    tail = 100 * (1 - level) / 2
    return float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail))


def median_abs(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x))) if len(x) else None


def safe_ratio(a, b):
    return float(a / b) if a is not None and b not in (None, 0.0) else None


def geometric_mean(values):
    v = _finite(values)
    v = v[v > 0]
    return float(np.exp(np.mean(np.log(v)))) if len(v) else None


def pooled_ratio(num, den, blocks, replicates, seed):
    """The harm checks' error ratio: per seed median |num| / median |den| (num, den: {seed: per-row errors on the
    same frozen rows}); pooled = their geometric mean; 95% interval by an episode-block bootstrap, stratified by seed
    (each replicate redraws every seed's episodes with replacement; blocks: {seed: episode of each row}). Returns the
    per-seed ratios, the pooled ratio and its interval."""
    seeds = sorted(num)
    per = {str(s): safe_ratio(median_abs(num[s]), median_abs(den[s])) for s in seeds}
    pooled = geometric_mean(list(per.values()))
    if not seeds:
        return dict(per_seed=per, pooled=None, interval95=[None, None], replicates=0)
    members = {}
    for s in seeds:
        _, inv = np.unique(np.asarray(blocks[s]), return_inverse=True)
        members[s] = [np.flatnonzero(inv == b) for b in range(inv.max() + 1)] if len(inv) else []
    g = np.random.default_rng(seed)
    boot = []
    for _ in range(int(replicates)):
        out = []
        for s in seeds:
            mem = members[s]
            if not mem:
                out.append(None)
                continue
            idx = np.concatenate([mem[i] for i in g.integers(0, len(mem), len(mem))])
            out.append(safe_ratio(median_abs(np.asarray(num[s])[idx]), median_abs(np.asarray(den[s])[idx])))
        v = geometric_mean(out) if all(o is not None for o in out) else None
        if v is not None:
            boot.append(v)
    lo, hi = interval(boot)
    return dict(per_seed=per, pooled=pooled, interval95=[lo, hi], replicates=int(len(boot)))


# ------------------------------------------------------------------------------------------------ ranks

def auc(score, positive):
    """P(score of a positive row > score of a negative row), ties counting one half; None if a class is empty."""
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    keep = np.isfinite(score)
    score, positive = score[keep], positive[keep]
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = stats.rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return None
    return float(stats.spearmanr(x[keep], y[keep])[0])


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y controlling for every column of controls: the Pearson correlation of the
    residuals of rank(x) and rank(y) after least squares on [1, rank(control) ...]."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    Z = [np.asarray(c, np.float64) for c in controls]
    keep = np.isfinite(x) & np.isfinite(y)
    for z in Z:
        keep &= np.isfinite(z)
    if keep.sum() < len(Z) + 3:
        return None
    rx, ry = stats.rankdata(x[keep]), stats.rankdata(y[keep])
    design = np.column_stack([np.ones(int(keep.sum()))] + [stats.rankdata(z[keep]) for z in Z])
    res = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
    ex, ey = res(rx), res(ry)
    if np.std(ex) <= 1e-9 * np.std(rx) or np.std(ey) <= 1e-9 * np.std(ry):
        return 0.0  # x (or y) is explained by the controls: nothing is left to correlate
    return float(np.corrcoef(ex, ey)[0, 1])


def top_mask(D, share=TOP_SHARE, tie=None):
    """The ceil(share x n) rows with the largest D (ties broken by the order tie, a permutation)."""
    D = np.asarray(D, np.float64)
    n = len(D)
    tie = np.arange(n) if tie is None else np.asarray(tie)
    order = np.lexsort((tie, D))
    k = int(math.ceil(share * n))
    mask = np.zeros(n, bool)
    if k:
        mask[order[-k:]] = True
    return mask


def placement_efficiency(dose, D):
    """(P - E) / (B - E) when P >= E, (P - E) / (E - W) otherwise; P = sum dose x D as placed, E = the mean of a random
    placement, B and W the best and the worst arrangement of the same doses. 0 when the doses or D are constant."""
    d, D = np.asarray(dose, np.float64), np.asarray(D, np.float64)
    P = float(d @ D)
    E = float(d.mean() * D.sum())
    sd, sD = np.sort(d), np.sort(D)
    B, W = float(sd @ sD), float(sd @ sD[::-1])
    if P >= E:
        return 0.0 if B - E <= 1e-15 * max(1.0, abs(B)) else (P - E) / (B - E)
    return 0.0 if E - W <= 1e-15 * max(1.0, abs(W)) else (P - E) / (E - W)


def place_by(dose, signal, tie=None):
    """The dose multiset re-assigned by a signal's ranks: the largest dose on the largest signal."""
    d, s = np.asarray(dose, np.float64), np.asarray(signal, np.float64)
    tie = np.arange(len(s)) if tie is None else np.asarray(tie)
    order = np.lexsort((tie, s))
    out = np.empty(len(d))
    out[order] = np.sort(d)
    return out


# ------------------------------------------------------------------------------------------------ detection

def concentration(dose, region):
    """C = mean dose on full-throttle-region rows / mean dose on the other rows."""
    d, r = np.asarray(dose, np.float64), np.asarray(region, bool)
    if not r.any() or r.all():
        return None
    return safe_ratio(float(d[r].mean()), float(d[~r].mean()))


def detection_seed(dose, D, noise_map, region, tie, share=TOP_SHARE):
    """D1 (AUC), C (for D2), D3 (partial Spearman) and placement efficiency for one seed's BCA run."""
    top = top_mask(D, share, tie)
    return dict(d1_auc=auc(dose, top), concentration=concentration(dose, region),
                d3_partial_spearman=partial_spearman(dose, D, [noise_map, np.asarray(region, np.float64)]),
                placement_efficiency=placement_efficiency(dose, D), spearman_dose_D=spearman(dose, D),
                dose_quartiles=[float(q) for q in np.quantile(np.asarray(dose, np.float64), [0.25, 0.5, 0.75])],
                dose_interquartile_range=float(np.subtract(*np.quantile(np.asarray(dose, np.float64), [0.75, 0.25]))),
                dose_mean=float(np.mean(dose)), top_rows=int(top.sum()), rows=int(len(D)))


def comparator_scores(dose, D, signals, tie, share=TOP_SHARE):
    """For each signal {name: per-row values}: the same dose multiset placed by it, scored by D1's AUC and placement
    efficiency (and the signal's own AUC)."""
    top = top_mask(D, share, tie)
    out = {}
    for name, s in signals.items():
        placed = place_by(dose, s, tie)
        out[name] = dict(d1_auc=auc(placed, top), placement_efficiency=placement_efficiency(placed, D),
                         signal_auc=auc(s, top), spearman_signal_D=spearman(s, D))
    return out


def state_map(ms, columns, tie, groups=10):
    """Rows in `groups` equal-count groups by m(s) (ties broken by tie); per group the mean of m(s) and of every
    column ({name: per-row values})."""
    ms = np.asarray(ms, np.float64)
    order = np.lexsort((np.asarray(tie), ms))
    bins = np.empty(len(ms), np.int64)
    bins[order] = np.arange(len(ms)) * groups // max(1, len(ms))
    rows = []
    for g in range(groups):
        m = bins == g
        row = dict(group=g + 1, rows=int(m.sum()), removed_share_mean=float(ms[m].mean()) if m.any() else None,
                   removed_share_range=[float(ms[m].min()), float(ms[m].max())] if m.any() else None)
        for name, v in columns.items():
            v = np.asarray(v, np.float64)[m]
            v = v[np.isfinite(v)]
            row[name] = float(v.mean()) if len(v) else None
        rows.append(row)
    return rows


def detection_family(per_seed, marks):
    """D1-D3 over seeds, one Holm family (one-sided). per_seed: list of dicts with d1_auc, concentration_ratio (C of
    the gapped run over C of the full-log run) and d3_partial_spearman. A seed with a missing value counts as the null
    (AUC 0.5, ratio 1, correlation 0), so a failed seed never helps."""
    d1 = [0.5 if s.get("d1_auc") is None else s["d1_auc"] for s in per_seed]
    d2 = [0.0 if s.get("concentration_ratio") in (None, 0) else math.log(s["concentration_ratio"]) for s in per_seed]
    d3 = [0.0 if s.get("d3_partial_spearman") is None else s["d3_partial_spearman"] for s in per_seed]
    p = dict(D1=p_greater(d1, 0.5), D2=p_greater(d2, 0.0), D3=p_greater(d3, 0.0))
    adj = holm(p)
    ratio_mean = mean([math.exp(v) for v in d2])
    out = dict(
        D1=dict(values=d1, mean=mean(d1), lower=t_lower(d1), p=p["D1"], p_holm=adj["D1"],
                passed=bool(mean(d1) is not None and mean(d1) >= marks["d1_auc_mean_min"]
                            and (t_lower(d1) or -1) > marks["d1_lower_gt"] and adj["D1"] < marks["holm_alpha"])),
        D2=dict(log_ratios=d2, ratio_seed_mean=ratio_mean, log_lower=t_lower(d2), p=p["D2"], p_holm=adj["D2"],
                passed=bool(ratio_mean is not None and ratio_mean >= marks["d2_ratio_min"]
                            and (t_lower(d2) if t_lower(d2) is not None else -1) > marks["d2_log_lower_gt"]
                            and adj["D2"] < marks["holm_alpha"])),
        D3=dict(values=d3, mean=mean(d3), lower=t_lower(d3), p=p["D3"], p_holm=adj["D3"],
                passed=bool(mean(d3) is not None and mean(d3) > marks["d3_mean_gt"]
                            and (t_lower(d3) if t_lower(d3) is not None else -1) > marks["d3_lower_gt"]
                            and adj["D3"] < marks["holm_alpha"])))
    out["passed"] = bool(out["D1"]["passed"] and out["D2"]["passed"] and out["D3"]["passed"])
    return out
