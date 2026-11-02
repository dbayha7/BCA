"""Pure decision arithmetic of G1 (G1_RULE.md). No model, simulator or file access; numpy only.

- The L1 band per anchor at the scored slot: -tol <= KC_j <= c rho_j + tol, on valid anchors (rho_j > 0).
- q*: the largest share k / n that a critic with per-anchor pass probability p_hat clears in every one of `seeds`
  screen seeds with probability >= `confidence`; the binomial tail is exact, in log space.
- The in-support resolution band: a quantile of |KC| at the matched in-support slot.
- The G1 label in its precedence.
"""
import math

import numpy as np

TOL, C = 0.15, 0.3
SHARE_POOLED, SHARE_EACH = 0.70, 0.60
SCREEN_SEEDS, Q_CONFIDENCE = 3, 0.90
BAND_QUANTILE = 0.90
ORDER = ("INCOMPLETE", "INCONCLUSIVE", "NOT REPRESENTABLE", "PASS")


def l1_band(rho, tol=TOL, c=C):
    """(lower, upper) per anchor: -tol and c rho + tol."""
    rho = np.asarray(rho, np.float64)
    return np.full(rho.shape, -tol), c * rho + tol


def l1(kc_s10, rho, tol=TOL, c=C):
    """Per-anchor L1 on valid anchors (rho > 0): flags, share, and the median check in the same band."""
    kc, rho = np.asarray(kc_s10, np.float64), np.asarray(rho, np.float64)
    valid = np.isfinite(rho) & (rho > 0) & np.isfinite(kc)
    lower, upper = l1_band(rho, tol, c)
    flags = valid & (kc >= lower) & (kc <= upper)
    n_valid = int(valid.sum())
    share = float(flags.sum() / n_valid) if n_valid else 0.0
    med_kc = float(np.median(kc[valid])) if n_valid else None
    med_rho = float(np.median(rho[valid])) if n_valid else None
    med_ok = bool(n_valid and -tol <= med_kc <= c * med_rho + tol)
    nonpos = np.isfinite(rho) & (rho <= 0) & np.isfinite(kc)
    return {"valid": valid, "flags": flags, "n_valid": n_valid, "n_pass": int(flags.sum()), "share": share,
            "median_kc": med_kc, "median_rho": med_rho, "median_upper": None if med_rho is None else c * med_rho + tol,
            "median_ok": med_ok,
            "under_capture_share": float(np.mean(kc[valid] < -tol)) if n_valid else None,
            "over_capture_share": float(np.mean(kc[valid] > upper[valid])) if n_valid else None,
            "nonpositive_rho": {"count": int(nonpos.sum()),
                                "plain_resolution_share": float(np.mean(np.abs(kc[nonpos]) <= tol)) if nonpos.any() else None}}


def _log_binom_pmf(n, k, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def binomial_tail(n, p, k):
    """P(Binomial(n, p) >= k), exact in log space."""
    n, k = int(n), int(k)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [_log_binom_pmf(n, i, float(p)) for i in range(k, n + 1)]
    top = max(logs)
    if top == -math.inf:
        return 0.0
    return float(min(1.0, math.exp(top) * sum(math.exp(x - top) for x in logs)))


def q_star(p_hat, n_valid, seeds=SCREEN_SEEDS, confidence=Q_CONFIDENCE):
    """The largest k / n_valid with binomial_tail(n_valid, p_hat, k) ** seeds >= confidence."""
    n = int(n_valid)
    if n <= 0:
        return None
    best = 0
    for k in range(0, n + 1):
        if binomial_tail(n, float(p_hat), k) ** int(seeds) >= confidence:
            best = k
        else:
            break
    return {"q_star": best / n, "k": best, "n_valid": n, "p_hat": float(p_hat), "seeds": int(seeds), "confidence": confidence}


def resolution_band(kc_in_support, quantile=BAND_QUANTILE):
    """The quantile ("higher") of |KC| at the matched in-support slot over anchors and seeds."""
    values = np.abs(np.asarray(kc_in_support, np.float64).ravel())
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {"band": float(np.quantile(values, quantile, method="higher")), "quantile": quantile, "n": int(values.size),
            "median": float(np.median(values))}


def labels(*, i0_ok, f0_ok, median_ok_by_seed, share_by_seed, share_pooled):
    """(headline, all labels). Every computed label is reported; the headline follows ORDER."""
    out = []
    if not i0_ok:
        out.append("INCONCLUSIVE (I0: the instrument check failed)")
    if not f0_ok:
        out.append("INCONCLUSIVE (F0: a non-finite loss or value)")
    bad = [f"seed {s}: median outside the band" for s, ok in median_ok_by_seed.items() if not ok]
    bad += [f"seed {s}: share {v:.3f} < {SHARE_EACH}" for s, v in share_by_seed.items() if v < SHARE_EACH]
    if share_pooled < SHARE_POOLED:
        bad.append(f"pooled share {share_pooled:.3f} < {SHARE_POOLED}")
    if bad:
        out.append("NOT REPRESENTABLE (" + "; ".join(bad) + ")")
    all_labels = out or ["PASS"]
    headline = min(all_labels, key=lambda s: ORDER.index(s.split(" (")[0]))
    return headline.split(" (")[0], all_labels
