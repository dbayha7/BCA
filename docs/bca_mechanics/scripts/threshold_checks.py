"""Mechanical checks of BCA's threshold, band and dose on synthetic scores (modules M4 to M7 of MODULES.md).

Nothing here reads study data and nothing is trained. The threshold is computed by the repository's own
calibration/wbcp.py (calibrate), so set PYTHONPATH to the repository root:
    PYTHONPATH=<root> python docs/bca_mechanics/scripts/threshold_checks.py
Needs NumPy only.

Checks (each prints PASS or FAIL):
  1. refusal by size: the share of random score sets whose 1,000-draw threshold is finite, per size n, against the
     binomial prediction P(at least 950 of 1,000 draws cross), each draw crossing with probability 1 - 0.9^n
     (calibration/reference.certifiable's reasoning; the exact, infinite-draw form is finite only for n >= 29).
  2. weights of 1 and a test mass of 1 reproduce the unweighted threshold bit for bit (the weighting test's mark W0).
  3. the exact form: with equal weights, BCA's 1,000-draw threshold decides like the exact binomial test (the toy
     notebooks' test 4a); the share of agreeing decisions and of equal threshold values is printed.
  4. the threshold does not fall as the test mass grows, on a grid t = 1.01^k with the same random stream
     (the per-step table's assumption; coverage.threshold_table checks the same thing on study data).
  5. a common factor on the weights and the test mass changes nothing.
  6. the band covers about 90% of fresh rows from the same law (module M6's "working as intended").
  7. the dose 1 + 0.5 U / (U + u) lies in [1, 1.5), does not fall as U grows, and gives 1 at U = +inf under the
     standard refusal rule and 1.5 under the most-caution rule (module M7; the formula of calibration/dose.py).
"""
import math
import sys
from fractions import Fraction

import numpy as np

try:
    from calibration import wbcp
except ImportError as error:
    sys.exit(f"set PYTHONPATH to the repository root (calibration/wbcp.py): {error}")

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
SEED = 20261020


def verdict(ok):
    return "PASS" if ok else "FAIL"


def exact_threshold(scores):
    """BCA's threshold with infinitely many draws: the largest x with P(Binomial(n, 0.1) <= x) <= 0.05 scores may lie
    above it (bca_channel_toy.ipynb, cell 8)."""
    s = np.sort(np.asarray(scores, np.float64))
    n = s.size
    cdf, best = 0.0, -1
    for x in range(n + 1):
        cdf += math.exp(math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
                        + x * math.log(ALPHA) + (n - x) * math.log1p(-ALPHA))
        if cdf > 1 - BETA:
            break
        best = x
    hpd = s[n - best - 1] if best >= 0 else math.inf
    lam_hat = s[math.ceil((1 - Fraction(str(ALPHA))) * n) - 1]
    return max(lam_hat, hpd)


def finite_probability(n):
    """P(at least 950 of 1,000 draws cross) when each draw crosses with probability 1 - 0.9^n: the chance that the
    1,000-draw threshold is finite for n scores (calibration/reference.certifiable's own reasoning). The exact form
    (infinitely many draws) is finite iff 0.9^n <= 0.05, that is n >= 29."""
    p = 1.0 - (1.0 - ALPHA) ** n
    # P(Binomial(1000, p) >= 950) by direct summation of the log-pmf
    total = 0.0
    for k in range(950, DRAWS + 1):
        total += math.exp(math.lgamma(DRAWS + 1) - math.lgamma(k + 1) - math.lgamma(DRAWS - k + 1)
                          + k * math.log(p) + (DRAWS - k) * math.log1p(-p))
    return total


def check_1_refusal_by_size():
    rng = np.random.default_rng([SEED, 1])
    ok = True
    print("check 1: share of 200 random score sets with a finite 1,000-draw threshold, by size n, against the binomial "
          "prediction P(at least 950 of 1,000 draws cross); the exact form is finite only for n >= 29")
    for n in (10, 20, 24, 26, 28, 29, 30, 33, 36, 45, 100):
        finite = 0
        for _ in range(200):
            scores = np.abs(rng.standard_normal(n))
            r = wbcp.calibrate(scores, np.random.default_rng(rng.integers(2 ** 31)), alpha=ALPHA, beta=BETA, draws=DRAWS)
            finite += math.isfinite(r.threshold)
        share = finite / 200
        predicted = finite_probability(n)
        exact = "finite" if (1.0 - ALPHA) ** n <= 1.0 - BETA else "+inf"
        print(f"   n = {n:3d}: finite in {share:5.1%} of sets; predicted {predicted:5.1%}; exact form: {exact}")
        if abs(share - predicted) > 0.12:          # 200 sets: the share's standard error is at most 0.035
            ok = False
        if n >= 36 and share < 1.0:
            ok = False
    print("  ", verdict(ok))
    return ok


def check_2_unit():
    rng = np.random.default_rng([SEED, 2])
    scores = np.abs(rng.standard_normal(843))
    a = wbcp.calibrate(scores, np.random.default_rng(7), alpha=ALPHA, beta=BETA, draws=DRAWS)
    b = wbcp.calibrate(scores, np.random.default_rng(7), np.ones(843), 1.0, alpha=ALPHA, beta=BETA, draws=DRAWS)
    same = (a.threshold == b.threshold and a.lambda_hat == b.lambda_hat and a.lambda_hpd == b.lambda_hpd
            and np.array_equal(a.crossings, b.crossings))
    print(f"check 2: weights of 1 reproduce the unweighted threshold bit for bit: {same} "
          f"(R = {a.threshold:.6f}, lambda_hat = {a.lambda_hat:.6f}, lambda_hpd = {a.lambda_hpd:.6f})")
    print("  ", verdict(same))
    return same


def check_3_exact_form():
    """The toy notebooks' test 4(a): the 1,000-draw threshold and the exact binomial form decide alike. A decision is
    "the threshold is at most a claimed value"; the claimed value is drawn uniformly between the smallest and the
    largest score. Also printed, not judged: how often the two threshold VALUES are equal, and the largest relative
    difference. Monte Carlo noise moves the credible quantile by a score or two; with 29 to 45 scores that quantile is
    the single largest score, so the value can move by tens of percent between random streams, while at the study's
    bank size (about 840 rows) it moves by about 2% at most."""
    rng = np.random.default_rng([SEED, 3])
    ok = True
    print("check 3: decisions of the 1,000-draw threshold against the exact binomial form, 300 random score sets per n")
    for n in (36, 45, 100, 843):
        agree = equal = 0
        rel = []
        for _ in range(300):
            scores = np.abs(rng.standard_normal(n)) * rng.uniform(0.5, 2.0)
            r = wbcp.calibrate(scores, np.random.default_rng(rng.integers(2 ** 31)), alpha=ALPHA, beta=BETA, draws=DRAWS)
            ex = exact_threshold(scores)
            claimed = rng.uniform(scores.min(), scores.max())
            agree += (r.threshold <= claimed) == (ex <= claimed)
            equal += r.threshold == ex
            rel.append(abs(r.threshold - ex) / ex)
        share, same, worst = agree / 300, equal / 300, max(rel)
        print(f"   n = {n:4d}: decisions agree in {share:5.1%}; threshold values equal in {same:5.1%}; "
              f"largest relative difference {worst:.3f}")
        # the pass rule is the toy notebooks' (decisions agree); the value differences are reported, not judged: for
        # 29 to 45 scores the credible threshold is the single largest score, which moves between random streams
        if share < 0.90 or (n >= 800 and share < 0.99):
            ok = False
    print("  ", verdict(ok))
    return ok


def check_4_monotone_in_test_mass():
    rng = np.random.default_rng([SEED, 4])
    scores = np.abs(rng.standard_normal(843))
    c = rng.uniform(0.1, 1.1, 843)
    weights = 0.0907 * np.exp(4 * c)       # the strong tilt's exact weight shape (experiments/shift_iw/world.ratio)
    grid = 1.01 ** np.arange(-201, 201)    # 0.135 .. 7.39, the strong tilt's range
    R = np.array([wbcp.calibrate(scores, np.random.default_rng(11), weights, float(t), alpha=ALPHA, beta=BETA,
                                 draws=DRAWS).threshold for t in grid])
    monotone = bool(np.all(R[1:] >= R[:-1]))
    finite = np.isfinite(R)
    first_inf = grid[np.argmax(~finite)] if (~finite).any() else None
    print(f"check 4: R(t) never falls along the grid: {monotone}; R at t = 1: "
          f"{R[np.argmin(np.abs(grid - 1.0))]:.4f}; +inf from t = {first_inf}")
    print("  ", verdict(monotone))
    return monotone


def check_5_common_factor():
    rng = np.random.default_rng([SEED, 5])
    scores = np.abs(rng.standard_normal(500))
    w = rng.uniform(0.2, 5.0, 500)
    a = wbcp.calibrate(scores, np.random.default_rng(3), w, 2.0, alpha=ALPHA, beta=BETA, draws=DRAWS)
    b = wbcp.calibrate(scores, np.random.default_rng(3), 1000.0 * w, 2000.0, alpha=ALPHA, beta=BETA, draws=DRAWS)
    same = a.threshold == b.threshold and np.array_equal(a.crossings, b.crossings)
    print(f"check 5: a common factor of 1000 on the weights and the test mass changes nothing: {same}")
    print("  ", verdict(same))
    return same


def check_6_coverage():
    rng = np.random.default_rng([SEED, 6])
    u, eta = 0.15, rng.uniform(0.5, 1.5, 843)
    residual = rng.standard_normal(843) * 0.12 * eta
    scores = np.abs(residual) / (np.maximum(eta, 1e-6) * u)
    R = wbcp.calibrate(scores, np.random.default_rng(5), alpha=ALPHA, beta=BETA, draws=DRAWS).threshold
    eta_new = rng.uniform(0.5, 1.5, 20000)
    fresh = rng.standard_normal(20000) * 0.12 * eta_new
    U = R * u * np.maximum(eta_new, 1e-6)
    covered = np.mean(np.abs(fresh) <= U)
    ok = 0.89 <= covered <= 0.94
    print(f"check 6: R = {R:.4f} on 843 synthetic held-out rows; the band q +- R u eta covers {covered:.3f} of 20,000 "
          f"fresh rows from the same law (target 0.90; the study's bar [0.89, 0.94])")
    print("  ", verdict(ok))
    return ok


def check_7_dose():
    u = 0.15
    U = np.concatenate([np.linspace(0.0, 5.0, 2001), [np.inf]])
    finite = np.isfinite(U)
    with np.errstate(invalid="ignore"):
        dose = np.where(finite, 1.0 + 0.5 * U / (U + u), 1.0)        # standard rule: a refused row gets 1
    dose_most = np.where(finite, dose, 1.5)                            # most caution: a refused row gets 1.5
    in_range = bool(np.all(dose >= 1.0) and np.all(dose < 1.5))
    monotone = bool(np.all(np.diff(dose[finite]) >= 0))
    print(f"check 7: dose in [1, 1.5): {in_range}; non-decreasing in U: {monotone}; at U = u: {1 + 0.5 * 0.5:.3f}; "
          f"at U = 10u: {1 + 0.5 * 10 / 11:.3f}; at U = +inf: {dose[-1]:.1f} (standard) / {dose_most[-1]:.1f} (most caution)")
    ok = in_range and monotone and dose[-1] == 1.0 and dose_most[-1] == 1.5
    print("  ", verdict(ok))
    return ok


def main():
    print(f"numpy {np.__version__}; calibration/wbcp.py from {wbcp.__file__}")
    results = [check_1_refusal_by_size(), check_2_unit(), check_3_exact_form(), check_4_monotone_in_test_mass(),
               check_5_common_factor(), check_6_coverage(), check_7_dose()]
    print()
    print("all checks:", verdict(all(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
