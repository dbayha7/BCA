"""Closed-form arithmetic for the measurement design (no data, no simulator, no learner).

1. Evidence floor of an equal-weight tail certificate (WBCP with uniform weights = Wilks = exact binomial):
   certify 'P(failure) <= alpha' with confidence beta iff P(Bin(n, alpha) <= f) <= 1 - beta (f failures in n tries).
2. Group-sequential looks at n = 10, 30, 100 with Bonferroni spending (1 - beta) / 3 per look.
3. Anytime-valid beta-binomial mixture e-process for H0: p >= alpha (zero-failure boundary and allowed failures).
4. Weighted floor: n_min / (n_eff / n) for the shift ratios recorded on the live page (0.73 moderate, 0.34 strong).
5. Seeds needed / minimum detectable paired effect with Student t (one-sided 5%, 80% power), given a seed sd.
6. Coverage precision with an episode design effect.
"""
import math
import numpy as np
from scipy import stats, special

BETA = 0.95
print("== 1. fixed-n exact binomial floor (beta 0.95)")
for alpha in (0.5, 0.25, 0.2, 0.1, 0.05):
    nmin = math.ceil(math.log(1 - BETA) / math.log(1 - alpha))
    row = [f"alpha={alpha}: n_min={nmin}"]
    for n in (10, 15, 29, 30, 59, 100, 300):
        fs = [f for f in range(n + 1) if stats.binom.cdf(f, n, alpha) <= 1 - BETA]
        row.append(f"n={n}: f<={max(fs) if fs else 'none'}")
    print("  " + "; ".join(row))

print("== 2. three looks (10, 30, 100), Bonferroni 0.05/3 per look")
for alpha in (0.5, 0.2, 0.1, 0.05):
    lvl = (1 - BETA) / 3
    nmin = math.ceil(math.log(lvl) / math.log(1 - alpha))
    row = [f"alpha={alpha}: n_min={nmin}"]
    for n in (10, 30, 100):
        fs = [f for f in range(n + 1) if stats.binom.cdf(f, n, alpha) <= lvl]
        row.append(f"n={n}: f<={max(fs) if fs else 'none'}")
    print("  " + "; ".join(row))

print("== 3. anytime-valid mixture e-process, H0: p >= alpha, reject when E >= 20")
def log_e(n, f, alpha):
    # Valid test supermartingale for H0: p >= alpha (p = failure probability): the mixture over q of the likelihood
    # ratio L(q) / L(alpha), with q uniform on [0, alpha]. For q < alpha and any null p >= alpha each factor has
    # expectation p q / alpha + (1 - p)(1 - q)/(1 - alpha) <= 1, so the mixture is a nonnegative supermartingale and
    # Ville's inequality gives P(sup E >= 20) <= 0.05 under every null p: valid at any stopping time.
    logden = (f * math.log(alpha) if f else 0.0) + ((n - f) * math.log(1 - alpha) if n - f else 0.0)
    lognum = special.betaln(f + 1, n - f + 1) + stats.beta.logcdf(alpha, f + 1, n - f + 1) - math.log(alpha)
    return lognum - logden
for alpha in (0.5, 0.2, 0.1, 0.05):
    n0 = next(n for n in range(1, 5000) if log_e(n, 0, alpha) >= math.log(20))
    allow = []
    for n in (10, 30, 100, 300):
        fs = [f for f in range(n + 1) if log_e(n, f, alpha) >= math.log(20)]
        allow.append(f"n={n}: f<={max(fs) if fs else 'none'}")
    print(f"  alpha={alpha} (prior uniform on [0, alpha]): zero-failure n={n0}; " + "; ".join(allow))

print("== 4. weighted floor (alpha 0.1, beta 0.95): 29 / (n_eff/n)")
for name, r in (("none", 1.0), ("moderate", 0.73), ("strong", 0.34), ("O10 strong", 0.35), ("O10 mild", 0.72)):
    print(f"  {name}: n_eff/n={r}: n_min ~ {math.ceil(29 / r)}")

print("== 5. paired seed tests: min detectable effect (one-sided 5%, power 80%) = delta s.t. noncentral t power 0.8")
def power(n, delta, sd, a=0.05):
    df = n - 1
    tc = stats.t.ppf(1 - a, df)
    nc = delta / (sd / math.sqrt(n))
    return 1 - stats.nct.cdf(tc, df, nc)
for sd in (1.0, 2.0, 4.0, 6.0, 10.0):
    mde = []
    for n in (5, 10, 15, 20):
        lo, hi = 0.0, 50 * sd
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if power(n, mid, sd) < 0.8 else (lo, mid)
        mde.append(f"n={n}: {hi:.2f}")
    print(f"  seed sd={sd}: " + "; ".join(mde))
for sd, delta in ((1.9, 1.0), (1.9, 2.0), (6.3, 5.0), (10.0, 5.0), (4.0, 3.0)):
    n = next(n for n in range(3, 400) if power(n, delta, sd) >= 0.8)
    print(f"  seeds needed: sd={sd}, delta={delta}: {n}")

print("== 6. coverage precision: SE of a pooled step coverage near p with E episodes of m steps, design effect")
for p in (0.9, 0.5):
    for rho in (0.0, 0.1, 0.3, 0.6):
        for E, m in ((200, 100), (1000, 100), (1000, 30)):
            deff = 1 + (m - 1) * rho
            se = math.sqrt(p * (1 - p) * deff / (E * m))
            print(f"  p={p} rho={rho} E={E} m={m}: deff={deff:.1f} SE={100*se:.2f} pp (95% half-width {196*se:.2f} pp)")

print("== 7. replicate evidence draws needed to show an error rate <= 7.3% when the true rate is 5% (CP upper 95%)")
for R in (200, 400, 800, 1000, 2000):
    k = int(round(0.05 * R))
    up = stats.beta.ppf(0.95, k + 1, R - k)
    print(f"  R={R}: observed {k} errors -> one-sided 95% upper {100*up:.2f}%")
