# Closed-form / Monte Carlo arithmetic on made-up numbers only (no ThrusterChain data).
# Pass probabilities of DESIGN.md marks under stated scenarios. Values in points taken from records:
# V_S ~ 106 (perfect-switch NET 106.1 = V_S - 0 ... ROUND1.md:13), no-stall clip cost 61.8 (fallback.md:172-174).
import numpy as np
from scipy import stats

rng = np.random.default_rng(12345)
T5 = stats.t.ppf(0.95, 4)

def passes(d, bar):
    m, s = d.mean(), d.std(ddof=1)
    L = m - T5 * s / np.sqrt(len(d)) if s > 0 else m
    return (m >= bar) and (L > 0)

# 1. A-R1 second part: NET(A-BCA) - NET(A-RAND) with ONE realized coin per seed-world (stream [20261019,3,k,w,s]),
#    A-BCA perfect (on in S, off in N), on-rate 0.5. Per seed diff = V_S*(1-c_S) + C_N*c_N.
for vs_sd in (0.0, 10.0):
    fails = 0; n = 200000
    for _ in range(n):
        VS = 106.0 + vs_sd * rng.standard_normal(5)
        CN = 61.8 + 0.3 * vs_sd * rng.standard_normal(5)
        cS = rng.random(5) < 0.5; cN = rng.random(5) < 0.5
        d = VS * (~cS) + CN * cN
        fails += not passes(d, 10.0)
    print(f"1. A-R1 vs realized A-RAND, perfect switch, V_S seed sd {vs_sd}: P(fail) = {fails / n:.3f}")
# with the expectation over coins instead: d = 0.5*V_S + 0.5*C_N, deterministic up to seed spread
print("   expectation form: per-seed diff = 0.5 V_S + 0.5 C_N ~ 83.9, passes with probability ~1")

# 2. A-V1 at a boundary null (true FSR exactly 0.05), R = 1000, CP one-sided 95% upper <= 0.073, 'every seed-world'
R = 1000
xmax = max(x for x in range(200) if stats.beta.ppf(0.95, x + 1, R - x) <= 0.073)
for p in (0.04, 0.05):
    one = stats.binom.cdf(xmax, R, p)
    print(f"2. A-V1/B-V1: pass iff errors <= {xmax}; true {p}: one {one:.3f}, all 5 {one**5:.3f}, all 10 {one**10:.3f}")

# 3. B-R1 release: per seed diff = C_N * 1[certified in N]; pass iff mean >= 10 and L > 0.
#    certification prob per seed-world from exact binomial (alpha_B 0.1, 95%) at n tries with true harm rate pi.
def fmax(n):
    ok = [f for f in range(n + 1) if stats.binom.cdf(f, n, 0.1) <= 0.05]
    return max(ok) if ok else -1
for n in (36, 50, 70, 100):
    f = fmax(n)
    row = []
    for pi in (0.01, 0.02, 0.03, 0.05, 0.08):
        c = stats.binom.cdf(f, n, pi) if f >= 0 else 0.0
        # pass needs >= 3 of 5 seeds certified (values 61.8 or 0): check exactly
        pk = 0.0
        for k in range(6):
            d = np.array([61.8] * k + [0.0] * (5 - k))
            if passes(d, 10.0):
                pk += stats.binom.pmf(k, 5, c)
        row.append(f"pi {pi}: cert {c:.2f}, P(B-R1 release passes) {pk:.2f}")
    print(f"3. n={n} (f<={f}): " + "; ".join(row))

# 4. C-R2 non-inferiority L >= -5 at true difference 0, 5 seeds, seed sd s
for s in (1.9, 4.0, 5.5, 8.8):
    se = s / np.sqrt(5)
    # P(mean - T5*sd_hat/sqrt5 >= -5): simulate
    d = rng.normal(0, s, (200000, 5))
    m = d.mean(1); sh = d.std(1, ddof=1)
    p = np.mean(m - T5 * sh / np.sqrt(5) >= -5)
    print(f"4. C-R2 at true diff 0, seed sd {s}: P(pass) = {p:.3f}")

# 5. Holm first-step threshold for a family of 4 and of 2, 5 seeds: t critical values
for k in (2, 4):
    print(f"5. Holm family {k}: first step one-sided level {0.05 / k:.4f}, t crit {stats.t.ppf(1 - 0.05 / k, 4):.3f} (vs {T5:.3f})")
for s in (1.9, 5.5, 8.8):
    se = s / np.sqrt(5)
    print(f"   sd {s}: mean needed for Holm-4 first step {stats.t.ppf(1 - 0.0125, 4) * se:.1f}, unadjusted {T5 * se:.1f}")

# 6. noncentral t check of the design's 1.36 sd (one-sided 5%, 80%, n=5)
from scipy.optimize import brentq
f = lambda d: 1 - stats.nct.cdf(T5, 4, d * np.sqrt(5)) - 0.8
print(f"6. min detectable effect at n=5: {brentq(f, 0.1, 5):.3f} sd")
