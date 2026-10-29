# Closed-form arithmetic only (no data). Pass probabilities of the design's replicate-rate marks.
import math


def logpmf(i, n, p):
    return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.exp(logpmf(i, n, p)) for i in range(0, k + 1)))


def cp_upper(x, n, conf=0.95):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if binom_cdf(x, n, mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


R = 1000
x = 40
while cp_upper(x + 1, R) <= 0.073:
    x += 1
print("A-V1/B-V1: R=1000, largest error count with CP upper <= 0.073:", x, "CP upper", round(cp_upper(x, R), 4))
for p in (0.03, 0.04, 0.045, 0.05):
    pp = binom_cdf(x, R, p)
    print(f"  true error rate {p}: P(pass one) = {pp:.3f}; all of 5 = {pp**5:.3f}; all of 10 = {pp**10:.3f}")
for n in (5000, 10000):
    f = int(0.075 * n)
    while 1 - cp_upper(f, n) < 0.927:
        f -= 1
    print(f"A-V2 pooled n={n}: pass iff failures <= {f} (observed coverage >= {1 - f / n:.4f})")
    for c in (0.93, 0.935, 0.94, 0.95):
        print(f"  true coverage {c}: P(pass pooled) = {binom_cdf(f, n, 1 - c):.3f}")
for n in (36, 50, 70, 100):
    f = max([j for j in range(0, n) if binom_cdf(j, n, 0.1) <= 0.05], default=None)
    print(f"exact binomial n={n}: certify iff harm events <= {f}; size at pi=0.1: {binom_cdf(f, n, 0.1):.4f}")
    print("   P(certify | pi):", ", ".join(f"{p}: {binom_cdf(f, n, p):.2f}" for p in (0.01, 0.02, 0.03, 0.05, 0.08)))
for n in (10, 30, 100):
    f = [j for j in range(0, n) if binom_cdf(j, n, 0.1) <= 0.05 / 3]
    print(f"look n={n}: certifiable with harm events <= {max(f) if f else 'none'}")
for rs in (0.2, 0.3, 0.33, 0.4, 0.5):
    print(f"RS={rs}: promise COR>=0.9 implies only CAC >= {(0.9 - rs) / (1 - rs):.3f}")
