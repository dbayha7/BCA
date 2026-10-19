# Closed-form arithmetic on made-up numbers only (no ThrusterChain data) for the revised DESIGN.md marks.
# Run: PYTHONDONTWRITEBYTECODE=1 smokevenv/bin/python -B revise_arith.py > revise_arith.out
import math

import numpy as np
from scipy import stats

rng = np.random.default_rng(20261019)


def cp_upper(x, n, level=0.95):
    return 1.0 if x >= n else float(stats.beta.ppf(level, x + 1, n - x))


def cp_lower(x, n, level=0.95):
    return 0.0 if x <= 0 else float(stats.beta.ppf(1 - level, x, n - x + 1))


def max_count(n, bar):
    """Largest error count x with one-sided 95% CP upper bound <= bar (binary search)."""
    lo, hi = -1, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cp_upper(mid, n) <= bar:
            lo = mid
        else:
            hi = mid
    return lo


print("1. Pooled validity mark: CP 95% upper bound of the pooled error count <= bar, plus a per-unit guard")
guard = int(stats.binom.ppf(0.999, 1000, 0.05))
print(f"   per-unit guard: 99.9th percentile of Binom(1000, 0.05) = {guard} errors in 1,000")
for K in (5, 10, 20):
    n = 1000 * K
    for bar in (0.065, 0.073):
        x = max_count(n, bar)
        row = []
        for p in (0.04, 0.05, 0.055, 0.06, 0.065, 0.07):
            pooled = stats.binom.cdf(x, n, p)
            unit = stats.binom.cdf(guard, 1000, p) ** K
            row.append(f"p={p}: pooled {pooled:.3f}, guard {unit:.3f}")
        print(f"   K={K} units (R*K={n}), bar {bar}: pass iff count <= {x}; " + "; ".join(row))

print("2. C-V3 per-bank check: 200 replicate banks x K seed-worlds; pooled share of banks with COR_b < 0.90")
for K in (10, 20):
    n = 200 * K
    x = max_count(n, 0.073)
    print(f"   K={K} (n={n}) bar 0.073: pass iff count <= {x}; P(pass | true 0.05) {stats.binom.cdf(x, n, 0.05):.3f}, "
          f"| 0.06 {stats.binom.cdf(x, n, 0.06):.3f}, | 0.08 {stats.binom.cdf(x, n, 0.08):.3f}")
print("   actual bank below 0.90 in at most 1 of 10 seed-worlds at a per-bank rate 0.05: "
      f"{stats.binom.cdf(1, 10, 0.05):.3f}")


def fmax(n, level):
    ok = [f for f in range(n + 1) if stats.binom.cdf(f, n, 0.1) <= 1 - level]
    return max(ok) if ok else -1


print("3. Looks with Bonferroni 0.05/3 (level 1 - 0.0167): largest harm count that certifies")
for n in (10, 30, 39, 40, 70, 100):
    print(f"   n={n}: f <= {fmax(n, 1 - 0.05 / 3)}")
print(f"   zero-failure floor at 0.05/3: {min(n for n in range(1, 200) if 0.9 ** n <= 0.05 / 3)}; "
      f"batch: {min(n for n in range(1, 200) if 0.9 ** n <= 0.05)}")

T5, T10 = stats.t.ppf(0.95, 4), stats.t.ppf(0.95, 9)


def passes(d, bar):
    m, s = d.mean(), d.std(ddof=1)
    lb = m - stats.t.ppf(0.95, len(d) - 1) * s / math.sqrt(len(d)) if s > 0 else m
    return m >= bar and lb > 0


print("4. Release power (exact binomial gate, alpha_B 0.1, level 0.95): per-seed certification and P(release passes)")
print("   release passes iff mean >= +10 and LB > 0 with per-seed values 61.8 (certified) or 0 (refused)")
for S in (5, 10):
    need = min(k for k in range(S + 1) if passes(np.array([61.8] * k + [0.0] * (S - k)), 10.0))
    for n in (36, 50, 70, 100, 150, 200, 300):
        f = fmax(n, 0.95)
        row = []
        for pi in (0.02, 0.03, 0.05, 0.07):
            c = stats.binom.cdf(f, n, pi) if f >= 0 else 0.0
            pk = 1 - stats.binom.cdf(need - 1, S, c)
            row.append(f"pi {pi}: cert {c:.2f} pass {pk:.2f}")
        print(f"   S={S} (needs {need} of {S} certified), n={n} (f<={f}): " + "; ".join(row))
print("   smallest n with per-seed certification >= 0.9:")
for pi in (0.02, 0.03, 0.04, 0.05):
    n = next(n for n in range(29, 2000) if fmax(n, 0.95) >= 0 and stats.binom.cdf(fmax(n, 0.95), n, pi) >= 0.9)
    print(f"   pi {pi}: n = {n}")

print("5. Holm critical values (one-sided 0.05 overall): first step of a family of k claims")
for k in (2, 3, 5):
    for S in (5, 10):
        t = stats.t.ppf(1 - 0.05 / k, S - 1)
        print(f"   k={k}, S={S}: level {0.05 / k:.4f}, t crit {t:.3f} (unadjusted {stats.t.ppf(0.95, S - 1):.3f}); "
              f"mean needed at sd 1.9 / 5.5 / 8.8: "
              + " / ".join(f"{t * s / math.sqrt(S):.1f}" for s in (1.9, 5.5, 8.8)))

print("6. Power of a mean >= bar and p <= level claim: P(pass) at true effect delta, seed sd s (simulation, 200,000)")
for S in (5, 10):
    for level, lab in ((0.05, "unadjusted"), (0.025, "Holm k=2 first"), (0.01, "Holm k=5 first")):
        tc = stats.t.ppf(1 - level, S - 1)
        out = []
        for delta, s in ((5.0, 1.9), (10.0, 5.5), (15.0, 8.8), (5.0, 5.5), (12.0, 8.8)):
            d = rng.normal(delta, s, (200000, S))
            m, sd = d.mean(1), d.std(1, ddof=1)
            ok = (m >= 5.0) & (m / (sd / math.sqrt(S)) >= tc)
            out.append(f"d={delta}/sd={s}: {ok.mean():.2f}")
        print(f"   S={S}, {lab} ({level}): " + "; ".join(out))

print("7. Smoothing law atoms: P(|N(0, 0.2^2)| >= 0.5) per thruster and the share of queries with any atom (4)")
p1 = 2 * stats.norm.sf(2.5)
print(f"   per thruster {p1:.4f}; any of 4 thrusters {1 - (1 - p1) ** 4:.4f}")

print("8. B-K1 permutation test: P(p <= 0.05 in >= 4 of 5 seeds | no information) "
      f"{1 - stats.binom.cdf(3, 5, 0.05):.2e}; >= 8 of 10: {1 - stats.binom.cdf(7, 10, 0.05):.2e}")
print("   smallest attainable p with 200 shuffles: 1/201 = %.4f" % (1 / 201))

print("9. A-R1 against the best world-blind rule (record values: hinge always 76.5 = 107.4 - 30.9; perfect switch "
      "106.1; one false switch costs 61.8)")
d = np.array([106.1 - 76.5] * 4 + [106.1 - 61.8 - 76.5])
m, s = d.mean(), d.std(ddof=1)
print(f"   one false switch in 5 seeds: per-seed diffs {np.round(d, 1).tolist()}, mean {m:.1f}, sd {s:.1f}, "
      f"LB {m - T5 * s / math.sqrt(5):.1f}")
d = np.array([106.1 - 76.5] * 5)
print(f"   perfect switch: per-seed diff {d[0]:.1f} in every seed (mean >= +10 and LB > 0 if the seed sd is below "
      f"{(d[0]) * math.sqrt(5) / T5:.1f})")

print("10. A-RAND by expectation: per-seed diff = (1 - r) V_S + r C_N with r = 0.5: "
      f"{0.5 * 106.1 + 0.5 * 61.8:.1f}")

print("11. C-R2 recast (dominated check): P(dominated reading | true C - plain = 0) at seed sd s, 5 seeds: "
      "mean(plain - C) >= 5 and LB > 0")
for s in (4.0, 5.5, 8.8):
    d = rng.normal(0.0, s, (200000, 5))
    m, sd = d.mean(1), d.std(1, ddof=1)
    ok = (m >= 5.0) & (m - T5 * sd / math.sqrt(5) > 0)
    print(f"   sd {s}: {ok.mean():.3f}")
