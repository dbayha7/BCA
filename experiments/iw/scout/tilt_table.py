"""Closed-form numbers for truncated exponential densities p_lam(c) = exp(lam c) / M(lam) on [0.1, 1.1]."""
import math
LO, HI = 0.1, 1.1
def M(l): return (HI - LO) if l == 0 else (math.exp(l * HI) - math.exp(l * LO)) / l
def mean(l):
    if l == 0: return (LO + HI) / 2
    return (HI * math.exp(l * HI) - LO * math.exp(l * LO)) / (l * M(l)) - 1 / l
def above(l, a): return (HI - a) / (HI - LO) if l == 0 else (math.exp(l * HI) - math.exp(l * a)) / (l * M(l))
def kish(l0, l, n=1000): return n / (M(l0) * M(2 * l - l0) / M(l) ** 2)
def wmax(l0, l): return max(math.exp((l - l0) * c) * M(l0) / M(l) for c in (LO, HI))
def wmin(l0, l): return min(math.exp((l - l0) * c) * M(l0) / M(l) for c in (LO, HI))
for l0 in (-2.0, -3.0):
    print(f"p_train lam {l0}: mean c {mean(l0):.3f}, P(c>0.9) {above(l0, 0.9):.3f}, P(c>0.7) {above(l0, 0.7):.3f}")
    for l in (-2, -1, 0, 1, 1.5, 2, 2.5, 3):
        print(f"  test lam {l:+.1f}: mean c {mean(l):.3f}  P(c>0.9) {above(l, 0.9):.3f}  P(c>0.7) {above(l, 0.7):.3f}"
              f"  Kish/1000 {kish(l0, l):6.1f}  w range [{wmin(l0, l):.3f}, {wmax(l0, l):.2f}]")
