"""C2 / C3 rules on the certified seeds: per-twin 'BCA ahead' (mean d >= 5, positive in >= ceil(0.8 n)), the mirrored
'twin ahead' (mean d <= -5, positive in <= 1), the tie otherwise; null rates, power, the conjunction's power, and the
coin chances of the sign thresholds. Independent normal per-seed differences; for the conjunction the three twins share
BCA's side (d_t = mu_t + e0 - e_t, e0 and e_t each with sd s/sqrt(2))."""
import math
import numpy as np

rng = np.random.default_rng(20261020)
R = 200_000


def pos_min(n):
    return math.ceil(0.8 * n)


def read(d):
    n = d.shape[1]
    m = d.mean(axis=1)
    k = (d > 0).sum(axis=1)
    ahead = (m >= 5.0) & (k >= pos_min(n))
    behind = (m <= -5.0) & (k <= 1)
    return ahead, behind, ~(ahead | behind)


def coin(n, k_min):
    return sum(math.comb(n, k) for k in range(k_min, n + 1)) / 2 ** n


print("sign thresholds (coin chances):")
for n in (10, 9, 8):
    km = pos_min(n)
    print(f"  n={n}: BCA ahead needs >= {km} positive: coin {coin(n, km):.4f};  twin ahead needs <= 1 positive: coin {(1 + n) / 2 ** n:.4f}")

print("\nper twin, one-sided rules on n certified seeds (rows: true edge, sd):")
for n in (10, 9, 8):
    for mu, sd in ((0, 5), (0, 7.5), (0, 10), (0, 15), (3, 5), (5, 5), (5, 10), (10, 5), (10, 10), (5.6, 1.9),
                   (-3.2, 5), (-4.5, 5.2), (-5, 5), (-10, 5), (-10, 10)):
        d = rng.normal(mu, sd, size=(R, n))
        a, b, t = read(d)
        print(f"  n={n:2d} edge {mu:+5.1f} sd {sd:4.1f}: BCA ahead {a.mean():.3f}  twin ahead {b.mean():.3f}  tie {t.mean():.3f}")

print("\nfamily of three twins, true tie, 'any twin ahead' (shared BCA side):")
for sd in (5, 7.5, 10, 15):
    n = 10
    s = sd / math.sqrt(2)
    e0 = rng.normal(0, s, size=(R, n))
    anyb = np.zeros(R, bool)
    anya = np.zeros(R, bool)
    for t in range(3):
        d = e0 - rng.normal(0, s, size=(R, n))
        a, b, _ = read(d)
        anyb |= b
        anya |= a
    print(f"  sd {sd:4.1f}: any twin ahead {anyb.mean():.4f}; any 'BCA ahead' {anya.mean():.4f}")

print("\nC2 conjunction (BCA ahead of all three), shared BCA side, n=10:")
for mus, sd in (((5, 5, 5), 5), ((5, 5, 5), 10), ((10, 10, 10), 5), ((10, 10, 10), 10), ((6, 0, -3.5), 5),
                ((10, 5, 5), 5), ((15, 10, 10), 10)):
    n = 10
    s = sd / math.sqrt(2)
    e0 = rng.normal(0, s, size=(R, n))
    allA = np.ones(R, bool)
    for mu in mus:
        d = mu + e0 - rng.normal(0, s, size=(R, n))
        a, _, _ = read(d)
        allA &= a
    print(f"  edges {mus} sd {sd:4.1f}: C2 holds {allA.mean():.3f}")
