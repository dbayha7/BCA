"""R2's expectation under per-step weights, from the design numbers on synthetic banks (no ThrusterChain data, no
network): exact strong weights w(c) = 0.0907 exp(4c), wbcp.calibrate with miscoverage 0.1, credibility 0.95, 1,000
draws; per-step thresholds tabulated on the grid (ratio 1.01); dose = 1 + 0.5 x / (1 + x), x = R x eta (the closed
form; refusal 1.5).

Two settings (argument 'gap' or 'no-gap'; default 'gap'):
  gap     the gapped host that R2 and W1 are judged on. The ball's removal is modelled by the in-ball probability given
          the current, g_in(c) = clip(0.70 (c - 0.24), 0, 1), which reproduces the seed-0 ball's in-ball shares
          0.157 / 0.258 / 0.364 (none / mild / strong; printed below). Bank: 1,000 rows from the training law, each kept
          with probability 1 - g_in(c) (about 843 kept); kept training rows and test steps outside the ball are thinned
          the same way. The single threshold's test mass is the gapped one (the mean of w over strong-law steps outside
          the ball, by quadrature, about 2.29), and the unit and W2's edge are set on the gapped laws.
  no-gap  version 4's first calculation (the full-data host): 843 bank rows and the training rows from the training
          law with nothing removed, the full-data test mass (2.84). Same streams as that calculation, so it reproduces
          its output (r2_per_step_expectation_no_gap.txt).

Scenarios: (a) the scale network already follows c (scores |z|, eta_i = g(c_i) / unit, g(c) = 0.3 + 2c); (b) the
scale ignores c (scores |z| g(c) / unit, eta = 1), with g = exp(k c) and k set so unweighted strong coverage (outside
the ball, in the gap setting) is about 0.88 (W2's edge) and about 0.85.

Printed per scenario (means over banks): thresholds (unweighted Ru, single Rs, per-step at the lightest and heaviest
grid masses), dose changes (mean, mean absolute, top tenth of rows by current, largest on any row, and at the heaviest
possible row c = 1.1), coverage of the strong law outside the ball, and the bank's weight facts: kept rows, total
weight, Kish size, the heaviest test atom's share of the bank's weight, and the exponential draw E0 above which one
posterior draw at the heaviest test mass fails to cross (its test atom then carries more than 10% of the draw's mass:
t E0 > (alpha / (1 - alpha)) x total, so E0 > total / (9 t)), with the probability exp(-E0) of that per draw.
"""

import json
import math
import sys

import numpy as np

from experiments.shift_iw import coverage as CV
from experiments.shift_iw import world as W

MODE = sys.argv[1] if len(sys.argv) > 1 else "gap"
if MODE not in ("gap", "no-gap"):
    raise SystemExit("usage: r2_per_step_expectation.py [gap|no-gap] [banks]")
GAP = MODE == "gap"
BANKS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
N_ROWS, N_TEST, RATIO, ALPHA = 20000, 40000, 1.01, 0.1
N_BANK = 1000 if GAP else 843      # drawn; the gap keeps about 843 of 1,000
SLOPE, START = 0.70, 0.24          # g_in(c) = clip(SLOPE (c - START), 0, 1)


def g_in(c):
    """The modelled in-ball probability given the current (gap setting); 0 everywhere in the no-gap setting."""
    c = np.asarray(c, np.float64)
    return np.clip(SLOPE * (c - START), 0.0, 1.0) if GAP else np.zeros_like(c)


def quad(f, law, n=200001):
    """The mean of f(c) over the law on [0.1, 1.1] by the trapezoid rule on a fine grid."""
    c = np.linspace(W.C_LO, W.C_HI, n)
    p = W.density(c, law)
    trap = lambda y: float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(c)))
    return trap(f(c) * p) / trap(p)


def outside_mean(f, law):
    """The mean of f(c) over the law's steps outside the ball (density p_law(c) (1 - g_in(c)))."""
    return quad(lambda c: f(c) * (1.0 - g_in(c)), law) / quad(lambda c: 1.0 - g_in(c), law)


def draw_kept(rng, n, law):
    """n currents from the law's steps outside the ball (thinning by 1 - g_in)."""
    out = []
    while sum(len(o) for o in out) < n:
        c = W.inverse_cdf(rng.uniform(size=2 * n), law)
        out.append(c[rng.uniform(size=c.size) >= g_in(c)])
    return np.concatenate(out)[:n]


if GAP:
    rng = np.random.default_rng([20261016, 9190])
    c_rows = draw_kept(rng, N_ROWS, "train")
    c_test = draw_kept(rng, N_TEST, "strong")
    z_test = np.abs(rng.standard_normal(N_TEST))
    T_SINGLE = outside_mean(lambda c: W.ratio(c, "strong"), "strong")
else:   # version 4's first calculation, stream for stream
    rng = np.random.default_rng([20261016, 9090])
    c_rows = W.inverse_cdf(rng.uniform(size=N_ROWS), "train")
    c_test = W.inverse_cdf(rng.uniform(size=N_TEST), "strong")
    z_test = np.abs(rng.standard_normal(N_TEST))
    T_SINGLE = W.full_test_mass("strong")
top = c_rows >= np.quantile(c_rows, 0.9)
w_rows = W.ratio(c_rows, "strong")
W_MAX = float(W.ratio(W.C_HI, "strong"))
grid = CV.mass_grid(np.array([W.ratio(W.C_LO, "strong"), W_MAX]), RATIO)


def dose(x):
    x = np.asarray(x, np.float64)
    return np.where(np.isfinite(x), 1.0 + 0.5 * x / (1.0 + x), 1.5)


def scenario(name, g, eta_follows):
    if GAP:
        zz = np.abs(np.random.default_rng([20261016, 9194]).standard_normal(200000))
        cc = draw_kept(np.random.default_rng([20261016, 9195]), 200000, "train")
    else:
        zz = np.abs(rng.standard_normal((1, 200000)))[0]
        cc = W.inverse_cdf(rng.uniform(size=200000), "train")
    unit = float(np.sqrt(np.mean((zz * g(cc)) ** 2)))     # RMS residual over kept training rows
    eta_hi = float(g(W.C_HI) / unit) if eta_follows else 1.0
    out = []
    for b in range(BANKS):
        r = np.random.default_rng([20261016, 9091, b])
        c = W.inverse_cdf(r.uniform(size=N_BANK), "train")
        z = np.abs(r.standard_normal(N_BANK))
        if GAP:
            keep = np.random.default_rng([20261016, 9093, b]).uniform(size=N_BANK) >= g_in(c)
            c, z = c[keep], z[keep]
        scores = z if eta_follows else z * g(c) / unit
        w = W.ratio(c, "strong")
        stream = [20261016, 9092, b]
        Ru = CV.calibrate(scores, stream).threshold
        Rs = CV.calibrate(scores, stream, w, T_SINGLE).threshold
        tab = CV.threshold_table(scores, stream, w, grid)
        Rrow, _ = CV.lookup(tab, w_rows)
        eta = g(c_rows) / unit if eta_follows else np.ones(N_ROWS)
        du, ds, dp = dose(Ru * eta), dose(Rs * eta), dose(Rrow * eta)
        R_hi = tab.thresholds[-1]                          # the heaviest possible row, c = 1.1, w = 7.39
        hi_u, hi_s, hi_p = (float(dose(R * eta_hi)) for R in (Ru, Rs, R_hi))
        # coverage of the strong law outside the ball (per-step and single), for orientation
        st = z_test if eta_follows else z_test * g(c_test) / unit
        Rt, _ = CV.lookup(tab, W.ratio(c_test, "strong"))
        total = float(w.sum())
        e0 = (ALPHA / (1.0 - ALPHA)) * total / W_MAX
        out.append(dict(bank_rows=float(len(c)), bank_total_weight=total, bank_mean_weight=total / len(c),
                        kish=CV.kish(w), heaviest_share_of_total=W_MAX / total, e0_fail=e0,
                        p_draw_fails=math.exp(-e0),
                        Ru=Ru, Rs=Rs, R_lo=tab.thresholds[0], R_hi=R_hi, monotone=tab.monotone,
                        refused_rows=float(np.mean(np.isinf(Rrow))),
                        mean_single=float((ds - du).mean()), mean_step=float((dp - du).mean()),
                        abs_step=float(np.abs(dp - du).mean()), top_step=float((dp - du)[top].mean()),
                        step_minus_single=float((dp - ds).mean()), top_step_minus_single=float((dp - ds)[top].mean()),
                        max_step=float(np.abs(dp - du).max()), x_u_mean=float((Ru * eta).mean()),
                        x_u_heaviest=float(Ru * eta_hi), heaviest_step=hi_p - hi_u,
                        heaviest_step_minus_single=hi_p - hi_s,
                        cov_u=float(np.mean(st <= Ru)), cov_s=float(np.mean(st <= Rs)),
                        cov_step=float(np.mean(st <= Rt))))
    keys = out[0].keys()
    summ = {k: float(np.mean([o[k] for o in out])) for k in keys}
    summ.update({f"{k}_sd": float(np.std([o[k] for o in out])) for k in ("mean_step", "top_step", "cov_step")})
    summ["ratio_hi_over_u"] = float(np.mean([o["R_hi"] / o["Ru"] for o in out]))
    summ["ratio_lo_over_u"] = float(np.mean([o["R_lo"] / o["Ru"] for o in out]))
    summ["ratio_hi_over_single"] = float(np.mean([o["R_hi"] / o["Rs"] for o in out]))
    summ["ratio_single_over_u"] = float(np.mean([o["Rs"] / o["Ru"] for o in out]))
    summ["all_monotone"] = all(o["monotone"] for o in out)
    summ["refused_rows_max"] = float(max(o["refused_rows"] for o in out))
    summ["unit"] = unit
    return name, summ


def k_for(target):
    """k such that the unweighted band (the 91.7% quantile of kept training-law scores |z| exp(k c)) covers `target`
    of the strong law's steps outside the ball."""
    lo, hi = 0.0, 4.0
    zz = np.abs(np.random.default_rng(7).standard_normal(200000))
    if GAP:
        ctr = draw_kept(np.random.default_rng(8), 200000, "train")
        cst = draw_kept(np.random.default_rng(9), 200000, "strong")
    else:
        ctr = W.inverse_cdf(np.random.default_rng(8).uniform(size=200000), "train")
        cst = W.inverse_cdf(np.random.default_rng(9).uniform(size=200000), "strong")
    for _ in range(40):
        k = 0.5 * (lo + hi)
        s_tr = zz * np.exp(k * ctr)
        q = np.quantile(s_tr, 0.917)            # the band's expected no-shift coverage, 91.7%
        cov = np.mean(zz * np.exp(k * cst) <= q)
        lo, hi = (k, hi) if cov > target else (lo, k)
    return 0.5 * (lo + hi)


print("setting", MODE, "banks", BANKS, "bank rows drawn", N_BANK, flush=True)
if GAP:
    print("in-ball share by law (model g_in):", json.dumps({law: round(quad(g_in, law), 4) for law in W.TILT_NAMES}))
    print("Z = P_train(outside the ball):", round(1.0 - quad(g_in, "train"), 4))
    print("mean strong weight over kept training-law rows (Z_test / Z):",
          round(outside_mean(lambda c: W.ratio(c, "strong"), "train"), 4))
print("single-threshold test mass:", round(T_SINGLE, 4), flush=True)
res = {}
ks = {}
for name, target in (("b_edge_0.88", 0.88), ("b_0.85", 0.85)):
    ks[name] = k_for(target)
for name, g, follows in [("a_scale_follows_c", lambda c: 0.3 + 2.0 * np.asarray(c, np.float64), True),
                         ("b_edge_0.88", (lambda k: (lambda c: np.exp(k * np.asarray(c, np.float64))))(ks["b_edge_0.88"]),
                          False),
                         ("b_0.85", (lambda k: (lambda c: np.exp(k * np.asarray(c, np.float64))))(ks["b_0.85"]), False)]:
    n, s = scenario(name, g, follows)
    if name in ks:
        s["k"] = ks[name]
    res[n] = s
    print(n, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
print("grid points", len(grid), grid[0], grid[-1])
