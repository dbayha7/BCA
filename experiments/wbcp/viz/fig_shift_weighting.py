"""Importance weights under distribution shift: uniform BQ-CP against WBCP on a real frozen pool.

What the figure shows
  a  The real hopper-medium-v2 held-out score pool (TD3+BC with BCA's scale fit, 100k updates, 500,285 rows) under
     the benchmark's density tilt at gamma = 1. The miscoverage curve L(lambda) = share of rows with score > lambda is
     drawn under the calibration law (every pool row equally likely) and under the tilted test law (row i weighted by
     a_i = exp(gamma z_i)). The threshold that is exact for the calibration law misses more test rows than alpha.
  b  One 1,103-row calibration bank drawn with the benchmark's own random streams. calibrate() is run without weights
     (uniform BQ-CP; the benchmark deploys its lambda_hpd) and with the exact weights w* and test mass wbar (WBCP). The
     two threshold posteriors (the per-draw alpha-crossings), each deployed threshold, its true miscoverage under the
     tilted law (exact, over the whole pool) and the Kish n_eff of the weights.
  c  The benchmark's failure rates against shift strength for the same tilt and bank size (iid banks, 2,000 banks
     per point, exact 95% intervals): uniform BQ-CP against WBCP with exact and with estimated weights. The two WBCP
     series are drawn 0.045 to either side of each gamma so they do not hide each other.

Where the data come from
  - Pool: runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000 (frozen.npz), loaded with
    experiments/wbcp/d4rl_benchmark.py load_pool() (line 125), population 'heldout'.
  - Tilt: d4rl_benchmark.py tilt_feature() (line 193, density = log distance to the 10th nearest neighbour among a
    seeded 10,000-row reference subset), standardize() (169) and make_tilt() (247), with the k-NN settings read from
    runs/wbcp_bench/main.json. Exact risk and lambda*: ExactRisk (256). The recomputed tilt is checked against the
    n_eff, oracle wbar and lambda* that main.json recorded, so panels a-b use the tilt behind panel c.
  - Bank: draw_calibration(stream(seed, "cal", n, trial)) as in run_trial() (line 443); BQ-CP with
    stream(seed, "bq", n, trial, score) and .lambda_hpd (453); exact weights by common_scale(tilt.log_mass[rows],
    log(sum a^2 / sum a)) (464) and WBCP with stream(seed, "oracle", ...) (473, _wbcp at 488) through
    calibration/wbcp.py calibrate() (line 57; lambda_hpd line 115, Kish n_eff line 116). The benchmark's BQ-CP arm
    records lambda_hpd; BCA itself deploys max(lambda_hat, lambda_hpd) (calibration/reference.py:176-179), and for
    the bank shown the clamp does not bind, so the two agree (lambda_hat is in the facts). Panel b plots each
    posterior as its empirical CDF: the share of draws whose alpha-crossing is <= lambda, which is Pr(L+ <= alpha)
    of Eq. (7), so each deployed threshold is where its curve reaches beta.
    The bank shown is the first in benchmark order whose outcome matches the benchmark majority (uniform fails, WBCP
    holds); the banks skipped are reported in the facts.
  - Panel c: runs/wbcp_bench/main.json, density tilt, n = 1,103, normalized score (the "Independent rows" design of
    results.ipynb section 5; DEPENDENCE.md Change 5 iid column). The finite-sample note uses
    runs/wbcp_dependence/hopper-u100000-weighting/a_density1.json (Change 11, results.ipynb section 8).

Runs in a few seconds: one k-NN pass over the pool and two calibrate() calls per bank tried.
"""

import json
import math

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from calibration.wbcp import calibrate
from experiments.wbcp import d4rl_benchmark as DB  # numpy and scipy only; no JAX
from experiments.wbcp.viz import style as S

KEY = "shift_weighting"
BENCH = S.ROOT / "runs" / "wbcp_bench" / "main.json"
SWEEP = S.ROOT / "runs" / "wbcp_dependence" / "hopper-u100000-weighting" / "a_density1.json"
TILT, GAMMA, N, SCORE = "density", 1.0, 1103, "normalized"
GAMMAS = (0.0, 0.5, 1.0, 2.0)


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def _settings():
    with open(BENCH) as handle:
        bench = json.load(handle)
    with open(SWEEP) as handle:
        sweep = json.load(handle)
    return bench, sweep


def _block(result, gamma, n=N):
    for block in result["blocks"]:
        if (block["tilt"], block["gamma"], block["n"], block["score"]) == (TILT, gamma, n, SCORE):
            return block
    raise KeyError((TILT, gamma, n, SCORE))


def _pool_and_tilt(bench):
    cfg = bench["settings"]
    pool = DB.load_pool(str(S.ROOT / cfg["frozen"]), cfg["population"])
    raw = DB.tilt_feature(TILT, pool.obs, pool.action, pool.policy_action, state_feature=cfg["state_feature"],
                          knn_k=cfg["knn_k"], knn_reference=cfg["knn_reference"], knn_seed=cfg["knn_seed"],
                          density_transform=cfg["density_transform"])
    tilt = DB.make_tilt(TILT, DB.standardize(raw), GAMMA)
    scores = pool.scores[SCORE]
    order = np.argsort(scores, kind="stable")
    risk = DB.ExactRisk(scores, tilt.mass, order)
    uniform = DB.ExactRisk(scores, np.ones(scores.size), order)
    recorded = _block(bench, GAMMA)
    checks = [(tilt.n_eff, recorded["tilt_n_eff"]), (tilt.oracle_wbar, recorded["oracle_wbar"]),
              (risk.lambda_star(cfg["alpha"]), recorded["lambda_star"]),
              (uniform.lambda_star(cfg["alpha"]), bench["lambda_star_uniform"][SCORE])]
    for mine, theirs in checks:
        if not math.isclose(mine, theirs, rel_tol=1e-9):
            raise RuntimeError(f"recomputed tilt differs from main.json: {mine} vs {theirs}")
    return pool, scores, tilt, risk, uniform


def _bank(cfg, scores, tilt, risk, trial):
    """One benchmark bank (run_trial, d4rl_benchmark.py:439-475): BQ-CP and exact-weight WBCP."""
    seed, options = cfg["seed"], dict(alpha=cfg["alpha"], beta=cfg["beta"], draws=cfg["draws"])
    rows = DB.draw_calibration(DB.stream(seed, "cal", N, trial), N, scores.size)
    bank = scores[rows]
    uniform = calibrate(bank, DB.stream(seed, "bq", N, trial, SCORE), **options)
    weights, wbar = DB.common_scale(tilt.log_mass[rows], math.log(tilt.square_total / tilt.total))
    weighted = calibrate(bank, DB.stream(seed, "oracle", N, trial, TILT, GAMMA, SCORE), weights,
                         max(wbar, np.finfo(np.float64).tiny), **options)
    lam_u, lam_w = uniform.lambda_hpd, weighted.threshold  # what the benchmark deploys for each arm
    miss_u, miss_w = (float(v) for v in risk(np.array([lam_u, lam_w])))
    return dict(trial=trial, uniform=uniform, weighted=weighted, lam_u=lam_u, lam_w=lam_w, miss_u=miss_u,
                miss_w=miss_w)


def make():
    bench, sweep = _settings()
    cfg = bench["settings"]
    alpha = cfg["alpha"]
    pool, scores, tilt, risk, uniform_risk = _pool_and_tilt(bench)
    lam_cal, lam_star = uniform_risk.lambda_star(alpha), risk.lambda_star(alpha)
    miss_at_cal = float(risk(np.array([lam_cal]))[0])
    max_w = tilt.mass.size / tilt.total  # largest w* = a / mean(a)

    tried = []
    for trial in range(50):
        bank = _bank(cfg, scores, tilt, risk, trial)
        tried.append(bank)
        if bank["miss_u"] > alpha >= bank["miss_w"]:
            break
    else:
        raise RuntimeError("no bank with the majority outcome among the first 50")

    fig, (ax_a, ax_b, ax_c) = S.figure(1, 3, gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.08]))
    facts = []

    def fact(what, value, source):
        facts.append(dict(what=what, value=value, source=source))

    pool_src = "recomputed in fig_shift_weighting.py from runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000 " \
               "with d4rl_benchmark.py tilt_feature/make_tilt/ExactRisk; matches runs/wbcp_bench/main.json"

    # ---- a: miscoverage curves under the calibration law and the tilted test law -------------------------------
    xlo, xhi = 0.9, 1.65
    grid = np.linspace(xlo, xhi, 800)

    def cal(v):
        return 100 * float(uniform_risk(np.array([v]))[0])

    def test(v):
        return 100 * float(risk(np.array([v]))[0])

    ax_a.plot(grid, 100 * uniform_risk(grid), color=S.ORANGE, ls=(0, (5, 2.5)), lw=3)
    ax_a.plot(grid, 100 * risk(grid), color=S.BLUE, lw=3.4)
    ax_a.axhline(100 * alpha, color=S.INK, lw=1.4, ls=(0, (2, 2)))
    ax_a.text(xhi - 0.005, 100 * alpha - 0.3, "α = 10%", ha="right", va="top", fontsize=S.SMALL, color=S.INK)
    ax_a.vlines(lam_cal, 0, 100 * miss_at_cal, color=S.SOFT, lw=1.3, ls=(0, (1, 2)))
    ax_a.plot([lam_cal], [100 * alpha], "o", color=S.ORANGE, ms=11, zorder=5)
    ax_a.plot([lam_cal], [100 * miss_at_cal], "o", color=S.BLUE, ms=11, zorder=5)
    ax_a.plot([lam_star], [100 * alpha], "o", color=S.INK, ms=11, zorder=5)
    ax_a.annotate(f"λ = {lam_cal:.2f} misses\n{_pct(miss_at_cal)} of test rows",
                  xy=(lam_cal, 100 * miss_at_cal), xytext=(lam_cal + 0.04, 100 * miss_at_cal + 0.7),
                  fontsize=S.SMALL, color=S.INK, va="bottom", ha="left",
                  arrowprops=dict(arrowstyle="-", color=S.SOFT, lw=1.2, shrinkA=2, shrinkB=7))
    ax_a.text(lam_star + 0.012, 100 * alpha + 0.3, f"λ* = {lam_star:.2f}", ha="left", va="bottom",
              fontsize=S.SMALL, color=S.INK)
    ax_a.text(1.37, test(1.37) + 0.45, "tilted test law", color=S.BLUE, fontsize=S.SMALL, ha="left", va="bottom")
    ax_a.text(1.08, cal(1.36) - 0.45, "calibration law", color=S.ORANGE, fontsize=S.SMALL, ha="left", va="top")
    ax_a.text(xhi - 0.005, 18.6, f"test weight of a row:\nw* = exp(γ z) / its mean\nlargest w* = {max_w:.0f}",
              color=S.SOFT, fontsize=S.SMALL, ha="right", va="top")
    ax_a.set_xlim(xlo, xhi)
    ax_a.set_ylim(2, 19)
    ax_a.set_xticks([1.0, 1.2, 1.4, 1.6])
    ax_a.set_xlabel("Threshold λ on the score |y − q| / σ")
    ax_a.set_ylabel("Rows with score > λ (%)")
    S.panel_label(ax_a, "a  Density tilt γ = 1 fattens the tail")

    fact("Pool rows (hopper-medium-v2 held-out half)", f"{pool.size:,}", "runs/wbcp_bench/main.json population.rows")
    fact("Shift shown in panels a-b", "density tilt, gamma = 1", "runs/wbcp_bench/main.json settings; DEPENDENCE.md Terms")
    fact("Target miscoverage alpha", "10%", "runs/wbcp_bench/main.json settings.alpha")
    fact("Threshold exact for the calibration law (uniform lambda*)", f"{lam_cal:.4f}",
         "runs/wbcp_bench/main.json lambda_star_uniform.normalized; " + pool_src)
    fact("Test-law miscoverage of that threshold", _pct(miss_at_cal, 2), pool_src)
    fact("Test-law lambda* (density gamma 1)", f"{lam_star:.4f}", "runs/wbcp_bench/main.json block lambda_star")
    fact("Largest exact weight w* / mean weight", f"{max_w:.1f}",
         "runs/wbcp_bench/main.json tilts.density.gammas['1.0'].max_oracle_w; DEPENDENCE.md Change 11 table (33)")
    fact("Tilted pool n_eff (share of rows)", f"{tilt.n_eff:,.0f} ({_pct(tilt.n_eff / pool.size)})",
         "runs/wbcp_bench/main.json block tilt_n_eff")

    # ---- b: one bank, uniform and weighted threshold posteriors (Eq. 7: Pr(L <= alpha) at lambda) ---------------
    shown = tried[-1]
    u, w = shown["uniform"], shown["weighted"]
    beta = cfg["beta"]
    # long dashes with short gaps: the uniform posterior is nearly vertical, and short dashes break up on its steps
    for post, color, ls in ((u.crossings, S.ORANGE, (0, (7, 1.6))), (w.crossings, S.BLUE, "-")):
        xs = np.concatenate([[xlo], np.clip(post, xlo, xhi), [xhi]])
        ys = np.concatenate([[0.0], np.arange(1, post.size + 1) / post.size, [np.mean(post <= xhi)]])
        ax_b.step(xs, ys, where="post", color=color, ls=ls, lw=3)
    ax_b.axhline(beta, color=S.INK, lw=1.4, ls=(0, (2, 2)))
    ax_b.text(xhi - 0.005, beta - 0.02, f"β = {beta:g}", ha="right", va="top", fontsize=S.SMALL, color=S.INK)
    ax_b.vlines(lam_star, 0, 1.0, color=S.INK, lw=2.2)
    ax_b.text(lam_star, 1.015, "λ*", ha="center", va="bottom", fontsize=S.SMALL, color=S.INK)
    ax_b.plot([shown["lam_u"]], [beta], "o", color=S.ORANGE, ms=11, zorder=5)
    ax_b.plot([shown["lam_w"]], [beta], "o", color=S.BLUE, ms=11, zorder=5)
    ax_b.text(1.255, 0.70, f"uniform BQ-CP\nλ = {shown['lam_u']:.2f}, n = {N:,}\nmisses {_pct(shown['miss_u'])}: fails",
              ha="left", va="top", fontsize=S.SMALL, color=S.ORANGE)
    ax_b.text(1.255, 0.36, f"WBCP, exact w*\nλ = {shown['lam_w']:.2f}, n_eff {w.n_eff:.0f}\n"
              f"misses {_pct(shown['miss_w'])}: holds", ha="left", va="top", fontsize=S.SMALL, color=S.BLUE)
    ax_b.set_xlim(xlo, xhi)
    ax_b.set_ylim(0, 1.09)
    ax_b.set_xticks([1.0, 1.2, 1.4, 1.6])
    ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax_b.set_xlabel("Threshold λ")
    ax_b.set_ylabel("Posterior Pr(miss ≤ α)")
    S.panel_label(ax_b, "b  One bank: two threshold posteriors")

    src_b = (f"computed in fig_shift_weighting.py: benchmark bank {shown['trial']} (d4rl_benchmark.py run_trial streams, "
             f"seed {cfg['seed']}), calibration/wbcp.py calibrate(alpha={alpha}, beta={cfg['beta']}, "
             f"draws={cfg['draws']}); miss = ExactRisk under the tilted law over the whole pool")
    fact("Bank shown (benchmark order)", f"bank {shown['trial']}, n = {N:,} iid rows", src_b)
    fact("Uniform BQ-CP threshold (lambda_hpd)", f"{shown['lam_u']:.4f}", src_b)
    fact("Uniform BQ-CP test-law miscoverage", _pct(shown["miss_u"], 2) + " (> 10%: fails)", src_b)
    fact("WBCP (exact w) threshold", f"{shown['lam_w']:.4f}", src_b)
    fact("WBCP (exact w) test-law miscoverage", _pct(shown["miss_w"], 2) + " (<= 10%: holds)", src_b)
    fact("WBCP Kish n_eff of the bank's weights", f"{w.n_eff:.1f} of {N:,}", src_b)
    fact("Posterior SD, uniform / WBCP", f"{u.sigma_post:.3f} / {w.sigma_post:.3f}", src_b)
    fact("Posterior draws per bank", f"{u.crossings.size:,}", src_b)
    fact("Empirical-quantile clamp lambda_hat, uniform / WBCP (bank shown; not drawn)",
         f"{u.lambda_hat:.4f} / {w.lambda_hat:.4f}: below lambda_hpd, so the clamp max(lambda_hat, lambda_hpd) that BCA "
         f"deploys (calibration/reference.py:176-179 uses Calibration.threshold) does not bind and equals the "
         f"benchmark's BQ-CP lambda_hpd (d4rl_benchmark.py:453)", src_b)
    for skipped in tried[:-1]:
        fact(f"Bank {skipped['trial']} (skipped by the selection rule)",
             f"uniform {skipped['lam_u']:.4f} misses {_pct(skipped['miss_u'], 2)}; WBCP {skipped['lam_w']:.4f} "
             f"misses {_pct(skipped['miss_w'], 2)}, n_eff {skipped['weighted'].n_eff:.0f}", src_b)
    fact("Benchmark mean thresholds at density gamma 1 (uniform / WBCP exact w)",
         f"{_block(bench, GAMMA)['arms']['BQ-CP']['threshold']:.3f} / "
         f"{_block(bench, GAMMA)['arms']['WBCP (oracle w)']['threshold']:.3f}",
         "runs/wbcp_bench/main.json block arms[*].threshold (context for the bank shown)")

    # ---- c: benchmark failure rates against shift strength ------------------------------------------------------
    blocks = [_block(bench, g) for g in GAMMAS]
    # the two WBCP arms are dodged sideways (by 0.045 in gamma) so their markers and intervals do not hide each other
    arms = (("BQ-CP", S.ORANGE, (0, (5, 2.5)), "o", S.ORANGE, 0.0),
            ("WBCP (oracle w)", S.BLUE, "-", "o", S.BLUE, -0.045),
            ("WBCP", S.BLUE, (0, (1.5, 1.5)), "D", S.PAPER, 0.045))
    x = np.array(GAMMAS)
    for arm, color, ls, marker, face, dodge in arms:
        rate = np.array([100 * b["arms"][arm]["fail"] for b in blocks])
        low = np.array([100 * b["arms"][arm]["ci"][0] for b in blocks])
        high = np.array([100 * b["arms"][arm]["ci"][1] for b in blocks])
        ax_c.errorbar(x + dodge, rate, yerr=[rate - low, high - rate], color=color, ls=ls, lw=2.8, marker=marker, ms=10,
                      mfc=face, mew=2.2, capsize=4, elinewidth=1.6)
        for g, b in zip(GAMMAS, blocks):
            a = b["arms"][arm]
            fact(f"Banks failing, {arm}, density gamma {g:g}, n = {N:,}",
                 f"{_pct(a['fail'], 2)} [{_pct(a['ci'][0], 1)}, {_pct(a['ci'][1], 1)}] ({a['failures']}/{a['trials']})"
                 + (f", abstains {_pct(a['abstain'], 1)}" if a["abstain"] > 0 else ""),
                 "runs/wbcp_bench/main.json (results.ipynb section 5 'Independent rows'; DEPENDENCE.md Change 5)")
    budget = 100 * (1 - cfg["beta"])
    ax_c.axhline(budget, color=S.INK, lw=1.4, ls=(0, (2, 2)))
    ax_c.text(2.13, budget * 1.05, "5% budget", ha="right", va="bottom", fontsize=S.SMALL, color=S.INK)
    ax_c.set_yscale("log")
    ax_c.set_ylim(1.1, 135)
    ax_c.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
    ax_c.yaxis.set_minor_locator(NullLocator())
    ax_c.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax_c.set_xlim(-0.15, 2.15)
    neff = [b["calibration_n_eff"]["oracle"] for b in blocks]
    ax_c.set_xticks(x, [f"{g:g}\n{m:,.0f}" for g, m in zip(GAMMAS, neff)])
    ax_c.set_xlabel("Shift γ  (second row: n_eff of exact w*)")
    ax_c.set_ylabel("Banks failing (%, log scale)")
    ax_c.text(1.08, 42, "uniform BQ-CP", color=S.ORANGE, fontsize=S.SMALL, ha="left", va="center")
    ax_c.text(1.08, 10.0, "WBCP: exact w* (solid),\nestimated ŵ (dotted)", color=S.BLUE, fontsize=S.SMALL,
              ha="left", va="bottom")
    ab = blocks[-1]["arms"]
    ax_c.text(1.85, 1.2, f"at γ = 2 WBCP abstains in\n{_pct(ab['WBCP (oracle w)']['abstain'], 0)} (exact) / "
              f"{_pct(ab['WBCP']['abstain'], 0)} (est.) of banks", ha="right", va="bottom", fontsize=S.SMALL,
              color=S.SOFT)
    S.panel_label(ax_c, "c  Benchmark: banks failing vs shift")

    for b, m in zip(blocks, neff):
        fact(f"Mean calibration n_eff (exact w), density gamma {b['gamma']:g}", f"{m:,.1f} of {N:,}",
             "runs/wbcp_bench/main.json block calibration_n_eff.oracle")
    fact("Failure budget 1 - beta", "5%", "runs/wbcp_bench/main.json settings.beta = 0.95")

    big = _block(sweep, GAMMA, 8824)["arms"]["WBCP (oracle w)"]
    small = _block(sweep, GAMMA, 1103)["arms"]["WBCP (oracle w)"]
    fact("Change 11 sweep, WBCP exact w at density gamma 1 (not drawn)",
         f"{_pct(small['fail'], 2)} at n = 1,103 -> {_pct(big['fail'], 2)} at n = 8,824 (4,000 banks each)",
         "runs/wbcp_dependence/hopper-u100000-weighting/a_density1.json; DEPENDENCE.md Change 11")
    return fig, facts


if __name__ == "__main__":
    figure, found = make()
    print(S.save(figure, KEY))
    for item in found:
        print(f"{item['what']}: {item['value']}")
