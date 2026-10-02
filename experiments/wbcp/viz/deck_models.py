"""Deck charts, group 'models': WBCP's theory and BCA's models as pictures, one message per chart.

Every chart is drawn with style.deck_figure and saved with style.save_deck under runs/wbcp_viz/deck/<key>.png.
Each public chart function returns (fig, facts); make_all() renders all six and returns
{key: {png, headline, takeaway, notes, facts}}.

  score_threshold   one real calibration bank (frozen TD3+BC hopper pool, BCA's sampler, n = 1,024, K = 6) as a
                    histogram, the deployed WBCP threshold and the rows above it.
  posterior_draws   25 of that bank's 1,000 Bayesian-bootstrap CDFs near the 1 - alpha line, their crossings and
                    lambda_hpd (fig_wbcp_threshold's regeneration of calibrate()'s draws).
  three_guarantees  per-bank true miss rate of three rules on the same real hopper banks (no shift, 1,103 iid rows,
                    the benchmark's own random streams): W-CRC (marginal), RCPS (high-probability), WBCP
                    (posterior-credible). Real data, not a synthetic illustration.
  shift_tilt        share of rows above a cut-off under the calibration law and under the density tilt gamma = 1
                    (fig_shift_weighting panel a).
  design_effect     theory: predicted failure 1 - Phi(1.645 / sqrt(D)), D = 1 + (K - 1) rho (DEPENDENCE.md
                    Change 2; experiments/wbcp/dependence.py predicted_failure).
  dose_curve        BCA's dose map m = 1 + b U / (U + u), b = 0.5, with the band where the five healthy TD3+BC pools'
                    real rows land (fig_bc_dose panel a).

Windows Python, NumPy / SciPy / Matplotlib only (no JAX). The heaviest chart (three_guarantees, 500 calibrate()
calls) takes about 20 s.
"""

import functools
import math
import sys
from statistics import NormalDist

import numpy as np

from experiments.wbcp.viz import style as S

if str(S.ROOT) not in sys.path:
    sys.path.insert(0, str(S.ROOT))

# type sizes on the 16.64 x 6.4 in deck canvas (points); nothing below 20
BIG, MID, TXT = 34, 26, 22
TAG = 20
GREY_BAR = "#CDD1D6"  # muted bars: lighter than S.MUTED so the highlighted bars and lines stand out

POOL_DIR = "runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000"
SRC_POOL = POOL_DIR + "/frozen.npz (heldout population, 500,285 rows, 1,094 episodes)"


def _tag(ax, text="Illustration", left=False):
    ax.text(0.008 if left else 0.995, 0.995, text, transform=ax.transAxes, ha="left" if left else "right", va="top",
            fontsize=TAG, color=S.MUTED,
            style="italic", bbox=dict(boxstyle="round,pad=0.25", facecolor=S.CARD, edgecolor=S.RULE, lw=1.0))


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


# ---------------------------------------------------------------------------------------------------------------
# shared data
# ---------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _hopper():
    from experiments.wbcp.d4rl_benchmark import ExactRisk, episode_groups, load_pool
    pool = load_pool(str(S.ROOT / POOL_DIR), "heldout")
    groups = episode_groups(pool.episode, pool.timestep)
    scores = pool.scores["normalized"]
    order = np.argsort(scores, kind="stable")
    risk = ExactRisk(scores, np.ones(pool.size), order)
    return pool, groups, risk


BANK_N = 1024  # configs/td3_bc.yaml datasets.hopper.reservation.size (fig_wbcp_threshold.SIZES[-1])


@functools.lru_cache(maxsize=None)
def _shown_bank():
    """The configured-size hopper bank of fig_wbcp_threshold (K = 6, BCA's sampler, seed 2026093001, trial 0)."""
    from experiments.wbcp.viz import fig_wbcp_threshold as TH
    pool, groups, risk = _hopper()
    bank = TH._bank(pool, groups, BANK_N)
    return TH, pool, bank, float(risk.lambda_star(TH.ALPHA))


def _bank_facts(TH, pool, bank):
    cal = bank["cal"]
    return [
        dict(what="Score pool: frozen TD3+BC hopper-medium-v2 critic (100k updates), held-out rows / episodes",
             value=f"{pool.size:,} rows, {np.unique(pool.episode).size:,} episodes", source=SRC_POOL),
        dict(what="Bank: BCA's sampler, configured hopper size and rows per episode",
             value=f"n = {bank['n']:,}, K = {TH.K}, {bank['episodes']} episodes",
             source="calibration/bank.py stratified_bank via d4rl_benchmark.draw_calibration(..., 'reservation'); "
                    "configs/td3_bc.yaml datasets.hopper.reservation; stream seed 2026093001 trial 0 "
                    "(fig_wbcp_threshold.py)"),
        dict(what="Score", value="|y - min(Q1, Q2)| / sigma (normalized)", source=POOL_DIR + "/frozen.json"),
        dict(what="alpha, beta, posterior draws", value=f"{TH.ALPHA}, {TH.BETA}, {TH.DRAWS:,}",
             source="configs/td3_bc.yaml bca.alpha / credibility / draws"),
        dict(what="Deployed threshold max(lambda_hat, lambda_hpd)",
             value=f"{cal.threshold:.3f} (lambda_hat {cal.lambda_hat:.3f}, lambda_hpd {cal.lambda_hpd:.3f})",
             source="computed: calibration/wbcp.py calibrate(), uniform weights, as calibration/reference.py:176"),
        dict(what="True miss rate of the deployed threshold on the pool rows outside the bank",
             value=_pct(bank["miss_dep"], 2), source="computed: share of the other pool rows scoring above it"),
    ]


# ---------------------------------------------------------------------------------------------------------------
# 1  score_threshold
# ---------------------------------------------------------------------------------------------------------------

def _score_threshold():
    TH, pool, bank, lam_star = _shown_bank()
    cal = bank["cal"]
    scores = bank["scores"]
    lam = float(cal.threshold)
    above_bank = float(np.mean(scores > lam))
    x1 = 3.0
    beyond = int(np.sum(scores > x1))

    fig, ax = S.deck_figure()
    width = 0.05
    edges = np.unique(np.concatenate([np.arange(0.0, x1 + 1e-9, width), [lam]]))
    counts, edges = np.histogram(scores, bins=edges)
    left, right = edges[:-1], edges[1:]
    hi = left >= lam - 1e-12
    # density scale so the split bin at lambda is not drawn short
    heights = counts / (right - left) * width
    ax.bar(left[~hi], heights[~hi], width=(right - left)[~hi], align="edge", color=GREY_BAR, edgecolor=S.PAPER,
           lw=0.8, zorder=2)
    ax.bar(left[hi], heights[hi], width=(right - left)[hi], align="edge", color=S.ORANGE, edgecolor=S.PAPER,
           lw=0.8, zorder=2)
    top = float(heights.max())
    ax.axvline(lam, color=S.BLUE, lw=5, zorder=4)
    ax.text(lam + 0.04, top * 1.02, f"cut-off  λ = {lam:.2f}", color=S.BLUE, fontsize=BIG, fontweight="semibold",
            ha="left", va="top")
    ax.text(lam + 0.30, top * 0.50, f"{_pct(bank['miss_dep'])} of new rows above", color=S.ORANGE,
            fontsize=MID, fontweight="semibold", ha="left", va="center")
    ax.text(lam + 0.30, top * 0.37, "target: at most 10%", color=S.SOFT, fontsize=TXT, ha="left", va="center")
    ax.annotate("", xy=(lam + 0.33, top * 0.10), xytext=(lam + 0.33, top * 0.29),
                arrowprops=dict(arrowstyle="-|>", color=S.ORANGE, lw=2.2, mutation_scale=22))
    ax.text(x1 - 0.02, top * 0.10, f"+{beyond} rows\nup to {scores.max():.1f} →", color=S.SOFT, fontsize=TXT,
            ha="right", va="bottom", linespacing=1.1)
    ax.set_xlim(0, x1)
    ax.set_ylim(0, top * 1.06)
    ax.set_xlabel("Error score |y − q| / σ")
    ax.set_ylabel(f"Rows (of {bank['n']:,})")

    facts = _bank_facts(TH, pool, bank) + [
        dict(what="Bank rows above the deployed threshold", value=f"{_pct(above_bank)} ({int(np.sum(scores > lam))})",
             source="computed from the bank's scores"),
        dict(what="Histogram: bin width; x-axis cut", value=f"0.05 (the bin at lambda is split); {x1}: {beyond} "
             f"bank scores beyond it, largest {scores.max():.2f}", source="computed"),
        dict(what="Pool's true 90% point lambda* (not drawn)", value=f"{lam_star:.3f}",
             source="computed: d4rl_benchmark.ExactRisk.lambda_star(0.1); DEPENDENCE.md reports 1.064"),
    ]
    vals = dict(lam=lam, miss=bank["miss_dep"], n=bank["n"], above=above_bank, lam_hat=float(cal.lambda_hat))
    return fig, facts, vals


# ---------------------------------------------------------------------------------------------------------------
# 2  posterior_draws
# ---------------------------------------------------------------------------------------------------------------

CURVES = 25


def _posterior_draws():
    TH, pool, bank, _ = _shown_bank()
    cal = bank["cal"]
    sorted_scores, cdf, lam = TH._posterior_cdfs(bank)  # checked against calibrate()'s crossings
    hpd = float(cal.lambda_hpd)
    alpha = TH.ALPHA

    fig, ax = S.deck_figure()
    x0, x1, y0, y1 = 0.92, 1.30, 0.86, 0.94
    xs = np.concatenate([[x0 - 1.0], sorted_scores, [x1 + 1.0]])
    for m in range(CURVES):
        ys = np.concatenate([[0.0], cdf[m], [cdf[m, -1]]])
        ax.step(xs, ys, where="post", color=S.BLUE_LIGHT, lw=2.0, zorder=2)
    paper = dict(boxstyle="square,pad=0.15", facecolor=S.PAPER, edgecolor="none", alpha=0.92)
    ax.axhline(1 - alpha, color=S.INK, lw=2.0, ls=(0, (2, 2.5)), zorder=3)
    ax.text(x0 + 0.004, 1 - alpha - 0.0012, "90% line", color=S.INK, fontsize=TXT, ha="left", va="top", bbox=paper,
            zorder=6)
    ax.scatter(lam[:CURVES], np.full(CURVES, 1 - alpha), s=170, color=S.BLUE, edgecolor=S.PAPER, lw=1.2, zorder=5)
    ax.axvline(hpd, color=S.BLUE, lw=5, zorder=4)
    ax.text(hpd + 0.007, y1 - 0.0015, f"λ = {hpd:.2f}", color=S.BLUE, fontsize=BIG, fontweight="semibold",
            ha="left", va="top", bbox=paper, zorder=6)
    n_left = int(np.sum(lam <= hpd))
    ax.text(hpd + 0.007, y0 + 0.004, f"{n_left / lam.size:.0%} of {lam.size:,} draws\ncross left of it",
            color=S.BLUE, fontsize=MID, ha="left", va="bottom", linespacing=1.1, zorder=6)
    ax.text(x0 + 0.004, y1 - 0.0015, f"{CURVES} of {lam.size:,} posterior draws\ndots: each draw's cut-off",
            color=S.SOFT, fontsize=TXT, ha="left", va="top", linespacing=1.15, bbox=paper, zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_yticks([0.87, 0.90, 0.93], ["87%", "90%", "93%"])
    ax.set_xlabel("Cut-off λ (error score)")
    ax.set_ylabel("Rows below λ")

    shown = lam[:CURVES]
    facts = _bank_facts(TH, pool, bank) + [
        dict(what="Draws drawn: the first 25 rows of calibrate()'s exponentials; their crossings (min-max)",
             value=f"{CURVES} of {lam.size:,}; {shown.min():.3f}-{shown.max():.3f}",
             source="computed: fig_wbcp_threshold._posterior_cdfs (regenerated from the same stream, checked equal "
                    "to Calibration.crossings)"),
        dict(what="lambda_hpd = smallest bank score with at least 95% of crossings at or below it",
             value=f"{hpd:.3f}; {n_left} of {lam.size:,} crossings <= it",
             source="computed: calibration/wbcp.py Eq. (7); Calibration.crossings"),
        dict(what="Crossings, 5th / 50th percentile", value=f"{np.quantile(lam, 0.05):.3f} / "
             f"{np.quantile(lam, 0.5):.3f}", source="computed: Calibration.crossings"),
        dict(what="Bank's empirical 90% point lambda_hat (Eq. 1; not drawn)", value=f"{cal.lambda_hat:.3f}",
             source="computed: Calibration.lambda_hat"),
    ]
    vals = dict(hpd=hpd, share=n_left / lam.size, draws=lam.size, lo=float(np.quantile(lam, 0.05)),
                med=float(np.quantile(lam, 0.5)), lam_hat=float(cal.lambda_hat), miss=bank["miss_dep"])
    return fig, facts, vals


# ---------------------------------------------------------------------------------------------------------------
# 3  three_guarantees
# ---------------------------------------------------------------------------------------------------------------

G_BANKS, G_N, G_SEED = 500, 1103, 2026093001  # first 500 banks of runs/wbcp_bench/main.json (n = 1,103, iid)


def _three_guarantees():
    from calibration.wbcp import calibrate
    from experiments.wbcp import d4rl_benchmark as DB
    from experiments.wbcp.reproduce_table1 import rcps_index, weighted_crc
    pool, _, risk = _hopper()
    scores = pool.scores["normalized"]
    alpha, beta = 0.1, 0.95
    j = rcps_index(G_N, alpha, 1 - beta)
    ones = np.ones(G_N)
    lam = np.empty((G_BANKS, 3))
    for trial in range(G_BANKS):  # run_trial (d4rl_benchmark.py:439-475) with no shift, so exact weights are all 1
        rows = DB.draw_calibration(DB.stream(G_SEED, "cal", G_N, trial), G_N, pool.size)
        bank = scores[rows]
        ordered = np.sort(bank, kind="stable")
        lam[trial, 0] = weighted_crc(ordered, ones, 1.0, alpha)
        lam[trial, 1] = ordered[j]
        lam[trial, 2] = calibrate(bank, DB.stream(G_SEED, "bq", G_N, trial, "normalized"), alpha=alpha, beta=beta,
                                  draws=1000).threshold
    miss = 100 * risk(lam.ravel()).reshape(lam.shape)
    fail = (miss > 100 * alpha).mean(axis=0)
    mean = miss.mean(axis=0)

    fig, ax = S.deck_figure()
    rows_ = [  # top to bottom: the message reads from the marginal rule down to WBCP
        ("Marginal", "W-CRC", 0, S.ORANGE, S.ORANGE_LIGHT),
        ("High-probability", "RCPS", 1, S.MUTED, "#D5D8DC"),
        ("Posterior-credible", "WBCP", 2, S.BLUE, S.BLUE_LIGHT),
    ]
    bins = np.arange(6.0, 13.0 + 1e-9, 0.2)
    gap = 1.25
    peak = max(np.histogram(miss[:, i], bins=bins)[0].max() for i in range(3))
    for r, (kind, rule, i, strong, light) in enumerate(rows_):
        base = (len(rows_) - 1 - r) * gap
        counts, edges = np.histogram(miss[:, i], bins=bins)
        h = counts / peak
        lo_e, hi_e = edges[:-1], edges[1:]
        over = lo_e >= 100 * alpha - 1e-9
        ax.bar(lo_e[~over], h[~over], width=0.2, bottom=base, align="edge", color=light, edgecolor=S.PAPER, lw=0.6,
               zorder=2)
        ax.bar(lo_e[over], h[over], width=0.2, bottom=base, align="edge", color=strong, edgecolor=S.PAPER, lw=0.6,
               zorder=2)
        ax.plot([bins[0], bins[-1]], [base, base], color=S.RULE, lw=1.5, zorder=1)
        ax.text(5.85, base + 0.42, kind, color=S.INK if i != 1 else S.SOFT, fontsize=TXT + 2, fontweight="semibold",
                ha="right", va="center")
        ax.text(5.85, base + 0.12, rule, color=strong, fontsize=TXT, ha="right", va="center")
        share = fail[i]
        ax.text(13.15, base + 0.42, f"{100 * share:.0f}%" if share >= 0.1 else f"{100 * share:.1f}%",
                color=strong, fontsize=BIG, fontweight="semibold", ha="left", va="center")
        if r == 0:
            ax.text(13.15, base + 0.06, "of banks\nover 10%", color=S.SOFT, fontsize=TXT, ha="left", va="center",
                    linespacing=1.0)
    ax.plot([100 * alpha] * 2, [-0.08, 2 * gap + 1.04], color=S.INK, lw=2.4, ls=(0, (2, 2.5)), zorder=3)
    ax.text(100 * alpha, 2 * gap + 1.08, "10% target", color=S.INK, fontsize=TXT, ha="center", va="bottom")
    ax.set_xlim(3.0, 15.2)
    ax.set_ylim(-0.08, 2 * gap + 1.32)
    ax.set_xticks([6, 8, 10, 12], ["6%", "8%", "10%", "12%"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(bins[0], bins[-1])
    ax.set_xlabel("Bank's true miss rate", x=(9.5 - 3.0) / (15.2 - 3.0))

    src = (f"computed: the first {G_BANKS} banks of runs/wbcp_bench/main.json's design (seed {G_SEED}, "
           f"d4rl_benchmark.stream 'cal'/'bq', n = {G_N:,} iid rows, normalized score, no shift); true miss = "
           "ExactRisk over the whole pool")
    facts = [
        dict(what="Pool", value="frozen TD3+BC hopper-medium-v2, 500,285 held-out rows", source=SRC_POOL),
        dict(what="Banks", value=f"{G_BANKS} iid banks of {G_N:,} rows, no shift (exact weights all 1)", source=src),
        dict(what="Marginal rule: W-CRC (weighted_crc with weights 1, test mass 1 = split conformal)",
             value=f"{_pct(fail[0])} of banks over 10%; mean miss {mean[0]:.2f}%", source=src + "; "
             "experiments/wbcp/reproduce_table1.py weighted_crc"),
        dict(what="High-probability rule: RCPS (Hoeffding-Bentkus, delta = 0.05), fixed order statistic",
             value=f"{_pct(fail[1])} of banks over 10%; mean miss {mean[1]:.2f}%; index {j} of {G_N}",
             source=src + "; reproduce_table1.py rcps_index"),
        dict(what="Posterior-credible rule: WBCP (calibrate, uniform weights = BQ-CP), deployed threshold",
             value=f"{_pct(fail[2])} of banks over 10%; mean miss {mean[2]:.2f}%", source=src + "; calibration/wbcp.py"),
        dict(what="Benchmark, all 2,000 banks, density gamma 0 (W-CRC there uses estimated weights)",
             value="W-CRC 48.9% (mean 9.97%), RCPS 1.5% (8.16%), BQ-CP 4.85% (8.54%), WBCP 4.9% (8.55%)",
             source="runs/wbcp_bench/main.json blocks density/0.0/1103/normalized"),
        dict(what="Failure budget for the posterior-credible and high-probability rules", value="5% of banks",
             source="beta = 0.95, configs/td3_bc.yaml bca.credibility"),
        dict(what="Histogram bins", value="0.2 points of miss rate, 6-13%; bars at or above 10% in the strong colour",
             source="computed"),
    ]
    vals = dict(fail=fail, mean=mean)
    return fig, facts, vals


# ---------------------------------------------------------------------------------------------------------------
# 4  shift_tilt
# ---------------------------------------------------------------------------------------------------------------

def _shift_tilt():
    from experiments.wbcp.viz import fig_shift_weighting as SW
    bench, _ = SW._settings()
    alpha = bench["settings"]["alpha"]
    pool, scores, tilt, risk, uniform = SW._pool_and_tilt(bench)  # checked against main.json
    lam_cal, lam_test = float(uniform.lambda_star(alpha)), float(risk.lambda_star(alpha))
    miss_test = float(risk(np.array([lam_cal]))[0])

    fig, ax = S.deck_figure()
    xlo, xhi = 0.9, 1.65
    grid = np.linspace(xlo, xhi, 600)
    ax.plot(grid, 100 * uniform(grid), color=S.MUTED, lw=4, zorder=3)
    ax.plot(grid, 100 * risk(grid), color=S.ORANGE, lw=5, zorder=4)
    ax.axhline(100 * alpha, color=S.INK, lw=2.0, ls=(0, (2, 2.5)), zorder=2)
    ax.text(xhi, 100 * alpha - 0.35, "10% target", color=S.INK, fontsize=TXT, ha="right", va="top")
    ax.vlines(lam_cal, 100 * alpha, 100 * miss_test, color=S.ORANGE, lw=2.5, ls=(0, (1, 1.6)), zorder=3)
    ax.plot([lam_cal], [100 * alpha], "o", color=S.MUTED, ms=16, zorder=5, mec=S.PAPER, mew=1.5)
    ax.plot([lam_cal], [100 * miss_test], "o", color=S.ORANGE, ms=16, zorder=5, mec=S.PAPER, mew=1.5)
    ax.text(lam_cal + 0.018, 100 * miss_test + 0.15, f"{100 * miss_test:.0f}%", color=S.ORANGE, fontsize=BIG + 6,
            fontweight="semibold", ha="left", va="bottom")
    ax.text(lam_cal - 0.015, 100 * alpha - 0.45, f"same cut-off\nλ = {lam_cal:.2f}", color=S.SOFT, fontsize=TXT,
            ha="right", va="top", linespacing=1.1)
    ax.text(1.40, 100 * float(risk(np.array([1.40]))[0]) + 0.5, "shifted test", color=S.ORANGE, fontsize=MID,
            fontweight="semibold", ha="left", va="bottom")
    ax.text(1.40, 100 * float(uniform(np.array([1.40]))[0]) - 0.6, "calibration", color=S.MUTED, fontsize=MID,
            fontweight="semibold", ha="left", va="top")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(2, 19)
    ax.set_xticks([1.0, 1.2, 1.4, 1.6])
    ax.set_yticks([5, 10, 15], ["5%", "10%", "15%"])
    ax.set_xlabel("Cut-off λ (error score)")
    ax.set_ylabel("Rows above λ")

    src = ("recomputed with fig_shift_weighting._pool_and_tilt (d4rl_benchmark tilt_feature/make_tilt/ExactRisk on "
           f"{POOL_DIR}); checked against runs/wbcp_bench/main.json")
    facts = [
        dict(what="Pool", value=f"frozen TD3+BC hopper-medium-v2, {pool.size:,} held-out rows", source=SRC_POOL),
        dict(what="Shift", value="density tilt gamma = 1: row weight exp(gamma z), z = standardized log distance to "
             "the 10th nearest neighbour", source="DEPENDENCE.md Terms; runs/wbcp_bench/main.json settings"),
        dict(what="Calibration-law 90% point (the 'same cut-off')", value=f"{lam_cal:.4f}",
             source="runs/wbcp_bench/main.json lambda_star_uniform.normalized; " + src),
        dict(what="Share of shifted-test rows above it", value=_pct(miss_test, 2), source=src),
        dict(what="Shifted-test 90% point (not drawn)", value=f"{lam_test:.4f}",
             source="runs/wbcp_bench/main.json block density/1.0 lambda_star"),
        dict(what="Kish n_eff of the tilt over the pool", value=f"{tilt.n_eff:,.0f} of {pool.size:,}",
             source="runs/wbcp_bench/main.json block tilt_n_eff"),
    ]
    vals = dict(lam_cal=lam_cal, lam_test=lam_test, miss=miss_test)
    return fig, facts, vals


# ---------------------------------------------------------------------------------------------------------------
# 5  design_effect
# ---------------------------------------------------------------------------------------------------------------

RHOS = ((0.016, "hopper", S.BLUE), (0.05, "", S.MUTED), (0.12, "walker2d", S.ORANGE))
Z_BETA = NormalDist().inv_cdf(0.95)  # 1.645


def _predicted(k, rho):
    d = 1.0 + (np.asarray(k, float) - 1.0) * rho
    return 100 * (1.0 - np.vectorize(NormalDist().cdf)(Z_BETA / np.sqrt(d)))


def _design_effect():
    fig, ax = S.deck_figure()
    ks = np.geomspace(1, 100, 300)
    ends = {}
    for rho, name, color in RHOS:
        y = _predicted(ks, rho)
        ax.plot(ks, y, color=color, lw=5 if name else 4, zorder=4 if name else 3)
        ends[rho] = float(y[-1])
        label = f"ρ = {rho:g}" + (f"  {name}" if name else "")
        ax.text(105, y[-1], label, color=color, fontsize=MID if name else TXT, fontweight="semibold" if name else
                "normal", ha="left", va="center")
    ax.axhline(5, color=S.INK, lw=2.4, ls=(0, (5, 3)), zorder=2)
    ax.text(1.05, 4.5, "5% budget", color=S.INK, fontsize=TXT, ha="left", va="top")
    k_w = 23  # walker2d's configured K (configs/td3_bc.yaml; DEPENDENCE.md Change 10)
    y_w = float(_predicted(k_w, 0.12))
    ax.plot([k_w], [y_w], "o", color=S.ORANGE, ms=17, mec=S.PAPER, mew=1.5, zorder=6)
    ax.text(k_w / 1.12, y_w + 0.9, f"K = {k_w}: {y_w:.0f}%", color=S.ORANGE, fontsize=BIG, fontweight="semibold",
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(0, 35)
    ax.set_yticks([0, 5, 10, 20, 30], ["0", "5%", "10%", "20%", "30%"])
    ax.set_xlabel("Rows per episode K")
    ax.set_ylabel("Banks failing (predicted)")
    _tag(ax, "Theory:  D = 1 + (K − 1)ρ", left=True)

    facts = [
        dict(what="Predicted failure", value="1 - Phi(1.645 / sqrt(D)), D = 1 + (K - 1) rho",
             source="experiments/wbcp/DEPENDENCE.md Change 2 (lines 182-196); experiments/wbcp/dependence.py "
                    "predicted_failure"),
        dict(what="rho shown", value="0.016 (hopper-medium), 0.05, 0.12 (walker2d-medium-replay)",
             source="DEPENDENCE.md Changes 2/10/12 (normalized score, TD3+BC critic); pen-expert is 0.054"),
        dict(what="Predicted failure at K = 1 (independent rows)", value="5.0%", source="computed"),
        dict(what="Predicted failure at K = 100", value=", ".join(f"rho {r:g}: {ends[r]:.1f}%" for r, _, _ in RHOS),
             source="computed"),
        dict(what="walker2d, configured K = 23 (rho 0.12): D and predicted failure",
             value=f"D = {1 + 22 * 0.12:.2f}, {y_w:.1f}% (observed for the configured TD3+BC bank: 17.6%)",
             source="computed; observed from results_td3_bc.ipynb configured-bank table / DEPENDENCE.md Change 10"),
        dict(what="hopper, configured K = 6 (rho 0.016), predicted (not marked)",
             value=f"{float(_predicted(6, 0.016)):.1f}% (observed 4.5%)", source="computed; results_td3_bc.ipynb"),
    ]
    vals = dict(y_w=y_w, k_w=k_w, y_h6=float(_predicted(6, 0.016)), ends=ends)
    return fig, facts, vals


# ---------------------------------------------------------------------------------------------------------------
# 6  dose_curve
# ---------------------------------------------------------------------------------------------------------------

def _dose_curve():
    import yaml
    from experiments.wbcp.viz import fig_bc_dose as BD
    with open(S.ROOT / "configs" / "td3_bc.yaml", encoding="utf-8") as handle:
        blend = float(yaml.safe_load(handle)["bca"]["blend"])
    pools = BD.load_pools(blend)  # checks the dose port against the evaluation JSONs
    lo = min(np.quantile(p["x"], 0.05) for p in pools)
    hi = max(np.quantile(p["x"], 0.95) for p in pools)
    d_lo, d_hi = 1.0 + blend * lo / (1.0 + lo), 1.0 + blend * hi / (1.0 + hi)
    means = [float(p["dose"].mean()) for p in pools]
    sds = [float(p["dose"].std()) for p in pools]

    # every pool counts equally in the pooled per-row dose distribution drawn on the right
    width = 0.01
    bins = np.arange(1.0, 1.5 + 1e-9, width)
    pooled = sum(np.histogram(p["dose"], bins=bins)[0] / p["dose"].size for p in pools) / len(pools)
    centers = (bins[:-1] + bins[1:]) / 2
    cum = np.cumsum(pooled)
    p05, p95 = (float(centers[np.searchsorted(cum, q)]) for q in (0.05, 0.95))
    near = float(pooled[(centers >= 1.25) & (centers <= 1.35)].sum())

    fig, ax = S.deck_figure()
    xmax, ylo, yhi = 10.0, 0.96, 1.58
    xs = np.linspace(0.0, xmax, 500)
    ax.axvspan(lo, hi, color=S.BLUE_LIGHT, alpha=0.35, lw=0, zorder=1)
    ax.plot(xs, 1.0 + blend * xs / (1.0 + xs), color=S.BLUE, lw=5.5, zorder=4, solid_capstyle="round")
    band = np.linspace(lo, hi, 60)
    ax.plot(band, 1.0 + blend * band / (1.0 + band), color=S.INK, lw=7.5, zorder=5, solid_capstyle="round")
    right, span = 9.95, 3.6  # the histogram grows leftwards from the right edge
    keep = pooled > 0
    ax.barh(centers[keep], span * pooled[keep] / pooled.max(), height=width, left=right - span * pooled[keep] /
            pooled.max(), color=S.BLUE, edgecolor="none", zorder=3)
    ax.text(right - span - 0.25, centers[np.argmax(pooled)], "real rows ≈ 1.3", color=S.BLUE, fontsize=BIG,
            fontweight="semibold", ha="right", va="center", zorder=6)
    ax.axhline(1.0 + blend, color=S.INK, lw=2.4, ls=(0, (5, 3)), zorder=2)
    ax.text(xmax, 1.0 + blend + 0.012, f"cap {1 + blend:g}", color=S.INK, fontsize=TXT, ha="right", va="bottom")
    ax.axhline(1.0, color=S.MUTED, lw=2.0, zorder=2)
    ax.text(xmax, 1.0 + 0.012, "1 = host's own BC", color=S.SOFT, fontsize=TXT, ha="right", va="bottom")
    ax.text((lo + hi) / 2, ylo + 0.012, "90% of\nrows", color=S.SOFT, fontsize=TXT, ha="center", va="bottom",
            linespacing=1.0, zorder=6)
    ax.text(3.0, 1.11, r"$m = 1 + 0.5\,\dfrac{U}{U+u}$", color=S.BLUE, fontsize=MID + 2, ha="left", va="center")
    ax.set_xlim(0, xmax)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.set_xlabel("Calibrated width U / u")
    ax.set_ylabel("BC dose m")

    facts = [
        dict(what="Dose map and blend b", value=f"m = 1 + b U/(U+u), U = R u max(eta, 1e-6); b = {blend:g}, cap {1 + blend:g}",
             source="calibration/dose.py l.53-60, l.75-76; configs/td3_bc.yaml bca.blend"),
        dict(what="Pools", value="5 healthy frozen TD3+BC pools, Q1 signal: hopper, walker2d, halfcheetah, maze2d, "
             "pen-expert", source="runs/wbcp_signal/frozen/<pool>/heads.npz; SIGNAL_STUDY.md step 2"),
        dict(what="Threshold R per pool", value="mean deployed WBCP threshold over 4,000 iid banks",
             source="runs/wbcp_signal/frozen/<pool>/evaluation_b4000_s2026100201.json designs[iid].signals.q1"),
        dict(what="Shaded x band: union of per-pool 5th-95th percentiles of U/u; mapped dose band",
             value=f"U/u {lo:.2f}-{hi:.2f}; dose {d_lo:.3f}-{d_hi:.3f}",
             source="computed with fig_bc_dose.load_pools (float32 port of dose.py, checked against the JSONs)"),
        dict(what="Per-pool mean dose (range over 5 pools) / per-row SD range", value=f"{min(means):.3f}-{max(means):.3f}"
             f" / {min(sds):.3f}-{max(sds):.3f}", source="computed; SIGNAL_STUDY.md l.219-223 gives 1.262-1.327 "
             "and 0.017-0.050"),
        dict(what="Right-hand histogram: per-row dose of the 5 pools, each pool weighted equally (bin 0.01); "
                  "5th-95th percentile; share between 1.25 and 1.35",
             value=f"{p05:.3f}-{p95:.3f}; {_pct(near)}", source="computed with fig_bc_dose.load_pools"),
    ]
    vals = dict(d_lo=d_lo, d_hi=d_hi, m_lo=min(means), m_hi=max(means), sd_lo=min(sds), sd_hi=max(sds),
                blend=blend, near=near)
    return fig, facts, vals


# ---------------------------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------------------------

def score_threshold():
    return _score_threshold()[:2]


def posterior_draws():
    return _posterior_draws()[:2]


def three_guarantees():
    return _three_guarantees()[:2]


def shift_tilt():
    return _shift_tilt()[:2]


def design_effect():
    return _design_effect()[:2]


def dose_curve():
    return _dose_curve()[:2]


def _text(key, v):
    if key == "score_threshold":
        return ("A rule picks one cut-off",
                f"Cut-off λ = {v['lam']:.2f}: {_pct(v['miss'])} of new rows above",
                f"This is one real calibration bank: {v['n']:,} rows of the frozen TD3+BC hopper critic, drawn the "
                f"way BCA draws its bank (6 rows per episode). Each bar counts rows by their normalized critic error. "
                f"WBCP deploys one cut-off, λ = {v['lam']:.2f}; rows above it are the misses. On the hopper rows "
                f"outside the bank it misses {_pct(v['miss'])}, inside the 10% target.")
    if key == "posterior_draws":
        return ("How WBCP picks the cut-off",
                f"WBCP deploys λ = {v['hpd']:.2f}: {v['share']:.0%} of draws cross below",
                f"Same bank. Each light curve is one Bayesian-bootstrap draw of the bank's score distribution; 25 of "
                f"the {v['draws']:,} draws are shown. Each draw crosses the 90% line at its own cut-off; half of "
                f"them lie below {v['med']:.2f}. WBCP deploys λ = {v['hpd']:.2f}, the cut-off that 95% of the "
                f"draws cross at or below, so it is 95% sure that at most 10% of rows lie above. The plain 90% point "
                f"of the bank is {v['lam_hat']:.2f}.")
    if key == "three_guarantees":
        f, m = v["fail"], v["mean"]
        return ("Three kinds of guarantee",
                f"Marginal W-CRC: {100 * f[0]:.0f}% of banks over 10%; WBCP: {100 * f[2]:.1f}%",
                f"Same {G_BANKS} real hopper banks (1,103 independent rows, no shift), three rules. W-CRC "
                f"promises only that the average miss rate is at most 10%; it averages {m[0]:.1f}%, so "
                f"{100 * f[0]:.0f}% of banks land above 10%. That is by design, and it is why W-CRC fails our "
                f"5%-of-banks test. RCPS promises 95% of banks below 10% and is conservative ({100 * f[1]:.1f}% "
                f"over); WBCP's 95% posterior statement lands {100 * f[2]:.1f}% over here, and 4.85% over all "
                f"2,000 banks of the benchmark.")
    if key == "shift_tilt":
        return ("Under shift, the tail gets fatter",
                f"Same cut-off misses {100 * v['miss']:.0f}% of shifted rows, not 10%",
                f"Real hopper pool. The grey curve is the calibration law: {v['lam_cal']:.2f} leaves exactly 10% "
                f"of rows above. The orange curve reweights the same rows toward sparse, rarely visited states "
                f"(density tilt, gamma 1). At the same cut-off, {_pct(v['miss'])} of test rows lie above; the test "
                f"law needs {v['lam_test']:.2f}. Importance weights are how WBCP moves the cut-off.")
    if key == "design_effect":
        return ("Correlated rows shrink the bank",
                f"walker2d at K = {v['k_w']}: predicted {v['y_w']:.0f}% of banks fail",
                f"Theory, not data: rows from one episode are correlated, with intra-class correlation ρ, so K rows "
                f"per episode inflate the variance by D = 1 + (K − 1)ρ and the predicted failure is "
                f"1 − Φ(1.645/√D). Hopper's ρ of 0.016 stays near the 5% budget at small K; walker2d's 0.12 at its "
                f"configured K = {v['k_w']} predicts {v['y_w']:.0f}%, and the configured TD3+BC bank observed 17.6%.")
    if key == "dose_curve":
        return ("Real doses sit near 1.3",
                f"Possible 1 to 1.5; real per-pool means {v['m_lo']:.2f}–{v['m_hi']:.2f}",
                f"BCA turns the calibrated width into a multiplier on the host's behaviour-cloning term, capped at "
                f"{1 + v['blend']:g}. On the five healthy TD3+BC pools every pool's middle 90% of rows falls "
                f"between {v['d_lo']:.2f} and {v['d_hi']:.2f}, and {_pct(v['near'], 0)} of rows lie between 1.25 "
                f"and 1.35; pool means are {v['m_lo']:.2f}–{v['m_hi']:.2f} with row SD "
                f"{v['sd_lo']:.2f}–{v['sd_hi']:.2f}. So the dose acts close to a constant extra BC weight.")
    raise KeyError(key)


CHARTS = {
    "score_threshold": _score_threshold,
    "posterior_draws": _posterior_draws,
    "three_guarantees": _three_guarantees,
    "shift_tilt": _shift_tilt,
    "design_effect": _design_effect,
    "dose_curve": _dose_curve,
}


def make_all(keys=None):
    import matplotlib.pyplot as plt
    out = {}
    for key in keys or CHARTS:
        fig, facts, vals = CHARTS[key]()
        path = S.save_deck(fig, key)
        plt.close(fig)
        headline, takeaway, notes = _text(key, vals)
        out[key] = dict(png=str(path), headline=headline, takeaway=takeaway, notes=notes, facts=facts)
    return out


if __name__ == "__main__":
    import time
    chosen = sys.argv[1:] or None
    for k in chosen or CHARTS:
        t0 = time.time()
        res = make_all([k])[k]
        print(f"{k}: {time.time() - t0:.1f} s -> {res['png']}")
        print("  ", res["headline"], "|", res["takeaway"])
