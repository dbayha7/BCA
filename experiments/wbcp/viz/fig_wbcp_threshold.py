"""How WBCP turns a calibration bank into a threshold: Algorithm 1 run on real calibration banks.

Data. The frozen TD3+BC hopper-medium-v2 score pool runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000
(500,285 held-out transitions in 1,094 episodes; score = |y - min(Q1, Q2)| / sigma, frozen.json
score_definition), read with experiments/wbcp/d4rl_benchmark.load_pool. Banks are drawn as BCA draws its
held-out bank: calibration/bank.stratified_bank via d4rl_benchmark.draw_calibration (spacing 'reservation'),
K = 6 rows per episode as configs/td3_bc.yaml sets for hopper, from the benchmark's own random streams
(d4rl_benchmark.stream, seed 2026093001 = the dependence-evidence runs' benchmark_seed, trial 0, fixed
before looking at any result). Thresholds come from calibration/wbcp.calibrate with uniform weights,
alpha = 0.1, beta = 0.95 and 1,000 draws, exactly as calibration/reference.freeze_reference calls it.

Panels.
  a  For the n = 256 bank: the first 30 of calibrate()'s 1,000 Bayesian-bootstrap draws of the CDF of the
     sorted scores (masses E_i over sum_j E_j + E_{n+1}; the test atom E_{n+1} is never covered), the
     empirical CDF, the 1 - alpha line, and each draw's alpha-crossing lambda^(m). The exponentials are
     regenerated from the same stream and checked to reproduce calibrate()'s crossings exactly.
  b  All 1,000 crossings of that bank (the threshold posterior). The crossings are atoms on the bank's own
     sorted scores, so they are drawn as stems (draws per atom), not binned. Shown with lambda_hat (Eq. 1),
     lambda_hpd (Eq. 7) and the deployed max of the two, the pool's true 90% point lambda*
     (d4rl_benchmark.ExactRisk.lambda_star), and each threshold's true miscoverage on the pool rows outside
     the bank.
  c  Banks of n = 64, 256 and 1,024 (1,024 is the configured TD3+BC hopper size). Each column shows the
     posterior's range (thin line), its middle 90% (bar: 50th to 950th of 1,000 sorted crossings, so the top is
     lambda_hpd), lambda_hat, the deployed threshold and its true miscoverage on the rest of the pool. The panel
     shows one bank per n, not a failure rate.
"""

import math

import numpy as np

from calibration.wbcp import calibrate, crossings
from experiments.wbcp.d4rl_benchmark import ExactRisk, draw_calibration, episode_groups, load_pool, stream
from experiments.wbcp.viz import style as S

POOL = S.ROOT / "runs" / "wbcp_frozen" / "hopper-medium-v2-s202609171-u100000"
SEED, TRIAL = 2026093001, 0  # benchmark_seed of calibration/dependence_evidence.json; first trial
K = 6  # configs/td3_bc.yaml, datasets.hopper.reservation.rows_per_episode
SIZES = (64, 256, 1024)  # 1,024 = configs/td3_bc.yaml, datasets.hopper.reservation.size
SHOWN = 256  # the bank of panels a and b
ALPHA, BETA, DRAWS = 0.1, 0.95, 1000  # configs/td3_bc.yaml bca: alpha, credibility, draws
CURVES = 30  # posterior draws drawn in panel a: the first 30 rows of calibrate()'s exponentials
SCORE = "normalized"
SRC_POOL = "runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000/frozen.npz (heldout population)"


def _bank(pool, groups, n):
    """One BCA-style bank of n rows, calibrated as calibration/reference.py:176 does, plus its true risk."""
    rows = draw_calibration(stream(SEED, "cal", n, TRIAL), n, pool.size, groups, K, "reservation")
    scores = pool.scores[SCORE][rows]
    cal = calibrate(scores, stream(SEED, "bq", n, TRIAL, SCORE), alpha=ALPHA, beta=BETA, draws=DRAWS)
    rest = np.ones(pool.size, bool)
    rest[rows] = False
    outside = pool.scores[SCORE][rest]
    return dict(n=n, rows=rows, scores=scores, cal=cal, episodes=int(np.unique(pool.episode[rows]).size),
                miss_dep=float(np.mean(outside > cal.threshold)), miss_hat=float(np.mean(outside > cal.lambda_hat)))


def _posterior_cdfs(bank):
    """Algorithm 1's draws for this bank, regenerated: calibrate() makes exactly one exponential draw of shape
    (draws, n + 1) when draws <= 4e6 // (n + 1) (calibration/wbcp.py:105-110), from the same stream."""
    n = bank["n"]
    assert DRAWS <= 4_000_000 // (n + 1)
    exps = stream(SEED, "bq", n, TRIAL, SCORE).standard_exponential((DRAWS, n + 1))
    sorted_scores = np.sort(bank["scores"], kind="stable")
    ones = np.ones(n)
    lam = crossings(sorted_scores, ones, 1.0, exps, ALPHA)  # uniform weights and test mass, wbcp.py:69
    if not np.array_equal(np.sort(lam), bank["cal"].crossings):
        raise AssertionError("regenerated draws do not reproduce calibrate()'s crossings")
    mass = exps[:, :-1]
    cdf = np.cumsum(mass, axis=1) / (mass.sum(axis=1) + exps[:, -1])[:, None]
    return sorted_scores, cdf, lam


def _pct(x):
    return f"{100 * x:.1f}%"


def make():
    pool = load_pool(str(POOL), "heldout")
    groups = episode_groups(pool.episode, pool.timestep)
    risk = ExactRisk(pool.scores[SCORE], np.ones(pool.size))
    lam_star = risk.lambda_star(ALPHA)
    banks = {n: _bank(pool, groups, n) for n in SIZES}
    shown = banks[SHOWN]
    cal = shown["cal"]
    sorted_scores, cdf, lam = _posterior_cdfs(shown)

    fig, (ax_a, ax_b, ax_c) = S.figure(1, 3, gridspec_kw=dict(width_ratios=[1.12, 1.0, 1.0]))
    fig.get_layout_engine().set(w_pad=0.25, wspace=0.08)
    lam_hat_tex, lam_hpd_tex, lam_star_tex = r"$\hat\lambda$", r"$\lambda_{\rm hpd}$", r"$\lambda^{*}$"

    # a: Bayesian-bootstrap CDFs --------------------------------------------------------------------------
    x0, x1, y0 = 0.35, 2.05, 0.6
    xs = np.concatenate([[x0], sorted_scores, [x1 + 1]])
    for m in range(CURVES):
        ys = np.concatenate([[0.0], cdf[m], [cdf[m, -1]]])
        ax_a.step(xs, ys, where="post", color=S.BLUE_LIGHT, lw=1.4, alpha=0.9, zorder=2)
    emp = np.concatenate([[0.0], np.arange(1, SHOWN + 1) / SHOWN, [1.0]])
    ax_a.step(xs, emp, where="post", color=S.ORANGE, lw=3.2, zorder=4)
    ax_a.axhline(1 - ALPHA, color=S.INK, lw=2.0, ls=(0, (6, 4)), zorder=3)
    ax_a.scatter(lam[:CURVES], np.full(CURVES, 1 - ALPHA), s=46, color=S.BLUE, edgecolor=S.PAPER, lw=0.8, zorder=5)
    ax_a.scatter([cal.lambda_hat], [1 - ALPHA], s=110, marker="D", color=S.ORANGE, edgecolor=S.PAPER, lw=1.2,
                 zorder=6)
    ax_a.set_xlim(x0, x1)
    ax_a.set_ylim(y0, 1.045)
    ax_a.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_xlabel(r"score $\lambda$ = |y $-$ q| / $\sigma$")
    ax_a.set_ylabel(r"mass of scores $\leq\lambda$")
    ax_a.text(x0 + 0.04, 1 - ALPHA + 0.008, r"1 $-$ $\alpha$", ha="left", va="bottom", fontsize=S.SMALL,
              color=S.INK)
    ax_a.text(0.92, 0.735, f"{CURVES} of the {DRAWS:,}\nposterior draws", ha="left", va="center",
              fontsize=S.SMALL, color=S.BLUE, linespacing=1.15)
    ax_a.text(0.60, 0.635, "empirical CDF", ha="left", va="center", fontsize=S.SMALL, color=S.ORANGE)
    ax_a.annotate(r"their crossings $\lambda^{(m)}$", xy=(float(np.max(lam[:CURVES])), 1 - ALPHA),
                  xytext=(1.22, 0.835), ha="left", va="center", fontsize=S.SMALL, color=S.BLUE,
                  arrowprops=dict(arrowstyle="-", color=S.BLUE, lw=1.2, shrinkA=2, shrinkB=6))
    ax_a.annotate(lam_hat_tex, xy=(cal.lambda_hat, 1 - ALPHA), xytext=(0.86, 0.965),
                  ha="right", va="center", fontsize=S.SMALL, color=S.ORANGE,
                  arrowprops=dict(arrowstyle="-", color=S.ORANGE, lw=1.2, shrinkA=2, shrinkB=7))
    # Each draw tops out at 1 - V_{n+1}: the test atom sits at the worst-case loss (+inf), so it is never covered.
    # The visible gap at the right edge also holds the bank's scores beyond x1, so the label claims only "never 1".
    ax_a.axhline(1.0, color=S.MUTED, lw=1.2, zorder=1)
    ax_a.text(x1 - 0.02, 1.024, r"draws never reach 1: test atom at +$\infty$", ha="right", va="center",
              fontsize=S.SMALL, color=S.BLUE)
    S.panel_label(ax_a, f"a  Posterior CDFs, one bank (n = {SHOWN})")

    # b: threshold posterior: the crossings are atoms on the bank's own sorted scores ----------------------
    hpd, dep, hat = cal.lambda_hpd, cal.threshold, cal.lambda_hat
    values, counts = np.unique(lam, return_counts=True)
    inside = values <= hpd
    ax_b.vlines(values[inside], 0, counts[inside], color=S.BLUE_LIGHT, lw=3.2, zorder=4)  # data above the
    ax_b.vlines(values[~inside], 0, counts[~inside], color=S.MUTED, lw=3.2, zorder=4)  # reference lines
    top = float(counts.max())
    bx0, bx1 = 0.78, 1.50
    ax_b.set_xlim(bx0, bx1)
    ax_b.set_ylim(0, top * 1.75)
    ax_b.axvline(hat, color=S.ORANGE, lw=3.0, ls=(0, (5, 3)), zorder=3)
    ax_b.axvline(lam_star, color=S.INK, lw=2.2, ls=(0, (1.5, 2.5)), zorder=3)
    ax_b.axvline(dep, color=S.BLUE, lw=3.4, zorder=5)
    ytxt = top * 1.72
    ax_b.text(hat - 0.012, ytxt, f"{lam_hat_tex} (Eq. 1)\nmisses\n{_pct(shown['miss_hat'])}", ha="right",
              va="top", fontsize=S.SMALL, color=S.ORANGE, linespacing=1.15)
    ax_b.text(lam_star + 0.008, ytxt, f"{lam_star_tex}: true 90% point", ha="left", va="top",
              rotation=90, fontsize=S.SMALL, color=S.INK)
    ax_b.text(dep + 0.014, ytxt, f"{lam_hpd_tex} (Eq. 7)\n= deployed\nmisses {_pct(shown['miss_dep'])}",
              ha="left", va="top", fontsize=S.SMALL, color=S.BLUE, linespacing=1.15)
    n_above = int(np.sum(lam > hpd))
    ax_b.text(dep + 0.014, top * 0.55, f"{n_above} draws\nabove", ha="left", va="center", fontsize=S.SMALL,
              color=S.SOFT, linespacing=1.15)
    ax_b.set_xlabel(r"crossing $\lambda^{(m)}$ (a score of the bank)")
    ax_b.set_ylabel(f"draws (of {DRAWS:,})")
    S.panel_label(ax_b, "b  Threshold posterior")

    # c: the posterior narrows as the bank grows --------------------------------------------------------------
    xs_c = np.arange(len(SIZES), dtype=float)  # n grows 4x per step: equal spacing is a log scale
    for x, n in zip(xs_c, SIZES):
        c = banks[n]["cal"]
        post = c.crossings[np.isfinite(c.crossings)]
        low = float(c.crossings[math.ceil((1 - BETA) * DRAWS) - 1])  # 50th of 1,000: 5% of draws below
        ax_c.plot([x, x], [post.min(), post.max()], color=S.MUTED, lw=1.6, zorder=2, solid_capstyle="butt")
        ax_c.plot([x, x], [low, c.lambda_hpd], color=S.BLUE_LIGHT, lw=16, zorder=3, solid_capstyle="butt")
        ax_c.scatter([x], [c.lambda_hat], s=120, marker="D", color=S.ORANGE, edgecolor=S.PAPER, lw=1.2, zorder=5)
        ax_c.scatter([x], [c.threshold], s=150, color=S.BLUE, edgecolor=S.PAPER, lw=1.2, zorder=6)
        lead = "deployed, " if n == SIZES[0] else ""
        ax_c.text(x + 0.2, c.threshold, f"{lead}misses {_pct(banks[n]['miss_dep'])}", ha="left", va="center",
                  fontsize=S.SMALL, color=S.BLUE, zorder=7,
                  bbox=dict(boxstyle="square,pad=0.08", facecolor=S.PAPER, edgecolor="none"))
    first = banks[SIZES[0]]["cal"]
    ax_c.text(xs_c[0] + 0.2, 1.72, "middle 90%\nof the draws", ha="left", va="center", fontsize=S.SMALL,
              color=S.SOFT, linespacing=1.15)
    top_draw = float(np.max(first.crossings[np.isfinite(first.crossings)]))
    ax_c.text(xs_c[0] + 0.12, top_draw, "all draws", ha="left", va="top", fontsize=S.SMALL, color=S.SOFT)
    ax_c.text(xs_c[0] - 0.2, first.lambda_hat, lam_hat_tex, ha="right", va="center", fontsize=S.SMALL,
              color=S.ORANGE)
    ax_c.axhline(lam_star, color=S.INK, lw=2.2, ls=(0, (1.5, 2.5)), zorder=1)
    ax_c.text(xs_c[-1] + 1.05, lam_star - 0.03, lam_star_tex, ha="right", va="top", fontsize=S.SMALL, color=S.INK)
    ax_c.set_xlim(-0.75, xs_c[-1] + 1.1)
    ax_c.set_ylim(0.7, 3.05)
    ax_c.set_xticks(xs_c)
    ax_c.set_xticklabels([f"{n:,}" for n in SIZES])
    ax_c.set_xlabel(f"bank size n (K = {K} rows per episode)")
    ax_c.set_ylabel(r"threshold $\lambda$")
    S.panel_label(ax_c, "c  Bigger bank, tighter threshold")

    facts = _facts(pool, banks, lam_star, n_above)
    facts.append(dict(what=f"Panel a x-axis cut at {x1}: bank scores beyond it (markdown); largest bank score",
                      value=f"{int(np.sum(sorted_scores > x1))} of {SHOWN}; {sorted_scores[-1]:.2f}",
                      source="computed from the n = 256 bank"))
    return fig, facts


def _facts(pool, banks, lam_star, n_above):
    shown = banks[SHOWN]
    cal = shown["cal"]
    facts = [
        dict(what="Score pool: TD3+BC hopper-medium-v2, 100k updates, held-out rows / episodes",
             value=f"{pool.size:,} rows, {np.unique(pool.episode).size:,} episodes", source=SRC_POOL),
        dict(what="Bank sampler and rows per episode K", value=f"stratified_bank (reservation), K = {K}",
             source="calibration/bank.py stratified_bank; configs/td3_bc.yaml datasets.hopper.reservation"),
        dict(what="Random streams of the banks and posteriors",
             value=f"d4rl_benchmark.stream(seed {SEED}, 'cal'|'bq', n, trial {TRIAL})",
             source="experiments/wbcp/d4rl_benchmark.py stream(); seed = benchmark_seed in "
                    "calibration/dependence_evidence.json"),
        dict(what="alpha (1 - alpha line in panel a), beta, posterior draws M",
             value=f"alpha = {ALPHA}, 1 - alpha = {1 - ALPHA:.1f}, beta = {BETA}, M = {DRAWS:,}",
             source="configs/td3_bc.yaml bca.alpha / credibility / draws"),
        dict(what="Posterior CDF draws shown in panel a", value=f"first {CURVES} of {DRAWS:,}",
             source="computed: calibrate()'s exponentials regenerated from the same stream; crossings checked "
                    "equal to Calibration.crossings"),
        dict(what="Panel a/b bank: size and episodes", value=f"n = {SHOWN}, {shown['episodes']} episodes",
             source="computed with draw_calibration(..., per_episode=6, spacing='reservation')"),
        dict(what="Pool's true 90% point lambda* (smallest score with miscoverage <= 0.1 over the pool)",
             value=f"{lam_star:.3f}", source="computed: d4rl_benchmark.ExactRisk(...).lambda_star(0.1); "
                                             "DEPENDENCE.md reports 1.064 for this pool"),
        dict(what=f"Draws whose crossing lies above lambda_hpd (n = {SHOWN})", value=f"{n_above}",
             source="computed from Calibration.crossings"),
    ]
    for n in SIZES:
        b = banks[n]
        c = b["cal"]
        facts += [
            dict(what=f"n = {n:,} bank: lambda_hat (Eq. 1), lambda_hpd (Eq. 7), deployed max",
                 value=f"{c.lambda_hat:.3f}, {c.lambda_hpd:.3f}, {c.threshold:.3f} ({b['episodes']} episodes)",
                 source="computed: calibration/wbcp.py calibrate() (uniform weights, as calibration/reference.py)"),
            dict(what=f"n = {n:,} bank: true miscoverage of the deployed threshold on the rest of the pool",
                 value=_pct(b["miss_dep"]), source="computed: share of pool rows outside the bank scoring above it"),
            dict(what=f"n = {n:,} bank: true miscoverage of lambda_hat alone on the rest of the pool",
                 value=_pct(b["miss_hat"]), source="computed: share of pool rows outside the bank scoring above it"),
            dict(what=f"n = {n:,} bank: panel c bar = middle 90% of the draws (50th to 950th smallest crossing); "
                      "thin line = all draws (min to max)",
                 value=f"{c.crossings[math.ceil((1 - BETA) * DRAWS) - 1]:.3f} to {c.lambda_hpd:.3f}; "
                       f"{np.min(c.crossings):.3f} to {np.max(c.crossings):.3f}",
                 source="computed: Calibration.crossings"),
            dict(what=f"n = {n:,} bank: posterior sd of the crossings (sigma_post); draws that never cross",
                 value=f"{c.sigma_post:.3f}; {int(np.sum(~np.isfinite(c.crossings)))}",
                 source="computed: Calibration.sigma_post, Calibration.crossings"),
        ]
    return facts


if __name__ == "__main__":
    figure, found = make()
    print(S.save(figure, "wbcp_threshold"))
    for fact in found:
        print(fact)
