"""Why the calibration bank is thinned: rows of one episode are correlated, so whole-episode banks fail.

Panel a draws one bank of each kind on real episode lengths: the 1,094 held-out hopper-medium-v2 episodes of the
benchmark's frozen pool (runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000/frozen.npz, array `episode`). The thinned
bank is calibration/bank.py stratified_bank itself (K = 5, n = 1,103, the bank Change 9 of
experiments/wbcp/DEPENDENCE.md benchmarks); the whole-episode bank is the rule calibration/reference.py:83-86 keeps for
population splits (seeded permutation, whole episodes until n rows are covered), re-implemented here in three lines
because reference.py imports JAX. The numbers next to them are the benchmark's: whole episodes
runs/wbcp_dependence/hopper-u100000/e1_blocks.json (Changes 1 and 3), BCA's sampler
runs/wbcp_dependence/hopper-u100000-reservation/r1_resv_k5.json (Change 9), with the Monte Carlo design effects of
predictions_k5/predictions.json in that directory. The drawn thinned bank uses today's stratified_bank, which removes the
surplus over n at random (calibration/bank.py:97-101, REMAINDER_TRIM 2026-10-01), so 2 of its 221 episodes keep 4 rows;
r1_resv_k5 predates that trim (no `bank_trim` in its settings) and its banks held 1,105 rows.

Panel b is the intra-class correlation rho of the miss indicator 1{score > lambda*} (one-way ANOVA,
experiments/wbcp/dependence.py:42-52) for the five datasets whose TD3+BC critic converged (DEPENDENCE.md Changes 2, 10,
12; pen-cloned and pen-human are left out there because their critics diverged), read from the predictions.json files
under runs/wbcp_dependence.

Panel c places every no-shift benchmark run on those five pools with independent rows, the stratified K-per-episode
sampler, or whole episodes (1,103 rows on hopper, the bank of Changes 1-9; 1,024 elsewhere, the TD3+BC-family configured
size; 4,000 banks each)
at its Monte Carlo design effect D, against the share of banks whose threshold missed more than 10% of test rows. The
rule is the uniform-weight posterior (BQ-CP), normalized score; with no shift the exact WBCP weights are all one. The
curve is dependence.py's normal approximation, failure = 1 - Phi(z_beta / sqrt(D)) (dependence.py:102-103), computed
here with the standard library. D = 1 for independent rows by definition.
"""

import json
import math
import sys
from statistics import NormalDist

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from experiments.wbcp.viz import style as S

if str(S.ROOT) not in sys.path:
    sys.path.insert(0, str(S.ROOT))
from calibration import bank  # noqa: E402  (pure NumPy)

DEP = S.ROOT / "runs" / "wbcp_dependence"
POOL = S.ROOT / "runs" / "wbcp_frozen" / "hopper-medium-v2-s202609171-u100000" / "frozen.npz"
SEED = 911  # the reservation seed of BCA's configs (DEPENDENCE.md Change 9); any fixed seed draws a typical bank
N_HOPPER, K_SHOWN, SHOWN = 1103, 5, 6
BETA = 0.95
ARM = "BQ-CP"

HC, MZ, PE = "all-datasets/predictions_halfcheetah_small", "all-datasets/predictions_maze2d_small", \
    "all-datasets/predictions_penexpert_small"
WK_C10, WK_C12 = "other-datasets/predictions_walker2d_n1024", "all-datasets/predictions_walker2d_small"
# (label, D4RL name, bank size n, rho source, runs: (design, result file, prediction file, design name in it))
DATASETS = [
    ("hopper", "hopper-medium-v2", 1103, "hopper-u100000/predictions.json", [
        ("iid", "hopper-u100000/e0_iid.json", None, None),
        ("whole", "hopper-u100000/e1_blocks.json", "hopper-u100000/predictions.json", "whole episodes"),
    ] + [(f"K={k}", f"hopper-u100000-spacing/e6_strat_k{k}.json", "hopper-u100000-spacing/predictions.json",
          f"{k} per episode, stratified") for k in (2, 5, 10, 25, 50)]),
    ("halfcheetah", "halfcheetah-medium-expert-v2", 1024, HC + "/predictions.json", [
        ("iid", "all-datasets/halfcheetah_iid.json", None, None),
        ("whole", "all-datasets/halfcheetah_blocks.json", HC + "/predictions.json", "whole episodes"),
    ] + [(f"K={k}", f"all-datasets/halfcheetah_strat{k}.json", HC + "/predictions.json",
          f"{k} per episode, stratified") for k in (2, 5, 10)]),
    ("pen-expert", "pen-expert-v1", 1024, PE + "/predictions.json", [
        ("iid", "all-datasets/penexpert_iid.json", None, None),
        ("whole", "all-datasets/penexpert_blocks.json", PE + "/predictions.json", "whole episodes"),
    ] + [(f"K={k}", f"all-datasets/penexpert_strat{k}.json", PE + "/predictions.json",
          f"{k} per episode, stratified") for k in (2, 5, 10)]),
    ("maze2d", "maze2d-large-v1", 1024, MZ + "/predictions.json", [
        ("iid", "all-datasets/maze2d_iid.json", None, None),
        ("whole", "all-datasets/maze2d_blocks.json", MZ + "/predictions.json", "whole episodes"),
    ] + [(f"K={k}", f"all-datasets/maze2d_strat{k}.json", MZ + "/predictions.json",
          f"{k} per episode, stratified") for k in (2, 5, 10)]),
    # walker2d: each run is matched to the prediction file its write-up table cites (Change 10: K = 5, 23 and whole
    # episodes; Change 12: K = 2, 10).
    ("walker2d", "walker2d-medium-replay-v2", 1024, WK_C10 + "/predictions.json", [
        ("iid", "other-datasets/w_iid_n1024.json", None, None),
        ("whole", "other-datasets/w_blocks_n1024.json", WK_C10 + "/predictions.json", "whole episodes"),
        ("K=2", "all-datasets/walker2d_strat2.json", WK_C12 + "/predictions.json", "2 per episode, stratified"),
        ("K=5", "other-datasets/w_strat5_n1024.json", WK_C10 + "/predictions.json", "5 per episode, stratified"),
        ("K=10", "all-datasets/walker2d_strat10.json", WK_C12 + "/predictions.json", "10 per episode, stratified"),
        ("K=23", "other-datasets/w_strat23_n1024.json", WK_C10 + "/predictions.json", "23 per episode, stratified"),
    ]),
]


def _json(rel):
    with open(DEP / rel, encoding="utf-8") as handle:
        return json.load(handle)


def _failure(rel, n):
    """The arm's no-shift failure on the normalized score at bank size n: (rate, 95% CI, banks)."""
    for block in _json(rel)["blocks"]:
        if block["gamma"] == 0 and block["score"] == "normalized" and block["n"] == n:
            arm = block["arms"][ARM]
            return arm["fail"], arm["ci"], arm["trials"]
    raise KeyError(f"{rel}: no gamma = 0 normalized block at n = {n}")


def _design(rel, name, n):
    for row in _json(rel)["scores"]["normalized"]["designs"]:
        if row["design"] == name and row["n"] == n:
            return row
    raise KeyError(f"{rel}: no design {name!r} at n = {n}")


def _predicted_failure(d):
    """dependence.py:102-103, stats.norm.sf(stats.norm.isf(1 - beta) / sqrt(D)), via the standard library."""
    normal = NormalDist()
    return 1.0 - normal.cdf(normal.inv_cdf(BETA) / math.sqrt(d))


def _banks():
    """Real hopper episode lengths, one whole-episode bank and one stratified_bank draw (fixed seed)."""
    with np.load(POOL) as pool:
        episode = pool["episode"]
    _, lengths = np.unique(episode, return_counts=True)
    # calibration/reference.py:83-86 (rows_per_episode=None): seeded permutation, episodes until n rows are covered.
    order = np.random.default_rng(SEED).permutation(len(lengths))
    whole = order[: int(np.searchsorted(np.cumsum(lengths[order]), N_HOPPER)) + 1]
    thin, offsets = bank.stratified_bank(lengths, N_HOPPER, K_SHOWN, np.random.default_rng(SEED))
    return lengths, whole, thin, offsets


def _fmt_pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def make():
    facts = []

    def fact(what, value, source):
        facts.append(dict(what=what, value=value, source=source))

    fig, axes = S.figure(1, 3, gridspec_kw=dict(width_ratios=[1.32, 0.78, 1.12]))
    ax_a, ax_b, ax_c = axes

    # ---- a: one bank of each kind on real hopper episodes --------------------------------------------------------
    lengths, whole, thin, offsets = _banks()
    pool_src = "runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000/frozen.npz (episode)"
    k5 = "runs/wbcp_dependence/hopper-u100000-reservation/predictions_k5/predictions.json"
    blocks_fail, blocks_ci, blocks_trials = _failure("hopper-u100000/e1_blocks.json", N_HOPPER)
    resv_fail, resv_ci, resv_trials = _failure("hopper-u100000-reservation/r1_resv_k5.json", N_HOPPER)
    d_whole_row = _design("hopper-u100000-reservation/predictions_k5/predictions.json", "whole episodes", N_HOPPER)
    d_resv_row = _design("hopper-u100000-reservation/predictions_k5/predictions.json",
                         "5 per episode, BCA reservation", N_HOPPER)
    d_whole, mean_whole = d_whole_row["design_effect_mc"], d_whole_row["mean_bank_size"]
    n_eff_whole = mean_whole / d_whole
    d_resv = d_resv_row["design_effect_mc"]
    whole_rows = int(lengths[whole].sum())
    thin_rows = int(sum(len(o) for o in offsets))
    surplus = int(sum(min(K_SHOWN, int(lengths[ep])) for ep in thin)) - thin_rows
    fact("pool episodes (hopper-medium-v2 held-out half) and their length range", f"{len(lengths)} episodes, "
         f"{int(lengths.min())}-{int(lengths.max())} steps", pool_src)
    fact("whole-episode bank drawn (seed 911, calibration/reference.py:83-86 rule, n = 1,103)",
         f"{len(whole)} episodes of lengths {', '.join(str(int(x)) for x in lengths[whole])}; {whole_rows} rows",
         pool_src + " + calibration/reference.py:83-86")
    fact("thinned bank drawn (calibration/bank.py stratified_bank, K = 5, n = 1,103, seed 911)",
         f"{len(thin)} episodes, {thin_rows} rows; first {SHOWN} episode lengths "
         f"{', '.join(str(int(x)) for x in lengths[thin[:SHOWN]])}; rows kept in those 6: "
         f"{', '.join(str(len(o)) for o in offsets[:SHOWN])}", pool_src + " + calibration/bank.py:45-102")
    fact("thinned bank drawn: surplus rows removed by the trim (K x episodes - n)",
         f"{surplus} ({K_SHOWN} x {len(thin)} - {thin_rows})", "calibration/bank.py:96-101 (computed on the draw)")
    fact("whole-episode banks, hopper n = 1,103: failure (uniform BQ-CP, normalized, no shift)",
         f"{_fmt_pct(blocks_fail)} [{_fmt_pct(blocks_ci[0])}, {_fmt_pct(blocks_ci[1])}] of {blocks_trials} banks",
         "runs/wbcp_dependence/hopper-u100000/e1_blocks.json; DEPENDENCE.md Change 3")
    fact("whole-episode banks, hopper: Monte Carlo design effect D and mean bank size",
         f"D = {d_whole:.3f}, mean bank {mean_whole:.1f} rows", k5)
    fact("whole-episode bank effective size n / D", f"{mean_whole:.1f} / {d_whole:.3f} = {n_eff_whole:.1f} rows",
         k5 + " (computed)")
    fact("BCA's thinned bank (stratified_bank K = 5, n = 1,103): failure",
         f"{_fmt_pct(resv_fail)} [{_fmt_pct(resv_ci[0])}, {_fmt_pct(resv_ci[1])}] of {resv_trials} banks",
         "runs/wbcp_dependence/hopper-u100000-reservation/r1_resv_k5.json; DEPENDENCE.md Change 9")
    fact("BCA's thinned bank: Monte Carlo design effect D (finite 1,094-episode pool; shown as D ≈ 1)",
         f"{d_resv:.3f}", k5)

    # Vertical layout of panel a in text rows (y grows downward): a header, the bars, then notes, per bank.
    rows = dict(head_whole=0.0)
    whole_y = [1.0 + i for i in range(len(whole))]
    rows["note_whole1"] = whole_y[-1] + 1.0
    rows["note_whole2"] = rows["note_whole1"] + 0.85
    rows["head_thin"] = rows["note_whole2"] + 1.75
    thin_y = [rows["head_thin"] + 1.0 + i for i in range(SHOWN)]
    rows["more"] = thin_y[-1] + 0.95
    rows["note_thin1"] = rows["more"] + 0.95
    rows["note_thin2"] = rows["note_thin1"] + 0.85
    bar_h = 0.56
    text_x = 830

    for yy, ep in zip(whole_y, whole):
        ax_a.barh(yy, lengths[ep], height=bar_h, left=0, color=S.ORANGE, edgecolor="none")
    for yy, ep, off in zip(thin_y, thin[:SHOWN], offsets[:SHOWN]):
        ax_a.barh(yy, lengths[ep], height=bar_h, left=0, color=S.RULE, edgecolor="none")
        ax_a.scatter(off + 0.5, np.full(len(off), yy), s=95, color=S.BLUE, zorder=3, edgecolor=S.PAPER,
                     linewidth=1.2)

    def line(key, text, color=S.SOFT, size=S.SMALL, weight="normal"):
        ax_a.text(0, rows[key], text, color=color, fontsize=size, fontweight=weight, va="center", ha="left")

    line("head_whole", f"Whole episodes: all {whole_rows:,} rows of {len(whole)}", S.ORANGE, S.BASE, "semibold")
    line("note_whole1", f"D = {d_whole:.1f}: a mean bank of {mean_whole:,.0f} rows")
    line("note_whole2", f"counts as only {n_eff_whole:.0f} independent rows")
    line("head_thin", f"Thinned: ≤ {K_SHOWN} spaced rows from each of {len(thin)}", S.BLUE, S.BASE, "semibold")
    line("more", f"… and {len(thin) - SHOWN} more ({surplus} surplus rows dropped at random)", S.MUTED)
    line("note_thin1", "D ≈ 1: as good as independent rows")
    line("note_thin2", "grey: withheld from training; dots: the bank")

    def score(yc, rate, color):
        ax_a.text(text_x, yc - 0.35, _fmt_pct(rate), color=color, fontsize=30, fontweight="semibold",
                  va="center", ha="left")
        ax_a.text(text_x, yc + 0.85, "of banks fail", color=S.SOFT, fontsize=S.SMALL, va="center", ha="left")

    score(np.mean(whole_y), blocks_fail, S.ORANGE)
    score(np.mean(thin_y), resv_fail, S.BLUE)

    ax_a.set_xlim(0, 1180)
    ax_a.set_ylim(rows["note_thin2"] + 0.7, rows["head_whole"] - 0.7)
    ax_a.spines["left"].set_visible(False)
    ax_a.spines["bottom"].set_bounds(0, 800)
    ax_a.yaxis.set_major_locator(NullLocator())
    ax_a.xaxis.set_major_locator(FixedLocator([0, 200, 400, 600, 800]))
    ax_a.set_xlabel("Step within the episode (hopper-medium)", loc="left")
    S.panel_label(ax_a, "a  One bank of each kind, real episodes")

    # ---- b: measured rho per dataset ----------------------------------------------------------------------------
    rhos = []
    for label, d4rl, n, rho_src, _ in DATASETS:
        rho = _json(rho_src)["scores"]["normalized"]["icc"]
        rhos.append((label, rho))
        fact(f"rho (ICC of the miss indicator), {d4rl}", f"{rho:.4f} (shown {rho:.3f})",
             f"runs/wbcp_dependence/{rho_src}; DEPENDENCE.md Changes 2, 10, 12")
    ypos = np.arange(len(rhos))
    for yy, (label, rho) in zip(ypos, rhos):
        ax_b.plot([0, rho], [yy, yy], color=S.ORANGE_LIGHT, lw=7, solid_capstyle="butt")
        ax_b.scatter([rho], [yy], s=150, color=S.ORANGE, zorder=3)
        ax_b.text(rho + 0.007, yy, f"{rho:.3f}", va="center", ha="left", fontsize=S.TICK, color=S.INK)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([label for label, _ in rhos], fontsize=S.TICK, color=S.INK)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_ylim(len(rhos) - 0.4, -1.9)
    ax_b.set_xlim(0, 0.16)
    ax_b.xaxis.set_major_locator(FixedLocator([0, 0.05, 0.10]))
    ax_b.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0")
    ax_b.spines["left"].set_visible(False)
    ax_b.set_xlabel("ρ, correlation of two rows'\nmisses within one episode", loc="left")
    ax_b.text(0.0, -1.25, "D ≈ 1 + (m − 1) ρ", fontsize=S.BASE, color=S.INK, va="center", ha="left")
    ax_b.text(0.0, -0.62, "for m rows per episode", fontsize=S.SMALL, color=S.SOFT, va="center", ha="left")
    S.panel_label(ax_b, "b  Within-episode correlation")

    # ---- c: failure against design effect -------------------------------------------------------------------------
    grid = np.geomspace(0.8, 120, 300)
    ax_c.plot(grid, [100 * _predicted_failure(d) for d in grid], color=S.INK, lw=2.0, zorder=1)
    ax_c.axhline(5, color=S.INK, lw=1.6, ls=(0, (4, 3)), zorder=1)
    points = []
    for label, d4rl, n, _, runs in DATASETS:
        for design, result, pred, name in runs:
            rate, ci, trials = _failure(result, n)
            if pred is None:
                d, d_src = 1.0, "D = 1 by definition (independent rows)"
            else:
                d = _design(pred, name, n)["design_effect_mc"]
                d_src = f"runs/wbcp_dependence/{pred} ({name}, n = {n})"
            points.append((design, d, rate, label))
            fact(f"{d4rl}, {design} bank, n = {n}: design effect D and failure",
                 f"D = {d:.3f}; fail {_fmt_pct(rate)} [{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}] of {trials} banks",
                 f"runs/wbcp_dependence/{result}; {d_src}")
    style = dict(iid=dict(color=S.PAPER, edgecolor=S.MUTED, s=120, linewidth=2.2, zorder=4),
                 whole=dict(color=S.ORANGE, edgecolor=S.PAPER, s=200, linewidth=1.2, zorder=3),
                 thin=dict(color=S.BLUE, edgecolor=S.PAPER, s=120, linewidth=1.2, zorder=3))
    for design, d, rate, _ in points:
        kind = design if design in ("iid", "whole") else "thin"
        ax_c.scatter([d], [100 * rate], **style[kind])

    ax_c.set_xscale("log")
    ax_c.set_xlim(0.8, 130)
    ax_c.set_ylim(0, 55)
    ax_c.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax_c.xaxis.set_minor_locator(NullLocator())
    ax_c.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax_c.yaxis.set_major_locator(FixedLocator([0, 5, 10, 20, 30, 40, 50]))
    ax_c.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax_c.set_xlabel("Design effect D (log scale)")
    ax_c.set_ylabel("Banks failing (test miss > 10%)")
    ax_c.text(1.0, 48.5, "whole episodes", color=S.ORANGE, fontsize=S.BASE, fontweight="semibold", ha="left",
              va="center")
    names = {label: (d, 100 * rate) for design, d, rate, label in points if design == "whole"}
    for label, (dx, dy), ha, va in [("walker2d", (-0.06, 0), "right", "center"),
                                    ("maze2d", (-0.06, 0), "right", "center"),
                                    ("halfcheetah", (0, -2.6), "center", "top"),
                                    ("hopper, pen-expert", (0.06, -2.2), "left", "top")]:
        key = label.split(",")[0]
        d, rate = names[key]
        ax_c.text(d * 10 ** dx, rate + dy, label, color=S.ORANGE, fontsize=S.SMALL, ha=ha, va=va)
    ax_c.text(2.7, 9.5, "K spaced rows per episode\n(K = 2 to 50)", color=S.BLUE, fontsize=S.BASE,
              fontweight="semibold", ha="left", va="center", linespacing=1.1)
    ax_c.text(0.86, 1.7, "independent rows", color=S.SOFT, fontsize=S.SMALL, ha="left", va="center")
    ax_c.text(125, 6.6, "5% allowed", color=S.INK, fontsize=S.SMALL, ha="right", va="bottom")
    ax_c.text(3.5, 15.3, "theory: 1 − Φ(1.645 / √D)", color=S.INK, fontsize=S.SMALL, ha="left", va="center")
    S.panel_label(ax_c, "c  Failure follows D on 5 datasets")

    fact("theory curve", "failure = 1 - Phi(1.645 / sqrt(D)); 1.645 = z at beta = 0.95",
         "experiments/wbcp/dependence.py:102-103; DEPENDENCE.md Change 2")
    fact("budget line", "5% of banks (1 - beta, beta = 0.95)", "calibration/bank.py:28 DEPENDENCE_BUDGET")
    fact("K values shown in c", "stratified K = 2, 5, 10 on every dataset; 25, 50 on hopper; 23 on walker2d",
         "runs/wbcp_dependence (files listed per point)")
    return fig, facts


if __name__ == "__main__":
    figure, listed = make()
    print(S.save(figure, "bank_dependence"))
    for item in listed:
        print(item)
