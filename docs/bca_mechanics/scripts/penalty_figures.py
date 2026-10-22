"""Draw the two penalty figures of docs/bca_mechanics/PENALTY_EVIDENCE.md from the numbers penalty_cases.py writes.

  figures/penalty_iid_vs_shift.svg      BCA's penalty by condition (IID against shift), standard against IW, per seed,
                                        each where it was used: threshold, coverage, dose, width at the actor's command,
                                        Stage 2's lower-edge width, the harm switch's height.
  figures/penalty_iw_minus_standard.svg IW minus standard, per seed: under the IID test (estimated weights), under the
                                        declared shift, in the dose, per 5,000-update chunk, the mechanism on the 30
                                        recorded banks, and the return cost without a shift.

Usage (from the repository root; NumPy and Matplotlib):
  python docs/bca_mechanics/scripts/penalty_cases.py --root . --hosts <results.json> --stage2-copies <dir> \\
         --write-json docs/bca_mechanics/data/penalty_cases.json
  python docs/bca_mechanics/scripts/penalty_figures.py --cases docs/bca_mechanics/data/penalty_cases.json \\
         --recompute docs/bca_mechanics/data/penalty_recompute.json --out docs/bca_mechanics/figures [--png <dir>]

Colors: three categorical slots of the reference palette (blue = no shift / standard, aqua = mild shift, orange =
strong shift); validated for color-vision deficiency with the dataviz skill's validator (all pairs pass; aqua needs
direct labels, which every panel carries through its legend and axis text). Text uses text tokens, never series colors.
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

QUALITIES = ["random", "medium-replay", "medium", "medium-expert", "expert"]
SHORT = {"random": "random", "medium-replay": "medium-\nreplay", "medium": "medium", "medium-expert": "medium-\nexpert", "expert": "expert"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
JIT = np.array([-0.06, -0.03, 0.0, 0.03, 0.06])  # per-seed offsets inside a group

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "text.color": INK, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 8.5, "xtick.major.size": 0, "ytick.major.size": 0,
})


def dots(ax, x, ys, color, marker="o", filled=True, size=28, z=3, **kw):
    ys = np.asarray(ys, float)
    xs = x + JIT[: len(ys)]
    if marker == "x":                             # an unfilled marker: color is its stroke
        ax.scatter(xs, ys, s=size, marker=marker, color=color, linewidths=1.3, zorder=z, **kw)
        return
    ax.scatter(xs, ys, s=size, marker=marker, facecolors=color if filled else SURFACE, edgecolors=color,
               linewidths=1.3, zorder=z, **kw)


def handle(color, marker="o", filled=True, label=""):
    return Line2D([], [], linestyle="none", marker=marker, markersize=6.5, markerfacecolor=color if filled else SURFACE,
                  markeredgecolor=color, markeredgewidth=1.3, label=label)


def tidy(ax, left_only=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if left_only:
        ax.grid(axis="x", visible=False)


def note(ax, text, xy=(0.01, 0.98)):
    ax.text(*xy, text, transform=ax.transAxes, va="top", ha="left", fontsize=8, color=INK2, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9))


# ----------------------------------------------------------------------------------------- figure 1: by condition

def figure_iid_vs_shift(C, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.135, wspace=0.22, hspace=0.42)
    fig.suptitle("BCA's penalty by test condition, standard against importance-weighted (IW), one dot per seed",
                 x=0.05, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.05, 0.925, "IID = test states follow the training law; shift = the mild or strong current mix. The standard penalty is read "
             "from the same band in every condition, so only coverage moves;\nthe penalty rises only where a shift is declared through "
             "weights.", fontsize=9, color=INK2, va="bottom")

    # A. threshold ratio
    ax = axes[0, 0]
    v5, c7 = C["case5_v5"], C["case7"]
    for i, q in enumerate(QUALITIES):
        dots(ax, i - 0.3, v5[q]["exact_none"], BLUE)
        dots(ax, i - 0.1, c7[q]["ratio"], BLUE, filled=False)
        dots(ax, i + 0.1, v5[q]["exact_mild"], AQUA, marker="s")
        dots(ax, i + 0.3, v5[q]["exact_strong"], ORANGE, marker="^")
    ax.axhline(1, color=INK, lw=1, zorder=2)
    ax.set_xticks(range(5), [SHORT[q] for q in QUALITIES])
    ax.set_ylabel("threshold R, IW / standard (1 = standard)")
    ax.set_ylim(0.85, 2.4)
    ax.set_title("A. Threshold R: IW / standard (weighting test, TD3+BC's critic)")
    ax.legend(handles=[handle(BLUE, label="IID, exact weights (= 1 on 25 of 25)"), handle(BLUE, filled=False, label="IID, estimated weights"),
                       handle(AQUA, "s", label="mild shift, exact weights"), handle(ORANGE, "^", label="strong shift, exact weights")],
              loc="upper left", ncol=1)
    tidy(ax)

    # B. coverage
    ax = axes[0, 1]
    cov = C["case2_coverage"]
    for i, q in enumerate(QUALITIES):
        dots(ax, i - 0.25, cov[q]["iid_standard"], BLUE)
        dots(ax, i, cov[q]["strong_standard"], ORANGE)
        dots(ax, i + 0.25, cov[q]["strong_iw_exact"], ORANGE, marker="s", filled=False)
    ax.axhline(0.90, color=INK, lw=1, zorder=2)
    ax.text(-0.45, 0.902, "promise 0.90", fontsize=8, color=INK2, ha="left", va="bottom")
    ax.set_xticks(range(5), [SHORT[q] for q in QUALITIES])
    ax.set_ylabel("coverage of the band on test steps")
    ax.set_ylim(0.78, 0.98)
    ax.set_title("B. Coverage of the standard band, IID vs strong shift")
    ax.legend(handles=[handle(BLUE, label="IID, standard (mark M1)"), handle(ORANGE, label="strong shift, standard (W2)"),
                       handle(ORANGE, "s", filled=False, label="strong shift, IW exact weights (W1)")], loc="lower left")
    tidy(ax)

    # C. dose
    ax = axes[0, 2]
    groups = [("TD3+BC\nweighting test\n(seed 0, last chunk)", None), ("TD3+BC\nmedium-expert\n(four-learner)", "td3_bc medium-expert"),
              ("CQL\nmedium", "cql medium"), ("CQL\nmedium-expert", "cql medium-expert"), ("ReBRAC\nmedium", "rebrac medium")]
    hosts, std1 = C["case5_hosts"], C["case1_hosts_standard_dose"]
    for i, (lab, key) in enumerate(groups):
        if key is None:
            s = [C["case8_chunks"][q]["standard"][-1] for q in QUALITIES]
            w = [C["case8_chunks"][q]["iw"][-1] for q in QUALITIES]
            dots(ax, i - 0.15, s, BLUE)
            dots(ax, i + 0.15, w, ORANGE, marker="^")
            continue
        if key in hosts:
            h = hosts[key]
            dots(ax, i - 0.3, h["standard_dose"], BLUE)
            dots(ax, i - 0.1, [a + b for a, b in zip(h["standard_dose"], h["shuffled"])], MUTED, filled=False)
            dots(ax, i + 0.1, [a + b for a, b in zip(h["standard_dose"], h["estimated"])], ORANGE, marker="^", filled=False)
            dots(ax, i + 0.3, h["iw_dose"], ORANGE, marker="^")
        else:
            dots(ax, i, std1[key], BLUE)
    ax.axhline(1.5, color=INK, lw=1, zorder=2)
    ax.text(-0.45, 1.503, "largest possible dose 1.5", fontsize=8, color=INK2, ha="left", va="bottom")
    ax.axhline(1.0, color=GRID, lw=1)
    ax.set_xticks(range(5), [g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("mean dose on kept training rows")
    ax.set_ylim(1.0, 1.56)
    ax.set_title("C. Dose m = 1 + 0.5 U / (U + u) on the training rows (IID)")
    ax.legend(handles=[handle(BLUE, label="standard"), handle(ORANGE, "^", label="IW, exact strong weights"),
                       handle(ORANGE, "^", filled=False, label="IW, estimated strong weights"),
                       handle(MUTED, filled=False, label="placebo: shuffled weights")], loc="lower left")
    note(ax, "Weighting-test group: one dot per data\nquality (seed 0); two separate training runs.", (0.30, 0.42))
    tidy(ax)

    # D. width at the actor's command
    ax = axes[1, 0]
    ch = C["case2_channel"]
    for i, q in enumerate(QUALITIES):
        rows = ch[q]
        ref = np.array([r["width_within"] for r in rows])
        dots(ax, i - 0.3, [r["width_past"] for r in rows] / ref, BLUE, marker="^")
        dots(ax, i - 0.1, [r["width_within_w"] for r in rows] / ref, ORANGE)
        dots(ax, i + 0.1, [r["width_past_w"] for r in rows] / ref, ORANGE, marker="^")
        dots(ax, i + 0.3, [r["needed_past_strong"] for r in rows], MUTED, marker="x", size=34)
    ax.set_yscale("log")
    ax.axhline(1, color=INK, lw=1, zorder=2)
    ax.text(4.45, 0.95, "standard band within 80% = 1", fontsize=8, color=INK2, ha="right", va="top")
    ax.set_yticks([0.5, 1, 1.5, 2, 3, 5, 10, 20], ["0.5", "1", "1.5", "2", "3", "5", "10", "20"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.45, 40)
    ax.set_xticks(range(5), [SHORT[q] for q in QUALITIES])
    ax.set_ylabel("band width at the actor's command, relative")
    ax.set_title("D. Band width at the actor's command, seen vs unseen")
    ax.legend(handles=[handle(BLUE, "^", label="standard, commands past 80% throttle (unseen)"),
                       handle(ORANGE, label="IW strong weights, within 80%"), handle(ORANGE, "^", label="IW strong weights, past 80%"),
                       handle(MUTED, "x", label="width needed past 80% for 90% coverage (strong law)")], loc="upper left")
    tidy(ax)

    # E. stage 2
    ax = axes[1, 1]
    s2 = C.get("case3_stage2")
    xs = ["recorded\ncommand\n(T1)", "80% cap\n(T1)", "actor's harmful\ncommand (T1)", "80% cap\n(T3, shifted)", "actor's harmful\ncommand (T3)"]
    keys = ["T1_recorded_all", "T1_cap_harmful", "T1_command_harmful", "T3_cap_harmful", "T3_command_harmful"]
    if s2:
        for ver, col in (("uniform", BLUE), ("weighted", ORANGE)):
            for r in s2[ver]:
                ax.plot(range(5), [r[k] for k in keys], color=col, lw=1.6, marker="o", ms=4.5, alpha=0.9, zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3], ["0.2", "0.3", "0.5", "1", "2", "3"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0.18, 3.2)
    else:
        ax.text(0.5, 0.5, "Stage 2 width scores not read (penalty_cases.py --stage2-copies)", ha="center", transform=ax.transAxes)
    ax.set_xticks(range(5), xs, fontsize=7.5)
    ax.set_ylabel("band width U (critic value units)")
    ax.set_title("E. Stage 2 lower edge: width at three commands, same state")
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=4.5, label="uniform (standard) band"),
                       Line2D([], [], color=ORANGE, lw=1.6, marker="o", ms=4.5, label="classifier-weighted band + refusal term")],
              loc="upper left")
    note(ax, "Standard: command minus cap +0.0003 to +0.012\non every seed (5 seeds, exploratory).", (0.02, 0.80))
    tidy(ax)

    # F. harm switch height
    ax = axes[1, 2]
    sw = C["case4_switch"]
    settings = [("step 1", "main", "step 1\nmain, seeds 0-4"), ("near-80", "near80", "near-80\nsetting, 0-4"),
                ("near-80", "random_rows", "random-rows\nsetting, 0-4"), ("near-80", "main", "fresh seeds\n10-14 (pre-registered)")]
    top = 14.0
    for i, (study, setting, lab) in enumerate(settings):
        for world, col, off in (("no_stall", BLUE, -0.18), ("stall", ORANGE, 0.18)):
            rows = [r for r in sw if r["study"] == study and r["setting"] == setting and r["world"] == world]
            fin = [r["height"] for r in rows if math.isfinite(r["height"])]
            inf = [r for r in rows if not math.isfinite(r["height"])]
            if fin:
                dots(ax, i + off, fin, col)
            if inf:
                ax.scatter(i + off + JIT[: len(inf)], [top] * len(inf), s=40, marker="^", facecolors=SURFACE, edgecolors=col, linewidths=1.3, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.025, 20)
    ax.axhline(top, color=GRID, lw=1)
    ax.text(3.45, top * 1.15, "+inf = refused (18-29 rows in the group)", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(4), [s[2] for s in settings], fontsize=7.5)
    ax.set_ylabel("height R_tail - R_rest (critic value units)")
    ax.set_title("F. Harm switch height: harmless vs harmful world")
    ax.legend(handles=[handle(BLUE, label="no-stall world (harmless)"), handle(ORANGE, label="stall world (harmful)"),
                       handle(INK2, "^", filled=False, label="infinite")], loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3,
              columnspacing=1.0, handletextpad=0.4)
    tidy(ax)

    fig.text(0.05, 0.012, "Sources: outputs/shift_iw/v5/<quality>/report.json (A, B, C seed 0 arms from explore/routes2/diagnose/dose_chunks.json); "
             "origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json (C); outputs/shift_iw/explore/channel/results (D); "
             "runs/shift_explore/stage2 width scores (E);\nnear-80 switch.json files and restructure/step1/results/results.json (F). "
             "Rebuilt by docs/bca_mechanics/scripts/penalty_cases.py; drawn by penalty_figures.py.", fontsize=7.5, color=INK2)
    return fig


# ------------------------------------------------------------------------------------ figure 2: IW minus standard

def figure_iw_minus_standard(C, R, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.11, wspace=0.24, hspace=0.45)
    fig.suptitle("Importance-weighted BCA minus standard BCA, one dot per seed: when is the weighted penalty higher, and why",
                 x=0.05, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.05, 0.925, "With weights of 1 the two are identical to the last bit (50 of 50 seed-hosts, 4 of 4 learners, 30 of 30 recorded "
             "banks; not drawn). Under the IID test the only differences come from\nestimated weights (A1); under a declared shift the "
             "penalty rises on every seed (A2, B, E).", fontsize=9, color=INK2, va="bottom")
    v5, c7 = C["case5_v5"], C["case7"]

    # A1. IID, estimated weights
    ax = axes[0, 0]
    for i, q in enumerate(QUALITIES):
        r = np.array(c7[q]["ratio"])
        dots(ax, i, 100 * (r - 1), BLUE, filled=False)
        for j, v in enumerate(r):
            if abs(v - 1) > 0.05:
                ax.annotate(f"seed {j}: {100 * (v - 1):+.1f}%", (i + JIT[j], 100 * (v - 1)), textcoords="offset points",
                            xytext=(8, -3 if v < 1 else 3), fontsize=7.5, color=INK2, va="center")
    ax.axhline(0, color=INK, lw=1, zorder=2)
    for y in (-5, 5):
        ax.axhline(y, color=MUTED, lw=1, zorder=2)
    ax.text(4.45, 5.2, "+5%", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.text(4.45, -5.2, "-5%", fontsize=7.5, color=INK2, ha="right", va="top")
    ax.set_xticks(range(5), [SHORT[q] for q in QUALITIES])
    ax.set_ylabel("threshold change, % (IW / standard - 1)")
    ax.set_ylim(-11, 11)
    ax.set_title("A1. IID test, classifier-estimated weights (exact = 1)")
    cnt = C["case7_counts"]
    note(ax, f"higher on {cnt['higher']} of 25, lower on {cnt['lower']}, equal on {cnt['equal']}; beyond 5% on {cnt['beyond5']};\n"
             f"the sign follows the fitted slope on the current in {cnt['slope_sign_agrees']} of {cnt['judged']}", (0.02, 0.98))
    tidy(ax)

    # A2. declared shift
    ax = axes[0, 1]
    for i, q in enumerate(QUALITIES):
        dots(ax, i - 0.2, 100 * (np.array(v5[q]["exact_mild"]) - 1), AQUA, marker="s")
        dots(ax, i + 0.2, 100 * (np.array(v5[q]["exact_strong"]) - 1), ORANGE, marker="^")
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.set_xticks(range(5), [SHORT[q] for q in QUALITIES])
    ax.set_ylabel("threshold change, % (IW / standard - 1)")
    ax.set_ylim(-5, 135)
    ax.set_title("A2. Declared shift, exact weights, same held-out scores")
    ax.set_xlabel("")
    ax.legend(handles=[handle(AQUA, "s", label="mild shift (above 0 on 25 of 25)"), handle(ORANGE, "^", label="strong shift (above 0 on 25 of 25)")],
              loc="upper left")
    tidy(ax)

    # B. dose change
    ax = axes[0, 2]
    labels = [SHORT[q] for q in QUALITIES] + ["CQL\nmedium", "CQL\nmedium-\nexpert", "ReBRAC\nmedium"]
    for i, q in enumerate(QUALITIES):
        dots(ax, i, v5[q]["dose_change_strong"], ORANGE, marker="^")
    for k, key in enumerate(("cql medium", "cql medium-expert", "rebrac medium")):
        h = C["case10"][key]
        i = 5 + k
        dots(ax, i - 0.25, h["strong"], ORANGE, marker="^")
        dots(ax, i, h["estimated"], ORANGE, marker="^", filled=False)
        dots(ax, i + 0.25, h["shuffled"], MUTED, filled=False)
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.axvline(4.5, color=GRID, lw=1)
    ax.text(2, 0.205, "weighting test (TD3+BC), frozen band, R2", fontsize=7.5, color=INK2, ha="center", va="top")
    ax.text(6, 0.205, "four-learner study, last refresh", fontsize=7.5, color=INK2, ha="center", va="top")
    ax.set_xticks(range(8), labels, fontsize=7.5)
    ax.set_ylabel("dose change on kept training rows, IW - standard")
    ax.set_ylim(-0.03, 0.21)
    ax.set_title("B. Dose change on training rows, strong-shift weights")
    ax.legend(handles=[handle(ORANGE, "^", label="exact strong weights"), handle(ORANGE, "^", filled=False, label="estimated strong weights"),
                       handle(MUTED, filled=False, label="placebo: shuffled weights")], loc="center left", bbox_to_anchor=(0.0, 0.62))
    tidy(ax)

    # C. the mechanism on recorded banks
    ax = axes[1, 0]
    banks = R["banks"]
    cats = [("B\ntest atom\nonly", lambda b: [b["B_test_mass_only"]["ratio"]]),
            ("C\nstrong weights,\nrandom rows", lambda b: [b["C_shuffled_strong"]["ratio_mean"]]),
            ("C\nmild weights,\nrandom rows", lambda b: [b["C_shuffled_mild"]["ratio_mean"]]),
            ("D\nheaviest on\nlargest", lambda b: [b["D_sorted_strong"]["heaviest_on_largest_scores"]["ratio"]]),
            ("D\nheaviest on\nsmallest", lambda b: [b["D_sorted_strong"]["heaviest_on_smallest_scores"]["ratio"]]),
            ("E\nIID slopes,\nrandom rows", lambda b: [b["E_none_slopes"]["ratio_min"], b["E_none_slopes"]["ratio_max"]])]
    rng = np.random.default_rng(3)
    for i, (lab, f) in enumerate(cats):
        for b in banks:
            ys = f(b)
            x = i + rng.uniform(-0.16, 0.16)
            ax.scatter([x] * len(ys), ys, s=16, facecolors=BLUE, edgecolors=SURFACE, linewidths=0.6, zorder=3, alpha=0.85)
            if i == 1:
                ax.plot([x, x], [b["C_shuffled_strong"]["ratio_min"], b["C_shuffled_strong"]["ratio_max"]], color=BLUE, lw=0.8, alpha=0.35, zorder=2)
            if i == 5:
                ax.plot([x, x], ys, color=BLUE, lw=0.8, alpha=0.35, zorder=2)
    strong = [v for q in QUALITIES for v in v5[q]["exact_strong"]]
    est = [v for q in QUALITIES for v in c7[q]["ratio"]]
    ax.axhspan(min(strong), max(strong), color=ORANGE, alpha=0.12, lw=0, zorder=1)
    ax.axhspan(min(est), max(est), color=AQUA, alpha=0.18, lw=0, zorder=1)
    ax.axhline(1, color=INK, lw=1, zorder=2)
    ax.set_yscale("log")
    ax.set_yticks([0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5], ["0.4", "0.5", "0.7", "1", "1.5", "2", "3", "5"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.33, 6.5)
    ax.set_xticks(range(6), [c[0] for c in cats], fontsize=7)
    ax.set_ylabel("threshold ratio, weighted / unweighted")
    ax.set_title("C. Why: 30 recorded banks, one input changed at a time")
    ax.legend(handles=[handle(BLUE, label="one recorded bank (805-851 held-out scores)"),
                       Patch(facecolor=ORANGE, alpha=0.3, label="recorded: exact strong weights, 25 seeds (1.11-2.25)"),
                       Patch(facecolor=AQUA, alpha=0.35, label="recorded: estimated IID weights, 25 seeds (0.91-1.05)")],
              loc="lower left", fontsize=7.5)
    note(ax, f"C = smaller effective sample (Kish 0.35 n) + test atom: mean x{R['summary']['C_strong_ratio_mean']:.2f};\n"
             f"the rest of the recorded rise is the tilt toward high-error rows (between C and D).", (0.02, 0.98))
    tidy(ax)

    # D. return cost on IID starts
    ax = axes[1, 1]
    cost = C["case8_cost"]
    labels = [SHORT[q] for q in QUALITIES] + ["CQL\nmedium", "CQL\nmedium-\nexpert"]
    for i, q in enumerate(QUALITIES):
        dots(ax, i - 0.15, cost[q]["capped_return"], ORANGE, marker="^")
        dots(ax, i + 0.15, cost[q]["lower_return"], AQUA, marker="s")
    for k, key in enumerate(("cql medium", "cql medium-expert")):
        dots(ax, 5 + k, cost[key]["return_per_seed"], ORANGE, marker="^")
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.axvline(4.5, color=GRID, lw=1)
    ax.set_xticks(range(7), labels, fontsize=7.5)
    ax.set_ylabel("return on IID starts, IW - standard (return units)")
    ax.set_ylim(-23, 11)
    ax.set_title("D. Return cost without a shift (exploratory; 7.6 units = 1 point)")
    ax.legend(handles=[handle(ORANGE, "^", label="dose (capped pull), TD3+BC; CQL claim 3"), handle(AQUA, "s", label="lower edge, TD3+BC")],
              loc="lower right")
    tidy(ax)

    # E. per chunk
    ax = axes[1, 2]
    upd = np.arange(15000, 100001, 5000)
    ends = []
    for i, q in enumerate(QUALITIES):
        d = np.array(C["case8_chunks"][q]["iw"]) - np.array(C["case8_chunks"][q]["standard"])
        ax.plot(upd / 1000, d, color=ORDINAL[i], lw=2, solid_capstyle="round", zorder=3)
        ends.append([float(d[-1]), q])
    ends.sort()                                   # direct labels at the line ends, pushed apart where they would touch
    for k in range(1, len(ends)):
        if ends[k][0] - ends[k - 1][0] < 0.006:
            ends[k][0] = ends[k - 1][0] + 0.006
    for y, q in ends:
        ax.text(101.5, y, q, fontsize=7.5, color=INK2, va="center")
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.set_xlim(12, 122)
    ax.set_ylim(-0.01, 0.12)
    ax.set_xlabel("training update (thousands)")
    ax.set_ylabel("mean dose per 5,000 updates, IW arm - standard arm")
    ax.set_title("E. Per 5,000 updates: IW arm minus standard arm (seed 0)")
    ax.legend(handles=[Line2D([], [], color=ORDINAL[i], lw=2, label=QUALITIES[i]) for i in range(5)], loc="upper left", ncol=3,
              title="data quality (90 of 90 chunks above 0)", title_fontsize=7.5)
    tidy(ax)

    fig.text(0.05, 0.02, "Sources: outputs/shift_iw/v5/<quality>/report.json (A1, A2, B); origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json (B, D); "
             "docs/bca_mechanics/data/penalty_recompute.json, the pinned-environment recompute of runs/shift_explore/{near80,stage2} held-out scores (C);\n"
             "outputs/shift_iw/explore/RESULTS.json (D); explore/routes2/diagnose/dose_chunks.json (E). "
             "Rebuilt by docs/bca_mechanics/scripts/penalty_cases.py; drawn by penalty_figures.py.", fontsize=7.5, color=INK2)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--recompute", required=True)
    ap.add_argument("--out", required=True, help="folder for the two SVG files")
    ap.add_argument("--png", default=None, help="also write PNG previews here")
    args = ap.parse_args()
    C = json.load(open(args.cases, encoding="utf8"))
    R = json.load(open(args.recompute, encoding="utf8"))
    os.makedirs(args.out, exist_ok=True)
    for name, fn in (("penalty_iid_vs_shift", figure_iid_vs_shift), ("penalty_iw_minus_standard", figure_iw_minus_standard)):
        fig = fn(C, R, args.out) if fn is figure_iw_minus_standard else fn(C, args.out)
        p = os.path.join(args.out, f"{name}.svg")
        fig.savefig(p, format="svg", metadata={"Date": None})
        print("wrote", p)
        if args.png:
            os.makedirs(args.png, exist_ok=True)
            fig.savefig(os.path.join(args.png, f"{name}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
