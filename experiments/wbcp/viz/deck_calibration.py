"""Deck charts, calibration group: is WBCP's calibration right? One message per chart, few words, big numbers.

Charts (key: message)
  archived_vs_wbcp  the archived BCA radius overspent the 5% budget; WBCP (with the test atom) stays under it
  paper_table1      Table 1 of Lou and Luo reproduces: every rule within 1.6 points of the paper
  paper_table2      Table 2 at n = 250 does not: WBCP fails 10.2% against the paper's 7.9%
  hopper_shift      on real hopper data, uniform BCA fails up to 100% of banks under shift; WBCP stays near 5%
  finite_sample     WBCP's excess under strong shift shrinks as the bank grows (exact weights)
  weights_cost      well-specified estimated weights match exact ones; misspecified ones fail 17-36%

Where the numbers come from (each chart also returns them as facts with their source)
  - Archived radius: experiments/wbcp/README.md, 'What changed relative to the archived BCA radius'. There is no run
    artifact behind these six numbers; they are typed in here.
  - Paper tables: our rows are read from runs/wbcp_paper/t1_g1.json and t2_g1_n250.json; the paper's rows are typed in
    from the same runs' .txt files and README.md (the paper prints them; no run produces them).
  - Hopper: runs/wbcp_bench/main.json (well-specified weights) and rawdisc.json (raw-feature discriminator), n = 1,103,
    normalized score, 2,000 banks per cell; runs/wbcp_dependence/hopper-u100000-weighting/a_*.json for the bank-size
    sweep (4,000 banks per cell). Every value read from a run is checked against the fact-checked deck values below.

Run from the repo root: python -m experiments.wbcp.viz.deck_calibration   (a few seconds; NumPy and matplotlib only)
"""

import json
import sys

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from experiments.wbcp.viz import style as S

if str(S.ROOT) not in sys.path:
    sys.path.insert(0, str(S.ROOT))

RUNS = S.ROOT / "runs"
T1 = RUNS / "wbcp_paper" / "t1_g1.json"
T2 = RUNS / "wbcp_paper" / "t2_g1_n250.json"
BENCH = RUNS / "wbcp_bench" / "main.json"
RAW = RUNS / "wbcp_bench" / "rawdisc.json"
SWEEP = RUNS / "wbcp_dependence" / "hopper-u100000-weighting"

VALUE = 28  # direct value labels
LABEL = 22  # series names and annotations
NOTE = 20  # the smallest text on a chart
BUDGET = 5.0
GREY_DARK = S.SOFT  # 'ours' for a muted group
GREY_LIGHT = "#C9CCD1"  # 'paper' for a muted group
ORANGE_DARK = "#9A5513"  # a second orange series, darker than S.ORANGE


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _rel(path):
    return path.relative_to(S.ROOT).as_posix()


def _check(name, mine, verified):
    """Every value read from a run must equal the fact-checked deck value (one decimal)."""
    if round(mine, 1) != verified:
        raise RuntimeError(f"{name}: run gives {mine:.3f}, the verified deck value is {verified}")
    return mine


def _bench_block(result, tilt, gamma, n=1103, score="normalized"):
    for block in result["blocks"]:
        if (block["tilt"], block["gamma"], block["n"], block["score"]) == (tilt, gamma, n, score):
            return block
    raise KeyError((tilt, gamma, n, score))


def _budget(ax, x, y=BUDGET, ha="right", va="bottom", dy=0.0, transform=None):
    """The 5% failure budget: a dashed INK line, labelled once."""
    ax.axhline(y, color=S.INK, lw=2.0, ls=(0, (5, 3)), zorder=1)
    ax.text(x, y + dy, "5% budget", ha=ha, va=va, fontsize=NOTE, color=S.INK, zorder=6,
            transform=transform or ax.transData)


def _bars(ax, xs, values, color, width, labels=True, label_color=None, fmt="{:.1f}", dy=0.6, size=VALUE,
          weight="semibold", **kw):
    ax.bar(xs, values, width=width, color=color, zorder=3, **kw)
    if labels:
        for x, v in zip(xs, values):
            ax.text(x, v + dy, fmt.format(v), ha="center", va="bottom", fontsize=size, fontweight=weight,
                    color=label_color or color, zorder=6, bbox=dict(facecolor=S.PAPER, edgecolor="none", pad=1.0))


def _clean(ax, grid=True):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=10)
    if grid:
        ax.yaxis.grid(True, color=S.RULE, lw=1.0, zorder=0)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------------------------------
def archived_vs_wbcp():
    """The archived radius (no test atom) against WBCP / BQ-CP, exchangeable data: failing trials by bank size."""
    sizes = ["n = 50", "n = 200", "n = 1,000"]
    archived = [9.7, 6.8, 5.6]
    wbcp = [3.4, 3.6, 4.4]
    src = "experiments/wbcp/README.md, 'What changed relative to the archived BCA radius' (no run artifact)"

    fig, ax = S.deck_figure()
    x = np.arange(3)
    w = 0.36
    _bars(ax, x - w / 2 - 0.01, archived, S.ORANGE, w, dy=0.2)
    _bars(ax, x + w / 2 + 0.01, wbcp, S.BLUE, w, labels=False)
    for xx, v in zip(x + w / 2 + 0.01, wbcp):  # inside the bar top: clear of the budget line at 5
        ax.text(xx, v - 0.25, f"{v:.1f}", ha="center", va="top", fontsize=VALUE, fontweight="semibold",
                color="white", zorder=6)
    # series names inside the first pair of bars
    ax.text(-w / 2 - 0.01, 0.45, "archived", ha="center", va="bottom", fontsize=LABEL, color="white",
            fontweight="semibold", zorder=6)
    ax.text(w / 2 + 0.01, 0.45, "WBCP", ha="center", va="bottom", fontsize=LABEL, color="white",
            fontweight="semibold", zorder=6)
    _budget(ax, 2.92, dy=0.12)
    ax.set_xlim(-0.55, 2.95)
    ax.set_ylim(0, 11.0)
    ax.set_yticks([0, 5, 10])
    ax.set_xticks(x, sizes)
    ax.set_ylabel("Trials failing (%)")
    _clean(ax)

    facts = [dict(what=f"Archived radius, exchangeable failure, {s}", value=f"{a}%", source=src)
             for s, a in zip(sizes, archived)]
    facts += [dict(what=f"WBCP / BQ-CP, exchangeable failure, {s}", value=f"{b}%", source=src)
              for s, b in zip(sizes, wbcp)]
    facts.append(dict(what="Archived radius: true credibility of its nominal 95%",
                      value="88.8% (n = 50), 94.3% (n = 200), 93.9% (n = 1,000)", source=src))
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
def _paired(ax, x, ours, paper, dark, light, w, ours_color=None, paper_color=None):
    _bars(ax, [x - w / 2 - 0.01], [ours], dark, w, label_color=ours_color or dark, dy=1.0)
    _bars(ax, [x + w / 2 + 0.01], [paper], light, w, label_color=paper_color or light, weight="normal", dy=1.0)


def paper_table1():
    """Table 1 (gamma = 1, n = 200): our failure rate against the paper's for the four rules."""
    run = _load(T1)["summary"]
    rules = ["BQ-CP", "RCPS", "W-CRC", "WBCP"]
    verified_ours = {"BQ-CP": 92.8, "RCPS": 85.7, "W-CRC": 41.2, "WBCP": 5.1}
    paper = {"BQ-CP": 92.3, "RCPS": 85.2, "W-CRC": 42.8, "WBCP": 5.0}  # printed in the paper
    ours = {r: _check(f"Table 1 {r}", 100 * run[r]["fail"], verified_ours[r]) for r in rules}

    fig, ax = S.deck_figure()
    w = 0.36
    for i, r in enumerate(rules):
        if r == "WBCP":
            _paired(ax, i, ours[r], paper[r], S.BLUE, S.BLUE_LIGHT, w, paper_color=S.BLUE)
        else:
            _paired(ax, i, ours[r], paper[r], GREY_DARK, GREY_LIGHT, w, paper_color=S.MUTED)
    ax.text(-w / 2 - 0.01, 78, "ours", ha="center", va="bottom", fontsize=LABEL, color="white",
            fontweight="semibold", zorder=6)
    ax.text(w / 2 + 0.01, 78, "paper", ha="center", va="bottom", fontsize=LABEL, color=S.INK, zorder=6)
    _budget(ax, 3.97, dy=1.2)
    ax.set_xlim(-0.55, 4.0)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(range(4), rules)
    ax.get_xticklabels()[-1].set(color=S.BLUE, fontweight="semibold")
    ax.set_ylabel("Trials failing (%)")
    _clean(ax)

    src_ours = f"{_rel(T1)} summary[rule].fail (10,000 trials)"
    src_paper = "Lou and Luo Table 1, as printed in runs/wbcp_paper/t1_g1.txt and experiments/wbcp/README.md"
    facts = []
    for r in rules:
        facts.append(dict(what=f"Table 1, gamma 1, n 200: {r} failure, ours", value=f"{ours[r]:.1f}%",
                          source=src_ours))
        facts.append(dict(what=f"Table 1, gamma 1, n 200: {r} failure, paper", value=f"{paper[r]:.1f}%",
                          source=src_paper))
    gap = max(abs(ours[r] - paper[r]) for r in rules)
    facts.append(dict(what="Largest gap ours vs paper over the four rules", value=f"{gap:.1f} points (W-CRC)",
                      source=src_ours + "; " + src_paper))
    facts.append(dict(what="Table 1 exact-weight row (Eq. 6 mass, not drawn), ours / paper",
                      value=f"{_check('Table 1 oracle', 100 * run['WBCP (oracle w)']['fail'], 4.9):.1f}% / 6.7%",
                      source=src_ours + "; " + src_paper))
    facts.append(dict(what="Table 1 WBCP failure, ours, 95% CI", value="5.1% [4.7, 5.5]",
                      source="runs/wbcp_paper/t1_g1.txt"))
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
def paper_table2():
    """Table 2 (count loss, gamma = 1, n = 250): WBCP with estimated and exact weights, ours against the paper."""
    run = _load(T2)["summary"]
    est = run["WBCP"]
    exact = run["WBCP (oracle w)"]
    ours_est = _check("Table 2 WBCP", 100 * est["fail"], 10.2)
    ours_exact = _check("Table 2 WBCP oracle", 100 * exact["fail"], 7.0)
    ci_est = [100 * c for c in est["ci"]]
    paper_est, paper_ci, paper_exact = 7.9, (7.4, 8.5), 5.6  # printed in the paper (README.md Table 2 section)
    wcrc_ours, wcrc_paper = 100 * run["W-CRC"]["fail"], 43.7

    fig, ax = S.deck_figure()
    # horizontal bars: rows top to bottom, WBCP (estimated weights) first, then exact weights
    h = 0.36
    rows = [(1.0, "WBCP", ours_est, paper_est, S.ORANGE, S.ORANGE_LIGHT, S.ORANGE, ci_est, paper_ci),
            (0.0, "WBCP,\nexact weights", ours_exact, paper_exact, GREY_DARK, GREY_LIGHT, S.MUTED, None, None)]
    for y, name, ours, paper, dark, light, paper_text, ci, pci in rows:
        for yy, v, color, txt, interval, weight in ((y + h / 2 + 0.01, ours, dark, dark, ci, "semibold"),
                                                    (y - h / 2 - 0.01, paper, light, paper_text, pci, "normal")):
            ax.barh(yy, v, height=h, color=color, zorder=3)
            end = v
            if interval is not None:
                ax.plot(interval, [yy, yy], color=S.INK, lw=2.4, zorder=5, solid_capstyle="butt")
                for e in interval:
                    ax.plot([e, e], [yy - 0.09, yy + 0.09], color=S.INK, lw=2.4, zorder=5)
                end = interval[1]
            ax.text(end + 0.25, yy, f"{v:.1f}", ha="left", va="center", fontsize=VALUE, fontweight=weight,
                    color=txt, zorder=6)
    ax.text(0.25, 1.0 + h / 2 + 0.01, "ours", ha="left", va="center", fontsize=LABEL, color="white",
            fontweight="semibold", zorder=6)
    ax.text(0.25, 1.0 - h / 2 - 0.01, "paper", ha="left", va="center", fontsize=LABEL, color=S.INK, zorder=6)
    ax.axvline(BUDGET, color=S.INK, lw=2.0, ls=(0, (5, 3)), zorder=4)
    ax.text(BUDGET + 0.12, 1.52, "5% budget", ha="left", va="bottom", fontsize=NOTE, color=S.INK)
    ax.set_ylim(-0.5, 1.62)
    ax.set_xlim(0, 12.6)
    ax.set_yticks([1.0, 0.0], ["WBCP", "WBCP,\nexact weights"])
    ax.tick_params(axis="y", labelsize=LABEL, length=0, pad=12)
    for label in ax.get_yticklabels():
        label.set_color(S.INK)
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel("Trials failing (%)")
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=S.RULE, lw=1.0, zorder=0)
    ax.set_axisbelow(True)

    src_ours = f"{_rel(T2)} summary (10,000 trials, count loss, alpha 0.4, K = 4)"
    src_paper = "Lou and Luo Table 2, as printed in runs/wbcp_paper/t2_g1_n250.txt and experiments/wbcp/README.md"
    facts = [
        dict(what="Table 2, gamma 1, n 250: WBCP (estimated weights) failure, ours [95% CI]",
             value=f"{ours_est:.1f}% [{ci_est[0]:.1f}, {ci_est[1]:.1f}]", source=src_ours),
        dict(what="Table 2, gamma 1, n 250: WBCP failure, paper [95% CI]",
             value=f"{paper_est}% [{paper_ci[0]}, {paper_ci[1]}]", source=src_paper),
        dict(what="Table 2, gamma 1, n 250: WBCP exact weights (Eq. 6 mass) failure, ours",
             value=f"{ours_exact:.1f}%", source=src_ours),
        dict(what="Table 2, gamma 1, n 250: WBCP oracle failure, paper", value=f"{paper_exact}%", source=src_paper),
        dict(what="W-CRC at n 250 (not drawn), ours vs paper", value=f"{wcrc_ours:.1f}% vs {wcrc_paper}%",
             source=src_ours + "; " + src_paper),
        dict(what="WBCP with 1,000-sample classifier and test-mass fits (not drawn)", value="7.6% [7.1, 8.2]; W-CRC 46.9%, exact-weight row 6.5%",
             source="runs/wbcp_paper/sens_fit1000_n250.txt; experiments/wbcp/README.md Table 2"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
HOPPER_TILTS = [  # (tick label, tilt, gamma, verified uniform, verified WBCP estimated); sorted by uniform failure
    ("none", "policy", 0.0, 4.9, 4.8),
    ("policy\nγ 1", "policy", 1.0, 5.5, 4.3),
    ("policy\nγ 2", "policy", 2.0, 7.7, 4.3),
    ("state\nγ 0.5", "state", 0.5, 24.9, 8.0),
    ("density\nγ 0.5", "density", 0.5, 42.6, 4.2),
    ("density\nγ 1", "density", 1.0, 89.5, 7.3),
    ("state\nγ 1", "state", 1.0, 98.4, 1.0),
    ("density\nγ 2", "density", 2.0, 100.0, 2.6),
]


def hopper_shift():
    """Hopper-medium, n = 1,103 independent rows: banks failing under each tilt, uniform BCA against WBCP.

    A broken axis: 0-10% at full height (WBCP and the budget live there), 10-100% compressed above it.
    """
    bench = _load(BENCH)
    uni, wb, abst = [], [], []
    for label, tilt, gamma, v_u, v_w in HOPPER_TILTS:
        arms = _bench_block(bench, tilt, gamma)["arms"]
        uni.append(_check(f"hopper {tilt} {gamma} BQ-CP", 100 * arms["BQ-CP"]["fail"], v_u))
        wb.append(_check(f"hopper {tilt} {gamma} WBCP", 100 * arms["WBCP"]["fail"], v_w))
        abst.append(100 * arms["WBCP"]["abstain"])

    fig, (top, bot) = S.deck_figure(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.25]))
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.03)
    x = np.arange(len(HOPPER_TILTS))
    w = 0.38
    lo, hi = (0, 10.6), (10.6, 128)
    for ax in (top, bot):
        ax.bar(x - w / 2 - 0.01, uni, width=w, color=S.ORANGE, zorder=3)
        ax.bar(x + w / 2 + 0.01, wb, width=w, color=S.BLUE, zorder=3)
    for i in range(len(x)):
        ax = top if uni[i] > lo[1] else bot
        dy = 2.0 if ax is top else 0.15
        ax.text(x[i] - w / 2 - 0.01, uni[i] + dy, f"{uni[i]:.0f}" if uni[i] >= 20 else f"{uni[i]:.1f}",
                ha="center", va="bottom", fontsize=VALUE - 2, fontweight="semibold", color=S.ORANGE, zorder=6,
                bbox=dict(facecolor=S.PAPER, edgecolor="none", pad=1.0))
        bot.text(x[i] + w / 2 + 0.01, wb[i] + 0.15, f"{wb[i]:.1f}", ha="center", va="bottom", fontsize=VALUE - 2,
                 fontweight="semibold", color=S.BLUE, zorder=6, bbox=dict(facecolor=S.PAPER, edgecolor="none", pad=1.0))
    # the two lowest WBCP rates are partly abstentions (counted as passes): say so on the bar
    for i in range(len(x)):
        if abst[i] >= 1.0:
            bot.text(x[i] + w / 2 + 0.01, 6.3, f"{abst[i]:.0f}%\nabstain", ha="center", va="bottom",
                     fontsize=NOTE, color=S.SOFT, linespacing=1.0, zorder=6)
    # direct key in the empty top-left corner (the first three tilts never reach 10%)
    top.text(-0.55, 112, "uniform BCA", ha="left", va="top", fontsize=LABEL + 4, fontweight="semibold",
             color=S.ORANGE)
    top.text(-0.55, 68, "WBCP", ha="left", va="top", fontsize=LABEL + 4, fontweight="semibold", color=S.BLUE)
    _budget(bot, len(x) + 0.2, ha="right", va="bottom", dy=0.15)
    top.set_ylim(*hi)
    bot.set_ylim(*lo)
    top.set_yticks([25, 50, 75, 100])
    bot.set_yticks([0, 5, 10])
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", length=0, labelbottom=False)
    bot.set_xlim(-0.6, len(x) + 0.22)
    bot.set_xticks(x, [t[0] for t in HOPPER_TILTS])
    for ax in (top, bot):
        _clean(ax)
    # break marks: short diagonal strokes on the y axis where the scale changes
    for ax, y in ((top, 0.0), (bot, 1.0)):
        ax.plot([-0.012, 0.012], [y - 0.035, y + 0.035], transform=ax.transAxes, color=S.SOFT, lw=2.0,
                clip_on=False)
    fig.supylabel("Banks failing (%)", fontsize=S.DECK_BASE, color=S.INK)

    src = f"{_rel(BENCH)} blocks (normalized score, n = 1,103, 2,000 banks; 'none' = policy gamma 0)"
    facts = []
    for (label, tilt, gamma, _, _), u, w_, a in zip(HOPPER_TILTS, uni, wb, abst):
        name = " ".join(label.split())
        facts.append(dict(what=f"Hopper {name}: uniform BCA (BQ-CP) banks failing", value=f"{u:.1f}%", source=src))
        facts.append(dict(what=f"Hopper {name}: WBCP (estimated weights) banks failing",
                          value=f"{w_:.1f}%" + (f" (abstains {a:.1f}%, counted as pass)" if a else ""),
                          source=src))
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
def finite_sample():
    """Exact-weight WBCP failure against bank size: two tilts aligned with the score and the policy control."""
    sizes = [1103, 2206, 4412, 8824]
    series = [  # (file, tilt, gamma, verified values)
        ("a_density1.json", "density", 1.0, [7.6, 6.3, 6.7, 6.0]),
        ("a_state05.json", "state", 0.5, [7.9, 7.0, 6.7, 6.0]),
        ("a_policy1.json", "policy", 1.0, [4.8, 5.1, 5.2, 4.9]),
    ]
    vals, cis = {}, {}
    for fname, tilt, gamma, verified in series:
        res = _load(SWEEP / fname)
        arms = [_bench_block(res, tilt, gamma, n)["arms"]["WBCP (oracle w)"] for n in sizes]
        vals[tilt] = [_check(f"sweep {tilt} n={n}", 100 * a["fail"], v) for a, n, v in zip(arms, sizes, verified)]
        cis[tilt] = [[100 * c for c in a["ci"]] for a in arms]

    fig, ax = S.deck_figure()
    style = {"density": (S.ORANGE, "o"), "state": (ORANGE_DARK, "s"), "policy": (S.MUTED, "o")}
    for tilt in ("policy", "state", "density"):  # density last: where the two meet, its marker matches the label
        color, marker = style[tilt]
        ax.plot(sizes, vals[tilt], color=color, lw=4.5 if tilt != "policy" else 3.5, marker=marker,
                ms=15 if tilt != "policy" else 12, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    d, s, p = vals["density"], vals["state"], vals["policy"]
    # values: where each aligned tilt starts and where both end
    ax.text(sizes[0] * 0.93, s[0], f"{s[0]:.1f}", ha="right", va="center", fontsize=VALUE, fontweight="semibold",
            color=ORANGE_DARK)
    ax.text(sizes[0] * 0.93, d[0] - 0.12, f"{d[0]:.1f}", ha="right", va="top", fontsize=VALUE,
            fontweight="semibold", color=S.ORANGE)
    ax.text(sizes[-1] * 1.07, d[-1], f"{d[-1]:.1f}", ha="left", va="center", fontsize=VALUE,
            fontweight="semibold", color=S.ORANGE)
    # series names along the lines
    ax.text(sizes[1], s[1] + 0.22, "state γ 0.5", ha="center", va="bottom", fontsize=LABEL, color=ORANGE_DARK,
            fontweight="semibold")
    ax.text(sizes[1] * 1.04, d[1] - 0.2, "density γ 1", ha="left", va="top", fontsize=LABEL, color=S.ORANGE,
            fontweight="semibold")
    ax.text(sizes[2], p[2] - 0.25, "control: policy γ 1", ha="center", va="top", fontsize=LABEL, color=S.MUTED)
    _budget(ax, sizes[0] / 1.4, ha="left", va="bottom", dy=0.06)
    ax.set_xscale("log", base=2)
    ax.set_xlim(sizes[0] / 1.42, sizes[-1] * 1.75)
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.set_ylim(3.0, 8.6)
    ax.yaxis.set_major_locator(FixedLocator([3, 4, 5, 6, 7, 8]))
    ax.set_xlabel("Bank size n")
    ax.set_ylabel("Banks failing (%)")
    _clean(ax)

    facts = []
    for fname, tilt, gamma, _ in series:
        src = f"{_rel(SWEEP / fname)} 'WBCP (oracle w)' (iid banks, normalized score, 4,000 banks per size); " \
              "DEPENDENCE.md Change 11 A"
        for n, v, ci in zip(sizes, vals[tilt], cis[tilt]):
            facts.append(dict(what=f"Exact-weight WBCP, {tilt} gamma {gamma:g}, n = {n:,}: banks failing [95% CI]",
                              value=f"{v:.1f}% [{ci[0]:.1f}, {ci[1]:.1f}]", source=src))
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
WEIGHT_TILTS = [  # (tick label, tilt, gamma, verified well-specified, misspecified, exact); sorted by misspecified
    ("state γ 0.5", "state", 0.5, 8.0, 16.6, 8.1),
    ("density γ 0.5", "density", 0.5, 4.2, 17.3, 4.3),
    ("state γ 1", "state", 1.0, 1.0, 28.0, 0.9),
    ("density γ 1", "density", 1.0, 7.3, 36.4, 7.5),
]


def weights_cost():
    """Hopper, n = 1,103: WBCP with exact, well-specified estimated and misspecified estimated weights."""
    bench, raw = _load(BENCH), _load(RAW)
    good, bad, exact, ab_good, ab_exact, ab_bad = [], [], [], [], [], []
    for label, tilt, gamma, v_g, v_b, v_e in WEIGHT_TILTS:
        g = _bench_block(bench, tilt, gamma)["arms"]
        b = _bench_block(raw, tilt, gamma)["arms"]
        good.append(_check(f"{label} well-specified", 100 * g["WBCP"]["fail"], v_g))
        bad.append(_check(f"{label} misspecified", 100 * b["WBCP"]["fail"], v_b))
        exact.append(_check(f"{label} exact", 100 * g["WBCP (oracle w)"]["fail"], v_e))
        ab_good.append(100 * g["WBCP"]["abstain"])
        ab_exact.append(100 * g["WBCP (oracle w)"]["abstain"])
        ab_bad.append(100 * b["WBCP"]["abstain"])

    fig, ax = S.deck_figure()
    x = np.arange(len(WEIGHT_TILTS))
    w = 0.26
    _bars(ax, x - w - 0.02, exact, GREY_LIGHT, w, label_color=S.MUTED, weight="normal", size=VALUE - 4)
    _bars(ax, x, good, S.BLUE, w, size=VALUE - 4)
    _bars(ax, x + w + 0.02, bad, S.ORANGE, w)
    # direct key, top left, in the series' own colours (stacked in the order the bars read: left to right)
    ax.text(-0.5, 38.5, "exact weights", ha="left", va="top", fontsize=LABEL, color=S.MUTED)
    ax.text(-0.5, 34.0, "estimated, well-specified", ha="left", va="top", fontsize=LABEL, color=S.BLUE,
            fontweight="semibold")
    ax.text(-0.5, 29.5, "estimated, misspecified", ha="left", va="top", fontsize=LABEL, color=S.ORANGE,
            fontweight="semibold")
    for i in range(len(x)):
        if ab_good[i] >= 1.0 or ab_exact[i] >= 1.0:
            ax.text(x[i] - w / 2 - 0.01, 6.6, f"{ab_exact[i]:.0f}% / {ab_good[i]:.0f}%\nabstain", ha="center",
                    va="bottom", fontsize=NOTE, color=S.SOFT, linespacing=1.0, zorder=6)
    _budget(ax, 3.98, dy=0.3)
    ax.set_xlim(-0.55, 4.0)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xticks(x, [t[0] for t in WEIGHT_TILTS])
    ax.set_ylabel("Banks failing (%)")
    _clean(ax)

    src = f"{_rel(BENCH)} (WBCP, 'WBCP (oracle w)') and {_rel(RAW)} (WBCP, raw-feature discriminator); " \
          "normalized score, n = 1,103, 2,000 banks"
    facts = []
    for (label, *_), g, b, e, ag, ae, abad in zip(WEIGHT_TILTS, good, bad, exact, ab_good, ab_exact, ab_bad):
        facts.append(dict(what=f"Hopper {label}: WBCP failing, exact / well-specified / misspecified weights",
                          value=f"{e:.1f}% / {g:.1f}% / {b:.1f}%", source=src))
        if ag or ae or abad:
            facts.append(dict(what=f"Hopper {label}: abstaining banks (counted as passes), exact / well-specified / "
                                   "misspecified", value=f"{ae:.1f}% / {ag:.1f}% / {abad:.1f}%", source=src))
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
SLIDES = {
    "archived_vs_wbcp": dict(
        make=archived_vs_wbcp,
        headline="The old radius overspent the budget",
        takeaway="Archived radius failed 9.7%; WBCP stays at 3.4-4.4%.",
        notes="Each bar is the share of trials, with no shift, whose threshold misses more than alpha; the budget "
              "is 5%. The archived radius drew its Bayesian bootstrap over the n observed scores only, with no test "
              "atom, so its nominal 95% was really 88.8% at n = 50, 94.3% at 200 and 93.9% at 1,000, and it failed "
              "9.7%, 6.8% and 5.6% of trials. Adding the test atom, which is WBCP with uniform weights (BQ-CP), "
              "gives 3.4%, 3.6% and 4.4%. The archived radius also certified every trial, even where the published "
              "rule abstains. These numbers are from the WBCP README; there is no run file behind them."),
    "paper_table1": dict(
        make=paper_table1,
        headline="Table 1: we match the paper",
        takeaway="WBCP fails 5.1% of trials; the paper reports 5.0%.",
        notes="This is the paper's synthetic regression under covariate shift: gamma 1, n = 200, 10,000 trials. "
              "Dark bars are our implementation, light bars the published numbers. Every rule lands within 1.6 "
              "points of the paper: BQ-CP 92.8 vs 92.3, RCPS 85.7 vs 85.2, W-CRC 41.2 vs 42.8, WBCP 5.1 vs 5.0. "
              "Only WBCP sits at the 5% budget (5.1%, interval 4.7 to 5.5). Not drawn: the exact-weight row, "
              "where we get 4.9% against the paper's 6.7%."),
    "paper_table2": dict(
        make=paper_table2,
        headline="Table 2, n = 250: WBCP lands high",
        takeaway="WBCP fails 10.2% here; the paper reports 7.9%.",
        notes="Table 2 is the count-loss experiment at gamma 1 and n = 250, 10,000 trials. Our WBCP fails 10.2% "
              "[9.6, 10.8], outside the paper's 7.9% [7.4, 8.5]; with exact weights we get 7.0% against 5.6%, so "
              "about 1.4 of the 2.3 points are there even without weight estimation, and the cause is not "
              "identified. With 1,000-sample fits for the weights WBCP drops to 7.6%, inside the paper's interval, "
              "but W-CRC (46.9 vs 43.7) and the exact-weight row (6.5 vs 5.6) do not come into line, so this "
              "does not show the paper used a bigger fit."),
    "hopper_shift": dict(
        make=hopper_shift,
        headline="Shift breaks uniform BCA; WBCP holds or abstains",
        takeaway="Uniform fails up to 100% of banks; WBCP at most 8%.",
        notes="Real hopper-medium data from D4RL, banks of 1,103 independent rows, 2,000 banks per tilt; the axis "
              "is broken at 10% and compressed above. Uniform BCA (BQ-CP) is fine with no shift (4.9%) but fails "
              "25% to 100% of banks once the test data tilt toward sparse or unusual states, while WBCP with "
              "estimated weights stays between 1.0% and 8.0%. At the two strongest tilts WBCP's low rate partly "
              "comes from abstaining: 31% and 12% of banks refuse to certify, which counts as a pass. State gamma 2 "
              "is left out: there WBCP abstains on every bank."),
    "finite_sample": dict(
        make=finite_sample,
        headline="The excess shrinks as the bank grows",
        takeaway="7.6-7.9% at n = 1,103 falls to 6.0% at 8x.",
        notes="Hopper, independent banks, exact weights, 4,000 banks per point. Under the two tilts aligned with "
              "the score WBCP fails 7.6% and 7.9% at BCA's bank size of 1,103, and 6.0% at 8,824; the policy "
              "control stays at 4.8-5.2%. The fall is not monotone (6.3 then 6.7 in between). A bank that happens to miss the few heavy-weight exceedances reports a "
              "spread about a third too small and certifies too tight a threshold. The excess shrinks as the bank "
              "grows, so it is a finite-sample effect of the method, not a bug; the paper reports one of the same "
              "size."),
    "weights_cost": dict(
        make=weights_cost,
        headline="Misspecified weights cost coverage; good ones don't",
        takeaway="Misspecified weights fail 17-36%; well-specified match exact.",
        notes="Hopper, n = 1,103, 2,000 banks per bar. Weights from a well-specified discriminator track the exact "
              "weights within 0.2 points at every tilt. A discriminator on raw features, which is misspecified, "
              "makes WBCP fail 16.6% to 36.4% of banks. At state gamma 1 the low rates come with abstention: 31% "
              "of banks with well-specified weights and 26% with exact weights abstain, against none with "
              "misspecified weights."),
}


def make_all():
    """Render every chart of the group; return {key: {png, headline, takeaway, notes, facts}}."""
    import matplotlib.pyplot as plt

    out = {}
    for key, slide in SLIDES.items():
        fig, facts = slide["make"]()
        path = S.save_deck(fig, key)
        plt.close(fig)
        out[key] = dict(png=str(path), headline=slide["headline"], takeaway=slide["takeaway"], notes=slide["notes"],
                        facts=facts)
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the facts carry gamma signs; a cp1252 console cannot print them
    for key, item in make_all().items():
        print(key, item["png"])
        for f in item["facts"]:
            print(f"   {f['what']}: {f['value']}")
