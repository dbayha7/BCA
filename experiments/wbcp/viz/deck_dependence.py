"""Deck charts, group 'dependence': why BCA's calibration bank is thinned, and where the thinning rule still fails.

Five single-message charts for the slides (S.deck_figure / S.save_deck, PNGs under runs/wbcp_viz/deck/):

1 whole_vs_thin       whole-episode banks against independent rows, by dataset
2 rho_by_dataset      within-episode correlation rho of the miss indicator, by dataset
3 hopper_K            hopper: failure against rows per episode K, with the pre-registered prediction
4 K_by_dataset        failure at K = 2 / 5 / 10 on the five healthy pools
5 configured_banks    the configured banks (raise K until the withholding cap fits) at both bank sizes

Every value is a fact-checked number from experiments/wbcp/DEPENDENCE.md or calibration/dependence_evidence.json (cited
per fact). The one value not stated in DEPENDENCE.md, walker2d's independent-row failure at n = 1,024, is read from
runs/wbcp_dependence/other-datasets/w_iid_n1024.json (BQ-CP, gamma = 0, normalized score, 4,000 banks).
"Failure" throughout is the share of banks whose threshold misses more than 10% of test rows (alpha = 0.1); the budget
is 5% (beta = 0.95). Uniform BCA (BQ-CP), normalized score, no shift, TD3+BC frozen pools, 4,000 banks per value.
Pure NumPy / matplotlib; each chart renders in well under a second.
"""

from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from experiments.wbcp.viz import style as S

DEP = "experiments/wbcp/DEPENDENCE.md"
REG = "calibration/dependence_evidence.json"

BIG = 30  # the key numbers
LABEL = 22  # direct labels and dataset names
NOTE = 20  # the floor
GREY_LIGHT = "#C6CBD0"  # muted context, lighter than S.MUTED
GREY_HATCH = "#DADDE0"


def _r1(value):
    """Round half up to one decimal, as the write-ups do (17.65 -> 17.7; float formatting would give 17.6)."""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _budget(ax, x, y, ha="left", va="bottom", transform=None, span=None):
    """The 5% budget line, dashed INK, with its label; `span` = (x0, x1) stops it short of right-hand labels."""
    style = dict(color=S.INK, lw=2.4, ls=(0, (6, 4)), zorder=1)
    if span is None:
        ax.axhline(5, **style)
    else:
        ax.plot(span, [5, 5], **style)
    kw = dict(transform=transform) if transform is not None else {}
    ax.text(x, y, "5% budget", color=S.INK, fontsize=NOTE, ha=ha, va=va, zorder=6, **kw)


def _pct(ax, ticks):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.yaxis.set_minor_locator(NullLocator())


def _xcats(ax, x, labels, colors=None):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=LABEL)
    for tick, color in zip(ax.get_xticklabels(), colors or [S.INK] * len(labels)):
        tick.set_color(color)
    ax.tick_params(axis="x", length=0, pad=10)


class _Facts(list):
    def add(self, what, value, source):
        self.append(dict(what=what, value=value, source=source))


# ---- 1 whole-episode banks vs independent rows ---------------------------------------------------------------------

def whole_vs_thin():
    facts = _Facts()
    # (label, whole-episode failure %, independent-row failure %, bank size, source of each)
    rows = [
        ("pen-expert", 27.4, 4.3, "1,024", f"{DEP} Change 12 table"),
        ("hopper", 27.8, 4.7, "1,103", f"{DEP} Change 3 (whole, 4,000 trials) and Change 4 (iid reference)"),
        ("halfcheetah", 33.1, 4.7, "1,024", f"{DEP} Change 12 table"),
        ("maze2d", 40.0, 4.8, "1,024", f"{DEP} Change 12 table"),
        ("walker2d", 48.0, 4.65, "1,024", f"{DEP} Change 10 (whole, 48.0% [46.4, 49.6]); independent rows from "
                                          "runs/wbcp_dependence/other-datasets/w_iid_n1024.json (BQ-CP fail 0.0465)"),
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = S.deck_figure()
    x = np.arange(len(rows))
    w, off = 0.36, 0.2
    whole = [r[1] for r in rows]
    iid = [r[2] for r in rows]
    ax.bar(x - off, whole, w, color=S.ORANGE, zorder=2)
    ax.bar(x + off, iid, w, color=S.BLUE, zorder=2)
    for xi, v in zip(x, whole):
        ax.text(xi - off, v + 1.0, f"{_r1(v)}%", ha="center", va="bottom", fontsize=BIG, fontweight="semibold",
                color=S.ORANGE)
    for label, wv, iv, n, src in rows:
        facts.add(f"{label}: failure, whole-episode bank vs independent rows (n = {n})",
                  f"{_r1(wv)}% vs {_r1(iv)}%", src)

    right = len(rows) - 1 + off + w / 2 + 0.08
    ax.text(right, 40, "whole\nepisodes", color=S.ORANGE, fontsize=LABEL, fontweight="semibold", ha="left",
            va="center", linespacing=1.05)
    ax.text(right, 12.5, "independent\nrows", color=S.BLUE, fontsize=LABEL, fontweight="semibold", ha="left",
            va="center", linespacing=1.05)
    _budget(ax, right, 4.0, va="top")
    _xcats(ax, x, [r[0] for r in rows])
    ax.set_xlim(-0.55, len(rows) - 1 + 1.2)
    ax.set_ylim(0, 54)
    _pct(ax, [0, 25, 50])
    ax.set_ylabel("Banks failing")
    facts.add("independent-row range across the five pools", "4.3-4.8%", "values above")
    facts.add("setting", "uniform BCA (BQ-CP), normalized min(Q1, Q2) residual score (BCA's score at the time), "
              "no shift, TD3+BC frozen pools, 4,000 banks per bar; failure = threshold misses > 10% of test rows; "
              "target bank size 1,103 rows on hopper (whole-episode banks average 1,362), 1,024 elsewhere",
              f"{DEP} Terms, Changes 3, 10, 12")
    facts.add("budget line", "5% of banks (1 - beta, beta = 0.95)", f"{DEP} intro; calibration/bank.py "
              "DEPENDENCE_BUDGET")
    return fig, facts


# ---- 2 within-episode correlation rho --------------------------------------------------------------------------------

def rho_by_dataset():
    facts = _Facts()
    healthy = [("hopper", 0.016), ("halfcheetah", 0.033), ("pen-expert", 0.054), ("maze2d", 0.073),
               ("walker2d", 0.120)]
    flagged = [("pen-human", 0.090), ("pen-cloned", 0.108)]
    fig, ax = S.deck_figure()
    xh = np.arange(len(healthy), dtype=float)
    xf = len(healthy) + 0.75 + np.arange(len(flagged), dtype=float)
    w = 0.62
    ax.bar(xh, [v for _, v in healthy], w, color=S.BLUE, zorder=2)
    ax.bar(xf, [v for _, v in flagged], w, color=GREY_HATCH, edgecolor=S.MUTED, hatch="//", linewidth=0, zorder=2)
    for i, (label, v) in enumerate(healthy):
        key = i in (0, len(healthy) - 1)
        ax.text(xh[i], v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=BIG if key else LABEL,
                fontweight="semibold" if key else "normal", color=S.BLUE if key else S.SOFT)
    for xi, (label, v) in zip(xf, flagged):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=NOTE, color=S.MUTED)
    ax.text(xf.mean(), 0.126, "diverging critics", ha="center", va="bottom", fontsize=NOTE, color=S.MUTED,
            style="italic")

    # hopper's level carried across to walker2d, and the ratio
    lo, hi = healthy[0][1], healthy[-1][1]
    ax.plot([xh[0] + w / 2, xh[-1] + w / 2 + 0.12], [lo, lo], color=S.INK, lw=1.6, ls=(0, (1.5, 2.5)), zorder=3)
    xa = xh[-1] + w / 2 + 0.12
    ax.annotate("", xy=(xa, hi), xytext=(xa, lo), arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=2.0,
                                                                  mutation_scale=22, shrinkA=0, shrinkB=0), zorder=3)
    ax.text(xa + 0.08, (lo + hi) / 2, f"{hi / lo:.1f}×", ha="left", va="center", fontsize=BIG, fontweight="semibold", color=S.INK)

    _xcats(ax, np.concatenate([xh, xf]), [l for l, _ in healthy] + [l for l, _ in flagged],
           [S.INK] * len(healthy) + [S.MUTED] * len(flagged))
    ax.set_xlim(-0.55, xf[-1] + 0.55)
    ax.set_ylim(0, 0.155)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.05, 0.10]))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}" if v else "0")
    ax.set_ylabel("Within-episode ρ")
    src = f"{DEP} Change 2 (hopper 0.0155, shown 0.016 in Changes 10/12), Change 10 and Change 12 tables"
    for label, v in healthy:
        facts.add(f"rho (ICC of the miss indicator), {label}, normalized score, TD3+BC pool", f"{v:.3f}", src)
    for label, v in flagged:
        state = "diverged" if label == "pen-human" else "still diverging"
        facts.add(f"rho, {label} (critic {state}, found post hoc; muted)", f"{v:.3f}",
                  src + "; critic health: Change 12 post hoc")
    facts.add("walker2d rho / hopper rho", f"{hi / lo:.1f}× from the shown values (7.7× from hopper's unrounded "
              "0.0155; DEPENDENCE.md: 'seven times hopper's')", f"{DEP} Summary and Change 10")
    facts.add("why a small rho matters (speaker notes)", "hopper whole-episode bank of 1,362 rows ~ 210 independent "
              "rows (pair-weighted rho 0.0115 over ~480 rows per episode, D ~ 6.5)", f"{DEP} Change 2")
    return fig, facts


# ---- 3 hopper: failure vs rows per episode K ----------------------------------------------------------------------

def hopper_K():
    facts = _Facts()
    ks = [1, 2, 5, 10, 25, 50, 100]
    observed = [4.9, 4.6, 5.2, 5.7, 8.0, 10.8, 14.7]
    predicted = [5.0, 5.1, 5.5, 6.2, 8.0, 10.7, 15.1]  # ICC design effect, pre-registered
    fig, ax = S.deck_figure()
    ax.plot(ks, predicted, color=GREY_LIGHT, lw=5, zorder=1, solid_capstyle="round")
    ax.text(29, 9.2, "predicted", color=S.MUTED, fontsize=LABEL, ha="right", va="bottom")
    good = [k <= 10 for k in ks]
    for k, v, ok in zip(ks, observed, good):
        size = 520 if k == 5 else 300
        ax.scatter([k], [v], s=size, color=S.BLUE if ok else S.ORANGE, edgecolor=S.PAPER, linewidth=2, zorder=4)
    ax.text(5, 5.2 + 1.1, "5.2%", ha="center", va="bottom", fontsize=BIG, fontweight="semibold", color=S.BLUE)
    ax.text(100, 14.7 + 1.0, "14.7%", ha="center", va="bottom", fontsize=BIG, fontweight="semibold",
            color=S.ORANGE)
    _budget(ax, 135, 4.55, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.78, 140)
    ax.xaxis.set_major_locator(FixedLocator(ks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.set_ylim(0, 17.8)
    _pct(ax, [0, 5, 10, 15])
    ax.set_xlabel("Rows per episode K")
    ax.set_ylabel("Banks failing")
    src = f"{DEP} Change 4 table (hopper-medium, n = 1,103, random positions, 4,000 trials per K)"
    for k, v, p in zip(ks, observed, predicted):
        facts.add(f"hopper, K = {k}: observed failure (predicted, ICC)", f"{_r1(v)}% ({_r1(p)}%)", src)
    facts.add("colour rule", "blue for K <= 10 (within about 1 point of 5%; K = 10: 5.7% [5.0, 6.4]), orange for "
              "K >= 25", src)
    facts.add("cost of K = 5", "221 episodes, about 10.7% of the dataset withheld from training; mean threshold "
              "unchanged (1.139-1.143)", src)
    return fig, facts


# ---- 4 failure at K = 2 / 5 / 10 on five datasets ------------------------------------------------------------------

def K_by_dataset():
    facts = _Facts()
    series = {  # stratified (spaced) rows, n = 1,103 on hopper, 1,024 elsewhere
        "hopper": [4.4, 5.2, 5.4],
        "halfcheetah": [4.7, 5.5, 6.5],
        "pen-expert": [5.6, 6.0, 7.8],
        "maze2d": [4.6, 6.2, 9.1],
        "walker2d": [5.3, 7.5, 12.2],
    }
    src = {"hopper": f"{DEP} Change 6 (stratified)", "walker2d": f"{DEP} Change 10 (K = 5) and Change 12 (K = 2, 10)"}
    fig, ax = S.deck_figure()
    x = np.arange(3)
    for name in ("halfcheetah", "pen-expert", "maze2d"):
        ax.plot(x, series[name], color=GREY_LIGHT, lw=3.5, marker="o", ms=9, zorder=2)
        ax.text(2.1, series[name][-1], name, color=S.MUTED, fontsize=NOTE, ha="left", va="center")
    ax.plot(x, series["hopper"], color=S.BLUE, lw=5, marker="o", ms=13, zorder=3)
    ax.text(2.1, series["hopper"][-1], "hopper", color=S.BLUE, fontsize=LABEL, fontweight="semibold",
            ha="left", va="center")
    ax.plot(x, series["walker2d"], color=S.ORANGE, lw=6, marker="o", ms=16, zorder=4)
    ax.text(2.1, series["walker2d"][-1], "walker2d", color=S.ORANGE, fontsize=LABEL, fontweight="semibold",
            ha="left", va="center")
    ax.text(1, series["walker2d"][1] + 0.6, "7.5%", color=S.ORANGE, fontsize=BIG, fontweight="semibold",
            ha="center", va="bottom")
    ax.text(2, series["walker2d"][2] + 0.6, "12.2%", color=S.ORANGE, fontsize=BIG, fontweight="semibold",
            ha="center", va="bottom")
    _budget(ax, 1.5, 4.6, ha="center", va="top", span=(-0.2, 2.06))
    ax.set_xticks(x)
    ax.set_xticklabels(["2", "5", "10"], fontsize=LABEL)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_xlim(-0.2, 2.75)
    ax.set_ylim(2.5, 14.4)
    _pct(ax, [5, 10])
    ax.set_xlabel("Rows per episode K")
    ax.set_ylabel("Banks failing")
    for name, vals in series.items():
        facts.add(f"{name}: failure at K = 2 / 5 / 10 (spaced rows, no shift)", " / ".join(f"{v}%" for v in vals),
                  src.get(name, f"{DEP} Change 12 table"))
    facts.add("pen-human, pen-cloned", "left out: critics diverged", f"{DEP} Change 12 post hoc")
    return fig, facts


# ---- 5 configured banks ---------------------------------------------------------------------------------------------

def configured_banks():
    facts = _Facts()
    # (label, TD3+BC-family size failure %, K, n; IQL size failure %, K, n), uniform_bca_failure x 100 in the registry
    rows = [
        ("hopper", 5.1, 5, "1,103", 5.025, 18, "8,192"),
        ("pen-expert", 5.45, 5, "1,024", 5.275, 7, "8,192"),
        ("halfcheetah", 5.325, 6, "1,024", 6.275, 17, "8,192"),
        ("maze2d", 6.45, 5, "1,024", 6.575, 5, "8,192"),
        ("walker2d", 17.65, 23, "1,024", 23.4, 67, "8,192"),
    ]
    fig, ax = S.deck_figure()
    x = np.arange(len(rows))
    w, off = 0.38, 0.2
    last = len(rows) - 1
    for i, (label, s, ks, ns, b, kb, nb) in enumerate(rows):
        hot = i == last
        ax.bar(x[i] - off, s, w, color=S.ORANGE_LIGHT if hot else GREY_LIGHT, zorder=2)
        ax.bar(x[i] + off, b, w, color=S.ORANGE if hot else S.MUTED, zorder=2)
        facts.add(f"{label}: failure at K = {ks}, n = {ns} and K = {kb}, n = {nb} (BCA sampler)",
                  f"{_r1(s)}% and {_r1(b)}%", f"{REG} uniform_bca_failure x 100 (td3_bc entries, 4,000 trials)")
    # walker2d carries its own detail: the failure on top, K and the bank size inside the bar
    _, s, ks, ns, b, kb, nb = rows[last]
    for xx, v, k, n, inside, nudge in ((x[last] - off, s, ks, ns, S.INK, -0.07),
                                       (x[last] + off, b, kb, nb, S.PAPER, 0.04)):
        ax.text(xx + nudge, v + 0.6, f"{_r1(v)}%", ha="center", va="bottom", fontsize=BIG, fontweight="semibold",
                color=S.ORANGE)
        ax.text(xx, v - 0.9, f"K {k}", ha="center", va="top", fontsize=LABEL, fontweight="semibold", color=inside)
        ax.text(xx, 0.9, f"{n}\nrows", ha="center", va="bottom", fontsize=NOTE, color=inside, linespacing=1.0)
    # the other four pools, summarized once
    others = [v for r in rows[:last] for v in (r[1], r[4])]
    x0, x1, yb = x[0] - off - w / 2, x[last - 1] + off + w / 2, 8.2
    ax.plot([x0, x0, x1, x1], [yb - 0.6, yb, yb, yb - 0.6], color=S.MUTED, lw=1.6, zorder=3)
    ax.text((x0 + x1) / 2, yb + 0.4, f"{_r1(min(others))}–{_r1(max(others))}%", ha="center", va="bottom",
            fontsize=26, color=S.SOFT)
    right = last + off + w / 2 + 0.08
    _budget(ax, right, 4.3, va="top")
    _xcats(ax, x, [r[0] for r in rows], [S.INK] * last + [S.ORANGE])
    ax.set_xlim(-0.55, last + 1.05)
    ax.set_ylim(0, 27)
    _pct(ax, [0, 10, 20])
    ax.set_ylabel("Banks failing")
    facts.add("other four pools, both sizes", f"{_r1(min(others))}-{_r1(max(others))}%", REG)
    facts.add("hopper bars", "K = 5 at n = 1,103 and K = 18 at n = 8,192 (registry runs), not the configured K = 6 / "
              "K = 17", REG)
    facts.add("other pools over budget (95% interval above 5%)", "maze2d 6.5% [5.7, 7.3] and 6.6% [5.8, 7.4]; "
              "halfcheetah at 8,192 rows 6.3% [5.5, 7.1]", f"{REG} ci95")
    facts.add("hopper at 8,192 rows is optimistic", "the bank covers 44% of the finite pool; a real deployment of "
              "IQL's K = 17-18 bank is expected to fail between 5% and 7%", f"{DEP} Change 9, result 4")
    facts.add("rule that sets K", "smallest K >= 5 whose reservation fits the withholding cap (10% at 1,024 rows, "
              "25% at 8,192) for 1,000 simulated seeds", f"{DEP} Change 9")
    facts.add("8,192-row banks", "IQL's configured size, run on TD3+BC pools (not IQL host results)", REG)
    return fig, facts


CHARTS = {
    "whole_vs_thin": dict(
        make=whole_vs_thin,
        headline="Whole-episode banks break the guarantee",
        takeaway="27–48% of banks fail; independent rows stay near 5%",
        notes="BCA first reserved its calibration bank as whole episodes. With no distribution shift at all, 27.4% "
              "to 48.0% of those banks miss more than 10% of test rows, against a 5% budget; independent rows of the "
              "same target size fail 4.3-4.8%. Uniform BCA on the normalized min(Q1, Q2) score, TD3+BC frozen pools, "
              "4,000 banks per bar, target size 1,103 rows on hopper and 1,024 elsewhere."),
    "rho_by_dataset": dict(
        make=rho_by_dataset,
        headline="Episode correlation varies 7.5× by dataset",
        takeaway="ρ runs from 0.016 (hopper) to 0.120 (walker2d)",
        notes="rho is the correlation between two rows' misses inside the same episode, measured on each TD3+BC "
              "frozen pool with the normalized score. It looks small, but episodes are long: on hopper a "
              "whole-episode bank of about 1,360 rows carries about as much information as 210 independent rows. "
              "On the five healthy pools rho runs from 0.016 on hopper to 0.120 on walker2d. Greyed out: a post-hoc "
              "check found the pen-human critic diverged and pen-cloned's still diverging, so 0.090 and 0.108 are "
              "left out of conclusions."),
    "hopper_K": dict(
        make=hopper_K,
        headline="On hopper, few rows per episode restore 5%",
        takeaway="Hopper K = 5: 5.2% fail; K = 100: 14.7%",
        notes="Instead of whole episodes, take K rows at random positions from each of many episodes. On hopper "
              "(1,103-row banks, 4,000 per K, no shift) failure stays within about a point of the 5% budget up to "
              "K = 10 (4.9, 4.6, 5.2, 5.7%) and climbs to 8.0, 10.8 and 14.7% at K = 25, 50 and 100. The grey line "
              "is the design-effect prediction written down before the run; it tracks the data to about a point. "
              "The cost is training data, not band width: K = 5 withholds about 11% of hopper's data."),
    "K_by_dataset": dict(
        make=K_by_dataset,
        headline="Hopper's K = 5 does not transfer",
        takeaway="walker2d fails 7.5% at K = 5, 12.2% at K = 10",
        notes="The same sweep with spaced rows on the five healthy pools, no shift, 4,000 banks each (1,103 rows on "
              "hopper, 1,024 elsewhere). Hopper stays at 4.4-5.4%, but walker2d-medium-replay, whose rho is 7.5 "
              "times hopper's, fails 7.5% at K = 5 and 12.2% at K = 10; maze2d, pen-expert and halfcheetah also "
              "exceed 5% at K = 10. So K has to come from each dataset's measured rho: about one or two rows per "
              "episode on walker2d."),
    "configured_banks": dict(
        make=configured_banks,
        headline="Raising K to fit the cap fails walker2d",
        takeaway="walker2d's forced K = 23 and 67 fail 17.7% and 23.4%",
        notes="BCA's configs choose K by a declared rule: start at 5 and raise K until the withheld episodes fit the "
              "training-data cap. On walker2d the cap forces K = 23 and 67, and those banks fail 17.7% and 23.4% "
              "with no shift (BCA's own sampler, TD3+BC pool, 4,000 banks each). The other four pools land at "
              "5.0-6.6% at both 1,024 and IQL's 8,192 rows, close to budget but not all inside it: maze2d (6.5%, "
              "6.6%) and halfcheetah at 8,192 rows (6.3%) are measurably over. Hopper's bars are its K = 5 "
              "(1,103 rows) and K = 18 runs, and the 8,192-row hopper figure is optimistic because the bank covers "
              "44% of the test pool."),
}


def make_all():
    """Render every chart of the group; return {key: {png, headline, takeaway, notes, facts}}."""
    import matplotlib.pyplot as plt

    out = {}
    for key, spec in CHARTS.items():
        fig, facts = spec["make"]()
        path = S.save_deck(fig, key)
        plt.close(fig)
        out[key] = dict(png=str(path), headline=spec["headline"], takeaway=spec["takeaway"], notes=spec["notes"],
                        facts=list(facts))
    return out


if __name__ == "__main__":
    for chart_key, result in make_all().items():
        print(chart_key, result["png"])
