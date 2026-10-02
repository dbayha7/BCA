"""Deck charts, signal group: which error signal BCA should calibrate, and whether its dose targets anything.

Seven single-message charts for slides the presenter talks over. Each chart function returns (fig, facts); facts
list every number drawn, with its source. make_all() renders them to runs/wbcp_viz/deck/<key>.png.

Sources (all read-only):
- min_vs_q1: signal study step 2 (experiments/signal/SIGNAL_STUDY.md l.217-223), read from
  runs/wbcp_signal/frozen/<pool>/evaluation_b4000_s2026100201.json via fig_critic_alignment.pool_rates and checked
  against the table.
- q1_optimistic: Illustration, the LQ harness slice of fig_critic_alignment.lq_slice (runs/wbcp_signal/lq/results.json
  meta, closed forms of experiments/signal/lq_harness.py).
- dose_near_13: per-row Q1-signal doses of the five healthy TD3+BC pools, fig_bc_dose.load_pools (heads.npz at each
  pool's mean deployed WBCP threshold), checked against SIGNAL_STUDY.md l.219-223.
- lq_scoreboard: step 3's 216 cells, SIGNAL_STUDY.md l.296-301.
- strength_ratio: post hoc, runs/wbcp_signal/strength_reanalysis.json bc_gradient_ratio (SIGNAL_STUDY.md l.398-404).
- pilot_power: step-4 pilot, whitelisted power rows only (runs/wbcp_signal/step4/pilot_report.json
  power[0].pooled_block_c, as quoted in runs/wbcp_signal/step4/expectations.md Addendum A part 2, item 8). The values
  are copied here; this module does not open the pilot report or anything under step4/main.
- shared_bias: runs/wbcp_signal/lq/results.json (q1 signal, shared_bias), checked against SIGNAL_STUDY.md l.328-335.
"""

import json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

from experiments.wbcp.viz import fig_bc_dose, fig_critic_alignment
from experiments.wbcp.viz import style as S

VAL, LAB, NOTE = 30, 23, 20  # key values, direct labels, small context text (points)
LQ = S.ROOT / "runs" / "wbcp_signal" / "lq" / "results.json"
STRENGTH = S.ROOT / "runs" / "wbcp_signal" / "strength_reanalysis.json"
MINUS = "−"


def _signed(v, digits=3):
    return ("+" if v > 0 else MINUS if v < 0 else "") + f"{abs(v):.{digits}f}"


def _tag(ax, text, x=0.995, y=0.995, ha="right", va="top"):
    """A small corner tag such as 'Illustration' or 'post hoc'."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=NOTE, color=S.SOFT, zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=S.CARD, edgecolor=S.RULE, linewidth=1.5))


# ---------------------------------------------------------------------------------------------------------------------
# 1. The min band misses Q1's error; the Q1 band holds the budget


STEP2 = {  # SIGNAL_STUDY.md l.219-223: Q1's residual under the min band, Q1's residual under the q1 band (% of banks)
    "hopper": (65.3, 4.6), "walker2d": (82.4, 4.3), "halfcheetah": (15.6, 4.0), "maze2d": (27.4, 4.4),
    "pen-expert": (10.5, 5.1)}


def min_vs_q1():
    rows = fig_critic_alignment.pool_rates()  # sorted by the min band's failure, largest first
    for r in rows:
        want = STEP2[r["label"]]
        got = (round(100 * r["min_band"], 1), round(100 * r["q1_band"], 1))
        if got != want:
            raise AssertionError(f"{r['label']}: {got} differs from SIGNAL_STUDY.md {want}")
    facts = []
    fig, ax = S.deck_figure()
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.4
    for yi, r in zip(y, rows):
        m, q = 100 * r["min_band"], 100 * r["q1_band"]
        ax.barh(yi + h / 2, m, height=h, color=S.ORANGE, zorder=2)
        ax.barh(yi - h / 2, q, height=h, color=S.BLUE, zorder=2)
        ax.text(m + 1.2, yi + h / 2, f"{m:.1f}%", ha="left", va="center", fontsize=VAL, fontweight="semibold",
                color=S.ORANGE)
        ax.text(6.4, yi - h / 2, f"{q:.1f}%", ha="left", va="center", fontsize=LAB, color=S.BLUE,
                fontweight="semibold")
        facts.append(dict(what=f"{r['label']}: banks failing for Q1's residual, min(Q1, Q2) band / Q1 band",
                          value=f"{m:.1f}% / {q:.1f}%",
                          source=r["path"] + " designs[iid].signals.{min,q1}.failure.q1.fail; "
                                             "experiments/signal/SIGNAL_STUDY.md l.219-223"))
    top = y[0]
    ax.text(7.0, top + h / 2, "min(Q1, Q2) band", ha="left", va="center", fontsize=LAB, color="white",
            fontweight="semibold", zorder=4)
    ax.text(13.6, top - h / 2, "Q1 band", ha="left", va="center", fontsize=LAB, color=S.BLUE, fontweight="semibold")
    ax.axvline(5, color=S.INK, lw=2.5, ls=(0, (5, 3)), zorder=3)
    ax.text(5.6, top + 0.78, "5% budget", ha="left", va="center", fontsize=LAB, color=S.INK)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=LAB, labelcolor=S.INK)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(y[-1] - 0.7, top + 1.05)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Banks failing (%)")
    facts += [
        dict(what="banks per pool, rows per bank, weights, alpha, beta",
             value=f"{rows[0]['banks']:,} banks of {rows[0]['n']:,} independent rows; uniform-weight WBCP; "
                   f"alpha {rows[0]['alpha']}, beta {rows[0]['beta']}; sigma refit per signal",
             source="runs/wbcp_signal/frozen/<pool>/evaluation_b4000_s2026100201.json settings, designs[iid]; "
                    "SIGNAL_STUDY.md l.201-215"),
        dict(what="5% line: the failure budget 1 - beta", value="5%", source="beta = 0.95"),
        dict(what="notes only: earlier alignment study (online sigma, 1,000 banks), min band failing for Q1",
             value="8.2-70.2%", source="experiments/wbcp/CRITIC_ALIGNMENT.md Results table; SIGNAL_STUDY.md l.248-252"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
# 2. Illustration: Q1 optimistic, the min equals the true Q, the actor climbs Q1


def q1_optimistic():
    L = fig_critic_alignment.lq_slice()
    t = L["t"]
    facts = []
    fig, ax = S.deck_figure()
    ax.axvspan(-L["logged"], L["logged"], color=S.MUTED, alpha=0.16, lw=0, zorder=0)
    ax.fill_between(t, L["q_true"], L["q1"], color=S.BLUE_LIGHT, alpha=0.45, lw=0, zorder=1)
    ax.plot(t, L["q_true"], color=S.ORANGE, lw=5, zorder=3, solid_capstyle="round")
    ax.plot(t, L["q1"], color=S.BLUE, lw=5, ls=(0, (5, 1.8)), zorder=4)
    ax.plot([L["t_true"]], [L["q_at_true"]], "o", ms=20, color=S.ORANGE, mec=S.PAPER, mew=3, zorder=5)
    ax.plot([L["t_q1"]], [L["q1_at_q1"]], "o", ms=20, color=S.BLUE, mec=S.PAPER, mew=3, zorder=5)
    ax.annotate("", xy=(L["t_q1"], L["q_true_at_q1"]), xytext=(L["t_q1"], L["q1_at_q1"]),
                arrowprops=dict(arrowstyle="<->", color=S.INK, lw=3, shrinkA=11, shrinkB=1,
                                mutation_scale=28), zorder=6)

    ymin, ymax = L["q_true"].min(), L["q1"].max()
    span = ymax - ymin
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.42 * span)
    ax.set_xlim(t[0], t[-1])
    top = ax.get_ylim()[1]
    ax.text(0, top - 0.04 * span, "logged\nactions", ha="center", va="top", fontsize=LAB, color=S.SOFT)
    # TD3+BC's actor also has a BC term, so Q1's peak is where its Q term pulls, not where it lands
    ax.text(L["t_q1"], L["q1_at_q1"] + 0.07 * span, "actor pulled here", ha="center", va="bottom", fontsize=LAB,
            color=S.BLUE, fontweight="semibold")
    ax.text(L["t_true"], L["q_at_true"] - 0.07 * span, "true best", ha="center", va="top", fontsize=LAB,
            color=S.ORANGE, fontweight="semibold")
    ax.text(L["t_q1"] + 0.07, (L["q_true_at_q1"] + L["q1_at_q1"]) / 2, "Q1's error:\nthe min can't see it",
            ha="left", va="center", fontsize=LAB, color=S.INK)
    q_at = lambda x, curve: float(np.interp(x, t, L[curve]))
    ax.text(-0.62, q_at(-0.62, "q1") + 0.06 * span, "Q1", ha="center", va="bottom", fontsize=VAL,
            color=S.BLUE, fontweight="semibold")
    ax.text(0.95, q_at(0.95, "q_true") - 0.30 * span, "min(Q1, Q2) = true Q", ha="center", va="top",
            fontsize=LAB, color=S.ORANGE, fontweight="semibold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Action")
    ax.set_ylabel("Value Q(s, a)")
    _tag(ax, "Illustration")
    src = ("computed by fig_critic_alignment.lq_slice from runs/wbcp_signal/lq/results.json meta with "
           "experiments/signal/lq_harness.py's closed forms")
    facts += [
        dict(what="setting: LQ harness case, behaviour, pi offset, kappa, state",
             value="q1_optimistic, poor, 1, 1, s = (0.5, 0.5, 0.5)",
             source="cells of the step-3 grid (fig_critic_alignment.py BEHAVIOR, OFFSET, KAPPA, STATE)"),
        dict(what="heads: Q1 = Q^pi + kappa ||a - beta(s)||^2, Q2 = Q^pi, so min(Q1, Q2) = Q^pi (true Q)",
             value="construction", source="experiments/signal/lq_harness.py:267-292 (case_critic, head_error)"),
        dict(what="logged-action span along the slice, beta(s) +/- 2 sigma_b", value=f"+/-{L['logged']:.1f}",
             source="behavior_noise 0.2, runs/wbcp_signal/lq/results.json meta.settings"),
        dict(what="true Q's peak on the slice (distance from beta(s))", value=f"{L['t_true']:.2f}", source=src),
        dict(what="Q1's peak, where the actor's Q term pulls (distance from beta(s))", value=f"{L['t_q1']:.2f}",
             source=src),
        dict(what="Q1's error at its peak, Q1 - Q^pi (arrow)", value=f"{L['err_at_q1']:.2f}", source=src),
        dict(what="axis numbers", value="hidden (illustration)", source="design choice"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
# 3. The dose is about 1.3 on every pool, with almost no row-to-row variation


DOSE_TABLE = {  # SIGNAL_STUDY.md l.219-223, dose mean +/- row SD (q1)
    "hopper": (1.296, 0.024), "walker2d": (1.327, 0.029), "halfcheetah": (1.298, 0.025), "maze2d": (1.262, 0.050),
    "pen-expert": (1.312, 0.017)}


def dose_near_13():
    blend = 0.5  # configs/td3_bc.yaml l.20; load_pools checks it against each evaluation file's setting
    pools = fig_bc_dose.load_pools(blend)
    for p in pools:
        mean, sd = DOSE_TABLE[p["name"]]
        if abs(p["dose"].mean() - mean) > 6e-4 or abs(p["dose"].std() - sd) > 6e-4:
            raise AssertionError(f"{p['name']}: dose differs from SIGNAL_STUDY.md")
    pools = sorted(pools, key=lambda p: p["dose"].mean())
    facts = []
    fig, ax = S.deck_figure()
    bins = np.arange(1.0, 1.5 + 1e-9, 0.004)
    mids = (bins[:-1] + bins[1:]) / 2
    for i, p in enumerate(pools):
        counts, _ = np.histogram(p["dose"], bins=bins)
        height = 0.82 * counts / counts.max()
        ax.fill_between(mids, i, i + height, step="mid", color=S.BLUE, lw=0, zorder=2)
        mean = p["dose"].mean()
        ax.plot([mean, mean], [i - 0.08, i + 0.9], color=S.INK, lw=2.5, zorder=3)
        ax.text(1.425, i + 0.3, f"{mean:.2f}", ha="left", va="center", fontsize=LAB, color=S.BLUE,
                fontweight="semibold")
        facts.append(dict(what=f"{p['name']}: per-row BC dose, mean / SD ({p['rows']:,} rows), Q1 signal",
                          value=f"{mean:.3f} / {p['dose'].std():.3f}",
                          source=f"computed by fig_bc_dose.load_pools from runs/wbcp_signal/frozen/{p['directory']}/"
                                 "heads.npz at the mean iid-bank threshold; SIGNAL_STUDY.md l.219-223"))
    n = len(pools)
    ax.axvline(1.0, color=S.INK, lw=2.5, ls=(0, (1.5, 2.5)), zorder=1)
    ax.axvline(1.5, color=S.INK, lw=2.5, ls=(0, (5, 3)), zorder=1)
    ax.text(1.008, n + 0.15, "1.0 = host's own BC", ha="left", va="bottom", fontsize=LAB, color=S.INK)
    ax.text(1.492, n + 0.15, "1.5 = cap", ha="right", va="bottom", fontsize=LAB, color=S.INK)
    ax.text(1.425, n - 0.38, "mean", ha="left", va="bottom", fontsize=NOTE, color=S.SOFT)
    ax.axvline(1.3, color=S.INK, lw=1.5, alpha=0.35, zorder=1)
    ax.text(1.3, n + 0.15, "≈ 1.3", ha="center", va="bottom", fontsize=VAL, color=S.INK, fontweight="semibold")
    ax.set_yticks(np.arange(n) + 0.3, [p["name"] for p in pools])
    ax.tick_params(axis="y", length=0, labelsize=LAB, labelcolor=S.INK)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.985, 1.515)
    ax.set_ylim(-0.25, n + 0.85)
    ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.set_xlabel("BC dose per row")
    means = [p["dose"].mean() for p in pools]
    sds = [p["dose"].std() for p in pools]
    facts += [
        dict(what="dose range: 1 (host's own BC weight) to the cap 1 + b", value="1.0 to 1.5 (b = 0.5)",
             source="calibration/dose.py l.53-60; configs/td3_bc.yaml l.20"),
        dict(what="pool means / row SDs across the five pools", value=f"{min(means):.2f}-{max(means):.2f} / "
             f"{min(sds):.3f}-{max(sds):.3f}", source="computed (above); SIGNAL_STUDY.md l.219-223"),
        dict(what="histograms: bin width; black tick", value="0.004 dose; pool mean",
             source="computed from heads.npz with the dose.py port"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
# 4. Step 3: BCA against its two same-mean controls, 216 cells


LQ_CELLS = [("worse", 121, S.ORANGE, "worse than both controls"), ("better", 60, S.BLUE, "better than both"),
            ("within", 34, S.MUTED, "within 2 SE"), ("mixed", 1, "#C9CDD2", "mixed")]


def lq_scoreboard():
    if sum(c for _, c, _, _ in LQ_CELLS) != 216:
        raise AssertionError("cells do not sum to 216")
    facts = []
    fig, ax = S.deck_figure()
    rows, gap, pad = 8, 0.9, 0.11
    x0 = 0.0
    for key, count, color, label in LQ_CELLS:
        cols = -(-count // rows)
        for k in range(count):
            c, r = divmod(k, rows)
            ax.add_patch(Rectangle((x0 + c + pad, r + pad), 1 - 2 * pad, 1 - 2 * pad, facecolor=color, lw=0))
        big = key in ("worse", "better")
        ax.text(x0 + pad, rows + 1.55, f"{count}", ha="left", va="bottom", fontsize=48 if big else VAL,
                fontweight="semibold", color=color if key != "mixed" else S.SOFT)
        ax.text(x0 + pad, rows + 0.35, label, ha="left", va="bottom", fontsize=LAB,
                color=S.INK if big else S.SOFT)
        x0 += cols + gap
        what = {"worse": "worse than both same-mean controls (constant and shuffled dose) by > 2 paired SE",
                "better": "better than both same-mean controls (constant and shuffled dose) by > 2 paired SE",
                "within": "within 2 paired SE of both same-mean controls",
                "mixed": "mixed against the two controls (SIGNAL_STUDY.md's label; not further broken down)"}[key]
        facts.append(dict(what="step-3 LQ cells where BCA is " + what, value=str(count),
                          source="experiments/signal/SIGNAL_STUDY.md l.296-299"))
    ax.text(x0 - gap + 0.1, -0.35, "1 square = 1 LQ cell", ha="right", va="top", fontsize=NOTE, color=S.SOFT)
    ax.set_xlim(-0.2, x0 + 2.2)
    ax.set_ylim(-1.3, rows + 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    facts += [
        dict(what="cells", value="216 = 6 cases x 3 signals x 2 behaviours x (kappa, offset) settings, 5 replicates "
                                 "each", source="SIGNAL_STUDY.md l.263-273, l.296-297"),
        dict(what="of the 60 better cells, those in cases with no Q1 error (clean, noisy_reward, q2_optimistic)",
             value="15 (not drawn)", source="SIGNAL_STUDY.md l.306-308"),
        dict(what="size of BCA's departure from the constant dose", value="median 6% of |BCA - none| (not drawn)",
             source="SIGNAL_STUDY.md l.309-310"),
        dict(what="notes only, post hoc: correlation of a strength-only model with BCA - constant, 216 cells; BCA's "
                  "BC strength over the constant's", value="0.85; 1-3% stronger",
             source="SIGNAL_STUDY.md, Post hoc: same mean, different strength"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
# 5. Post hoc: same mean dose, not the same strength


STRENGTH_ROWS = [("independent_errors", "good", "independent errors, good"),
                 ("independent_errors", "poor", "independent errors, poor"),
                 ("shared_bias", "poor", "shared bias, poor"), ("shared_bias", "good", "shared bias, good"),
                 ("q1_optimistic", "poor", "Q1 optimistic, poor"), ("q1_optimistic", "good", "Q1 optimistic, good")]
STRENGTH_VERIFIED = {  # medians, verified chart (strength_reanalysis.json bc_gradient_ratio)
    "independent_errors/good": (2.6081, 1.0114), "independent_errors/poor": (2.2556, 1.0214),
    "shared_bias/good": (1.1859, 1.0071), "shared_bias/poor": (1.2430, 1.0137),
    "q1_optimistic/good": (0.9915, 1.0035), "q1_optimistic/poor": (1.0066, 1.0113)}


def strength_ratio():
    with open(STRENGTH, encoding="utf-8") as handle:
        ratios = json.load(handle)["bc_gradient_ratio"]
    data = []
    for case, behaviour, label in STRENGTH_ROWS:
        key = f"{case}/{behaviour}"
        o, b = ratios[key]["oracle_over_constant"]["median"], ratios[key]["bca_over_constant"]["median"]
        if abs(o - STRENGTH_VERIFIED[key][0]) > 1e-4 or abs(b - STRENGTH_VERIFIED[key][1]) > 1e-4:
            raise AssertionError(f"{key}: strength_reanalysis.json differs from the verified medians")
        data.append((key, label, o, b))
    if [d[2] for d in data] != sorted([d[2] for d in data], reverse=True):
        raise AssertionError("rows are not sorted by the oracle's ratio")
    facts = []
    fig, ax = S.deck_figure()
    y = np.arange(len(data))[::-1].astype(float)
    off = 0.17
    for yi, (key, label, o, b) in zip(y, data):
        key_row = key.startswith("independent_errors")
        color = S.ORANGE if key_row else S.ORANGE_LIGHT
        ax.plot([1, o], [yi + off] * 2, color=color, lw=9, solid_capstyle="butt", zorder=2)
        ax.plot(o, yi + off, "o", ms=20, color=color, mec=S.PAPER, mew=2.5, zorder=4)
        ax.plot(b, yi - off, "D", ms=15, color=S.BLUE, mec=S.PAPER, mew=2, zorder=4)
        ax.text(max(o, 1) + 0.07, yi + off, f"{o:.1f}×", ha="left", va="center",
                fontsize=VAL if key_row else LAB, fontweight="semibold" if key_row else "normal",
                color=S.ORANGE if key_row else S.SOFT)
        facts.append(dict(what=f"{label} data: BC-term gradient norm / constant dose's, median (oracle / BCA)",
                          value=f"{o:.2f} / {b:.3f}",
                          source=f"runs/wbcp_signal/strength_reanalysis.json bc_gradient_ratio['{key}']."
                                 "{oracle,bca}_over_constant.median; SIGNAL_STUDY.md l.398-404"))
    top = y[0]
    ax.text(2.2, top + off + 0.3, "oracle dose", ha="center", va="bottom", fontsize=LAB,
            color=S.ORANGE, fontweight="semibold")
    bca = [d[3] for d in data]
    ax.text(1.12, top - off - 0.04, f"BCA dose {min(bca):.2f}–{max(bca):.2f}×", ha="left", va="center",
            fontsize=LAB, color=S.BLUE, fontweight="semibold")
    ax.axvline(1, color=S.INK, lw=2.5, ls=(0, (5, 3)), zorder=1)
    ax.text(1.02, top + 0.95, "1× = constant dose, same mean", ha="left", va="center", fontsize=NOTE,
            color=S.INK)
    ax.set_yticks(y, [d[1] for d in data])
    ax.tick_params(axis="y", length=0, labelsize=NOTE + 1, labelcolor=S.INK)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.85, 3.05)
    ax.set_ylim(y[-1] - 0.6, top + 1.25)
    ax.set_xticks([1, 1.5, 2, 2.5, 3], ["1×", "1.5×", "2×", "2.5×", "3×"])
    ax.set_xlabel("BC strength vs constant dose")
    _tag(ax, "post hoc", y=0.02, va="bottom")
    facts += [
        dict(what="BCA dose ratio range (medians, six case x behaviour groups)",
             value=f"{min(bca):.3f}-{max(bca):.3f}", source="runs/wbcp_signal/strength_reanalysis.json"),
        dict(what="what is compared", value="BC-term gradient norm at the starting actor, q1 signal, LQ harness, "
             "median over every replicate of every kappa/offset cell of the case; post hoc, not pre-registered",
             source="experiments/signal/strength_reanalysis.py; SIGNAL_STUDY.md l.390-419"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
# 6. Step-4 pilot: the positive control has power only with poor data


PILOT = {  # C = G(OR2) - mean G(SHUF), fraction of the optimality gap; pilot replicate 99, block C localized cells
    ("expert", "P1L"): -0.0036, ("expert", "P2L"): -0.033, ("poor", "P1L"): 0.1659, ("poor", "P2L"): 0.0818}
PILOT_SOURCE = ("runs/wbcp_signal/step4/pilot_report.json power[0].pooled_block_c (whitelisted power rows); "
                "runs/wbcp_signal/step4/expectations.md Addendum A part 2, item 8")
FORECAST = 0.03  # expectations.md Part 0 item 8: OR2 - SHUF >= 0.03 for P1L and P2L at expert


def pilot_power():
    facts = []
    fig, ax = S.deck_figure()
    xs = {("expert", "P1L"): 0.0, ("expert", "P2L"): 1.0, ("poor", "P1L"): 2.7, ("poor", "P2L"): 3.7}
    for (level, placement), x in xs.items():
        v = PILOT[(level, placement)]
        color = S.ORANGE if level == "expert" else S.BLUE
        ax.bar(x, v, width=0.78, color=color, zorder=2)
        above = v >= 0
        ax.text(x, v + (0.007 if above else -0.007), _signed(v), ha="center", va="bottom" if above else "top",
                fontsize=VAL, fontweight="semibold", color=color)
        # placement name at the bar's base: inside a positive bar, in the open space above a negative one
        ax.text(x, 0.006, placement, ha="center", va="bottom", fontsize=LAB, fontweight="semibold",
                color="white" if above else S.SOFT, zorder=4)
        facts.append(dict(what=f"{level} data, {placement}: C = G(OR2) - G(SHUF), pooled over block C's two "
                               "localized cells, fixed point, S2 (fraction of the optimality gap)",
                          value=f"{v:+.4f}", source=PILOT_SOURCE))
    ax.axhline(0, color=S.SOFT, lw=1.5, zorder=1)
    ax.axhline(FORECAST, color=S.INK, lw=2.5, ls=(0, (5, 3)), zorder=3)
    ax.text(1.85, FORECAST + 0.005, "0.03 forecast", ha="center", va="bottom", fontsize=LAB, color=S.INK,
            bbox=dict(facecolor=S.PAPER, edgecolor="none", pad=1.5))
    ax.set_xticks([0.5, 3.2], ["expert data", "poor data"])
    ax.tick_params(axis="x", length=0, labelsize=27, pad=12)
    for label, color in zip(ax.get_xticklabels(), (S.ORANGE, S.BLUE)):
        label.set_color(color)
        label.set_fontweight("semibold")
    ax.set_xlim(-0.65, 4.35)
    ax.set_ylim(-0.065, 0.2)
    ax.set_yticks([0, 0.1], ["0", "0.1"])
    ax.set_ylabel("Oracle gain over shuffled")
    ax.spines["bottom"].set_visible(False)
    _tag(ax, "pilot: 1 replicate")
    facts += [
        dict(what="0.03 line: the pre-registered power forecast (expert data, both placements)", value="0.03",
             source="runs/wbcp_signal/step4/expectations.md Part 0, item 8"),
        dict(what="placements", value="P1L: BC amplification at logged actions; P2L: trust weighting at logged "
             "actions", source="runs/wbcp_signal/step4/expectations.md Part 2 (l.226-229)"),
        dict(what="notes only: poor data per cell, C for P1L / P2L (q1opt-local; tilt-local)",
             value="0.036 / 0.030; 0.296 / 0.133", source="runs/wbcp_signal/step4/expectations.md Addendum A part 2, item 8"),
        dict(what="notes only: G2 moved to poor data after the pilot", value="Amendment P",
             source="runs/wbcp_signal/step4/expectations.md Addendum A part 3"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------
# 7. shared_bias: residual coverage stays about 91% while the true Q1 error is missed


SHARED_VERIFIED = {  # SIGNAL_STUDY.md l.330-335: % of rows where the q1 band misses the true Q1 error, offsets 0.5/1/2
    ("good", 0.5): (0.0, 0.3, 100), ("good", 1.0): (1.1, 100, 100), ("good", 2.0): (86.6, 100, 100),
    ("poor", 0.5): (0.0, 0.0, 97.7), ("poor", 1.0): (0.0, 45.0, 100), ("poor", 2.0): (1.0, 100, 100)}
OFFSETS = (0.5, 1.0, 2.0)


def _pct(v):
    if round(v, 1) == 0:
        return "0%"
    return f"{v:.0f}%" if v >= 9.95 else f"{v:.1f}%"


def shared_bias():
    with open(LQ, encoding="utf-8") as handle:
        results = json.load(handle)["results"]
    cell = {(r["behavior"], r["kappa"], r["pi_offset"]): r["metrics"] for r in results
            if r["case"] == "shared_bias" and r["signal"] == "q1"}
    facts = []
    fig, ax = S.deck_figure()
    cmap = LinearSegmentedColormap.from_list("miss", [S.CARD, S.ORANGE_LIGHT, S.ORANGE])
    col_x = {"own": 0.0, 0.5: 1.35, 1.0: 2.35, 2.0: 3.35}
    w, hgt = 0.94, 0.9
    row_y, y = [], 6.6
    for behaviour in ("good", "poor"):
        for kappa in (0.5, 1.0, 2.0):
            row_y.append((behaviour, kappa, y))
            y -= 1.0
        y -= 0.35
    own_cov = []
    for behaviour, kappa, yy in row_y:
        misses = []
        for off in OFFSETS:
            m = cell[(behaviour, kappa, off)]
            misses.append(100 * m["miss_q1err_logged"]["mean"])
            own_cov.append(100 * (1 - m["miss_own_logged"]["mean"]))
        want = SHARED_VERIFIED[(behaviour, kappa)]
        if any(abs(round(a, 1) - b) > 0.051 for a, b in zip(misses, want)):
            raise AssertionError(f"shared_bias {behaviour} kappa {kappa}: {misses} differs from {want}")
        covs = own_cov[-3:]
        for off, v in zip(OFFSETS, misses):
            ax.add_patch(FancyBboxPatch((col_x[off] + 0.03, yy + 0.05), w - 0.06, hgt - 0.1,
                                        boxstyle="round,pad=0,rounding_size=0.06", facecolor=cmap(v / 100),
                                        edgecolor=S.RULE if v < 5 else "none", lw=1.2))
            ax.text(col_x[off] + w / 2, yy + hgt / 2, _pct(v), ha="center", va="center", fontsize=LAB,
                    color="white" if v > 55 else S.INK, fontweight="semibold" if v > 55 else "normal")
        ax.text(-0.12, yy + hgt / 2, f"κ = {kappa:g}", ha="right", va="center", fontsize=LAB, color=S.INK)
        facts.append(dict(what=f"shared_bias, {behaviour} data, kappa {kappa:g}: % of logged rows where the q1 band "
                               "misses the true Q1 error, offsets 0.5 / 1 / 2",
                          value=" / ".join(f"{v:.2f}" for v in misses),
                          source="runs/wbcp_signal/lq/results.json shared_bias q1 metrics.miss_q1err_logged.mean; "
                                 "SIGNAL_STUDY.md l.330-335"))
        facts.append(dict(what=f"shared_bias, {behaviour} data, kappa {kappa:g}: band's coverage of its own "
                               "residual at logged rows, offsets 0.5 / 1 / 2 (not drawn per cell)",
                          value=" / ".join(f"{c:.2f}" for c in covs),
                          source="runs/wbcp_signal/lq/results.json shared_bias q1 1 - metrics.miss_own_logged.mean"))
    # one coverage block per behaviour group: the mean over its 9 cells (each cell's value is in facts)
    for g, (behaviour, ys) in enumerate((("good", [r[2] for r in row_y[:3]]), ("poor", [r[2] for r in row_y[3:]]))):
        ax.text(-0.95, (ys[0] + ys[-1] + hgt) / 2, f"{behaviour}\ndata", ha="right", va="center", fontsize=LAB,
                color=S.INK, fontweight="semibold", linespacing=1.1)
        group_cov = float(np.mean(own_cov[9 * g:9 * (g + 1)]))
        ax.add_patch(FancyBboxPatch((col_x["own"] + 0.03, ys[-1] + 0.05), w - 0.06, ys[0] - ys[-1] + hgt - 0.1,
                                    boxstyle="round,pad=0,rounding_size=0.06", facecolor=S.BLUE, lw=0))
        ax.text(col_x["own"] + w / 2, (ys[0] + ys[-1] + hgt) / 2, f"{group_cov:.0f}%", ha="center", va="center",
                fontsize=VAL, color="white", fontweight="semibold")
        facts.append(dict(what=f"shared_bias, {behaviour} data: band's coverage of its own residual at logged rows, "
                               "mean over the 9 kappa x offset cells (blue block)",
                          value=f"{group_cov:.2f}%",
                          source="runs/wbcp_signal/lq/results.json shared_bias q1 1 - metrics.miss_own_logged.mean"))
    head = row_y[0][2] + hgt + 0.15
    ax.text(col_x["own"] + w / 2, head, "residual\ncovered", ha="center", va="bottom", fontsize=LAB,
            color=S.BLUE, fontweight="semibold", linespacing=1.05)
    ax.text((col_x[0.5] + col_x[2.0] + w) / 2, head + 0.62, "true Q1 error missed", ha="center", va="bottom",
            fontsize=LAB, color=S.ORANGE, fontweight="semibold")
    for off in OFFSETS:
        ax.text(col_x[off] + w / 2, head, f"offset {off:g}", ha="center", va="bottom", fontsize=NOTE, color=S.SOFT)
    ax.set_xlim(-2.2, 4.75)
    ax.set_ylim(row_y[-1][2] - 0.1, head + 1.35)
    ax.axis("off")
    facts += [
        dict(what="band's own residual coverage across all 18 cells (min-max)",
             value=f"{min(own_cov):.1f}-{max(own_cov):.1f}%", source="runs/wbcp_signal/lq/results.json; "
             "SIGNAL_STUDY.md l.325-326 ('about 91%')"),
        dict(what="nominal residual coverage", value="90% (alpha = 0.1)", source="SIGNAL_STUDY.md l.348-349"),
        dict(what="setting", value="LQ harness, shared_bias case, q1 band (all three signals identical here), "
             "logged actions, 5 replicates per cell", source="SIGNAL_STUDY.md l.263-273, l.325-335"),
    ]
    return fig, facts


# ---------------------------------------------------------------------------------------------------------------------


CHARTS = {
    "min_vs_q1": dict(
        make=min_vs_q1,
        headline="The min band misses Q1's error",
        takeaway="min band fails 10.5–82.4% of banks; Q1 band 4.0–5.1%",
        notes="On frozen TD3+BC critics for the five healthy D4RL pools, we calibrated WBCP on 4,000 banks of 1,024 "
              "independent rows each, with sigma refit for each signal. A band calibrated on BCA's current signal, "
              "the min(Q1, Q2) residual, fails for Q1's residual in 10.5% (pen-expert) to 82.4% (walker2d) of banks, "
              "against a 5% budget. Calibrating on Q1 itself fails in 4.0-5.1% of banks and still covers the min "
              "residual (at most 2.9% fail), at 2-10% more width. The earlier alignment study, with each pool's "
              "online sigma and 1,000 banks, found 8.2-70.2% for the min band; refitting sigma made the miss larger."),
    "q1_optimistic": dict(
        make=q1_optimistic,
        headline="The actor climbs Q1; the min hides it",
        takeaway="Illustration: Q1 pulls the actor twice as far; min sees none",
        notes="This is an illustration from the known-MDP linear-quadratic harness: the q1_optimistic case, poor "
              "data, kappa 1, where Q2 is exact and Q1 grows optimistic away from the logged actions. There "
              "min(Q1, Q2) equals the true Q, so the residual BCA calibrates carries none of Q1's error, 1.57 at "
              "Q1's peak. TD3+BC's actor climbs Q1, so along this slice its Q term pulls it 1.25 from the behaviour "
              "action while the true best is 0.63 away. Real D4RL heads disagree in both directions, so this one-sided case is stylized."),
    "dose_near_13": dict(
        make=dose_near_13,
        headline="The dose sits near 1.3 on every pool",
        takeaway="Every pool averages 1.26–1.33 on a 1.0–1.5 scale",
        notes="BCA's dose is m = 1 + 0.5 U/(U + u), so it can only lie between 1 and 1.5. On the five healthy "
              "TD3+BC pools with the Q1 signal, at each pool's mean deployed WBCP threshold, the per-row dose "
              "averages 1.26 (maze2d) to 1.33 (walker2d), with a row-to-row SD of only 0.017-0.050. In practice the "
              "hook is close to a uniform 1.3x BC weight."),
    "lq_scoreboard": dict(
        make=lq_scoreboard,
        headline="BCA loses to its controls twice as often",
        takeaway="Worse than both same-mean controls in 121 of 216 cells",
        notes="Step 3 compared BCA's change in true value J with a constant dose and a shuffled dose at the same "
              "mean, in 216 LQ cells (six error cases, three signals, good and poor data, kappa and offset "
              "settings, 5 replicates each). BCA was worse than both by more than 2 paired SE in 121 cells, better "
              "in 60, within 2 SE in 34 and mixed in 1. The differences are small, a median 6% of BCA's effect "
              "against no BCA, and 15 of the 60 wins are in cases where Q1 has no error to target. Post hoc, BCA's "
              "slightly stronger BC update (1-3%) explains most of the gaps: a strength-only model correlates 0.85 with "
              "them across all 216 cells."),
    "strength_ratio": dict(
        make=strength_ratio,
        headline="Same mean dose, not the same strength",
        takeaway="Post hoc: oracle's BC pull up to 2.6× the constant's; BCA's ≈1.0×",
        notes="Post hoc, not pre-registered. In the LQ harness the oracle dose, built from the true Q1 error, had a "
              "BC gradient a median 2.61x the same-mean constant dose's with good data and 2.26x with poor data in "
              "independent_errors, 1.19-1.24x in shared_bias and about 1x in q1_optimistic; BCA's was "
              "1.00-1.02x. So the oracle's large effects on J are mostly strength, not targeting, and step 4 "
              "matches realized strength instead of the mean dose."),
    "pilot_power": dict(
        make=pilot_power,
        headline="Pilot: oracle beats shuffled only on poor data",
        takeaway="Poor data: oracle +0.17, +0.08 over shuffled; expert: ≤ 0",
        notes="Step 4's pilot, one replicate excluded from all analysis, checked whether the harness can detect "
              "targeting at all: oracle weights against shuffled weights at matched strength, in block C's two "
              "localized cells (fixed point, strength S2), in units of the optimality gap. With expert data the "
              "oracle gained nothing, -0.004 for P1L (BC amplification) and -0.033 for P2L (trust weighting), "
              "against a forecast of at least 0.03; with poor data, not forecast, it gained 0.166 and 0.082, mostly from "
              "the tilt-local cell. After seeing this, gate G2 was moved to poor data, so it now certifies detection "
              "only when BC is harmful."),
    "shared_bias": dict(
        make=shared_bias,
        headline="No residual can see a shared bias",
        takeaway="Residual coverage 91%; true Q1 error missed on up to 100%",
        notes="In the LQ harness's shared_bias case both heads carry the same Bellman-consistent bias, so all three "
              "signals are identical. The q1 band covers its own residual on about 91% of logged rows in every "
              "cell, close to the nominal 90%. Yet it misses the true Q1 error on 0-1% of rows at the mildest "
              "settings and on 45-100% once kappa and the policy offset grow, for example 86.6% at kappa 2, offset "
              "0.5, good data. No residual-based signal can detect this bias."),
}


def make_all():
    """Render every chart of the group to runs/wbcp_viz/deck/<key>.png; return {key: {png, headline, ...}}."""
    import matplotlib.pyplot as plt

    out = {}
    for key, spec in CHARTS.items():
        fig, facts = spec["make"]()
        path = S.save_deck(fig, key)
        plt.close(fig)
        out[key] = dict(png=str(path), headline=spec["headline"], takeaway=spec["takeaway"], notes=spec["notes"],
                        facts=facts)
    return out


if __name__ == "__main__":
    for key, item in make_all().items():
        print(key, item["png"], len(item["facts"]), "facts")
