"""Deck charts, group 'hosts': BCA's calibration across the four hosts (TD3+BC, CQL, ReBRAC, IQL) x seven datasets.

Every chart carries one message for a talked-over slide. Charts:
  host_heatmap     configured-bank failure (%) per dataset x host, flagged critics hatched
  rho_heatmap      within-episode dependence rho per dataset x host, with the spread across healthy hosts
  predictions_met  share of pre-registered Stage B predictions met on healthy critics, per host
  critic_health    which of the 28 critics are healthy and which are flagged

Values (all read and cross-checked by hand):
  - configured bank, uniform BCA, normalized score, BCA's own sampler, 4,000 banks per cell: section 10 of
    experiments/wbcp/results_td3_bc.ipynb (the table of all four hosts), equal to section 6 of each host's own
    notebook results_{cql,rebrac,iql}.ipynb. TD3+BC, CQL and ReBRAC use n = 1,024 (248 on pen-human); IQL its own
    configured bank, n = 8,192 (1,194 on pen-human), so its K is larger (configs/iql.yaml).
  - rho (normalized score): the Stage B tables of runs/wbcp_hosts/{cql,rebrac,iql}/expectations.md and their TD3+BC
    reference line; the same numbers are in section 10 of results_td3_bc.ipynb.
  - predictions met: section 5 scorecards of the four notebooks, restricted to healthy critics; also
    EXPERIMENTS.md entry 7.
  - critic health: cell 1 of each notebook and the Stage B 'Critic health' paragraphs of the expectation files;
    EXPERIMENTS.md entry 8.

Run from the repo root:  python -m experiments.wbcp.viz.deck_hosts
"""

import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, LogNorm, to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import experiments.wbcp.viz.style as S  # noqa: E402

HOSTS = ["TD3+BC", "CQL", "ReBRAC", "IQL"]
# Rows in the order of mean configured-bank failure across hosts, so the failure heatmap reads top to bottom from
# 'holds' to 'fails'. The same order is used in every grid of this group.
DATASETS = ["hopper", "halfcheetah", "pen-expert", "maze2d", "pen-cloned", "walker2d", "pen-human"]

NB = "experiments/wbcp/results_td3_bc.ipynb section 10 (all hosts); results_{cql,rebrac,iql}.ipynb section 6"

# Configured bank, uniform BCA, BCA sampler, normalized score: % of 4,000 banks failing.
FAIL = {
    "TD3+BC": {"hopper": 4.5, "halfcheetah": 5.3, "walker2d": 17.6, "maze2d": 6.4, "pen-cloned": 6.3,
               "pen-expert": 5.4, "pen-human": 33.3},
    "CQL": {"hopper": 5.1, "halfcheetah": 4.7, "walker2d": 10.5, "maze2d": 5.6, "pen-cloned": 8.4,
            "pen-expert": 5.9, "pen-human": 49.4},
    "ReBRAC": {"hopper": 4.3, "halfcheetah": 5.3, "walker2d": 18.3, "maze2d": 7.5, "pen-cloned": 8.3,
               "pen-expert": 7.1, "pen-human": 34.2},
    "IQL": {"hopper": 5.5, "halfcheetah": 5.2, "walker2d": 18.1, "maze2d": 5.1, "pen-cloned": 12.1,
            "pen-expert": 5.6, "pen-human": 52.4},
}
# Configured K per dataset (configs/<host>.yaml via the notebooks): TD3+BC, CQL, ReBRAC share n = 1,024; IQL n = 8,192.
K_TD3_FAMILY = {"hopper": 6, "halfcheetah": 6, "walker2d": 23, "maze2d": 5, "pen-cloned": 5, "pen-expert": 5,
                "pen-human": 124}
K_IQL = {"hopper": 17, "halfcheetah": 17, "walker2d": 67, "maze2d": 5, "pen-cloned": 11, "pen-expert": 7,
         "pen-human": 199}

# Within-episode correlation of misses at lambda*, normalized score.
RHO = {
    "TD3+BC": {"hopper": 0.016, "halfcheetah": 0.033, "walker2d": 0.120, "maze2d": 0.073, "pen-cloned": 0.108,
               "pen-expert": 0.054, "pen-human": 0.090},
    "CQL": {"hopper": 0.002, "halfcheetah": 0.028, "walker2d": 0.048, "maze2d": 0.057, "pen-cloned": 0.168,
            "pen-expert": 0.055, "pen-human": 0.201},
    "ReBRAC": {"hopper": 0.012, "halfcheetah": 0.034, "walker2d": 0.143, "maze2d": 0.153, "pen-cloned": 0.177,
               "pen-expert": 0.165, "pen-human": 0.105},
    "IQL": {"hopper": 0.012, "halfcheetah": 0.018, "walker2d": 0.050, "maze2d": 0.050, "pen-cloned": 0.213,
            "pen-expert": 0.064, "pen-human": 0.388},
}
RHO_SRC = ("runs/wbcp_hosts/{cql,rebrac,iql}/expectations.md Stage B tables and their TD3+BC reference line; "
           "results_td3_bc.ipynb section 10")

# Flagged critics and the reason each notebook prints (cell 1, 'Critic not converged').
FLAGGED = {
    "TD3+BC": {"pen-cloned": "Q out of\nrange", "pen-human": "diverged"},
    "CQL": {"maze2d": "still\ngrowing", "pen-cloned": "diverged", "pen-expert": "diverged",
            "pen-human": "Q out of\nrange"},
    "ReBRAC": {"maze2d": "still\ngrowing"},
    "IQL": {},
}
FLAG_SRC = ("cell 1 of results_{td3_bc,cql,rebrac,iql}.ipynb; Stage B 'Critic health' in "
            "runs/wbcp_hosts/{cql,rebrac,iql}/expectations.md; EXPERIMENTS.md entry 8")

# Pre-registered Stage B predictions met on healthy critics (met, total).
PRED = {"TD3+BC": (77, 80), "CQL": (46, 48), "ReBRAC": (91, 96), "IQL": (92, 98)}
PRED_SRC = ("section 5 scorecards of results_{td3_bc,cql,rebrac,iql}.ipynb restricted to healthy critics; "
            "EXPERIMENTS.md entry 7")

TRIALS = 4000
# A valid rule fails 5% of the time; with 4,000 banks the observed rate falls below this in 97.5% of runs.
NOISE_TOP = 5.0 + 1.96 * math.sqrt(0.05 * 0.95 / TRIALS) * 100  # 5.68
PASS_FILL = "#D3E1F2"  # lighter than ORANGE_LIGHT, so pass and fail also differ in lightness
DARK_ORANGE = "#9E5214"
FAIL_CMAP = LinearSegmentedColormap.from_list("fail", ["#F6DEC4", S.ORANGE_LIGHT, S.ORANGE, DARK_ORANGE])
RHO_CMAP = LinearSegmentedColormap.from_list("rho", ["#FBF3EA", S.ORANGE_LIGHT, S.ORANGE, DARK_ORANGE])
MUTED_CMAP = LinearSegmentedColormap.from_list("muted", ["#F1F0EC", "#D9DADA"])

TXT = 22  # cell numbers and labels (pt)
BIG = 30


def _fact(what, value, source):
    return {"what": what, "value": value, "source": source}


def _luminance(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _grid_axes(fig, left, bottom, width, height, nrows, ncols):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)
    ax.axis("off")
    return ax


def _tile(ax, col, row, color, gap=0.05, hatch=False):
    box = FancyBboxPatch((col + gap, row + gap), 1 - 2 * gap, 1 - 2 * gap,
                         boxstyle="round,pad=0,rounding_size=0.06", facecolor=color, edgecolor="none",
                         mutation_aspect=1.0)
    ax.add_patch(box)
    if hatch:
        with matplotlib.rc_context({"hatch.color": S.INK, "hatch.linewidth": 1.4}):
            ax.add_patch(FancyBboxPatch((col + gap, row + gap), 1 - 2 * gap, 1 - 2 * gap,
                                        boxstyle="round,pad=0,rounding_size=0.06", facecolor="none",
                                        edgecolor=(0, 0, 0, 0), hatch="//", linewidth=0, alpha=0.30))


def _headers(ax, ncols, labels, y=-0.12, size=TXT, colors=None):
    for j, lab in enumerate(labels):
        ax.text(j + 0.5, y, lab, ha="center", va="bottom", fontsize=size, color=(colors or {}).get(lab, S.INK),
                fontweight="semibold")


def _row_labels(ax, labels, x=-0.12, size=TXT, colors=None):
    for i, lab in enumerate(labels):
        ax.text(x, i + 0.5, lab, ha="right", va="center", fontsize=size, color=(colors or {}).get(lab, S.INK))


def _cell_text(ax, col, row, text, fill, flagged, size=TXT, weight="semibold"):
    color = "white" if _luminance(fill) < 0.5 else S.INK
    bbox = None
    if flagged:  # keep the number readable on top of the hatch
        bbox = dict(boxstyle="round,pad=0.18", facecolor=fill, edgecolor="none")
    ax.text(col + 0.5, row + 0.5, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight,
            bbox=bbox)


def _key(fig, x, y, items, size=20, dy=0.085):
    """A small swatch key: items = [(kind, color, label)], kind in {'fill', 'hatch'}."""
    ax = fig.add_axes([x, y - dy * len(items), 0.03, dy * len(items)])
    ax.set_xlim(0, 1)
    ax.set_ylim(len(items), 0)
    ax.axis("off")
    for i, (kind, color, label) in enumerate(items):
        outline = kind == "outline"
        ax.add_patch(FancyBboxPatch((0.05, i + 0.18), 0.9, 0.64, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="none" if outline else color,
                                    edgecolor=S.INK if outline else "none", linewidth=3.0 if outline else 0,
                                    clip_on=False))
        if kind == "hatch":
            with matplotlib.rc_context({"hatch.color": S.INK, "hatch.linewidth": 1.4}):
                ax.add_patch(Rectangle((0.05, i + 0.18), 0.9, 0.64, facecolor="none", edgecolor=(0, 0, 0, 0),
                                       hatch="//", alpha=0.30, clip_on=False))
        ax.text(1.35, i + 0.5, label, ha="left", va="center", fontsize=size, color=S.SOFT, clip_on=False)
    return ax


# ----------------------------------------------------------------------------------------------------------------------
# 1  host_heatmap


def host_heatmap():
    fig, _ = S.deck_figure(constrained_layout=False)
    fig.axes[0].remove()
    nr, nc = len(DATASETS), len(HOSTS)
    ax = _grid_axes(fig, 0.155, 0.035, 0.53, 0.84, nr, nc)
    norm = LogNorm(vmin=NOISE_TOP, vmax=55)
    for j, h in enumerate(HOSTS):
        for i, d in enumerate(DATASETS):
            v = FAIL[h][d]
            fill = PASS_FILL if v < NOISE_TOP else FAIL_CMAP(0.12 + 0.88 * norm(v))
            flagged = d in FLAGGED[h]
            _tile(ax, j, i, fill, hatch=flagged)
            _cell_text(ax, j, i, f"{v:.1f}%", fill, flagged)
    _headers(ax, nc, HOSTS)
    bad = [DATASETS.index("walker2d"), DATASETS.index("pen-human")]
    _row_labels(ax, DATASETS, colors={DATASETS[i]: DARK_ORANGE for i in bad})
    # Frame the two rows that fail on every host.
    top, bottom = min(bad), max(bad) + 1
    ax.add_patch(FancyBboxPatch((0.0, top), nc, bottom - top, boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="none", edgecolor=S.INK, linewidth=3.0, clip_on=False, zorder=4))
    _key(fig, 0.73, 0.62, [("fill", PASS_FILL, "within noise of 5%"),
                           ("fill", S.ORANGE, "over 5%"),
                           ("hatch", "#E9E6DE", "flagged critic")], dy=0.10)

    facts = []
    for h in HOSTS:
        facts.append(_fact(f"{h}: configured-bank failure % (uniform BCA, BCA sampler, normalized score), "
                           + ", ".join(DATASETS),
                           ", ".join(f"{FAIL[h][d]:.1f}" for d in DATASETS), NB))
    facts.append(_fact("blue cutoff: a valid rule's 95% Monte Carlo range at 4,000 banks reaches",
                       f"{NOISE_TOP:.2f}% (cells <= 5.6% are blue; every blue cell's printed 95% interval reaches "
                       "5%, e.g. ReBRAC hopper 4.3% [3.7, 5.0]; every orange cell's lies above 5%)",
                       "binomial SE at p = 0.05, 4,000 trials; intervals in results_td3_bc.ipynb section 10"))
    facts.append(_fact("flagged critics (hatched)", "; ".join(f"{h}: {', '.join(FLAGGED[h]) or 'none'}"
                                                             for h in HOSTS), FLAG_SRC))
    facts.append(_fact("configured K, TD3+BC/CQL/ReBRAC (n = 1,024; pen-human n = 248)",
                       ", ".join(f"{d} {K_TD3_FAMILY[d]}" for d in DATASETS), "results_td3_bc.ipynb sections 1 and 10"))
    facts.append(_fact("configured K, IQL (n = 8,192; pen-human n = 1,194)",
                       ", ".join(f"{d} {K_IQL[d]}" for d in DATASETS), "results_td3_bc.ipynb section 10"))
    facts.append(_fact("walker2d range across hosts", "10.5-18.3%", NB))
    facts.append(_fact("pen-human range across hosts", "33.3-52.4%", NB))
    return fig, facts


# ----------------------------------------------------------------------------------------------------------------------
# 2  rho_heatmap


def _healthy_spread(d):
    vals = [RHO[h][d] for h in HOSTS if d not in FLAGGED[h]]
    return max(vals) / min(vals), len(vals)


def rho_heatmap():
    fig, _ = S.deck_figure(constrained_layout=False)
    fig.axes[0].remove()
    nr, nc = len(DATASETS), len(HOSTS)
    ax = _grid_axes(fig, 0.155, 0.035, 0.53, 0.84, nr, nc)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=0.40)
    # Rows where the host moves rho materially (spread >= 3x among healthy critics and largest healthy rho >= 0.1)
    # keep the orange ramp, with their healthy min and max outlined; the other rows are muted to a grey ramp.
    # hopper's 8x is between values below 0.02, so it is muted.
    big_rows = {d for d in DATASETS if _healthy_spread(d)[0] >= 2.95
                and max(RHO[h][d] for h in HOSTS if d not in FLAGGED[h]) >= 0.1}
    for j, h in enumerate(HOSTS):
        for i, d in enumerate(DATASETS):
            v = RHO[h][d]
            flagged = d in FLAGGED[h]
            fill = RHO_CMAP(norm(v)) if d in big_rows else MUTED_CMAP(norm(v))
            _tile(ax, j, i, fill, hatch=flagged)
            label = f"{v:.3f}".lstrip("0")
            if d in big_rows:
                _cell_text(ax, j, i, label, fill, flagged)
            else:
                ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=TXT, color=S.SOFT,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor=fill, edgecolor="none") if flagged else None)
    for d in big_rows:
        i = DATASETS.index(d)
        healthy = [h for h in HOSTS if d not in FLAGGED[h]]
        for h in (min(healthy, key=lambda h: RHO[h][d]), max(healthy, key=lambda h: RHO[h][d])):
            ax.add_patch(FancyBboxPatch((HOSTS.index(h) + 0.035, i + 0.035), 0.93, 0.93,
                                        boxstyle="round,pad=0,rounding_size=0.07", facecolor="none",
                                        edgecolor=S.INK, linewidth=3.0, zorder=4))
    _headers(ax, nc, HOSTS)
    _row_labels(ax, DATASETS, colors={d: (DARK_ORANGE if d in big_rows else S.MUTED) for d in DATASETS})

    # The spread across healthy hosts: the host alone moves rho by this factor.
    ax_s = _grid_axes(fig, 0.695, 0.035, 0.10, 0.84, nr, 1)
    ax_s.text(0.5, -0.12, "max / min", ha="center", va="bottom", fontsize=20, color=S.SOFT)
    for i, d in enumerate(DATASETS):
        r, _ = _healthy_spread(d)
        big = d in big_rows
        ax_s.text(0.5, i + 0.5, f"{r:.1f}×", ha="center", va="center", fontsize=30 if big else 22,
                  color=DARK_ORANGE if big else S.MUTED, fontweight="semibold" if big else "normal")
    _key(fig, 0.815, 0.66, [("outline", "none", "min, max"), ("hatch", "#E9E6DE", "flagged critic")],
         dy=0.10)

    facts = []
    facts.append(_fact("highlight rule (orange rows, min and max healthy cells outlined)",
                       "spread >= 3x among healthy critics and largest healthy rho >= 0.1: "
                       + ", ".join(d for d in DATASETS if d in big_rows)
                       + "; hopper's 8x is between values below 0.02, so it is muted", "deck_hosts.py rho_heatmap"))
    facts.append(_fact("pen-human pool size", "about 13 episodes, so its rho is noisy; 2 healthy critics",
                       "runs/wbcp_hosts/{cql,rebrac,iql}/expectations.md Stage A item 7"))
    facts.append(_fact("same withheld episodes on every host", "identical population hashes (verify_pools)",
                       "runs/wbcp_hosts/rebrac/expectations.md Stage B; iql/expectations.md Stage B"))
    for h in HOSTS:
        facts.append(_fact(f"{h}: rho (normalized score), " + ", ".join(DATASETS),
                           ", ".join(f"{RHO[h][d]:.3f}" for d in DATASETS), RHO_SRC))
    for d in DATASETS:
        r, k = _healthy_spread(d)
        facts.append(_fact(f"{d}: max/min rho across the {k} healthy critics", f"{r:.2f}x", RHO_SRC))
    facts.append(_fact("Stage A forecast 'rho ranks datasets as TD3+BC does (Spearman >= 0.7)'",
                       "missed on every host: CQL 0.61, ReBRAC 0.57, IQL about 0.63",
                       "runs/wbcp_hosts/{cql,rebrac,iql}/expectations.md Stage A check; EXPERIMENTS.md entry 7"))
    facts.append(_fact("hopper is the lowest-rho dataset", "on every host", RHO_SRC))
    return fig, facts


# ----------------------------------------------------------------------------------------------------------------------
# 3  predictions_met


def predictions_met():
    fig, ax = S.deck_figure()
    order = sorted(HOSTS, key=lambda h: -PRED[h][0] / PRED[h][1])  # TD3+BC, CQL, ReBRAC, IQL
    y = list(range(len(order)))
    for yi, h in zip(y, order):
        met, tot = PRED[h]
        share = met / tot
        ax.barh(yi, share, height=0.62, color=S.BLUE, edgecolor="none")
        ax.barh(yi, 1 - share, left=share, height=0.62, color=S.ORANGE, edgecolor="none")
        ax.text(share - 0.012, yi, f"{100 * share:.0f}%", ha="right", va="center", fontsize=BIG, color="white",
                fontweight="semibold")
        ax.text(1.025, yi, f"{met} / {tot}", ha="left", va="center", fontsize=BIG, color=S.INK,
                fontweight="semibold")
    ax.set_yticks(y, order)
    ax.tick_params(axis="y", length=0, labelsize=26, pad=12, labelcolor=S.INK)
    ax.set_xticks([])
    ax.set_xlim(0, 1.21)
    ax.set_ylim(len(order) - 0.45, -0.75)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.text(1.0, -0.62, "missed", ha="right", va="bottom", fontsize=20, color=S.ORANGE, fontweight="semibold")
    ax.text(1.025, -0.62, "met / predicted", ha="left", va="bottom", fontsize=20, color=S.SOFT)

    tm = sum(PRED[h][0] for h in HOSTS)
    tt = sum(PRED[h][1] for h in HOSTS)
    facts = [_fact(f"{h}: Stage B predictions met on healthy critics", f"{PRED[h][0]} of {PRED[h][1]} "
                   f"({100 * PRED[h][0] / PRED[h][1]:.1f}%)", PRED_SRC) for h in order]
    facts.append(_fact("all four hosts, healthy critics", f"{tm} of {tt} ({100 * tm / tt:.1f}%)", PRED_SRC))
    facts.append(_fact("where the 16 misses are", "15 are banks drawn from a few episodes (whole-episode banks; "
                       "pen-human K = 124/199); the other is IQL walker2d K = 67, raw score", PRED_SRC))
    facts.append(_fact("pre-registered tolerance", "about +/-2 points, or +/-5 where the design effect D > 3",
                       "results_*.ipynb section 5; runs/wbcp_hosts/*/expectations.md"))
    facts.append(_fact("all critics, flagged included (notebook scorecards)",
                       "TD3+BC 105/112, CQL 101/112, ReBRAC 107/112, IQL 92/98: 405 of 434 (93.3%)",
                       "'Scorecard' line of section 5, results_{td3_bc,cql,rebrac,iql}.ipynb"))
    facts.append(_fact("timing", "CQL, ReBRAC, IQL predictions written before any benchmark run; TD3+BC's after "
                       "DEPENDENCE.md Change 12 had run and read many of the same designs (older samplers)",
                       "runs/wbcp_hosts/td3_bc/expectations.md 'What is already known'"))
    return fig, facts


# ----------------------------------------------------------------------------------------------------------------------
# 4  critic_health


def critic_health():
    fig, _ = S.deck_figure(constrained_layout=False)
    fig.axes[0].remove()
    nr, nc = len(DATASETS), len(HOSTS)
    ax = _grid_axes(fig, 0.155, 0.035, 0.50, 0.84, nr, nc)
    for j, h in enumerate(HOSTS):
        for i, d in enumerate(DATASETS):
            reason = FLAGGED[h].get(d)
            fill = S.ORANGE if reason else PASS_FILL
            _tile(ax, j, i, fill)
            if reason:
                one_line = reason.replace("\n", " ")
                ax.text(j + 0.5, i + 0.5, one_line if len(one_line) <= 13 else reason, ha="center", va="center",
                        fontsize=20, color="white", fontweight="semibold", linespacing=1.0)
    _headers(ax, nc, HOSTS)
    _row_labels(ax, DATASETS)
    n_flag = sum(len(FLAGGED[h]) for h in HOSTS)
    # The count doubles as the key: a big number, its swatch, its word.
    ax_note = fig.add_axes([0.71, 0.05, 0.28, 0.82])
    ax_note.axis("off")
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    for y, n, word, fill, color in [(0.80, nr * nc - n_flag, "healthy", PASS_FILL, S.BLUE),
                                    (0.52, n_flag, "flagged", S.ORANGE, S.ORANGE)]:
        ax_note.text(0.30, y, f"{n}", ha="right", va="center", fontsize=60, color=color, fontweight="semibold")
        ax_note.add_patch(FancyBboxPatch((0.37, y - 0.05), 0.10, 0.10, boxstyle="round,pad=0,rounding_size=0.015",
                                         facecolor=fill, edgecolor="none"))
        ax_note.text(0.52, y, word, ha="left", va="center", fontsize=26, color=color, fontweight="semibold")
    ax_note.text(0.37, 0.41, "excluded from\nconclusions", ha="left", va="top", fontsize=20, color=S.SOFT,
                 linespacing=1.15)
    # Honesty notes: the thresholds are post hoc, and IQL could only get the weaker check (it logs no Q series).
    ax_note.text(0.02, 0.0, "post-hoc thresholds\nIQL: range check only", ha="left", va="bottom", fontsize=20,
                 color=S.MUTED, linespacing=1.15)

    facts = [_fact(f"{h}: flagged critics", ", ".join(f"{d} ({r.replace(chr(10), ' ')})"
                                                      for d, r in FLAGGED[h].items()) or "none", FLAG_SRC)
             for h in HOSTS]
    facts.append(_fact("flagged critics, total", f"{n_flag} of {nr * nc}", FLAG_SRC))
    facts.append(_fact("health thresholds", "set post hoc, after TD3+BC's pen pools were found to diverge",
                       "EXPERIMENTS.md entry 8; runs/wbcp_hosts/cql/expectations.md Stage B"))
    facts.append(_fact("IQL's check is weaker", "IQL logs no Q series, so only the loss ratio and the Q-range check "
                       "apply; 'still growing' cannot be detected on IQL", "runs/wbcp_hosts/iql/expectations.md Stage B"))
    facts.append(_fact("when health was read", "CQL, ReBRAC, IQL: before any benchmark run; TD3+BC: known from "
                       "earlier work before its host matrix", "runs/wbcp_hosts/{cql,rebrac,iql,td3_bc}/expectations.md"))
    facts.append(_fact("examples", "CQL pen-cloned Q 4.4e7 and pen-expert Q 3.9e7 (diverged); CQL pen-human Q -7,058 "
                       "below its floor of -86; CQL maze2d Q +96% over the last 20% of training",
                       "runs/wbcp_hosts/cql/expectations.md Stage B"))
    return fig, facts


# ----------------------------------------------------------------------------------------------------------------------

CHARTS = {
    "host_heatmap": (
        host_heatmap,
        "Same two datasets fail on every host",
        "walker2d fails 10–18%, pen-human 33–52%, on all four hosts.",
        "Each cell is the share of 4,000 simulated banks whose certified threshold misses more than 10% of test "
        "rows, for the bank each host actually deploys (uniform BCA, BCA's sampler, normalized score); a valid rule "
        "fails 5%. Hopper and halfcheetah stay within noise of 5% on all four hosts. walker2d fails 10.5-18.3% and "
        "pen-human 33.3-52.4% everywhere, because their banks take many rows per episode (K = 23 and 124; IQL's "
        "8,192-row bank needs 67 and 199). Hatched cells are flagged critics: pen-human's is flagged on TD3+BC and "
        "CQL, but it fails as badly on the two healthy hosts (34.2% ReBRAC, 52.4% IQL). IQL uses its own, larger "
        "bank.",
    ),
    "rho_heatmap": (
        rho_heatmap,
        "The host changes ρ, not just the data",
        "Same episodes, another host: ρ moves about 3× on healthy critics.",
        "rho is the within-episode correlation of misses; it sets how much a K-per-episode bank is inflated. We "
        "pre-registered that rho is mostly a property of the dataset, and that forecast missed on every host "
        "(rank correlation with TD3+BC's 0.57-0.63 against 0.7). Hopper is lowest everywhere, but on the same "
        "withheld episodes the host alone moves rho 3.0x on walker2d (0.048 to 0.143) and 3.1x on pen-expert among "
        "healthy critics; pen-human's 3.7x rests on two healthy critics and about 13 episodes, and hopper's 8x is "
        "between tiny values (0.002 vs 0.016). So K has to be chosen per host and dataset.",
    ),
    "predictions_met": (
        predictions_met,
        "95% of predictions came true",
        "306 of 322 pre-registered predictions met on healthy critics.",
        "Before each host's benchmark runs we wrote down a Monte Carlo failure prediction for every bank design, "
        "with a tolerance of about two points (five where the design effect exceeds 3). On healthy critics they "
        "held 77 of 80 times for TD3+BC, 46 of 48 for CQL, 91 of 96 for ReBRAC and 92 of 98 for IQL; with the "
        "flagged critics included it is 405 of 434 (93%). 15 of the 16 misses are banks drawn from a few "
        "episodes (whole-episode banks, pen-human's K = 124 or 199), a risk the pre-registrations named; the other "
        "is IQL walker2d at K = 67 on the raw score. TD3+BC's predictions were written after the dependence study "
        "had already run many of the same designs, so its 77 of 80 is the weakest test of the four.",
    ),
    "critic_health": (
        critic_health,
        "7 of 28 critics failed the health check",
        "7 of 28 flagged, by post-hoc thresholds; none on IQL.",
        "Each host's critic was checked after 100,000 updates, on CQL, ReBRAC and IQL before any benchmark was read. "
        "Seven are flagged: CQL "
        "on maze2d and all three pen datasets, ReBRAC on maze2d, and TD3+BC on pen-cloned and pen-human, by "
        "divergence, Q outside its possible range, or Q still growing. Their calibration numbers are exact for "
        "those scores but are kept out of every conclusion. The thresholds were set after TD3+BC's pen critics "
        "diverged, so they are post hoc, and IQL logs no Q series, so on IQL only the range check could run.",
    ),
}


def make_all():
    out = {}
    for key, (fn, headline, takeaway, notes) in CHARTS.items():
        fig, facts = fn()
        path = S.save_deck(fig, key)
        plt.close(fig)
        out[key] = {"png": str(path), "headline": headline, "takeaway": takeaway, "notes": notes, "facts": facts}
    return out


if __name__ == "__main__":
    import json

    res = make_all()
    print(json.dumps({k: v["png"] for k, v in res.items()}, indent=1))
