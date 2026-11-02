"""Activity inside vs outside the withheld region, from the completed evaluation-only rerun.

Read-only: uses the 144 rerun context files and DATA_PLAN.json metadata. It never
decodes dataset or checkpoint arrays. Writes figures (PNG/SVG/PDF), CSV table views
and a manifest with source hashes into this folder.

Withheld region (the "Hole"): high load (obs[10]+obs[11] >= 0.4) and |a[0]| <= 0.25.
Three training seeds are the independent units; every figure shows them separately.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RERUN = ROOT / "experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1"
EVAL = RERUN / "run/evaluation"
FIG, TAB = HERE / "figures", HERE / "tables"

SEEDS = (0, 1, 2)
CONDITIONS = ("full", "hole", "random")
CONTEXTS = ("ordinary", "strong_current")
ARMS = ("native", "bca", "distance")
STEPS = (0, 10000, 15000, 20000)
FIXED_A0 = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
BAND = np.abs(FIXED_A0) <= 0.25          # slots 2-4 of the six-candidate bank
LOAD_EDGE, BAND_EDGE = 0.4, 0.25

# Reference palette (validated: first three categorical slots, all-pairs, light surface).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLOR = {"full": "#2a78d6", "hole": "#eb6834", "random": "#1baf7a"}
LABEL = {"full": "Full data", "hole": "Hole (region withheld)", "random": "Random removal (same count)"}
CONTEXT_LABEL = {"ordinary": "Ordinary current", "strong_current": "Strong current"}
REGION_WASH = dict(facecolor=MUTED, alpha=0.13, edgecolor="none", zorder=0)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": SURFACE,
})

_cache: dict = {}
_sources: dict = {}


def context_file(seed, condition, arm, step, context):
    stage = "warmup" if step in (0, 10000) else arm
    path = EVAL / f"seed{seed}_{condition}__{stage}__{step:05d}__{context}.json"
    if path not in _cache:
        raw = path.read_bytes()
        _sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        _cache[path] = json.loads(raw)
    return _cache[path]


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=200 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def table(name, header, rows):
    TAB.mkdir(exist_ok=True)
    with (TAB / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def header(fig, title, subtitle, legend=None, line=False):
    """Title, subtitle and optional shared legend in reserved space above the panels."""
    height = fig.get_figheight()
    lines = subtitle.count("\n") + 1
    fig.text(0.01, 1 - 0.10 / height, title, ha="left", va="top", fontsize=11, color=INK)
    fig.text(0.01, 1 - 0.38 / height, subtitle, ha="left", va="top", fontsize=8, color=INK2, linespacing=1.35)
    used = 0.38 + 0.17 * lines + 0.06
    if legend:
        handles = [plt.Line2D([], [], color=COLOR[c], marker="o", linestyle="-" if line else "none",
                              linewidth=2 if line else 0, markersize=7, markeredgecolor=SURFACE,
                              markeredgewidth=1.2) for c in legend]
        fig.legend(handles, [LABEL[c] for c in legend], loc="upper left", bbox_to_anchor=(0.005, 1 - used / height),
                   ncol=len(legend), handletextpad=0.4, columnspacing=1.6, labelcolor=INK2, borderaxespad=0)
        used += 0.30
    fig.subplots_adjust(top=1 - (used + 0.32) / height)


def withheld_rectangle(ax, x0=LOAD_EDGE, x1=10.0):
    ax.add_patch(Rectangle((x0, -BAND_EDGE), x1 - x0, 2 * BAND_EDGE, **REGION_WASH))


# ---------------------------------------------------------------- 1. training data
def fig_training_rows():
    plan = json.loads((RERUN / "DATA_PLAN.json").read_text(encoding="utf-8"))
    _sources[(RERUN / "DATA_PLAN.json").relative_to(ROOT).as_posix()] = hashlib.sha256(
        (RERUN / "DATA_PLAN.json").read_bytes()).hexdigest()
    rows = []
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    for i, c in enumerate(CONDITIONS):
        info = plan["conditions"][c]
        inside, total = info["training_rows_inside_hole"], info["training_rows"]
        rows.append([c, inside, total - inside, total, round(100 * inside / total, 2)])
        ax.barh(i, inside, height=0.5, color=COLOR[c], zorder=2)
        ax.text(inside + 150, i, f"{inside:,} of {total:,} rows ({100 * inside / total:.1f}%)",
                va="center", ha="left", color=INK2, fontsize=8.5)
    ax.set_yticks(range(3), [LABEL[c] for c in CONDITIONS])
    ax.invert_yaxis()
    ax.set_xlim(0, 14500)
    ax.set_xlabel("Training rows inside the withheld region (high load and |a0| <= 0.25)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Training data inside the withheld region", loc="left")
    save(fig, "01_training_rows_in_region")
    table("01_training_rows_in_region", ["condition", "rows_inside_region", "rows_outside_region",
                                         "training_rows", "percent_inside"], rows)


# ---------------------------------------------------------------- 2. visited action map
def visited(seed, condition, arm, step, context):
    cache = context_file(seed, condition, arm, step, context)["free"]["cached_visited"]
    obs = np.asarray(cache["observations"], np.float64).reshape(-1, 14)
    act = np.asarray(cache["actions"], np.float64).reshape(-1, 4)
    return obs[:, 10] + obs[:, 11], act[:, 0]


def fig_visited_map(context, step=20000, arm="native"):
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 7.4), sharex=True, sharey=True)
    rows = []
    loads_all = np.concatenate([visited(s, c, arm, step, context)[0] for s in SEEDS for c in CONDITIONS])
    x_max = max(LOAD_EDGE + 0.1, float(np.quantile(loads_all, 0.995)) * 1.05)
    for r, s in enumerate(SEEDS):
        for k, c in enumerate(CONDITIONS):
            ax = axes[r, k]
            load, a0 = visited(s, c, arm, step, context)
            inside = (load >= LOAD_EDGE) & (np.abs(a0) <= BAND_EDGE)
            high = load >= LOAD_EDGE
            withheld_rectangle(ax, x1=x_max * 2)
            ax.axvline(LOAD_EDGE, color=AXIS, linewidth=0.8, zorder=1)
            ax.scatter(load[~inside], a0[~inside], s=16, color=COLOR[c], alpha=0.55, linewidths=0, zorder=2)
            ax.scatter(load[inside], a0[inside], s=22, color=COLOR[c], edgecolors=INK, linewidths=0.6, zorder=3)
            free = context_file(s, c, arm, step, context)["free"]
            share = free["high_step_hole_fraction"]
            ax.text(0.02, 0.97, f"inside: {inside.sum()} of {high.sum()} high-load rows shown\n"
                                f"all 100 steps: {100 * share:.0f}% of high-load steps",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK2)
            rows.append([context, s, c, arm, step, int(high.sum()), int(inside.sum()),
                         free["high_steps"], free["hole_steps"], round(share, 4)])
            if r == 0:
                ax.set_title(LABEL[c], color=INK)
            if k == 0:
                ax.set_ylabel(f"Seed {s}\nfirst action coordinate a0")
            if r == 2:
                ax.set_xlabel("Load (obs[10] + obs[11])")
            ax.set_xlim(left=min(0.0, float(loads_all.min())), right=x_max)
            ax.set_ylim(-1.05, 1.05)
    header(fig, f"Where the trained native policy acts — {CONTEXT_LABEL[context].lower()}, {step // 1000}k checkpoint",
           "Gray area = withheld region (load >= 0.4 and |a0| <= 0.25); outlined dots fall inside it.\n"
           "Dots: visited states at steps 0, 25, 50 and 75 of 32 shared-start rollouts per panel.")
    save(fig, f"02_visited_action_map_{context}")
    return rows


# ---------------------------------------------------------------- 3. band usage over training
def band_share(seed, condition, arm, step, context):
    return context_file(seed, condition, arm, step, context)["free"]["high_step_hole_fraction"]


def fig_band_usage():
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.3})
    rows = []
    for r, ctx in enumerate(CONTEXTS):
        for k, s in enumerate(SEEDS):
            ax = axes[r, k]
            for c in CONDITIONS:
                ys = [100 * band_share(s, c, "native", t, ctx) for t in STEPS]
                ax.plot(np.array(STEPS) / 1000, ys, color=COLOR[c], linewidth=2, marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
                rows += [[ctx, s, c, "native", t, round(y, 2)] for t, y in zip(STEPS, ys)]
            ax.axvline(10, color=AXIS, linewidth=0.8, zorder=1)
            if r == 0:
                ax.set_title(f"Seed {s}")
            if k == 0:
                ax.set_ylabel(f"{CONTEXT_LABEL[ctx]}\nsteps inside region (%)")
            if r == 1 and k == 1:
                ax.set_xlabel("Training checkpoint (thousand critic updates)")
            ax.set_xticks([0, 10, 15, 20])
            ax.set_ylim(-3, 103)
    header(fig, "How often the native policy acts inside the withheld region during training",
           "Share of high-load steps (all 100 steps, 32 shared starts) with |a0| <= 0.25.\n"
           "0k = untrained, 10k = shared fork, 15k and 20k = native branch.", legend=CONDITIONS, line=True)
    save(fig, "03_region_usage_over_training")
    table("03_region_usage_over_training", ["context", "seed", "condition", "arm", "checkpoint",
                                            "percent_high_load_steps_inside_region"], rows)


def fig_band_usage_arms():
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.2), sharey=True, gridspec_kw={"hspace": 0.35})
    rows = []
    for r, ctx in enumerate(CONTEXTS):
        for k, arm in enumerate(ARMS):
            ax = axes[r, k]
            for i, c in enumerate(CONDITIONS):
                ys = [100 * band_share(s, c, arm, 20000, ctx) for s in SEEDS]
                xs = i + np.array([-0.12, 0.0, 0.12])
                ax.scatter(xs, ys, s=40, color=COLOR[c], edgecolors=SURFACE, linewidths=1.2, zorder=3)
                ax.plot([i - 0.22, i + 0.22], [np.mean(ys)] * 2, color=INK2, linewidth=1.2, zorder=2)
                rows += [[ctx, s, c, arm, 20000, round(y, 2)] for s, y in zip(SEEDS, ys)]
            ax.set_xticks(range(3), ["Full", "Hole", "Random"])
            ax.set_xlim(-0.6, 2.6)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title({"native": "Native", "bca": "BCA", "distance": "Distance control"}[arm])
            if k == 0:
                ax.set_ylabel(f"{CONTEXT_LABEL[ctx]}\nsteps inside region (%)")
            ax.set_ylim(-3, 103)
    header(fig, "Region usage at the final 20k checkpoint, by method",
           "Each dot is one training seed (left to right: 0, 1, 2); the short line is the seed mean.")
    save(fig, "04_region_usage_by_method")
    table("04_region_usage_by_method", ["context", "seed", "condition", "arm", "checkpoint",
                                        "percent_high_load_steps_inside_region"], rows)


# ---------------------------------------------------------------- 4. own-actor first action
def fig_actor_first_action(step=20000):
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.6), sharey=True, gridspec_kw={"hspace": 0.4})
    rows = []
    for r, ctx in enumerate(CONTEXTS):
        for k, s in enumerate(SEEDS):
            ax = axes[r, k]
            ax.axhspan(-BAND_EDGE, BAND_EDGE, **REGION_WASH)
            for i, c in enumerate(CONDITIONS):
                probe = context_file(s, c, "native", step, ctx)["q"]["per_probe"]
                high = np.asarray(probe["high_load"], bool)
                a0 = np.asarray(probe["actions"], np.float64)[:, 0, 0][high]
                jitter = (np.arange(a0.size) % 7 - 3) * 0.035
                ax.scatter(i + jitter, a0, s=18, color=COLOR[c], edgecolors=SURFACE, linewidths=0.8, zorder=3)
                inside = np.abs(a0) <= BAND_EDGE
                ax.text(i, 1.08, f"{inside.sum()}/{a0.size}", ha="center", va="bottom", fontsize=7.5, color=INK2)
                rows += [[ctx, s, c, "native", step, j, round(float(v), 5), bool(abs(v) <= BAND_EDGE)]
                         for j, v in enumerate(a0)]
            ax.set_xticks(range(3), ["Full", "Hole", "Random"])
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(-1.1, 1.25)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(f"Seed {s}", pad=14)
            if k == 0:
                ax.set_ylabel(f"{CONTEXT_LABEL[ctx]}\nactor's chosen a0")
    header(fig, "The native actor's chosen first action at high-load states (20k)",
           "Gray band = withheld action range |a0| <= 0.25. Numbers above each strip: anchors whose chosen a0 "
           "falls inside the band.")
    save(fig, "05_actor_first_action_high_load")
    table("05_actor_first_action_high_load", ["context", "seed", "condition", "arm", "checkpoint",
                                              "high_load_anchor_index", "a0", "inside_band"], rows)


# ---------------------------------------------------------------- 5. value landscape
def fig_value_landscape():
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.4), sharex=True, gridspec_kw={"hspace": 0.3})
    rows = []
    for r, ctx in enumerate(CONTEXTS):
        for k, s in enumerate(SEEDS):
            ax = axes[r, k]
            ax.axvspan(-BAND_EDGE, BAND_EDGE, **REGION_WASH)
            probe = context_file(s, "full", "native", 20000, ctx)["q"]["per_probe"]
            high = np.asarray(probe["high_load"], bool)
            qpi = np.asarray(probe["qpi"], np.float64)[high][:, 1:]
            centred = qpi - qpi.mean(axis=1, keepdims=True)
            for line in centred:
                ax.plot(FIXED_A0, line, color=MUTED, alpha=0.25, linewidth=0.8, zorder=1)
            ax.plot(FIXED_A0, centred.mean(axis=0), color=INK, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            best = np.bincount(centred.argmax(axis=1), minlength=5)
            for j, a in enumerate(FIXED_A0):
                rows.append([ctx, s, float(a), bool(BAND[j]), round(float(qpi[:, j].mean()), 4),
                             round(float(centred[:, j].mean()), 4), int(best[j]), int(high.sum())])
            ax.text(0.02, 0.03, f"best fixed action inside band at {best[BAND].sum()} of {high.sum()} anchors",
                    transform=ax.transAxes, fontsize=7.5, color=INK2)
            if r == 0:
                ax.set_title(f"Seed {s}")
            if k == 0:
                ax.set_ylabel(f"{CONTEXT_LABEL[ctx]}\nvalue vs anchor mean")
            if r == 1 and k == 1:
                ax.set_xlabel("Fixed first action a0 (other coordinates 0)")
            ax.set_xticks(FIXED_A0)
    header(fig, "True value of each fixed first action at high-load states",
           "Exact discounted Q^pi from the simulator, continuing with one fixed reference policy per seed (full-data "
           "native, 20k).\nEach anchor is centred on its own mean. Thin gray = one anchor; black = mean; gray band = "
           "withheld range.")
    save(fig, "06_true_value_by_action_high_load")
    table("06_true_value_by_action_high_load", ["context", "seed", "a0", "inside_band", "mean_qpi",
                                                "mean_qpi_minus_anchor_mean", "anchors_where_best",
                                                "high_load_anchors"], rows)


# ---------------------------------------------------------------- 6. critic error by action
def signed_error(seed, condition, step, context, load_side):
    probe = context_file(seed, condition, "native", step, context)["q"]["per_probe"]
    high = np.asarray(probe["high_load"], bool)
    mask = high if load_side == "high" else ~high
    q1 = np.asarray(probe["q1"], np.float64)[mask][:, 1:]
    qpi = np.asarray(probe["qpi"], np.float64)[mask][:, 1:]
    return q1 - qpi


def fig_critic_error(context, step=20000):
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.6), sharex=True, sharey="row", gridspec_kw={"hspace": 0.3})
    rows = []
    for r, side in enumerate(("high", "low")):
        for k, s in enumerate(SEEDS):
            ax = axes[r, k]
            if side == "high":
                ax.axvspan(-BAND_EDGE, BAND_EDGE, **REGION_WASH)
            ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
            for i, c in enumerate(CONDITIONS):
                err = signed_error(s, c, step, context, side)
                if err.shape[0] == 0:
                    if i == 0:
                        ax.text(0.5, 0.5, "no low-load anchors in this context", transform=ax.transAxes,
                                ha="center", va="center", fontsize=8, color=INK2)
                    continue
                offset = (i - 1) * 0.03
                ax.plot(FIXED_A0 + offset, err.mean(axis=0), color=COLOR[c], linewidth=2, marker="o",
                        markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
                lo, hi = np.quantile(err, [0.25, 0.75], axis=0)
                ax.vlines(FIXED_A0 + offset, lo, hi, color=COLOR[c], linewidth=1, alpha=0.6, zorder=2)
                for j, a in enumerate(FIXED_A0):
                    rows.append([context, s, c, step, side, float(a), bool(BAND[j] and side == "high"),
                                 round(float(err[:, j].mean()), 4), round(float(np.abs(err[:, j]).mean()), 4),
                                 round(float((err[:, j] > 0.1).mean()), 4), int(err.shape[0])])
            if r == 0:
                ax.set_title(f"Seed {s}")
            if k == 0:
                ax.set_ylabel(("High-load anchors\n(withholding applies)" if side == "high"
                               else "Low-load anchors\n(nothing withheld)") + "\ncritic minus true value")
            if r == 1 and k == 1:
                ax.set_xlabel("Fixed first action a0")
            ax.set_xticks(FIXED_A0)
    header(fig, f"Native critic error by action, inside vs outside the withheld region — {CONTEXT_LABEL[context].lower()}",
           "Q1 minus each policy's own exact Q^pi; above 0 = over-estimate. Dot = mean over anchors, bar = middle 50%.\n"
           "Gray band (top row) = withheld region; the bottom row is the same actions at low load, where nothing "
           "was withheld.", legend=CONDITIONS, line=True)
    save(fig, f"07_critic_error_by_action_{context}")
    return rows


# ---------------------------------------------------------------- 7. common-reference error
def fig_common_reference(step=20000):
    common = json.loads((EVAL / "COMMON_REFERENCE.json").read_text(encoding="utf-8"))
    _sources[(EVAL / "COMMON_REFERENCE.json").relative_to(ROOT).as_posix()] = hashlib.sha256(
        (EVAL / "COMMON_REFERENCE.json").read_bytes()).hexdigest()
    regions = [("high_band_fixed", "High\nband*"), ("high_retained_flanks", "High\nflanks"),
               ("low_band_fixed", "Low\nband"), ("low_retained_flanks", "Low\nflanks")]
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.8), sharey="row", gridspec_kw={"hspace": 0.55})
    rows = []
    for r, ctx in enumerate(CONTEXTS):
        for k, s in enumerate(SEEDS):
            ax = axes[r, k]
            ax.axvspan(-0.5, 0.5, **REGION_WASH)
            ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
            for i, c in enumerate(CONDITIONS):
                entry = common[f"seed{s}_{c}__native__{step}__{ctx}"]["panels"]
                means = [entry[key]["raw_score_deviation"]["mean_signed_error"] for key, _ in regions]
                maes = [entry[key]["raw_score_deviation"]["mae"] for key, _ in regions]
                xs = np.arange(4) + (i - 1) * 0.22
                present = [m is not None for m in means]
                ax.scatter(xs[present], [m for m in means if m is not None], s=34, color=COLOR[c],
                           edgecolors=SURFACE, linewidths=1.2, zorder=3)
                rows += [[ctx, s, c, step, key, "NA" if m is None else round(float(m), 4),
                          "NA" if a is None else round(float(a), 4)]
                         for (key, _), m, a in zip(regions, means, maes)]
            ax.set_xticks(range(4), [lab for _, lab in regions], fontsize=7.5)
            if r == 1 and k == 1:
                ax.set_xlabel("Load side and action region (* = withheld)")
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(f"Seed {s}")
            if k == 0:
                ax.set_ylabel(f"{CONTEXT_LABEL[ctx]}\ncritic minus reference")
    header(fig, "Native critic deviation against one shared target, by region (20k)",
           "Same fixed actions and anchors, scored against one reference policy per seed (full-data native, 20k),\n"
           "so every critic is compared on an identical target. Gray = withheld region. Strong current has no "
           "low-load anchors.", legend=CONDITIONS)
    save(fig, "08_common_reference_deviation_by_region")
    table("08_common_reference_deviation_by_region", ["context", "seed", "condition", "checkpoint", "region",
                                                      "mean_signed_deviation", "mean_absolute_deviation"], rows)


# ---------------------------------------------------------------- 8. return vs region usage
def fig_return_vs_usage(step=20000):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    rows = []
    for k, ctx in enumerate(CONTEXTS):
        ax = axes[k]
        for c in CONDITIONS:
            for s in SEEDS:
                free = context_file(s, c, "native", step, ctx)["free"]
                x, y = 100 * free["high_step_hole_fraction"], free["mean_return"]
                ax.scatter(x, y, s=46, color=COLOR[c], edgecolors=SURFACE, linewidths=1.2, zorder=3)
                ax.annotate(f"{s}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
                rows.append([ctx, s, c, step, round(x, 2), round(y, 4)])
        ax.set_title(CONTEXT_LABEL[ctx])
        ax.set_xlabel("High-load steps inside withheld region (%)")
        if k == 0:
            ax.set_ylabel("Mean discounted return (higher is better)")
    header(fig, "Region usage and return of the native policy (20k)",
           "Each dot is one seed (number beside it). Nine dots per panel; descriptive only.", legend=CONDITIONS)
    save(fig, "09_return_vs_region_usage")
    table("09_return_vs_region_usage", ["context", "seed", "condition", "checkpoint",
                                        "percent_high_load_steps_inside_region", "mean_discounted_return"], rows)


def main():
    fig_training_rows()
    visited_rows = []
    for ctx in CONTEXTS:
        visited_rows += fig_visited_map(ctx)
    table("02_visited_action_map", ["context", "seed", "condition", "arm", "checkpoint", "high_load_rows_shown",
                                    "inside_region_rows_shown", "high_load_steps_all", "inside_region_steps_all",
                                    "fraction_high_load_steps_inside"], visited_rows)
    fig_band_usage()
    fig_band_usage_arms()
    fig_actor_first_action()
    fig_value_landscape()
    error_rows = []
    for ctx in CONTEXTS:
        error_rows += fig_critic_error(ctx)
    table("07_critic_error_by_action", ["context", "seed", "condition", "checkpoint", "load_side", "a0",
                                        "inside_withheld_region", "mean_signed_error", "mean_absolute_error",
                                        "fraction_over_estimated", "anchors"], error_rows)
    fig_common_reference()
    fig_return_vs_usage()
    manifest = {"schema": "bca.ood_region_activity.v1",
                "withheld_region": "obs[10]+obs[11] >= 0.4 and |a[0]| <= 0.25",
                "sources_sha256": dict(sorted(_sources.items())),
                "figures": sorted(p.name for p in FIG.glob("*.png")),
                "tables": sorted(p.name for p in TAB.glob("*.csv")),
                "scope": "Read-only analysis of saved evaluation outputs; no dataset or checkpoint arrays decoded; "
                         "three training seeds are the independent units"}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(manifest["figures"]), "tables": len(manifest["tables"]),
                      "sources": len(_sources)}))


if __name__ == "__main__":
    main()
