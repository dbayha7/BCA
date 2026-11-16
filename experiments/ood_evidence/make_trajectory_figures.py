"""Trajectory figures: what the data hold, what each trained policy does, and the count. Trains nothing.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/ood_evidence/make_trajectory_figures.py

One figure per environment, written to docs/ood_environments/figures:

    fig_honeypot_trajectories.png, fig_bottleneck_trajectories.png, fig_published_trajectories.png

Each figure has five panels.
    1  The data, drawn on the map. Honeypot and Bottleneck data are single steps (256 per cell), so
       the panel shows how often each move was executed at each cell. The published gridworld's data
       are whole runs, so the panel draws recorded runs.
    2  The unseen move, close up: its rows in the data, its true value, and the standard learner's
       critic's value, beside the best other move at the same cell.
    3, 4  Runs of the trained policies, played move by move with episode_sampler.py: 12 runs from
       each of the 27 saved policies of a method (9 datasets x 3 training seeds), so no single run
       is picked by hand.
    5  The registered outcome, exact for every saved policy, one dot per dataset.

Samples (saved_policies.py): Honeypot and Bottleneck, results/confirm_v5 of gridworld_ood_v1,
datasets 7501 to 7509; published gridworld, results/confirm_version_a of gridworld_critical_v1,
best-policy data, datasets 6501 to 6509.

Every number the outcome panels show is exact (from the environment's formulas), and is printed and
compared with the saved judged tables; the script stops with an error if one differs.
"""
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "ood_environments" / "figures"
sys.path.insert(0, str(HERE))
import episode_sampler as S  # noqa: E402
import saved_policies as SP  # noqa: E402

env, cenv = S.env, S.cenv
import collect  # noqa: E402
import evaluation as ev  # noqa: E402

RUN_SEED = 2026111503          # draws for the illustrated runs only; no number in a chart depends on it
PER_POLICY = 12                # runs drawn per saved policy

# ---------------------------------------------------------------- style: the palette of make_figures.py
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a79f"
BAD, GOODC, WARN, WALL, TINT = "#d03b3b", "#0ca30c", "#fab219", "#52514e", "#dbe9fb"
SEQ = LinearSegmentedColormap.from_list("rows", ("#eef4fd", "#86b6ef", "#3987e5", "#184f95"))
COLOR = {"data": INK2, "baseline": BLUE, "support_blind_width": ORANGE, "support_shuffled": GRAY, "complete": GRAY}
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.labelcolor": INK2, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8, "legend.frameon": False,
    "legend.fontsize": 8, "figure.dpi": 200, "savefig.dpi": 200, "savefig.facecolor": "white"})
CHECKS, VALUES = [], []


def check(label, value, expected, tol):
    CHECKS.append((label, float(value), float(expected), float(tol)))


def note(label, value):
    VALUES.append((label, value))


# ---------------------------------------------------------------- maps
def map_axes(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(10, 0)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(10) + 0.5, labels=[str(i) for i in range(10)])
    ax.set_yticks(np.arange(10) + 0.5, labels=[str(i) for i in range(10)])
    ax.tick_params(length=0, labelsize=7)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, pad=20)


def draw_cells(ax, fills, labels):
    for cell in range(100):
        r, c = divmod(cell, 10)
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=fills.get(cell, "#ffffff"), edgecolor=GRID, linewidth=0.8))
    for cell, (text, color) in labels.items():
        r, c = divmod(cell, 10)
        ax.text(c + 0.5, r + 0.55, text, ha="center", va="center", fontsize=10, fontweight="bold", color=color,
                zorder=7, path_effects=[pe.withStroke(linewidth=1.6, foreground="white")] if color == INK else None)


def under(ax, lines, width=64):
    """Plain text under a map, wrapped to the map's width. Returns the number of lines written."""
    wrapped = [part for line in lines for part in textwrap.wrap(line, width)]
    for k, line in enumerate(wrapped):
        ax.text(0, 10.42 + 0.52 * k, line, fontsize=8, color=INK2, va="top", ha="left", clip_on=False)
    return len(wrapped)


def inhouse_map(task, tinted, extra=None):
    fills = {c: WALL for c in task.walls}
    labels = {env.GOAL: ("G", "#ffffff"), **(extra or {})}
    fills[env.GOAL] = GOODC
    for hz in task.hazards:
        fills[hz] = BAD
        labels[hz] = ("H", "#ffffff")
    if task.low_exit is not None:
        fills[task.low_exit] = WARN
        labels[task.low_exit] = ("E", INK)
    for c in tinted:
        fills[c] = TINT
    return fills, labels


def published_map(task):
    fills = {c: WALL for c in task.walls}
    labels = {task.goal: ("G", "#ffffff"), task.start: ("S", INK)}
    fills[task.goal] = GOODC
    for c in task.lava:
        fills[c] = BAD
        labels[c] = ("L", "#ffffff")
    for c in (cenv.CRITICAL, cenv.SECOND_CRITICAL):
        fills[c] = TINT
    return fills, labels


def arrow(ax, src, dst, color=BAD, lw=2.0, start=0.16, stop=0.70):
    (r0, c0), (r1, c1) = divmod(int(src), 10), divmod(int(dst), 10)
    x0, y0, dx, dy = c0 + 0.5, r0 + 0.5, c1 - c0, r1 - r0
    patch = FancyArrowPatch((x0 + start * dx, y0 + start * dy), (x0 + stop * dx, y0 + stop * dy), arrowstyle="-|>",
                            mutation_scale=11, lw=lw, color=color, zorder=6, shrinkA=0, shrinkB=0)
    patch.set_path_effects([pe.withStroke(linewidth=lw + 2.2, foreground="white")])
    ax.add_patch(patch)


# ---------------------------------------------------------------- panel 1: the data
MARKER = ("^", ">", "v", "<")                     # up, right, down, left: environment.py's direction order


def coverage_panel(ax, task, counts, withheld, title):
    """Honeypot and Bottleneck: one small mark per cell and move, shaded by the rows in which it was executed."""
    fills, labels = inhouse_map(task, ())
    draw_cells(ax, fills, labels)
    free = task.free_cells()
    norm = Normalize(0, float(counts[free].max()))
    for cell in free:
        r, c = divmod(cell, 10)
        for direction, (dx, dy) in enumerate(env._DELTA):
            if (cell, direction) in withheld:
                continue
            ax.scatter(c + 0.5 + 0.29 * dx, r + 0.5 + 0.29 * dy, marker=MARKER[direction], s=17,
                       color=SEQ(norm(counts[cell, direction])), edgecolor=AXIS, linewidth=0.3, zorder=3)
    for cell, direction in withheld:
        arrow(ax, cell, task.move(cell, direction)[0])
    map_axes(ax, title)
    bar = plt.colorbar(ScalarMappable(norm, SEQ), cax=ax.inset_axes([0.0, -0.075, 0.46, 0.022]), orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, length=2)
    ax.text(4.9, 10.62, "rows in which the move was executed", fontsize=7.5, color=INK2, va="center", clip_on=False)
    ax.plot([0.15, 0.6], [11.55, 11.55], color=BAD, lw=2, clip_on=False)
    ax.text(0.8, 11.55, "move with 0 rows in all nine datasets (withheld)", fontsize=7.5, color=INK2, va="center",
            clip_on=False)


def recorded_runs(seeds, per_dataset):
    """Published gridworld: the first `per_dataset` recorded runs of each dataset, as (cells, ending)."""
    runs = []
    for seed in seeds:
        data = SP.published_dataset(seed)
        order = np.lexsort((data["step"], data["episode"]))
        episode, s, s2, done = (data[k][order] for k in ("episode", "s", "s2", "done"))
        for number in np.unique(episode)[:per_dataset]:
            rows = np.flatnonzero(episode == number)
            cells = np.append(s[rows], s2[rows[-1]]).astype(int)
            if done[rows[-1]] > 0:
                ending = S.GOAL_END if cells[-1] == cenv.GOAL else S.BAD_END
            else:
                ending = S.TIMEOUT
            runs.append((cells, ending))
    return runs


# ---------------------------------------------------------------- panels 3 and 4: runs of the trained policies
def sample_runs(world, records, starts, seed):
    """PER_POLICY runs from every saved policy, spread evenly over the start cells."""
    runs, per_start = [], PER_POLICY // len(starts)
    for k, record in enumerate(records):
        for start in starts:
            result = S.play(world, record["commands"], start, per_start,
                            rng=np.random.default_rng([seed, k, int(start)]), keep_paths=True)
            for i in range(per_start):
                runs.append((result["paths"][i, :result["length"][i] + 1].astype(int), int(result["ending"][i])))
    return runs


def draw_runs(ax, runs, color, seed):
    """Each run is one line, shifted a little so that runs on the same cells stay visible."""
    rng = np.random.default_rng(seed)
    for cells, ending in runs:
        dx, dy = rng.uniform(-0.3, 0.3, 2)
        xs, ys = cells % 10 + 0.5 + dx, cells // 10 + 0.5 + dy
        bad, out_of_time = ending == S.BAD_END, ending == S.TIMEOUT
        ax.plot(xs, ys, color=MUTED if out_of_time else color, alpha=0.22, lw=0.9, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
        if bad:                                   # only the last step is red, so bad endings are not overdrawn
            ax.plot(xs[-2:], ys[-2:], color=BAD, alpha=0.95, lw=1.5, zorder=5, solid_capstyle="round")
            ax.plot(xs[-1], ys[-1], marker="X", color=BAD, markersize=5.5, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=6)
        elif out_of_time:
            ax.plot(xs[-1], ys[-1], marker="o", color=MUTED, markersize=3.5, zorder=6)


def route_of(cells):
    """Name the route of one run that reached the goal in the published gridworld.

    cells: the cells the run visited, in order, start first (numpy integers, cell = 10 * row + column).
    Return a short label. Runs with the same label are counted together in the line under each map,
    for example "reached the goal: 300 (via cell 83: 270, via cell 67: 30)".

    The rule: the first of the two critical cells the run reaches. The wall row (row 6) can be passed
    only at cells 61 and 62 on the left or at cell 67 in the middle, and from the lower left the only
    way east is through cell 83 (lava blocks rows 7 and 9). So every run that reaches the goal passes
    cell 67 or cell 83, and the first one it reaches says which way it came down. A run that wanders
    into the other cell afterwards keeps its first label. `ways_to_the_goal` checks the map fact.
    """
    for cell in cells:
        if cell == cenv.SECOND_CRITICAL:
            return f"via cell {cenv.SECOND_CRITICAL}"
        if cell == cenv.CRITICAL:
            return f"via cell {cenv.CRITICAL}"
    raise ValueError("a run reached the goal without passing cell 67 or cell 83, which the map does not allow")


def outline(ax, cells):
    """A dark outline around cells that a panel refers to."""
    for cell in cells:
        r, c = divmod(int(cell), 10)
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=7))


def decisions_per_cell(task, policy):
    """Expected number of decisions an episode from the start makes at each cell (published gridworld, exact)."""
    inner = task.chain(policy)
    inner[:, list(task.terminals)] = 0.0
    here = np.zeros(S.CELLS)
    here[task.start] = 1.0
    made = np.zeros(S.CELLS)
    for _ in range(cenv.HORIZON):
        made += here
        here = here @ inner
    return made


def ways_to_the_goal(task, closed):
    """True if the goal can still be reached from the start when the cells in `closed` are shut."""
    seen, frontier = {task.start}, [task.start]
    while frontier:
        cell = frontier.pop()
        for move in task.legal_moves(cell):
            nxt = task.destination(cell, move)
            if nxt == task.goal:
                return True
            if nxt not in seen and nxt not in closed and nxt not in task.lava:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def count_line(runs, names, routes=False):
    """Text under a runs panel: how the drawn runs ended. `names` maps an ending to its words."""
    parts = []
    for ending, words in names.items():
        mine = [cells for cells, e in runs if e == ending]
        if callable(words):                                   # an ending split by the cell it happened in
            for label, n in words(mine):
                parts.append(f"{label}: {n}")
            continue
        text = f"{words}: {len(mine)}"
        if routes and ending == S.GOAL_END:
            split = Counter(route_of(cells) for cells in mine)
            if len(split) > 1:
                # non-breaking spaces keep each "via cell 83: 259" on one line when the text wraps
                text += " (" + ", ".join(f"{label}: {n}".replace(" ", " ") for label, n in split.most_common()) + ")"
        parts.append(text)
    return "Drawn runs. " + " · ".join(p for p in parts if not p.endswith(": 0")) + "."


def runs_legend(ax, color, bad_words, timeouts, lines_above):
    """Key under a runs panel, below `lines_above` lines of text."""
    handles = [Line2D([0], [0], color=color, lw=1.6, alpha=0.7, label="one run"),
               Line2D([0], [0], color=BAD, lw=1.5, marker="X", markersize=5.5, markeredgecolor="white", markevery=[1],
                      label=bad_words)]
    if timeouts:
        handles.append(Line2D([0], [0], color=MUTED, lw=1.2, marker="o", markersize=3.5, label="ran out of time"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.011 - 0.052 * lines_above),
              ncol=len(handles), handlelength=1.6, columnspacing=1.2, fontsize=7.5)


# ---------------------------------------------------------------- panel 2: the unseen move, close up
def closeup(ax, rows, title, xlabel, pad, reading):
    """rows: (label, true value, critic's value). A diamond for the truth, a dot for the critic.

    The panel is shorter than the map beside it, and each row's label is written inside it.
    `reading` is one or two plain sentences stating what the panel shows, written under it."""
    box = ax.get_position()
    ax.set_position([box.x0 + 0.015, box.y0 + 0.40 * box.height, box.width - 0.015, 0.55 * box.height])
    for y, (label, true, critic) in enumerate(rows):
        ax.text(0.012, y - 0.47, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5,
                color=INK, linespacing=1.25)
        ax.plot([true, critic], [y, y], color=GRAY, lw=2.2, zorder=1, solid_capstyle="round")
        ax.scatter([true], [y], marker="D", s=44, color=INK, zorder=3)
        ax.scatter([critic], [y], marker="o", s=58, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        left = true <= critic
        ax.text(true - pad if left else true + pad, y, f"{true:.2f}", ha="right" if left else "left", va="center",
                fontsize=8.5, color=INK)
        ax.text(critic + pad if left else critic - pad, y, f"{critic:.2f}", ha="left" if left else "right", va="center",
                fontsize=8.5, color=INK)
        ax.text((true + critic) / 2, y + 0.13, f"error {critic - true:+.2f}", ha="center", va="top", fontsize=8,
                color=INK2)
    ax.set_yticks([])
    ax.set_ylim(len(rows) - 0.5, -0.85)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title(title)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, markersize=6, label="true value"),
                       Line2D([], [], marker="o", ls="", color=BLUE, markersize=7, label="the critic's value")],
              loc="upper left", bbox_to_anchor=(-0.02, -0.26), ncol=2, columnspacing=1.2, handletextpad=0.3)
    ax.text(0.0, -0.44, "Error = the critic's value minus the true value.", transform=ax.transAxes, fontsize=8,
            color=INK2, va="top")
    ax.text(0.0, -0.58, "\n".join(textwrap.wrap(reading, 60)), transform=ax.transAxes, fontsize=8.5, color=INK,
            va="top", linespacing=1.3)


# ---------------------------------------------------------------- panel 5: the registered outcome
def dataset_means(records, method, value):
    """One number per dataset (its three training seeds averaged), in dataset order."""
    mine = [r for r in records if r["method"] == method]
    seeds = sorted({r["data_seed"] for r in mine})
    return np.array([np.mean([value(r) for r in mine if r["data_seed"] == s]) for s in seeds])


def column(ax, x, values, color, fmt, pad, marker="o"):
    """One dot per dataset, a bar at their mean, the mean written above."""
    values = np.atleast_1d(np.asarray(values, float))
    jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.zeros(1)
    ax.scatter(x + jitter, values, s=20, color=color, edgecolor="white", linewidth=0.7, zorder=3, marker=marker)
    mean = float(values.mean())
    ax.plot([x - 0.27, x + 0.27], [mean, mean], color=INK, lw=1.6, zorder=4, solid_capstyle="round")
    ax.text(x, values.max() + pad, fmt.format(mean), va="bottom", ha="center", fontsize=8.5, color=INK)
    return mean


def outcome_axes(ax, title, ylabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


DOT_KEY = [Line2D([], [], marker="o", ls="", color=MUTED, markersize=4.5, label="one dataset (its three training runs averaged)"),
           Line2D([0], [0], color=INK, lw=1.6, label="average of the nine datasets (the printed number)")]


def layout(suptitle):
    fig = plt.figure(figsize=(8.6, 11.6))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.56), hspace=0.50, wspace=0.16,
                            left=0.06, right=0.97, top=0.898, bottom=0.075)
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, :]))
    axes[4].axhline(0, color=AXIS, lw=0.8, zorder=1)
    fig.suptitle(suptitle, x=0.06, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    return fig, axes


def finish(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def by_method(records):
    return {m: [r for r in records if r["method"] == m] for m in SP.METHOD_NAMES}


# ================================================================ Honeypot and Bottleneck
SAVED = json.loads((SP.INHOUSE_BLOCK / "both.json").read_text())


def executed_counts(setting):
    """[dataset, cell, direction] rows in which each move was executed; also checks the withheld moves."""
    task = env.honeypot() if setting == "honeypot" else env.bottleneck_r2(float(setting[-2:]) / 10)
    withheld = [(c, d) for c, ds in collect.targeted_moves(task).items() for d in ds]
    out = []
    for seed in SP.INHOUSE_DATASETS:
        data = SP.inhouse_dataset(setting, seed)
        counts = np.zeros((100, 4), int)
        np.add.at(counts, (data["s"].astype(int), data["direction"].astype(int)), 1)
        for cell, direction in withheld:
            could = task.probabilities(data["a"][data["s"] == cell].astype(float))[:, direction] > 0
            assert counts[cell, direction] == 0 and not could.any(), f"{setting} {seed}: the withheld move has rows"
        out.append(counts)
    return task, withheld, np.array(out)


def single_move_table(task, record):
    """True value and critic's value of each single move at each cell, for one saved policy."""
    corner = [ev.target_corner(d) for d in range(4)]
    return task.direction_values(task.value(record["commands"])), record["q1"][:, corner]


def inhouse_closeup_rows(task, withheld, counts, records, unseen_label, other_label):
    """Average, over the standard learner's policies and the decision cells, of the withheld move and of the
    best other single move at the same cell (best by true value)."""
    seeds = list(SP.INHOUSE_DATASETS)
    unseen, other, rows_other, best_moves = [], [], [], Counter()
    for record in records:
        truth, critic = single_move_table(task, record)
        for cell, direction in withheld:
            unseen.append((truth[cell, direction], critic[cell, direction]))
            rest = [d for d in range(4) if d != direction]
            best = rest[int(np.argmax(truth[cell, rest]))]
            best_moves[(cell, env.DIRECTION_NAMES[best])] += 1
            other.append((truth[cell, best], critic[cell, best]))
            rows_other.append(counts[seeds.index(record["data_seed"]), cell, best])
    unseen, other = np.mean(unseen, axis=0), np.mean(other, axis=0)
    note(f"  close-up, best other move chosen (cell, move): count", dict(best_moves))
    return [(f"{unseen_label}: 0 rows in the data", *unseen),
            (f"{other_label}: {np.mean(rows_other):.0f} rows on average", *other)], unseen, other


def fig_honeypot():
    task, withheld, counts = executed_counts("honeypot")
    world = S.InHouseWorld(task)
    cells = sorted({c for c, _ in withheld})
    records = SP.inhouse("honeypot")
    groups = by_method(records)

    def hazard(r):
        return 100 * float(ev.unseen_hazard_probability(task, r["commands"])[cells].mean())

    fig, (a1, a2, a3, a4, a5) = layout("Honeypot: a harmful move that the data never show")
    coverage_panel(a1, task, counts.mean(axis=0), set(withheld),
                   "The data: 256 single steps at every cell.\nThe four steps into hazard 34 are missing")

    rows, unseen, other = inhouse_closeup_rows(task, withheld, counts, groups["baseline"], "Step into hazard 34",
                                               "Best other move at the same four cells")
    closeup(a2, rows, "The unseen move, close up\n(standard learner's critic, 27 trained policies)", "value", pad=0.6,
            reading=f"The critic has no row for the step into the hazard. It rates that step {unseen[1]:.2f}, almost "
                    f"as high as the best other move ({other[1]:.2f}). Its true value is {unseen[0]:.2f}.")
    a2.set_xlim(-7.2, 9.2)
    check("Honeypot: critic's error at the withheld move", unseen[1] - unseen[0], 9.88, 0.4)
    note("Honeypot close-up: true, critic at the withheld move", unseen.round(3).tolist())
    note("Honeypot close-up: true, critic at the best other move", other.round(3).tolist())

    endings = {S.GOAL_END: "reached the goal",
               S.BAD_END: lambda runs: [("ended in hazard 34", sum(c[-1] == env.UNSEEN_HAZARD for c in runs)),
                                        ("ended in another hazard", sum(c[-1] != env.UNSEEN_HAZARD for c in runs))],
               S.TIMEOUT: "ran out of time"}
    for ax, method, seed in ((a3, "baseline", 1), (a4, "support_blind_width", 2)):
        fills, labels = inhouse_map(task, cells)
        draw_cells(ax, fills, labels)
        runs = sample_runs(world, groups[method], cells, RUN_SEED + seed)
        draw_runs(ax, runs, COLOR[method], RUN_SEED + 10 + seed)
        exact = np.mean([hazard(r) for r in groups[method]])
        map_axes(ax, f"{SP.METHOD_NAMES[method]}: {len(runs)} runs from the\nfour cells beside hazard 34")
        lines = under(ax, [count_line(runs, endings), f"Exact, all 27 policies: {exact:.2f}% of episodes end in hazard 34."])
        runs_legend(ax, COLOR[method], "run that ends in a hazard", any(e == S.TIMEOUT for _, e in runs), lines)
        note(f"Honeypot runs drawn, {method}", count_line(runs, endings))

    demonstrator, _ = collect.demonstrator_commands(task, collect.targeted_moves(task))
    complete = [r for r in SP.inhouse("honeypot", arm="complete") if r["method"] == "baseline"]
    columns = [("The data's own policy\n(the demonstrator)", COLOR["data"], [hazard({"commands": demonstrator})], "D"),
               ("Standard\nlearner", COLOR["baseline"], dataset_means(records, "baseline", hazard), "o"),
               ("BCA\npolicy guard", COLOR["support_blind_width"], dataset_means(records, "support_blind_width", hazard), "o"),
               ("Shuffled\ncontrol", COLOR["support_shuffled"], dataset_means(records, "support_shuffled", hazard), "o"),
               ("Standard learner when\nthe move is in the data", COLOR["complete"], dataset_means(complete, "baseline", hazard), "s")]
    for x, (label, color, values, marker) in enumerate(columns):
        mean = column(a5, x, values, color, "{:.2f}%", 0.45, marker)
        note(f"Honeypot outcome, {label!r}", round(mean, 3))
    for key, method in (("baseline", "baseline"), ("version_a", "support_blind_width"), ("shuffled_control", "support_shuffled")):
        saved = 100 * np.array(SAVED["honeypot"]["hazard_endings_by_dataset"][key])
        check(f"Honeypot: per-dataset hazard endings match the saved table, {method}",
              np.abs(dataset_means(records, method, hazard) - saved).max(), 0.0, 1e-9)
    a5.set_xticks(range(len(columns)), labels=[c[0] for c in columns], fontsize=8)
    a5.set_xlim(-0.6, len(columns) - 0.4)
    a5.set_ylim(-0.7, 16.5)
    outcome_axes(a5, "Episodes that end in hazard 34 (lower is better)", "% of episodes from the four cells")
    fig.legend(handles=DOT_KEY, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.004))
    finish(fig, "fig_honeypot_trajectories.png")


def fig_bottleneck(shown=3.5):
    settings = {reward: f"bottleneck_r2_exit{int(round(10 * reward)):02d}" for reward in env.LOW_EXIT_REWARDS}
    task, withheld, counts = executed_counts(settings[shown])
    world = S.InHouseWorld(task)
    records = SP.inhouse(settings[shown])
    groups = by_method(records)

    fig, (a1, a2, a3, a4, a5) = layout("Bottleneck: a helpful move that the data never show")
    coverage_panel(a1, task, counts.mean(axis=0), set(withheld),
                   f"The data: 256 single steps at every cell.\nThe step right at B (the bridge) is missing")
    a1.text(env.B % 10 + 0.5, env.B // 10 + 0.5, "B", ha="center", va="center", fontsize=8, fontweight="bold", color=INK,
            zorder=7, path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

    rows, unseen, other = inhouse_closeup_rows(task, withheld, counts, groups["baseline"], "Step right onto the bridge",
                                               "Step up into the low exit")
    closeup(a2, rows, f"The unseen move, close up (low exit pays {shown})\n(standard learner's critic, 27 trained policies)",
            "value", pad=0.16,
            reading=f"The critic has no row for the bridge and rates it {abs(unseen[1] - unseen[0]):.2f} "
                    f"{'below' if unseen[1] < unseen[0] else 'above'} its true value. It still rates the bridge "
                    f"({unseen[1]:.2f}) a little above the low exit ({other[1]:.2f}), the best other move at B.")
    a2.set_xlim(1.6, 7.2)
    note("Bottleneck close-up: true, critic at the bridge", unseen.round(3).tolist())
    note("Bottleneck close-up: true, critic at the best other move", other.round(3).tolist())

    endings = {S.GOAL_END: "crossed and reached the goal", S.LOW_EXIT_END: "took the low exit",
               S.BAD_END: "ended in a hazard", S.TIMEOUT: "ran out of time"}
    for ax, method, seed in ((a3, "baseline", 1), (a4, "support_blind_width", 2)):
        fills, labels = inhouse_map(task, (env.B,), {env.B: ("B", INK)})
        draw_cells(ax, fills, labels)
        runs = sample_runs(world, groups[method], (env.B,), RUN_SEED + 20 + seed)
        draw_runs(ax, runs, COLOR[method], RUN_SEED + 30 + seed)
        exact = 100 * np.mean([task.outcomes(r["commands"])["goal"][env.B] for r in groups[method]])
        map_axes(ax, f"{SP.METHOD_NAMES[method]}: {len(runs)} runs from B\n(low exit pays {shown})")
        lines = under(ax, [count_line(runs, endings), f"Exact, all 27 policies: {exact:.1f}% of episodes from B reach the goal."])
        runs_legend(ax, COLOR[method], "run that ends in a hazard", any(e == S.TIMEOUT for _, e in runs), lines)
        note(f"Bottleneck runs drawn, {method}", count_line(runs, endings))

    series = (("The data's own policy (the demonstrator)", "data", "D"), ("Standard learner", "baseline", "o"),
              ("BCA policy guard", "support_blind_width", "o"), ("Shuffled control", "support_shuffled", "o"),
              ("Standard learner when the bridge is in the data", "complete", "s"))
    for g, (reward, setting) in enumerate(settings.items()):
        exit_task = env.bottleneck_r2(reward)

        def goal(r):
            return 100 * float(exit_task.outcomes(r["commands"])["goal"][env.B])

        mine = SP.inhouse(setting)
        complete = [r for r in SP.inhouse(setting, arm="complete") if r["method"] == "baseline"]
        demonstrator, _ = collect.demonstrator_commands(exit_task, collect.targeted_moves(exit_task))
        for k, (label, key, marker) in enumerate(series):
            values = ([goal({"commands": demonstrator})] if key == "data"
                      else dataset_means(complete, "baseline", goal) if key == "complete"
                      else dataset_means(mine, key, goal))
            mean = column(a5, g * (len(series) + 1) + k, values, COLOR[key], "{:.0f}%", 2.5, marker)
            note(f"Bottleneck outcome, low exit {reward}, {label!r}", round(mean, 3))
        for key, method in (("baseline", "baseline"), ("version_a", "support_blind_width")):
            saved = 100 * np.array(SAVED["bottleneck"]["by_exit"][str(reward)]["crossings_by_dataset"][key])
            check(f"Bottleneck: per-dataset crossings match the saved table, exit {reward}, {method}",
                  np.abs(dataset_means(mine, method, goal) - saved).max(), 0.0, 1e-9)
    width = len(series) + 1
    a5.set_xticks([g * width + (len(series) - 1) / 2 for g in range(3)],
                  labels=[f"low exit pays {reward}" for reward in settings], fontsize=8.5)
    a5.set_xlim(-0.8, 3 * width - 1.2)
    a5.set_ylim(-4, 118)
    outcome_axes(a5, "Episodes from B that cross the bridge and reach the goal (higher is better)", "% of episodes started at B")
    handles = [Line2D([], [], marker=marker, ls="", color=COLOR[key], markersize=5.5, label=label)
               for label, key, marker in series]
    a5.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.5, columnspacing=1.0,
              handletextpad=0.2)
    fig.legend(handles=DOT_KEY, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    finish(fig, "fig_bottleneck_trajectories.png")


# ================================================================ published gridworld
def fig_published(recipe="optimal"):
    task = cenv.published()
    world = S.PublishedWorld(task)
    records = SP.published(recipe)
    groups = by_method(records)
    moves, _ = task.best_policy()
    best = cenv.policy_from_moves(moves)
    best_goal = float(task.outcomes(best)["goal"][task.start])
    # The close-up shows the two cells where the policies differ (cell83_decomposition.py): beside the lava, off
    # the data's route, where the move "right" goes into lava. At cell 83 every saved policy commands "right".
    beside = {72: cenv.RIGHT, 92: cenv.RIGHT}
    assert all(task.destination(c, m) in task.lava for c, m in beside.items())

    def score(r):
        return float(task.outcomes(r["commands"])["goal"][task.start]) / best_goal

    # The map fact route_of relies on: each critical cell alone leaves a way open, both shut leave none.
    critical = (cenv.CRITICAL, cenv.SECOND_CRITICAL)
    assert all(ways_to_the_goal(task, {c}) for c in critical) and not ways_to_the_goal(task, set(critical))
    note("Published map: the goal is reachable with cell 67 shut, with cell 83 shut, with both shut",
         [ways_to_the_goal(task, {cenv.CRITICAL}), ways_to_the_goal(task, {cenv.SECOND_CRITICAL}),
          ways_to_the_goal(task, set(critical))])

    fig, (a1, a2, a3, a4, a5) = layout("Published gridworld: moves into lava that the data never command")
    endings = {S.GOAL_END: "reached the goal", S.BAD_END: "ended in lava", S.TIMEOUT: "ran out of time"}

    fills, labels = published_map(task)
    draw_cells(a1, fills, labels)
    recorded = recorded_runs(SP.PUBLISHED_DATASETS, 36)
    draw_runs(a1, recorded, COLOR["data"], RUN_SEED + 40)
    map_axes(a1, f"The data: {len(recorded)} recorded runs of the best policy\n(36 from each of the nine datasets)")
    lines = under(a1, [count_line(recorded, endings, routes=True)])
    runs_legend(a1, COLOR["data"], "run that ends in lava", any(e == S.TIMEOUT for _, e in recorded), lines)
    note("Published recorded runs drawn", count_line(recorded, endings, routes=True))

    outline(a1, beside)
    commanded, happened, safe = ({c: [] for c in beside} for _ in range(3))
    for seed in SP.PUBLISHED_DATASETS:
        data = SP.published_dataset(seed, recipe)
        for c, m in beside.items():
            here = data["s"] == c
            commanded[c].append(int((data["command"][here] == m).sum()))
            happened[c].append(int((data["executed"][here] == m).sum()))
            safe[c].append(int((data["command"][here] == moves[c]).sum()))
    assert all(sum(v) == 0 for v in commanded.values()), "the data are expected never to command these moves"
    pairs = [(c, m, int(moves[c])) for c, m in beside.items()]          # cell, the move into lava, the best policy's move
    truth, critic = [], []
    for r in groups["baseline"]:
        true_values = task.command_values(task.value(r["commands"]))
        truth.append([true_values[c][[m, b]] for c, m, b in pairs])
        critic.append([r["q1"][c][[m, b]] for c, m, b in pairs])
    truth, critic = np.array(truth), np.array(critic)                  # [policy, cell, (lava move, best move)]
    misranked = (critic[:, :, 0] > critic[:, :, 1]).sum(axis=0)        # policies whose critic prefers the lava move
    mean_truth, mean_critic = truth.mean(axis=(0, 1)), critic.mean(axis=(0, 1))

    def lava_directed(method):
        """% of the decisions made at the two cells that command the move into lava, over the method's 27 policies."""
        made = aimed = 0.0
        for r in groups[method]:
            decisions = decisions_per_cell(task, r["commands"])
            for c, m in beside.items():
                made += decisions[c]
                aimed += decisions[c] * task.command_probabilities(r["commands"][c])[m]
        return 100 * aimed / made

    def per_dataset(table, digits):
        return " and ".join(f"{np.mean(table[c]):.{digits}f}" for c in beside)

    share = {method: lava_directed(method) for method in SP.METHOD_NAMES}
    best_names = [cenv.MOVE_NAMES[b] for _, _, b in pairs]
    rows = [(f'Command "right" into lava at cells 72 and 92: commanded 0 times;\nit happened '
             f"{per_dataset(happened, 1)} times per dataset through the noise", mean_truth[0], mean_critic[0]),
            (f'The best policy\'s command there ("{best_names[0]}" at 72, "{best_names[1]}" at 92):\ncommanded '
             f"{per_dataset(safe, 0)} times per dataset", mean_truth[1], mean_critic[1])]
    closeup(a2, rows, "The never-commanded move, close up\n(cells 72 and 92, outlined on the maps)",
            "value (1 = certain to reach the goal now)", pad=0.022,
            reading=f"At these two cells the standard learner's critic rates the move into lava above the safe move in "
                    f"{misranked[0]} of {len(truth)} trained policies at cell 72 and {misranked[1]} of {len(truth)} at "
                    f"cell 92. The standard learner commands it on {share['baseline']:.1f}% of its decisions there, the "
                    f"BCA policy guard on {share['support_blind_width']:.1f}% and the shuffled control on "
                    f"{share['support_shuffled']:.1f}%.")
    a2.set_xlim(-0.12, 0.95)
    note("Published close-up, cells 72 and 92: lava move commanded per dataset", {c: float(np.mean(v)) for c, v in commanded.items()})
    note("Published close-up, cells 72 and 92: lava move happened per dataset", {c: round(float(np.mean(v)), 2) for c, v in happened.items()})
    note("Published close-up, cells 72 and 92: best command issued per dataset", {c: round(float(np.mean(v)), 1) for c, v in safe.items()})
    note("Published close-up: true, critic at the lava move", [round(float(mean_truth[0]), 3), round(float(mean_critic[0]), 3)])
    note("Published close-up: true, critic at the best policy's move", [round(float(mean_truth[1]), 3), round(float(mean_critic[1]), 3)])
    note("Published close-up: policies whose critic prefers the lava move, at cell 72 and at cell 92", misranked.tolist())
    note("Published close-up: % of decisions at the two cells that command the lava move", {k: round(v, 2) for k, v in share.items()})

    for ax, method, seed in ((a3, "baseline", 1), (a4, "support_blind_width", 2)):
        draw_cells(ax, fills, labels)
        outline(ax, beside)
        runs = sample_runs(world, groups[method], (task.start,), RUN_SEED + 50 + seed)
        draw_runs(ax, runs, COLOR[method], RUN_SEED + 60 + seed)
        outcomes = [task.outcomes(r["commands"]) for r in groups[method]]
        lava = 100 * np.mean([o["lava"][task.start] for o in outcomes])
        map_axes(ax, f"{SP.METHOD_NAMES[method]}: {len(runs)} runs from the start S")
        lines = under(ax, [count_line(runs, endings, routes=True),
                   f"Exact, all 27 policies: score {np.mean([score(r) for r in groups[method]]):.3f}; {lava:.1f}% end in lava."])
        runs_legend(ax, COLOR[method], "run that ends in lava", any(e == S.TIMEOUT for _, e in runs), lines)
        note(f"Published runs drawn, {method}", count_line(runs, endings, routes=True))

    saved = {(r["data_seed"], r["train_seed"], r["method"]): r["score"]
             for r in json.loads((SP.PUBLISHED_BLOCK / "confirmation_records.json").read_text()) if r["recipe"] == recipe}
    columns = [("The data's own policy\n(the best policy)", COLOR["data"], [1.0], "D"),
               ("Standard\nlearner", COLOR["baseline"], dataset_means(records, "baseline", score), "o"),
               ("BCA\npolicy guard", COLOR["support_blind_width"], dataset_means(records, "support_blind_width", score), "o"),
               ("Shuffled\ncontrol", COLOR["support_shuffled"], dataset_means(records, "support_shuffled", score), "o")]
    for x, (label, color, values, marker) in enumerate(columns):
        mean = column(a5, x, values, color, "{:.3f}", 0.004, marker)
        note(f"Published outcome, {label!r}", round(mean, 4))
    for method, expected in (("baseline", 0.953), ("support_blind_width", 0.990), ("support_shuffled", 0.984)):
        gap = max(abs(score(r) - saved[(r["data_seed"], r["train_seed"], method)]) for r in groups[method])
        check(f"Published: every run's score matches the saved record, {method}", gap, 0.0, 1e-9)
        check(f"Published: mean score, {method}", np.mean([score(r) for r in groups[method]]), expected, 0.0006)
    a5.set_xticks(range(len(columns)), labels=[c[0] for c in columns], fontsize=8)
    a5.set_xlim(-0.6, len(columns) - 0.4)
    a5.set_ylim(0.90, 1.012)
    outcome_axes(a5, "Score from the start (higher is better; the axis starts at 0.90)",
                 "goal rate ÷ the best policy's goal rate")
    fig.legend(handles=DOT_KEY, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.004))
    finish(fig, "fig_published_trajectories.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    fig_honeypot()
    fig_bottleneck()
    fig_published()
    print("\nPLOTTED VALUES")
    for label, value in VALUES:
        print(f"  {label}: {value}")
    print("\nCHECKS (value, expected, allowed gap)")
    failed = False
    for label, value, expected, tol in CHECKS:
        ok = abs(value - expected) <= tol
        failed |= not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {value:.6g} vs {expected:.6g} (+-{tol:g})")
    if failed:
        sys.exit("a plotted number differs from its checked value")
