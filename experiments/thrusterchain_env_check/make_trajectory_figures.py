"""ThrusterChain trajectory figures: what the data hold, what each trained policy commands, and the score.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_env_check/make_trajectory_figures.py

One figure per data quality, written to docs/ood_environments/figures:

    fig_thrusterchain_medium_expert_trajectories.png, fig_thrusterchain_medium_trajectories.png

The same layout as the gridworld figures (experiments/ood_evidence/make_trajectory_figures.py), for a
continuous system with 12 setups.
    1  The data: every logged command on thrusters 1 and 2 (dataset seed 0); red = cut from training.
    2  The cut command, close up: thruster 1's command when the water pushes hard, logged against kept.
    3  Twelve small panels: thruster 1's command over an episode for each setup (TD3+BC, CQL, IQL, ReBRAC;
       alone, with standard BCA, with importance-weighted BCA) on the 10 fixed viewer starts of the
       strong-current-law bank, training seed 0 (the only seed whose episodes the study saved), with the
       stall-aware reference controller behind them.
    4  The score on that bank, one dot per training seed (0 to 4).

Trains nothing and plays no learned policy. Every trained-policy number comes from saved_runs_extract.npz,
a read-only copy of what the host study saved (extract_saved_runs.py). The reference controller's traces
are computed here from the pinned starts (pinned_reference.npz); it is a fixed controller, not a learner.
Every score difference the figures imply is printed and compared with the study's own summary
(outputs/shift_hosts_v4/RESULTS_SUMMARY.md); the script stops with an error if one differs.
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "ood_environments" / "figures"
sys.path.insert(0, str(ROOT))
import experiments.shift  # noqa: E402,F401  (dependency path and version checks)
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

# ---------------------------------------------------------------- style: the palette of the gridworld figures
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY, BAD = "#2a78d6", "#eb6834", "#1baf7a", "#a9a79f", "#d03b3b"
SEQ = LinearSegmentedColormap.from_list("rows", ("#eef4fd", "#86b6ef", "#3987e5", "#184f95"))
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.labelcolor": INK2, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8, "legend.frameon": False,
    "legend.fontsize": 8, "figure.dpi": 200, "savefig.dpi": 200, "savefig.facecolor": "white"})

LEARNERS = {"td3_bc": "TD3+BC", "cql": "CQL", "iql": "IQL", "rebrac": "ReBRAC"}
VARIANTS = {"alone": ("Host alone", BLUE), "standard": ("With standard BCA", ORANGE),
            "iw": ("With importance-weighted BCA", AQUA)}
SEEDS = range(5)
HARD_WATER = 0.65                       # world.py: the hold demand above which the recording controller holds
# The study's own summary (outputs/shift_hosts_v4/RESULTS_SUMMARY.md): standard BCA minus alone, IW minus standard.
SUMMARY = {("medium-expert", "td3_bc"): (0.515, 0.203), ("medium-expert", "iql"): (0.654, 0.198),
           ("medium-expert", "rebrac"): (-0.146, 0.104), ("medium", "td3_bc"): (-4.748, -2.684),
           ("medium", "iql"): (0.472, -0.049)}
CHECKS, VALUES = [], []


def check(label, value, expected, tol):
    CHECKS.append((label, float(value), float(expected), float(tol)))


def note(label, value):
    VALUES.append((label, value))


def figure_for(quality, saved, meta, reference):
    record = meta["qualities"][quality]
    bank = meta["banks"].index(meta["shown_bank"])
    knee = TC.KNEE
    action, removed, water = (saved[f"{quality}/data/{k}"] for k in ("action", "removed", "water"))
    gain = RC.textbook_gain(TC.STALL)
    demand = water @ gain[0, TC.CURRENT:]                       # thruster 1's hold demand: how hard the water pushes
    hard = demand > HARD_WATER
    holds = action[:, 0] == -np.float32(0.8)
    kept = ~removed
    past = np.abs(action) > knee
    note(f"{quality}: logged rows, removed, kept", [len(action), int(removed.sum()), int(kept.sum())])
    note(f"{quality}: hold rows logged, kept", [int(holds.sum()), int((holds & kept).sum())])
    note(f"{quality}: share of kept rows with a thruster past 80%", round(float(past[kept].any(1).mean()), 4))
    check(f"{quality}: logged rows match the study's record", len(action), record["removal"]["training_rows"], 0)
    check(f"{quality}: kept rows match the study's record", kept.sum(), record["removal"]["kept_training_rows"], 0)
    check(f"{quality}: kept hold rows match the study's record", (holds & kept).sum(), record["removal"]["kept_at_exactly_80"], 0)
    check(f"{quality}: every hold row is in hard water", (holds & ~hard).sum(), 0, 0)

    def points(learner, variant):
        return np.array([saved[f"{quality}/{learner}-{variant}/seed{s}/points"][bank] for s in SEEDS])

    def past_share(learner, variant):
        return float(np.mean([saved[f"{quality}/{learner}-{variant}/seed{s}/past_80_share"][bank] for s in SEEDS]))

    fig = plt.figure(figsize=(8.6, 14.6))
    outer = fig.add_gridspec(3, 1, height_ratios=(3.6, 5.5, 1.9), hspace=0.24, left=0.085, right=0.975, top=0.925,
                             bottom=0.085)
    top = outer[0].subgridspec(1, 2, wspace=0.26)
    a1, a2 = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    fig.suptitle(f"ThrusterChain, {quality} data: the helpful command that was cut from the data",
                 x=0.085, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)

    # ---- panel 1: the data on the map of commands
    a1.hexbin(action[kept, 0], action[kept, 1], gridsize=44, cmap=SEQ, bins="log", mincnt=1, linewidths=0,
              extent=(-1, 1, -1, 1))
    a1.scatter(action[removed, 0], action[removed, 1], s=2.0, color=BAD, alpha=0.35, linewidths=0, zorder=3)
    hold = record["hold_command"]
    a1.scatter([hold[0]], [hold[1]], marker="D", s=46, facecolor="white", edgecolor=INK, linewidth=1.4, zorder=5)
    a1.annotate("the 80% hold", (hold[0], hold[1]), xytext=(-0.62, 0.42), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    for x in (-knee, knee):
        a1.axvline(x, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=4)
    a1.text(-knee, 1.03, "80%", ha="center", va="bottom", fontsize=7.5, color=INK2)
    a1.text(knee, 1.03, "80%", ha="center", va="bottom", fontsize=7.5, color=INK2)
    box = a1.get_position()
    a1.set_position([box.x0, box.y0 + 0.20 * box.height, box.width, 0.80 * box.height])
    a1.set_xlim(-1.03, 1.03)
    a1.set_ylim(-1.03, 1.03)
    a1.set_aspect("equal")
    a1.set_xlabel("thruster 1 command", fontsize=8)
    a1.set_ylabel("thruster 2 command", fontsize=8)
    a1.set_title(f"The data: {len(action):,} logged steps, by their commands\non thrusters 1 and 2", pad=16)
    a1.legend(handles=[Line2D([], [], marker="h", ls="", color="#3987e5", markersize=7, label=f"kept for training ({int(kept.sum()):,})"),
                       Line2D([], [], marker="o", ls="", color=BAD, markersize=4, label=f"cut from training ({int(removed.sum()):,})")],
              loc="upper left", bbox_to_anchor=(-0.03, -0.14), ncol=2, columnspacing=1.0, handletextpad=0.2)

    # ---- panel 2: the cut command, close up
    box = a2.get_position()
    a2.set_position([box.x0, box.y0 + 0.50 * box.height, box.width, 0.50 * box.height])
    bins = np.linspace(-1, 1, 81)
    a2.hist([action[hard & kept, 0], action[hard & removed, 0]], bins=bins, stacked=True, color=["#86b6ef", BAD],
            edgecolor="white", linewidth=0.2)
    a2.set_yscale("log")
    a2.axvline(-knee, color=INK2, lw=0.8, ls=(0, (4, 3)))
    a2.annotate(f"the 80% hold:\n{int(holds.sum()):,} steps logged,\n{int((holds & kept).sum())} kept",
                (-knee, holds.sum() * 0.5), xytext=(-0.52, holds.sum() * 0.75), fontsize=8, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    a2.set_xlim(-1.03, 1.03)
    a2.set_xlabel("thruster 1 command, on steps where the water pushes hard", fontsize=8)
    a2.set_ylabel("logged steps (log scale)", fontsize=8)
    a2.grid(axis="y", color=GRID, linewidth=0.7)
    a2.set_axisbelow(True)
    a2.set_title("The cut command, close up")
    a2.legend(handles=[Line2D([], [], marker="s", ls="", color="#86b6ef", markersize=7, label="kept for training"),
                       Line2D([], [], marker="s", ls="", color=BAD, markersize=7, label="cut from training")],
              loc="upper left", bbox_to_anchor=(-0.03, -0.27), ncol=2, columnspacing=1.0, handletextpad=0.2)
    reading =(f"When the water pushes hard, the recording controller holds thruster 1 at exactly 80%. The log has "
               f"{int(holds.sum()):,} such steps; {int((holds & kept).sum())} are left in the training data. Random "
               f"commands past 80%, where the propeller stalls, were kept: {100 * past[kept].any(1).mean():.0f}% of "
               f"training steps have one. The study saved no critic value at the cut command, so this panel "
               f"cannot show the critic's error there.")
    a2.text(0.0, -0.52, "\n".join(textwrap.wrap(reading, 62)), transform=a2.transAxes, fontsize=8.5, color=INK,
            va="top", linespacing=1.3)

    # ---- the twelve setups: thruster 1's command over an episode
    grid = outer[1].subgridspec(4, 3, hspace=0.16, wspace=0.07)
    starts = saved[f"{quality}/td3_bc-alone/seed0/start"]
    ref = TR.rollout(RC.stall_aware(TC.STALL), reference[starts], TC.STALL, record=True).commands[:, :, 0]
    steps = np.arange(TC.STALL.length)
    first = None
    for r, (learner, learner_name) in enumerate(LEARNERS.items()):
        for c, (variant, (variant_name, color)) in enumerate(VARIANTS.items()):
            ax = fig.add_subplot(grid[r, c], sharex=first, sharey=first)
            first = first or ax
            key = f"{quality}/{learner}-{variant}/seed0"
            assert np.array_equal(saved[f"{key}/start"], starts), "every setup must show the same starts"
            commands = saved[f"{key}/commands"][:, :, 0]
            ax.plot(steps, ref.T, color=INK2, lw=0.6, alpha=0.45, zorder=2)
            ax.plot(steps, commands.T, color=color, lw=0.9, alpha=0.6, zorder=3)
            over = np.abs(commands) > knee
            ax.scatter(np.broadcast_to(steps, commands.shape)[over], commands[over], s=3.2, color=BAD, linewidths=0, zorder=4)
            ax.axhline(-knee, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
            pts = points(learner, variant)
            ax.text(0.985, 0.955, f"5 seeds: {pts.mean():.1f} points; a thruster past 80%\non {100 * past_share(learner, variant):.0f}% of steps",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2, linespacing=1.15)
            note(f"{quality}: {learner}-{variant} points (5 seeds), past-80% share", [round(float(pts.mean()), 3),
                                                                                    round(past_share(learner, variant), 4)])
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7.5, length=2)
            if r == 0:
                ax.set_title(variant_name, fontsize=9, pad=5)
            if c == 0:
                ax.set_ylabel(f"{learner_name}\nthruster 1 command", fontsize=8)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if r == len(LEARNERS) - 1:
                ax.set_xlabel("step of the episode", fontsize=8)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
    first.set_xlim(0, TC.STALL.length - 1)
    first.set_ylim(-1.08, 1.0)
    fig.text(0.085, outer[1].get_position(fig).y1 + 0.019,
             "What each trained policy commands: thruster 1 over an episode\n"
             "(the 10 fixed starts of the strong-current-law bank, training seed 0)", fontsize=9.5, fontweight="bold",
             color=INK, va="bottom", linespacing=1.25)
    handles = [Line2D([0], [0], color=color, lw=1.6, label="one episode of the trained policy") for _, color in [VARIANTS["alone"]]]
    handles += [Line2D([0], [0], color=INK2, lw=1.0, alpha=0.7, label="the same starts under the stall-aware reference controller (100 points)"),
                Line2D([], [], marker="o", ls="", color=BAD, markersize=3.5, label="thruster 1 past 80% (the propeller stalls)"),
                Line2D([0], [0], color=INK2, lw=0.8, ls=(0, (4, 3)), label="80% throttle")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, outer[1].get_position(fig).y0 - 0.028), ncol=2,
               fontsize=7.5, columnspacing=1.4, handlelength=1.8)

    # ---- the score, one dot per training seed
    a5 = fig.add_subplot(outer[2])
    box = a5.get_position()
    a5.set_position([box.x0, box.y0, box.width, 0.70 * box.height])
    low = 60.0
    below = {}
    for g, (learner, learner_name) in enumerate(LEARNERS.items()):
        means = {}
        for k, (variant, (variant_name, color)) in enumerate(VARIANTS.items()):
            x, pts = g * 4 + k, points(learner, variant)
            means[variant] = pts.mean()
            shown = pts[pts >= low]
            if len(shown):
                a5.scatter(x + np.linspace(-0.2, 0.2, len(shown)), np.sort(shown), s=20, color=color, edgecolor="white",
                           linewidth=0.7, zorder=3)
            if pts.mean() >= low:
                a5.plot([x - 0.3, x + 0.3], [pts.mean()] * 2, color=INK, lw=1.6, zorder=4, solid_capstyle="round")
                a5.text(x, pts.max() + 1.2, f"{pts.mean():.1f}", ha="center", va="bottom", fontsize=8, color=INK)
            if (pts < low).any():
                a5.scatter([x], [low + 0.9], marker="v", s=30, color=color, zorder=3)
                span = (f"at {pts.min():.0f}" if (pts < low).sum() == 1
                        else f"{pts.min():.0f} to {pts[pts < low].max():.0f}")
                below.setdefault(learner_name, []).append(
                    f"{variant_name[0].lower() + variant_name[1:]}: {int((pts < low).sum())} of 5 seeds "
                    f"({span}), average {pts.mean():.1f}")
        if (quality, learner) in SUMMARY:
            want = SUMMARY[(quality, learner)]
            check(f"{quality} {learner}: standard BCA minus alone matches the study's summary", means["standard"] - means["alone"], want[0], 0.0015)
            check(f"{quality} {learner}: IW BCA minus standard BCA matches the study's summary", means["iw"] - means["standard"], want[1], 0.0015)
        note(f"{quality}: {learner} standard minus alone, IW minus standard",
             [round(float(means["standard"] - means["alone"]), 3), round(float(means["iw"] - means["standard"]), 3)])
    a5.set_xticks([g * 4 + 1 for g in range(len(LEARNERS))], labels=list(LEARNERS.values()), fontsize=9)
    a5.tick_params(axis="x", length=0)
    a5.set_xlim(-0.8, len(LEARNERS) * 4 - 1.2)
    a5.set_ylim(low, 104)
    a5.grid(axis="y", color=GRID, linewidth=0.7)
    a5.set_axisbelow(True)
    a5.set_ylabel("points", fontsize=8)
    a5.set_title("Score on the strong-current-law bank, one dot per training seed\n"
                 "(0 = the policy of the random-quality data, 100 = the stall-aware controller; the axis starts at 60)")
    handles = [Line2D([], [], marker="o", ls="", color=color, markersize=5.5, label=name) for name, color in VARIANTS.values()]
    handles += [Line2D([0], [0], color=INK, lw=1.6, label="average of the five seeds (the printed number)"),
                Line2D([], [], marker="v", ls="", color=MUTED, markersize=5.5, label="seeds below the axis")]
    a5.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, -0.13), ncol=3, fontsize=7.5, columnspacing=1.2,
              handletextpad=0.3)
    if below:
        text = "Seeds below the axis. " + " ".join(f"{name}, " + "; ".join(parts) + "." for name, parts in below.items())
        a5.text(0.0, -0.52, "\n".join(textwrap.wrap(text, 150)), transform=a5.transAxes, fontsize=7.5, color=INK2,
                va="top", linespacing=1.3)
        note(f"{quality}: off the axis", text)

    OUT.mkdir(parents=True, exist_ok=True)
    name = f"fig_thrusterchain_{quality.replace('-', '_')}_trajectories.png"
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    meta = json.loads((HERE / "saved_runs_extract.json").read_text())
    gain = RC.textbook_gain(TC.STALL)
    for value, expected in zip(gain[0, TC.CURRENT:], (0.910, 0.822, -0.320)):
        check("thruster 1's hold demand uses the coefficients in world.py", value, expected, 0.0006)
    with np.load(HERE / "saved_runs_extract.npz", allow_pickle=False) as saved, \
            np.load(HERE / "pinned_reference.npz", allow_pickle=False) as pinned:
        reference = pinned[f"{meta['shown_bank']}/starts"]
        for quality in meta["qualities"]:
            figure_for(quality, saved, meta, reference)
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
