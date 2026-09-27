"""Render the accepted first TD3+BC pair from saved summaries only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def render(host, bca):
    h, b = read(host / "audit.json"), read(bca / "audit.json")
    pair = read(bca / "comparison.json")
    assert h["accepted"] and b["accepted"] and pair["completed_paired_seeds"] == 1
    assert h["run_id"] == "td3_bc-hopper-host-s202609171"
    assert b["run_id"] == "td3_bc-hopper-bca-noiw-s202609171"
    banks = [read(p / "evaluations.json") for p in [host, bca]]
    blocks = [read(p / "metric_blocks.json") for p in [host, bca]]
    refreshes = read(bca / "refreshes.json")
    colors, labels = ["#526d8c", "#c35232"], ["Plain host", "BCA, no IW"]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 14.3), layout="constrained")
    fig.suptitle("TD3+BC / Hopper — first standard BCA pair at 1M updates\n"
                 "Seed 202609171 • one of five paired training seeds", fontsize=19)
    for index, method in enumerate(banks):
        x = [r["step"] / 1e6 for r in method[:-1]]
        means = [np.mean([e["normalized_score"] for e in r["episodes"]]) for r in method[:-1]]
        axes.flat[0].plot(x, means, color=colors[index], lw=1.0, label=labels[index])
        final = [e["normalized_score"] for e in method[-1]["episodes"]]
        axes.flat[1].plot(range(1, 21), final, ".-", color=colors[index], lw=.65,
                          label=f"{labels[index]}: {np.mean(final):.3f}")
    axes.flat[0].set(title="All 200 periodic banks", xlabel="Host updates (millions)", ylabel="Mean normalized score")
    axes.flat[1].set(title="All 20 paired final episodes", xlabel="Declared episode index", ylabel="Normalized score")
    axes.flat[0].legend(frameon=False, fontsize=8)
    axes.flat[1].legend(frameon=False, fontsize=8)
    delta = pair["paired_episode_deltas"]
    axes.flat[2].bar(range(1, 21), delta, color=[colors[1] if x > 0 else colors[0] for x in delta])
    axes.flat[2].axhline(0, color="#777777", lw=.8)
    axes.flat[2].set(title=f"BCA minus host: {pair['paired_episode_wins']}/20 wins",
                     xlabel="Declared episode index", ylabel="Paired final score difference")
    panels = [(3, "q_mean", "Actor-action Q1", "Mean Q1 on actor rows"),
              (4, "lambda", "Native Q normalization", "Alpha / mean(abs(Q1))"),
              (5, "actor_loss", "Actor objective", "Mean native actor loss"),
              (6, "critic_loss", "Critic objective", "Sum of twin MSEs"),
              (7, "bc_loss", "Consumed BC term", "Host error / BCA dose-weighted error")]
    for index, key, title, ylabel in panels:
        for method, color, label in zip(blocks, colors, labels):
            axes.flat[index].plot([r["step"] / 1e6 for r in method],
                                  [r[key]["mean"] for r in method], color=color, lw=1, label=label)
        axes.flat[index].set(title=title, xlabel="Host updates (millions)", ylabel=ylabel)
    method = blocks[1]
    x = [r["step"] / 1e6 for r in method]
    axes.flat[8].plot(x, [r["scale_loss"]["mean"] for r in method], color=colors[1], lw=1)
    axes.flat[8].set(title="BCA scale-fitting objective", xlabel="Host updates (millions)", ylabel="Mean scale loss; host N/A")
    rx = [r["step"] / 1e6 for r in refreshes]
    axes.flat[9].plot(rx, [r["residual_unit"] for r in refreshes], color=colors[1], label="Frozen refresh unit")
    axes.flat[9].scatter([c["step"] / 1e6 for c in b["checkpoints"]],
                         [c["live_residual_scale"] for c in b["checkpoints"]],
                         color="#305f74", s=30, label="Live EMA at checkpoints", zorder=3)
    axes.flat[9].set(title="BCA residual-scale records", xlabel="Host updates (millions)", ylabel="Residual unit; host N/A")
    axes.flat[9].legend(frameon=False, fontsize=8)
    axes.flat[10].plot(x, [r["actor_bc_multiplier_mean"]["mean"] for r in method], color=colors[1], lw=1)
    axes.flat[10].axhline(1, color=colors[0], lw=.8, label="Host implicit unit strength")
    axes.flat[10].set(title="BCA consumed actor BC multiplier", xlabel="Host updates (millions)", ylabel="Mean multiplier on actor rows")
    axes.flat[10].legend(frameon=False, fontsize=8)
    for key, color, style, label in [("conformal_radius", "#305f74", "-", "Conformal"),
                                     ("bayesian_radius", colors[1], "-", "Bayesian"),
                                     ("radius", "#242f40", ":", "Combined maximum")]:
        axes.flat[11].plot(rx, [r["radii"][key][0] for r in refreshes], color=color, ls=style, lw=1.2, label=label)
    axes.flat[11].set(title="Both BCA radius components", xlabel="Host updates (millions)", ylabel="Dimensionless radius; host N/A")
    axes.flat[11].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.14)
    fig.supxlabel("Metrics: 1,000-update means; actor fields exclude skipped placeholders and retain active zeros.\n"
                  "1M accepted scale fits; zero abstentions; 198 refreshes. Episode spread is not training-seed uncertainty.\n"
                  "No causal or OOD-ranking claim. Unit-warmup numerical differences remain unresolved; no replay.", fontsize=10)
    fig.savefig(bca / "paired_overview.png", dpi=160)
    fig.savefig(bca / "paired_overview.svg")
    plt.close(fig)
    svg = bca / "paired_overview.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", type=Path, required=True)
    p.add_argument("--bca", type=Path, required=True)
    a = p.parse_args()
    render(a.host, a.bca)
