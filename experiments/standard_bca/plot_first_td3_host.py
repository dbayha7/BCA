"""Plot the closed first host audit, without loading models or running simulators."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render(folder):
    audit = json.loads((folder / "audit.json").read_text())
    banks = json.loads((folder / "evaluations.json").read_text())
    blocks = json.loads((folder / "metric_blocks.json").read_text())
    assert audit["accepted"] and audit["run_id"] == "td3_bc-hopper-host-s202609171"
    assert len(banks) == 201 and len(blocks) == 1000
    blue, ink, gray = "#147d92", "#182d45", "#526477"
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelcolor": ink, "text.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), layout="constrained")
    fig.suptitle("TD3+BC / Hopper / Plain host — 1M updates\nSeed 202609171 • BCA pair and four other seeds pending", fontsize=18)
    a = axes.flat[0]
    a.plot([b["step"]/1e6 for b in banks[:-1]],
           [np.mean([e["normalized_score"] for e in b["episodes"]]) for b in banks[:-1]],
           color=blue, lw=1.1)
    a.scatter([1], [audit["summary"]["final_mean"]], color="#d8663c", zorder=4,
              label="20-episode final mean")
    a.set(title="Learning curve", xlabel="Host updates (millions)", ylabel="D4RL normalized score")
    a.legend(fontsize=8, frameon=False)
    a = axes.flat[1]
    final = [e["normalized_score"] for e in banks[-1]["episodes"]]
    a.scatter(np.arange(1, 21), final, color=blue, s=25)
    a.axhline(np.mean(final), color="#d8663c", label=f"Mean {np.mean(final):.2f}", lw=1.4)
    a.set(title="All 20 final episodes", xlabel="Declared episode index (1–20)", ylabel="D4RL normalized score")
    a.legend(frameon=False, fontsize=8)
    x = [b["step"]/1e6 for b in blocks]
    for index, field, title, ylabel in [
        (2, "q_mean", "Actor-action Q1", "Mean Q1 (actor-update rows)"),
        (3, "lambda", "Native Q normalization", "Lambda = alpha / mean(|Q1|)"),
        (4, "actor_loss", "Actor objective", "Mean native actor loss"),
        (5, "critic_loss", "Critic objective", "Sum of twin MSEs"),
        (6, "bc_loss", "Behavior cloning term", "Mean action-coordinate squared error")]:
        a = axes.flat[index]
        y = [b[field]["mean"] for b in blocks]
        a.plot(x, y, color=blue, lw=1.1)
        a.set(title=title, xlabel="Host updates (millions)", ylabel=ylabel)
    for index, title, detail in [
        (7, "Scale fitting", "N/A for the plain host\nNo calibrator; zero fitting operations\nAccepted/abstained fits: N/A"),
        (8, "Bayesian / conformal radii", "N/A for the plain host\nNo posterior; zero refreshes\nBoth components remain in the BCA arm")]:
        a = axes.flat[index]
        a.set_title(title)
        a.text(.5, .5, detail, ha="center", va="center", color=gray, fontsize=11,
               transform=a.transAxes, linespacing=1.7)
        a.set_xticks([]); a.set_yticks([])
        for spine in a.spines.values(): spine.set_visible(False)
    for a in list(axes.flat)[:7]:
        a.grid(alpha=.16)
    fig.supxlabel("Metric traces: means over 1,000 host updates; actor metrics use only 500 active rows per block.\nEpisode spread is not training-seed uncertainty. No BCA benefit or OOD-ranking claim.", fontsize=10)
    fig.savefig(folder / "overview.png", dpi=160)
    fig.savefig(folder / "overview.svg")
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    render(p.parse_args().folder)
