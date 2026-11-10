"""How much of BCA's change in behavior does plain sharpening of the baseline reproduce?

No training. For each matched pair, the saved baseline ("native") command at every
cell is multiplied by ONE number per policy and clipped back into [-1,1], with the
number chosen so the policy's average absolute preference equals that of its BCA
partner. The sharpened baseline keeps the baseline's ranking of directions at every
cell; only its decisiveness changes. It is then evaluated exactly, like the others.

Written before the first run (2026-11-09), on policies whose outcomes were already seen,
so this is exploratory:

  Prediction. If BCA's local gains are mostly generic sharpening, the sharpened
  baseline reproduces at least half of BCA's mean change in (a) the probability of
  DOWN at Honeypot cell 24 and (b) the 200-step goal probability from cell 24, in the
  targeted data arm.
  Refuted if it reproduces less than half of either. Then the gain needs BCA's change
  of direction, not just its decisiveness.
  No prediction is made for the Bottleneck B cell; it is reported as measured.

Usage:
    python -I -B sharpening_counterfactual.py --evidence <stress folder> --out <outputs folder>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gridworld_min as gw  # noqa: E402
from measure_action_probabilities import CONDITIONS, hazard_frontier, load_models  # noqa: E402

BETA, HORIZON = 2.0, 200
READOUTS = {"honeypot": {"p_down_at_24": (24, (gw.DOWN,))},
            "bottleneck": {"p_right_at_B": (44, (gw.RIGHT,)), "p_up_hazard_at_B": (44, (gw.UP,)),
                           "p_right_at_gate": (45, (gw.RIGHT,)), "p_up_wall_at_C": (46, (gw.UP,))}}
STARTS = {"honeypot": (24,), "bottleneck": (44, 90)}


def sharpen(commands, cells, target_mean_abs):
    """Scale every command by one factor (then clip) to reach the target average |preference|."""
    def level(factor):
        return float(np.abs(np.clip(factor * commands[cells], -1, 1)).mean())
    low, high = 0.0, 1.0
    while level(high) < target_mean_abs and high < 1e6:
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if level(mid) < target_mean_abs else (low, mid)
    factor = 0.5 * (low + high)
    return np.clip(factor * commands, -1, 1), factor


def readouts(scenario, commands, frontier):
    out = {}
    for label, (cell, directions) in READOUTS[scenario].items():
        out[label] = float(gw.movement_probabilities(commands[cell], BETA)[list(directions)].sum())
    goal, hazard = gw.finite_outcomes(scenario, commands, BETA, HORIZON)
    for start in STARTS[scenario]:
        out[f"goal_200_from_{start}"] = float(goal[start])
        out[f"hazard_200_from_{start}"] = float(hazard[start])
    for cell, direction in frontier.items():
        out[f"hazard_direction_at_{cell}"] = float(gw.movement_probabilities(commands[cell], BETA)[direction])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    models = load_models(args.evidence)
    pairs = defaultdict(dict)
    for model in models.values():
        pairs[model["pair"]][model["arm"]] = model

    per_pair = []
    for pair, arms in sorted(pairs.items()):
        native, bca = arms["native_aux"], arms["bca_direct"]
        scenario = native["scenario"]
        cells = gw.free_cells(scenario)
        frontier = hazard_frontier(scenario)
        target = float(np.abs(bca["commands"][cells]).mean())
        sharpened, factor = sharpen(native["commands"], cells, target)
        per_pair.append({"pair": pair, "scenario": scenario, "condition": native["condition"],
                         "dataset": native["dataset"], "seed": native["seed"], "scale_factor": factor,
                         "native_mean_abs_preference": float(np.abs(native["commands"][cells]).mean()),
                         "bca_mean_abs_preference": target,
                         "native": readouts(scenario, native["commands"], frontier),
                         "sharpened_native": readouts(scenario, sharpened, frontier),
                         "bca": readouts(scenario, bca["commands"], frontier)})

    summary = {}
    for scenario in CONDITIONS:
        frontier = hazard_frontier(scenario)
        toward = [c for c, d in frontier.items() if d in (gw.UP, gw.RIGHT)]
        away = [c for c, d in frontier.items() if d in (gw.DOWN, gw.LEFT)]
        summary[scenario] = {}
        for condition in CONDITIONS[scenario]:
            rows = [r for r in per_pair if r["scenario"] == scenario and r["condition"] == condition]
            block = {"pairs": len(rows), "mean_scale_factor": float(np.mean([r["scale_factor"] for r in rows]))}
            names = list(READOUTS[scenario]) + [f"{k}_200_from_{s}" for s in STARTS[scenario] for k in ("goal", "hazard")]
            for name in names:
                n, s, b = (np.array([r[arm][name] for r in rows]) for arm in ("native", "sharpened_native", "bca"))
                total = float(b.mean() - n.mean())
                block[name] = {"native_mean": float(n.mean()), "sharpened_native_mean": float(s.mean()),
                               "bca_mean": float(b.mean()),
                               "share_of_bca_mean_change_reproduced": (float(s.mean() - n.mean()) / total) if total else None,
                               "pairs_where_sharpening_moves_the_same_way_as_bca":
                                   int(np.sum(np.sign(s - n) == np.sign(b - n)))}
            if scenario == "honeypot":
                for label, group in (("hazard_toward_goal_corner", toward), ("hazard_away_from_goal_corner", away)):
                    n, s, b = (np.array([[r[arm][f"hazard_direction_at_{c}"] for c in group] for r in rows])
                               for arm in ("native", "sharpened_native", "bca"))
                    block[label] = {"native_mean": float(n.mean()), "sharpened_native_mean": float(s.mean()),
                                    "bca_mean": float(b.mean()),
                                    "pair_cells": int(n.size),
                                    "pair_cells_where_sharpening_raises_hazard_probability": int((s > n).sum()),
                                    "pair_cells_where_bca_raises_hazard_probability": int((b > n).sum()),
                                    "largest_rise_under_sharpening": float((s - n).max()),
                                    "largest_rise_under_bca": float((b - n).max())}
            summary[scenario][condition] = block
    (args.out / "sharpening_counterfactual.json").write_text(
        json.dumps({"beta": BETA, "horizon": HORIZON, "summary": summary, "per_pair": per_pair},
                   indent=2, allow_nan=False))
    print("wrote", args.out / "sharpening_counterfactual.json")


if __name__ == "__main__":
    main()
