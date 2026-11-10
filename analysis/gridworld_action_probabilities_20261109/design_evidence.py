"""Evidence used to judge the Honeypot and Bottleneck designs. Read-only, no training.

It answers five setup questions from saved artifacts:
  1. How much room for improvement did each decision cell have (oracle headroom)?
  2. How large is the true safe-vs-dangerous value gap compared with the critic's error?
  3. Where is the width small: at the demonstrated (oracle) command, or elsewhere?
  4. What data actually surround the bridge in each Bottleneck dataset arm?
  5. How much do these probabilities move between trainings that differ only trivially?

Usage:
    python -I -B design_evidence.py --evidence <stress folder> --measured <outputs folder> --out <outputs folder>
"""
import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gridworld_min as gw  # noqa: E402

BETA = 2.0
SEEDS = (1234, 2234, 3234)
SAFE_24 = (1.0, 1.0, -1.0, -1.0)     # up + right
RISKY_24 = (-1.0, -1.0, 1.0, -1.0)   # down, into hazard 34
RIGHT_ONLY = (-1.0, 1.0, -1.0, -1.0)
UP_ONLY = (1.0, -1.0, -1.0, -1.0)
WIDTH_PROBES = {
    "honeypot": {(24, RISKY_24): "state24_dangerous_command", (24, SAFE_24): "state24_safe_command"},
    "bottleneck": {(44, RIGHT_ONLY): "B_enter_bridge_command", (44, UP_ONLY): "B_hazard_command",
                   (45, RIGHT_ONLY): "gate_exit_command", (46, RIGHT_ONLY): "C_onward_command"},
}


def mean(values):
    return float(np.mean(values)) if len(values) else None


def oracle_headroom():
    out = {}
    for scenario, starts in (("honeypot", (24, 31, 33, 90)), ("bottleneck", (44, 45, 46, 90))):
        commands, value = gw.oracle(scenario, BETA)
        goal, hazard = gw.finite_outcomes(scenario, commands, BETA, 200)
        per_direction = gw.direction_values(scenario, value)
        cells = {}
        for cell in gw.free_cells(scenario):
            probs = gw.movement_probabilities(commands[cell], BETA)
            hazard_dirs = [d for d in range(4) if gw.direction_role(scenario, cell, d) == "hazard"]
            if hazard_dirs or cell in starts:
                cells[str(cell)] = {
                    "oracle_command": commands[cell].tolist(),
                    "oracle_movement_probabilities": probs.tolist(),
                    "oracle_step_hazard_probability": float(sum(probs[d] for d in hazard_dirs)),
                    "value_of_each_direction": per_direction[cell].tolist(),
                    "oracle_goal_200": float(goal[cell]), "oracle_hazard_200": float(hazard[cell]),
                    "oracle_value": float(value[cell])}
        out[scenario] = {"cells": cells}
    commands, value = gw.oracle("honeypot", BETA)
    out["honeypot"]["state_24_true_gap_under_oracle_continuation"] = {
        "safe_command_value": gw.command_value("honeypot", 24, SAFE_24, value, BETA),
        "risky_command_value": gw.command_value("honeypot", 24, RISKY_24, value, BETA)}
    return out


def candidate_bottleneck_layouts():
    """Every row the single opening could sit on, judged by geometry and the oracle only.

    A row is usable if B (left of the opening) and C (right of it) are ordinary cells.
    For each usable row: is B, the gate or C next to a hazard, what the best command is
    at B and at B's neighbors (the cells whose data a learner must generalize from when
    B itself has no rows), and how well the best possible policy does.
    """
    out = {}
    for row in range(gw.SIZE):
        scenario = "bottleneck" if row == 4 else f"bottleneck_row{row}"
        b, gate, c = 10 * row + 4, 10 * row + 5, 10 * row + 6
        free = set(gw.free_cells(scenario))
        if b not in free or c not in free:
            out[str(row)] = {"usable": False, "reason": "B or C is a wall, hazard or goal cell"}
            continue
        commands, value = gw.oracle(scenario, BETA)
        goal, hazard = gw.finite_outcomes(scenario, commands, BETA, 200)

        def roles(cell):
            return {gw.DIRECTION_NAMES[d]: gw.direction_role(scenario, cell, d) for d in range(4)}

        def preferred(cell):
            return [gw.DIRECTION_NAMES[d] for d in range(4) if commands[cell][d] > 0]

        neighbors = {}
        for d in range(4):
            nxt, _, done = gw.move(scenario, b, d)
            if nxt != b and not done and nxt != gate:
                neighbors[str(nxt)] = preferred(nxt)
        out[str(row)] = {
            "usable": True, "B": b, "gate": gate, "C": c,
            "hazard_next_to": [name for name, cell in (("B", b), ("gate", gate), ("C", c))
                               if "hazard" in roles(cell).values()],
            "B_directions": roles(b), "C_directions": roles(c),
            "oracle_preferred_at_B": preferred(b), "oracle_preferred_at_C": preferred(c),
            "oracle_preferred_at_supported_neighbors_of_B": neighbors,
            "neighbors_that_point_into_a_hazard_if_copied_at_B":
                [cell for cell, dirs in neighbors.items()
                 if any(roles(b)[name] == "hazard" for name in dirs)],
            "oracle_value_at_B": float(value[b]),
            "oracle_goal_200_from_B": float(goal[b]), "oracle_hazard_200_from_B": float(hazard[b]),
            "oracle_goal_200_from_A90": float(goal[90]), "oracle_hazard_200_from_A90": float(hazard[90]),
            "oracle_value_at_A90": float(value[90])}
    return out


def bank_analysis(evidence, models):
    """One pass over the corrected fixed-action bank at the original execution setting."""
    oracle_corner = {s: gw.oracle(s, BETA)[0] for s in ("honeypot", "bottleneck")}
    frontier = {s: {c: [d for d in range(4) if gw.direction_role(s, c, d) == "hazard"]
                    for c in gw.free_cells(s)} for s in oracle_corner}
    groups = defaultdict(list)   # (model, state) -> corner rows
    own = {}                     # (model, state) -> row of the model's own command
    labels = set()
    with gzip.open(evidence / "evaluation_v1/fixed_actions_common.jsonl.gz", "rt") as handle:
        for line in handle:
            if '"temperature":2.0' not in line:
                continue
            row = json.loads(line)
            labels.add(row["action_label"])
            key = (row["model"], row["state"])
            if row["action_label"] == "box_corner":
                groups[key].append(row)
            elif row["action_label"].startswith(models[row["model"]]["arm"].split("_")[0]):
                own[key] = row

    def find(rows, command):
        hits = [r for r in rows if tuple(r["action"]) == tuple(command)]
        assert len(hits) == 1, (command, len(hits))
        return hits[0]

    def describe(row):
        return {"q1": row["q1"], "width": row["width"], "q1_minus_3w": row["q1_minus_3W_probe"],
                "true_q_own_policy": row["true_Q_own_online_actor_infinite"],
                "true_q_native_policy": row["true_Q_common_native_online_actor_infinite"],
                "q1_error": row["raw_q1_signed_error"]}

    margins = defaultdict(list)
    width_by_corner = defaultdict(list)
    probes = defaultdict(dict)             # model -> probe label -> (width, q1 error)
    model_widths = defaultdict(list)       # model -> every corner width at every free cell
    for (model_id, state), rows in groups.items():
        meta = models[model_id]
        scenario = meta["scenario"]
        assert len(rows) == 16
        bucket = (scenario, meta["condition"], meta["arm"])
        model_widths[model_id].extend(r["width"] for r in rows)
        for (cell, command), label in WIDTH_PROBES[scenario].items():
            if cell == state:
                hit = find(rows, command)
                probes[model_id][label] = (hit["width"], hit["raw_q1_signed_error"])
        if (scenario, state) == ("honeypot", 24):
            a, b = describe(find(rows, SAFE_24)), describe(find(rows, RISKY_24))
            margins[bucket + ("state24_safe_minus_risky",)].append({"first": a, "second": b})
        if (scenario, state) == ("bottleneck", 44):
            right, up = describe(find(rows, RIGHT_ONLY)), describe(find(rows, UP_ONLY))
            margins[bucket + ("B_right_minus_up",)].append({"first": right, "second": up})
        widths = np.array([r["width"] for r in rows])
        q1 = np.array([r["q1"] for r in rows])
        score = np.array([r["q1_minus_3W_probe"] for r in rows])
        truth = np.array([r["true_Q_own_online_actor_infinite"] for r in rows])
        best = int(np.argmax(truth))  # best corner for this model's own continuation
        oracle_index = next(i for i, r in enumerate(rows)
                            if tuple(r["action"]) == tuple(oracle_corner[scenario][state]))
        record = {"oracle_width_rank": int((widths < widths[oracle_index]).sum()) + 1,
                  "oracle_width": float(widths[oracle_index]),
                  "mean_corner_width": float(widths.mean()),
                  "q1_picks_best_corner": bool(int(np.argmax(q1)) == best),
                  "penalized_picks_best_corner": bool(int(np.argmax(score)) == best),
                  "regret_q1": float(truth[best] - truth[int(np.argmax(q1))]),
                  "regret_penalized": float(truth[best] - truth[int(np.argmax(score))]),
                  "frontier": bool(frontier[scenario][state])}
        if frontier[scenario][state]:
            d = frontier[scenario][state][0]
            into = [i for i, r in enumerate(rows) if r["action"][d] == 1.0]
            away = [i for i, r in enumerate(rows) if r["action"][d] == -1.0]
            record["width_hazard_preferred_minus_avoided"] = float(widths[into].mean() - widths[away].mean())
            # Control: the same contrast for the three directions that do NOT enter a hazard.
            others = []
            for other in range(4):
                if other != d:
                    plus = [i for i, r in enumerate(rows) if r["action"][other] == 1.0]
                    minus = [i for i, r in enumerate(rows) if r["action"][other] == -1.0]
                    others.append(float(widths[plus].mean() - widths[minus].mean()))
            record["width_other_direction_preferred_minus_avoided"] = float(np.mean(others))
        if (model_id, state) in own:
            record["own_command_width"] = own[(model_id, state)]["width"]
        width_by_corner[bucket].append(record)

    margin_summary = {}
    for key, items in margins.items():
        fields = ("q1", "width", "q1_minus_3w", "true_q_own_policy", "q1_error")
        entry = {"n": len(items)}
        for side in ("first", "second"):
            entry[side] = {f: mean([i[side][f] for i in items]) for f in fields}
        for f in ("true_q_own_policy", "q1", "q1_minus_3w"):
            gaps = [i["first"][f] - i["second"][f] for i in items]
            entry[f"gap_{f}"] = {"mean": mean(gaps), "min": float(min(gaps)), "n_positive": int(sum(g > 0 for g in gaps))}
        margin_summary["|".join(key)] = entry

    width_summary = {}
    for key, items in width_by_corner.items():
        hazard = [i for i in items if i["frontier"]]
        width_summary["|".join(key)] = {
            "cells": len(items),
            "share_where_oracle_corner_has_smallest_width": mean([i["oracle_width_rank"] == 1 for i in items]),
            "share_where_oracle_corner_is_in_smallest_four": mean([i["oracle_width_rank"] <= 4 for i in items]),
            "mean_rank_of_oracle_corner_width_1_to_16": mean([i["oracle_width_rank"] for i in items]),
            "mean_width_at_oracle_corner": mean([i["oracle_width"] for i in items]),
            "mean_width_over_all_corners": mean([i["mean_corner_width"] for i in items]),
            "mean_width_at_own_command": mean([i["own_command_width"] for i in items if "own_command_width" in i]),
            "share_q1_picks_best_corner": mean([i["q1_picks_best_corner"] for i in items]),
            "share_q1_minus_3w_picks_best_corner": mean([i["penalized_picks_best_corner"] for i in items]),
            "mean_regret_q1": mean([i["regret_q1"] for i in items]),
            "mean_regret_q1_minus_3w": mean([i["regret_penalized"] for i in items]),
            "hazard_cells": len(hazard),
            "hazard_cells_mean_width_hazard_preferred_minus_avoided":
                mean([i["width_hazard_preferred_minus_avoided"] for i in hazard]),
            "hazard_cells_share_where_preferring_hazard_has_smaller_width":
                mean([i["width_hazard_preferred_minus_avoided"] < 0 for i in hazard]),
            "hazard_cells_mean_width_other_direction_preferred_minus_avoided":
                mean([i["width_other_direction_preferred_minus_avoided"] for i in hazard]),
            "hazard_cells_mean_regret_q1": mean([i["regret_q1"] for i in hazard]),
            "hazard_cells_mean_regret_q1_minus_3w": mean([i["regret_penalized"] for i in hazard])}
    # Does the width grow where the data were removed? Widths carry a per-model unit,
    # so each is also divided by that model's average width over all cells and corners.
    response = defaultdict(lambda: defaultdict(list))
    for model_id, found in probes.items():
        meta = models[model_id]
        typical = float(np.mean(model_widths[model_id]))
        for label, (width, error) in found.items():
            response["|".join((meta["scenario"], label, meta["arm"]))][meta["condition"]].append(
                (width, width / typical, error))
    width_response = {key: {condition: {"mean_width": mean([v[0] for v in values]),
                                        "mean_width_relative_to_model_average": mean([v[1] for v in values]),
                                        "mean_q1_error": mean([v[2] for v in values]),
                                        "share_where_abs_error_exceeds_width": mean([abs(v[2]) > v[0] for v in values]),
                                        "share_where_abs_error_exceeds_3_widths": mean([abs(v[2]) > 3 * v[0] for v in values])}
                            for condition, values in sorted(by_condition.items())}
                      for key, by_condition in sorted(response.items())}
    return {"action_labels_seen": sorted(labels), "value_gaps": margin_summary,
            "width_by_corner": width_summary, "width_at_deleted_region": width_response}


def bottleneck_data_support(evidence):
    root = evidence / "archive/td3_gridworld/qualification_v4/fragmented_bottleneck/prepared"
    teacher, _ = gw.oracle("bottleneck", BETA)
    out = {"oracle_commands": {str(c): teacher[c].tolist() for c in (43, 54, 44, 45, 46, 47, 56)}}
    agreement = []
    for seed in SEEDS:
        out[str(seed)] = {}
        for arm in ("full", "fragments", "random_rows"):
            data = np.load(root / f"{arm}_{seed}.npz", allow_pickle=False)
            s, s2, a, uniform = data["s"], data["s2"], data["a"], data["uniform_command"]
            x = s % 10
            entry = {"rows": int(len(s)),
                     "rows_left_of_wall": int((x < 5).sum()), "rows_right_of_wall": int((x > 5).sum()),
                     "rows_at_source": {str(c): int((s == c).sum()) for c in (43, 54, 44, 45, 46, 47, 56)},
                     "rows_from_B_into_hazard_34": int(((s == 44) & (s2 == 34)).sum()),
                     "rows_from_B_into_gate": int(((s == 44) & (s2 == 45)).sum()),
                     "rows_from_gate_into_C": int(((s == 45) & (s2 == 46)).sum()),
                     "rows_arriving_at_B": int((s2 == 44).sum()), "rows_arriving_at_gate": int((s2 == 45).sum()),
                     "rows_into_hazard_34_from_other_cells": int(((s2 == 34) & (s != 44)).sum()),
                     "mean_recorded_command": {}}
            for cell in (43, 54, 46, 47):
                rows = a[s == cell]
                if len(rows):
                    center = rows.mean(axis=0)
                    entry["mean_recorded_command"][str(cell)] = {
                        "command": center.tolist(),
                        "movement_probabilities": gw.movement_probabilities(center, BETA).tolist()}
            out[str(seed)][arm] = entry
            if arm == "full":  # the stored teacher flag lets us check my oracle against the collector's
                for cell in gw.free_cells("bottleneck"):
                    rows = a[(s == cell) & ~uniform]
                    if len(rows) >= 8:
                        agreement.append(bool(np.all(np.sign(rows.mean(axis=0)) == teacher[cell])))
    out["oracle_check"] = {"cells_compared": len(agreement), "share_matching_collector_teacher": mean(agreement)}
    return out


def noise_yardstick(measured):
    """Honeypot cells other than 24 have identical rows in all three data arms, so a
    native policy's difference between arms there is training noise, not a data effect."""
    table = defaultdict(dict)
    paired = defaultdict(dict)
    with open(measured / "per_model_cells.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scenario"] != "honeypot" or float(row["beta"]) != BETA:
                continue
            direction = next(n for n in gw.DIRECTION_NAMES if row[f"role_{n}"] == "hazard")
            key = (int(row["cell"]), row["dataset"], row["seed"])
            table[(row["arm"],) + key][row["condition"]] = float(row[f"p_{direction}"])
            paired[key + (row["condition"],)][row["arm"]] = float(row[f"p_{direction}"])
    out = {}
    for arm in ("native_aux", "bca_direct"):
        gaps = [abs(v[a] - v[b]) for (m, cell, *_), v in table.items() if m == arm and cell != 24
                for a, b in (("full", "dangerous_actions"), ("full", "random_rows"),
                             ("dangerous_actions", "random_rows"))]
        out[f"{arm}_between_arm_gap_at_untouched_cells"] = {
            "comparisons": len(gaps), "median": float(np.median(gaps)),
            "p90": float(np.percentile(gaps, 90)), "max": float(np.max(gaps))}
    at24 = [abs(v["full"] - v["random_rows"]) for (m, cell, *_), v in table.items()
            if m == "native_aux" and cell == 24]
    out["native_full_vs_random_gap_at_24"] = {"comparisons": len(at24), "median": float(np.median(at24)),
                                              "max": float(np.max(at24))}
    for label, keep in (("untouched_cells", lambda c: c != 24), ("cell_24", lambda c: c == 24)):
        deltas = [abs(v["bca_direct"] - v["native_aux"]) for (cell, *_), v in paired.items() if keep(cell)]
        out[f"bca_minus_native_absolute_shift_{label}"] = {
            "comparisons": len(deltas), "median": float(np.median(deltas)),
            "p90": float(np.percentile(deltas, 90)), "max": float(np.max(deltas))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.evidence / "evaluation_v1/model_manifest.json").read_text())
    models = {m["id"]: m for m in manifest}
    result = {"oracle_headroom": oracle_headroom(),
              "candidate_bottleneck_layouts": candidate_bottleneck_layouts(),
              "bottleneck_data_support": bottleneck_data_support(args.evidence),
              "noise_yardstick_percentage_points_as_fraction": noise_yardstick(args.measured),
              "fixed_action_bank": bank_analysis(args.evidence, models)}
    (args.out / "design_evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("wrote", args.out / "design_evidence.json")


if __name__ == "__main__":
    main()
