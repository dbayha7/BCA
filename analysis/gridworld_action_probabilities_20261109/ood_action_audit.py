"""Are the "missing" actions in the archived Honeypot and Bottleneck data truly unseen?

Read-only, no training. An action counts as out-of-distribution here only if
  (1) the retained data hold no command at that cell that could produce the move, AND
  (2) the retained data never show the move being executed at that cell, AND
  (3) other moves at that same cell are still well supported.

Part A counts (1)-(3) in the archived data arms.
Part B shows why (2) cannot hold under the archived execution rule, and sizes a
       cutoff rule ("a direction given less than c is never executed") that allows it.
Part C re-evaluates the saved policies under that cutoff, as a changed-deployment
       check only: how much of the baseline-to-BCA difference is slips below the cutoff?

Usage:
    python -I -B ood_action_audit.py --evidence <stress folder> --out <outputs folder>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gridworld_min as gw  # noqa: E402
from measure_action_probabilities import CONDITIONS, load_models  # noqa: E402

BETA, HORIZON, CUTOFF = 2.0, 200, 0.05
SEEDS = (1234, 2234, 3234)
HAZARD_34_NEIGHBORS = {24: gw.DOWN, 33: gw.RIGHT, 35: gw.LEFT, 44: gw.UP}


def truncated(command, beta=BETA, cutoff=CUTOFF):
    """Archived softmax, except a direction given less than `cutoff` is never executed."""
    probs = gw.movement_probabilities(command, beta)
    kept = np.where(probs >= cutoff, probs, 0.0)
    return kept / kept.sum(axis=-1, keepdims=True)


def chain(scenario, commands, law):
    n = gw.SIZE * gw.SIZE
    P, reward, goal_in, hazard_in = np.zeros((n, n)), np.zeros(n), np.zeros(n), np.zeros(n)
    for cell in gw.free_cells(scenario):
        for direction, prob in enumerate(law(commands[cell])):
            nxt, r, done = gw.move(scenario, cell, direction)
            reward[cell] += prob * r
            goal_in[cell] += prob * (nxt == gw.GOAL)
            hazard_in[cell] += prob * (nxt in gw.HAZARDS)
            if not done:
                P[cell, nxt] += prob
    return P, reward, goal_in, hazard_in


def outcomes(scenario, commands, law):
    P, _, goal_in, hazard_in = chain(scenario, commands, law)
    goal, hazard = np.zeros(len(goal_in)), np.zeros(len(goal_in))
    for _ in range(HORIZON):
        goal, hazard = goal_in + P @ goal, hazard_in + P @ hazard
    return goal, hazard


def best_policy(scenario, law, gamma=0.95):
    corner_probs = np.array([law(c) for c in gw.CORNERS])
    value, commands = np.zeros(100), np.zeros((100, 4))
    for _ in range(3000):
        previous = value.copy()
        per_direction = gw.direction_values(scenario, previous, gamma)
        for cell in gw.free_cells(scenario):
            scores = corner_probs @ per_direction[cell]
            best = int(np.argmax(scores))
            value[cell], commands[cell] = scores[best], gw.CORNERS[best]
        if np.max(np.abs(value - previous)) < 1e-12:
            break
    return commands


def part_a(evidence):
    archive = evidence / "archive/td3_gridworld"
    floor, _ = gw.probability_bounds(BETA)
    out = {"honeypot_cell_24_step_down": {}, "honeypot_entries_into_hazard_34_complete_data": {},
           "bottleneck_B_step_right": {}}
    for seed in SEEDS:
        arms = {"complete": archive / f"qualification_v3/full_controls/data_honeypot_{seed}.npz",
                "targeted": archive / f"qualification_v3/honeypot_action_support_prepared/dangerous_actions_data{seed}.npz"}
        block = {}
        for arm, path in arms.items():
            data = np.load(path, allow_pickle=False)
            here = data["s"] == 24
            p_down = gw.movement_probabilities(data["a"][here], BETA)[:, gw.DOWN]
            block[arm] = {"rows_at_cell": int(here.sum()),
                          "commands_giving_down_at_least_half": int((p_down >= 0.5).sum()),
                          "largest_down_probability_among_commands": float(p_down.max()),
                          "rows_where_down_was_executed": int((data["s2"][here] == 34).sum()),
                          "expected_executions_given_the_commands": float(p_down.sum()),
                          "fewest_expected_executions_any_commands_could_give": float(here.sum() * floor),
                          "rows_where_another_move_was_executed": int((data["s2"][here] != 34).sum())}
        out["honeypot_cell_24_step_down"][str(seed)] = block
        data = np.load(arms["complete"], allow_pickle=False)
        out["honeypot_entries_into_hazard_34_complete_data"][str(seed)] = {
            str(cell): int(((data["s"] == cell) & (data["s2"] == 34)).sum()) for cell in HAZARD_34_NEIGHBORS}
        block = {}
        for arm in ("full", "fragments", "random_rows"):
            data = np.load(archive / f"qualification_v4/fragmented_bottleneck/prepared/{arm}_{seed}.npz",
                           allow_pickle=False)
            here = data["s"] == 44
            block[arm] = {"rows_at_B": int(here.sum()),
                          "rows_where_right_was_executed": int((data["s2"][here] == 45).sum()),
                          "rows_where_another_move_was_executed": int((data["s2"][here] != 45).sum())}
        out["bottleneck_B_step_right"][str(seed)] = block
    return out


def part_b(evidence):
    archive = evidence / "archive/td3_gridworld"
    floor, ceiling = gw.probability_bounds(BETA)
    rng = np.random.default_rng(20261109)
    uniform = rng.uniform(-1, 1, size=(400000, 4))
    p_uniform = gw.movement_probabilities(uniform, BETA)[:, 0]
    out = {"archived_rule": {"smallest_probability_any_command_gives_a_direction": floor,
                             "largest": ceiling,
                             "fewest_expected_executions_in_256_visits": 256 * floor,
                             "fewest_expected_executions_in_800_visits": 800 * floor},
           "share_of_uniform_commands_giving_one_direction_at_least": {
               str(t): float((p_uniform >= t).mean()) for t in (0.5, 0.25, 0.10, 0.05, 0.02)}}
    # How much probability do the demonstrator's own commands give to directions it does not want?
    # The Bottleneck files flag which rows came from the noisy best command.
    teacher = gw.oracle("bottleneck", BETA)[0]
    unwanted = []
    for seed in SEEDS:
        data = np.load(archive / f"qualification_v4/fragmented_bottleneck/prepared/full_{seed}.npz",
                       allow_pickle=False)
        rows = ~data["uniform_command"]
        probs = gw.movement_probabilities(data["a"][rows], BETA)
        unwanted.append(probs[teacher[data["s"][rows]] < 0])
    unwanted = np.concatenate(unwanted)
    out["probability_demonstrator_commands_give_an_unwanted_direction"] = {
        "values": int(unwanted.size), "median": float(np.median(unwanted)),
        "p99": float(np.percentile(unwanted, 99)), "p99_9": float(np.percentile(unwanted, 99.9)),
        "max": float(unwanted.max()), "share_at_or_above_cutoff": float((unwanted >= CUTOFF).mean())}
    # Smallest probability a WANTED direction gets from a corner command: 1, 2 or 3 wanted directions.
    wanted = []
    for corner in gw.CORNERS:
        if 0 < (corner > 0).sum() < 4:
            wanted.append(float(gw.movement_probabilities(corner, BETA)[corner > 0].min()))
    out["smallest_probability_a_wanted_direction_gets_from_a_corner_command"] = min(wanted)
    out["cutoff"] = CUTOFF
    # Under the cutoff rule: what the best possible policy achieves.
    for scenario, starts in (("honeypot", (24, 33, 44, 90)), ("bottleneck", (44, 90)), ("bottleneck_row5", (54, 90))):
        commands = best_policy(scenario, truncated)
        goal, hazard = outcomes(scenario, commands, truncated)
        archived_goal, archived_hazard = outcomes(scenario, gw.oracle(scenario, BETA)[0],
                                                  lambda c: gw.movement_probabilities(c, BETA))
        out[f"best_possible_{scenario}"] = {
            str(s): {"goal_200_cutoff_rule": float(goal[s]), "hazard_200_cutoff_rule": float(hazard[s]),
                     "goal_200_archived_rule": float(archived_goal[s]),
                     "hazard_200_archived_rule": float(archived_hazard[s])} for s in starts}
    # At the four cells around hazard 34: how many of the archived commands would a
    # "never issue a command that could enter the hazard" demonstrator have to redraw?
    redraw = {}
    for seed in SEEDS:
        data = np.load(archive / f"qualification_v3/full_controls/data_honeypot_{seed}.npz", allow_pickle=False)
        redraw[str(seed)] = {}
        for cell, direction in HAZARD_34_NEIGHBORS.items():
            p = gw.movement_probabilities(data["a"][data["s"] == cell], BETA)[:, direction]
            redraw[str(seed)][str(cell)] = {"commands": int(p.size),
                                            "would_be_redrawn": int((p >= CUTOFF).sum()),
                                            "kept_commands_that_could_still_enter": 0}
    out["honeypot_commands_redrawn_around_hazard_34"] = redraw
    return out


def part_c(evidence):
    """Saved policies, changed execution only. Not a retraining and not a new result about learning."""
    models = load_models(evidence)
    pairs = defaultdict(dict)
    for model in models.values():
        pairs[model["pair"]][model["arm"]] = model
    archived = lambda c: gw.movement_probabilities(c, BETA)  # noqa: E731
    rows = []
    for pair, arms in sorted(pairs.items()):
        scenario = arms["native_aux"]["scenario"]
        start = 24 if scenario == "honeypot" else 44
        record = {"pair": pair, "scenario": scenario, "condition": arms["native_aux"]["condition"], "start": start}
        for arm, model in arms.items():
            for name, law in (("archived", archived), ("cutoff", truncated)):
                goal, hazard = outcomes(scenario, model["commands"], law)
                record[f"{arm}_{name}_goal"] = float(goal[start])
                record[f"{arm}_{name}_hazard"] = float(hazard[start])
        rows.append(record)
    summary = {}
    for scenario in CONDITIONS:
        summary[scenario] = {}
        for condition in CONDITIONS[scenario]:
            sub = [r for r in rows if r["scenario"] == scenario and r["condition"] == condition]
            entry = {"pairs": len(sub), "start": sub[0]["start"]}
            for name in ("archived", "cutoff"):
                n = np.array([r[f"native_aux_{name}_goal"] for r in sub])
                b = np.array([r[f"bca_direct_{name}_goal"] for r in sub])
                entry[name] = {"baseline_goal": float(n.mean()), "bca_goal": float(b.mean()),
                               "gain": float((b - n).mean()), "pairs_bca_higher": int((b > n + 1e-12).sum()),
                               "pairs_bca_lower": int((b < n - 1e-12).sum()),
                               "pairs_equal": int((np.abs(b - n) <= 1e-12).sum()),
                               "baseline_hazard": float(np.mean([r[f"native_aux_{name}_hazard"] for r in sub])),
                               "bca_hazard": float(np.mean([r[f"bca_direct_{name}_hazard"] for r in sub]))}
            summary[scenario][condition] = entry
    return {"summary": summary, "per_pair": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"definition": __doc__.split("Part A")[0].strip(),
              "A_archived_data": part_a(args.evidence),
              "B_execution_rule": part_b(args.evidence),
              "C_saved_policies_under_cutoff_rule": part_c(args.evidence)}
    (args.out / "ood_action_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("wrote", args.out / "ood_action_audit.json")


if __name__ == "__main__":
    main()
