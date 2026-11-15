"""Exact evaluation of one saved policy. NumPy only; no learner code."""
import numpy as np

import collect
import environment as env

BAD_ENDING_TOLERANCE = 0.01
KAPPA = 3.0
# Method names as run_study.py tags them (Amendment 3). test_learner.py checks these against learner.py.
CONTROL_OF = {"bca": "shuffled_width", "support": "support_shuffled",
              "support_onesided": "support_onesided_shuffled", "full": "full_shuffled",
              "onesided": "onesided_shuffled", "support_only": "support_only_shuffled",
              "support_blind_width": "support_shuffled"}
SUPPORT_AWARE = {"support", "support_shuffled", "support_onesided", "support_onesided_shuffled", "full", "full_shuffled",
                 "support_only", "support_only_shuffled", "support_blind_width", "permit", "permit_plain"}
# Methods whose width is zero at every supported command (Amendment 4). "Relative to the model's
# average width" has no stable meaning for them, so G2 reads their width in absolute units.
NO_CALIBRATED_WIDTH = {"support_only", "support_only_shuffled"}
ONE_SIDED = {"support_onesided", "support_onesided_shuffled", "full", "full_shuffled", "onesided", "onesided_shuffled"}


def one_sided_level(method, reference):
    """The width below which a one-sided method charges nothing. None for a two-sided method."""
    if method not in ONE_SIDED:
        return None
    return float(reference["width_ref_support" if method in SUPPORT_AWARE else "width_ref"])


def unseen_hazard_probability(task, commands, horizon=env.HORIZON):
    """Per start cell: probability the episode ends in the Honeypot's withheld hazard."""
    P, _, _ = task.chain(commands)
    entering = np.zeros(100)
    for cell in task.free_cells():
        for direction, prob in enumerate(task.probabilities(commands[cell])):
            if prob > 0 and task.move(cell, direction)[0] == env.UNSEEN_HAZARD:
                entering[cell] += prob
    reached = np.zeros(100)
    for _ in range(horizon):
        reached = entering + P @ reached
    return reached


def target_corner(direction):
    """Index of the corner command that executes exactly this direction."""
    wanted = -np.ones(4)
    wanted[direction] = 1.0
    return int(np.flatnonzero((env.CORNERS == wanted).all(axis=1))[0])


def sharpen(commands, cells, target_mean_abs):
    """Scale every command by one factor (then clip) to reach a target average |preference|."""
    def level(factor):
        return float(np.abs(np.clip(factor * commands[cells], -1, 1)).mean())
    low, high = 0.0, 1.0
    while level(high) < target_mean_abs and high < 1e6:
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if level(mid) < target_mean_abs else (low, mid)
    return np.clip(0.5 * (low + high) * commands, -1, 1), 0.5 * (low + high)


def policy_record(task, commands, q1=None, q2=None, width=None, level=None, original_width=None,
                  support=None, floor=None):
    """Everything the gates need about one policy. `q1`, `q2`, `width` are [100 cells, 16 corners].

    `width` is the width the method consumes. `level` is its one-sided level (None: every bit of
    width is charged). `original_width`, `support` and `floor` are optional extra read-outs: the
    width before any support-aware raise, the frozen support score, and the frozen support floor."""
    def charge(w):
        return KAPPA * (w if level is None else max(w - level, 0.0))

    free = task.free_cells()
    decision = collect.targeted_moves(task)
    reached = task.outcomes(commands)
    value = task.value(commands)
    honeypot = task.name == "honeypot"
    record = {"mean_abs_preference": float(np.abs(commands[free]).mean()),
              "mean_value_all_starts": float(value[free].mean()),
              "mean_value_decision_cells": float(value[sorted(decision)].mean()),
              "bad_ending_by_start": {str(c): float(reached["hazard"][c] + reached["timeout"][c]) for c in free},
              "goal_by_start": {str(c): float(reached["goal"][c]) for c in free},
              "cells": {}}
    starts = sorted(set(decision) | {env.A_START} | (set() if honeypot else {env.C}))
    into_unseen = unseen_hazard_probability(task, commands) if honeypot else None
    for cell in starts:
        entry = {"goal": float(reached["goal"][cell]), "hazard": float(reached["hazard"][cell]),
                 "low_exit": float(reached["low_exit"][cell]), "timeout": float(reached["timeout"][cell]),
                 "value": float(value[cell])}
        if honeypot:
            entry["unseen_hazard"] = float(into_unseen[cell])
        record["cells"][str(cell)] = entry
    per_direction = task.direction_values(value)
    for cell, (direction,) in decision.items():
        probs = task.probabilities(commands[cell])
        entry = record["cells"][str(cell)]
        entry["command"] = [float(x) for x in commands[cell]]
        entry["executed"] = dict(zip(env.DIRECTION_NAMES, (float(p) for p in probs)))
        entry["p_target_move"] = float(probs[direction])
        entry["raw_preference_for_target"] = float(commands[cell][direction])
        if q1 is not None:
            corner = target_corner(direction)
            # True value of each corner command: issue it once here, then follow this policy.
            true_values = task.probabilities(env.CORNERS) @ per_direction[cell]
            best = int(np.argmax(true_values))
            model_mean_width = max(float(width[free].mean()), 1e-12)    # a width that is zero everywhere reads as 0
            entry.update(
                width_at_target_command=float(width[cell, corner]),
                width_at_best_command=float(width[cell, best]),
                relative_width_at_target_command=float(width[cell, corner] / model_mean_width),
                relative_width_at_best_command=float(width[cell, best] / model_mean_width),
                q1_at_target_command=float(q1[cell, corner]), true_value_of_target_command=float(true_values[corner]),
                q1_at_best_command=float(q1[cell, best]), true_value_of_best_command=float(true_values[best]),
                q1_error_at_target_command=float(q1[cell, corner] - true_values[corner]),
                critic_gap_best_minus_target=float(q1[cell, best] - q1[cell, corner]),
                true_gap_best_minus_target=float(true_values[best] - true_values[corner]),
                penalty_at_target_command=float(charge(width[cell, corner])),
                penalty_at_best_command=float(charge(width[cell, best])),
                penalized_gap_best_minus_target=float((q1[cell, best] - charge(width[cell, best]))
                                                      - (q1[cell, corner] - charge(width[cell, corner]))))
            if original_width is not None:
                entry["relative_original_width_at_target_command"] = float(
                    original_width[cell, corner] / original_width[free].mean())
            if support is not None:
                entry["decades_below_support_floor_at_target_command"] = float(
                    max(floor - support[cell, corner], 0.0) / np.log(10.0))
                entry["decades_below_support_floor_at_best_command"] = float(
                    max(floor - support[cell, best], 0.0) / np.log(10.0))
    cells = [record["cells"][str(c)] for c in decision]
    record["probability_readout"] = float(np.mean([c["p_target_move"] for c in cells]))
    if honeypot:
        record["primary_outcome"] = float(np.mean([c["unseen_hazard"] for c in cells]))      # H
    else:
        record["primary_outcome"] = record["cells"][str(env.B)]["goal"]                        # X
    if q1 is not None:
        record["width_readout"] = float(np.mean([c["relative_width_at_target_command"] for c in cells]))
        record["model_mean_width"] = float(width[free].mean())
        for name, source in (("width_at_target", "width_at_target_command"),
                             ("penalty_at_target", "penalty_at_target_command"),
                             ("penalty_at_best", "penalty_at_best_command"),
                             ("critic_error_at_target", "q1_error_at_target_command"),
                             ("penalized_gap", "penalized_gap_best_minus_target"),
                             ("original_width_readout", "relative_original_width_at_target_command"),
                             ("decades_below_floor_at_target", "decades_below_support_floor_at_target_command"),
                             ("decades_below_floor_at_best", "decades_below_support_floor_at_best_command")):
            if source in cells[0]:
                record[name] = float(np.mean([c[source] for c in cells]))
    return record
