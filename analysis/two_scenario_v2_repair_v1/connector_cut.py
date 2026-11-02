"""Fixed projection-slab analysis of saved v2 transitions; no simulator calls.

This is an analysis proposal, not the mask currently used by v2 training.
It does not write a filtered dataset or change any reward, state, time or done.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "experiments/thrusterchain_bca/two_scenario_v2"
MANIFEST_SHA256 = "c4e3198289e9ea8b329bbaedbf8cdc5e1c7cea19249beb121ffaa100af53336d"
LOWER, UPPER = 0.2, 0.8
ROLE_NAMES = {0: "fitting", 1: "calibration", 2: "reserved"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticated_data_module():
    path = V2 / "MANIFEST.json"
    if _sha(path) != MANIFEST_SHA256:
        raise ValueError("v2 manifest differs from the inspected source")
    pins = json.loads(path.read_text(encoding="utf-8"))["files"]
    for filename in ("data.py", "task.py"):
        source = V2 / filename
        if _sha(source) != pins.get(source.relative_to(REPO).as_posix()):
            raise ValueError("v2 source does not match its seal: " + filename)
    module = importlib.import_module("experiments.thrusterchain_bca.two_scenario_v2.data")
    if Path(module.__file__).resolve() != (V2 / "data.py").resolve() or Path(module.T.__file__).resolve() != (V2 / "task.py").resolve():
        raise ValueError("v2 data/task import resolved outside the authenticated source")
    return module


def _axis(geometry):
    b, c = np.asarray(geometry["B"], np.float64), np.asarray(geometry["C"], np.float64)
    if b.shape != (13,) or c.shape != (13,) or not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("B/C must be finite complete states")
    delta = c[:10] - b[:10]
    length2 = float(delta @ delta)
    if not np.isfinite(length2) or length2 <= 0:
        raise ValueError("B and C must be physically distinct")
    return b[:10], delta, length2


def progress(states, geometry):
    x = np.asarray(states, np.float64)
    if x.ndim == 0 or x.shape[-1] != 13 or not np.isfinite(x).all():
        raise ValueError("Progress requires finite full 13-state rows")
    b, delta, length2 = _axis(geometry)
    return np.sum((x[..., :10] - b) * delta, axis=-1) / length2


def retained_mask(arrays, geometry):
    qs, qn = progress(arrays["states"], geometry), progress(arrays["next_states"], geometry)
    if qs.shape != qn.shape:
        raise ValueError("Source/next rows differ")
    # Closed interval intersection includes both endpoints and direct jumps,
    # in either direction. No time or observed outcome enters this rule.
    return ~((np.minimum(qs, qn) <= UPPER) & (np.maximum(qs, qn) >= LOWER))


def _projection_side(q):
    return np.where(q < LOWER, 0, np.where(q > UPPER, 2, 1))


def _hash_array(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def analyze(arrays, geometry, protocol):
    D = authenticated_data_module()
    schema = D.validate_arrays(arrays, require_all_parents=False)
    expected_roles = [int(protocol[key]) for key in ("fit_parent_episodes", "calibration_parent_episodes", "reserved_parent_episodes")]
    if schema["parents"] != int(protocol["parent_episodes"]) or schema["role_parent_counts"] != expected_roles:
        raise ValueError("Saved parent roles do not match the supplied protocol")
    if not np.array_equal(np.unique(arrays["episode_ids"]), np.arange(protocol["parent_episodes"])):
        raise ValueError("Original parent identities are incomplete")
    rules = protocol["setup_rules"]
    if (int(rules["minimum_approach_parents"]), int(rules["minimum_fragment_parents"]),
        int(rules["calibration_eligible_parent_minimum"]), int(rules["minimum_removed_parents"])) != (8, 8, 128, 8):
        raise ValueError("This analysis preserves the frozen 8/8/128/8 requirements")
    if float(protocol["geometry_recipe"]["connector_trim_fraction"]) != LOWER:
        raise ValueError("Slab endpoints must inherit the frozen trim")
    mask = retained_mask(arrays, geometry)
    qs, qn = progress(arrays["states"], geometry), progress(arrays["next_states"], geometry)
    retained_indices = np.flatnonzero(mask)
    fragment_ids = D.fragment_ids_for_mask(arrays, mask)
    row_fragment = np.full(len(mask), -1, np.int64)
    row_fragment[retained_indices] = fragment_ids
    source_goal = D.T.contains(arrays["states"], geometry["goal"])
    next_goal = D.T.contains(arrays["next_states"], geometry["goal"])
    danger_edge = D.T.edge_intersects(arrays["states"], arrays["next_states"], geometry["danger"])
    success_edge = arrays["terminals"] & next_goal & ~danger_edge
    summaries, starts = [], []
    for parent in np.unique(arrays["episode_ids"]):
        idx = np.flatnonzero(arrays["episode_ids"] == parent)
        keep = idx[mask[idx]]
        removed = idx[~mask[idx]]
        fids = np.unique(row_fragment[keep])
        starts.append(float(qs[idx[0]]))
        success = bool(success_edge[idx[-1]])
        first_fragment = keep[row_fragment[keep] == row_fragment[idx[0]]] if mask[idx[0]] else np.array([], np.int64)
        last_fragment = keep[row_fragment[keep] == row_fragment[idx[-1]]] if mask[idx[-1]] else np.array([], np.int64)
        upstream = bool(len(first_fragment) and len(removed) and np.all(np.maximum(qs[first_fragment], qn[first_fragment]) < LOWER))
        downstream = bool(success and len(last_fragment) and len(removed) and np.all(np.minimum(qs[last_fragment], qn[last_fragment]) > UPPER))
        both = bool(upstream and downstream and row_fragment[idx[0]] != row_fragment[idx[-1]])
        summaries.append({"parent_id": int(parent), "role": ROLE_NAMES[int(arrays["roles"][idx[0]])],
                          "rows": int(len(idx)), "removed_rows": int(len(removed)), "retained_rows": int(len(keep)),
                          "retained_fragments": int(len(fids)), "raw_goal_success": success,
                          "intact_retained_success": bool(success and np.all(mask[idx])),
                          "upstream_start_fragment": upstream, "downstream_goal_fragment": downstream,
                          "genuine_upstream_and_downstream_goal_fragments": both,
                          "upstream_fragment_row_ids": arrays["row_ids"][first_fragment].astype(int).tolist() if upstream else [],
                          "downstream_goal_fragment_row_ids": arrays["row_ids"][last_fragment].astype(int).tolist() if downstream else [],
                          "original_first_time": int(arrays["times"][idx[0]]), "original_final_time": int(arrays["times"][idx[-1]])})
    role_counts = {}
    for role, name in ROLE_NAMES.items():
        rows = arrays["roles"] == role
        parents = [p for p in summaries if p["role"] == name]
        selected = rows & mask
        role_counts[name] = {
            "parents": len(parents), "rows": int(rows.sum()), "retained_rows": int(selected.sum()),
            "removed_rows": int((rows & ~mask).sum()),
            "retained_parents": int(len(np.unique(arrays["episode_ids"][selected]))),
            "removed_parents": int(len(np.unique(arrays["episode_ids"][rows & ~mask]))),
            "retained_fragments": sum(p["retained_fragments"] for p in parents),
            "raw_goal_success_parents": sum(p["raw_goal_success"] for p in parents),
            "intact_retained_success_parents": sum(p["intact_retained_success"] for p in parents),
            "upstream_fragment_parents": sum(p["upstream_start_fragment"] for p in parents),
            "downstream_goal_fragment_parents": sum(p["downstream_goal_fragment"] for p in parents),
            "genuine_fragment_pair_parents": sum(p["genuine_upstream_and_downstream_goal_fragments"] for p in parents),
            "raw_source_goal_rows": int((rows & source_goal).sum()),
            "raw_next_goal_rows": int((rows & next_goal).sum()),
            "retained_source_goal_rows": int((selected & source_goal).sum()),
            "retained_next_goal_rows": int((selected & next_goal).sum()),
            "retained_goal_terminal_rows": int((selected & success_edge).sum()),
            "retained_other_terminal_rows": int((selected & arrays["terminals"] & ~success_edge).sum()),
        }
    support = {}
    fit = mask & (arrays["roles"] == D.FIT)
    for name, t in (("B", 25), ("C", 30)):
        diff = arrays["states"] - np.asarray(geometry[name], np.float64)
        near = fit & (arrays["times"] == t)
        near &= np.linalg.norm(diff[:, :10], axis=1) <= float(rules["approach_physical_l2_radius"])
        near &= np.abs(diff[:, 10]) <= float(rules["approach_current_radius"])
        near &= np.linalg.norm(diff[:, 11:], axis=1) <= float(rules["approach_wave_l2_radius"])
        parents = np.unique(arrays["episode_ids"][near])
        support[name] = {"time": t, "retained_fitting_rows": int(near.sum()),
                         "distinct_parents": int(len(parents)), "parent_ids": parents.astype(int).tolist(),
                         "passed": bool(len(parents) >= 8)}
    b, delta, length2 = _axis(geometry)
    length = float(np.sqrt(length2))
    goal_center = np.asarray(geometry["goal"]["center"], np.float64)
    radius = float(geometry["goal"]["radius"])
    if goal_center.shape != (10,) or not np.isfinite(goal_center).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid original goal ball")
    goal_center_q = float((goal_center - b) @ delta / length2)
    goal_lower = goal_center_q - radius / length
    start_center_q = float(progress(np.asarray(geometry["A"]), geometry))
    jitter = float(protocol["collection"]["physical_start_jitter"])
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("Invalid original start-law box")
    start_law_upper = start_center_q + jitter * float(np.sum(np.abs(delta))) / length2
    separation = {"observed_initial_q_min": min(starts), "observed_initial_q_max": max(starts),
                  "all_observed_starts_before_slab": bool(max(starts) < LOWER),
                  "initial_physical_box_q_upper": start_law_upper,
                  "entire_original_start_box_before_slab": bool(start_law_upper < LOWER),
                  "goal_center_q": goal_center_q, "entire_goal_ball_q_lower": goal_lower,
                  "entire_goal_ball_after_slab": bool(goal_lower > UPPER),
                  "proof_scope": "Every path from the specified start box to the goal ball must cross the removed slab under the discrete chord-edge convention; no unique route is asserted."}
    sides_s, sides_n = _projection_side(qs), _projection_side(qn)
    cut_matrix = [[int(((~mask) & (sides_s == i) & (sides_n == j)).sum()) for j in range(3)] for i in range(3)]
    checks = {
        "all_observed_starts_before_slab": separation["all_observed_starts_before_slab"],
        "entire_original_start_box_before_slab": separation["entire_original_start_box_before_slab"],
        "entire_goal_ball_after_slab": separation["entire_goal_ball_after_slab"],
        "zero_retained_forbidden_edges": bool(not np.any(~retained_mask({"states": arrays["states"][mask], "next_states": arrays["next_states"][mask]}, geometry))),
        "zero_intact_successful_fitting_trajectories": role_counts["fitting"]["intact_retained_success_parents"] == 0,
        "eight_genuine_fitting_fragment_pairs": role_counts["fitting"]["genuine_fragment_pair_parents"] >= 8,
        "eight_fitting_parents_with_removed_labels": role_counts["fitting"]["removed_parents"] >= 8,
        "B_retained_full_state_time_support": support["B"]["passed"],
        "C_retained_full_state_time_support": support["C"]["passed"],
        "128_eligible_calibration_parents": role_counts["calibration"]["retained_parents"] >= 128,
    }
    return {
        "schema": "two_scenario_v2.fixed_projection_cut_analysis.v1",
        "mask_definition": {"projection": "dot(x[:10]-B[:10],C[:10]-B[:10])/squared_norm(C[:10]-B[:10])", "closed_band": [LOWER, UPPER],
                            "row_order": "original saved archive row order", "retained_mask_encoding": "uint8,1=retained,0=removed", "edge_rule": "closed projection interval intersection in either direction"},
        "counts": {"rows": len(mask), "retained_rows": int(mask.sum()), "removed_rows": int((~mask).sum()),
                   "retained_fragments": int(len(np.unique(fragment_ids)))},
        "by_role": role_counts, "by_parent": summaries, "support": support, "separation": separation,
        "cut_edge_projection_sides": {"order": ["before", "closed_band", "after"], "removed_source_by_next": cut_matrix},
        "hashes": {"retained_mask_uint8_sha256": _hash_array(mask, np.uint8),
                   "retained_row_ids_int64le_sha256": _hash_array(arrays["row_ids"][mask], "<i8"),
                   "removed_row_ids_int64le_sha256": _hash_array(arrays["row_ids"][~mask], "<i8"),
                   "retained_fragment_ids_int64le_sha256": _hash_array(fragment_ids, "<i8")},
        "checks": checks, "failed_criteria": [key for key, ok in checks.items() if not ok],
        "decision": "saved_data_mask_checks_passed" if all(checks.values()) else "saved_data_mask_checks_failed",
        "preservation": {"raw_arrays_mutated": False, "filtered_dataset_written": False, "new_transitions_or_joins": 0,
                         "original_row_ids_times_states_next_observations_rewards_terminals_preserved": True},
        "execution": {"simulator_steps": 0, "model_calls": 0, "training_updates": 0, "calibrations": 0},
        "training_ready": False,
        "limitations": ["This mask uses the already inspected v2 development data.",
                        "The existing v2 learner still recomputes its capsule mask; this analysis does not change training.",
                        "Separation is a support-construction claim, not evidence of BCA benefit or a unique optimal route.",
                        "Marginal calibration outside the missing slab does not guarantee coverage inside it."],
    }
