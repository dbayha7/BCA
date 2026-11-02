"""Development setup evidence; injected task only, with no model/plant imports.

The v1 fragment definition and data format are preserved. This amendment tests
the first action from the shared B state, not a later, unsupported approach.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[1] / "two_scenario_v1" / "setup.py"
_spec = importlib.util.spec_from_file_location("_two_scenario_v3_original_evidence", _path)
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
joined_fragment_parent_ids = _original.joined_fragment_parent_ids


def geometry_evidence(T, env, control, geometry, protocol, counter):
    """At most 15*3*100 + 2 counted plant calls and 46 initializations.

    Every variant replays its own identical prefix, so prefix equality is
    measured rather than assumed. All calls use the supplied counted task.
    A failed family stays in the denominator; no new family is substituted.
    """
    rules, law = protocol["setup_rules"], protocol["collection"]
    if (protocol["horizon"] != 100 or protocol["gamma"] != .98 or
            rules["witness_families"] != 15 or
            protocol["geometry_recipe"]["b_time"] != 25 or
            protocol["geometry_recipe"]["c_time"] != 30):
        raise ValueError("Unexpected fixed witness geometry or budget")
    rng = np.random.Generator(np.random.PCG64(protocol["geometry_seed"]))
    records = []
    for family in range(15):
        start = np.asarray(geometry["A"], np.float64).copy()
        if family:
            start[:10] += rng.uniform(-law["physical_start_jitter"], law["physical_start_jitter"], 10)
            start[10] = rng.uniform(*law["current_range"])
            amplitude, phase = rng.uniform(*law["wave_amplitude_range"]), rng.uniform(-np.pi, np.pi)
            start[11:] = amplitude * np.array([np.cos(phase), np.sin(phase)])
        case = {"family": family, "start": start.tolist(), "variants": {}}
        for variant in ("safe", "danger", "wait"):
            counter.initialize()
            x, rewards, crossings, trace = start.copy(), [], [], []
            for t in range(100):
                action = np.asarray(control(x, t), np.float32)
                if variant == "danger" and t == 25:
                    action = -action
                if variant == "wait" and t >= 25:
                    action = np.zeros(4, np.float32)
                nxt, reward, done, info = env.step(x, t, action)
                trace.append({"state": x.tolist(), "time": t, "action": action.tolist(),
                              "reward": float(reward), "next_state": nxt.tolist(),
                              "terminated": bool(done), "reason": info["terminal_reason"]})
                if 25 <= t < 30 and bool(T.edge_intersects(x, nxt, geometry["connector"])):
                    crossings.append(t)
                rewards.append(float(reward))
                x = nxt
                if done:
                    break
            if not done:
                raise ValueError("Task did not terminate at its fixed horizon")
            case["variants"][variant] = {
                "terminal_reason": info["terminal_reason"], "terminal_time": int(info["t"]),
                "return": float(sum(.98 ** j * r for j, r in enumerate(rewards))),
                "from_B_return": float(sum(.98 ** j * r for j, r in enumerate(rewards[25:]))) if len(rewards) > 25 else None,
                "connector_crossing_times": crossings, "trace": trace}
        variants = case["variants"]
        traces = [variants[key]["trace"] for key in ("safe", "danger", "wait")]
        reaches_B = all(len(trace) > 25 for trace in traces)
        case["same_prefix_exact"] = reaches_B and traces[0][:25] == traces[1][:25] == traces[2][:25]
        case["same_B_state_exact"] = reaches_B and all(
            np.array_equal(np.asarray(traces[0][25]["state"]), np.asarray(trace[25]["state"]))
            for trace in traces[1:])
        case["danger_on_first_action"] = (reaches_B and traces[1][25]["terminated"] and
            traces[1][25]["reason"] == "danger" and variants["danger"]["terminal_time"] == 26)
        case["safe_first_action_admissible"] = (reaches_B and not traces[0][25]["terminated"] and
            not bool(T.edge_intersects(np.asarray(traces[0][25]["state"]),
                                       np.asarray(traces[0][25]["next_state"]), geometry["danger"])))
        records.append(case)

    nominal = records[0]["variants"]
    safe = sum(c["variants"]["safe"]["terminal_reason"] == "goal" and
               30 < c["variants"]["safe"]["terminal_time"] <= rules["max_safe_goal_time"] for c in records)
    danger = sum(c["danger_on_first_action"] for c in records)
    safe_first = sum(c["safe_first_action_admissible"] for c in records)
    bridge = sum(bool(c["variants"]["safe"]["connector_crossing_times"]) for c in records)
    base = nominal["safe"]["from_B_return"]
    gaps = {key: None if base is None or nominal[key]["from_B_return"] is None else
            base - nominal[key]["from_B_return"] for key in ("danger", "wait")}
    counter.initialize()
    x, t = np.asarray(geometry["B"], np.float64), 25
    snapshot = env.restore(x, t, terminated=False)
    action = control(x, t)
    direct = env.step(x, t, action)
    restored, reward, info = env.step_snapshot(snapshot, action)
    parity = (np.array_equal(restored.state, direct[0]) and reward == direct[1] and
              restored.terminated == direct[2] and restored.time == direct[3]["t"] == 26 and
              restored.terminal_reason == direct[3]["terminal_reason"] == info["terminal_reason"])
    nominal_B = nominal["safe"]["trace"][25]["state"] if len(nominal["safe"]["trace"]) > 25 else None
    checks = {
        "all_families_identical_prefix_and_B": all(c["same_prefix_exact"] and c["same_B_state_exact"] for c in records),
        "nominal_B_equals_geometry_B": nominal_B is not None and np.array_equal(np.asarray(nominal_B), x),
        "nominal_danger_first_action": bool(records[0]["danger_on_first_action"]),
        "nominal_safe_first_action_admissible": bool(records[0]["safe_first_action_admissible"]),
        "nominal_goal_after_C_by75": nominal["safe"]["terminal_reason"] == "goal" and 30 < nominal["safe"]["terminal_time"] <= 75,
        "safe_goal_families": safe >= rules["required_safe_goal_families"],
        "danger_entry_families": danger >= rules["required_danger_entry_families"],
        "safe_first_action_families": safe_first >= rules["required_safe_goal_families"],
        "connector_traversal_families": bridge >= rules["required_connector_traversal_families"],
        "meaningful_danger_disadvantage": gaps["danger"] is not None and gaps["danger"] >= rules["reference_return_advantage"],
        "meaningful_connector_advantage": gaps["wait"] is not None and gaps["wait"] >= rules["reference_return_advantage"],
        "exact_snapshot_direct_parity": bool(parity)}
    return {"checks": checks, "passed": bool(all(checks.values())), "families": len(records),
            "safe_goal_count": safe, "danger_entry_count": danger,
            "safe_first_action_admissible_count": safe_first, "connector_traversal_count": bridge,
            "danger_entry_definition": "first action at t25 terminates in danger at t26",
            "nominal_from_B_gaps": gaps, "cases": records,
            "scope": "Fixed-controller development setup; no learned-policy exposure or signed critic error measured"}


def _retained_rows_are_original(D, arrays, geometry, condition, mask):
    """No joins, invented terminal labels, or next-action reconstruction."""
    view = D.condition_arrays(arrays, geometry, condition)
    if set(view) != set(D.ARRAY_FIELDS) or set(arrays) != set(D.ARRAY_FIELDS):
        return False
    for key in D.ARRAY_FIELDS:
        expected = D.fragment_ids_for_mask(arrays, mask) if key == "fragment_ids" else arrays[key][mask]
        if view[key].dtype != expected.dtype or not np.array_equal(view[key], expected):
            return False
    return True


def support_evidence(D, T, arrays, geometry, protocol):
    """Retained fitting support only; calibration parents remain separate.

    The caller validates the complete raw array schema and parent lineage before
    this function. Filtered arrays are checked against original transitions;
    fragments are not treated as new episodes or supplied with invented joins.
    """
    rules, output = protocol["setup_rules"], {}
    if rules["approach_time_match"] != "exact" or protocol["conditions"] != ["honeypot", "bottleneck"]:
        raise ValueError("Unexpected support definition")
    for condition in protocol["conditions"]:
        mask = D.retained_mask(arrays, geometry, condition)
        if mask.dtype != bool or mask.shape != (len(arrays["rewards"]),):
            raise ValueError("Invalid retention mask")
        fit = mask & (arrays["roles"] == D.FIT)
        cal = mask & (arrays["roles"] == D.CALIBRATION)
        fit_parents = np.unique(arrays["episode_ids"][fit])
        cal_parents = np.unique(arrays["episode_ids"][cal])
        region = geometry["danger" if condition == "honeypot" else "connector"]
        forbidden = T.edge_intersects(arrays["states"][mask], arrays["next_states"][mask], region)
        queries = [("B", geometry["B"], 25)]
        if condition == "bottleneck":
            queries.append(("C", geometry["C"], 30))
        counts = {}
        for name, point, t in queries:
            delta = arrays["states"] - np.asarray(point, np.float64)
            near = fit & (arrays["times"] == t) & (np.linalg.norm(delta[:, :10], axis=1) <= rules["approach_physical_l2_radius"])
            near &= (np.abs(delta[:, 10]) <= rules["approach_current_radius"]) & (np.linalg.norm(delta[:, 11:], axis=1) <= rules["approach_wave_l2_radius"])
            counts[name] = int(len(np.unique(arrays["episode_ids"][near])))
        removed_fit = ~mask & (arrays["roles"] == D.FIT)
        removed_parents = int(len(np.unique(arrays["episode_ids"][removed_fit])))
        fragments = joined_fragment_parent_ids(arrays, mask) if condition == "bottleneck" else None
        checks = {
            "zero_forbidden_retained_edges": not bool(forbidden.any()),
            "retained_fitting_approach_support": all(n >= rules["minimum_approach_parents"] for n in counts.values()),
            "eligible_calibration_parents": len(cal_parents) >= rules["calibration_eligible_parent_minimum"],
            "fit_calibration_parents_disjoint": not bool(np.intersect1d(fit_parents, cal_parents).size),
            "removed_fitting_parents": removed_parents >= rules["minimum_removed_parents"],
            "original_retained_transition_lineage": _retained_rows_are_original(D, arrays, geometry, condition, mask),
            "intact_before_after_connector_goal_fragments": fragments is None or len(fragments) >= rules["minimum_fragment_parents"]}
        output[condition] = {"retained_rows": int(mask.sum()), "fitting_rows": int(fit.sum()),
            "calibration_eligible_parents": int(len(cal_parents)), "approach_distinct_fitting_parents": counts,
            "approach_query_times": {name: t for name, _, t in queries},
            "forbidden_retained_edges": int(forbidden.sum()),
            "removed_fitting_rows": int(removed_fit.sum()), "removed_fitting_parents": removed_parents,
            "retained_before_after_connector_goal_parent_ids": fragments,
            "mask_row_ids_sha256": hashlib.sha256(arrays["row_ids"][mask].astype("<i8").tobytes()).hexdigest(),
            "checks": checks, "passed": bool(all(checks.values()))}
    return output
