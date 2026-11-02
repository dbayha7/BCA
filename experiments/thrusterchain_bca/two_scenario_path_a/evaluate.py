"""Final evaluation adapter with injected authenticated policies and counted task.

No model, simulator, JAX, fitting, approval, or command-line launch is imported
here. The production worker supplies the sealed task, physical counter, clock
limit and exact checkpoint/normalization-bound callbacks before calling us.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
V2 = REPO / "experiments/thrusterchain_bca/two_scenario_v2"
V2_MANIFEST = "c4e3198289e9ea8b329bbaedbf8cdc5e1c7cea19249beb121ffaa100af53336d"
ANCHOR_TIMES = {"honeypot": 29, "bottleneck": 25}
ANCHOR_RECIPES = {"honeypot": "controller_until25_then_opposite25_through28",
                  "bottleneck": "controller_until25"}
COMMON_ACTIONS = ["controller", "opposite_controller", "zero"]
PHASE_CAPS = {"returns": 153600, "own_qpi": 57600, "common_reference": 21600, "prefix_parity": 4800}


def require(value, message):
    if not value:
        raise ValueError(message)


def _source_modules():
    """Reuse the original sealed helpers; no canonical plant is imported."""
    path = V2 / "MANIFEST.json"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == V2_MANIFEST, "Wrong v2 source manifest")
    pins = json.loads(path.read_text(encoding="utf-8"))["files"]
    result = []
    for filename in ("evaluation.py", "task.py"):
        path = V2 / filename
        require(hashlib.sha256(path.read_bytes()).hexdigest() == pins[path.relative_to(REPO).as_posix()], "Changed v2 helper")
        name = "_path_a_evaluation_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        result.append(module)
    return tuple(result)


def plan(protocol):
    """Validate the frozen matrix and distinguish tight bounds from reserved caps."""
    expected = {"horizon": 100, "gamma": .98, "kappa": 3., "return_episodes_per_policy": 128,
                "anchors_per_seed_condition": 16, "common_action_slots": 3,
                "include_own_actor_probe": True, "common_reference_panel": True,
                "error_tolerance": 1., "numerical_tie_tolerance": 1e-10,
                "condition_anchor_times": ANCHOR_TIMES,
                "condition_anchor_recipes": ANCHOR_RECIPES, "common_actions": COMMON_ACTIONS}
    for key, value in expected.items():
        require(protocol.get(key) == value, "Frozen evaluation setting differs: " + key)
    require(protocol["conditions"] == ["honeypot", "bottleneck"] and protocol["arms"] == ["native", "bca"], "Wrong matrix")
    require(len(protocol["seeds"]) == len(protocol["evaluation_seeds"]) == 3, "Three paired independent seeds required")
    eval_seeds = [int(s[k]) for s in protocol["evaluation_seeds"] for k in ("anchors", "returns")]
    require(len(set(eval_seeds)) == 6, "Anchor and return streams must be separate")
    fitting_seeds = {int(v) for seed in protocol["seeds"] for v in seed.values()}
    fitting_seeds.update((int(protocol["collection"]["collection_seed"]), int(protocol["collection"]["split_seed"]),
                          int(protocol["calibration_seed"])))
    require(set(eval_seeds).isdisjoint(fitting_seeds), "Evaluation streams overlap collection/training/calibration")
    require(protocol["evaluation_transition_cap"] == sum(PHASE_CAPS.values()) == 237600, "Wrong overall evaluation cap")
    for phase, key in (("returns", "evaluation_return_steps"), ("own_qpi", "evaluation_own_qpi_steps"),
                       ("common_reference", "evaluation_common_reference_steps"), ("prefix_parity", "evaluation_prefix_parity_steps")):
        require(protocol[key] == PHASE_CAPS[phase], "Changed reserved phase cap")
    tight = {"returns": 12 * 128 * 100,
             "own_qpi": 3 * 2 * 16 * 4 * sum(100 - t for t in ANCHOR_TIMES.values()),
             "common_reference": 3 * 16 * 3 * sum(100 - t for t in ANCHOR_TIMES.values()),
             "prefix_parity": 3 * 16 * sum(t + 1 for t in ANCHOR_TIMES.values())}
    require(all(tight[k] <= PHASE_CAPS[k] for k in tight), "Plan exceeds reserved phase cap")
    return {"policies": 12, "return_episodes": 1536, "anchor_banks": 6, "anchors": 96,
            "own_qpi_references": 768, "common_reference_qpi_references": 288,
            "reserved_transition_cap": 237600, "tight_transition_upper_bound": sum(tight.values()),
            "phase_caps": dict(PHASE_CAPS), "tight_phase_upper_bounds": tight,
            "unspent_capacity_is_not_authority_for_extra_queries": True,
            "independent_unit": "3 training seeds; 128 paired return episodes nested within each seed/condition"}


def _hash_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def start_bank(geometry, law, seed, count):
    """Pure independent draws from the frozen initial law; no environment reset."""
    x = np.asarray(geometry["A"], np.float64)
    require(x.shape == (13,) and np.isfinite(x).all(), "Full A state required")
    require(type(count) is int and count > 0, "Positive fixed bank size required")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    rows = []
    for _ in range(count):
        start = x.copy()
        start[:10] += rng.uniform(-law["physical_start_jitter"], law["physical_start_jitter"], 10)
        start[10] = rng.uniform(*law["current_range"])
        amplitude, phase = rng.uniform(*law["wave_amplitude_range"]), rng.uniform(-np.pi, np.pi)
        start[11:] = amplitude * np.array([np.cos(phase), np.sin(phase)])
        rows.append(start)
    return np.stack(rows)


def support_labels(task_module, states, next_states, geometry, condition):
    """Construction labels for observed edges, never a learned density claim.

    Honeypot removes closed danger-intersecting edges. Bottleneck removes the
    entire closed projection slab [0.2,0.8], including jumps and either direction;
    it deliberately does not use v2's old finite connector capsule.
    """
    x, y = np.asarray(states, np.float64), np.asarray(next_states, np.float64)
    require(x.shape == y.shape and x.shape[-1] == 13 and np.isfinite(x).all() and np.isfinite(y).all(), "Invalid complete edges")
    if condition == "honeypot":
        return task_module.edge_intersects(x, y, geometry["danger"])
    require(condition == "bottleneck", "Unknown support condition")
    b, c = np.asarray(geometry["B"], np.float64), np.asarray(geometry["C"], np.float64)
    require(b.shape == c.shape == (13,), "Full B/C states required")
    d = c[:10] - b[:10]
    norm2 = float(d @ d)
    require(np.isfinite(norm2) and norm2 > 0, "Degenerate separating direction")
    qs = np.sum((x[..., :10] - b[:10]) * d, axis=-1) / norm2
    qn = np.sum((y[..., :10] - b[:10]) * d, axis=-1) / norm2
    return (np.minimum(qs, qn) <= .8) & (np.maximum(qs, qn) >= .2)


def _action(control, state, t):
    a = np.asarray(control(state.copy(), t))
    require(a.shape == (4,) and a.dtype == np.float32 and np.isfinite(a).all() and np.all(np.abs(a) <= 1), "Controller must supply deployed actions")
    return a.copy()


def build_anchors(task, controller, starts, condition, persist):
    """One prefix and one exact replay of its last step per fixed start.

    No prefix is dropped or replaced when it terminates early. The caller's
    physical meter counts every call, including a replay that fails parity.
    """
    target = ANCHOR_TIMES[condition]
    rows, traces = [], []
    for index, start in enumerate(starts):
        snapshot = task.restore(start.copy(), 0, terminated=False)
        trace = []
        for t in range(target):
            action = _action(controller, snapshot.state, t)
            if condition == "honeypot" and 25 <= t <= 28:
                action = -action
            previous = snapshot
            snapshot, reward, info = task.step_snapshot(snapshot, action)
            trace.append({"state": previous.state.copy(), "time": t, "action": action.copy(),
                          "next_state": snapshot.state.copy(), "reward": float(reward),
                          "terminated": bool(snapshot.terminated), "reason": snapshot.terminal_reason})
            if snapshot.terminated:
                persist("invalid_anchor_%02d" % index, {"index": index, "condition": condition,
                        "target_time": target, "trace": trace, "reason": "prefix terminated; no replacement allowed"})
                raise ValueError("Fixed anchor prefix terminated before the required live anchor")
        replay, replay_reward, replay_info = task.step_snapshot(previous, action)
        exact = (np.array_equal(replay.state, snapshot.state) and replay_reward == reward and
                 replay.time == snapshot.time == target and replay.terminated == snapshot.terminated and
                 replay.terminal_reason == snapshot.terminal_reason)
        persist("anchor_%02d" % index, {"index": index, "condition": condition, "trace": trace,
                "anchor_state": snapshot.state.copy(), "anchor_time": target,
                "last_prefix_step_exact_replay": bool(exact)})
        require(exact, "Anchor final-prefix-step parity failed")
        rows.append(snapshot.state.copy()); traces.append(trace)
    return np.stack(rows), np.full(len(rows), target, np.int64), traces


def controller_qpi(task, controller, states, times, actions):
    """Common policy uses exact physical states, never reconstructed float32 obs."""
    n, k = actions.shape[:2]
    values, first_next = np.zeros((n, k), np.float64), np.empty((n, k, 13), np.float64)
    completed = 0
    for i in range(n):
        for slot in range(k):
            live = task.restore(states[i].copy(), int(times[i]), terminated=False)
            offset = 0
            while not live.terminated:
                require(offset < 100 - int(times[i]), "Controller reference exceeded finite horizon")
                action = actions[i, slot].copy() if offset == 0 else _action(controller, live.state, live.time)
                live, reward, info = task.step_snapshot(live, action)
                completed += 1
                if offset == 0:
                    first_next[i, slot] = live.state
                values[i, slot] += .98 ** offset * float(reward)
                offset += 1
    return values, first_next, completed


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def _signed_detail(q, u, truth, labels):
    raw, adjusted = q - truth, q - 3. * u - truth
    def stratum(mask):
        return {"n": int(mask.sum()), "raw_error": raw[mask].tolist(), "adjusted_error": adjusted[mask].tolist(),
                "raw_over": int((raw[mask] > 1.).sum()), "raw_under": int((raw[mask] < -1.).sum()),
                "raw_near": int((np.abs(raw[mask]) <= 1.).sum()),
                "adjusted_over": int((adjusted[mask] > 1.).sum()), "adjusted_under": int((adjusted[mask] < -1.).sum()),
                "adjusted_near": int((np.abs(adjusted[mask]) <= 1.).sum()),
                "raw_mae": None if not mask.any() else float(np.abs(raw[mask]).mean()),
                "adjusted_mae": None if not mask.any() else float(np.abs(adjusted[mask]).mean())}
    return {"omitted_edge": stratum(labels), "retained_edge": stratum(~labels),
            "over_to_under": int(((raw > 1.) & (adjusted < -1.)).sum()),
            "near_to_under": int(((np.abs(raw) <= 1.) & (adjusted < -1.)).sum()),
            "existing_underestimation_worsened": int(((raw < -1.) & (adjusted < raw - 1.)).sum())}


def _common_ranking(q, u, values):
    def choose(a):
        return np.asarray([int(np.flatnonzero(row >= row.max() - 1e-10)[0]) for row in a])
    a, b = choose(q[:, :3]), choose(q[:, :3] - 3. * u[:, :3])
    idx = np.arange(len(a)); gain = values[idx, b] - values[idx, a]
    return {"raw_indices": a.tolist(), "adjusted_indices": b.tolist(), "mean_value_gain": float(gain.mean()),
            "beneficial_switches": int((gain > 1.).sum()), "harmful_switches": int((gain < -1.).sum()),
            "raw_mean_regret": float((values.max(axis=1) - values[idx, a]).mean()),
            "adjusted_mean_regret": float((values.max(axis=1) - values[idx, b]).mean()),
            "reference": "shared controller; descriptive choice quality, not this critic's own-policy error"}


def _policy_identity(policies):
    expected = {(seed, condition, arm) for seed in range(3) for condition in ANCHOR_TIMES for arm in ("native", "bca")}
    require(set(policies) == expected, "Exactly twelve authenticated final policies are required")
    for key, policy in policies.items():
        seed, condition, arm = key
        require(all(callable(policy[k]) for k in ("actor", "q1", "width")), "Missing frozen inference callback")
        meta = policy["identity"]
        require((meta["seed_index"], meta["condition"], meta["arm"]) == key and
                meta["cumulative_critic_updates"] == 20000 and meta["training_kappa"] == (3. if arm == "bca" else 0.), "Wrong final checkpoint cell")
        for field in ("checkpoint_sha256", "normalization_sha256", "dataset_sha256"):
            require(isinstance(meta[field], str) and len(meta[field]) == 64 and all(c in "0123456789abcdef" for c in meta[field]), "Invalid identity hash")
    for seed in range(3):
        for condition in ANCHOR_TIMES:
            a, b = (policies[(seed, condition, arm)]["identity"] for arm in ("native", "bca"))
            require(all(a[k] == b[k] for k in ("normalization_sha256", "dataset_sha256")), "Unmatched pair data/normalization")


def evaluate_study(task, controller, geometry, protocol, policies, counter, persist):
    """Explicit worker call, after target-runtime restore checks and launch approval.

    ``counter.begin(phase)``/``counter.receipt()`` must refer to the physical
    meter already wrapped around task.plant_step, with PHASE_CAPS and the shared
    wall deadline enforced. ``persist(name,json_value)`` must write exclusively
    and fail on persistence error. No policy/critic/uncertainty updates occur.
    Callback input is raw float32 obs14: the loader owns authenticated frozen
    normalization and checkpoint state; metadata alone does not authenticate it.
    """
    specification = plan(protocol)
    _policy_identity(policies)
    E, T = _source_modules()
    cells, phase_completed = [], dict.fromkeys(PHASE_CAPS, 0)
    policy_evaluations = return_episodes = actor_callback_calls = 0
    def emit(name, value): persist(name, _jsonable(value))
    emit("evaluation_plan", specification)
    try:
        for seed in range(3):
            for condition in protocol["conditions"]:
                prefix = "%s_seed%d" % (condition, seed)
                return_starts = start_bank(geometry, protocol["collection"], protocol["evaluation_seeds"][seed]["returns"], 128)
                anchor_starts = start_bank(geometry, protocol["collection"], protocol["evaluation_seeds"][seed]["anchors"], 16)
                ids = [prefix + "/return/%03d" % i for i in range(128)]
                emit(prefix + "_banks", {"return_starts": return_starts, "anchor_starts": anchor_starts,
                     "return_ids": ids, "return_start_sha256": _hash_array(return_starts),
                     "anchor_start_sha256": _hash_array(anchor_starts), "anchor_recipe": ANCHOR_RECIPES[condition],
                     "cross_condition_random_numbers_shared": True})
                counter.begin("prefix_parity")
                states, times, traces = build_anchors(task, controller, anchor_starts, condition,
                    lambda name, value: emit(prefix + "_" + name, value))
                phase_completed["prefix_parity"] += 16 * (ANCHOR_TIMES[condition] + 1)
                base = np.stack([_action(controller, x, int(t)) for x, t in zip(states, times)])
                common_actions = np.stack((base, -base, np.zeros_like(base)), axis=1)
                counter.begin("common_reference")
                common_qpi, common_next, used = controller_qpi(task, controller, states, times, common_actions)
                phase_completed["common_reference"] += used
                common_labels = support_labels(T, np.repeat(states[:, None, :], 3, axis=1), common_next, geometry, condition)
                emit(prefix + "_common_reference", {"states": states, "times": times, "actions": common_actions,
                     "first_next_states": common_next, "omitted_edge": common_labels, "qpi": common_qpi,
                     "reference": "unchanged controller from exact full states; gamma .98; true terminal; no bootstrap"})
                returns, arms = {}, {}
                for arm in ("native", "bca"):
                    policy = policies[(seed, condition, arm)]
                    counter.begin("returns")
                    run = E.rollout(task, return_starts.copy(), np.zeros(128, np.int64), policy["actor"], max_transitions=12800)
                    require(run["counts"]["complete_episodes"] == 128, "Incomplete fixed return bank")
                    phase_completed["returns"] += run["counts"]["transitions_completed"]
                    return_episodes += run["counts"]["complete_episodes"]
                    actor_callback_calls += run["counts"]["actor_calls"]
                    returns[arm] = run["returns"]
                    exposure = []
                    reasons = []
                    for episode in run["traces"]:
                        x = np.stack([r["state"] for r in episode]); y = np.stack([r["info"]["state"] for r in episode])
                        exposure.append(support_labels(T, x, y, geometry, condition))
                        reasons.append(episode[-1]["info"]["terminal_reason"])
                    emit(prefix + "_" + arm + "_returns", {"identity": policy["identity"], "starts": return_starts,
                         "ids": ids, "returns": run["returns"], "traces": run["traces"], "omitted_edges": exposure,
                         "terminal_reasons": reasons, "counts": run["counts"]})
                    obs = np.stack([task.observe(x, int(t)) for x, t in zip(states, times)])
                    own = np.asarray(policy["actor"](obs))
                    actor_callback_calls += 1
                    require(own.dtype == np.float32 and own.shape == (16, 4) and np.isfinite(own).all() and np.all(np.abs(own) <= 1), "Invalid own actor probe")
                    actions = np.concatenate((common_actions, own[:, None, :]), axis=1)
                    flat_obs, flat_actions = np.repeat(obs, 4, axis=0), actions.reshape(64, 4)
                    q = np.asarray(policy["q1"](flat_obs, flat_actions), np.float64)
                    width = np.asarray(policy["width"](flat_obs, flat_actions), np.float64)
                    require(q.shape == width.shape == (64,), "Raw Q and width callbacks must return one scalar per complete action")
                    q, width = q.reshape(16, 4), width.reshape(16, 4)
                    require(np.isfinite(q).all() and np.isfinite(width).all() and (width >= 0).all(), "Nonfinite Q/width: no replacement or recalibration")
                    counter.begin("own_qpi")
                    probe = E.rollout(task, np.repeat(states, 4, axis=0), np.repeat(times, 4), policy["actor"],
                                     first_actions=flat_actions, max_transitions=16 * 4 * (100 - ANCHOR_TIMES[condition]))
                    phase_completed["own_qpi"] += probe["counts"]["transitions_completed"]
                    actor_callback_calls += probe["counts"]["actor_calls"]
                    truth = probe["returns"].reshape(16, 4)
                    first_next = np.stack([row[0]["info"]["state"] for row in probe["traces"]]).reshape(16, 4, 13)
                    require(np.array_equal(first_next[:, :3], common_next), "Shared-action first transition differs across references")
                    labels = support_labels(T, np.repeat(states[:, None, :], 4, axis=1), first_next, geometry, condition)
                    diagnostics = E.score_diagnostics(q, width, truth, error_tolerance=1., tie_tolerance=1e-10, common_slots=3)
                    arms[arm] = {"return_mean": float(run["returns"].mean()), "episodes": 128,
                        "goal_episodes": reasons.count("goal"), "danger_episodes": reasons.count("danger"),
                        "horizon_episodes": reasons.count("horizon"),
                        "episodes_with_omitted_edge": sum(bool(x.any()) for x in exposure),
                        "omitted_edges": sum(int(x.sum()) for x in exposure),
                        "observed_policy_edges": sum(len(x) for x in exposure),
                        "own_policy_errors_and_ranking": diagnostics, "signed_by_support": _signed_detail(q, width, truth, labels),
                        "common_reference_ranking": _common_ranking(q, width, common_qpi)}
                    emit(prefix + "_" + arm + "_qpi", {"identity": policy["identity"], "states": states, "times": times,
                         "actions": actions, "first_next_states": first_next, "omitted_edge": labels, "raw_q1": q,
                         "width": width, "adjusted_actor_score": q - 3. * width, "own_policy_qpi": truth,
                         "common_controller_qpi": common_qpi, "probe_traces": probe["traces"],
                         "counts": probe["counts"], "metrics": arms[arm],
                         "native_adjusted_score_is_diagnostic_only": arm == "native"})
                    policy_evaluations += 1
                paired = E.paired_returns(returns["native"], returns["bca"], ids, ids)
                cell = {"seed_index": seed, "condition": condition, "paired_returns": paired, "arms": arms}
                emit(prefix + "_paired", cell)
                cells.append(cell)
        require(len(cells) == 6 and policy_evaluations == 12 and return_episodes == 1536 and
                all(phase_completed[k] <= PHASE_CAPS[k] for k in PHASE_CAPS), "Incomplete or over-budget evaluation")
        summary = {"status": "completed_frozen_evaluation", "cells": cells, "counts": counter.receipt(),
                   "completed_policy_evaluations": policy_evaluations, "completed_paired_cells": len(cells),
                   "completed_return_episodes": return_episodes, "actor_callback_calls": actor_callback_calls,
                   "logical_completed_transitions_by_phase": phase_completed, "plan": specification,
                   "per_condition": {condition: {"paired_seed_differences": [c["paired_returns"]["paired_mean_difference"] for c in cells if c["condition"] == condition],
                       "mean_across_three_seeds": float(np.mean([c["paired_returns"]["paired_mean_difference"] for c in cells if c["condition"] == condition]))} for condition in ANCHOR_TIMES},
                   "interpretation": "Exploratory paired policy comparison; episodes/anchors are nested, not extra independent seeds. No action-only or Bayesian-uniqueness claim."}
        emit("evaluation_report", summary)
        return summary
    except BaseException as exc:
        failure = {"status": "incomplete_evaluation", "error": type(exc).__name__ + ": " + str(exc),
                   "completed_cells": len(cells), "counts": counter.receipt(),
                   "completed_policy_evaluations": policy_evaluations, "completed_return_episodes": return_episodes,
                   "actor_callback_calls": actor_callback_calls,
                   "logical_completed_transitions_by_phase": phase_completed,
                   "last_rollout_counts": getattr(exc, "evaluation_counts", None), "no_automatic_retry": True}
        try: emit("evaluation_failure", failure)
        except BaseException as write_error:
            print(json.dumps(_jsonable({**failure, "persistence_error": str(write_error)})), flush=True)
        raise
