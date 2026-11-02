"""Additive technical smoke on sealed final banks; no execution at import time.

Only an explicitly authorized runtime worker calls check_and_evaluate. It must
wrap the task's plant with the physical meter and enforce its wall deadline.
This module never loads models, fits, approves, launches, or changes banks.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V2 = REPO / "experiments/thrusterchain_bca/two_scenario_v2"
EVALUATE_SHA256 = "726fc84d17748dc237b57ab2865506f3eb0077d3985ac5a14ac3a6cc8c8444ff"
ORIGINAL_MANIFEST_SHA256 = "2f10a640684f6ea7040df9fe0388544207e1c4562a547a2ff24a04d31bf1a366"
GEOMETRY_SHA256 = "942271e7ed260356cfc4b4543fa79c585c0e2721bf19514bcbcfa581946c2411"
EVIDENCE_SHA256 = "f1da8d7eb8f3d91fb1b8181790ccfeba6dc213cd57d9deeb56ee49267da445c5"
PHASE_CAPS = {"prefix_parity": 2688, "bank_edges": 288, "forced_qpi": 144, "returns": 200}


def require(value, message):
    if not value: raise ValueError(message)


def _read_bound(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Changed sealed input: " + str(path))
    return json.loads(raw)


def _sealed_inputs(geometry, protocol):
    """Authenticate old helpers/banks before import; root seals this amendment."""
    path = HERE / "evaluate.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == EVALUATE_SHA256, "Sealed final evaluator changed")
    manifest = _read_bound(HERE / "MANIFEST.json", ORIGINAL_MANIFEST_SHA256)
    p = HERE / "protocol.json"
    frozen = _read_bound(p, manifest["files"][p.relative_to(REPO).as_posix()])
    keys = ("evaluation_seeds", "collection", "conditions", "condition_anchor_times", "condition_anchor_recipes",
            "common_actions", "geometry_recipe", "horizon", "gamma", "anchors_per_seed_condition")
    require(all(protocol[k] == frozen[k] for k in keys), "Smoke must use the exact final evaluation bank recipe")
    require(geometry == _read_bound(V2 / "setup_attempt_1/geometry.json", GEOMETRY_SHA256), "Geometry changed")
    spec = importlib.util.spec_from_file_location("_path_a_sealed_smoke_evaluate", path)
    E = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = E
    spec.loader.exec_module(E)
    E.plan(protocol)
    R, T = E._source_modules()
    evidence = _read_bound(V2 / "setup_attempt_1/geometry_evidence.json", EVIDENCE_SHA256)
    require(evidence["families"] == 15 and evidence["cases"][0]["family"] == 0, "Wrong nominal evidence family")
    trace = evidence["cases"][0]["variants"]["danger"]["trace"]
    row = trace[29]
    require(len(trace) == 30 and row["time"] == 29 and row["terminated"] and row["reason"] == "danger" and
            not trace[28]["terminated"] and row["state"] == trace[28]["next_state"], "Wrong saved live danger anchor")
    return E, R, T, row


class _RecordedTask:
    """Capture existing production-rollout calls, without additional queries."""
    def __init__(self, task):
        self.task, self.events = task, []

    def observe(self, state, time):
        return self.task.observe(state, time)

    def step(self, state, time, action):
        nxt, reward, done, info = self.task.step(state, time, action)
        self.events.append({"state": state.copy(), "time": int(time), "action": action.copy(),
            "next_state": nxt.copy(), "reward": float(reward), "terminated": bool(done), "info": info})
        return nxt, reward, done, info


def check_and_evaluate(task, controller, geometry, protocol, policies, counter, persist):
    """Use disposable native/BCA actor/q1/width callbacks supplied by the worker.

    The task's physical meter must expose begin/receipt with PHASE_CAPS, total
    3320 and the shared wall deadline. Persist is exclusive and must raise on
    failure. All six fixed banks are checked even when coverage fails. No bank,
    action, seed or failed case is replaced. Earlier technical smoke exposure
    does not turn these development banks into untouched confirmation data.
    """
    require(set(policies) == {"native", "bca"} and
            all(callable(policies[a][k]) for a in policies for k in ("actor", "q1", "width")),
            "Both disposable production callback sets are required")
    E, R, T, known = _sealed_inputs(geometry, protocol)
    banks, arm_reports = [], {}
    def emit(name, value): persist(name, E._jsonable(value))
    emit("smoke_evaluation_plan", {"phase_caps": PHASE_CAPS, "total_transition_cap": 3320,
         "final_banks": 6, "anchors": 96, "common_edges_per_bank": 48, "common_edges": 288,
         "coverage_rule": "Every fixed bank must have at least one retained and one omitted common-action edge",
         "scope": "Technical evaluation-path smoke; no reward/critic-error efficacy acceptance criterion"})
    try:
        for seed in range(3):
            for condition in protocol["conditions"]:
                name = "smoke_%s_seed%d" % (condition, seed)
                starts = E.start_bank(geometry, protocol["collection"], protocol["evaluation_seeds"][seed]["anchors"], 16)
                emit(name + "_starts", {"starts": starts, "seed": protocol["evaluation_seeds"][seed]["anchors"],
                     "recipe": E.ANCHOR_RECIPES[condition], "start_sha256": E._hash_array(starts)})
                counter.begin("prefix_parity")
                states, times, prefix_traces = E.build_anchors(task, controller, starts, condition,
                    lambda key, value: emit(name + "_" + key, value))
                base = np.stack([E._action(controller, x, int(t)) for x, t in zip(states, times)])
                actions = np.stack((base, -base, np.zeros_like(base)), axis=1)
                counter.begin("bank_edges")
                next_states = np.empty((16, 3, 13), np.float64)
                events = []
                for i in range(16):
                    for slot in range(3):
                        before = task.restore(states[i].copy(), int(times[i]), terminated=False)
                        after, reward, info = task.step_snapshot(before, actions[i, slot].copy())
                        next_states[i, slot] = after.state
                        event = {"anchor": i, "slot": slot, "state": before.state.copy(), "time": before.time,
                                 "action": actions[i, slot].copy(), "next_state": after.state.copy(),
                                 "reward": float(reward), "terminated": bool(after.terminated),
                                 "terminal_reason": after.terminal_reason, "next_time": after.time}
                        events.append(event)
                        emit(name + "_edge_%02d_%d" % (i, slot), event)
                labels = E.support_labels(T, np.repeat(states[:, None, :], 3, axis=1), next_states, geometry, condition)
                omitted, retained = int(labels.sum()), int((~labels).sum())
                bank = {"seed_index": seed, "condition": condition, "anchors": 16, "common_edges": len(events),
                        "states": states, "times": times, "actions": actions, "actual_next_states": next_states,
                        "omitted_edges": labels, "retained_edges": ~labels, "omitted_count": omitted,
                        "retained_count": retained, "passed": len(events) == 48 and omitted >= 1 and retained >= 1}
                emit(name + "_coverage", bank)
                banks.append(bank)

        anchor = np.asarray(known["state"], np.float64)
        bad_action = np.asarray(known["action"], np.float32)
        safe_action = E._action(controller, anchor, 29)
        obs = np.stack([task.observe(anchor, 29)] * 2)
        first_actions = np.stack((bad_action, safe_action))
        emit("smoke_own_policy_anchor", {"state": anchor, "time": 29, "first_actions": first_actions,
             "known_bad_next_state": known["next_state"], "known_bad_reward": known["reward"],
             "known_bad_terminal": True, "known_bad_reason": "danger", "source": "sealed nominal family0 danger trace row29"})
        for arm in ("native", "bca"):
            policy = policies[arm]
            raw = np.asarray(policy["q1"](obs, first_actions), np.float64)
            width = np.asarray(policy["width"](obs, first_actions), np.float64)
            require(raw.shape == width.shape == (2,) and np.isfinite(raw).all() and np.isfinite(width).all() and
                    np.all(width >= 0), "Disposable Q/width callback has invalid shape or values")
            references = {}
            counter.begin("forced_qpi")
            for label, action, cap in (("bad", bad_action, 1), ("safe", safe_action, 71)):
                recorder = _RecordedTask(task)
                try:
                    run = R.rollout(recorder, anchor[None, :].copy(), np.array([29], np.int64), policy["actor"],
                                    first_actions=action[None, :].copy(), max_transitions=cap)
                except BaseException as exc:
                    emit("smoke_%s_%s_incomplete" % (arm, label), {"events": recorder.events,
                         "error": type(exc).__name__ + ": " + str(exc),
                         "rollout_counts": getattr(exc, "evaluation_counts", None), "physical_counts": counter.receipt()})
                    raise
                reference = {"returns": run["returns"], "events": recorder.events, "final_times": run["final_times"],
                             "counts": run["counts"], "reference": "forced first action then this disposable actor; gamma .98; true terminal"}
                if label == "bad":
                    event = recorder.events[0]
                    reference["exact_saved_bad_replay"] = bool(len(recorder.events) == 1 and run["final_times"][0] == 30 and
                        np.array_equal(event["next_state"], np.asarray(known["next_state"], np.float64)) and
                        event["reward"] == known["reward"] and event["terminated"] and event["info"]["terminal_reason"] == "danger")
                emit("smoke_%s_%s_reference" % (arm, label), reference)
                if label == "bad":
                    require(reference["exact_saved_bad_replay"], "Actual saved bad step failed exact replay; no technical pass")
                references[label] = reference
            counter.begin("returns")
            recorder = _RecordedTask(task)
            try:
                free = R.rollout(recorder, np.asarray(geometry["A"], np.float64)[None, :], np.array([0], np.int64),
                                 policy["actor"], max_transitions=100)
            except BaseException as exc:
                emit("smoke_%s_free_incomplete" % arm, {"events": recorder.events,
                     "error": type(exc).__name__ + ": " + str(exc),
                     "rollout_counts": getattr(exc, "evaluation_counts", None), "physical_counts": counter.receipt()})
                raise
            qpi = np.array([references[k]["returns"][0] for k in ("bad", "safe")], np.float64)
            report = {"arm": arm, "raw_q1": raw, "width": width, "adjusted_score": raw - 3. * width,
                      "own_policy_qpi": qpi, "raw_error": raw - qpi, "adjusted_error": raw - 3. * width - qpi,
                      "free_return": float(free["returns"][0]), "free_events": recorder.events,
                      "free_final_time": int(free["final_times"][0]), "free_counts": free["counts"],
                      "bad_reference": references["bad"], "safe_reference": references["safe"],
                      "efficacy_acceptance_applied": False}
            emit("smoke_" + arm + "_evaluation", report)
            arm_reports[arm] = report
        counts = counter.receipt()
        require(counts.get("execution_counts_known") is True and counts["attempted"] == counts["completed"] and
                set(counts["completed"]) == set(PHASE_CAPS) and
                all(0 <= counts["completed"][key] <= limit for key, limit in PHASE_CAPS.items()) and
                counts["completed"]["prefix_parity"] == 2688 and counts["completed"]["bank_edges"] == 288,
                "Incomplete or inconsistent physical smoke accounting")
        passed = len(banks) == 6 and all(b["passed"] for b in banks) and len(arm_reports) == 2
        result = {"status": "completed_technical_evaluation_smoke", "passed": bool(passed),
                  "final_banks": 6, "anchors": 96, "common_edges": 288,
                  "bank_coverage": [{k: b[k] for k in ("seed_index", "condition", "anchors", "common_edges", "omitted_count", "retained_count", "passed")} for b in banks],
                  "completed_disposable_arms": len(arm_reports), "counts": counts, "transition_cap": 3320,
                  "policy_performance_is_not_a_smoke_pass_criterion": True,
                  "no_resampling_or_adaptive_fill": True, "development_banks_now_technically_inspected": True}
        emit("smoke_evaluation_report", result)
        return result
    except BaseException as exc:
        failure = {"status": "incomplete_technical_evaluation_smoke", "passed": False,
                   "error": type(exc).__name__ + ": " + str(exc), "completed_banks": len(banks),
                   "completed_disposable_arms": len(arm_reports), "counts": counter.receipt(), "no_automatic_retry": True}
        try: emit("smoke_evaluation_failure", failure)
        except BaseException as write_error:
            print(json.dumps(E._jsonable({**failure, "persistence_error": str(write_error)})), flush=True)
        raise
