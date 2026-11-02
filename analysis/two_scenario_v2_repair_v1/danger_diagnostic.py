"""One manual, bounded same-state diagnostic; never a training entrypoint."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V2 = REPO / "experiments/thrusterchain_bca/two_scenario_v2"
V3 = REPO / "experiments/thrusterchain_bca/two_scenario_v3"
CORE = REPO / "experiments/thrusterchain_bca/titan_20261026/payload/transfer_bundle/core_v4"
EST = timezone(timedelta(hours=-5))
BOUND = {
    "MANIFEST.json": "c4e3198289e9ea8b329bbaedbf8cdc5e1c7cea19249beb121ffaa100af53336d",
    "setup_attempt_1/geometry.json": "942271e7ed260356cfc4b4543fa79c585c0e2721bf19514bcbcfa581946c2411",
    "setup_attempt_1/geometry_evidence.json": "f1da8d7eb8f3d91fb1b8181790ccfeba6dc213cd57d9deeb56ee49267da445c5",
    "setup_attempt_1/RESULT.json": "1e9ba131c35b3e4cafbb8815e3f6338ba6001cc35e1f3e05d2be7518f89941f1",
}
RULES = {
    "anchors": 15, "time": 29, "bad_terminal_time": 30, "horizon": 100, "gamma": .98,
    "all_bad_replays_exact": True, "nominal_safe_goal_required": True,
    "nominal_safe_minus_bad_return_minimum": 5., "safe_goal_count_minimum": 12,
    "plant_call_cap": 1080, "state_initialization_cap": 30,
    "internal_seconds": 45, "external_timeout_seconds": 55, "kill_grace_seconds": 5,
    "manual_execution_charge_seconds": 60, "preparation_charge_seconds": 30,
    "phase_charge_seconds": 90, "previous_charged_seconds": 1800,
    "cumulative_charged_seconds": 1890, "remaining_reserved_seconds": 12510,
    "previous_simulator_steps": 189042, "remaining_setup_steps_after_full_diagnostic": 70678,
    "unchanged_future_evaluation_steps": 237600,
    "training_authorized": False, "automatic_retry": False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def no_v3_attempt():
    require(not list(V3.glob("setup_attempt_*")),
            "A v3 attempt exists; shared budget may be stale. Do not run this diagnostic.")


def source_check():
    no_v3_attempt()
    for relative, expected in BOUND.items():
        require(sha(V2 / relative) == expected, "Bound v2 artifact changed: " + relative)
    manifest = read(V2 / "MANIFEST.json")
    # Authenticate the checker before importing it, then use its original full check.
    relative = (V2 / "setup.py").relative_to(REPO).as_posix()
    require(sha(V2 / "setup.py") == manifest["files"][relative], "v2 checker changed")
    setup = load_module("_same_state_diagnostic_v2_setup", V2 / "setup.py")
    protocol = setup.check()
    result = read(V2 / "setup_attempt_1/RESULT.json")
    require(result["prepared_identity_unchanged"] and result["execution_counts_known"] and
            result["status"] == "setup_passed_pending_geometry_protocol_review", "Incomplete v2 source result")
    require(result["counts"]["attempted"] == result["counts"]["completed"] and
            sum(result["counts"]["completed"].values()) == 188832, "v2 accounting changed")
    require(protocol["gamma"] == .98 and protocol["horizon"] == 100 and
            protocol["setup_rules"]["reference_return_advantage"] == 5. and
            protocol["setup_rules"]["required_safe_goal_families"] == 12, "Original criteria changed")
    return protocol


def contract():
    return {"schema": "two_scenario.v2.same_state_danger.v1", "rules": RULES,
            "bound_v2_artifacts": BOUND,
            "files": {name: sha(HERE / name) for name in ("danger_diagnostic.py", "test_danger_diagnostic.py")},
            "source_closure": "Exact v2 manifest and unchanged v2 setup.check()",
            "scope": "Development fixed-controller returns from all 15 saved danger states; not learned Q or BCA efficacy"}


def prepare():
    source_check()
    require(not (HERE / "danger_attempt_1").exists(), "Preserve the prior diagnostic attempt")
    save(HERE / "DANGER_CONTRACT.json", contract())
    return {"status": "prepared_not_executed", "contract_sha256": sha(HERE / "DANGER_CONTRACT.json")}


def check():
    protocol = source_check()
    require(read(HERE / "DANGER_CONTRACT.json") == contract(), "Diagnostic code, tests or contract changed")
    return protocol


def anchor_bank(evidence):
    import numpy as np
    require(evidence["families"] == 15 and len(evidence["cases"]) == 15, "All 15 source cases are required")
    anchors = []
    for family, case in enumerate(evidence["cases"]):
        require(case["family"] == family, "Source family ordering changed")
        danger = case["variants"]["danger"]
        trace = danger["trace"]
        require(len(trace) == 30 and danger["terminal_time"] == 30, "Expected saved danger at time 30")
        row = trace[29]
        state, nxt = np.asarray(row["state"], np.float64), np.asarray(row["next_state"], np.float64)
        action = np.asarray(row["action"], np.float64)
        require(state.shape == nxt.shape == (13,) and action.shape == (4,), "Incomplete full-state anchor")
        require(np.all(np.isfinite(state)) and np.all(np.isfinite(nxt)) and np.all(np.isfinite(action)) and
                np.isfinite(row["reward"]), "Nonfinite anchor")
        require(np.array_equal(action, action.astype(np.float32).astype(np.float64)) and np.all(np.abs(action) <= 1),
                "Saved command is not an exact deployed float32 action")
        require(row["time"] == 29 and row["terminated"] is True and row["reason"] == "danger" and
                not trace[28]["terminated"] and np.array_equal(state, np.asarray(trace[28]["next_state"])),
                "Anchor time, live-state lineage or danger label changed")
        anchors.append({"family": family, "source_row": 29, "state": row["state"], "time": 29,
                        "bad_action": row["action"], "expected_next_state": row["next_state"],
                        "expected_reward": row["reward"], "expected_terminated": True,
                        "expected_terminal_reason": "danger", "expected_next_time": 30})
    return anchors


class Counter:
    def __init__(self, plant, deadline):
        self.plant, self.deadline = plant, deadline
        self.attempted = self.completed = self.starts = 0

    def initialize(self):
        self.deadline()
        require(self.starts < 30, "Diagnostic initialization cap exhausted")
        self.starts += 1

    def __call__(self, state, action):
        self.deadline()
        require(state.shape == (13,) and action.shape == (4,), "Exactly one full-state call required")
        require(self.attempted < 1080, "Diagnostic transition cap exhausted")
        self.attempted += 1
        value = self.plant(state, action)
        self.completed += 1
        return value

    def receipt(self):
        return {"plant_calls_attempted": self.attempted, "plant_calls_completed": self.completed,
                "state_initializations_attempted": self.starts,
                "execution_counts_known": self.attempted == self.completed,
                "critic_updates": 0, "actor_updates": 0, "auxiliary_updates": 0,
                "calibrations": 0, "gym_reset_calls": 0}


class ReplayMismatch(RuntimeError):
    """Saved transition cannot be reproduced; do not interpret new outcomes."""


def run_cases(env, control, anchors, counter, output):
    """Injected fake-task-testable path; write anchors before the first call."""
    import numpy as np
    require(len(anchors) == 15 and [a["family"] for a in anchors] == list(range(15)), "Do not select anchors")
    save(output / "ANCHOR_BANK.json", anchors)
    cases = []
    for anchor in anchors:
        family, state, t = anchor["family"], np.asarray(anchor["state"], np.float64), anchor["time"]
        require(t == 29 and state.shape == (13,), "Wrong anchor time or full state")
        case = {"family": family, "time": t, "state": state.tolist(), "bad": None, "safe": None}
        def step(snapshot, action, variant):
            next_snapshot, reward, info = env.step_snapshot(snapshot, action)
            event = {"family": family, "variant": variant, "time": snapshot.time,
                     "state": snapshot.state.tolist(), "action": np.asarray(action).tolist(),
                     "next_state": next_snapshot.state.tolist(), "next_time": next_snapshot.time,
                     "reward": float(reward), "terminated": next_snapshot.terminated,
                     "terminal_reason": next_snapshot.terminal_reason, "counts": counter.receipt()}
            append(output / "transitions.jsonl", event)
            return next_snapshot, float(reward), event
        try:
            counter.initialize()
            live = env.restore(state.copy(), t, terminated=False)
            bad, reward, event = step(live, np.asarray(anchor["bad_action"], np.float32), "bad")
            exact = (np.array_equal(bad.state, np.asarray(anchor["expected_next_state"], np.float64)) and
                     reward == anchor["expected_reward"] and bad.terminated is True and
                     bad.terminal_reason == "danger" and bad.time == 30)
            case["bad"] = {"return": reward, "trace": [event], "exact_replay": bool(exact)}
            if not exact:
                raise ReplayMismatch("Saved bad-step replay is not exact; diagnostic stopped without new anchors")
            counter.initialize()
            live = env.restore(state.copy(), t, terminated=False)
            rewards, trace = [], []
            while not live.terminated:
                require(29 <= live.time < 100 and len(rewards) < 71, "Safe continuation horizon changed")
                action = np.asarray(control(live.state, live.time), np.float32)
                live, reward, event = step(live, action, "safe")
                rewards.append(reward)
                trace.append(event)
            value = float(sum(.98 ** j * reward for j, reward in enumerate(rewards)))
            case["safe"] = {"return": value, "trace": trace, "terminal_time": live.time,
                            "terminal_reason": live.terminal_reason}
            case["safe_minus_bad_return"] = value - case["bad"]["return"]
        except BaseException as exc:
            case["error"] = type(exc).__name__ + ": " + str(exc)
            # No case is dropped: preserve an incomplete case before propagating.
            try:
                save(output / ("case_%02d.json" % family), case)
            except BaseException as write_exc:
                print(json.dumps({"status": "case_persistence_failed", "family": family,
                                  "error": str(write_exc), "original_error": str(exc),
                                  "counts": counter.receipt()}), flush=True)
            raise
        save(output / ("case_%02d.json" % family), case)
        cases.append(case)
    safe_goals = sum(c["safe"]["terminal_reason"] == "goal" for c in cases)
    checks = {"all15_bad_replays_exact": all(c["bad"]["exact_replay"] for c in cases),
              "nominal_safe_goal": cases[0]["safe"]["terminal_reason"] == "goal",
              "nominal_safe_minus_bad_at_least5": cases[0]["safe_minus_bad_return"] >= 5.,
              "at_least12_safe_goals": safe_goals >= 12}
    return {"denominator": 15, "completed_cases": len(cases), "checks": checks,
            "safe_goal_count": safe_goals, "passed": bool(all(checks.values())),
            "per_family_safe_minus_bad_return": [c["safe_minus_bad_return"] for c in cases],
            "scope": "Same saved time 29/full-state fixed-controller continuation returns; no learned Q, policy or BCA efficacy claim"}


def load_runtime(protocol, deadline):
    """Only --execute calls this function; tests replace it with a fake plant."""
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "gymnasium")}
    require(versions == {"numpy": "2.5.4", "gymnasium": "1.0.0"}, "Numerical runtime differs; no substitution")
    require(not any(k == "jax" or k.startswith("jax.") for k in sys.modules), "No learner/JAX permitted")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(CORE))
    from experiments.thrusterchain_bca.two_scenario_v2 import task as T
    from thrusterchain import environment as canonical
    require(Path(canonical.__file__).resolve() == CORE / "thrusterchain/environment.py", "Canonical module path collision")
    require(Path(T.__file__).resolve() == V2 / "task.py", "Task module path collision")
    physics = canonical.Physics(stall=True, stall_slope=3.)
    source_result = read(V2 / "setup_attempt_1/RESULT.json")
    require(physics.as_dict() == source_result["physics"], "Physics differs from saved v2")
    require(canonical.provenance() == source_result["runtime"]["canonical"], "Canonical provenance differs")
    counter = Counter(lambda x, a: canonical.dynamics(x, a, physics), deadline)
    A, B = canonical.matrices(physics)
    control = T.tracking_controller(A, B, protocol["geometry_recipe"])
    env = T.Task(counter, read(V2 / "setup_attempt_1/geometry.json"))
    runtime = {"python": sys.version, "executable": sys.executable, "versions": versions,
               "physics": physics.as_dict(), "canonical": canonical.provenance()}
    return env, control, counter, runtime


def execute():
    start = time.monotonic()
    # Refuse Windows before importing a plant or creating output/reservation.
    require(os.name == "posix", "Manual existing Linux runtime required; no assistant WSL route")
    protocol = check()
    output = HERE / "danger_attempt_1"
    output.mkdir(exist_ok=False)
    save(output / "ATTEMPT.json", {"started_est": datetime.now(EST).isoformat(), "pid": os.getpid(),
         "contract_sha256": sha(HERE / "DANGER_CONTRACT.json"), "rules": RULES, "no_automatic_retry": True,
         "missing_final_receipt": "Incomplete: counts unknown; persisted completed calls are a lower bound and 1080 is the upper bound. Never assume zero."})
    counter = None
    result = {"status": "failed_or_interrupted_diagnostic", "decision": None, "training_authorized": False,
              "error": None, "contract_sha256": sha(HERE / "DANGER_CONTRACT.json"), "denominator": 15}
    def deadline():
        if time.monotonic() - start >= 45:
            raise TimeoutError("45-second internal diagnostic deadline")
    def stop(signum, frame):
        raise TimeoutError("Diagnostic deadline or signal " + str(signum))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for s in handlers:
        signal.signal(s, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 45 - (time.monotonic() - start)))
    try:
        env, control, counter, result["runtime"] = load_runtime(protocol, deadline)
        anchors = anchor_bank(read(V2 / "setup_attempt_1/geometry_evidence.json"))
        result["evidence"] = run_cases(env, control, anchors, counter, output)
        result.update(status="completed_same_state_diagnostic",
                      decision="passed_development_diagnostic" if result["evidence"]["passed"] else "failed_development_diagnostic")
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc(),
                      failure_kind="fatal_saved_replay_integrity" if isinstance(exc, ReplayMismatch) else "source_runtime_interruption_or_persistence")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    try:
        check()
        result["prepared_identity_unchanged"] = True
    except BaseException as exc:
        result.update(status="failed_or_interrupted_diagnostic", decision=None,
                      prepared_identity_unchanged=False, postcheck_error=str(exc))
    counts = counter.receipt() if counter is not None else Counter(None, lambda: None).receipt()
    result.update(counts=counts, execution_counts_known=counts["execution_counts_known"],
                  elapsed_seconds=time.monotonic() - start, finished_est=datetime.now(EST).isoformat(),
                  case_files_written=len(list(output.glob("case_*.json"))),
                  cumulative_transition_lower=189042 + counts["plant_calls_completed"],
                  cumulative_transition_upper=189042 + counts["plant_calls_attempted"],
                  phase_charged_seconds=90, cumulative_charged_seconds=1890,
                  remaining_reserved_seconds=12510, no_automatic_retry=True)
    if result["elapsed_seconds"] > 55:
        result.update(status="failed_or_interrupted_diagnostic", decision=None, timing_error="Manual process exceeded 55 seconds")
    try:
        save(output / "RESULT.json", result)
    except BaseException as exc:
        result.update(status="result_persistence_failed", decision=None, persistence_error=str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare()))
    elif args.check:
        check()
        print(json.dumps({"status": "checked_not_executed", "contract_sha256": sha(HERE / "DANGER_CONTRACT.json"),
                          "attempt_exists": (HERE / "danger_attempt_1").exists(), "training_authorized": False}))
    else:
        report = execute()
        raise SystemExit(0 if report["status"] == "completed_same_state_diagnostic" else 1)
