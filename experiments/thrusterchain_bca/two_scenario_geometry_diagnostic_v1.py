"""One unchanged-recipe geometry diagnostic; manual execution only, no learner."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = HERE / "two_scenario_v1"
CONTRACT = HERE / "two_scenario_geometry_diagnostic_v1.json"
OUTPUT = HERE / "two_scenario_geometry_diagnostic_v1_attempt_1"
TEST = HERE / "test_two_scenario_geometry_diagnostic_v1.py"
EST = timezone(timedelta(hours=-5))
PARENT_MANIFEST = "a48e465c133093ada9a79955161d4b10fce8b393d05c0a5eb088afb3109bf7cd"
PARENT_RESULT = "7a488838fd77593c45c5c099854f3d7942f16d8ba2bf62e3947836c087269b0c"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    """Preserve nonfinite evidence as explicit strings, never invalid JSON."""
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def persist(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_check():
    require(digest(ORIGINAL / "MANIFEST.json") == PARENT_MANIFEST, "Original manifest changed")
    require(digest(ORIGINAL / "setup_attempt_1/RESULT.json") == PARENT_RESULT, "Original failed receipt changed")
    setup = load_module(ORIGINAL / "setup.py", "two_scenario_original_setup_diagnostic")
    protocol = setup.check()
    return setup, protocol


def contract_value():
    _, protocol = parent_check()
    return {"schema": "two_scenario.geometry_diagnostic.v1", "setup_approval_message":
            "Sentinel_bc18442c3ac88191aaeb71b51a717761", "reaffirmation_message":
            "Sentinel_95d79ed6189c81918a8f398b2ef02a28", "parent_manifest_sha256": PARENT_MANIFEST,
            "parent_result_sha256": PARENT_RESULT, "script_sha256": digest(__file__),
            "fixture_sha256": digest(TEST), "recipe": protocol["geometry_recipe"],
            "unchanged_recipe_only": True, "transition_cap": 105, "initialization_cap": 2,
            "process_cap_seconds": 30, "prior_completed_transitions": 105,
            "prior_measured_seconds": 2.1510453130003953, "prior_setup_charged_seconds": 840,
            "preparation_allowance_seconds": 60, "diagnostic_charged_seconds": 30,
            "remaining_reserved_seconds": 13470, "cumulative_transition_cap_after_diagnostic": 210,
            "unconsumed_setup_transition_allowance_after_full_diagnostic": 260590,
            "later_evaluation_reserved_transitions": 237600, "automatic_retry": False,
            "training_authorized": False, "collection_authorized": False,
            "note": "No allocation is reclaimed; no corrected geometry or full setup retry is authorized."}


def verify():
    actual = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(actual == contract_value(), "Diagnostic contract or bound evidence changed")
    return actual


def measurements(task_module, trace, recipe):
    """Pure arithmetic on captured states; does not call a plant/controller."""
    import numpy as np
    require(len(trace) == 105, "Exactly 100 route and five excursion transitions required")
    b, c, g = int(recipe["b_time"]), int(recipe["c_time"]), int(recipe["goal_time"])
    require((b, c) == (25, 30), "Only the unchanged frozen recipe is supported")
    route = np.asarray([trace[0]["state"]] + [r["next_state"] for r in trace[:100]], np.float64)
    excursion = np.asarray([trace[100]["state"]] + [r["next_state"] for r in trace[100:]], np.float64)
    require(route.shape == (101, 13) and excursion.shape == (6, 13), "Wrong captured state shapes")
    center = excursion[-1, :10]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a, d = route[:-1, :10] - center, np.diff(route[:, :10], axis=0)
        den = np.sum(d * d, axis=1)
        f = np.clip(np.divide(-np.sum(a * d, axis=1), den, out=np.zeros_like(den), where=den > 0), 0, 1)
        separation = float(np.min(np.linalg.norm(a + f[:, None] * d, axis=1)))
        # Preserve the original min operand order, including its NaN behavior.
        radius = min(float(recipe["danger_radius_max"]), float(recipe["danger_radius_fraction"]) * separation)
        delta = route[c, :10] - route[b, :10]
        length = float(np.linalg.norm(delta))
        connector_radius = float(recipe["connector_radius_fraction"]) * length
        goal_distance = float(np.linalg.norm(route[g, :10] - center))
    trim = float(recipe["connector_trim_fraction"])
    geometry = {"A": route[0], "B": route[b], "C": route[c], "D": route[g],
                "nominal_route": route, "excursion": excursion,
                "goal": {"center": route[g, :10], "radius": float(recipe["goal_radius"])},
                "danger": {"center": center, "radius": radius},
                "connector": {"kind": "capsule", "start": route[b, :10] + trim * delta,
                              "end": route[c, :10] - trim * delta, "radius": connector_radius}}
    minimum = float(recipe["minimum_region_radius"])
    checks = {"captured_states_finite": bool(np.isfinite(route).all() and np.isfinite(excursion).all()),
              "danger_separation_finite": math.isfinite(separation), "danger_radius_finite": math.isfinite(radius),
              "danger_radius_minimum": bool(radius >= minimum), "connector_radius_finite": math.isfinite(connector_radius),
              "connector_radius_minimum": bool(connector_radius >= minimum),
              "goal_not_reached_before_C": None, "goal_danger_disjoint": None,
              "B_outside_connector": None, "C_outside_connector": None}
    if checks["captured_states_finite"] and math.isfinite(goal_distance):
        checks["goal_not_reached_before_C"] = not bool(np.any(task_module.contains(route[:c + 1], geometry["goal"])))
        checks["goal_danger_disjoint"] = bool(goal_distance > radius + recipe["goal_radius"])
        if math.isfinite(connector_radius) and connector_radius > 0 and length > 0:
            checks["B_outside_connector"] = not bool(task_module.contains(route[b], geometry["connector"]))
            checks["C_outside_connector"] = not bool(task_module.contains(route[c], geometry["connector"]))
    return {"geometry": geometry, "values": {"danger_route_separation": separation, "danger_radius": radius,
            "connector_length": length, "connector_radius": connector_radius, "goal_danger_center_distance": goal_distance,
            "minimum_region_radius": minimum}, "checks": checks,
            "failed_criteria": [k for k, v in checks.items() if v is False],
            "not_evaluable": [k for k, v in checks.items() if v is None],
            "original_size_guard_rejects": bool(not np.isfinite(radius) or radius < minimum or connector_radius < minimum),
            "note": "Diagnostic predicates do not replace or relax the original constructor checks."}


class Capture:
    def __init__(self, task_module, plant, recipe, root, deadline):
        self.task, self.plant, self.recipe, self.root, self.deadline = task_module, plant, recipe, Path(root), deadline
        self.attempted = self.completed = self.initializations = 0
        self.trace, self.measurements_persist_error = [], None

    def counts(self):
        return {"attempted": self.attempted, "completed": self.completed, "initializations": self.initializations,
                "execution_counts_known": self.attempted == self.completed, "collection_steps": 0,
                "critic_updates": 0, "actor_updates": 0, "auxiliary_updates": 0, "calibrations": 0, "gym_reset_calls": 0}

    def __call__(self, state, action):
        self.deadline()
        require(self.attempted < 105, "Diagnostic transition cap exhausted")
        require(getattr(state, "shape", None) == (13,) and getattr(action, "shape", None) == (4,), "Exactly one transition per call")
        i = self.attempted
        if i in (0, 100):
            self.initializations += 1
        self.attempted += 1
        out = self.plant(state, action)
        self.completed += 1  # Completed work stays counted even if subsequent persistence fails.
        nxt = out[0] if isinstance(out, tuple) else out
        record = {"phase": "nominal" if i < 100 else "excursion", "time": i if i < 100 else 25 + i - 100,
                  "state": state.tolist(), "action": action.tolist(), "next_state": nxt.tolist()}
        self.trace.append(record)
        try:
            with (self.root / "transitions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(safe(record), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if self.completed == 105:
                persist(self.root / "measurements.json", measurements(self.task, self.trace, self.recipe))
        except BaseException as exc:
            self.measurements_persist_error = type(exc).__name__ + ": " + str(exc)
            raise RuntimeError("Geometry diagnostic persistence failed: " + self.measurements_persist_error) from exc
        return out


def execute():
    start = time.monotonic()
    require(os.name == "posix", "Manual existing Linux runtime only; no assistant execution route")
    contract = verify()
    OUTPUT.mkdir(exist_ok=False)
    persist(OUTPUT / "ATTEMPT.json", {"started_est": datetime.now(EST).isoformat(), "pid": os.getpid(),
            "contract_sha256": digest(CONTRACT), "charged_seconds": 30, "no_automatic_retry": True})
    result = {"status": "incomplete_diagnostic", "error": None, "training_authorized": False,
              "contract_sha256": digest(CONTRACT), "parent_result_sha256": PARENT_RESULT}
    capture = None
    def deadline():
        if time.monotonic() - start >= 24:
            raise TimeoutError("Diagnostic internal deadline")
    def stop(signum, frame):
        raise TimeoutError("Diagnostic signal " + str(signum))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    for s in handlers:
        signal.signal(s, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 24 - (time.monotonic() - start)))
    try:
        versions = {n: importlib.metadata.version(n) for n in ("numpy", "gymnasium")}
        require(versions == {"numpy": "2.5.4", "gymnasium": "1.0.0"}, "Frozen numerical runtime mismatch")
        setup, protocol = parent_check()
        sys.path.insert(0, str(REPO)); sys.path.insert(0, str(setup.CORE))
        from experiments.thrusterchain_bca.two_scenario_v1 import task as task_module
        from thrusterchain import environment as canonical
        require(Path(canonical.__file__).resolve() == setup.CORE / "thrusterchain/environment.py", "Canonical path mismatch")
        require(not any(k == "jax" or k.startswith("jax.") for k in sys.modules), "Learner/JAX must not be imported")
        physics = canonical.Physics(stall=True, stall_slope=3.)
        result["runtime"] = {"executable": sys.executable, "python": sys.version, "versions": versions,
                             "canonical": canonical.provenance(), "physics": physics.as_dict()}
        A, B = canonical.matrices(physics)
        controller = task_module.tracking_controller(A, B, contract["recipe"])
        capture = Capture(task_module, lambda x, a: canonical.dynamics(x, a, physics), contract["recipe"], OUTPUT, deadline)
        try:
            geometry = task_module.construct_geometry(capture, controller, contract["recipe"])
        except ValueError as exc:
            require(capture.completed == 105 and (OUTPUT / "measurements.json").exists(), "Construction failed before complete diagnostic evidence")
            result.update(status="diagnostic_complete_geometry_rejected", original_constructor_error=str(exc))
        else:
            persist(OUTPUT / "constructor_geometry.json", geometry)
            result["status"] = "diagnostic_complete_geometry_accepted_unexpectedly"
        result["measurements_sha256"] = digest(OUTPUT / "measurements.json")
        result["trace_sha256"] = digest(OUTPUT / "transitions.jsonl")
    except BaseException as exc:
        result.update(status="incomplete_diagnostic", error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    try:
        verify()
        result["prepared_identity_unchanged"] = True
    except BaseException as exc:
        result.update(status="incomplete_diagnostic", prepared_identity_unchanged=False, postcheck_error=str(exc))
    result["counts"] = capture.counts() if capture else {"attempted": 0, "completed": 0, "initializations": 0, "execution_counts_known": True}
    result["diagnostic_persistence_error"] = capture.measurements_persist_error if capture else None
    result.update(elapsed_seconds=time.monotonic() - start, finished_est=datetime.now(EST).isoformat(),
                  prior_measured_seconds=contract["prior_measured_seconds"], cumulative_completed_transitions=105 + result["counts"]["completed"],
                  remaining_reserved_seconds=13470, cumulative_charged_seconds=930, no_automatic_retry=True)
    result.update(cumulative_attempted_transitions=105 + result["counts"]["attempted"],
                  cumulative_measured_seconds=contract["prior_measured_seconds"] + result["elapsed_seconds"],
                  unconsumed_setup_transition_allowance=260800 - 105 - result["counts"]["attempted"],
                  timing_note="Measured Python runtime excludes unmeasured shell overhead; reserved charges remain 60 + 840 + 30 seconds.")
    if result["elapsed_seconds"] > 30:
        result.update(status="incomplete_diagnostic", error="Diagnostic exceeded 30-second allowance")
    try:
        persist(OUTPUT / "RESULT.json", result)
    except BaseException as exc:
        result.update(status="incomplete_diagnostic", result_persistence_error=type(exc).__name__ + ": " + str(exc))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "check", "execute"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        require(not OUTPUT.exists(), "Cannot prepare after a diagnostic attempt")
        persist(CONTRACT, contract_value())
    elif args.check:
        verify()
        print(json.dumps({"status": "prepared_manual_diagnostic_only", "contract_sha256": digest(CONTRACT),
                          "attempt_exists": OUTPUT.exists(), "training_authorized": False}))
    else:
        report = execute()
        print(json.dumps(safe(report), indent=2, allow_nan=False), flush=True)
        raise SystemExit(0 if report["status"].startswith("diagnostic_complete_") else 1)
