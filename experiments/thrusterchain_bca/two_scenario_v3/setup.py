"""Bounded development amendment: cached geometry, physical validation and data only."""
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
REPO = HERE.parents[2]
V1 = HERE.with_name("two_scenario_v1")
DIAGNOSTIC = HERE.with_name("two_scenario_geometry_diagnostic_v1_attempt_1")
EST = timezone(timedelta(hours=-5))
APPROVAL = "Sentinel_bc18442c3ac88191aaeb71b51a717761"
MEASUREMENTS_HASH = "410f4c0cf3075732fea04036841c0e0bc5be5e715129d1065c5422d45f73c3c1"
TRACE_HASH = "bda39267a277bd195a0707651a6d60b861755ee2c80ca4dd52137140cf122292"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_safe(value):
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def parent_check():
    original = load(V1 / "setup.py", "two_scenario_v3_original_setup")
    original.check()
    diagnostic = load(HERE.parent / "two_scenario_geometry_diagnostic_v1.py", "two_scenario_v3_diagnostic")
    diagnostic.verify()
    require(sha(DIAGNOSTIC / "measurements.json") == MEASUREMENTS_HASH and
            sha(DIAGNOSTIC / "transitions.jsonl") == TRACE_HASH, "Saved construction evidence changed")
    r = read(DIAGNOSTIC / "RESULT.json")
    require(r["status"] == "diagnostic_complete_geometry_rejected" and r["prepared_identity_unchanged"] and
            r["counts"]["execution_counts_known"] and r["cumulative_completed_transitions"] == 210,
            "Prior execution accounting is not the reviewed terminal evidence")
    return original


def no_overlap():
    # This is a local receipt check, not a distributed lock across machines/tasks.
    require(not (HERE.with_name("two_scenario_v2") / "setup_attempt_1").exists(),
            "Concurrent alternative v2 has a reservation; reconcile work/budget before v3")


def prepare():
    require(not (HERE / "MANIFEST.json").exists(), "Preserve an existing v3 seal")
    require(not (HERE / "setup_attempt_1").exists(), "Cannot prepare an executed attempt")
    parent_check(); no_overlap()
    pins = dict(read(V1 / "MANIFEST.json")["files"])
    paths = [V1 / "MANIFEST.json", V1 / "SETUP_APPROVAL.json", V1 / "setup_attempt_1/RESULT.json",
             V1 / "setup_attempt_1/ATTEMPT.json", HERE.parent / "two_scenario_geometry_diagnostic_v1.py",
             HERE.parent / "test_two_scenario_geometry_diagnostic_v1.py",
             HERE.parent / "two_scenario_geometry_diagnostic_v1.json"]
    paths += [DIAGNOSTIC / name for name in ("RESULT.json", "ATTEMPT.json", "measurements.json", "transitions.jsonl")]
    paths += [p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md")]
    for p in paths:
        pins[p.relative_to(REPO).as_posix()] = sha(p)
    save(HERE / "MANIFEST.json", {"schema": "two_scenario.v3.setup", "files": pins,
         "allocation": read(HERE / "BUILD_ALLOCATION.json"), "development_informed": True,
         "training_authorized": False, "physical_setup_verified": False, "automatic_retry": False})
    save(HERE / "SETUP_APPROVAL.json", approval_value())


def approval_value():
    return {"user_message": APPROVAL, "reaffirmation_message": "Sentinel_95d79ed6189c81918a8f398b2ef02a28",
            "manifest_sha256": sha(HERE / "MANIFEST.json"), "setup_only": True,
            "development_amendment": "Fixed0.01 connector width and first prescribed opposing-action hazard; review before training",
            "transition_cap": 260502, "manual_process_cap_seconds": 840,
            "phase_charge_seconds": 870, "prior_completed_transitions": 210,
            "prior_charged_seconds": 930, "training_authorized": False}


def check():
    m = read(HERE / "MANIFEST.json")
    for relative, expected in m["files"].items():
        p = (REPO / relative).resolve()
        require(p.is_relative_to(REPO) and sha(p) == expected, "Changed source/evidence: " + relative)
    require(m["allocation"] == read(HERE / "BUILD_ALLOCATION.json"), "Allocation changed")
    require(read(HERE / "SETUP_APPROVAL.json") == approval_value(), "Wrong amendment approval")
    p = read(HERE / "protocol.json")
    require(not p["training_authorized"] and not p["scientific_evaluation_authorized"], "Setup cannot release training")
    parent_check()
    return p


class Counter:
    CAPS = {"geometry": 4502, "collection": 128000, "replay": 128000}
    START_CAPS = {"geometry": 46, "collection": 1280, "replay": 0}

    def __init__(self, plant, deadline, progress=None):
        self.plant, self.deadline, self.progress, self.phase = plant, deadline, progress, "geometry"
        self.attempted = dict.fromkeys(self.CAPS, 0)
        self.completed = dict.fromkeys(self.CAPS, 0)
        self.starts = dict.fromkeys(self.CAPS, 0)

    def begin(self, phase):
        require(phase in self.CAPS, "Unknown phase")
        self.phase = phase
        self.deadline()

    def initialize(self, unused=None):
        require(self.starts[self.phase] < self.START_CAPS[self.phase], "Initialization cap exhausted")
        self.starts[self.phase] += 1
        self.deadline()

    def __call__(self, state, action):
        self.deadline()
        require(getattr(state, "shape", None) == (13,) and getattr(action, "shape", None) == (4,), "One full transition required")
        require(self.attempted[self.phase] < self.CAPS[self.phase] and sum(self.attempted.values()) < 260502, "Transition cap exhausted")
        self.attempted[self.phase] += 1
        out = self.plant(state, action)
        self.completed[self.phase] += 1
        if self.progress and sum(self.completed.values()) % 1000 == 0:
            self.progress(self.receipt())
        return out

    def receipt(self):
        return {"attempted": dict(self.attempted), "completed": dict(self.completed),
                "state_initializations_attempted": dict(self.starts), "gym_reset_calls": 0,
                "execution_counts_known": self.attempted == self.completed,
                "critic_updates": 0, "actor_updates": 0, "auxiliary_updates": 0, "calibrations": 0}


def execute_setup():
    start = time.monotonic()
    require(os.name == "posix", "Manual existing Linux runtime required; assistant WSL remains denied")
    protocol = check(); no_overlap()
    root = HERE / "setup_attempt_1"
    root.mkdir(exist_ok=False)
    save(root / "ATTEMPT.json", {"started_est": datetime.now(EST).isoformat(), "pid": os.getpid(),
         "manifest_sha256": sha(HERE / "MANIFEST.json"), "approval_sha256": sha(HERE / "SETUP_APPROVAL.json"),
         "prior_charged_seconds": 930, "new_phase_charge_seconds": 870, "no_automatic_retry": True})
    result = {"status": "failed_or_interrupted_setup", "error": None, "training_authorized": False,
              "manifest_sha256": sha(HERE / "MANIFEST.json"), "development_informed": True}
    counter = None
    def deadline():
        if time.monotonic() - start >= 825:
            raise TimeoutError("Setup internal deadline")
    def stop(signum, frame):
        raise TimeoutError("Setup signal " + str(signum))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    for s in handlers:
        signal.signal(s, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 825 - (time.monotonic() - start)))
    try:
        versions = {n: importlib.metadata.version(n) for n in ("numpy", "gymnasium")}
        require(versions == {"numpy": "2.5.4", "gymnasium": "1.0.0"}, "Frozen numerical runtime mismatch")
        original = parent_check()
        sys.path.insert(0, str(REPO)); sys.path.insert(0, str(original.CORE))
        from experiments.thrusterchain_bca.two_scenario_v3 import amendment as A, evidence as E
        from thrusterchain import environment as canonical
        require(Path(canonical.__file__).resolve() == original.CORE / "thrusterchain/environment.py", "Canonical path collision")
        require(not any(k == "jax" or k.startswith("jax.") for k in sys.modules), "Setup must not import learner/JAX")
        construction = A.geometry_from_saved(read(DIAGNOSTIC / "measurements.json"),
                       [json.loads(line) for line in (DIAGNOSTIC / "transitions.jsonl").read_text().splitlines()], protocol["geometry_recipe"])
        save(root / "geometry_diagnostics.json", construction)
        require(all(v is True for v in construction["checks"].values()), "Amended geometry separation failed; no search or simulator work")
        geometry = construction["geometry"]
        save(root / "geometry.json", geometry)
        physics = canonical.Physics(stall=True, stall_slope=3.)
        result["runtime"] = {"executable": sys.executable, "python": sys.version, "versions": versions,
                             "canonical": canonical.provenance(), "physics": physics.as_dict()}
        def progress(counts):
            with (root / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"saved_est": datetime.now(EST).isoformat(), "counts": counts}) + "\n")
        counter = Counter(lambda x, a: canonical.dynamics(x, a, physics), deadline, progress)
        matrices = canonical.matrices(physics)
        control = A.T.tracking_controller(*matrices, protocol["geometry_recipe"])
        env = A.T.Task(counter, geometry)
        evidence = E.geometry_evidence(A.T, env, control, geometry, protocol, counter)
        save(root / "geometry_evidence.json", evidence)
        require(evidence["passed"], "Fixed geometry/value/accessibility gate failed; collection not started")
        counter.begin("collection")
        arrays = A.collect(env, control, geometry, protocol["collection"], on_parent_start=counter.initialize)
        metadata = {"manifest_sha256": result["manifest_sha256"], "geometry_sha256": sha(root / "geometry.json"),
                    "protocol_sha256": sha(HERE / "protocol.json"), "physics": physics.as_dict(),
                    "development_informed": True, "no_oracle_labels_in_learners": True, "training_authorized": False}
        A.D.save(root / "parent_data.npz", arrays, metadata)
        result["dataset_sha256"] = sha(root / "parent_data.npz")
        counter.begin("replay")
        result["replay"] = A.D.exact_replay(arrays, env)
        save(root / "replay.json", result["replay"])
        support = E.support_evidence(A.D, A.T, arrays, geometry, protocol)
        save(root / "data_evidence.json", support)
        require(all(v["passed"] for v in support.values()), "Retained support/exclusion/fragment/calibration gate failed")
        result["status"] = "setup_passed_pending_geometry_protocol_review"
    except BaseException as exc:
        result.update(status="failed_or_interrupted_setup", error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    try:
        check()
        result["prepared_identity_unchanged"] = True
    except BaseException as exc:
        result.update(status="failed_or_interrupted_setup", prepared_identity_unchanged=False, postcheck_error=str(exc))
    result["counts"] = counter.receipt() if counter else Counter(None, lambda: None).receipt()
    result.update(elapsed_seconds=time.monotonic() - start, finished_est=datetime.now(EST).isoformat(),
                  prior_measured_seconds=6.074587407000308, cumulative_charged_seconds=1800,
                  remaining_reserved_seconds=12600, later_evaluation_reserved_transitions=237600,
                  prior_geometry_initializations=4, no_automatic_retry=True)
    result["cumulative_attempted_transitions"] = 210 + sum(result["counts"]["attempted"].values())
    result["cumulative_completed_transitions"] = 210 + sum(result["counts"]["completed"].values())
    result["unconsumed_setup_transition_allowance"] = 260800 - result["cumulative_attempted_transitions"]
    result["cumulative_measured_seconds"] = result["prior_measured_seconds"] + result["elapsed_seconds"]
    result["timing_note"] = "Measured Python runtimes exclude shell overhead; conservative phase charges are preserved."
    if result["elapsed_seconds"] > 840:
        result.update(status="failed_or_interrupted_setup", error="Exceeded840-second manual allowance")
    try:
        save(root / "RESULT.json", result)
    except BaseException as exc:
        result.update(status="failed_or_interrupted_setup", result_persistence_error=str(exc))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "check", "execute-setup"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        check(); no_overlap()
        print(json.dumps({"status": "prepared_not_physically_validated", "manifest_sha256": sha(HERE / "MANIFEST.json"),
                          "attempt_exists": (HERE / "setup_attempt_1").exists(), "training_authorized": False}))
    else:
        report = execute_setup()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        raise SystemExit(0 if report["status"] == "setup_passed_pending_geometry_protocol_review" else 1)
