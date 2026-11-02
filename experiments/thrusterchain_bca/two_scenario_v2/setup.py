"""Bounded task/data setup only. No learner, CUDA, fitting or training entrypoint."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CORE = REPO / "experiments/thrusterchain_bca/titan_20261026/payload/transfer_bundle/core_v4"
EST = timezone(timedelta(hours=-5))
APPROVAL = ("I approve the two-scenario v2 bounded setup on the ThrusterChain stall3 plant: frozen geometry with connector radius fraction 0.15, one 1,280-parent collection and one exact replay, at most 260,800 plant transitions and 840 seconds; no policy training and no scientific evaluation are authorized")  # the user writes SETUP_APPROVAL.json with these words


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare():
    require(not (HERE / "MANIFEST.json").exists(), "Preserve the frozen preparation")
    pins = {}
    for root in (HERE, CORE):
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md") and "__pycache__" not in p.parts:
                require("setup_attempt_1" not in p.parts, "Cannot seal after a setup attempt")
                pins[p.relative_to(REPO).as_posix()] = sha(p)
    # The unchanged numerical learner is review evidence; it is never imported by setup.
    v = REPO / "experiments/thrusterchain_bca/zeus_20261026_validation"
    payload = CORE.parent.parent
    for relative in ("central_quartile_training_v1_evaluation_rerun1/runtime.py",
                     "central_quartile_training_v1_evaluation_rerun1/state_exact.py"):
        p = v / relative
        pins[p.relative_to(REPO).as_posix()] = sha(p)
    for relative in ("unifloral_td3bc_validation_v1/candidate.py", "unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py"):
        p = payload / relative
        pins[p.relative_to(REPO).as_posix()] = sha(p)
    save(HERE / "MANIFEST.json", {"schema": "two_scenario.setup.v2", "files": pins,
         "allocation": read(HERE / "BUILD_ALLOCATION.json"), "training_authorized": False,
         "geometry_is_unverified": True, "automatic_retry": False})
    # v2: the approval is the user's, written at their terminal (README.md); prepare() writes only the manifest.


def check():
    m = read(HERE / "MANIFEST.json")
    require(m["allocation"] == read(HERE / "BUILD_ALLOCATION.json"), "Allocation changed")
    for relative, expected in m["files"].items():
        p = (REPO / relative).resolve()
        require(p.is_relative_to(REPO) and sha(p) == expected, "Source or contract changed: " + relative)
    a = read(HERE / "SETUP_APPROVAL.json")
    require(a == {"approval": APPROVAL, "manifest_sha256": sha(HERE / "MANIFEST.json"),
                  "setup_only": True, "transition_cap": 260800, "process_cap_seconds": 840,
                  "training_authorized": False}, "Wrong setup approval: SETUP_APPROVAL.json must carry the v2 words and this manifest's hash")
    p = read(HERE / "protocol.json")
    require(p["training_authorized"] is False and p["scientific_evaluation_authorized"] is False,
            "Setup cannot release training or final evaluation")
    return p


class Counter:
    CAPS = {"geometry": 4800, "collection": 128000, "replay": 128000}

    def __init__(self, plant, deadline, progress=None):
        self.plant, self.deadline, self.progress = plant, deadline, progress
        self.phase = "geometry"
        self.attempted = dict.fromkeys(self.CAPS, 0)
        self.completed = dict.fromkeys(self.CAPS, 0)
        self.starts = dict.fromkeys(self.CAPS, 0)

    def begin(self, phase):
        require(phase in self.CAPS, "Unknown counted phase")
        self.phase = phase
        self.deadline()

    def initialize(self, unused=None):
        cap = {"geometry": 48, "collection": 1280, "replay": 0}[self.phase]
        require(self.starts[self.phase] < cap, "State-initialization cap exhausted")
        self.starts[self.phase] += 1
        self.deadline()

    def __call__(self, state, action):
        self.deadline()
        require(getattr(state, "shape", None) == (13,) and getattr(action, "shape", None) == (4,),
                "This counter admits exactly one full-state transition per call")
        require(self.attempted[self.phase] < self.CAPS[self.phase], "Phase transition cap exhausted")
        require(sum(self.attempted.values()) < 260800, "Overall setup transition cap exhausted")
        self.attempted[self.phase] += 1
        value = self.plant(state, action)
        self.completed[self.phase] += 1
        if self.progress and sum(self.completed.values()) % 1000 == 0:
            self.progress(self.receipt())
        return value

    def receipt(self):
        return {"attempted": dict(self.attempted), "completed": dict(self.completed),
                "state_initializations_attempted": dict(self.starts), "gym_reset_calls": 0,
                "execution_counts_known": self.attempted == self.completed,
                "critic_updates": 0, "actor_updates": 0, "auxiliary_updates": 0, "calibrations": 0}


def geometry_evidence(T, env, control, geometry, protocol, counter):
    import numpy as np
    rules, law = protocol["setup_rules"], protocol["collection"]
    rng = np.random.Generator(np.random.PCG64(protocol["geometry_seed"]))
    records = []
    for family in range(rules["witness_families"]):
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
                action = control(x, t)
                if variant == "danger" and 25 <= t < 30:
                    action = -action
                if variant == "wait" and t >= 25:
                    action = np.zeros(4, np.float32)
                nxt, reward, done, info = env.step(x, t, action)
                trace.append({"state": x.tolist(), "time": t, "action": np.asarray(action).tolist(),
                              "reward": reward, "next_state": nxt.tolist(), "terminated": done,
                              "reason": info["terminal_reason"]})
                if 25 <= t < 30 and bool(T.edge_intersects(x, nxt, geometry["connector"])):
                    crossings.append(t)
                rewards.append(reward)
                x = nxt
                if done:
                    break
            case["variants"][variant] = {"terminal_reason": info["terminal_reason"], "terminal_time": info["t"],
                "return": float(sum(.98 ** j * r for j, r in enumerate(rewards))),
                "from_B_return": float(sum(.98 ** j * r for j, r in enumerate(rewards[25:]))) if len(rewards) > 25 else None,
                "connector_crossing_times": crossings, "trace": trace}
        records.append(case)
    nominal = records[0]["variants"]
    safe = sum(c["variants"]["safe"]["terminal_reason"] == "goal" and
               30 < c["variants"]["safe"]["terminal_time"] <= rules["max_safe_goal_time"] for c in records)
    danger = sum(c["variants"]["danger"]["terminal_reason"] == "danger" for c in records)
    bridge = sum(bool(c["variants"]["safe"]["connector_crossing_times"]) for c in records)
    base = nominal["safe"]["from_B_return"]
    gaps = {key: None if base is None or nominal[key]["from_B_return"] is None else
            base - nominal[key]["from_B_return"] for key in ("danger", "wait")}
    # Snapshot/direct parity at an actual live time; both calls are physically counted.
    counter.initialize()
    x, t = np.asarray(geometry["B"], np.float64), 25
    snapshot = env.restore(x, t, terminated=False)
    a = control(x, t)
    direct = env.step(x, t, a)
    restored, reward, info = env.step_snapshot(snapshot, a)
    parity = np.array_equal(restored.state, direct[0]) and reward == direct[1] and restored.terminated == direct[2] and restored.time == 26
    checks = {"nominal_goal_after_C_by75": nominal["safe"]["terminal_reason"] == "goal" and 30 < nominal["safe"]["terminal_time"] <= 75,
              "safe_goal_families": safe >= rules["required_safe_goal_families"],
              "danger_entry_families": danger >= rules["required_danger_entry_families"],
              "connector_traversal_families": bridge >= rules["required_connector_traversal_families"],
              "meaningful_danger_disadvantage": gaps["danger"] is not None and gaps["danger"] >= rules["reference_return_advantage"],
              "meaningful_connector_advantage": gaps["wait"] is not None and gaps["wait"] >= rules["reference_return_advantage"],
              "exact_snapshot_direct_parity": bool(parity)}
    return {"checks": checks, "passed": all(checks.values()), "families": len(records),
            "safe_goal_count": safe, "danger_entry_count": danger, "connector_traversal_count": bridge,
            "nominal_from_B_gaps": gaps, "cases": records,
            "scope": "Fixed-controller task validity; no learned-policy exposure or signed critic error was measured"}


def joined_fragment_parent_ids(arrays, mask):
    """Evidence of genuine retained fragments, not a new transition between them."""
    import numpy as np
    parents = []
    for parent in np.unique(arrays["episode_ids"][arrays["roles"] == 0]):
        ids = np.flatnonzero(arrays["episode_ids"] == parent)
        removed = ids[~mask[ids]]
        times = arrays["times"][removed]
        blocks = np.split(times, np.flatnonzero(np.diff(times) != 1) + 1)
        retained_times = set(int(t) for t in arrays["times"][ids[mask[ids]]])
        for block in blocks:
            if (len(block) and np.any((block >= 25) & (block < 30)) and
                    int(block[0]) - 1 in retained_times and int(block[-1]) + 1 in retained_times and
                    bool(np.all(mask[ids[arrays["times"][ids] < block[0]]])) and
                    bool(np.all(mask[ids[arrays["times"][ids] > block[-1]]])) and
                    bool(mask[ids[-1]]) and
                    bool(arrays["terminals"][ids[-1]]) and float(arrays["rewards"][ids[-1]]) > 0):
                parents.append(int(parent))
                break
    return parents


def support_evidence(D, T, arrays, geometry, protocol):
    import numpy as np
    rules, output = protocol["setup_rules"], {}
    for condition in protocol["conditions"]:
        mask = D.retained_mask(arrays, geometry, condition)
        fit = mask & (arrays["roles"] == D.FIT)
        cal = mask & (arrays["roles"] == D.CALIBRATION)
        region = geometry["danger" if condition == "honeypot" else "connector"]
        forbidden = T.edge_intersects(arrays["states"][mask], arrays["next_states"][mask], region)
        queries = [("B", geometry["B"], 25)]
        queries += [("danger_approach", geometry["excursion"][-2], 29)] if condition == "honeypot" else [("C", geometry["C"], 30)]
        counts = {}
        for name, point, t in queries:
            delta = arrays["states"] - np.asarray(point, np.float64)
            near = fit & (arrays["times"] == t) & (np.linalg.norm(delta[:, :10], axis=1) <= rules["approach_physical_l2_radius"])
            near &= (np.abs(delta[:, 10]) <= rules["approach_current_radius"]) & (np.linalg.norm(delta[:, 11:], axis=1) <= rules["approach_wave_l2_radius"])
            counts[name] = int(len(np.unique(arrays["episode_ids"][near])))
        eligible = int(len(np.unique(arrays["episode_ids"][cal])))
        removed_fit = ~mask & (arrays["roles"] == D.FIT)
        removed_parents = int(len(np.unique(arrays["episode_ids"][removed_fit])))
        fragments = joined_fragment_parent_ids(arrays, mask) if condition == "bottleneck" else None
        output[condition] = {"retained_rows": int(mask.sum()), "fitting_rows": int(fit.sum()),
            "calibration_eligible_parents": eligible, "approach_distinct_fitting_parents": counts,
            "forbidden_retained_edges": int(forbidden.sum()),
            "removed_fitting_rows": int(removed_fit.sum()), "removed_fitting_parents": removed_parents,
            "retained_before_after_connector_goal_parent_ids": fragments,
            "mask_row_ids_sha256": hashlib.sha256(arrays["row_ids"][mask].astype("<i8").tobytes()).hexdigest(),
            "passed": not forbidden.any() and all(n >= rules["minimum_approach_parents"] for n in counts.values()) and
                      eligible >= rules["calibration_eligible_parent_minimum"] and
                      removed_parents >= rules["minimum_removed_parents"] and
                      (fragments is None or len(fragments) >= rules["minimum_fragment_parents"])}
    return output


def execute_setup():
    start = time.monotonic()
    protocol = check()
    require(os.name == "posix", "Manual existing Linux runtime required; no assistant WSL route is used")
    root = HERE / "setup_attempt_1"
    root.mkdir(exist_ok=False)
    save(root / "ATTEMPT.json", {"started_est": datetime.now(EST).isoformat(), "pid": os.getpid(),
         "manifest_sha256": sha(HERE / "MANIFEST.json"), "approval_sha256": sha(HERE / "SETUP_APPROVAL.json"),
         "charged_seconds": 840, "no_automatic_retry": True})
    result = {"status": "failed_or_interrupted_setup", "training_authorized": False, "error": None,
              "manifest_sha256": sha(HERE / "MANIFEST.json"), "execution_counts_known": True}
    counter = None
    def deadline():
        require(time.monotonic() - start < 825, "Setup computation deadline reached")
    def stop(signum, frame):
        raise TimeoutError("Setup deadline or signal " + str(signum))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for s in handlers:
        signal.signal(s, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 825 - (time.monotonic() - start)))
    try:
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "gymnasium")}
        require(versions == {"numpy": "2.5.4", "gymnasium": "1.0.0"}, "Setup numerical runtime differs; no substitution/install")
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(CORE))
        from experiments.thrusterchain_bca.two_scenario_v2 import task as T, data as D
        from thrusterchain import environment as canonical
        require(Path(canonical.__file__).resolve() == CORE / "thrusterchain/environment.py", "Canonical module path collision")
        require(not any(k == "jax" or k.startswith("jax.") for k in sys.modules), "Setup must not initialize the learner/JAX")
        result["runtime"] = {"python": sys.version, "executable": sys.executable, "versions": versions,
                             "canonical": canonical.provenance()}
        physics = canonical.Physics(stall=True, stall_slope=3.)
        result["physics"] = physics.as_dict()
        def progress(counts):
            with (root / "progress.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"saved_est": datetime.now(EST).isoformat(), "counts": counts}) + "\n")
        counter = Counter(lambda x, a: canonical.dynamics(x, a, physics), deadline, progress)
        A, B = canonical.matrices(physics)
        control = T.tracking_controller(A, B, protocol["geometry_recipe"])
        counter.initialize(); counter.initialize()
        geometry = T.construct_geometry(counter, control, protocol["geometry_recipe"])
        save(root / "geometry.json", geometry)
        env = T.Task(counter, geometry)
        evidence = geometry_evidence(T, env, control, geometry, protocol, counter)
        save(root / "geometry_evidence.json", evidence)
        require(evidence["passed"], "Frozen geometry/value/accessibility check failed; collection was not started")
        counter.begin("collection")
        arrays = D.collect(env, control, geometry, protocol["collection"], on_parent_start=counter.initialize)
        metadata = {"manifest_sha256": result["manifest_sha256"], "geometry_sha256": sha(root / "geometry.json"),
                    "protocol_sha256": sha(HERE / "protocol.json"), "physics": physics.as_dict(),
                    "no_oracle_labels_in_learners": True, "training_authorized": False}
        D.save(root / "parent_data.npz", arrays, metadata)
        result["dataset_sha256"] = sha(root / "parent_data.npz")
        counter.begin("replay")
        result["replay"] = D.exact_replay(arrays, env)
        support = support_evidence(D, T, arrays, geometry, protocol)
        save(root / "data_evidence.json", support)
        require(all(v["passed"] for v in support.values()), "Retained support or calibration eligibility failed")
        result["status"] = "setup_passed_pending_geometry_protocol_review"
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    try:
        check()
        result["prepared_identity_unchanged"] = True
    except BaseException as exc:
        result.update(status="failed_or_interrupted_setup", prepared_identity_unchanged=False, postcheck_error=str(exc))
    result["counts"] = counter.receipt() if counter is not None else {
        "attempted": {}, "completed": {}, "state_initializations_attempted": {}, "gym_reset_calls": 0,
        "execution_counts_known": True, "critic_updates": 0, "actor_updates": 0, "auxiliary_updates": 0, "calibrations": 0}
    result["execution_counts_known"] = result["counts"]["execution_counts_known"]
    result.update(elapsed_seconds=time.monotonic() - start, finished_est=datetime.now(EST).isoformat(),
                  remaining_reserved_seconds=13500, remaining_reserved_transitions=237600,
                  no_automatic_retry=True, training_authorized=False)
    if result["elapsed_seconds"] > 840:
        result.update(status="failed_or_interrupted_setup", error="Setup exceeded its840second allowance")
    save(root / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute-setup", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        check()
        print(json.dumps({"status": "prepared_not_physically_validated", "manifest_sha256": sha(HERE / "MANIFEST.json"),
                          "setup_reserved": (HERE / "setup_attempt_1").exists(), "training_authorized": False}))
    else:
        report = execute_setup()
        print(json.dumps({k: report[k] for k in ("status", "error", "counts", "elapsed_seconds", "training_authorized")}, indent=2))
        raise SystemExit(0 if report["status"] == "setup_passed_pending_geometry_protocol_review" else 1)
