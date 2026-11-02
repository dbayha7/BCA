"""Bounded native/BCA comparison worker; scientific imports occur only in --job execution.

This adapter reuses the committed UniFloral training, exact restore and time-aware
evaluation functions. It never edits or imports the concurrent native_screen package.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
VALIDATION = HERE.parents[1]
KINDS = {"preflight", "warmup", "branch", "bca_branch", "evaluate"}


def source_roots():
    return (HERE, VALIDATION / "knee_edge_ood_v3", VALIDATION / "knee_s10_candidate_v1",
            HERE.parent, VALIDATION / "bca_dose_screen_v1")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_integrity(record):
    """Fail closed before imports and after work, including verification exceptions."""
    require(isinstance(record, dict) and isinstance(record.get("files"), dict)
            and bool(record["files"]), "Missing source integrity record")
    require(Path(record["package_dir"]).resolve() == HERE, "Wrong worker package")
    paths = {str(Path(p).resolve()): digest for p, digest in record["files"].items()}
    require(str(Path(__file__).resolve()) in paths, "Worker is not in the source seal")
    changed = [p for p, digest in paths.items() if not Path(p).is_file() or sha(p) != digest]
    for root in source_roots():
        for path in root.iterdir():
            importable = path.is_file() and path.suffix in (".py", ".pyc", ".pth", ".pyd", ".so")
            if importable and str(path.resolve()) not in paths:
                changed.append("unsealed importable: " + str(path))
            if path.is_dir() and (path / "__init__.py").is_file():
                initializer = str((path / "__init__.py").resolve())
                if initializer not in paths:
                    changed.append("unsealed package: " + initializer)
    require(not changed, "Source integrity failure: " + ", ".join(changed))
    return {"verified_files": len(paths), "changed": []}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def force_kappa_three(helper):
    """Override the historical helper's captured default, preserving native kappa."""
    original = helper.patch_runtime
    helper.KAPPA_A = 3.0
    helper.patch_runtime = lambda runtime, kappa=3.0: original(runtime, kappa=3.0)


def configure(job):
    """Import pinned sources and install only this comparison's explicit data/seed bindings."""
    config = job["config"]
    require(job["kind"] in KINDS, "Unknown job kind")
    require(job.get("world", "stall3") == "stall3" and job.get("arm", "cutoff") == "cutoff",
            "This comparison supports only stall3/CUTOFF")
    require(len(config["training"]) == 3, "Exactly three paired training seeds required")
    require(len(config["evaluation_seeds"]) == 3, "Exactly three evaluation seed rows required")
    v3 = VALIDATION / "knee_edge_ood_v3"
    candidate1 = VALIDATION / "knee_s10_candidate_v1"
    dose = VALIDATION / "bca_dose_screen_v1"
    for path in reversed((v3, candidate1, HERE.parent, dose)):
        sys.path.insert(0, str(path))
    require("recorder" not in sys.modules, "A recorder was imported before the widened recorder binding")
    import recorder_wide
    sys.modules["recorder"] = recorder_wide
    # Old workers check their own --job package at import. Our complete seal has
    # already been verified; import them as libraries, never as their old CLI.
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve())]
        import stage1_worker as W1
        import screen_worker as SW
    finally:
        sys.argv = original_argv
    import bindings as B
    import dataset_knee as D
    import numpy as np
    import prefix_rule as PR
    import qpi
    import units
    require(Path(W1.__file__).resolve() == (v3 / "stage1_worker.py").resolve(), "Wrong training worker")
    require(Path(SW.__file__).resolve() == (dose / "screen_worker.py").resolve(), "Wrong BCA worker")
    require(D.recorder is recorder_wide and W1.E.recorder is recorder_wide, "Widened recorder was not bound")
    D._sealed_dataset_entropy = lambda: int(config["dataset_entropy"])

    def data_seeds(dataset, test_seeds):
        require(test_seeds is None and type(dataset) is D.KneeDataset, "Unexpected data or test-seed override")
        require(dataset.metadata["seed"] == config["dataset_entropy"], "Wrong dataset entropy")
        return dict(config["data_seeds"]), hashlib.sha256(
            json.dumps(config["data_seeds"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    B._seeds_for = data_seeds
    B.M.normalization_rows = lambda masks: np.flatnonzero(masks["cutoff"])
    W1.protocol_view = lambda ignored: {"seeds": [dict(row) for row in config["training"]]}
    cache = {}

    def load_world(ignored, world):
        require(world == "stall3", "Unapproved physics world")
        if not cache:
            path = (REPO / config["dataset_relative"]).resolve()
            require(path.is_relative_to(REPO.resolve()), "Dataset path escapes repository")
            require(sha(path) == config["dataset_archive_sha256"], "Dataset archive hash differs")
            dataset = D.load(path)  # Full exact lineage and dynamics validation; counted by SimulatorCounter.
            require(dataset.digest == config["dataset_content_sha256"], "Dataset content hash differs")
            require(dataset.metadata["world"] == world and dataset.metadata["provenance"] == "research",
                    "Wrong dataset world/provenance")
            require(dataset.metadata["recorder_sha256"] == sha(candidate1 / "recorder_wide.py"),
                    "Dataset recorder source differs")
            require(dataset.metadata["behavior"]["name"] == "knee_s10_wide_window_v1", "Wrong behavior recorder")
            plan = B.plan_streams(dataset, world)
            require(plan["plans"]["cutoff"]["receipt"]["protected_transition_rows_bound_to_learner"] == 0,
                    "Protected rows entered a learner stream")
            cache.update(dataset=dataset, plan=plan)
        return cache["dataset"], cache["plan"]

    W1.load_world = load_world
    # The historical helper's default argument captured 63 at definition time:
    # updating its constant alone would silently retain the high dose.
    force_kappa_three(SW)
    W1.step03.set_worker_environment()
    return W1, SW, B, D, np, PR, qpi, units


def evaluate(job, out, deadline, modules):
    W1, SW, B, D, np, PR, qpi, units = modules
    config, i = job["config"], int(job["seed_index"])
    runtime, evaluator, _, _ = W1.step03.rerun1()
    dataset, plan = W1.load_world(job, "stall3")
    seeds = config["evaluation_seeds"][i]
    # Existing sampling functions are used with explicitly new independent
    # streams; their historical 'smoke' marker describes that API, not our data.
    starts = W1.E.bank_states("stage1_e2_bank", int(seeds["returns"]))
    require(starts.shape == (128, 13), "Return bank must have 128 starts")
    return_ids = [f"seed{i}:episode{k}:" + hashlib.sha256(
        np.ascontiguousarray(row, dtype="<f8").tobytes()).hexdigest() for k, row in enumerate(starts)]
    anchors = [PR.build_anchor("stage1_e2_anchors", j, int(seeds["anchors"])) for j in range(48)]
    states = np.stack([a["state"] for a in anchors]).astype(np.float64)
    obs = np.stack([a["obs"] for a in anchors]).astype(np.float32)
    times = np.asarray([a["t"] for a in anchors], np.int64)
    require(np.array_equal(times, np.repeat([0, 25, 50], 16)), "Wrong anchor time strata")
    slots = ("edge_knee", "s10", "edge_700")
    common = W1.E.P.build(48, slots).astype(np.float32)
    budget = W1.Budget(int(job["transitions"]), float(job["timeout_seconds"]))
    result = {"kind": "evaluate", "seed_index": i, "world": "stall3", "data_arm": "cutoff",
              "evaluation_seeds": seeds, "sampling_status": "new exploratory evaluation streams",
              "reference": "forced full action then the same learned policy, without knee clipping",
              "gamma": 0.98, "horizon": 100, "anchors": {"states": states, "observations": obs,
              "times": times, "common_actions": common, "common_slots": list(slots),
              "ids": [{k: a[k] for k in ("bank", "item", "reset_seed", "t", "current")} for a in anchors],
              "prefix_actions": [a["prefix_actions"] for a in anchors]}, "starts": starts, "runs": {}}
    for method in ("native", "bca"):
        deadline()
        run_dirs = {"warmup": job["warmup_dir"],
                    "branch": job["native_branch_dir"] if method == "native" else job["bca_branch_dir"]}
        loader = W1.policies if method == "native" else SW.policies_bca
        _, policies, assertions, _ = loader(job, runtime, evaluator, dataset, plan, "cutoff", run_dirs, ("020000",))
        policy = policies["020000"]
        roll = W1.E.rollouts(starts, policy.act, "stall3", "pi", budget)
        own_action = policy.act(obs)
        actions = np.concatenate([common, own_action[:, None, :]], axis=1).astype(np.float32)
        qpi_values, calls, receipt = qpi.raw_values_from_states(states, times, actions, policy.act,
                                                              "stall3", clip_all=False)
        q1, q2, qmin = W1.E.critic(policy, obs, actions)
        width = np.asarray(policy.width(np.repeat(obs, 4, axis=0), actions.reshape(-1, 4)), np.float64).reshape(48, 4)
        require(np.isfinite(qpi_values).all() and np.isfinite(width).all(), "Nonfinite Q reference/width")
        policy.assert_unchanged()
        result["runs"][method] = {"returns": roll["returns"], "return_ids": return_ids,
            "return_receipt": roll["receipt"],
            "free_actions": roll["executed"], "free_observations": roll["obs"],
            "exposure": W1.E.step_rates(roll["executed"], roll["obs"]),
            "actions": actions, "slots": list(slots) + ["own_actor"], "q1": q1, "q2": q2,
            "twin_min": qmin, "qpi": qpi_values, "width": width,
            "penalized_q1_kappa3": q1 - 3.0 * width, "qpi_receipt": receipt,
            "qpi_actor_calls": calls, "assertions": assertions,
            "model_unchanged": True, "branch_report_sha256": sha(Path(run_dirs["branch"]) / "report.json")}
    return result


def run_job(job_path):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out = job_path.parent
    started = time.monotonic()
    counter = None
    report = {"job": job.get("id"), "kind": job.get("kind"), "status": "failed", "unit_label": "INCOMPLETE",
              "started_utc": datetime.now(timezone.utc).isoformat(), "worker_sha256": sha(__file__),
              "job_sha256": sha(job_path), "config_sha256": hashlib.sha256(json.dumps(
                  job.get("config"), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "kappa": 3.0, "no_automatic_retry": True}
    try:
        report["integrity_before"] = verify_integrity(job.get("integrity"))
        modules = configure(job)
        W1, SW, B, D, np, PR, qpi, units = modules
        require(type(job.get("transitions")) is int and job["transitions"] >= 0, "Missing exact simulator cap")
        counter = units.SimulatorCounter(job["transitions"])
        deadline = W1.Deadline(job["timeout_seconds"])
        with counter:
            kind = job["kind"]
            if kind == "evaluate":
                value = evaluate(job, out, deadline, modules)
            elif kind == "bca_branch":
                value = SW.bca_branch(job, out, deadline)
            else:
                value = {"preflight": W1.preflight, "warmup": W1.warmup, "branch": W1.branch}[kind](job, out, deadline)
            deadline()
        report.update(value, status="completed", unit_label="COMPLETE")
        if job["kind"] == "bca_branch":
            report["dose_provenance"] = "Historical dose-screen helper reused; effective kappa explicitly fixed to 3, not 63"
    except BaseException as exc:
        report.update(status="failed", unit_label="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report["interrupted"] = type(exc).__name__
    try:
        report["integrity_after"] = verify_integrity(job.get("integrity"))
    except BaseException as exc:
        report["integrity_after"] = {"error": f"{type(exc).__name__}: {exc}"}
        report.update(status="failed", unit_label="INCOMPLETE", postcheck_error=f"{type(exc).__name__}: {exc}")
    report.update(transitions=counter.steps if counter is not None else None,
                  simulator_steps_exact=counter.steps if counter is not None else None,
                  simulator_steps_attempted=counter.attempted if counter is not None else None,
                  execution_counts_known=counter is not None,
                  elapsed_seconds=time.monotonic() - started,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    write_new(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    result = run_job(args.job)
    print(json.dumps({k: result.get(k) for k in ("status", "job", "kind", "transitions", "error", "postcheck_error")}))
    raise SystemExit(0 if result.get("status") == "completed" else 1)
