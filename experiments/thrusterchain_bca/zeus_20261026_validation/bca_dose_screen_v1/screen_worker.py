"""BCA dose screen v1: GPU jobs, one per process (PROTOCOL.md). User-run in the recorded WSL GPU environment; agents
never run this file and never open its checkpoints.

It reuses the knee_edge_ood_v3 Stage 1 worker (imported by path, unchanged on disk) with four declared changes that
live in this process only:
1. rerun1 runtime.make_adapter is wrapped so the "bca" arm gets kappa_A (DOSE.json) instead of the frozen 3.0;
2. `bca_branch` is the Stage 1 branch job with arm "bca" in place of "native" (same fork, same sealed branch seeds);
3. the evaluation restores branch checkpoints with the "bca" arm (policies_bca);
4. stage1_eval.anchor_values also records U at the anchors' probe slots.
Integrity: this worker verifies its own job's record (this package plus the v3 dependency closure) before any package
import and again after the work, and only then imports the v3 worker, from which it hides --job so that the v3
worker's own check, which would name the v3 folder, does not run twice.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "knee_edge_ood_v3"
for _p in (str(V3), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import integrity                                             # noqa: E402  (v3 integrity.py, standard library only)
integrity.verify(sys.argv, __file__)                         # every recorded file, before any package import


def v3_added_files(record):
    """The v3 folder comes first on sys.path, so an importable file added there could shadow an import: scan it too."""
    return [p for p in integrity.changed_files({**record, "package_dir": str(V3)}) if p.startswith("added: ")]


if "--job" in sys.argv:
    _job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text(encoding="utf-8"))
    _added = v3_added_files(_job["integrity"])
    if _added:
        raise SystemExit("importable files added to the v3 folder: " + ", ".join(_added))

_argv, sys.argv = sys.argv, [a for a in sys.argv if a != "--job"]
try:
    import numpy as np                                       # noqa: E402
    import stage1_eval as E                                  # noqa: E402
    import stage1_worker as W1                               # noqa: E402
    import probes as P                                       # noqa: E402
    import step03                                            # noqa: E402
    import bindings as B                                     # noqa: E402
    from core import require, sha_file                       # noqa: E402
finally:
    sys.argv = _argv

DOSE = json.loads((HERE / "DOSE.json").read_text(encoding="utf-8"))
KAPPA_A = float(DOSE["kappa_A"])


def patch_runtime(runtime, kappa=KAPPA_A):
    """Change 1: the bca arm's kappa. Idempotent; returns the runtime."""
    if getattr(runtime, "_dose_screen_kappa", None) == kappa:
        return runtime
    original = getattr(runtime, "_dose_screen_original_make_adapter", runtime.make_adapter)

    def make_adapter(context, state, arm, k, *rest, **kw):
        return original(context, state, arm, kappa if arm == "bca" else k, *rest, **kw)
    runtime._dose_screen_original_make_adapter = original
    runtime.make_adapter = make_adapter
    runtime._dose_screen_kappa = kappa
    return runtime


def rerun1():
    runtime, evaluate_worker, gradient_audit, rerun_inputs = step03.rerun1()
    return patch_runtime(runtime), evaluate_worker, gradient_audit, rerun_inputs


def bca_branch(job, out, deadline):
    """Change 2: W1.branch with arm "bca" at kappa_A; the paired native branch's minibatch hash is compared."""
    W1.smoke_options(job)
    runtime, _, gradient_audit, _ = rerun1()
    dataset, plan = W1.load_world(job, job["world"])
    adapter, state, warm, assertions, expected, probe, _ = W1.restore_warm(runtime, job, dataset, plan, job["warmup_dir"])
    adapter, state = runtime.branch(adapter, state, W1.protocol_view(job), job["seed_index"], "bca",
                                    warm["matching"]["distance_coefficient"])
    require(adapter.arm == "bca" and adapter.kappa == KAPPA_A, f"Branch kappa {adapter.kappa} is not kappa_A {KAPPA_A}")
    branch_assertions = B.assert_adapter(adapter, expected)
    audits, checkpoints = {}, {}

    def on_step(current, metrics):
        if int(metrics["counts"]["critic"]) == 15_000:
            audits["15000"] = W1.run_audit(gradient_audit, runtime, adapter, current, probe, out, 15_000)
            checkpoints["015000"] = W1.save_state(runtime, out, "015000", current)

    state, metrics, minibatches = W1.train(runtime, adapter, state, W1.BRANCH_UPDATES, out, deadline, job, on_step)
    final = runtime.validate(adapter, state)
    require(final["counts"]["critic"] == 20_000 and final["counts"]["refresh_accepted"] == 2, "Branch counts differ")
    require(set(audits) == {"15000"}, "The 15k audit and checkpoint were not taken")
    audits["20000"] = W1.run_audit(gradient_audit, runtime, adapter, state, probe, out, 20_000)
    checkpoints["020000"] = W1.save_state(runtime, out, "020000", state)
    native = json.loads((Path(job["native_branch_dir"]) / "report.json").read_text(encoding="utf-8"))
    paired = {"binding_equal": expected == native["binding"],
              "warmup_report_equal": sha_file(Path(job["warmup_dir"]) / "report.json") == native["warmup_report_sha256"],
              "minibatch_stream_equal": minibatches["phase0_minibatch_sha256"] == native["minibatches"]["phase0_minibatch_sha256"]}
    require(all(paired.values()), f"The BCA branch is not paired with its native branch: {paired}")
    return {"kind": "bca_branch", "kappa": adapter.kappa, "binding": expected, "assertions": assertions,
            "branch_assertions": branch_assertions,
            "warmup_report_sha256": sha_file(Path(job["warmup_dir"]) / "report.json"),
            "native_branch_report_sha256": sha_file(Path(job["native_branch_dir"]) / "report.json"),
            "distance_coefficient": warm["matching"]["distance_coefficient"],
            "counts": runtime.host(final["counts"]), "checks": runtime.host(final["checks"]),
            "initial_parameter_hashes": warm["initial_parameter_hashes"], "normalization": warm["normalization"],
            "gradient_audits": audits, "checkpoints": checkpoints, "minibatches": minibatches,
            "pairing": paired, "dose_sha256": sha_file(HERE / "DOSE.json"),
            "last_metrics": runtime.host(metrics)}


def policies_bca(job, runtime, evaluate_worker, dataset, plan, arm, run_dirs, checkpoints):
    """Change 3: W1.policies with the branch template built for the "bca" arm at kappa_A, after checking that the
    branch directory holds a completed kappa_A BCA branch made under this DOSE.json."""
    report = json.loads((Path(run_dirs["branch"]) / "report.json").read_text(encoding="utf-8"))
    require(report.get("kind") == "bca_branch" and report.get("status") == "completed" and report.get("kappa") == KAPPA_A
            and report.get("dose_sha256") == sha_file(HERE / "DOSE.json"), "Evaluation branch is not a completed kappa_A BCA branch")
    patch_runtime(runtime)
    original_branch = runtime.branch

    def branch(adapter, state, protocol, seed_index, _arm, coefficient):
        return original_branch(adapter, state, protocol, seed_index, "bca", coefficient)
    runtime.branch = branch
    try:
        return _ORIGINAL_POLICIES(job, runtime, evaluate_worker, dataset, plan, arm, run_dirs, checkpoints)
    finally:
        runtime.branch = original_branch


def anchor_values_with_u(policy, anchors, world, budget, *, stall_trained):
    """Change 4: stage1_eval.anchor_values plus U at the same gated probes."""
    out = _ORIGINAL_ANCHOR_VALUES(policy, anchors, world, budget, stall_trained=stall_trained)
    obs = np.stack([np.asarray(a["obs"], np.float32) for a in anchors])
    gated = P.build(len(anchors), E.GATED_SLOTS)
    n, slots = gated.shape[:2]
    rows = np.repeat(obs, slots, axis=0)
    out["U"] = np.asarray(policy.width(rows, gated.reshape(-1, 4)), np.float64).reshape(n, slots)
    return out


_ORIGINAL_POLICIES = W1.policies
_ORIGINAL_ANCHOR_VALUES = E.anchor_values


def evaluate(job, out, deadline):
    W1.policies, E.anchor_values = policies_bca, anchor_values_with_u
    try:
        return W1.evaluate(job, out, deadline)
    finally:
        W1.policies, E.anchor_values = _ORIGINAL_POLICIES, _ORIGINAL_ANCHOR_VALUES


KINDS = {"preflight": W1.preflight, "bca_branch": bca_branch, "evaluate": evaluate}


def run_job(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job_path).resolve().parent
    started = time.monotonic()
    counter = None
    report = {"job": job["id"], "kind": job["kind"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "screen": "bca_dose_screen_v1", "kappa_A": KAPPA_A, "dose_sha256": sha_file(HERE / "DOSE.json")}
    try:
        require(job["kind"] in KINDS, "Unknown job kind")
        step03.set_worker_environment()
        deadline = W1.Deadline(job["timeout_seconds"])
        import units
        limit = int(job.get("transitions") or 0) or None
        counter = units.SimulatorCounter(limit)
        with counter:
            report.update(KINDS[job["kind"]](job, out, deadline))
        if limit is not None:
            report["transitions_prospective"] = report.get("transitions")
            report["transitions"] = counter.steps
        report.update(status="completed", unit_label="COMPLETE")
    except BaseException as exc:
        report.update(status="failed", unit_label="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report["interrupted"] = type(exc).__name__
    W1._close_report(report, counter, job)
    added = v3_added_files(job.get("integrity") or {"files": {}})
    if added:
        report["integrity_after"]["v3_added"] = added
        if report.get("status") == "completed":
            report.update(status="failed", unit_label="INCOMPLETE", error="importable files added to the v3 folder")
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    step03._write_new(out / "report.json", W1.sanitize(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True)
    result = run_job(parser.parse_args().job)
    sys.exit(0 if result.get("status") == "completed" else 1)
