"""Native screen for candidate knee_s10_v2 under r5: GPU and CPU jobs, one per process (SCREEN_RULE_r5.md). User-run
in the recorded WSL environment; agents never run this file and never open its checkpoints.

It reuses knee_edge_ood_v3's Stage 1 worker (unchanged on disk) with the r4 package's declared in-process changes
(`../native_screen/screen_bind.py`: recorder, dataset entropy, data-plan seeds, training seeds, E2 bank, E2 anchors,
the BCA dose, and, for research jobs only, the amendment A1 dataset loader), plus four job kinds of its own:
`bca_branch` and `evaluate_bca` (the dose screen's, at kappa = KAPPA), `i0` (the stage-0 instrument unit) and `g1`
(the reported calibration unit of REVISION_r5.md section 5). Smoke jobs (the dry run) read their own test collection
through the unpatched loader, which regenerates data collected in the same environment.

Binding 8 (amendment A2): for GPU jobs the recorded device kind (knee_edge_ood_v3 frozen_checks.RECORDED_DEVICE_KIND,
read by the preflight and by every training bind) becomes the one allowed kind this machine has
(`device_kinds.ALLOWED_DEVICE_KINDS`: Zeus's laptop GPU or Titan's desktop GPU), and every GPU report records it.
Binding 9 (amendment A3): pi_ref's LQR gains are loaded from the stored artifact `pi_ref_gains.npy` (bytes pinned
by pi_ref.GAINS_SHA256) instead of being recomputed, which another CPU does one last bit off.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = HERE.parent
NATIVE = CAND / "native_screen"
V3 = CAND.parent / "knee_edge_ood_v3"
F2 = CAND / "formal_g2"
for _p in (str(V3), str(F2), str(NATIVE), str(CAND / "g1"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import integrity                                             # noqa: E402  (v3 integrity.py, standard library only)
integrity.verify(sys.argv, __file__)


def v3_added_files(record):
    return [p for p in integrity.changed_files({**record, "package_dir": str(V3)}) if p.startswith("added: ")]


_kind, _provenance = None, "research"
if "--job" in sys.argv:
    _job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text(encoding="utf-8"))
    _kind, _provenance = _job.get("kind"), _job.get("provenance", "research")
    _added = v3_added_files(_job["integrity"])
    if _added:
        raise SystemExit("importable files added to the v3 folder: " + ", ".join(_added))
if _kind in ("g1", "i0"):
    os.environ["JAX_PLATFORMS"] = "cpu"                      # the CPU units, before any JAX import

_argv, sys.argv = sys.argv, [a for a in sys.argv if a != "--job"]
try:
    import numpy as np                                       # noqa: E402
    import screen_bind as SB                                 # noqa: E402  first: substitutes the recorder
    import stage1_eval as E                                  # noqa: E402
    import stage1_worker as W1                               # noqa: E402
    import probes as P                                       # noqa: E402
    import step03                                            # noqa: E402
    import bindings as B                                     # noqa: E402
    import device_kinds as DK                                # noqa: E402
    import pi_ref                                            # noqa: E402
    from core import require, sha_file                       # noqa: E402
finally:
    sys.argv = _argv


def bind_pi_ref_gains():
    """Binding 9 (amendment A3): pi_ref's LQR gains are the stored artifact `pi_ref_gains.npy`, whose bytes hash to
    pi_ref.GAINS_SHA256, the pin v1 computed on Zeus. pi_ref recomputes the Riccati recursion at first use and
    refuses on a hash mismatch; another CPU reproduces it one last bit off, so the sealed pin is kept by loading the
    pinned bytes instead of recomputing them."""
    import hashlib
    k = np.ascontiguousarray(np.load(HERE / "pi_ref_gains.npy", allow_pickle=False))
    require(k.shape == (pi_ref.HORIZON, 4, 13) and k.dtype == np.float64, "Stored pi_ref gains have the wrong shape or dtype")
    require(hashlib.sha256(k.tobytes()).hexdigest() == pi_ref.GAINS_SHA256, "Stored pi_ref gains do not hash to the pinned GAINS_SHA256")
    pi_ref._K = k
    return pi_ref.GAINS_SHA256


PI_REF_GAINS_SHA256 = bind_pi_ref_gains()


def bind_device_kind():
    """Binding 8 (amendment A2): the recorded device kind is the one allowed kind this machine has. Called after the
    worker environment is set and before the preflight or any training bind reads the recorded kind."""
    import jax
    kinds = sorted({d.device_kind for d in jax.devices()})
    require(len(kinds) == 1 and kinds[0] in DK.ALLOWED_DEVICE_KINDS,
            f"Device kinds {kinds} are not exactly one allowed kind {sorted(DK.ALLOWED_DEVICE_KINDS)}")
    step03.K.RECORDED_DEVICE_KIND = kinds[0]
    return kinds[0]

KAPPA = SB.KAPPA
SB.patch_dataset_entropy()
SB.patch_bindings()
SB.patch_eval()
if _provenance == "research":
    SB.patch_dataset_loader()                                # change 7: research data by digest, never regenerated
_ORIGINAL_ANCHORS, _ORIGINAL_POLICIES, _ORIGINAL_ANCHOR_VALUES = W1.anchors, W1.policies, E.anchor_values
W1.protocol_view = SB.protocol_view
W1.anchors = lambda smoke_seed=None: SB.anchors(_ORIGINAL_ANCHORS, smoke_seed)


def rerun1():
    runtime, evaluate_worker, gradient_audit, rerun_inputs = step03.rerun1()
    return SB.patch_runtime(runtime), evaluate_worker, gradient_audit, rerun_inputs


def bca_branch(job, out, deadline):
    """The Stage 1 branch job with arm "bca" at KAPPA from the same 10k fork, paired with its native branch."""
    W1.smoke_options(job)
    runtime, _, gradient_audit, _ = rerun1()
    dataset, plan = W1.load_world(job, job["world"])
    adapter, state, warm, assertions, expected, probe, _ = W1.restore_warm(runtime, job, dataset, plan, job["warmup_dir"])
    adapter, state = runtime.branch(adapter, state, W1.protocol_view(job), job["seed_index"], "bca",
                                    warm["matching"]["distance_coefficient"])
    require(adapter.arm == "bca" and adapter.kappa == KAPPA, f"Branch kappa {adapter.kappa} is not {KAPPA}")
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
            "gradient_audits": audits, "checkpoints": checkpoints, "minibatches": minibatches, "pairing": paired,
            "seeds_sha256": sha_file(SB.SS.SEEDS_PATH), "last_metrics": runtime.host(metrics)}


def policies_bca(job, runtime, evaluate_worker, dataset, plan, arm, run_dirs, checkpoints):
    report = json.loads((Path(run_dirs["branch"]) / "report.json").read_text(encoding="utf-8"))
    require(report.get("kind") == "bca_branch" and report.get("status") == "completed" and report.get("kappa") == KAPPA
            and report.get("seeds_sha256") == sha_file(SB.SS.SEEDS_PATH), "Evaluation branch is not a completed BCA branch of this screen")
    SB.patch_runtime(runtime)
    original_branch = runtime.branch

    def branch(adapter, state, protocol, seed_index, _arm, coefficient):
        return original_branch(adapter, state, protocol, seed_index, "bca", coefficient)
    runtime.branch = branch
    try:
        return _ORIGINAL_POLICIES(job, runtime, evaluate_worker, dataset, plan, arm, run_dirs, checkpoints)
    finally:
        runtime.branch = original_branch


def anchor_values_with_u(policy, anchors, world, budget, *, stall_trained):
    out = _ORIGINAL_ANCHOR_VALUES(policy, anchors, world, budget, stall_trained=stall_trained)
    obs = np.stack([np.asarray(a["obs"], np.float32) for a in anchors])
    gated = P.build(len(anchors), E.GATED_SLOTS)
    n, slots = gated.shape[:2]
    out["U"] = np.asarray(policy.width(np.repeat(obs, slots, axis=0), gated.reshape(-1, 4)), np.float64).reshape(n, slots)
    return out


def evaluate_bca(job, out, deadline):
    W1.policies, E.anchor_values = policies_bca, anchor_values_with_u
    try:
        return {**W1.evaluate(job, out, deadline), "kind": "evaluate_bca", "kappa": KAPPA}
    finally:
        W1.policies, E.anchor_values = _ORIGINAL_POLICIES, _ORIGINAL_ANCHOR_VALUES


def g1(job, out, deadline):
    require(os.environ.get("JAX_PLATFORMS") == "cpu", "The g1 unit runs on the CPU JAX runtime")
    import g1_unit_r5
    return g1_unit_r5.run(job, out, deadline)


def i0(job, out, deadline):
    import i0_unit
    return i0_unit.run(job, out, deadline)


KINDS = {"preflight": W1.preflight, "warmup": W1.warmup, "branch": W1.branch, "bc_only": W1.bc_only, "r0": W1.r0,
         "evaluate": W1.evaluate, "bca_branch": bca_branch, "evaluate_bca": evaluate_bca, "g1": g1, "i0": i0}
CPU_KINDS = ("g1", "i0")


def run_job(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job_path).resolve().parent
    started = time.monotonic()
    counter = None
    report = {"job": job["id"], "kind": job["kind"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "screen": "knee_s10_candidate_v2/native_screen_r5", "kappa": KAPPA, "provenance": job.get("provenance", "research"),
              "bind": SB.receipt()}
    try:
        require(job["kind"] in KINDS, "Unknown job kind")
        if job["kind"] in CPU_KINDS:                         # the CPU units: the recorded environment with the CPU platform
            require("jax" not in sys.modules, "JAX was imported before the CPU environment was set")
            os.environ.update({**step03.WORKER_ENVIRONMENT, "JAX_PLATFORMS": "cpu"})
            report["environment"] = {"JAX_PLATFORMS": "cpu", "others": "step03.WORKER_ENVIRONMENT"}
        else:
            step03.set_worker_environment()
            report["device_kind"] = bind_device_kind()      # binding 8 (A2)
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
