"""Stage 2, the BCA comparison (approval A5; user-run in the recorded WSL GPU environment).

Requires a Stage 1 PASS produced under the committed A4 package, and the A5 precondition recorded with it (no-stall
CUTOFF J_E1 gain >= +50 at 20k in every seed). This supervisor never imports JAX; every GPU job is a separate
`stage2_worker.py` process under matched_policy_pilot_v1's supervise().

Units, in order (a 5.0 rerun repeats only units the supervisor did not judge completed):
  precheck (D1/D2 at the 10k forks; DETECTION-FAIL stops before any branch), 21 branches (per seed: stall CUTOFF
  BCA / distance / native replicate, stall RANDOM BCA / replicate, no-stall CUTOFF BCA / replicate), 3 evaluations.
Caps: 1,800 s per branch job, 4 h of training, 1 h and 18M transitions of evaluation (precheck and evaluations).

Run in WSL (instructions and the approval command: A5_README.md):
  /home/david/bca-operator-audit-20261026/bin/python -B -u <this file> --execute
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

import attempts as A
import stage1
import stage1_math as SM1
import stage2_math as SM
import step03
import step04
from core import require, sha_file
from stage1_worker import sanitize
import units as U
from stage2_worker import FORKS

RUN = HERE / "stage2_run"
MANIFEST = HERE / "A5_MANIFEST.json"
APPROVAL = HERE / "A5_APPROVAL.json"
A5_REQUIRED_FILES = {"stage2.py", "stage2_worker.py", "stage2_math.py", "test_stage2.py", "A5_README.md"}
APPROVAL_WORDS = "I approve A5 (Stage 2, BCA comparison) for knee_edge_ood_v2"
A4_MANIFEST_SHA256 = "e3a8982cd5abaf346931eba2515d6599511c02135e0a58b0d77feb051d3b9a89"     # v2 A4, sealed
JOB_SECONDS, TRAIN_SECONDS = 1_800, 4 * 3_600
EVAL_SECONDS, EVAL_TRANSITIONS = 3_600, 18_000_000
PREFLIGHT_SECONDS = 600
SEEDS = (0, 1, 2)
CELL_KEYS = {"stall_cutoff": "stall_cutoff", "stall_random": "stall_random", "nostall_cutoff": "nostall_cutoff"}
ARM_KEYS = {"bca": "bca", "distance": "distance", "native": "rep"}


def job_specs():
    jobs = [{"id": "precheck", "kind": "precheck"}]
    for i in SEEDS:
        for cell, (data_arm, world, arms) in FORKS.items():
            jobs += [{"id": f"branch_s{i}_{cell}_{arm}", "kind": "branch", "seed_index": i, "data_arm": data_arm,
                      "world": world, "branch_arm": arm, "cell": cell} for arm in arms]
    jobs += [{"id": f"evaluate_s{i}", "kind": "evaluate", "seed_index": i} for i in SEEDS]
    return jobs


JOB_IDS = [j["id"] for j in job_specs()]


# ------------------------------------------------------------------------------------------------ inputs

def stage1_outcome():
    """The last Stage 1 attempt: PASS under the committed A4 package, with the A5 precondition; returns every Stage 1
    unit directory the evaluation and forks use."""
    attempts = A.history(stage1.RUN)
    require(attempts and (attempts[-1] / "report.json").is_file(), "The latest Stage 1 (A4) attempt has no report")
    last = json.loads((attempts[-1] / "report.json").read_text(encoding="utf-8"))
    require(last.get("label") == "PASS", f"Stage 1 ended {last.get('label')}; A5 cannot be requested")
    require(last.get("approval", {}).get("manifest_sha256") == A4_MANIFEST_SHA256 and A4_MANIFEST_SHA256 is not None,
            "The Stage 1 report was not produced under the committed A4 package")
    require(bool(last.get("a5_precondition", {}).get("ok")), "Stage 2 not requested (precondition): no-stall CUTOFF adequacy")
    require(set(last.get("unit_dirs", {})) == set(stage1.JOB_IDS), "The Stage 1 report does not name every unit directory")
    dirs = {}
    for unit in stage1.JOB_IDS:
        path = Path(last["unit_dirs"][unit])
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        verdict = U.read(path / "SUPERVISOR_RESULT.json") or {}
        require(report.get("status") == "completed" and verdict.get("completed") and not (path / "NOT_REPRODUCED.json").exists(),
                f"Stage 1 unit {unit} is not a completed, reproduced unit")
        dirs[unit] = str(path)
    return {"report": str(attempts[-1] / "report.json"), "report_sha256": sha_file(attempts[-1] / "report.json"),
            "a5_precondition": last["a5_precondition"], "dirs": dirs, "collection_dir": last["a3"]["collection_dir"]}


def supervisor():
    return stage1.supervisor()


def default_dispatch(job_dir, cap):
    command = [sys.executable, "-B", "-u", str(HERE / "stage2_worker.py"), "--job", str(job_dir / "job.json")]
    env = dict(os.environ)
    env.update(step03.WORKER_ENVIRONMENT)
    return supervisor()(command, job_dir, cap, env=env, job_id=job_dir.name)


def run_unit(spec, directory, dirs, caps, options, stage1_dirs, dispatch):
    if spec["kind"] == "branch":
        cap, transitions = min(JOB_SECONDS, TRAIN_SECONDS - caps["training_seconds"]), 0
    else:
        cap, transitions = EVAL_SECONDS - caps["evaluation_seconds"], EVAL_TRANSITIONS - caps["evaluation_transitions"]
    if cap <= 0 or (spec["kind"] != "branch" and transitions <= 0):
        raise step03.CapExhausted(f"stage cap exhausted before {spec['id']}")
    job_dir = directory / spec["id"]
    job_dir.mkdir(parents=False, exist_ok=False)
    job = {**spec, "timeout_seconds": float(cap), "transitions": int(transitions), **options,
           "runs": {k: str(v) for k, v in dirs.items()}, "stage1_runs": stage1_dirs}
    if spec["kind"] == "branch":
        i, data_arm, world = spec["seed_index"], spec["data_arm"], spec["world"]
        job["warmup_dir"] = stage1_dirs[f"warmup_s{i}_{data_arm}_{world}"]
        job["stage1_branch_dir"] = stage1_dirs[f"branch_s{i}_{data_arm}_{world}"]
    step03._write_new(job_dir / "job.json", job)
    outcome = dispatch(job_dir, float(cap))
    report = stage1._read(job_dir / "report.json")
    elapsed = float(outcome.get("elapsed_seconds", cap))
    if spec["kind"] == "branch":
        caps["training_seconds"] += elapsed
    else:
        caps["evaluation_seconds"] += elapsed
        caps["evaluation_transitions"] += int((report or {}).get("transitions", transitions))   # no report: charge the allowance
    complete = bool(report and report.get("status") == "completed" and outcome.get("exit_code") == 0
                    and not outcome.get("timed_out") and not outcome.get("interrupted"))
    step03._write_new(job_dir / "SUPERVISOR_RESULT.json", {**outcome, "completed": complete, "cap_seconds": cap})
    return (report if complete else None), outcome


# ------------------------------------------------------------------------------------------------ the attempt

def execute(*, dispatch=None):
    plan = A.check(approval_path=APPROVAL, manifest_path=MANIFEST, run_dir=RUN, words=APPROVAL_WORDS, units=JOB_IDS,
                   sealed_files={j: f"{j}/report.json" for j in JOB_IDS}, required_files=A5_REQUIRED_FILES, here=HERE)
    upstream = A.verify_upstream(HERE, {**stage1.UPSTREAM_PINS, "A4_MANIFEST.json": A4_MANIFEST_SHA256})
    s1 = stage1_outcome()
    sequence_ok, sequence = U.manifest_sequence_ok(plan)
    require(sequence_ok, f"At most one code-only fix (5.0): the package changed more than once over the attempts {sequence}")
    sealed, reference, unsealed = U.classify(RUN, plan)
    directory = RUN / f"attempt_{plan['attempt']}"
    directory.mkdir(parents=True, exist_ok=False)
    step03._write_new(directory / "ATTEMPT.json", {**plan, "stage1": s1, "upstream": upstream, "sealed_after_supervisor": sealed,
                                                   "must_reproduce": reference, "unsealed_by_supervisor": unsealed})
    dispatch = dispatch or default_dispatch
    options = {"provenance": "research", "collection_dir": s1["collection_dir"], "integrity": stage1.U.integrity_record()}
    report = {"stage": "2", "approval": plan, "stage1": s1, "units_run": [], "reproduced_units": [], "unit_dirs": {},
              "failed_unit": "preflight"}
    caps = {"training_seconds": 0.0, "evaluation_seconds": 0.0, "evaluation_transitions": 0}
    dirs, reports = {}, {}
    try:
        pre_dir = directory / "preflight"
        pre_dir.mkdir()
        step03._write_new(pre_dir / "job.json", {"id": "preflight", "kind": "preflight", "timeout_seconds": PREFLIGHT_SECONDS,
                                                 **options})
        outcome = dispatch(pre_dir, PREFLIGHT_SECONDS)
        pre = stage1._read(pre_dir / "report.json")
        report["preflight"] = pre
        if not (pre and pre.get("status") == "completed" and outcome.get("exit_code") == 0):
            report.update(label="INCOMPLETE", reasons=["preflight failed (exempt; no value computed)"])
            return _finish(directory, report)
        report["failed_unit"] = "supervisor"
        for spec in job_specs():
            unit = spec["id"]
            report["failed_unit"] = unit
            try:
                dirs[unit], reports[unit], mode = U.take(
                    unit, run_dir=RUN, directory=directory, sealed=sealed, reference=reference,
                    fingerprint_now=U.fingerprint("stage2_worker", spec["kind"]), caps=caps,
                    cap_key="training_seconds" if spec["kind"] == "branch" else "evaluation_seconds",
                    job_now={**spec, **{k: v for k, v in options.items() if k != "collection_dir"}},
                    run=lambda spec=spec: run_unit(spec, directory, dirs, caps, options, s1["dirs"], dispatch))
            except U.UnitFailed as exc:
                report["units_run"].append(unit)
                report.update(label="INCOMPLETE", reasons=[str(exc)], caps=caps)
                return _finish(directory, report)
            report["unit_dirs"][unit] = str(dirs[unit])
            if mode != "reused":
                report["units_run"].append(unit)
            if mode == "reproduced":
                report["reproduced_units"].append(unit)
            if unit == "precheck":
                report["failed_unit"] = "precheck_decision"
                detection = detect(reports["precheck"])
                report["detection"] = detection
                if not all(d["D1"] and d["D2"] for d in detection["by_seed"].values()):
                    label, reasons = SM.label(v1_reasons=[], detection_by_seed=detection["by_seed"], gains={}, screen=None,
                                              damage_by_seed={})
                    report.update(label=label, reasons=reasons, failed_unit=None, caps=caps)
                    step03._write_new(directory / "decision.json", sanitize({"label": label, "reasons": reasons,
                                                                              "detection": detection}))
                    return _finish(directory, report)
        report["failed_unit"] = "decision"
        decision = decide(reports["precheck"], {i: reports[f"evaluate_s{i}"] for i in SEEDS},
                          {u: reports[u] for u in JOB_IDS if u.startswith("branch_")})
        step03._write_new(directory / "decision.json", sanitize(decision))
        report.update(label=decision["label"], reasons=decision["reasons"], caps=caps)
        report["failed_unit"] = "decision" if decision["label"] == "INCOMPLETE" else None
    except step03.CapExhausted as exc:
        report.update(label="INCOMPLETE", reasons=[f"cap: {exc}"], caps=caps)
    except BaseException as exc:
        report.update(label="INCOMPLETE", reasons=[f"{type(exc).__name__}: {exc}"], caps=caps, traceback=traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report["interrupted"] = type(exc).__name__
            _finish(directory, report)
            raise
    return _finish(directory, report)


def _finish(directory, report):
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    A.finish(report)
    step03._write_new(directory / "report.json", sanitize(report))
    return report


# ------------------------------------------------------------------------------------------------ decision

def _u(seed_arm, slots, gated):
    u = stage1.arr(seed_arm["U"])
    return {name: u[gated, k] for k, name in enumerate(slots)}


def detect(pre):
    gated = np.asarray(pre["gated"], bool)
    slots = pre["slots"]
    by_seed, reported = {}, {}
    for s in SEEDS:
        d = pre["seeds"][str(s)] if str(s) in pre["seeds"] else pre["seeds"][s]
        cut, ran = _u(d["cutoff"], slots, gated), _u(d["random"], slots, gated)
        by_seed[s] = SM.detection(cut, ran)
        u, q1 = stage1.arr(d["cutoff"]["U"])[gated], stage1.arr(d["cutoff"]["Q1"])[gated]
        score = {name: (q1 - 3.0 * u)[:, k] for k, name in enumerate(slots)}
        mag = d["cutoff"]["magnitude_U"]
        keep = {arm: gated & ~np.asarray(d[arm]["own_a123_past_knee"], bool) for arm in ("cutoff", "random")}
        own = {arm: {name: stage1.arr(d[arm]["own_U"])[keep[arm], k] for k, name in enumerate(slots)} for arm in ("cutoff", "random")}
        own_ok = keep["cutoff"].any() and keep["random"].any()
        reported[s] = {**SM.detection_reported(cut, score),
                       "magnitude_control_mean_U": {k: float(np.mean(stage1.arr(v)[gated])) for k, v in mag.items()},
                       "actor_own_a123": {"excluded_past_knee": {arm: int((gated & ~keep[arm]).sum()) for arm in keep},
                                          "detection": SM.detection(own["cutoff"], own["random"]) if own_ok else None}}
    return {"by_seed": by_seed, "reported": reported}


def decide(pre, evaluations, branches):
    detection = detect(pre)
    j, gains, damage, repairs, ninediff, nostall_extra = {}, {}, {}, {}, {}, {}
    for s in SEEDS:
        cells = evaluations[s]["cells"]
        j[s] = {}
        for cell, entry in cells.items():
            for arm, b in entry["branches"].items():
                own = b["free"]
                j[s][(CELL_KEYS[cell], ARM_KEYS[arm])] = own["e2/pi"]["J"]
                gains.setdefault(f"{cell}/{arm}", []).append(own["e1/pi"]["J"] - entry["J_E1_untrained"])
            ninediff[f"s{s}_{cell}"] = entry["branches"]["native"]["free"]["e2/pi"]["J"] - entry["stage1_native_on_stage2_bank"]["e2/pi"]["J"]
        sc = cells["stall_cutoff"]["branches"]
        rep, bca = sc["native"]["free"], sc["bca"]["free"]
        damage[s] = bool(rep["e2_rates"]["r_deep_minus"] >= SM1.K2_BAR and rep["e2/delta_clip"] >= SM1.K3_BAR)
        ra, ba = sc["native"]["anchors"], sc["bca"]["anchors"]
        names = ra["gated_slots"]
        gated = np.asarray([t in (0, 25, 50) for t in ra["times"]])
        rep_kc = SM1.kc(stage1.arr(ra["q1"]), stage1.arr(ra["qpi_clip"]), names, "s10")
        bca_score = stage1.arr(ba["q1"]) - 3.0 * stage1.arr(ba["U"])
        bca_kc = SM1.kc(bca_score, stage1.arr(ba["qpi_clip"]), names, "s10")
        valid = np.asarray(ra["valid"], bool) & np.asarray(ba["valid"], bool) & gated
        repairs[s] = SM.repair({"bca_entry": bca["e2_rates"]["r_deep_minus"], "rep_entry": rep["e2_rates"]["r_deep_minus"],
                                "bca_delta_clip": bca["e2/delta_clip"], "rep_delta_clip": rep["e2/delta_clip"],
                                "valid_both": valid, "rep_kc_s10": rep_kc, "bca_kc_s10_score": bca_kc})
        nc = cells["nostall_cutoff"]["branches"]
        nostall_extra[s] = {"bca_minus_rep_entry": nc["bca"]["free"]["e2_rates"]["r_deep_minus"] - nc["native"]["free"]["e2_rates"]["r_deep_minus"],
                            "bca_minus_rep_delta_clip": nc["bca"]["free"]["e2/delta_clip"] - nc["native"]["free"]["e2/delta_clip"]}
    v1 = [f"{u}: replicate minibatches repeat Stage 1" for u, r in branches.items()
          if r.get("branch_arm") == "native" and not r.get("minibatches_differ_from_stage1_native")]
    screen = SM.screens(j)
    label, reasons = SM.label(v1_reasons=v1, detection_by_seed=detection["by_seed"], gains=gains, screen=screen,
                              damage_by_seed=damage)
    return {"label": label, "reasons": reasons, "detection": detection, "screens": screen, "gains_e1": gains,
            "damage_reproduced": damage, "recovery": SM.recovery(j), "repair": repairs,
            "repair_all_seeds": SM.repair_all_seeds(repairs),
            "replicate_minus_stage1_native_J": ninediff, "nostall_cutoff_reported": nostall_extra,
            "J_e2": {s: {f"{c}/{a}": v for (c, a), v in d.items()} for s, d in j.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    result = execute()
    print(json.dumps({"label": result.get("label"), "reasons": result.get("reasons")}, indent=2, default=step03._plain))
