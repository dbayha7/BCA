"""Stage 1, the native damage gate (approval A4; user-run in the recorded WSL GPU environment).

Requires step 0.4 (A3) to have ended PASS; the A4 approval must carry the written acknowledgement of the standing
prediction "Stage 1 V0(d) may end INCONCLUSIVE". This supervisor never imports JAX: every GPU job is a separate
`stage1_worker.py` process under matched_policy_pilot_v1's supervise() (an owned process group, killed at its cap).

Units, in order (each is one job; a 5.0 rerun repeats only units without a completed report):
  12 warmups (seed x world x arm), R0, 12 branches, 3 BC-only, 3 evaluations (one per seed index).
Caps (fail closed; whichever binds first wins): 1,800 s per training job, 5 h of training, 1 h and 18M transitions
of evaluation (R0 and the three evaluations). R0: INCONCLUSIVE when adequacy fails, FAIL-early when the stall CUTOFF
reach rate is below 0.05 in every seed (branches are then not trained).

Run in WSL (instructions and the approval command: A4_README.md):
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
import stage1_eval
import stage1_math as SM
import units as U
from stage1_worker import sanitize
import step03
import step04
import step04_math as SM4
from core import VALIDATION, module, require, sha_file
from knee_region import GATED_PAST_KNEE, S9

RUN = HERE / "stage1_run"
MANIFEST = HERE / "A4_MANIFEST.json"
APPROVAL = HERE / "A4_APPROVAL.json"
A4_REQUIRED_FILES = {"integrity.py", "stage1.py", "stage1_worker.py", "stage1_eval.py", "stage1_math.py", "units.py", "stage1_smoke.py",
                     "test_stage1.py", "A4_README.md"}
APPROVAL_WORDS = "I approve A4 (Stage 1, native damage gate) for knee_edge_ood_v2"
PREDICTION_ACK = ("I acknowledge the written predictions: Stage 1 V0(d) may end INCONCLUSIVE, and after the v1 step 0.3 "
                  "INCONCLUSIVE, Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE")
A3_MANIFEST_SHA256 = "f681abb063d4f0d107e25a2bc9bfcc04b29e7eeed5130045b38b79c5777d0548"     # v2 A3, sealed
UPSTREAM_PINS = {"MANIFEST.json": step03.A1_MANIFEST_SHA256, "A2_MANIFEST.json": step04.A2_MANIFEST_SHA256,
                 "A3_MANIFEST.json": A3_MANIFEST_SHA256}
JOB_SECONDS, TRAIN_SECONDS = 1_800, 5 * 3_600
EVAL_SECONDS, EVAL_TRANSITIONS = 3_600, 18_000_000
PREFLIGHT_SECONDS = 600
SEEDS, WORLDS, ARMS = (0, 1, 2), ("stall3", "nostall"), ("cutoff", "random")
TRAINING_KINDS, EVALUATION_KINDS = ("warmup", "branch", "bc_only"), ("r0", "evaluate")
GATED = np.arange(48)                         # stage1_e2_anchors items 0-47 are t in {0, 25, 50} (blocks of 16)


def run_name(i, arm, world):
    return f"s{i}_{arm}_{world}"


def job_specs():
    jobs = [{"id": f"warmup_{run_name(i, a, w)}", "kind": "warmup", "seed_index": i, "arm": a, "world": w}
            for i in SEEDS for w in WORLDS for a in ARMS]
    jobs.append({"id": "r0", "kind": "r0"})
    jobs += [{"id": f"branch_{run_name(i, a, w)}", "kind": "branch", "seed_index": i, "arm": a, "world": w}
             for i in SEEDS for w in WORLDS for a in ARMS]
    jobs += [{"id": f"bc_only_s{i}", "kind": "bc_only", "seed_index": i} for i in SEEDS]
    jobs += [{"id": f"evaluate_s{i}", "kind": "evaluate", "seed_index": i} for i in SEEDS]
    return jobs


JOB_IDS = [j["id"] for j in job_specs()]


# ------------------------------------------------------------------------------------------------ approval inputs

def a3_outcome(approval):
    """Step 0.4's last report must be PASS; the approval acknowledges the standing prediction. Returns the sealed
    collection directory (the attempt that completed the collection unit)."""
    attempts = A.history(step04.RUN)
    require(attempts and (attempts[-1] / "report.json").is_file(), "The latest step 0.4 (A3) attempt has no report")
    reports = attempts[-1:]
    last = json.loads((reports[-1] / "report.json").read_text(encoding="utf-8"))
    require(last.get("label") == "PASS", f"Step 0.4 ended {last.get('label')}; A4 cannot be requested")
    require(last.get("approval", {}).get("manifest_sha256") == A3_MANIFEST_SHA256,
            "The step 0.4 report was not produced under the committed A3 package")
    require(PREDICTION_ACK in approval.get("approval", ""), "The A4 approval must acknowledge the standing prediction")
    sealed = last["approval"].get("sealed_units", {})
    collection_dir = step04.RUN / sealed["collection"] if "collection" in sealed else reports[-1]
    record = json.loads((collection_dir / "collection.json").read_text(encoding="utf-8"))
    require(record["unit_label"] == "COMPLETE", "The sealed collection unit is not complete")
    return {"label": last["label"], "report": str(reports[-1] / "report.json"),
            "collection_dir": str(collection_dir), "collection_sha256": sha_file(collection_dir / "collection.json")}


def supervisor():
    return module("_knee_stage1_supervisor", VALIDATION / "matched_policy_pilot_v1/launch.py").supervise


def default_dispatch(job_dir, cap):
    command = [sys.executable, "-B", "-u", str(HERE / "stage1_worker.py"), "--job", str(job_dir / "job.json")]
    env = dict(os.environ)
    env.update(step03.WORKER_ENVIRONMENT)
    return supervisor()(command, job_dir, cap, env=env, job_id=job_dir.name)


# ------------------------------------------------------------------------------------------------ the attempt

reproduction_key, _read = U.reproduction_key, U.read


def run_unit(spec, directory, dirs, caps, options, dispatch):
    """Dispatch one job under its cap; returns (report or None, supervisor outcome)."""
    kind = spec["kind"]
    if kind in TRAINING_KINDS:
        cap = min(JOB_SECONDS, TRAIN_SECONDS - caps["training_seconds"])
        transitions = 0
    else:
        cap = EVAL_SECONDS - caps["evaluation_seconds"]
        transitions = EVAL_TRANSITIONS - caps["evaluation_transitions"]
    if cap <= 0 or (kind in EVALUATION_KINDS and transitions <= 0):
        raise step03.CapExhausted(f"stage cap exhausted before {spec['id']}")
    job_dir = directory / spec["id"]
    job_dir.mkdir(parents=False, exist_ok=False)
    job = {**spec, "timeout_seconds": float(cap), "transitions": int(transitions), "runs": {k: str(v) for k, v in dirs.items()},
           **options}
    if kind == "branch":
        job["warmup_dir"] = str(dirs[spec["id"].replace("branch_", "warmup_")])
    step03._write_new(job_dir / "job.json", job)
    outcome = dispatch(job_dir, float(cap))
    report = _read(job_dir / "report.json")
    elapsed = float(outcome.get("elapsed_seconds", cap))
    if kind in TRAINING_KINDS:
        caps["training_seconds"] += elapsed
    else:
        caps["evaluation_seconds"] += elapsed
        caps["evaluation_transitions"] += int((report or {}).get("transitions", transitions))
    complete = bool(report and report.get("status") == "completed" and outcome.get("exit_code") == 0
                    and not outcome.get("timed_out") and not outcome.get("interrupted"))
    step03._write_new(job_dir / "SUPERVISOR_RESULT.json", {**outcome, "completed": complete, "cap_seconds": cap})
    return (report if complete else None), outcome


def execute(*, provenance="research", test_seeds=None, training_seeds=None, visited_seed=None, dispatch=None):
    plan = A.check(approval_path=APPROVAL, manifest_path=MANIFEST, run_dir=RUN, words=APPROVAL_WORDS, units=JOB_IDS,
                   sealed_files={j: f"{j}/report.json" for j in JOB_IDS}, required_files=A4_REQUIRED_FILES, here=HERE)
    upstream = A.verify_upstream(HERE, UPSTREAM_PINS)
    a3 = a3_outcome(plan["approval"])
    sequence_ok, sequence = U.manifest_sequence_ok(plan)
    require(sequence_ok, f"At most one code-only fix (5.0): the package changed more than once over the attempts {sequence}")
    sealed, reference, unsealed = U.classify(RUN, plan)
    directory = RUN / f"attempt_{plan['attempt']}"
    directory.mkdir(parents=True, exist_ok=False)
    step03._write_new(directory / "ATTEMPT.json", {**plan, "a3": a3, "upstream": upstream, "sealed_after_supervisor": sealed,
                                                   "must_reproduce": reference, "unsealed_by_supervisor": unsealed})
    dispatch = dispatch or default_dispatch
    options = {"provenance": provenance, "collection_dir": a3["collection_dir"], "test_seeds": test_seeds,
               "training_seeds": training_seeds, "visited_seed": visited_seed, "integrity": U.integrity_record()}
    report = {"stage": "1", "approval": plan, "a3": a3, "units_run": [], "reproduced_units": [], "unit_dirs": {},
              "failed_unit": "preflight"}
    caps = {"training_seconds": 0.0, "evaluation_seconds": 0.0, "evaluation_transitions": 0}
    dirs, reports = {}, {}
    try:
        pre_dir = directory / "preflight"
        pre_dir.mkdir()
        step03._write_new(pre_dir / "job.json", {"id": "preflight", "kind": "preflight", "timeout_seconds": PREFLIGHT_SECONDS,
                                                 **options})
        outcome = dispatch(pre_dir, PREFLIGHT_SECONDS)
        pre = _read(pre_dir / "report.json")
        report["preflight"] = pre
        if not (pre and pre.get("status") == "completed" and outcome.get("exit_code") == 0):
            report.update(label="INCOMPLETE", reasons=["preflight failed (exempt; no value computed)"])
            return _finish(directory, report)
        report["failed_unit"] = "supervisor"           # from here on every failure is counted (5.0)
        for spec in job_specs():
            unit = spec["id"]
            report["failed_unit"] = unit
            try:
                dirs[unit], reports[unit], mode = U.take(
                    unit, run_dir=RUN, directory=directory, sealed=sealed, reference=reference,
                    fingerprint_now=U.fingerprint("stage1_worker", spec["kind"]), caps=caps,
                    cap_key="training_seconds" if spec["kind"] in TRAINING_KINDS else "evaluation_seconds",
                    job_now={**spec, **{k: v for k, v in options.items() if k != "collection_dir"}},
                    run=lambda spec=spec: run_unit(spec, directory, dirs, caps, options, dispatch))
            except U.UnitFailed as exc:
                report["units_run"].append(unit)
                report.update(label="INCOMPLETE", reasons=[str(exc)], caps=caps)
                return _finish(directory, report)
            report["unit_dirs"][unit] = str(dirs[unit])
            if mode != "reused":
                report["units_run"].append(unit)
            if mode == "reproduced":
                report["reproduced_units"].append(unit)
            if unit == "r0":
                r0 = reports["r0"]
                report["r0"] = {k: r0[k] for k in ("seeds", "adequacy_ok", "fail_early")}
                if not r0["adequacy_ok"] or r0["fail_early"]:
                    label, reasons, _ = SM.label(v1_reasons=[], r0=r0, per_seed={})
                    report.update(label=label, reasons=reasons, failed_unit=None, caps=caps,
                                  reading=SM.FAIL_EARLY_READING if label == "FAIL-early" else None)
                    return _finish(directory, report)
        report["failed_unit"] = "decision"
        decision = decide({i: reports[f"evaluate_s{i}"] for i in SEEDS}, reports["r0"],
                          {u: reports[u] for u in JOB_IDS if u.startswith(("warmup_", "branch_"))})
        step03._write_new(directory / "decision.json", sanitize(decision))
        report.update(label=decision["label"], reasons=decision["reasons"], reading=decision["reading"],
                      reading_text=SM.READINGS.get(decision["reading"]), caps=caps,
                      a5_precondition=decision["a5_precondition"])
        report["failed_unit"] = "decision" if decision["label"] == "INCOMPLETE" else None
    except step03.CapExhausted as exc:
        report.update(label="INCOMPLETE", reasons=[f"cap: {exc}"], caps=caps)
    except BaseException as exc:
        report.update(label="INCOMPLETE", reasons=[f"{type(exc).__name__}: {exc}"], caps=caps,
                      traceback=traceback.format_exc())
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

def arr(values):
    """Floats from a sanitized JSON list (non-finite values were written as strings)."""
    names = {"nan": np.nan, "positive_infinity": np.inf, "negative_infinity": -np.inf}
    return np.asarray([[names.get(v, v) for v in row] if isinstance(row, list) else names.get(row, row)
                       for row in values], np.float64)


def anchor_inputs(entry, stall_trained):
    """Per-gated-anchor arrays of one run at 20k (E2 anchors)."""
    a = entry["anchors_e2"]
    names = a["gated_slots"]
    qpi, q1 = arr(a["qpi_clip"])[GATED], arr(a["q1"])[GATED]
    out = {"valid": np.asarray(a["valid"], bool)[GATED], "kc_s05": SM.kc(q1, qpi, names, "s05"),
           "kc_s10": SM.kc(q1, qpi, names, "s10"), "names": names, "qpi": qpi, "q1": q1,
           "kc_s20": SM.kc(q1, qpi, names, "s20") if "s20" in names else None,      # v2 change 3 (reported)
           "kc_by_critic": {c: {s: SM.kc(arr(a[k])[GATED], qpi, names, s) for s in ("s05", "s10")}
                            for c, k in (("Q1", "q1"), ("Q2", "q2"), ("twin_min", "qmin"))},
           "q1_error_knee": np.abs(q1[:, names.index("edge_knee")] - qpi[:, names.index("edge_knee")])}
    if stall_trained:
        ns, nn = arr(a["qpi_nostall"])[GATED], a["nostall_slots"]
        res = SM4.neighbor_residuals(qpi[:, [names.index(n) for n in S9]], {n: qpi[:, names.index(n)] for n in GATED_PAST_KNEE},
                                     {n: ns[:, nn.index(n)] for n in ("s05", "s10", "s20")})
        out.update(res=res, item5_ok=SM.item5_ok(res["item5_error"]), rho_s05=res["rho_hat"]["s05"],
                   rho_s10=res["rho_hat"]["s10"], rho_max_p10=res["rho_hat_max"]["p10"])
    return out


def equality_check(entry):
    """V1 backstop (the evaluate unit already enforced it): Q_nostall equals Q_stall on the supported S9 slots."""
    worst = 0.0
    for key in ("anchors_e2", "anchors_e1"):
        a = entry[key]
        ok, dev = stage1_eval.stall_equals_nostall({**a, "qpi_clip": arr(a["qpi_clip"]), "qpi_nostall": arr(a["qpi_nostall"])})
        worst = max(worst, dev)
        if not ok:
            return False, worst
    return True, worst


def decide(evaluations, r0, training):
    v1, per_seed, reported = [], {}, {}
    for i in SEEDS:
        runs = evaluations[i]["runs"]
        c, r = runs[run_name(i, "cutoff", "stall3")], runs[run_name(i, "random", "stall3")]
        nc = runs[run_name(i, "cutoff", "nostall")]
        for name in (run_name(i, "cutoff", "stall3"), run_name(i, "random", "stall3")):
            ok, worst = equality_check(runs[name]["020000"])
            if not ok:
                v1.append(f"{name}: stall = no-stall on the supported slots fails (largest scaled deviation {worst:.3g})")
        ci, ri = anchor_inputs(c["020000"], True), anchor_inputs(r["020000"], True)
        nci = anchor_inputs(nc["020000"], False)
        nr = runs[run_name(i, "random", "nostall")]

        def numbers(run, inputs):
            own = run["020000"]["own_world"]
            return {**inputs, "gain_e1": own["e1/pi"]["J"] - run["J_E1_untrained"],
                    "r_deep_rate": own["e2_rates"]["r_deep_minus"], "delta_clip": own["e2/delta_clip"],
                    "a123_past_rate": own["e2_rates"]["a123_past_knee"]}
        cn, rn = numbers(c, ci), numbers(r, ri)
        swap = {"delta_clip": c["020000"]["nostall_world"]["e2/delta_clip"],
                "a123_past_rate": c["020000"]["nostall_world"]["e2_rates"]["a123_past_knee"]}
        per_seed[i] = SM.seed_rules(cn, rn, swap)
        sets = SM.anchor_sets(ci["valid"], ri["valid"], ci["item5_ok"], ri["item5_ok"])
        both = ci["valid"] & nci["valid"]
        j = {(arm, w, ew): runs[run_name(i, arm, w)]["020000"]["own_world" if ew == w else "nostall_world"]["e2/pi"]["J"]
             for arm in ARMS for w, ew in (("stall3", "stall3"), ("stall3", "nostall"), ("nostall", "nostall"))}
        reported[i] = {
            "l2_items_cutoff": SM4.l2_rule(ci["res"], ci["qpi"][:, ci["names"].index("edge_knee")],
                                           ci["qpi"][:, ci["names"].index("s10")], sets["common"])[0] if sets["common"].any() else None,
            "l2_items_random": SM4.l2_rule(ri["res"], ri["qpi"][:, ri["names"].index("edge_knee")],
                                           ri["qpi"][:, ri["names"].index("s10")], sets["common"])[0] if sets["common"].any() else None,
            "K1b_median_kc_s10_stall_minus_nostall_cutoff": SM.median(ci["kc_s10"][both] - nci["kc_s10"][both]),
            "cutoff_minus_random_entry": cn["r_deep_rate"] - rn["r_deep_rate"],
            "nostall_cutoff": {"entry": nc["020000"]["own_world"]["e2_rates"]["r_deep_minus"],
                               "delta_clip": nc["020000"]["own_world"]["e2/delta_clip"],
                               "gain_e1": nc["020000"]["own_world"]["e1/pi"]["J"] - nc["J_E1_untrained"]},
            "nostall_random": {"entry": nr["020000"]["own_world"]["e2_rates"]["r_deep_minus"],
                               "delta_clip": nr["020000"]["own_world"]["e2/delta_clip"],
                               "gain_e1": nr["020000"]["own_world"]["e1/pi"]["J"] - nr["J_E1_untrained"]},
            "kc_medians_common": {arm: {c: {s: SM.median(v[sets["common"]]) for s, v in by.items()}
                                        for c, by in x["kc_by_critic"].items()} for arm, x in (("cutoff", ci), ("random", ri))},
            "v2_kc_s20_full_throttle": {arm: None if x["kc_s20"] is None else
                                        {"median_common": SM.median(x["kc_s20"][sets["common"]]),
                                         "share_le_0_5_common": float(np.mean(x["kc_s20"][sets["common"]] <= 0.5))
                                         if sets["common"].any() else None}
                                        for arm, x in (("cutoff", ci), ("random", ri))},
            "q1_abs_error_at_minus_knee": {arm: SM.median(x["q1_error_knee"][x["valid"]]) for arm, x in (("cutoff", ci), ("random", ri))},
            "delta_clipall_e2": {name: runs[name]["020000"]["own_world"]["e2/delta_clipall"] for name in runs},
            "checkpoint_15k": {name: {k: runs[name]["015000"]["own_world"][k] for k in ("e2/delta_clip", "e2_rates")}
                               for name in runs},
            "visited_kc_medians": {name: visited_kc(runs[name]["020000"]["visited"]) for name in runs},
            "target_smoothing_w_rate": {name: runs[name]["020000"]["target_smoothing_w_rate"] for name in runs},
            "visited_vs_training_smd": {name: runs[name]["020000"]["visited_vs_training_smd"] for name in runs},
            "training_input_shift_smd_stall_vs_nostall": evaluations[i].get("training_input_shift_smd"),
            "bc_only": evaluations[i].get("bc_only"),
            "secondary": SM.secondary(j),
            "gradient_audits": {u: rep.get("gradient_audits") for u, rep in training.items() if f"_s{i}_" in u},
        }
    gains = [reported[i]["nostall_cutoff"]["gain_e1"] for i in SEEDS]
    label, reasons, reading = SM.label(v1_reasons=v1, r0=r0, per_seed=per_seed)
    return {"label": label, "reasons": reasons, "reading": reading, "failed_unit": "decision" if v1 else None,
            "per_seed": per_seed, "reported": reported,
            "a5_precondition": {"no_stall_cutoff_gain_e1": gains, "ok": bool(all(g >= SM.GAIN_BAR for g in gains)),
                                "meaning": "Stage 2 may be requested only after a Stage 1 PASS and with this precondition"}}


def visited_kc(v):
    names, qpi, q1 = v["slots"], arr(v["qpi_clip"]), arr(v["q1"])
    return {s: SM.median(SM.kc(q1, qpi, names, s)) for s in ("s05", "s10")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    result = execute()
    print(json.dumps({"label": result.get("label"), "reasons": result.get("reasons"), "reading": result.get("reading")},
                     indent=2, default=step03._plain))
