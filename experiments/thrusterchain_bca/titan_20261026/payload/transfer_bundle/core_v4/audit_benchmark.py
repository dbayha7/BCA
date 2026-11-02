"""Ten topical post-change acceptance passes against one exact source snapshot.

These are concrete software/numerical passes, not ten independent scientific
replications or research-level evidence. Independent reviews are stored separately.
"""
import argparse
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from datetime import datetime, timezone
from run_benchmark import source_manifest, write_json
from thrusterchain.datasets import canonical_json, load_dataset
from thrusterchain.regular import load_frozen_policy
from thrusterchain.planning import array_hash
from thrusterchain.reporting import validate_control_records
from thrusterchain.identification_validation import validate_identification_receipt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"tests"))


def load_json(path):
    def reject_constant(value):
        raise AssertionError("nonstandard JSON numeric constant: " + value)
    def unique_keys(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise AssertionError("duplicate JSON key: " + key)
            result[key]=value
        return result
    def finite_float(value):
        number=float(value)
        if not np.isfinite(number): raise AssertionError("overflowing JSON number: " + value)
        return number
    return json.loads(Path(path).read_text(),parse_constant=reject_constant,
                      parse_float=finite_float,object_pairs_hook=unique_keys)

def flatten(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item

def test_cases(module, include=None, exclude=None):
    suite=unittest.defaultTestLoader.loadTestsFromName(module)
    cases=list(flatten(suite))
    if include: cases=[x for x in cases if any(k in x.id() for k in include)]
    if exclude: cases=[x for x in cases if not any(k in x.id() for k in exclude)]
    if not cases: raise RuntimeError("empty acceptance pass "+module)
    return cases

def artifact_checks(run):
    run=Path(run)
    if (run/"INVALID_RUN.json").exists(): raise AssertionError("run explicitly invalid")
    results=load_json(run/"benchmark_results.json")
    control_validation=validate_control_records(results, run/"control_episode_records.csv")
    source_before=load_json(run/"source_manifest_before.json")
    source_after=load_json(run/"source_manifest_after.json")
    current=source_manifest()
    if source_before["source_sha256"]!=source_after["source_sha256"]: raise AssertionError("run source drift")
    if current["source_sha256"]!=source_after["source_sha256"]: raise AssertionError("source changed after run")
    data=load_dataset(run/"offline_dataset.npz")
    policy,receipt=load_frozen_policy(run/"frozen_bc")
    if results["dataset_sha256"]!=data.digest: raise AssertionError("dataset mismatch")
    if receipt.payload["dataset_sha256"]!=data.digest: raise AssertionError("freeze receipt mismatch")
    calibration=load_json(run/"imitation_final_calibration.json")
    for key,expected in (("receipt_sha256",receipt.digest),("policy_sha256",policy.digest),("dataset_sha256",data.digest)):
        if calibration[key]!=expected: raise AssertionError("calibration identity mismatch: "+key)
    if calibration["split"]!="final_calibration" or calibration["policy_updated"]:
        raise AssertionError("invalid final calibration role")
    references=load_json(run/"reference_freeze_receipts.json")
    with np.load(run/"identified_model.npz",allow_pickle=False) as model:
        if array_hash(model["A"],model["B"])!=references["identified_model_mpc"]["model_hash"]:
            raise AssertionError("identified model artifact mismatch")
        identification_validation=validate_identification_receipt(
            references["identified_model_mpc"],data,model["A"],model["B"])
    if results["research_evidence"] or results["newly_trained_nine_cell_policy_grid_complete"]:
        raise AssertionError("smoke falsely labeled research")
    if set(results["baselines"])!={"behavior_cloning","random","true_model_mpc","identified_model_mpc"}:
        raise AssertionError("baseline missing")
    for name,result in results["baselines"].items():
        if any(type(result[key]) is not int or result[key] != 0 for key in ("parameter_updates", "normalization_updates")):
            raise AssertionError("updates during evaluation")
        guard=result["inference_guard"]
        if any(type(guard[key]) is not int or guard[key] != 0 for key in ("training_calls","dataset_collection_calls","write_attempts","forbidden_api_calls")):
            raise AssertionError("prohibited operation observed")
        if set(result["families"])!={"recorded","strong_current","big_waves","big_kicks"}:
            raise AssertionError("scenario family missing")
        if name=="behavior_cloning":
            if type(result["inference_audit"]["forbidden_api_calls"]) is not int or result["inference_audit"]["forbidden_api_calls"]!=0:
                raise AssertionError("BC attempted prohibited API")
            if result["policy_sha256_before"]!=policy.digest or result["policy_sha256_after"]!=policy.digest:
                raise AssertionError("BC policy identity mismatch")
            if result["receipt_sha256"]!=receipt.digest:
                raise AssertionError("BC selection receipt identity mismatch")
        elif canonical_json(result["receipt"])!=canonical_json(references[name]):
            raise AssertionError("reference result receipt mismatch")
        for family,row in result["families"].items():
            if len(row["returns"])!=results["evaluation_episodes_per_family"]:
                raise AssertionError("incomplete episodes")
            if not np.isfinite(row["returns"]).all() or not np.isfinite(row["discounted_returns"]).all():
                raise AssertionError("nonfinite returns")
            if len(row["discounted_returns"])!=len(row["returns"]) or len(row["episode_seeds"])!=len(row["returns"]):
                raise AssertionError("incomplete parallel evaluation arrays")
            if row["episode_seeds"]!=results["baselines"]["behavior_cloning"]["families"]["recorded"]["episode_seeds"]:
                raise AssertionError("unpaired evaluation seeds")
    diagnostic=load_json(run/"diagnostic_results.json")
    if len(diagnostic["cells"])!=9 or diagnostic["guarantee_claimed"]:
        raise AssertionError("invalid smoke diagnostic interpretation")
    if set(diagnostic["four_arms"])!={"unweighted_standard","unweighted_most_caution","weighted_standard","weighted_most_caution"}:
        raise AssertionError("incomplete fixed-fallback comparisons")
    return dict(dataset_sha256=data.digest,policy_sha256=policy.digest,
        source_sha256=current["source_sha256"],baseline_count=4,
        all_intercepted_guard_counts_zero=True, control_record_validation=control_validation,
        identification_receipt_validation=identification_validation,
        profile="smoke",research_evidence=False)

def run_review(output, run):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    snapshot=source_manifest(); write_json(output/"source_snapshot.json",snapshot)
    provenance=["archival","authentication","original_transition","batch_dynamics"]
    numerical=["float32","reward_uses","boundary_rollouts","optional_jax"]
    dataset=["grouped_splits","continuing_time_limit","group_leakage","replay_and_time",
             "dataset_immutable","generation_refuses"]
    training=["offline_fitting","invalid_ridge"]
    plans=[
      ("Canonical provenance and physical parity",test_cases("test_environment",include=provenance)),
      ("Physical invariants and action arithmetic",test_cases("test_environment",include=numerical)),
      ("Gymnasium API, scenario claims and packaged notices",test_cases("test_environment",exclude=provenance+numerical)+test_cases("test_revision_claims")+test_cases("test_wheel_notices")),
      ("Offline lineage, NumPy compatibility and report integrity",test_cases("test_regular",include=dataset)+test_cases("test_numpy_compatibility")+test_cases("test_reporting_integrity")+test_cases("test_revision_v3_csv")+test_cases("test_revision_v3_metadata")),
      ("Offline fit, normalization, integer budgets and selection isolation",test_cases("test_regular",include=training)+test_cases("test_training_protocol")+test_cases("test_sgd_adapter")),
      ("Frozen inference, separate adaptation, resumption and cooperative guards",test_cases("test_regular",exclude=dataset+training)+test_cases("test_inference_guard")+test_cases("test_revision_planner_guard")+test_cases("test_training_runner")+test_cases("test_training_cli")),
      ("Constrained references and train-only identification",test_cases("test_planning")+test_cases("test_revision_v3_identification")),
      ("Controlled coverage laws, support and count planning",test_cases("test_coverage")),
      ("Terminal calibration, quantiles and held-out units",test_cases("test_calibration")),
    ]
    entries=[]
    for i,(name,cases) in enumerate(plans,1):
        before=source_manifest()["source_sha256"]
        log=io.StringIO()
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite(cases))
        after=source_manifest()["source_sha256"]
        (output/f"pass_{i:02d}.txt").write_text(log.getvalue(),encoding="utf8")
        entry=dict(pass_number=i,name=name,source_sha256=before,
            command=f"python -B audit_benchmark.py --output {output} --run {run}",
            test_ids=[x.id() for x in cases],tests_run=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
            status="pass" if result.wasSuccessful() and before==after==snapshot["source_sha256"] else "fail")
        entries.append(entry); print(f"Pass {i}: {entry['status']} — {name}",flush=True)
    try:
        evidence=artifact_checks(run); status="pass"
    except Exception as exc:
        evidence={"error":str(exc)}; status="fail"
    entries.append(dict(pass_number=10,name="End-to-end receipts, complete evaluation and artifact claims",
        source_sha256=snapshot["source_sha256"],status=status,evidence=evidence,
        command=f"python -B audit_benchmark.py --output {output} --run {run}"))
    report=dict(schema="thrusterchain.ten_post_change_passes.v1",
        completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=snapshot["source_sha256"],
        passes=entries,all_passed=all(x["status"]=="pass" for x in entries),
        independence="Nine grouped software/numerical test passes plus end-to-end artifact pass. Independent human/agent reviews are separate artifacts.",
        scientific_status="integration verification only; research policy comparison pending",
        design_archive_verified=False)
    write_json(output/"ten_pass_review.json",report)
    (output/"README.md").write_text("# Ten post-change acceptance passes\n\n"+
      "Exact source snapshot: "+snapshot["source_sha256"]+"\n\n"+
      "\n".join(f"{x['pass_number']}. {x['name']}: **{x['status']}**" for x in entries)+
      "\n\nThese are topical acceptance passes, not ten independent scientific studies. Source hashes, exact test IDs and logs are in this folder. Independent reviewers have separately saved findings under audits/reviews. Overall publication/release readiness is not asserted.\n",encoding="utf8")
    return 0 if report["all_passed"] else 1

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--run",required=True,type=Path)
    args=parser.parse_args()
    raise SystemExit(run_review(args.output,args.run))

