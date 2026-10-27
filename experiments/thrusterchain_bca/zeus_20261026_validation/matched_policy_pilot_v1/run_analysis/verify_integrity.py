"""Read-only post-run hashes/accounting; no model import or scientific execution."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = []


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    checks.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


spec = importlib.util.spec_from_file_location("pilot_readonly_launch", ROOT / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
protocol, prepared = launch.verify_prepared()
identity = launch.verify_inputs()  # File hashes/metadata only, no restore or forwards.
check("frozen_prepared_manifest", sha(ROOT / "PREPARED.json") == "e7ce56db788e96703531c3aaf506d462b9218437495b597c220c476a03033579")
protocol_sha = sha(ROOT / "protocol.json")
result = read(ROOT / "RUN_RESULT.json")
check("completed_under_global_cap", result["status"] == "completed_bounded_matched_policy_pilot" and result["elapsed_seconds"] < protocol["total_timeout_seconds"])
check("protocol_and_prepared_unchanged", result["protocol_sha256"] == protocol_sha and result["prepared_identity_unchanged"])
check("all_job_identities", [j["job"] for j in result["jobs"]] == protocol["run_order"] + ["evaluation"])
totals = {}
training = {}
for job in result["jobs"]:
    name = job["job"]
    folder = ROOT / "run" / name
    report = read(folder / "report.json")
    supervisor = read(folder / "SUPERVISOR_RESULT.json")
    check(name + " completed", job["completed"] and job["exit_code"] == 0 and not job["timed_out"] and not job["interrupted"] and job["execution_counts_known"])
    check(name + " report seal", sha(folder / "report.json") == job["report_sha256"] == supervisor["report_sha256"])
    check(name + " worker cap", job["elapsed_seconds"] < 600)
    check(name + " protocol", report["protocol_sha256"] == protocol_sha)
    if name == "evaluation":
        continue
    expected = {"critic": 2000, "actor": 1000, "actor_target": 1000, "critic_target": 1000,
                "outer_steps": 1000, "auxiliary": 2000, "aux_fit_attempts": 2000,
                "refresh_attempts": 1, "refresh_accepted": 1}
    check(name + " exact updates", report["additional_updates"] == expected == job["ledger"]["committed_additional_updates"])
    for key, value in expected.items():
        totals[key] = totals.get(key, 0) + value
    check(name + " no training simulator use", report["ledger"]["simulator_transitions"] == 0 and report["ledger"]["evaluation_episodes"] == 0 and report["ledger"]["attempted_uncommitted_updates"] == {})
    check(name + " valid terminal checks", all(report["checks"].values()) and report["original_checkpoint_unchanged"]["full_state_unchanged"])
    actor = read(folder / "actor_manifest.json")
    state = read(folder / "state_manifest.json")
    check(name + " manifest seal", sha(folder / "actor_manifest.json") == report["actor_manifest_sha256"] and sha(folder / "state_manifest.json") == report["state_manifest_sha256"])
    for key in ("actor_params", "final_state"):
        artifact = actor[key]
        check(name + " " + key, sha(folder / artifact["path"]) == artifact["sha256"] and (folder / artifact["path"]).stat().st_size == artifact["bytes"])
    check(name + " original UniFloral identity", actor["native_source_sha256"] == protocol["native_source_sha256"] and actor["native_revision"] == protocol["native_revision"] and actor["initial_checkpoint_state_sha256"] == protocol["checkpoint_state_sha256"])
    restore = read(folder / "restore_receipt.json")
    check(name + " exact restored leaves", restore["restore"]["all_numeric_leaves_match"] and restore["restore"]["state_sha256"] == protocol["checkpoint_state_sha256"])
    check(name + " GPU runtime", restore["runtime"]["backend"] == "gpu" and restore["runtime"]["executable"] == "/home/david/bca-operator-audit-20261026/bin/python")
    refresh = read(folder / "refresh_receipt.json")
    check(name + " midpoint calibration", refresh["relative_critic"] == 1000 and refresh["absolute_critic"] == 21000 and refresh["details"]["flags"]["posterior_inputs_valid"] and refresh["details"]["flags"]["posterior_certified"])
    training[name] = {"rng_fork": report["rng_fork"], "initial_reference": report["initial_reference"],
        "final_reference": report["final_reference"], "normalization": actor["normalization"],
        "last_actor_metrics": report["last_update_metrics"]["native"], "seconds": job["elapsed_seconds"]}
for index in (0, 1):
    check("stream paired RNG " + str(index), all(training[f"seed{index}_{arm}"]["rng_fork"] == training[f"seed{index}_native"]["rng_fork"] for arm in ("bca", "distance")))
check("all shared initial calibration and normalization", all(t["initial_reference"] == training["seed0_native"]["initial_reference"] and t["normalization"] == training["seed0_native"]["normalization"] for t in training.values()))
evaluation = read(ROOT / "run/evaluation/report.json")
check("evaluation counts", evaluation["counts"]["completed_transition_rows"] == 22400 and evaluation["counts"]["completed_episodes"] == 224 and evaluation["counts"]["inflight_transition_rows"] == 0 and evaluation["counts"]["optimizer_updates"] == 0 and evaluation["counts"]["calibration_updates"] == 0 and evaluation["failure_count"] == 0)
check("evaluation immutability", evaluation["actor_parameters_unchanged"] and evaluation["immutability"]["full_state_unchanged"])
for name, artifact in {**evaluation["policy_results"], "reset_states": evaluation["reset_states"]}.items():
    check("evaluation cached artifact " + name, sha(ROOT / "run/evaluation" / artifact["path"]) == artifact["sha256"])
summary = {"status": "passed_readonly_post_run_integrity", "checks_passed": len(checks),
    "protocol_sha256": protocol_sha, "run_result_sha256": sha(ROOT / "RUN_RESULT.json"),
    "evaluation_report_sha256": sha(ROOT / "run/evaluation/report.json"),
    "elapsed_seconds": result["elapsed_seconds"], "total_training_updates": totals,
    "evaluation_counts": evaluation["counts"], "training": training,
    "checks": checks, "scope": "Saved hashes/JSON only; zero new training, model forwards, calibration, simulation, WSL or confirmation access."}
(ROOT / "run_analysis/INTEGRITY_REVIEW.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: summary[key] for key in ("status", "checks_passed", "protocol_sha256", "run_result_sha256", "evaluation_report_sha256", "elapsed_seconds", "total_training_updates")}))
