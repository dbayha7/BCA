"""Read-only, reproducible analysis of 22 local Zeus jobs + eight supplied results.

No model imports, checkpoint execution, optimization, source edits, or remote
access occur. Missing identities/results fail validation; no job is dropped.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

TITAN_SHA256 = "1f65b23ab78d06a16390941a9b146b6ec80e7cde6d44e0e790e92f64b15ca2ee"
CONTEXTS = ("stall_low", "stall_high", "monotone_low", "monotone_high")
RETENTIONS = ("full", "rare", "zero")
ARMS = ("td3_bc", "bca")
SPLITS = ("final", "validation")
EXPECTED_TITAN = {f"replicate_{r:02d}/{retention}/{arm}" for r in range(2, 6)
                  for retention, arm in (("full", "bca"), ("zero", "td3_bc"))}
DEFAULT_RUN = Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_diagnostics\training_runs\core_20261020T2250Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def close(actual, expected, name):
    require(np.allclose(actual, expected, rtol=1e-11, atol=1e-10), f"arithmetic mismatch: {name}")


def sample_stats(values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (5,) and np.isfinite(values).all(), "require all five finite replicate values")
    return dict(values=values.tolist(), mean=float(values.mean()), sample_sd=float(values.std(ddof=1)))


def paired_stats(baseline, bca):
    baseline, bca = np.asarray(baseline, float), np.asarray(bca, float)
    require(baseline.shape == bca.shape == (5,) and np.isfinite(baseline).all() and np.isfinite(bca).all(),
            "paired arms require five finite values each; broadcasting/missing pairs forbidden")
    stats = sample_stats(bca - baseline)
    margin = float(student_t.ppf(.975, 4) * stats["sample_sd"] / np.sqrt(5))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=5)))
    permutation_means = np.mean(signs * np.asarray(stats["values"]), axis=1)
    p = float(np.mean(np.abs(permutation_means) >= abs(stats["mean"]) - 1e-12))
    return dict(mean_difference=stats["mean"], sample_sd_difference=stats["sample_sd"],
                differences=stats["values"], ci95=[stats["mean"] - margin, stats["mean"] + margin],
                n=5, df=4, positive_differences=int(np.sum(bca > baseline)),
                sign_flip_two_sided_p=p,
                inference="exploratory unadjusted paired-t interval, df=4, trained seed unit; conditional on fixed banks; approximate normal differences assumption",
                sign_flip_assumption="all 32 sign flips under symmetric/exchangeable paired null; host confounding prevents clean method-causal interpretation")


class Evidence:
    def __init__(self):
        self.files = {}

    def file(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None:
            require(digest == expected.lower(), f"file hash mismatch: {path}")
        self.files[str(path)] = dict(sha256=digest, bytes=path.stat().st_size)
        return digest

    def json(self, path, expected=None):
        self.file(path, expected)
        return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_summary(summary, split, banks, oracle, job=None):
    require(summary["split"] == split, "evaluation split differs")
    require(summary["bank_sha256"] == banks[split]["arrays_sha256"], "evaluation bank differs")
    objective = summary["objective"]
    require(objective["horizon"] == 100 and objective["gamma"] == 1.0
            and objective["terminal_bootstrap"] is False and objective["terminal_reward"] == 0,
            "objective differs")
    n = 400 if split == "final" else 80
    require(summary["pooled"]["starts"] == n and summary["pooled"]["completed"] == n, "missing completed starts")
    require(set(summary["strata"]) == set(CONTEXTS), "context stratum missing")
    for context in CONTEXTS:
        require(summary["strata"][context]["starts"] == n // 4, "stratum count differs")
    for metric in ("mean_return", "failure_rate", "high_action_fraction", "critical_fraction"):
        close(summary["pooled"][metric], np.mean([summary["strata"][c][metric] for c in CONTEXTS]), metric)
    if job is not None:
        require(summary["job"] == job, "summary job identity differs")
        require(summary["checkpoint"]["update"] == (100000 if split == "final" else 50000), "wrong checkpoint budget")
    if split == "final" and job is not None:
        q = summary["critic_diagnostics"]
        require(q["status"] == "completed" and q["bank_sha256"] == banks["query"]["arrays_sha256"], "critic diagnostic identity/status differs")
        require(q["query_states"] == 48 and q["candidates"] == 6, "query design differs")
        cal = summary["calibration_diagnostics"]
        require(cal["query_bank_sha256"] == banks["query"]["arrays_sha256"], "calibration query bank differs")
        require(cal["status"] == ("evaluated" if job["arm"] == "bca" else "absent"), "calibration status requires explicit handling")
        if job["arm"] == "bca":
            require(cal["checkpoint_update"] == 100000 and cal["last_refresh_update"] == 95000
                    and cal["seed"] == 102001, "calibration checkpoint/noise identity differs")
        for context in ("pooled",) + CONTEXTS:
            metrics = summary["pooled"] if context == "pooled" else summary["strata"][context]
            lo, hi = oracle["means"][context]
            close(metrics["regret_lower"], lo - metrics["mean_return"], "regret lower")
            close(metrics["regret_upper"], hi - metrics["mean_return"], "regret upper")


def verify_local_arrays(path, summary, evidence):
    evidence.file(path, summary["file_sha256"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(array_hash(arrays) == summary["arrays_sha256"], "local evaluation arrays hash differs")
    for context in ("pooled",) + CONTEXTS:
        mask = np.ones(len(arrays["context"]), bool) if context == "pooled" else arrays["context"] == CONTEXTS.index(context)
        metrics = summary["pooled"] if context == "pooled" else summary["strata"][context]
        for key, array_key in (("mean_return", "return_value"), ("failure_rate", "failure"),
                               ("high_action_fraction", "high_action_fraction"), ("critical_fraction", "critical_fraction"),
                               ("regret_lower", "regret_lower"), ("regret_upper", "regret_upper")):
            if key in metrics:
                close(metrics[key], np.mean(arrays[array_key][mask]), f"local {context} {key}")
        close(metrics["return_std"], np.std(arrays["return_value"][mask], ddof=1), "within-bank SD")
    require(np.array_equal(arrays["failure"], (arrays["max_abs_position"] > 3) | ~arrays["completed"]), "failure rule differs")
    require(np.all(arrays["finite"]) and np.all(arrays["steps"] == 100), "invalid/incomplete local rollouts")
    if "q_truth" in arrays:
        q = summary["critic_diagnostics"]
        predictions = arrays["q_predictions"]
        for k, name in enumerate(("critic_1", "critic_2", "twin_min")):
            pred = predictions[..., k] if k < 2 else predictions.min(axis=-1)
            error = pred - arrays["q_truth"]
            for key, actual in (("rmse", np.sqrt(np.mean(error ** 2))), ("mae", np.mean(np.abs(error))), ("mean_error", np.mean(error))):
                close(q["q_error"][name][key], actual, key)
        close(q["bellman_residual"]["rmse"], np.sqrt(np.mean(arrays["q_bellman_residual"] ** 2)), "deterministic Bellman residual")
        close(q["action_ranking"]["mean_true_regret_of_predicted_best"], np.mean(arrays["q_ranking_regret"]), "ranking regret")
    if "calibration_covered" in arrays:
        close(summary["calibration_diagnostics"]["coverage"], np.mean(arrays["calibration_covered"]), "native noisy residual coverage")
    return arrays


def metric_values(summary, context):
    values = dict(summary["pooled"] if context == "pooled" else summary["strata"][context])
    values.pop("return_std", None)  # within-rollout spread is not replicate SD
    q = summary.get("critic_diagnostics", {})
    if context == "pooled" and q.get("status") == "completed":
        for critic, metrics in q["q_error"].items():
            values.update({f"{critic}_{key}": value for key, value in metrics.items()})
        values.update(bellman_residual_rmse=q["bellman_residual"]["rmse"],
                      bellman_residual_mae=q["bellman_residual"]["mae"],
                      action_ranking_regret=q["action_ranking"]["mean_true_regret_of_predicted_best"],
                      action_ranking_accuracy=q["action_ranking"]["pairwise_accuracy_excluding_true_ties"])
    cal = summary.get("calibration_diagnostics", {})
    if cal.get("status") == "evaluated":
        cm = cal["pooled"] if context == "pooled" else cal["strata"][context]
        values.update({f"calibration_{key}": value for key, value in cm.items()})
        if context == "pooled":
            values.update(band_refusal_rate=cal["band_refusal_rate"], frozen_threshold=cal["frozen_threshold"])
    return values


def aggregate(run, titan_path, *, source_root=None, evidence_only=False):
    run = Path(run).resolve()
    evidence = Evidence()
    titan = evidence.json(titan_path, TITAN_SHA256)
    config = evidence.json(run / "config.json")
    provenance = evidence.json(run / "provenance.json")
    matrix = evidence.json(run / "job_matrix.json")
    prepared = evidence.json(run / "prepared_manifest.json")
    require(len(matrix) == 30 and len({j["job_id"] for j in matrix}) == 30, "require exactly 30 unique planned jobs")
    require(config["source_episodes"] == 1024 and config["reservation_episodes"] == 128
            and config["settings"]["updates"] == 100000 and config["settings"]["batch_size"] == 256,
            "frozen training design differs")
    require(set(t["job_id"] for t in titan["jobs"]) == EXPECTED_TITAN and len(titan["jobs"]) == 8, "supplied report job allocation differs")
    require(titan["source_run"] == run.name and titan["completed_jobs"] == 8
            and titan["actual_training_backend"] == "cpu", "supplied report run/backend differs")
    require(all(titan["verification"][key] is True for key in
                ("source_closure", "all_64_frozen_preparation_files", "all_final_checkpoint_hashes",
                 "all_midpoint_snapshot_hashes", "all_evaluation_array_hashes", "all_evaluation_statuses_completed")),
            "supplied report does not affirm all expected source/config/evaluation checks")
    require(all(titan["backend_decision"]["cpu"][arm]["within_tolerance"] is True for arm in ARMS),
            "supplied report CPU fixture did not pass")
    omitted_files = []
    prepared_verified = 0
    for relative, expected in prepared["files"].items():
        path = run / relative
        if evidence_only and not path.exists() and (relative.startswith("audit/") or relative.endswith("/learner.npz")):
            omitted_files.append(dict(kind="training_data_or_source_audit", path=relative, recorded_sha256=expected))
        else:
            evidence.file(path, expected)
            prepared_verified += 1
    require(len(prepared["files"]) == 64 and prepared["ready"], "frozen preparation incomplete")
    repository = Path(source_root).resolve() if source_root is not None else run.parents[3]
    for source, expected in provenance["source_sha256"].items():
        relative = source if "/" in source else source.replace(".", "/") + ".py"
        evidence.file(repository / relative, expected)
    banks = {split: evidence.json(run / "banks" / split / "bank.json") for split in ("final", "validation", "query")}
    for split, bank in banks.items():
        evidence.file(run / "banks" / split / bank["file"], bank["file_sha256"])
    optima = evidence.json(run / "oracle/final.json")
    require(optima["bank_sha256"] == banks["final"]["arrays_sha256"] and len(optima["solutions"]) == 400, "oracle bank differs")
    lower, upper = np.asarray(optima["return_lower"]), np.asarray(optima["return_upper"])
    require(lower.shape == upper.shape == (400,) and np.all(upper >= lower), "oracle interval malformed")
    with np.load(run / "banks/final/bank.npz", allow_pickle=False) as archive:
        final_context = archive["contexts"]
    oracle = dict(bank_sha256=optima["bank_sha256"], starts=400,
                  means={c: [float(np.mean(lower[m])), float(np.mean(upper[m]))]
                         for c, m in [("pooled", np.ones(400, bool))] + [(c, final_context == k) for k, c in enumerate(CONTEXTS)]},
                  maximum_fw_gap=max(s["fw_gap"] for s in optima["solutions"]),
                  maximum_replay_error=max(s["replay_error"] for s in optima["solutions"]),
                  certificate="floating-point convex Frank-Wolfe gap; not outward-rounded intervals or population optimality")
    titan_jobs = {job["job_id"]: job for job in titan["jobs"]}
    jobs, count_rows, seed_rows = [], [], []
    for job in matrix:
        jid, retention, arm = job["job_id"], job["retention"], job["arm"]
        prep = evidence.json(run / "data" / f"seed_{job['data_seed']}" / retention / "preparation.json")
        pm = prep["manifest"]
        require(pm["calibration_rows"] == 128 and pm["training_episodes"] == 896 and pm["budget_topups"] == 0, "preparation counts differ")
        if jid in titan_jobs:
            reported = titan_jobs[jid]
            require(reported["job"] == job and reported["status"] == "completed", "supplied job identity/status differs")
            evals = reported["evaluations"]
            record = dict(job_id=jid, job=job, host="Titan", backend="cpu", status=reported["status"],
                source=dict(kind="user_supplied_report", path=str(Path(titan_path).resolve()), sha256=TITAN_SHA256,
                            independently_verified_arrays=False, independently_verified_checkpoint=False,
                            verification_scope="report file hash, job/bank/checkpoint metadata consistency and paired initial identities checked locally; source/config/array/checkpoint verification on Titan is reported, not independently rerun"),
                updates=reported["accepted_critic_updates"], actor_updates=reported["actor_updates"],
                training_seconds=reported["training_seconds"], checkpoint=reported["final_checkpoint"],
                initial_native_sha256=reported["initial_native_sha256"], initial_prng_sha256=reported["initial_prng_sha256"],
                final_prng_sha256=reported["final_prng_sha256"], refresh_audit=None)
            require(record["checkpoint"] == evals["final"]["checkpoint"], "reported final checkpoint differs")
        else:
            manifest_path = run / "jobs" / jid / "manifest.json"
            manifest = evidence.json(manifest_path)
            require(manifest["status"] == "completed", f"training incomplete: {jid}")
            require(manifest["identity"]["data"] == pm, f"prepared data identity differs: {jid}")
            require(manifest["identity"]["arm"] == arm and manifest["identity"]["seed"] == job["optimization_seed"], "optimization identity differs")
            for source, expected in manifest["identity"]["source_sha256"].items():
                require(provenance["source_sha256"][source] == expected, "training source identity differs")
            require(manifest["runtime"]["devices"] == ["TFRT_CPU_0"], "unexpected local backend")
            record = dict(job_id=jid, job=job, host="Zeus", backend="cpu", status=manifest["status"],
                source=dict(kind="local_manifest_and_evaluation_arrays", path=str(manifest_path), sha256=sha256(manifest_path),
                            independently_verified_arrays=True, independently_verified_checkpoint=True,
                            verification_scope="local file/array/checkpoint hashes and summary arithmetic verified; no new rollout or training in this analysis"),
                updates=manifest["final_update"], actor_updates=manifest["counters"]["actor"],
                training_seconds=manifest["training_seconds"], checkpoint=manifest["checkpoint"],
                initial_native_sha256=manifest["initial_native_sha256"], initial_prng_sha256=manifest["initial_prng_sha256"],
                final_prng_sha256=manifest["prng_sha256"], refresh_audit=dict(count=len(manifest["refreshes"]),
                    numerically_certified=sum(bool(x.get("posterior_certified")) for x in manifest["refreshes"]),
                    inputs_valid=all(x.get("posterior_inputs_valid") for x in manifest["refreshes"])))
            evals = {}
            for split in SPLITS:
                summary_path = run / "evaluation/jobs" / jid / split / "summary.json"
                summary = evidence.json(summary_path)
                evals[split] = summary
                status = evidence.json(run / "evaluation_status/jobs" / jid / (split + ".json"))
                require(status["status"] == "completed", "local evaluation status incomplete")
                checkpoint = summary["checkpoint"]
                matching = [s for s in manifest["snapshots"] if s["update"] == checkpoint["update"]]
                require(len(matching) == 1, "checkpoint snapshot absent")
                for field in ("file", "sha256", "actor_sha256", "update"):
                    require(checkpoint[field] == matching[0][field], f"evaluation snapshot identity differs: {field}")
                checkpoint_path = manifest_path.parent / checkpoint["file"]
                if evidence_only and not checkpoint_path.exists():
                    omitted_files.append(dict(kind="checkpoint", path=str(checkpoint_path.relative_to(run)), recorded_sha256=checkpoint["sha256"]))
                    record["source"]["independently_verified_checkpoint"] = False
                    record["source"]["verification_scope"] = "bundled local manifest/snapshot/evaluation metadata and array arithmetic verified; checkpoint bytes omitted, so checkpoint identity is metadata-only in this rerun"
                else:
                    evidence.file(checkpoint_path, checkpoint["sha256"])
                verify_local_arrays(summary_path.parent / summary["file"], summary, evidence)
        require(record["updates"] == 100000 and record["actor_updates"] == 50000, "terminal update counts differ")
        record.update(evaluation=evals, preparation=pm,
                      preparation_source="canonical frozen pretraining manifest; Titan use is supported by supplied report's preparation-verification assertion")
        for split in SPLITS:
            verify_summary(evals[split], split, banks, oracle, job=job)
            for context in ("pooled",) + CONTEXTS:
                seed_rows.append(dict(job_id=jid, replicate=job["replicate"], retention=retention, arm=arm,
                                      host=record["host"], split=split, context=context, **metric_values(evals[split], context)))
        jobs.append(record)
        if arm == "td3_bc":
            count_rows.append(dict(replicate=job["replicate"], retention=retention, data_seed=job["data_seed"],
                **{k: pm[k] for k in ("retained_rows", "training_rows", "calibration_rows", "training_episodes", "calibration_episodes",
                                      "training_critical_rows", "calibration_critical_rows", "bca_refusal", "native_numerical_certifiability")},
                calibration_context_counts=pm["calibration_context_counts"], calibration_critical_context_counts=pm["calibration_critical_context_counts"]))
    require(len(jobs) == 30 and len({j["job_id"] for j in jobs}) == 30, "need all 30 unique matched jobs")
    index = {j["job_id"]: j for j in jobs}
    for r in range(1, 6):
        for retention in RETENTIONS:
            baseline, bca = [index[f"replicate_{r:02d}/{retention}/{arm}"] for arm in ARMS]
            for key in ("initial_native_sha256", "initial_prng_sha256", "final_prng_sha256"):
                require(baseline[key] == bca[key], "paired host initialization/PRNG identity differs")
            require(baseline["preparation"] == bca["preparation"], "paired preparation differs")
    comparisons, method_rows = [], []
    for split, retention, context in itertools.product(SPLITS, RETENTIONS, ("pooled",) + CONTEXTS):
        paired = []
        for r in range(1, 6):
            arm_jobs = [index[f"replicate_{r:02d}/{retention}/{a}"] for a in ARMS]
            vals = [metric_values(j["evaluation"][split], context)["mean_return"] for j in arm_jobs]
            paired.append(dict(replicate=r, td3_bc=vals[0], bca=vals[1], delta=vals[1] - vals[0],
                               td3_bc_host=arm_jobs[0]["host"], bca_host=arm_jobs[1]["host"]))
        comparisons.append(dict(split=split, retention=retention, context=context, replicates=paired,
            host_confounded_pairs=sum(x["td3_bc_host"] != x["bca_host"] for x in paired),
            **paired_stats([p["td3_bc"] for p in paired], [p["bca"] for p in paired])))
        for arm in ARMS:
            values = [metric_values(index[f"replicate_{r:02d}/{retention}/{arm}"]["evaluation"][split], context) for r in range(1, 6)]
            common = set.intersection(*(set(v) for v in values))
            metrics = {key: sample_stats([v[key] for v in values]) for key in sorted(common)
                       if all(isinstance(v[key], (float, int)) and not isinstance(v[key], bool) for v in values)}
            method_rows.append(dict(split=split, retention=retention, arm=arm, context=context, n=5, metrics=metrics))
    controls = []
    for split, name in itertools.product(SPLITS, config["controls"]):
        path = run / "evaluation/controls" / split / name / "summary.json"
        summary = evidence.json(path)
        verify_summary(summary, split, banks, oracle)
        verify_local_arrays(path.parent / summary["file"], summary, evidence)
        derived = ({context: [value - metric_values(summary, context)["mean_return"] for value in oracle["means"][context]]
                    for context in ("pooled",) + CONTEXTS} if split == "final" else None)
        controls.append(dict(name=name, split=split, evaluation=summary, derived_regret_bounds=derived,
                             role="untrained known-model finite-horizon LQR gains followed by clipping; not a data-matched learned benchmark"))
    limits = [
        "Five independent training seeds are the replication unit. The 400 shared final starts are not 400 independent training replicates; uncertainty is conditional on this evaluation bank.",
        "Intervals are exploratory unadjusted paired-t 95% intervals with df=4 and approximate normal-difference assumptions. Multiple contexts/retentions are not multiplicity-corrected. The 32-sign-flip diagnostic additionally assumes a symmetric/exchangeable paired null.",
        "Full retention replicates 2–5 have BCA on Titan and TD3+BC on Zeus; zero retention replicates 2–5 have the inverse allocation. Both hosts used CPU and Titan's report states the fixed fixture passed tolerance, but host/OS/runtime differences remain confounded with method. Rare retention and replicate 1 are same-host Zeus comparisons.",
        "Eight Titan jobs come from the user-supplied, SHA256-verified report. Their raw arrays, checkpoints and remote source/config verification were not independently accessed or rerun here; available bank, job, checkpoint and paired initialization/PRNG metadata were checked.",
        "Known-model fixed-cap and regime-cap controllers use the dynamics and finite-horizon LQR gains; they are informative untrained controls, not data-matched learned baselines.",
        "Full means no deliberate deletion, not full continuous visitation. Behavior has categorical command atoms; learned continuous actions do not acquire a continuous-density support guarantee.",
        "The pooled calibration bank has 128 rows from 128 reserved episodes, reused adaptively. Rare critical counts are 1,3,0,0,1 and zero counts are all zero. Numerical posterior readiness/certifiability does not certify critical-region or population coverage.",
        "True-Q error uses forced action then the deployed deterministic policy to H; it is Q^pi, not Q*. The deterministic Bellman residual uses the deployed actor and online twin-min critic. Empirical calibrated coverage targets the distinct native noisy TD3 target-network residual with the frozen 95k reference. None implies long-horizon Q coverage.",
        "Each method's Q target follows its own deployed policy, and the sixth candidate is its own actor action. Cross-method Q-error differences therefore do not isolate critic estimation quality under one common target. The designed query distribution differs from the pooled behavior calibration distribution; below-nominal empirical query coverage does not by itself refute a conformal theorem.",
        "Q diagnostics are pooled over the 48-state × 6-candidate designed bank. Matched per-context Q summaries are unavailable for Titan, so no all-30 stratum Q statistics or zero imputations are supplied.",
        "The failure event is any absolute position coordinate state[0:10:2] exceeding 3 before/after a step; episodes continue to H=100. Zero observed failures is not a safety certificate.",
        "Continuous regret uses the same 400 fixed-start convex QPs and numerical Frank-Wolfe bounds. It is not an analytic continuous-population optimum or outward-rounded interval proof.",
        "100000-update terminal checkpoints remain primary. 50000-update validation uses a separate 80-start bank and is descriptive only; its difference from final conflates checkpoint and bank. No checkpoint selection, tuning, retraining, or causal outcome-driven recommendation is performed."
    ]
    result = dict(schema="thrusterchain-combined-results-analysis-v1", complete=True, source_run=run.name,
        generated_utc=datetime.now(timezone.utc).isoformat(), config=config, provenance=provenance,
        jobs=jobs, paired_comparisons=comparisons, method_summary=method_rows, controls=controls,
        calibration_data_counts=count_rows, oracle=oracle, banks=banks, limitations=limits,
        verification=dict(planned_unique_jobs=30, local_jobs=22, supplied_report_jobs=8,
                          terminal_updates=3000000, actor_updates=1500000, final_evaluations=30, midpoint_evaluations=30,
                          prepared_file_hashes_verified=prepared_verified, all_pairs_initialization_and_prng_match=True,
                          all_local_checkpoint_and_evaluation_array_hashes_verified=not any(f["kind"] == "checkpoint" for f in omitted_files),
                          all_local_evaluation_array_hashes_verified=True,
                          all_local_summary_arithmetic_verified=True,
                          verification_mode="bundled_evidence_subset" if evidence_only else "full_local_audit",
                          omitted_evidence_files=omitted_files,
                          titan_report_sha256=TITAN_SHA256, titan_reported_verification=titan["verification"],
                          titan_config_source_verification="reported source closure and frozen preparation checks; direct config digest absent from supplied summary report",
                          no_remote_titan_access=True, analysis_only=True),
        evidence=dict(files=evidence.files, analysis_script_sha256=sha256(__file__)),
        metric_semantics=dict(reward="-(sum(state[:10]**2)+0.3*sum(action**2))", horizon=100, gamma=1,
            paired_difference="BCA minus TD3+BC; positive is better return", seed_sd="sample SD of five trained seed means, ddof=1",
            q_rmse_mean="mean of five per-seed query RMSEs, not RMSE after pooling errors",
            high_action_fraction="fraction of steps with any |command|>KNEE; above-knee is useful in monotone high-load context, not intrinsically unsafe",
            failure="max absolute position >3 or incomplete episode; no early termination"))
    return result, seed_rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def report_text(result):
    comparisons = [c for c in result["paired_comparisons"] if c["split"] == "final" and c["context"] == "pooled"]
    methods = {(m["retention"], m["arm"], m["context"]): m["metrics"] for m in result["method_summary"] if m["split"] == "final"}
    lines = ["# Continuous ThrusterChain: completed core comparison", "",
        "All 30 planned jobs reached 100,000 critic updates and 50,000 actor updates. Terminal results show no pooled return advantage for BCA in any retention condition. Both known-model, untrained capped-LQR controls outperform every learned policy on the shared final bank.", "",
        "The primary endpoint is raw undiscounted return over H=100, averaged equally across four contexts and 400 common starts. Larger (less negative) is better. Each mean and SD below uses five independent training replicates.", "",
        "| Retention | TD3+BC mean ± SD | BCA mean ± SD | Paired BCA − TD3+BC | Exploratory 95% paired-t CI | Positive pairs |", "|---|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        a, b = [methods[(c["retention"], arm, "pooled")]["mean_return"] for arm in ARMS]
        lines.append(f"| {c['retention']} | {a['mean']:.3f} ± {a['sample_sd']:.3f} | {b['mean']:.3f} ± {b['sample_sd']:.3f} | {c['mean_difference']:+.3f} | [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | {c['positive_differences']}/5 |")
    lines += ["", "Intervals use paired seed differences, df=4; they are exploratory, unadjusted for multiple comparisons, approximate, and conditional on this fixed evaluation bank. Zero retention has five negative differences, but four pairs compare different hosts. Its interval excluding zero does not isolate a BCA causal effect. Rare retention is entirely Zeus and its pooled interval includes zero.", "",
              "## Every terminal seed result", "", "| Retention | Replicate | TD3+BC return | BCA return | Difference | Hosts TD3 / BCA |", "|---|---:|---:|---:|---:|---|"]
    for c in comparisons:
        for p in c["replicates"]:
            lines.append(f"| {c['retention']} | {p['replicate']} | {p['td3_bc']:.6f} | {p['bca']:.6f} | {p['delta']:+.6f} | {p['td3_bc_host']} / {p['bca_host']} |")
    lines += ["", "## Contexts, controls, and regret", "", "| Retention | Stall low Δ | Stall high Δ | Monotone low Δ | Monotone high Δ |", "|---|---:|---:|---:|---:|"]
    for retention in RETENTIONS:
        means = [next(c["mean_difference"] for c in result["paired_comparisons"] if c["split"] == "final" and c["retention"] == retention and c["context"] == context) for context in CONTEXTS]
        lines.append("| " + retention + " | " + " | ".join(f"{x:+.3f}" for x in means) + " |")
    lines += ["", "BCA's rare-retention stall/high-load mean improves, while monotone/high-load return worsens in all three retention settings. This is a descriptive context pattern, not a selected endpoint or a multiplicity-adjusted discovery.", ""]
    for control in result["controls"]:
        if control["split"] == "final":
            lo, hi = control["derived_regret_bounds"]["pooled"]
            lines.append(f"- Known-model `{control['name']}`: return {control['evaluation']['pooled']['mean_return']:.6f}, mean fixed-start regret bounds [{lo:.6f}, {hi:.6f}].")
    lines += [f"- Shared 400-start QP optimum mean return bounds: [{result['oracle']['means']['pooled'][0]:.12f}, {result['oracle']['means']['pooled'][1]:.12f}]. Maximum numerical Frank-Wolfe gap: {result['oracle']['maximum_fw_gap']:.3g}.", "",
              "These controls use known dynamics and finite-horizon LQR gains, so they are not data-matched learned benchmarks. Full retention learned mean regret is about 77–81; rare/zero mean regret about 117–123. All per-seed/context bounds and failure frequencies are in the CSVs and JSON. The failure event is any |position|>3, with episodes still run to H; it is not an episode termination rule or a general safety criterion.", "",
              "## Value error and empirical calibration", "", "| Retention | Arm | True Q RMSE | Deterministic Bellman RMSE | Ranking regret | Failure frequency | Residual coverage |", "|---|---|---:|---:|---:|---:|---:|"]
    for retention, arm in itertools.product(RETENTIONS, ARMS):
        m = methods[(retention, arm, "pooled")]
        coverage = f"{m['calibration_coverage']['mean']:.3f}" if "calibration_coverage" in m else "absent"
        lines.append(f"| {retention} | {arm} | {m['twin_min_rmse']['mean']:.3f} | {m['bellman_residual_rmse']['mean']:.3f} | {m['action_ranking_regret']['mean']:.3f} | {100*m['failure_rate']['mean']:.3f}% | {coverage} |")
    lines += ["", "The table averages five per-seed diagnostics; it does not pool their errors before computing RMSE. True Q is a forced command followed by the deployed policy to H, not optimal Q. Small one-step Bellman residuals coexist with much larger long-horizon Q error, especially after critical-row deletion. Ranking regret compares the critic-selected command against the best true Q among six candidates.", "",
              "Each method's true Q follows its own policy and its sixth candidate is its own actor command, so cross-method Q-error differences do not isolate critic quality under one shared target. The 288 query pairs are 48 designed states crossed with six candidates, not 288 independent draws.", "",
              "Coverage targets a different residual: the native noisy TD3 target-network target minus current twin-min Q, compared with the frozen checkpoint band (last refresh 95k). Every BCA run has empirical query coverage below nominal 0.9; this does not establish any long-horizon Q coverage. The designed query law differs from the behavior calibration law, so this is not by itself a refutation of a conformal theorem. The pooled 128-row calibration bank is adaptively reused. Rare critical counts are 1, 3, 0, 0, 1; zero counts are all 0. Numerical readiness is not critical-region certification. All BCA final band refusal rates are zero and native usability is one; no executed actions are refused, and these values are not statistical support guarantees.", "",
              "## Provenance and limits", "",
              "The analysis combines 22 locally verified Zeus results with eight results in the user-supplied Titan report. The report SHA256 is `" + TITAN_SHA256 + "`. Local source/preparation/checkpoint/evaluation hashes and summary arithmetic were verified. Titan job, bank, checkpoint and paired initialization/PRNG metadata were checked against available records; Titan raw arrays and checkpoints were not independently accessed or rerun. Both hosts trained on CPU. The reported fixture tolerance check does not remove host/OS/runtime confounding.", "",
              "The 50k validation results are descriptive and use a separate 80-start bank. No checkpoint selection, tuning, retraining, or modifications to frozen records occurred. Per-context matched Q summaries are unavailable for the eight supplied jobs; they are not imputed. Full context return/coverage summaries remain available.", "",
              "Reproduce against the complete local run with `python aggregate_results.py --run-dir <run-directory> --titan-report ../inputs/titan_results.json --output .`. A portable evidence bundle can use `--source-root <source-closure-directory> --evidence-only`; training archives and checkpoint bytes may be omitted only in that explicit mode, and every omission is recorded rather than claimed verified. The run folder keeps its original name `core_20261020T2250Z`. This script uses NumPy/SciPy only. `aggregate.json` preserves all summaries and source identities; CSVs expose every seed/context, paired differences, method statistics, calibration counts, and controls. Analytical next steps are to inspect the reported context/Q/calibration failures and design any future study separately; this completed run does not establish a BCA benefit.", ""]
    if result["verification"]["verification_mode"] == "bundled_evidence_subset":
        lines += ["This reproduction used an evidence subset. Evaluation arrays and included files were verified; omitted training archives and checkpoint bytes were not reverified. See `verification.omitted_evidence_files` for exact identities. The original full-local audit and independent audit must be retained separately as historical evidence.", ""]
        lines = [line.replace("Local source/preparation/checkpoint/evaluation hashes and summary arithmetic were verified.", "Included local source/preparation/evaluation hashes and summary arithmetic were verified; checkpoint bytes omitted from this reproduction were not reverified.") for line in lines]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--run-dir", dest="run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--titan", "--titan-report", dest="titan", type=Path, default=Path(__file__).resolve().parents[1] / "inputs/titan_results.json")
    parser.add_argument("--source-root", type=Path, help="repository root or portable frozen source closure root")
    parser.add_argument("--evidence-only", action="store_true", help="allow intentionally omitted training archives/source audit/checkpoints, recording each omission")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    require(args.run.resolve() not in args.output.resolve().parents and args.output.resolve() != args.run.resolve(), "analysis must not write inside canonical run")
    result, rows = aggregate(args.run, args.titan, source_root=args.source_root, evidence_only=args.evidence_only)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "RESULTS.md").write_text(report_text(result), encoding="utf-8")
    write_csv(args.output / "per_seed_metrics.csv", rows)
    write_csv(args.output / "paired_comparisons.csv", result["paired_comparisons"])
    write_csv(args.output / "method_summary.csv", [dict(split=m["split"], retention=m["retention"], arm=m["arm"], context=m["context"], metric=k, **v) for m in result["method_summary"] for k, v in m["metrics"].items()])
    write_csv(args.output / "calibration_counts.csv", result["calibration_data_counts"])
    write_csv(args.output / "controls.csv", [dict(control=c["name"], split=c["split"], context=context, **metric_values(c["evaluation"], context), derived_regret_bounds=None if c["derived_regret_bounds"] is None else c["derived_regret_bounds"][context]) for c in result["controls"] for context in ("pooled",) + CONTEXTS])
    print(json.dumps(dict(complete=True, jobs=len(result["jobs"]), evidence_files=len(result["evidence"]["files"]),
                         pooled=[c for c in result["paired_comparisons"] if c["split"] == "final" and c["context"] == "pooled"]), indent=2))


if __name__ == "__main__":
    main()
