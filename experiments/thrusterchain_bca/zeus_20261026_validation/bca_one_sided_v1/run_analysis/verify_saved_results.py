"""Independent saved-arithmetic review. Stdlib + NumPy only; no model imports.

Never runs a learner, calibrator, simulator, random draw or new method. Existing
posterior crossings and all original selections are checked against their saved
inputs. Writes only this directory's review receipt, with exclusive creation.
"""
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
RUN = PILOT / "run"
PAYLOAD = PILOT.parents[1] / "titan_20261026/payload"
AUDIT = PAYLOAD / "unifloral_operator_alignment_v1/results_20261026T221700_762125Z"
EXPORT = PAYLOAD / "unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/inference_export"
EXPECTED_REPORT = "320113721d15e54bed4047e9574e94fac6d9ed78eca6d140575f386ae065b011"
checks = 0
hashes = {}


def require(condition, label):
    global checks
    if not bool(condition):
        raise AssertionError(label)
    checks += 1


def equal(actual, expected, label, atol=0., rtol=0.):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol), label)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash(path, expected):
    observed = sha(path)
    require(observed == expected, "SHA256: " + str(path))
    hashes[str(path.relative_to(PILOT.parents[4])).replace("\\", "/")] = observed


def epsilon(a, b):
    return 1e-6 + 1e-6 * np.maximum(np.abs(a), np.abs(b))


def main():
    report = read(RUN / "report.json")
    check_hash(RUN / "report.json", EXPECTED_REPORT)
    prepared = read(PILOT / "PREPARED.json")
    for name, digest in prepared["files"].items():
        check_hash(PILOT / name, digest)
    for name, digest in report["artifact_sha256"].items():
        check_hash(RUN / name, digest)
    terminal = read(PILOT / "RUN_RESULT.json")
    require(terminal["exit_code"] == 0 and not terminal["timed_out"] and terminal["execution_counts_known"], "terminal status")
    require(terminal["report_sha256"] == EXPECTED_REPORT and terminal["prepared_identity_unchanged"], "terminal identity")
    require(report["status"] == "completed_bounded_one_sided_bca_validation", "worker complete")
    require(report["elapsed_seconds"] < 300, "worker time cap")

    identity = report["restore"]["identity"]
    manifest_entries = 0
    for folder, rec in identity["sources"].items():
        manifest = PAYLOAD / folder / "prepared_files.json"
        check_hash(manifest, rec["manifest_sha256"])
        files = read(manifest)["files"]
        require(len(files) == rec["verified_files"], "source manifest count")
        manifest_entries += len(files)
        for rel, digest in files.items():
            check_hash(PAYLOAD / folder / rel, digest)
    provenance = PAYLOAD / "unifloral_td3bc_validation_v1/provenance/unifloral.json"
    check_hash(provenance, identity["native_manifest_sha256"])
    native = read(provenance)
    require(native["revision"] == identity["native_revision"], "native revision")
    vendor = PAYLOAD / "unifloral_td3bc_validation_v1/upstream"
    require({p.relative_to(vendor).as_posix() for p in vendor.rglob("*") if p.is_file()} == set(native["source_sha256"]), "native file inventory")
    require(len(native["source_sha256"]) == identity["native_snapshot_files"] == 40, "native file count")
    for rel, digest in native["source_sha256"].items():
        check_hash(vendor / rel, digest)
    for rel, digest in identity["evidence"].items():
        check_hash(PAYLOAD / rel, digest)
    checkpoint = PAYLOAD / "unifloral_checkpoint_v1/train_20261023T183248_552877Z/checkpoints/critic_020000_terminal"
    check_hash(checkpoint / "COMMITTED.json", identity["checkpoint_commit_sha256"])
    for name, digest in identity["checkpoint_files"].items():
        check_hash(checkpoint / name, digest)
    check_hash(EXPORT / "COMMITTED.json", identity["export_commit_sha256"])
    for name, digest in identity["export_files"].items():
        check_hash(EXPORT / name, digest)
    check_hash(PAYLOAD / "transfer_bundle/research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz", identity["dataset_sha256"])
    original = read(EXPORT / "calibration_receipt.json")
    original_manifest = read(EXPORT / "eligible_checkpoint.json")
    snapshot = report["original_snapshot_before"]
    require(snapshot == report["original_snapshot_after"], "original snapshots equal")
    require(snapshot["full_state_runtime_encoding_sha256"] == original_manifest["full_training_state_sha256"], "original snapshot/export link")
    require(snapshot["full_state_unchanged"] and snapshot["inference_unchanged"], "original snapshot flags")
    equal(report["unit"], original["residual_scale"], "fixed original unit")
    require(not report["unit_updated"] and report["full_training_hold"], "unit/training scope")
    ledger = report["ledger"]
    expected_counts = dict(auxiliary_updates_attempted=128, auxiliary_updates_completed=128,
        actor_updates=0, critic_updates=0, target_updates=0, original_auxiliary_updates=0,
        original_calibration_refreshes=0, new_calibration_calls=1, simulator_transitions=0,
        new_rollouts=0, confirmation_reads=0, logged_current_critic_rows=512,
        logged_target_actor_rows=512, logged_target_critic_rows=512, native_label_draws=3,
        auxiliary_gradient_rows=32768, auxiliary_diagnostic_rows=896,
        development_auxiliary_rows=1152, development_value_rows_read=1152)
    require(ledger == expected_counts, "exact ledger")
    require(len(report["training_losses"]) == 128 and np.all(np.isfinite(report["training_losses"])), "saved finite loss count")
    require(report["data"]["calibration_reused_from_original_BCA"] and not report["data"]["fresh_confirmation_claim_permitted"], "reuse disclosure")
    require(not report["data"]["pairwise_episode_overlap"], "logged bank episode separation")
    stage_rows = [json.loads(line) for line in (RUN / "stages.jsonl").read_text(encoding="utf-8").splitlines()]
    require(stage_rows[-1]["ledger"] == ledger, "terminal journal ledger")
    sealed_stage = next(s for s in stage_rows if s["stage"] == "new_auxiliary_reference_sealed")
    require(sealed_stage["ledger"]["development_value_rows_read"] == 0, "reference sealed before values")

    sealed = read(RUN / "AUXILIARY_REFERENCE_SEALED.json")
    check_hash(RUN / "aux_params.msgpack", sealed["params_sha256"])
    check_hash(RUN / "calibration.json", sealed["calibration_sha256"])
    require(not sealed["development_values_loaded"] and sealed["original_state_unchanged"], "aux seal flags")
    calibration = read(RUN / "calibration.json")
    require(calibration == report["calibration"], "calibration/report equality")
    with np.load(RUN / "calibration_arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(RUN / "logged_labels.npz", allow_pickle=False) as z:
        labels = {k: z[k] for k in z.files}
    unit = report["unit"]
    eta = arrays["calibration_eta"].astype(np.float64)
    computed_scores = np.maximum(labels["calibration_q1"].astype(np.float64) - labels["calibration_target"].astype(np.float64), 0.) / (unit * np.maximum(eta, 1e-6))
    equal(arrays["scores"], computed_scores, "128 directional calibration scores")
    crossings = arrays["crossings"]
    require(crossings.shape == (1000,) and np.all(np.isfinite(crossings)) and np.all(crossings[:-1] <= crossings[1:]), "1000 finite ordered saved crossings")
    require(np.all(np.isin(crossings, arrays["scores"])), "crossings lie on score grid")
    empirical_index = math.ceil((1 - Fraction("0.1")) * 128) - 1
    posterior_index = math.ceil(Fraction("0.95") * 1000) - 1
    q_emp = float(np.sort(computed_scores, kind="stable")[empirical_index])
    q_post = float(crossings[posterior_index])
    q_bayes = max(q_emp, q_post)
    equal([q_emp, q_post, q_bayes, 128.], [calibration["empirical_threshold"], calibration["posterior_threshold"], calibration["threshold"], calibration["n_eff"]], "saved threshold order statistics")
    logged = {}
    for bank, n in (("fit", 256), ("calibration", 128), ("validation", 128)):
        e = arrays[bank + "_eta"]
        q, y = labels[bank + "_q1"], labels[bank + "_target"]
        require(e.shape == q.shape == y.shape == (n,) and np.all(e >= .01) and np.all(np.isfinite(e + q + y)), "logged array domain " + bank)
        logged[bank] = {}
        for name, threshold in (("bayesian", q_bayes), ("ordinary", q_emp), ("raw_local", 1.)):
            radius = threshold * (unit * np.maximum(e.astype(np.float64), 1e-6))
            coverage = y.astype(np.float64) >= q.astype(np.float64) - radius
            calculated = dict(rows=n, covered=int(coverage.sum()), directional_coverage=float(coverage.mean()), mean_radius=float(radius.mean()), max_radius=float(radius.max()))
            for key, value in calculated.items():
                equal(value, report["logged_readouts"][bank][name][key], "logged " + bank + "/" + name + "/" + key)
            logged[bank][name] = calculated
        if bank in report["after_local"]:
            normalized = y / np.float32(unit) - q / np.float32(unit) + e
            x = np.float32(20) * normalized
            sigmoid = np.exp(-np.logaddexp(np.float32(0), -x))
            loss = (float(np.mean(sigmoid)) - .9) ** 2 + .005 * float(np.mean(e ** 2))
            post = report["after_local"][bank]
            equal(loss, post["uniform_weight_objective"], "postfit objective " + bank, atol=2e-7)
            equal(float(np.mean(e)), post["mean_eta"], "postfit mean eta " + bank, atol=2e-7)
            equal(float(np.mean(np.float32(unit) * e)), post["raw_local_mean_radius"], "postfit raw width " + bank, atol=5e-7)
            equal(float(np.mean(y >= q - np.float32(unit) * e)), post["raw_local_directional_coverage"], "postfit raw coverage " + bank)

    raw = read(AUDIT / "raw_banks.json")["development"]
    with np.load(AUDIT / "raw_banks.npz", allow_pickle=False) as z:
        def decode(name):
            d = raw[name]
            a = z[d["npz_key"]]
            require(a.dtype.str == d["dtype"] and list(a.shape) == d["shape"] and hashlib.sha256(a.tobytes()).hexdigest() == d["array_bytes_sha256"], "cached descriptor " + name)
            return a.astype(np.float64)
        q1 = decode("q_current")[:, 0].reshape(128, 9)
        old_u = decode("u").reshape(128, 9)
        actions = decode("actions").reshape(128, 9, 4)
        actor = decode("saved_actor_actions")
        values = decode("saved_values").reshape(128, 9)
    equal(actions[:, 0], actor, "actor candidate zero")
    selection = read(RUN / "development_selections.json")
    outcomes = read(RUN / "development_outcomes.json")
    check_hash(RUN / "development_selections.json", read(RUN / "DEVELOPMENT_SELECTIONS_SEALED.json")["sha256"])
    require(not read(RUN / "DEVELOPMENT_SELECTIONS_SEALED.json")["saved_values_read"], "selection seal ordering")
    require(selection["metadata"] == outcomes["metadata"], "metadata equality")
    require([(m["group_id"], m["candidate_id"]) for m in raw["metadata"]] == [(g, c) for g in range(128) for c in range(9)], "original ordering")
    expected_metadata = [{k: raw["metadata"][i*9][k] for k in ("group_id", "batch", "time", "episode_seed")} for i in range(128)]
    require(selection["metadata"] == expected_metadata, "metadata linkage")
    equal(outcomes["candidate_values"], values, "cached V linkage")
    scores = {k: np.asarray(v, dtype=np.float64) for k, v in selection["scores"].items()}
    local = q1 - scores["raw_local"]  # Derived from sealed scores; no new model forward.
    require(np.all(local > 0), "positive saved local radius")
    distance = np.mean((actions - actor[:, None, :]) ** 2, axis=2)
    expected_scores = {"no_penalty": q1, "original_bca": q1-old_u,
        "one_sided_bayesian": q1-q_bayes*local, "one_sided_ordinary": q1-q_emp*local,
        "raw_local": q1-local, "actor_distance": q1-.1*19.29619572569095*distance,
        "raw_local_matched_effective_strength": q1-q_bayes*local,
        "ordinary_matched_effective_strength": q1-(q_bayes/q_emp)*(q_emp*local),
        "original_scale_at_new_threshold": q1-(q_bayes/original["threshold"])*old_u}
    require(set(scores) == set(expected_scores) and len(selection["methods"]) == 10, "all control methods")
    for name, expected in expected_scores.items():
        equal(scores[name], expected, "score mapping " + name, atol=5e-13)
    selected = {}
    for name, records in selection["methods"].items():
        require(len(records) == 128, "selection row count")
        ids = []
        for i, record in enumerate(records):
            if name == "frozen_actor":
                require(record == {"index": 0, "near_tie": False, "selection_rule": "fixed_original_actor_candidate_0"}, "fixed actor selection")
            else:
                row = scores[name][i]
                best = float(row.max())
                near = np.flatnonzero(np.abs(row-best) <= 1e-6+1e-6*abs(best))
                require(record["index"] == int(near[0]) and record["exact_argmax_index"] == int(row.argmax()) and record["near_best_indices"] == near.tolist() and record["near_tie"] == (len(near)>1), "saved tie/selection " + name)
                equal([record["maximum_score"], record["selected_score"], record["top_two_gap"]], [best, row[near[0]], np.sort(row)[-1]-np.sort(row)[-2]], "saved selection scalars")
            ids.append(record["index"])
        selected[name] = np.array(ids)
        equal(ids, outcomes["selected_indices"][name], "saved selected indices")
        equal(values[np.arange(128), ids], outcomes["selected_values"][name], "saved selected values")
    matched = {}
    for name in ("raw_local_matched_effective_strength", "ordinary_matched_effective_strength"):
        equal(selected[name], selected["one_sided_bayesian"], "matched choices")
        equal(outcomes["selected_values"][name], outcomes["selected_values"]["one_sided_bayesian"], "matched chosen V")
        matched[name] = "all 128 indices and values identical to one_sided_bayesian"
    chosen_v = {k: values[np.arange(128), ids] for k, ids in selected.items()}
    summary_map = {(r["stratum"], r["method"]): r for r in report["development_strata"]}
    require(len(summary_map) == 150 and len({r["stratum"] for r in report["development_strata"]}) == 15, "15 strata x 10 controls present")
    csv_rows = list(csv.DictReader((RUN / "all_prespecified_strata.csv").open(encoding="utf-8", newline="")))
    csv_map = {(r["stratum"], r["method"]): r for r in csv_rows}
    require(set(csv_map) == set(summary_map), "CSV/report summary inventory")
    key_rows = []
    for stratum in ("overall", "development_1", "development_2"):
        indices = np.arange(128) if stratum == "overall" else np.array([i for i,m in enumerate(expected_metadata) if m["batch"] == stratum])
        require(len(indices) == (128 if stratum == "overall" else 64), "key stratum size")
        for name, val in chosen_v.items():
            gain = val[indices]-chosen_v["no_penalty"][indices]
            band = epsilon(val[indices], chosen_v["no_penalty"][indices])
            old_band = epsilon(val[indices], chosen_v["original_bca"][indices])
            switches = selected[name][indices] != selected["no_penalty"][indices]
            calculated = dict(states=len(indices), mean_selected_value=float(val[indices].mean()), mean_gain_vs_q1=float(gain.mean()),
                mean_gain_vs_original_bca=float((val[indices]-chosen_v["original_bca"][indices]).mean()),
                mean_gain_vs_actor_distance=float((val[indices]-chosen_v["actor_distance"][indices]).mean()),
                mean_regret=float((values.max(axis=1)[indices]-val[indices]).mean()), mean_numerical_band=float(band.mean()),
                mean_numerical_band_vs_original_bca=float(old_band.mean()), switches=int(switches.sum()),
                helpful_switches=int(((gain>band)&switches).sum()), harmful_switches=int(((gain < -band)&switches).sum()),
                inconclusive_switches=int(((np.abs(gain)<=band)&switches).sum()), minimum_gain_vs_q1=float(gain.min()),
                maximum_gain_vs_q1=float(gain.max()), near_tie_states=sum(selection["methods"][name][i]["near_tie"] for i in indices))
            for key, val in calculated.items():
                equal(val, summary_map[stratum,name][key], "key summary " + stratum + "/" + name + "/" + key, atol=1e-12)
                equal(val, float(csv_map[stratum,name][key]), "CSV key summary", atol=1e-12)
            key_rows.append(dict(stratum=stratum, method=name, **calculated))
    primary = [r for r in key_rows if r["method"] == "one_sided_bayesian" and r["stratum"] != "overall"]
    require(primary == report["primary_batch_rows"], "primary rows")
    screen = q_bayes > 0 and all(r["mean_gain_vs_q1"] > r["mean_numerical_band"] and r["mean_gain_vs_original_bca"] > r["mean_numerical_band_vs_original_bca"] for r in primary)
    require(screen == report["exploratory_two_batch_utility_screen"] == False, "failed prespecified utility screen")
    result = dict(schema="unifloral.bca_one_sided.independent_saved_review.v1", status="pass_saved_arithmetic_verification",
        utc=datetime.now(timezone.utc).isoformat(), verifier_sha256=sha(Path(__file__)), python=sys.version, numpy=np.__version__,
        reproduction_command=r".\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/bca_one_sided_v1/run_analysis/verify_saved_results.py",
        report_sha256=EXPECTED_REPORT, checked_assertions=checks, reviewed_file_sha256=hashes,
        source_manifest_entries=manifest_entries, native_snapshot_files=40,
        saved_score_methods_checked=9, selection_records_checked=1280, key_summary_rows_recomputed=30,
        total_report_rows_inventory_checked=150, calibration_rows_checked=128, saved_posterior_crossings_checked=1000,
        empirical_order_statistic_one_based=empirical_index+1, posterior_order_statistic_one_based=posterior_index+1,
        calibration=calibration, logged_postfit_readouts=logged, key_summaries=key_rows, matched_controls=matched,
        exploratory_two_batch_utility_screen=bool(screen),
        screen_failure="development_2 gain versus original BCA is -0.01636913260187317, below its positive numerical band 5.0599485602699126e-05",
        ledger=ledger, worker_elapsed_seconds=report["elapsed_seconds"], material_execution_incompleteness=[],
        limitations=["Saved evidence checks, not independent attestation of historical runtime execution or every GPU forward.",
            "Posterior crossings checked for grid membership, sorting and threshold order statistics only; exponential draws are not saved and calibration was not called again.",
            "Development eta is not separately saved. Score mappings use local radius recovered from Q1 minus saved raw-local score; no network output was regenerated.",
            "Pre-fit eta and target-critic outputs are not saved, so their forward values and pre-fit objectives were not independently regenerated.",
            "GPU float32 postfit objectives are checked by NumPy within 2e-7; float64 score mappings within 5e-13; key summary arithmetic within 1e-12.",
            "Original calibration episodes are adaptively reused. Local validation is exploratory, not fresh confirmation or selected-action/OOD/true-return coverage.",
            "Runtime is the disclosed Zeus CUDA environment outside upstream JAX/NumPy constraints. Template/replay forwards are separate from diagnostic ledger counts.",
            "No pilot rerun, random draws, fitting, calibration, models, GPU work, new metrics, selectors, simulation or confirmation access occurred in this review."],
        full_training_hold=True, no_automatic_followup=True)
    with (HERE / "INDEPENDENT_REVIEW.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "checks": checks, "key_summaries": 30, "selection_records": 1280,
        "utility_screen": bool(screen), "review_sha256": sha(HERE / "INDEPENDENT_REVIEW.json")}, indent=2))


if __name__ == "__main__":
    main()
