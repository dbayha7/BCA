"""Exclusive outcome-free preparation, then separately reviewed frozen execution."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import traceback

import replication as R


def now():
    return datetime.now(timezone.utc).isoformat()


def fresh(path):
    path = Path(path).resolve()
    R.require(path != R.ROOT and R.ROOT in path.parents, "output must be a new child of replication directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_npz(path, arrays):
    import numpy as np
    with Path(path).open("xb") as handle: np.savez_compressed(handle, **arrays)


def load_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive: return {key: archive[key].copy() for key in archive.files}


def prepare(output):
    output = fresh(output)
    status = dict(status="preparing", started_utc=now(), truth_labels_used=0, plant_transitions=0)
    R.write_json(output/"status.json", status)
    module, engine, before = None, None, None
    try:
        module, engine, prior = R.load_engine()
        import numpy as np
        import scipy
        import jax
        before = module.parameter_hashes(engine)
        bank = R.prepare_bank(module, engine)
        scores = R.score_arrays(bank["q1"], bank["width"], bank["distance"], bank["width_usable"], bank["permutation"])
        valid, reasons = R.valid_state_mask(bank), R.invalid_state_reasons(bank)
        draws = R.bootstrap_plan(bank["context"], bank["time"])
        R.require(int(np.sum(100-bank["time"])*12) == 21696, "planned budget differs")
        save_npz(output/"inputs.npz", bank)
        save_npz(output/"scores.npz", scores)
        save_npz(output/"bootstrap_plan.npz", dict(state_indices=draws))
        R.write_json(output/"input_status.json", dict(planned_states=32, valid_states=int(valid.sum()),
            state_invalid_reasons={str(i): reason for i, reason in enumerate(reasons) if reason},
            native_width_refusals=int(bank["width_refusal"].sum()),
            support_counts={key: int(np.sum(bank["support_class"] == key)) for key in ("common_on_grid", "rare_on_grid", "unsupported")},
            coefficient=R.COEFFICIENT, coefficient_reestimated=False, truth_labels_used=0,
            planned_returns=384, runnable_returns=int(valid.sum()*12), planned_transitions=21696,
            runnable_transitions=int(np.sum(100-bank["time"][valid])*12)))
        R.write_json(output/"engine_identity.json", engine.metadata)
        reference = engine.result.state.posterior
        R.write_json(output/"reference.json", dict(R=float(reference.threshold), unit=float(reference.residual_scale),
            ready=bool(reference.ready), effective_calibration_rows=float(reference.n_eff),
            miscoverage=engine.settings.miscoverage, credibility=engine.settings.credibility,
            reference_update=engine.reference_update, source_calibration_rows=128,
            width="actual native frozen_level_dose width; full produced precision, no rounded multiplier"))
        after = module.parameter_hashes(engine)
        R.require(before == after == prior["parameters"], "preparation changed frozen parameters")
        manifest = dict(schema="direct-q-replication-prepared-v1", status="prepared_no_truth_labels", created_utc=now(),
            truth_labels_used=0, plant_transitions=0, source_sha256=R.source_hashes(), prior_validation=prior,
            parameters_before=before, parameters_after=after, runtime=engine.direct_runtime,
            versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__),
            artifact_sha256={p.name: R.digest(p) for p in sorted(output.iterdir()) if p.name != "status.json"})
        R.write_json(output/"preparation_manifest.json", manifest)
        status.update(status="prepared_no_truth_labels", finished_utc=now(),
            preparation_manifest_sha256=R.digest(output/"preparation_manifest.json"))
    except Exception as error:
        status.update(status="failed_before_truth", error=repr(error), traceback=traceback.format_exc(), finished_utc=now())
        if engine is not None: status.update(parameters_before=before, parameters_after=module.parameter_hashes(engine))
        R.write_json(output/"status.json", status)
        raise
    R.write_json(output/"status.json", status)


def authenticate(prepared, freeze):
    prepared = Path(prepared).resolve()
    manifest, review = R.read_json(prepared/"preparation_manifest.json"), R.read_json(freeze)
    R.require(review.get("approved_for_oracle_execution") is True, "root-reviewed freeze must authorize labels")
    R.require(R.digest(prepared/"preparation_manifest.json") == review["preparation_manifest_sha256"], "reviewed input manifest changed")
    R.require(R.source_hashes() == manifest["source_sha256"] == review["source_sha256"], "frozen code/protocol changed")
    R.require(manifest["status"] == "prepared_no_truth_labels" and manifest["truth_labels_used"] == 0, "preparation not outcome-free")
    for name, expected in manifest["artifact_sha256"].items(): R.require(R.digest(prepared/name) == expected, "prepared artifact changed: "+name)
    R.require(R.authenticate_prior() == manifest["prior_validation"], "prior source/model/replay receipt changed")
    return manifest


def rollout_saved_inputs(module, engine, bank):
    """No fixtures; only finite predeclared states, original indices retained."""
    import numpy as np
    valid = R.valid_state_mask(bank)
    shape = (32, 12)
    result = dict(return_value=np.full(shape, np.nan), completed=np.zeros(shape, bool), finite=np.zeros(shape, bool),
        failure=np.ones(shape, bool), steps=np.zeros(shape, int), evaluated=np.zeros(shape, bool))
    if not valid.any(): return result
    selected = {key: value[valid] for key, value in bank.items()}
    actual = module.truth_rollouts(engine, selected)
    for key, value in actual.items():
        if key not in result:
            result[key] = np.full(shape, np.nan) if value.dtype.kind == "f" else np.zeros(shape, value.dtype)
        result[key][valid] = value
    result["evaluated"][valid] = True
    return result


def candidate_records(bank, truth, scores, rollout, state_rows):
    rows = []
    for i in range(32):
        for j in range(12):
            row = dict(state_id=i, candidate_id=j, context=R.CONTEXTS[int(bank["context"][i])],
                time=int(bank["time"][i]), regime=int(bank["regimes"][i]), load=float(bank["states"][i, 10]),
                state_valid=state_rows[i]["valid"], state_invalid_reasons=";".join(state_rows[i]["invalid_reasons"]),
                candidate_slot="grid" if j < 6 else "actor" if j == 6 else "uniform_continuous",
                support_class=str(bank["support_class"][i, j]), beta=float(bank["beta"][i, j]),
                q1=float(bank["q1"][i, j]), true_policy_q=float(truth[i, j]), signed_q1_error=float(bank["q1"][i, j]-truth[i, j]),
                width=float(bank["width"][i, j]), width_usable=bool(bank["width_usable"][i, j]),
                native_width_refusal=bool(bank["width_refusal"][i, j]), eta=float(bank["eta"][i, j]),
                distance=float(bank["distance"][i, j]), shuffled_width=float(bank["shuffled_width"][i, j]),
                shuffled_source_index=int(bank["permutation"][i, j]),
                rollout_evaluated=bool(rollout["evaluated"][i, j]), rollout_completed=bool(rollout["completed"][i, j]),
                failure_event=bool(rollout["failure"][i, j]), steps=int(rollout["steps"][i, j]))
            row.update({"action_"+str(k): float(bank["actions"][i, j, k]) for k in range(4)})
            row.update({"score_"+key: float(scores[key][i, j]) for key in R.METHODS})
            rows.append(row)
    return rows


def flattened_state_rows(bank, records):
    rows = []
    for record in records:
        i = record["state_id"]
        row = {key: value for key, value in record.items() if key not in ("methods", "invalid_reasons")}
        row.update(context=R.CONTEXTS[int(bank["context"][i])], time=int(bank["time"][i]), invalid_reasons=";".join(record["invalid_reasons"]))
        for method, metrics in record["methods"].items(): row.update({method+"__"+key: value for key, value in metrics.items()})
        rows.append(row)
    return rows


def run(prepared, freeze, output):
    manifest = authenticate(prepared, freeze)
    output = fresh(output)
    status = dict(status="running", started_utc=now(), main_transitions=0, completed_main_returns=0,
        planned_main_returns=384, planned_main_transitions=21696, new_validation_transitions=0, new_training_updates=0)
    R.write_json(output/"status.json", status)
    module, engine, before = None, None, None
    try:
        module, engine, prior = R.load_engine()
        import numpy as np
        before = module.parameter_hashes(engine)
        R.require(before == manifest["parameters_before"] == manifest["parameters_after"], "frozen model differs from preparation")
        bank, scores = load_npz(Path(prepared)/"inputs.npz"), load_npz(Path(prepared)/"scores.npz")
        draws = load_npz(Path(prepared)/"bootstrap_plan.npz")["state_indices"]
        R.write_json(output/"execution_receipt.json", dict(started_utc=status["started_utc"],
            preparation_manifest_sha256=R.digest(Path(prepared)/"preparation_manifest.json"), freeze_sha256=R.digest(freeze),
            source_sha256=R.source_hashes(), parameters_before=before, prior_validation=prior,
            planned_main_returns=384, planned_main_transitions=21696, new_validation_transitions=0))
        rollout = rollout_saved_inputs(module, engine, bank)
        truth = rollout["return_value"]
        save_npz(output/"raw_results.npz", {**bank, **{"score_"+k: v for k, v in scores.items()},
            **{"rollout_"+k: v for k, v in rollout.items()}, "true_policy_q": truth})
        status.update(main_transitions=int(rollout["steps"].sum()), completed_main_returns=int(rollout["completed"].sum()),
            evaluated_main_returns=int(rollout["evaluated"].sum()), nonfinite_evaluated_returns=int(np.sum(rollout["evaluated"] & ~np.isfinite(truth))),
            failure_events_evaluated=int(np.sum(rollout["failure"] & rollout["evaluated"])))
        R.require(status["main_transitions"] <= 21696, "transition budget exceeded")
        R.write_json(output/"status.json", status)
        bank.update(rollout_completed=rollout["completed"], rollout_finite=rollout["finite"], rollout_steps=rollout["steps"])
        summary, state_rows, bootstrap = R.analyze(bank, truth, scores, draws)
        summary.update(schema="direct-q-replication-results-v1", generated_utc=now(), fixed_coefficient=R.COEFFICIENT,
            provenance=dict(source_sha256=R.source_hashes(), preparation_manifest_sha256=R.digest(Path(prepared)/"preparation_manifest.json"),
                freeze_sha256=R.digest(freeze), prior_validation=prior),
            native_width_refusals=int(bank["width_refusal"].sum()), actual_transition_counts=dict(main=status["main_transitions"], new_validation=0),
            limitations=R.read_json(R.ROOT/"protocol.json")["limitations"], state_records=state_rows)
        save_npz(output/"bootstrap_statistics.npz", bootstrap)
        R.write_json(output/"paired_results.json", summary)
        R.write_csv(output/"state_records.csv", flattened_state_rows(bank, state_rows))
        R.write_csv(output/"candidate_records.csv", candidate_records(bank, truth, scores, rollout, state_rows))
        R.write_csv(output/"metric_summary.csv", [dict(metric=name, **value) for name, value in summary["metrics"].items()])
        after = module.parameter_hashes(engine)
        R.require(before == after == prior["parameters"], "frozen parameters changed")
        status.update(status="complete" if summary["valid_states"] == 32 else "complete_with_invalid_states", finished_utc=now(),
            valid_states=summary["valid_states"], invalid_states=summary["invalid_states"], parameters_before=before, parameters_after=after,
            output_sha256={p.name: R.digest(p) for p in sorted(output.iterdir()) if p.name != "status.json"})
    except Exception as error:
        status.update(status="failed", finished_utc=now(), error=repr(error), traceback=traceback.format_exc())
        if engine is not None: status.update(parameters_before=before, parameters_after=module.parameter_hashes(engine))
        R.write_json(output/"status.json", status)
        raise
    R.write_json(output/"status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run"); p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare": prepare(args.output)
    else: run(args.prepared, args.freeze, args.output)
