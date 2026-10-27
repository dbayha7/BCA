"""Exclusive prepare/run phases. Neither phase runs on import."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import traceback

import diagnostic as D


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = [D.ROOT / name for name in ("diagnostic.py", "run_diagnostic.py", "protocol.json")]
    return {p.name: D.digest(p) for p in paths}


def fresh_dir(path):
    path = Path(path).resolve()
    D.require(path != D.ROOT and D.ROOT in path.parents, "output must be a new child of this diagnostic directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_npz(path, arrays):
    import numpy as np
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def load_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def reference_record(engine):
    ref = engine.result.state.posterior
    def numeric(value):
        value = float(value)
        return value if __import__("math").isfinite(value) else str(value)
    return dict(R=numeric(ref.threshold), unit=numeric(ref.residual_scale), ready=bool(ref.ready),
                reference_update=engine.reference_update, miscoverage=engine.settings.miscoverage,
                credibility=engine.settings.credibility,
                calibration_rows=engine.result.record["identity"]["data"]["calibration_rows"],
                effective_calibration_rows=numeric(ref.n_eff),
                reference_sha256=D.parameter_hashes(engine)["frozen_reference"])


def prepare(output):
    """No dynamics/oracle return calls: generate, score and freeze inputs only."""
    output = fresh_dir(output)
    D.write_json(output / "status.json", dict(status="preparing", started_utc=now(), truth_labels_used=0))
    try:
        engine = D.load_engine()
        import numpy as np
        import scipy
        import jax
        before = D.parameter_hashes(engine)
        adaptation = D.bank_inputs(engine, 2, 3, 4)
        evaluation = D.bank_inputs(engine, 8, 1, 2)
        coefficient = D.matching_coefficient(adaptation["width"], adaptation["distance"], adaptation["width_usable"])
        scores = D.score_arrays(evaluation, coefficient)
        D.require(int(np.sum(100-evaluation["time"])*12) == 21696, "main budget changed")
        save_npz(output / "adaptation_inputs.npz", adaptation)
        save_npz(output / "evaluation_inputs.npz", evaluation)
        save_npz(output / "score_inputs.npz", scores)
        D.write_json(output / "coefficient_receipt.json", dict(created_utc=now(), truth_labels_used=0,
            rule="pooled within-state centered RMS(U)/RMS(D)", result=coefficient,
            adaptation_inputs_sha256=D.digest(output / "adaptation_inputs.npz")))
        D.write_json(output / "engine_identity.json", engine.metadata)
        after = D.parameter_hashes(engine)
        D.require(before == after, "prepare mutated frozen parameters")
        D.write_json(output / "preparation_manifest.json", dict(schema="direct-q-preparation-v1", status="prepared_no_truth_labels",
            created_utc=now(), truth_labels_used=0, source_sha256=source_hashes(),
            actual_engine_sha256=D.digest(D.WORKSPACE / "action_only_20261021/engine.py"),
            original_engine_freeze_sha256=D.digest(D.WORKSPACE / "action_only_20261021/freeze.json"),
            artifact_sha256={p.name: D.digest(p) for p in sorted(output.iterdir()) if p.name != "status.json"},
            parameters_before=before, parameters_after=after, reference=reference_record(engine),
            runtime=engine.direct_runtime, versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__),
            fresh_state_meaning="new independently sampled physical arrays; no prior state rows reused or selected",
            support_counts={key: int(np.sum(evaluation["support_class"] == key)) for key in ("common_on_grid", "rare_on_grid", "unsupported")},
            width_refusals=int(evaluation["width_refusal"].sum()), coefficient_status=coefficient["status"],
            planned_main_returns=384, planned_main_transitions=21696, maximum_main_transitions=38400, fixture_transitions=608))
        D.write_json(output / "status.json", dict(status="prepared_no_truth_labels", finished_utc=now(), truth_labels_used=0,
            preparation_manifest_sha256=D.digest(output / "preparation_manifest.json")))
    except Exception as error:
        D.write_json(output / "status.json", dict(status="failed_before_truth", finished_utc=now(), error=repr(error), traceback=traceback.format_exc()))
        raise
    return output


def authenticate_preparation(prepared, freeze):
    prepared, freeze = Path(prepared).resolve(), Path(freeze).resolve()
    reviewed = json.loads(freeze.read_text(encoding="utf-8"))
    D.require(reviewed.get("approved_for_oracle_execution") is True, "reviewed freeze must explicitly approve oracle execution")
    D.require(reviewed["preparation_manifest_sha256"] == D.digest(prepared / "preparation_manifest.json"), "preparation receipt changed after review")
    manifest = json.loads((prepared / "preparation_manifest.json").read_text(encoding="utf-8"))
    D.require(manifest["source_sha256"] == source_hashes() == reviewed["source_sha256"], "protocol/code changed after preparation or review")
    D.require(manifest["truth_labels_used"] == 0 and manifest["status"] == "prepared_no_truth_labels", "not an outcome-free preparation")
    for filename, expected in manifest["artifact_sha256"].items():
        D.require(D.digest(prepared / filename) == expected, "prepared artifact changed: " + filename)
    for name, key in (("engine.py", "actual_engine_sha256"), ("freeze.json", "original_engine_freeze_sha256")):
        D.require(D.digest(D.WORKSPACE / "action_only_20261021" / name) == manifest[key], "original engine authentication changed")
    return manifest


def candidate_rows(bank, truth, scores, rollout):
    rows = []
    for i in range(32):
        for j in range(12):
            error = float(bank["q1"][i, j]-truth[i, j])
            row = dict(state_id=i, candidate_id=j, context=D.CONTEXTS[int(bank["context"][i])],
                regime=int(bank["regimes"][i]), load=float(bank["states"][i, 10]), time=int(bank["time"][i]),
                slot="grid" if j < 6 else "actor" if j == 6 else "uniform_continuous",
                support_class=str(bank["support_class"][i, j]), categorical_beta=float(bank["beta"][i, j]),
                grid_id=int(bank["grid_id"][i, j]), q1=float(bank["q1"][i, j]), q2=float(bank["q2"][i, j]),
                true_policy_q=float(truth[i, j]), signed_q1_error=error, positive_q1_overestimation=max(error, 0.),
                width=float(bank["width"][i, j]), eta=float(bank["eta"][i, j]), distance=float(bank["distance"][i, j]),
                width_usable=bool(bank["width_usable"][i, j]), width_refusal=bool(bank["width_refusal"][i, j]),
                rollout_completed=bool(rollout["completed"][i, j]), rollout_finite=bool(rollout["finite"][i, j]),
                failure_event=bool(rollout["failure"][i, j]), transitions=int(rollout["steps"][i, j]))
            row.update({"action_"+str(k): float(bank["actions"][i, j, k]) for k in range(4)})
            row.update({"score_"+key: float(value[i, j]) for key, value in scores.items()})
            rows.append(row)
    return rows


def error_summaries(bank, truth):
    import numpy as np
    masks = [("pooled", np.ones(32, bool))]
    masks += [(name, bank["context"] == i) for i, name in enumerate(D.CONTEXTS)]
    masks += [("regime_"+str(g), bank["regimes"] == g) for g in (0, 1)]
    masks += [("load_"+str(load), bank["states"][:, 10] == load) for load in (.3, 1.1)]
    masks += [("time_"+str(t), bank["time"] == t) for t in (0, 50, 99)]
    error = bank["q1"]-truth
    records = []
    for group, states in masks:
        for support in ("all", "common_on_grid", "rare_on_grid", "unsupported"):
            mask = np.broadcast_to(states[:, None], error.shape).copy()
            if support != "all": mask &= bank["support_class"] == support
            values = error[mask]
            records.append(dict(group=group, support=support, rows=int(mask.sum()),
                mean_signed_q1_error=float(values.mean()) if len(values) else None,
                mean_positive_overestimation=float(np.maximum(values, 0).mean()) if len(values) else None,
                positive_overestimation_rows=int(np.sum(values > 0)),
                q1_rmse=float(np.sqrt(np.mean(values**2))) if len(values) else None,
                width_refusals=int(np.sum(bank["width_refusal"] & mask))))
    return records


def run(prepared, freeze, output):
    manifest = authenticate_preparation(prepared, freeze)
    output = fresh_dir(output)
    started = now()
    status = dict(status="running", started_utc=started, completed_main_returns=0, main_transitions=0, fixture_transitions=0)
    D.write_json(output / "status.json", status)
    engine, before = None, None
    try:
        engine = D.load_engine()
        import numpy as np
        before = D.parameter_hashes(engine)
        D.require(before == manifest["parameters_before"] == manifest["parameters_after"], "frozen parameters differ from preparation")
        bank = load_npz(Path(prepared) / "evaluation_inputs.npz")
        scores = load_npz(Path(prepared) / "score_inputs.npz")
        # This receipt is written before the first oracle call, including fixtures.
        D.write_json(output / "execution_receipt.json", dict(started_utc=started,
            preparation_manifest_sha256=D.digest(Path(prepared) / "preparation_manifest.json"),
            reviewed_freeze_sha256=D.digest(freeze), source_sha256=source_hashes(), parameters_before=before,
            planned_main_returns=384, planned_main_transitions=21696, fixture_transitions=608))
        fixtures = D.validate_truth(engine)
        status["fixture_transitions"] = fixtures["plant_transitions"]
        D.write_json(output / "fixture_results.json", fixtures)
        rollout = D.truth_rollouts(engine, bank)
        truth = rollout["return_value"]
        save_npz(output / "raw_results.npz", {**bank, **{"score_"+k: v for k, v in scores.items()},
            **{"rollout_"+k: v for k, v in rollout.items()}, "true_policy_q": truth,
            "signed_q1_error": bank["q1"]-truth, "positive_q1_overestimation": np.maximum(bank["q1"]-truth, 0.)})
        rows = candidate_rows(bank, truth, scores, rollout)
        D.write_csv(output / "candidate_records.csv", rows)
        status.update(completed_main_returns=int(rollout["completed"].sum()), main_transitions=int(rollout["steps"].sum()),
            nonfinite_main_returns=int(np.sum(~np.isfinite(truth))), failure_events=int(rollout["failure"].sum()),
            failure_rule="any absolute physical position>3 at pre/post state, or incomplete/nonfinite continuation; no early termination for threshold event")
        D.require(rollout["completed"].all() and np.isfinite(truth).all(), "incomplete/nonfinite oracle outcomes preserved; inferential aggregation refused")
        D.require(status["main_transitions"] == 21696, "unexpected main transition budget")
        analysis, decisions = D.analyze(bank, truth, scores)
        analysis.update(schema="direct-q-ranking-results-v1", generated_utc=now(), training_replications=1,
            scope="conditional on designed state/context/time law, fixed candidate realization and one trained checkpoint; not Q coverage or training-seed uncertainty",
            paired_unit="physical state; bootstrap within context x time strata, 10000 draws; tiny strata n2-3",
            argmax="first exact maximum in fixed candidate order", pairwise_rank_tie_atol=1e-6,
            reference=manifest["reference"], coefficient_receipt=json.loads((Path(prepared)/"coefficient_receipt.json").read_text()),
            error_summaries=error_summaries(bank, truth), support_counts=manifest["support_counts"],
            width_refusals=int(bank["width_refusal"].sum()), numerical_validation=fixtures,
            provenance=dict(preparation_manifest_sha256=D.digest(Path(prepared)/"preparation_manifest.json"),
                reviewed_freeze_sha256=D.digest(freeze), engine_identity_sha256=D.digest(Path(prepared)/"engine_identity.json")))
        D.write_json(output / "paired_results.json", analysis)
        D.write_csv(output / "paired_decisions.csv", decisions)
        D.write_csv(output / "within_state_correlations.csv", analysis["within_state_width_positive_error_spearman"])
        D.write_csv(output / "error_summaries.csv", analysis["error_summaries"])
        after = D.parameter_hashes(engine)
        D.require(before == after, "oracle evaluation mutated frozen parameters")
        status.update(status="complete", finished_utc=now(), parameters_before=before, parameters_after=after,
            new_training_updates=0, output_sha256={p.name: D.digest(p) for p in sorted(output.iterdir()) if p.name != "status.json"})
    except Exception as error:
        status.update(status="failed", finished_utc=now(), error=repr(error), traceback=traceback.format_exc())
        if engine is not None:
            status.update(parameters_before=before, parameters_after=D.parameter_hashes(engine))
        D.write_json(output / "status.json", status)
        raise
    D.write_json(output / "status.json", status)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run"); p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare": prepare(args.output)
    else: run(args.prepared, args.freeze, args.output)
