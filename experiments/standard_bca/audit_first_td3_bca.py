"""Audit the first standard-study TD3+BC Hopper no-IW BCA run from saved data, on CPU.

This script has one explicit run identity and never calls a learner, model or
simulator. Run under the recorded Flax/numpy/h5py environment on the execution host.
"""
from pathlib import Path
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
from flax import serialization

RUN_ID = "td3_bc-hopper-bca-noiw-s202609171"
MANIFEST = "13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.asarray(value)
    h = hashlib.sha256(json.dumps([a.dtype.str, a.shape]).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


RADIUS_FIELDS = ["radius", "conformal_radius", "bayesian_radius", "posterior_quantiles",
                 "effective_sample_size", "supported_count", "inputs_valid", "finite"]


def leaves(value):
    if isinstance(value, dict):
        return [leaf for k in sorted(value) for leaf in leaves(value[k])]
    return [] if value is None else [value]


def posterior_hash(post):
    # Preserve the frozen PosteriorState/PosteriorRadius named-tuple field order.
    fields = leaves(post["cal_params"]) + [post["residual_scale"], post["group_edges"]]
    fields += [post["radii"][k] for k in RADIUS_FIELDS] + [post["ready"]]
    return digest([array_hash(v) for v in fields])


def json_arrays(value):
    if isinstance(value, dict):
        return {k: json_arrays(v) for k, v in value.items()}
    return np.asarray(value).tolist()


def check_radii(radii):
    values = {k: np.asarray(v) for k, v in radii.items()}
    require(set(values) == set(RADIUS_FIELDS), "Unexpected radius fields.")
    require(all(v.shape == ((1, 128) if k == "posterior_quantiles" else (1,))
                and np.isfinite(v).all() for k, v in values.items()), "Radius shape/finiteness.")
    require(values["inputs_valid"].all() and values["finite"].all(), "Invalid posterior radius.")
    require(values["supported_count"][0] == 1103
            and values["effective_sample_size"][0] == 1103, "Unweighted heldout support differs.")
    require(np.array_equal(values["radius"], np.maximum(values["conformal_radius"],
            values["bayesian_radius"])), "Combined radius differs from maximum.")
    require(np.array_equal(values["bayesian_radius"],
            np.sort(values["posterior_quantiles"], axis=1)[:, 121]), "Bayesian 95% rank differs.")
    require(all((values[k] >= 0).all() for k in RADIUS_FIELDS[:4]), "Negative radius/draw.")


def audit(root, repo, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    lane = root / "queue-local-v1"
    run = lane / "runs" / RUN_ID
    require(sha(source / "manifest.json") == MANIFEST, "Frozen manifest changed.")
    manifest = read(source / "manifest.json")
    item, = [x for x in manifest["runs"] if x["row"]["run_id"] == RUN_ID]
    row = item["row"]
    for name, wanted in manifest["source_sha256"].items():
        require(sha(source / name) == wanted, "Changed frozen source: " + name)
    sys.path.insert(0, str(repo))
    from experiments.ood.standard_receipt import bind_receipt
    receipt = bind_receipt(source / "manifest.json", lane, source, RUN_ID)
    save(out / "process_exit.json", receipt)
    result = read(run / "result.json")
    queue = read(lane / "queue_status.json")
    closed, = [x for x in queue["completed"] if x["run_id"] == RUN_ID]
    require(closed["result_sha256"] == sha(run / "result.json"), "Queue/result mismatch.")
    actual = read(lane / "attempts" / RUN_ID / "actual_exit.json")
    prep = read(run / "preparation.json")
    require(prep["accepted"] is True and prep["learner_updates"] == 0, "Preparation failed.")
    m = prep["metadata"]
    for name, wanted in manifest["source_sha256"].items():
        require(m["source_files"][name] == wanted, "Preparation source mismatch.")
    require(m["settings"]["args"] == {**row["native_args"], "wandb_project": "unifloral",
            "wandb_team": "flair", "wandb_group": "debug"}, "Native arguments/defaults changed.")
    require(m["settings"]["specification"] == row["specification"], "Method changed.")
    for key, value in row["protocol"].items():
        require(m["settings"]["protocol"][key] == value, "Protocol changed: " + key)
    require(digest(m["settings"]) == m["settings_sha256"], "Settings digest mismatch.")
    cache = Path(m["raw_identity"]["path"])
    require(sha(cache) == row["cache"]["sha256"] == m["raw_identity"]["sha256"],
            "Actual dataset cache mismatch.")
    with h5py.File(cache, "r") as stream:
        raw = {k: stream[k][()] for k in m["raw_array_hashes"]}
    require({k: array_hash(v) for k, v in raw.items()} == m["raw_array_hashes"],
            "Raw array identity mismatch.")
    ids = np.asarray(m["dependency_maps"]["raw_current"], np.int64)
    expected_ids = np.flatnonzero(~raw["timeouts"][:-1].astype(bool))
    require(np.array_equal(ids, expected_ids), "Native conversion row mismatch.")
    converted = dict(observations=raw["observations"][ids].astype(np.float32),
                     next_observations=raw["observations"][ids + 1].astype(np.float32),
                     actions=raw["actions"][ids].astype(np.float32),
                     rewards=raw["rewards"][ids].astype(np.float32),
                     terminals=raw["terminals"][ids].astype(bool))
    require({k: array_hash(v) for k, v in converted.items()} == m["converted_array_hashes"],
            "Converted array identities differ.")
    train_ids = np.asarray(m["training_converted_ids"], np.int64)
    hold_ids = np.asarray(m["heldout_converted_ids"], np.int64)
    require(len(train_ids) == 998895 and len(hold_ids) == 1103
            and len(m["reference_converted_ids"]) == 1024, "Unexpected data partition.")
    require(np.array_equal(np.sort(np.r_[train_ids, hold_ids]), np.arange(len(ids))),
            "Training/holdout rows are not a disjoint exhaustive partition.")
    obs = converted["observations"]
    mean, std = obs[train_ids].mean(0), obs[train_ids].std(0) + .001
    require(np.array_equal(mean, np.asarray(m["obs_mean"], np.float32))
            and np.array_equal(std, np.asarray(m["obs_std"], np.float32)),
            "Training-only observation normalization differs.")
    require(m["reward_normalization"] == "disabled", "Unexpected reward transformation.")
    arrays = [(obs - mean) / std, converted["actions"], converted["rewards"],
              (converted["next_observations"] - mean) / std,
              converted["terminals"].astype(np.float32)]
    ref_ids = np.asarray(m["reference_converted_ids"], np.int64)
    require(len(np.unique(ref_ids)) == 1024 and np.isin(ref_ids, train_ids).all(),
            "Reference rows are not unique training rows.")
    for key, indices in [("training", train_ids), ("heldout", hold_ids), ("reference", ref_ids)]:
        require(digest([array_hash(x[indices]) for x in arrays]) == m["run_input_hashes"][key],
                "Reconstructed prepared input identity differs: " + key)
    accepted = read(repo / "docs/validation/standard-bca-preflight.json")
    cell, = [x for x in accepted["data_preparation"]["cells"]
             if (x["host"], x["dataset"]) == ("td3_bc", "hopper")]
    require(m["run_input_hashes"]["training"] == cell["paired_data_fingerprints"][0],
            "Accepted paired training fingerprint differs.")
    for key, indices in [("training_ids", train_ids), ("heldout_ids", hold_ids),
                         ("reference_ids", ref_ids)]:
        require(digest([array_hash(indices.astype(np.int32))]) == m["run_input_hashes"][key],
                "Prepared ID hash differs: " + key)
    require(row["calibration_weighting"] == "none"
            and row["specification"]["posterior"]["affinity"]["mode"] == "off",
            "Unexpected importance tilt.")
    host_folder = repo / "outputs/standard_bca/td3_bc/hopper/host/s202609171/verified-v4"
    host = read(host_folder / "audit.json")
    require(host["accepted"] is True and host["run_id"] == "td3_bc-hopper-host-s202609171",
            "Wrong closed host audit.")
    for key in ["training", "heldout", "training_ids", "heldout_ids", "obs_mean", "obs_std"]:
        require(m["run_input_hashes"][key] == host["run_input_hashes"][key],
                "Paired host data differs: " + key)
    del raw, converted, arrays, obs
    # Read counters directly; never initialize an actor or query checkpoint weights.
    sys.path.insert(0, str(source))
    from runtime.validation import checkpoint_counts, counters
    checkpoints = []
    for c in result["checkpoints"]:
        path = run / c["path"]
        require(sha(path) == c["sha256"], "Checkpoint file hash differs.")
        tree = serialization.msgpack_restore(path.read_bytes())
        counts = checkpoint_counts(tree, "td3", c["step"], 2)
        require(counts == c["counters"], "Reported counters differ from saved bytes.")
        state = tree["state"]
        require(all(state[k] is not None for k in ["calibrator", "posterior", "residual_scale"]),
                "BCA calibration state missing.")
        require(counts["accepted_scale_fits"] == c["step"], "Accepted scale-fit counter differs.")
        post = state["posterior"]
        require(bool(post["ready"]) and np.shape(post["group_edges"]) == (0,)
                and float(post["residual_scale"]) > 0 and float(state["residual_scale"]) > 0,
                "Posterior readiness/unit differs.")
        check_radii(post["radii"])
        matching_host, = [x for x in host["checkpoints"] if x["step"] == c["step"]]
        require(np.array_equal(tree["training_rng"], matching_host["training_rng"]),
                "Saved paired training RNG differs.")
        targets = {}
        for name in ["actor_target", "critic_target"]:
            target = counters(state["native"][name])
            require(target["/step"] == c["step"] // 2
                    and all(v == 0 for k, v in target.items() if k.endswith("/count")),
                    "Target counters differ.")
            targets[name] = target
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            return value is None or bool(np.isfinite(np.asarray(value)).all())
        require(finite(tree), "Nonfinite checkpoint state.")
        checkpoints.append({**c, "decoded_counters": counts, "target_counters": targets,
                            "training_rng": np.asarray(tree["training_rng"]).tolist(),
                            "live_residual_scale": float(state["residual_scale"]),
                            "frozen_residual_scale": float(post["residual_scale"]),
                            "posterior_snapshot_sha256": posterior_hash(post),
                            "radii": json_arrays(post["radii"])})
        del tree
    require(sha(run / "events.jsonl.gz") == result["events_sha256"], "Journal hash mismatch.")
    kinds = collections.Counter()
    blocks, evaluations, refreshes = [], [], []
    accepted_fits = abstained_fits = 0
    current_posterior_hash = None
    previous = 0
    actor_fields = ["q_mean", "lambda", "actor_loss", "bc_loss"]
    extra_fields = ["bc_multiplier_mean", "bc_support_fraction", "posterior_ready",
                    "scale_ess_abstained", "scale_fit_accepted", "scale_inputs_valid", "scale_loss"]
    with gzip.open(run / "events.jsonl.gz", "rt") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["kind"]
            kinds[kind] += 1
            require(event["execution"] == {"method": "bca"}, "Journal method mismatch.")
            if kind == "prepared":
                require(event["step"] == 0 and event["metadata"] == m, "Prepared journal mismatch.")
            elif kind == "accepted_scan":
                step = event["step"]
                require(step == previous + 1000, "Scan boundary mismatch.")
                if current_posterior_hash is None:
                    current_posterior_hash = event["posterior_snapshot_sha256"]
                require(event["posterior_snapshot_sha256"] == current_posterior_hash,
                        "Frozen posterior changed between refreshes.")
                metrics = event["metrics"]
                require(set(metrics) == set(actor_fields + ["critic_loss", "inputs_valid"] + extra_fields),
                        "Unexpected host metrics.")
                values = {k: np.asarray(v) for k, v in metrics.items()}
                require(all(v.shape == (1000,) and np.isfinite(v).all() for v in values.values())
                        and values["inputs_valid"].all(), "Invalid scan rows.")
                require(values["scale_inputs_valid"].all() and values["scale_fit_accepted"].all()
                        and not values["scale_ess_abstained"].any(), "Rejected/abstained scale fit.")
                accepted_fits += int(values["scale_fit_accepted"].sum())
                abstained_fits += int(values["scale_ess_abstained"].sum())
                ready = previous >= 10000
                require((values["posterior_ready"] == ready).all()
                        and (values["bc_support_fraction"] == 1).all(), "Readiness/support differs.")
                doses = values["bc_multiplier_mean"]
                require(((doses >= 1) & (doses <= 1.5)).all()
                        and (ready or (doses == 1).all()), "Dose range/warmup differs.")
                active = np.arange(previous + 1, step + 1) % 2 == 0
                block = dict(step=step, host_rows=1000, actor_rows=int(active.sum()))
                for key in actor_fields + ["critic_loss", "scale_loss", "bc_multiplier_mean"]:
                    selected = values[key][active] if key in actor_fields else values[key]
                    if key in actor_fields:
                        require((values[key][~active] == 0).all(), "Unexpected skipped placeholder.")
                    block[key] = dict(mean=float(selected.mean()), min=float(selected.min()),
                                      max=float(selected.max()), zeros=int((selected == 0).sum()))
                block["actor_bc_multiplier_mean"] = dict(mean=float(doses[active].mean()),
                    min=float(doses[active].min()), max=float(doses[active].max()))
                blocks.append(block)
                previous = step
            elif kind == "refresh":
                wanted = row["protocol"]["refresh_events"][len(refreshes)]
                require(event["step"] == previous == wanted["step"]
                        and event["refresh_seed"] == wanted["seed"], "Refresh schedule differs.")
                require(event["training_reference_ids"] == m["reference_converted_ids"]
                        and event["heldout_ids"] == m["heldout_converted_ids"]
                        and event["reference_sha256"] == m["run_input_hashes"]["reference"]
                        and event["heldout_sha256"] == m["run_input_hashes"]["heldout"],
                        "Frozen refresh data differs.")
                require(event["posterior_weighting"] == "unweighted"
                        and event["posterior_ready"] is True
                        and all(event["metrics"].values())
                        and event["component_diagnostics"]["level_engagement_valid"] is True,
                        "Invalid or weighted posterior refresh.")
                check_radii(event["radii"])
                require(float(event["residual_unit"]) > 0, "Invalid frozen residual scale.")
                current_posterior_hash = event["posterior_snapshot_sha256"]
                refreshes.append({k: v for k, v in event.items()
                                  if k not in ["training_reference_ids", "heldout_ids"]})
            elif kind in ["periodic", "final"]:
                require(event["step"] == previous, "Evaluation precedes its accepted updates.")
                evaluations.append(event)
            elif kind == "completed":
                require(event["step"] == previous == 1000000
                        and event["final_episodes_actual"] == 20,
                        "Invalid completed journal event.")
            else:
                raise ValueError("Unexpected host journal event: " + kind)
    require(kinds == dict(prepared=1, accepted_scan=1000, refresh=198, periodic=200, final=1, completed=1)
            and previous == 1000000 and kind == "completed", "Incomplete journal.")
    require(accepted_fits == 1000000 and abstained_fits == 0, "Scale-fit totals differ.")
    # CPU PRNG arithmetic only; no learner/model or simulator call.
    import jax
    for event in refreshes:
        key = jax.random.fold_in(jax.random.PRNGKey(event["refresh_seed"]), event["step"])
        require(np.array_equal(np.asarray(key), event["refresh_key"]), "Refresh RNG key differs.")
    for checkpoint in checkpoints:
        event = next(r for r in reversed(refreshes) if r["step"] <= checkpoint["step"])
        require(checkpoint["posterior_snapshot_sha256"] == event["posterior_snapshot_sha256"]
                and checkpoint["radii"] == event["radii"]
                and checkpoint["frozen_residual_scale"] == event["residual_unit"],
                "Checkpoint posterior and final preceding refresh differ.")
    require([{k: v for k, v in e.items() if k != "execution"} for e in evaluations]
            == result["evaluations"], "Result/journal evaluation disagreement.")
    for bank, wanted in zip(evaluations, row["protocol"]["evaluation_events"]):
        require(all(bank[k] == wanted[k] for k in ["kind", "step", "episode_seeds"]),
                "Evaluation bank identity differs.")
        require(bank["episode_count"] == len(bank["episodes"]) == len(wanted["episode_seeds"]),
                "Evaluation episode count differs.")
        require(not bank["repeated_episode_seeds"], "Unexpected repeated seeds.")
        transform = bank["score_transform"]
        require(transform == m["evaluation_score_transform"], "Score transform differs.")
        for i, e in enumerate(bank["episodes"]):
            require(e["seed"] == wanted["episode_seeds"][i] and e["episode_index"] == i
                    and 0 < e["length"] <= 1000, "Episode identity/length mismatch.")
            score = 100 * (e["raw_return"] - transform["reference_min"]) / (
                    transform["reference_max"] - transform["reference_min"])
            require(np.isfinite([score, e["normalized_score"]]).all()
                    and abs(score - e["normalized_score"]) < 1e-10, "Score arithmetic differs.")
    final = np.asarray([e["normalized_score"] for e in evaluations[-1]["episodes"]])
    curve = [float(np.mean([e["normalized_score"] for e in bank["episodes"]]))
             for bank in evaluations[:-1]]
    summary = dict(final_mean=float(final.mean()), final_episode_median=float(np.median(final)),
                   final_episode_sd=float(final.std(ddof=1)), final_episode_min=float(final.min()),
                   final_episode_max=float(final.max()), periodic_curve_mean=float(np.mean(curve)),
                   final_checkpoint_periodic_mean=curve[-1], training_seeds=1,
                   training_seed_uncertainty=None, bca_comparison=None)
    evidence_files = ["resolved.json", "source.json", "preparation.json", "result.json",
                      "exit.json", "events.jsonl.gz"]
    audit = dict(schema="standard-td3-hopper-bca-noiw-audit-v1", accepted=True, run_id=RUN_ID,
                 checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 manifest_sha256=MANIFEST, frozen_source_files_checked=108,
                 scientific_source_commit="6e912b0ec3e3346ba6c628b6eef6fddedc74ffc0",
                 actual_worker_exit=actual, host_updates=1000000, critic_updates=1000000,
                 actor_updates=500000, target_updates=500000, checkpoints=checkpoints,
                 journal_counts=dict(kinds), evaluation_episodes=2020,
                 training_rows=len(train_ids), holdout_rows=len(hold_ids), reference_rows=1024,
                 data_cache_sha256=row["cache"]["sha256"], run_input_hashes=m["run_input_hashes"],
                 calibrator_fitting_operations=accepted_fits, posterior_refreshes=len(refreshes),
                 accepted_scale_fits=accepted_fits, abstained_scale_fits=abstained_fits,
                 importance_weighting="off; Bayesian bootstrap fitting masses retained",
                 last_refresh=refreshes[-1],
                 summary=summary, model_queries_added=0, simulator_steps_added=0,
                 learner_updates_added=0, ood_collection_ready=False,
                 evidence_sha256={n: sha(run / n) for n in evidence_files},
                 metric_aggregation="1,000 host-update blocks; actor fields use one-based even / zero-based odd updates. Skipped placeholders excluded; active zeros retained.",
                 note="One paired training seed; four additional paired seeds and OOD acceptance pending.")
    host_banks = read(host_folder / "evaluations.json")
    require(len(host_banks) == len(evaluations), "Paired bank count differs.")
    for left, right in zip(host_banks, evaluations):
        require(all(left[k] == right[k] for k in ["kind", "step", "episode_seeds", "score_transform"]),
                "Paired evaluation identity differs.")
    host_final = np.asarray([e["normalized_score"] for e in host_banks[-1]["episodes"]])
    deltas = final - host_final
    comparison = dict(schema="standard-td3-hopper-pair-v1", training_seed=202609171,
        completed_paired_seeds=1, declared_paired_seeds=5, training_seed_uncertainty=None,
        host_summary=host["summary"], bca_summary=summary,
        final_delta=float(deltas.mean()), curve_delta=summary["periodic_curve_mean"]-host["summary"]["periodic_curve_mean"],
        paired_episode_wins=int((deltas > 0).sum()), paired_episode_ties=int((deltas == 0).sum()),
        paired_episode_deltas=deltas.tolist(), saved_training_rng_equal=True,
        evidence="Descriptive single-seed contrast; reset variation is not training-seed uncertainty.",
        host_artifact_sha256={n:sha(host_folder/n) for n in ["audit.json", "evaluations.json", "metric_blocks.json"]})
    audit["last_10000_update_means"] = {key: float(np.mean([b[key]["mean"] for b in blocks[-10:]]))
        for key in actor_fields + ["critic_loss", "scale_loss", "bc_multiplier_mean", "actor_bc_multiplier_mean"]}
    audit["bayesian_radius_exceeds_conformal_refreshes"] = sum(r["radii"]["bayesian_radius"][0] > r["radii"]["conformal_radius"][0] for r in refreshes)
    audit["frozen_batch_floor_dose_changed_refreshes"] = sum(r["component_diagnostics"]["level_floor_dose_changed_fraction"] > 0 for r in refreshes)
    audit["saved_training_rng_matches_host"] = True
    audit["comparison"] = comparison
    save(out / "refreshes.json", refreshes)
    save(out / "comparison.json", comparison)
    save(out / "evaluations.json", evaluations)
    save(out / "metric_blocks.json", blocks)
    save(out / "audit.json", audit)
    print(json.dumps(dict(summary=summary,final_delta=comparison["final_delta"],curve_delta=comparison["curve_delta"],paired_episode_wins=comparison["paired_episode_wins"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.repo, a.output)
