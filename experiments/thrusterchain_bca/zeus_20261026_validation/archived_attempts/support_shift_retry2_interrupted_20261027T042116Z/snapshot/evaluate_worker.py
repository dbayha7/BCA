"""Final-only support-pilot evaluation; imports perform no model or simulator work."""
from pathlib import Path
import json
import time
import numpy as np

from common import HERE, PAYLOAD, atomic_json, digest, module, read, require, sha

prior = module("_support_pilot_rollout_helpers", HERE.parent / "matched_policy_pilot_v1/evaluate_worker.py")
CELLS = ["seed0_broad_count_matched", "seed0_restricted", "seed1_restricted", "seed1_broad_count_matched"]
ARMS = ("native", "bca", "distance")
SEEDS = list(range(2026104000, 2026104032))
POLICIES = [cell + "__" + arm for cell in CELLS for arm in ("untrained",) + ARMS]


def validate_protocol(protocol):
    expected = {"schema": "unifloral.support_shift_pilot.v1", "cells": CELLS,
        "gamma": .98, "horizon": 100, "evaluation_seeds": SEEDS,
        "evaluation_policies": 16, "evaluation_episodes": 512,
        "simulator_transition_cap": 51200, "evaluation_timeout_seconds": 600,
        "evaluation_bootstrap_draws": 2000, "evaluation_bootstrap_seed": 2026103091,
        "return_band_atol": 1e-6, "return_band_rtol": 1e-6,
        "support_cutoff": .5, "support_coordinate": 0, "gap_action_tolerance": 1e-6,
        "native_revision": "f2dc1278eae18ed3e22c92255119c54885414d17",
        "native_source_sha256": "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee",
        "dataset_sha256": "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226"}
    for key, expected_value in expected.items():
        require(type(protocol.get(key)) is type(expected_value) and protocol[key] == expected_value,
                "Evaluation protocol differs: " + key)


def gap_exposure(actions):
    actions = np.asarray(actions)
    require(actions.shape == (32, 100, 4) and np.all(np.isfinite(actions))
            and np.all(np.abs(actions) <= 1), "Malformed deployed action history")
    outside = actions[:, :, 0].astype(np.float64) > .5 + 1e-6
    excess = np.maximum(actions[:, :, 0].astype(np.float64) - .5, 0.)
    return {"action_rows": 3200, "above_cutoff_action_rows": int(outside.sum()),
        "above_cutoff_fraction": float(outside.mean()),
        "episodes_with_gap_action": int(np.any(outside, axis=1).sum()),
        "above_cutoff_fraction_by_episode": outside.mean(axis=1).tolist(),
        "mean_excess_distance": float(excess.mean()),
        "mean_squared_excess_distance": float(np.mean(excess ** 2)),
        "maximum_excess_distance": float(excess.max()),
        "boundary": .5, "visit_tolerance": 1e-6,
        "scope": "Observed action extrapolation exposure; not an OOD calibration guarantee"}


def native_action_rows(value):
    value = np.asarray(value)
    require(value.ndim == 2 and value.shape[1] == 4 and value.dtype == np.float32
            and np.all(np.isfinite(value)) and np.all(np.abs(value) <= 1),
            "Native requested actions must be bounded finite float32 rows")
    return value


def decision_report(returns, exposures, protocol):
    require(set(returns) == set(POLICIES) and set(exposures) == set(POLICIES), "All 16 declared policies are required")
    indices = np.random.Generator(np.random.PCG64(2026103091)).integers(0, 32, size=(2000, 32))
    learning, comparisons = {}, {}
    for cell in CELLS:
        learning[cell] = prior.paired_comparison(returns[cell + "__native"], returns[cell + "__untrained"], protocol, indices)
        comparisons[cell] = {arm: prior.paired_comparison(returns[cell + "__bca"],
            returns[cell + "__" + arm], protocol, indices) for arm in ("native", "distance")}
    adequate = all(item["positive_screen"] for item in learning.values())
    positive = all(item["positive_screen"] for by_arm in comparisons.values() for item in by_arm.values())
    interactions, mechanism = {}, {}
    for seed in (0, 1):
        broad, restricted = f"seed{seed}_broad_count_matched", f"seed{seed}_restricted"
        interactions[str(seed)] = {}
        for control in ("native", "distance"):
            broad_gain = np.asarray(comparisons[broad][control]["paired_differences"])
            restricted_gain = np.asarray(comparisons[restricted][control]["paired_differences"])
            difference = restricted_gain - broad_gain
            interactions[str(seed)][control] = {"mean_relative_benefit_change": float(difference.mean()),
                "paired_differences": difference.tolist(),
                "episode_paired_bootstrap_95pct": np.quantile(difference[indices].mean(axis=1), [.025,.975]).tolist(),
                "scope": "Restricted minus broad BCA-relative gain; descriptive, state/time/reward composition also differs"}
        exposed = any(exposures[restricted + "__" + arm]["above_cutoff_action_rows"] > 0 for arm in ("native", "distance"))
        mechanism[restricted] = {"baseline_gap_exposure_observed": exposed,
            "interpretation": "gap exposure observed; no causal or coverage guarantee" if exposed else "extrapolation mechanism untested regardless of return gains"}
    decision = ("stop_inadequate_learning_without_extension" if not adequate else
                "promising_exploratory_support_shift_screen" if positive else "stop_fixed_support_shift_pilot_without_tuning")
    return {"decision": decision, "learning_adequacy_passed": adequate,
        "primary_screen_passed": adequate and positive, "all_bca_comparisons_positive": positive,
        "learning_comparisons": learning, "bca_comparisons": comparisons,
        "restricted_minus_broad_relative_benefit": interactions,
        "extrapolation_interpretation": mechanism,
        "bootstrap": {"draws": 2000, "seed": 2026103091, "generator": "NumPy PCG64",
            "shared_paired_indices": True, "indices_sha256": digest(indices.astype("<i8").tobytes())},
        "interval_scope": "Conditional paired-episode intervals for fixed actors; two independent initialization outcomes reported separately, not broad superiority inference"}


def normalization_checked(value):
    require(isinstance(value, dict) and value.get("epsilon") == .001, "Native normalization epsilon differs")
    mean, std = np.asarray(value.get("mean")), np.asarray(value.get("std"))
    require(mean.shape == std.shape == (14,) and np.all(np.isfinite(mean))
            and np.all(np.isfinite(std)) and np.all(std >= 0), "Malformed native normalization")
    return {"mean": mean.tolist(), "std": std.tolist(), "epsilon": .001}


def validate_actor_manifest(manifest, cell, arm, protocol, protocol_sha):
    condition, seed_index = cell[6:], int(cell[4])
    require(manifest.get("cell") == cell and manifest.get("condition") == condition
            and manifest.get("seed_index") == seed_index, "Actor cell identity differs")
    require(manifest.get("protocol_sha256") == protocol_sha and manifest.get("completed") is True,
            "Actor protocol/completion differs")
    for field in ("native_source_sha256", "native_revision"):
        require(manifest.get(field) == protocol[field], "Actor native source differs")
    if arm == "untrained":
        require(manifest.get("kind") == "untrained" and manifest.get("arm") == "native", "Untrained actor identity differs")
        expected = {"critic": 0, "actor": 0, "auxiliary": 0, "refresh_accepted": 0}
    else:
        require(manifest.get("kind") == "branch" and manifest.get("arm") == arm, "Final actor identity differs")
        expected = {"critic": 12000, "actor": 6000, "auxiliary": 12000,
                    "actor_target": 6000, "critic_target": 6000, "refresh_accepted": 2,
                    "last_refresh_critic": 11000, "reference_ready": True, "phase": 0}
    require(all(manifest.get("counts", {}).get(k) == v for k, v in expected.items()), "Actor update/schedule counts differ")
    return normalization_checked(manifest.get("normalization"))


def training_seal(directory, cell, stage, protocol_sha):
    """Bind supervisor completion to report, manifest and both state/actor bytes."""
    directory = Path(directory)
    supervisor_raw = prior._checked(directory, "SUPERVISOR_RESULT.json")
    report_raw = prior._checked(directory, "report.json")
    supervisor, report = json.loads(supervisor_raw), json.loads(report_raw)
    kind = "warmup" if stage == "warmup" else "branch"
    require(supervisor.get("job") == cell + "/" + stage and supervisor.get("completed") is True
            and supervisor.get("exit_code") == 0 and supervisor.get("timed_out") is False
            and supervisor.get("interrupted") is False and supervisor.get("report_sha256") == digest(report_raw),
            "Supervisor/report seal differs: " + cell + "/" + stage)
    require(report.get("status") == "completed_support_" + kind and report.get("protocol_sha256") == protocol_sha
            and report.get("execution_counts_known") is True and report.get("cell") == cell,
            "Training report identity differs")
    hashes = {"SUPERVISOR_RESULT.json": digest(supervisor_raw), "report.json": digest(report_raw)}
    names = ["actor_manifest.json", "state_manifest.json"]
    if stage == "warmup":
        names += ["untrained_actor_manifest.json"]
    manifests = {}
    for name in names:
        raw = prior._checked(directory, name)
        require(report.get(name.removesuffix(".json") + "_sha256") == digest(raw), "Training manifest seal differs: " + name)
        hashes[name] = digest(raw)
        manifests[name] = json.loads(raw)
    state_manifest = manifests["state_manifest.json"]
    require(state_manifest.get("artifact") == manifests["actor_manifest.json"].get("final_state")
            and state_manifest.get("protocol_sha256") == protocol_sha and state_manifest.get("cell") == cell
            and state_manifest.get("kind") == kind, "Full-state manifest binding differs")
    for name in ("actor_manifest.json", "untrained_actor_manifest.json"):
        if name not in manifests:
            continue
        manifest = manifests[name]
        keys = ["actor_params"] + (["final_state"] if name == "actor_manifest.json" else [])
        for key in keys:
            artifact = manifest[key]
            raw = prior._checked(directory, artifact["path"])
            require(artifact == {"path": artifact["path"], "sha256": digest(raw), "bytes": len(raw)}, "Sealed actor/state file differs")
            hashes[artifact["path"]] = digest(raw)
    return manifests, hashes


def assemble_exports(protocol, jobs_root, protocol_sha):
    """Pure file verification before any model, reset or simulation call."""
    exports, identities, hashes, norms = {}, {}, {}, {}
    for cell in CELLS:
        condition = cell[6:]
        for stage in ("warmup",) + ARMS:
            directory = Path(jobs_root) / cell / stage
            manifests, sealed = training_seal(directory, cell, stage, protocol_sha)
            arm = "untrained" if stage == "warmup" else stage
            manifest = manifests["untrained_actor_manifest.json" if stage == "warmup" else "actor_manifest.json"]
            norm = validate_actor_manifest(manifest, cell, arm, protocol, protocol_sha)
            if condition in norms:
                require(norm == norms[condition], "Condition normalization changed across seeds/arms")
            norms[condition] = norm
            policy = cell + "__" + arm
            exports[policy] = {"directory": directory, "manifest": manifest, "normalization": norm}
            identities[policy] = {"directory": str(directory), "actor_params": manifest["actor_params"],
                                  "sealed_hashes": sealed, "normalization": norm}
            hashes.update({str(directory / name): wanted for name, wanted in sealed.items()})
    for seed in (0,1):
        left = exports[f"seed{seed}_broad_count_matched__untrained"]["manifest"]["parameter_leaves"]
        right = exports[f"seed{seed}_restricted__untrained"]["manifest"]["parameter_leaves"]
        require(left == right, "Paired fresh actor initial parameters differ across data conditions")
    require(exports["seed0_restricted__untrained"]["manifest"]["parameter_leaves"] != exports["seed1_restricted__untrained"]["manifest"]["parameter_leaves"],
            "Independent actor initializations unexpectedly identical")
    return exports, identities, hashes, norms


def run(protocol, output_dir, job):
    started = time.monotonic()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "EVALUATION_ATTEMPT.json").open("xb") as stream:
        stream.write(prior.encoded({"schema": "unifloral.support.evaluation_attempt.v1", "automatic_retry": False, "policies": POLICIES}))
    ledger = prior.Ledger(output / "progress.json", 51200, 600, started=started)
    try:
        validate_protocol(protocol)
        protocol_raw = (HERE / "protocol.json").read_bytes()
        require(json.loads(protocol_raw) == protocol, "Supplied evaluation protocol differs")
        protocol_sha = digest(protocol_raw)
        plan_raw = (HERE / "DATA_PLAN.json").read_bytes()
        plan = json.loads(plan_raw)
        require(digest(plan_raw) == protocol["data_plan_sha256"] and plan["evaluation_reset_seeds"] == SEEDS
                and plan["known_reset_seed_collisions"] == [], "Data/reset plan differs")
        exports, identities, input_hashes, normalizations = assemble_exports(protocol, job["jobs_root"], protocol_sha)
        ledger.deadline_check()
        native_loader = module("native_loader", PAYLOAD / "unifloral_td3bc_validation_v1/native_loader.py")
        native = native_loader.load_native(PAYLOAD / "unifloral_td3bc_validation_v1/upstream",
                                          PAYLOAD / "unifloral_td3bc_validation_v1/provenance/unifloral.json")
        import jax
        import jax.numpy as jnp
        from flax import serialization
        require(jax.default_backend() == "gpu", "Approved isolated GPU runtime required")
        # No init, optimizer creation, broad-checkpoint restore or calibration.
        applies = {}
        for condition, norm in normalizations.items():
            model = native.DeterministicTanhActor(4, jnp.asarray(norm["mean"], jnp.float32), jnp.asarray(norm["std"], jnp.float32))
            applies[condition] = jax.jit(lambda params, obs, model=model: model.apply(params, obs).sample(seed=None))
        actors, params, before = {}, {}, {}
        for policy, export in exports.items():
            manifest = export["manifest"]
            value = serialization.msgpack_restore(prior._checked(export["directory"], manifest["actor_params"]["path"]))
            description = prior.leaf_descriptions(value)
            require(description == manifest["parameter_leaves"], "Actor numeric leaves differ from sealed manifest")
            params[policy], before[policy] = value, description
            apply = applies[manifest["condition"]]
            actors[policy] = lambda obs, value=value, apply=apply: native_action_rows(apply(value, jnp.asarray(obs, jnp.float32)))
        core, core_receipt = prior.load_core()
        physics = core.Physics(stall=True, stall_slope=1.5)
        starts = []
        for seed in SEEDS:
            ledger.before_reset()
            env = core.ThrusterChainEnv(stall=True, start="recorded", stall_slope=1.5,
                observation_mode="time_aware", task_semantics="finite_horizon")
            try:
                obs, info = env.reset(seed=seed)
                state = env.state
                ledger.after_reset()
                require(env.t == 0 and info["remaining_steps"] == 100 and info["bootstrap_mask"] == 1.
                    and info["continuation_discount"] == .98 and np.array_equal(obs, prior.observation(state[None], 0, 100)[0]),
                    "Canonical reset semantics differ")
                starts.append(state)
            finally:
                env.close()
        starts = np.stack(starts)
        reset_file = prior.atomic_npz(output / "reset_states.npz", reset_seeds=np.asarray(SEEDS, np.int64), states=starts)
        def check_state(states, step):
            require(not np.any(core.canonical.float_radius(states) > core.canonical.FLOAT_BOUND), "Canonical float bound exceeded")
            obs = prior.observation(states, step, 100)
            require(not np.any(obs < np.r_[core.canonical.OBS_LOW, np.float32(0)])
                and not np.any(obs > np.r_[core.canonical.OBS_HIGH, np.float32(1)]), "Canonical observation bounds exceeded")
        returns, exposures, files = {}, {}, {}
        for policy in POLICIES:
            result = prior.rollout(actors[policy], lambda s, a: core.dynamics(s, a, physics), starts,
                                   100, .98, ledger, policy, check_state)
            # Native requested actions already are float32; the deployment cast
            # is exact. Persist both roles, without injecting perturbations.
            files[policy] = prior.atomic_npz(output / (policy + ".npz"), **result,
                requested_actions=result["actions"], executed_actions=result["actions"],
                reset_seeds=np.asarray(SEEDS, np.int64))
            returns[policy], exposures[policy] = result["returns"], gap_exposure(result["actions"])
        for policy, value in params.items():
            require(prior.leaf_descriptions(value) == before[policy], "Evaluation mutated actor parameters")
        for path, wanted in input_hashes.items():
            require(sha(path) == wanted, "Training export changed during evaluation")
        require(sha(HERE / "protocol.json") == protocol_sha and sha(HERE / "DATA_PLAN.json") == protocol["data_plan_sha256"], "Evaluation protocol/plan changed")
        require(prior.core_identity() == {k: core_receipt[k] for k in ("bundle_manifest_sha256", "verified_core_files", "canonical_source_sha256", "matrix_sha256")}, "Canonical source changed")
        require(ledger.data["completed_transition_rows"] == 51200 and ledger.data["completed_episodes"] == 512
                and ledger.data["reset_calls_completed"] == 32, "Final evaluation counts differ")
        ledger.deadline_check()
        ledger.data["status"] = "completed_support_evaluation"
        ledger.persist()
        report = {"status": "completed_support_evaluation", "protocol_sha256": protocol_sha,
            "data_plan_sha256": protocol["data_plan_sha256"], **decision_report(returns, exposures, protocol),
            "counts": ledger.data.copy(), "execution_counts_known": True, "failure_count": 0,
            "actor_identities": identities, "actor_parameters_unchanged": True,
            "initialization": "Two fresh seeds paired across conditions; no restored broad-trained checkpoint",
            "normalization_by_condition": normalizations, "core": core_receipt,
            "native_source": native._source_receipt, "reset_states": reset_file,
            "policy_results": files, "returns_by_policy": {k:v.tolist() for k,v in returns.items()},
            "mean_discounted_returns": {k:float(v.mean()) for k,v in returns.items()},
            "gap_exposure": exposures, "model_initializations_in_evaluator": 0,
            "limitations": ["Exploratory within-task support pilot; reused development/calibration provenance, no confirmation claim.",
                "Action-conditioned filtering also changes state/time/reward composition; retained states reflect original broad behavior.",
                "Gradient matching is only on fixed logged probes; no matched live gradient or Bayesian-specific attribution.",
                "Two initialization outcomes do not establish general superiority; episode intervals are conditional.",
                "Excluded-action exposure is descriptive and supplies no calibration or OOD guarantee.",
                "Previously disclosed runtime versions remain outside upstream JAX/NumPy constraints."],
            "automatic_retry": False}
        atomic_json(output / "report.json", report)
        return report
    except BaseException as error:
        ledger.data.update(status="failed_support_evaluation", failure_count=1,
            failed_policies=int(ledger.data.get("status") == "evaluating"), error=type(error).__name__ + ": " + str(error))
        ledger.persist()
        atomic_json(output / "report.json", {"status": "failed_support_evaluation", "error": ledger.data["error"],
            "counts": ledger.data.copy(), "execution_counts_known": ledger.data["execution_counts_known"],
            "decision": "incomplete_no_utility_conclusion", "automatic_retry": False})
        raise
