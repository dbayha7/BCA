"""Bounded auxiliary-only one-sided BCA worker; invoked only by manual launcher.

Importing this module performs no JAX/model/data/calibration work. run() fits a
new copied auxiliary state, never the restored original checkpoint. There are
no simulator imports or policy/critic optimizer calls.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
HANDOFF = HERE.parents[1] / "titan_20261026"
OPERATOR = HANDOFF / "payload/unifloral_operator_alignment_v1"
AUDIT = OPERATOR / "results_20261026T221700_762125Z"
HELPER = HERE.parent / "operator_analysis/cached_score_mechanism_v1/run_cached_comparison.py"
EXPECTED = {
    "root_seed": 2026102671, "auxiliary_updates": 128, "worker_timeout_seconds": 300,
    "local_width_penalty": .005, "local_cal_beta": 20., "aux_learning_rate": .001,
    "kappa": 1., "actor_distance_coefficient": .1,
    "alpha": .1, "credibility": .95, "posterior_draws": 1000,
}
AUDIT_HASHES = {
    "raw_banks.json": "2c24e5844d6d38ffd26c1a662b149fb7b453ce8cacce1eb429c008152d6c6e7e",
    "raw_banks.npz": "cb87c6fe809e94d38715fc45df07605a386222fb9bc2c9a33059c708463d576b",
    "report.json": "c11b32408b59815cb7a75585b9c3aeacb97655827e40ab1110250a981adcefd7",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def safe(value):
    """JSON-safe receipts; raw posterior draws remain lossless in NPZ."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return "positive_infinity" if value == np.inf else "negative_infinity" if value == -np.inf else "nan"
    return value


def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(protocol, output_dir):
    """Return completed/refused report; exceptions retain journal and ledger."""
    for key, value in EXPECTED.items():
        require(protocol.get(key) == value and not isinstance(protocol.get(key), bool), "Frozen worker setting differs: " + key)
    output = Path(output_dir).resolve()
    require(output.is_dir(), "Supervisor must create output directory")
    require(not any((output / n).exists() for n in ("report.json", "aux_params.msgpack", "worker_started.json")), "Existing worker evidence")
    start = time.monotonic()
    ledger = {
        "auxiliary_updates_attempted": 0, "auxiliary_updates_completed": 0,
        "actor_updates": 0, "critic_updates": 0, "target_updates": 0,
        "original_auxiliary_updates": 0, "original_calibration_refreshes": 0,
        "new_calibration_calls": 0, "simulator_transitions": 0, "new_rollouts": 0,
        "confirmation_reads": 0, "logged_current_critic_rows": 0,
        "logged_target_actor_rows": 0, "logged_target_critic_rows": 0,
        "native_label_draws": 0, "auxiliary_gradient_rows": 0,
        "auxiliary_diagnostic_rows": 0, "development_auxiliary_rows": 0,
        "development_value_rows_read": 0,
    }
    new_json(output / "worker_started.json", {"scope": "auxiliary_only", "settings": EXPECTED})

    def stage(name, **details):
        require(time.monotonic() - start < 300, "Worker elapsed limit reached")
        with (output / "stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe({"stage": name, "elapsed_seconds": time.monotonic() - start,
                                          "ledger": dict(ledger), **details}), allow_nan=False) + "\n")

    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        import optax
        from flax import serialization
        from flax.training.train_state import TrainState
        require(jax.default_backend() == "gpu", "Frozen scientific runtime requires GPU")
        require(jax.config.jax_enable_x64 is False and jax.config.jax_disable_jit is False,
                "Frozen JAX flags differ")
        math = module("_one_sided_worker_math", HERE / "one_sided_math.py")
        data = module("_one_sided_worker_data", HERE / "logged_data.py")
        restore = module("_one_sided_worker_restore", HERE / "restore_logged.py")
        old_protocol = json.loads((OPERATOR / "protocol.json").read_text(encoding="utf-8"))
        stage("restore_started")
        bindings = restore.load(old_protocol)
        original_before = bindings.assert_unchanged()
        banks = data.load_logged_banks()
        for name, count in (("fit", 256), ("calibration", 128), ("validation", 128)):
            bank = banks[name]
            shapes = {"obs": (count, 14), "action": (count, 4), "reward": (count,), "next_obs": (count, 14), "done": (count,)}
            for key, shape in shapes.items():
                require(bank[key].shape == shape and bank[key].dtype == np.float32
                        and np.all(np.isfinite(bank[key])), "Logged bank contract differs: " + name + "/" + key)
            require(np.all(np.isin(bank["done"], [0, 1])) and np.all(np.abs(bank["action"]) <= 1), "Mask/action invalid")
            require(len(bank["group_ids"]) == count and len(bank["row_ids"]) == count, "Missing provenance")
        groups = [set(banks[n]["group_ids"].tolist()) for n in ("fit", "calibration", "validation")]
        require(not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2], "Episode leakage")
        require(len(groups[1]) == len(groups[2]) == 128, "Calibration/validation need one row per episode")
        new_json(output / "data_receipt.json", banks["receipt"])
        for key, attr in (("obs", "obs"), ("action", "action"), ("reward", "reward"), ("next_obs", "next_obs"), ("done", "done")):
            require(np.array_equal(banks["fit"][key], np.asarray(getattr(bindings.fitting_transition, attr))), "Fitting bank mismatch: " + key)
        stage("restore_and_data_verified")

        context = bindings.context
        cal_model = context.models[2]
        original_aux = bindings.state.aux
        require(serialization.to_bytes(original_aux.calibrator.params) == serialization.to_bytes(original_aux.posterior.cal_params),
                "Terminal auxiliary/reference parameter boundary differs")
        unit = float(np.asarray(original_aux.residual_scale))
        require(np.isfinite(unit) and unit > 0 and unit == float(np.asarray(original_aux.posterior.residual_scale)), "Invalid frozen residual unit")
        require(context.aux_config.cal_lr == .001 and context.aux_config.cal_beta == 20
                and context.aux_config.width_penalty == .005, "Source auxiliary settings differ")
        require(cal_model.state_dep is True and cal_model.hidden == 64 and cal_model.depth == 2, "Local architecture differs")
        params = jax.tree_util.tree_map(lambda x: jnp.array(x), original_aux.calibrator.params)
        auxiliary = TrainState.create(apply_fn=cal_model.apply, params=params, tx=optax.adam(.001))
        root_key = jax.random.PRNGKey(2026102671)
        labels = {}
        for name, domain in (("fit", 11), ("calibration", 12), ("validation", 13)):
            bank = banks[name]
            n = len(bank["reward"])
            ledger["logged_current_critic_rows"] += n
            q = np.asarray(bindings.policy.q(bank["obs"], bank["action"]), dtype=np.float32)
            require(q.shape == (n, 2) and np.all(np.isfinite(q)), "Current Q shape/numerics")
            ledger["logged_target_actor_rows"] += n
            target_actor = bindings.target_act(bank["next_obs"])
            noise_key = jax.random.fold_in(root_key, domain)
            noise = jax.random.normal(noise_key, shape=(n, 4), dtype=jnp.float32) * jnp.float32(.2)
            target_action = jnp.clip(jnp.asarray(target_actor) + jnp.clip(noise, -.5, .5), -1., 1.)
            ledger["logged_target_critic_rows"] += n
            ledger["native_label_draws"] += 1
            target_q = bindings.target_q(bank["next_obs"], np.asarray(target_action, dtype=np.float32))
            y = jnp.asarray(bank["reward"]) + jnp.float32(.98) * (jnp.float32(1) - jnp.asarray(bank["done"])) * jnp.min(jnp.asarray(target_q), axis=-1)
            labels[name] = {"q1": q[:, 0], "target": np.asarray(y, dtype=np.float32),
                            "noise_key": np.asarray(noise_key), "target_action": np.asarray(target_action)}
            require(np.all(np.isfinite(labels[name]["target"])), "Nonfinite logged target")
        require(all(ledger[k] == 512 for k in ("logged_current_critic_rows", "logged_target_actor_rows", "logged_target_critic_rows")), "Logged model-row cap mismatch")
        np.savez_compressed(output / "logged_labels.npz", **{name + "_" + key: value for name, label in labels.items() for key, value in label.items()})
        stage("logged_labels_frozen", interpretation="One native-law smoothing draw per logged bank; fixed during auxiliary fit")

        @jax.jit
        def eta_fn(parameters, observations, actions):
            return cal_model.apply(parameters, observations, actions)

        def eta_bank(parameters, name):
            bank = banks[name]
            ledger["auxiliary_diagnostic_rows"] += len(bank["reward"])
            eta = np.asarray(eta_fn(parameters, jnp.asarray(bank["obs"]), jnp.asarray(bank["action"])))
            require(eta.shape == (len(bank["reward"]),) and np.all(np.isfinite(eta)) and np.all(eta >= .01), "Invalid positive scale")
            return eta

        def local_diagnostic(eta, name):
            label = labels[name]
            w = np.ones(len(eta), dtype=np.float32) / len(eta)
            loss = float(np.asarray(math.scale_loss(jnp.asarray(eta), jnp.asarray(label["q1"]), jnp.asarray(label["target"]),
                                                  jnp.asarray(unit, jnp.float32), jnp.asarray(w))))
            require(np.isfinite(loss), "Nonfinite local diagnostic")
            return {"uniform_weight_objective": loss, "mean_eta": float(np.mean(eta)),
                    "raw_local_mean_radius": float(np.mean(unit * eta)),
                    "raw_local_directional_coverage": float(np.mean(label["target"] >= label["q1"] - unit * eta))}

        before_local = {name: local_diagnostic(eta_bank(auxiliary.params, name), name) for name in ("fit", "validation")}
        fit = banks["fit"]
        fit_obs, fit_actions = jnp.asarray(fit["obs"]), jnp.asarray(fit["action"])
        fit_q, fit_target = jnp.asarray(labels["fit"]["q1"]), jnp.asarray(labels["fit"]["target"])
        fixed_unit = jnp.asarray(unit, jnp.float32)

        @jax.jit
        def fit_step(aux, key):
            weights = context.pipeline.bayesian_bootstrap_weights(key, 256)
            def objective(parameters):
                eta = cal_model.apply(parameters, fit_obs, fit_actions)
                return math.scale_loss(eta, fit_q, fit_target, fixed_unit, weights,
                                       alpha=.1, beta=20., width_penalty=.005)
            loss, gradient = jax.value_and_grad(objective)(aux.params)
            finite = jnp.isfinite(loss) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(gradient)]))
            proposed = aux.apply_gradients(grads=gradient)
            finite = finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves((proposed.params, proposed.opt_state))]))
            return jax.lax.cond(finite, lambda _: proposed, lambda _: aux, operand=None), loss, finite

        losses = []
        for step in range(128):
            ledger["auxiliary_updates_attempted"] += 1
            ledger["auxiliary_gradient_rows"] += 256
            key = jax.random.fold_in(jax.random.fold_in(root_key, 21), step)
            auxiliary, loss, finite = fit_step(auxiliary, key)
            require(bool(np.asarray(finite)), "Nonfinite fit proposal refused without committing it")
            ledger["auxiliary_updates_completed"] += 1
            losses.append(float(np.asarray(loss)))
            if step in (0, 31, 63, 95, 127):
                stage("auxiliary_fit_progress", updates_completed=step + 1, loss=losses[-1])
        require(int(np.asarray(auxiliary.step)) == 128, "Auxiliary update count mismatch")
        original_after_fit = bindings.assert_unchanged()
        require(original_before == original_after_fit, "Original checkpoint changed during auxiliary fit")
        after_eta = {name: eta_bank(auxiliary.params, name) for name in ("fit", "calibration", "validation")}
        after_local = {name: local_diagnostic(after_eta[name], name) for name in ("fit", "validation")}
        scores = math.directional_scores(labels["calibration"]["q1"], labels["calibration"]["target"], after_eta["calibration"], unit)
        posterior_key = np.asarray(jax.random.fold_in(root_key, 31), dtype=np.uint32)
        posterior_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(posterior_key.tolist())))
        ledger["new_calibration_calls"] += 1
        calibration = context.pipeline.calibration.wbcp.calibrate(scores, posterior_rng, alpha=.1, beta=.95, draws=1000)
        require(ledger["new_calibration_calls"] == 1, "Calibration call cap exceeded")
        q_bayes, q_emp = float(calibration.threshold), float(calibration.lambda_hat)
        calibration_receipt = {
            "threshold": q_bayes, "empirical_threshold": q_emp, "posterior_threshold": float(calibration.lambda_hpd),
            "n_eff": float(calibration.n_eff), "posterior_draws": 1000,
            "finite_posterior_crossings": int(np.isfinite(calibration.crossings).sum()),
            "alpha": .1, "credibility": .95, "calibration_rows": 128, "unique_calibration_episodes": 128,
            "zero_threshold_valid": q_bayes == 0, "certified_numerically": bool(calibration.certified),
            "meaning": "One-sided native-target score calibration; not selected-action or true-return coverage",
        }
        with (output / "aux_params.msgpack").open("xb") as stream:
            stream.write(serialization.to_bytes(auxiliary.params))
        np.savez_compressed(output / "calibration_arrays.npz", scores=scores, crossings=calibration.crossings,
                            **{name + "_eta": eta for name, eta in after_eta.items()})
        new_json(output / "calibration.json", calibration_receipt)
        new_json(output / "AUXILIARY_REFERENCE_SEALED.json", {
            "params_sha256": sha(output / "aux_params.msgpack"), "calibration_sha256": sha(output / "calibration.json"),
            "fixed_unit": unit, "auxiliary_updates": 128, "original_state_unchanged": True,
            "development_values_loaded": False, "fit_labels": "logged fitting transitions only",
        })
        stage("new_auxiliary_reference_sealed", numerical_refusal=not np.isfinite(q_bayes))
        common = {"settings": EXPECTED, "ledger": ledger, "restore": bindings.receipt,
                  "data": banks["receipt"], "unit": unit, "unit_updated": False,
                  "before_local": before_local, "after_local": after_local,
                  "training_losses": losses, "calibration": calibration_receipt,
                  "full_training_hold": True, "actor_critic_updates": 0,
                  "original_snapshot_before": original_before}
        if not np.isfinite(q_bayes):
            common.update(status="refused_nonfinite_one_sided_reference", action_scoring_executed=False,
                          original_snapshot_after=bindings.assert_unchanged(), elapsed_seconds=time.monotonic() - start)
            new_json(output / "report.json", common)
            return safe(common)
        require(q_bayes >= 0 and np.isfinite(q_emp) and q_emp >= 0, "Invalid finite calibration threshold")
        logged_readouts = {}
        for name in ("fit", "calibration", "validation"):
            eta, label = after_eta[name], labels[name]
            logged_readouts[name] = {}
            for method_name, threshold in (("bayesian", q_bayes), ("ordinary", q_emp), ("raw_local", 1.)):
                radius = math.radius(eta, unit, threshold)
                covered = label["target"].astype(np.float64) >= label["q1"].astype(np.float64) - radius
                logged_readouts[name][method_name] = {"rows": len(eta), "covered": int(covered.sum()),
                    "directional_coverage": float(covered.mean()), "mean_radius": float(radius.mean()), "max_radius": float(radius.max())}
        common["logged_readouts"] = logged_readouts
        stage("logged_validation_complete")

        # Development data are evaluation-only and are loaded only after sealing.
        for name, digest in AUDIT_HASHES.items():
            require(sha(AUDIT / name) == digest, "Cached audit input identity changed: " + name)
        require(sha(HELPER) == "8e0d61ccb2b06ac7357b8ed8db39e27d5f3d9273a49f0085a172e0d4ee0c411a", "Selection helpers changed")
        helpers = module("_one_sided_selection_helpers", HELPER)
        raw = json.loads((AUDIT / "raw_banks.json").read_text(encoding="utf-8"))["development"]
        metadata = raw["metadata"]
        require([(m["group_id"], m["candidate_id"]) for m in metadata] == [(g, c) for g in range(128) for c in range(9)], "Development order changed")
        state_metadata = [{k: metadata[g * 9][k] for k in ("group_id", "batch", "time", "episode_seed")} for g in range(128)]
        with np.load(AUDIT / "raw_banks.npz", allow_pickle=False) as archive:
            def decode(name):
                desc = raw[name]
                a = archive[desc["npz_key"]]
                require(a.dtype.str == desc["dtype"] and list(a.shape) == desc["shape"]
                        and hashlib.sha256(a.tobytes()).hexdigest() == desc["array_bytes_sha256"], "Cached array mismatch: " + name)
                require(np.all(np.isfinite(a)), "Nonfinite cached array: " + name)
                return a
            obs, actions = decode("observations"), decode("actions")
            q1 = decode("q_current")[:, 0].astype(np.float64).reshape(128, 9)
            old_u = decode("u").astype(np.float64).reshape(128, 9)
            actors = decode("saved_actor_actions")
            require(np.array_equal(actions.reshape(128, 9, 4)[:, 0], actors), "Actor candidate mismatch")
            ledger["development_auxiliary_rows"] += 1152
            dev_eta = np.asarray(eta_fn(auxiliary.params, jnp.asarray(obs), jnp.asarray(actions)))
            require(dev_eta.shape == (1152,) and np.all(np.isfinite(dev_eta)) and np.all(dev_eta >= .01), "Invalid candidate scale")
            local = (unit * np.maximum(dev_eta.astype(np.float64), 1e-6)).reshape(128, 9)
            distance = np.mean((actions.astype(np.float64).reshape(128, 9, 4) - actors.astype(np.float64)[:, None, :]) ** 2, axis=2)
            old_q = float(np.asarray(original_aux.posterior.threshold))
            method_scores = {
                "no_penalty": q1, "original_bca": q1 - old_u,
                "one_sided_bayesian": q1 - q_bayes * local,
                "one_sided_ordinary": q1 - q_emp * local,
                "raw_local": q1 - local,
                "actor_distance": q1 - .1 * 19.29619572569095 * distance,
                "raw_local_matched_effective_strength": q1 - q_bayes * local,
            }
            availability = {}
            if q_emp > 0:
                method_scores["ordinary_matched_effective_strength"] = q1 - (q_bayes / q_emp) * (q_emp * local)
            elif q_bayes == 0:
                method_scores["ordinary_matched_effective_strength"] = q1.copy()
            else:
                availability["ordinary_matched_effective_strength"] = "unavailable: empirical threshold zero but Bayesian threshold positive"
            if old_q > 0:
                method_scores["original_scale_at_new_threshold"] = q1 - (q_bayes / old_q) * old_u
            elif q_bayes == 0:
                method_scores["original_scale_at_new_threshold"] = q1.copy()
            else:
                availability["original_scale_at_new_threshold"] = "unavailable: original threshold zero but new threshold positive"
            choices = {name: [helpers.select(row) for row in values] for name, values in method_scores.items()}
            choices["frozen_actor"] = [{"index": 0, "near_tie": False,
                                        "selection_rule": "fixed_original_actor_candidate_0"} for _ in range(128)]
            new_json(output / "development_selections.json", {"methods": choices, "scores": {k: v.tolist() for k, v in method_scores.items()},
                                                               "availability": availability, "metadata": state_metadata})
            new_json(output / "DEVELOPMENT_SELECTIONS_SEALED.json", {"sha256": sha(output / "development_selections.json"), "saved_values_read": False})
            values = decode("saved_values").astype(np.float64).reshape(128, 9)
            ledger["development_value_rows_read"] += 1152
        for name in ("raw_local_matched_effective_strength", "ordinary_matched_effective_strength"):
            if name in choices:
                require([c["index"] for c in choices[name]] == [c["index"] for c in choices["one_sided_bayesian"]], "Matched-strength selection discrepancy")
        chosen = {name: np.array([c["index"] for c in records]) for name, records in choices.items()}
        chosen_v = {name: values[np.arange(128), ids] for name, ids in chosen.items()}
        oracle = values.max(axis=1)
        summaries = []
        for stratum, indices in helpers.strata(state_metadata):
            require(len(indices) == (128 if stratum == "overall" else 16 if "/" in stratum
                                    else 64 if stratum.startswith("development_") else 32), "Stratum size mismatch")
            for name, val in chosen_v.items():
                gain = val[indices] - chosen_v["no_penalty"][indices]
                band = helpers.value_epsilon(val[indices], chosen_v["no_penalty"][indices])
                original_band = helpers.value_epsilon(val[indices], chosen_v["original_bca"][indices])
                switches = chosen[name][indices] != chosen["no_penalty"][indices]
                summaries.append({"stratum": stratum, "method": name, "states": len(indices),
                    "mean_selected_value": float(val[indices].mean()), "mean_gain_vs_q1": float(gain.mean()),
                    "mean_gain_vs_original_bca": float((val[indices] - chosen_v["original_bca"][indices]).mean()),
                    "mean_gain_vs_actor_distance": float((val[indices] - chosen_v["actor_distance"][indices]).mean()),
                    "mean_regret": float((oracle[indices] - val[indices]).mean()), "mean_numerical_band": float(band.mean()),
                    "mean_numerical_band_vs_original_bca": float(original_band.mean()),
                    "switches": int(switches.sum()), "helpful_switches": int(((gain > band) & switches).sum()),
                    "harmful_switches": int(((gain < -band) & switches).sum()), "inconclusive_switches": int(((np.abs(gain) <= band) & switches).sum()),
                    "minimum_gain_vs_q1": float(gain.min()), "maximum_gain_vs_q1": float(gain.max()),
                    "near_tie_states": sum(choices[name][i]["near_tie"] for i in indices)})
        new_json(output / "development_outcomes.json", {"metadata": state_metadata, "candidate_values": values.tolist(),
                    "selected_values": {k: v.tolist() for k, v in chosen_v.items()}, "selected_indices": {k: v.tolist() for k, v in chosen.items()}})
        with (output / "all_prespecified_strata.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
        primary = [r for r in summaries if r["method"] == "one_sided_bayesian" and r["stratum"] in ("development_1", "development_2")]
        require(len(primary) == 2, "Missing primary development batch")
        snapshot_after = bindings.assert_unchanged()
        require(snapshot_after == original_before, "Original state changed")
        require(ledger["auxiliary_updates_completed"] == 128 and ledger["auxiliary_gradient_rows"] == 32768
                and ledger["development_auxiliary_rows"] == 1152 and ledger["new_calibration_calls"] == 1, "Final budget mismatch")
        common.update(status="completed_bounded_one_sided_bca_validation", action_scoring_executed=True,
            development_strata=summaries, matched_controls_availability=availability,
            exploratory_two_batch_utility_screen=(q_bayes > 0 and all(
                r["mean_gain_vs_q1"] > r["mean_numerical_band"]
                and r["mean_gain_vs_original_bca"] > r["mean_numerical_band_vs_original_bca"] for r in primary)),
            zero_radius_degenerate=q_bayes == 0,
            primary_batch_rows=primary, original_snapshot_after=snapshot_after, elapsed_seconds=time.monotonic() - start,
            claim_scope="Exploratory local auxiliary adaptation and calibration; no policy-training/OOD/selected-action coverage guarantee")
        stage("completed")
        common["artifact_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name not in ("stages.jsonl", "report.json")}
        new_json(output / "report.json", common)
        return safe(common)
    except BaseException as exc:
        new_json(output / "worker_failure.json", {"error": type(exc).__name__ + ": " + str(exc),
                                                  "ledger": ledger, "elapsed_seconds": time.monotonic() - start})
        raise
