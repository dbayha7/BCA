"""Eight fixed fitting blocks; zero updates, no new candidates or scale fitting.

Only the explicit GPU worker calls run(). Imports here are standard-library
only. Raw gradient trees stay in memory and are averaged before the derived
full-256 coefficient is applied. Production code and checkpoints are unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError("Helper was imported from a different path: " + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Missing helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def run(output_dir, checkpoint_manifest, protocol):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import serialization

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError("Worker must create an exclusive output directory")
    stages = (output / "stages.jsonl").open("x", encoding="utf-8", buffering=1)
    checks, cache, patches, completed = [], {}, [], []
    started = time.monotonic()
    rtol, atol = float(protocol["rtol"]), float(protocol["atol"])
    factor = float(protocol["fixed_contrast_factor"])
    zero = {name: 0 for name in ("optimizer_updates", "calibrator_updates", "calibration_refreshes",
                               "simulator_steps", "confirmation_starts", "forbidden_api_calls",
                               "candidate_banks_generated", "scales_refitted")}
    zero.update(verified=False, frozen_state_unchanged=False, fitting_bank_unchanged=False,
                frozen_state_sha256_before=None, frozen_state_sha256_after=None,
                development_labels_read=False, confirmation_data_read=False,
                gradient_blocks_attempted=0, gradient_blocks_completed=0)

    def host(value):
        if isinstance(value, dict):
            return {str(k): host(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [host(v) for v in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        array = np.asarray(value)
        if array.ndim:
            return host(array.tolist())
        value = array.item()
        if isinstance(value, float) and not np.isfinite(value):
            return "nan" if np.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")
        return value

    def emit(value):
        stages.write(json.dumps(host(value), sort_keys=True, allow_nan=False) + "\n")
        stages.flush()

    def require(condition, message):
        if not bool(condition):
            raise AssertionError(message)

    def record(name, fn, prerequisites=()):
        emit({"event": "started", "name": name, "elapsed_seconds": time.monotonic() - started})
        missing = [key for key in prerequisites if key not in cache]
        if missing:
            result = {"name": name, "status": "skip", "reason": "Failed prerequisites: " + ", ".join(missing)}
        else:
            try:
                result = {"name": name, "status": "pass", "details": host(fn())}
            except Exception as exc:
                result = {"name": name, "status": "fail", "error_type": type(exc).__name__,
                          "error": str(exc), "traceback": traceback.format_exc(limit=8)}
        checks.append(result)
        emit({"event": "finished", "elapsed_seconds": time.monotonic() - started, **result})

    def block(value):
        return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)

    def flat(tree):
        values = [np.asarray(x, np.float64).reshape(-1) for x in jax.tree_util.tree_leaves(tree)]
        return np.concatenate(values) if values else np.zeros(0, np.float64)

    def norm(tree):
        return float(np.linalg.norm(flat(tree)))

    def finite(tree, label):
        require(all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree)), label + " contains nonfinite values")

    def combine(*terms):
        return jax.tree_util.tree_map(lambda *values: sum(coefficient * value for (coefficient, _), value in zip(terms, values)),
                                      *(tree for _, tree in terms))

    def close(a, b, label):
        a, b = np.asarray(a), np.asarray(b)
        require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)), label + " invalid shape/numerics")
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=False, err_msg=label)
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0

    def norm_close(actual, expected, label):
        a, b = flat(actual), flat(expected)
        require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)), label + " invalid shape/numerics")
        error = float(np.linalg.norm(a - b))
        bound = atol + rtol * max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
        require(error <= bound, f"{label}: norm error {error} exceeds {bound}")
        return {"error_norm": error, "allowed_norm_error": bound, "rtol": rtol, "atol": atol}

    def geometry(left, right):
        a, b = flat(left), flat(right)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        dot = float(np.dot(a, b))
        sum_norm = float(np.linalg.norm(a + b))
        return {"left_norm": na, "right_norm": nb, "dot": dot,
                "cosine": None if na == 0 or nb == 0 else dot / (na * nb),
                "right_to_left_norm_ratio": None if na == 0 else nb / na,
                "ratio_status": "unavailable_zero_denominator" if na == 0 else "available",
                "sum_norm": sum_norm, "sum_norm_over_norm_sum": None if na + nb == 0 else sum_norm / (na + nb)}

    def signed_cosine_summary(values):
        defined = [value for value in values if value is not None]
        require(all(np.isfinite(value) and abs(value) <= 1 + atol + rtol for value in defined),
                "Cosine summary requires finite values within numerical cosine bounds")
        return {"count": len(values), "defined": len(defined), "unavailable": len(values) - len(defined),
                "min": min(defined) if defined else None, "max": max(defined) if defined else None,
                "median": float(np.median(defined)) if defined else None,
                "mean": float(np.mean(defined)) if defined else None,
                "interpretation": "Signed descriptive alignment; negative cosines are valid, not a failed diagnostic"}

    def digest(value):
        return hashlib.sha256(serialization.to_bytes(value)).hexdigest()

    def guard(obj, name):
        if not hasattr(obj, name):
            return
        original = getattr(obj, name)
        def forbidden(*args, **kwargs):
            zero["forbidden_api_calls"] += 1
            raise RuntimeError("Forbidden zero-update API: " + name)
        patches.append((obj, name, original))
        setattr(obj, name, forbidden)

    def initialize():
        require(rtol == 2e-5 and atol == 2e-6, "Tolerance changed")
        require(protocol["fitting_rows"] == 256 and protocol["block_size"] == 32, "Block budget changed")
        require(protocol["kappa_grid"] == [0, .03, .1, .3, 1, 3, 10], "Kappa grid changed")
        require(factor == 31.372587203979492 and float(np.float32(factor)) == factor, "Frozen factor changed or is not native-representable")
        require(protocol["native_alpha"] == 2.5 and protocol["native_q_epsilon"] == 1e-7, "Native constants changed")
        contract = _module("_bca_blocks_contract", ROOT / "contract.py")
        plan = contract.block_plan(protocol["fitting_rows"], protocol["block_size"], protocol["block_starts"])
        cache["plan"] = plan
        reference_root = (ROOT.parent / protocol["reference_results"]).resolve()
        require(reference_root.is_relative_to(ROOT.parent.resolve()), "Reference receipt path escapes workspace")
        report_raw = (reference_root / "report.json").read_bytes()
        checks_raw = (reference_root / "checks.json").read_bytes()
        require(hashlib.sha256(report_raw).hexdigest() == protocol["reference_report_sha256"], "Prior report hash changed")
        require(hashlib.sha256(checks_raw).hexdigest() == protocol["reference_checks_sha256"], "Prior checks hash changed")
        old_report, old = json.loads(report_raw), json.loads(checks_raw)
        require(old_report["status"] == "pass" and old_report["exit_code"] == 0 and old_report["zero_work_verified"] is True,
                "Prior fitting-only GPU run did not pass")
        require(old["status"] == "pass" and old["passed"] is True and len(old["checks"]) == 11
                and all(row["status"] == "pass" for row in old["checks"]), "Prior checks are incomplete")
        require(old_report["checks_sha256"] == protocol["reference_checks_sha256"], "Prior report/check identity differs")
        require(old_report["identity_before"] == old_report["identity_after"], "Prior sources changed")
        for name in ("optimizer_updates", "calibrator_updates", "calibration_refreshes", "simulator_steps", "confirmation_starts", "forbidden_api_calls"):
            require(type(old["zero_work"][name]) is int and old["zero_work"][name] == 0, "Prior zero-work field differs: " + name)
        require(old["zero_work"]["verified"] is True and old["zero_work"]["frozen_state_unchanged"] is True,
                "Prior frozen-state verification missing")
        reference = {row["name"]: row["details"] for row in old["checks"]}
        require(reference["detached_fitting_statistics_live_action_route"]["factor"] == factor, "Factor is not the saved fitted scalar")
        prior = ROOT.parent / "unifloral_frozen_usefulness_v1_seedfix"
        inference = _module("_bca_blocks_inference", prior / "inference.py")
        load_protocol = json.loads((prior / "protocol.json").read_text(encoding="utf-8"))
        policy, bank, receipt = inference.load(Path(checkpoint_manifest), load_protocol)
        _, components, repeated = inference.read_contract(Path(checkpoint_manifest), load_protocol)
        require(receipt["file_hashes"] == repeated["file_hashes"], "Consumed export bytes changed during load")
        old_checkpoint = reference["authenticated_fitting_checkpoint_and_zero_update_guards"]["checkpoint"]
        require(receipt["manifest_sha256"] == old_checkpoint["manifest_sha256"] and receipt["file_hashes"] == old_checkpoint["file_hashes"],
                "Checkpoint differs from the first-block anchor")
        require(components["scale_bank"]["sha256"] == reference["authenticated_fitting_checkpoint_and_zero_update_guards"]["fitting_bank_sha256"],
                "Fitting bank differs from the first-block anchor")
        payload = serialization.msgpack_restore(components["parameters"]["bytes"])
        stage1, stage2 = ROOT.parent / "unifloral_td3bc_validation_v1", ROOT.parent / "unifloral_bca_pipeline_validation_v1"
        loader = _module("native_loader", stage1 / "native_loader.py")
        native = loader.load_native(stage1 / "upstream", stage1 / "provenance/unifloral.json")
        bridge = _module("bca_bridge", stage2 / "bca_bridge.py")
        bca = bridge.load_pipeline(stage2)
        candidate = _module("_bca_blocks_candidate", stage1 / "candidate.py")
        mean, std = jnp.asarray(payload["obs_mean"]), jnp.asarray(payload["obs_std"])
        actor, critic = native.DeterministicTanhActor(4, mean, std), native.DualQNetwork(mean, std)
        models = (None, None, bca.Calibrator(mean, std, state_dep=True))
        params, ref = policy.actor_params, policy.reference
        args = native.Args(batch_size=32, gamma=float(receipt["manifest"]["gamma"]))
        agent = SimpleNamespace(dual_q=SimpleNamespace(params=policy.critic_params))
        data_obs, data_actions = jnp.asarray(bank["observations"]), jnp.asarray(bank["actions"])
        dataset = native.Transition(data_obs, data_actions, jnp.zeros(256, jnp.float32), data_obs, jnp.zeros(256, jnp.float32))
        width_fn = lambda rr, obs, actions: bca.live_width(models, rr, obs, actions)[0]
        fn0 = candidate.make_microsteps(args, actor.apply, critic.apply, dataset, width_fn=width_fn, kappa_penalty=0.)
        fn1 = candidate.make_microsteps(args, actor.apply, critic.apply, dataset, width_fn=width_fn, kappa_penalty=1.)

        def components_fn(pp, obs, recorded):
            actions = jax.vmap(lambda row: actor.apply(pp, row).sample(seed=None))(obs)
            q = jax.vmap(lambda oo, ac: critic.apply(policy.critic_params, oo, ac)[0])(obs, actions)
            u, usable, valid = bca.live_width(models, ref, obs, actions)
            bc = jnp.square(actions - recorded).mean(axis=-1).mean()
            values = jnp.stack((q.mean(), u.mean(), bc))
            return values, {"values": values, "mean_abs_q": jnp.abs(q).mean(), "actions": actions,
                            "width_usable": usable, "width_valid": valid}

        def actual_loss(pp, obs, recorded, function):
            n = obs.shape[0]
            batch = native.Transition(obs, recorded, jnp.zeros(n, obs.dtype), obs, jnp.zeros(n, obs.dtype))
            return function.actor_loss(pp, agent, batch, ref)

        def action_q(actions, obs):
            return critic.apply(policy.critic_params, obs, actions)[..., 0].sum()

        def action_u(actions, obs):
            return bca.live_width(models, ref, obs, actions)[0].sum()

        ref_fields = (ref.cal_params, ref.residual_scale, ref.threshold, ref.lambda_hat, ref.lambda_hpd, ref.n_eff)
        def reference_width(fields, obs, actions):
            rr = ref._replace(cal_params=fields[0], residual_scale=fields[1], threshold=fields[2],
                              lambda_hat=fields[3], lambda_hpd=fields[4], n_eff=fields[5])
            return bca.live_width(models, rr, obs, actions)[0].mean()

        cache.update(contract=contract, plan=plan, reference=reference, policy=policy, bank=bank,
            bank_digest=digest(bank), receipt=receipt, params=params, data_obs=data_obs, data_actions=data_actions,
            ref_fields=ref_fields, components=jax.jit(jax.jacrev(components_fn, argnums=0, has_aux=True)),
            vg0=jax.jit(jax.value_and_grad(lambda pp, obs, ac: actual_loss(pp, obs, ac, fn0), has_aux=True)),
            vg1=jax.jit(jax.value_and_grad(lambda pp, obs, ac: actual_loss(pp, obs, ac, fn1), has_aux=True)),
            action_gradients=jax.jit(lambda ac, obs: (jax.grad(action_q)(ac, obs), jax.grad(action_u)(ac, obs))),
            reference_gradient=jax.jit(jax.grad(reference_width, argnums=0)))
        zero["frozen_state_sha256_before"] = policy.digest()
        require(zero["frozen_state_sha256_before"] == old["zero_work"]["frozen_state_sha256_after"], "Frozen numerical state differs from anchor")
        for name in ("fit_scale", "refresh", "freeze_reference", "initialize_pipeline", "initial_reference"):
            guard(bca, name)
        for name in ("create_train_state", "make_train_step", "eval_agent"):
            guard(native, name)
        guard(loader, "initialize")
        guard(bridge, "initialize_pipeline")
        from flax.training.train_state import TrainState
        import optax
        guard(TrainState, "apply_gradients")
        guard(optax, "apply_updates")
        guard(optax, "incremental_update")
        cache["ready"] = True
        return {"checkpoint": receipt, "reference_report_sha256": protocol["reference_report_sha256"],
                "reference_checks_sha256": protocol["reference_checks_sha256"], "block_plan": plan,
                "fixed_contrast_factor": factor, "factor_source": "Prior passing fitting-only receipt; unchanged float32 scalar",
                "fitting_bank_sha256": components["scale_bank"]["sha256"], "fitting_bank_digest": cache["bank_digest"],
                "candidate_banks_generated": 0, "scales_refitted": 0, "guarded_api_count": len(patches),
                "actor_batch_unused_fields": "Synthetic zero reward/done and next_obs=obs placeholders; never used by actor_loss",
                "jit_reuse": "Same compiled functions, dynamic 32-row observation/action arguments across all eight blocks"}

    def grids(gq, gu, gbc, coefficient, action_q, action_u, action_bc):
        qnorm = np.linalg.norm(np.asarray(action_q, np.float64), axis=1)
        unorm = np.linalg.norm(np.asarray(action_u, np.float64), axis=1)
        result = []
        for kappa in protocol["kappa_grid"]:
            row = {"kappa_penalty": kappa}
            for label, multiplier in (("raw_u", 1.), ("fixed_factor_u", factor)):
                effective = float(kappa) * multiplier
                qterm, penalty = combine((-coefficient, gq)), combine((coefficient * effective, gu))
                value_gradient = combine((1, qterm), (1, penalty))
                total = combine((1, value_gradient), (1, gbc))
                ratios = [None if q == 0 else effective * u / q for q, u in zip(qnorm, unorm)]
                action_cosines = [geometry(-np.asarray(q), effective * np.asarray(u))["cosine"] for q, u in zip(action_q, action_u)]
                row[label] = {"effective_raw_u_coefficient": effective,
                    "parameter_q_vs_penalty": geometry(qterm, penalty),
                    "parameter_q_plus_penalty_vs_bc": geometry(value_gradient, gbc),
                    "parameter_total_gradient_norm": norm(total),
                    "action_penalty_to_q_norm_ratios": ratios,
                    "action_ratio_summary": cache["contract"].ratio_summary(ratios),
                    "action_q_vs_penalty_cosines": action_cosines,
                    "action_q_vs_penalty_cosine_summary": signed_cosine_summary(action_cosines),
                    "action_bc_to_native_weighted_q_norm_ratios": [None if q == 0 or coefficient == 0 else float(np.linalg.norm(np.asarray(bc, np.float64))) / (coefficient * q)
                                                                  for q, bc in zip(qnorm, action_bc)]}
            result.append(row)
        return result

    def anchor_comparison(details):
        old = cache["reference"]["actual_candidate_loss_and_gradient_identity"]
        errors = {}
        for name in ("loss_kappa0", "loss_kappa1", "mean_abs_q1_same32_actor_actions", "q_scale_coefficient", "q_mean", "u_mean", "bc_mean_over_four_then_rows"):
            errors[name] = close(details[name], old[name], "Block0 anchor " + name)
        for name in ("mean_q1", "mean_u", "bc"):
            errors["parameter_" + name] = close(details["parameter_gradient_norms"][name], old["parameter_gradient_norms"][name], "Block0 parameter norm " + name)
        for name in ("q1", "u", "bc_per_row"):
            for statistic in ("min", "median", "max", "mean"):
                errors["action_" + name + "_" + statistic] = close(details["action_gradient_norms"][name][statistic], old["action_gradient_norms"][name][statistic], "Block0 action norm anchor")
        old_grid = cache["reference"]["fixed_kappa_grid_gradient_geometry"]["grid"]
        for current, previous in zip(details["grid"], old_grid):
            require(current["kappa_penalty"] == previous["kappa_penalty"], "Anchor grid ordering changed")
            raw = current["raw_u"]
            prefix = "kappa_" + str(current["kappa_penalty"])
            errors[prefix + "_parameter_ratio"] = close(raw["parameter_q_vs_penalty"]["right_to_left_norm_ratio"], previous["parameter_q_vs_penalty"]["right_to_left_norm_ratio"], "Block0 parameter ratio anchor")
            errors[prefix + "_total_norm"] = close(raw["parameter_total_gradient_norm"], previous["parameter_total_gradient_norm"], "Block0 total gradient anchor")
            errors[prefix + "_action_ratios"] = close(raw["action_penalty_to_q_norm_ratios"], previous["action_penalty_to_q_norm_ratios"], "Block0 action ratios anchor")
        errors["fixed_factor_parameter_ratio"] = close(details["grid"][4]["fixed_factor_u"]["parameter_q_vs_penalty"]["right_to_left_norm_ratio"],
            cache["reference"]["detached_fitting_statistics_live_action_route"]["hypothetical_kappa1_parameter_q_vs_penalty"]["right_to_left_norm_ratio"], "Block0 frozen-factor parameter ratio")
        return {"passed": True, "comparisons": len(errors), "maximum_absolute_errors": errors,
                "rtol": rtol, "atol": atol, "scope": "Saved first-block scalar/norm/ratio anchor; prior full gradient trees were not stored"}

    def evaluate_block(start, stop):
        zero["gradient_blocks_attempted"] += 1
        require(zero["gradient_blocks_attempted"] <= 8 and stop - start == 32, "Gradient block cap exceeded")
        obs, recorded = cache["data_obs"][start:stop], cache["data_actions"][start:stop]
        jacobian, aux = block(cache["components"](cache["params"], obs, recorded))
        gq, gu, gbc = [jax.tree_util.tree_map(lambda x, index=index: x[index], jacobian) for index in range(3)]
        require(bool(aux["width_usable"]) and bool(aux["width_valid"]), "Actual frozen width is unusable or invalid")
        (loss0, native_aux), g0 = block(cache["vg0"](cache["params"], obs, recorded))
        (loss1, penalty_aux), g1 = block(cache["vg1"](cache["params"], obs, recorded))
        aq, au = block(cache["action_gradients"](aux["actions"], obs))
        abc = 2 * (aux["actions"] - recorded) / 4.
        reference_grad = block(cache["reference_gradient"](cache["ref_fields"], obs, aux["actions"]))
        finite((jacobian, aux["values"], aux["mean_abs_q"], g0, g1, aq, au, abc, reference_grad), "Block gradients/values")
        require(norm(reference_grad) == 0., "Frozen reference received a gradient")
        c = float(native_aux["q_scale_coefficient"])
        close(c, penalty_aux["q_scale_coefficient"], "Same coefficient in actual kappa0/kappa1")
        expected_c = jax.lax.stop_gradient(protocol["native_alpha"] / (aux["mean_abs_q"] + protocol["native_q_epsilon"]))
        close(c, expected_c, "Coefficient must use this block's actor-Q mean magnitude")
        identity = norm_close(combine((1, g1), (-1, g0)), combine((c, gu)), "Actual block gradient increment")
        native_identity = norm_close(g0, combine((-c, gq), (1, gbc)), "Actual native block decomposition")
        increment_error = close(loss1 - loss0, c * aux["values"][1], "Actual block loss increment")
        cq = np.linalg.norm(np.asarray(aq, np.float64), axis=1)
        cu = np.linalg.norm(np.asarray(au, np.float64), axis=1)
        cb = np.linalg.norm(np.asarray(abc, np.float64), axis=1)
        grid = grids(gq, gu, gbc, c, aq, au, abc)
        details = {"start": start, "stop_exclusive": stop, "rows": stop - start,
            "block_inputs_sha256": digest({"observations": obs, "recorded_actions": recorded}),
            "mean_abs_q1_same32_actor_actions": float(aux["mean_abs_q"]), "q_scale_coefficient": c,
            "near_zero_q_informational": {
                "threshold": protocol["near_zero_q_multiple"] * protocol["native_q_epsilon"],
                "mean_abs_q_at_or_below_threshold": float(aux["mean_abs_q"]) <= protocol["near_zero_q_multiple"] * protocol["native_q_epsilon"],
                "role": "Informational only; does not reject a block or change native epsilon"},
            "q_mean": float(aux["values"][0]), "u_mean": float(aux["values"][1]), "bc_mean_over_four_then_rows": float(aux["values"][2]),
            "loss_kappa0": float(loss0), "loss_kappa1": float(loss1),
            "gradient_identity": identity, "native_gradient_decomposition": native_identity,
            "loss_increment_max_absolute_error": increment_error,
            "parameter_gradient_norms": {"mean_q1": norm(gq), "mean_u": norm(gu), "bc": norm(gbc)},
            "action_gradient_norms": {name: cache["contract"].ratio_summary(values.tolist()) for name, values in (("q1", cq), ("u", cu), ("bc_per_row", cb))},
            "reference_gradient_norm": norm(reference_grad), "action_u_gradient_norm": norm(au),
            "live_action_gradient_status": "nonzero" if norm(au) > 0 else "zero_observed_not_evidence_of_routing_failure_or_usefulness",
            "fixed_contrast_factor": factor, "grid": grid,
            "scope": "Actual kappa0/1 actor_loss, algebraic fixed-state grids; no optimizer or seven-run claim"}
        cache["policy"].assert_unchanged()
        require(digest(cache["bank"]) == cache["bank_digest"], "Fitting bank changed")
        if start == 0:
            details["prior_first_block_comparison"] = anchor_comparison(details)
            cache["block0_verified"] = True
        completed.append({"start": start, "count": stop - start, "gq": gq, "gu": gu, "gbc": gbc,
                          "mean_abs_q": float(aux["mean_abs_q"]), "values": aux["values"],
                          "aq": aq, "au": au, "abc": abc, "details": details})
        cache["block_" + str(start)] = True
        zero["gradient_blocks_completed"] += 1
        return details

    def whole_batch():
        ordered = sorted(completed, key=lambda value: value["start"])
        require(len(ordered) == 8 and [row["start"] for row in ordered] == list(protocol["block_starts"]), "Full fitting coverage is incomplete")
        counts = [row["count"] for row in ordered]
        weights = cache["contract"].normalized_weights(counts)
        averages = {name: combine(*[(weight, row[name]) for weight, row in zip(weights, ordered)])
                    for name in ("gq", "gu", "gbc")}
        m = cache["contract"].weighted_mean([row["mean_abs_q"] for row in ordered], counts)
        c = float(jax.lax.stop_gradient(protocol["native_alpha"] / (jnp.float32(m) + protocol["native_q_epsilon"])))
        aq, au, abc = [jnp.concatenate([row[name] for row in ordered], axis=0) for name in ("aq", "au", "abc")]
        grid = grids(averages["gq"], averages["gu"], averages["gbc"], c, aq, au, abc)
        cancellation = {}
        for name in ("gq", "gu", "gbc"):
            average_norm = cache["contract"].weighted_mean([norm(row[name]) for row in ordered], counts)
            cancellation[name] = {"norm_of_weighted_mean_gradient": norm(averages[name]),
                                  "weighted_mean_of_block_gradient_norms": average_norm,
                                  "ratio": None if average_norm == 0 else norm(averages[name]) / average_norm}
        block_ratio_summary = []
        for index, kappa in enumerate(protocol["kappa_grid"]):
            entry = {"kappa_penalty": kappa}
            for label in ("raw_u", "fixed_factor_u"):
                entry[label] = {"parameter_penalty_to_q_ratio_across_blocks": cache["contract"].ratio_summary(
                    [row["details"]["grid"][index][label]["parameter_q_vs_penalty"]["right_to_left_norm_ratio"] for row in ordered]),
                    "parameter_total_gradient_norm_across_blocks": cache["contract"].ratio_summary(
                    [row["details"]["grid"][index][label]["parameter_total_gradient_norm"] for row in ordered])}
            block_ratio_summary.append(entry)
        finite((averages, c), "Derived full256 gradient")
        return {"scope": "Derived full256 mean objective, not an additional direct full256 autodiff evaluation",
            "direct_full256_gradient_evaluated": False, "rows": sum(counts), "block_weights": weights,
            "mean_abs_q_from_equal_row_block_means": m, "native_float32_mean_abs_q_used": float(jnp.float32(m)),
            "q_scale_coefficient_applied_once_after_raw_component_averaging": c,
            "component_means": {name: cache["contract"].weighted_mean([float(row["values"][index]) for row in ordered], counts)
                                for index, name in enumerate(("q_mean", "u_mean", "bc_mean"))},
            "raw_component_gradient_norms": {name: norm(tree) for name, tree in averages.items()},
            "between_block_component_cancellation": cancellation,
            "grid": grid, "descriptive_per_block_ratio_ranges": block_ratio_summary,
            "per_block_normalized_gradients_or_ratios_averaged_to_define_full_batch": False,
            "interpretation": "All raw component gradients share frozen parameters; c_all is detached and computed from all256 actor-Q magnitudes. Block float32 reduction rounding may differ from a direct256 JAX path.",
            "no_heterogeneity_acceptance_threshold": True}

    try:
        record("authenticated_fitting_bank_fixed_factor_and_guards", initialize)
        for start, stop in cache.get("plan", ()):
            prerequisites = ("ready",) if start == 0 else ("ready", "block0_verified")
            record(f"fitting_block_{start:03d}_{stop:03d}", lambda start=start, stop=stop: evaluate_block(start, stop), prerequisites)
        record("derived_full256_gradient_and_descriptive_block_ranges", whole_batch,
               tuple("block_" + str(start) for start in protocol["block_starts"]))
    finally:
        if "policy" in cache:
            try:
                cache["policy"].assert_unchanged()
                zero["frozen_state_sha256_after"] = cache["policy"].digest()
                zero["frozen_state_unchanged"] = zero["frozen_state_sha256_after"] == zero["frozen_state_sha256_before"]
                zero["fitting_bank_unchanged"] = digest(cache["bank"]) == cache["bank_digest"]
                zero["verified"] = bool(cache.get("ready")) and zero["frozen_state_unchanged"] and zero["fitting_bank_unchanged"] and zero["forbidden_api_calls"] == 0
            except Exception as exc:
                checks.append({"name": "frozen_state_integrity", "status": "fail", "error": str(exc)})
        emit({"event": "zero_work_verification", "zero_work": zero})
        for obj, name, original in reversed(patches):
            setattr(obj, name, original)
        stages.close()
    passed = bool(checks) and all(row["status"] == "pass" for row in checks) and zero["verified"]
    return {"schema": "unifloral.fitting_gradient_blocks_checks.v1", "status": "pass" if passed else "fail",
            "passed": passed, "checks": checks, "zero_work": zero,
            "phase": protocol.get("phase", "fitting_only_gradient_blocks"), "elapsed_seconds": time.monotonic() - started,
            "fixed_contrast_factor": factor, "coefficient_selected": False, "production_normalization_changed": False,
            "tolerances": {"rtol": rtol, "atol": atol, "gradient_identity": "L2 error <= atol + rtol*max(left norm,right norm)"},
            "limitations": ["Single frozen checkpoint and its stored fitting bank; no efficacy or confirmation result.",
                            "All kappa-grid gradients are algebraic reconstructions checked against actual kappa0/1 losses.",
                            "The fixed scalar is equivalent to multiplying kappa, not an identified adaptive normalization rule.",
                            "The derived full256 gradient is not an average of normalized block gradients or ratios.",
                            "Finite zero gradient/ratio denominators are explicitly unavailable; heterogeneity is descriptive, not a pass/fail criterion."]}
