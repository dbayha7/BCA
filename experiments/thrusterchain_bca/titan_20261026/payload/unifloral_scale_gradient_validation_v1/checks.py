"""Fitting-only, zero-update diagnostics for an authenticated frozen checkpoint.

Scientific imports and calculations occur only inside run(), invoked by the
explicit user-run worker. No evaluation labels, simulator, fitting, refresh,
optimizer step, or production normalization change is part of this module.
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
            raise RuntimeError("Helper module came from a different path: " + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def run(output_dir, checkpoint_manifest, protocol):
    """Return JSON-safe details; the worker owns checks.json, we own stages.jsonl."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import serialization

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError("The worker must create an exclusive output directory")
    stream = (output / "stages.jsonl").open("x", encoding="utf-8", buffering=1)
    checks, cache, patches = [], {}, []
    rtol, atol = float(protocol["rtol"]), float(protocol["atol"])
    started = time.monotonic()
    zero = {name: 0 for name in ("optimizer_updates", "calibrator_updates", "calibration_refreshes",
                               "simulator_steps", "confirmation_starts", "forbidden_api_calls")}
    zero.update(verified=False, frozen_state_unchanged=False,
                frozen_state_sha256_before=None, frozen_state_sha256_after=None,
                development_labels_read=False, confirmation_data_read=False)

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
        item = array.item()
        if isinstance(item, float) and not np.isfinite(item):
            return "nan" if np.isnan(item) else ("positive_infinity" if item > 0 else "negative_infinity")
        return item

    def emit(item):
        stream.write(json.dumps(host(item), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()

    def require(condition, message):
        if not bool(condition):
            raise AssertionError(message)

    def record(name, fn, requires=()):
        emit({"event": "started", "name": name, "elapsed_seconds": time.monotonic() - started})
        missing = [key for key in requires if key not in cache]
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
        require(all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree)), label + " is nonfinite")

    def combine(*terms):
        return jax.tree_util.tree_map(lambda *xs: sum(c * x for c, x in zip((t[0] for t in terms), xs)),
                                      *(t[1] for t in terms))

    def norm_close(actual, expected, label):
        a, b = flat(actual), flat(expected)
        require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)), label + " invalid shape/numerics")
        error = float(np.linalg.norm(a - b))
        bound = atol + rtol * max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
        require(error <= bound, f"{label}: norm error {error} exceeds {bound}")
        return {"error_norm": error, "allowed_norm_error": bound, "rtol": rtol, "atol": atol}

    def close(actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)), label + " invalid shape/numerics")
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=False, err_msg=label)
        return {"maximum_absolute_error": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0,
                "rtol": rtol, "atol": atol}

    def describe(values):
        x = np.asarray(values, np.float64).reshape(-1)
        require(x.size > 0 and np.all(np.isfinite(x)), "Summary requires finite nonempty values")
        return {"count": int(x.size), "min": float(x.min()), "median": float(np.median(x)),
                "max": float(x.max()), "mean": float(x.mean())}

    def native_factor(value):
        if value is None or not np.isfinite(value) or value <= 0:
            return {"status": "unavailable", "factor_native": None, "reason": "No finite positive host factor"}
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            cast = np.float32(value)
        if not np.isfinite(cast) or cast <= 0:
            return {"status": "unavailable", "factor_native": None,
                    "reason": "Host factor overflows or underflows in native float32; no clipping or fallback"}
        return {"status": "available", "factor_native": float(cast), "reason": None}

    def geometry(a, b):
        x, y = flat(a), flat(b)
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        dot = float(np.dot(x, y))
        return {"left_norm": nx, "right_norm": ny, "dot": dot,
                "cosine": None if nx == 0 or ny == 0 else dot / (nx * ny),
                "right_to_left_norm_ratio": None if nx == 0 else ny / nx,
                "ratio_status": "unavailable_zero_denominator" if nx == 0 else "available",
                "sum_norm": float(np.linalg.norm(x + y)),
                "sum_norm_over_norm_sum": None if nx + ny == 0 else float(np.linalg.norm(x + y)) / (nx + ny)}

    def digest(value):
        return hashlib.sha256(serialization.to_bytes(value)).hexdigest()

    def guard(obj, name):
        if not hasattr(obj, name):
            return
        old = getattr(obj, name)
        def forbidden(*args, **kwargs):
            zero["forbidden_api_calls"] += 1
            raise RuntimeError("Forbidden zero-update API: " + name)
        patches.append((obj, name, old))
        setattr(obj, name, forbidden)

    def initialize():
        require(rtol == 2e-5 and atol == 2e-6, "Validation tolerance changed")
        require(protocol["fitting_rows"] == 256 and protocol["gradient_rows"] == 32, "Unexpected bank budget")
        require(protocol["kappa_grid"] == [0, .03, .1, .3, 1, 3, 10] and protocol["positive_scales"] == [.1, 1, 10], "Grid changed")
        require(float(protocol["native_alpha"]) == 2.5 and float(protocol["native_q_epsilon"]) == 1e-7, "Native constants changed")
        manifest_path = Path(checkpoint_manifest).resolve()
        require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol["checkpoint_manifest_sha256"], "Checkpoint identity changed")
        contract = _module("_bca_fitting_scale_contract", ROOT / "contract.py")
        prior = ROOT.parent / "unifloral_frozen_usefulness_v1_seedfix"
        inference = _module("_bca_fitting_scale_inference", prior / "inference.py")
        # The prior protocol supplies ONLY the authenticated checkpoint contract.
        # No evaluator or prior output/label file is imported or read.
        load_protocol = json.loads((prior / "protocol.json").read_text(encoding="utf-8"))
        policy, bank, receipt = inference.load(manifest_path, load_protocol)
        _, components, repeated_receipt = inference.read_contract(manifest_path, load_protocol)
        require(receipt["file_hashes"] == repeated_receipt["file_hashes"], "Checkpoint bytes changed during loading")
        payload = serialization.msgpack_restore(components["parameters"]["bytes"])
        native_loader = _module("native_loader", ROOT.parent / "unifloral_td3bc_validation_v1/native_loader.py")
        native = native_loader.load_native(ROOT.parent / "unifloral_td3bc_validation_v1/upstream",
                                           ROOT.parent / "unifloral_td3bc_validation_v1/provenance/unifloral.json")
        bridge = _module("bca_bridge", ROOT.parent / "unifloral_bca_pipeline_validation_v1/bca_bridge.py")
        bca = bridge.load_pipeline(ROOT.parent / "unifloral_bca_pipeline_validation_v1")
        candidate = _module("_bca_fitting_scale_candidate", ROOT.parent / "unifloral_td3bc_validation_v1/candidate.py")
        mean, std = jnp.asarray(payload["obs_mean"]), jnp.asarray(payload["obs_std"])
        actor = native.DeterministicTanhActor(4, mean, std)
        critic = native.DualQNetwork(mean, std)
        calibrator = bca.Calibrator(mean, std, state_dep=True)
        models = (None, None, calibrator)
        obs = jnp.asarray(bank["observations"][:32], jnp.float32)
        recorded = jnp.asarray(bank["actions"][:32], jnp.float32)
        batch = native.Transition(obs, recorded, jnp.zeros(32, jnp.float32), obs, jnp.zeros(32, jnp.float32))
        args = native.Args(batch_size=32, gamma=float(receipt["manifest"]["gamma"]))
        require(args.td3_alpha == protocol["native_alpha"], "Native actor coefficient differs")
        agent = SimpleNamespace(dual_q=SimpleNamespace(params=policy.critic_params))
        cache.update(policy=policy, bank=bank, receipt=receipt, native=native, bca=bca, candidate=candidate,
                     actor=actor, critic=critic, models=models, obs=obs, recorded=recorded, batch=batch,
                     args=args, agent=agent, params=policy.actor_params, reference=policy.reference, contract=contract)
        zero["frozen_state_sha256_before"] = policy.digest()
        for name in ("fit_scale", "refresh", "freeze_reference", "initialize_pipeline", "initial_reference"):
            guard(bca, name)
        for name in ("create_train_state", "make_train_step", "eval_agent"):
            guard(native, name)
        guard(native_loader, "initialize")
        guard(bridge, "initialize_pipeline")
        from flax.training.train_state import TrainState
        import optax
        guard(TrainState, "apply_gradients")
        guard(optax, "apply_updates")
        guard(optax, "incremental_update")
        cache["guards_installed"] = True
        return {"checkpoint": receipt, "fitting_bank_sha256": components["scale_bank"]["sha256"],
                "fitting_rows": 256, "gradient_rows": 32, "raw_observations_normalized_once": True,
                "actor_batch_placeholder_fields": "reward=0, next_obs=obs, done=0 are unused by actor_loss; no recorded TD labels present",
                "guarded_api_count": len(patches), "no_optimizer_state_created": True}

    def make_bank():
        policy, bank, contract = cache["policy"], cache["bank"], cache["contract"]
        actor_actions = policy.act(bank["observations"])
        generator = np.random.default_rng(protocol["candidate_seed"])
        random_actions = generator.uniform(-1, 1, (256, 6, 4)).astype(np.float32)
        banks = []
        for i in range(256):
            proposed = [actor_actions[i], bank["actions"][i], np.zeros(4, np.float32), *random_actions[i]]
            seen, distinct = set(), []
            for action in proposed:
                action = np.asarray(action, np.float32)
                # Numeric exact equality merges signed zero too.
                key = tuple(float(x) for x in action)
                if key not in seen:
                    seen.add(key)
                    distinct.append(action.copy())
            banks.append(np.stack(distinct))
        counts = [len(row) for row in banks]
        require(sum(counts) <= protocol["max_candidate_rows"] and max(counts) <= 9, "Candidate cap exceeded")
        actions = np.concatenate(banks)
        observations = np.repeat(bank["observations"], counts, axis=0)
        q = policy.q(observations, actions)
        u, usable, valid = policy.width(observations, actions)
        require(bool(usable) and bool(valid), "Actual reference is not usable and valid")
        finite((q, u), "Fitting candidate Q/U")
        qr, ur, dr, ar = [], [], [], []
        cursor = 0
        for index, count in enumerate(counts):
            qr.append(q[cursor:cursor + count, 0].astype(np.float64).tolist())
            ur.append(u[cursor:cursor + count].astype(np.float64).tolist())
            dr.append(np.abs(q[cursor:cursor + count, 0].astype(np.float64) - q[cursor:cursor + count, 1]).tolist())
            ar.append(np.square(banks[index].astype(np.float64) - actor_actions[index]).mean(axis=1).tolist())
            cursor += count
        scaling = contract.candidate_scale(qr, ur, protocol["relative_contrast_floor"])
        native_scaling = native_factor(scaling["factor"])
        stats = {name: contract.contrast_statistics(rows) for name, rows in
                 (("q1", qr), ("actual_u", ur), ("raw_abs_twin_disagreement", dr), ("actor_squared_distance", ar))}
        cache.update(q_rows=qr, u_rows=ur, banks=banks, bank_actions=actions, bank_observations=observations,
                     candidate_counts=counts, proposed_scale=scaling, proposed_native_scale=native_scaling, candidate_stats=stats)
        return {"candidate_seed": protocol["candidate_seed"], "counts": counts, "total_candidate_rows": sum(counts),
                "candidate_sha256": digest({"actions": actions, "observations": observations, "counts": np.asarray(counts)}),
                "statistics": stats, "proposed_detached_factor": scaling, "native_float32_factor": native_scaling,
                "weighting": "Equal states, then equal candidate actions in each state; no labels or selection",
                "candidate_rule": protocol["candidate_rule"]}

    def gradients():
        actor, critic, bca = cache["actor"], cache["critic"], cache["bca"]
        obs, recorded, params, ref = cache["obs"], cache["recorded"], cache["params"], cache["reference"]
        def actions(pp):
            return jax.vmap(lambda row: actor.apply(pp, row).sample(seed=None))(obs)
        def parts(pp):
            aa = actions(pp)
            qq = jax.vmap(lambda oo, ac: critic.apply(cache["policy"].critic_params, oo, ac)[0])(obs, aa)
            uu = bca.live_width(cache["models"], ref, obs, aa)[0]
            bc = jnp.square(aa - recorded).mean(axis=-1).mean()
            return jnp.stack((qq.mean(), uu.mean(), bc))
        values, jacobian = block(jax.jit(lambda pp: (parts(pp), jax.jacrev(parts)(pp)))(params))
        gq, gu, gbc = [jax.tree_util.tree_map(lambda x, i=i: x[i], jacobian) for i in range(3)]
        aa = block(actions(params))
        qq = block(jax.vmap(lambda oo, ac: critic.apply(cache["policy"].critic_params, oo, ac)[0])(obs, aa))
        m = float(np.asarray(jnp.abs(qq).mean()))
        c = float(np.asarray(jax.lax.stop_gradient(cache["args"].td3_alpha / (jnp.abs(qq).mean() + 1e-7))))
        def aq(acts):
            return critic.apply(cache["policy"].critic_params, obs, acts)[..., 0].sum()
        def au(acts):
            return bca.live_width(cache["models"], ref, obs, acts)[0].sum()
        aqg, aug = block(jax.jit(lambda acts: (jax.grad(aq)(acts), jax.grad(au)(acts)))(aa))
        abc = 2 * (aa - recorded) / 4.0
        finite((values, gq, gu, gbc, aqg, aug, abc), "Actual gradients")
        require(norm(gu) > 0 and norm(aug) > 0, "Actual U gradient is zero; live-action check is inconclusive")
        def actual_loss(pp, q_multiplier, unit_multiplier, kappa):
            q_apply = lambda variables, oo, ac: q_multiplier * critic.apply(variables, oo, ac)
            width = lambda reference, oo, ac: bca.live_width(cache["models"], reference, oo, ac)[0]
            functions = cache["candidate"].make_microsteps(cache["args"], actor.apply, q_apply,
                cache["batch"], width_fn=width, kappa_penalty=kappa)
            scaled_ref = ref._replace(residual_scale=ref.residual_scale * unit_multiplier)
            return functions.actor_loss(pp, cache["agent"], cache["batch"], scaled_ref)
        vg0 = jax.jit(jax.value_and_grad(lambda pp: actual_loss(pp, jnp.float32(1), jnp.float32(1), 0.0), has_aux=True))
        vg1 = jax.jit(jax.value_and_grad(lambda pp, qg, ug: actual_loss(pp, qg, ug, 1.0), has_aux=True))
        (loss0, aux0), grad0 = block(vg0(params))
        (loss1, aux1), grad1 = block(vg1(params, jnp.float32(1), jnp.float32(1)))
        require(float(aux0["q_scale_coefficient"]) == float(aux1["q_scale_coefficient"]), "Native c changed between kappa0/1")
        close(aux1["q_scale_coefficient"], c, "Same first32 actor-action coefficient")
        identity = norm_close(combine((1, grad1), (-1, grad0)), combine((c, gu)), "Actual candidate gradient identity")
        native_identity = norm_close(grad0, combine((-c, gq), (1, gbc)), "Actual native actor gradient decomposition")
        value_identity = close(loss1 - loss0, c * values[1], "Actual candidate loss increment")
        cache.update(actions_fn=actions, parts_fn=parts, gq=gq, gu=gu, gbc=gbc, aq=aqg, au=aug, abc=abc,
                     actor_actions=aa, mean_abs_q=m, coefficient=c, part_values=values,
                     grad0=grad0, grad1=grad1, loss0=loss0, loss1=loss1, vg1=vg1, gradients_ready=True)
        return {"mean_abs_q1_same32_actor_actions": m, "q_scale_coefficient": c,
                "q_mean": values[0], "u_mean": values[1], "bc_mean_over_four_then_rows": values[2],
                "loss_kappa0": loss0, "loss_kappa1": loss1,
                "gradient_identity": identity, "native_gradient_decomposition": native_identity,
                "loss_increment": value_identity,
                "parameter_gradient_norms": {"mean_q1": norm(gq), "mean_u": norm(gu), "bc": norm(gbc)},
                "action_gradient_norms": {"q1": describe(np.linalg.norm(np.asarray(aqg), axis=1)),
                                          "u": describe(np.linalg.norm(np.asarray(aug), axis=1)),
                                          "bc_per_row": describe(np.linalg.norm(np.asarray(abc), axis=1))},
                "actual_candidate_calls": "kappa0 and kappa1 actor_loss only; no microstep or optimizer dispatch"}

    def grid():
        c, gq, gu, gbc = cache["coefficient"], cache["gq"], cache["gu"], cache["gbc"]
        qnorm = np.linalg.norm(np.asarray(cache["aq"], np.float64), axis=1)
        unorm = np.linalg.norm(np.asarray(cache["au"], np.float64), axis=1)
        cq = cache["candidate_stats"]["q1"]["centered_rms"]
        cu = cache["candidate_stats"]["actual_u"]["centered_rms"]
        output = []
        for kappa in protocol["kappa_grid"]:
            qterm, penalty = combine((-c, gq)), combine((c * kappa, gu))
            combined = combine((1, qterm), (1, penalty))
            total = combine((1, combined), (1, gbc))
            ratios = [None if q == 0 else kappa * u / q for q, u in zip(qnorm, unorm)]
            contrast_ratio = None if cq == 0 else kappa * cu / cq
            if contrast_ratio is not None and not np.isfinite(contrast_ratio):
                contrast_ratio = None
            output.append({"kappa_penalty": kappa, "parameter_q_vs_penalty": geometry(qterm, penalty),
                "candidate_bank_centered_penalty_to_q_ratio": contrast_ratio,
                "candidate_bank_ratio_status": "unavailable_zero_or_nonfinite_denominator_ratio" if contrast_ratio is None else "available",
                "candidate_bank_ratio_definition": "kappa*C_U/C_Q on the full fitting candidate bank; separate from derivative ratios",
                "parameter_q_plus_penalty_vs_bc": geometry(combined, gbc),
                "parameter_total_gradient_norm": norm(total),
                "action_penalty_to_q_norm_ratios": ratios,
                "action_ratio_summary_defined": describe([x for x in ratios if x is not None]) if any(x is not None for x in ratios) else None,
                "action_unavailable_zero_q_gradient_rows": sum(x is None for x in ratios),
                "action_q_vs_penalty_cosines": [geometry(-np.asarray(q), kappa * np.asarray(u))["cosine"] for q, u in zip(cache["aq"], cache["au"])]})
        return {"grid": output, "calculation": "Fixed-state algebraic gradients from actual verified kappa0/1 identity; not seven training runs",
                "sign_convention": "Minimized loss has Q term -c gradQ, penalty +c*kappa gradU, plus gradBC"}

    def reference_routes():
        ref, obs, aa = cache["reference"], cache["obs"], cache["actor_actions"]
        differentiable = (ref.cal_params, ref.residual_scale, ref.threshold, ref.lambda_hat, ref.lambda_hpd, ref.n_eff)
        def width_from_fields(fields):
            rr = ref._replace(cal_params=fields[0], residual_scale=fields[1], threshold=fields[2],
                              lambda_hat=fields[3], lambda_hpd=fields[4], n_eff=fields[5])
            return cache["bca"].live_width(cache["models"], rr, obs, aa)[0].mean()
        result = block(jax.jit(jax.grad(width_from_fields))(differentiable))
        require(norm(result) == 0.0, "Frozen-reference gradient is nonzero")
        return {"all_reference_gradient_norm": norm(result), "action_gradient_norm": norm(cache["au"]),
                "frozen_fields": ["cal_params", "residual_scale", "threshold", "lambda_hat", "lambda_hpd", "n_eff"],
                "ready_bool": "Held fixed; nondifferentiable", "live_action_nonzero": True}

    def ranking(base_rows, other_rows):
        ambiguous, changed, near_top = [], [], []
        for index, (base, other) in enumerate(zip(base_rows, other_rows)):
            b, x = np.asarray(base, np.float64), np.asarray(other, np.float64)
            selected, alternate = int(np.argmax(b)), int(np.argmax(x))
            propagated = 2 * (atol + rtol * float(np.max(np.abs(b))))
            ordered = np.sort(b)
            if len(ordered) > 1 and float(ordered[-1] - ordered[-2]) <= propagated:
                near_top.append(index)
            if selected != alternate:
                changed.append(index)
                gap = float(b[selected] - b[alternate])
                require(gap <= propagated, "Ranking changed outside propagated numerical tolerance")
                ambiguous.append({"row": index, "base_gap": gap, "propagated_tolerance": propagated})
        return {"changed_indices": changed, "numerically_ambiguous_count": len(near_top),
                "near_top_rows": near_top, "changed_ambiguous_count": len(ambiguous),
                "changed_ambiguous_details": ambiguous, "outside_tolerance_changes": 0,
                "rule": "Exact argmax; changed indices must lie within twice the fixed value-comparison tolerance"}

    def positive_units():
        bca, ref, contract = cache["bca"], cache["reference"], cache["contract"]
        obs, acts = jnp.asarray(cache["bank_observations"]), jnp.asarray(cache["bank_actions"])
        qbase = cache["policy"].q(obs, acts)[:, 0]
        ubase = cache["policy"].width(obs, acts)[0]
        counts = cache["candidate_counts"]
        split = lambda values: [np.asarray(x, np.float64).tolist() for x in np.split(np.asarray(values), np.cumsum(counts)[:-1])]
        def coepsilon_loss(pp, scale):
            parts = cache["parts_fn"](pp)
            aa = cache["actions_fn"](pp)
            qq = jax.vmap(lambda oo, ac: cache["critic"].apply(cache["policy"].critic_params, oo, ac)[0])(cache["obs"], aa) * scale
            rr = ref._replace(residual_scale=ref.residual_scale * scale)
            uu = bca.live_width(cache["models"], rr, cache["obs"], aa)[0]
            cc = jax.lax.stop_gradient(protocol["native_alpha"] / (jnp.abs(qq).mean() + scale * protocol["native_q_epsilon"]))
            return -cc * qq.mean() + cc * uu.mean() + parts[2]
        coepsilon_vg = jax.jit(jax.value_and_grad(coepsilon_loss))
        scaled_action_width_grad = jax.jit(jax.grad(lambda live_actions, scale: bca.live_width(
            cache["models"], ref._replace(residual_scale=ref.residual_scale * scale),
            cache["obs"], live_actions)[0].sum()))
        rows = []
        for scale in protocol["positive_scales"]:
            s = jnp.float32(scale)
            rr = ref._replace(residual_scale=ref.residual_scale * s)
            ug, usable, valid = block(bca.live_width(cache["models"], rr, obs, acts))
            require(bool(usable) and bool(valid), "Rescaled actual reference is invalid")
            width_equality = close(ug, s * ubase, "Coherent actual residual-unit width scale")
            actual_action_gradient = block(scaled_action_width_grad(cache["actor_actions"], s))
            width_gradient_equality = norm_close(actual_action_gradient, s * cache["au"],
                                                "Actual rescaled U action gradient")
            qg = np.asarray(s * qbase)
            proposed = contract.candidate_scale(split(qg), split(ug), protocol["relative_contrast_floor"])
            require(proposed["status"] == cache["proposed_scale"]["status"], "Scale availability changed under coherent units")
            require(native_factor(proposed["factor"])["status"] == cache["proposed_native_scale"]["status"],
                    "Native factor representability changed under coherent units")
            if proposed["status"] == "available":
                close(proposed["factor"], cache["proposed_scale"]["factor"], "Centered Q/U factor unit invariance")
            (loss, aux), grad = block(cache["vg1"](cache["params"], s, s))
            response = contract.native_epsilon_response(cache["mean_abs_q"], scale, protocol["native_alpha"], protocol["native_q_epsilon"])
            beta = response["value_gradient_multiplier"]
            prediction = combine((-cache["coefficient"] * beta, cache["gq"]),
                                 (cache["coefficient"] * beta, cache["gu"]), (1, cache["gbc"]))
            gradient_check = norm_close(grad, prediction, "Native fixed-epsilon predicted gradient")
            co_value, co_grad = block(coepsilon_vg(cache["params"], s))
            co_check = norm_close(co_grad, cache["grad1"], "Counterfactual co-scaled-epsilon gradient invariance")
            close(co_value, cache["loss1"], "Counterfactual co-scaled-epsilon loss invariance")
            ranks = []
            for k in protocol["kappa_grid"]:
                original = np.asarray(qbase - np.float32(k) * ubase, np.float32)
                scaled_scores = np.asarray(qg - np.float32(k) * np.asarray(ug), np.float32)
                recovered = np.asarray(scaled_scores / np.float32(scale), np.float32)
                close(recovered, original, "Positive-unit score equality after division")
                ranks.append({"kappa_penalty": k, **ranking(split(original), split(scaled_scores))})
            rows.append({"positive_scale": scale, "actual_frozen_residual_scale": rr.residual_scale,
                         "width_check": width_equality, "width_action_gradient_check": width_gradient_equality,
                         "candidate_factor": proposed["factor"],
                         "native_epsilon_response": response, "actual_q_scale_coefficient": aux["q_scale_coefficient"],
                         "native_gradient_check": gradient_check, "co_scaled_epsilon_diagnostic": co_check,
                         "actual_loss": loss, "rankings": ranks})
        return {"cases": rows, "bc_gradient_unchanged": True,
                "scope": "Frozen forward/derivative unit algebra only; initial residual unit1, EMA std floor1e-6, Adam epsilon and fitting dynamics are unchanged"}

    def synthetic_td():
        # The export has no target networks, rewards or next observations.
        # Current frozen weights are declared target stand-ins in this algebra fixture.
        ref, bca = cache["reference"], cache["bca"]
        obs = cache["obs"]
        batch = cache["native"].Transition(obs, cache["recorded"], jnp.linspace(-1., 1., 32, dtype=jnp.float32),
                                         jnp.roll(obs, 1, axis=0), (jnp.arange(32) % 2).astype(jnp.float32))
        view = SimpleNamespace(actor_target=SimpleNamespace(params=cache["params"]),
                               critic_target=SimpleNamespace(params=cache["policy"].critic_params))
        args = SimpleNamespace(discount=.98, policy_noise=cache["args"].policy_noise, noise_clip=cache["args"].noise_clip)
        key = jax.random.fold_in(jax.random.PRNGKey(protocol["candidate_seed"]), 0x5444)
        def target(scale):
            actor_view = SimpleNamespace(apply=lambda pp, oo: cache["actor"].apply(pp, oo).sample(seed=None))
            critic_view = SimpleNamespace(apply=lambda pp, oo, ac: scale * cache["critic"].apply(pp, oo, ac))
            return bca.native_target(args, (actor_view, critic_view, None), view,
                                     batch._replace(reward=scale * batch.reward), key)
        target_jit = jax.jit(target)
        original = block(target_jit(jnp.float32(1)))
        current_min = cache["critic"].apply(cache["policy"].critic_params, obs, batch.action).min(axis=-1)
        eta = cache["models"][2].apply(ref.cal_params, obs, batch.action)
        original_score = jnp.abs(original - current_min) / bca.positive_scale(eta, ref.residual_scale)
        cases = []
        for scale in protocol["positive_scales"]:
            s = jnp.float32(scale)
            y = block(target_jit(s))
            cases.append({"positive_scale": scale,
                "target_scaling": close(y, s * original, "Synthetic reward/TD target units"),
                "unit_normalized_residual": close(jnp.abs(y - s * current_min) / bca.positive_scale(eta, s * ref.residual_scale),
                                                   original_score, "Synthetic normalized residual units"),
                "terminal_reward_only": close(np.asarray(y)[1::2], np.asarray(s * batch.reward)[1::2], "Synthetic terminal mask")})
        return {"cases": cases, "fixture": "Synthetic rewards [-1,1], alternating terminal masks, rolled fitting observations; current networks stand in for targets",
                "not_measured_training_td_invariance": True, "native_target_definition_used_unchanged": True,
                "noise_key": "fold_in(PRNGKey(candidate_seed),0x5444), shared across unit cases"}

    def constants_and_centering():
        bca, ref, obs = cache["bca"], cache["reference"], cache["obs"]
        actor, critic, candidate = cache["actor"], cache["critic"], cache["candidate"]
        constant_calibrator = SimpleNamespace(apply=lambda pp, oo, ac: jnp.ones(ac.shape[:-1], ac.dtype))
        constant_models = (None, None, constant_calibrator)
        cases = []
        for label, rr in (("constant_positive_width", ref),
                          ("zero_width", ref._replace(threshold=jnp.float32(0), lambda_hat=jnp.float32(0), lambda_hpd=jnp.float32(0)))):
            widths, usable, valid = block(bca.live_width(constant_models, rr, obs, cache["actor_actions"]))
            require(bool(usable) and bool(valid), "Constant/zero fixture does not satisfy actual width validation")
            function = candidate.make_microsteps(cache["args"], actor.apply, critic.apply, cache["batch"],
                width_fn=lambda rr, oo, ac: bca.live_width(constant_models, rr, oo, ac)[0], kappa_penalty=1.0)
            (_, _), grad = block(jax.jit(jax.value_and_grad(
                lambda pp: function.actor_loss(pp, cache["agent"], cache["batch"], rr), has_aux=True))(cache["params"]))
            cases.append({"fixture": label, "actual_width": widths[0],
                          "gradient": norm_close(grad, cache["grad0"], "Constant/zero U leaves actor gradient unchanged")})
        means = np.asarray([np.mean(row) for row in cache["u_rows"]], np.float32)
        row_constants = jnp.asarray(means[:32])
        centered_fn = candidate.make_microsteps(cache["args"], actor.apply, critic.apply, cache["batch"],
            width_fn=lambda rr, oo, ac: bca.live_width(cache["models"], rr, oo, ac)[0] - jax.lax.stop_gradient(row_constants),
            kappa_penalty=1.0)
        (_, _), centered_grad = block(jax.jit(jax.value_and_grad(
            lambda pp: centered_fn.actor_loss(pp, cache["agent"], cache["batch"], ref), has_aux=True))(cache["params"]))
        gradient_check = norm_close(centered_grad, cache["grad1"], "Detached action-independent subtraction gradient")
        ranks = []
        for k in protocol["kappa_grid"]:
            left, centered = [], []
            for q, u, constant in zip(cache["q_rows"], cache["u_rows"], means):
                q, u = np.asarray(q, np.float32), np.asarray(u, np.float32)
                a = q - np.float32(k) * u
                b = q - np.float32(k) * (u - constant)
                close(b - np.float32(k) * constant, a, "State-constant subtraction after removing offset")
                left.append(a)
                centered.append(b)
            ranks.append({"kappa_penalty": k, **ranking(left, centered)})
        return {"fixtures": cases, "actual_U_centering_gradient_check": gradient_check, "rankings": ranks,
                "scope": "Centering is a score/derivative diagnostic, not a replacement nonnegative uncertainty width"}

    def edge_cases():
        contract = cache["contract"]
        base = [[-1., 1.], [-2., 2.]]
        constant = [[1., 1.], [2., 2.]]
        zeros = [[0., 0.], [0., 0.]]
        tiny = [[1. - 1e-8, 1. + 1e-8], [2. - 1e-8, 2. + 1e-8]]
        unavailable = []
        for label, q, u in (("zero_u", base, zeros), ("constant_u", base, constant),
                            ("zero_q_contrasts", zeros, base), ("constant_q", constant, base),
                            ("near_constant_u", base, tiny)):
            result = contract.candidate_scale(q, u, protocol["relative_contrast_floor"])
            require(result["status"] == "unavailable" and result["factor"] is None, "Invalid contrast received fallback factor")
            unavailable.append({"fixture": label, **result})
        for bad in (float("nan"), float("inf")):
            try:
                contract.candidate_scale([[bad, 1.]], [[1., 2.]], protocol["relative_contrast_floor"])
            except ValueError:
                pass
            else:
                raise AssertionError("Nonfinite scales were accepted")
        representations = []
        for label, q, u in (("overflow_factor", [[-1., 1.]], [[-1e-100, 1e-100]]),
                            ("underflow_factor", [[-1e-100, 1e-100]], [[-1., 1.]])):
            factor = contract.candidate_scale(q, u, protocol["relative_contrast_floor"])
            require(factor["status"] == "available", "Synthetic host ratio should be finite positive")
            native_status = native_factor(factor["factor"])
            require(native_status["status"] == "unavailable" and native_status["factor_native"] is None,
                    "Unrepresentable factor received a native fallback")
            representations.append({"fixture": label, "host_factor": factor["factor"], **native_status})
        # Exact action-space zero-Q fixture with nonzero gradient: q(a)=mean(a-a0).
        anchor = cache["actor_actions"]
        def edge_q(actions, offset):
            return jnp.mean(actions - anchor, axis=-1) + offset
        rows = []
        for offset in (0., protocol["near_zero_q_multiple"] * protocol["native_q_epsilon"]):
            qq = edge_q(anchor, jnp.float32(offset))
            qg = jax.grad(lambda ac: edge_q(ac, jnp.float32(offset)).sum())(anchor)
            m = float(jnp.abs(qq).mean())
            cc = float(jax.lax.stop_gradient(protocol["native_alpha"] / (jnp.abs(qq).mean() + protocol["native_q_epsilon"])))
            finite((qq, qg, cc), "Zero/near-zero Q edge")
            require(norm(qg) > 0, "Zero-Q edge gradient is vacuous")
            responses = [contract.native_epsilon_response(m, scale, protocol["native_alpha"], protocol["native_q_epsilon"])
                         for scale in protocol["positive_scales"]]
            rows.append({"q_offset": offset, "mean_abs_q": m, "native_c": cc,
                         "action_q_gradient_norm": norm(qg), "native_weighted_action_q_gradient_norm": norm(cc * qg),
                         "unit_responses": responses,
                         "fixture": "Synthetic q(a)=mean(a-a0)+offset at exact a0; nonzero derivative at zero Q"})
        # Actual legacy live_width fallback is exposed, while the proposed gate rejects it.
        invalid = []
        ref = cache["reference"]
        for label, rr in (("unready_finite", ref._replace(ready=jnp.asarray(False))),
                          ("ready_infinite", ref._replace(threshold=jnp.float32(jnp.inf), lambda_hpd=jnp.float32(jnp.inf)))):
            width, usable, valid = block(cache["bca"].live_width(cache["models"], rr, cache["obs"], anchor))
            require(not bool(usable), "Invalid fixture unexpectedly usable")
            strict_accept = bool(rr.ready) and bool(usable) and bool(valid) and np.isfinite(float(rr.threshold))
            require(not strict_accept, "Proposed strict diagnostic accepted invalid calibration")
            invalid.append({"fixture": label, "legacy_width_max": float(np.max(np.asarray(width))),
                            "legacy_valid": bool(valid), "legacy_usable": bool(usable),
                            "diagnostic_status": "invalid_calibration_not_zero_penalty_success"})
        return {"unavailable_scales": unavailable, "nonfinite_inputs_rejected": True,
                "native_float32_factor_rejections": representations,
                "zero_nearzero_q": rows, "invalid_reference_policy": invalid,
                "production_behavior_changed": False}

    def detached_statistics():
        proposed = cache["proposed_scale"]
        if proposed["status"] != "available" or cache["proposed_native_scale"]["status"] != "available":
            return {"diagnostic_status": "unavailable", "reason": proposed["reason"] or cache["proposed_native_scale"]["reason"],
                    "factor": None, "actual_factor_gradient_checks_performed": False}
        shape = (256, max(cache["candidate_counts"]))
        qm, um, mask = (np.zeros(shape, np.float32) for _ in range(3))
        for i, (q, u) in enumerate(zip(cache["q_rows"], cache["u_rows"])):
            qm[i, :len(q)], um[i, :len(u)], mask[i, :len(q)] = q, u, 1
        qm, um, mask = map(jnp.asarray, (qm, um, mask))
        def contrast(matrix):
            count = mask.sum(axis=1)
            mean = (matrix * mask).sum(axis=1) / count
            variance = (jnp.square(matrix - mean[:, None]) * mask).sum(axis=1) / count
            return jnp.sqrt(variance.mean())
        def penalty(qstats, ustats, live_actions):
            factor = jax.lax.stop_gradient(contrast(qstats) / contrast(ustats))
            return factor * cache["bca"].live_width(cache["models"], cache["reference"], cache["obs"], live_actions)[0].mean()
        stats_q_grad, stats_u_grad, live_grad = block(jax.jit(jax.grad(penalty, argnums=(0, 1, 2)))(qm, um, cache["actor_actions"]))
        require(norm(stats_q_grad) == 0 and norm(stats_u_grad) == 0, "Detached fitting statistics received gradients")
        factor = float(contrast(qm) / contrast(um))
        require(native_factor(factor)["status"] == "available", "Computed JAX factor is not finite positive float32")
        close(factor, proposed["factor"], "JAX masked/equal-state contrast factor matches contract")
        routed = norm_close(live_grad, cache["au"] * factor / 32., "Detached factor preserves live action derivative")
        parameter_penalty = combine((cache["coefficient"] * factor, cache["gu"]))
        return {"diagnostic_status": "available", "factor": factor,
                "q_statistics_gradient_norm": norm(stats_q_grad), "u_statistics_gradient_norm": norm(stats_u_grad),
                "live_action_gradient_norm": norm(live_grad), "live_action_identity": routed,
                "hypothetical_kappa1_parameter_q_vs_penalty": geometry(combine((-cache["coefficient"], cache["gq"])), parameter_penalty),
                "factor_role": "Detached fitting-bank candidate only; neither selected nor installed in production"}

    def differentiating_scale_counterexample():
        def example(x, detached):
            q = jnp.asarray([0., 2.], jnp.float32)
            u = jnp.stack((x, 2 * x))
            cq, cu = jnp.std(q), jnp.std(u)
            factor = cq / cu
            if detached:
                factor = jax.lax.stop_gradient(factor)
            return factor * u.mean()
        x = jnp.float32(1)
        frozen = jax.grad(lambda a: example(a, True))(x)
        through = jax.grad(lambda a: example(a, False))(x)
        close(frozen, 3., "Detached toy gradient")
        close(through, 0., "Through-scale toy cancellation")
        return {"synthetic_example": "Q=[0,2], U(a)=[a,2a], a=1, C_Q/C_U=2/a",
                "same_penalty_value": float(example(x, True)), "detached_scale_gradient": frozen,
                "gradient_through_scale": through,
                "interpretation": "Differentiating a normalizer changes the intended penalty gradient; this is a mathematical counterexample, not measured actor behavior"}

    try:
        record("authenticated_fitting_checkpoint_and_zero_update_guards", initialize)
        record("fixed_fitting_candidate_bank_and_contrast_scales", make_bank, ("guards_installed",))
        record("actual_candidate_loss_and_gradient_identity", gradients, ("proposed_scale",))
        record("fixed_kappa_grid_gradient_geometry", grid, ("gradients_ready",))
        record("frozen_reference_and_live_action_gradient_routes", reference_routes, ("gradients_ready",))
        record("coherent_positive_reward_units_and_native_epsilon", positive_units, ("gradients_ready",))
        record("explicit_synthetic_reward_and_td_unit_algebra", synthetic_td, ("gradients_ready",))
        record("constant_zero_width_and_state_constant_subtraction", constants_and_centering, ("gradients_ready",))
        record("finite_edge_cases_and_fail_closed_diagnostic_policy", edge_cases, ("gradients_ready",))
        record("detached_fitting_statistics_live_action_route", detached_statistics, ("gradients_ready",))
        record("counterexample_to_differentiating_through_scale", differentiating_scale_counterexample)
    finally:
        if "policy" in cache:
            try:
                cache["policy"].assert_unchanged()
                zero["frozen_state_sha256_after"] = cache["policy"].digest()
                zero["frozen_state_unchanged"] = zero["frozen_state_sha256_before"] == zero["frozen_state_sha256_after"]
                zero["verified"] = bool(cache.get("guards_installed")) and zero["frozen_state_unchanged"] and zero["forbidden_api_calls"] == 0
            except Exception as exc:
                checks.append({"name": "frozen_state_integrity", "status": "fail", "error": str(exc)})
        emit({"event": "zero_work_verification", "zero_work": zero})
        for obj, name, old in reversed(patches):
            setattr(obj, name, old)
        stream.close()
    passed = bool(checks) and all(item["status"] == "pass" for item in checks) and zero["verified"]
    return {"schema": "unifloral.fitting_scale_gradient_checks.v1", "status": "pass" if passed else "fail",
            "passed": passed, "checks": checks, "zero_work": zero,
            "phase": protocol["phase"], "elapsed_seconds": time.monotonic() - started,
            "tolerances": {"rtol": rtol, "atol": atol, "gradient_identity": "L2 error <= atol + rtol*max(left norm,right norm)"},
            "coefficient_selected": False, "production_normalization_changed": False,
            "limitations": protocol["limitations"]}
