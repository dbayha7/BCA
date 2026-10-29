"""Fresh central-quartile matched-training adapter; scientific imports occur only inside functions.

Pinned native/uncertainty kernels are reused verbatim. Only initialization seeds,
data provisioning, schedule and actor penalty binding are adapted here.
"""
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
import math

from common import HERE, PAYLOAD, module, require, digest

COUNT_KEYS = ("critic", "actor", "actor_target", "critic_target", "outer_steps",
              "auxiliary", "aux_fit_attempts", "refresh_attempts", "refresh_accepted")


def config(value):
    return asdict(value) if is_dataclass(value) else dict(vars(value))


def host(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): host(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [host(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    array = np.asarray(value)
    if array.ndim:
        return host(array.tolist())
    item = array.item()
    if isinstance(item, float) and not math.isfinite(item):
        return "nan" if math.isnan(item) else ("positive_infinity" if item > 0 else "negative_infinity")
    return item


def block(jax, value):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


def load_native():
    loader = module("_central_training_native_loader", PAYLOAD / "unifloral_td3bc_validation_v1/native_loader.py")
    return loader.load_native(PAYLOAD / "unifloral_td3bc_validation_v1/upstream",
                              PAYLOAD / "unifloral_td3bc_validation_v1/provenance/unifloral.json")


def runtime_receipt(native):
    import importlib.metadata
    import sys
    jax = native.jax
    versions = {name: importlib.metadata.version(name) for name in (
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "flax",
        "optax", "distrax", "chex", "tensorflow-probability", "scipy")}
    expected = dict(zip(versions, ("0.6.2", "0.6.2", "0.6.2", "0.6.2", "2.5.4", "0.10.7",
        "0.2.8", "0.1.5", "0.1.90", "0.25.0", "1.18.1")))
    require(versions == expected, "Frozen runtime versions differ")
    require(jax.default_backend() == "gpu" and not jax.config.jax_disable_jit and not jax.config.jax_enable_x64,
            "Expected GPU/JIT/float32 JAX runtime")
    require(str(jax.config.jax_default_matmul_precision) == "highest", "Matmul precision differs")
    return {"backend": jax.default_backend(), "executable": sys.executable, "python": sys.version,
            "devices": [str(d) for d in jax.devices()], "device_kinds": [d.device_kind for d in jax.devices()],
            "packages": versions, "native_source": native.__file__, "source_receipt": native._source_receipt,
            "upstream_runtime_constraints_exception": "Same disclosed JAX/NumPy exception as Titan and prior Zeus pilot"}


def parameter_leaves(tree):
    import numpy as np
    from flax import serialization
    rows = []
    def visit(node, path):
        if isinstance(node, dict):
            for key in sorted(node):
                visit(node[key], path + [str(key)])
        elif isinstance(node, (tuple, list)):
            for index, value in enumerate(node):
                visit(value, path + [str(index)])
        else:
            array = np.asarray(node)
            rows.append({"path": "/".join(path), "shape": list(array.shape), "dtype": array.dtype.str,
                         "sha256": digest(array.tobytes(order="C"))})
    visit(serialization.to_state_dict(tree), [])
    return rows


def tree_digest(tree):
    import json
    return digest(json.dumps(parameter_leaves(tree), sort_keys=True, separators=(",", ":")).encode())


def restore_exact(adapter, template, raw):
    """Restore complete state, including phase-held batch, optimizer and RNG.

    The caller must separately authenticate the checkpoint/normalization/data
    manifest before supplying bytes. Static apply functions remain those of the
    freshly authenticated template; serialized parameters alone are insufficient.
    """
    from flax import serialization
    restored = serialization.from_bytes(template, raw)
    require(serialization.to_bytes(restored) == raw, "Full-state restore not byte exact")
    validate(adapter, restored)
    return restored


def create(protocol, seed_index, condition):
    """Production initialization: authenticated runtime, fresh state, retained replay."""
    require(condition in ("full", "hole", "random"), "Unknown conditional data arm")
    require(type(seed_index) is int and seed_index in (0, 1, 2), "Three declared seed indexes required")
    native = load_native()
    receipt = runtime_receipt(native)
    data = module("_central_training_condition_data", HERE / "data.py").load_condition(native, condition)
    return _initialize_data(protocol, seed_index, condition, native, data, receipt)


def _initialize_data(protocol, seed_index, condition, native, data, receipt):
    """Shared construction for production and separately bounded synthetic checks.

    Native/auxiliary initialization reads one common retained-intersection
    normalization fixture. Only condition-retained replay reaches update kernels.
    This does not alter the pinned model definitions or normalization operation.
    """
    import numpy as np
    jax, jnp = native.jax, native.jnp
    training = module("_central_training_training_types", PAYLOAD / "unifloral_checkpoint_v1/training.py")
    candidate = module("_central_training_candidate", PAYLOAD / "unifloral_td3bc_validation_v1/candidate.py")
    bridge = module("_central_training_bca_bridge", PAYLOAD / "unifloral_bca_pipeline_validation_v1/bca_bridge.py")
    loader = module("_central_training_native_loader", PAYLOAD / "unifloral_td3bc_validation_v1/native_loader.py")
    train = training._transitions(native, data.train, "eligible training", len(data.train.obs))
    heldout = training._transitions(native, data.heldout, "eligible calibration", 128)
    probe = training._transitions(native, data.probe, "eligible probes", len(data.probe.obs))
    normalization = training._transitions(native, data.normalization_fixture, "common retained-intersection normalization", len(data.normalization_fixture.obs))
    ids = {k: np.asarray(v, np.int64) for k, v in data.heldout_ids.items()}
    require(set(ids) == {"row_ids", "episode_ids"} and all(v.shape == (128,) and len(np.unique(v)) == 128 for v in ids.values()), "Calibration identities invalid")
    seed = protocol["seeds"][seed_index]
    args = native.Args(seed=seed["native_seed"], dataset="thrusterchain-finite-horizon-v1",
                       num_updates=10000, gamma=.98, batch_size=256)
    rng, agent = loader.initialize(native, args, normalization, seed["native_seed"])
    pipeline = bridge.load_pipeline(PAYLOAD / "unifloral_bca_pipeline_validation_v1")
    models, aux, aux_config, aux_args = pipeline.initialize_pipeline(agent, normalization, args, seed["aux_seed"])
    require(float(np.asarray(aux.residual_scale)) == 1.0, "Original residual initialization changed")
    require((aux_config.cal_lr, aux_config.cal_beta, aux_config.width_penalty, aux_config.scale_ema,
             aux_config.posterior.alpha, aux_config.posterior.credibility, aux_config.posterior.draws) ==
             (.001, 20., .005, .99, .1, .95, 1000), "Original uncertainty settings differ")
    functions = candidate.make_microsteps(args, agent.actor.apply_fn, agent.dual_q.apply_fn, train,
                                         width_fn=None, kappa_penalty=0.)
    # Keep reference pytree shape fixed through warmup/restore; static kappa0 never reads it.
    micro = functions.initial_state(rng, agent, reference=aux.posterior)
    zero = jnp.int32(0)
    state = training.TrainingState(micro, aux,
        jax.random.fold_in(jax.random.PRNGKey(seed["aux_seed"]), training.AUX_STREAM_DOMAIN),
        zero, zero, zero, zero, zero, zero, jnp.array([10000], jnp.int32), jnp.int32(10000))
    model = agent.actor.apply_fn.__self__
    initial_hashes = {"actor": tree_digest(agent.actor.params), "critic": tree_digest(agent.dual_q.params),
                      "auxiliary_scale": tree_digest(aux.calibrator.params)}
    optimizer_steps = {"actor": int(agent.actor.step), "critic": int(agent.dual_q.step),
                      "actor_target": int(agent.actor_target.step), "critic_target": int(agent.dual_q_target.step),
                      "auxiliary": int(aux.calibrator.step)}
    require(all(x == 0 for x in optimizer_steps.values()), "Fresh optimizer counters not zero")
    context = SimpleNamespace(native=native, args=args, training=training, candidate=candidate,
        pipeline=pipeline, models=models, aux_config=aux_config, aux_args=aux_args,
        train_transition=train, heldout_transition=heldout, heldout_ids=ids, probe=probe, data=data,
        obs_mean=model.obs_mean, obs_std=model.obs_std, runtime=receipt,
        initial_parameter_hashes=initial_hashes, initial_optimizer_steps=optimizer_steps,
        initial_optimizer_hashes={"actor": tree_digest(agent.actor.opt_state), "critic": tree_digest(agent.dual_q.opt_state),
                                  "auxiliary_scale": tree_digest(aux.calibrator.opt_state)},
        normalization={"mean": np.asarray(model.obs_mean).tolist(), "std": np.asarray(model.obs_std).tolist(),
                       "epsilon": .001, "source": "Common hole/random retained-intersection training observations; identical all conditions",
                       "rows": len(normalization.obs),
                       "fixture_digest": tree_digest(normalization.obs)},
        fit_step=jax.jit(lambda a,b,k: pipeline.fit_scale(aux_args, aux_config, models, a,b,k)))
    adapter = make_adapter(context, state, "native", 0., agent.actor.params, [10000], 10000)
    validate(adapter, state)
    return adapter, state


def width_function(context, arm, anchor_params):
    jax, jnp = context.native.jax, context.native.jnp
    if arm == "native":
        return None
    if arm == "bca":
        def width(reference, obs, actions):
            value, usable, valid = context.pipeline.live_width(context.models, reference, obs, actions)
            return jnp.where(usable & valid, value, jnp.full_like(value, jnp.nan))
        return width
    require(arm == "distance", "Unknown actor penalty")
    anchor = jax.tree_util.tree_map(jax.lax.stop_gradient, anchor_params)
    actor_apply = context.models[0].apply
    def width(reference, obs, actions):
        del reference
        anchor_actions = jax.lax.stop_gradient(actor_apply(anchor, obs))
        return jnp.asarray(19.29619572569095, actions.dtype) * jnp.mean((actions-anchor_actions)**2, axis=-1)
    return width


def make_adapter(context, state, arm, kappa, anchor_params, schedule, budget):
    jax, jnp = context.native.jax, context.native.jnp
    functions = context.candidate.make_microsteps(context.args,
        state.microstep.agent_state.actor.apply_fn, state.microstep.agent_state.dual_q.apply_fn,
        context.train_transition, width_fn=width_function(context, arm, anchor_params), kappa_penalty=kappa)
    def equal(a,b):
        aa, sa = jax.tree_util.tree_flatten(a); bb, sb = jax.tree_util.tree_flatten(b)
        if sa != sb:
            raise ValueError("Reference structure differs")
        return jnp.all(jnp.stack([jnp.array_equal(x,y) for x,y in zip(aa,bb)]))
    def sampler(key):
        _, subkey = jax.random.split(key)
        return jax.random.randint(subkey, (context.args.batch_size,), 0, len(context.train_transition.obs))
    state_checks, finite = context.training._validation_kernels()
    return SimpleNamespace(context=context, arm=arm, kappa=float(kappa), schedule=list(schedule), budget=budget,
        functions=functions, microstep_step=jax.jit(functions.step), finite=finite, state_checks=state_checks,
        reference_equal=jax.jit(equal), sample_indices=jax.jit(sampler), args_snapshot=config(context.args),
        aux_snapshot=config(context.aux_config))


def branch(adapter, state, protocol, seed_index, arm, coefficient):
    import numpy as np
    context = adapter.context
    jax, jnp = context.native.jax, context.native.jnp
    require(arm in ("native", "bca", "distance"), "Unknown branch arm")
    require(type(seed_index) is int and seed_index in (0,1,2), "Invalid seed index")
    c = validate(adapter, state)["counts"]
    require(c["critic"] == 10000 and c["phase"] == 0 and c["refresh_accepted"] == 1 and c["reference_ready"], "Incomplete warmup cannot fork")
    require(np.isfinite(coefficient) and coefficient > 0, "Invalid fixed coefficient")
    seed = protocol["seeds"][seed_index]
    native_key = jax.random.fold_in(state.microstep.rng, seed["branch_native_seed"])
    aux_key = jax.random.fold_in(state.aux_rng, seed["branch_aux_seed"])
    anchor = state.microstep.agent_state.actor.params
    state = state._replace(microstep=state.microstep._replace(rng=native_key, reference=state.aux.posterior),
        aux_rng=aux_key, refresh_at=jnp.array([10000,15000], jnp.int32), critic_budget=jnp.int32(20000))
    kappa = {"native":0., "bca":3., "distance":float(coefficient)}[arm]
    result = make_adapter(context, state, arm, kappa, anchor, [10000,15000], 20000)
    validate(result, state)
    return result, state


def validate(adapter, state):
    import numpy as np
    context = adapter.context
    require(config(context.args) == adapter.args_snapshot and config(context.aux_config) == adapter.aux_snapshot, "Scientific settings changed")
    c = context.training.counts(state)
    require(0 <= c["critic"] <= adapter.budget == c["critic_budget"] and np.asarray(state.refresh_at).tolist() == adapter.schedule, "Budget/schedule mismatch")
    require(c["phase"] == c["critic"] % 2 and all(c[k] == c["critic"] // 2 for k in ("actor","actor_target","critic_target","outer_steps")), "Native phase/counters disagree")
    require(c["auxiliary"] == c["aux_fit_attempts"] == c["critic"], "Auxiliary work mismatch")
    due = sum(n <= c["critic"] for n in adapter.schedule)
    require(all(c[k] == due for k in ("refresh_attempts","refresh_accepted","next_refresh_index")), "Calibration counters mismatch")
    require(c["last_refresh_critic"] == (adapter.schedule[due-1] if due else 0), "Calibration boundary mismatch")
    a = state.microstep.agent_state
    for ts, expected in ((a.actor,c["actor"]),(a.dual_q,c["critic"]),(a.actor_target,c["actor_target"]),
                         (a.dual_q_target,c["critic_target"]),(state.aux.calibrator,c["auxiliary"])):
        require(int(np.asarray(ts.step)) == expected, "Optimizer step mismatch")
    flags = host(adapter.state_checks(state))
    require(all(flags[k] for k in ("native_and_aux_finite","duplicate_native_equal","reference_storage_valid","positive_residual_unit")), "Nonfinite/invalid complete state")
    require(bool(np.asarray(adapter.reference_equal(state.microstep.reference,state.aux.posterior))), "Actor reference binding differs")
    if due:
        require(c["reference_ready"] and flags["ready_reference_finite"], "Calibrated reference invalid/infinite")
    else:
        require(not c["reference_ready"] and flags["warmup_reference_shape"], "Invalid intentional unready warmup reference")
    return {"counts":c,"checks":flags}


def step(adapter, state, *, progress=None):
    """One atomic native critic + original auxiliary fit + due calibration."""
    import numpy as np
    context = adapter.context
    jax, jnp = context.native.jax, context.native.jnp
    attempted = {"critic":0,"actor":0,"actor_target":0,"critic_target":0,"auxiliary":0,"refresh":0}
    completed = dict.fromkeys(attempted, 0)
    diagnostics = {}; stage = "preflight"
    def notify(event):
        if progress is not None:
            progress({"stage":stage, "event":event, "attempted":dict(attempted),
                      "completed":dict(completed)})
    try:
        c = validate(adapter,state)["counts"]
        require(c["critic"] < adapter.budget,"Budget complete")
        stage = "native_microstep"; attempted["critic"] = 1
        for key in ("actor","actor_target","critic_target"):
            attempted[key] = int(c["phase"] == 1)
        notify("before_dispatch")
        micro, metrics = block(jax,adapter.microstep_step(state.microstep))
        for key in ("critic","actor","actor_target","critic_target"):
            completed[key] = attempted[key]
        notify("after_dispatch")
        diagnostics["native"] = host(metrics)
        require(bool(np.asarray(adapter.finite((micro.agent_state,micro.batch,micro.rng,metrics)))),"Nonfinite native result")
        auxiliary = state.aux._replace(native=context.pipeline.native_view(micro.agent_state))
        next_key,fit_key,refresh_key = jax.random.split(state.aux_rng,3)
        stage = "auxiliary_fit"; attempted["auxiliary"] = 1
        notify("before_dispatch")
        auxiliary,metrics = block(jax,context.fit_step(auxiliary,micro.batch,fit_key))
        completed["auxiliary"] = int(np.asarray(auxiliary.calibrator.step))-int(np.asarray(state.aux.calibrator.step))
        notify("after_dispatch")
        diagnostics["auxiliary"] = host(metrics)
        require(bool(np.asarray(metrics["scale_inputs_valid"])) and bool(np.asarray(metrics["scale_fit_accepted"])),"Original auxiliary fit rejected")
        critic = c["critic"]+1; index = c["next_refresh_index"]
        refresh_due = index < len(adapter.schedule) and critic == adapter.schedule[index]
        if refresh_due:
            stage = "calibration_refresh"; attempted["refresh"] = 1
            notify("before_dispatch")
            auxiliary,flags,details = context.pipeline.refresh(context.aux_args,context.aux_config,context.models,
                auxiliary,context.heldout_transition,refresh_key,heldout_ids=context.heldout_ids["row_ids"])
            block(jax,auxiliary)
            completed["refresh"] = 1
            notify("after_dispatch")
            diagnostics["refresh"] = {"flags":host(flags),"diagnostics":host(details)}
            ref = auxiliary.posterior
            require(bool(flags["posterior_inputs_valid"]) and bool(flags["posterior_certified"]) and bool(details["certified"])
                and bool(np.asarray(context.pipeline.reference_valid(ref))) and bool(np.asarray(ref.ready))
                and all(np.isfinite(float(np.asarray(getattr(ref,k)))) for k in ("threshold","lambda_hat","lambda_hpd","residual_scale","n_eff")), "Calibration rejected/infinite")
        # Boundary actor used previous reference. Only subsequent actors see this reference.
        one, inc = jnp.int32(1),jnp.int32(int(refresh_due))
        proposed = state._replace(microstep=micro._replace(reference=auxiliary.posterior),aux=auxiliary,aux_rng=next_key,
            aux_fit_attempts=state.aux_fit_attempts+one,aux_fit_accepted=state.aux_fit_accepted+one,
            refresh_attempts=state.refresh_attempts+inc,refresh_accepted=state.refresh_accepted+inc,
            next_refresh_index=state.next_refresh_index+inc,last_refresh_critic=jnp.int32(critic) if refresh_due else state.last_refresh_critic)
        stage = "commit_validation"
        checked = validate(adapter,proposed)
        return proposed,{"committed":True,"arm":adapter.arm,"kappa":adapter.kappa,"refresh_due":refresh_due,
                         "counts":checked["counts"],**diagnostics}
    except Exception as exc:
        raise context.training.TrainingStepError(stage+": "+str(exc),last_committed_state=state,
            stage=stage,attempted_counts=attempted,diagnostics=diagnostics) from exc


def match(adapter,state):
    """One immutable-reference gradient match; no optimizer, calibration or RNG updates."""
    import numpy as np
    context = adapter.context
    jax,jnp = context.native.jax,context.native.jnp
    design = module("_central_training_distance_design", HERE / "design.py")
    ref = jax.tree_util.tree_map(jax.lax.stop_gradient,state.aux.posterior)
    require(bool(np.asarray(ref.ready)),"Match requires completed calibration")
    def one(obs,action):
        value,usable,valid = context.pipeline.live_width(context.models,ref,obs[None,:],action[None,:])
        return jnp.where(usable & valid,value[0],jnp.nan)
    gradients = jax.jit(jax.vmap(jax.grad(one,argnums=1)))(context.probe.obs,context.probe.action)
    actor = state.microstep.agent_state.actor
    anchors = actor.apply_fn(actor.params,context.probe.obs).sample(seed=None)
    distance = design.distance_action_gradient(np.asarray(context.probe.action),np.asarray(anchors))
    receipt = design.match_coefficient(np.asarray(gradients),distance)
    return {**receipt,"probe_rows":len(context.probe.obs),"reference_digest":tree_digest(state.aux.posterior),
            "anchor_actor_digest":tree_digest(actor.params),"unscaled_bca_gradient_digest":tree_digest(gradients),
            "distance_gradient_digest":tree_digest(distance),"optimizer_updates":0,"calibration_calls":0,
            "no_rematch":True,
            "matching_scope": "RMS action gradients at logged training actions only; excludes actor Jacobian and Q normalization",
            "on_policy_strength_matched":False, "distance_action_gradient_at_fork_actor":0.0}, np.asarray(gradients), distance


def _compare_trees(jax, left, right, *, rtol=2e-5, atol=2e-6):
    """Use the previously frozen native parity tolerances; integers stay exact."""
    import numpy as np
    from flax import serialization
    a, sa = jax.tree_util.tree_flatten(serialization.to_state_dict(left))
    b, sb = jax.tree_util.tree_flatten(serialization.to_state_dict(right))
    require(sa == sb, "Comparison tree structures differ")
    maximum = 0.
    for x, y in zip(a, b):
        x, y = np.asarray(x), np.asarray(y)
        require(x.shape == y.shape, "Comparison leaf shapes differ")
        require(np.all(np.isfinite(x)) and np.all(np.isfinite(y)), "Nonfinite comparison leaf")
        if x.dtype.kind in "biu" and y.dtype.kind in "biu":
            np.testing.assert_array_equal(x, y)
        else:
            require(x.dtype == y.dtype, "Comparison floating dtypes differ")
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
        if x.size:
            maximum = max(maximum, float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))))
    return maximum


def _check_live_gradients(adapter, state):
    """Gradient-only real or toy model checks; no optimizer or RNG updates."""
    import numpy as np
    context = adapter.context
    jax, jnp = context.native.jax, context.native.jnp
    agent, ref = state.microstep.agent_state, state.aux.posterior
    before = tree_digest(state)
    batch = jax.tree_util.tree_map(lambda x: x[:8], context.probe)
    frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, ref)
    width = width_function(context, "bca", agent.actor.params)
    def wmean(actions):
        return width(frozen, batch.obs, actions).mean()
    action_grad = jax.grad(wmean)(batch.action)
    direct_grad = jax.grad(lambda actions: context.pipeline.live_width(
        context.models, frozen, batch.obs, actions)[0].mean())(batch.action)
    action_error = _compare_trees(jax, action_grad, direct_grad)
    require(np.linalg.norm(np.asarray(action_grad, np.float64)) > 0, "No live BCA action gradient in fixed synthetic fixture")
    # A fixed direction and two prespecified step sizes avoid selecting a
    # favorable coordinate or tuning finite-difference tolerances afterward.
    direction = jnp.sin(jnp.arange(batch.action.size, dtype=jnp.float32)+1).reshape(batch.action.shape)
    direction = direction / jnp.linalg.norm(direction)
    scale = jax.lax.stop_gradient(jnp.maximum(1., jnp.abs(wmean(batch.action))))
    automatic = jnp.sum(action_grad * direction) / scale
    differences = []
    for epsilon in (.001, .0005):
        plus = float(wmean(batch.action+epsilon*direction))
        minus = float(wmean(batch.action-epsilon*direction))
        finite_difference = (plus-minus)/(2*epsilon*float(scale))
        # Model arithmetic remains float32. Cancellation of almost equal width
        # means can dominate a small directional derivative even though the
        # final subtraction is performed on host float64. Report that limit
        # rather than claiming an unresolved finite difference verifies AD.
        roundoff_bound = (2*np.finfo(np.float32).eps*(abs(plus)+abs(minus))
                          /(2*epsilon*float(scale)))
        error = abs(float(automatic)-finite_difference)
        tolerance = .03*abs(float(automatic))+3e-5+roundoff_bound
        require(error <= tolerance, "Live-width finite difference exceeds truncation and float32 roundoff allowance")
        resolved = abs(float(automatic)) > 4*roundoff_bound
        differences.append({"epsilon":epsilon, "automatic":float(automatic),
            "finite_difference":finite_difference, "roundoff_bound":roundoff_bound,
            "absolute_error":error, "total_tolerance":tolerance,
            "resolved_above_float32_roundoff":resolved,
            "interpretation":"resolved_directional_check" if resolved else "inconclusive_float32_cancellation"})
    bca = context.candidate.make_microsteps(context.args, agent.actor.apply_fn, agent.dual_q.apply_fn,
        context.train_transition, width_fn=width, kappa_penalty=3.)
    native = context.candidate.make_microsteps(context.args, agent.actor.apply_fn, agent.dual_q.apply_fn,
        context.train_transition, width_fn=None, kappa_penalty=0.)
    bca_grad = jax.grad(lambda p:bca.actor_loss(p, agent, batch, ref)[0])(agent.actor.params)
    native_grad = jax.grad(lambda p:native.actor_loss(p, agent, batch, ref)[0])(agent.actor.params)
    _, metrics = native.actor_loss(agent.actor.params, agent, batch, ref)
    coefficient = jax.lax.stop_gradient(metrics["q_scale_coefficient"])
    def penalty(params):
        actions = jax.vmap(lambda obs:agent.actor.apply_fn(params, obs).sample(seed=None))(batch.obs)
        return coefficient * 3. * width(frozen, batch.obs, actions).mean()
    expected = jax.grad(penalty)(agent.actor.params)
    actual = jax.tree_util.tree_map(lambda a,b:a-b, bca_grad, native_grad)
    actor_error = _compare_trees(jax, actual, expected)
    require(any(np.any(np.asarray(x) != 0) for x in jax.tree_util.tree_leaves(expected)), "BCA actor gradient did not receive live action width")
    reference_grad = jax.grad(lambda p:bca.actor_loss(agent.actor.params, agent, batch,
        ref._replace(cal_params=p))[0])(ref.cal_params)
    require(all(np.all(np.asarray(x) == 0) for x in jax.tree_util.tree_leaves(reference_grad)), "Calibration parameters received actor gradient")
    scalar_grad = jax.grad(lambda scalars:bca.actor_loss(agent.actor.params, agent, batch,
        ref._replace(threshold=scalars[0], residual_scale=scalars[1]))[0])(
            jnp.array([ref.threshold, ref.residual_scale]))
    np.testing.assert_array_equal(scalar_grad, np.zeros(2))
    require(before == tree_digest(state), "Gradient check mutated complete state")
    return {"action_gradient_max_abs_error":action_error, "actor_penalty_gradient_max_abs_error":actor_error,
        "frozen_reference_gradient_zero":True, "live_action_gradient_nonzero":True,
        "finite_differences":differences, "finite_difference_rtol":.03, "finite_difference_atol":3e-5,
        "finite_difference_resolved":all(x['resolved_above_float32_roundoff'] for x in differences),
        "optimizer_updates":0, "state_unchanged":True}


def synthetic_model_check(protocol, *, progress=None):
    """Manually invoked actual-model check with a failure-safe setup receipt.

    Generated tuples only; no production dataset, checkpoint or simulator.
    Attempted counts include native parity comparisons and duplicated restore
    updates. A failure preserves attempted upper bounds and verified counts.
    Never called by create(), imports, CPU toy tests or production training.
    """
    import time
    started = time.monotonic()
    keys = ("critic", "actor", "auxiliary", "calibration")
    ledger = {"attempted":dict.fromkeys(keys,0), "verified":dict.fromkeys(keys,0)}
    result = {"status":"running_synthetic_runtime_check", "stage":"runtime_setup", "production_data_rows":0,
        "simulator_transitions":0, "real_checkpoint_restores":0, "execution_counts_known":True,
        "caps":{"critic_updates":64, "actor_updates":32, "auxiliary_updates":64,
                "calibration_calls":4, "seconds":600}}
    def notify():
        if progress is not None:
            import copy
            progress(copy.deepcopy({"stage":result['stage'], "counts":ledger,
                "execution_counts_known":result['execution_counts_known'],
                "inflight":ledger['attempted'] != ledger['verified']}))
    try:
        notify()
        _synthetic_model_check_body(protocol, result, ledger, started, notify)
        result.update(status="passed_synthetic_runtime_check", stage="completed")
    except BaseException as exc:
        result.update(status="failed_synthetic_runtime_check", error=type(exc).__name__+": "+str(exc))
        if ledger['attempted'] != ledger['verified']:
            result['execution_counts_known'] = False
    result.update(counts=ledger, elapsed_seconds=time.monotonic()-started)
    notify()
    # Setup can fail before NumPy is importable; this failure receipt uses only
    # Python containers/scalars and must still survive in that situation.
    return result


def real_source_initialization_check(protocol, *, deadline, progress=None):
    """Read-only real-source adapter check: five fresh states, zero updates.

    This is separately counted from generated-data optimizer fixtures. Data
    decoding and static initialization are charged before create(); if create
    fails its exact completed decode/model count is unknown, never assumed zero.
    """
    import copy
    import gc
    import time
    calls = ((0,'full'),(0,'hole'),(0,'random'),(1,'full'),(2,'full'))
    ledger = dict(initializations_attempted=0, initializations_completed=0,
        source_episodes_reserved=0, source_episodes_completed=0,
        source_tuples_reserved=0, source_tuples_completed=0,
        protected_transition_rows=0, optimizer_updates=0, calibration_updates=0,
        simulator_transitions=0)
    result = dict(status='running_real_source_initialization_check', stage='before_initialization',
        execution_counts_known=True, counts=ledger, records=[],
        caps=dict(initializations=5, source_episodes=4480, source_tuples=448000),
        data_scope='Read original train/calibration tuples for fresh static initialization only; no fitting')
    def notify():
        if progress is not None: progress(copy.deepcopy(result))
    try:
        notify()
        for seed_index,condition in calls:
            require(time.monotonic() < deadline, 'Read-only initialization deadline reached')
            require(ledger['initializations_attempted'] < 5, 'Initialization cap reached')
            ledger['initializations_attempted'] += 1
            ledger['source_episodes_reserved'] += 896
            ledger['source_tuples_reserved'] += 89600
            result.update(stage='initialization_dispatch', current={'seed_index':seed_index,'condition':condition})
            notify()
            adapter,state=create(protocol,seed_index,condition)
            context=adapter.context
            receipt=context.data.receipt
            decoded=receipt['decoded_roles']
            require(decoded['episodes']==896 and decoded['transition_tuples']==89600
                and decoded['selection_or_final_calibration_transition_rows']==0
                and decoded['development_probe_transition_rows']==0,
                'Initialization source-role accounting differs')
            ledger['source_episodes_completed'] += decoded['episodes']
            ledger['source_tuples_completed'] += decoded['transition_tuples']
            ledger['initializations_completed'] += 1
            result['stage']='initialization_completed'
            notify()
            counts=context.training.counts(state)
            require(all(counts[k]==0 for k in ('critic','actor','auxiliary','refresh_attempts','refresh_accepted'))
                and not counts['reference_ready'] and counts['phase']==0,
                'Read-only initialization unexpectedly updated or calibrated state')
            require(all(x==0 for x in context.initial_optimizer_steps.values()), 'Nonzero initial optimizer state')
            require(receipt['data_plan_sha256']==protocol['data_plan_sha256'], 'Initialization data plan differs')
            require(context.normalization['rows']==60924, 'Common normalization row count differs')
            result['records'].append(dict(seed_index=seed_index,condition=condition,
                normalization=context.normalization, common_normalization=receipt['common_normalization'],
                initial_parameter_hashes=context.initial_parameter_hashes,
                initial_optimizer_hashes=context.initial_optimizer_hashes,
                initial_optimizer_steps=context.initial_optimizer_steps, counts=counts,
                native_source=context.runtime['native_source'],
                data_plan_sha256=receipt['data_plan_sha256']))
            del adapter,state,context
            gc.collect()
            notify()
        first=result['records'][0]
        for row in result['records']:
            require(row['normalization']==first['normalization'] and
                    row['common_normalization']==first['common_normalization'],
                    'Common normalization or row provenance differs during actual initialization')
        for row in result['records'][:3]:
            require(row['initial_parameter_hashes']==first['initial_parameter_hashes'] and
                    row['initial_optimizer_hashes']==first['initial_optimizer_hashes'],
                    'Paired actual initialization weights/optimizers differ')
        for network in ('actor','critic'):
            hashes=[result['records'][i]['initial_parameter_hashes'][network] for i in (0,3,4)]
            require(len(set(hashes))==3, 'Independent actual initialization seeds produced identical networks')
        require(time.monotonic() < deadline, 'Read-only initialization deadline reached')
        result.update(status='passed_real_source_initialization_check',stage='completed')
    except BaseException as exc:
        result.update(status='failed_real_source_initialization_check',error=type(exc).__name__+': '+str(exc))
        result['execution_counts_known']=ledger['initializations_attempted']==ledger['initializations_completed']
    notify()
    return result


def _synthetic_model_check_body(protocol, result, ledger, started, notify):
    import time
    import numpy as np
    from flax import serialization
    native = load_native()
    receipt = runtime_receipt(native)
    result["runtime"] = receipt
    result["stage"] = "synthetic_initialization"
    notify()
    # Fixed synthetic generator is not a production training/evaluation seed.
    rng = np.random.Generator(np.random.PCG64(731531))
    def bank(n):
        obs = rng.uniform(-.5,.5,(n,14)).astype(np.float32)
        obs[:,-1] = np.tile(np.linspace(.01,1,100,dtype=np.float32), (n+99)//100)[:n]
        action = rng.uniform(-.8,.8,(n,4)).astype(np.float32)
        reward = (-.01*np.sum(obs[:,:10]**2,axis=1)-.03*np.sum(action**2,axis=1)).astype(np.float32)
        nxt = obs.copy(); nxt[:,-1] = np.maximum(0.,obs[:,-1]-.01)
        done = (nxt[:,-1] == 0).astype(np.float32)
        return native.Transition(obs,action,reward,nxt,done)
    replay, heldout, probe = bank(512), bank(128), bank(128)
    data = SimpleNamespace(train=replay, heldout=heldout, probe=probe,
        normalization_fixture=replay,
        heldout_ids={"row_ids":np.arange(128)+1000000, "episode_ids":np.arange(128)+10000},
        source_row_ids=np.arange(512), receipt={"synthetic_only":True})
    adapter,state = _initialize_data(protocol,0,"synthetic_only",native,data,receipt)
    jax,jnp = native.jax,native.jnp
    state=state._replace(refresh_at=jnp.array([4],jnp.int32),critic_budget=jnp.int32(4))
    adapter=make_adapter(adapter.context,state,"native",0.,state.microstep.agent_state.actor.params,[4],4)
    def reserve(critic, actor, auxiliary, calibration):
        require(time.monotonic()-started < 600, "Synthetic model-check 600-second cap reached")
        for key, amount, cap in (("critic",critic,64),("actor",actor,32),("auxiliary",auxiliary,64),
                                ("calibration",calibration,4)):
            require(ledger['attempted'][key]+amount <= cap, "Synthetic "+key+" cap reached")
        for key, amount in zip(("critic","actor","auxiliary","calibration"),(critic,actor,auxiliary,calibration)):
            ledger['attempted'][key] += amount
        notify()
    def checked_step(a,s):
        last_attempted = dict.fromkeys(ledger['attempted'], 0)
        last_completed = dict.fromkeys(ledger['verified'], 0)
        def dispatch(event):
            mapping = {'critic':'critic','actor':'actor','auxiliary':'auxiliary','calibration':'refresh'}
            changes = {k:event['attempted'][v]-last_attempted[k] for k,v in mapping.items()}
            reserve(changes['critic'],changes['actor'],changes['auxiliary'],changes['calibration'])
            for key,mapped in mapping.items():
                ledger['verified'][key] += event['completed'][mapped]-last_completed[key]
                last_attempted[key] = event['attempted'][mapped]
                last_completed[key] = event['completed'][mapped]
            result['dispatch_stage'] = event['stage']
            notify()
        try:
            after, metrics = step(a,s,progress=dispatch)
        except BaseException:
            result['execution_counts_known'] = ledger['attempted'] == ledger['verified']
            notify()
            raise
        return after,metrics
    result["stage"] = "actual_native_parity"
    notify()
    template=state
    direct=(state.microstep.rng,state.microstep.agent_state)
    context=adapter.context
    direct_step=jax.jit(native.make_train_step(context.args,direct[1].actor.apply_fn,
        direct[1].dual_q.apply_fn,context.train_transition))
    parity=[]
    for outer in range(2):
        state,_=checked_step(adapter,state)
        state,metrics=checked_step(adapter,state)
        reserve(2,1,0,0)
        previous=direct[1]
        try:
            direct,direct_metrics=block(jax,direct_step(direct,None))
        except BaseException:
            result['execution_counts_known'] = False
            notify()
            raise
        critic_delta=int(direct[1].dual_q.step)-int(previous.dual_q.step)
        actor_delta=int(direct[1].actor.step)-int(previous.actor.step)
        ledger['verified']['critic']+=critic_delta
        ledger['verified']['actor']+=actor_delta
        notify()
        require((critic_delta,actor_delta)==(2,1),"Native comparison update counters differ")
        np.testing.assert_array_equal(direct[0],state.microstep.rng)
        error=_compare_trees(jax,direct[1],state.microstep.agent_state)
        metric_errors={}
        for key,value in direct_metrics.items():
            mapped="q_scale_coefficient" if key == "lambda" else key
            # The microstep receipt is host-converted JSON scalars. Restore its
            # known native metric dtype for this comparison only; model/state
            # dtype assertions above remain strict.
            expected_metric = np.asarray(metrics['native'][mapped], dtype=np.asarray(value).dtype)
            metric_errors[mapped]=_compare_trees(jax,value,expected_metric)
        parity.append({"outer":outer+1,"state_max_abs_error":error,"metric_max_abs_errors":metric_errors,
            "rng_exact":True,"float_rtol":2e-5,"float_atol":2e-6})
    result['native_parity']=parity
    result['stage']='live_bca_gradient'
    notify()
    result['live_bca_gradient']=host(_check_live_gradients(adapter,state))
    warmup=state
    raw=serialization.to_bytes(warmup)
    restored=restore_exact(adapter,template,raw)
    require(tree_digest(warmup)==tree_digest(restored),"Synthetic restore tree differs")
    matching,_,_=match(adapter,warmup)
    result.update(common_normalization=adapter.context.normalization,
        warmup_full_state_sha256=digest(raw),matching=matching,branches=[])
    result['stage']='branch_restore_and_refresh'
    notify()
    for arm,kappa in (("native",0.),("bca",3.),("distance",matching['distance_coefficient'])):
        anchor=warmup.microstep.agent_state.actor.params
        branch=warmup._replace(refresh_at=jnp.array([4,6],jnp.int32),critic_budget=jnp.int32(8))
        aa=make_adapter(context,branch,arm,kappa,anchor,[4,6],8)
        saved=serialization.to_bytes(branch)
        restored_branch=restore_exact(aa,branch,saved)
        first,_=checked_step(aa,branch)
        again,_=checked_step(aa,restored_branch)
        require(tree_digest(first)==tree_digest(again),"Restored next step differs: "+arm)
        current=first
        for _ in range(3): current,_=checked_step(aa,current)
        checked=validate(aa,current)
        result['branches'].append({"arm":arm,"counts":checked['counts'],"restored_next_step_equal":True})
