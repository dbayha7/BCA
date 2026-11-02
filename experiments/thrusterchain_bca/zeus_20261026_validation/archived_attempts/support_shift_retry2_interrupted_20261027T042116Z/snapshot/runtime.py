"""Fresh support-pilot adapter; scientific imports occur only inside functions.

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
    loader = module("_support_native_loader", PAYLOAD / "unifloral_td3bc_validation_v1/native_loader.py")
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


def create(protocol, seed_index, condition):
    """Fresh models/optimizers only. No old checkpoint, update, refresh or simulator."""
    import numpy as np
    native = load_native()
    receipt = runtime_receipt(native)
    jax, jnp = native.jax, native.jnp
    data = module("_support_condition_data", HERE / "data.py").load_condition(native, condition)
    training = module("_support_training_types", PAYLOAD / "unifloral_checkpoint_v1/training.py")
    candidate = module("_support_candidate", PAYLOAD / "unifloral_td3bc_validation_v1/candidate.py")
    bridge = module("_support_bca_bridge", PAYLOAD / "unifloral_bca_pipeline_validation_v1/bca_bridge.py")
    loader = module("_support_native_loader", PAYLOAD / "unifloral_td3bc_validation_v1/native_loader.py")
    train = training._transitions(native, data.train, "eligible training", 57752)
    heldout = training._transitions(native, data.heldout, "eligible calibration", 128)
    probe = training._transitions(native, data.probe, "eligible probes", 768)
    training._transitions(native, data.scale_bank, "eligible provenance-only scale bank", 256)
    ids = {k: np.asarray(v, np.int64) for k, v in data.heldout_ids.items()}
    require(set(ids) == {"row_ids", "episode_ids"} and all(v.shape == (128,) and len(np.unique(v)) == 128 for v in ids.values()), "Calibration identities invalid")
    seed = protocol["seeds"][seed_index]
    args = native.Args(seed=seed["native_seed"], dataset="thrusterchain-finite-horizon-v1",
                       num_updates=6000, gamma=.98, batch_size=256)
    rng, agent = loader.initialize(native, args, train, seed["native_seed"])
    pipeline = bridge.load_pipeline(PAYLOAD / "unifloral_bca_pipeline_validation_v1")
    models, aux, aux_config, aux_args = pipeline.initialize_pipeline(agent, train, args, seed["aux_seed"])
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
                       "epsilon": .001, "source": "Native population statistics of this condition's retained training observations"},
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
    c = validate(adapter, state)["counts"]
    require(c["critic"] == 10000 and c["phase"] == 0 and c["refresh_accepted"] == 1 and c["reference_ready"], "Incomplete warmup cannot fork")
    require(np.isfinite(coefficient) and coefficient > 0, "Invalid fixed coefficient")
    seed = protocol["seeds"][seed_index]
    native_key = jax.random.fold_in(state.microstep.rng, seed["branch_native_seed"])
    aux_key = jax.random.fold_in(state.aux_rng, seed["branch_aux_seed"])
    anchor = state.microstep.agent_state.actor.params
    state = state._replace(microstep=state.microstep._replace(rng=native_key, reference=state.aux.posterior),
        aux_rng=aux_key, refresh_at=jnp.array([10000,11000], jnp.int32), critic_budget=jnp.int32(12000))
    kappa = {"native":0., "bca":3., "distance":float(coefficient)}[arm]
    result = make_adapter(context, state, arm, kappa, anchor, [10000,11000], 12000)
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


def step(adapter, state):
    """One atomic native critic + original auxiliary fit + due calibration."""
    import numpy as np
    context = adapter.context
    jax, jnp = context.native.jax, context.native.jnp
    attempted = {"critic":0,"actor":0,"actor_target":0,"critic_target":0,"auxiliary":0,"refresh":0}
    diagnostics = {}; stage = "preflight"
    try:
        c = validate(adapter,state)["counts"]
        require(c["critic"] < adapter.budget,"Budget complete")
        stage = "native_microstep"; attempted["critic"] = 1
        for key in ("actor","actor_target","critic_target"):
            attempted[key] = int(c["phase"] == 1)
        micro, metrics = block(jax,adapter.microstep_step(state.microstep))
        diagnostics["native"] = host(metrics)
        require(bool(np.asarray(adapter.finite((micro.agent_state,micro.batch,micro.rng,metrics)))),"Nonfinite native result")
        auxiliary = state.aux._replace(native=context.pipeline.native_view(micro.agent_state))
        next_key,fit_key,refresh_key = jax.random.split(state.aux_rng,3)
        stage = "auxiliary_fit"; attempted["auxiliary"] = 1
        auxiliary,metrics = block(jax,context.fit_step(auxiliary,micro.batch,fit_key))
        diagnostics["auxiliary"] = host(metrics)
        require(bool(np.asarray(metrics["scale_inputs_valid"])) and bool(np.asarray(metrics["scale_fit_accepted"])),"Original auxiliary fit rejected")
        critic = c["critic"]+1; index = c["next_refresh_index"]
        refresh_due = index < len(adapter.schedule) and critic == adapter.schedule[index]
        if refresh_due:
            stage = "calibration_refresh"; attempted["refresh"] = 1
            auxiliary,flags,details = context.pipeline.refresh(context.aux_args,context.aux_config,context.models,
                auxiliary,context.heldout_transition,refresh_key,heldout_ids=context.heldout_ids["row_ids"])
            block(jax,auxiliary)
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
    design = module("_support_design",HERE / "design.py")
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
            "no_rematch":True}, np.asarray(gradients), distance
