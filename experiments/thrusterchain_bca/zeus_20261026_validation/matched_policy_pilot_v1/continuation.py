"""Explicit paired continuation adapter for the original UniFloral BCA method.

No model construction or data access occurs on import. The complete restored
TrainingState is copied, with declared RNG forks, extended schedule/budget and
an explicitly bound actor reference. Original source files remain unchanged.
"""
from dataclasses import asdict, is_dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE.parents[1] / "titan_20261026/payload"
ARMS = {"native": 0.0, "bca": 3.0, "distance": 0.1}
DISTANCE_FACTOR = 19.29619572569095


class ContinuationStepError(RuntimeError):
    def __init__(self, message, *, last_committed_state, stage, attempted_counts, diagnostics):
        super().__init__(message)
        self.last_committed_state = last_committed_state
        self.stage = stage
        self.attempted_counts = attempted_counts
        self.diagnostics = diagnostics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        result = sys.modules[name]
        require(Path(result.__file__).resolve() == path, "Module path collision")
        return result
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def _config(value):
    return asdict(value) if is_dataclass(value) else dict(vars(value))


def host(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): host(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [host(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return np.asarray(value).tolist()


def _block(jax, value):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


def make_width_fn(arm, actor_apply_fn, frozen_actor_params, pipeline, models,
                  distance_factor=DISTANCE_FACTOR):
    import jax
    import jax.numpy as jnp
    require(arm in ARMS and distance_factor == DISTANCE_FACTOR, "Unknown fixed arm/distance factor")
    if arm == "native":
        return None
    if arm == "bca":
        def bca_width(reference, observations, live_actions):
            width, usable, valid = pipeline.live_width(models, reference, observations, live_actions)
            # Nonfinite proposals are refused atomically; never consume a zero fallback.
            return jnp.where(usable & valid, width, jnp.full_like(width, jnp.nan))
        return bca_width
    anchor = jax.tree_util.tree_map(jax.lax.stop_gradient, frozen_actor_params)
    def distance_width(reference, observations, live_actions):
        del reference
        anchor_actions = actor_apply_fn(anchor, observations).sample(seed=None)
        anchor_actions = jax.lax.stop_gradient(anchor_actions)
        return jnp.asarray(distance_factor, dtype=live_actions.dtype) * jnp.mean(
            jnp.square(live_actions - anchor_actions), axis=-1)
    return distance_width


def initialize_continuation(context, original_state, arm, native_seed, aux_seed,
                            critic_updates=2000, refresh_at=1000, *,
                            candidate_factory=None, training_api=None):
    """Pure continuation construction; reduced fixture budgets never reach worker.

    The worker separately pins the real 2000/1000 budget and source identities.
    Injected factories/APIs support synthetic tests without changing real logic.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    require(arm in ARMS, "Unknown arm")
    require(type(critic_updates) is int and critic_updates > 0 and critic_updates % 2 == 0,
            "Continuation budget must be positive/even")
    require(type(refresh_at) is int and 0 < refresh_at < critic_updates and refresh_at % 2 == 0,
            "Halfway refresh must be internal/even")
    require(all(type(seed) is int and 0 <= seed < 2**32 for seed in (native_seed, aux_seed)), "Invalid RNG stream IDs")
    if candidate_factory is None:
        candidate_factory = _module("_matched_native_candidate", PAYLOAD / "unifloral_td3bc_validation_v1/candidate.py").make_microsteps
    if training_api is None:
        training_api = _module("_operator_original_training", PAYLOAD / "unifloral_checkpoint_v1/training.py")
    initial = training_api.counts(original_state)
    require(initial["phase"] == 0 and initial["critic"] == initial["critic_budget"], "Only a complete terminal checkpoint may fork")
    old_schedule = np.asarray(original_state.refresh_at).tolist()
    require(initial["next_refresh_index"] == len(old_schedule)
            and initial["last_refresh_critic"] == initial["critic"], "Initial refresh boundary mismatch")
    schedule = old_schedule + [initial["critic"] + refresh_at]
    reference = original_state.aux.posterior
    require(bool(np.asarray(context.pipeline.reference_valid(reference))) and bool(np.asarray(reference.ready)), "Initial reference invalid")
    require(all(np.isfinite(float(np.asarray(getattr(reference, key)))) for key in
                ("threshold", "lambda_hat", "lambda_hpd", "residual_scale", "n_eff")), "Initial reference nonfinite")
    actor = original_state.microstep.agent_state.actor
    critic = original_state.microstep.agent_state.dual_q
    width = make_width_fn(arm, actor.apply_fn, actor.params, context.pipeline, context.models)
    functions = candidate_factory(context.args, actor.apply_fn, critic.apply_fn,
                                  context.train_transition, width_fn=width, kappa_penalty=ARMS[arm])
    native_key = jax.random.fold_in(original_state.microstep.rng, native_seed)
    aux_key = jax.random.fold_in(original_state.aux_rng, aux_seed)
    micro = original_state.microstep._replace(rng=native_key, reference=reference)
    state = original_state._replace(microstep=micro, aux_rng=aux_key,
        refresh_at=jnp.asarray(schedule, dtype=jnp.int32),
        critic_budget=jnp.asarray(initial["critic"] + critic_updates, dtype=jnp.int32))

    def finite(tree):
        result = jnp.asarray(True)
        for leaf in jax.tree_util.tree_leaves(tree):
            result = result & jnp.all(jnp.isfinite(leaf))
        return result

    def equal(left, right):
        a, sa = jax.tree_util.tree_flatten(left)
        b, sb = jax.tree_util.tree_flatten(right)
        if sa != sb or len(a) != len(b):
            raise ValueError("Duplicated state pytree mismatch")
        result = jnp.asarray(True)
        for x, y in zip(a, b):
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("Duplicated state leaf shape/dtype mismatch")
            result = result & jnp.array_equal(x, y)
        return result

    def checks(s):
        agent = s.microstep.agent_state
        view = s.aux.native
        return {"finite": finite(s),
                "native_view_equal": equal((agent.actor, agent.actor_target, agent.dual_q, agent.dual_q_target),
                                           (view.actor, view.actor_target, view.critic, view.critic_target)),
                "actor_reference_equal": equal(s.microstep.reference, s.aux.posterior),
                "reference_valid": context.pipeline.reference_valid(s.aux.posterior),
                "positive_unit": s.aux.residual_scale > 0}
    adapter = SimpleNamespace(source=context, training=training_api, arm=arm, functions=functions,
        microstep_step=jax.jit(functions.step), finite=jax.jit(finite), checks=jax.jit(checks),
        initial_counts=initial, schedule=schedule, budget=initial["critic"] + critic_updates,
        critic_updates=critic_updates, refresh_relative=refresh_at,
        args_snapshot=_config(context.args), aux_config_snapshot=_config(context.aux_config),
        rng_receipt={"operation": "fold_in saved checkpoint keys with paired stream IDs",
                     "native_seed": native_seed, "aux_seed": aux_seed,
                     "saved_native_rng": host(original_state.microstep.rng), "saved_aux_rng": host(original_state.aux_rng),
                     "forked_native_rng": host(native_key), "forked_aux_rng": host(aux_key)})
    validate(adapter, state)
    return adapter, state


def validate(adapter, state):
    """Continuation invariants; original terminal-only validator is not relaxed."""
    import numpy as np
    c = adapter.training.counts(state)
    require(_config(adapter.source.args) == adapter.args_snapshot
            and _config(adapter.source.aux_config) == adapter.aux_config_snapshot, "Scientific context mutated")
    require(c["critic_budget"] == adapter.budget and adapter.initial_counts["critic"] <= c["critic"] <= adapter.budget,
            "Continuation critic budget violated")
    require(np.asarray(state.refresh_at).tolist() == adapter.schedule, "Continuation schedule changed")
    actor = c["critic"] // 2
    require(c["phase"] == c["critic"] % 2 and all(c[k] == actor for k in
            ("actor", "actor_target", "critic_target", "outer_steps")), "Native cadence/counters differ")
    require(c["auxiliary"] == c["aux_fit_attempts"] == c["critic"], "Shadow auxiliary count differs")
    due = sum(t <= c["critic"] for t in adapter.schedule)
    require(all(c[k] == due for k in ("refresh_attempts", "refresh_accepted", "next_refresh_index")), "Refresh accounting differs")
    require(c["last_refresh_critic"] == adapter.schedule[due - 1], "Last refresh differs")
    agent = state.microstep.agent_state
    for value, count in ((agent.dual_q, c["critic"]), (agent.actor, actor),
                         (agent.actor_target, actor), (agent.dual_q_target, actor), (state.aux.calibrator, c["auxiliary"])):
        require(int(np.asarray(value.step)) == count, "TrainState/optimizer counter mismatch")
    b = state.microstep.batch
    require(getattr(b, "_fields", None) == ("obs", "action", "reward", "next_obs", "done"), "Native transition schema changed")
    n = adapter.source.args.batch_size
    for x, shape in zip(b, ((n, 14), (n, 4), (n,), (n, 14), (n,))):
        require(x.shape == shape and x.dtype == np.dtype(np.float32), "Held replay batch shape/dtype changed")
    for key in (state.microstep.rng, state.aux_rng):
        require(np.asarray(key).shape == (2,) and np.asarray(key).dtype == np.dtype(np.uint32), "RNG representation changed")
    require(bool(np.asarray(state.aux.posterior.ready)), "Reference unready")
    flags = host(adapter.checks(state))
    require(all(flags.values()), "Continuation state invalid: " + str(flags))
    return {"counts": c, "checks": flags}


def step(adapter, state):
    """Commit one native critic microstep, matching auxiliary fit and due refresh."""
    import numpy as np
    context = adapter.source
    jax, jnp = context.native.jax, context.native.jnp
    attempted = {"critic": 0, "actor": 0, "actor_target": 0, "critic_target": 0, "auxiliary": 0, "refresh": 0}
    diagnostics = {}
    stage = "preflight"
    try:
        c = validate(adapter, state)["counts"]
        require(c["critic"] < adapter.budget, "Declared continuation budget complete")
        stage = "native_microstep"
        attempted["critic"] = 1
        for key in ("actor", "actor_target", "critic_target"):
            attempted[key] = int(c["phase"] == 1)
        micro, native_metrics = _block(jax, adapter.microstep_step(state.microstep))
        diagnostics["native"] = host(native_metrics)
        require(bool(np.asarray(adapter.finite((micro, native_metrics)))), "Nonfinite native result")
        auxiliary = state.aux._replace(native=context.pipeline.native_view(micro.agent_state))
        next_key, fit_key, refresh_key = jax.random.split(state.aux_rng, 3)
        stage = "auxiliary_fit"
        attempted["auxiliary"] = 1
        auxiliary, fit_metrics = _block(jax, context.fit_step(auxiliary, micro.batch, fit_key))
        diagnostics["auxiliary"] = host(fit_metrics)
        require(bool(np.asarray(fit_metrics["scale_inputs_valid"])) and bool(np.asarray(fit_metrics["scale_fit_accepted"])),
                "Original auxiliary fit rejected; no native-only fallback")
        critic = c["critic"] + 1
        index = c["next_refresh_index"]
        refresh_due = index < len(adapter.schedule) and critic == adapter.schedule[index]
        if refresh_due:
            stage = "calibration_refresh"
            attempted["refresh"] = 1
            auxiliary, flags, details = context.pipeline.refresh(
                context.aux_args, context.aux_config, context.models, auxiliary,
                context.heldout_transition, refresh_key, heldout_ids=context.heldout_ids["row_ids"])
            _block(jax, auxiliary)
            diagnostics["refresh"] = {"flags": host(flags), "diagnostics": host(details)}
            ref = auxiliary.posterior
            require(bool(flags["posterior_inputs_valid"]) and bool(flags["posterior_certified"])
                    and bool(details["certified"]) and bool(np.asarray(context.pipeline.reference_valid(ref)))
                    and bool(np.asarray(ref.ready)), "Calibration refresh refused")
            require(all(np.isfinite(float(np.asarray(getattr(ref, k)))) for k in
                        ("threshold", "lambda_hat", "lambda_hpd", "residual_scale", "n_eff")), "Nonfinite reference refused")
        # The boundary actor already ran with the prior reference; the next actor
        # sees the newly frozen posterior. Controls carry the same shadow work.
        micro = micro._replace(reference=auxiliary.posterior)
        one, inc = jnp.asarray(1, jnp.int32), jnp.asarray(int(refresh_due), jnp.int32)
        proposed = state._replace(microstep=micro, aux=auxiliary, aux_rng=next_key,
            aux_fit_attempts=state.aux_fit_attempts + one, aux_fit_accepted=state.aux_fit_accepted + one,
            refresh_attempts=state.refresh_attempts + inc, refresh_accepted=state.refresh_accepted + inc,
            next_refresh_index=state.next_refresh_index + inc,
            last_refresh_critic=jnp.asarray(critic, jnp.int32) if refresh_due else state.last_refresh_critic)
        stage = "commit_validation"
        counts = validate(adapter, proposed)["counts"]
        return proposed, {"committed": True, "arm": adapter.arm, "kappa": ARMS[adapter.arm],
                          "refresh_due": refresh_due, "counts": counts, **diagnostics}
    except Exception as exc:
        raise ContinuationStepError(stage + ": " + type(exc).__name__ + ": " + str(exc),
            last_committed_state=state, stage=stage, attempted_counts=attempted, diagnostics=diagnostics) from exc
