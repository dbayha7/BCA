"""Isolated native UniFloral kappa=0 training with actual BCA auxiliary fits.

Importing this module uses only the standard library. Native computation and
the unchanged uncertainty bridge are loaded only by create(). No simulator,
evaluation labels, dataset generator, or production core runner is imported.

One committed call is one native critic update plus one auxiliary fit. On the
second critic of a pair, native actor/target updates occur before the auxiliary
fit. A due eager refresh follows that fit. All state is committed together.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT.parent / "unifloral_td3bc_validation_v1"
STAGE2 = ROOT.parent / "unifloral_bca_pipeline_validation_v1"
NATIVE_SHA256 = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"
AUX_STREAM_DOMAIN = 1431323732


class TrainingState(NamedTuple):
    microstep: Any
    aux: Any
    aux_rng: Any
    aux_fit_attempts: Any
    aux_fit_accepted: Any
    refresh_attempts: Any
    refresh_accepted: Any
    next_refresh_index: Any
    last_refresh_critic: Any
    refresh_at: Any
    critic_budget: Any


class TrainingStepError(RuntimeError):
    """No tentative state was committed; explicit attempted work is retained."""

    def __init__(self, message, *, last_committed_state, stage, attempted_counts, diagnostics=None):
        super().__init__(message)
        self.last_committed_state = last_committed_state
        self.stage = stage
        self.attempted_counts = deepcopy(attempted_counts)
        self.diagnostics = {} if diagnostics is None else deepcopy(diagnostics)


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError("Refusing an existing helper from a different path: " + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _configuration(config):
    required = {"mode", "gamma", "batch_size", "seed", "aux_seed", "refresh_at", "critic_budget", "kappa_penalty"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Checkpoint config must contain exactly the agreed scientific fields")
    mode = config["mode"]
    if mode not in ("production", "tiny"):
        raise ValueError("Mode must be production or tiny")
    expected = {"gamma": 0.98, "seed": 2026102302, "aux_seed": 2026102361,
                "batch_size": 256 if mode == "production" else 8,
                "critic_budget": 20000 if mode == "production" else 6,
                "refresh_at": [10000, 15000, 20000] if mode == "production" else [2, 4, 6],
                "kappa_penalty": 0.0}
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError("Unsupported checkpoint configuration: " + key)
    for key in ("batch_size", "seed", "aux_seed", "critic_budget"):
        if type(config[key]) is not int:
            raise ValueError(key + " must be an integer")
    if type(config["refresh_at"]) is not list or any(type(n) is not int for n in config["refresh_at"]):
        raise ValueError("refresh_at must be the declared integer list")
    if isinstance(config["gamma"], bool) or isinstance(config["kappa_penalty"], bool):
        raise ValueError("gamma and kappa_penalty must be numeric scalars")
    return deepcopy(config)


def _digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _host(value):
    """JSON-safe device diagnostics, including a failed nonfinite calculation."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _host(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_host(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    array = np.asarray(value)
    if array.ndim:
        return _host(array.tolist())
    item = array.item()
    if isinstance(item, float) and not np.isfinite(item):
        return "nan" if np.isnan(item) else ("positive_infinity" if item > 0 else "negative_infinity")
    return item


def _block(jax, tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _integer(value, label):
    import numpy as np
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iu":
        raise ValueError(label + " must be a scalar integer counter")
    return int(raw)


def counts(state):
    """Committed counters only; failed dispatched work is on TrainingStepError."""
    import numpy as np
    micro = state.microstep
    return {
        "critic": _integer(micro.critic_calls, "critic"),
        "actor": _integer(micro.actor_calls, "actor"),
        "actor_target": _integer(micro.actor_target_updates, "actor target"),
        "critic_target": _integer(micro.critic_target_updates, "critic target"),
        "outer_steps": _integer(micro.outer_steps, "outer steps"),
        "phase": _integer(micro.phase, "phase"),
        "auxiliary": _integer(state.aux_fit_accepted, "accepted auxiliary fits"),
        "aux_fit_attempts": _integer(state.aux_fit_attempts, "auxiliary fit attempts"),
        "refresh_attempts": _integer(state.refresh_attempts, "refresh attempts"),
        "refresh_accepted": _integer(state.refresh_accepted, "accepted refreshes"),
        "last_refresh_critic": _integer(state.last_refresh_critic, "last refresh critic"),
        "next_refresh_index": _integer(state.next_refresh_index, "next refresh index"),
        "reference_ready": bool(np.asarray(state.aux.posterior.ready)),
        "critic_budget": _integer(state.critic_budget, "critic budget"),
    }


@lru_cache(maxsize=1)
def _validation_kernels():
    """Cache device-side checks so every call need not copy all network leaves."""
    import jax
    import jax.numpy as jnp

    def finite(tree):
        result = jnp.asarray(True)
        for leaf in jax.tree_util.tree_leaves(tree):
            result = result & jnp.all(jnp.isfinite(leaf))
        return result

    def equal(left, right):
        a, sa = jax.tree_util.tree_flatten(left)
        b, sb = jax.tree_util.tree_flatten(right)
        if sa != sb or len(a) != len(b):
            raise ValueError("Duplicate native view has a different pytree structure")
        result = jnp.asarray(True)
        for x, y in zip(a, b):
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("Duplicate native view has a different leaf shape/dtype")
            result = result & jnp.array_equal(x, y)
        return result

    def state_checks(state):
        micro, aux = state.microstep, state.aux
        native = micro.agent_state
        native_tuple = (native.actor, native.actor_target, native.dual_q, native.dual_q_target)
        view_tuple = (aux.native.actor, aux.native.actor_target, aux.native.critic, aux.native.critic_target)
        ref = aux.posterior
        valid_ref = finite((ref.cal_params, ref.residual_scale, ref.n_eff))
        valid_ref = valid_ref & (ref.residual_scale > 0) & (ref.n_eff >= 0)
        for value in (ref.threshold, ref.lambda_hat, ref.lambda_hpd):
            valid_ref = valid_ref & ~jnp.isnan(value) & (value >= 0)
        valid_ref = valid_ref & (~ref.ready | (ref.threshold == jnp.maximum(ref.lambda_hat, ref.lambda_hpd)))
        return {
            "native_and_aux_finite": finite((micro.rng, native, micro.batch, micro.q_loss_first,
                                               aux.native, aux.calibrator, aux.residual_scale, state.aux_rng)),
            "duplicate_native_equal": equal(native_tuple, view_tuple),
            "reference_storage_valid": valid_ref,
            "ready_reference_finite": finite((ref.threshold, ref.lambda_hat, ref.lambda_hpd)),
            "warmup_reference_shape": (jnp.isposinf(ref.threshold) & jnp.isposinf(ref.lambda_hat)
                                        & jnp.isposinf(ref.lambda_hpd) & (ref.n_eff == 0)
                                        & (ref.residual_scale == 1.0)),
            "positive_residual_unit": aux.residual_scale > 0,
        }

    return jax.jit(state_checks), jax.jit(finite)


def validate(state):
    """Check full continuation state, allowing only intentional unready infinity.

    This validates numerical/state invariants, not dataset provenance or an
    inference-export eligibility claim. Checkpoint bindings cover sources/data.
    """
    import numpy as np
    if type(state) is not TrainingState:
        raise TypeError("Expected complete TrainingState, not network weights alone")
    c = counts(state)
    if any(c[name] < 0 for name in c if name != "reference_ready"):
        raise ValueError("Negative committed counter")
    if c["critic_budget"] not in (6, 20000) or c["critic"] > c["critic_budget"]:
        raise ValueError("Committed critic count exceeds the fixed budget")
    expected_schedule = [2, 4, 6] if c["critic_budget"] == 6 else [10000, 15000, 20000]
    schedule = np.asarray(state.refresh_at)
    if schedule.dtype.kind not in "iu" or schedule.shape != (3,) or schedule.tolist() != expected_schedule:
        raise ValueError("Stored refresh schedule differs from the fixed checkpoint plan")
    actor_count = c["critic"] // 2
    if c["phase"] != c["critic"] % 2 or any(c[name] != actor_count for name in
                                               ("actor", "actor_target", "critic_target", "outer_steps")):
        raise ValueError("Native two-critic phase/actor/target counters disagree")
    if c["auxiliary"] != c["critic"] or c["aux_fit_attempts"] != c["critic"]:
        raise ValueError("Every committed critic must have exactly one accepted auxiliary fit")
    due_count = sum(n <= c["critic"] for n in expected_schedule)
    if any(c[name] != due_count for name in ("refresh_attempts", "refresh_accepted", "next_refresh_index")):
        raise ValueError("Refresh counters disagree with the fixed committed schedule")
    if c["last_refresh_critic"] != (expected_schedule[due_count - 1] if due_count else 0):
        raise ValueError("Last refresh checkpoint is inconsistent")
    native = state.microstep.agent_state
    for name, train_state, expected in (
        ("critic", native.dual_q, c["critic"]), ("actor", native.actor, actor_count),
        ("actor target", native.actor_target, actor_count),
        ("critic target", native.dual_q_target, actor_count),
        ("auxiliary", state.aux.calibrator, c["auxiliary"]),
    ):
        if _integer(train_state.step, name + " TrainState.step") != expected:
            raise ValueError(name + " optimizer/target step differs from committed counters")
    if state.microstep.reference is not None:
        raise ValueError("Kappa-zero checkpoint must keep auxiliary reference out of native actor state")
    batch = state.microstep.batch
    batch_size = 8 if c["critic_budget"] == 6 else 256
    if getattr(batch, "_fields", None) != ("obs", "action", "reward", "next_obs", "done"):
        raise ValueError("Pending native batch must retain the complete Transition schema")
    for field, array, shape in zip(batch._fields, batch,
            ((batch_size, 14), (batch_size, 4), (batch_size,), (batch_size, 14), (batch_size,))):
        if array.shape != shape or array.dtype != np.dtype(np.float32):
            raise ValueError("Pending native batch lost its shape/dtype: " + field)
    if np.asarray(state.aux.residual_scale).shape != ():
        raise ValueError("Current auxiliary residual unit must be scalar")
    for name, key in (("native", state.microstep.rng), ("auxiliary", state.aux_rng)):
        raw = np.asarray(key)
        if raw.shape != (2,) or raw.dtype != np.dtype(np.uint32):
            raise ValueError(name + " RNG must preserve its uint32 key")
    reference = state.aux.posterior
    if np.asarray(reference.ready).shape != () or np.asarray(reference.ready).dtype != np.dtype(bool):
        raise ValueError("Reference ready flag must be a scalar boolean")
    for name in ("residual_scale", "threshold", "lambda_hat", "lambda_hpd", "n_eff"):
        if np.asarray(getattr(reference, name)).shape != ():
            raise ValueError("Reference field must be scalar: " + name)
    flags = _host(_validation_kernels()[0](state))
    if not all(flags[name] for name in ("native_and_aux_finite", "duplicate_native_equal",
                                      "reference_storage_valid", "positive_residual_unit")):
        raise ValueError("Invalid checkpoint numerical state: " + json.dumps(flags, sort_keys=True))
    if due_count == 0:
        if c["reference_ready"] or not flags["warmup_reference_shape"]:
            raise ValueError("Only the declared unready initial reference is allowed during warmup")
    elif not c["reference_ready"] or not flags["ready_reference_finite"]:
        raise ValueError("Refresh must be ready and finite; no infinite-threshold fallback is permitted")
    return {"passed": True, "counts": c, "checks": flags,
            "reference_threshold": _host(reference.threshold),
            "unready_warmup_intentional": due_count == 0}


def _transitions(native, value, label, expected_rows=None):
    import numpy as np
    if not isinstance(value, native.Transition):
        raise TypeError(label + " must use the pinned native Transition type")
    n = len(value.obs)
    if n < 1 or (expected_rows is not None and n != expected_rows):
        raise ValueError(label + " row count is invalid")
    shapes = ((n, 14), (n, 4), (n,), (n, 14), (n,))
    for field, array, shape in zip(value._fields, value, shapes):
        array = np.asarray(array)
        if array.shape != shape or array.dtype != np.dtype(np.float32) or not np.all(np.isfinite(array)):
            raise ValueError(label + "." + field + " must be finite float32 with exact shape " + str(shape))
    if np.any(np.abs(np.asarray(value.action)) > 1) or not np.all(np.isin(np.asarray(value.done), (0.0, 1.0))):
        raise ValueError(label + " action bounds or terminal masks are invalid")
    return native.Transition(*(native.jnp.asarray(x, native.jnp.float32) for x in value))


def create(native, train_transition, heldout_transition, heldout_ids, config):
    """Create native/auxiliary state, independent RNG and fixed eager scheduler.

    The heldout mapping must contain unique row IDs and 128 distinct episode
    IDs. The supervisor's authenticated data loader establishes disjointness
    from the training episodes; this interface is not given training group IDs.
    """
    import numpy as np
    config = _configuration(config)
    receipt = getattr(native, "_source_receipt", {})
    if receipt.get("source_sha256") != NATIVE_SHA256 or receipt.get("passed") is not True:
        raise ValueError("A verified pinned UniFloral computational slice is required")
    loader = _module("native_loader", STAGE1 / "native_loader.py")
    candidate = _module("candidate", STAGE1 / "candidate.py")
    bridge = _module("bca_bridge", STAGE2 / "bca_bridge.py")
    pipeline = bridge.load_pipeline(STAGE2)
    jax, jnp = native.jax, native.jnp
    train = _transitions(native, train_transition, "training")
    heldout = _transitions(native, heldout_transition, "heldout", 128)
    if not isinstance(heldout_ids, dict) or set(heldout_ids) != {"row_ids", "episode_ids"}:
        raise ValueError("Heldout IDs require row_ids and episode_ids provenance")
    ids = {}
    for name in ("row_ids", "episode_ids"):
        array = np.asarray(heldout_ids[name])
        if array.shape != (128,) or array.dtype.kind not in "iu" or np.any(array < 0) or len(np.unique(array)) != 128:
            raise ValueError("Heldout " + name + " must be 128 unique nonnegative integer IDs")
        ids[name] = np.frombuffer(np.asarray(array, dtype=np.int64).tobytes(), dtype=np.int64)
    args = native.Args(seed=config["seed"], dataset="thrusterchain-finite-horizon-v1",
                       num_updates=config["critic_budget"] // 2, gamma=config["gamma"],
                       batch_size=config["batch_size"])
    rng, agent = loader.initialize(native, args, train, config["seed"])
    models, auxiliary, aux_config, aux_args = pipeline.initialize_pipeline(
        agent, train, args, config["aux_seed"])
    functions = candidate.make_microsteps(args, agent.actor.apply_fn, agent.dual_q.apply_fn,
                                         train, width_fn=None, kappa_penalty=0.0)
    micro = functions.initial_state(rng, agent, reference=None)
    zero = jnp.asarray(0, jnp.int32)
    aux_rng = jax.random.fold_in(jax.random.PRNGKey(config["aux_seed"]), AUX_STREAM_DOMAIN)
    state = TrainingState(micro, auxiliary, aux_rng, zero, zero, zero, zero, zero, zero,
                          jnp.asarray(config["refresh_at"], jnp.int32),
                          jnp.asarray(config["critic_budget"], jnp.int32))
    model = agent.actor.apply_fn.__self__
    mean, std = model.obs_mean, model.obs_std
    context = SimpleNamespace(
        native=native, args=args, config=deepcopy(config), pipeline=pipeline, models=models,
        aux_config=aux_config, aux_args=aux_args, train_transition=train,
        heldout_transition=heldout, heldout_ids=ids, obs_mean=mean, obs_std=std,
        normalization={"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist(),
                       "epsilon": 0.001, "source": "Native population statistics of training observations"},
        microstep_fns=functions, microstep_step=jax.jit(functions.step),
        # Exposed only for the bounded native baseline gate. The scheduler below
        # dispatches microstep_step and never calls this two-critic outer step.
        native_step=jax.jit(native.make_train_step(args, agent.actor.apply_fn, agent.dual_q.apply_fn, train)),
        fit_step=jax.jit(lambda aux, batch, key: pipeline.fit_scale(
            aux_args, aux_config, models, aux, batch, key)),
        config_sha256=_digest_config(config), args_snapshot=asdict(args),
        auxiliary_rng_schedule={"initial_seed": config["aux_seed"], "stream_domain": AUX_STREAM_DOMAIN,
                                "split_per_critic": "next, fit, reserved_refresh",
                                "refresh_key_reserved_even_when_not_due": True},
    )
    validate(state)
    return context, state


def native_args(context):
    """Return a copy, so external inspection cannot mutate scheduler arguments."""
    return replace(context.args)


def step(context, state):
    """One atomic committed critic+fit, plus due actor/target/refresh work.

    A failed computation may have dispatched optimizers, but does not replace
    the supplied immutable state. No retry or fail-open fallback is performed.
    Save last_committed_state on TrainingStepError; record attempted_counts.
    KeyboardInterrupt propagates so the supervisor can save its current state.
    """
    import numpy as np
    jax, jnp = context.native.jax, context.native.jnp
    attempted = {"critic": 0, "actor": 0, "auxiliary": 0, "refresh": 0}
    diagnostics = {}
    stage = "preflight"
    try:
        validate(state)
        if _digest_config(_configuration(context.config)) != context.config_sha256 or asdict(context.args) != context.args_snapshot:
            raise ValueError("Context scientific configuration changed after initialization")
        c = counts(state)
        if c["critic_budget"] != context.config["critic_budget"] or np.asarray(state.refresh_at).tolist() != context.config["refresh_at"]:
            raise ValueError("Restored state budget/schedule differs from its bound context")
        if c["critic"] >= c["critic_budget"]:
            raise ValueError("Fixed critic budget is complete; no further dispatch permitted")
        stage = "native_microstep"
        attempted["critic"] = 1
        attempted["actor"] = int(c["phase"] == 1)
        micro, native_metrics = _block(jax, context.microstep_step(state.microstep))
        diagnostics["native"] = _host(native_metrics)
        if not bool(np.asarray(_validation_kernels()[1]((micro.agent_state, native_metrics)))):
            raise ValueError("Native optimizer result or metrics are nonfinite")
        auxiliary = state.aux._replace(native=context.pipeline.native_view(micro.agent_state))
        next_key, fit_key, refresh_key = jax.random.split(state.aux_rng, 3)
        stage = "auxiliary_fit"
        attempted["auxiliary"] = 1
        auxiliary, fit_metrics = _block(jax, context.fit_step(auxiliary, micro.batch, fit_key))
        diagnostics["auxiliary"] = _host(fit_metrics)
        if not bool(np.asarray(fit_metrics["scale_inputs_valid"])) or not bool(np.asarray(fit_metrics["scale_fit_accepted"])):
            raise ValueError("Actual auxiliary fit was rejected; no native-only fallback")
        index = c["next_refresh_index"]
        critic = c["critic"] + 1
        refresh_due = index < len(context.config["refresh_at"]) and critic == context.config["refresh_at"][index]
        refresh_metrics, refresh_diagnostics = None, None
        if refresh_due:
            stage = "calibration_refresh"
            attempted["refresh"] = 1
            auxiliary, refresh_metrics, refresh_diagnostics = context.pipeline.refresh(
                context.aux_args, context.aux_config, context.models, auxiliary,
                context.heldout_transition, refresh_key, heldout_ids=context.heldout_ids["row_ids"])
            _block(jax, auxiliary)
            diagnostics["refresh"] = {"flags": _host(refresh_metrics), "diagnostics": _host(refresh_diagnostics)}
            ref = auxiliary.posterior
            finite_ready = bool(np.asarray(ref.ready)) and all(np.isfinite(float(np.asarray(getattr(ref, name))))
                for name in ("threshold", "lambda_hat", "lambda_hpd", "residual_scale", "n_eff"))
            if (not bool(refresh_metrics["posterior_inputs_valid"])
                    or not bool(refresh_metrics["posterior_certified"])
                    or not bool(refresh_diagnostics["certified"])
                    or not bool(np.asarray(context.pipeline.reference_valid(ref))) or not finite_ready):
                raise ValueError("Refresh was rejected, uncertified, or ready-infinite; checkpoint commit refused")
        one = jnp.asarray(1, jnp.int32)
        refresh_increment = jnp.asarray(int(refresh_due), jnp.int32)
        proposed = TrainingState(
            micro, auxiliary, next_key, state.aux_fit_attempts + one, state.aux_fit_accepted + one,
            state.refresh_attempts + refresh_increment, state.refresh_accepted + refresh_increment,
            state.next_refresh_index + refresh_increment,
            jnp.asarray(critic, jnp.int32) if refresh_due else state.last_refresh_critic,
            state.refresh_at, state.critic_budget,
        )
        stage = "commit_validation"
        report = validate(proposed)
        metrics = {"committed": True, "kappa_penalty": 0.0, "native": diagnostics["native"],
                   "auxiliary": diagnostics["auxiliary"], "refresh_due": refresh_due,
                   "refresh": diagnostics.get("refresh"), "counts": report["counts"]}
        return proposed, metrics
    except Exception as exc:
        raise TrainingStepError(f"{stage}: {type(exc).__name__}: {exc}", last_committed_state=state,
                                stage=stage, attempted_counts=attempted, diagnostics=diagnostics) from exc


__all__ = ["TrainingState", "TrainingStepError", "create", "step", "counts", "validate", "native_args"]
