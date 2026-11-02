"""Pinned uncertainty-only definitions bound to an existing UniFloral agent.

Importing this file uses only the standard library. Scientific imports happen
inside load_pipeline(), which is for the explicitly user-run validation worker.
Historical host imports, initialization and training updates are never executed.
This is a component-test bridge, not a production training scheduler.
"""
import ast
import builtins
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parent
PROVENANCE_SHA256 = "f4ddaa7f4fdfc38457a44015a1fa6d4c65ff7d9a086067bafed3fab07300b3ff"
NATIVE_DEFINITIONS = ("_real", "Config", "State", "_ids", "native_target", "fit_scale", "refresh")
DIRECT_DEFINITIONS = ("live_width",)
_PIPELINES = {}


class NativeReadOnlyView(NamedTuple):
    """Immutable field aliases; contains the original UniFloral TrainStates."""
    actor: Any
    actor_target: Any
    critic: Any
    critic_target: Any


def native_view(agent):
    return NativeReadOnlyView(agent.actor, agent.actor_target, agent.dual_q, agent.dual_q_target)


@dataclass(frozen=True)
class DeterministicActorView:
    apply_fn: Any

    def apply(self, params, observations):
        # The pinned native actor is a transformed Deterministic distribution.
        # Match the verified native actor-loss readout; no sampling key is used.
        return self.apply_fn(params, observations).sample(seed=None)


@dataclass(frozen=True)
class CriticView:
    apply_fn: Any

    def apply(self, params, observations, actions):
        return self.apply_fn(params, observations, actions)


def _verified_sources(root):
    root = Path(root).resolve()
    provenance_bytes = (root / "uncertainty_provenance.json").read_bytes()
    if hashlib.sha256(provenance_bytes).hexdigest() != PROVENANCE_SHA256:
        raise ValueError("Pinned uncertainty provenance hash mismatch")
    provenance = json.loads(provenance_bytes)
    source_root = (root / "uncertainty_source").resolve()
    raw_sources, checked = {}, {}
    for relative, item in provenance["files"].items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError("Uncertainty source path escapes its root")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item["sha256"] or len(raw) != item["bytes"]:
            raise ValueError("Pinned uncertainty source mismatch: " + relative)
        raw_sources[relative] = raw
        checked[relative] = actual
    actual_files = {p.relative_to(source_root).as_posix()
                    for p in source_root.rglob("*") if p.is_file()}
    if actual_files != set(checked):
        raise ValueError("Uncertainty snapshot has missing or additional files")
    receipt = {"passed": True, "provenance_sha256": PROVENANCE_SHA256,
               "files": checked, "declarations": provenance["declared_fixture_bindings"],
               "selected_nodes": {k: v["selected_nodes"] for k, v in provenance["files"].items()
                                  if "selected_nodes" in v},
               "historical_host_imported": False,
               "scope": "Unchanged uncertainty AST and exact calibration modules; no historical host update"}
    return root, provenance, raw_sources, receipt


def verify_sources(root=None):
    """Read/hash only. Does not import JAX or execute copied scientific code."""
    return _verified_sources(ROOT if root is None else root)[3]


def _canonical_ast(value):
    """Version-stable identities for the AST fields shared by Python 3.12/3.14.

    ast.dump's default empty-list behavior changed in Python 3.13. Keep every
    field, including empty lists and None. Hexadecimal floats avoid depending
    on decimal display conventions; this does not modify the compiled AST.
    """
    if isinstance(value, ast.AST):
        return {"node": type(value).__name__,
                "fields": {name: _canonical_ast(item) for name, item in ast.iter_fields(value)}}
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError("Unsupported canonical AST field: " + type(value).__name__)


def _ast_sha256(node):
    encoded = json.dumps(_canonical_ast(node), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_tree(raw, names, evidence):
    tree = ast.parse(raw.decode("utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(selected) != len(names) or {node.name for node in selected} != set(names):
        raise ValueError("Missing or duplicate uncertainty definition")
    expected = {item["name"]: item["ast_sha256"] for item in evidence}
    for node in selected:
        actual = _ast_sha256(node)
        if actual != expected[node.name]:
            raise ValueError("Uncertainty AST identity mismatch: " + node.name)
    # Original nodes retain their decorators, bodies, line numbers and constants.
    # No imports or other top-level statements from the historical host survive.
    return ast.Module(body=selected, type_ignores=[])


def load_pipeline(root=None):
    root, provenance, raw_sources, receipt = _verified_sources(ROOT if root is None else root)
    cache_key = str(root)
    if cache_key in _PIPELINES:
        return _PIPELINES[cache_key]

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training.train_state import TrainState
    from numbers import Real

    suffix = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    prefix = "_unifloral_bca_uncertainty_" + suffix
    if prefix in sys.modules:
        raise RuntimeError("Private uncertainty namespace already occupied")
    package = types.ModuleType(prefix)
    package.__path__ = [str(root / "uncertainty_source")]
    sys.modules[prefix] = package
    calibration = types.ModuleType(prefix + ".calibration")
    calibration.__path__ = [str(root / "uncertainty_source/calibration")]
    calibration.__package__ = prefix + ".calibration"
    sys.modules[calibration.__name__] = calibration
    package.calibration = calibration
    original_import = builtins.__import__

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Preserve exact copied absolute imports without installing a global
        # 'calibration' alias or changing sys.path. External packages use their
        # own ordinary import machinery.
        if level == 0 and name == "calibration":
            for item in fromlist:
                if item != "*" and not hasattr(calibration, item):
                    raise ImportError("Unloaded private calibration module: " + item)
            return calibration
        if level == 0 and name.startswith("calibration."):
            item = name.split(".", 1)[1]
            module = getattr(calibration, item, None)
            if module is None:
                raise ImportError("Unloaded private calibration module: " + item)
            return module if fromlist else calibration
        if name == "runtime" or name.startswith("runtime.") or name == "algorithms" or name.startswith("algorithms."):
            raise ImportError("Historical host import is prohibited in uncertainty bridge")
        return original_import(name, globals, locals, fromlist, level)

    local_builtins = dict(vars(builtins), __import__=local_import)

    def exact_module(short_name, relative, module=None):
        module = types.ModuleType(prefix + "." + short_name) if module is None else module
        module.__file__ = str(root / "uncertainty_source" / relative)
        module.__package__ = module.__name__.rpartition(".")[0]
        module.__dict__["__builtins__"] = local_builtins
        sys.modules[module.__name__] = module
        exec(compile(raw_sources[relative], module.__file__, "exec"), module.__dict__)
        return module

    exact_module("calibration", "calibration/__init__.py", calibration)
    for short_name in ("wbcp", "bank", "network", "dose", "reference"):
        module = exact_module("calibration." + short_name, "calibration/" + short_name + ".py")
        setattr(calibration, short_name, module)

    network, reference = calibration.network, calibration.reference
    selected = types.ModuleType(prefix + ".selected_uncertainty")
    selected.__file__ = str(root / "uncertainty_source/historical/td3_bc_bca.py")
    selected.__dict__.update(jax=jax, jnp=jnp, np=np, dataclass=dataclass,
                             Real=Real, Any=Any, NamedTuple=NamedTuple,
                             Calibrator=network.Calibrator,
                             bayesian_bootstrap_weights=network.bayesian_bootstrap_weights,
                             soft_coverage=network.soft_coverage,
                             WBCPConfig=reference.WBCPConfig,
                             initial_reference=reference.initial_reference,
                             freeze_reference=reference.freeze_reference,
                             reference_valid=reference.reference_valid,
                             tree_finite=calibration.dose.tree_finite)
    selected.__dict__["__builtins__"] = local_builtins
    sys.modules[selected.__name__] = selected
    for relative, names in (("historical/td3_bc_bca.py", NATIVE_DEFINITIONS),
                            ("historical/direct_extension.py", DIRECT_DEFINITIONS)):
        tree = _selected_tree(raw_sources[relative], names, provenance["files"][relative]["selected_nodes"])
        exec(compile(tree, str(root / "uncertainty_source" / relative), "exec"), selected.__dict__)

    pipeline = types.SimpleNamespace(
        Config=selected.Config, WBCPConfig=reference.WBCPConfig, State=selected.State,
        Calibrator=network.Calibrator, initial_reference=reference.initial_reference,
        fit_scale=selected.fit_scale, refresh=selected.refresh, live_width=selected.live_width,
        native_target=selected.native_target, reference_valid=reference.reference_valid,
        freeze_reference=reference.freeze_reference, positive_scale=reference.positive_scale,
        soft_coverage=network.soft_coverage,
        bayesian_bootstrap_weights=network.bayesian_bootstrap_weights,
        tree_finite=calibration.dose.tree_finite, certifiable=reference.certifiable,
        calibration=calibration, receipt=receipt, native_view=native_view,
        jax=jax, jnp=jnp, np=np, optax=optax, TrainState=TrainState)
    pipeline.initialize_pipeline = lambda native_agent, dataset, native_args, seed: _initialize(
        pipeline, native_agent, dataset, native_args, seed)
    _PIPELINES[cache_key] = pipeline
    return pipeline


def _initialize(pipeline, native_agent, dataset, native_args, seed):
    """Only the calibrator is initialized; every native TrainState is reused.

    Observation binding is explicitly different from the old ThrusterChain
    adapter: all raw observation coordinates use the native normalization once.
    Production warmup/refresh cadence and strict-infinite policy are absent.
    """
    jax, jnp, np = pipeline.jax, pipeline.jnp, pipeline.np
    if dataset.obs.ndim != 2 or dataset.action.ndim != 2 or len(dataset.obs) == 0:
        raise ValueError("Expected nonempty raw observation/action matrices")
    if len(dataset.action) != len(dataset.obs):
        raise ValueError("Observation/action row counts differ")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Auxiliary initialization seed must be a uint32-range integer")
    actor_model = getattr(native_agent.actor.apply_fn, "__self__", None)
    if type(actor_model).__name__ != "DeterministicTanhActor":
        raise TypeError("Expected the pinned UniFloral DeterministicTanhActor bound apply method")
    obs_mean, obs_std = actor_model.obs_mean, actor_model.obs_std
    expected_mean = dataset.obs.mean(axis=0)
    expected_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    for state in (native_agent.actor, native_agent.actor_target, native_agent.dual_q, native_agent.dual_q_target):
        model = getattr(state.apply_fn, "__self__", None)
        if model is None or not hasattr(model, "obs_mean") or not hasattr(model, "obs_std"):
            raise TypeError("Native apply methods must carry their observation statistics")
        if not np.array_equal(np.asarray(model.obs_mean), np.asarray(expected_mean)) or not np.array_equal(
                np.asarray(model.obs_std), np.asarray(expected_std)):
            raise ValueError("Native observation statistics differ from the supplied fixture")
    if not np.all(np.isfinite(np.asarray(obs_mean))) or not np.all(np.isfinite(np.asarray(obs_std))):
        raise ValueError("Native observation statistics must be finite")

    config = pipeline.Config(arm="bca", posterior=pipeline.WBCPConfig(), blend=1.0)
    aux_args = types.SimpleNamespace(discount=float(native_args.gamma),
                                    policy_noise=float(native_args.policy_noise),
                                    noise_clip=float(native_args.noise_clip))
    if not 0 <= aux_args.discount <= 1 or not np.isfinite(aux_args.discount):
        raise ValueError("Invalid native discount")
    if any(not np.isfinite(x) or x < 0 for x in (aux_args.policy_noise, aux_args.noise_clip)):
        raise ValueError("Invalid native target-noise settings")
    cal = pipeline.Calibrator(obs_mean, obs_std, state_dep=True)
    obs = jnp.zeros((1, dataset.obs.shape[1]), dtype=dataset.obs.dtype)
    action = jnp.zeros((1, dataset.action.shape[1]), dtype=dataset.action.dtype)
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 1128352841)
    cal_state = pipeline.TrainState.create(
        apply_fn=cal.apply, params=cal.init(key, obs, action), tx=pipeline.optax.adam(config.cal_lr))
    state = pipeline.State(native_view(native_agent), cal_state, jnp.asarray(1.0, jnp.float32),
                           pipeline.initial_reference(cal_state.params))
    models = (DeterministicActorView(native_agent.actor.apply_fn),
              CriticView(native_agent.dual_q.apply_fn), cal)
    return models, state, config, aux_args


def initialize_pipeline(native_agent, dataset, native_args, seed):
    """Convenience entry point using sources beside this bridge file."""
    return load_pipeline().initialize_pipeline(native_agent, dataset, native_args, seed)
