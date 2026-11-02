"""Hash-checked, unchanged UniFloral computational definitions; no CLI imports.

This explicit import boundary is a harness adaptation, not the upstream script.
It does not replace missing dependencies or validate D4RL / Gym / MuJoCo.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

REVISION = "f2dc1278eae18ed3e22c92255119c54885414d17"
MANIFEST_SHA256 = "7d0c0eccda6730c0b40bf9085cdbc31f565acbd8010db337b01ef5e64cf26a58"
SOURCE_SHA256 = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"
NAMES = {"Args", "AgentTrainState", "Transition", "SoftQNetwork", "DualQNetwork",
         "DeterministicTanhActor", "create_train_state", "make_train_step"}
IMPORTS = {"collections", "dataclasses", "distrax", "flax.linen",
           "flax.training.train_state", "jax", "jax.numpy", "optax"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(vendor_path, manifest_path):
    vendor_path, manifest_path = Path(vendor_path), Path(manifest_path)
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Pinned UniFloral manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION:
        raise ValueError("Pinned UniFloral revision mismatch")
    checked = {}
    for relative, expected in manifest["source_sha256"].items():
        path = (vendor_path / relative).resolve()
        if not path.is_relative_to(vendor_path.resolve()):
            raise ValueError("Manifest path outside vendor snapshot")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Pinned source hash mismatch: {relative}")
        checked[relative] = actual
    actual_files = {p.relative_to(vendor_path).as_posix() for p in vendor_path.rglob("*") if p.is_file()}
    if actual_files != set(checked):
        raise ValueError("Vendor snapshot has missing or additional files")
    return {"passed": True, "revision": REVISION, "manifest_sha256": MANIFEST_SHA256,
            "source_sha256": SOURCE_SHA256, "files": checked}


def selected_tree(source):
    tree = ast.parse(source)
    nodes, found, segments = [], set(), []
    for node in tree.body:
        keep = False
        if isinstance(node, ast.Import):
            keep = all(alias.name in IMPORTS for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            keep = node.module in IMPORTS
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            keep = node.name in NAMES
            if keep:
                found.add(node.name)
        elif isinstance(node, ast.Assign):
            ids = {target.id for target in node.targets if isinstance(target, ast.Name)}
            keep = bool(ids & NAMES)
            if keep:
                found.update(ids & NAMES)
        if keep:
            nodes.append(node)
            segment = ast.get_source_segment(source, node)
            segments.append({"line": node.lineno, "end_line": node.end_lineno,
                             "sha256": hashlib.sha256(segment.encode()).hexdigest()})
    if found != NAMES:
        raise ValueError(f"Unexpected computational definitions: {found ^ NAMES}")
    return ast.Module(body=nodes, type_ignores=[]), segments


def load_native(vendor_path, manifest_path):
    receipt = verify_source(vendor_path, manifest_path)
    source_path = Path(vendor_path) / "algorithms" / "td3_bc.py"
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("Native source bytes changed between verification and compilation")
    source = source_bytes.decode("utf-8")
    tree, segments = selected_tree(source)
    module = types.ModuleType("unifloral_pinned_td3bc_computational_slice")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    # Definitions are compiled without rewriting their AST nodes. This is reached
    # only by the explicitly user-run subprocess after environment checks.
    exec(compile(tree, str(source_path), "exec"), module.__dict__)
    module._source_receipt = {**receipt, "selected_nodes": segments,
        "scope": "Unmodified computational AST nodes; CLI, evaluation and data ingestion excluded",
        "omitted_side_effect": "Upstream XLA_FLAGS assignment; worker sets the same value before imports"}
    return module


def initialize(native, args, dataset, seed):
    """Native main lines 305–324, using fixture shapes instead of Gym spaces.

    Network definitions, native random split, identical target seeds, population
    observation statistics and TrainState creation are retained literally.
    """
    jax, jnp = native.jax, native.jnp
    rng = jax.random.PRNGKey(seed)
    num_actions = dataset.action.shape[-1]
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros(dataset.obs.shape[1:])
    dummy_action = jnp.zeros(num_actions)
    actor_net = native.DeterministicTanhActor(num_actions, obs_mean, obs_std)
    q_net = native.DualQNetwork(obs_mean, obs_std)
    rng, rng_actor, rng_q = jax.random.split(rng, 3)
    agent_state = native.AgentTrainState(
        actor=native.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        actor_target=native.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        dual_q=native.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=native.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
    )
    return rng, agent_state
