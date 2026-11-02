"""Exact terminal restore with role-filtered logged template provisioning.

Adapted from the hash-pinned operator restore. Only path/data provisioning differs;
all native computational modules, full-state canonical checks, normalization and
export equality are retained. No original source or checkpoint is modified.

Read-only exact terminal restore for the frozen operator-alignment audit.

Import and verify() use the standard library only. load() is worker-only: it
recreates the authenticated static template, restores every committed numeric
leaf, and checks the current policy/reference against the existing export.
It never dispatches a learner step, optimizer update, calibration fit or refresh.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "titan_20261026/payload/unifloral_operator_alignment_v1"
BASE = ROOT.parent
ORIGINAL = BASE / "unifloral_checkpoint_v1"
RUN = ORIGINAL / "train_20261023T183248_552877Z"
CHECKPOINT = RUN / "checkpoints/critic_020000_terminal"
STAGE1 = BASE / "unifloral_td3bc_validation_v1"
STAGE2 = BASE / "unifloral_bca_pipeline_validation_v1"
EVALUATOR = BASE / "unifloral_frozen_usefulness_v1_seedfix"
RECOVERY = BASE / "unifloral_export_recovery_v2"
RECOVERED = RECOVERY / "recovery_20261023T214750_256386Z"
EXPORT = RECOVERED / "inference_export"
_DEFAULT_BUNDLE = (Path(r"C:\Users\David Bayha\Downloads\titan_transfer_20261022_v1\titan_transfer_20261022_v1")
                   if os.name == "nt" else Path("/mnt/c/Users/David Bayha/Downloads/titan_transfer_20261022_v1/titan_transfer_20261022_v1"))
# Portable location only: the same file SHA/digest/split checks are mandatory.
BUNDLE = (BASE / "transfer_bundle").resolve()
DATA = BUNDLE / "research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz"
GATE = ORIGINAL / "checks_20261023T180300_619035Z/report.json"
STATE_SHA = "874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8"
COMMIT_SHA = "686b034f2793c2a6e70be5dc72d1595f22b22977916d41bee634f18843114570"
METADATA_SHA = "b2a4b47570974319efbc7548083bda1f3362ce0c325eb8c5443bc9a71c58329b"
EXPORT_SHA = "a54856669e51fd86234c9e9624c62135cd111b0bd7fad07cc1c4fb8e252c899c"
EXPORT_COMMIT_SHA = "c9b58f1c12faeb318cae876b78a725a9ec094fc97a9a1ca1a558817221bf9392"
DATA_SHA = "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226"
DATA_DIGEST = "6b58db68263dff0d7e9b8d6dfa7c1ca5d4127ed9ee6a3b9069add1c565e88331"
NATIVE_REVISION = "f2dc1278eae18ed3e22c92255119c54885414d17"
NATIVE_PROVENANCE_SHA = "7d0c0eccda6730c0b40bf9085cdbc31f565acbd8010db337b01ef5e64cf26a58"
NATIVE_SOURCE_SHA = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"
MANIFESTS = {
    "unifloral_checkpoint_v1": "92b1497f1d5169b985c6ba1cbf497807807a533ef6c5da055e5cb4359d6e8b86",
    "unifloral_td3bc_validation_v1": "1e26f713822020275179845f30c8691467951745d6c7c829ff017b7045904516",
    "unifloral_bca_pipeline_validation_v1": "6eb91c5bcc6c2edddc1ab93c19424307e12a3c8363bf9063fa4bbe2664180478",
    "unifloral_frozen_usefulness_v1": "9219ea2797b14c054c8e7a4787e290ce247c2682ec865a6e1f978b80cc2d9509",
    "unifloral_frozen_usefulness_v1_seedfix": "c21f3ade3dddcb06068168de0a3c509847f4f0d8f4eb6a499d5b3a8b7d851fc2",
    "unifloral_export_recovery_v2": "05caa0b2db6c701fb20199e269c3b30faac04c6d914ff70faaf435e0549478b1",
}
EVIDENCE = {
    "unifloral_checkpoint_v1/train_20261023T183248_552877Z/bindings.json": "35adffbcfbde2bc06cab8cd36776b959d2e9d430f9dc1b6144eff225dea35754",
    "unifloral_checkpoint_v1/train_20261023T183248_552877Z/report.json": "85e99816b10903fbf1976f42ef0681428341487b67b43ca8c3160b6071391cee",
    "unifloral_checkpoint_v1/checks_20261023T180300_619035Z/report.json": "00c0937af8c9997354f4020bea10572ad8287f815a8eedb9dc8a9b3081d0725b",
    "unifloral_checkpoint_v1/checks_20261023T180300_619035Z/checks.json": "ba4f2d6dce0e45d13bbeabf4b05cc32c01ce7c07a1047d28bd37bad337520c10",
    "unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/report.json": "ab5d8b696b01729542c615fa7f178fcd7ea3e793a9bec3d9cbd36a12e6b750dd",
    "unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/inference_parity.json": "7de3014783569d838f7274334f88d0d732a212372330ecc21d31940c46757dcb",
    "unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/RECOVERY_LINK.json": "29941b3fdae2bafd27425fd63dcd7c9f21b638e4dc3142d1b06fdc6aa1f833aa",
}
COUNTS = {"actor": 10000, "actor_target": 10000, "aux_fit_attempts": 20000,
          "auxiliary": 20000, "critic": 20000, "critic_budget": 20000,
          "critic_target": 10000, "last_refresh_critic": 20000,
          "next_refresh_index": 3, "outer_steps": 10000, "phase": 0,
          "reference_ready": True, "refresh_accepted": 3, "refresh_attempts": 3}
CONFIG = {"mode": "production", "gamma": 0.98, "batch_size": 256,
          "seed": 2026102302, "aux_seed": 2026102361,
          "refresh_at": [10000, 15000, 20000], "critic_budget": 20000, "kappa_penalty": 0.0}
ROLES = {"parameters": "parameters.msgpack", "scale_bank": "scale_bank.npz",
         "training_receipt": "training_receipt.json", "normalization_receipt": "normalization_receipt.json",
         "calibration_receipt": "calibration_receipt.json"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON number")
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=floating)


def checked_file(base, relative):
    base = Path(base).resolve()
    rel = Path(relative)
    if rel.is_absolute() or any(part in ("..", "") for part in rel.parts) or ":" in str(relative):
        raise ValueError("Unsafe artifact path")
    path = base / rel
    if path.is_symlink() or not path.resolve().is_relative_to(base) or not path.is_file():
        raise ValueError("Missing or unsafe artifact: " + str(path))
    return path.read_bytes()


def _commit(directory, kind, expected_sha, names):
    directory = Path(directory)
    raw = checked_file(directory, "COMMITTED.json")
    if sha(raw) != expected_sha:
        raise ValueError("Committed artifact marker changed")
    commit = read_json(raw)
    if (commit.get("schema") != "unifloral.committed_artifact.v1" or commit.get("kind") != kind
            or set(commit.get("files", {})) != set(names)
            or {p.name for p in directory.iterdir()} != set(names) | {"COMMITTED.json"}):
        raise ValueError("Committed artifact file set differs")
    hashes = {}
    for name, entry in commit["files"].items():
        value = checked_file(directory, name)
        if entry != {"sha256": sha(value), "bytes": len(value)}:
            raise ValueError("Committed component identity differs: " + name)
        hashes[name] = sha(value)
    return hashes


def _indices(split):
    values = split.get("scale_bank_training_row_indices")
    if (not isinstance(values, list) or len(values) != 256
            or any(type(v) is not int or not 0 <= v < 76800 for v in values)
            or len(set(values)) != 256 or split.get("scale_bank_seed") != 2026102363
            or split.get("train_episodes") != 768 or split.get("train_rows") != 76800
            or split.get("calibration_rows") != 128):
        raise ValueError("Saved fitting-bank role or exact row selection differs")
    return values


def _terminal(metadata, bindings):
    if (metadata.get("schema") != "unifloral.full_training_checkpoint.v1"
            or encoded(metadata.get("counts")) != encoded(COUNTS)
            or encoded(metadata.get("bindings")) != encoded(bindings)
            or encoded(bindings.get("config")) != encoded(CONFIG)
            or metadata.get("state_sha256") != STATE_SHA
            or metadata.get("state_bytes") != 10328701):
        raise ValueError("Not the fixed complete terminal checkpoint")
    if metadata.get("bindings_sha256") != sha(encoded(bindings) + b"\n"):
        raise ValueError("Checkpoint binding digest differs")
    data = bindings.get("datasetidentity", {})
    if data.get("sha256") != DATA_SHA or data.get("digest") != DATA_DIGEST:
        raise ValueError("Checkpoint dataset identity differs")
    _indices(data.get("split_receipt", {}))
    norm = bindings.get("normalization", {})
    for key in ("mean", "std"):
        value = norm.get(key)
        if (not isinstance(value, list) or len(value) != 14
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
                or (key == "std" and any(x < 0 for x in value))):
            raise ValueError("Checkpoint normalization differs")


def _protocol(protocol):
    fixed = {"gamma": 0.98, "horizon": 100, "fitting_rows": 256,
             "checkpoint_manifest_sha256": EXPORT_SHA}
    for key, value in fixed.items():
        if type(protocol.get(key)) is not type(value) or protocol.get(key) != value:
            raise ValueError("Operator checkpoint protocol differs: " + key)
    for key, value in {"full_checkpoint_state_sha256": STATE_SHA, "dataset_sha256": DATA_SHA}.items():
        if key in protocol and protocol[key] != value:
            raise ValueError("Operator checkpoint identity differs: " + key)


def verify(protocol=None):
    """Stdlib-only source/data/commit identities; no outcomes or confirmation reads."""
    if protocol is not None:
        _protocol(protocol)
    sources = {}
    for folder, expected in MANIFESTS.items():
        base = BASE / folder
        raw = checked_file(base, "prepared_files.json")
        if sha(raw) != expected:
            raise ValueError("Pinned manifest changed: " + folder)
        files = read_json(raw).get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Empty source manifest")
        for name, wanted in files.items():
            if sha(checked_file(base, name)) != wanted:
                raise ValueError("Pinned source changed: " + folder + "/" + name)
        sources[folder] = {"manifest_sha256": expected, "verified_files": len(files)}
    raw = checked_file(STAGE1, "provenance/unifloral.json")
    if sha(raw) != NATIVE_PROVENANCE_SHA:
        raise ValueError("Native provenance changed")
    native = read_json(raw)
    files = native.get("source_sha256", {})
    if (native.get("revision") != NATIVE_REVISION or len(files) != 40
            or files.get("algorithms/td3_bc.py") != NATIVE_SOURCE_SHA):
        raise ValueError("Native snapshot declaration differs")
    vendor = STAGE1 / "upstream"
    entries = list(vendor.rglob("*"))
    if any(p.is_symlink() for p in entries) or {p.relative_to(vendor).as_posix() for p in entries if p.is_file()} != set(files):
        raise ValueError("Native snapshot file set differs")
    for relative, wanted in files.items():
        if sha(checked_file(vendor, relative)) != wanted:
            raise ValueError("Native source changed: " + relative)
    evidence = {}
    for relative, wanted in EVIDENCE.items():
        raw = checked_file(BASE, relative)
        if sha(raw) != wanted:
            raise ValueError("Checkpoint/recovery evidence changed: " + relative)
        evidence[relative] = wanted
    committed = _commit(CHECKPOINT, "training_checkpoint", COMMIT_SHA, {"metadata.json", "state.msgpack"})
    if committed != {"metadata.json": METADATA_SHA, "state.msgpack": STATE_SHA}:
        raise ValueError("Terminal checkpoint identity differs")
    bindings = read_json(checked_file(RUN, "bindings.json"))
    metadata = read_json(checked_file(CHECKPOINT, "metadata.json"))
    _terminal(metadata, bindings)
    if bindings["sources"] != {name: sources[name] for name in bindings["sources"]}:
        raise ValueError("Terminal checkpoint source binding differs")
    if sha(DATA.read_bytes()) != DATA_SHA:
        raise ValueError("Pinned training dataset changed")
    export_hashes = _commit(EXPORT, "inference_export", EXPORT_COMMIT_SHA, set(ROLES.values()) | {"eligible_checkpoint.json"})
    if export_hashes["eligible_checkpoint.json"] != EXPORT_SHA:
        raise ValueError("Export identity differs")
    manifest = read_json(checked_file(EXPORT, "eligible_checkpoint.json"))
    if manifest.get("files") != {role: {"path": name, "sha256": export_hashes[name]} for role, name in ROLES.items()}:
        raise ValueError("Export role bindings differ")
    expected = {"gamma": .98, "horizon": 100, "terminal_bootstrap": False, "reward_scale": 1.0,
                "observation_mode": "time_aware_14d", "native_revision": NATIVE_REVISION,
                "native_source_sha256": NATIVE_SOURCE_SHA, "synthetic_fixture": False,
                "policy_kind": "frozen_deterministic_current_actor", "finite_ready_refreshes": 3}
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in expected.items()):
        raise ValueError("Export scientific contract differs")
    link = read_json(checked_file(RECOVERED, "RECOVERY_LINK.json"))
    if (link.get("status") != "pass" or link.get("raw_checkpoint_payload_sha256") != STATE_SHA
            or link.get("export_manifest_sha256") != EXPORT_SHA or link.get("exact_content_match") is not True
            or link.get("numeric_leaves") != 215 or link.get("state_and_counters_unchanged") is not True):
        raise ValueError("Exact recovered-checkpoint/export linkage differs")
    return {"sources": sources, "native_snapshot_files": 40, "native_revision": NATIVE_REVISION,
            "native_manifest_sha256": NATIVE_PROVENANCE_SHA, "evidence": evidence,
            "checkpoint_commit_sha256": COMMIT_SHA, "checkpoint_files": committed,
            "export_commit_sha256": EXPORT_COMMIT_SHA, "export_files": export_hashes,
            "dataset_sha256": DATA_SHA, "dataset_digest": DATA_DIGEST,
            "terminal_counts": COUNTS.copy(), "fitting_rows": 256,
            "fitting_role": "Previously selected recorded training rows; not independent calibration or test rows",
            "fitting_row_indices_sha256": sha(encoded(_indices(bindings["datasetidentity"]["split_receipt"])))}


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        value = sys.modules[name]
        if Path(value.__file__).resolve() != path:
            raise RuntimeError("Conflicting loaded helper: " + name)
        return value
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def load(protocol):
    """Worker-only exact restore and read-only forward bindings; no new RNG draws."""
    identity = verify(protocol)
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import serialization
    original = _module("_operator_original_runner", ORIGINAL / "run.py")
    runtime = original.runtime()
    sources = original.identities()
    gate = original.verify_gate(GATE, sources)
    native_loader = _module("native_loader", STAGE1 / "native_loader.py")
    data = _module("_one_sided_logged_data", HERE / "logged_data.py")
    training = _module("_operator_original_training", ORIGINAL / "training.py")
    saved_io = _module("_operator_original_checkpoint_io", ORIGINAL / "checkpoint_io.py")
    native = native_loader.load_native(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
    train, heldout, heldout_ids, data_info = data.load_training_template(native, DATA)
    context, template = training.create(native, train, heldout, heldout_ids, CONFIG)
    bindings = {"config": CONFIG, "sources": sources,
        "datasetidentity": {"sha256": data_info["dataset_sha256"], "digest": data_info["digest"], "split_receipt": data_info["split_receipt"]},
        "normalization": {"mean": np.asarray(context.obs_mean).tolist(), "std": np.asarray(context.obs_std).tolist()},
        "native_args": asdict(training.native_args(context)), "auxiliary_config": asdict(context.aux_config),
        "auxiliary_rng_schedule": context.auxiliary_rng_schedule, "gate_identity": gate,
        "training_stage_reviewed": True,
        "review_scope": "Explicit user launch of this declared bounded stage after a hash-bound passing correctness gate"}
    if saved_io._json(bindings) != saved_io._json(read_json(checked_file(RUN, "bindings.json"))):
        raise ValueError("Restoration template bindings differ from the committed run")
    state, restored = saved_io.load(CHECKPOINT, template, bindings)
    training.validate(state)
    if encoded(training.counts(state)) != encoded(COUNTS):
        raise ValueError("Restored terminal counters differ")
    inference = _module("_operator_frozen_inference", EVALUATOR / "inference.py")
    policy, scale_bank, inference_receipt = inference.load(EXPORT / "eligible_checkpoint.json", read_json(checked_file(EVALUATOR, "protocol.json")))
    agent = state.microstep.agent_state

    def description(value):
        return saved_io._describe(serialization.to_state_dict(value))

    def exact(label, left, right):
        if description(left) != description(right):
            raise ValueError("Exact checkpoint/export mismatch: " + label)

    exact("actor", agent.actor.params, policy.actor_params)
    exact("critic", agent.dual_q.params, policy.critic_params)
    exact("reference", state.aux.posterior, policy.reference)
    for key in scale_bank:
        exact("fitting_bank_" + key, data_info["scale_bank"][key], scale_bank[key])
    indices = np.asarray(_indices(data_info["split_receipt"]), dtype=np.int64)
    fitting = native.Transition(*(np.asarray(value)[indices].copy() for value in train))
    groups = np.repeat(np.asarray(data_info["train_groups"], np.int64), 100)[indices].copy()
    if not np.array_equal(groups, scale_bank["group_ids"]) or set(groups.tolist()) & set(data_info["calibration_groups"]):
        raise ValueError("Fitting-bank groups do not identify training-only rows")
    for label, observed, wanted in (("observations", fitting.obs, scale_bank["observations"]), ("actions", fitting.action, scale_bank["actions"])):
        if observed.dtype != wanted.dtype or not np.array_equal(observed, wanted):
            raise ValueError("Saved fitting transitions disagree with exported " + label)
    for value in (*fitting, groups, indices, *scale_bank.values()):
        value.setflags(write=False)
    before = sha(serialization.to_bytes(state))
    content = description(state)
    metadata = read_json(checked_file(CHECKPOINT, "metadata.json"))
    if content != metadata["state_leaves"]:
        raise ValueError("Restored full state differs from committed numeric content")
    digests = {name: sha(encoded(description(value))) for name, value in {
        "current_actor": agent.actor.params, "current_critic": agent.dual_q.params,
        "target_actor": agent.actor_target.params, "target_critic": agent.dual_q_target.params,
        "frozen_reference": state.aux.posterior}.items()}
    target_actor = jax.jit(lambda obs: agent.actor_target.apply_fn(agent.actor_target.params, obs).sample(seed=None))
    target_critic = jax.jit(lambda obs, action: agent.dual_q_target.apply_fn(agent.dual_q_target.params, obs, action))

    def observations(value):
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[1] != 14 or value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ValueError("Target model expects finite float32 observations with 14 columns")
        return value

    def target_act(obs):
        obs = observations(obs)
        result = np.asarray(target_actor(jnp.asarray(obs)), dtype=np.float32)
        if result.shape != (len(obs), 4) or not np.all(np.isfinite(result)) or np.any(np.abs(result) > 1):
            raise ValueError("Invalid target actor output")
        return result

    def target_q(obs, actions):
        obs = observations(obs)
        actions = np.asarray(actions)
        if actions.shape != (len(obs), 4) or actions.dtype != np.float32 or not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1):
            raise ValueError("Target critic expects finite deployed float32 actions")
        result = np.asarray(target_critic(jnp.asarray(obs), jnp.asarray(actions)))
        if result.shape != (len(obs), 2) or not np.all(np.isfinite(result)):
            raise ValueError("Invalid target critic output")
        return result

    def assert_unchanged():
        policy.assert_unchanged()
        if sha(serialization.to_bytes(state)) != before or training.counts(state) != COUNTS:
            raise ValueError("Frozen full checkpoint or counters changed")
        if verify(protocol) != identity:
            raise ValueError("Checkpoint/source/data identity changed")
        return {"full_state_unchanged": True, "inference_unchanged": True,
                "full_state_runtime_encoding_sha256": before, "optimizer_updates": 0,
                "calibrator_updates": 0, "calibration_refreshes": 0}

    receipt = {"template_data_reader": {"path": str(HERE / "logged_data.py"), "sha256": sha((HERE / "logged_data.py").read_bytes()), "selection_and_final_calibration_transition_rows_decoded": 0}, "identity": identity, "runtime": runtime, "restore": restored, "inference": inference_receipt,
        "canonical_state_leaves_sha256": sha(encoded(content)), "parameter_tree_sha256": digests,
        "full_state_runtime_encoding_sha256": before, "current_export_parameters_exact": True,
        "frozen_reference_exact": True, "fitting_transitions_exact": True,
        "fitting_rows": 256, "fitting_unique_groups": len(set(groups.tolist())),
        "fitting_role": identity["fitting_role"], "final_refresh_critic": 20000,
        "reference_critic_identity": "Terminal current critic: final refresh follows final native actor/critic/target update and auxiliary fit",
        "snapshot_receipt": {"final_refresh_verified": True, "current_equals_reference": True,
            "checkpoint_created_utc": metadata["created_utc"], "counts": training.counts(state),
            "proof": "Pinned training.step dispatches native microstep, auxiliary fit, then due refresh; terminal export requires last_refresh_critic=critic=20000 and phase=0",
            "training_manifest_sha256": identity["sources"]["unifloral_checkpoint_v1"]["manifest_sha256"],
            "current_critic_tree_sha256": digests["current_critic"],
            "reference_critic_tree_sha256": digests["current_critic"],
            "separate_reference_critic_snapshot_stored": False},
        "separate_reference_critic_snapshot_stored": False,
        "original_refresh_noise_reconstructed": False,
        "smoothing_draw_scope": "Caller supplies new prespecified native-law draws; original refresh draw is not reconstructed",
        "static_template_initialized_then_exactly_restored": True,
        "optimizer_updates": 0, "calibrator_updates": 0, "calibration_refreshes": 0}
    assert_unchanged()
    return SimpleNamespace(context=context, state=state, policy=policy, scale_bank=scale_bank,
        fitting_transition=fitting, fitting_group_ids=groups, fitting_row_indices=indices,
        target_act=target_act, target_q=target_q, q_reference=policy.q,
        receipt=receipt, assert_unchanged=assert_unchanged)
