"""Isolated mixed-engine dispatcher draft, not connected to the study launcher.

GPU parity and full-study integration are pending.  This module must only be
used by bounded validation until independently reviewed and validated on GPU.
It preserves each engine's actual receipt and rejects foreign resume chains.
"""
import copy
import hashlib
from pathlib import Path

import jax

import study
from candidate_cql.adapter import CachedCQLAdapter, execution_identity
from candidate_td3 import engine as td3
from compiled import engine as scan
from thrusterchain.training_protocol import LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import (STREAMS, _effective_settings, _hash, _identity, _rng,
                                           _with_runner_state, train_offline, train_transitions)

ROUTES = {"td3bc": td3.ENGINE, "cql": "eager-cql-cached-polyak-v1",
          "iql": scan.ENGINE, "rebrac": scan.ENGINE, "unifloral-unified": scan.ENGINE}


def binding():
    """Capture identities for all routes. Freeze this object before a test run."""
    return {"schema": "thrusterchain.candidate_dispatch_binding.v1", "routes": dict(ROUTES),
            "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "study_sources": study.source_identity(), "jax_settings": study.execution_settings(jax),
            "td3": {"source": td3.source_sha256(), "dependencies": td3.dependency_hashes(),
                    "runtime": td3.runtime_signature()},
            "cql": execution_identity(), "scan_source_sha256": scan._engine_sha256()}


def require_binding(frozen):
    if not isinstance(frozen, dict) or frozen != binding():
        raise RuntimeError("frozen dispatcher, source, route or runtime identity changed")


def _steps(method, rounds):
    if method not in ROUTES or type(rounds) is not int or rounds <= 0:
        raise ValueError("invalid method or critic-round budget")
    if method == "unifloral-unified" and rounds % 2:
        raise ValueError("unified UniFloral requires an even critic-round budget")
    return rounds // 2 if method == "unifloral-unified" else rounds


def _check_checkpoint(cp, method, frozen, dataset, seed, batch_size):
    if not isinstance(cp, LearnerCheckpoint):
        raise RuntimeError("expected a learner checkpoint")
    state = cp.metadata.get("runner_state", {})
    expected = {"phase": "offline", "dataset_sha256": dataset.digest,
                "train_split_sha256": dataset.split_digest("train"), "seed": seed,
                "batch_size": batch_size, "learning_settings": {}}
    if any(state.get(k) != v for k, v in expected.items()):
        raise RuntimeError("checkpoint dataset, split, seed, batch or phase mismatch")
    if "sampler_rng_state" not in state or type(state.get("total_runner_updates")) is not int:
        raise RuntimeError("checkpoint sampler or update cursor missing")
    if state["total_runner_updates"] != cp.optimizer_step_counts.get(cp.primary_optimizer):
        raise RuntimeError("checkpoint update cursor disagrees with optimizer")
    adapter_identity = cp.metadata.get("adapter", {}).get("execution_identity")
    route = frozen["routes"][method]
    if method == "cql":
        if (route != "eager-cql-cached-polyak-v1" or adapter_identity != frozen["cql"]
                or state.get("engine") != route
                or state.get("dispatcher_sha256") != frozen["dispatcher_sha256"]
                or state.get("dispatcher_binding_sha256") != _hash(frozen)):
            raise RuntimeError("checkpoint is not the frozen CQL candidate route")
    elif method == "td3bc":
        if (route != td3.ENGINE or state.get("engine") != route
                or state.get("engine_sha256") != frozen["td3"]["source"]
                or state.get("source_dependencies_sha256") != frozen["td3"]["dependencies"]
                or state.get("runtime_signature") != frozen["td3"]["runtime"]
                or adapter_identity is not None):
            raise RuntimeError("checkpoint is not the frozen TD3 candidate route")
    elif (route != scan.ENGINE or state.get("engine") != route
          or state.get("engine_sha256") != frozen["scan_source_sha256"] or adapter_identity is not None):
        raise RuntimeError("checkpoint is not the frozen compiled route")


def _verify_receipt(native, cp, previous, dataset, method, frozen, updates, batch_size):
    raw = dict(native)
    digest = raw.pop("receipt_sha256", None)
    before = dict(previous.optimizer_step_counts) if previous is not None else {}
    counts = dict(cp.optimizer_step_counts)
    deltas = {key: value - before.get(key, 0) for key, value in counts.items()}
    identity = _identity(dataset)
    common = (digest == _hash(raw) and all(native.get(k) == v for k, v in identity.items())
              and native.get("adapter_id") == cp.adapter_id
              and native.get("primary_optimizer") == cp.primary_optimizer
              and native.get("executed_primary_optimizer_updates") == updates
              and native.get("optimizer_step_counts") == counts
              and native.get("optimizer_step_deltas") == deltas
              and all(type(value) is int and value >= 0 for value in deltas.values())
              and cp.optimizer_steps - (previous.optimizer_steps if previous else 0) == sum(deltas.values())
              and native.get("environment_interactions") == 0
              and native.get("checkpoint_sha256") == cp.digest)
    if not common:
        raise RuntimeError("native receipt digest, dataset, counts or checkpoint mismatch")
    predecessor = previous.digest if previous is not None else None
    if method == "cql":
        declared_budget = {"updates": updates, "batch_size": batch_size}
        config = {"budget": declared_budget, "seed": cp.metadata["runner_state"]["seed"],
                  "learning_settings": {}}
        budget_record = native.get("budget", {})
        budget = budget_record.get("executed", {})
        effective = _effective_settings(cp)
        if (native.get("schema") != "thrusterchain.offline_training.v4"
                or native.get("resumed_from_sha256") != predecessor
                or native.get("requested_primary_optimizer_updates") != updates
                or native.get("status") != "trained" or native.get("config") != config
                or native.get("config_sha256") != _hash(config)
                or native.get("effective_learning_settings") != effective
                or native.get("effective_learning_settings_sha256") != _hash(effective)
                or budget_record.get("source") != "explicit_offline_optimizer_budget"
                or budget_record.get("requested") != declared_budget
                or budget_record.get("effective") != declared_budget
                or budget.get("updates") != updates
                or budget.get("primary_optimizer_updates") != updates
                or budget.get("batch_size") != batch_size or budget.get("environment_interactions") != 0
                or budget.get("optimizer_steps_total") != sum(deltas.values())
                or native.get("optimizer_steps_before") != (previous.optimizer_steps if previous else 0)
                or native.get("optimizer_steps_after") != cp.optimizer_steps
                or native.get("executed_optimizer_steps_total") != sum(deltas.values())
                or native.get("sampling", {}).get("bit_generator") != "PCG64"
                or native.get("sampling", {}).get("with_replacement") is not True
                or native.get("sampling", {}).get("stream") != "offline_batches"
                or native.get("sampling", {}).get("stream_key") != STREAMS["offline_batches"]):
            raise RuntimeError("CQL native receipt mismatch")
    elif method == "td3bc":
        if (native.get("schema") != "thrusterchain.candidate_td3_chunk.v1"
                or native.get("engine") != td3.ENGINE or native.get("engine_sha256") != frozen["td3"]["source"]
                or native.get("source_dependencies_sha256") != frozen["td3"]["dependencies"]
                or native.get("runtime_signature") != frozen["td3"]["runtime"]
                or native.get("resume_from_sha256") != predecessor
                or native.get("batch_size") != batch_size or native.get("dataset_draws") != updates * batch_size):
            raise RuntimeError("TD3 candidate receipt mismatch")
    else:
        data = {"dataset_sha256": dataset.digest, "train_split_sha256": dataset.split_digest("train")}
        rounds = updates * (2 if method == "unifloral-unified" else 1)
        study.verify_native_receipt(native, cp, previous, data, method,
                                    {"chunk_critic_rounds": rounds, "batch_size": batch_size})


def _native_cql_checkpoint(cp, frozen):
    """Recover the native runner output exactly, removing dispatcher metadata."""
    state = cp.metadata["runner_state"]
    expected = {"engine": frozen["routes"]["cql"], "dispatcher_sha256": frozen["dispatcher_sha256"],
                "dispatcher_binding_sha256": _hash(frozen)}
    if any(state.get(k) != v for k, v in expected.items()):
        raise RuntimeError("CQL checkpoint dispatcher marker mismatch")
    for key in expected:
        del state[key]
    return _with_runner_state(cp, state)


def _tag_cql_checkpoint(cp, frozen):
    state = cp.metadata["runner_state"]
    if any(k in state for k in ("engine", "dispatcher_sha256", "dispatcher_binding_sha256")):
        raise RuntimeError("native CQL checkpoint has an unexpected dispatcher marker")
    state.update({"engine": frozen["routes"]["cql"], "dispatcher_sha256": frozen["dispatcher_sha256"],
                  "dispatcher_binding_sha256": _hash(frozen)})
    return _with_runner_state(cp, state)


def verify_published_receipt(receipt, cp, previous, dataset, method, frozen, updates, batch_size):
    """Verify the published envelope and its unmodified native receipt."""
    if method != "cql":
        return _verify_receipt(receipt, cp, previous, dataset, method, frozen, updates, batch_size)
    raw = dict(receipt)
    digest = raw.pop("receipt_sha256", None)
    native_cp = _native_cql_checkpoint(cp, frozen)
    if (digest != _hash(raw) or receipt.get("schema") != "thrusterchain.candidate_dispatch_chunk.v1"
            or receipt.get("method") != "cql" or receipt.get("engine") != frozen["routes"]["cql"]
            or receipt.get("dispatcher_sha256") != frozen["dispatcher_sha256"]
            or receipt.get("dispatcher_binding_sha256") != _hash(frozen)
            or receipt.get("checkpoint_sha256") != cp.digest
            or receipt.get("native_checkpoint_sha256") != native_cp.digest
            or receipt.get("resume_from_sha256") != (previous.digest if previous is not None else None)
            or not isinstance(receipt.get("native_receipt"), dict)):
        raise RuntimeError("CQL dispatcher receipt or checkpoint lineage mismatch")
    return _verify_receipt(receipt["native_receipt"], native_cp, previous, dataset, method, frozen, updates, batch_size)


def _verify_previous_envelope(previous, receipt, method, frozen):
    """Check the immediate predecessor is accompanied by its actual receipt.

    Integrators must additionally validate the complete predecessor chain.
    """
    if not isinstance(receipt, dict):
        raise RuntimeError("previous checkpoint requires its published receipt")
    raw = dict(receipt)
    digest = raw.pop("receipt_sha256", None)
    if digest != _hash(raw) or receipt.get("checkpoint_sha256") != previous.digest:
        raise RuntimeError("previous checkpoint receipt digest mismatch")
    if method == "cql":
        native = receipt.get("native_receipt")
        native_cp = _native_cql_checkpoint(previous, frozen)
        if (receipt.get("schema") != "thrusterchain.candidate_dispatch_chunk.v1"
                or receipt.get("method") != "cql" or receipt.get("engine") != frozen["routes"]["cql"]
                or receipt.get("dispatcher_sha256") != frozen["dispatcher_sha256"]
                or receipt.get("dispatcher_binding_sha256") != _hash(frozen)
                or receipt.get("native_checkpoint_sha256") != native_cp.digest or not isinstance(native, dict)):
            raise RuntimeError("previous CQL dispatcher receipt identity mismatch")
        native_raw = dict(native)
        native_digest = native_raw.pop("receipt_sha256", None)
        state = previous.metadata["runner_state"]
        updates = native.get("executed_primary_optimizer_updates")
        deltas = native.get("optimizer_step_deltas")
        if (type(updates) is not int or updates < 1 or not isinstance(deltas, dict)
                or set(deltas) != set(previous.optimizer_step_counts)
                or any(type(v) is not int or v < 0 or v > previous.optimizer_step_counts[k]
                       for k, v in deltas.items())):
            raise RuntimeError("previous CQL optimizer deltas invalid")
        budget = {"updates": updates, "batch_size": state["batch_size"]}
        config = {"budget": budget, "seed": state["seed"], "learning_settings": {}}
        executed = native.get("budget", {}).get("executed", {})
        effective = _effective_settings(previous)
        if (native_digest != _hash(native_raw) or native.get("schema") != "thrusterchain.offline_training.v4"
                or native.get("checkpoint_sha256") != native_cp.digest
                or native.get("dataset_sha256") != state["dataset_sha256"]
                or native.get("split_sha256") != {"train": state["train_split_sha256"]}
                or native.get("preprocessing_source") != "train_only"
                or native.get("learner_visible_split") != "train"
                or native.get("adapter_id") != previous.adapter_id
                or native.get("primary_optimizer") != previous.primary_optimizer
                or native.get("optimizer_step_counts") != dict(previous.optimizer_step_counts)
                or native.get("resumed_from_sha256") != receipt.get("resume_from_sha256")
                or native.get("requested_primary_optimizer_updates") != updates
                or native.get("status") != "trained" or native.get("config") != config
                or native.get("config_sha256") != _hash(config)
                or native.get("effective_learning_settings") != effective
                or native.get("effective_learning_settings_sha256") != _hash(effective)
                or native.get("optimizer_steps_before") != previous.optimizer_steps - sum(deltas.values())
                or native.get("optimizer_steps_after") != previous.optimizer_steps
                or native.get("executed_optimizer_steps_total") != sum(deltas.values())
                or deltas.get(previous.primary_optimizer) != updates
                or native.get("budget", {}).get("requested") != budget
                or native.get("budget", {}).get("effective") != budget
                or native.get("budget", {}).get("source") != "explicit_offline_optimizer_budget"
                or executed.get("updates") != updates or executed.get("primary_optimizer_updates") != updates
                or executed.get("optimizer_steps_total") != sum(deltas.values())
                or executed.get("batch_size") != state["batch_size"]
                or executed.get("environment_interactions") != 0
                or native.get("environment_interactions") != 0
                or native.get("sampling", {}).get("bit_generator") != "PCG64"
                or native.get("sampling", {}).get("with_replacement") is not True
                or native.get("sampling", {}).get("stream") != "offline_batches"
                or native.get("sampling", {}).get("stream_key") != STREAMS["offline_batches"]):
            raise RuntimeError("previous CQL native receipt mismatch")
    else:
        state = previous.metadata["runner_state"]
        expected_schema = ("thrusterchain.candidate_td3_chunk.v1" if method == "td3bc"
                           else "thrusterchain.compiled_chunk.v1")
        expected_engine_sha = frozen["td3"]["source"] if method == "td3bc" else frozen["scan_source_sha256"]
        if (receipt.get("schema") != expected_schema or receipt.get("engine") != frozen["routes"][method]
                or receipt.get("engine_sha256") != expected_engine_sha
                or receipt.get("dataset_sha256") != state["dataset_sha256"]
                or receipt.get("split_sha256") != {"train": state["train_split_sha256"]}
                or receipt.get("optimizer_step_counts") != dict(previous.optimizer_step_counts)
                or receipt.get("adapter_id") != previous.adapter_id
                or receipt.get("primary_optimizer") != previous.primary_optimizer
                or receipt.get("batch_size") != state["batch_size"]
                or receipt.get("environment_interactions") != 0
                or (method != "td3bc" and receipt.get("method") != method)):
            raise RuntimeError("previous receipt engine or checkpoint identity mismatch")


def train_chunk(dataset, method, seed, batch_size, critic_rounds, frozen, previous=None, previous_receipt=None):
    """Run a bounded draft chunk with an engine-specific published receipt.

    This is not an authorized full-study entry point.  Callers must not use
    base/native runners to resume returned checkpoints outside this dispatcher.
    The CQL receipt wraps the native receipt unchanged and binds the marked
    checkpoint; other routes return their native receipt unchanged.
    """
    require_binding(frozen)
    updates = _steps(method, critic_rounds)
    if type(batch_size) is not int or batch_size <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("invalid batch size or seed")
    if previous is not None:
        _check_checkpoint(previous, method, frozen, dataset, seed, batch_size)
        _verify_previous_envelope(previous, previous_receipt, method, frozen)
    elif previous_receipt is not None:
        raise RuntimeError("receipt supplied without a previous checkpoint")
    sampler = _rng(seed, "offline_batches")
    if previous is not None:
        sampler.bit_generator.state = copy.deepcopy(previous.metadata["runner_state"]["sampler_rng_state"])
    sampler.integers(0, len(train_transitions(dataset)), size=(updates, batch_size))
    expected_sampler = sampler.bit_generator.state
    if method == "td3bc":
        cp, native = td3.train_chunk(dataset, seed, batch_size, updates, previous)
    elif method == "cql":
        native_cp, native = train_offline(dataset, CachedCQLAdapter, budget=OfflineBudget(updates, batch_size),
                                          seed=seed, resume_from=previous)
        _verify_receipt(native, native_cp, previous, dataset, method, frozen, updates, batch_size)
        cp = _tag_cql_checkpoint(native_cp, frozen)
        receipt = {"schema": "thrusterchain.candidate_dispatch_chunk.v1", "method": "cql",
                   "engine": frozen["routes"]["cql"], "dispatcher_sha256": frozen["dispatcher_sha256"],
                   "dispatcher_binding_sha256": _hash(frozen), "checkpoint_sha256": cp.digest,
                   "native_checkpoint_sha256": native_cp.digest,
                   "resume_from_sha256": previous.digest if previous is not None else None,
                   "native_receipt": native}
        receipt["receipt_sha256"] = _hash(receipt)
    else:
        cp, native = scan.train_chunk(dataset, method, seed, batch_size, updates, previous)
    require_binding(frozen)
    _check_checkpoint(cp, method, frozen, dataset, seed, batch_size)
    if cp.metadata["runner_state"]["sampler_rng_state"] != expected_sampler:
        raise RuntimeError("checkpoint PCG64 cursor differs from independent expected stream")
    if method != "cql":
        receipt = native
    verify_published_receipt(receipt, cp, previous, dataset, method, frozen, updates, batch_size)
    previous_rounds = 0 if previous is None else previous.optimizer_step_counts[previous.primary_optimizer] * (2 if method == "unifloral-unified" else 1)
    study.verify_counts(method, cp, previous_rounds + critic_rounds)
    return cp, receipt
