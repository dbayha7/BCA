"""Verify a complete on-disk mixed-engine checkpoint chain.

This is an isolated draft and is not used by the full-study launcher. GPU
parity, launch integration, and a final end-to-end smoke remain outstanding.
The trusted manifest digest must be obtained from outside the directory being
verified; a mutable manifest and its adjacent sidecar alone do not attest one
another. Verification checks provenance/integrity, not model correctness.
"""
import hashlib
import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path

import study
from candidate_dispatch import dispatcher as dispatch
from thrusterchain.training_protocol import LearnerCheckpoint
from thrusterchain.training_runner import _rng, train_transitions

SCHEMA = "thrusterchain.mixed_study_manifest.v1"


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _published(root, chunk, budget):
    """Find published rounds, failing closed on failure or pending evidence."""
    if root.exists() and not root.is_dir():
        raise RuntimeError("training root is not a directory")
    result = {}
    if root.exists():
        for entry in root.iterdir():
            if entry.name == "failure.json":
                raise RuntimeError("terminal training failure recorded; do not resume or claim success")
            if entry.name.startswith(".pending-"):
                raise RuntimeError("pending training publication requires inspection before retry")
            if entry.name.startswith("round_"):
                if entry.is_symlink() or not entry.is_dir() or not re.fullmatch(r"round_[0-9]{7}", entry.name):
                    raise RuntimeError("invalid published round entry")
                step = int(entry.name[6:])
                if step < chunk or step > budget or step % chunk or step in result:
                    raise RuntimeError("published round outside the frozen budget")
                result[step] = entry
    return result


def _validate_configuration(cp, method, learner_seed, obs_dim, rounds):
    """Check pinned configuration and every optimizer's expected schedule."""
    from thrusterchain.__main__ import HOST_ADAPTERS
    from algorithms import td3_bc, iql, cql, rebrac
    from host_integration.cql_adapter import _args_record
    from host_integration.unifloral_adapter import _args as unified_args

    if cp.adapter_id != HOST_ADAPTERS[method][2]:
        raise RuntimeError("checkpoint adapter ID differs from frozen method")
    info = cp.metadata.get("adapter", {})
    if (not isinstance(info, dict) or info.get("seed") != learner_seed or info.get("obs_dim") != obs_dim
            or info.get("learning_settings") != {}):
        raise RuntimeError("checkpoint initialization identity or learning settings differ")
    if method == "td3bc":
        args = asdict(replace(td3_bc.Args(), seed=learner_seed, dataset="thrusterchain-v6", discount=.98))
        expected = {"critic": rounds, "actor": rounds // args["policy_freq"]}
        primary = "critic"
    elif method == "rebrac":
        args = asdict(replace(rebrac.Args(), seed=learner_seed, dataset="thrusterchain-v6", gamma=.98))
        expected = {"critic": rounds, "actor": (rounds + args["policy_freq"] - 1) // args["policy_freq"]}
        primary = "critic"
    elif method == "iql":
        args = asdict(replace(iql.Args(), seed=learner_seed, dataset="thrusterchain-v6", discount=.98))
        expected = {"critic": rounds, "actor": rounds, "value": rounds}
        primary = "critic"
    elif method == "cql":
        args = _args_record(replace(cql.Args(), seed=learner_seed, dataset="thrusterchain-v6", discount=.98,
                                    bc_steps=0, cql_reward_scale=1., cql_reward_bias=0.))
        expected = {key: rounds for key in ("actor", "critic1", "critic2", "log_alpha", "log_alpha_prime")}
        primary = "critic1"
    elif method == "unifloral-unified":
        args = asdict(unified_args(learner_seed))
        expected = {"critic": rounds, "actor": rounds // 2}
        primary = "actor"
    else:
        raise ValueError("unknown method")
    # Metadata is JSON-canonicalized when checkpoints are created; normalize
    # tuples and other JSON containers identically before comparing arguments.
    args = json.loads(study.encoded(args))
    if info.get("args") != args:
        raise RuntimeError("checkpoint resolved configuration differs from pinned method")
    if cp.primary_optimizer != primary or dict(cp.optimizer_step_counts) != expected:
        raise RuntimeError("checkpoint optimizer counters differ from the pinned update schedule")


def verify(root, expected_manifest_sha256, case_id, method, *, terminal):
    """Verify the full chain for one declared cell and return its progress.

    The mixed manifest must contain `spec`, `sources`, `runtime`,
    `dispatch_binding`, and `chain_verifier_sha256`, in addition to the schema.
    `terminal=True` requires every declared chunk. `terminal=False` allows a
    contiguous prefix, including no published chunks, but never permits holes,
    oversized/partial chunks, extra published rounds, or altered receipts. Any
    failure marker or pending publication requires separate inspection and
    makes this verifier refuse success and resumption. The caller must
    separately enforce the global train/evaluate barrier.
    """
    root = Path(root).resolve()
    if type(terminal) is not bool:
        raise ValueError("terminal must be explicitly true or false")
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("an externally frozen lowercase manifest SHA-256 is required")
    if study.sha(root / "manifest.json") != expected_manifest_sha256:
        raise RuntimeError("manifest differs from the externally frozen digest")
    manifest = study.read_manifest(root)  # also checks the saved sidecar and spec
    if (set(manifest) != {"schema", "spec", "sources", "runtime", "dispatch_binding", "chain_verifier_sha256"}
            or manifest.get("schema") != SCHEMA
            or manifest.get("chain_verifier_sha256") != source_sha256()
            or manifest.get("sources") != study.source_identity()
            or manifest.get("runtime") != study.environment()):
        raise RuntimeError("mixed manifest source or runtime identity mismatch")
    binding = manifest["dispatch_binding"]
    dispatch.require_binding(binding)
    spec = manifest["spec"]
    matches = [case for case in spec["cases"] if case["id"] == case_id]
    if len(matches) != 1 or method not in spec["methods"]:
        raise ValueError("case or method absent from the frozen specification")
    case = matches[0]
    data_receipt_path = root / "datasets" / case_id / "receipt.json"
    if data_receipt_path.is_symlink() or not data_receipt_path.is_file():
        raise RuntimeError("dataset receipt is not an ordinary file")
    data_receipt_hash = study.sha(data_receipt_path)
    dataset = study.data_for(root, case)
    data_receipt = study.read_json(data_receipt_path)
    metadata = dataset.metadata
    provenance_ok = (metadata.get("provenance") == "research" if spec["profile"] == "full"
                     else metadata.get("provenance") in ("smoke", "research"))
    if (set(data_receipt) != {"case", "dataset_sha256", "file_sha256", "train_split_sha256", "episodes", "schema"}
            or data_receipt.get("episodes") != spec["episodes_per_dataset"]
            or data_receipt.get("schema") != metadata.get("schema")
            or metadata.get("behavior", {}).get("name") != case["behavior"]
            or metadata.get("seed") != case["data_seed"]
            or metadata.get("episodes") != spec["episodes_per_dataset"]
            or metadata.get("horizon") != 100 or metadata.get("gamma") != .98
            or metadata.get("task_semantics") != "finite_horizon"
            or metadata.get("observation_mode") != "time_aware"
            or metadata.get("start_family") != "recorded"
            or metadata.get("physics") != {"stall": True, "stall_slope": 1.5}
            or not provenance_ok
            or dataset.split_digest("train") != data_receipt.get("train_split_sha256")):
        raise RuntimeError("dataset metadata or receipt differs from the frozen case and task")
    training_batch = train_transitions(dataset)
    n = len(training_batch)
    if n < 1:
        raise RuntimeError("empty training split")
    sampler = _rng(case["train_seed"], "offline_batches")
    learner_seed = int(_rng(case["train_seed"], "learner_initialization").integers(0, 2**63 - 1))
    budget, chunk, batch = spec["critic_rounds"], spec["chunk_critic_rounds"], spec["batch_size"]
    training_root = root / "training" / case_id / method
    published = _published(training_root, chunk, budget)
    expected_steps = list(range(chunk, budget + 1, chunk))
    actual_steps = sorted(published)
    if actual_steps != expected_steps[:len(actual_steps)]:
        raise RuntimeError("noncontiguous published checkpoint chain")
    if terminal and actual_steps != expected_steps:
        raise RuntimeError("terminal chain does not meet the complete critic-round budget")
    previous = None
    fingerprints = []
    updates = dispatch._steps(method, chunk)
    for step in actual_steps:
        directory = published[step]
        receipt_path, checkpoint_path = directory / "receipt.json", directory / "checkpoint.json"
        if (receipt_path.is_symlink() or checkpoint_path.is_symlink()
                or not receipt_path.is_file() or not checkpoint_path.is_file()):
            raise RuntimeError("published chunk is missing an ordinary receipt or checkpoint")
        receipt_hash, checkpoint_hash = study.sha(receipt_path), study.sha(checkpoint_path)
        receipt = study.read_json(receipt_path)
        cp = LearnerCheckpoint.load(checkpoint_path)
        state = cp.metadata.get("runner_state", {})
        wall = receipt.get("wall_seconds")
        if (receipt.get("schema") != "thrusterchain.study_chunk.v1"
                or receipt.get("case") != case or receipt.get("method") != method
                or receipt.get("critic_rounds") != step or receipt.get("chunk_critic_rounds") != chunk
                or receipt.get("primary_steps_total") != dispatch._steps(method, step)
                or receipt.get("optimizer_counts") != dict(cp.optimizer_step_counts)
                or receipt.get("checkpoint_sha256") != cp.digest
                or receipt.get("checkpoint_file_sha256") != checkpoint_hash
                or type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0
                or state.get("total_runner_updates") != dispatch._steps(method, step)):
            raise RuntimeError("published chunk identity, cost or checkpoint bytes mismatch")
        dispatch._check_checkpoint(cp, method, binding, dataset, case["train_seed"], batch)
        _validate_configuration(cp, method, learner_seed, training_batch.observations.shape[1], step)
        dispatch.verify_published_receipt(receipt.get("native_receipt"), cp, previous, dataset,
                                          method, binding, updates, batch)
        study.verify_counts(method, cp, step)
        # Same RNG call and shape as the runner, independently replayed from
        # the frozen seed; replacement draws need not be unique source records.
        sampler.integers(0, n, size=(updates, batch))
        if state.get("sampler_rng_state") != sampler.bit_generator.state:
            raise RuntimeError("checkpoint PCG64 cursor is not the declared sampling stream")
        fingerprints.append((receipt_path, receipt_hash, checkpoint_path, checkpoint_hash))
        previous = cp
    # Refuse source/runtime changes made while this (potentially long) audit ran.
    if (study.sha(root / "manifest.json") != expected_manifest_sha256
            or study.read_manifest(root) != manifest or study.source_identity() != manifest["sources"]
            or study.environment() != manifest["runtime"] or source_sha256() != manifest["chain_verifier_sha256"]):
        raise RuntimeError("manifest, source or runtime changed during chain verification")
    dispatch.require_binding(binding)
    if (_published(training_root, chunk, budget) != published
            or any(not a.is_file() or not c.is_file() or a.is_symlink() or c.is_symlink()
                   or study.sha(a) != ah or study.sha(c) != ch for a, ah, c, ch in fingerprints)):
        raise RuntimeError("published chain changed during verification")
    if data_receipt_path.is_symlink() or not data_receipt_path.is_file() or study.sha(data_receipt_path) != data_receipt_hash:
        raise RuntimeError("dataset receipt changed during chain verification")
    final_data = study.data_for(root, case)
    if final_data.digest != dataset.digest or final_data.split_digest("train") != dataset.split_digest("train"):
        raise RuntimeError("training data changed during chain verification")
    return {"case": case_id, "method": method, "terminal": terminal,
            "critic_rounds_verified": actual_steps[-1] if actual_steps else 0,
            "chunks_verified": len(actual_steps), "manifest_sha256": expected_manifest_sha256,
            "final_checkpoint_sha256": previous.digest if previous is not None else None}
