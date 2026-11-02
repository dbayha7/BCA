"""Exclusive committed checkpoints; no project/scientific imports at import time.

The caller recreates the static Flax apply/optimizer template through training.create
with the identical bound config, source, data and normalization, then restores every
dynamic field. A hash identifies bytes; it does not attest provenance declarations.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


class CheckpointError(ValueError):
    pass


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _parse(raw):
    def invalid(value):
        raise CheckpointError("Nonfinite JSON constant: " + value)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CheckpointError("Duplicate metadata key: " + key)
            result[key] = value
        return result

    result = json.loads(raw.decode("utf-8"), parse_constant=invalid, object_pairs_hook=pairs)
    _json(result)  # Also rejects float overflow represented as Infinity.
    return result


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bindings(value):
    if not isinstance(value, dict) or not {"config", "sources", "datasetidentity", "normalization"}.issubset(value):
        raise CheckpointError("Bindings must include config, sources, datasetidentity and normalization")
    return _parse(_json(value))


def _atomic_new_file(path, raw):
    """Replace only the empty reservation created by this writer in its new dir."""
    with path.open("xb"):
        pass
    temporary = path.with_name("." + path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish_directory(path, files, kind):
    """New directory, atomic individual files, last-write commit marker.

    A failed write is preserved as an unusable partial directory. Never reuse it.
    Readers accept only an exact committed file set. This is not a guarantee
    against storage hardware failure or an unrelated writer modifying the files.
    """
    path = Path(path).resolve()
    if not path.parent.is_dir():
        raise CheckpointError("Checkpoint parent directory must already exist")
    if (not isinstance(files, dict) or not files or "COMMITTED.json" in files
            or any(Path(name).name != name or name in ("", ".", "..") for name in files)
            or any(not isinstance(raw, bytes) for raw in files.values())):
        raise CheckpointError("Artifact publication requires flat filenames and byte payloads")
    path.mkdir(exist_ok=False)
    entries = {name: {"sha256": _sha(raw), "bytes": len(raw)} for name, raw in files.items()}
    for name, raw in files.items():
        _atomic_new_file(path / name, raw)
    commit = {"schema": "unifloral.committed_artifact.v1", "kind": kind, "files": entries}
    commit_raw = _json(commit)
    _atomic_new_file(path / "COMMITTED.json", commit_raw)
    return {"directory": str(path), "commit_sha256": _sha(commit_raw), "files": entries}


def _read_directory(path, expected_kind):
    path = Path(path).resolve()
    if not path.is_dir():
        raise CheckpointError("Missing checkpoint directory")
    marker = path / "COMMITTED.json"
    if not marker.is_file() or marker.is_symlink():
        raise CheckpointError("Partial checkpoint: no regular commit marker")
    commit_raw = marker.read_bytes()
    commit = _parse(commit_raw)
    if commit.get("schema") != "unifloral.committed_artifact.v1" or commit.get("kind") != expected_kind:
        raise CheckpointError("Unsupported committed artifact")
    entries = commit.get("files")
    if not isinstance(entries, dict) or set(entries) != {"state.msgpack", "metadata.json"}:
        raise CheckpointError("Checkpoint commit has an unexpected file set")
    if {entry.name for entry in path.iterdir()} != set(entries) | {"COMMITTED.json"}:
        raise CheckpointError("Partial or modified checkpoint directory contains missing/extra files")
    files = {}
    for name, expected in entries.items():
        target = path / name
        if not target.is_file() or target.is_symlink():
            raise CheckpointError("Checkpoint component must be a regular file: " + name)
        raw = target.read_bytes()
        if expected != {"sha256": _sha(raw), "bytes": len(raw)}:
            raise CheckpointError("Checkpoint component hash/length mismatch: " + name)
        files[name] = raw
    return files, {"directory": str(path), "commit_sha256": _sha(commit_raw), "files": entries}


def _int(value):
    import numpy as np
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind not in "iu":
        raise CheckpointError("Training counter must be a scalar integer")
    return int(value)


def actual_counts(state):
    import numpy as np
    micro, native = state.microstep, state.microstep.agent_state
    ready = np.asarray(state.aux.posterior.ready)
    if ready.shape != () or ready.dtype.kind != "b":
        raise CheckpointError("Reference ready flag must be a scalar Boolean")
    result = {"critic": _int(micro.critic_calls), "actor": _int(micro.actor_calls),
        "actor_target": _int(micro.actor_target_updates), "critic_target": _int(micro.critic_target_updates),
        "outer_steps": _int(micro.outer_steps), "phase": _int(micro.phase),
        "auxiliary": _int(state.aux.calibrator.step), "aux_fit_attempts": _int(state.aux_fit_attempts),
        "refresh_attempts": _int(state.refresh_attempts), "refresh_accepted": _int(state.refresh_accepted),
        "last_refresh_critic": _int(state.last_refresh_critic), "next_refresh_index": _int(state.next_refresh_index),
        "reference_ready": bool(ready), "critic_budget": _int(state.critic_budget)}
    if (any(value < 0 for value in result.values())
            or result["phase"] != result["critic"] % 2
            or result["actor"] != result["critic"] // 2
            or any(result[key] != result["actor"] for key in ("outer_steps", "actor_target", "critic_target"))
            or _int(native.dual_q.step) != result["critic"] or _int(native.actor.step) != result["actor"]
            or _int(state.aux_fit_accepted) != result["auxiliary"]
            or result["aux_fit_attempts"] < result["auxiliary"]
            or result["refresh_attempts"] < result["refresh_accepted"]):
        raise CheckpointError("Inconsistent training optimizer/phase counters")
    return result


def _normalization(state, bindings):
    import numpy as np
    expected = bindings["normalization"]
    if not isinstance(expected, dict) or set(expected) != {"mean", "std"}:
        raise CheckpointError("Normalization binding requires exact mean/std lists")
    native = state.microstep.agent_state
    for trained in (native.actor, native.actor_target, native.dual_q, native.dual_q_target, state.aux.calibrator):
        model = getattr(trained.apply_fn, "__self__", None)
        if model is None or not hasattr(model, "obs_mean") or not hasattr(model, "obs_std"):
            raise CheckpointError("Static template lacks bound native normalization")
        for name, attr in (("mean", "obs_mean"), ("std", "obs_std")):
            value = np.asarray(getattr(model, attr))
            if (value.shape != (14,) or value.dtype != np.dtype("float32")
                    or not np.all(np.isfinite(value))
                    or not np.array_equal(value.astype(np.float64), np.asarray(expected[name], np.float64))
                    or (name == "std" and np.any(value < 0))):
                raise CheckpointError("Static template normalization differs from binding")


def _describe(value, contents=True):
    import numpy as np
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CheckpointError("Serialized state dictionary keys must be strings")
        return {"dict": [[key, _describe(value[key], contents)] for key in sorted(value)]}
    if isinstance(value, (list, tuple)):
        return {"sequence": [_describe(item, contents) for item in value]}
    if value is None:
        return {"none": True}
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise CheckpointError("Checkpoint state must contain only numeric arrays/scalars and None")
    record = {"dtype": array.dtype.str, "shape": list(array.shape)}
    if contents:
        record["sha256"] = _sha(array.tobytes(order="C"))
    return record


def _template_compatible(value, schema):
    """Fresh TrainState Python-int counters may restore an exact stored int array.

    This exception applies only to literal Python integers in the fresh template,
    never to template arrays or to the saved/restored leaf verification.
    """
    import numpy as np
    if type(value) is int:
        return (isinstance(schema, dict) and schema.get("shape") == []
                and set(schema) == {"dtype", "shape"}
                and np.dtype(schema["dtype"]).kind in "iu")
    if isinstance(value, dict):
        if not isinstance(schema, dict) or set(schema) != {"dict"}:
            return False
        items = schema["dict"]
        return (isinstance(items, list) and [key for key, _ in items] == sorted(value)
                and all(_template_compatible(value[key], child) for key, child in items))
    if isinstance(value, (list, tuple)):
        return (isinstance(schema, dict) and set(schema) == {"sequence"}
                and len(value) == len(schema["sequence"])
                and all(_template_compatible(item, child) for item, child in zip(value, schema["sequence"])))
    return _describe(value, False) == schema


def save(path, state, bindings, counts):
    """Save all dynamic state; refuse an existing destination, even if partial."""
    from flax import serialization
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError("Checkpoint destination already exists: " + str(path))
    bindings = _bindings(bindings)
    _normalization(state, bindings)
    observed = actual_counts(state)
    if _json(counts) != _json(observed):
        raise CheckpointError("Supplied counts do not match the actual saved state")
    state_dict = serialization.to_state_dict(state)
    payload = serialization.to_bytes(state)
    metadata = {"schema": "unifloral.full_training_checkpoint.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "bindings": bindings,
        "bindings_sha256": _sha(_json(bindings)), "counts": observed,
        "state_sha256": _sha(payload), "state_bytes": len(payload),
        "state_schema": _describe(state_dict, False), "state_leaves": _describe(state_dict),
        "state_scope": "Complete microstep+held batch+native/target Adam states+native RNG+auxiliary optimizer/RNG+frozen reference+cadence counters",
        "static_template": "Recreate with unchanged training.create and identical config/source/data/normalization before restoring; static apply functions and optimizer transformations are not serialized"}
    receipt = publish_directory(path, {"state.msgpack": payload, "metadata.json": _json(metadata)}, "training_checkpoint")
    return {**receipt, "counts": observed, "bindings_sha256": metadata["bindings_sha256"], "state_sha256": _sha(payload)}


def load(path, template, expected_bindings):
    """Restore exact numeric state into a freshly recreated matching template."""
    from flax import serialization
    expected_bindings = _bindings(expected_bindings)
    files, receipt = _read_directory(path, "training_checkpoint")
    metadata = _parse(files["metadata.json"])
    if metadata.get("schema") != "unifloral.full_training_checkpoint.v1":
        raise CheckpointError("Unsupported training checkpoint schema")
    if (_json(metadata.get("bindings")) != _json(expected_bindings)
            or metadata.get("bindings_sha256") != _sha(_json(expected_bindings))):
        raise CheckpointError("Checkpoint config/source/data/normalization bindings differ")
    payload = files["state.msgpack"]
    if metadata.get("state_sha256") != _sha(payload) or metadata.get("state_bytes") != len(payload):
        raise CheckpointError("Checkpoint state identity differs from metadata")
    _normalization(template, expected_bindings)
    if not _template_compatible(serialization.to_state_dict(template), metadata.get("state_schema")):
        raise CheckpointError("Restoration template dynamic schema differs")
    restored = serialization.from_bytes(template, payload)
    if _describe(serialization.to_state_dict(restored)) != metadata.get("state_leaves"):
        raise CheckpointError("Restored leaf shape/dtype/content differs from the saved state")
    if _json(actual_counts(restored)) != _json(metadata.get("counts")):
        raise CheckpointError("Restored counters differ from metadata")
    _normalization(restored, expected_bindings)
    return restored, {**receipt, "counts": metadata["counts"],
        "bindings_sha256": metadata["bindings_sha256"], "state_sha256": metadata["state_sha256"],
        "all_numeric_leaves_match": True}
