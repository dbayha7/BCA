"""Episode-preserving collection and missing-region masks for the new task.

There are no simulator imports or calls at import time. Collection/replay are
explicit functions for the user-launched bounded setup, not preparation tests.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

try:
    from . import task as T
except ImportError:
    # File-loaded launchers must not resolve some other experiment's generic
    # module named "task" from sys.modules or an unrelated sys.path entry.
    import importlib.util
    from pathlib import Path
    import sys
    _spec = importlib.util.spec_from_file_location("_two_scenario_data_task_v1", Path(__file__).with_name("task.py"))
    T = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = T
    _spec.loader.exec_module(T)

FIT, CALIBRATION, RESERVED = 0, 1, 2
PARENTS = 1280
ROLE_COUNTS = (768, 256, 256)
ARRAY_FIELDS = ("states", "next_states", "observations", "next_observations", "actions", "rewards", "terminals",
                "episode_ids", "row_ids", "times", "roles", "fragment_ids", "families")


def episode_roles(seed):
    result = np.empty(PARENTS, np.int8)
    # Balance recorder families within each independent parent role before any
    # trajectory or outcome is seen. Each family has 192/64/64 parents.
    for family in range(4):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), family])))
        order = rng.permutation(np.arange(family, PARENTS, 4))
        result[order[:192]], result[order[192:256]], result[order[256:]] = FIT, CALIBRATION, RESERVED
    return result


def retained_mask(arrays, geometry, condition):
    if condition == "full":
        return np.ones(len(arrays["rewards"]), bool)
    names = {"honeypot": "danger", "bottleneck": "connector"}
    if condition not in names:
        raise ValueError("Unknown condition")
    return ~T.edge_intersects(arrays["states"], arrays["next_states"], geometry[names[condition]])


def fragment_ids_for_mask(arrays, mask):
    """Each missing timestep starts a new fragment; original times are retained."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (len(arrays["rewards"]),):
        raise ValueError("Invalid retention mask")
    selected = np.flatnonzero(mask)
    ids = np.empty(len(selected), np.int64)
    previous, fragment = None, -1
    for i, row in enumerate(selected):
        key = (int(arrays["episode_ids"][row]), int(arrays["times"][row]))
        if previous is None or key[0] != previous[0] or key[1] != previous[1] + 1:
            fragment += 1
        ids[i], previous = fragment, key
    return ids


def condition_arrays(arrays, geometry, condition):
    mask = retained_mask(arrays, geometry, condition)
    result = {key: np.array(arrays[key][mask], copy=True) for key in ARRAY_FIELDS}
    result["fragment_ids"] = fragment_ids_for_mask(arrays, mask)
    for value in result.values():
        value.setflags(write=False)
    return result


def validate_arrays(arrays, *, require_all_parents=True):
    if set(arrays) != set(ARRAY_FIELDS):
        raise ValueError("Unexpected or missing transition fields")
    n = len(arrays["rewards"])
    shapes = {"states": (n, 13), "next_states": (n, 13), "observations": (n, 14),
              "next_observations": (n, 14), "actions": (n, 4)}
    for key in ARRAY_FIELDS:
        value = np.asarray(arrays[key])
        if value.shape != shapes.get(key, (n,)) or value.dtype.kind not in "fiub" or not np.all(np.isfinite(value)):
            raise ValueError("Invalid transition field " + key)
    if arrays["states"].dtype != np.float64 or arrays["next_states"].dtype != np.float64:
        raise ValueError("Physical states must preserve float64")
    if any(arrays[key].dtype != np.float32 for key in ("observations", "next_observations", "actions")):
        raise ValueError("Deployment observations/actions must be float32")
    if arrays["terminals"].dtype != bool:
        raise ValueError("Terminal labels must be boolean")
    for key in ("episode_ids", "row_ids", "times", "roles", "fragment_ids", "families"):
        if arrays[key].dtype.kind not in "iu":
            raise ValueError("IDs, roles and times must be integer")
    if len(np.unique(arrays["row_ids"])) != n or np.any(np.abs(arrays["actions"]) > 1):
        raise ValueError("Duplicate rows or out-of-bound actions")
    if np.any((arrays["times"] < 0) | (arrays["times"] >= 100)) or np.any(~np.isin(arrays["roles"], [0, 1, 2])):
        raise ValueError("Invalid time or role")
    for i in range(n):
        if not np.array_equal(arrays["observations"][i], T.observe(arrays["states"][i], arrays["times"][i])):
            raise ValueError("Observation/state/time mismatch")
        if not np.array_equal(arrays["next_observations"][i], T.observe(arrays["next_states"][i], arrays["times"][i] + 1)):
            raise ValueError("Next observation/state/time mismatch")
    parents = np.unique(arrays["episode_ids"])
    if require_all_parents and not np.array_equal(parents, np.arange(PARENTS)):
        raise ValueError("Expected exactly 1280 parent episodes")
    counts = [0, 0, 0]
    for parent in parents:
        idx = np.flatnonzero(arrays["episode_ids"] == parent)
        roles = np.unique(arrays["roles"][idx])
        if len(roles) != 1:
            raise ValueError("A parent episode crosses data roles")
        if not np.all(arrays["families"][idx] == int(parent) % 4):
            raise ValueError("Recorder family differs within or from its frozen parent assignment")
        counts[int(roles[0])] += 1
        if np.any(np.diff(idx) != 1) or np.any(np.diff(arrays["times"][idx]) != 1):
            raise ValueError("Raw parent transitions are not contiguous")
        if np.any(arrays["terminals"][idx[:-1]]) or not arrays["terminals"][idx[-1]]:
            raise ValueError("Raw parent has missing or premature terminal boundary")
        if len(idx) > 1 and not np.array_equal(arrays["next_states"][idx[:-1]], arrays["states"][idx[1:]]):
            raise ValueError("Broken exact state lineage")
    if require_all_parents and tuple(counts) != ROLE_COUNTS:
        raise ValueError("Wrong independent parent split counts")
    if require_all_parents:
        for family in range(4):
            family_roles = [int(arrays["roles"][np.flatnonzero(arrays["episode_ids"] == p)[0]])
                            for p in parents if p % 4 == family]
            if tuple(np.bincount(family_roles, minlength=3)) != (192, 64, 64):
                raise ValueError("Recorder families are not balanced within parent roles")
    return {"rows": n, "parents": len(parents), "role_parent_counts": counts}


def recorder_action(controller, state, time, family, rng, geometry, config):
    """Frozen four-family rule; family 3 recovers one step before family 2.

    This is a prescribed recorder intervention, not an observed safe-action
    claim. The setup must verify its realized safety and retained support.
    """
    if family not in (0, 1, 2, 3):
        raise ValueError("Unknown recorder family")
    action = np.asarray(controller(state, time), np.float64)
    sigma = 0.0 if family == 3 else float(config["wide_action_noise"] if family == 1 else config["action_noise"])
    noise = rng.normal(0.0, sigma, 4)
    end = int(geometry["c_time"]) - (1 if family == 3 else 0)
    if family in (2, 3) and int(geometry["b_time"]) <= time < end:
        action = -action
    return T.deployed_action(action + noise)


def collect(task, controller, geometry, config, on_parent_start=None):
    """One fixed four-family parent bank; all conditions later mask this bank.

    Families: low-noise nominal, broader-noise nominal, a fixed opposite-control
    excursion, and a zero-action-noise excursion recovering one step earlier.
    Parent roles are assigned by a
    separate stream. No policy-trained or evaluator labels guide collection.
    """
    if int(config["parents"]) != PARENTS:
        raise ValueError("Collection is fixed to 1280 independent parents")
    roles, rows = episode_roles(config["split_seed"]), []
    start = T.state_vector(geometry["A"])
    for parent in range(PARENTS):
        if on_parent_start is not None:
            # The callback records an initialization attempt before any draws;
            # it can stop the bounded collection without starting this parent.
            on_parent_start(parent)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(config["collection_seed"]), parent])))
        family = parent % 4
        t = 0
        x = T.state_vector(start)
        x[:10] += rng.uniform(-float(config["physical_start_jitter"]), float(config["physical_start_jitter"]), 10)
        # Full exogenous coordinates are recorded, not hidden context.
        x[10] = rng.uniform(*config["current_range"])
        amp, phase = rng.uniform(*config["wave_amplitude_range"]), rng.uniform(-np.pi, np.pi)
        x[11:] = amp * np.array([np.cos(phase), np.sin(phase)])
        for time in range(t, T.HORIZON):
            action = recorder_action(controller, x, time, family, rng, geometry, config)
            nxt, reward, terminal, _ = task.step(x, time, action)
            rows.append((x.copy(), nxt.copy(), T.observe(x, time), T.observe(nxt, time + 1), action,
                         reward, terminal, parent, len(rows), time, int(roles[parent]), parent, family))
            x = nxt
            if terminal:
                break
    columns = list(zip(*rows))
    dtypes = (np.float64, np.float64, np.float32, np.float32, np.float32, np.float64, bool,
              np.int64, np.int64, np.int32, np.int8, np.int64, np.int8)
    arrays = {key: np.asarray(values, dtype=dtype) for key, values, dtype in zip(ARRAY_FIELDS, columns, dtypes)}
    validate_arrays(arrays)
    return arrays


def exact_replay(arrays, task):
    """One replay of each recorded edge, preserving true state/time and labels."""
    validate_arrays(arrays)
    for i in range(len(arrays["rewards"])):
        nxt, reward, terminal, _ = task.step(arrays["states"][i], int(arrays["times"][i]), arrays["actions"][i])
        if not np.array_equal(nxt, arrays["next_states"][i]) or np.float64(reward).tobytes() != arrays["rewards"][i].tobytes() or terminal != bool(arrays["terminals"][i]):
            raise ValueError("Exact replay mismatch at original row " + str(arrays["row_ids"][i]))
    return {"replayed_rows": len(arrays["rewards"]), "exact": True}


def digest(arrays, metadata):
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save(path, arrays, metadata):
    validate_arrays(arrays)
    envelope = {"metadata": metadata, "content_sha256": digest(arrays, metadata)}
    with open(path, "xb") as output:
        np.savez_compressed(output, **arrays, envelope=np.array(json.dumps(envelope, sort_keys=True, allow_nan=False)))


def load(path, expected_archive_sha256=None):
    raw = open(path, "rb").read()
    if expected_archive_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_archive_sha256:
        raise ValueError("Dataset archive hash mismatch")
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_FIELDS) | {"envelope"}:
            raise ValueError("Dataset archive schema mismatch")
        arrays = {key: np.array(archive[key], copy=True) for key in ARRAY_FIELDS}
        envelope = json.loads(str(archive["envelope"].item()))
    validate_arrays(arrays)
    if digest(arrays, envelope["metadata"]) != envelope["content_sha256"]:
        raise ValueError("Dataset content hash mismatch")
    for key, a in arrays.items():
        arrays[key] = np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
    return arrays, envelope
