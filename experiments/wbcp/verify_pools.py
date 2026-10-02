"""Verify the frozen WBCP score pools: score file, checkpoint, D4RL file, withheld split and preparation hashes.

Step 1c of the signal study (runs/wbcp_signal/expectations.md). Step 2 reuses the TD3+BC
checkpoints as frozen critics, and every cross-host comparison assumes that each host withheld
the same rows of the same file, so before either runs, every pool written by freeze_scores.py
(TD3+BC), freeze_cql.py, freeze_rebrac.py or freeze_iql.py is checked against what it records.
Nothing is trained or scored; the script only reads files.

Per pool (a directory with frozen.json and frozen.npz):
1. npz: the SHA256 of frozen.npz equals frozen.json npz_sha256 (every builder writes it through
   freeze_scores.write_artifact), and the arrays pass freeze_scores.check_contract.
2. checkpoint: checkpoint_<step>.msgpack exists, and its SHA256 and size equal frozen.json
   checkpoint.sha256 and size_bytes (the field every builder's write_checkpoint records).
3. data: the cached D4RL file (<data-dir>/<dataset_file.filename>) hashes to the SHA256 that
   frozen.json records (dataset_file.sha256, declared_sha256, dataset_sha256), that
   configs/<algorithm>.yaml declares for the dataset, and that resolved.json's cache declares.
4. split: the withheld population is re-derived from the file with freeze_scores.py's rule.
   The converted rows are runtime.td3_bc._convert's map with raw timeouts (every raw row but the
   last, minus timeout rows: freeze_scores.converted_rows, freeze_iql.dependency_maps), their
   effective components are runtime.td3_bc.dependency_maps, the target is
   freeze_scores.split_row's, and the selection is runtime.td3_bc._reserve on blank transitions
   with the recorded terminal mask, exactly as runtime.td3_bc.validate_prepared repeats it
   (calibration.reference.reserve_calibration with rows_per_episode=None and the recorded
   train_fraction and split seed). frozen.npz row must equal the withheld rows, in_training
   must be false throughout, the training rows must be the complement, episode and timestep
   must be the component labels (freeze_scores.components), and frozen.json's row counts and
   target must agree. The population SHA256 is sha256 of the sorted rows as little-endian
   int64, reserve_calibration's withheld_indices_sha256 rule, so it is comparable across hosts
   and equals what CQL and ReBRAC record.
5. derived: hashes and statistics in frozen.json re-derived without training. From the file
   and the split: the training/held-out ID hashes (preparation.training_ids_sha256,
   run_input_hashes.training_ids / heldout_ids, the reservation digests, IQL's index and
   episode fingerprints), the raw-array digest raw_dataset_sha256 (train.prepare's five-key
   read), IQL's raw terminal/timeout and converted-data fingerprints (prepare_dataset's
   calibration.iql_reference.fingerprint of d4rl.qlearning_dataset's arrays, which are the raw
   rows of the timeout map, checked by freeze_iql.dependency_maps), the training-only
   observation mean/std (runtime.common.compute_mean_std, as every host fits it) with their
   run_input_hashes, and frozen.npz obs and action, which must be the population's dataset
   rows under that normalization. ReBRAC's calibrator statistics
   (run_input_hashes.cal_obs_mean / cal_obs_std) are jnp reductions on the preparing device,
   so their recorded hashes are compared with the statistics the checkpoint stores. Hashes of
   the prepared pools (reward handling, transforms, environment constants) are 'not
   re-derivable without re-preparation'; with --reprepare the host builder's own prepare()
   (freeze_scores.prepare, freeze_cql.prepare, freeze_rebrac.prepare, freeze_iql.prepare; no
   training) re-prepares resolved.json with the current code and compares those too, so a
   mismatch there means the current code would not feed a restored checkpoint the inputs it
   was trained on (ReBRAC's calibrator statistics: within 1e-5 of the checkpoint's cal_obs_std,
   since a CPU re-preparation of a GPU preparation differs in the last float32 bits).
   metadata_sha256, settings_sha256 and source_files_sha256 depend on the code version and
   are not compared.
Also reported: run_record.json (device, exit code; a nonzero exit fails), a failure.json in
the pool (fails), and IQL pools that scored the online Q heads, flagged 'superseded
(pre-qf_target fix)': they predate a221438/fdba570 and will be retrained, but are verified.
Across pools: the short-train pools of one dataset (frozen.json dataset, the environment) must
declare one dataset file (dataset_file filename and SHA256) and hold one population; either
difference fails, even when every pool passes against its own host's config, because the study
assumes identical data across hosts. Checkpoint-mode pools (freeze_scores.py --checkpoint, only
under --all) score all rows or a run's thinned bank by design and are left out of that
comparison and of check 4.

Pools are found under --root (default runs/wbcp_frozen); names containing smoke, gputest or
-u2000 are skipped unless --all, and a directory without frozen.json (a pool still being
written) is skipped. Pool directories may also be given explicitly. A run that verifies no pool
at all (an empty --root, a mistyped path, only unfinished pools) fails. Pools are processed one
dataset file at a time, so at most one file's arrays are held. Writes <output>/verification.json
and <output>/verification.md, prints the table, and exits 1 if any check fails.

JAX_PLATFORMS=cpu python experiments/wbcp/verify_pools.py --output runs/wbcp_signal/verify_pools
JAX_PLATFORMS=cpu python experiments/wbcp/verify_pools.py runs/wbcp_frozen/cql-* --reprepare --output OUT
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name, _value in {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "WANDB_MODE": "disabled",
                      "D4RL_SUPPRESS_IMPORT_ERROR": "1", "MUJOCO_GL": "egl"}.items():
    os.environ.setdefault(_name, _value)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import runtime.td3_bc as R  # noqa: E402
from calibration.reference import qlearning_episode_ids  # noqa: E402
from experiments.wbcp import freeze_scores as F  # noqa: E402
from runtime.config import read_config  # noqa: E402
from runtime.provenance import sha  # noqa: E402

SCHEMA = "wbcp-pool-verification-v1"
DEFAULT_ROOT = ROOT / "runs" / "wbcp_frozen"
DEFAULT_DATA = F.DEFAULT_DATA
CONFIG_DIR = ROOT / "configs"
SKIP = ("smoke", "gputest", "-u2000")
HOSTS = {"td3_bc": "TD3+BC", "cql": "CQL", "rebrac": "ReBRAC", "iql": "IQL"}
RAW_KEYS = ("observations", "actions", "rewards", "terminals", "timeouts")  # train.prepare's read for TD3+BC/CQL/ReBRAC
CODE_DEPENDENT = ("preparation.metadata_sha256", "preparation.settings_sha256", "preparation.source_files_sha256")
RESERVATION_DIGESTS = ("calibration_indices_sha256", "withheld_indices_sha256")
RESERVATION_RECORD = {"cql": "preparation.reservation",  # where each host keeps reserve_calibration's metadata
                      "rebrac": "preparation.reservation_record.inherited_primitive_metadata"}
NOT_DERIVABLE = "not re-derivable without re-preparation"
# ReBRAC's calibrator statistics are jnp.mean / jnp.std over the training observations
# (runtime.rebrac.prepare), reduced on the preparing device: a GPU preparation and a CPU one
# differ in the last float32 bits (11 ulps on pen-human). The recorded hash is compared exactly
# with the statistics the checkpoint stores; a re-preparation is compared numerically.
DEVICE_DEPENDENT = {"preparation.run_input_hashes.cal_obs_mean": "cal_obs_mean",
                    "preparation.run_input_hashes.cal_obs_std": "cal_obs_std"}
DEVICE_TOLERANCE = 1e-5  # largest |re-prepared - checkpoint| in units of the checkpoint's cal_obs_std
SUPERSEDED = ("superseded (pre-qf_target fix): scored and scale-fitted on the online Q heads, not the Polyak target "
              "heads IQL's actor reads (a221438, fdba570); to be retrained")


def discover(root, everything=False):
    """(pools, skipped) under root: pool directories, and (directory, reason) for the ones left out."""
    pools, skipped = [], []
    for directory in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if not everything and any(token in directory.name for token in SKIP):
            skipped.append((directory, "smoke/gputest/-u2000 run (use --all)"))
        elif not (directory / "frozen.json").is_file():
            skipped.append((directory, "frozen.json missing (pool still being written?)"))
        else:
            pools.append(directory)
    return pools, skipped


def host_of(meta):
    """The host key: freeze_scores.py (TD3+BC) writes no host field, the other builders do."""
    return meta.get("host", meta["algorithm"])


def superseded(meta):
    """IQL pools scored before the qf_target fix say so in their own score definition."""
    q = meta.get("score_definition", {}).get("q", "")
    return SUPERSEDED if host_of(meta) == "iql" and "target" not in q else None


def population_sha256(rows):
    """sha256 of sorted row indices as little-endian int64: reserve_calibration's withheld_indices_sha256 rule."""
    return hashlib.sha256(np.sort(np.asarray(rows, np.int64)).astype("<i8").tobytes()).hexdigest()


def _resolved(directory):
    path = Path(directory) / "resolved.json"
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else None


def _flatten(prefix, value, out):
    """Dotted leaves of nested dicts (lists stay leaves)."""
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}", item, out)
    elif value is not None:
        out[prefix] = value
    return out


def _show(value):
    """Scalars as they are; lists and dicts as a short digest, so the report stays small."""
    return "digest " + R._digest(value)[:16] if isinstance(value, (list, dict)) else value


def _load_arrays(directory):
    with np.load(Path(directory) / "frozen.npz") as npz:
        return {name: npz[name] for name in npz.files}


# 1-3: file identities -------------------------------------------------------------------------

def check_npz(directory, meta, arrays):
    recorded, path = meta.get("npz_sha256"), Path(directory) / "frozen.npz"
    actual = sha(path) if path.is_file() else None
    result = dict(ok=actual is not None and actual == recorded, recorded=recorded, sha256=actual, contract=None)
    if arrays is None:
        result.update(ok=False, contract="frozen.npz unreadable")
        return result
    try:
        F.check_contract(arrays)
        result["contract"] = "ok"
    except (ValueError, FloatingPointError) as error:
        result.update(ok=False, contract=str(error))
    return result


def check_checkpoint(directory, meta):
    entry = meta.get("checkpoint") or {}
    if "sha256" not in entry:
        return dict(ok=False, error="frozen.json records no checkpoint sha256")
    name = Path(entry["path"]).name
    path = next((p for p in (Path(directory) / name, Path(entry["path"])) if p.is_file()), None)
    found = sorted(p.name for p in Path(directory).glob("checkpoint_*.msgpack"))
    result = dict(step=entry.get("step"), path=None if path is None else str(path), recorded=entry["sha256"],
                  recorded_size_bytes=entry.get("size_bytes"), checkpoints_in_directory=found, sha256=None)
    if path is None:
        return dict(result, ok=False, error="checkpoint file missing")
    result.update(sha256=sha(path), size_bytes=path.stat().st_size)
    recorded = [entry["sha256"]] + ([entry["recorded_sha256"]] if entry.get("recorded_sha256") else [])
    result["ok"] = (all(result["sha256"] == r for r in recorded) and name == f"checkpoint_{entry.get('step')}.msgpack"
                    and entry.get("size_bytes") in (None, result["size_bytes"]))
    return result


def declared_cache(meta, config_dir):
    """configs/<algorithm>.yaml's cache entry for the pool's environment (the builders' CONFIG files)."""
    config = read_config(Path(config_dir) / f"{meta['algorithm']}.yaml")
    entries = [d["cache"] for d in config["datasets"].values() if d.get("environment") == meta.get("dataset")]
    return entries[0] if len(entries) == 1 else None


def check_data(directory, meta, data_dir, config_dir, file_sha):
    record = meta.get("dataset_file") or {}
    path = F._data_dir(data_dir) / record.get("filename", "")
    cache = declared_cache(meta, config_dir)
    resolved = (_resolved(directory) or {}).get("cache") or {}
    actual = file_sha(path)
    declared = {"frozen.json dataset_file.sha256": record.get("sha256"),
                "frozen.json dataset_file.declared_sha256": record.get("declared_sha256"),
                "frozen.json dataset_sha256": meta.get("dataset_sha256"),
                f"configs/{meta['algorithm']}.yaml cache.sha256": None if cache is None else cache["sha256"],
                "resolved.json cache.sha256": resolved.get("sha256")}
    mismatches = [k for k, v in declared.items() if v is not None and v != actual]
    if cache is None:
        mismatches.append(f"dataset {meta.get('dataset')} not declared once in configs/{meta['algorithm']}.yaml")
    elif cache.get("filename") != record.get("filename"):
        mismatches.append("configs filename differs from frozen.json dataset_file.filename")
    if resolved and resolved.get("filename") != record.get("filename"):
        mismatches.append("resolved.json cache.filename differs from frozen.json dataset_file.filename")
    size = path.stat().st_size if actual is not None else None
    if record.get("size_bytes") not in (None, size):
        mismatches.append("size_bytes")
    if actual is None:
        mismatches.insert(0, "file missing: " + str(path))
    return dict(ok=not mismatches, path=str(path), sha256=actual, size_bytes=size, declared=declared,
                mismatches=mismatches)


# 4: the split ----------------------------------------------------------------------------------

def load_dataset(path, train_fraction, split_seed):
    """Everything the split and derived checks need from one cached file, read once.

    The selection is runtime.td3_bc._reserve on blank transitions with the recorded terminal
    mask, the call runtime.td3_bc.validate_prepared repeats: with explicit component IDs the
    selection reads only the component boundaries. Raw rewards and terminals are kept as
    d4rl's OfflineEnv.get_dataset returns them (an (N, 1) column flattened to (N,)), the arrays
    IQL fingerprints; raw_dataset_sha256 digests the file's arrays unchanged.
    """
    import h5py

    path = Path(path)
    converted = F.converted_rows({"cache": {"filename": path.name}}, path.parent)
    with h5py.File(path, "r") as f:
        raw = {k: f[k][()] for k in RAW_KEYS}
    flat = {k: raw[k][:, 0] if raw[k].shape == (len(raw["observations"]), 1) else raw[k]
            for k in ("rewards", "terminals")}  # get_dataset's reshape
    rows = np.flatnonzero(~np.asarray(raw["timeouts"][:-1], bool))  # _convert's map when timeouts exist
    if len(rows) != converted:
        raise ValueError("timeout row map differs from freeze_scores.converted_rows")
    maps = R.dependency_maps(rows, np.asarray(flat["terminals"])[rows])
    reservation = F.split_row({"protocol": {}}, converted, train_fraction, split_seed)["protocol"]["reservation"]
    blank = np.zeros((converted, 1), np.float32)
    done = np.asarray(maps["terminal_mask"], np.float32)
    training, withheld, heldout, record = R._reserve(maps, R.Reservation(**reservation),
                                                      R.C.Transition(blank, blank, blank[:, 0], blank, done))
    component, timestep = F.components(SimpleNamespace(metadata={"dependency_maps": maps}))
    obs = np.asarray(raw["observations"], np.float32)
    mean, std = R.C.compute_mean_std(obs[rows[training]], eps=0.001)
    return SimpleNamespace(
        path=path, converted=converted, rows=rows, component=component, timestep=timestep, reservation=reservation,
        training=training, withheld=withheld, heldout=heldout, record=record, mean=mean, std=std, obs=obs,
        actions=np.asarray(raw["actions"], np.float32), raw_rewards=np.asarray(flat["rewards"]),
        raw_terminals=np.asarray(flat["terminals"]), raw_timeouts=np.asarray(raw["timeouts"]),
        raw_dataset_sha256=R._digest({k: R._array_hash(raw[k]) for k in sorted(raw)}))


def split_parameters(meta):
    split = meta.get("split") or {}
    return split.get("train_fraction"), split.get("split_seed")


def check_split(meta, arrays, ds):
    rows, n = np.asarray(arrays["row"], np.int64), ds.converted
    inside = bool(len(rows) and rows.min() >= 0 and rows.max() < n)
    complement = np.setdiff1d(np.arange(n), rows)
    recorded_rows = meta.get("rows") or {}
    target = ds.reservation["target_size"]
    recorded_target = (meta.get("split") or {}).get("target_size",
                                                    (meta.get("preparation") or {}).get("reservation", {}).get("target_size"))
    checks = {
        "row equals the re-derived withheld rows": np.array_equal(rows, ds.withheld.astype(np.int64)),
        "population is every withheld row": np.array_equal(ds.heldout, ds.withheld),
        "in_training false throughout": not np.any(arrays["in_training"]),
        "training rows are the complement": np.array_equal(complement, ds.training.astype(np.int64)),
        "episode is the effective component": inside and np.array_equal(arrays["episode"], ds.component[rows]),
        "timestep counts rows within the component": inside and np.array_equal(arrays["timestep"], ds.timestep[rows]),
        "recorded target": recorded_target in (None, target),
        "rows.converted": recorded_rows.get("converted") == n,
        "rows.training": recorded_rows.get("training") == len(ds.training),
        "rows.population": recorded_rows.get("population") == len(rows),
        "rows.withheld": recorded_rows.get("withheld") in (None, len(ds.withheld)),
        "rows.converted_components": recorded_rows.get("converted_components") in (None, int(ds.component[-1]) + 1),
        "rows.population_episodes": recorded_rows.get("population_episodes") in (None, len(np.unique(arrays["episode"]))),
    }
    failed = [name for name, ok in checks.items() if not ok]
    train_fraction, split_seed = split_parameters(meta)
    return dict(ok=not failed, failed=failed, train_fraction=train_fraction, split_seed=split_seed, target_size=target,
                recorded_target_size=recorded_target, converted=n, training=len(ds.training), population=len(rows),
                population_sha256=population_sha256(rows), derived_population_sha256=population_sha256(ds.withheld))


# 5: derived hashes -----------------------------------------------------------------------------

def recorded_fields(meta):
    """The hash and statistics fields of frozen.json that check 5 classifies, by dotted path."""
    host, p, fields = host_of(meta), meta.get("preparation") or {}, {}
    _flatten("preparation.run_input_hashes", p.get("run_input_hashes"), fields)
    fields.update({f"preparation.{k}": v for k, v in p.items() if k.endswith("_sha256")})
    if host in RESERVATION_RECORD:
        record = _flatten(RESERVATION_RECORD[host], p.get("reservation_record", {}).get("inherited_primitive_metadata")
                          if host == "rebrac" else p.get("reservation"), {})
        fields.update({k: v for k, v in record.items() if k.rsplit(".", 1)[-1] in RESERVATION_DIGESTS})
    if host == "iql":
        m = p.get("metadata") or {}
        fields.update({f"preparation.metadata.{k}": v for k, v in m.items()
                       if k.endswith("_sha256") or k in ("obs_mean", "obs_std_with_epsilon")})
        if (m.get("reward") or {}).get("return_range") is not None:
            fields["preparation.metadata.reward.return_range"] = m["reward"]["return_range"]
    norm = meta.get("obs_normalization") or {}
    fields.update({f"obs_normalization.{k}": norm[k] for k in ("obs_mean", "obs_std") if k in norm})
    return fields


def derived_values(meta, arrays, ds):
    """frozen.json fields recomputed from the cached file and the re-derived split, as each host computes them.

    TD3+BC stores ids and statistics as jnp arrays in PreparedData and hashes them with
    runtime.td3_bc._tree_hash (ReBRAC's ids hash identically); CQL hashes its statistics with
    _array_hash; IQL fingerprints its int32 indices and raw terminal/timeout episode IDs
    (calibration.iql_reference.fingerprint, calibration.reference.qlearning_episode_ids), and
    prepare_dataset fingerprints get_dataset's raw terminals/timeouts and the tuple of
    d4rl.qlearning_dataset's five arrays in sorted key order: the timeout map's raw rows (next
    observations one row on), cast per row to float32 (terminals to bool) as qlearning_dataset does.
    """
    host, values = host_of(meta), {}
    values["preparation.training_ids_sha256"] = R._array_hash(np.asarray(ds.training, np.int64))
    if host in ("td3_bc", "rebrac"):
        values["preparation.run_input_hashes.training_ids"] = R._tree_hash(jnp.asarray(ds.training))
        values["preparation.run_input_hashes.heldout_ids"] = R._tree_hash(jnp.asarray(ds.heldout))
    if host == "td3_bc":
        values["preparation.run_input_hashes.obs_mean"] = R._tree_hash(jnp.asarray(ds.mean))
        values["preparation.run_input_hashes.obs_std"] = R._tree_hash(jnp.asarray(ds.std))
    if host == "cql":
        values["preparation.run_input_hashes.obs_mean"] = R._array_hash(ds.mean)
        values["preparation.run_input_hashes.obs_std"] = R._array_hash(ds.std)
    if host in ("td3_bc", "cql"):
        values["preparation.raw_dataset_sha256"] = ds.raw_dataset_sha256
    if host in RESERVATION_RECORD:
        inherited = ds.record["inherited_primitive_metadata"]
        values.update({f"{RESERVATION_RECORD[host]}.{k}": inherited[k] for k in RESERVATION_DIGESTS})
    if host == "iql":
        from calibration.iql_reference import fingerprint

        ids = qlearning_episode_ids({"terminals": ds.raw_terminals, "timeouts": ds.raw_timeouts})
        converted = dict(observations=ds.obs[ds.rows], actions=ds.actions[ds.rows],
                         next_observations=ds.obs[ds.rows + 1], rewards=ds.raw_rewards[ds.rows].astype(np.float32),
                         terminals=ds.raw_terminals[ds.rows].astype(bool))
        values["preparation.metadata.converted_numeric_data_sha256"] = fingerprint(
            tuple(converted[k] for k in sorted(converted)))
        del converted
        values.update({"preparation.metadata.raw_terminal_timeout_sha256": fingerprint(
                           {"terminals": ds.raw_terminals, "timeouts": ds.raw_timeouts}),
                       "preparation.population_ids_sha256": R._array_hash(np.asarray(ds.withheld, np.int64)),
                       "preparation.metadata.train_indices_sha256": fingerprint(ds.training),
                       "preparation.metadata.withheld_indices_sha256": fingerprint(ds.withheld),
                       "preparation.metadata.calibration_indices_sha256": fingerprint(ds.heldout),
                       "preparation.metadata.episode_ids_sha256": fingerprint(ids),
                       "preparation.metadata.obs_mean": ds.mean.tolist(),
                       "preparation.metadata.obs_std_with_epsilon": ds.std.tolist()})
    values.update({"obs_normalization.obs_mean": ds.mean.tolist(), "obs_normalization.obs_std": ds.std.tolist()})
    return R._json_value(values)


def checkpoint_statistics(directory, meta):
    """ReBRAC's calibrator statistics as its checkpoint stores them (state.cal_obs_mean / cal_obs_std)."""
    from flax import serialization

    path = Path(directory) / Path(meta["checkpoint"]["path"]).name
    state = serialization.msgpack_restore(path.read_bytes())["state"]
    return {field: np.asarray(state[key], np.float32) for field, key in DEVICE_DEPENDENT.items()}


def reprepared_values(meta, directory, data_dir):
    """The fields check 5 classifies, from the host builder's own prepare() on resolved.json (no training).

    ReBRAC's device-dependent calibrator statistics come back as their largest deviation from
    the checkpoint's, in units of the checkpoint's cal_obs_std, not as hashes.
    """
    host, row = host_of(meta), _resolved(directory)
    if row is None:
        raise FileNotFoundError("resolved.json missing")
    values = {}
    with contextlib.redirect_stdout(io.StringIO()):  # the runtimes' legacy reward-transform print
        if host == "iql":
            from experiments.wbcp import freeze_iql as I

            data = I.prepare(row, data_dir).data
            m, training, withheld = data.metadata, data.train_indices, data.withheld_indices
            _flatten("preparation.metadata", m, values)
        else:
            if host == "td3_bc":
                module = F
            elif host == "cql":
                from experiments.wbcp import freeze_cql as module
            else:
                from experiments.wbcp import freeze_rebrac as module
            prepared = module.prepare(row, data_dir)[1]
            m = prepared.metadata
            training = m.get("training_converted_ids", m.get("training_converted_indices"))
            withheld = m.get("withheld_converted_ids", m.get("withheld_converted_indices"))
            _flatten("preparation.run_input_hashes", m["run_input_hashes"], values)
            if "raw_dataset_sha256" in m:
                values["preparation.raw_dataset_sha256"] = m["raw_dataset_sha256"]
            if host in RESERVATION_RECORD:
                record = m["reservation"]["inherited_primitive_metadata"] if host == "rebrac" else m["reservation"]
                values.update({f"{RESERVATION_RECORD[host]}.{k}": record[k] for k in RESERVATION_DIGESTS})
            if host == "rebrac":
                stored = checkpoint_statistics(directory, meta)
                scale = stored["preparation.run_input_hashes.cal_obs_std"]
                for field, key in DEVICE_DEPENDENT.items():
                    deviation = np.abs(np.asarray(getattr(prepared, key), np.float32) - stored[field]) / scale
                    values[field] = dict(scaled_deviation_from_checkpoint=float(deviation.max()))
    values["preparation.training_ids_sha256"] = R._array_hash(np.asarray(training, np.int64))
    values["re-prepared population rows"] = population_sha256(withheld)
    return R._json_value(values)


def _agrees(field, source, value, recorded):
    if source == "re-preparation" and field in DEVICE_DEPENDENT:
        return value["scaled_deviation_from_checkpoint"] <= DEVICE_TOLERANCE
    return value == recorded


def _show_derived(field, source, value):
    if source == "re-preparation" and field in DEVICE_DEPENDENT:
        return (f"{value['scaled_deviation_from_checkpoint']:.2g} cal_obs_std from the checkpoint's statistics "
                f"(device-dependent float32 reduction; tolerance {DEVICE_TOLERANCE:g})")
    return _show(value)


def check_derived(meta, arrays, ds, reprepared=None, reprepare_error=None, checkpoint=None):
    """Classify every recorded field: match, mismatch, not re-derivable, or code-dependent (not compared).

    Sources: the cached file and the re-derived split ('dataset'), the checkpoint's stored state
    ('checkpoint': ReBRAC's calibrator statistics) and the builder's re-preparation.
    """
    recorded = recorded_fields(meta)
    derived = derived_values(meta, arrays, ds) if ds is not None else {}
    checkpoint = checkpoint or {}
    reprepared = reprepared or {}
    rows = np.asarray(arrays["row"], np.int64)
    if ds is not None:  # frozen.npz obs/action: the population's dataset rows under the re-derived normalization
        recorded["frozen.npz obs"] = R._array_hash(arrays["obs"])
        recorded["frozen.npz action"] = R._array_hash(arrays["action"])
        inside = bool(len(rows) and rows.min() >= 0 and rows.max() < ds.converted)
        raw_rows = ds.rows[rows] if inside else None
        derived["frozen.npz obs"] = (R._array_hash(R.C.normalize_states(ds.obs[raw_rows], ds.mean, ds.std))
                                     if inside else "rows outside the dataset")
        derived["frozen.npz action"] = R._array_hash(ds.actions[raw_rows]) if inside else "rows outside the dataset"
    if "re-prepared population rows" in reprepared:
        recorded["re-prepared population rows"] = population_sha256(rows)
    fields = []
    for name, value in recorded.items():
        entry = dict(field=name, recorded=_show(value))
        sources = {source: values[name] for source, values in
                   (("dataset", derived), ("checkpoint", checkpoint), ("re-preparation", reprepared)) if name in values}
        if name in CODE_DEPENDENT:
            entry["status"] = "code-dependent: not compared"
        elif not sources:
            entry["status"] = NOT_DERIVABLE + (f" (re-preparation failed: {reprepare_error})" if reprepare_error else "")
        else:
            agree = all(_agrees(name, source, v, value) for source, v in sources.items())
            entry.update(status="match" if agree else "mismatch",
                         derived={source: _show_derived(name, source, v) for source, v in sources.items()})
        fields.append(entry)
    counts = {status: sum(f["status"] == status for f in fields) for status in ("match", "mismatch")}
    counts["not re-derivable"] = sum(f["status"].startswith(NOT_DERIVABLE) for f in fields)
    counts["code-dependent"] = sum(f["status"].startswith("code-dependent") for f in fields)
    return dict(ok=counts["mismatch"] == 0 and ds is not None, counts=counts, fields=fields,
                reprepare_error=reprepare_error)


# pools -----------------------------------------------------------------------------------------

def run_record(directory):
    path = Path(directory) / "run_record.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf8"))
    return {k: record.get(k) for k in ("device", "jax_backend", "jax_platforms", "exit_code", "started_utc",
                                       "finished_utc")}


def verify_pool(directory, meta, *, ds, ds_error, data_dir, config_dir, file_sha, reprepare=False):
    started = time.perf_counter()
    directory = Path(directory)
    host = host_of(meta)
    try:
        arrays = _load_arrays(directory)
    except (OSError, ValueError, KeyError) as error:
        arrays, arrays_error = None, f"{type(error).__name__}: {error}"
    else:
        arrays_error = None
    checks = dict(npz=check_npz(directory, meta, arrays), checkpoint=check_checkpoint(directory, meta),
                  data=check_data(directory, meta, data_dir, config_dir, file_sha))
    source = meta.get("source", "short-train")
    if source != "short-train":
        checks["split"] = dict(ok=None, note="checkpoint-mode pool: no population split to re-derive")
    elif arrays is None or ds is None:
        checks["split"] = dict(ok=False, error=arrays_error or ds_error)
    else:
        checks["split"] = check_split(meta, arrays, ds)
    if arrays is None:
        checks["derived"] = dict(ok=False, error=arrays_error)
    else:
        reprepared, error, stored = None, None, None
        if reprepare:
            try:
                reprepared = reprepared_values(meta, directory, data_dir)
            except Exception as failure:  # reported as not re-derivable, never as a pass
                error = f"{type(failure).__name__}: {failure}"
        if host == "rebrac" and checks["checkpoint"]["ok"]:
            stored = {field: R._tree_hash(value) for field, value in checkpoint_statistics(directory, meta).items()}
        checks["derived"] = check_derived(meta, arrays, ds if source == "short-train" else None, reprepared, error,
                                          stored)
        if source != "short-train":
            checks["derived"].update(ok=None, note="checkpoint-mode pool: compared with re-preparation only")
    record = run_record(directory)
    failures = [f"{name}" for name, check in checks.items() if check.get("ok") is False]
    if record is not None and record.get("exit_code") not in (None, 0):
        failures.append(f"run_record exit_code {record['exit_code']}")
    if (directory / "failure.json").is_file():
        failures.append("failure.json present")
    population = checks["split"].get("population_sha256") or (
        population_sha256(arrays["row"]) if arrays is not None else None)
    record_file = meta.get("dataset_file") or {}
    return dict(directory=str(directory), name=directory.name, host=host, dataset=meta.get("dataset"),
                dataset_file=record_file.get("filename"),
                dataset_file_sha256=record_file.get("sha256", meta.get("dataset_sha256")), source=source,
                updates=meta.get("updates"), created_utc=meta.get("created_utc"), git=meta.get("git"),
                superseded=superseded(meta), run_record=record, checks=checks, population_sha256=population,
                ok=not failures, failures=failures, seconds=time.perf_counter() - started)


def _pool_key(item):
    """Dataset file, then split parameters (so each file is read once per split), then host and name."""
    directory, meta = item
    host = host_of(meta)
    return ((meta.get("dataset_file") or {}).get("filename", ""), json.dumps(split_parameters(meta)),
            list(HOSTS).index(host) if host in HOSTS else len(HOSTS), Path(directory).name)


def verify(pools, *, data_dir=DEFAULT_DATA, config_dir=CONFIG_DIR, reprepare=False, skipped=()):
    """Verify pool directories one dataset file at a time; returns the report (see the module docstring)."""
    started = time.perf_counter()
    items, skipped = [], [dict(directory=str(d), reason=r) for d, r in skipped]
    for directory in map(Path, pools):
        if not (directory / "frozen.json").is_file():
            skipped.append(dict(directory=str(directory), reason="frozen.json missing (pool still being written?)"))
            continue
        items.append((directory, json.loads((directory / "frozen.json").read_text(encoding="utf8"))))
    hashes, reports, current = {}, [], (None, None, None)

    def file_sha(path):
        if path not in hashes:
            hashes[path] = sha(path) if path.is_file() else None
        return hashes[path]

    for directory, meta in sorted(items, key=_pool_key):
        filename = (meta.get("dataset_file") or {}).get("filename", "")
        key = (filename, *split_parameters(meta))
        if key != current[0]:  # one dataset file's arrays at a time
            current = (key, None, None)
            if meta.get("source", "short-train") == "short-train":
                try:
                    current = (key, load_dataset(F._data_dir(data_dir) / filename, *split_parameters(meta)), None)
                except Exception as error:
                    current = (key, None, f"{type(error).__name__}: {error}")
        reports.append(verify_pool(directory, meta, ds=current[1], ds_error=current[2], data_dir=data_dir,
                                   config_dir=config_dir, file_sha=file_sha, reprepare=reprepare))
    current = None
    datasets = {}  # by environment, not by file: a host that declared another file for it must stand out
    for report in reports:
        if report["source"] != "short-train":  # checkpoint-mode pools score all rows or a K-thinned bank by design
            continue
        group = datasets.setdefault(report["dataset"], dict(files=set(), pools={}))
        group["files"].add((report["dataset_file"], report["dataset_file_sha256"]))
        group["pools"][report["name"]] = report["population_sha256"]
    for group in datasets.values():
        files, populations = sorted(group.pop("files"), key=str), set(group["pools"].values())
        group.update(dataset_files=[dict(filename=f, sha256=s) for f, s in files],
                     population_sha256=sorted(populations, key=str), same_file=len(files) == 1,
                     identical=len(populations) == 1)
    failures = [f"{r['name']}: {f}" for r in reports for f in r["failures"]]
    failures += [f"{name}: pools declare different dataset files" for name, g in datasets.items() if not g["same_file"]]
    failures += [f"{name}: pools hold different populations" for name, g in datasets.items() if not g["identical"]]
    if not reports:  # unfinished pools are skipped one by one, but a run that checked nothing does not pass
        failures.append("no pools verified")
    return dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                data_dir=str(F._data_dir(data_dir)), config_dir=str(config_dir), reprepare=reprepare,
                git=F.git_identity(), script_sha256=sha(Path(__file__).resolve()), pools=reports, skipped=skipped,
                datasets=datasets, ok=not failures, failures=failures, seconds=time.perf_counter() - started)


# report ----------------------------------------------------------------------------------------

def _mark(check):
    return {True: "ok", False: "FAIL", None: "n/a"}[check.get("ok")]


def markdown(report):
    lines = [f"# Frozen pool verification ({report['created_utc']})", "",
             f"{len(report['pools'])} pools verified, {sum(not r['ok'] for r in report['pools'])} failed, "
             f"{len(report['skipped'])} skipped. Data: `{report['data_dir']}`. Re-preparation: "
             f"{'on' if report['reprepare'] else 'off'}. Overall: {'PASS' if report['ok'] else 'FAIL'}.", "",
             "| pool | host | dataset | npz ok | checkpoint ok | data ok | split ok | derived | population sha256 | notes |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["pools"]:
        c, d = r["checks"], r["checks"]["derived"]
        counts = d.get("counts") or {}
        derived = (f"{_mark(d)}: {counts.get('match', 0)} match, {counts.get('mismatch', 0)} mismatch, "
                   f"{counts.get('not re-derivable', 0)} n/r" if counts else _mark(d))
        record = r["run_record"]
        notes = ([f"{record.get('device')}, exit {record.get('exit_code')}"] if record else ["no run_record.json"])
        notes += ["superseded (pre-qf_target fix)"] if r["superseded"] else []
        notes += [f"failed: {', '.join(r['failures'])}"] if r["failures"] else []
        notes += [f"split: {', '.join(c['split']['failed'])}"] if c["split"].get("failed") else []
        notes += [f"mismatch: {f['field']}" for f in d.get("fields", []) if f["status"] == "mismatch"]
        if d.get("reprepare_error"):
            notes.append("re-preparation failed")
        lines.append(f"| {r['name']} | {HOSTS.get(r['host'], r['host'])} | {r['dataset']} | {_mark(c['npz'])} | "
                     f"{_mark(c['checkpoint'])} | {_mark(c['data'])} | {_mark(c['split'])} | {derived} | "
                     f"{(r['population_sha256'] or '-')[:12]} | {'; '.join(notes)} |")
    lines += ["", "| dataset | dataset file (sha256) | pools | population sha256 | one file | identical population |",
              "|---|---|---|---|---|---|"]
    for name, g in report["datasets"].items():
        files = ", ".join(f"{f['filename']} ({(f['sha256'] or '-')[:12]})" for f in g["dataset_files"])
        lines.append(f"| {name} | {files} | {len(g['pools'])} | "
                     f"{', '.join((s or '-')[:12] for s in g['population_sha256'])} | "
                     f"{'yes' if g['same_file'] else 'NO'} | {'yes' if g['identical'] else 'NO'} |")
    if report["skipped"]:
        lines += ["", "Skipped: " + "; ".join(f"{Path(s['directory']).name} ({s['reason']})" for s in report["skipped"])]
    if report["failures"]:
        lines += ["", "Failures: " + "; ".join(report["failures"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pools", nargs="*", type=Path, help="pool directories (default: every pool under --root)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="directory of frozen pools")
    parser.add_argument("--all", action="store_true", help="also verify smoke, gputest and -u2000 pools")
    parser.add_argument("--reprepare", action="store_true",
                        help="also re-prepare each pool with its builder's prepare() (no training) and compare")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="cached D4RL HDF5 directory")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="directory of <algorithm>.yaml configs")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for verification.json / verification.md")
    opt = parser.parse_args(argv)
    if opt.output.exists():
        parser.error("--output already exists")
    if opt.pools:
        pools, skipped = opt.pools, []
    else:
        pools, skipped = discover(opt.root, opt.all)
    report = verify(pools, data_dir=opt.data_dir, config_dir=opt.config_dir, reprepare=opt.reprepare,
                    skipped=skipped)
    table = markdown(report)
    opt.output.mkdir(parents=True)
    with (opt.output / "verification.json").open("x", encoding="utf8") as f:
        json.dump(R._json_value(report), f, indent=1, allow_nan=False)
        f.write("\n")
    with (opt.output / "verification.md").open("x", encoding="utf8") as f:
        f.write(table)
    print(table)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
