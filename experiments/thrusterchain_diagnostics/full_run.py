"""Frozen full-core orchestration; all output belongs to a new run directory.

Prepare once, then run each worker's disjoint jobs sequentially, solve the shared
final oracle once, and evaluate. ``run --max-jobs 1`` bounds a worker session
without changing the 30-job plan. Workers use the frozen three-way matrix stride.
All commands are resumable except prepare, which deliberately refuses existing
directories. No command selects a checkpoint using an observed return.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
import uuid


SCHEMA = "thrusterchain-full-core-v1"
TERMINAL = {"completed", "failed", "refused"}
RETENTIONS = ("full", "rare", "zero")
ARMS = ("td3_bc", "bca")
_EVENT_FILE = "events.jsonl"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event(directory, kind, **details):
    row = dict(time=_now(), event=kind, **details)
    with (Path(directory) / _EVENT_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)


def default_config():
    from .training import TrainingSettings
    return dict(schema=SCHEMA, arms=list(ARMS), retentions=list(RETENTIONS),
        data_seeds=list(range(1101, 1106)), optimization_seeds=list(range(2101, 2106)),
        reservation_seeds=list(range(3101, 3106)), source_episodes=1024,
        reservation_episodes=128, rare_probability=.05, settings=asdict(TrainingSettings()),
        validation=dict(seed=72001, per_context=20, update=50_000),
        final=dict(seed=82001, per_context=100, update=100_000),
        query=dict(seed=92001, per_context=4, times=[0, 50, 99]),
        calibration_evaluation=dict(seed=102001, split="final", query_bank="query"),
        oracle=dict(split="final", required=True, shared=True),
        execution=dict(training_host="zeus", parallel_jobs=3),
        controls=["fixed_cap", "regime_cap"],
        selection="none; terminal primary and midpoint descriptive", normalization="per-retention training complement")


def validate_config(config):
    """This CLI executes the authorized full core, not an adaptive sweep."""
    expected = default_config()
    if set(config) != set(expected):
        raise ValueError("configuration fields differ from the full-core schema")
    for key in expected:
        if key == "settings":
            # Chunk size affects checkpoint frequency, not the scientific budget.
            got = dict(config[key])
            chunk = got.pop("chunk", None)
            want = dict(expected[key])
            want.pop("chunk")
            if got != want or not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
                raise ValueError("training settings differ from the frozen full-core protocol")
        elif config[key] != expected[key]:
            raise ValueError("full-core configuration differs at " + key)
    return config


def job_matrix(config):
    return [dict(job_id=f"replicate_{index + 1:02d}/{retention}/{arm}", replicate=index + 1,
                 data_seed=data_seed, optimization_seed=config["optimization_seeds"][index],
                 reservation_seed=config["reservation_seeds"][index], retention=retention, arm=arm)
            for index, data_seed in enumerate(config["data_seeds"])
            for retention in config["retentions"] for arm in config["arms"]]


def _source_hashes():
    from . import training as T
    repository = Path(__file__).resolve().parents[2]
    suite = Path(__file__).resolve().parent
    paths = [suite / name for name in ("__init__.py", "continuous.py", "continuous_data.py",
             "training.py", "training_data.py", "training_eval.py", "training_calibration.py",
             "full_run.py", "TRAINING_PROTOCOL.md")]
    return dict(T._source_hashes(), **{path.relative_to(repository).as_posix(): _sha(path)
                                    for path in sorted(paths) if path.is_file()})


def _environment():
    packages = {}
    for name in ("numpy", "scipy", "jax", "jaxlib", "flax", "optax", "chex"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True, timeout=10)
        revision = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    return dict(host=socket.gethostname(), platform=platform.platform(), python=sys.version,
                executable=sys.executable, packages=packages, git_revision=revision,
                environment={name: os.environ.get(name) for name in (
                    "JAX_PLATFORM_NAME", "JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")})


def _process_identity(pid):
    """Return (alive/dead/unknown, creation token). Never signal Windows jobs."""
    if not isinstance(pid, int) or pid <= 0:
        return "unknown", None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ("dead", None) if ctypes.get_last_error() == 87 else ("unknown", None)
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return "unknown", None
            if code.value != 259:
                return "dead", None
            fields = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(field) for field in fields)):
                return "alive", None
            token = str((fields[0].dwHighDateTime << 32) | fields[0].dwLowDateTime)
            return "alive", token
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead", None
    except PermissionError:
        return "unknown", None
    proc = Path(f"/proc/{pid}/stat")
    try:
        token = proc.read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        token = None
    return "alive", token


def _proven_stale(record):
    if record.get("host", "").casefold() != socket.gethostname().casefold():
        return False
    state, token = _process_identity(record.get("pid"))
    return state == "dead" or (state == "alive" and token is not None
                               and record.get("creation_token") is not None
                               and record["creation_token"] != token)


@contextmanager
def run_lock(directory, *, name="driver"):
    """OS lock plus PID identity prevents races while reclaiming stale metadata."""
    path = Path(directory) / ("." + name + ".lock")
    stream = path.open("a+b")
    if path.stat().st_size == 0:
        stream.write(b" ")
        stream.flush()
    stream.seek(0)
    acquired = False
    claimed = False
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        stream.seek(0)
        content = stream.read().decode("utf-8").strip()
        old = json.loads(content) if content else {}
        if old.get("active") and not _proven_stale(old):
            raise RuntimeError(f"{name} lock holder is alive or cannot be proven stale: {old}")
        _, token = _process_identity(os.getpid())
        record = dict(active=True, pid=os.getpid(), host=socket.gethostname(),
                      creation_token=token, acquired=_now())
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(record).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
        claimed = True
        yield
    except OSError as error:
        if not acquired:
            raise RuntimeError(f"another process owns {name} lock: {path}") from error
        raise
    finally:
        if claimed:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps(dict(record, active=False, released=_now())).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _settings(config, reservation_seed):
    from .training import TrainingSettings
    return replace(TrainingSettings(**config["settings"]), reservation_seed=reservation_seed)


def _dataset_path(directory, job):
    return Path(directory) / "data" / f"seed_{job['data_seed']}" / job["retention"]


def _prepared(dataset, settings):
    from . import training as T
    return T.prepare_training(dataset.learner_rows, settings, data_metadata=dataset.metadata,
                              calibration_episode_ids=dataset.metadata["calibration_episode_ids"])


def prepare(directory, config=None):
    global _EVENT_FILE
    _EVENT_FILE = "events.jsonl"
    from . import training as T, training_data as D, training_eval as E
    directory = Path(directory).resolve()
    config = validate_config(default_config() if config is None else config)
    directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory):
        _write(directory / "config.json", config, exclusive=True)
        _write(directory / "job_matrix.json", job_matrix(config), exclusive=True)
        _write(directory / "provenance.json", dict(schema=SCHEMA, created=_now(),
               source_sha256=_source_hashes(), environment=_environment()), exclusive=True)
        _event(directory, "prepare_started", jobs=len(job_matrix(config)))
        try:
            for data_seed, reserve_seed in zip(config["data_seeds"], config["reservation_seeds"]):
                source = D.generate_source_data(seed=data_seed, n_episodes=config["source_episodes"],
                    reservation_size=config["reservation_episodes"], reservation_seed=reserve_seed)
                D.save_source(source, directory / "audit" / f"seed_{data_seed}")
                for retention in config["retentions"]:
                    dataset = D.retain_source(source, retention, rare_probability=config["rare_probability"])
                    destination = directory / "data" / f"seed_{data_seed}" / retention
                    D.save_dataset(dataset, destination)
                    prepared = _prepared(dataset, _settings(config, reserve_seed))
                    _write(destination / "preparation.json", dict(manifest=prepared.manifest,
                        training_transition_sha256=T.tree_sha256(prepared.train),
                        calibration_transition_sha256=T.tree_sha256(prepared.bank)), exclusive=True)
                _event(directory, "source_prepared", data_seed=data_seed)
            for split in ("validation", "final"):
                E.save_bank(E.make_start_bank(split=split, seed=config[split]["seed"],
                    per_context=config[split]["per_context"]), directory / "banks" / split)
            E.save_bank(E.make_query_bank(per_context=config["query"]["per_context"]), directory / "banks" / "query")
            paths = [directory / name for name in ("config.json", "job_matrix.json", "provenance.json")]
            for group in ("audit", "data", "banks"):
                paths.extend(path for path in (directory / group).rglob("*") if path.is_file())
            manifest = dict(schema=SCHEMA, ready=True, created=_now(),
                files={path.relative_to(directory).as_posix(): _sha(path) for path in sorted(paths)})
            _write(directory / "prepared_manifest.json", manifest, exclusive=True)
            _event(directory, "prepare_completed", files=len(manifest["files"]))
        except BaseException as error:
            _event(directory, "prepare_failed", error=type(error).__name__ + ": " + str(error))
            raise
    return manifest


def verify_run(directory):
    directory = Path(directory).resolve()
    manifest = _read(directory / "prepared_manifest.json")
    if manifest.get("schema") != SCHEMA or not manifest.get("ready"):
        raise ValueError("run preparation is incomplete")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or _sha(path) != expected:
            raise ValueError("frozen artifact changed: " + name)
    config = validate_config(_read(directory / "config.json"))
    if _read(directory / "job_matrix.json") != job_matrix(config):
        raise ValueError("job matrix differs from the frozen configuration")
    if _read(directory / "provenance.json")["source_sha256"] != _source_hashes():
        raise ValueError("source changed since preparation; preserve this run and use a new identity")
    return config


def _job_status(directory, job, status, **details):
    record = dict(job=job, status=status, time=_now(), **details)
    _write(Path(directory) / "jobs" / job["job_id"] / "driver_status.json", record)
    _event(directory, "job_" + status, job_id=job["job_id"], **details)
    return record


def _recoverable_failure(reason):
    return any(str(reason).startswith(name + ":") for name in (
        "OSError", "IOError", "PermissionError", "FileNotFoundError", "TimeoutError", "MemoryError"))


def run_jobs(directory, *, max_jobs=None, worker_index=0, workers=3, retry_failed=False):
    global _EVENT_FILE
    _EVENT_FILE = f"events.worker_{worker_index}.jsonl"
    from . import training as T, training_data as D
    directory = Path(directory).resolve()
    if max_jobs is not None and (isinstance(max_jobs, bool) or max_jobs < 1):
        raise ValueError("max_jobs must be positive")
    if not isinstance(worker_index, int) or not 0 <= worker_index < workers:
        raise ValueError("worker_index must be in [0, workers)")
    with run_lock(directory, name=f"worker_{worker_index}"):
        config = verify_run(directory)
        if workers != config["execution"]["parallel_jobs"]:
            raise ValueError("worker partition count differs from the frozen execution configuration")
        if socket.gethostname().casefold() != config["execution"]["training_host"].casefold():
            raise RuntimeError("full-core training is authorized only on Zeus")
        _event(directory, "run_session_started", environment=_environment(), max_jobs=max_jobs,
               worker_index=worker_index, workers=workers)
        attempted = 0
        for job in job_matrix(config)[worker_index::workers]:
            job_dir = directory / "jobs" / job["job_id"]
            status_path = job_dir / "driver_status.json"
            previous = _read(status_path) if status_path.exists() else {}
            if previous.get("status") in TERMINAL:
                if not (retry_failed and previous["status"] == "failed"
                        and _recoverable_failure(previous.get("reason"))):
                    continue
            if max_jobs is not None and attempted >= max_jobs:
                break
            attempted += 1
            started = time.perf_counter()
            if previous.get("status") == "failed":
                _event(directory, "operator_requested_recovery", job_id=job["job_id"], previous=previous)
            _job_status(directory, job, "running")
            try:
                training_manifest = job_dir / "manifest.json"
                if training_manifest.exists() and _read(training_manifest).get("status") == "failed":
                    native_failure = _read(training_manifest).get("failure", "")
                    if not (retry_failed and _recoverable_failure(native_failure)):
                        raise RuntimeError("previous native training attempt failed; no automatic retry of an invalid update")
                dataset_path = _dataset_path(directory, job)
                dataset = D.load_dataset(dataset_path)
                settings = _settings(config, job["reservation_seed"])
                prepared = _prepared(dataset, settings)
                frozen = _read(dataset_path / "preparation.json")
                if (prepared.manifest != frozen["manifest"] or T.tree_sha256(prepared.train) != frozen["training_transition_sha256"]
                        or T.tree_sha256(prepared.bank) != frozen["calibration_transition_sha256"]):
                    raise ValueError("prepared training/calibration arrays differ from the frozen identity")
                result = T.train(prepared, settings, arm=job["arm"], seed=job["optimization_seed"],
                    out_dir=job_dir, resume=True,
                    progress=lambda row, job=job: _event(directory, "training_progress", job_id=job["job_id"], **row))
                record = result.record
                if record["status"] == "refused":
                    _job_status(directory, job, "refused", reason=record.get("reason"), final_update=0,
                                session_seconds=time.perf_counter() - started)
                    continue
                if record["status"] != "completed" or record["final_update"] != settings.updates:
                    raise RuntimeError("training returned without reaching the required accepted-update budget")
                loaded = T.load_trained(job_dir)
                if loaded.record["checkpoint"]["update"] != settings.updates:
                    raise RuntimeError("terminal checkpoint did not verify at the full budget")
                _job_status(directory, job, "completed", final_update=record["final_update"],
                    checkpoint=record["checkpoint"], training_seconds=record.get("training_seconds"),
                    session_seconds=time.perf_counter() - started)
            except KeyboardInterrupt:
                _job_status(directory, job, "interrupted", reason="KeyboardInterrupt; last committed checkpoint preserved",
                            session_seconds=time.perf_counter() - started)
                raise
            except Exception as error:
                _job_status(directory, job, "failed", reason=type(error).__name__ + ": " + str(error),
                            traceback=traceback.format_exc(), session_seconds=time.perf_counter() - started)
        _event(directory, "run_session_finished", attempted=attempted, worker_index=worker_index)
    return status(directory)


def solve_oracle(directory):
    global _EVENT_FILE
    _EVENT_FILE = "events.oracle.jsonl"
    from . import training_eval as E
    directory = Path(directory).resolve()
    with run_lock(directory, name="oracle"):
        verify_run(directory)
        bank = E.load_bank(directory / "banks" / "final")
        destination = directory / "oracle" / "final.json"
        if destination.exists():
            result = _read(destination)
            if result["bank_sha256"] != bank.metadata["arrays_sha256"]:
                raise ValueError("cached oracle bank identity differs")
            return result
        # Individual solved starts survive interruption; no trained policy enters this stage.
        from .continuous import FiniteHorizonQP
        rows = []
        problem = FiniteHorizonQP(bank.objective)
        for index, (state, regime) in enumerate(zip(bank.states, bank.regimes)):
            path = directory / "oracle" / "starts" / f"{index:04d}.json"
            if path.exists():
                record = _read(path)
                if record["bank_sha256"] != bank.metadata["arrays_sha256"] or record["index"] != index:
                    raise ValueError("cached oracle row identity differs")
                row = record["solution"]
            else:
                row = problem.solve(state, int(regime)).as_dict()
                _write(path, dict(index=index, bank_sha256=bank.metadata["arrays_sha256"], solution=row), exclusive=True)
            rows.append(row)
            _event(directory, "oracle_progress", completed=index + 1, total=len(bank.states))
        result = dict(schema="thrusterchain-shared-continuous-optima-v1", bank_sha256=bank.metadata["arrays_sha256"],
            return_lower=[row["return_interval"][0] for row in rows],
            return_upper=[row["return_interval"][1] for row in rows], solutions=rows,
            scope="fixed shared starts only; not exact continuous population optimum")
        _write(destination, result, exclusive=True)
    return result


def _evaluation_task(directory, name, evaluate):
    """Commit a complete evaluation directory atomically; preserve interrupted work."""
    from . import training_eval as E
    destination = Path(directory) / "evaluation" / name
    if not destination.resolve().is_relative_to((Path(directory) / "evaluation").resolve()):
        raise ValueError("evaluation must remain within its run directory")
    if (destination / "summary.json").exists():
        record = _read(destination / "summary.json")
        if _sha(destination / record["file"]) != record["file_sha256"]:
            raise ValueError("saved evaluation array hash differs: " + name)
        if not _diagnostic_failed(record):
            return record
    temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    try:
        summary, arrays = evaluate()
        record = E.save_evaluation(summary, arrays, temporary)
        if destination.exists():
            preserved = destination.with_name(destination.name + ".previous-" + uuid.uuid4().hex)
            destination.replace(preserved)
        temporary.replace(destination)
        _event(directory, "evaluation_completed", name=name)
        return record
    except Exception as error:
        _event(directory, "evaluation_failed", name=name, reason=type(error).__name__ + ": " + str(error))
        return None


def _diagnostic_failed(record):
    return any(record.get(key, {}).get("status") == "failed"
               for key in ("critic_diagnostics", "calibration_diagnostics"))


def evaluate_jobs(directory, *, controls_only=False):
    global _EVENT_FILE
    _EVENT_FILE = "events.evaluation.jsonl"
    from . import training as T, training_eval as E
    directory = Path(directory).resolve()
    with run_lock(directory, name="evaluation"):
        config = verify_run(directory)
        banks = {split: E.load_bank(directory / "banks" / split) for split in ("validation", "final", "query")}
        oracle_path = directory / "oracle" / "final.json"
        optimum = _read(oracle_path) if oracle_path.exists() else None
        for split in ("validation", "final"):
            for control in config["controls"]:
                # Always complete fixed controls, including while the shared oracle is pending.
                _evaluation_task(directory, f"controls/{split}/{control}", lambda split=split, control=control:
                    E.evaluate_policy(E.batched_control(control), banks[split]))
        if controls_only:
            return status(directory)
        for job in job_matrix(config):
            job_dir = directory / "jobs" / job["job_id"]
            training_path = job_dir / "manifest.json"
            driver_path = job_dir / "driver_status.json"
            driver = _read(driver_path) if driver_path.exists() else {"status": "pending"}
            record = _read(training_path) if training_path.exists() else {}
            for split in ("validation", "final"):
                update = config[split]["update"]
                name = f"jobs/{job['job_id']}/{split}"
                if record.get("final_update", 0) < update or (split == "final" and driver["status"] != "completed"):
                    reason = "training " + driver["status"] + "; required checkpoint unavailable"
                    _write(directory / "evaluation_status" / (name + ".json"), dict(status="unavailable", reason=reason))
                    continue
                if split == "final" and optimum is None:
                    _write(directory / "evaluation_status" / (name + ".json"), dict(status="pending", reason="shared final oracle required"))
                    continue
                def evaluate(job_dir=job_dir, split=split, update=update):
                    trained = T.load_trained(job_dir, update=update)
                    if trained.record["checkpoint"]["update"] != update:
                        raise ValueError("evaluation checkpoint is not the predeclared checkpoint")
                    result, arrays = E.evaluate_policy(trained.policy, banks[split],
                        critic=trained.critic if split == "final" else None,
                        query_bank=banks["query"] if split == "final" else None,
                        optimum=optimum if split == "final" else None)
                    if split == "final":
                        from .training_calibration import evaluate_calibration
                        try:
                            calibration, calibration_arrays = evaluate_calibration(trained, banks["query"],
                                seed=config["calibration_evaluation"]["seed"])
                            result["calibration_diagnostics"] = calibration
                            arrays.update({"calibration_" + key: value for key, value in calibration_arrays.items()})
                        except Exception as error:
                            result["calibration_diagnostics"] = dict(status="failed", reason=type(error).__name__ + ": " + str(error))
                    result.update(job=job, checkpoint=trained.record["checkpoint"])
                    return result, arrays
                result = _evaluation_task(directory, name, evaluate)
                diagnostic_status = result.get("critic_diagnostics", {}).get("status") if result else None
                _write(directory / "evaluation_status" / (name + ".json"),
                       dict(status="completed" if result and not _diagnostic_failed(result) else "failed",
                            critic_diagnostics=diagnostic_status,
                            calibration_diagnostics=result.get("calibration_diagnostics", {}).get("status") if result else None))
        report = status(directory)
        _write(directory / "report.json", report)
    return report


def status(directory):
    """Every planned seed remains in the denominator, including explicit failures."""
    directory = Path(directory)
    config = _read(directory / "config.json")
    records, counts = [], {}
    for job in job_matrix(config):
        path = directory / "jobs" / job["job_id"] / "driver_status.json"
        record = _read(path) if path.exists() else dict(job=job, status="pending")
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        record["evaluation"] = {}
        record["evaluation_status"] = {}
        for split in ("validation", "final"):
            summary_path = directory / "evaluation" / "jobs" / job["job_id"] / split / "summary.json"
            record["evaluation"][split] = _read(summary_path) if summary_path.exists() else None
            evaluation_status = directory / "evaluation_status" / "jobs" / job["job_id"] / (split + ".json")
            record["evaluation_status"][split] = _read(evaluation_status) if evaluation_status.exists() else {"status": "pending"}
        records.append(record)
    comparisons = []
    for retention in config["retentions"]:
        for context in ("pooled", "stall_low", "stall_high", "monotone_low", "monotone_high"):
            pairs = []
            for replicate in range(1, 6):
                selected = {row["job"]["arm"]: row for row in records
                            if row["job"]["replicate"] == replicate and row["job"]["retention"] == retention}
                values = {}
                for arm, row in selected.items():
                    evaluation = row["evaluation"]["final"]
                    values[arm] = None if evaluation is None else (evaluation["pooled"] if context == "pooled"
                                                                   else evaluation["strata"][context])["mean_return"]
                difference = None if any(values.get(arm) is None for arm in ARMS) else values["bca"] - values["td3_bc"]
                pairs.append(dict(replicate=replicate, returns=values, bca_minus_td3_bc=difference))
            complete = [pair["bca_minus_td3_bc"] for pair in pairs if pair["bca_minus_td3_bc"] is not None]
            comparisons.append(dict(retention=retention, context=context, replicates=pairs,
                complete_pairs=len(complete), planned_pairs=5,
                mean_difference=statistics.mean(complete) if len(complete) == 5 else None,
                sample_std_difference=statistics.stdev(complete) if len(complete) == 5 else None,
                inference="mean and sample SD across five trained replicates; no interval claimed"))
    controls = {f"{split}/{kind}": (directory / "evaluation" / "controls" / split / kind / "summary.json").exists()
                for split in ("validation", "final") for kind in config["controls"]}
    all_terminal = all(row["status"] in TERMINAL for row in records)
    eval_complete = all(all(row["evaluation"].values()) and all(
        item["status"] == "completed" for item in row["evaluation_status"].values())
        if row["status"] == "completed" else all(item["status"] in ("completed", "unavailable")
        for item in row["evaluation_status"].values()) for row in records)
    return dict(schema=SCHEMA, time=_now(), planned_jobs=len(records), statuses=counts,
        training_terminal=all_terminal, complete=all_terminal and eval_complete and all(controls.values())
        and (directory / "oracle" / "final.json").exists(), controls=controls,
        shared_final_oracle=(directory / "oracle" / "final.json").exists(), jobs=records, comparisons=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("config", help="write the full immutable configuration template")
    template.add_argument("--output", required=True)
    for command in ("prepare", "run", "oracle", "evaluate", "status"):
        child = sub.add_parser(command)
        child.add_argument("--output", required=True, type=Path, help="new directory for prepare; existing run otherwise")
        if command == "prepare":
            child.add_argument("--config", type=Path)
        if command == "run":
            child.add_argument("--max-jobs", type=int)
            child.add_argument("--worker-index", type=int, default=0)
            child.add_argument("--workers", type=int, default=3)
            child.add_argument("--retry-failed", action="store_true",
                help="operator-reviewed recovery of I/O/resource failures only; preserves the prior reason and resumes")
        if command == "evaluate":
            child.add_argument("--controls-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "config":
        _write(args.output, default_config(), exclusive=True)
        return 0
    if args.command == "prepare":
        result = prepare(args.output, _read(args.config) if args.config else None)
    elif args.command == "run":
        result = run_jobs(args.output, max_jobs=args.max_jobs, worker_index=args.worker_index, workers=args.workers,
                          retry_failed=args.retry_failed)
    elif args.command == "oracle":
        result = solve_oracle(args.output)
    elif args.command == "evaluate":
        result = evaluate_jobs(args.output, controls_only=args.controls_only)
    else:
        result = status(args.output)
    if "jobs" in result:
        result = {key: result[key] for key in ("planned_jobs", "statuses", "training_terminal", "complete", "shared_final_oracle")}
    elif "solutions" in result:
        result = dict(bank_sha256=result["bank_sha256"], starts=len(result["solutions"]))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
