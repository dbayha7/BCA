"""Explicit one-owner Colab transport profile; frozen scientific code is reused.

Importing this module performs no model, package-installation or simulator work.
Operator evidence cannot establish a distributed lock: conflicting/unknown
machine status is rejected and must be resolved outside this notebook.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
RELATIVE = Path("experiments/thrusterchain_bca/two_scenario_path_a")
PROFILE_DIR = Path("analysis/path_a_colab_v1")
TITAN_MANIFEST_SHA256 = "32718856c6b4714ea1a4004554d61b80223d502e3517b5563c372a97047a15ed"
BOOT_PINS = {
    "launch.py": "7760811f7b0994b2a407f68ae59e703447acc9beefa79fc7b853a3c889fd0609",
    "smoke_launch.py": "17ea43520a3362413d4504e692b6cc6327aac4d929b2aac9fc5b2c1ade32faf8",
    "recovery_launch.py": "b661ae7a969a166d0647950486d8567c84c945f9b194cd0b4158fbbf787774a6",
    "titan_launch.py": "d3063e8e49dbdc41373f79e8bcd237d10f22e3b9af159c1dbf9ce80c02a61538"}
ZERO_WORK = dict(critic=0, actor=0, auxiliary=0, calibrations=0, simulator_steps=0, reset_calls=0)
MIGRATION_ATTESTATION = ("I checked the source and Titan; neither Titan validation nor recovery has started. "
                         "Zeus remains held. Colab is the sole selected runner.")
APPROVAL_STATEMENT = ("I approve this manifest-bound Colab migration only after checking that Titan validation "
    "and recovery never started and Zeus remains held. Colab is the sole runner. Preserve the frozen Path A "
    "science and completed warmup: one 120-second zero-update compatibility check, then only on success the "
    "17 remaining training jobs and evaluation. Debit installation, preparation and waits from the existing "
    "10660 seconds, including the prior 60-second allowance; science never exceeds 10480 seconds. "
    "No automatic retry, extra scientific work, Path B or budget reset.")


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def boot_id(): return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def elapsed(ledger):
    require(ledger.get("schema") == "path_a.colab_setup.v1" and ledger.get("timing_complete") is True and
            ledger.get("unknown_runtime_seconds") == 0 and ledger.get("includes_install_upload_extract_and_waits") is True,
            "Incomplete setup timing; never substitute a fresh allowance")
    require(ledger.get("boot_id") == boot_id(), "Runtime restarted or timing belongs to another host; no automatic resume")
    mono, wall = time.monotonic() - ledger["monotonic_started"], time.time() - ledger["wall_started"]
    require(all(math.isfinite(x) and x >= 0 for x in (mono, wall)), "Invalid/backward setup clock")
    return max(mono, wall)


def budget(ledger, validation_elapsed=0):
    require(math.isfinite(validation_elapsed) and 0 <= validation_elapsed <= 115, "Unknown validation cost")
    # Elapsed includes validation itself; reserve its unconsumed portion as well.
    remaining = math.floor(10660 - 60 - elapsed(ledger) - (120 - validation_elapsed))
    require(remaining > 15, "Existing allowance exhausted; do not extend or reset it")
    return {"scientific_cap_seconds": min(10480, remaining), "validation_cap_seconds": 120,
            "prior_reserved_seconds": 60, "shared_allowance_seconds": 10660,
            "setup_and_waits_charged": True, "unknown_runtime_seconds": 0}


def migration(status):
    expected = {"schema": "path_a.colab_migration.v1", "selected_owner": "Colab",
        "titan_validation": "not_started", "titan_recovery": "not_started", "zeus_hold_preserved": True,
        "other_execution": "none", "additional_scientific_work": 0, "additional_runtime_seconds_known": True,
        "additional_runtime_seconds": 0, "attestation": MIGRATION_ATTESTATION}
    require(all(status.get(k) == v for k, v in expected.items()),
            "Migration status is unknown/conflicting/consumed; this profile cannot resume another attempt")
    require(isinstance(status.get("operator"), str) and status["operator"].strip(), "Named operator required")
    observed = datetime.fromisoformat(status["checked_at_est"])
    require(observed.utcoffset() is not None and observed.utcoffset().total_seconds() == -18000,
            "Record the actual source-status observation in EST")
    evidence = status.get("evidence")
    require(isinstance(evidence, list) and evidence, "Captured source-status evidence is required")
    for item in evidence:
        require(Path(item["path"]).is_absolute() and sha(item["path"]) == item["sha256"], "Migration evidence differs")
    return status


def load_engine(repo):
    here = Path(repo).resolve() / RELATIVE
    for name, digest in BOOT_PINS.items():
        require(sha(here / name) == digest, "Frozen bootstrap source differs: " + name)
    require(sha(here / "TITAN_MANIFEST.json") == TITAN_MANIFEST_SHA256, "Wrong frozen Titan closure")
    for name in ("launch", "smoke_launch", "recovery_launch"):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == here / (name + ".py"), "Module-path collision")
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("_path_a_colab_frozen_titan", here / "titan_launch.py")
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    return engine


class Profile:
    def __init__(self, repo, *, engine=None):
        self.repo = Path(repo).resolve()
        self.directory = self.repo / PROFILE_DIR
        self.here = self.repo / RELATIVE
        self.validation = self.here / "colab_validation_1"
        self.output = self.here / "run_colab_recovery_1"
        self.T = engine if engine is not None else load_engine(self.repo)
        self.R, self.L, self.S = self.T.R, self.T.L, self.T.S
        self.original_check = self.T.check

    def no_other_attempt(self):
        require(not any((self.here / n).exists() for n in ("titan_validation_1", "run_titan_recovery_1", "run_recovery_1")),
                "A competing/prior migration attempt or source hold was copied here; no migration")

    def check(self, approval=False):
        protocol = self.original_check(approval=False)  # Original approvals are historical evidence only.
        self.no_other_attempt()
        manifest = read(self.directory / "COLAB_MANIFEST.json")
        require(manifest["schema"] == "path_a.colab_profile.v1" and manifest["repository"] == str(self.repo) and
                manifest["titan_manifest_sha256"] == TITAN_MANIFEST_SHA256 and manifest["jobs"] == self.R.jobs(),
                "Colab profile/scientific plan differs")
        for path, digest in manifest["files"].items():
            require(sha(path) == digest, "Colab profile/evidence changed: " + path)
        migration(read(manifest["migration_status_path"]))
        elapsed(read(manifest["setup_ledger_path"]))
        if approval:
            a = read(self.directory / "COLAB_APPROVAL.json")
            require(a.get("statement") == APPROVAL_STATEMENT and a.get("approve_colab_recovery") is True and
                    a.get("manifest_sha256") == sha(self.directory / "COLAB_MANIFEST.json") and
                    a.get("operator") == read(manifest["migration_status_path"])["operator"],
                    "Fresh explicit Colab approval is missing or differs; Titan approval is insufficient")
        return protocol, manifest

    def identity(self):
        _, manifest = self.check(approval=True)
        require(os.name == "posix" and sys.executable == manifest["executable"] and sys.version_info[:2] == (3, 12),
                "Use the exact selected isolated Python3.12 Colab interpreter")
        return {"machine_profile": "Colab", "repository": str(self.repo), "executable": sys.executable,
                "python": sys.version, "hostname": platform.node(), "boot_id": boot_id(),
                "gpu_identity": "Recorded from actual JAX runtime; no device-name substitution"}

    def proof(self):
        _, manifest = self.check(approval=True)
        final = read(self.validation / "RESULT.json")
        report = read(self.validation / "report.json")
        attempt = read(self.validation / "ATTEMPT.json")
        require(final.get("status") == "passed_colab_validation" and final.get("identity_unchanged") is True and
                final["elapsed_seconds"] <= 115 and final["report_sha256"] == sha(self.validation / "report.json") and
                report.get("passed") is True and report.get("new_work") == ZERO_WORK and
                report.get("status") == "passed_zero_update_titan_compatibility",
                "No authenticated successful Colab compatibility result")
        require(attempt["manifest_sha256"] == sha(self.directory / "COLAB_MANIFEST.json") and
                attempt["approval_sha256"] == sha(self.directory / "COLAB_APPROVAL.json") and
                attempt["identity"] == final["identity"] == self.identity(), "Colab validation identity changed")
        budget(read(manifest["setup_ledger_path"]), final["elapsed_seconds"])
        return {n: sha(self.validation / n) for n in ("ATTEMPT.json", "RESULT.json", "report.json", "progress.jsonl")}

    @contextmanager
    def alarm(self, seconds):
        def stop(signum, frame): raise InterruptedError("Colab owned process signal " + str(signum))
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)}
        for s in previous: signal.signal(s, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, seconds))
        try: yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, h in previous.items(): signal.signal(s, h)

    def validate(self):
        start = time.monotonic()
        protocol, manifest = self.check(approval=True)
        identity = self.identity()
        budget(read(manifest["setup_ledger_path"]))
        require(not self.output.exists(), "Scientific Colab attempt already exists")
        self.validation.mkdir(exist_ok=False)
        save(self.validation / "ATTEMPT.json", {"identity": identity, "started_est": datetime.now(self.L.EST).isoformat(),
            "manifest_sha256": sha(self.directory / "COLAB_MANIFEST.json"),
            "approval_sha256": sha(self.directory / "COLAB_APPROVAL.json"), "process_cap_seconds": 120})
        result = dict(status="failed_colab_validation", identity=identity, identity_unchanged=False, error=None,
                      no_automatic_retry=True, new_work=ZERO_WORK)
        end = start + 105
        def deadline(): require(time.monotonic() < end, "Colab validation deadline exceeded")
        def persist(event):
            with (self.validation / "progress.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(self.L.plain(event), allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
        try:
            with self.alarm(end-time.monotonic()):
                persist({"stage": "colab_compatibility_start", "new_work": ZERO_WORK})
                deadline()
                binding, training, services, review = self.S._services(protocol)
                compatibility = self.L.load("_colab_frozen_compatibility", self.here / "titan_compatibility.py")
                worker = self.L.load("_colab_frozen_worker", self.here / "worker.py")
                old = compatibility.EXPECTED_EXECUTABLE
                try:
                    # Explicit transport path only. All package/device/numerical/restore checks are unchanged.
                    compatibility.EXPECTED_EXECUTABLE = manifest["executable"]
                    report = compatibility.check_compatibility(protocol, binding, training, services, review,
                        worker, deadline, persist, warmup_dir=self.R.OLD / self.R.REUSED)
                finally: compatibility.EXPECTED_EXECUTABLE = old
                save(self.validation / "report.json", report)
                require(report["passed"] is True and report["new_work"] == ZERO_WORK and
                        report["status"] == "passed_zero_update_titan_compatibility", "Compatibility did not pass")
                deadline(); self.check(approval=True)
                require(self.identity() == identity, "Colab identity changed")
                result.update(status="passed_colab_validation", identity_unchanged=True,
                              report_sha256=sha(self.validation / "report.json"))
        except BaseException as exc:
            result.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        result["elapsed_seconds"] = time.monotonic()-start
        if result["elapsed_seconds"] > 115: result.update(status="failed_colab_validation", error="Validation time limit exceeded")
        save(self.validation / "RESULT.json", result)
        return result

    @contextmanager
    def recovery_bindings(self, proof):
        old = self.R.OUT, self.R.check, self.R.old_processes_absent
        def checked(approval=False):
            protocol, _ = self.check(approval=approval)
            require(self.proof() == proof, "Colab validation proof changed")
            return protocol
        self.R.OUT, self.R.check, self.R.old_processes_absent = self.output, checked, self.identity
        try: yield
        finally: self.R.OUT, self.R.check, self.R.old_processes_absent = old

    def worker(self, job_id):
        self.check(approval=True); proof = self.proof()
        job = read(self.output / job_id / "JOB.json")
        require(job["colab_manifest_sha256"] == sha(self.directory / "COLAB_MANIFEST.json") and
                job["colab_validation_proof"] == proof, "Worker is not bound to Colab approval/proof")
        with self.recovery_bindings(proof): return self.R.worker_main(job_id)

    def child(self, job, directory, end, log):
        process = None
        try:
            require(time.monotonic() < end, "Deadline exhausted before dispatch")
            process = subprocess.Popen([sys.executable, "-B", "-u", str(self.directory / "colab_profile.py"),
                "--repo", str(self.repo), "--worker", job["id"]], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(directory / "PROCESS.json", {"pid": process.pid})
            return process.wait(timeout=max(.001, end-time.monotonic()))
        except BaseException:
            if process is not None:
                if process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                        process.wait()
                else: process.wait()
            raise

    def execute(self):
        start = time.monotonic()
        _, manifest = self.check(approval=True)
        identity, proof = self.identity(), self.proof()
        cap = budget(read(manifest["setup_ledger_path"]), read(self.validation / "RESULT.json")["elapsed_seconds"])["scientific_cap_seconds"]
        end = start + cap - 15
        smoke = self.S.smoke_identity()
        prior_steps = 189655 + sum(self.S.smoke_passed()["physical"]["counts"]["completed"].values())
        self.output.mkdir(exist_ok=False)
        save(self.output / "ATTEMPT.json", {"identity": identity, "supervisor_pid": os.getpid(),
            "colab_manifest_sha256": sha(self.directory / "COLAB_MANIFEST.json"), "colab_validation_proof": proof,
            "recovery_manifest_sha256": sha(self.here / "RECOVERY_MANIFEST.json"), "smoke_proof": smoke,
            "science_cap_seconds_after_all_setup_and_waits": cap, "started_est": datetime.now(self.L.EST).isoformat()})
        accepted, current = [], None
        result = dict(status="incomplete_colab_recovery", jobs=[], error=None, no_automatic_retry=True,
                      training_hold_remains=True, science_cap_seconds_after_all_setup_and_waits=cap)
        try:
            with self.alarm(end-time.monotonic()), self.recovery_bindings(proof):
                self.R.copy_warmup()
                for job in self.R.jobs():
                    self.R.check(approval=True); self.R.verify_mirror()
                    require(time.monotonic() < end and self.identity() == identity and self.S.smoke_identity() == smoke,
                            "Deadline/source/runtime proof changed")
                    directory = self.output / job["id"]; directory.mkdir(exist_ok=False)
                    save(directory / "JOB.json", {**job, "remaining_seconds": end-time.monotonic(),
                        "recovery_manifest_sha256": sha(self.here / "RECOVERY_MANIFEST.json"), "smoke_proof": smoke,
                        "colab_manifest_sha256": sha(self.directory / "COLAB_MANIFEST.json"), "colab_validation_proof": proof})
                    current = job
                    print(json.dumps({"status": "starting_colab_job", "id": job["id"]}), flush=True)
                    with (directory / "worker.log").open("x", encoding="utf-8") as log:
                        rc = self.child(job, directory, end, log)
                    result["jobs"].append({"id": job["id"], "exit_code": rc})
                    accepted.append(self.L.accepted_job(directory, rc)); current = None
                    save(directory / "SUPERVISOR.json", {"status": "accepted", "exit_code": rc})
                    print(json.dumps({"status": "accepted_colab_job", "id": job["id"]}), flush=True)
                self.R.check(approval=True); self.R.verify_mirror()
                require(time.monotonic() < end, "Final acceptance exceeded deadline")
                result.update(status="completed_colab_path_a_recovery", new_execution_counts_known=True)
        except BaseException as exc:
            result.update(error=type(exc).__name__ + ": " + str(exc), new_execution_counts_known=False)
        result.update(self.R.accounting(accepted, current, prior_simulator_steps=prior_steps,
            evaluation_output=self.output / "evaluation"), elapsed_seconds=time.monotonic()-start,
            all_declared_new_jobs_completed=result["status"] == "completed_colab_path_a_recovery")
        save(self.output / "RESULT.json", result)
        return result


def prepare(repo, migration_status_path, setup_ledger_path, executable):
    p = Profile(repo)
    p.original_check(approval=False); p.no_other_attempt()
    status_path, ledger_path = Path(migration_status_path).resolve(), Path(setup_ledger_path).resolve()
    status = migration(read(status_path)); budget(read(ledger_path))
    require(PurePosixPath(executable).is_absolute(), "Select the actual isolated Linux interpreter")
    require(not any(x.exists() for x in (p.directory / "COLAB_MANIFEST.json", p.validation, p.output)), "No reseal/reset/retry")
    paths = [p.directory / "colab_profile.py", status_path, ledger_path] + [Path(e["path"]) for e in status["evidence"]]
    save(p.directory / "COLAB_MANIFEST.json", {"schema": "path_a.colab_profile.v1", "repository": str(p.repo),
        "executable": str(executable), "titan_manifest_sha256": TITAN_MANIFEST_SHA256, "jobs": p.R.jobs(),
        "migration_status_path": str(status_path), "setup_ledger_path": str(ledger_path),
        "files": {str(x): sha(x) for x in paths}, "fresh_colab_approval_required": True,
        "source_status_is_operator_evidence_not_distributed_lock": True,
        "validation_seconds": 120, "maximum_scientific_seconds": 10480, "prior_reserved_seconds": 60})
    return {"status": "prepared_not_authorized_or_executed", "manifest_sha256": sha(p.directory / "COLAB_MANIFEST.json")}


def authorize(repo, operator, statement):
    p = Profile(repo); _, m = p.check()
    require(statement == APPROVAL_STATEMENT and operator == read(m["migration_status_path"])["operator"], "Explicit Colab authorization required")
    require(not p.validation.exists() and not p.output.exists(), "Never authorize a prior attempt")
    save(p.directory / "COLAB_APPROVAL.json", {"approve_colab_recovery": True, "statement": statement,
        "operator": operator, "manifest_sha256": sha(p.directory / "COLAB_MANIFEST.json"),
        "authorized_est": datetime.now(p.L.EST).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repo", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "validate", "execute"): group.add_argument("--" + name, action="store_true")
    group.add_argument("--worker")
    args = parser.parse_args(); profile = Profile(args.repo)
    if args.check:
        profile.check(); result = {"status": "source_checked_only", "target_runtime_verified": False}
    elif args.validate: result = profile.validate()
    elif args.worker: result = profile.worker(args.worker)
    else: result = profile.execute()
    print(json.dumps(result), flush=True)
    if args.validate: raise SystemExit(0 if result["status"] == "passed_colab_validation" else 1)
    if args.execute: raise SystemExit(0 if result["status"] == "completed_colab_path_a_recovery" else 1)
    if args.worker: raise SystemExit(0 if result["status"] == "completed" else 1)
