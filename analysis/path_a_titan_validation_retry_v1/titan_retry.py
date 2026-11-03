"""Additive supervised Titan validation retry; frozen scientific code is unchanged."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
EST = timezone(timedelta(hours=-5))
SCHEMA = "path_a.titan_validation_retry.v1"
INSTALL_SCHEMA = "path_a.titan_validation_retry_install.v1"
TARGET_REPO = "/mnt/c/Users/David Bayha/BCA-PathA"
TARGET_PYTHON = "/home/dbayha/bca-gpu/bin/python"
PRIOR_PID = 34106
TITAN_MANIFEST = "32718856c6b4714ea1a4004554d61b80223d502e3517b5563c372a97047a15ed"
PATH_A = Path("experiments/thrusterchain_bca/two_scenario_path_a")
OWN = Path("analysis/path_a_titan_validation_retry_v1")
BOOT_PINS = {
    "launch.py": "7760811f7b0994b2a407f68ae59e703447acc9beefa79fc7b853a3c889fd0609",
    "smoke_launch.py": "17ea43520a3362413d4504e692b6cc6327aac4d929b2aac9fc5b2c1ade32faf8",
    "recovery_launch.py": "b661ae7a969a166d0647950486d8567c84c945f9b194cd0b4158fbbf787774a6",
    "titan_launch.py": "d3063e8e49dbdc41373f79e8bcd237d10f22e3b9af159c1dbf9ce80c02a61538",
}
ZERO_WORK = dict(critic=0, actor=0, auxiliary=0, calibrations=0, simulator_steps=0, reset_calls=0)
APPROVAL_TEXT = (
    "I approve one supervised zero-update Titan validation retry, preserving validation attempt 1 "
    "and all frozen Path A inputs. Only a complete passing attempt 2 authorizes the previously "
    "approved unchanged 17 training jobs plus evaluation in a distinct reservation. The remaining "
    "10660 seconds is not reset: charge 60 seconds previous preparation, 120 seconds validation 1, "
    "at least 60 seconds new preparation and 120 seconds validation 2; production is at most "
    "10300 seconds and additional measured or bounded preparation execution reduces it. Idle time "
    "between commands is not charged. No additional scientific work, "
    "automatic retry, helper substitution or modified numerical criteria is authorized.")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(EST).isoformat()


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def allocation():
    return {"shared_remaining_seconds": 10660, "prior_preparation_reserved_seconds": 60,
            "prior_validation_charged_seconds": 120, "new_preparation_minimum_seconds": 60,
            "retry_validation_charged_seconds": 120, "validation_child_deadline_seconds": 105,
            "validation_external_timeout_seconds": 115, "kill_grace_seconds": 5,
            "maximum_production_seconds": 10300, "production_internal_margin_seconds": 15,
            "production_external_margin_seconds": 5, "new_validation_work": dict(ZERO_WORK),
            "preparation_receipt_tail_bound_seconds": 5, "dispatch_receipt_exec_tail_bound_seconds": 5,
            "time_accounting": "per_execution_phase_wall_time_and_explicit_bounds; no_between_command_idle_charge",
            "training_jobs": 17, "evaluation_jobs": 1, "new_evaluation_transition_cap": 237600,
            "cumulative_optimizer_caps": {"critic": 190014, "actor": 95008, "auxiliary": 190014, "calibrations": 22},
            "cumulative_simulator_ceiling": 498400, "no_automatic_retry": True,
            "scientific_functions_changed": False}


def approval_expected(repo):
    return {"approval": APPROVAL_TEXT, "retry_manifest_sha256": sha(Path(repo) / OWN / "RETRY_MANIFEST.json"),
            "validation_attempt": 2, "prior_pid": PRIOR_PID, "zero_update_retry_only": True,
            "conditional_existing_recovery": True}


def remaining(ledger, dispatch=None):
    dispatch = dispatch or {"execution_seconds": 0, "receipt_exec_tail_bound_seconds": 0}
    numbers = (ledger["installation_elapsed_seconds"], ledger["unrecorded_setup_upper_seconds"],
               ledger["prepare_execution_seconds"], ledger["prepare_receipt_tail_bound_seconds"],
               dispatch["execution_seconds"], dispatch["receipt_exec_tail_bound_seconds"])
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                for x in numbers), "Unknown or negative preparation timing")
    require(ledger["prepare_receipt_tail_bound_seconds"] == 5
            and dispatch["receipt_exec_tail_bound_seconds"] in (0, 5), "Execution tail bound differs")
    prep = sum(numbers)
    charge = math.ceil(max(60, prep))
    cap = 10360 - charge
    require(0 < cap <= 10300, "Remaining approved process allowance exhausted")
    return {"production_cap_seconds": cap, "preparation_charged_seconds": charge,
            "preparation_elapsed_and_upper_bound_seconds": prep,
            "upper_bound_is_measured_runtime": False, "idle_between_commands_charged": False,
            "validation_charged_separately_not_double_counted": True,
            "validation_1_charged_seconds": 120, "validation_2_charged_seconds": 120,
            "previous_preparation_reserved_seconds": 60}


def dispatch_production(repo, started=None):
    """Arm the timeout before frozen imports/hashing; inner supervisor authenticates."""
    started = time.monotonic() if started is None else started
    repo = Path(repo).resolve()
    here, validation = repo / OWN, repo / PATH_A / "titan_validation_2"
    ledger = read(here / "RETRY_LEDGER.json")
    result = read(validation / "RESULT.json")
    require(result.get("passed") is True, "No passing validation receipt for dispatch")
    timeout = shutil.which("timeout")
    require(timeout is not None, "GNU timeout is required for the external production deadline")
    receipt = {"schema": SCHEMA, "started_est": now(), "ledger_sha256": sha(here / "RETRY_LEDGER.json"),
               "validation_result_sha256": sha(validation / "RESULT.json"),
               "execution_seconds": time.monotonic() - started, "receipt_exec_tail_bound_seconds": 5,
               "tail_bound_is_measured_runtime": False, "no_automatic_retry": True}
    budget = remaining(ledger, receipt)
    require(budget["production_cap_seconds"] > 15, "Insufficient allowance for production supervision")
    save(here / "DISPATCH_RECEIPT.json", receipt)
    command = [timeout, "--signal=TERM", "--kill-after=5s",
               str(budget["production_cap_seconds"] - 5) + "s", sys.executable, "-B", "-u",
               str(here / "titan_retry.py"), "--production-supervisor"]
    # Preliminary metadata sets only a <=10300-second ceiling, never permission.
    # No user pause or frozen source import occurs before this external clock.
    tail = time.monotonic() - started - receipt["execution_seconds"]
    if not 0 <= tail <= 5:
        save(here / "DISPATCH_FAILED.json", {"status": "failed_execution_tail", "observed_tail_seconds": tail})
        raise ValueError("Dispatch completion exceeded its reserved execution tail; preserve receipt")
    save(here / "DISPATCH_COMPLETED.json", {"status": "completed_dispatch_checks",
         "receipt_sha256": sha(here / "DISPATCH_RECEIPT.json"), "checked_tail_seconds": tail})
    os.execv(timeout, command)
    raise RuntimeError("External production timeout did not replace this process")


def load_engine(repo):
    root = Path(repo) / PATH_A
    require(sha(root / "TITAN_MANIFEST.json") == TITAN_MANIFEST, "Original Titan manifest differs")
    for name, digest in BOOT_PINS.items():
        require(sha(root / name) == digest, "Frozen launcher differs: " + name)
    for name in ("launch", "smoke_launch", "recovery_launch"):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == (root / (name + ".py")).resolve(),
                    "Imported launcher belongs to another checkout: " + name)
    spec = importlib.util.spec_from_file_location("_path_a_retry_frozen_titan", root / "titan_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Retry:
    def __init__(self, repo=None, engine=None, started=None):
        self.invoked = time.monotonic() if started is None else started  # Before frozen imports/hashing.
        self.repo = Path(repo or Path(__file__).resolve().parents[2]).resolve()
        self.here, self.study = self.repo / OWN, self.repo / PATH_A
        self.first, self.validation = self.study / "titan_validation_1", self.study / "titan_validation_2"
        self.out = self.study / "run_titan_recovery_validation2"
        self.T = engine or load_engine(self.repo)
        self.R, self.L, self.S = self.T.R, self.T.L, self.T.S
        self.frozen_check = self.T.check

    def conflicts(self):
        require(not next(self.repo.rglob("path_a_a1.py"), None),
                "Unreviewed path_a_a1.py exists; do not use or overwrite it")
        forbidden = (self.study / "run_titan_recovery_1", self.study / "run_recovery_1")
        require(not any(p.exists() for p in forbidden), "Another production reservation exists")
        for pattern in ("colab_validation_*", "run_colab_recovery_*"):
            require(not any(self.study.glob(pattern)), "Another machine/profile attempt exists")
        require(all(p in (self.first, self.validation) for p in self.study.glob("titan_validation_*")),
                "Another Titan validation attempt exists")
        require(all(p == self.out for p in self.study.glob("run_titan_recovery_*")),
                "Another Titan recovery attempt exists")

    def first_identity(self):
        require(self.first.is_dir() and {p.name for p in self.first.iterdir()} == {"ATTEMPT.json", "progress.jsonl"},
                "First validation is not the reviewed incomplete two-file attempt")
        attempt = read(self.first / "ATTEMPT.json")
        require(attempt["pid"] == PRIOR_PID and attempt["titan_manifest_sha256"] == TITAN_MANIFEST
                and attempt["titan_approval_sha256"] == sha(self.study / "TITAN_APPROVAL.json"),
                "First validation lineage changed")
        events = [json.loads(line) for line in (self.first / "progress.jsonl").read_text().splitlines() if line.strip()]
        require(events and events[-1].get("stage") == "before_template_initialization",
                "First validation advanced beyond the reviewed stage")
        require(events[-1]["counts"]["template_initializations_completed"] == 0,
                "First validation model work no longer matches the reviewed evidence")
        return {name: sha(self.first / name) for name in ("ATTEMPT.json", "progress.jsonl")}

    def source_check(self, approval=True):
        protocol = self.frozen_check(approval=True)
        self.conflicts()
        manifest = read(self.here / "RETRY_MANIFEST.json")
        require(manifest["schema"] == SCHEMA and manifest["titan_manifest_sha256"] == TITAN_MANIFEST
                and manifest["allocation"] == allocation(), "Retry allocation/lineage changed")
        for relative, digest in manifest["files"].items():
            path = (self.repo / relative).resolve()
            require(path.is_relative_to(self.here) and sha(path) == digest, "Retry file differs: " + relative)
        require((OWN / "titan_retry.py").as_posix() in manifest["files"], "Retry implementation is not sealed")
        if approval:
            require(read(self.here / "RETRY_APPROVAL.json") == approval_expected(self.repo),
                    "Manifest-bound retry approval is missing or changed")
        install = read(self.here / "INSTALL_RECEIPT.json")
        require(install["schema"] == INSTALL_SCHEMA and install["status"] == "installed_without_overwriting"
                and install["repository"] == TARGET_REPO
                and install["manifest_sha256"] == sha(self.here / "RETRY_MANIFEST.json")
                and install["timing_known"] is True and install["path_a_a1_absent"] is True
                and install["first_validation_pid"] == PRIOR_PID and install["no_scientific_work"] is True
                and install["first_validation_files"] == self.first_identity(),
                "Installer evidence is incomplete or inconsistent")
        duration = install["installation_elapsed_seconds"]
        require(isinstance(duration, (int, float)) and not isinstance(duration, bool)
                and math.isfinite(duration) and duration >= 0, "Installation timing unknown")
        return protocol

    def target(self):
        return self.T.target_identity()

    def prior_inactive(self):
        self.target()
        try:
            os.kill(PRIOR_PID, 0)
        except ProcessLookupError:
            return {"pid": PRIOR_PID, "absent_at_est": now(), "scope": "recorded Titan PID only"}
        except PermissionError as exc:
            raise ValueError("Cannot establish recorded Titan PID inactivity") from exc
        raise ValueError("Recorded Titan PID is present (possibly reused); no retry dispatch")

    def bindings(self):
        return {name: sha(self.here / name) for name in
                ("RETRY_MANIFEST.json", "RETRY_APPROVAL.json", "INSTALL_RECEIPT.json",
                 "RETRY_LEDGER.json", "PREPARE_COMPLETED.json")}

    def prepare(self, upper, confirm_attempts, confirm_helper):
        require(confirm_attempts and confirm_helper, "Explicit no-other-attempt and helper-absent/not-used confirmations required")
        require(isinstance(upper, (int, float)) and not isinstance(upper, bool)
                and math.isfinite(upper) and upper >= 0, "Unrecorded setup upper bound is required")
        self.source_check()
        identity, absent = self.target(), self.prior_inactive()
        require(not self.validation.exists() and not self.out.exists(), "Retry already reserved")
        ledger = {"schema": SCHEMA, "boot_id_at_preparation": boot_id(),
                  "prepared_est": now(), "target_identity": identity, "prior_process_observation": absent,
                  "installation_elapsed_seconds": read(self.here / "INSTALL_RECEIPT.json")["installation_elapsed_seconds"],
                  "unrecorded_setup_upper_seconds": upper, "upper_bound_is_measured_runtime": False,
                  "confirmed_no_other_attempts": True, "confirmed_helper_absent_not_used": True,
                  "first_validation_files": self.first_identity(),
                  "bindings": {n: sha(self.here / n) for n in
                               ("RETRY_MANIFEST.json", "RETRY_APPROVAL.json", "INSTALL_RECEIPT.json")}}
        ledger["prepare_execution_seconds"] = time.monotonic() - self.invoked
        ledger["prepare_receipt_tail_bound_seconds"] = 5
        ledger["prepare_tail_bound_is_measured_runtime"] = False
        budget = remaining(ledger)
        save(self.here / "RETRY_LEDGER.json", ledger)
        tail = time.monotonic() - self.invoked - ledger["prepare_execution_seconds"]
        if not 0 <= tail <= 5:
            save(self.here / "PREPARE_FAILED.json", {"status": "failed_execution_tail", "observed_tail_seconds": tail})
            raise ValueError("Preparation completion exceeded its reserved execution tail; preserve ledger")
        save(self.here / "PREPARE_COMPLETED.json", {"status": "completed_preparation_checks",
             "ledger_sha256": sha(self.here / "RETRY_LEDGER.json"), "checked_tail_seconds": tail})
        return {"status": "prepared_not_executed", **budget}

    def check(self):
        protocol = self.source_check()
        ledger = read(self.here / "RETRY_LEDGER.json")
        complete = read(self.here / "PREPARE_COMPLETED.json")
        require(not (self.here / "PREPARE_FAILED.json").exists()
                and complete["status"] == "completed_preparation_checks"
                and complete["ledger_sha256"] == sha(self.here / "RETRY_LEDGER.json")
                and 0 <= complete["checked_tail_seconds"] <= 5, "Preparation never completed within its execution bound")
        require(ledger["schema"] == SCHEMA
                and ledger["first_validation_files"] == self.first_identity()
                and ledger["confirmed_no_other_attempts"] is True
                and ledger["confirmed_helper_absent_not_used"] is True
                and ledger["bindings"] == {n: sha(self.here / n) for n in
                    ("RETRY_MANIFEST.json", "RETRY_APPROVAL.json", "INSTALL_RECEIPT.json")},
                "Preparation ledger is stale, incomplete or changed")
        require(ledger["target_identity"] == self.target(), "Prepared target identity differs")
        return protocol

    @contextmanager
    def signals(self):
        def stopped(signum, frame):
            raise InterruptedError("Retry supervisor signal " + str(signum))
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous = {s: signal.getsignal(s) for s in signals}
        for s in signals:
            signal.signal(s, stopped)
        try:
            yield
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)

    def child(self, command, directory, end, stdout, stderr):
        proc = None
        result = {"exit_code": None, "signal": None, "timed_out": False, "interrupted": False, "error": None}
        try:
            require(time.monotonic() < end, "Deadline reached before child dispatch")
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            save(directory / "PROCESS.json", {"pid": proc.pid, "started_est": now()})
            result["exit_code"] = proc.wait(timeout=max(.001, end - time.monotonic()))
        except BaseException as exc:
            result.update(error=type(exc).__name__ + ": " + str(exc),
                          timed_out=isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)),
                          interrupted=isinstance(exc, (KeyboardInterrupt, InterruptedError)))
        finally:
            if proc is not None:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait(timeout=1)
                result["exit_code"] = proc.wait(timeout=1)
                if result["exit_code"] < 0:
                    result["signal"] = -result["exit_code"]
        return result

    def validation_worker(self):
        self.check()
        self.prior_inactive()
        require(read(self.validation / "ATTEMPT.json")["bindings"] == self.bindings(), "Wrong validation supervisor")
        previous = self.T.VALIDATION
        self.T.VALIDATION = self.validation / "worker"
        try:
            return self.T.validate()
        finally:
            self.T.VALIDATION = previous

    def worker_proof(self):
        directory = self.validation / "worker"
        final, report, attempt = (read(directory / n) for n in ("RESULT.json", "report.json", "ATTEMPT.json"))
        require(final["status"] == "passed_titan_validation" and final["identity_unchanged"] is True
                and 0 <= final["elapsed_seconds"] <= 115
                and final["report_sha256"] == sha(directory / "report.json"),
                "Missing, failed or changed frozen validation result")
        require(report["status"] == "passed_zero_update_titan_compatibility" and report["passed"] is True
                and report["new_work"] == ZERO_WORK, "Frozen zero-update compatibility did not pass")
        require(attempt["titan_manifest_sha256"] == TITAN_MANIFEST
                and attempt["titan_approval_sha256"] == sha(self.study / "TITAN_APPROVAL.json")
                and attempt["target_identity"] == final["target_identity"] == self.target(),
                "Frozen validation target/lineage changed")
        return {n: sha(directory / n) for n in ("ATTEMPT.json", "RESULT.json", "report.json", "progress.jsonl")}

    def validate(self):
        end = self.invoked + 105
        self.target()
        self.validation.mkdir(exist_ok=False)
        result = {"status": "failed_supervised_titan_validation", "passed": False, "error": None,
                  "charged_seconds": 120, "no_automatic_retry": True, "new_work": dict(ZERO_WORK),
                  "partial_inference_counts_known": False,
                  "nested_original_worker_allowance_is_historical_not_authoritative": True}
        try:
            with self.signals():
                self.check()
                self.prior_inactive()
                remaining(read(self.here / "RETRY_LEDGER.json"))
                require(not self.out.exists(), "Production already reserved")
                save(self.validation / "ATTEMPT.json", {"started_est": now(), "pid": os.getpid(),
                     "bindings": self.bindings(), "first_validation_files": self.first_identity(),
                     "allocation": allocation(), "target_identity": self.target()})
                with (self.validation / "stdout.log").open("x") as stdout, (self.validation / "stderr.log").open("x") as stderr:
                    result["child"] = self.child([sys.executable, "-B", "-u", str(self.here / "titan_retry.py"),
                                                 "--validation-worker"], self.validation, end, stdout, stderr)
                child = result["child"]
                require(child["exit_code"] == 0 and not child["error"] and not child["timed_out"] and not child["interrupted"],
                        "Validation child failed; inspect saved exit and stderr")
                result["worker_proof"] = self.worker_proof()
                self.check()
                require(time.monotonic() - self.invoked <= 115, "Validation exceeded receipt deadline")
                result.update(status="passed_supervised_titan_validation", passed=True,
                              partial_inference_counts_known=True, bindings=self.bindings())
        except BaseException as exc:
            result.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        result.update(elapsed_seconds=time.monotonic() - self.invoked, finished_est=now())
        if result["passed"]:
            try:
                result["production_budget_at_receipt"] = remaining(read(self.here / "RETRY_LEDGER.json"))
                result["production_budget_recomputed_at_launch"] = True
            except BaseException as exc:
                result.update(status="failed_supervised_titan_validation", passed=False,
                              error=type(exc).__name__ + ": " + str(exc))
        save(self.validation / "RESULT.json", result)
        return result

    def proof(self):
        final, attempt = read(self.validation / "RESULT.json"), read(self.validation / "ATTEMPT.json")
        child = final.get("child", {})
        require(final["status"] == "passed_supervised_titan_validation" and final["passed"] is True
                and 0 <= final["elapsed_seconds"] <= 115 and child.get("exit_code") == 0
                and not child.get("error") and not child.get("timed_out") and not child.get("interrupted")
                and attempt["bindings"] == final["bindings"] == self.bindings()
                and final["worker_proof"] == self.worker_proof(), "No complete authenticated retry pass")
        return {n: sha(self.validation / n) for n in ("ATTEMPT.json", "RESULT.json", "stdout.log", "stderr.log", "PROCESS.json")}

    @contextmanager
    def recovery_bindings(self, proof):
        previous = self.R.OUT, self.R.check, self.R.old_processes_absent
        def checked(approval=False):
            protocol = self.check()
            require(self.proof() == proof, "Retry proof changed")
            return protocol
        self.R.OUT, self.R.check, self.R.old_processes_absent = self.out, checked, self.target
        try:
            yield
        finally:
            self.R.OUT, self.R.check, self.R.old_processes_absent = previous

    def worker(self, job_id):
        self.check()
        proof = self.proof()
        job, attempt = read(self.out / job_id / "JOB.json"), read(self.out / "ATTEMPT.json")
        require(job["retry_bindings"] == attempt["retry_bindings"] == self.bindings()
                and job["validation2_proof"] == attempt["validation2_proof"] == proof, "Wrong retry production job")
        require(job["dispatch_receipt_sha256"] == attempt["dispatch_receipt_sha256"]
                == sha(self.here / "DISPATCH_RECEIPT.json"), "Dispatch execution receipt changed")
        require(job["dispatch_completed_sha256"] == attempt["dispatch_completed_sha256"]
                == sha(self.here / "DISPATCH_COMPLETED.json"), "Dispatch completion receipt changed")
        with self.recovery_bindings(proof):
            return self.R.worker_main(job_id)

    def execute(self):
        self.check()
        self.prior_inactive()
        proof = self.proof()
        dispatch = read(self.here / "DISPATCH_RECEIPT.json")
        dispatch_digest = sha(self.here / "DISPATCH_RECEIPT.json")
        dispatch_complete = read(self.here / "DISPATCH_COMPLETED.json")
        completed_digest = sha(self.here / "DISPATCH_COMPLETED.json")
        require(not (self.here / "DISPATCH_FAILED.json").exists()
                and dispatch_complete["status"] == "completed_dispatch_checks"
                and dispatch_complete["receipt_sha256"] == dispatch_digest
                and 0 <= dispatch_complete["checked_tail_seconds"] <= 5, "Dispatch did not complete within its execution bound")
        require(dispatch["schema"] == SCHEMA and dispatch["ledger_sha256"] == sha(self.here / "RETRY_LEDGER.json")
                and dispatch["validation_result_sha256"] == sha(self.validation / "RESULT.json")
                and dispatch["receipt_exec_tail_bound_seconds"] == 5, "Dispatch execution receipt changed")
        budget = remaining(read(self.here / "RETRY_LEDGER.json"), dispatch)
        end = self.invoked + budget["production_cap_seconds"] - 15
        smoke, identity = self.S.smoke_identity(), self.target()
        self.out.mkdir(exist_ok=False)
        result = {"status": "incomplete_titan_retry_recovery", "jobs": [], "error": None,
                  "no_automatic_retry": True, "training_hold_remains": True, "budget": budget}
        accepted, current = [], None
        try:
            save(self.out / "ATTEMPT.json", {"started_est": now(), "supervisor_pid": os.getpid(),
                 "retry_bindings": self.bindings(), "validation2_proof": proof, "budget": budget,
                 "dispatch_receipt_sha256": dispatch_digest,
                 "dispatch_completed_sha256": completed_digest,
                 "recovery_manifest_sha256": sha(self.study / "RECOVERY_MANIFEST.json"), "smoke_proof": smoke,
                 "target_identity": identity, "no_automatic_retry": True})
            with self.signals(), self.recovery_bindings(proof):
                self.R.copy_warmup()
                for job in self.R.jobs():
                    self.R.check(approval=True)
                    self.R.verify_mirror()
                    require(self.S.smoke_identity() == smoke and self.target() == identity, "Recovery identity changed")
                    require(sha(self.here / "DISPATCH_RECEIPT.json") == dispatch_digest
                            and sha(self.here / "DISPATCH_COMPLETED.json") == completed_digest,
                            "Dispatch execution receipt changed")
                    left = end - time.monotonic()
                    require(left > 0, "Shared science deadline exhausted")
                    directory = self.out / job["id"]
                    directory.mkdir(exist_ok=False)
                    save(directory / "JOB.json", {**job, "remaining_seconds": left, "retry_bindings": self.bindings(),
                         "validation2_proof": proof, "recovery_manifest_sha256": sha(self.study / "RECOVERY_MANIFEST.json"),
                         "dispatch_receipt_sha256": dispatch_digest,
                         "dispatch_completed_sha256": completed_digest,
                         "smoke_proof": smoke})
                    current = job
                    print(json.dumps({"status": "starting_titan_retry_job", "id": job["id"], "remaining_seconds": left}), flush=True)
                    with (directory / "worker.log").open("x") as log:
                        outcome = self.child([sys.executable, "-B", "-u", str(self.here / "titan_retry.py"), "--worker", job["id"]],
                                             directory, end, log, subprocess.STDOUT)
                    result["jobs"].append({"id": job["id"], **outcome})
                    save(directory / "SUPERVISOR.json", outcome)
                    require(not outcome["error"], "Recovery child supervision failed")
                    accepted.append(self.L.accepted_job(directory, outcome["exit_code"]))
                    current = None
                    print(json.dumps({"status": "accepted_titan_retry_job", "id": job["id"]}), flush=True)
                self.R.check(approval=True)
                self.R.verify_mirror()
                require(sha(self.here / "DISPATCH_RECEIPT.json") == dispatch_digest
                        and sha(self.here / "DISPATCH_COMPLETED.json") == completed_digest,
                        "Dispatch execution receipt changed")
                require(time.monotonic() < end, "Final acceptance exceeded science deadline")
                result.update(status="completed_titan_retry_recovery", new_execution_counts_known=True)
        except BaseException as exc:
            result.update(error=type(exc).__name__ + ": " + str(exc), new_execution_counts_known=False)
        result.update(self.R.accounting(accepted, current, prior_simulator_steps=192975, evaluation_output=self.out / "evaluation"),
                      elapsed_seconds=time.monotonic() - self.invoked, finished_est=now(),
                      all_declared_new_jobs_completed=result["status"] == "completed_titan_retry_recovery")
        save(self.out / "RESULT.json", result)
        return result


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare", "check", "validate", "validation-worker", "execute", "production-supervisor"):
        group.add_argument("--" + flag, action="store_true")
    group.add_argument("--worker")
    parser.add_argument("--unrecorded-setup-upper-seconds", type=float)
    parser.add_argument("--confirm-no-other-attempts", action="store_true")
    parser.add_argument("--confirm-helper-absent-not-used", action="store_true")
    args = parser.parse_args()
    if args.execute:
        dispatch_production(Path(__file__).resolve().parents[2], started=started)
        raise RuntimeError("Production dispatcher unexpectedly returned")
    retry = Retry(started=started)
    if args.prepare:
        result = retry.prepare(args.unrecorded_setup_upper_seconds, args.confirm_no_other_attempts, args.confirm_helper_absent_not_used)
    elif args.check:
        retry.source_check()
        result = {"status": "source_checked_not_executed"}
    elif args.validate:
        result = retry.validate()
    elif args.validation_worker:
        result = retry.validation_worker()
    elif args.worker:
        result = retry.worker(args.worker)
    elif args.production_supervisor:
        result = retry.execute()
    else:
        raise RuntimeError("Unrecognized retry operation")
    print(json.dumps(result), flush=True)
    good = {"source_checked_not_executed", "prepared_not_executed", "passed_supervised_titan_validation",
            "passed_titan_validation", "completed", "completed_titan_retry_recovery"}
    return 0 if result.get("status") in good else 1


if __name__ == "__main__":
    raise SystemExit(main())
