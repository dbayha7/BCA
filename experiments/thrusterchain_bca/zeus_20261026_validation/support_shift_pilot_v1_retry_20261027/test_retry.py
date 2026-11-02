"""Synthetic retry isolation/lineage tests; no real /proc, models or GPU calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch

gate = launch.retry_gate
IDENTITY = {"archive_manifest_sha256": "archive", "origin_prepared_sha256": "origin",
            "protocol_sha256": "protocol", "retry_lineage_sha256": "retry"}
OBSERVED = {"matching_processes": [], "recorded_origin_pids": [{"pid": 372, "status": "absent"}]}


class RetryTests(unittest.TestCase):
    def test_actual_original_science_and_archive_lineage(self):
        identity = gate.lineage()
        self.assertEqual(identity["protocol_sha256"], gate.PROTOCOL_SHA)
        accounting = gate.accounting()
        self.assertEqual(accounting["attempt_ordinal"], 2)
        self.assertTrue(accounting["prior_attempt_counts_are_lower_bounds"])
        self.assertFalse(accounting["prior_attempt_final_counts_known"])
        self.assertEqual(accounting["prior_attempt_last_committed_ledger"]["committed_additional_updates"]["critic"], 200)

    def test_roots_and_subprocess_job_paths_are_retry_only(self):
        self.assertEqual(launch.HERE, Path(launch.__file__).resolve().parent)
        self.assertEqual(launch.HERE.name, "support_shift_pilot_v1_retry_20261027")
        jobs = launch.job_specs(launch.read(launch.HERE / "protocol.json"))
        for job in jobs:
            for field in ("warmup_path", "jobs_root"):
                if field in job:
                    self.assertTrue(Path(job[field]).is_relative_to(launch.HERE / "run"))
        targets = gate.process_targets()
        self.assertIn(launch.HERE / "launch.py", targets)
        self.assertIn(launch.HERE.parent / "support_shift_pilot_v1/launch.py", targets)

    def test_missing_or_malformed_clearance_cannot_launch(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
            root = Path(directory)
            with patch.object(gate, "lineage", return_value=IDENTITY), patch.object(gate, "verify_no_processes") as check:
                with self.assertRaises(FileNotFoundError): gate.require_clearance(root)
                (root / "ORIGIN_STOPPED.json").write_text('{"status":')
                with self.assertRaises(json.JSONDecodeError): gate.require_clearance(root)
                check.assert_not_called()

    def test_check_mode_writes_only_successful_clearance(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
            root = Path(directory)
            with patch.object(launch, "HERE", root), patch.object(launch.sys, "argv", ["launch.py", "--check-origin-stopped"]), \
                 patch.object(launch, "verify_prepared", return_value=({}, {})), patch.object(launch, "verify_inputs"), \
                 patch.object(gate, "lineage", return_value=IDENTITY), patch.object(gate, "verify_no_processes", return_value=OBSERVED), \
                 patch.object(launch, "reserve") as reserve, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.main(), 0)
                reserve.assert_not_called()
            self.assertEqual([p.name for p in root.iterdir()], ["ORIGIN_STOPPED.json"])
            receipt = gate.read(root / "ORIGIN_STOPPED.json")
            self.assertFalse(receipt["attempt_reserved"])
            self.assertFalse(receipt["scientific_execution_started"])
            with patch.object(gate, "lineage", return_value=IDENTITY), patch.object(gate, "verify_no_processes", return_value=OBSERVED) as check:
                self.assertEqual(gate.require_clearance(root), OBSERVED)
                check.assert_called_once_with(root)

    def test_live_or_denied_check_writes_nothing(self):
        for error in (ValueError("existing scoped process"), PermissionError("synthetic denial")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
                root = Path(directory)
                with patch.object(gate, "lineage", return_value=IDENTITY), patch.object(gate, "verify_no_processes", side_effect=error):
                    with self.assertRaises(type(error)): gate.check_origin_stopped(root)
                self.assertEqual(list(root.iterdir()), [])

    def test_user_recorded_pid_receipt_requires_fresh_scoped_check(self):
        receipt = {"schema": "unifloral.support_retry.user_pid_check.v1",
            "status": "recorded_origin_pids_absent_user_report", "checked_utc": "2026-10-27T04:10:00+00:00",
            "source": "trusted_user_terminal_report", "global_scoped_process_absence_verified": False,
            "recorded_origin_pids": [{"pid": 372, "status": "absent"}, {"pid": 486, "status": "absent"}],
            "attempt_reserved": False, "scientific_execution_started": False, **IDENTITY}
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
            root = Path(directory)
            (root / "ORIGIN_STOPPED.json").write_text(json.dumps(receipt))
            with patch.object(gate, "lineage", return_value=IDENTITY), patch.object(gate, "verify_no_processes", return_value=OBSERVED) as check:
                self.assertEqual(gate.require_clearance(root), OBSERVED)
                check.assert_called_once_with(root)
            with patch.object(gate, "lineage", return_value=IDENTITY), patch.object(gate, "verify_no_processes", side_effect=ValueError("live process")):
                with self.assertRaisesRegex(ValueError, "live process"): gate.require_clearance(root)

    def test_user_recorded_pid_receipt_rejects_missing_or_mismatched_pids(self):
        receipt = {"schema": "unifloral.support_retry.user_pid_check.v1",
            "status": "recorded_origin_pids_absent_user_report", "checked_utc": "2026-10-27T04:10:00+00:00",
            "source": "trusted_user_terminal_report", "global_scoped_process_absence_verified": False,
            "attempt_reserved": False, "scientific_execution_started": False, **IDENTITY}
        for rows in (None, [{"pid": 372, "status": "absent"}],
                     [{"pid": 372, "status": "absent"}, {"pid": 487, "status": "absent"}]):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
                root = Path(directory)
                (root / "ORIGIN_STOPPED.json").write_text(json.dumps({**receipt, "recorded_origin_pids": rows}))
                with patch.object(gate, "lineage", return_value=IDENTITY), patch.object(gate, "verify_no_processes") as check:
                    with self.assertRaisesRegex(ValueError, "recorded-PID"): gate.require_clearance(root)
                    check.assert_not_called()

    def test_parent_rejects_gate_before_reservation(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
            root = Path(directory)
            (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            with patch.object(launch, "EXPECTED_ENV", str(root)), patch.object(launch.sys, "prefix", str(root)), \
                 patch.object(launch.sys, "platform", "linux"), patch.object(gate, "require_clearance", side_effect=ValueError("live old worker")), \
                 patch.object(launch, "reserve") as reserve:
                with self.assertRaisesRegex(ValueError, "live old worker"): launch.execute({}, {})
                reserve.assert_not_called()

    def test_scanner_detects_old_and_retry_paths_and_records_reused_pid(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
            proc = Path(directory)
            targets = (launch.HERE.parent / "support_shift_pilot_v1/launch.py", launch.HERE / "launch.py")
            for pid, target in zip((372, 900), targets):
                folder = proc / str(pid); folder.mkdir()
                (folder / "comm").write_text("python\n")
                (folder / "cmdline").write_bytes(b"python\0" + str(target).encode() + b"\0--execute\0")
            other = proc / "486"; other.mkdir(); (other / "comm").write_text("bash\n")
            observed = gate.inspect_processes(proc_root=proc, own_pid=123, own_uid=(proc / "372").stat().st_uid)
            self.assertEqual({r["pid"] for r in observed["matching_processes"]}, {372, 900})
            self.assertEqual(observed["recorded_origin_pids"][1]["status"], "present_non_python")

    def test_scanner_permission_error_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
            proc = Path(directory); folder = proc / "372"; folder.mkdir()
            with patch.object(Path, "read_text", side_effect=PermissionError("synthetic denial")):
                with self.assertRaisesRegex(RuntimeError, "denied"):
                    gate.inspect_processes(proc_root=proc, own_pid=123, own_uid=folder.stat().st_uid)

    def test_changed_science_and_prepared_sources_rejected(self):
        original_sha = gate.sha
        def changed_protocol(path):
            return "changed" if Path(path) == gate.HERE / "protocol.json" else original_sha(path)
        with patch.object(gate, "sha", side_effect=changed_protocol):
            with self.assertRaisesRegex(ValueError, "Scientific protocol changed"): gate.lineage()
        with patch.object(gate, "lineage", return_value=IDENTITY), patch.object(launch, "sha", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "Prepared file changed"): launch.verify_prepared()

    def test_terminal_report_links_prior_accounting_without_merging(self):
        import test_launch
        fixture = test_launch.LaunchTests(methodName="test_all_success_and_terminal_progress")
        fixture.setUp()
        try:
            with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_retry_") as directory:
                code, jobs, report = fixture.sequence(Path(directory))
            self.assertEqual(code, 0)
            self.assertEqual(len(jobs), 17)
            self.assertEqual(report["retry_lineage"]["attempt_ordinal"], 2)
            self.assertFalse(report["retry_lineage"]["prior_attempt_final_counts_known"])
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
