"""Cheap isolated supervisor fixtures; no JAX, real dataset reads or simulator."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import validation_supervisor as retry


class RetryTests(unittest.TestCase):
    def setUp(self):
        if os.name == 'nt':
            # This managed Windows sandbox denies os.replace even in its temp
            # directory (WinError 5). Test state-machine logic using ordinary
            # fixture writes; do not claim target atomic-rename validation.
            def fixture_json(path, value):
                Path(path).write_text(json.dumps(value), encoding='utf-8')
            writer = patch.object(retry, 'atomic_json', fixture_json)
            writer.start()
            self.addCleanup(writer.stop)

    def fixture(self, root):
        root = Path(root)
        wrapper = root / 'retry'
        wrapper.mkdir()
        manifest = {'original_protocol_sha256': 'fixture', 'original_prepared_sha256': 'fixture', 'archive_directory': 'archived'}
        receipt = {'prior_supervisor_elapsed_seconds_lower_bound': 19.75672,
                   'prior_attempts': [{'ordinal': 1, 'final_counts_known': False}, {'ordinal': 2, 'final_counts_known': True}]}
        return root, wrapper, manifest, receipt

    def test_atomic_reservation_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, wrapper, manifest, receipt = self.fixture(tmp)
            output = retry.reserve(root, wrapper, manifest, receipt)
            saved = retry.read(output / 'RESERVATION.json')
            self.assertEqual(saved['attempt_ordinal'], 3)
            self.assertEqual(saved['prior_archive'], 'archived')
            self.assertFalse(saved['production_attempt_reserved'])
            self.assertEqual(saved['worker_cap_seconds'], 900)
            with self.assertRaises(ValueError):
                retry.reserve(root, wrapper, manifest, receipt)
            self.assertFalse((root / 'ATTEMPT.json').exists())

    def test_one_shot_marker_survives_missing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, wrapper, manifest, receipt = self.fixture(tmp)
            (wrapper / 'VALIDATION_ATTEMPT_3.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                retry.reserve(root, wrapper, manifest, receipt)
            self.assertFalse((root / 'runtime_validation_v1').exists())

    def test_failure_before_worker_entry_gets_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail(*args, **kwargs):
                raise OSError('fixture spawn failure')
            outcome = retry.supervise([], Path(tmp), time.monotonic()+5, retry.Signals(), popen=fail)
            self.assertIn('fixture spawn failure', outcome['error'])
            self.assertIsNone(outcome['exit_code'])
            self.assertTrue((Path(tmp) / 'CHILD_OUTCOME.json').exists())

    def test_timeout_reaps_real_fixture_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = retry.supervise([sys.executable, '-c', 'import time; time.sleep(30)'],
                Path(tmp), time.monotonic()+.3, retry.Signals())
            self.assertTrue(outcome['timed_out'])
            self.assertIsNotNone(outcome['exit_code'])
            self.assertLess(outcome['elapsed_seconds'], 5)

    def test_signals_during_popen_and_cleanup_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = retry.Signals()
            children = []
            def spawning(*args, **kwargs):
                child = subprocess.Popen(*args, **kwargs)
                children.append(child)
                state.record(signal.SIGINT)
                state.record(signal.SIGINT)
                return child
            original = retry.launch.prior.terminate_owned
            def cleanup(child):
                state.record(signal.SIGINT)
                original(child)
            with patch.object(retry.launch.prior, 'terminate_owned', cleanup):
                outcome = retry.supervise([sys.executable, '-c', 'import time; time.sleep(30)'],
                    Path(tmp), time.monotonic()+5, state, popen=spawning)
            self.assertIsNotNone(children[0].poll())
            self.assertTrue(outcome['interrupted'])
            self.assertEqual(len(outcome['signals']), 3)
            self.assertTrue((Path(tmp) / 'CHILD_OUTCOME.json').exists())

    def test_signal_before_spawn_dispatches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = retry.Signals()
            state.record(signal.SIGINT)
            with patch.object(retry.subprocess, 'Popen') as spawn:
                outcome = retry.supervise([], Path(tmp), time.monotonic()+5, state, popen=spawn)
            spawn.assert_not_called()
            self.assertTrue(outcome['interrupted'])

    def test_postcheck_interruption_keeps_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest, receipt = self.fixture(tmp)
            retry.atomic_json(root / 'report.json', {'status': 'passed_synthetic_runtime_validation',
                'execution_counts_known': True})
            state = retry.Signals()
            def interrupted(*args):
                provisional = retry.read(root / 'RESULT.json')
                self.assertEqual(provisional['status'], 'child_finished_postcheck_pending')
                self.assertFalse(provisional['passed'])
                state.record(signal.SIGINT)
                state.record(signal.SIGINT)
                raise KeyboardInterrupt('fixture postcheck interruption')
            result = retry.finalize(root, dict(exit_code=0, timed_out=False, interrupted=False,
                error=None, elapsed_seconds=3, signals=[]), manifest, receipt,
                time.monotonic()+5, state, check=interrupted)
            self.assertFalse(result['passed'])
            self.assertTrue(result['interrupted'])
            self.assertIn('KeyboardInterrupt', result['postcheck_error'])
            self.assertEqual(retry.read(root / 'RESULT.json')['status'], 'worker_finished')

    def test_success_does_not_erase_unknown_prior_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest, receipt = self.fixture(tmp)
            retry.atomic_json(root / 'report.json', {'status': 'passed_synthetic_runtime_validation',
                'execution_counts_known': True})
            result = retry.finalize(root, dict(exit_code=0, timed_out=False, interrupted=False,
                error=None, elapsed_seconds=3, signals=[]), manifest, receipt,
                time.monotonic()+5, retry.Signals(), check=lambda *args: True)
            self.assertTrue(result['passed'])
            self.assertTrue(result['execution_counts_known'])
            self.assertFalse(result['cumulative_execution_counts_known'])
            self.assertIsNone(result['cumulative_elapsed_seconds'])
            self.assertAlmostEqual(result['cumulative_elapsed_seconds_lower_bound'], 22.75672)
            self.assertTrue(result['training_hold_remains'])

    def test_no_report_preserves_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest, receipt = self.fixture(tmp)
            retry.atomic_json(root / 'progress.json', {'stage': 'synthetic_initialization'})
            result = retry.finalize(root, dict(exit_code=-9, timed_out=True, interrupted=False,
                error=None, elapsed_seconds=3, signals=[]), manifest, receipt,
                time.monotonic()+5, retry.Signals(), check=lambda *args: True)
            self.assertFalse(result['passed'])
            self.assertFalse(result['execution_counts_known'])
            self.assertEqual(result['accounting_scope'], 'last_progress_lower_bound')

    def test_postcheck_deadline_and_signal_stop_before_bulk_read(self):
        with self.assertRaises(TimeoutError):
            retry.postcheck({'original_prepared_sha256': 'unused'}, time.monotonic()-1, retry.Signals())
        state = retry.Signals()
        state.record(signal.SIGINT)
        with self.assertRaises(InterruptedError):
            retry.postcheck({'original_prepared_sha256': 'unused'}, time.monotonic()+5, state)

    def test_real_signal_handler_is_record_only_and_restored(self):
        previous = signal.getsignal(signal.SIGINT)
        with retry.Signals() as state:
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            self.assertEqual(state.received, [signal.SIGINT, signal.SIGINT])
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    def test_changed_retry_manifest_cannot_pass_postcheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retry.atomic_json(root / 'PREPARED.json', {'files': {}, 'dependencies': {}, 'inputs': {}})
            manifest = {'original_prepared_sha256': retry.sha(root / 'PREPARED.json'), 'files': {'retry.py': 'original'}}
            changed = dict(manifest, files={'retry.py': 'changed_and_resealed'})
            with patch.object(retry, 'ROOT', root), patch.object(retry, 'verify_retry', return_value=(changed, {})):
                with self.assertRaisesRegex(ValueError, 'Retry manifest changed'):
                    retry.postcheck(manifest, time.monotonic()+5, retry.Signals())

    def test_cleanup_failure_does_not_claim_child_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest, receipt = self.fixture(tmp)
            retry.atomic_json(root / 'report.json', {'status': 'passed_synthetic_runtime_validation',
                'execution_counts_known': True})
            result = retry.finalize(root, dict(exit_code=0, timed_out=False, interrupted=False,
                error=None, cleanup_error='fixture reap failed', elapsed_seconds=3, signals=[]),
                manifest, receipt, time.monotonic()+5, retry.Signals(), check=lambda *args: True)
            self.assertFalse(result['passed'])
            self.assertFalse(result['execution_counts_known'])
            self.assertEqual(result['status'], 'child_cleanup_failed_liveness_uncertain')

    def test_supervisor_error_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest, receipt = self.fixture(tmp)
            retry.atomic_json(root / 'report.json', {'status': 'passed_synthetic_runtime_validation',
                'execution_counts_known': True})
            result = retry.finalize(root, dict(exit_code=0, timed_out=False, interrupted=False,
                error='fixture supervisor failure', elapsed_seconds=3, signals=[]),
                manifest, receipt, time.monotonic()+5, retry.Signals(), check=lambda *args: True)
            self.assertFalse(result['passed'])

    def test_actual_cli_help_is_import_safe(self):
        result = subprocess.run([sys.executable, '-B', str(Path(retry.__file__)), '--help'],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--execute-validation', result.stdout)


if __name__ == '__main__':
    unittest.main()
