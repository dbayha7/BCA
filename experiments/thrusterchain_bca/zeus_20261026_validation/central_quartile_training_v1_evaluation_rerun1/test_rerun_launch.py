"""Single-job rerun launcher with fake supervisors; no subprocesses, models or WSL calls."""
import contextlib
import io
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch

LINEAGE = {'continuation_prepared_sha256': 'p'*64, 'continuation_terminal_sha256': 't'*64,
           'training_physical_accounting': {'cumulative_recorded_lower_bound': {'critic': 360600}}}
PLAN = {'original_elapsed_seconds': 16451.765985543, 'continuation_elapsed_seconds': 2452.264564082}


class RerunLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.p = launch.read(Path(launch.__file__).parent/'protocol.json')
        for name in ('protocol.json', 'AUTHORIZATION.json', 'RERUN_INPUTS.json', 'RECOVERY.json'):
            launch.save_new(self.root/name, self.p if name == 'protocol.json' else {'fixture_only': name})
        self.env = self.root/'synthetic_env'; self.env.mkdir()
        (self.env/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        handlers = {name: signal.getsignal(getattr(signal, name)) for name in launch.SIGNALS if hasattr(signal, name)}
        self.addCleanup(lambda: [signal.signal(getattr(signal, n), h) for n, h in handlers.items()])

    def sequence(self, *, report_status='completed_central_quartile_evaluation', interrupted=False, cap=7200,
                 overrun_at=None, signal_in_postcheck=False, postcheck_identity_fails=False):
        clock, calls, verifies = [1000.], [], []
        def verify():
            verifies.append(clock[0])
            if len(verifies) == overrun_at: clock[0] += cap+1
            if len(verifies) == 2:
                if signal_in_postcheck: signal.raise_signal(signal.SIGINT)
                if postcheck_identity_fails: raise ValueError('Prepared file changed or escaped: launch.py')
            return self.p, {}
        def supervise(command, output, worker_cap, envvars, job_id, **kwargs):
            dispatch = launch.read(output/'DISPATCH.json'); calls.append(job_id)
            self.assertEqual(command[-1], 'evaluation'); self.assertIn('--worker-token', command)
            self.assertEqual(dispatch['job']['kind'], 'evaluation')
            self.assertTrue(dispatch['job']['jobs_root'].endswith('central_quartile_training_v1_continuation1'+('\\' if '\\' in dispatch['job']['jobs_root'] else '/')+'run'))
            self.assertLessEqual(worker_cap, min(7200, 1000+cap-clock[0]))
            self.assertEqual(kwargs['deadline'], dispatch['deadline_monotonic'])
            clock[0] += 600
            if interrupted:
                return dict(exit_code=-9, timed_out=False, interrupted=True, error='synthetic stop', elapsed_seconds=1)
            launch.save_new(output/'report.json', {'status': report_status, 'execution_counts_known': True,
                'decision': 'completed_fixed_development_study_no_automatic_successor'
                    if report_status.startswith('completed') else 'incomplete_no_utility_conclusion',
                'reproduction': {'tier1_exact': 143, 'tier2_near_exact': 0}, 'ledger': {'optimizer_updates': 0}})
            return dict(exit_code=0 if report_status.startswith('completed') else 1, timed_out=False,
                        interrupted=False, error=None, elapsed_seconds=600)
        recovery = launch.recovery.recovery()
        with contextlib.ExitStack() as stack:
            for target, name, value in ((launch, 'HERE', self.root), (launch, 'EXPECTED_ENV', str(self.env)),
                                        (launch.sys, 'platform', 'linux'), (launch.sys, 'prefix', str(self.env))):
                stack.enter_context(patch.object(target, name, value))
            for target, name, kwargs in (
                    (launch, 'require_execution_authorization', {'return_value': {'fixture_only': True}}),
                    (launch, 'scoped_process_check', {'return_value': None}),
                    (launch, 'verify_prepared', {'side_effect': verify}),
                    (launch, 'verify_inputs', {'return_value': {}}),
                    (launch, 'verify_warmup_pairs', {'return_value': {'paired': True}}),
                    (launch, 'supervise', {'side_effect': supervise}),
                    (launch.recovery, 'verify_prior', {'return_value': recovery}),
                    (launch.recovery, 'recovery', {'return_value': recovery}),
                    (launch.rerun, 'verify_continuation', {'return_value': LINEAGE}),
                    (launch.rerun, 'rerun_seconds', {'return_value': cap}),
                    (launch.rerun, 'plan', {'return_value': PLAN}),
                    (launch.shutil, 'disk_usage', {'return_value': SimpleNamespace(free=10**15)}),
                    (launch.time, 'monotonic', {'side_effect': lambda: clock[0]})):
                stack.enter_context(patch.object(target, name, **kwargs))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = launch.execute(self.p, {})
        return code, calls, launch.read(self.root/'RUN_RESULT.json')

    def test_success_runs_one_evaluation_and_discloses_both_continuations(self):
        code, calls, result = self.sequence()
        self.assertEqual((code, calls, result['status']), (0, ['evaluation'], 'completed_evaluation_rerun'))
        self.assertTrue(result['all_declared_jobs_completed'] and result['execution_counts_known'])
        self.assertTrue(result['prepared_identity_unchanged'])
        self.assertEqual(result['new_training_updates'], 0)
        self.assertEqual(result['reproduction'], {'tier1_exact': 143, 'tier2_near_exact': 0})
        d = result['disclosure']
        self.assertEqual((d['continuations'], d['study_evaluation_attempts']), (2, 2))
        self.assertFalse(d['cumulative_execution_counts_known'])
        self.assertTrue(d['original_calendar_deadline_waived'])
        self.assertEqual(d['training_physical_accounting'], LINEAGE['training_physical_accounting'])
        self.assertEqual(d['repeat_training_work_lower_bound'], {'actor': 300, 'auxiliary': 600, 'critic': 600})
        deadline = launch.read(self.root/'DEADLINE.json')
        self.assertEqual(deadline['seconds_cap'], 7200)
        self.assertAlmostEqual(deadline['unspent_study_allowance_seconds'], 24295.969450375, places=6)
        self.assertEqual(launch.read(self.root/'PROGRESS.json')['status'], 'completed_evaluation_rerun')
        self.assertEqual(result['signals_deferred_after_scientific_work'], [])

    def test_failed_reproduction_or_evaluation_is_incomplete_without_retry(self):
        code, calls, result = self.sequence(report_status='failed_central_quartile_evaluation')
        self.assertEqual((code, calls, result['status']), (1, ['evaluation'], 'stopped_after_incomplete_job'))
        self.assertEqual(result['decision'], 'incomplete_no_utility_conclusion')
        self.assertFalse(result['all_declared_jobs_completed'])

    def test_interrupted_worker_is_not_restarted(self):
        code, calls, result = self.sequence(interrupted=True)
        self.assertEqual((code, calls, result['status']), (1, ['evaluation'], 'stopped_after_incomplete_job'))
        self.assertFalse(result['execution_counts_known'])
        self.assertTrue(result['no_automatic_restart'])

    def test_identity_hashing_time_counts_before_dispatch(self):
        code, calls, result = self.sequence(cap=20, overrun_at=1)
        self.assertEqual((code, calls), (1, []))
        self.assertIn('deadline', result['error'].lower())

    def test_postcheck_past_cap_cannot_report_success(self):
        code, calls, result = self.sequence(cap=700, overrun_at=2)
        self.assertEqual(calls, ['evaluation'])
        self.assertEqual((code, result['status']), (1, 'completed_work_postcheck_exceeded_deadline'))

    def test_interrupt_during_postcheck_is_deferred_and_receipt_is_written(self):
        code, calls, result = self.sequence(signal_in_postcheck=True)
        self.assertEqual(result['signals_deferred_after_scientific_work'], [int(signal.SIGINT)])
        self.assertEqual((code, result['status']), (0, 'completed_evaluation_rerun'))

    def test_changed_identity_in_postcheck_is_reported(self):
        code, _, result = self.sequence(postcheck_identity_fails=True)
        self.assertEqual((code, result['status']), (1, 'prepared_source_changed'))
        self.assertFalse(result['prepared_identity_unchanged'])

    def test_only_the_evaluation_job_exists_and_workers_cannot_train(self):
        self.assertEqual([j['id'] for j in launch.job_specs(self.p)], ['evaluation'])
        launch.save_new(self.root/'ATTEMPT.json', {'token': 'owned', 'supervisor_pid': 123})
        with patch.object(launch, 'HERE', self.root), patch.object(launch, 'EXPECTED_ENV', str(self.env)), \
                patch.object(launch.sys, 'platform', 'linux'), patch.object(launch.sys, 'prefix', str(self.env)), \
                patch.object(launch, 'verify_prepared', return_value=(self.p, {})), \
                patch.object(launch, 'require_execution_authorization', return_value={}), \
                patch.object(launch.os, 'getppid', return_value=123), patch.object(launch, 'module') as loader:
            for job in ('seed2_hole/bca', 'seed0_full/warmup'):
                with self.subTest(job=job), self.assertRaisesRegex(ValueError, 'evaluates only'):
                    launch.worker_entry('owned', job)
            loader.assert_not_called()

    def test_reservation_is_exclusive(self):
        launch.reserve(self.root, 'once', 1000.)
        original = (self.root/'ATTEMPT.json').read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            launch.reserve(self.root, 'twice', 2000.)
        self.assertEqual((self.root/'ATTEMPT.json').read_bytes(), original)


if __name__ == '__main__': unittest.main()
