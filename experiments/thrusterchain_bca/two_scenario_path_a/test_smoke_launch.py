import copy
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_launch as M


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class SmokeLaunchTests(unittest.TestCase):
    def fixture(self, root):
        write(root / 'AMENDMENT_MANIFEST.json', {'fixture': True})
        out = root / 'smoke_attempt_1'
        out.mkdir()
        banks = [{'seed_index': s, 'condition': c, 'anchors': 16, 'common_edges': 48,
                  'omitted_count': 16, 'retained_count': 32, 'passed': True}
                 for s in range(3) for c in ('honeypot', 'bottleneck')]
        actual = {'prefix_parity': 2688, 'bank_edges': 288, 'forced_qpi': 40, 'returns': 100}
        report = {'status': 'passed_runtime_smoke', 'passed': True,
            'amendment_manifest_sha256': M.L.sha(root / 'AMENDMENT_MANIFEST.json'),
            'learning': {'status': 'passed_learning_smoke', 'execution_counts_known': True,
                'counts': {'attempted': M.LEARNING_CAPS, 'completed': M.LEARNING_CAPS},
                'restored_actor_next_step_exact': {'native': True, 'bca': True}},
            'physical': {'status': 'completed_technical_evaluation_smoke', 'passed': True,
                'completed_disposable_arms': 2, 'bank_coverage': banks,
                'counts': {'execution_counts_known': True, 'attempted': actual, 'completed': actual}}}
        write(out / 'RESULT.json', {'status': 'passed_runtime_smoke', 'mode': 'smoke',
                                   'execution_counts_known': True, 'elapsed_seconds': 100})
        self.seal_fixture(out, report)
        return out, report

    def seal_fixture(self, out, report):
        write(out / 'report.json', report)
        write(out / 'ARTIFACTS.json', {'report.json': M.L.sha(out / 'report.json')})
        write(out / 'WORKER_EXIT.json', {'status': 'completed', 'identity_unchanged': True,
             'report_sha256': M.L.sha(out / 'report.json'), 'artifacts_sha256': M.L.sha(out / 'ARTIFACTS.json')})

    def test_budget_preserves_total_and_explicit_extra_training(self):
        a = M.allocation()
        self.assertEqual(1950 + 60 + 900 + 11490, 14400)
        self.assertEqual(189655 + 3320 + 237600, a['planned_cumulative_simulator_maximum'])
        self.assertEqual(a['cumulative_optimizer_caps'], {'critic': 180014, 'actor': 90008, 'auxiliary': 180014, 'calibrations': 21})
        self.assertEqual(sum(M.PHYSICAL_CAPS.values()), 3320)
        self.assertEqual(M.L.sha(HERE / 'MANIFEST.json'), M.ORIGINAL_MANIFEST)

    def test_exact_complete_smoke_is_accepted_with_terminal_receipt(self):
        with TemporaryDirectory() as d, patch.object(M, 'HERE', Path(d)):
            out, report = self.fixture(Path(d))
            self.assertEqual(M.smoke_passed(out), report)
            self.assertEqual(set(M.smoke_identity()), {'RESULT.json', 'WORKER_EXIT.json', 'report.json', 'ARTIFACTS.json'})

    def test_no_production_before_smoke_and_no_reservation(self):
        with TemporaryDirectory() as d, patch.object(M, 'HERE', Path(d)), patch.object(M.os, 'name', 'posix'), \
             patch.object(M, 'check', return_value={}), patch.object(M, 'smoke_identity', side_effect=ValueError('No passed smoke')), \
             patch.object(M.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'No passed smoke'):
                M.supervise('production')
            spawn.assert_not_called()
            self.assertFalse((M.HERE / 'run_amendment_1').exists())

    def test_worker_exit_alone_or_wrong_budget_cannot_authorize(self):
        with TemporaryDirectory() as d, patch.object(M, 'HERE', Path(d)):
            out, report = self.fixture(Path(d))
            write(out / 'RESULT.json', {'status': 'incomplete', 'mode': 'smoke', 'execution_counts_known': False, 'elapsed_seconds': 50})
            with self.assertRaises(ValueError): M.smoke_passed(out)
            self.assertTrue(M.smoke_passed(out, require_supervisor=False)['passed'])
            write(out / 'RESULT.json', {'status': 'passed_runtime_smoke', 'mode': 'smoke', 'execution_counts_known': True, 'elapsed_seconds': 901})
            with self.assertRaises(ValueError): M.smoke_passed(out)

    def test_all_banks_and_actual_coverage_calls_are_required(self):
        with TemporaryDirectory() as d, patch.object(M, 'HERE', Path(d)):
            out, original = self.fixture(Path(d))
            for case in ('duplicate_bank', 'empty_stratum', 'zero_physical', 'failed_actor_restore'):
                report = copy.deepcopy(original)
                if case == 'duplicate_bank': report['physical']['bank_coverage'][1] = report['physical']['bank_coverage'][0]
                elif case == 'empty_stratum': report['physical']['bank_coverage'][0].update(omitted_count=0, retained_count=48)
                elif case == 'zero_physical':
                    for field in ('attempted', 'completed'): report['physical']['counts'][field]['prefix_parity'] = 0
                else: report['learning']['restored_actor_next_step_exact']['bca'] = False
                self.seal_fixture(out, report)
                with self.subTest(case=case), self.assertRaises(ValueError): M.smoke_passed(out)

    def test_changed_report_or_raw_evidence_invalidates_pass(self):
        with TemporaryDirectory() as d, patch.object(M, 'HERE', Path(d)):
            out, report = self.fixture(Path(d))
            write(out / 'report.json', {})
            with self.assertRaises(ValueError): M.smoke_passed(out)
            self.seal_fixture(out, report)
            proof = M.smoke_identity()
            terminal = json.loads((out / 'RESULT.json').read_text())
            terminal['elapsed_seconds'] = 101
            write(out / 'RESULT.json', terminal)
            self.assertNotEqual(M.smoke_identity(), proof)

    def test_process_receipt_failure_terminates_and_waits_owned_child(self):
        child = SimpleNamespace(pid=12345, poll=Mock(return_value=None), wait=Mock(return_value=0))
        with patch.object(M.subprocess, 'Popen', return_value=child), \
             patch.object(M.L, 'save', side_effect=OSError('receipt write failed')), \
             patch.object(M.os, 'killpg', create=True) as kill:
            with self.assertRaises(OSError): M.run_child({'id': 'smoke'}, HERE, M.time.monotonic()+5, None)
            kill.assert_called_once_with(12345, signal.SIGTERM)
            child.wait.assert_called_once_with(timeout=5)

    def test_deadline_failure_terminates_exact_child_without_retry(self):
        child = SimpleNamespace(pid=12345, poll=Mock(return_value=None),
                                wait=Mock(side_effect=[M.subprocess.TimeoutExpired('fixture', 1), 0]))
        with patch.object(M.subprocess, 'Popen', return_value=child) as spawn, patch.object(M.L, 'save'), \
             patch.object(M.os, 'killpg', create=True) as kill:
            with self.assertRaises(M.subprocess.TimeoutExpired):
                M.run_child({'id': 'smoke'}, HERE, M.time.monotonic()+1, None)
            spawn.assert_called_once()
            kill.assert_called_once_with(12345, signal.SIGTERM)
            self.assertEqual(child.wait.call_count, 2)


if __name__ == '__main__':
    unittest.main()
