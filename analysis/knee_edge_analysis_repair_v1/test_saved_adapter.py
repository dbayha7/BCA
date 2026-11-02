"""All inputs here are synthetic fixtures in temporary directories, never experiment data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import analysis_math as M
import analyze_saved as A


def anchor_fixture():
    line = np.array([2.0 * M.X[s] + 3 for s in M.SLOTS])
    clip = np.tile(line, (64, 1))
    clip[:, 9:] -= 0.5
    def receipt(world, clipped):
        return {'evaluator': 'raw_dynamics_loop', 'gamma': 0.98, 'horizon': 100,
                'bootstrap': 'none after step 99',
                'clip': 'all four commands to +-KNEE' if clipped else 'none (deployed command, |a| <= 1 asserted)',
                'continuation': {'world': world, 'physics': {
                    'stall': world == 'stall3', 'stall_slope': 3.0, 'knee': M.X['knee'],
                    'gain': 0.3, 'throttle_cost': 0.3, 'wave_turn': 0.27, 'gamma': 0.98, 'length': 100}}}
    return {'times': M.TIMES, 'gated_slots': list(M.SLOTS), 'nostall_slots': list(M.SLOTS),
            'q1': clip.tolist(), 'qpi_clip': clip.tolist(), 'qpi_unclipped': clip.tolist(),
            'qpi_nostall': np.tile(line, (64, 1)).tolist(), 'valid': [True] * 64,
            'receipt_clip': receipt('stall3', True), 'receipt_unclipped': receipt('stall3', False),
            'receipt_nostall': receipt('nostall', True)}


class SavedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.attempt = self.root / 'approved_attempt'
        self.attempt.mkdir()
        self.contract = {'allowed_attempt': 'approved_attempt', 'a4_manifest_sha256': 'f' * 64,
                         'source_sha256': {}, 'evaluation_source_sha256': {'stage1_eval.py': 'a' * 64}}
        self.root_patch = patch.object(A, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.write('report.json', {'EXPLORATORY': {'disclosure': 'fixture only'}, 'protocol_label': None,
                                  'exploratory_label': 'INCONCLUSIVE', 'finished_utc': 'fixture', 'failed_unit': None})
        self.write('ATTEMPT.json', {'manifest_sha256': 'f' * 64})
        for seed in range(3):
            name = f'evaluate_s{seed}'
            self.write(name + '/job.json', {'id': name, 'kind': 'evaluate', 'seed_index': seed,
                'provenance': 'research', 'integrity': {'files': {
                    '/fixture/knee_edge_ood_v2/stage1_eval.py': 'a' * 64}}})
            self.write(name + '/report.json', {'status': 'completed', 'unit_label': 'COMPLETE', 'job': name,
                'kind': 'evaluate', 'seed_index': seed, 'smoke': None, 'producer_sha256': 'b' * 64,
                'runs': {f's{seed}_{arm}_stall3': {'020000': {'anchors_e2': anchor_fixture()}}
                         for arm in ('cutoff', 'random')}})
            self.write(name + '/SUPERVISOR_RESULT.json', {'completed': True, 'exit_code': 0,
                                                        'timed_out': False, 'interrupted': False})

    def write(self, relative, value):
        p = self.attempt / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding='utf-8')

    def change(self, relative, key, value):
        obj = json.loads((self.attempt / relative).read_text())
        obj[key] = value
        self.write(relative, obj)

    def inspect(self):
        return A.inspect_attempt(self.attempt, self.contract)

    def test_complete_fixture_preserves_inconclusive_protocol_and_no_unlock(self):
        result = self.inspect()
        self.assertTrue(result['v0f_all_seeds_pass'])
        self.assertEqual(result['original_exploratory_label'], 'INCONCLUSIVE')
        self.assertIsNone(result['protocol_decision'])
        self.assertFalse(result['can_unlock_stage2'])
        self.assertFalse(result['producer_fingerprint_independently_authenticated'])
        self.assertEqual(len(result['input_sha256']), 11)

    def test_missing_terminal_waits_without_reading_partial_arrays(self):
        empty = self.root / 'empty_attempt'
        empty.mkdir()
        contract = {**self.contract, 'allowed_attempt': 'empty_attempt'}
        with self.assertRaises(A.WaitingForData): A.inspect_attempt(empty, contract)

    def test_incomplete_attempt_rejected(self):
        self.change('report.json', 'failed_unit', 'evaluate_s2')
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_worker_failure_rejected(self):
        self.change('evaluate_s1/report.json', 'status', 'failed')
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_timeout_rejected(self):
        self.change('evaluate_s1/SUPERVISOR_RESULT.json', 'timed_out', True)
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_smoke_rejected(self):
        self.change('evaluate_s1/job.json', 'smoke', {'budget': 1})
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_seed_override_rejected(self):
        self.change('evaluate_s1/job.json', 'visited_seed', 1)
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_wrong_manifest_rejected(self):
        self.change('ATTEMPT.json', 'manifest_sha256', '0' * 64)
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_nonreproduction_marker_rejected(self):
        self.write('evaluate_s1/NOT_REPRODUCED.json', {})
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_source_digest_mismatch_rejected(self):
        self.change('evaluate_s1/job.json', 'integrity', {'files': {'/fixture/knee_edge_ood_v2/stage1_eval.py': '0' * 64}})
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_malformed_integrity_rejected(self):
        self.change('evaluate_s1/job.json', 'integrity', [])
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_nonmapping_json_rejected(self):
        self.write('evaluate_s1/report.json', [])
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_nonstandard_nonfinite_json_rejected(self):
        self.write('evaluate_s1/report.json', {'value': float('nan')})
        with self.assertRaises(M.InvalidData): self.inspect()

    def test_unapproved_attempt_rejected(self):
        with self.assertRaises(M.InvalidData): A.inspect_attempt(self.root / 'other', self.contract)

    def test_resolved_outside_input_rejected_without_open(self):
        with self.assertRaises(M.InvalidData): A.read_json(self.root / 'outside.json', {}, self.attempt)

    def test_input_change_during_analysis_rejected(self):
        original = A.analyze_seed
        def mutate(*args):
            result = original(*args)
            if args[0] == 2:
                self.change('report.json', 'exploratory_label', 'changed')
            return result
        with patch.object(A, 'analyze_seed', side_effect=mutate):
            with self.assertRaises(M.InvalidData): self.inspect()

    def test_source_change_rejected(self):
        source = self.root / 'source.py'
        source.write_text('first')
        contract = {'source_sha256': {'source.py': A.digest(b'first')}}
        A.check_sources(contract)
        source.write_text('changed')
        with self.assertRaises(M.InvalidData): A.check_sources(contract)

    def test_lineage_change_rejected(self):
        receipt = self.root / 'approval.json'
        receipt.write_text('original')
        contract = {'source_sha256': {}, 'lineage_sha256': {'approval.json': A.digest(b'original')}}
        A.check_sources(contract)
        receipt.write_text('changed')
        with self.assertRaises(M.InvalidData): A.check_sources(contract)

    def test_contract_path_outside_repository_rejected(self):
        contract = {**self.contract, 'allowed_attempt': '../outside'}
        with self.assertRaises(M.InvalidData): A.inspect_attempt(self.root.parent / 'outside', contract)

    def test_output_guard(self):
        with self.assertRaises(M.InvalidData): A.run(self.attempt, self.root / 'outside')
        with patch.object(A, 'HERE', self.root):
            existing = self.root / 'outputs/existing'
            existing.mkdir(parents=True)
            with self.assertRaises(M.InvalidData): A.run(self.attempt, existing)

    def test_run_end_to_end_fixture_receipt_and_no_updates(self):
        package = self.root / 'isolated_analysis'
        package.mkdir()
        (package / 'SOURCE_CONTRACT.json').write_text(json.dumps(self.contract))
        for name in ('analysis_math.py', 'analyze_saved.py'):
            (package / name).write_text('fixture placeholder for source hash only')
        with patch.object(A, 'HERE', package):
            out = package / 'outputs/fixture1'
            result = A.run(self.attempt, out)
        self.assertTrue((out / 'RESULT.json').is_file())
        for key in ('scientific_model_updates', 'calibration_updates', 'simulator_transitions'):
            self.assertEqual(result[key], 0)
        self.assertEqual(result['observed_est'][-6:], '-05:00')


if __name__ == '__main__':
    unittest.main()
