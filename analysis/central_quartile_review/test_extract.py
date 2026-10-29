import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import extract


def write_json(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return {'path': path.name, 'bytes': len(raw), 'sha256': extract.sha(raw)}


class ExtractionGuardTest(unittest.TestCase):
    def test_stopped_receipt_rejected_before_report_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            write_json(p / 'RUN_RESULT.json', {'status': 'stopped_after_incomplete_job'})
            with self.assertRaisesRegex(ValueError, 'terminal status'):
                extract.main(p)

    def test_artifact_detects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            desc = write_json(p / 'one.json', {'a': 1})
            self.assertTrue(extract.artifact(p, desc))
            (p / 'one.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                extract.artifact(p, desc)
            desc['path'] = '../one.json'
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                extract.artifact(p, desc)

    def test_complete_synthetic_report_and_late_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'run' / 'evaluation'
            out.mkdir(parents=True)
            files = {}
            for context in ('ordinary', 'strong_current'):
                arr = write_json(out / (context + '.npz'), {'fake': 'array'})
                files[context + '_bank'] = write_json(out / (context + '_bank.json'), {'arrays': arr})
            files['common_reference'] = write_json(out / 'common.json', {})
            for s in range(3):
                for condition in ('full', 'hole', 'random'):
                    stages = [('warmup', 0), ('warmup', 10000)] + [(a, t) for a in ('native', 'bca', 'distance') for t in (15000, 20000)]
                    for stage, step in stages:
                        for context in ('ordinary', 'strong_current'):
                            label = f'seed{s}_{condition}__{stage}__{step:05d}'
                            x = {'seed': s, 'condition': condition, 'arm': 'native' if stage == 'warmup' else stage,
                                 'checkpoint': step, 'context': context, 'complete_state_unchanged': True,
                                 'free': {'returns': [0.] * 32, 'mean_return': 0., 'mean_undiscounted_return': 0.,
                                          'high_step_hole_fraction': 0., 'episodes_entering_hole': 0}}
                            if step:
                                x.update(q={'panels': {}, 'raw_decisions': {}, 'adjusted_decisions': {},
                                            'signed_candidate_margins': {}}, visited={'panels': {}})
                            files[label + '/' + context] = write_json(out / (label + '__' + context + '.json'), x)
            reproduction = {'recorded_contexts': 143, 'contexts_checked': 144, 'tier1_exact': 143,
                            'tier2_near_exact': 0, 'new_contexts': ['seed2_random__distance__20000/strong_current'],
                            'banks_exact': ['ordinary', 'strong_current']}
            fields = ('learning_adequacy_passed', 'policy_screen_passed', 'accuracy_screen_passed',
                      'conditional_accuracy_testable', 'learning', 'policy_comparisons', 'return_interactions',
                      'accuracy_native_q1_minus_bca_adjusted_mae', 'accuracy_hole_minus_random',
                      'native_hole_minus_random_return', 'native_hole_minus_random_mae',
                      'policy_rescue_screen_passed', 'accuracy_rescue_screen_passed', 'high_load_anchor_families')
            report = {k: {} for k in fields}
            report.update(status='completed_central_quartile_evaluation', execution_counts_known=True, error=None,
                          study_evaluation_attempt=2, reproduction=reproduction, snapshot_results=files)
            report['policy_comparisons'] = {'ordinary/20000/hole/bca_minus_native': {'paired_values_by_seed': [[0.] * 32] * 3}}
            report_desc = write_json(out / 'report.json', report)
            terminal = {'status': 'completed_evaluation_rerun', 'prepared_identity_unchanged': True,
                        'all_declared_jobs_completed': True, 'execution_counts_known': True, 'error': None,
                        'jobs': [{'completed': True, 'execution_counts_known': True, 'report_sha256': report_desc['sha256']}],
                        'reproduction': reproduction, 'disclosure': {}}
            write_json(root / 'RUN_RESULT.json', terminal)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                extract.main(root)
            self.assertEqual(len(json.loads(output.getvalue())['snapshots']), 144)
            (out / files[next(k for k in files if '/' in k)]['path']).write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                extract.main(root)


if __name__ == '__main__':
    unittest.main()
