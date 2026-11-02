"""End-to-end run() control flow with fake models and simulator: the gate sees every bank and context."""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_worker as worker
import lock_tolerant
import runtime as real_runtime

PACKAGE = Path(worker.__file__).parent
PROTOCOL = worker.read(PACKAGE/'protocol.json')
NEW = worker.rerun_inputs.NEW_CONTEXT
FINAL = dict(transition_steps_attempted=1975304, transition_steps_completed=1975304, resets_attempted=129,
             resets_completed=129, actor_rows_attempted=1953792, actor_rows_completed=1953792,
             critic_rows_attempted=40320, critic_rows_completed=40320)
MATCHING = {'reference_digest': 'digest', 'anchor_actor_digest': 'digest', 'distance_coefficient': 2.5}


class RecordingGate:
    def check_bank(self, context, raw, arrays): pass
    def check_context(self, key, raw): return 'new'
    def finish(self): return {'recording_only': True}
    def receipt(self): return {'recording_only': True}


class FakePolicy:
    def __init__(self, runtime, adapter, state): self.state = state
    def assert_unchanged(self): pass


class FakeRuntime:
    host = staticmethod(real_runtime.host)
    def create(self, protocol, seed, condition):
        context = SimpleNamespace(data=SimpleNamespace(receipt={'protected_transition_rows_decoded': 0}),
                                  runtime={'device': 'fixture'})
        return SimpleNamespace(context=context), object()
    def branch(self, adapter, warm, protocol, seed, arm, coefficient): return adapter, object()
    def tree_digest(self, value): return 'digest'


def returns(spec_id, context):
    return np.random.default_rng(sum(map(ord, spec_id+context))).normal(size=32)


class WorkerFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.offsets, self.progress_calls = {}, [0]

    def fakes(self):
        core = SimpleNamespace(canonical=SimpleNamespace(START_LAWS={'recorded': {'current': [0.1, 0.7]},
                                                                     'strong_current': {'current': [0.7, 1.1]}}))
        reference = SimpleNamespace(new_ledger=lambda cap, actor_cap: dict(FINAL),
            runtime_parity=lambda core, x0, ledger, deadline, progress: {'parity': 'fixture'},
            actor_actions=lambda policy, obs, ledger, deadline, progress: np.zeros((32, 4)),
            authenticate_core=lambda core: None)
        def bank(core, reference, *, context, q_seeds, free_seeds, prefix_seed, start_law, ledger, deadline, progress):
            progress(dict(ledger)); shift = 0. if context == 'ordinary' else 1.
            return {'context': context, 'bank_id': 'bank-'+context, 'states': np.zeros((32, 13))+shift,
                    'times': np.asarray(PROTOCOL['evaluation']['anchor_times'], np.int64),
                    'observations': np.ones((32, 5))+shift, 'free_starts': np.zeros((32, 13))-shift,
                    'q_seeds': np.asarray(q_seeds, np.int64), 'free_seeds': np.asarray(free_seeds, np.int64),
                    'prefix_seed': int(prefix_seed), 'start_law': dict(start_law), 'fixed_a0': [-.5, -.25, 0, .25, .5],
                    'freshness': 'fixture'}
        def evaluate(core, reference, policy, bank, ledger, *, trained, adjusted_score, deadline, progress):
            progress(dict(ledger)); self.progress_calls[0] += 1
            sid, context = policy.state.id, bank['context']
            out = {'context': context, 'bank_id': bank['bank_id'], 'trained': bool(trained),
                   'free': {'returns': (returns(sid, context)+self.offsets.get((sid, context), 0.)).tolist(),
                            'episodes': 32, 'flags': [True, False], 'label': 'fixture'}, 'q': None}
            if trained:
                ledger['width_rows_attempted'] += 320; ledger['width_rows_completed'] += 320
                out['q'] = {'per_probe': {'actions': np.zeros((32, 6, 4)).tolist(), 'qpi': np.zeros((32, 6)).tolist(),
                                          'q1': np.zeros((32, 6)).tolist()}, 'panels': {'over': 3}}
            return out
        exports = {spec['id']: {'id': spec['id'], 'matching': dict(MATCHING)} for spec in worker.snapshot_plan(PROTOCOL)}
        def restore(runtime, adapter, template, export):
            return SimpleNamespace(id=export['id'], aux=SimpleNamespace(posterior=None),
                microstep=SimpleNamespace(agent_state=SimpleNamespace(actor=SimpleNamespace(params=None))))
        return core, reference, bank, evaluate, exports, restore

    def run_once(self, name, gate=None, recorded=None):
        import launch
        root = self.base/name; output = root/'run/evaluation'; output.mkdir(parents=True)
        for file in ('protocol.json', 'DATA_PLAN.json'):
            shutil.copyfile(PACKAGE/file, root/file)
        core, reference, bank, evaluate, exports, restore = self.fakes()
        patches = [patch.object(worker, 'HERE', root), patch.object(launch, 'require_execution_authorization'),
            patch.object(worker, 'assemble_exports', return_value=(exports, {})),
            patch.object(worker, 'load_simulator', return_value=(core, reference, {'core': 'fixture'})),
            patch.object(worker, 'module', side_effect=lambda n, p: FakeRuntime() if n == '_quartile_runtime' else None),
            patch.object(worker, 'restore_snapshot', side_effect=restore), patch.object(worker, 'FrozenPolicy', FakePolicy),
            patch.object(worker.evaluation, 'build_context_bank', side_effect=bank),
            patch.object(worker.evaluation, 'evaluate_snapshot', side_effect=evaluate),
            patch.object(worker.evaluation, 'q_panels', return_value={'distance_panel': 1}),
            patch.object(worker, 'summarize', return_value={'decision': 'completed_fixed_development_study_no_automatic_successor'}),
            patch.object(worker, 'common_reference_panels', return_value={'fixture': True})]
        if gate is not None:
            patches.append(patch.object(worker, 'ReproductionGate', gate))
        if recorded is not None:
            source, files = recorded
            def read_recorded(descriptor):
                raw = (source/descriptor['path']).read_bytes()
                worker.require(worker.digest(raw) == descriptor['sha256'], 'Recorded result bytes differ')
                return raw
            report = {'snapshot_results': files, 'status': 'failed_central_quartile_evaluation',
                      'phase': 'snapshot_inference', 'completed_snapshot_contexts': 143,
                      'error': 'PermissionError: [Errno 13] Permission denied: progress.json',
                      'decision': 'incomplete_no_utility_conclusion'}
            patches += [patch.object(worker.rerun_inputs, 'recorded_evaluation', return_value=report),
                        patch.object(worker.rerun_inputs, 'recorded_bytes', side_effect=read_recorded)]
        for item in patches:
            item.start(); self.addCleanup(item.stop)
        try:
            return worker.run(PROTOCOL, output, dict(jobs_root='fixture', timeout_seconds=300,
                                                     deadline_monotonic=time.monotonic()+300)), output
        finally:
            for item in reversed(patches): item.stop()

    def first_attempt(self):
        report, output = self.run_once('first', gate=RecordingGate)
        self.assertEqual(report['status'], 'completed_central_quartile_evaluation')
        files = {k: v for k, v in report['snapshot_results'].items() if k not in (NEW, 'common_reference')}
        self.assertEqual(len(files), 145)
        return output, files

    def test_exact_rerun_completes_with_143_tier1_and_one_new_context(self):
        recorded = self.first_attempt()
        report, output = self.run_once('rerun', recorded=recorded)
        self.assertEqual(report['status'], 'completed_central_quartile_evaluation')
        r = report['reproduction']
        self.assertEqual((r['tier1_exact'], r['tier2_near_exact'], r['contexts_checked']), (143, 0, 144))
        self.assertEqual(r['new_contexts'], [NEW])
        self.assertEqual(r['bank_archives_byte_identical'], {'ordinary': True, 'strong_current': True})
        self.assertEqual(report['study_evaluation_attempt'], 2)
        self.assertEqual(len([k for k in report['snapshot_results'] if '/' in k]), 144)
        self.assertEqual(json.loads((output/'report.json').read_bytes()), report)
        # Progress is throttled: far fewer writes than progress callbacks.
        self.assertLess(report['write_lock_tolerance']['progress_writes'], 20)
        self.assertGreater(self.progress_calls[0], 140)
        self.assertEqual(json.loads((output/'EVALUATION_ATTEMPT.json').read_text())['study_evaluation_attempt'], 2)

    def test_last_bit_differences_pass_as_tier2_and_are_disclosed(self):
        recorded = self.first_attempt()
        self.offsets[('seed1_hole__bca__20000', 'ordinary')] = 1e-9
        report, _ = self.run_once('rerun', recorded=recorded)
        r = report['reproduction']
        self.assertEqual(report['status'], 'completed_central_quartile_evaluation')
        self.assertEqual((r['tier1_exact'], r['tier2_near_exact']), (142, 1))
        self.assertEqual(r['near_exact_contexts'], ['seed1_hole__bca__20000/ordinary'])

    def test_difference_beyond_tolerance_stops_incomplete_with_partial_receipt(self):
        recorded = self.first_attempt()
        self.offsets[('seed1_hole__bca__20000', 'ordinary')] = 1e-3
        report, output = self.run_once('rerun', recorded=recorded)
        self.assertEqual(report['status'], 'failed_central_quartile_evaluation')
        self.assertEqual(report['decision'], 'incomplete_no_utility_conclusion')
        self.assertTrue(report['error'].startswith('ValueError: K2: context not reproduced'))
        self.assertIn('seed1_hole__bca__20000/ordinary', report['error'])
        self.assertLess(report['completed_snapshot_contexts'], 144)
        self.assertEqual(report['reproduction']['tier1_exact'], report['completed_snapshot_contexts'])
        self.assertFalse((output/'COMMON_REFERENCE.json').exists())

    def test_momentary_and_persistent_locks_do_not_stop_the_evaluation(self):
        recorded = self.first_attempt()
        real, calls = os.replace, [0]
        def flaky(source, target):
            calls[0] += 1
            if Path(target).name == 'progress.json' or calls[0] % 3 == 0:
                raise PermissionError(13, 'Permission denied')
            return real(source, target)
        with patch.object(lock_tolerant.os, 'replace', side_effect=flaky), \
                patch.object(lock_tolerant.time, 'sleep'):
            report, output = self.run_once('rerun', recorded=recorded)
        self.assertEqual(report['status'], 'completed_central_quartile_evaluation')
        locks = report['write_lock_tolerance']
        self.assertGreater(locks['lock_errors'], 0)
        self.assertGreater(locks['skipped_progress_writes'], 0)
        self.assertEqual(locks['progress_writes'], 0)
        self.assertEqual(report['reproduction']['tier1_exact'], 143)
        self.assertFalse((output/'progress.json').exists())


if __name__ == '__main__': unittest.main()
