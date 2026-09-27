"""Execution-layer tests use exact toy dynamics, never trained BCA outcomes."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.ood.collect import (Attempt, Ledger, OracleSimulator, bind_pair,
    collect_states, freeze_candidates, run_outcomes, support_bank, support_query,
    read_artifact, write_artifact, bind_pair_files, file_hash, sha)
from experiments.ood.oracle import enumerate_returns


def policy(value):
    return lambda observations, key=None: np.full((len(observations), 1), value, np.float64)


def score(obs, actions):
    n = len(actions)
    return dict(width=np.ones(n), scale=np.ones(n), dose=np.ones(n),
                usable=np.ones(n, bool), radius=1., bayesian_radius=1.,
                conformal_radius=.9, unit=1.)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def attempt(self, name='attempt', limit=3):
        return Attempt.oracle(self.root/name, self.root/'ledger.jsonl',
                              episodes=2, captures=(0, 2), horizon=3,
                              simulator_limit=limit)

    def pipeline(self, attempt, host=0., bca=1.):
        policies = dict(host=policy(host), bca=policy(bca))
        collect_states(attempt, partial(OracleSimulator, limit=attempt.settings['simulator_limit']), policies)
        support = support_bank(np.array([[0.], [1.], [2.], [3.]]),
                               np.array([[-1.], [0.], [-1.], [0.]]),
                               np.arange(4), np.arange(4), seed=91, max_rows=4, neighbors=1)
        freeze_candidates(attempt, policies, score, support)
        return policies

    def test_missing_training_acceptance_and_stale_binding_refused(self):
        with self.assertRaises(ValueError):
            bind_pair({}, {}, {}, {})
        # The binding validates evidence, not a caller-supplied ready boolean.
        with self.assertRaises(ValueError):
            bind_pair({'ready_for_collection': True}, {}, {}, {})

    def real_evidence(self):
        root = Path(__file__).resolve().parents[2]
        readiness = json.loads((root/'docs/validation/ood-actions-first-readiness.json').read_text(encoding='utf-8'))
        audits, accepted, processes = {}, {}, {}
        for method, cp in readiness['checkpoints'].items():
            p = root/cp['accepted_audit']['path']
            audits[method] = json.loads(p.read_text(encoding='utf-8'))
            accepted[method] = json.loads((root/cp['acceptance_receipt']['path']).read_text(encoding='utf-8'))
            processes[method] = json.loads((p.parent/'process_exit.json').read_text(encoding='utf-8'))
        return readiness, audits, accepted, processes

    def test_closed_real_evidence_join_is_not_execution_acceptance(self):
        inputs = self.real_evidence()
        bound = bind_pair(*inputs)
        self.assertTrue(bound['training_accepted'])
        self.assertFalse(bound['ready_for_collection'])
        self.assertFalse(bound['trained_adapter_accepted'])
        for mutation in ('unaccepted', 'checkpoint', 'source', 'process', 'preparation', 'counter'):
            r, a, c, p = deepcopy(inputs)
            if mutation == 'unaccepted': a['host']['accepted'] = False
            if mutation == 'checkpoint': r['checkpoints']['host']['checkpoint_sha256'] = '0'*64
            if mutation == 'source': r['scientific_manifest_sha256'] = '0'*64
            if mutation == 'process': p['host']['exit_code'] = 1
            if mutation == 'preparation': r['checkpoints']['host']['preparation_input_hashes']['training'] = '0'*64
            if mutation == 'counter': a['host']['critic_updates'] = 999999
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_pair(r, a, c, p)

    def test_changed_frozen_source_refused_before_binding(self):
        root = Path(__file__).resolve().parents[2]
        readiness = root/'docs/validation/ood-actions-first-readiness.json'
        with patch('experiments.ood.collect.file_hash', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'Changed frozen training source'):
                bind_pair_files(root, readiness, {})

    def test_exclusive_attempt_and_complete_state_schema(self):
        attempt = self.attempt()
        with self.assertRaises(FileExistsError):
            self.attempt()
        sim = OracleSimulator()
        sim.reset(1)
        state = sim.capture()
        for field in state:
            bad = deepcopy(state)
            del bad[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                sim.restore(bad)
        with self.assertRaises(ValueError):
            attempt.require_scientific_execution()

    def test_persistent_charge_precedes_call_and_survives_interruption(self):
        ledger = Ledger(self.root/'l.jsonl', {'engineering': 2})
        def interrupted():
            self.assertEqual(ledger.totals()['engineering']['environment'], 1)
            raise KeyboardInterrupt('fixture interruption')
        with self.assertRaises(KeyboardInterrupt):
            ledger.call('engineering', 'test', 4, interrupted)
        reopened = Ledger(self.root/'l.jsonl', {'engineering': 2})
        self.assertEqual(reopened.totals()['engineering'], {'environment': 1, 'physics': 4})
        reopened.call('engineering', 'test', 4, lambda: None)
        with self.assertRaises(ValueError):
            reopened.call('engineering', 'test', 4, lambda: self.fail('over-budget call'))

    def test_lossless_artifact_and_tampering(self):
        p = self.root/'bank.json'
        obj = dict(rng=('MT', np.array([0, 4294967295], np.uint32)), inactive=None,
                   action=np.array([-.0, 1], np.float32))
        expected = write_artifact(p, obj)
        restored = read_artifact(p, expected)
        self.assertIsInstance(restored['rng'], tuple)
        np.testing.assert_array_equal(restored['rng'][1], obj['rng'][1])
        self.assertEqual(restored['action'].tobytes(), obj['action'].tobytes())
        with self.assertRaises(FileExistsError):
            write_artifact(p, obj)
        p.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_artifact(p, expected)

    def test_training_only_episode_split_and_missing_boundaries(self):
        obs = np.arange(20.).reshape(10, 2)
        actions = np.linspace(-1, 1, 10).reshape(10, 1)
        ids = np.arange(10)
        bank = support_bank(obs, actions, ids, np.repeat(np.arange(5), 2), seed=7,
                            max_rows=8, neighbors=2)
        self.assertFalse(set(bank['reference_episodes']) & set(bank['validation_episodes']))
        self.assertEqual(len(bank['validation_indices']), 1)
        self.assertIsNotNone(bank['threshold'])
        missing = support_bank(obs, actions, ids, None, seed=7, max_rows=8, neighbors=2)
        self.assertIsNone(missing['threshold'])
        self.assertFalse(missing['thresholded_labels_available'])
        with self.assertRaises(ValueError):
            support_bank(obs, actions, np.zeros(10, int), None, seed=7)

    def test_all_slots_precommitted_and_duplicate_scores_share_one_vote(self):
        a = self.attempt()
        self.pipeline(a, host=1., bca=1.)
        bank = a.read_stage('candidates')
        for row in bank['rows']:
            if row['status'] != 'captured':
                continue
            self.assertEqual(row['proposed'].shape, (10, 1))
            self.assertEqual(row['step_keys'].shape, (3, 2))
            self.assertEqual(row['alias'][0], row['alias'][1])
            self.assertEqual(row['random'][0], row['random'][1])
            self.assertTrue((abs(row['actions']) <= 1).all())
        self.assertFalse((a.path/'outcomes.started.json').exists())

    def test_changed_candidates_and_rng_refused_before_outcome_calls(self):
        for kind in ('action', 'key'):
            a = self.attempt(kind)
            policies = self.pipeline(a)
            before = a.ledger.totals()
            p = a.path/'candidates.json'
            raw = p.read_text(encoding='utf-8')
            # Any byte change, including one key, invalidates the committed bank.
            p.write_text(raw+' ', encoding='utf-8')
            with self.assertRaises(ValueError):
                run_outcomes(a, partial(OracleSimulator, limit=3), policies)
            self.assertEqual(a.ledger.totals(), before)

    def test_exact_returns_missing_captures_and_same_state_contrasts(self):
        a = self.attempt()
        policies = self.pipeline(a)
        run_outcomes(a, partial(OracleSimulator, limit=3), policies)
        states = a.read_stage('states')['rows']
        candidates = a.read_stage('candidates')['rows']
        results = a.read_stage('outcomes')['rows']
        self.assertEqual(len(states), 8)
        self.assertEqual(sum(r['status'] == 'missing' for r in states), 2)
        for state, candidate in zip(states, candidates):
            if state['status'] != 'captured':
                continue
            selected = [r for r in results if r['state_id'] == state['state_id']]
            self.assertEqual(len(selected), 2)
            for row in selected:
                continuation = 0. if row['continuation'] == 'host' else 1.
                remaining = min(3, 3-state['snapshot']['elapsed'])
                expected = enumerate_returns(state['snapshot']['x'], candidate['actions'][:, 0],
                                             continuation=continuation, horizon=remaining)
                np.testing.assert_allclose(row['returns'], expected, rtol=0, atol=1e-14)
                np.testing.assert_allclose(row['harm'], np.asarray(expected)[0]-expected,
                                           rtol=0, atol=1e-14)
                self.assertLessEqual(max(row['lengths']), remaining)
        initial = next(r for r in results if r['state_id']=='host/0/0' and r['continuation']=='host')
        self.assertAlmostEqual(initial['returns'][0], -3.)
        self.assertAlmostEqual(initial['returns'][1], -.1)
        self.assertAlmostEqual(initial['returns'][2], -12.1)
        self.assertLess(initial['harm'][1], 0)  # unfamiliar but helpful
        self.assertGreater(initial['harm'][2], 1)  # familiar but harmful
        with self.assertRaises(FileExistsError):
            run_outcomes(a, partial(OracleSimulator, limit=3), policies)

    def test_interrupted_stage_cannot_restart(self):
        a = self.attempt()
        def broken(obs, key=None):
            raise RuntimeError('injected actor failure')
        with self.assertRaises(RuntimeError):
            collect_states(a, partial(OracleSimulator, limit=3), dict(host=broken,bca=policy(0.)))
        self.assertTrue((a.path/'states.failed.json').exists())
        with self.assertRaises(FileExistsError):
            collect_states(a, partial(OracleSimulator, limit=3), dict(host=policy(0.), bca=policy(0.)))

    def test_changed_rng_source_and_real_constructor_refused(self):
        a = self.attempt()
        a.streams['collection'][0] += 1
        with self.assertRaisesRegex(ValueError, 'random streams'):
            collect_states(a, partial(OracleSimulator, limit=3), dict(host=policy(0.), bca=policy(0.)))
        b = self.attempt('source')
        with patch('experiments.ood.collect.file_hash', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                collect_states(b, partial(OracleSimulator, limit=3), dict(host=policy(0.), bca=policy(0.)))
        c = self.attempt('constructor')
        with self.assertRaisesRegex(ValueError, 'no arbitrary factory'):
            collect_states(c, lambda: self.fail('unapproved constructor ran'), dict(host=policy(0.), bca=policy(0.)))


if __name__ == '__main__':
    unittest.main()
