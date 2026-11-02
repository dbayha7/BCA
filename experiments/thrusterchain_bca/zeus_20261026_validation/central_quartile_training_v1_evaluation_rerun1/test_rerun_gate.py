"""Approved reproduction gate (kill rules K1-K3) on synthetic recorded files; no real results read."""
import copy
import io
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_worker as worker
from lock_tolerant import serialized


def context_value(offset=0.):
    return {'context': 'ordinary', 'bank_id': 'abc', 'trained': True, 'cell': 'seed0_full', 'seed': 0,
            'free': {'returns': [1.25 + offset, -3.5, 0.0], 'episodes': 32, 'mean_return': -0.75},
            'q': {'panels': {'over': 3, 'flag': False, 'mae': None}, 'labels': ['x', 'y']}}


def bank_arrays():
    return {'states': np.arange(26, dtype=np.float64).reshape(2, 13), 'times': np.asarray([0, 25], np.int64),
            'q_seeds': np.asarray([5, 6], np.int64)}


def npz(arrays, epoch=None):
    buffer = io.BytesIO()
    if epoch is None:
        np.savez(buffer, **arrays)
    else:
        with patch('time.time', return_value=epoch):
            np.savez(buffer, **arrays)
    return buffer.getvalue()


class Tier2CompareTests(unittest.TestCase):
    def test_identical_and_within_tolerance(self):
        self.assertEqual(worker.tier2_compare(context_value(), context_value()), (None, 0.))
        where, worst = worker.tier2_compare(context_value(), context_value(1e-7))
        self.assertIsNone(where); self.assertGreater(worst, 0)
        self.assertLessEqual(worst, 1e-6)
        self.assertIsNone(worker.tier2_compare({'v': 1000.}, {'v': 1000. + 9e-4})[0])

    def test_violations_are_located(self):
        cases = [(context_value(), context_value(1e-3), '$.free.returns[0]'),
                 ({'v': 1}, {'v': 1.0}, '$.v'), ({'v': True}, {'v': 1}, '$.v'),
                 ({'v': None}, {'v': 0.0}, '$.v'), ({'v': 'a'}, {'v': 'b'}, '$.v'),
                 ({'v': 3}, {'v': 4}, '$.v'), ({'v': False}, {'v': True}, '$.v'),
                 ({'a': 1}, {'a': 1, 'b': 2}, '$'), ({'v': [1., 2.]}, {'v': [1.]}, '$.v'),
                 ({'v': 'nan'}, {'v': 'positive_infinity'}, '$.v')]
        for old, new, where in cases:
            with self.subTest(old=old, new=new):
                self.assertEqual(worker.tier2_compare(old, new)[0], where)

    def test_tolerance_scales_with_magnitude(self):
        self.assertIsNone(worker.tier2_compare({'v': 0.}, {'v': 1e-6})[0])
        self.assertEqual(worker.tier2_compare({'v': 0.}, {'v': 1.1e-6})[0], '$.v')
        self.assertIsNone(worker.tier2_compare({'v': 1e6}, {'v': 1e6 + 1.})[0])
        self.assertEqual(worker.tier2_compare({'v': 1e6}, {'v': 1e6 + 1.01})[0], '$.v')


class GateTests(unittest.TestCase):
    def setUp(self):
        self.store = {}
        def put(name, raw):
            self.store[name] = raw
            return {'path': name, 'sha256': worker.digest(raw), 'bytes': len(raw)}
        self.put = put
        files = {}
        self.arrays = {'ordinary': bank_arrays(), 'strong_current': {k: v + 1 for k, v in bank_arrays().items()}}
        for context, arrays in self.arrays.items():
            descriptor = put(context + '_bank.npz', npz(arrays, epoch=0))
            files[context + '_bank'] = put(context + '_bank.json', serialized(
                {'context': context, 'bank_id': 'id-' + context, 'start_law': {'current': [0.1, 0.7]},
                 'arrays': descriptor}))
        self.keys = [f'k{i}/ordinary' for i in range(3)]
        for i, key in enumerate(self.keys):
            files[key] = put(f'k{i}__ordinary.json', serialized(context_value()))
        report = {'snapshot_results': files, 'status': 'failed_central_quartile_evaluation',
                  'phase': 'snapshot_inference', 'completed_snapshot_contexts': 143,
                  'error': 'PermissionError: [Errno 13] Permission denied: progress.json',
                  'decision': 'incomplete_no_utility_conclusion'}
        def recorded_bytes(descriptor):
            raw = self.store[descriptor['path']]
            worker.require(worker.digest(raw) == descriptor['sha256'], 'Recorded result bytes differ')
            return raw
        for name, value in (('recorded_evaluation', lambda: copy.deepcopy(report)),
                            ('recorded_bytes', recorded_bytes)):
            patcher = patch.object(worker.rerun_inputs, name, value); patcher.start(); self.addCleanup(patcher.stop)
        self.gate = worker.ReproductionGate()

    def bank_json(self, context, **change):
        meta = {'context': context, 'bank_id': 'id-' + context, 'start_law': {'current': [0.1, 0.7]}}
        meta.update(change)
        return serialized({**meta, 'arrays': {'path': context + '_bank.npz', 'sha256': 'new-zip-time', 'bytes': 1}})

    def pass_banks(self):
        for context, arrays in self.arrays.items():
            self.gate.check_bank(context, self.bank_json(context), arrays)

    def test_bank_archives_are_deterministic_and_arrays_decide(self):
        # numpy.savez writes zip entries with the fixed 1980-01-01 default date, not the clock.
        self.assertEqual(npz(self.arrays['ordinary'], epoch=0), npz(self.arrays['ordinary'], epoch=10**9))
        self.pass_banks()  # descriptor differs from the recorded archive hash; equal arrays still pass
        receipt = self.gate.receipt()
        self.assertEqual(receipt['banks_exact'], ['ordinary', 'strong_current'])
        self.assertEqual(receipt['bank_archives_byte_identical'], {'ordinary': False, 'strong_current': False})
        gate = worker.ReproductionGate()
        recorded = json.loads(self.store['ordinary_bank.json'])
        gate.check_bank('ordinary', self.store['ordinary_bank.json'], self.arrays['ordinary'])
        self.assertEqual(gate.receipt()['bank_archives_byte_identical'], {'ordinary': True})
        self.assertEqual(recorded['arrays']['sha256'], worker.digest(npz(self.arrays['ordinary'])))

    def test_bank_changes_stop_the_run(self):
        changed = {k: v.copy() for k, v in self.arrays['ordinary'].items()}
        changed['states'][0, 0] = np.nextafter(changed['states'][0, 0], 1)
        retyped = {**self.arrays['ordinary'], 'times': self.arrays['ordinary']['times'].astype(np.int32)}
        extra = {**self.arrays['ordinary'], 'other': np.zeros(1)}
        for raw, arrays in ((self.bank_json('ordinary', bank_id='other'), self.arrays['ordinary']),
                            (self.bank_json('ordinary'), changed), (self.bank_json('ordinary'), retyped),
                            (self.bank_json('ordinary'), extra)):
            gate = worker.ReproductionGate()
            with self.subTest(), self.assertRaisesRegex(ValueError, '^K1:'):
                gate.check_bank('ordinary', raw, arrays)
        with self.assertRaisesRegex(ValueError, 'Unexpected or repeated bank'):
            self.gate.check_bank('elsewhere', self.bank_json('ordinary'), self.arrays['ordinary'])

    def test_exact_near_exact_new_and_finish(self):
        self.pass_banks()
        self.assertEqual(self.gate.check_context(self.keys[0], serialized(context_value())), 'exact')
        self.assertEqual(self.gate.check_context(self.keys[1], serialized(context_value(1e-9))), 'near_exact')
        self.assertEqual(self.gate.check_context(self.keys[2], serialized(context_value())), 'exact')
        self.assertEqual(self.gate.check_context(worker.rerun_inputs.NEW_CONTEXT, serialized(context_value(5.))), 'new')
        receipt = self.gate.finish()
        self.assertEqual((receipt['tier1_exact'], receipt['tier2_near_exact'], receipt['contexts_checked']), (2, 1, 4))
        self.assertEqual(receipt['near_exact_contexts'], [self.keys[1]])
        self.assertEqual(receipt['new_contexts'], [worker.rerun_inputs.NEW_CONTEXT])
        self.assertGreater(receipt['largest_tier2_scaled_deviation'], 0)
        self.assertEqual(receipt['first_evaluation_attempt']['completed_snapshot_contexts'], 143)

    def test_context_violations_stop_the_run(self):
        with self.assertRaisesRegex(ValueError, '^K2:.*free.returns'):
            self.gate.check_context(self.keys[0], serialized(context_value(1e-3)))
        with self.assertRaisesRegex(ValueError, '^K3: unrecorded'):
            self.gate.check_context('seed9_full__bca__20000/ordinary', serialized(context_value()))
        self.gate.check_context(self.keys[1], serialized(context_value()))
        with self.assertRaisesRegex(ValueError, 'produced twice'):
            self.gate.check_context(self.keys[1], serialized(context_value()))

    def test_finish_requires_every_bank_context_and_the_new_context(self):
        with self.assertRaisesRegex(ValueError, 'K1: not every'):
            self.gate.finish()
        self.pass_banks()
        self.gate.check_context(self.keys[0], serialized(context_value()))
        with self.assertRaisesRegex(ValueError, 'K3: recorded contexts not reproduced'):
            self.gate.finish()
        for key in self.keys[1:]:
            self.gate.check_context(key, serialized(context_value()))
        with self.assertRaisesRegex(ValueError, 'K3: missing context was not produced'):
            self.gate.finish()

    def test_tampered_recorded_file_cannot_be_used(self):
        self.store['k0__ordinary.json'] = serialized(context_value(1e-9))
        with self.assertRaisesRegex(ValueError, 'Recorded result bytes differ'):
            self.gate.check_context(self.keys[0], serialized(context_value(1e-9)))

    def test_tampered_recorded_file_fails_even_when_new_bytes_match_descriptor(self):
        self.store['k0__ordinary.json'] = serialized(context_value(1e-9))
        with self.assertRaisesRegex(ValueError, 'Recorded result bytes differ'):
            self.gate.check_context(self.keys[0], serialized(context_value()))


if __name__ == '__main__': unittest.main()
