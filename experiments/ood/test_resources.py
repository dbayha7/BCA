"""Resource tests execute no model or physical simulator."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from experiments.ood.resources import ResourceLedger, scope_caps, prior_adapter_usage


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'ledger.sqlite'
        self.caps = {'global': [20, 80], 'engineering/a': [10, 40],
                     'engineering/b': [10, 40]}

    def tearDown(self):
        self.temp.cleanup()

    def open(self):
        return ResourceLedger(self.path, self.caps, 'frozen-protocol')

    def test_reservation_commits_before_callback_and_is_never_refunded(self):
        with self.open() as ledger:
            def fail():
                with self.open() as reader:
                    self.assertEqual(reader.audit()['reserved']['global'], [1, 4])
                raise RuntimeError('uncertain physical call')
            with self.assertRaises(RuntimeError):
                ledger.call('attempt:constructor', ['engineering/a'], 1, 4, fail)
            self.assertEqual(ledger.audit()['reserved']['global'], [1, 4])
            with self.assertRaises(ValueError):
                ledger.call('attempt:constructor', ['engineering/a'], 1, 4, lambda: self.fail('retry'))

    def test_shared_prior_charge_counts_once_globally_and_in_each_relevant_cell(self):
        with self.open() as ledger:
            ledger.reserve('old', ['engineering/a', 'engineering/b'], 7, 28, {'receipt': 'pinned'})
            self.assertEqual(ledger.audit()['reserved'],
                             {'global': [7, 28], 'engineering/a': [7, 28], 'engineering/b': [7, 28]})
            with self.assertRaises(ValueError):
                ledger.reserve('excess', ['engineering/b'], 4, 16)
            self.assertEqual(ledger.audit()['entries'], 1)

    def test_reopening_preserves_usage_and_different_caps_or_protocol_refused(self):
        with self.open() as ledger:
            ledger.reserve('first', ['engineering/a'], 2, 8)
        with self.open() as ledger:
            self.assertEqual(ledger.audit()['reserved']['global'], [2, 8])
        with self.assertRaises(ValueError):
            ResourceLedger(self.path, {'global': [100, 400]}, 'frozen-protocol')
        with self.assertRaises(ValueError):
            ResourceLedger(self.path, self.caps, 'other')

    def test_two_connections_share_atomic_caps(self):
        with self.open() as first, self.open() as second:
            first.reserve('a', ['engineering/a'], 6, 24)
            with self.assertRaises(ValueError):
                second.reserve('b', ['engineering/a'], 5, 20)
            second.reserve('c', ['engineering/a'], 4, 16)
            self.assertEqual(first.audit()['reserved']['global'], [10, 40])

    def test_physics_cap_and_malformed_reservations_fail_before_callback(self):
        with self.open() as ledger:
            for scopes, env, physics in [(['engineering/a'], True, 4),
                    (['engineering/a'], 1, 41), (['unknown'], 1, 4),
                    (['engineering/a', 'engineering/a'], 1, 4), ([], 1, 4),
                    (['global'], 1, 4), (['engineering/a'], 0, 0)]:
                with self.subTest(scopes=scopes, env=env, physics=physics), self.assertRaises(ValueError):
                    ledger.call('bad', scopes, env, physics, lambda: self.fail('unreserved call'))
            self.assertEqual(ledger.audit()['entries'], 0)

    def test_aggregate_corruption_detected_at_stage_audit_and_reopen(self):
        with self.open() as ledger:
            ledger.reserve('first', ['engineering/a'], 1, 4)
            # Simulate external damage. Legitimate clients never edit aggregates.
            with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
                db.execute("UPDATE totals SET environment=0 WHERE scope='global'")
            with self.assertRaises(ValueError):
                ledger.audit()
        with self.assertRaises(ValueError):
            self.open()

    def test_journal_cannot_be_updated_or_deleted(self):
        with self.open() as ledger:
            ledger.reserve('first', ['engineering/a'], 1, 4)
            with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute('DELETE FROM entries')
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute("UPDATE entries SET payload='changed'")
            self.assertEqual(ledger.audit()['entries'], 1)

    def test_fixed_resource_contract(self):
        caps = scope_caps(['td3_bc/hopper/s1'], ['td3_bc/hopper'])
        self.assertEqual(caps['engineering/td3_bc/hopper'], [10000, 40000])
        self.assertEqual(caps['collection/td3_bc/hopper/s1'], [12800, 51200])
        self.assertEqual(caps['outcomes/td3_bc/hopper/s1'], [1280000, 5120000])
        self.assertEqual(caps['repeat_checks/td3_bc/hopper/s1'], [5120, 20480])
        self.assertNotIn('coverage/td3_bc/hopper/s1', caps)

    def test_prior_history_shared_attribution_preserves_original_failures(self):
        import json
        root = Path(__file__).resolve().parents[2]
        prior = json.loads((root/'experiments/ood/validation.json').read_text())
        usage = prior_adapter_usage(prior)
        self.assertEqual(sum(x['environment'] for x in usage), 72)
        self.assertEqual(sum(x['physics'] for x in usage), 288)
        self.assertEqual(len(usage), 12)
        self.assertEqual([x['actual_exit'] for x in usage[:6]], [1,1,1,1,1,1])
        self.assertEqual(sum(x['environment'] for x in usage if 'engineering/td3_bc/hopper' in x['scopes']), 36)


if __name__ == '__main__':
    unittest.main()
