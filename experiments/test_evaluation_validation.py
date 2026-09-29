"""Regression tests for host-specific saved episode schemas; no learners or simulator."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from runtime.validation import verify_events


def fixture(host):
    events = [NS(kind='periodic', step=2, episode_seeds=(11,)),
              NS(kind='final', step=2, episode_seeds=(12,))]
    protocol = NS(num_updates=2, scan_block_size=1, refresh_steps=(), evaluation_events=events)
    records = [dict(kind='accepted_scan', step=1), dict(kind='accepted_scan', step=2)]
    values = {'score': 3., 'return': 2.} if host == 'cql' else {'normalized_score': 3., 'raw_return': 2.}
    banks = [dict(kind=e.kind, step=e.step, episode_seeds=list(e.episode_seeds),
                  episode_count=1, episodes=[dict(values)]) for e in events]
    return protocol, records, banks


class EvaluationSchemas(unittest.TestCase):
    def test_each_host_schema_is_accepted_without_mutation(self):
        for host in ('cql', 'td3', 'rebrac'):
            args = fixture(host)
            before = deepcopy(args)
            verify_events(*args, host=host)
            self.assertEqual(args, before)

    def test_original_default_remains_compatible(self):
        verify_events(*fixture('td3'))

    def test_other_host_schema_is_not_silently_substituted(self):
        for host, wrong in [('cql', 'td3'), ('td3', 'cql'), ('rebrac', 'cql')]:
            with self.assertRaises(ValueError):
                verify_events(*fixture(wrong), host=host)

    def test_each_required_field_and_finite_gate(self):
        for host in ('cql', 'td3', 'rebrac'):
            for key in fixture(host)[2][0]['episodes'][0]:
                for bad in (None, float('nan'), float('inf'), '3', True):
                    args = fixture(host)
                    args[2][0]['episodes'][0][key] = bad
                    with self.assertRaises(ValueError):
                        verify_events(*args, host=host)
                args = fixture(host)
                del args[2][0]['episodes'][0][key]
                with self.assertRaises(ValueError):
                    verify_events(*args, host=host)

    def test_existing_schedule_and_bank_gates(self):
        for host in ('cql', 'td3', 'rebrac'):
            for change in ('scan', 'refresh', 'seed', 'count', 'bank'):
                p, r, e = fixture(host)
                if change == 'scan': r.pop()
                if change == 'refresh': r.append(dict(kind='refresh', step=1))
                if change == 'seed': e[0]['episode_seeds'] = [999]
                if change == 'count': e[0]['episode_count'] = 2
                if change == 'bank': e.pop()
                with self.assertRaises(ValueError): verify_events(p, r, e, host=host)

    def test_unknown_host_is_rejected(self):
        with self.assertRaises(ValueError): verify_events(*fixture('td3'), host='unknown')


if __name__ == '__main__':
    unittest.main()
