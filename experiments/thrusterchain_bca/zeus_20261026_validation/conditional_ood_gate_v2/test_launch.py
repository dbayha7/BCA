import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import launch


class LaunchTests(unittest.TestCase):
    def test_worker_budget_and_parent_are_required(self):
        with patch.object(launch.os, 'getppid', return_value=100):
            launch.require_supervised_worker(900, {'pid': 100})
            for value in (0, -1, 901, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    launch.require_supervised_worker(value, {'pid': 100})
            with self.assertRaises(ValueError):
                launch.require_supervised_worker(900, {'pid': 99})

    def test_unreadable_receipt_preserves_raw_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'report.json'
            p.write_text('interrupted partial JSON')
            value, error = launch.optional_receipt(p)
            self.assertIsNone(value)
            self.assertIn('JSONDecodeError', error)
            self.assertEqual(p.read_text(), 'interrupted partial JSON')

    def test_reservation_is_exclusive_and_failed_evidence_survives(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'run'
            launch.reserve(target, {'identity': 'synthetic'})
            original = (target / 'ATTEMPT.json').read_bytes()
            (target / 'failure.json').write_text('{"synthetic":true}')
            with self.assertRaises(FileExistsError):
                launch.reserve(target, {'identity': 'other'})
            self.assertEqual((target / 'ATTEMPT.json').read_bytes(), original)
            self.assertTrue((target / 'failure.json').exists())

    def test_manifest_tampering_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gate = root / 'gate'
            gate.mkdir()
            (gate / 'contract.json').write_text('{"caps":{"global_seconds":900}}')
            h = launch.sha(gate / 'contract.json')
            launch.write(gate / 'DECISION_FROZEN.json', {'contract_sha256': h}, exclusive=True)
            (root / 'code.py').write_text('original')
            launch.write(gate / 'PREPARED.json', {'contract_sha256': h, 'files': {'code.py': launch.sha(root / 'code.py')}}, exclusive=True)
            with patch.object(launch, 'HERE', gate), patch.object(launch, 'REPO', root):
                self.assertEqual(launch.verify_prepared()['verified_files'], 1)
                (root / 'code.py').write_text('changed')
                with self.assertRaises(ValueError):
                    launch.verify_prepared()

    def test_contract_change_does_not_refreeze(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'contract.json').write_text('{}')
            launch.write(root / 'DECISION_FROZEN.json', {'contract_sha256': 'not the hash'}, exclusive=True)
            with patch.object(launch, 'HERE', root):
                with self.assertRaises(ValueError):
                    launch.decision_contract()


if __name__ == '__main__':
    unittest.main()
