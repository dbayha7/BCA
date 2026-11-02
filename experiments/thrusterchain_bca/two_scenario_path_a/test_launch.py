import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch as L
import worker as W


class LaunchTests(unittest.TestCase):
    def test_matrix_and_cumulative_budget(self):
        p, jobs = L.configuration(), L.jobs()
        self.assertEqual(len(jobs), 19)
        self.assertEqual(len({j['id'] for j in jobs}), 19)
        self.assertEqual(sum(j['arm'] == 'warmup' for j in jobs), 6)
        self.assertEqual(sum(j['arm'] in ('native', 'bca') for j in jobs), 12)
        self.assertEqual(1890 + p['final_preparation_reserved_seconds'] + p['total_process_cap_seconds'], 14400)
        self.assertEqual(189655 + p['evaluation_transition_cap'], 427255)
        self.assertFalse(p['training_authorized'])
        self.assertFalse((HERE / 'LAUNCH_APPROVAL.json').exists())

    def test_no_old_approval_or_reservation_inheritance(self):
        with TemporaryDirectory() as directory, patch.object(L, 'HERE', Path(directory)):
            L.save(Path(directory) / 'MANIFEST.json', {'test': True})
            expected = L.approval_expected()
            self.assertEqual(expected['path'], 'A')
            self.assertEqual(expected['process_cap_seconds'], 12450)
            self.assertIn('No Path B execution', expected['approval'])
            with self.assertRaises(FileNotFoundError):
                L.read(Path(directory) / 'LAUNCH_APPROVAL.json')
            L.save(Path(directory) / 'x.json', {'value': 1})
            with self.assertRaises(FileExistsError):
                L.save(Path(directory) / 'x.json', {'value': 2})

    def _completed_training(self, path):
        caps = {'critic': 10000, 'actor': 5000, 'auxiliary': 10000, 'calibrations': 1}
        L.save(path / 'report.json', {'status': 'completed_training_job', 'execution_counts_known': True,
                                    'counts': {'attempted': caps, 'completed': caps}, 'simulator_steps': 0})
        L.save(path / 'WORKER_EXIT.json', {'status': 'completed', 'identity_unchanged': True,
                                         'report_sha256': L.sha(path / 'report.json')})

    def test_success_exit_never_overrides_postcheck_failure(self):
        with TemporaryDirectory() as directory:
            p = Path(directory)
            self._completed_training(p)
            self.assertEqual(L.accepted_job(p, 0)['counts']['completed']['critic'], 10000)
            exit_receipt = L.read(p / 'WORKER_EXIT.json')
            exit_receipt['identity_unchanged'] = False
            (p / 'WORKER_EXIT.json').write_text(json.dumps(exit_receipt))
            with self.assertRaisesRegex(ValueError, 'postcheck'):
                L.accepted_job(p, 0)

    def test_report_replacement_and_nonzero_exit_rejected(self):
        with TemporaryDirectory() as directory:
            p = Path(directory)
            self._completed_training(p)
            with self.assertRaises(ValueError):
                L.accepted_job(p, 1)
            (p / 'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Report differs'):
                L.accepted_job(p, 0)

    def test_changed_source_rejected(self):
        with TemporaryDirectory() as directory, patch.object(L, 'REPO', Path(directory)):
            p = Path(directory) / 'source.py'
            p.write_text('x=1')
            pins = {'source.py': L.sha(p)}
            L.verify_pins(pins)
            p.write_text('x=2')
            with self.assertRaises(ValueError):
                L.verify_pins(pins)

    def test_physical_counter_failure_is_not_zero(self):
        def fail(x, a):
            raise RuntimeError('injected plant failure')
        c = W.PhysicalCounter(fail, lambda: None, lambda value: None)
        c.begin('own_qpi')
        with self.assertRaises(RuntimeError):
            c(np.zeros(13), np.zeros(4))
        self.assertEqual(c.attempted['own_qpi'], 1)
        self.assertEqual(c.completed['own_qpi'], 0)
        self.assertFalse(c.receipt()['execution_counts_known'])

    def test_phase_cap_stops_before_extra_plant_call(self):
        calls = []
        c = W.PhysicalCounter(lambda x, a: calls.append(1), lambda: None, lambda value: None)
        c.begin('prefix_parity')
        c.attempted['prefix_parity'] = c.completed['prefix_parity'] = 4800
        with self.assertRaises(ValueError):
            c(np.zeros(13), np.zeros(4))
        self.assertEqual(calls, [])

    def test_callback_q1_axis_and_frozen_width(self):
        class Distribution:
            def __init__(self, obs): self.obs = obs
            def sample(self, seed=None): return np.zeros((len(self.obs), 4), np.float32)
        actor = SimpleNamespace(params={}, apply_fn=lambda params, obs: Distribution(obs))
        critic = SimpleNamespace(params={}, apply_fn=lambda params, obs, act: np.tile([3., 9.], (len(obs), 1)))
        native = SimpleNamespace(jax=SimpleNamespace(jit=lambda fn: fn))
        pipeline = SimpleNamespace(live_width=lambda *args: (np.full(len(args[-1]), 2.), True, True))
        adapter = SimpleNamespace(context=SimpleNamespace(native=native, pipeline=pipeline, models=()))
        state = SimpleNamespace(microstep=SimpleNamespace(agent_state=SimpleNamespace(actor=actor, dual_q=critic)),
                                aux=SimpleNamespace(posterior={}))
        callbacks = W.callbacks(SimpleNamespace(tree_digest=lambda value: 'fixed'), adapter, state, {})
        obs, action = np.zeros((5, 14), np.float32), np.zeros((5, 4), np.float32)
        np.testing.assert_array_equal(callbacks['q1'](obs, action), np.full(5, 3.))
        self.assertEqual(callbacks['actor'](obs).dtype, np.float32)
        np.testing.assert_array_equal(callbacks['width'](obs, action), np.full(5, 2.))
        callbacks['assert_unchanged']()


if __name__ == '__main__':
    unittest.main()
