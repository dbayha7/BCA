"""Mock-only refusal tests; no MuJoCo construction or model queries."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

from experiments.ood.real_connection import AccountedSimulator
from experiments.ood.resources import ResourceLedger, scope_caps


class DummyMujoco:
    def do_simulation(self, control, frames):
        return None


class DummyAdapter:
    constructions = 0
    fail_step = False
    double_constructor = False

    def __init__(self, dataset, **kwargs):
        type(self).constructions += 1
        env = DummyMujoco()
        env.do_simulation(np.zeros(3, np.float32), 4)
        if self.double_constructor:
            env.do_simulation(np.zeros(3, np.float32), 4)
        self.identity = {'kind':'mock'}

    def capture(self):
        return {'mock_state': 1}

    def step(self, action):
        if self.fail_step:
            raise RuntimeError('physical call uncertain')
        return {'mock_record': np.asarray(action)}

    def close(self):
        pass


class AccountedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.ledger = ResourceLedger(self.path/'ledger.sqlite', scope_caps([], ['td3_bc/hopper']), 'fixture')
        module = types.ModuleType('gym.envs.mujoco.mujoco_env')
        module.MujocoEnv = DummyMujoco
        self.patches = [patch.dict('sys.modules', {'gym.envs.mujoco.mujoco_env': module}),
                        patch('experiments.ood.simulator.LocomotionAdapter', DummyAdapter)]
        for p in self.patches:
            p.start()
        DummyAdapter.constructions = 0
        DummyAdapter.fail_step = DummyAdapter.double_constructor = False

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.ledger.__exit__()
        self.temp.cleanup()

    def make(self, **kw):
        args = dict(ledger=self.ledger, scope='engineering/td3_bc/hopper', token='attempt',
                    dataset='hopper-medium-v2', path=self.path/'sim', max_transitions=8)
        args.update(kw)
        return AccountedSimulator(**args)

    def test_constructor_reserved_once_before_physics_and_observed(self):
        sim = self.make()
        self.assertEqual(self.ledger.audit()['reserved']['global'], [1,4])
        self.assertTrue((self.path/'sim/constructor-control-before-physics.json').exists())
        sim.close()
        with self.assertRaises(ValueError):
            self.make(path=self.path/'other')
        self.assertEqual(DummyAdapter.constructions, 1)

    def test_unknown_scope_dataset_and_budget_refused_before_constructor(self):
        for kwargs in ({'scope':'outcomes/pair'}, {'dataset':'walker2d-medium-replay-v2'},
                       {'max_transitions':9}, {'max_transitions':True}):
            with self.assertRaises(ValueError):
                self.make(**kwargs)
        self.assertEqual(DummyAdapter.constructions, 0)
        self.assertEqual(self.ledger.audit()['entries'], 0)

    def test_extra_constructor_step_stops_before_extra_physics(self):
        DummyAdapter.double_constructor = True
        with self.assertRaises(ValueError):
            self.make()
        self.assertEqual(self.ledger.audit()['reserved']['global'], [1,4])
        self.assertTrue((self.path/'sim/constructor-control-before-physics.json').exists())
        self.assertFalse((self.path/'sim/constructor-completed.json').exists())

    def test_failed_step_retained_poisoned_and_not_refunded(self):
        sim = self.make()
        DummyAdapter.fail_step = True
        with self.assertRaises(RuntimeError):
            sim.step(np.zeros(3, np.float32))
        self.assertTrue((self.path/'sim/call-01-failed.json').exists())
        with self.assertRaises(ValueError):
            sim.step(np.zeros(3, np.float32))
        self.assertEqual(self.ledger.audit()['reserved']['global'], [2,8])
        sim.close()

    def test_step_ceiling_stops_before_callback(self):
        sim = self.make(max_transitions=1)
        sim.step(np.zeros(3, np.float32))
        with self.assertRaises(ValueError):
            sim.step(np.zeros(3, np.float32))
        self.assertEqual(self.ledger.audit()['reserved']['global'], [2,8])
        sim.close()


if __name__ == '__main__':
    unittest.main()
