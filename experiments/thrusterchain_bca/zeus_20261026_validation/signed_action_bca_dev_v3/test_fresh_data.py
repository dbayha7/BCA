import json
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import fresh_data as f


class FakeEnv:
    def __init__(self):
        self.t = 0
    def reset(self, seed):
        self.t = 0
        return self.obs(), {}
    def obs(self):
        o = np.zeros(14, np.float32)
        o[10] = .5
        o[-1] = (100 - self.t) / 100
        return o
    def close(self):
        pass


class FakeReference:
    @staticmethod
    def _check_time(deadline):
        pass
    @staticmethod
    def _env(core):
        return FakeEnv()
    @staticmethod
    def _check_env(*args):
        pass
    @staticmethod
    def _step(env, action, ledger, stage, deadline, progress):
        assert action.dtype == np.float32 and stage == 'calibration'
        env.t += 1
        ledger['transition_steps_attempted'] += 1
        ledger['transition_steps_completed'] += 1
        return env.obs(), float(action[0])


class FreshDataTest(unittest.TestCase):
    def test_seed_collision_is_rejected(self):
        recorder = f.module('_recorder_for_test', f.RECORDER)
        protocol = json.loads((f.RERUN / 'protocol.json').read_text())
        derived = recorder._episode_reset_seed(f.ROOT_SEED, 0)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            f.seed_audit(recorder, [derived], protocol)

    def test_collection_roles_selection_and_intact_tuples(self):
        recorder = f.module('_recorder_for_collection_test', f.RECORDER)
        logged = SimpleNamespace(EXPECTED_DATA_SHA='fixture', provenance=lambda: {
            'vectors': {'episode_seeds': np.arange(10)},
            'metadata': {'behavior': {'name': 'continuous_region_mixture_v1', 'epsilon': .25}}})
        def load(name, path):
            return recorder if 'recorder' in name else logged
        native = SimpleNamespace(Transition=namedtuple('Transition', 'obs action reward next_obs done'))
        ledger = {'max_resets': 256, 'resets_attempted': 0, 'resets_completed': 0,
                  'transition_steps_attempted': 0, 'transition_steps_completed': 0}
        with tempfile.TemporaryDirectory() as tmp, patch.object(f, 'module', side_effect=load):
            cal, check, receipt = f.collect(native, None, FakeReference, ledger, None, tmp)
            self.assertEqual(ledger['transition_steps_completed'], 12800)
            self.assertEqual(len(cal.obs), 64)
            self.assertEqual(len(check.obs), 64)
            self.assertFalse(set(receipt['calibration_row_ids']) & set(receipt['check_row_ids']))
            self.assertTrue(np.all(np.abs(cal.action[:, 0]) > .25))
            self.assertTrue(np.all(np.abs(check.action[:, 0]) > .25))
            with np.load(Path(tmp) / receipt['artifact']) as bank:
                t = bank['selected_times']
                np.testing.assert_array_equal(cal.obs, bank['observations'][np.arange(64), t[:64]])
                np.testing.assert_array_equal(check.next_obs, bank['observations'][np.arange(64, 128), t[64:] + 1])
                np.testing.assert_array_equal(cal.done, (t[:64] == 99).astype(np.float32))


if __name__ == '__main__':
    unittest.main()
