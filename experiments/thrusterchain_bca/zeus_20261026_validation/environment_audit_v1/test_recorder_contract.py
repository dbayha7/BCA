"""Pure recorder/role tests: no dataset decode, model or simulator calls."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parents[1] / 'titan_20261026/payload/transfer_bundle/core_v4'


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recorder = module_at('environment_audit_recorder', CORE / 'thrusterchain/datasets.py')
logged = module_at('environment_audit_role_reader', HERE.parent / 'bca_one_sided_v1/logged_data.py')


class RecorderContractTests(unittest.TestCase):
    def test_uniform_density_requires_quarter_mixture(self):
        actions = np.zeros((5,4))
        actions[:,0] = [-1.,-.25,0.,.5,1.]
        np.testing.assert_allclose(np.exp(recorder.behavior_log_density(actions,.25)),1/16,rtol=1e-15)
        density = np.exp(recorder.behavior_log_density(actions,.1))
        np.testing.assert_allclose(density[:4],.9/12,rtol=1e-15)
        self.assertAlmostEqual(density[4],.1/4)
        self.assertTrue(np.isneginf(recorder.behavior_log_density([1.1,0,0,0])))

    def test_quantization_measure_is_not_continuous_density(self):
        semantics = recorder.DENSITY_SEMANTICS
        self.assertEqual(semantics['executed_action_measure'],'discrete_float32_quantization_mass_not_evaluated')
        self.assertEqual(semantics['joint_state_action_density'],'unavailable')
        self.assertEqual(semantics['transition_measure'],'deterministic_dirac')

    def test_episode_streams_are_reproducible_separate_and_extension_stable(self):
        a = recorder._generator_metadata(179,10)
        b = recorder._generator_metadata(179,20)
        self.assertEqual(a['episodes'],b['episodes'][:10])
        self.assertEqual(len({x['reset_seed'] for x in b['episodes']}),20)
        for episode in [0,1,9]:
            reset = recorder._stream_seed_sequence(179,'reset',episode).generate_state(4)
            action = recorder._stream_seed_sequence(179,'actions',episode).generate_state(4)
            self.assertFalse(np.array_equal(reset,action))
        first = np.random.Generator(np.random.PCG64(recorder._stream_seed_sequence(179,'actions',0)))
        second = np.random.Generator(np.random.PCG64(recorder._stream_seed_sequence(179,'actions',0)))
        # Consuming the separate reset stream must not move action generation.
        np.random.Generator(np.random.PCG64(recorder._stream_seed_sequence(179,'reset',0))).random(17)
        for _ in range(16):
            x = recorder.sample_requested_action(first)
            y = recorder.sample_requested_action(second)
            np.testing.assert_array_equal(x[0],y[0])
            self.assertEqual(x[1:],y[1:])
            self.assertTrue(np.all(np.abs(x[0])<=1))

    def test_group_splits_reproducible_but_extension_not_fixed_membership(self):
        codes,counts = recorder._split_assignment(1280,179)
        np.testing.assert_array_equal(counts,[768,256,128,128])
        np.testing.assert_array_equal(codes,recorder._split_assignment(1280,179)[0])
        extended,_ = recorder._split_assignment(1290,179)
        self.assertFalse(np.array_equal(codes,extended[:1280]))

    def test_protected_roles_rejected_before_archive_access(self):
        class ForbiddenArchive:
            def open(self,*args):
                raise AssertionError('Protected role must fail before a file is opened')
        codes = np.zeros(1280,dtype=np.uint8)
        codes[17:19] = [2,3]
        for index in [17,18]:
            with self.assertRaisesRegex(ValueError,'Forbidden selection/final-calibration'):
                logged.read_episodes(ForbiddenArchive(),'observations',np.array([index]),codes)

    def test_invalid_density_arguments_fail_explicitly(self):
        for epsilon in [True,-.1,1.1,float('nan')]:
            with self.assertRaises(recorder.DatasetError):
                recorder.behavior_log_density(np.zeros(4),epsilon)
        for action in [[0,0,0],[float('nan'),0,0,0],[True]*4]:
            with self.assertRaises(recorder.DatasetError):
                recorder.behavior_log_density(action)


if __name__ == '__main__':
    unittest.main()
