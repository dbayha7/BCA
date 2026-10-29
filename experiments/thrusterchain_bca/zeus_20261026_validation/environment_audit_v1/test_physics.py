import unittest
import physics_checks


class PurePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=physics_checks.run_checks()

    def test_all_fixed_properties_pass(self):
        self.assertEqual(self.result['status'],'passed_fixed_physics_checks',self.result)
        self.assertEqual(len(self.result['checks']),11)
        self.assertTrue(all(v['passed'] for v in self.result['checks'].values()))

    def test_budget_and_no_environment_or_model(self):
        self.assertLessEqual(self.result['counts']['numpy_dynamics_rows'],128)
        self.assertLessEqual(self.result['counts']['jax_dynamics_rows'],32)
        for k in ['environments_constructed','resets','rollout_episodes','dataset_rows_decoded','model_calls','optimizer_updates']:
            self.assertEqual(self.result['counts'][k],0)

    def test_gym_stub_forbids_constructing_real_environment(self):
        with physics_checks.pure_canonical() as (src,loader):
            with self.assertRaisesRegex(AssertionError,'forbids environment spaces'):
                src.ThrusterChainEnv()


if __name__=='__main__':
    unittest.main()
