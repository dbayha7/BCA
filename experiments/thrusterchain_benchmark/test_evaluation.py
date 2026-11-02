"""Tests for bank identity, real environment semantics, and paired reporting."""

import unittest
import numpy as np

from experiments.shift import thruster_chain as tc
from experiments.thrusterchain_benchmark import evaluation as ev


class EvaluationTests(unittest.TestCase):
    def test_banks_are_reproducible_prefixes_and_roles_are_distinct(self):
        a = ev.starts(ev.BankSpec(3, "selection", 3), "recorded")
        b = ev.starts(ev.BankSpec(3, "selection", 5), "recorded")
        final = ev.starts(ev.BankSpec(3, "final", 3), "recorded")
        np.testing.assert_array_equal(a, b[:3])
        self.assertFalse(np.array_equal(a, final))
        kicks = ev.starts(ev.BankSpec(3, "selection", 3), "big_kicks")
        np.testing.assert_allclose(kicks[:, :tc.DS], 3 * a[:, :tc.DS])
        np.testing.assert_array_equal(kicks[:, tc.DS:], a[:, tc.DS:])

    def test_real_environment_rollout_and_pairing(self):
        seen = []
        def factory():
            def policy(obs):
                self.assertEqual(obs.dtype, np.float32)
                seen.append(obs.copy())
                return np.array([0.123456789, 0, 0, 0], np.float64)
            return policy
        spec = ev.BankSpec(4, "final", 2)
        result = ev.evaluate(factory, spec, "recorded")
        self.assertEqual(len(seen), 200)
        np.testing.assert_array_equal(result["steps"], [100, 100])
        env = tc.ThrusterChainEnv()
        state = ev.starts(spec, "recorded")[0]
        env.reset(options={"state": state})
        _, reward, _, _, info = env.step([0.123456789, 0, 0, 0])
        self.assertEqual(info["command"][0], float(np.float32(0.123456789)))
        np.testing.assert_array_equal(seen[1], info["state"].astype(np.float32))
        expected = reward
        for _ in range(99):
            _, reward, _, _, _ = env.step([0.123456789, 0, 0, 0])
            expected += reward
        self.assertEqual(result["returns"][0], expected)
        paired = ev.paired_difference(result, result)
        self.assertEqual(paired["mean"], 0)
        self.assertEqual(paired["episode_se"], 0)
        other = ev.evaluate(factory, ev.BankSpec(4, "selection", 2), "recorded")
        with self.assertRaises(ValueError):
            ev.paired_difference(result, other)


if __name__ == "__main__":
    unittest.main()
