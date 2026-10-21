"""Native residual-band semantics, without fitting or changing a learner."""

from dataclasses import asdict
import json
import unittest

import numpy as np

from experiments.thrusterchain_diagnostics import training as T
from experiments.thrusterchain_diagnostics.training_eval import make_query_bank, _observation_batch


class TrainingCalibrationDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.thrusterchain_diagnostics.training_calibration import evaluate_calibration
        cls.evaluate = staticmethod(evaluate_calibration)
        cls.settings = T.TrainingSettings()
        cls.bank = make_query_bank(per_context=1)
        cls.results = {}
        # Nontrivial training-only normalization catches accidental raw inputs.
        mean = np.linspace(-0.2, 0.2, 15, dtype=np.float32)
        std = np.linspace(0.8, 1.2, 15, dtype=np.float32)
        mean[13:], std[13:] = 0.0, 1.0
        for arm in ("td3_bc", "bca"):
            carry, models, _, _ = T.initialize(cls.settings, arm, 7102)
            record = dict(status="completed", identity=dict(
                arm=arm, seed=7102, settings=asdict(cls.settings)))
            cls.results[arm] = T.TrainingResult(carry, models, mean.copy(), std.copy(), record)

    def with_reference(self, threshold):
        import jax.numpy as jnp
        result = self.results["bca"]
        reference = result.state.posterior._replace(
            threshold=jnp.asarray(threshold, jnp.float32),
            lambda_hat=jnp.asarray(threshold, jnp.float32),
            lambda_hpd=jnp.asarray(threshold, jnp.float32),
            n_eff=jnp.asarray(128.0, jnp.float32),
            ready=jnp.asarray(True))
        state = result.state._replace(posterior=reference)
        carry = (result.carry[0], state, result.carry[2])
        return T.TrainingResult(carry, result.models, result.mean, result.std, result.record)

    def test_host_has_no_fabricated_calibration_output(self):
        summary, arrays = self.evaluate(self.results["td3_bc"], self.bank)
        self.assertEqual(summary["status"], "absent")
        self.assertEqual(arrays, {})
        json.dumps(summary, allow_nan=False)

    def test_refused_bca_has_no_policy_or_calibration_output(self):
        base = self.results["bca"]
        result = T.TrainingResult(None, None, base.mean, base.std,
            dict(status="refused", reason="insufficient calibration episodes",
                 identity=base.record["identity"]))
        summary, arrays = self.evaluate(result, self.bank)
        self.assertEqual(summary["status"], "refused")
        self.assertEqual(arrays, {})
        json.dumps(summary, allow_nan=False)

    def test_unready_reference_is_unit_dose_without_coverage_claim(self):
        summary, arrays = self.evaluate(self.results["bca"], self.bank)
        self.assertEqual(summary["status"], "unready")
        self.assertIsNone(summary["coverage"])
        self.assertEqual(summary["band_refusal_rate"], 1.0)
        self.assertFalse(summary["executed_policy_refusal"])
        self.assertEqual(np.asarray(arrays["dose"]).shape, (12, 6))
        np.testing.assert_array_equal(arrays["dose"], np.ones((12, 6)))
        np.testing.assert_array_equal(arrays["width"], np.zeros((12, 6)))
        # Native pre-ready zero widths are usable; this is not certification.
        self.assertTrue(np.asarray(arrays["dose_usable"]).all())
        json.dumps(summary, allow_nan=False)

    def test_infinite_reference_is_vacuous_and_returns_native_unit_dose(self):
        summary, arrays = self.evaluate(self.with_reference(np.inf), self.bank)
        self.assertEqual(summary["status"], "uncertified")
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual(summary["band_refusal_rate"], 1.0)
        self.assertFalse(summary["executed_policy_refusal"])
        self.assertTrue(np.isposinf(arrays["width"]).all())
        self.assertFalse(np.asarray(arrays["dose_usable"]).any())
        np.testing.assert_array_equal(arrays["dose"], np.ones((12, 6)))
        json.dumps(summary, allow_nan=False)

    def test_terminal_target_is_raw_reward_and_residual_is_explicit(self):
        summary, arrays = self.evaluate(self.with_reference(1.0), self.bank)
        self.assertEqual(summary["status"], "evaluated")
        self.assertEqual(summary["band_refusal_rate"], 0.0)
        self.assertFalse(summary["executed_policy_refusal"])
        times = np.asarray(arrays["time"])
        self.assertEqual(times.shape, (12, 6))
        terminal = times == 99
        self.assertEqual(int(terminal.sum()), 24)
        np.testing.assert_array_equal(np.asarray(arrays["target"])[terminal],
                                      np.asarray(arrays["reward"])[terminal])
        np.testing.assert_allclose(arrays["residual"], np.asarray(arrays["target"]) - arrays["q"],
                                   atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(arrays["absolute_residual"], np.abs(arrays["residual"]))
        expected_coverage = float(np.mean(np.asarray(arrays["absolute_residual"]) <= arrays["width"]))
        self.assertEqual(summary["coverage"], expected_coverage)
        self.assertTrue(np.isfinite(arrays["width"]).all())
        self.assertTrue(np.asarray(arrays["dose_usable"]).all())
        self.assertTrue(np.all(np.asarray(arrays["dose"]) >= 1.0))
        self.assertTrue(np.all(np.asarray(arrays["dose"]) <= 1.0 + self.settings.blend))
        json.dumps(summary, allow_nan=False)

    def test_six_candidates_include_deployed_actor_and_current_twin_min(self):
        result = self.with_reference(1.0)
        _, arrays = self.evaluate(result, self.bank)
        obs = _observation_batch(self.bank.states, self.bank.regimes, self.bank.times,
                                 self.bank.objective, True)
        actions = np.asarray(arrays["actions"])
        self.assertEqual(actions.shape, (12, 6, 4))
        np.testing.assert_array_equal(actions[:, 0], np.zeros((12, 4)))
        np.testing.assert_array_equal(actions[:, -1], result.policy(obs).astype(np.float32))
        predictions = result.critic(np.repeat(obs, 6, axis=0), actions.reshape(-1, 4))
        np.testing.assert_allclose(arrays["q"], predictions.min(axis=-1).reshape(12, 6),
                                   atol=1e-6, rtol=1e-6)

    def test_diagnostic_is_deterministic_and_does_not_mutate_learner_or_query_bank(self):
        result = self.with_reference(1.0)
        before = T.tree_sha256(result.carry)
        mean, std = result.mean.copy(), result.std.copy()
        bank_before = {key: value.copy() for key, value in self.bank.arrays.items()}
        first_summary, first = self.evaluate(result, self.bank, seed=102001)
        second_summary, second = self.evaluate(result, self.bank, seed=102001)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(set(first), set(second))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        self.assertEqual(T.tree_sha256(result.carry), before)
        np.testing.assert_array_equal(result.mean, mean)
        np.testing.assert_array_equal(result.std, std)
        for key, value in bank_before.items():
            np.testing.assert_array_equal(self.bank.arrays[key], value, err_msg=key)


if __name__ == "__main__":
    unittest.main()
