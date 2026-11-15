"""Tests of Version C (BCA_IW_PLAN.md). Needs JAX.

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_version_c.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_baselines as B  # noqa: E402  (puts the shared learner and the repository root on the path)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
import published_collect as C  # noqa: E402
import version_c as VC  # noqa: E402
from calibration import wbcp  # noqa: E402
from calibration.reference import WBCPConfig  # noqa: E402

ROWS = 512


def small_data(seed=7):
    data = C.generate(E.published(), "mixed", seed)
    keep = np.arange(ROWS)
    return B.to_device({k: v[keep] for k, v in data.items()})


def ready_state(seed=0, cap=0.5, cert=0.0, threshold=1.0):
    """A state whose reference is ready, with a hand-made table: R rises from 1 to 3 over log mass -2 to 2."""
    state = L.init_state(seed, 100, B.ACTION_DIM)
    reference = dict(state["reference"], ready=jnp.asarray(True), threshold=jnp.float32(threshold),
                     width_cap=jnp.float32(cap), unit=jnp.float32(1.0),
                     iw_logm=jnp.linspace(-2.0, 2.0, VC.GRID_POINTS, dtype=jnp.float32),
                     iw_R=jnp.linspace(1.0, 3.0, VC.GRID_POINTS, dtype=jnp.float32),
                     iw_logm_cert=jnp.float32(cert), iw_ready=jnp.asarray(True))
    return dict(state, reference=reference)


class Table(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.scores = np.abs(rng.normal(size=400))
        self.logit = rng.normal(2.0, 1.0, 400)
        self.words = np.array([3, 5, 7, 11], np.uint32)

    def test_unit_weights_and_unit_mass_give_the_standard_threshold_exactly(self):
        standard = wbcp.calibrate(self.scores, np.random.default_rng(self.words)).threshold
        raw, used = VC.threshold_table(self.scores, self.words, np.ones(400), [1.0])
        self.assertEqual(raw[0], standard)
        self.assertEqual(used[0], standard)

    def test_the_threshold_never_falls_as_the_mass_grows_and_ends_uncertifiable(self):
        weights = VC.weights_of(self.logit)
        raw, used = VC.threshold_table(self.scores, self.words, weights, VC.mass_grid(weights))
        self.assertTrue(np.all(used[1:] >= used[:-1]))
        self.assertTrue(np.isfinite(used[0]))
        self.assertTrue(np.isinf(used[-1]))

    def test_the_last_certifiable_mass_is_where_the_calibration_says(self):
        """Nothing is certifiable once the test mass exceeds total weight x alpha / (1 - alpha)."""
        weights = VC.weights_of(self.logit)
        fields, record = VC.iw_fields(self.scores, self.logit, self.words, 1.5)
        boundary = np.log(weights.sum() * 0.1 / 0.9)
        self.assertLessEqual(float(fields["iw_logm_cert"]), boundary + 1e-6)
        # The Bayesian part of the calibration is stricter than that bound, but a typical recorded
        # command must still be certifiable.
        self.assertGreater(float(fields["iw_logm_cert"]), float(np.median(np.log(weights))))
        self.assertTrue(np.all(np.isfinite(fields["iw_R"])))
        self.assertEqual(fields["iw_logm"].shape, (VC.GRID_POINTS,))
        self.assertTrue(record["monotone"])

    def test_a_rarer_command_gets_a_larger_threshold(self):
        fields, _ = VC.iw_fields(self.scores, self.logit, self.words, 1.5)
        self.assertGreater(fields["iw_R"][-1], fields["iw_R"][0])


class Charge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = L.calibrator(100)
        cls.state = ready_state()
        cells = np.arange(0, 100, 3)
        cls.obs = jnp.asarray(np.eye(100, dtype=np.float32)[cells])
        rng = np.random.default_rng(0)
        cls.action = jnp.asarray(rng.uniform(-1, 1, (len(cells), B.ACTION_DIM)), jnp.float32)

    def numpy_charge(self, reference):
        logm = -np.asarray(L.support_logit(reference["support"], self.obs, self.action), np.float64)
        threshold = np.interp(logm, np.asarray(reference["iw_logm"]), np.asarray(reference["iw_R"]))
        eta = np.asarray(self.cal.apply(reference["scale"], self.obs, self.action), np.float64)
        scale = np.maximum(eta, 1e-6) * float(reference["unit"])
        certified = np.maximum(threshold - float(reference["threshold"]), 0.0) * scale
        decades = np.maximum(logm - float(reference["iw_logm_cert"]), 0.0) / np.log(10.0)
        beyond = np.maximum(certified, float(reference["width_cap"]) * (1.0 + decades))
        return np.where(logm > float(reference["iw_logm_cert"]), beyond, certified), logm

    def test_the_charge_matches_an_independent_numpy_version(self):
        for cert in (-5.0, 0.0, 5.0):
            reference = dict(self.state["reference"], iw_logm_cert=jnp.float32(cert))
            wanted, _ = self.numpy_charge(reference)
            got = np.asarray(VC.charge(self.cal, reference, self.obs, self.action))
            np.testing.assert_allclose(got, wanted, rtol=2e-5, atol=2e-6)

    def test_no_charge_before_the_table_is_ready_or_at_the_standard_threshold(self):
        blank = dict(self.state["reference"], iw_ready=jnp.asarray(False))
        self.assertEqual(float(jnp.abs(VC.charge(self.cal, blank, self.obs, self.action)).max()), 0.0)
        flat = dict(self.state["reference"], iw_R=jnp.ones((VC.GRID_POINTS,), jnp.float32), iw_logm_cert=jnp.float32(50.0))
        self.assertEqual(float(jnp.abs(VC.charge(self.cal, flat, self.obs, self.action)).max()), 0.0)

    def test_a_command_that_cannot_be_certified_is_charged_at_least_the_cap(self):
        reference = dict(self.state["reference"], iw_logm_cert=jnp.float32(-50.0))
        got = np.asarray(VC.charge(self.cal, reference, self.obs, self.action))
        self.assertTrue(np.all(np.asarray(VC.cannot_certify(reference, self.obs, self.action))))
        self.assertTrue(np.all(got >= float(reference["width_cap"])))

    def test_the_charge_has_a_slope_toward_the_data_where_it_cannot_certify(self):
        reference = dict(self.state["reference"], iw_logm_cert=jnp.float32(-50.0))
        grad = jax.grad(lambda a: jnp.sum(VC.charge(self.cal, reference, self.obs, a)))(self.action)
        self.assertGreater(float(jnp.abs(grad).max()), 0.0)

    def test_the_unsupported_loss_only_pushes_scores_down(self):
        state = self.state
        critics = (state["q1"], state["q2"])
        reference = dict(state["reference"], iw_logm_cert=jnp.float32(-50.0))
        q = np.asarray(L.critic_apply(state["q1"], self.obs, self.action))
        cap = float(reference["width_cap"])
        high = jnp.asarray(q + L.KAPPA * cap + 1.0)      # target minus the capped charge is above every score
        self.assertEqual(float(VC.unsupported_loss(critics, self.cal, reference, self.obs, self.action, high)), 0.0)
        low = jnp.asarray(q - 10.0)           # bound far below: every flagged score is pushed down
        self.assertGreater(float(VC.unsupported_loss(critics, self.cal, reference, self.obs, self.action, low)), 0.0)
        certified = dict(state["reference"], iw_logm_cert=jnp.float32(50.0))
        self.assertEqual(float(VC.unsupported_loss(critics, self.cal, certified, self.obs, self.action, low)), 0.0)

    def test_the_target_is_lowered_by_kappa_times_the_charge_at_the_next_command(self):
        state = self.state
        reference = dict(state["reference"], iw_logm_cert=jnp.float32(-50.0))
        targets = (state["actor_t"], state["q1_t"], state["q2_t"])
        reward, done = jnp.zeros(self.obs.shape[0]), jnp.zeros(self.obs.shape[0])
        noise = jnp.zeros_like(self.action)
        plain = L.bellman_target(*targets, self.obs, reward, done, noise)
        lowered = VC.target_c(self.cal, reference, *targets, self.obs, reward, done, noise)
        next_command = L.smoothed_action(state["actor_t"], self.obs, noise)
        capped = np.asarray(VC.charge(self.cal, reference, self.obs, next_command, capped=True))
        np.testing.assert_allclose(capped, float(reference["width_cap"]), rtol=1e-6)      # all uncertified here
        wanted = np.asarray(plain) - L.GAMMA * L.KAPPA * capped
        np.testing.assert_allclose(np.asarray(lowered), wanted, rtol=1e-5, atol=1e-5)
        uncapped = np.asarray(VC.charge(self.cal, reference, self.obs, next_command))
        self.assertTrue(np.all(uncapped > capped))


class Step(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = small_data()
        cls.key = jax.random.PRNGKey(5)
        cls.shared = staticmethod(jax.jit(L.make_step("one_hot", ROWS)))
        cls.mine = staticmethod(jax.jit(VC.make_step("one_hot", ROWS)))

    def run_steps(self, step, state, mode, count=6):
        for t in range(1, count + 1):
            state = step(state, self.data, self.key, jnp.int32(mode), jnp.int32(t))
        return state

    def test_every_other_mode_gives_the_shared_learners_numbers_to_float_rounding(self):
        """Not bit for bit: the compiler rounds a few sums differently once the extra terms exist
        (differences of about 4e-9). So only VERSION_C runs are trained with this step."""
        for mode in (L.NATIVE, L.SUPPORT_BLIND_WIDTH, L.SUPPORT_SHUFFLED):
            state = ready_state()
            a = self.run_steps(self.shared, {k: v for k, v in state.items()}, mode)
            b = self.run_steps(self.mine, state, mode)
            for name in ("actor", "q1", "q2", "scale", "support", "actor_t", "q1_t"):
                for x, y in zip(jax.tree_util.tree_leaves(a[name]), jax.tree_util.tree_leaves(b[name])):
                    np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-6,
                                               err_msg=f"mode {mode}, {name}")

    def test_version_c_is_the_baseline_until_its_table_is_ready(self):
        state = ready_state()
        blank = dict(state, reference=dict(state["reference"], iw_ready=jnp.asarray(False)))
        a = self.run_steps(self.mine, blank, L.NATIVE)
        b = self.run_steps(self.mine, blank, VC.VERSION_C)
        for name in ("actor", "q1", "q2"):
            for x, y in zip(jax.tree_util.tree_leaves(a[name]), jax.tree_util.tree_leaves(b[name])):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_version_c_changes_the_critic_and_the_policy_once_ready(self):
        state = ready_state(cert=-50.0)
        a = self.run_steps(self.mine, state, L.NATIVE)
        b = self.run_steps(self.mine, state, VC.VERSION_C)
        for name in ("actor", "q1"):
            gap = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(a[name]),
                                                                  jax.tree_util.tree_leaves(b[name])))
            self.assertGreater(gap, 1e-6, name)
        for leaf in jax.tree_util.tree_leaves(b):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))) or leaf.dtype == jnp.bool_)

    def test_the_segment_runs_many_runs_with_mixed_modes(self):
        segment = VC.make_segment("one_hot", ROWS)
        states = L.stack([ready_state(seed) for seed in (0, 1, 2)])
        data = L.stack([self.data] * 3)
        keys = jnp.stack([jax.random.PRNGKey(s) for s in (0, 1, 2)])
        modes = jnp.asarray([L.NATIVE, VC.VERSION_C, L.SUPPORT_BLIND_WIDTH])
        out = segment(states, data, keys, modes, jnp.arange(1, 5, dtype=jnp.int32))
        self.assertEqual(np.asarray(out["step"]).tolist(), [4, 4, 4])
        self.assertEqual(out["reference"]["iw_R"].shape, (3, VC.GRID_POINTS))


if __name__ == "__main__":
    unittest.main()
