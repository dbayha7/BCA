"""Synthetic checks of actual Q1 fit/refresh against the pinned target kernel."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import q1_bca

ROOT = Path(__file__).resolve().parents[4]
BRIDGE = ROOT / 'experiments/thrusterchain_bca/titan_20261026/payload/unifloral_bca_pipeline_validation_v1/bca_bridge.py'


class Batch(NamedTuple):
    obs: object
    action: object
    reward: object
    next_obs: object
    done: object


class Params(NamedTuple):
    params: object


class Native(NamedTuple):
    actor: object
    actor_target: object
    critic: object
    critic_target: object


class CalState(NamedTuple):
    params: object
    step: object

    def apply_gradients(self, *, grads):
        new_params = jax.tree_util.tree_map(lambda p, g: p - .001 * g, self.params, grads)
        return self._replace(params=new_params, step=self.step + 1)


class Actor:
    def apply(self, params, observations):
        return jnp.broadcast_to(params, (observations.shape[0], 4))


class Critic:
    def apply(self, params, observations, actions):
        base = observations[:, 0] + actions[:, 0]
        return jnp.stack((base + params[0], base + params[1]), axis=-1)


class Scale:
    def apply(self, params, observations, actions):
        return jnp.exp(params['bias'] + .1 * actions[:, 0])


class DeterministicTanhActor:
    def __init__(self, mean, std):
        self.obs_mean, self.obs_std = mean, std

    def apply(self, params, obs):
        return obs


class BoundCritic:
    def __init__(self, mean, std):
        self.obs_mean, self.obs_std = mean, std

    def apply(self, params, obs, action):
        return jnp.zeros((len(obs), 2))


class Agent(NamedTuple):
    actor: object
    actor_target: object
    dual_q: object
    dual_q_target: object


def load_bridge():
    spec = importlib.util.spec_from_file_location('_q1_bca_test_bridge', BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.load_pipeline()


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_bridge()

    def setUp(self):
        self.pipe = q1_bca.make_q1_pipeline(self.original)
        self.args = SimpleNamespace(policy_noise=0., noise_clip=.5, discount=.98)
        self.config = self.original.Config(arm='bca', posterior=self.original.WBCPConfig(), blend=1.0)
        self.batch = Batch(obs=jnp.array([[1., 0.], [2., 0.], [4., 0.]], dtype=jnp.float32),
            action=jnp.array([[.1,0,0,0], [.2,0,0,0], [.3,0,0,0]], dtype=jnp.float32),
            reward=jnp.array([1., 2., 3.], dtype=jnp.float32),
            next_obs=jnp.array([[5., 0.], [6., 0.], [7., 0.]], dtype=jnp.float32),
            done=jnp.array([0., 1., 0.], dtype=jnp.float32))
        self.native = Native(actor=Params(jnp.zeros(4)), actor_target=Params(jnp.array([.25,0,0,0])),
            critic=Params(jnp.array([10., -90.])), critic_target=Params(jnp.array([2., 4.])))
        self.models = (Actor(), Critic(), Scale())
        cal = CalState({'bias': jnp.array(.1)}, jnp.int32(0))
        self.state = self.original.State(self.native, cal, jnp.float32(1.),
                                         self.original.initial_reference(cal.params))

    def test_q1_center_and_native_twin_min_target(self):
        key = jax.random.PRNGKey(8)
        q, y = self.pipe.q1_and_target(self.args, self.models, self.native, self.batch, key)
        np.testing.assert_allclose(q, [11.1, 12.2, 14.3], atol=1e-6)
        np.testing.assert_allclose(y, [1+.98*7.25, 2, 3+.98*9.25], atol=1e-6)
        original_target = self.original.native_target(self.args, self.models, self.native, self.batch, key, 1.)
        np.testing.assert_array_equal(y, original_target)
        # Both online Q1 and the target must be detached from scale fitting.
        grad = jax.grad(lambda online: self.pipe.q1_and_target(self.args, self.models,
            self.native._replace(critic=Params(online)), self.batch, key)[0].sum())(self.native.critic.params)
        np.testing.assert_array_equal(grad, jnp.zeros(2))

    def test_fitting_is_jittable_and_uses_q1_residual_units(self):
        key = jax.random.PRNGKey(24)
        step = jax.jit(lambda state, batch, rng: self.pipe.fit_scale(
            self.args, self.config, self.models, state, batch, rng))
        fitted, metrics = step(self.state, self.batch, key)
        self.assertTrue(bool(metrics['scale_fit_accepted']))
        self.assertEqual(int(fitted.calibrator.step), 1)
        np.testing.assert_array_equal(fitted.native.critic.params, self.native.critic.params)
        q, target = self.pipe.q1_and_target(self.args, self.models, self.native, self.batch,
                                            jax.random.fold_in(key, 1179210836))
        expected = .99 + .01 * float(jnp.maximum(jnp.std(target-q), 1e-6))
        self.assertAlmostEqual(float(fitted.residual_scale), expected, places=6)
        qmin = self.models[1].apply(self.native.critic.params, self.batch.obs, self.batch.action).min(axis=-1)
        self.assertFalse(np.allclose(np.asarray(target-q), np.asarray(target-qmin)))

    def test_exact_original_fit_parity_when_online_heads_agree(self):
        # On a case where min(Q1,Q2) equals Q1, every fit intermediate and the
        # resulting auxiliary state should coincide with the pinned code.
        native = self.native._replace(critic=Params(jnp.array([10., 10.])))
        state = self.state._replace(native=native)
        key = jax.random.PRNGKey(193)
        original = jax.jit(lambda s: self.original.fit_scale(
            self.args, self.config, self.models, s, self.batch, key))(state)
        aligned = jax.jit(lambda s: self.pipe.fit_scale(
            self.args, self.config, self.models, s, self.batch, key))(state)
        for a, b in zip(jax.tree_util.tree_leaves(original), jax.tree_util.tree_leaves(aligned)):
            np.testing.assert_array_equal(a, b)

    def test_refresh_uses_q1_in_calibration_and_checks_row_identity(self):
        observed = {}
        p = SimpleNamespace(**vars(self.original))
        def freeze(params, unit, scale, residual, key, config):
            observed['residual'] = np.asarray(residual)
            return self.state.posterior, True, {'certified': True}
        p.freeze_reference = freeze
        pipe = q1_bca.make_q1_pipeline(p)
        rng = jax.random.PRNGKey(4)
        result, flags, _ = pipe.refresh(self.args, self.config, self.models, self.state, self.batch,
                                        rng, heldout_ids=np.array([3, 4, 5]))
        self.assertTrue(flags['posterior_inputs_valid'])
        self.assertTrue(flags['posterior_certified'])
        q, target = pipe.q1_and_target(self.args, self.models, self.native, self.batch,
                                       jax.random.fold_in(rng, 1213156420))
        np.testing.assert_allclose(observed['residual'], target-q, atol=1e-6)
        np.testing.assert_array_equal(result.native.critic.params, self.native.critic.params)
        with self.assertRaises(ValueError):
            pipe.refresh(self.args, self.config, self.models, self.state, self.batch,
                         rng, heldout_ids=np.array([3, 3, 5]))

    def test_shape_mismatch_refused_and_width_is_original(self):
        class BadCritic:
            def apply(self, params, obs, actions):
                return jnp.zeros((len(obs),))
        with self.assertRaises(ValueError):
            self.pipe.q1_and_target(self.args, (Actor(), BadCritic(), Scale()), self.native,
                                    self.batch, jax.random.PRNGKey(2))
        self.assertIs(self.pipe.live_width, self.original.live_width)

    def test_real_reference_freeze_uses_q1_with_live_action_gradient(self):
        # A distinct, episode-ID-labelled synthetic calibration bank; no model
        # or simulator evaluation. Critic heads differ by 100 units, so the
        # asserted scale below would fail under twin-min calibration.
        idx = jnp.arange(64) % len(self.batch.obs)
        held = jax.tree_util.tree_map(lambda x: x[idx], self.batch)
        state, flags, diagnostics = self.pipe.refresh(self.args, self.config, self.models, self.state, held,
                jax.random.PRNGKey(92), heldout_ids=np.arange(64))
        self.assertTrue(flags['posterior_inputs_valid'])
        self.assertTrue(flags['posterior_certified'])
        self.assertTrue(bool(state.posterior.ready))
        self.assertTrue(np.isfinite(float(state.posterior.threshold)))
        # Q1 residuals on this fixture are all well below the min-Q residuals.
        q, y = self.pipe.q1_and_target(self.args, self.models, self.native, held,
                                       jax.random.fold_in(jax.random.PRNGKey(92), 1213156420))
        eta = self.models[2].apply(state.calibrator.params, held.obs, held.action)
        self.assertLess(float(state.posterior.threshold), 20.)
        self.assertGreater(float(jnp.max(jnp.abs(y-q) / eta)), 0.)
        obs = held.obs[:2]
        action = held.action[:2]
        def live_sum(a):
            width, usable, valid = self.pipe.live_width(self.models, state.posterior, obs, a)
            return jnp.where(usable & valid, width.sum(), jnp.nan)
        g = jax.grad(live_sum)(action)
        self.assertTrue(np.isfinite(np.asarray(g)).all())
        self.assertGreater(float(jnp.linalg.norm(g)), 0.)
        eps = 1e-3
        unit = jnp.zeros_like(action).at[0, 0].set(eps)
        finite_difference = (live_sum(action+unit)-live_sum(action-unit))/(2*eps)
        np.testing.assert_allclose(g[0, 0], finite_difference, rtol=.003, atol=.003)

    def test_manifest_normalized_fresh_initialization_matches_original_bridge(self):
        # Real pinned bridge and Calibrator, synthetic fixture and model
        # bindings. When fixture statistics do agree, the development-only
        # initializer must produce exactly the same fresh auxiliary state.
        p = self.original
        bridge = sys.modules['_q1_bca_test_bridge']
        fixture = SimpleNamespace(obs=jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 10,
                                  action=jnp.zeros((8, 4), jnp.float32))
        mean, std = fixture.obs.mean(axis=0), fixture.obs.std(axis=0)
        actor, critic = DeterministicTanhActor(mean, std), BoundCritic(mean, std)
        actor_ts = p.TrainState.create(apply_fn=actor.apply, params={'v': jnp.array(0.)},
                                       tx=p.optax.adam(.001))
        critic_ts = p.TrainState.create(apply_fn=critic.apply, params={'v': jnp.array(0.)},
                                        tx=p.optax.adam(.001))
        agent = Agent(actor_ts, actor_ts, critic_ts, critic_ts)
        args = SimpleNamespace(gamma=.98, policy_noise=.2, noise_clip=.5)
        ctx = SimpleNamespace(pipeline=p, data=SimpleNamespace(normalization_fixture=fixture), args=args,
                              models=(bridge.DeterministicActorView(actor.apply),
                                      bridge.CriticView(critic.apply), None))
        old_models, old_state, old_cfg, old_args = p.initialize_pipeline(agent, fixture, args, 12345)
        new_models, new_state, new_cfg, new_args = q1_bca.fresh_auxiliary_exact_normalization(
            ctx, agent, 12345, expected_mean=np.asarray(mean), expected_std=np.asarray(std))
        self.assertEqual(old_cfg, new_cfg)
        self.assertEqual(vars(old_args), vars(new_args))
        self.assertEqual(int(new_state.calibrator.step), 0)
        for old, new in zip(jax.tree_util.tree_leaves(old_state), jax.tree_util.tree_leaves(new_state)):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(old_models[2].apply(old_state.calibrator.params, fixture.obs, fixture.action),
                                      new_models[2].apply(new_state.calibrator.params, fixture.obs, fixture.action))
        wrong = np.asarray(mean).copy(); wrong[0] += .01
        with self.assertRaisesRegex(ValueError, 'authenticated manifest'):
            q1_bca.fresh_auxiliary_exact_normalization(ctx, agent, 12345, expected_mean=wrong,
                                                       expected_std=np.asarray(std))
        other_critic = BoundCritic(mean + .01, std)
        wrong_agent = agent._replace(dual_q_target=critic_ts.replace(apply_fn=other_critic.apply))
        with self.assertRaisesRegex(ValueError, 'authenticated manifest'):
            q1_bca.fresh_auxiliary_exact_normalization(ctx, wrong_agent, 12345,
                                                       expected_mean=np.asarray(mean), expected_std=np.asarray(std))


if __name__ == '__main__':
    unittest.main()
