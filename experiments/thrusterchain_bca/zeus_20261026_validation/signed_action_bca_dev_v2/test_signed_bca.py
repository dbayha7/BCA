"""Signed fitting/calibration tests with the pinned TD target on synthetic data."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'action_sensitive_bca_dev_v1'))
import test_q1_bca as fixture
import signed_bca as signed


class SignedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = fixture.load_bridge()

    def setUp(self):
        self.args = SimpleNamespace(policy_noise=0., noise_clip=.5, discount=.98)
        self.batch = fixture.Batch(obs=jnp.array([[1., 0.], [2., 0.], [4., 0.]], dtype=jnp.float32),
            action=jnp.array([[.1,0,0,0], [.2,0,0,0], [.3,0,0,0]], dtype=jnp.float32),
            reward=jnp.array([1., 2., 3.], dtype=jnp.float32),
            next_obs=jnp.array([[5., 0.], [6., 0.], [7., 0.]], dtype=jnp.float32),
            done=jnp.array([0., 1., 0.], dtype=jnp.float32))
        self.native = fixture.Native(actor=fixture.Params(jnp.zeros(4)),
            actor_target=fixture.Params(jnp.array([.25,0,0,0])),
            critic=fixture.Params(jnp.array([10., -90.])),
            critic_target=fixture.Params(jnp.array([2., 4.])))
        model = signed.SignedResidual(jnp.zeros(2), jnp.ones(2))
        self.models = (fixture.Actor(), fixture.Critic(), model)
        params = model.init(jax.random.PRNGKey(10), self.batch.obs, self.batch.action)
        cal = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(.001))
        self.state = self.pipeline.State(self.native, cal, jnp.float32(1.),
                                          self.pipeline.initial_reference(cal.params))

    def test_signed_q1_target_and_jitted_update_preserve_native(self):
        q, y = signed.q1_and_target(self.pipeline, self.args, self.models, self.state,
                                    self.batch, jax.random.PRNGKey(0))
        np.testing.assert_allclose(q, [11.1,12.2,14.3], atol=1e-6)
        np.testing.assert_allclose(y, [1+.98*7.25,2,3+.98*9.25], atol=1e-6)
        fit = jax.jit(lambda s,b,k: signed.fit_step(self.pipeline,self.args,self.models,s,b,k))
        fitted, metrics = fit(self.state,self.batch,jax.random.PRNGKey(11))
        self.assertTrue(bool(metrics['signed_fit_accepted']))
        self.assertEqual(int(fitted.calibrator.step),1)
        for a,b in zip(jax.tree_util.tree_leaves(self.state.native),jax.tree_util.tree_leaves(fitted.native)):
            np.testing.assert_array_equal(a,b)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(
            jax.tree_util.tree_leaves(self.state.calibrator.params),
            jax.tree_util.tree_leaves(fitted.calibrator.params))))
        grad = jax.grad(lambda online: signed.q1_and_target(self.pipeline,self.args,self.models,
            self.state._replace(native=self.native._replace(critic=fixture.Params(online))),
            self.batch,jax.random.PRNGKey(3))[0].sum())(self.native.critic.params)
        np.testing.assert_array_equal(grad,jnp.zeros(2))

    def test_signed_calibration_and_split_order_statistic(self):
        indices = jnp.arange(64) % 3
        batch = jax.tree_util.tree_map(lambda x:x[indices],self.batch)
        result = signed.refresh(self.pipeline,self.args,self.models,self.state,batch,
                                jax.random.PRNGKey(9),heldout_ids=np.arange(64))
        np.testing.assert_allclose(result['residual'],np.array(result['q1'])-np.array(result['target']))
        np.testing.assert_allclose(result['scores'],(np.array(result['residual'])-np.array(result['mu']))/
                                   np.array(result['sigma']),rtol=1e-6)
        self.assertEqual(result['split_rank_1based'],59)
        self.assertEqual(result['q_split'],np.sort(result['scores'])[58])
        self.assertTrue(np.isfinite(result['q_bayes']))
        with self.assertRaisesRegex(ValueError,'episode IDs'):
            signed.refresh(self.pipeline,self.args,self.models,self.state,batch,jax.random.PRNGKey(9),
                           heldout_ids=np.zeros(64,dtype=int))

    def test_nonfinite_fit_rejects_without_updating(self):
        bad = self.batch._replace(reward=self.batch.reward.at[0].set(jnp.nan))
        updated, metrics = jax.jit(lambda s,b: signed.fit_step(
            self.pipeline,self.args,self.models,s,b,jax.random.PRNGKey(22)))(self.state,bad)
        self.assertFalse(bool(metrics['signed_fit_accepted']))
        self.assertEqual(int(updated.calibrator.step),0)
        for a,b in zip(jax.tree_util.tree_leaves(self.state),jax.tree_util.tree_leaves(updated)):
            np.testing.assert_array_equal(a,b)

    def test_initialization_checks_checkpoint_normalization(self):
        p = self.pipeline
        bridge = sys.modules['_q1_bca_test_bridge']
        data = SimpleNamespace(obs=jnp.arange(32,dtype=jnp.float32).reshape(8,4)/10,
                               action=jnp.zeros((8,4),jnp.float32))
        mean, std = data.obs.mean(axis=0), data.obs.std(axis=0)
        actor, critic = fixture.DeterministicTanhActor(mean,std), fixture.BoundCritic(mean,std)
        actor_ts = p.TrainState.create(apply_fn=actor.apply,params={'v':jnp.array(0.)},tx=p.optax.adam(.001))
        critic_ts = p.TrainState.create(apply_fn=critic.apply,params={'v':jnp.array(0.)},tx=p.optax.adam(.001))
        agent = fixture.Agent(actor_ts,actor_ts,critic_ts,critic_ts)
        context = SimpleNamespace(pipeline=p,data=SimpleNamespace(normalization_fixture=data),
            args=SimpleNamespace(gamma=.98,policy_noise=.2,noise_clip=.5),
            models=(bridge.DeterministicActorView(actor.apply),bridge.CriticView(critic.apply),None))
        models,state,_,_ = signed.initialize(context,agent,123,expected_mean=np.asarray(mean),expected_std=np.asarray(std))
        self.assertEqual(int(state.calibrator.step),0)
        self.assertEqual(models[2].apply(state.calibrator.params,data.obs,data.action)[0].shape,(8,))
        wrong = np.asarray(mean).copy(); wrong[0] += .01
        with self.assertRaisesRegex(ValueError,'authenticated manifest'):
            signed.initialize(context,agent,123,expected_mean=wrong,expected_std=np.asarray(std))

    def test_action_gradient_matches_finite_difference_and_clip(self):
        obs, a = self.batch.obs[:1],self.batch.action[:1]
        def fn(action):
            return signed.predict(self.models,self.state,obs,action,5.)[0].sum()
        g = jax.grad(fn)(a)
        step = jnp.zeros_like(a).at[0,0].set(.001)
        fd = (fn(a+step)-fn(a-step))/.002
        self.assertTrue(np.isfinite(np.asarray(g)).all())
        np.testing.assert_allclose(g[0,0],fd,rtol=.01,atol=.002)
        penalty,mu,sigma = signed.predict(self.models,self.state,obs,a,-1e6)
        np.testing.assert_array_equal(penalty,jnp.zeros_like(penalty))
        self.assertTrue(bool(jnp.all(sigma>=.01)))
        np.testing.assert_array_equal(jax.grad(lambda x:signed.predict(
            self.models,self.state,obs,x,-1e6)[0].sum())(a),jnp.zeros_like(a))


if __name__ == '__main__':
    unittest.main()
