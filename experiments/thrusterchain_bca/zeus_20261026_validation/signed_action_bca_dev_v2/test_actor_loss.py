"""No-update tests of pinned actor objective and frozen reference binding."""
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'action_sensitive_bca_dev_v1'))
import compatibility
import revised_compat
import signed_bca
import actor_loss
from test_actor_penalty import candidate_module, Distribution


class Batch(NamedTuple):
    obs: object
    action: object


def actor(params,obs): return Distribution(jnp.tanh(params @ obs))


def critic(params,obs,action):
    del obs
    value=jnp.sum(params*action)
    return jnp.stack((value,.5*value))


class ActorLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(484)
        obs=rng.normal(size=(800,14)).astype(np.float32)
        times=np.repeat(np.arange(0,100,10),80);obs[:,-1]=1-times/100
        actions=rng.uniform(-.5,.5,(800,4)).astype(np.float32)
        episodes=np.arange(800,dtype=np.int64)
        base=compatibility.fit_reference(obs,actions,episodes,times,k=64)
        cls.reference=revised_compat.Reference(base,1e10,1.)
        cls.obs=obs[:4];cls.actions=actions[:4];cls.ids=episodes[:4];cls.times=times[:4]
        cls.model=signed_bca.SignedResidual(jnp.zeros(14),jnp.ones(14))
        cls.model_params=cls.model.init(jax.random.PRNGKey(39),jnp.asarray(cls.obs),jnp.asarray(cls.actions))
        cls.bound=actor_loss.bind_training_batch(cls.reference,cls.obs,cls.times,cls.ids,
                                                  signed_params=cls.model_params,threshold=4.)
        cls.batch=Batch(jnp.asarray(cls.obs),jnp.asarray(cls.actions))
        cls.params=jnp.asarray(rng.normal(scale=.05,size=(4,14)),dtype=jnp.float32)
        cls.agent=SimpleNamespace(dual_q=SimpleNamespace(params=jnp.array([1.,.7,1.1,.9])))
        cls.args=SimpleNamespace(num_critic_updates_per_step=2,batch_size=4,td3_alpha=2.5)
        cls.candidate=candidate_module()

    def test_default_off_exact_native_value_and_gradient(self):
        original=self.candidate.make_microsteps(self.args,actor,critic,self.batch,width_fn=None,kappa_penalty=0.)
        off=actor_loss.make_actor_loss(self.candidate,self.args,actor,critic,self.batch)
        expected=original.actor_loss(self.params,self.agent,self.batch,None)[0]
        actual=off(self.params,self.agent,self.batch)[0]
        np.testing.assert_array_equal(actual,expected)
        want=jax.grad(lambda p:original.actor_loss(p,self.agent,self.batch,None)[0])(self.params)
        got=jax.grad(lambda p:off(p,self.agent,self.batch)[0])(self.params)
        np.testing.assert_array_equal(got,want)

    def test_exact_added_penalty_with_live_actor_gradient(self):
        baseline=actor_loss.make_actor_loss(self.candidate,self.args,actor,critic,self.batch)
        signed=actor_loss.make_actor_loss(self.candidate,self.args,actor,critic,self.batch,
                                           enabled=True,signed_model=self.model,eta=.7)
        b,metrics=baseline(self.params,self.agent,self.batch)
        actual,detail=jax.jit(lambda p,b,r,i:signed(p,self.agent,b,r,i))(
            self.params,self.batch,self.bound,jnp.asarray(self.ids))
        self.assertTrue(bool(detail['binding_valid']))
        scale=jax.lax.stop_gradient(metrics['q_scale_coefficient'])
        def additive(p):
            a=jax.vmap(lambda s:actor(p,s).sample())(self.batch.obs)
            mu,sigma=self.model.apply(self.model_params,self.batch.obs,a)
            bp=jnp.maximum(0.,mu+4*sigma)
            d=revised_compat.compatibility_cost_jax(a,self.bound['neighbors'],self.bound['bandwidth'])
            return scale*(bp+.7*d).mean()
        np.testing.assert_allclose(actual-b,additive(self.params),rtol=1e-5,atol=1e-5)
        bg=jax.grad(lambda p:baseline(p,self.agent,self.batch)[0])(self.params)
        sg=jax.grad(lambda p:signed(p,self.agent,self.batch,self.bound,jnp.asarray(self.ids))[0])(self.params)
        np.testing.assert_allclose(sg-bg,jax.grad(additive)(self.params),rtol=2e-5,atol=2e-5)
        epsilon=.001; delta=jnp.zeros_like(self.params).at[0,1].set(epsilon)
        # Finite difference only the additive term with the native coefficient
        # explicitly fixed. Recomputing that coefficient across perturbations
        # would incorrectly differentiate through stop_gradient.
        fd=(additive(self.params+delta)-additive(self.params-delta))/(2*epsilon)
        np.testing.assert_allclose((sg-bg)[0,1],fd,rtol=.01,atol=.005)

    def test_reference_frozen_and_wrong_binding_fails_closed(self):
        loss=actor_loss.make_actor_loss(self.candidate,self.args,actor,critic,self.batch,
                                         enabled=True,signed_model=self.model,eta=.7)
        def wrt_neighbors(v):
            b=dict(self.bound);b['neighbors']=v
            return loss(self.params,self.agent,self.batch,b,jnp.asarray(self.ids))[0]
        np.testing.assert_array_equal(jax.grad(wrt_neighbors)(self.bound['neighbors']),
                                      jnp.zeros_like(self.bound['neighbors']))
        def wrt_params(v):
            b=dict(self.bound);b['signed_params']=v
            return loss(self.params,self.agent,self.batch,b,jnp.asarray(self.ids))[0]
        for x in jax.tree_util.tree_leaves(jax.grad(wrt_params)(self.model_params)):
            np.testing.assert_array_equal(x,jnp.zeros_like(x))
        value,metrics=loss(self.params,self.agent,self.batch,self.bound,jnp.asarray(self.ids[::-1].copy()))
        self.assertTrue(bool(jnp.isnan(value)));self.assertFalse(bool(metrics['binding_valid']))
        bad=self.batch._replace(obs=self.batch.obs.at[0,0].add(.001))
        value,metrics=loss(self.params,self.agent,bad,self.bound,jnp.asarray(self.ids))
        self.assertTrue(bool(jnp.isnan(value)));self.assertFalse(bool(metrics['binding_valid']))
        leak=dict(self.bound)
        leak['neighbor_episode_ids']=leak['neighbor_episode_ids'].at[0,0].set(int(self.ids[0]))
        value,metrics=loss(self.params,self.agent,self.batch,leak,jnp.asarray(self.ids))
        self.assertTrue(bool(jnp.isnan(value)));self.assertFalse(bool(metrics['binding_valid']))
        with self.assertRaises(ValueError):loss(self.params,self.agent,self.batch,None,None)

    def test_bound_neighbors_exclude_entire_source_episode(self):
        found=revised_compat.query_neighbors(self.reference,self.obs,self.times,exclude_episode_ids=self.ids)
        self.assertFalse(np.any(found['episode_ids']==self.ids[:,None]))
        np.testing.assert_array_equal(self.bound['neighbors'],found['actions'])


if __name__ == '__main__': unittest.main()
