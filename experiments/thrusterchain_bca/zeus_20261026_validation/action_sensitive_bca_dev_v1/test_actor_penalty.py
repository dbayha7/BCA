"""Tests of the actual pinned candidate actor-loss binding; no model training."""
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import importlib.util
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from actor_penalty import make_combined_width, make_actor_microsteps, make_combined_width_from_cost
import compatibility


class Batch(NamedTuple):
    obs: object
    action: object


class Distribution:
    def __init__(self, action): self.action = action
    def sample(self, seed=None): return self.action


class Pipeline:
    def live_width(self, models, reference, obs, actions):
        del models, obs
        values = reference['scale'] * jnp.sum(actions**2, axis=-1)
        return values, jnp.ones(values.shape, bool), jnp.ones(values.shape, bool)


def lookup(obs):
    # Deliberately dependent on obs so tests can detect forbidden reference flow.
    return jnp.broadcast_to(obs[:, None, :], (obs.shape[0], 2, 4))


def cost(actions, neighbors, bandwidth):
    return jnp.mean(((actions[:, None, :] - neighbors) / bandwidth)**2, axis=(1, 2))


def actor(params, obs): return Distribution(params * obs)


def critic(params, obs, action):
    del obs
    return jnp.stack((jnp.sum(params * action), .5 * jnp.sum(params * action)))


def candidate_module():
    path = Path(__file__).resolve().parents[2] / 'titan_20261026/payload/unifloral_td3bc_validation_v1/candidate.py'
    spec = importlib.util.spec_from_file_location('_diagnostic_native_candidate', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ActorBindingTests(unittest.TestCase):
    def setUp(self):
        self.obs = jnp.array([[.2,.3,.4,.5],[-.1,.2,-.3,.4]], dtype=jnp.float32)
        self.actions = jnp.array([[.1,-.2,.3,-.4],[.5,.4,.3,.2]], dtype=jnp.float32)
        self.reference = {'scale': jnp.float32(2)}
        self.h = jnp.array([1.,2.,1.,2.], dtype=jnp.float32)

    def test_combined_value_and_live_action_gradient(self):
        f = make_combined_width(Pipeline(), None, lookup, cost, self.h, kappa=3, eta=2)
        value = f(self.reference, self.obs, self.actions)
        expected = 2*jnp.sum(self.actions**2, axis=-1) + (2/3)*cost(self.actions, lookup(self.obs), self.h)
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-6)
        actual = jax.jit(jax.grad(lambda a: f(self.reference, self.obs, a).sum()))(self.actions)
        eps=1e-3; plus=np.array(self.actions); minus=np.array(self.actions); plus[0,1]+=eps; minus[0,1]-=eps
        finite=(float(f(self.reference,self.obs,jnp.array(plus)).sum())-float(f(self.reference,self.obs,jnp.array(minus)).sum()))/(2*eps)
        self.assertAlmostEqual(float(actual[0,1]),finite,delta=3e-4)
        frozen = jax.grad(lambda p: f({'scale':p},self.obs,self.actions).sum())(jnp.float32(2))
        self.assertEqual(float(frozen),0.0)

    def test_zero_eta_skips_neighbor_lookup_and_matches_width(self):
        def fail(_): raise AssertionError('zero eta should not query compatibility')
        f = make_combined_width(Pipeline(), None, fail, cost, self.h, eta=0)
        np.testing.assert_allclose(f(self.reference,self.obs,self.actions),
                                   2*jnp.sum(self.actions**2,axis=-1))

    def test_native_actor_loss_preserved_at_zero_width(self):
        native=candidate_module(); args=SimpleNamespace(num_critic_updates_per_step=2,batch_size=2,td3_alpha=2.5)
        data=Batch(self.obs,self.actions); agent=SimpleNamespace(dual_q=SimpleNamespace(params=jnp.ones(4)))
        original=native.make_microsteps(args,actor,critic,data,width_fn=None,kappa_penalty=0)
        combined=make_actor_microsteps(native,args,actor,critic,data,Pipeline(),None,lookup,cost,self.h,kappa=3,eta=0)
        params=jnp.array([.3,-.4,.5,-.6],dtype=jnp.float32); ref={'scale':jnp.float32(0)}
        old=original.actor_loss(params,agent,data,ref)[0];new=combined.actor_loss(params,agent,data,ref)[0]
        old_g=jax.grad(lambda p:original.actor_loss(p,agent,data,ref)[0])(params)
        new_g=jax.grad(lambda p:combined.actor_loss(p,agent,data,ref)[0])(params)
        np.testing.assert_array_equal(old,new);np.testing.assert_array_equal(old_g,new_g)

    def test_pinned_actor_receives_exact_additive_penalty_and_gradient(self):
        native=candidate_module();args=SimpleNamespace(num_critic_updates_per_step=2,batch_size=2,td3_alpha=2.5)
        data=Batch(self.obs,self.actions);agent=SimpleNamespace(dual_q=SimpleNamespace(params=jnp.ones(4)))
        baseline=native.make_microsteps(args,actor,critic,data,width_fn=None,kappa_penalty=0)
        combined=make_actor_microsteps(native,args,actor,critic,data,Pipeline(),None,lookup,cost,self.h,kappa=3,eta=2)
        params=jnp.array([.3,-.4,.5,-.6],dtype=jnp.float32)
        old,metrics=baseline.actor_loss(params,agent,data,self.reference)
        new=combined.actor_loss(params,agent,data,self.reference)[0]
        c=jax.lax.stop_gradient(metrics['q_scale_coefficient'])
        def additive(p):
            actions=p*self.obs
            return c*(3*jnp.mean(2*jnp.sum(actions**2,axis=-1)) + 2*jnp.mean(cost(actions,lookup(self.obs),self.h)))
        np.testing.assert_allclose(new-old,additive(params),rtol=1e-6,atol=1e-6)
        old_g=jax.grad(lambda p:baseline.actor_loss(p,agent,data,self.reference)[0])(params)
        new_g=jax.jit(jax.grad(lambda p:combined.actor_loss(p,agent,data,self.reference)[0]))(params)
        np.testing.assert_allclose(new_g-old_g,jax.grad(additive)(params),rtol=2e-6,atol=2e-6)

    def test_invalid_coefficients_refuse(self):
        for k,e in ((0,1),(3,-1),(float('nan'),1),(3,float('inf'))):
            with self.assertRaises(ValueError):
                make_combined_width(Pipeline(),None,lookup,cost,self.h,kappa=k,eta=e)

    def test_real_state_to_compatibility_cost_integrates_with_actor_width(self):
        rng=np.random.default_rng(17)
        observations=rng.normal(size=(800,14)).astype(np.float32)
        times=np.repeat(np.arange(0,100,10),80); observations[:,-1]=1.0-times/100.0
        actions=rng.uniform(-.5,.5,(800,4)).astype(np.float32)
        ref=compatibility.fit_reference(observations,actions,np.arange(800),times,k=64)
        jref=compatibility.jax_reference(ref)
        qobs=jnp.asarray(observations[:2]);qact=self.actions
        cost_fn=lambda s,a:compatibility.compatibility_from_observations_jax(jref,s,a,k=64)
        width=make_combined_width_from_cost(Pipeline(),None,cost_fn,kappa=3,eta=2)
        expected=2*jnp.sum(qact*qact,axis=-1)+(2/3)*cost_fn(qobs,qact)
        np.testing.assert_allclose(jax.jit(width)(self.reference,qobs,qact),expected,rtol=1e-6)
        grad=jax.jit(jax.grad(lambda a:width(self.reference,qobs,a).sum()))(qact)
        self.assertTrue(np.isfinite(np.asarray(grad)).all())
        self.assertGreater(float(jnp.linalg.norm(grad)),0.)


if __name__ == '__main__': unittest.main()
