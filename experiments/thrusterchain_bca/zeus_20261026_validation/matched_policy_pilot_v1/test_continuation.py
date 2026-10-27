"""Synthetic CPU contracts only: no native models, data, fitting or simulation.

The pinned microstep functions receive tiny algebraic actor/critic functions and
synthetic optimizer states. No checkpoint loader or experimental worker runs.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE.parents[1] / "titan_20261026/payload"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CANDIDATE = load("_matched_fixture_candidate", PAYLOAD / "unifloral_td3bc_validation_v1/candidate.py")
TRAINING = load("_matched_fixture_training", PAYLOAD / "unifloral_checkpoint_v1/training.py")
CONTINUATION = load("_matched_fixture_continuation", HERE / "continuation.py")


class Transition(NamedTuple):
    obs: object
    action: object
    reward: object
    next_obs: object
    done: object


class Agent(NamedTuple):
    actor: object
    dual_q: object
    actor_target: object
    dual_q_target: object


class NativeView(NamedTuple):
    actor: object
    actor_target: object
    critic: object
    critic_target: object


class Reference(NamedTuple):
    cal_params: object
    residual_scale: object
    threshold: object
    lambda_hat: object
    lambda_hpd: object
    n_eff: object
    ready: object


class Auxiliary(NamedTuple):
    native: object
    calibrator: object
    residual_scale: object
    posterior: object


class DeterministicAction(NamedTuple):
    action: object

    def sample(self, seed=None):
        del seed
        return self.action


def actor_apply(params, observations):
    return DeterministicAction(jnp.tanh(observations @ params["w"] + params["b"]))


def critic_apply(params, observations, actions):
    x = jnp.concatenate((observations, actions), axis=-1)
    return x @ params["w"] + params["b"]


def train_state(apply_fn, params, count):
    state = TrainState.create(apply_fn=apply_fn, params=params,
                              tx=optax.sgd(.001, momentum=.9))
    # Nonzero synthetic momentum makes accidental optimizer reinitialization visible.
    gradient = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * .03, params)
    return state.apply_gradients(grads=gradient).replace(step=jnp.int32(count))


def fixture():
    obs = jnp.linspace(-.4, .6, 16*14, dtype=jnp.float32).reshape(16, 14)
    data = Transition(obs, jnp.full((16, 4), .1, jnp.float32),
                      jnp.full((16,), -.2, jnp.float32), obs+.01,
                      jnp.zeros((16,), jnp.float32))
    actor = train_state(actor_apply,
        {"w": jnp.full((14, 4), .02, jnp.float32),
         "b": jnp.array([.02, -.01, .03, -.04], jnp.float32)}, 10000)
    critic = train_state(critic_apply,
        {"w": jnp.full((18, 2), .04, jnp.float32),
         "b": jnp.array([1., 1.2], jnp.float32)}, 20000)
    agent = Agent(actor, critic, actor.replace(step=jnp.int32(10000)),
                  critic.replace(step=jnp.int32(10000)))
    args = SimpleNamespace(num_critic_updates_per_step=2, batch_size=8,
        td3_alpha=2.5, policy_noise=.2, noise_clip=.5, gamma=.98,
        polyak_step_size=.005)
    batch = jax.tree_util.tree_map(lambda x: x[:8], data)
    return args, data, agent, batch


def assert_tree_equal(test, left, right):
    a, sa = jax.tree_util.tree_flatten(left)
    b, sb = jax.tree_util.tree_flatten(right)
    test.assertEqual(sa, sb)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def continuation_fixture(*, reject_fit=False, invalid_refresh=False):
    args, data, agent, batch = fixture()
    config = SimpleNamespace(cal_lr=.001, cal_beta=20., width_penalty=.005)
    aux_train = train_state(lambda p, o, a: p["gain"]+jnp.sum(a*0,axis=-1),
                            {"gain": jnp.float32(1.)}, 20000)
    reference = Reference(aux_train.params, jnp.float32(1.), jnp.float32(1.),
        jnp.float32(1.), jnp.float32(1.), jnp.float32(128.), jnp.bool_(True))
    def native_view(native):
        return NativeView(native.actor,native.actor_target,native.dual_q,native.dual_q_target)
    def reference_valid(ref):
        return (ref.residual_scale>0) & (ref.n_eff>=0) & (ref.threshold>=0) & (ref.threshold==jnp.maximum(ref.lambda_hat,ref.lambda_hpd))
    def live_width(models,ref,obs,actions):
        del models,obs
        value=ref.threshold*ref.residual_scale*(1+jnp.mean(actions**2,axis=-1))
        return value,ref.ready & jnp.isfinite(ref.threshold),reference_valid(ref)
    events = {"fit_keys": [], "refresh_keys": [], "refresh_counters": []}
    def fit_step(aux, used_batch, key):
        del used_batch
        events["fit_keys"].append(np.asarray(key).copy())
        updated=aux.calibrator.apply_gradients(grads={"gain": jnp.float32(.01)})
        return aux._replace(calibrator=updated), {
            "scale_inputs_valid": jnp.bool_(True), "scale_fit_accepted": jnp.bool_(not reject_fit)}
    def refresh(aux_args,aux_config,models,aux,heldout,key,*,heldout_ids):
        del aux_args,aux_config,models,heldout,heldout_ids
        events["refresh_keys"].append(np.asarray(key).copy())
        events["refresh_counters"].append((int(aux.native.actor.step),int(aux.calibrator.step)))
        threshold=jnp.float32(jnp.inf if invalid_refresh else 2.)
        ref=aux.posterior._replace(cal_params=aux.calibrator.params,threshold=threshold,
                                   lambda_hat=threshold,lambda_hpd=threshold)
        return aux._replace(posterior=ref), {
            "posterior_inputs_valid": True,"posterior_certified": True}, {"certified": True}
    pipeline=SimpleNamespace(native_view=native_view,reference_valid=reference_valid,
                              live_width=live_width,refresh=refresh)
    context=SimpleNamespace(native=SimpleNamespace(jax=jax,jnp=jnp),args=args,
        aux_config=config,aux_args=SimpleNamespace(),models=(),pipeline=pipeline,
        train_transition=data,heldout_transition=batch,heldout_ids={"row_ids": np.arange(8)},
        fit_step=fit_step)
    functions=CANDIDATE.make_microsteps(args,actor_apply,critic_apply,data,kappa_penalty=0.)
    micro=functions.initial_state(jax.random.PRNGKey(11),agent)
    aux=Auxiliary(native_view(agent),aux_train,jnp.float32(1.),reference)
    original=TRAINING.TrainingState(micro,aux,jax.random.PRNGKey(13),
        jnp.int32(20000),jnp.int32(20000),jnp.int32(3),jnp.int32(3),jnp.int32(3),
        jnp.int32(20000),jnp.array([10000,15000,20000],jnp.int32),jnp.int32(20000))
    return context,original,events


def initialize(context, original, arm="bca", native_seed=2681, aux_seed=2683):
    return CONTINUATION.initialize_continuation(context,original,arm,native_seed,aux_seed,
        critic_updates=4,refresh_at=2,candidate_factory=CANDIDATE.make_microsteps,training_api=TRAINING)


class NativeCandidateContracts(unittest.TestCase):
    def test_static_zero_penalty_native_loss_and_gradient(self):
        args, data, agent, batch = fixture()
        def forbidden(*unused):
            raise AssertionError("Static kappa0 path called penalty")
        funcs = CANDIDATE.make_microsteps(args, actor_apply, critic_apply, data,
                                          width_fn=forbidden, kappa_penalty=0.)
        def expected(params):
            actions = jax.vmap(lambda o: actor_apply(params, o).sample(None))(batch.obs)
            q = jax.vmap(lambda o,a: critic_apply(agent.dual_q.params,o,a))(batch.obs,actions)[:,0]
            coefficient = jax.lax.stop_gradient(args.td3_alpha/(jnp.abs(q).mean()+1e-7))
            return -coefficient*q.mean()+jnp.square(actions-batch.action).mean()
        actual, metrics = funcs.actor_loss(agent.actor.params, agent, batch, None)
        np.testing.assert_allclose(actual, expected(agent.actor.params), atol=1e-7, rtol=1e-6)
        grad = jax.grad(lambda p: funcs.actor_loss(p,agent,batch,None)[0])(agent.actor.params)
        expected_grad = jax.grad(expected)(agent.actor.params)
        for a,b in zip(jax.tree_util.tree_leaves(grad),jax.tree_util.tree_leaves(expected_grad)):
            np.testing.assert_allclose(a,b,atol=2e-7,rtol=2e-6)
        self.assertEqual(float(metrics["actor_penalty_term"]),0.)

    def test_q1_denominator_detached_and_penalty_placement(self):
        args,data,agent,batch=fixture()
        def width(ref,obs,actions):
            del obs
            return ref * (1.+jnp.mean(actions**2,axis=-1))
        funcs=CANDIDATE.make_microsteps(args,actor_apply,critic_apply,data,width_fn=width,kappa_penalty=3.)
        ref=jnp.float32(.7)
        loss,metrics=funcs.actor_loss(agent.actor.params,agent,batch,ref)
        actions=actor_apply(agent.actor.params,batch.obs).action
        q=critic_apply(agent.dual_q.params,batch.obs,actions)[:,0]
        coefficient=jax.lax.stop_gradient(args.td3_alpha/(jnp.abs(q).mean()+1e-7))
        expected=-coefficient*q.mean()+jnp.square(actions-batch.action).mean()+coefficient*3*width(ref,batch.obs,actions).mean()
        np.testing.assert_allclose(loss,expected,rtol=1e-6,atol=1e-7)
        self.assertEqual(float(jax.grad(lambda r: funcs.actor_loss(agent.actor.params,agent,batch,r)[0])(ref)),0.)
        np.testing.assert_allclose(metrics["q_scale_coefficient"],coefficient,rtol=1e-6)

    def test_native_microstep_rng_batch_and_cadence(self):
        args,data,agent,_=fixture()
        funcs=CANDIDATE.make_microsteps(args,actor_apply,critic_apply,data,kappa_penalty=0.)
        original=funcs.initial_state(jax.random.PRNGKey(7),agent)
        first,_=jax.jit(funcs.step)(original)
        second,_=jax.jit(funcs.step)(first)
        self.assertEqual(int(first.critic_calls),20001)
        self.assertEqual(int(first.actor_calls),10000)
        self.assertEqual(int(first.phase),1)
        self.assertEqual(int(second.critic_calls),20002)
        self.assertEqual(int(second.actor_calls),10001)
        self.assertEqual(int(second.actor_target_updates),10001)
        self.assertEqual(int(second.critic_target_updates),10001)
        self.assertEqual(int(second.phase),0)
        assert_tree_equal(self,first.batch,second.batch)
        expected_rng,_=jax.random.split(original.rng)  # replay draw
        expected_rng,_=jax.random.split(expected_rng)  # first critic target draw
        expected_rng,_=jax.random.split(expected_rng)  # second critic target draw
        np.testing.assert_array_equal(second.rng,expected_rng)
        assert_tree_equal(self,original.agent_state,agent)


class ContinuationContracts(unittest.TestCase):
    def test_distance_definition_gradient_and_frozen_anchor(self):
        context,original,_=continuation_fixture()
        actor=original.microstep.agent_state.actor
        obs=context.train_transition.obs[:8]
        anchor=actor_apply(actor.params,obs).sample(None)
        fn=CONTINUATION.make_width_fn("distance",actor_apply,actor.params,context.pipeline,())
        width=fn(None,obs,anchor)
        np.testing.assert_array_equal(width,np.zeros(8,np.float32))
        gradient=jax.grad(lambda a:fn(None,obs,a).sum())(anchor)
        np.testing.assert_array_equal(gradient,np.zeros((8,4),np.float32))
        actions=anchor+jnp.float32(.2)
        factor=np.float32(CONTINUATION.DISTANCE_FACTOR)
        np.testing.assert_allclose(fn(None,obs,actions),factor*jnp.mean((actions-anchor)**2,axis=-1),atol=1e-7,rtol=1e-6)
        actual=jax.grad(lambda a:fn(None,obs,a).sum())(actions)
        np.testing.assert_allclose(actual,factor*.5*(actions-anchor),atol=1e-7,rtol=1e-6)
        def with_anchor(params):
            local=CONTINUATION.make_width_fn("distance",actor_apply,params,context.pipeline,())
            return local(None,obs,actions).sum()
        for grad in jax.tree_util.tree_leaves(jax.grad(with_anchor)(actor.params)):
            np.testing.assert_array_equal(grad,np.zeros(grad.shape,grad.dtype))
        funcs=CANDIDATE.make_microsteps(context.args,actor_apply,critic_apply,context.train_transition,
                                       width_fn=fn,kappa_penalty=.1)
        changed=jax.tree_util.tree_map(lambda x:x+jnp.float32(.02),actor.params)
        _,metrics=funcs.actor_loss(changed,original.microstep.agent_state,
                                  jax.tree_util.tree_map(lambda x:x[:8],context.train_transition),original.aux.posterior)
        np.testing.assert_allclose(metrics["actor_penalty_term"],
            metrics["q_scale_coefficient"]*.1*metrics["width_mean"],atol=1e-7,rtol=1e-6)

    def test_initial_fork_preserves_full_state_and_binds_reference(self):
        context,original,_=continuation_fixture()
        adapter,state=initialize(context,original)
        assert_tree_equal(self,state.microstep.agent_state,original.microstep.agent_state)
        assert_tree_equal(self,state.aux,original.aux)
        assert_tree_equal(self,state.microstep.batch,original.microstep.batch)
        assert_tree_equal(self,state.microstep.reference,original.aux.posterior)
        np.testing.assert_array_equal(state.microstep.rng,jax.random.fold_in(original.microstep.rng,2681))
        np.testing.assert_array_equal(state.aux_rng,jax.random.fold_in(original.aux_rng,2683))
        np.testing.assert_array_equal(state.refresh_at,[10000,15000,20000,20002])
        self.assertEqual(int(state.critic_budget),20004)
        self.assertEqual(int(state.next_refresh_index),3)
        self.assertIsNone(original.microstep.reference)
        self.assertEqual(int(original.critic_budget),20000)
        self.assertEqual(CONTINUATION.validate(adapter,state)["counts"]["critic"],20000)

    def test_paired_rng_batches_counts_and_refresh_order(self):
        runs={}
        for arm in ("native","bca","distance"):
            context,original,events=continuation_fixture()
            adapter,state=initialize(context,original,arm)
            original_leaves=[np.asarray(x).copy() for x in jax.tree_util.tree_leaves(original)]
            batches=[];keys=[];actor_widths=[]
            for index in range(4):
                if index%2:
                    actions=actor_apply(state.microstep.agent_state.actor.params,state.microstep.batch.obs).action
                    expected_width=np.asarray(state.aux.posterior.threshold*(1+jnp.mean(actions**2,axis=-1))).mean()
                state,metrics=CONTINUATION.step(adapter,state)
                batches.append(np.asarray(state.microstep.batch.obs).copy())
                keys.append(np.asarray(state.microstep.rng).copy())
                self.assertEqual(metrics["refresh_due"],index==1)
                if arm=="bca" and index%2:
                    np.testing.assert_allclose(metrics["native"]["width_mean"],expected_width,rtol=1e-6)
                    actor_widths.append(metrics["native"]["width_mean"])
            counts=TRAINING.counts(state)
            self.assertEqual(counts["critic"],20004)
            self.assertEqual(counts["actor"],10002)
            self.assertEqual(counts["actor_target"],10002)
            self.assertEqual(counts["critic_target"],10002)
            self.assertEqual(counts["auxiliary"],20004)
            self.assertEqual(counts["refresh_accepted"],4)
            self.assertEqual(counts["next_refresh_index"],4)
            self.assertEqual(counts["last_refresh_critic"],20002)
            self.assertEqual(events["refresh_counters"],[(10001,20002)])
            assert_tree_equal(self,state.microstep.reference,state.aux.posterior)
            for saved,leaf in zip(original_leaves,jax.tree_util.tree_leaves(original)):
                np.testing.assert_array_equal(saved,np.asarray(leaf))
            if arm=="bca":self.assertGreater(actor_widths[1],1.9*actor_widths[0])
            runs[arm]=(batches,keys,events)
        for arm in ("bca","distance"):
            for i in range(4):
                np.testing.assert_array_equal(runs[arm][0][i],runs["native"][0][i])
                np.testing.assert_array_equal(runs[arm][1][i],runs["native"][1][i])
                np.testing.assert_array_equal(runs[arm][2]["fit_keys"][i],runs["native"][2]["fit_keys"][i])
            np.testing.assert_array_equal(runs[arm][2]["refresh_keys"][0],runs["native"][2]["refresh_keys"][0])

    def test_stream_forks_differ_and_schedule_is_immutable(self):
        context,original,_=continuation_fixture()
        adapter,a=initialize(context,original,"native",2681,2683)
        _,b=initialize(context,original,"native",2682,2684)
        self.assertFalse(np.array_equal(a.microstep.rng,b.microstep.rng))
        self.assertFalse(np.array_equal(a.aux_rng,b.aux_rng))
        with self.assertRaises(ValueError):
            CONTINUATION.validate(adapter,a._replace(refresh_at=jnp.array([10000,15000,20000,20004],jnp.int32)))

    def test_invalid_initial_reference_refuses_all_arms(self):
        context,original,_=continuation_fixture()
        bad=original.aux.posterior._replace(threshold=jnp.float32(jnp.inf),lambda_hpd=jnp.float32(jnp.inf))
        original=original._replace(aux=original.aux._replace(posterior=bad))
        for arm in CONTINUATION.ARMS:
            with self.subTest(arm=arm),self.assertRaises(ValueError):initialize(context,original,arm)

    def test_failed_auxiliary_preserves_last_committed_state(self):
        context,original,_=continuation_fixture(reject_fit=True)
        adapter,state=initialize(context,original)
        with self.assertRaises(CONTINUATION.ContinuationStepError) as raised:
            CONTINUATION.step(adapter,state)
        error=raised.exception
        self.assertIs(error.last_committed_state,state)
        self.assertEqual(error.stage,"auxiliary_fit")
        self.assertEqual(error.attempted_counts,{"critic":1,"actor":0,"actor_target":0,"critic_target":0,"auxiliary":1,"refresh":0})
        self.assertEqual(int(state.microstep.critic_calls),20000)

    def test_invalid_refresh_refuses_atomic_commit_including_controls(self):
        for arm in CONTINUATION.ARMS:
            with self.subTest(arm=arm):
                context,original,_=continuation_fixture(invalid_refresh=True)
                adapter,state=initialize(context,original,arm)
                state,_=CONTINUATION.step(adapter,state)
                with self.assertRaises(CONTINUATION.ContinuationStepError) as raised:
                    CONTINUATION.step(adapter,state)
                error=raised.exception
                self.assertIs(error.last_committed_state,state)
                self.assertEqual(error.stage,"calibration_refresh")
                self.assertEqual(error.attempted_counts,{"critic":1,"actor":1,"actor_target":1,"critic_target":1,"auxiliary":1,"refresh":1})
                self.assertEqual(int(state.microstep.critic_calls),20001)
                self.assertEqual(int(state.aux.posterior.threshold),1)

    def test_final_budget_refuses_dispatch(self):
        context,original,events=continuation_fixture()
        adapter,state=initialize(context,original,"native")
        for _ in range(4):state,_=CONTINUATION.step(adapter,state)
        with self.assertRaises(CONTINUATION.ContinuationStepError) as raised:
            CONTINUATION.step(adapter,state)
        self.assertEqual(raised.exception.stage,"preflight")
        self.assertTrue(all(x==0 for x in raised.exception.attempted_counts.values()))
        self.assertIs(raised.exception.last_committed_state,state)
        self.assertEqual(len(events["fit_keys"]),4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
