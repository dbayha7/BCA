"""CPU algebraic fixtures only: no real models, data, checkpoint or simulator."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState
from common import HERE, module, read
import runtime
import train_worker

fixture_module = module("_support_synthetic_prior",HERE.parent / "matched_policy_pilot_v1/test_continuation.py")


def fixture(*, invalid_refresh=False, reject_fit=False):
    f = fixture_module
    context, old, events = f.continuation_fixture(reject_fit=reject_fit, invalid_refresh=invalid_refresh)
    def fresh(ts):
        return TrainState.create(apply_fn=ts.apply_fn,params=ts.params,tx=ts.tx)
    a = old.microstep.agent_state
    agent = a._replace(actor=fresh(a.actor),dual_q=fresh(a.dual_q),actor_target=fresh(a.actor_target),dual_q_target=fresh(a.dual_q_target))
    cal = fresh(old.aux.calibrator)
    inf = jnp.float32(jnp.inf)
    ref = old.aux.posterior._replace(cal_params=cal.params,threshold=inf,lambda_hat=inf,lambda_hpd=inf,
                                    n_eff=jnp.float32(0),ready=jnp.bool_(False))
    aux = old.aux._replace(native=context.pipeline.native_view(agent),calibrator=cal,posterior=ref)
    native = f.CANDIDATE.make_microsteps(context.args,f.actor_apply,f.critic_apply,context.train_transition,kappa_penalty=0.)
    micro = native.initial_state(jax.random.PRNGKey(11),agent,reference=ref)
    z = jnp.int32(0)
    state = f.TRAINING.TrainingState(micro,aux,jax.random.PRNGKey(13),z,z,z,z,z,z,jnp.array([4],jnp.int32),jnp.int32(4))
    old_refresh = context.pipeline.refresh
    def refresh(*args,**kwargs):
        aux,flags,details = old_refresh(*args,**kwargs)
        return aux._replace(posterior=aux.posterior._replace(ready=jnp.bool_(True),n_eff=jnp.float32(128))),flags,details
    context.pipeline.refresh = refresh
    context.training,context.candidate = f.TRAINING,f.CANDIDATE
    context.models = (SimpleNamespace(apply=lambda p,o:f.actor_apply(p,o).sample(None)),)
    context.probe = context.train_transition
    adapter = runtime.make_adapter(context,state,"native",0.,agent.actor.params,[4],4)
    return adapter,state,events


def equal_tree(test,left,right):
    fixture_module.assert_tree_equal(test,left,right)


class RuntimeContracts(unittest.TestCase):
    def test_fresh_zero_and_unready_reference(self):
        adapter,state,_ = fixture()
        result = runtime.validate(adapter,state)
        self.assertEqual(result["counts"]["critic"],0)
        self.assertFalse(result["counts"]["reference_ready"])
        self.assertTrue(result["checks"]["warmup_reference_shape"])
        self.assertTrue(all(int(ts.step)==0 for ts in (state.microstep.agent_state.actor,state.microstep.agent_state.dual_q,state.aux.calibrator)))

    def test_native_kernel_parity_batch_reuse_and_reconstructed_indices(self):
        adapter,state,_ = fixture()
        original = state
        direct = state.microstep
        sampled = np.asarray(adapter.sample_indices(direct.rng))
        expected_batch = jax.tree_util.tree_map(lambda x:x[sampled],adapter.context.train_transition)
        for _ in range(4):
            # Both paths use the declared JIT mode; eager/JIT floating point
            # lowering differences are unrelated to scheduler parity.
            direct,_ = adapter.microstep_step(direct)
            state,metrics = runtime.step(adapter,state)
            equal_tree(self,direct.agent_state,state.microstep.agent_state)
            equal_tree(self,direct.rng,state.microstep.rng)
            if int(state.microstep.critic_calls) <= 2:
                equal_tree(self,state.microstep.batch,expected_batch)
        self.assertEqual(metrics["counts"]["critic"],4)
        self.assertEqual(metrics["counts"]["actor"],2)
        self.assertEqual(metrics["counts"]["auxiliary"],4)
        self.assertEqual(metrics["counts"]["refresh_accepted"],1)
        self.assertEqual(int(original.microstep.critic_calls),0)
        with self.assertRaises(Exception):
            runtime.step(adapter,state)

    def test_midpoint_reference_applies_to_next_actor(self):
        adapter,state,events = fixture()
        for _ in range(4):
            state,_ = runtime.step(adapter,state)
        anchor = state.microstep.agent_state.actor.params
        state = state._replace(refresh_at=jnp.array([4,6],jnp.int32),critic_budget=jnp.int32(8))
        bca = runtime.make_adapter(adapter.context,state,"bca",3.,anchor,[4,6],8)
        state,_ = runtime.step(bca,state)
        before = state
        native_direct,_ = bca.microstep_step(before.microstep)
        state,metrics = runtime.step(bca,state)
        equal_tree(self,native_direct.agent_state,state.microstep.agent_state)
        self.assertTrue(metrics["refresh_due"])
        self.assertEqual(events["refresh_counters"][-1],(3,6))
        equal_tree(self,state.microstep.reference,state.aux.posterior)

    def test_invalid_refresh_preserves_last_committed_state(self):
        adapter,state,_ = fixture(invalid_refresh=True)
        for _ in range(3):
            state,_ = runtime.step(adapter,state)
        before = runtime.tree_digest(state)
        with self.assertRaises(fixture_module.TRAINING.TrainingStepError) as caught:
            runtime.step(adapter,state)
        self.assertEqual(caught.exception.stage,"calibration_refresh")
        self.assertEqual(caught.exception.attempted_counts["refresh"],1)
        self.assertEqual(before,runtime.tree_digest(caught.exception.last_committed_state))

    def test_rejected_auxiliary_fit_no_native_only_commit(self):
        adapter,state,_ = fixture(reject_fit=True)
        with self.assertRaises(fixture_module.TRAINING.TrainingStepError) as caught:
            runtime.step(adapter,state)
        self.assertEqual(caught.exception.stage,"auxiliary_fit")
        self.assertEqual(int(caught.exception.last_committed_state.microstep.critic_calls),0)

    def test_distance_gradient_live_anchor_zero(self):
        adapter,state,_ = fixture()
        ctx=adapter.context; actor=state.microstep.agent_state.actor
        width = runtime.width_function(ctx,"distance",actor.params)
        obs=ctx.train_transition.obs[:8]; actions=actor.apply_fn(actor.params,obs).sample(None)
        np.testing.assert_array_equal(width(None,obs,actions),np.zeros(8))
        zero_grad=jax.grad(lambda a:width(None,obs,a).sum())(actions)
        np.testing.assert_array_equal(zero_grad,np.zeros((8,4)))
        probe=actions+.05
        actual=jax.grad(lambda a:width(None,obs,a).sum())(probe)
        expected=2*19.29619572569095/4*(probe-actions)
        np.testing.assert_allclose(actual,expected,rtol=1e-6)

    def test_live_bca_action_gradient_and_static_native_zero(self):
        adapter,state,_ = fixture()
        for _ in range(4):
            state,_=runtime.step(adapter,state)
        ctx=adapter.context; ref=state.aux.posterior
        width=runtime.width_function(ctx,"bca",state.microstep.agent_state.actor.params)
        actions=ctx.probe.action; obs=ctx.probe.obs
        grad=jax.grad(lambda a:width(ref,obs,a).sum())(actions)
        np.testing.assert_allclose(grad,ref.threshold*2*actions/4,rtol=1e-6)
        self.assertIsNone(runtime.width_function(ctx,"native",{}))

    def test_match_is_fixed_and_state_immutable(self):
        adapter,state,_=fixture()
        for _ in range(4):
            state,_=runtime.step(adapter,state)
        before=runtime.tree_digest(state)
        receipt,bca,distance=runtime.match(adapter,state)
        expected=np.sqrt(np.mean(np.sum((3*bca.astype(np.float64))**2,axis=1)))/np.sqrt(np.mean(np.sum(distance**2,axis=1)))
        self.assertAlmostEqual(receipt["distance_coefficient"],expected,places=12)
        self.assertEqual(before,runtime.tree_digest(state))
        self.assertEqual(receipt["optimizer_updates"],0)
        self.assertEqual(receipt["calibration_calls"],0)

    def test_full_state_roundtrip_preserves_optimizer_rng(self):
        adapter,state,_=fixture()
        template=state
        for _ in range(4):
            state,_=runtime.step(adapter,state)
        blob=serialization.to_bytes(state)
        restored=serialization.from_bytes(template,blob)
        equal_tree(self,state,restored)
        runtime.validate(adapter,restored)
        self.assertEqual(blob,serialization.to_bytes(restored))

    def test_worker_rejects_budget_and_cell_changes(self):
        p=read(HERE / "protocol.json")
        j={"kind":"warmup","seed_index":0,"condition":"restricted","cell":"seed0_restricted"}
        self.assertEqual(train_worker.validate_job(p,j),10000)
        bad=copy.deepcopy(p); bad["warmup_critic_updates"]=10002
        with self.assertRaises(ValueError):train_worker.validate_job(bad,j)
        with self.assertRaises(ValueError):train_worker.validate_job(p,{**j,"cell":"seed1_restricted"})


if __name__ == "__main__":
    unittest.main()
