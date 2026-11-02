"""CPU algebraic fixtures only: no real models, data, checkpoint or simulator."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import time
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState
from common import HERE, module, read
import runtime

fixture_module = module("_central_training_synthetic_prior",HERE.parent / "matched_policy_pilot_v1/test_continuation.py")


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
    def initialization_fixture(self, protocol, seed, condition, mismatch=False):
        counts=dict(critic=0,actor=0,auxiliary=0,refresh_attempts=0,refresh_accepted=0,
                    reference_ready=False,phase=0)
        ctx=SimpleNamespace(data=SimpleNamespace(receipt={'data_plan_sha256':'plan',
            'decoded_roles':{'episodes':896,'transition_tuples':89600,
                'selection_or_final_calibration_transition_rows':0,'development_probe_transition_rows':0},
            'common_normalization':{'rows':60924,'digest':'same'}}),
            training=SimpleNamespace(counts=lambda state:counts),
            normalization={'rows':60924,'mean':1 if mismatch and condition=='hole' else 0},
            initial_optimizer_steps={'actor':0,'critic':0,'aux':0},
            initial_parameter_hashes={'actor':f'a{seed}','critic':f'q{seed}','auxiliary':f'g{seed}'},
            initial_optimizer_hashes={'actor':'oa','critic':'oq','auxiliary':'og'},
            runtime={'native_source':'pinned-source'})
        return SimpleNamespace(context=ctx),object()

    def test_real_source_initialization_plan_without_any_update_calls(self):
        events=[]
        with patch.object(runtime,'create',side_effect=self.initialization_fixture) as create:
            result=runtime.real_source_initialization_check({'data_plan_sha256':'plan'},
                deadline=time.monotonic()+30,progress=events.append)
        self.assertEqual(result['status'],'passed_real_source_initialization_check')
        self.assertEqual([(c.args[1],c.args[2]) for c in create.call_args_list],
                         [(0,'full'),(0,'hole'),(0,'random'),(1,'full'),(2,'full')])
        self.assertEqual(result['counts']['source_tuples_completed'],448000)
        self.assertEqual(result['counts']['optimizer_updates'],0)
        self.assertEqual(events[1]['counts']['initializations_attempted'],1)
        self.assertEqual(events[1]['counts']['initializations_completed'],0)

    def test_real_source_initialization_rejects_preprocessing_difference(self):
        with patch.object(runtime,'create',side_effect=lambda *a:self.initialization_fixture(*a,mismatch=True)):
            result=runtime.real_source_initialization_check({'data_plan_sha256':'plan'},deadline=time.monotonic()+30)
        self.assertEqual(result['status'],'failed_real_source_initialization_check')
        self.assertIn('normalization',result['error'])
        self.assertTrue(result['execution_counts_known'])

    def test_real_source_initialization_failure_preserves_decode_upper_bound(self):
        with patch.object(runtime,'create',side_effect=RuntimeError('fixture startup failed')):
            result=runtime.real_source_initialization_check({'data_plan_sha256':'plan'},deadline=time.monotonic()+30)
        self.assertFalse(result['execution_counts_known'])
        self.assertEqual(result['counts']['source_tuples_reserved'],89600)
        self.assertEqual(result['counts']['source_tuples_completed'],0)

    def test_synthetic_setup_failure_has_zero_counts_and_progress(self):
        events=[]
        with patch.object(runtime, 'load_native', side_effect=ModuleNotFoundError('fixture missing runtime')):
            result=runtime.synthetic_model_check({}, progress=events.append)
        self.assertEqual(result['status'],'failed_synthetic_runtime_check')
        self.assertIn('fixture missing runtime',result['error'])
        self.assertTrue(result['execution_counts_known'])
        self.assertTrue(all(v==0 for v in result['counts']['attempted'].values()))
        self.assertEqual(events[0]['stage'],'runtime_setup')
        self.assertEqual(events[-1]['counts'],result['counts'])

    def test_synthetic_interruption_preserves_last_dispatched_lower_bound(self):
        events=[]
        def interrupted(protocol,result,ledger,started,notify):
            result['stage']='fixture_inflight'
            ledger['attempted']['critic']=1
            notify()
            raise KeyboardInterrupt('fixture interruption')
        with patch.object(runtime,'_synthetic_model_check_body',side_effect=interrupted):
            result=runtime.synthetic_model_check({},progress=events.append)
        self.assertFalse(result['execution_counts_known'])
        self.assertEqual(result['counts']['attempted']['critic'],1)
        self.assertEqual(result['counts']['verified']['critic'],0)
        self.assertTrue(events[-1]['inflight'])

    def test_dispatch_progress_surrounds_native_aux_and_calibration(self):
        adapter,state,_=fixture()
        for _ in range(3): state,_=runtime.step(adapter,state)
        events=[]
        state,_=runtime.step(adapter,state,progress=events.append)
        self.assertEqual([(e['stage'],e['event']) for e in events],[(stage,event)
            for stage in ('native_microstep','auxiliary_fit','calibration_refresh')
            for event in ('before_dispatch','after_dispatch')])
        self.assertEqual(events[0]['attempted']['critic'],1)
        self.assertEqual(events[0]['completed']['critic'],0)
        self.assertEqual(events[-1]['attempted'],events[-1]['completed'])

    def test_host_metric_conversion_does_not_relax_state_dtype_check(self):
        native=jnp.float32(1.25)
        with self.assertRaisesRegex(ValueError,'dtypes differ'):
            runtime._compare_trees(jax,native,float(native))
        host_metric=np.asarray(float(native),dtype=np.asarray(native).dtype)
        self.assertEqual(runtime._compare_trees(jax,native,host_metric),0.)

    def test_live_gradient_check_actual_interfaces_on_toy_models(self):
        adapter,state,_=fixture()
        for _ in range(4): state,_=runtime.step(adapter,state)
        result=runtime._check_live_gradients(adapter,state)
        self.assertTrue(result['frozen_reference_gradient_zero'])
        self.assertTrue(result['state_unchanged'])

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

    def test_restored_next_step_equals_uninterrupted_after_midpoint(self):
        adapter,state,_=fixture()
        for _ in range(4): state,_=runtime.step(adapter,state)
        anchor=state.microstep.agent_state.actor.params
        state=state._replace(refresh_at=jnp.array([4,6],jnp.int32),critic_budget=jnp.int32(8))
        adapter=runtime.make_adapter(adapter.context,state,"bca",3.,anchor,[4,6],8)
        # Save at odd phase5 so stored minibatch, first critic loss, RNG and
        # optimizer momentum are all required for exact continuation.
        state,_=runtime.step(adapter,state)
        blob=serialization.to_bytes(state)
        restored=runtime.restore_exact(adapter,state,blob)
        left,left_metrics=runtime.step(adapter,state)
        right,right_metrics=runtime.step(adapter,restored)
        equal_tree(self,left,right)
        self.assertTrue(left_metrics['refresh_due'])
        self.assertEqual(left_metrics['counts'],right_metrics['counts'])

    def test_production_fork_seeds_schedule_and_complete_state(self):
        adapter,state,_=fixture()
        for _ in range(4): state,_=runtime.step(adapter,state)
        # Counters are enlarged only in this algebraic fixture; there are no
        # additional optimizer calls or real model/data objects.
        a=state.microstep.agent_state
        a=a._replace(actor=a.actor.replace(step=jnp.int32(5000)),
            dual_q=a.dual_q.replace(step=jnp.int32(10000)),
            actor_target=a.actor_target.replace(step=jnp.int32(5000)),
            dual_q_target=a.dual_q_target.replace(step=jnp.int32(5000)))
        micro=state.microstep._replace(agent_state=a,critic_calls=jnp.int32(10000),
            actor_calls=jnp.int32(5000),actor_target_updates=jnp.int32(5000),
            critic_target_updates=jnp.int32(5000),outer_steps=jnp.int32(5000))
        aux=state.aux._replace(native=adapter.context.pipeline.native_view(a),
            calibrator=state.aux.calibrator.replace(step=jnp.int32(10000)))
        state=state._replace(microstep=micro,aux=aux,aux_fit_attempts=jnp.int32(10000),
            aux_fit_accepted=jnp.int32(10000),last_refresh_critic=jnp.int32(10000),
            refresh_at=jnp.array([10000],jnp.int32),critic_budget=jnp.int32(10000))
        adapter=runtime.make_adapter(adapter.context,state,"native",0.,a.actor.params,[10000],10000)
        original=runtime.tree_digest(state)
        protocol={'seeds':[{'branch_native_seed':101,'branch_aux_seed':102}]*3}
        branches=[runtime.branch(adapter,state,protocol,2,arm,1.2) for arm in ('native','bca','distance')]
        for new_adapter,new_state in branches:
            self.assertEqual(new_adapter.schedule,[10000,15000])
            self.assertEqual(new_adapter.budget,20000)
            runtime.validate(new_adapter,new_state)
            equal_tree(self,new_state.microstep.agent_state,state.microstep.agent_state)
        for (_,left),(_,right) in zip(branches,branches[1:]):
            equal_tree(self,left,right)
        self.assertEqual(original,runtime.tree_digest(state))
        with self.assertRaises(ValueError): runtime.branch(adapter,state,protocol,3,'native',1.)


if __name__ == "__main__":
    unittest.main()
