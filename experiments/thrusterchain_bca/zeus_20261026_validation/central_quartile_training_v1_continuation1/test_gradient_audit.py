"""Algebraic CPU tests using pinned actor-loss code and synthetic arrays only."""
import hashlib
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_DEFAULT_MATMUL_PRECISION']='highest'
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
import gradient_audit as audit
import test_runtime as fixtures
import runtime


def ready_fixture(arm='bca'):
    adapter,state,_=fixtures.fixture()
    # Only four tiny algebraic fixture updates make the reference ready.
    for _ in range(4): state,_=runtime.step(adapter,state)
    ctx=adapter.context
    ctx.args.batch_size=256
    raw=ctx.train_transition
    probe=jax.tree_util.tree_map(lambda x:jnp.tile(x,(48,)+(1,)*(x.ndim-1)),raw)
    ctx.probe=probe
    ids=np.arange(768,dtype=np.int64)
    ctx.data=SimpleNamespace(source_row_ids=ids,receipt={'gradient_probe_source_row_ids_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()})
    anchor=state.microstep.agent_state.actor.params
    adapter=runtime.make_adapter(ctx,state,arm,{'native':0.,'bca':3.,'distance':.2}[arm],anchor,[4],4)
    return adapter,state,ids,anchor


class GradientAuditTests(unittest.TestCase):
    def test_symmetric_tails_bc_can_pull_into_hole_without_critic(self):
        actions=np.array([[.4,0,0,0],[-.4,0,0,0]])
        # E logged a0=0 under symmetric retained tails; squared BC alone pulls
        # positive actor inward and negative actor inward, no critic required.
        bc=2*actions/4
        q=-bc  # deliberately opposite loss-gradient direction
        result=audit.inward_summary(actions,np.ones(2,bool),{'bc':bc,'q':q})
        self.assertEqual(result['bc']['inward_rows'],2)
        self.assertEqual(result['q']['outward_rows'],2)

    def test_zero_vector_cosines_are_undefined_not_false_alignment(self):
        r=audit.vector_summary([[1,0],[-1,0],[0,0]])
        self.assertEqual(r['cosine']['q/bc'],-1.)
        self.assertIsNone(r['cosine']['q/actual_penalty'])
        self.assertEqual(r['combined_l2_norm'],0.)

    def test_pinned_bca_loss_components_and_no_state_mutation(self):
        adapter,state,ids,anchor=ready_fixture()
        before=runtime.tree_digest(state);progress=[]
        result=audit.audit(adapter,state,ids,progress=progress.append)
        self.assertEqual(result['probe_rows'],768)
        self.assertEqual(result['ledger']['parameter_gradient_batches_completed'],3)
        self.assertEqual(result['ledger']['optimizer_updates'],0)
        self.assertEqual(before,runtime.tree_digest(state))
        self.assertEqual(result['complete_state_digest_before_after'],before)
        self.assertEqual(result['ledger']['q_action_gradient_rows_completed'],768)
        grads=result['per_row']['action_loss_gradients']
        np.testing.assert_array_equal(grads['prospective_bca3'],grads['actual_penalty'])
        self.assertGreater(result['mean_parameter_gradient']['l2_norm']['actual_penalty'],0)
        self.assertEqual(progress[-1],result['ledger'])
        batch=jax.tree_util.tree_map(lambda x:x[:256],adapter.context.probe)
        total=jax.grad(lambda params:adapter.functions.actor_loss(params,state.microstep.agent_state,
                        batch,state.microstep.reference)[0])(state.microstep.agent_state.actor.params)
        direct=np.concatenate([np.asarray(x,np.float64).ravel() for x in jax.tree_util.tree_leaves(total)])
        self.assertAlmostEqual(result['mean_parameter_gradient']['combined_l2_norm'],np.linalg.norm(direct),places=5)

    def test_native_shadow_bca_is_not_applied_penalty(self):
        adapter,state,ids,anchor=ready_fixture('native')
        result=audit.audit(adapter,state,ids)
        self.assertEqual(result['mean_parameter_gradient']['l2_norm']['actual_penalty'],0)
        np.testing.assert_array_equal(result['per_row']['action_loss_gradients']['actual_penalty'],np.zeros((768,4)))
        self.assertTrue(np.any(np.asarray(result['per_row']['action_loss_gradients']['prospective_bca3'])!=0))

    def test_distance_anchor_zero_and_requires_explicit_anchor(self):
        adapter,state,ids,anchor=ready_fixture('distance')
        with self.assertRaisesRegex(ValueError,'frozen fork'):audit.audit(adapter,state,ids)
        result=audit.audit(adapter,state,ids,anchor_params=anchor)
        np.testing.assert_array_equal(result['per_row']['action_loss_gradients']['actual_penalty'],np.zeros((768,4)))
        self.assertEqual(result['ledger']['distance_anchor_action_rows_completed'],768)

    def test_bad_source_ids_fail_before_any_diagnostic_dispatch(self):
        adapter,state,ids,_=ready_fixture()
        progress=[]
        with self.assertRaisesRegex(ValueError,'digest differs'):
            audit.audit(adapter,state,ids[::-1],progress=progress.append)
        self.assertEqual(progress,[])

    def test_failure_preserves_prospective_dispatch_ledger(self):
        adapter,state,ids,_=ready_fixture()
        progress=[]
        with patch.object(audit,'_parameter_vectors',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError) as caught:audit.audit(adapter,state,ids,progress=progress.append)
        led=caught.exception.gradient_audit_ledger
        self.assertEqual(led['parameter_gradient_batches_attempted'],1)
        self.assertEqual(led['parameter_gradient_batches_completed'],0)
        self.assertEqual(progress[-1],led)


if __name__=='__main__':unittest.main()
