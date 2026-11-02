"""Bounded CPU parity and native PCG64 continuation for CQL candidate."""
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from candidate_cql.adapter import CachedCQLAdapter,execution_identity
from host_integration.cql_adapter import CQLAdapter
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_protocol import OfflineBudget
from thrusterchain.training_runner import train_offline,train_transitions

class CandidateCQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=generate_behavior_dataset(behavior='mixed_random_strong',episodes=10,seed=744,provenance='smoke')
        cls.batch=train_transitions(cls.data)

    def state_close(self,a,b):
        la=jax.tree_util.tree_flatten_with_path(a)[0]; lb=jax.tree_util.tree_flatten_with_path(b)[0]
        self.assertEqual([jax.tree_util.keystr(p) for p,_ in la],[jax.tree_util.keystr(p) for p,_ in lb])
        for (p,x),(_,y) in zip(la,lb):
            x,y=np.asarray(x),np.asarray(y)
            self.assertEqual(x.shape,y.shape); self.assertEqual(x.dtype,y.dtype)
            if x.dtype.kind in 'biu': np.testing.assert_array_equal(x,y)
            else: np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5,err_msg=jax.tree_util.keystr(p))

    def test_native_fixed_terminal_batches(self):
        native,candidate=CQLAdapter(),CachedCQLAdapter()
        native.initialize(self.batch,seed=20); candidate.initialize(self.batch,seed=20)
        for row in ([0,99,102,199,2,2,1],[199,2,99,0,102,2,1]):
            batch=self.batch.select(np.array(row))
            nm,cm=native.update(batch),candidate.update(batch)
            self.state_close(native._state,candidate._state)
            np.testing.assert_array_equal(native._rng,candidate._rng)
            self.assertEqual(native.optimizer_step_counts,candidate.optimizer_step_counts)
            self.assertEqual(set(nm),set(cm))
            for key in nm: np.testing.assert_allclose(nm[key],cm[key],atol=1e-6,rtol=1e-5,err_msg=key)

    def test_real_pcg64_sampling_resume_and_identity(self):
        budget=OfflineBudget(4,8)
        native,nr=train_offline(self.data,CQLAdapter,budget=budget,seed=845)
        full,fr=train_offline(self.data,CachedCQLAdapter,budget=budget,seed=845)
        first,_=train_offline(self.data,CachedCQLAdapter,budget=OfflineBudget(2,8),seed=845)
        resumed,rr=train_offline(self.data,CachedCQLAdapter,budget=OfflineBudget(2,8),seed=845,resume_from=first)
        self.assertEqual(full.to_bytes(),resumed.to_bytes())
        self.assertEqual(dict(native.optimizer_step_counts),dict(full.optimizer_step_counts))
        self.assertEqual(native.metadata['runner_state']['sampler_rng_state'],full.metadata['runner_state']['sampler_rng_state'])
        self.assertEqual(full.metadata['adapter']['execution_identity'],execution_identity())
        self.assertEqual(fr['sampling']['bit_generator'],'PCG64')
        self.assertEqual(rr['resumed_from_sha256'],first.digest)
        a,b=CQLAdapter(),CachedCQLAdapter(); a.restore(native); b.restore(full)
        self.state_close(a._state,b._state); np.testing.assert_array_equal(a._rng,b._rng)
        # The existing evaluator restores via the native adapter id. Verify
        # numeric state and metadata round-trip, but do not infer that this
        # alone validates study receipt or manifest identity.
        evaluation_adapter=CQLAdapter(); evaluation_adapter.restore(full)
        self.assertEqual(evaluation_adapter.checkpoint().to_bytes(),full.to_bytes())
        self.state_close(b._state,evaluation_adapter._state)
        with patch('candidate_cql.adapter.source_identity',return_value={'variant':'different'}):
            with self.assertRaisesRegex(ValueError,'source or runtime identity'): CachedCQLAdapter().restore(first)
        with patch.dict('os.environ',{'XLA_FLAGS':'--test-simulated-runtime-drift'}):
            with self.assertRaisesRegex(ValueError,'source or runtime identity'): CachedCQLAdapter().restore(first)

    def test_nonfinite_update_rolls_back_state_and_rng(self):
        candidate=CachedCQLAdapter(); candidate.initialize(self.batch,seed=20)
        before=candidate.checkpoint().to_bytes()
        update=candidate._candidate_update
        def nonfinite(*args):
            state,metrics=update(*args)
            metrics=dict(metrics); metrics['qf_loss']=jnp.float32(float('nan'))
            return state,metrics
        candidate._candidate_update=nonfinite
        with self.assertRaisesRegex(FloatingPointError,'nonfinite'):
            candidate.update(self.batch.select(np.array([0,99,102,199])))
        self.assertEqual(before,candidate.checkpoint().to_bytes())

if __name__=='__main__': unittest.main()
