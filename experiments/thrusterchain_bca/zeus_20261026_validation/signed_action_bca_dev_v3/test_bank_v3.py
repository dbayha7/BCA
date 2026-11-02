import unittest
from types import SimpleNamespace

import numpy as np
import bank_v3 as b


class BankTests(unittest.TestCase):
    def test_actions_preserve_coordinates_and_original_fixed_vectors(self):
        actor = np.asarray([[.9, -.9, .1, -.3], [1., 0., 0., 0.]], np.float32)
        near = np.asarray([[-.4, .5, -.6, .7], [.7, .6, .5, .4]], np.float32)
        bank = b.candidate_bank(actor, near)
        self.assertEqual(bank.shape, (2, 15, 4))
        np.testing.assert_array_equal(bank[:, 0], actor)
        np.testing.assert_array_equal(bank[:, 9], near)
        for i in range(4):
            for sign, slot in ((-1, 1+2*i), (1, 2+2*i)):
                other = [k for k in range(4) if k != i]
                np.testing.assert_array_equal(bank[:, slot, other], actor[:, other])
                np.testing.assert_allclose(bank[:, slot, i], np.clip(actor[:, i] + sign*.2, -1, 1))
        np.testing.assert_array_equal(bank[:, 10:, 1:], np.zeros((2,5,3), np.float32))
        np.testing.assert_array_equal(bank[0, 10:, 0], np.asarray(b.FIXED_A0, np.float32))
        np.testing.assert_array_equal(bank[1, 2], bank[1, 0])  # clipped duplicate kept

    def test_selection_excludes_natural_from_competing_and_breaks_ties_by_index(self):
        actor = np.tile(np.asarray([[.1, .2, .3, .4]], np.float32), (64, 1))
        near = np.tile(np.asarray([[.9, .9, .9, .9]], np.float32), (64, 1))
        a = np.tile(b.candidate_bank(actor, near)[None], (6,1,1,1))
        q = np.zeros((6,64,15), np.float64)
        q[:,:,0], q[:,:,1] = 10., 9.
        times = np.repeat(b.TIMES,16)
        s = b.select_indices(times,q,a)
        expected = np.concatenate([np.arange(i*16,i*16+8) for i in range(4)])
        np.testing.assert_array_equal(s['indices'],expected)
        self.assertEqual(s['strata'], (['natural']*4+['ambiguous_q1']*4)*4)
        q[:, :4, 1] = -1000.  # Natural selection cannot change from a different score.
        changed = b.select_indices(times,q,a)
        np.testing.assert_array_equal(changed['indices'],expected)

    def test_ambiguity_uses_distinct_actions_and_six_checkpoint_median(self):
        actor = np.tile(np.asarray([[1., 0., 0., 0.]], np.float32), (64,1))
        near = np.tile(np.asarray([[.9, .9, .9, .9]], np.float32), (64,1))
        a = np.tile(b.candidate_bank(actor, near)[None], (6,1,1,1))
        q = np.zeros((6,64,15), np.float64)
        q[:,:,0], q[:,:,1], q[:,:,2] = 10., 9., 1000. # slot 2 duplicates actor
        s = b.select_indices(np.repeat(b.TIMES,16),q,a)
        np.testing.assert_allclose(s['per_checkpoint_normalized_gap'],1/11)
        self.assertTrue(np.all(s['duplicate_action_slots'] >= 1))
        q[:3,5,1] = 9.8
        self.assertLess(s['median_normalized_gap'][5], .1)
        changed = b.select_indices(np.repeat(b.TIMES,16),q,a)
        self.assertLess(changed['median_normalized_gap'][5],s['median_normalized_gap'][5])
        with self.assertRaises(ValueError):
            b.select_indices(np.repeat(b.TIMES,16),q[:5],a)

    def test_pool_budget_and_time_contract_without_simulator(self):
        class Env:
            def __init__(self, seed):
                self.state=np.zeros(13,np.float64);self.state[0]=seed;self.t=0
            def close(self): pass
        class Eval:
            @staticmethod
            def _reset(core,ref,seed,law,ledger,deadline,progress):
                ledger['resets_completed']+=1
                return Env(seed)
        class Ref:
            @staticmethod
            def _step(env, action, ledger, stage, deadline, progress):
                assert action.shape == (4,) and action.dtype == np.float32 and stage == 'prefix'
                env.t+=1;env.state[1]+=float(action[0]);ledger['transition_steps_completed']+=1
            @staticmethod
            def observation(states,times):
                return np.column_stack([states,1-times/100]).astype(np.float32)
        ledger={'transition_steps_completed':0,'resets_completed':0}
        out=b.build_pool(None,Ref,Eval,context='ordinary',seeds=np.arange(64),prefix_seed=17,
            start_law={'current':0.,'wave':0.,'kick':0.},ledger=ledger)
        self.assertEqual(ledger, {'transition_steps_completed':2400, 'resets_completed':64})
        np.testing.assert_array_equal(out['times'],np.repeat(b.TIMES,16))
        np.testing.assert_array_equal(out['natural_indices'],np.array([16*j+i for j in range(4) for i in range(4)]))
        self.assertEqual(out['observations'].shape,(64,14))


if __name__ == '__main__':
    unittest.main()
