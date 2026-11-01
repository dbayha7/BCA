"""Pure bank tests: no simulator, fitted model or return labels."""
import unittest
import numpy as np
import common_bank as bank


class CommonBankTests(unittest.TestCase):
    def test_exact_midpoint_and_complete_historical_actions(self):
        n = np.asarray([[-.8, .6, -.2, .4], [0, 0, 0, 0]], np.float32)
        b = np.asarray([[-.6, .8, .2, 0.], [.4, -.4, .4, -.4]], np.float32)
        x = bank.candidate_bank(n, b)
        self.assertEqual(x.shape, (2, 15, 4))
        np.testing.assert_array_equal(x[:, 0], n)
        np.testing.assert_array_equal(x[:, 1], b)
        center = (n+b)*np.float32(.5)
        for dim in range(4):
            for j, offset in enumerate((-.2, .2)):
                expected = center.copy()
                expected[:, dim] = np.clip(center[:, dim]+np.float32(offset),-1,1)
                np.testing.assert_array_equal(x[:, 2+2*dim+j],expected)
        expected_fixed=np.zeros((2,5,4),np.float32)
        expected_fixed[:,:,0]=np.asarray(bank.FIXED_A0,np.float32)
        np.testing.assert_array_equal(x[:,10:],expected_fixed)

    def test_both_scoring_paths_use_identical_immutable_candidates(self):
        n=np.asarray([[-.4,.2,.1,.0],[.1,.2,.3,.4]],np.float32)
        b=np.asarray([[.2,.1,.4,.0],[-.1,-.2,-.3,-.4]],np.float32)
        x=bank.candidate_bank(n,b)
        seen=[]
        def frozen_critic(actions, weight):
            seen.append(actions.copy())
            return actions @ weight
        first=frozen_critic(x,np.array([1,2,3,4],np.float32))
        second=frozen_critic(x,np.array([4,3,2,1],np.float32))
        np.testing.assert_array_equal(seen[0],seen[1])
        self.assertEqual(first.shape,second.shape)
        self.assertFalse(x.flags.writeable)
        with self.assertRaises(ValueError):
            x[0,0,0]=9
        with self.assertRaises(ValueError):
            x.flags.writeable=True

    def test_swap_only_exchanges_actor_slots_and_duplicates_are_preserved(self):
        n=np.asarray([[1.,1.,1.,1.],[0.,0.,0.,0.],[-.8,.6,-.2,.4]],np.float32)
        b=np.asarray([[1.,1.,1.,1.],[0.,0.,0.,0.],[-.6,.8,.2,0.]],np.float32)
        x=bank.candidate_bank(n,b)
        swapped=bank.candidate_bank(b,n)
        np.testing.assert_array_equal(x[:,0],swapped[:,1])
        np.testing.assert_array_equal(x[:,1],swapped[:,0])
        np.testing.assert_array_equal(x[:,2:],swapped[:,2:])
        self.assertEqual(x.shape[1],15)
        # At the all-ones boundary the actor slots and four +.2 slots coincide.
        self.assertEqual(bank.duplicate_slot_counts(x)[0],5)
        # At zero, both actor slots and the fixed zero action coincide.
        self.assertEqual(bank.duplicate_slot_counts(x)[1],2)

    def test_nonfinite_dtype_alignment_and_bounds_fail(self):
        valid=np.zeros((2,4),np.float32)
        for n,b in [(valid.astype(np.float64),valid), (valid,valid[:1]),
                    (np.full((2,4),np.nan,np.float32),valid),
                    (np.full((2,4),1.01,np.float32),valid),
                    (np.empty((0,4),np.float32),np.empty((0,4),np.float32))]:
            with self.assertRaises(ValueError): bank.candidate_bank(n,b)

    def test_no_values_or_scores_enter_bank(self):
        import inspect
        self.assertEqual(list(inspect.signature(bank.candidate_bank).parameters),
                         ['native_actions','bca_actions'])


if __name__=='__main__': unittest.main()
