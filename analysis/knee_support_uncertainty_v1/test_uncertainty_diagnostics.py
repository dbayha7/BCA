import unittest
import numpy as np
import uncertainty_diagnostics as D


class DiagnosticsFixtures(unittest.TestCase):
    def test_rank_ties(self):
        np.testing.assert_array_equal(D.ranks([3,1,1,2]),[4,1.5,1.5,3])
    def test_auc_orientation_and_ties(self):
        self.assertEqual(D.auc([1,2,3,4],[0,0,1,1]),1)
        self.assertEqual(D.auc([4,3,2,1],[0,0,1,1]),0)
        self.assertEqual(D.auc([1,1,1,1],[0,0,1,1]),.5)
    def test_auc_one_class(self):
        self.assertIsNone(D.auc([1,2],[1,1]));self.assertIsNone(D.auc([1,2],[0,0]))
    def test_conditional_excludes_one_class(self):
        r=D.conditional_auc([[1,2],[3,4]],[[0,1],[1,1]])
        self.assertEqual(r['mean'],1);self.assertEqual(r['eligible_banks'],1);self.assertEqual(r['one_class_banks'],1)
    def test_constant_correlations(self):
        r=D.local_correlations([[2,2],[4,4]],[[1,2],[1,2]])
        self.assertIsNone(r['bank_centered_pearson']);self.assertIsNone(r['mean_bank_spearman']);self.assertEqual(r['constant_banks'],2)
    def test_local_removes_state_offset(self):
        r=D.local_correlations([[0,1],[100,101]],[[1,0],[101,100]])
        self.assertAlmostEqual(r['bank_centered_pearson'],-1);self.assertAlmostEqual(r['mean_bank_spearman'],-1)
    def test_signed_zero_definition(self):
        r=D.sign_summary([-2,0,3]);self.assertEqual((r['over'],r['under'],r['exact_zero']),(1,1,1));self.assertAlmostEqual(r['mae'],5/3)
    def test_bank_identity_fail_closed(self):
        D.validate_bank_pairing({('stall3',0,0):{'identity':['a']},('nostall',0,0):{'identity':['b']}})
        with self.assertRaises(ValueError):D.validate_bank_pairing({('stall3',0,0):{'identity':['a']},('stall3',21,0):{'identity':['b']}})
    def test_controls_same_penalty_range_no_value_fit(self):
        q=np.array([[0,2,3,4.]]);u=np.array([[1,2,3,4.]]);dist=np.array([[0,.1,.2,.3]]);w=dist>0;v=np.array([[1,4,2,3.]])
        r,c=D.choice_controls(q,u,dist,w,v,2)
        np.testing.assert_array_equal(c['bca'],c['distance_range_matched']);self.assertEqual(c['cutoff'][0],0)
        r2,c2=D.choice_controls(q,u,dist,w,-v,2)
        for key in c:np.testing.assert_array_equal(c[key],c2[key])
    def test_constant_width_range_control_is_raw(self):
        q=np.array([[0,2,3,4.]]);u=np.ones_like(q);dist=np.array([[0,.1,.2,.3]]);w=dist>0
        r,c=D.choice_controls(q,u,dist,w,q,189)
        np.testing.assert_array_equal(c['raw'],c['distance_range_matched']);np.testing.assert_array_equal(c['raw'],c['bca'])
    def test_first_slot_tie_and_harm(self):
        q=np.array([[4,4,3,2.]])
        r,c=D.choice_controls(q,np.zeros_like(q),np.array([[0,.1,.2,.3]]),np.array([[0,1,1,1]],bool),np.array([[1,0,2,3.]]),0)
        self.assertEqual(c['raw'][0],0);self.assertEqual(r['raw']['exact_score_tie_banks'],1)
    def test_no_supported_candidate_rejected(self):
        with self.assertRaises(ValueError):D.choice_controls(np.ones((1,4)),np.ones((1,4)),np.ones((1,4)),np.ones((1,4),bool),np.ones((1,4)),21)
    def test_exact_float32_cutoff(self):
        k=np.float32(.8);a=np.array([k,np.nextafter(k,np.float32(1))],np.float32).astype(np.float64)
        np.testing.assert_array_equal(abs(a)>D.KNEE,[False,True])


if __name__=='__main__':unittest.main()
