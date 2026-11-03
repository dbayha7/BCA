import unittest
import numpy as np
from analyze import pair_arrays, events, discrimination, auc, corr, effective_kappa

class PairFixtures(unittest.TestCase):
    def test_native_is_zero_despite_inherited_config(self):
        self.assertEqual(effective_kappa({'arm':'native','kappa':63}),0);self.assertEqual(effective_kappa({'arm':'bca','kappa':189}),189)
        with self.assertRaises(ValueError):effective_kappa({'arm':'other','kappa':0})
    def fixture(self,k=2):
        return pair_arrays([[5,7,3,4]],[[4,5,2,4]],[[4,3,7,4]],[[1,3,0,1]],k)
    def test_algebra(self):
        a=self.fixture();np.testing.assert_array_equal(a['e'],a['m']-a['g']);np.testing.assert_array_equal(a['e'],a['min_error']+a['dh']);np.testing.assert_array_equal(a['adjusted_error'],a['e']-a['p'])
    def test_common_q_bias_cancels(self):
        x=self.fixture();y=pair_arrays([[105,107,103,104]],[[104,105,102,104]],[[4,3,7,4]],[[1,3,0,1]],2)
        for k in ['g','m','e','dh','min_margin','adjusted_margin']:np.testing.assert_array_equal(x[k],y[k])
    def test_harm_not_positive_absolute_error(self):
        a=pair_arrays([[-20,-15,-30,-20]],[[-20,-15,-30,-20]],[[4,3,7,4]],[[1,2,1,1]],0)
        ev=events(a,0);self.assertTrue(ev['raw_false_preference'][0,0]);self.assertLess(a['withheld_raw_error'][0,0],0)
    def test_correction_and_rescue(self):
        a=self.fixture();e=events(a,0);self.assertEqual(int(e['corrected_harmful'].sum()),1);self.assertEqual(int(e['near'].sum()),1);self.assertEqual(int(e['still_missed_useful'].sum()),1)
    def test_strict_gap_and_margin_ties(self):
        a=self.fixture();a['g']=np.array([[-.1,.1,0]]);a['m']=np.zeros((1,3));a['adjusted_margin']=np.zeros((1,3));e=events(a,.1);self.assertEqual(int(e['near'].sum()),3);self.assertEqual(int(events(a,0)['raw_missed_useful'].sum()),1)
    def test_invalid_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):pair_arrays([[1,2]],[[1,2]],[[1,2]],[[1,2]],1)
        with self.assertRaises(ValueError):pair_arrays([[1,2,3,4]],[[1,2,3,4]],[[1,2,np.nan,4]],[[1,2,3,4]],1)
    def test_auc_orientation_ties_oneclass(self):
        self.assertEqual(auc([3,2,1],[1,1,0]),1);self.assertEqual(auc([3,2,1],[0,0,1]),0);self.assertEqual(auc([1,1],[1,0]),.5);self.assertIsNone(auc([1,2],[1,1]))
    def test_grouping_and_exclusion(self):
        d=discrimination(np.array([[2,1,0],[2,1,0.]]),np.array([[-1,1,0],[-1,-1,-1.]]),0)
        self.assertEqual(d['eligible_mixed_banks'],1);self.assertEqual(d['eligible_pairs'],5);self.assertEqual(d['within_bank_auc'],1)
    def test_constant_correlation(self):self.assertIsNone(corr([1,1],[1,2]))
    def test_negative_width_contrast_can_promote_harm(self):
        a=pair_arrays([[0,-1,0,0]],[[0,-1,0,0]],[[0,-2,1,0]],[[2,0,2,2]],1);self.assertTrue(events(a,0)['new_harmful_preference'][0,0])
    def test_q1_min_head_contrast_need_not_positive(self):
        a=pair_arrays([[5,4,3,2]],[[0,4,3,2]],[[0,1,2,3]],[[1,1,1,1]],1);self.assertTrue((a['dh']<0).all())

if __name__=='__main__':unittest.main()
