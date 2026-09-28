import unittest
import numpy as np
from candidate_design import thresholds, band, applied_unit, distances, stream_seed, proposal_pool, select_bands, attach_policy_slots
from contrasts import paired_contrasts, resource_envelope, sequential_mean


class Contrasts(unittest.TestCase):
    def test_known_answer(self):
        r=paired_contrasts(100.,100.,70.,90.)
        self.assertEqual(r['absolute_advantage'],20.)
        self.assertEqual(r['degradation_advantage'],20.)

    def test_lower_ceiling_trap(self):
        r=paired_contrasts(100.,80.,90.,75.)
        self.assertEqual(r['degradation_advantage'],5.)
        self.assertEqual(r['absolute_advantage'],-15.)
        self.assertEqual(r['anchor_advantage'],-20.)

    def test_improvement_not_clipped(self):
        r=paired_contrasts(10.,10.,20.,25.)
        self.assertEqual(r['host_degradation'],-10.)
        self.assertEqual(r['bca_degradation'],-15.)

    def test_equal(self):
        self.assertEqual(paired_contrasts(5.,5.,3.,3.)['degradation_advantage'],0.)

    def test_invalid(self):
        for v in (float('nan'),float('inf'),True,'1'):
            with self.assertRaises(ValueError): paired_contrasts(v,1.,2.,3.)

    def test_overflow(self):
        with self.assertRaises(ValueError): paired_contrasts(1e308,-1e308,1.,1.)

    def test_original_sequential_arithmetic(self):
        self.assertEqual(sequential_mean([1e16,1.,-1e16]),0.)
        with self.assertRaises(ValueError): sequential_mean([])

    def test_budget(self):
        r=resource_envelope()
        self.assertEqual(r['new_environment'],36_791_360)
        self.assertEqual(r['combined_environment'],38_089_713)
        self.assertEqual(r['combined_physics'],152_358_852)
        self.assertEqual(r['unallocated_environment'],1_908_687)
        with self.assertRaises(ValueError): resource_envelope(slots=15)


class Selection(unittest.TestCase):
    def setUp(self):
        self.anchor=np.zeros(3,np.float32)
        self.neighbors=np.zeros((32,3),np.float32)
        self.pool=proposal_pool(self.anchor,123)

    def test_threshold_known(self):
        np.testing.assert_allclose(thresholds(np.arange(101)/100),(.95,.99),rtol=0,atol=1e-15)

    def test_degenerate_refused(self):
        for v in ([0,0],[1,1],[-1,1],[0,np.nan]):
            with self.assertRaises(ValueError): thresholds(v)

    def test_boundaries(self):
        self.assertEqual([band(x,.2,.5) for x in (0,.2,.3,.5,.6)],['near','near','moderate','moderate','strong'])
        with self.assertRaises(ValueError): band(.1,.5,.2)

    def test_distance_known(self):
        np.testing.assert_array_equal(distances(np.ones((1,3)),self.neighbors),[1.])

    def test_float32_transform(self):
        a=np.asarray([-1.,.12345678,1.],np.float32)
        expected=np.clip(np.float32(-1)+(a+np.float32(1))*np.float32(.5)*np.float32(2),-1,1)
        np.testing.assert_array_equal(applied_unit(a),expected)
        with self.assertRaises(ValueError): applied_unit(a.astype(np.float64))

    def test_reproducible_pool(self):
        np.testing.assert_array_equal(self.pool['proposed'],proposal_pool(self.anchor,123)['proposed'])
        self.assertEqual(self.pool['proposed'].shape,(8192,3))
        self.assertTrue((abs(self.pool['applied'])<=1).all())

    def test_quota_and_generation_order(self):
        r=select_bands(self.anchor,self.pool,self.neighbors,.1,.5)
        self.assertTrue(r['complete'])
        self.assertEqual([len(x) for x in r['selected'].values()],[4,4,4])
        for name,rows in r['selected'].items():
            indices=[x['proposal_index'] for x in rows]
            self.assertEqual(indices,sorted(indices))
            self.assertTrue(all(band(x['distance'],.1,.5)==name for x in rows))

    def test_missing_no_adaptation(self):
        r=select_bands(self.anchor,self.pool,self.neighbors,1.1,1.2)
        self.assertFalse(r['complete'])
        self.assertEqual(r['missing'],dict(near=0,moderate=4,strong=4))
        self.assertEqual(r['pool_size'],8192)

    def test_corrupt_transform(self):
        self.pool['applied'][0,0]=.12345
        with self.assertRaises(ValueError): select_bands(self.anchor,self.pool,self.neighbors,.1,.5)

    def test_duplicate_policy_reused(self):
        selection=select_bands(self.anchor,self.pool,self.neighbors,.1,.5)
        rows=attach_policy_slots(selection,self.anchor,self.anchor)
        self.assertEqual(len(rows),14)
        self.assertEqual([r['alias_of'] for r in rows[-2:]],[0,0])

    def test_stream_names_and_no_internal_collisions(self):
        seeds=[stream_seed(h,e,s,p,i) for h in ('td3_bc','rebrac') for e in ('hopper','walker2d') for s in range(202609171,202609176)
               for p in ('reset','candidate','continuation','random_score') for i in range(64 if p=='reset' else 256)]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertEqual(stream_seed('td3_bc','hopper',202609171,'candidate',1),stream_seed('td3_bc','hopper',202609171,'candidate',1))
        with self.assertRaises(ValueError): stream_seed('td3_bc','hopper',0,'candidate',0)

    def test_no_outcome_interface(self):
        with self.assertRaises(TypeError): select_bands(self.anchor,self.pool,self.neighbors,.1,.5,returns=[100.])


if __name__=='__main__': unittest.main(verbosity=2)
