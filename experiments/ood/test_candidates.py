"""No model queries/simulator steps: candidate rounding, aliases and scores."""
import unittest
import json,tempfile
from pathlib import Path
import numpy as np
from experiments.ood.candidates import action_bank,score_bank
from experiments.ood.collect import sha

class CandidatesTest(unittest.TestCase):
    def setUp(self):
        self.a=np.array([.4131605,-.5376547,.70012856],np.float32)
        self.bounds=dict(native_low=np.full(3,-1,np.float32),native_high=np.ones(3,np.float32))
    def bank(self):return action_bank(self.a,self.a,self.a,1600000000,self.bounds)
    def scores(self):return dict(width=np.ones(10),scale=np.ones(10),dose=np.ones(10),usable=np.ones(10,bool),
        radius=np.array(2.),bayesian_radius=np.array(2.),conformal_radius=np.array(1.),unit=np.array(1.))
    def test_ten_slots_and_fixed_direction(self):
        x=self.bank();self.assertEqual(x['sent'].shape,(10,3));self.assertEqual(x['sent'].dtype,np.float32)
        self.assertEqual(sha(x),sha(self.bank()))
        np.testing.assert_allclose(x['preclip_rms'][4:],[.05,.05,.15,.15,.30,.30],rtol=0,atol=1e-15)
        np.testing.assert_allclose(np.square(x['direction']).mean(),1.,rtol=0,atol=1e-15)
    def test_transform_not_identity_and_duplicates(self):
        x=self.bank();self.assertFalse(np.array_equal(x['sent'][0],x['applied'][0]))
        self.assertEqual(x['alias'][:3],[0,0,0])
        np.testing.assert_array_equal(x['sent'][0],self.a)
        lo=self.bounds['native_low'];hi=self.bounds['native_high']
        np.testing.assert_array_equal(x['applied'],np.clip(lo+(x['sent']+1)*.5*(hi-lo),lo,hi))
    def test_clipping_and_cast_preserved(self):
        x=action_bank(np.ones(3,np.float32),self.a,self.a,1600000000,self.bounds)
        self.assertGreater(x['clipping_fraction'][4:].sum(),0)
        self.assertTrue((abs(x['sent'])<=1).all());self.assertTrue((abs(x['applied'])<=1).all())
        self.assertTrue((x['cast_rounding_max_abs']>=0).all())
    def test_reject_invalid_dtype_bounds_seed(self):
        for a,seed,bounds in [(self.a.astype(np.float64),1,self.bounds),(self.a,True,self.bounds),
                             (self.a,1,dict(native_low=np.zeros(3,np.float32),native_high=np.ones(3,np.float32)))]:
            with self.assertRaises(ValueError):action_bank(a,self.a,self.a,seed,bounds)
    def test_alias_scores_share_one_random_value(self):
        x=self.bank();s=score_bank(x,self.scores(),np.ones(10),1.,1600000100)
        self.assertEqual(s['random'][0],s['random'][1]);self.assertEqual(s['random'][0],s['random'][2])
        self.assertFalse(s['support_distant'].any());self.assertIsNone(s['native_width'])
    def test_missing_threshold_stays_unavailable(self):
        self.assertIsNone(score_bank(self.bank(),self.scores(),np.ones(10),None,1600000100)['support_distant'])
    def test_bad_alias_score_or_missing_radius_fails(self):
        scores=self.scores();scores['width'][1]=2
        with self.assertRaises(ValueError):score_bank(self.bank(),scores,np.ones(10),None,1)
        scores=self.scores();del scores['conformal_radius']
        with self.assertRaises(ValueError):score_bank(self.bank(),scores,np.ones(10),None,1)
    def test_unusable_not_replaced(self):
        scores=self.scores();scores['usable'][9]=False
        with self.assertRaises(ValueError):score_bank(self.bank(),scores,np.ones(10),None,1)
    def test_failed_state_process_blocks_stage(self):
        from experiments.ood.candidate_stage import accepted_states
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ex=p/'actual_exit.json'
            ex.write_text(json.dumps(dict(actual_returncode=1,timeout=False,interruption_signal=None)))
            with self.assertRaises(ValueError):accepted_states(p,p/'states',ex)
    def test_unrelated_successful_process_blocks_stage(self):
        from experiments.ood.candidate_stage import accepted_states
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ex=p/'actual_exit.json'
            ex.write_text(json.dumps(dict(actual_returncode=0,timeout=False,interruption_signal=None)))
            (p/'dispatch.json').write_text(json.dumps(dict(command=['python','--phase','gate','--attempt',str(p/'states')])))
            with self.assertRaises(ValueError):accepted_states(p,p/'states',ex)
    def test_missing_independent_acceptance_blocks_stage(self):
        from experiments.ood.candidate_stage import accepted_states
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ex=p/'actual_exit.json';states=p/'states';states.mkdir()
            ex.write_text(json.dumps(dict(actual_returncode=0,timeout=False,interruption_signal=None)))
            (p/'dispatch.json').write_text(json.dumps(dict(command=['python','--phase','states','--attempt',str(states)])))
            with self.assertRaises(FileNotFoundError):accepted_states(p,states,ex)

if __name__=='__main__':unittest.main()
