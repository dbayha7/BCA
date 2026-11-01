import unittest
from types import SimpleNamespace
import numpy as np

import diagnostic_runner as d


class DiagnosticResultTests(unittest.TestCase):
    def test_choice_and_error_change_on_same_reference(self):
        raw=np.tile([2.,1.,0.,-1.,-2.,-3.],(32,1))
        reference=np.tile([0.,3.,0.,0.,0.,0.],(32,1))
        adjusted=raw.copy(); adjusted[:,0]=0.
        obs=np.zeros((32,14),np.float32); obs[:,10]=.5
        acts=np.zeros((32,6,4),np.float32)
        out=d.summarize_bank(raw,reference,{'raw':raw,'adjusted':adjusted},obs,acts)
        self.assertEqual(out['high_load_anchor_families'],32)
        self.assertEqual(out['scores']['adjusted']['choice_change_vs_raw']['changed'],32)
        self.assertEqual(out['scores']['adjusted']['choice_change_vs_raw']['improved'],32)
        self.assertEqual(out['scores']['adjusted']['choice_change_vs_raw']['harmed'],0)
        self.assertEqual(out['scores']['adjusted']['error']['high_band_fixed']['rows'],96)
        self.assertEqual(out['scores']['adjusted']['decision']['mean_regret'],0.)

    def test_parity_requires_shape_and_bounded_difference(self):
        d.parity(np.array([10.000001]),np.array([10.]),'example')
        with self.assertRaises(ValueError): d.parity(np.array([10.1]),np.array([10.]),'example')
        with self.assertRaises(ValueError): d.parity(np.zeros((1,1)),np.zeros(1),'example')

    def test_remaining_time_conversion(self):
        obs=np.zeros((4,14),np.float32);obs[:,-1]=[1.,.75,.5,.25]
        np.testing.assert_array_equal(d.time_index(obs),[0,25,50,75])
        obs[0,-1]=.333
        with self.assertRaises(ValueError):d.time_index(obs)

    def test_screen_requires_two_seeds_per_group_and_both_controls(self):
        cases=[]
        for arm in d.ARMS:
            for seed in d.SEEDS:
                def context():
                    good=seed<2
                    return {'scores':{'combined':{'choice_change_vs_raw':{'changed':int(good),'improved':int(good),'harmed':0},
                                'decision':{'mean_regret':.1}},
                            'compatibility_only':{'decision':{'mean_regret':.2}},
                            'strength_matched_q1':{'decision':{'mean_regret':.3}}}}
                cases.append({'seed':seed,'arm':arm,'contexts':{x:context() for x in d.CONTEXTS}})
        self.assertTrue(d.mechanism_screen(cases)['passed'])
        for case in cases:
            if case['arm']=='native' and case['seed']==1:
                for v in case['contexts'].values(): v['scores']['strength_matched_q1']['decision']['mean_regret']=.1
        self.assertFalse(d.mechanism_screen(cases)['passed'])

    def test_calibration_scores_reuse_refresh_noise_key(self):
        import jax
        import jax.numpy as jnp
        seen=[]
        def q1_and_target(args,models,native,heldout,key):
            seen.append(np.asarray(key))
            return jnp.arange(64,dtype=jnp.float32),jnp.arange(64,dtype=jnp.float32)+2
        model=SimpleNamespace(apply=lambda params,obs,act:jnp.full((64,),2.,jnp.float32))
        aux=SimpleNamespace(native='frozen',calibrator=SimpleNamespace(params={}),residual_scale=jnp.array(3.,jnp.float32))
        held=SimpleNamespace(obs=jnp.zeros((64,14)),action=jnp.zeros((64,4)))
        key=jax.random.PRNGKey(91)
        rows=d.calibration_values(jax,SimpleNamespace(q1_and_target=q1_and_target),
            SimpleNamespace(positive_scale=lambda pred,unit:jnp.maximum(pred,1e-6)*unit),
            (None,None,model),None,aux,held,key)
        np.testing.assert_array_equal(seen[0],np.asarray(jax.random.fold_in(key,1213156420)))
        np.testing.assert_allclose(rows['normalized_absolute_score'],np.full(64,1/3))
        self.assertEqual(len(rows['online_q1']),64)
        critic=SimpleNamespace(apply=lambda params,obs,act:jnp.stack(
            [jnp.arange(64,dtype=jnp.float32),jnp.arange(64,dtype=jnp.float32)-1],axis=1))
        aux.native=SimpleNamespace(critic=SimpleNamespace(params={}))
        twin=d.calibration_values(jax,SimpleNamespace(q1_and_target=q1_and_target),
            SimpleNamespace(positive_scale=lambda pred,unit:jnp.maximum(pred,1e-6)*unit),
            (None,critic,model),None,aux,held,key,center='twin_min')
        np.testing.assert_allclose(twin['normalized_absolute_score'],np.full(64,.5))
        np.testing.assert_array_equal(twin['target'],rows['target'])


if __name__=='__main__':unittest.main()
