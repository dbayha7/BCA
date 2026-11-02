import unittest
from unittest.mock import patch
import numpy as np
import runner


class TestScreen(unittest.TestCase):
    @staticmethod
    def make_case(seed, arm, opportunities):
        values=np.zeros((32,6));values[:,1]=1.
        raw=np.zeros((32,6));raw[:,0]=1.
        changed=np.zeros((32,6));changed[:,1]=1.
        if not opportunities: values[:,1]=0.
        scores={'signed_bayes':raw,'revised_compatibility':raw,
                'strength_matched_q1':raw,'combined':changed}
        block={'per_probe':{'high_load':[True]*32,'qpi':values.tolist(),'q1':raw.tolist(),
                            'scores':{k:v.tolist() for k,v in scores.items()}},
               'scores':{'q1_aligned':{'error':{'high_band_fixed':{'mae':2.}}},
                         'combined':{'error':{'high_band_fixed':{'mae':1.}}}}}
        return {'seed':seed,'arm':arm,'contexts':{'ordinary':block,'strong_current':block}}

    def test_pass_requires_both_checkpoint_groups(self):
        cases=[self.make_case(s,a,True) for s in runner.old.SEEDS for a in runner.old.ARMS]
        self.assertTrue(runner.screen(cases)['passed'])
        for case in cases:
            if case['arm']=='bca': case['contexts']['ordinary']['scores']['combined']['error']['high_band_fixed']['mae']=3.
        # Both context labels share this synthetic block.
        self.assertFalse(runner.screen(cases)['passed'])

    def test_no_opportunities_is_inconclusive(self):
        cases=[self.make_case(s,a,False) for s in runner.old.SEEDS for a in runner.old.ARMS]
        result=runner.screen(cases)
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'],'inconclusive')
        self.assertFalse(result['contexts']['ordinary']['cells']['native/seed0']['opportunity'])

    def test_signed_functions_remain_checkpoint_specific_on_new_shape(self):
        import jax
        import jax.numpy as jnp
        def fake_predict(model,state,obs,action,threshold):
            val=jnp.sum(action,axis=1)+model+state+threshold
            return val,val,val
        with patch.object(runner.signed_bca,'predict',fake_predict):
            first=runner.signed_function(jax,1.,2.,3.)
            second=runner.signed_function(jax,10.,20.,30.)
            x=jnp.zeros((2,4)); obs=jnp.zeros((2,14))
            np.testing.assert_array_equal(first(obs,x)[0],[6.,6.])
            np.testing.assert_array_equal(second(obs,x)[0],[60.,60.])
            # New shape forces a new JAX trace after the second factory call.
            x=jnp.zeros((3,4)); obs=jnp.zeros((3,14))
            np.testing.assert_array_equal(first(obs,x)[0],[6.,6.,6.])
            np.testing.assert_array_equal(second(obs,x)[0],[60.,60.,60.])


if __name__=='__main__': unittest.main()
