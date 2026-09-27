"""Known-answer tests for the no-IW first stage; no simulator or real data."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from runtime.config import ROOT, resolve, typed
from calibration import iql_scale as W
from algorithms import iql_bca as I

class NoIWTests(unittest.TestCase):
    def test_all_four_configs_disable_importance_only(self):
        for host in ('td3_bc','rebrac','cql','iql'):
            rows=[resolve(ROOT/f'configs/{host}.yaml',m,202609171,'/tmp/not-executed','hopper') for m in ('host','bca')]
            self.assertEqual(rows[1]['calibration_weighting'],'none')
            self.assertIn('noiw',rows[1]['run_id'])
            objects=typed(rows[1])
            if host=='iql':
                driver,module,options=objects
                args,variants,arms,_=driver.resolved_arguments(options,module)
                self.assertEqual(variants[0].iw.mode,'off')
                self.assertEqual([a.name for a in arms],['host','bca'])
                self.assertTrue(all(a.beta==args.beta and a.gain==1 for a in arms))
                self.assertEqual(rows[0]['options']['host_parameters'],rows[1]['options']['host_parameters'])
                self.assertEqual(args.posterior.mode,'full')
            else:
                runner,args,spec,protocol=objects
                config=spec if host=='cql' else spec.config()
                self.assertEqual(config.arm,'bca');self.assertEqual(config.iw.mode,'off')
                self.assertIsNotNone(config.posterior)
                self.assertEqual(rows[0]['native_args'],rows[1]['native_args'])
                self.assertEqual(len(protocol.refresh_steps if host=='cql' else protocol.refresh_events),198)
    def test_iql_uniform_importance_preserves_bayesian_masses(self):
        key=jax.random.PRNGKey(19); a=jnp.array([-9.,0.,1.,5.])
        with patch.object(W.IW,'capped_awr_log_weights',side_effect=AssertionError('No AWR query allowed')):
            first=W.scale_weighting(a,key,W.ScaleIWConfig(mode='off'))
            second=W.scale_weighting(a*100,key,W.ScaleIWConfig(mode='off'))
        np.testing.assert_array_equal(first.iw.mean_one_weights,np.ones(4))
        np.testing.assert_array_equal(first.prior,first.product.weights)
        np.testing.assert_array_equal(first.product.weights,second.product.weights)
        self.assertEqual(float(first.iw.ess_fraction),1.)
        self.assertTrue(bool(first.iw.ess_feasible))
        self.assertFalse(np.allclose(first.prior,np.ones(4)/4))
    def test_iql_iw_design_stays_distinct(self):
        unweighted,arms=I.default_design(3.,fitting_mode='off')
        weighted,other=I.default_design(3.,fitting_mode='awr')
        self.assertEqual(arms,other)
        self.assertEqual(unweighted[0].iw.mode,'off')
        self.assertEqual(weighted[0].iw.mode,'awr')
        with self.assertRaises(ValueError): I.default_design(3.,fitting_mode='best_on_test')
    def test_noiw_invalid_input_is_not_silently_accepted(self):
        with self.assertRaises(ValueError): W.ScaleIWConfig(mode='invented')
        result=W.scale_weighting(jnp.array([0.,jnp.nan]),jax.random.PRNGKey(1),W.ScaleIWConfig(mode='off'))
        self.assertFalse(bool(result.raw.inputs_valid))

if __name__=='__main__': unittest.main()
