"""Small deterministic contracts; no model loading, training or main experiment."""
import json
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibration_methods as CM

NATIVE = Path(os.environ.get('BCA_NATIVE_WBCP', str(Path(__file__).resolve().parents[2] /
    'results_20261021/release/ThrusterChain_results_20261021/source/calibration/wbcp.py')))


def independent_tail_threshold(scores, weights, query, seed, draws, alpha, beta):
    """Enumerate losses ABOVE each candidate, independently of cumulative crossing."""
    order = np.argsort(scores, kind='stable')
    scores, weights = np.asarray(scores)[order], np.asarray(weights)[order]
    empirical = next(s for s in np.unique(scores) if weights[scores>s].sum() <= alpha*weights.sum())
    E = np.random.default_rng(seed).standard_exponential((draws,len(scores)+1))
    crossings=[]
    for row in E:
        masses=row[:-1]*weights
        unseen=row[-1]*query
        feasible=[s for s in np.unique(scores) if masses[scores>s].sum()+unseen <= alpha*(masses.sum()+unseen)]
        crossings.append(min(feasible) if feasible else math.inf)
    posterior=sorted(crossings)[math.ceil(beta*draws)-1]
    return max(empirical,posterior), empirical, posterior


class CalibrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native=CM.load_native_wbcp(NATIVE)

    def calibrate(self,scores,weights,masses,**kwargs):
        return CM.calibrate_queries(scores,weights,masses,native=self.native,posterior_seed=[19,71],draws=128,**kwargs)

    def test_hand_computed_query_atom_crossings(self):
        E=np.array([[1,1,1,1],[1,1,1,4],[2,1,1,.5],[1,1,1,0]],float)
        actual=self.native.crossings(np.array([1.,3.,9.]),np.array([1.,2.,1.]),1.,E,.4)
        np.testing.assert_array_equal(actual,[3.,np.inf,3.,3.])

    def test_caller_owned_exponentials_and_manual_credible_threshold(self):
        # Three of four hand crossings equal 3, one is infinity; beta=.75 picks 3.
        E=np.array([[1,1,1,1],[1,1,1,4],[2,1,1,.5],[1,1,1,0]],float)
        result=CM.calibrate([1.,3.,9.],[1.,2.,1.],[1.,1.],E,native=self.native,alpha=.4,credibility=.75)
        np.testing.assert_array_equal(result.threshold,[3.,3.])
        self.assertEqual(result.lambda_hat[0],3.)
        self.assertEqual(result.lambda_hpd[0],3.)
        self.assertEqual(len(result.unique_results),1)
        self.assertEqual(result.unique_results[0]['finite_posterior_fraction'],.75)
        self.assertEqual(len(result.settings['exponential_draws_sha256']),64)

    def test_explicit_exponentials_match_seeded_native_with_chunking(self):
        scores=np.linspace(.1,2.,64)
        weights=np.tile([0.,.5,1.,2.],16)
        E=np.random.default_rng([19,71]).standard_exponential((128,65))
        actual=CM.calibrate(scores,weights,[.1,1.,10.],E,native=self.native)
        expected=self.calibrate(scores,weights,[.1,1.,10.])
        self.assertEqual(actual.unique_results,expected.unique_results)
        old=self.native._CHUNK_ELEMENTS
        try:
            self.native._CHUNK_ELEMENTS=65*7
            chunked=CM.calibrate(scores,weights,[.1,1.,10.],E,native=self.native)
        finally:
            self.native._CHUNK_ELEMENTS=old
        self.assertEqual(actual.unique_results,chunked.unique_results)

    def test_independent_tail_enumeration(self):
        scores=np.arange(1.,9.)
        weights=np.array([1.,2.,1.,2.,1.,2.,1.,2.])
        queries=np.array([.2,1.,4.])
        result=self.calibrate(scores,weights,queries,alpha=.4,beta=.5)
        for i,t in enumerate(queries):
            expected=independent_tail_threshold(scores,weights,t,[19,71],128,.4,.5)
            self.assertEqual((result.threshold[i],result.lambda_hat[i],result.lambda_hpd[i]),expected)

    def test_all_one_reduction_same_posterior(self):
        scores=np.linspace(.1,2.,128)
        pair=CM.calibrate_pair(scores,np.ones(128),np.ones(5),native=self.native,posterior_seed=[1,2],draws=1000)
        np.testing.assert_array_equal(pair['unweighted'].threshold,pair['exact_iw'].threshold)
        self.assertEqual(pair['unweighted'].unique_results,pair['exact_iw'].unique_results)
        native=self.native.calibrate(scores,np.random.default_rng([1,2]),draws=1000)
        self.assertEqual(pair['unweighted'].threshold[0],native.threshold)

    def test_common_scaling_is_invariant(self):
        scores=np.linspace(.1,4,64)
        weights=np.tile([0.,.5,1.,4.],16)
        first=self.calibrate(scores,weights,[.25,1.,4.])
        scaled=self.calibrate(scores,weights*8,np.array([.25,1.,4.])*8)
        np.testing.assert_array_equal(first.threshold,scaled.threshold)
        self.assertEqual(first.n_eff,scaled.n_eff)
        self.assertEqual([x['crossings_sha256'] for x in first.unique_results],
                         [x['crossings_sha256'] for x in scaled.unique_results])

    def test_exact_unique_queries_match_direct_native_calls(self):
        scores=np.linspace(.2,3,64)
        weights=np.tile([1.,2.],32)
        masses=np.array([1.,1.,np.nextafter(1.,2.),10.,1.])
        result=self.calibrate(scores,weights,masses)
        self.assertEqual(len(result.unique_results),3)  # near masses are NOT rounded together
        for i,t in enumerate(masses):
            exact=self.native.calibrate(scores,np.random.default_rng([19,71]),weights,float(t),draws=128)
            self.assertEqual(result.threshold[i],exact.threshold)
        self.assertEqual(result.bank_size,64)

    def test_zero_weights_retained_and_ess(self):
        scores=np.r_[np.ones(63),999.]
        weights=np.r_[np.ones(63),0.]
        result=self.calibrate(scores,weights,[.01])
        self.assertEqual(result.threshold[0],1.)
        self.assertEqual(result.bank_size,64)
        self.assertEqual(result.positive_weights,63)
        self.assertEqual(result.n_eff,63.)
        direct=self.native.calibrate(scores,np.random.default_rng([19,71]),weights,.01,draws=128)
        self.assertEqual(result.unique_results[0]['finite_posterior_fraction'],np.isfinite(direct.crossings).mean())

    def test_all_zero_bank_and_structural_refusal_are_distinct(self):
        result=self.calibrate(np.ones(64),np.zeros(64),[1.,np.inf])
        self.assertEqual(result.reason.tolist(),['no_positive_calibration_weight','unsupported_query'])
        self.assertEqual(result.n_eff,0.)
        self.assertEqual(len(result.unique_results),0)
        evaluation=CM.evaluate_queries(result,[2.,3.],[1.,2.],unit=1.)
        np.testing.assert_array_equal(evaluation.dose,[1.,1.])
        self.assertEqual(evaluation.summary['coverage_including_infinite'],1.)
        self.assertEqual(evaluation.summary['usable_coverage'],0.)
        self.assertIsNone(evaluation.summary['finite_conditional_coverage'])
        self.assertIsNone(evaluation.summary['mean_finite_half_width'])
        self.assertEqual(evaluation.summary['mean_half_width_including_infinite'],'inf')
        self.assertFalse(evaluation.summary['importance_law_valid_on_queries'])
        self.assertEqual(evaluation.summary['fallback_queries'],2)
        json.dumps(result.summary(),allow_nan=False)
        json.dumps(evaluation.summary,allow_nan=False)

    def test_finite_mass_posterior_refusal_is_not_structural(self):
        result=self.calibrate([1.],[1.],[1e6])
        self.assertEqual(result.reason[0],'posterior_infinite')
        self.assertEqual(result.positive_weights,1)
        self.assertEqual(result.lambda_hat[0],1.)

    def test_support_mask_applies_to_both_methods(self):
        pair=CM.calibrate_pair(np.ones(64),np.ones(64),[1.,1.],native=self.native,
            posterior_seed=7,query_supported=np.array([True,False]),draws=64)
        for result in pair.values():
            self.assertEqual(result.reason[1],'unsupported_query')
            self.assertTrue(np.isinf(result.threshold[1]))

    def test_weighting_can_move_threshold_in_both_directions(self):
        for scores,weights,expected in [
            (np.r_[np.ones(124),np.full(4,10.)],np.r_[np.ones(124),np.full(4,16.)],(1.,10.)),
            (np.r_[np.ones(64),np.full(64,10.)],np.r_[np.full(64,8.),np.full(64,.125)],(10.,1.))]:
            pair=CM.calibrate_pair(scores,weights,[1.],native=self.native,posterior_seed=19,draws=1000)
            self.assertEqual((pair['unweighted'].threshold[0],pair['exact_iw'].threshold[0]),expected)

    def test_dose_direction_bounds_and_paired_refusal_accounting(self):
        baseline=self.calibrate(np.ones(128),np.ones(128),[.01,.01,.01])
        weighted=self.calibrate(np.ones(128),np.ones(128),[.01,.01,np.inf])
        plain=CM.evaluate_queries(baseline,[.5,2.,4.],[0.1,1.,4.],unit=1.,absolute_error=[0.,1.,3.])
        iw=CM.evaluate_queries(weighted,[.5,2.,4.],[0.1,1.,4.],unit=1.)
        np.testing.assert_allclose(plain.dose,[1+.5*.1/1.1,1.25,1.4])
        self.assertTrue(np.all((plain.dose>=1)&(plain.dose<=1.5)))
        self.assertEqual(iw.dose[-1],1.)
        self.assertAlmostEqual(iw.summary['coverage_including_infinite'],2/3)
        self.assertAlmostEqual(iw.summary['usable_coverage'],1/3)
        self.assertEqual(iw.summary['finite_conditional_coverage'],.5)
        paired=CM.paired_metrics(plain,iw)
        self.assertAlmostEqual(paired['delta_coverage_including_infinite'],1/3)
        self.assertEqual(paired['delta_usable_coverage'],0.)
        self.assertEqual(paired['only_weighted_refused'],1)
        self.assertEqual(paired['delta_mean_half_width_jointly_accepted'],0.)
        json.dumps(paired,allow_nan=False)

    def test_invalid_inputs_fail_instead_of_flooring(self):
        for query in ([0.],[-1.],[np.nan]):
            with self.assertRaises(ValueError): self.calibrate([1.],[1.],query)
        for weights in ([-1.],[np.inf],[np.nan]):
            with self.assertRaises(ValueError): self.calibrate([1.],weights,[1.])
        with self.assertRaises(ValueError): self.calibrate([1.,2.],[1.],[1.])
        with self.assertRaises(ValueError): self.calibrate([-1.],[1.],[1.])
        with self.assertRaises(ValueError): self.calibrate([1.],[1.],[1.],query_supported=[1])
        with self.assertRaises(ValueError): self.calibrate([1.],[1e-308],[1e308])
        with self.assertRaises(ValueError):
            CM.calibrate_queries([1.],[1.],[1.],native=self.native,posterior_seed=None)
        with self.assertRaises(ValueError):
            CM.calibrate([1.],[1.],[1.],np.zeros((4,2)),native=self.native)

    def test_advisor_source_map_references_actual_file(self):
        evidence=CM.advisor_source_map(self.native)
        self.assertEqual(Path(evidence['native']['calibrate']['path']),NATIVE.resolve())
        self.assertEqual(len(evidence['stages']),8)
        self.assertEqual(len(evidence['native']['calibrate']['sha256']),64)


if __name__=='__main__':
    unittest.main()
