"""Independent numerical answers and adverse reporting fixtures; no simulator."""

import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.ood import analyze as A


class RankingTests(unittest.TestCase):
    def test_return_loss_and_strict_threshold(self):
        np.testing.assert_allclose(A.harm(-3., [-12.1, -3., -.1]), [9.1, 0, -2.9])
        self.assertEqual(A.harm(10, [9, 8.999]).tolist(), [1., 1.0009999999999994])
        self.assertEqual(A.auc([False, True], [1, 2]), 1.)

    def test_pair_enumeration_ties_reversal_and_infinity(self):
        labels = [0, 1, 1, 0]
        scores = [1., 1., 3., 2.]
        # Four harmful/nonharmful pairs: tie, loss, win, win.
        self.assertEqual(A.auc(labels, scores), .625)
        self.assertEqual(A.auc(labels, np.ones(4)), .5)
        self.assertEqual(A.auc(labels, -np.array(scores)), .375)
        self.assertEqual(A.auc([0, 1], [np.inf, np.inf]), .5)
        self.assertEqual(A.auc([0, 1], [0, np.inf]), 1.)

    def test_missing_class_is_unavailable(self):
        self.assertIsNone(A.auc([0, 0], [0, 1]))
        self.assertIsNone(A.auc([1, 1], [0, 1]))
        self.assertIsNone(A.auc([], []))
        with self.assertRaises(ValueError):
            A.auc([0, 2], [0, 1])
        with self.assertRaises(ValueError):
            A.auc([0, 1], [0, np.nan])

    def test_ap_uses_whole_tied_groups(self):
        self.assertEqual(A.average_precision([0, 1, 1, 0], [1, 1, 1, 1]), .5)
        self.assertEqual(A.average_precision([0, 1], [0, 1]), 1.)
        self.assertAlmostEqual(A.average_precision([0, 1], [1, 0]), .5)
        self.assertIsNone(A.average_precision([0, 0], [0, 1]))

    def test_aliases_reference_exclusion_and_conflicting_alias_refusal(self):
        p = dict(actions=[[0], [0], [-1], [-1], [1]], returns=[-3, -3, -12.1, -12.1, -.1],
                 width=[1, 1, 3, 3, 2], support=[0, 0, .1, .1, 1], random=[0, 0, 1, 1, 2])
        r = A.state_metrics(p)
        self.assertEqual(r['aliases'], [[0, 1], [2, 3], [4]])
        self.assertEqual(r['alternatives'], 2)
        self.assertEqual(r['harmful'], 1)
        self.assertEqual(r['width_auc'], 1.)
        self.assertEqual(r['support_auc'], 0.)
        p['returns'][3] = 0
        with self.assertRaises(ValueError):
            A.state_metrics(p)

    def test_pooled_can_disagree_with_within_state(self):
        self.assertEqual(np.mean([A.auc([0, 1], [0, 1]), A.auc([0, 1], [10, 11])]), 1.)
        self.assertEqual(A.auc([0, 1, 0, 1], [0, 1, 10, 11]), .75)

    def test_positive_radius_does_not_fix_order_or_ties(self):
        for radius in [0.01, 1, 500]:
            self.assertEqual(A.auc([0, 1], radius*np.array([3., 1.])), 0.)
            self.assertEqual(A.auc([0, 1], radius*np.array([2., 2.])), .5)

    def test_risk_retention_does_not_split_score_ties(self):
        r = A.risk_retention([0, 2, 4], [1, 1, 3])
        self.assertEqual([x['retained_count'] for x in r], [2, 3])
        self.assertEqual([x['mean_harm'] for x in r], [1., 2.])
        self.assertEqual([x['harmful_fraction'] for x in r], [.5, 2/3])
        self.assertEqual(len(A.risk_retention([0, 2], [1, 1])), 1)

    def test_loss_rank_constant_unavailable(self):
        self.assertEqual(A.loss_rank([1, 2, 3], [3, 2, 1]), -1.)
        self.assertIsNone(A.loss_rank([1, 2], [1, 1]))


class CoverageTests(unittest.TestCase):
    def test_independent_radius_matches_unchanged_production_on_saved_draws(self):
        import jax
        import jax.numpy as jnp
        from calibration.posterior import posterior_radius, PosteriorConfig
        scores=np.array([1,2,3,4],np.float32)
        weights=np.array([[100,1,1,1],[1,1,1,100],[1,1,100,1],[1,100,1,1]],np.float32)
        for alpha in [.1,.4]:
            ref=A.radius_reference(scores,weights,alpha=alpha,credibility=.75)
            actual=posterior_radius(jnp.asarray(scores),jax.random.PRNGKey(17),
                PosteriorConfig(alpha=alpha,credibility=.75,draws=4),bootstrap_weights=jnp.asarray(weights))
            self.assertEqual(float(actual.conformal_radius),ref['conformal'])
            self.assertEqual(float(actual.bayesian_radius),ref['bayesian'])
            self.assertEqual(float(actual.radius),ref['radius'])
            np.testing.assert_array_equal(actual.posterior_quantiles,ref['posterior_quantiles'])

    def test_finite_conformal_rank_boundaries_and_ties(self):
        self.assertEqual(A.conformal_radius(np.arange(1, 10), .1), 9.)
        self.assertTrue(math.isinf(A.conformal_radius(np.arange(1, 9), .1)))
        self.assertEqual(A.conformal_radius(np.ones(9), .1), 1.)
        self.assertEqual(A.conformal_radius([1, 2, 3, 4], .4), 3.)
        with self.assertRaises(ValueError):
            A.conformal_radius([-1, 2], .1)

    def test_tolerance_rank_against_independent_binomial_enumeration(self):
        self.assertTrue(math.isinf(A.tolerance_radius(np.arange(1, 29), .9, .95)))
        self.assertEqual(A.tolerance_radius(np.arange(1, 30), .9, .95), 29.)
        for n in [5, 10, 30]:
            for p in [.5, .9]:
                candidates = [k for k in range(1, n+1) if sum(
                    math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k)) >= .95]
                expected = candidates[0] if candidates else math.inf
                self.assertEqual(A.tolerance_radius(np.arange(1,n+1), p, .95), expected)

    def test_explicit_bayesian_draws_keep_both_radii(self):
        scores = [1., 2., 3., 4.]
        weights = [[100, 1, 1, 1], [1, 1, 1, 100], [1, 1, 100, 1], [1, 100, 1, 1]]
        r = A.radius_reference(scores, weights, alpha=.4, credibility=.75)
        np.testing.assert_equal(r['posterior_quantiles'], [1, 4, 3, 2])
        self.assertEqual(r['bayesian'], 3.)
        self.assertEqual(r['conformal'], 3.)
        self.assertEqual(r['radius'], 3.)
        self.assertTrue(math.isinf(A.radius_reference(scores, weights, alpha=.1)['radius']))

    def test_coverage_equality_infinity_and_exact_extreme_intervals(self):
        r = A.coverage([0, 2, 3, 100], [0, 2, 2, np.inf])
        self.assertEqual(r['failures'], 1)
        self.assertEqual(r['miscoverage'], .25)
        self.assertEqual(r['infinite_bounds'], 1)
        self.assertTrue(math.isinf(r['mean_width']))
        z = A.coverage(np.zeros(10), np.ones(10))
        self.assertEqual(z['miscoverage_interval'][0], 0.)
        self.assertAlmostEqual(z['miscoverage_interval'][1], 1-.025**.1)
        z = A.coverage(np.ones(10), np.zeros(10))
        self.assertEqual(z['miscoverage_interval'][1], 1.)
        self.assertAlmostEqual(z['miscoverage_interval'][0], .025**.1)
        with self.assertRaises(ValueError):
            A.coverage([np.nan], [1])


class ClusterTests(unittest.TestCase):
    def test_shared_reset_states_are_counted_once_and_missing_strata_are_explicit(self):
        from experiments.ood.report import fixture_records, STRATA
        full=[r for r in fixture_records() if r['seed']==0]
        result=A.ranking_summary(full,STRATA)
        self.assertEqual(len(full),48)
        self.assertEqual(result['unique_valid_states'],36)
        self.assertEqual(result['harmful_alternatives'],72)
        self.assertEqual(result['within_all_strata'],1.)
        self.assertTrue(result['broad_claim_allowed'])
        sparse=[r for r in fixture_records('sparse') if r['seed']==0]
        result=A.ranking_summary(sparse,STRATA)
        self.assertEqual(result['within_available_strata'],1.)
        self.assertIsNone(result['within_all_strata'])
        self.assertEqual(len(result['missing_strata']),3)
        self.assertFalse(result['broad_claim_allowed'])
        empty=[r for r in fixture_records('no_harm') if r['seed']==0]
        result=A.ranking_summary(empty,STRATA)
        self.assertEqual(result['unique_valid_states'],0)
        self.assertIsNone(result['pooled_auc'])
        with self.assertRaises(ValueError):
            A.ranking_summary(full+full[:1],STRATA)

    def test_sampler_keeps_seed_reset_pairs_and_crossed_blocks(self):
        banks = {0: [10, 11, 12], 1: [10, 11, 12]}
        draws = A.cluster_indices(banks, draws=12, seed=17, crossed=True)
        self.assertEqual(draws, A.cluster_indices(banks, draws=12, seed=17, crossed=True))
        for draw in draws:
            self.assertEqual(draw[0][1], draw[1][1])
            self.assertEqual(len(draw), 2)
            self.assertEqual(len(draw[0][1]), 3)
        with self.assertRaises(ValueError):
            A.cluster_indices(banks, draws=2, seed=17, crossed=False)
        with self.assertRaises(ValueError):
            A.cluster_indices({0:[10,11], 1:[11,12]}, draws=2, seed=17, crossed=True)

    def test_seed_pairing_strata_and_missing_draws(self):
        records=[]
        for seed in range(5):
            for block in range(2):
                for stratum in range(4):
                    # Pairwise difference is exactly .2 everywhere, with seed variation.
                    records.append(dict(seed=seed, block=seed*10+block, stratum=stratum,
                        width_auc=.4+.1*seed, support_auc=.2+.1*seed))
        r=A.paired_ranking_bootstrap(records, expected_strata=list(range(4)), draws=200, seed=17)
        self.assertAlmostEqual(r['width_minus_support']['estimate'], .2)
        np.testing.assert_allclose(r['width_minus_support']['interval'], [.2,.2], atol=1e-15)
        self.assertEqual(r['valid_draws'],200)
        self.assertGreaterEqual(r['width_minus_chance']['family_interval'][1],
                                r['width_minus_chance']['interval'][1])
        missing=[x for x in records if x['stratum']!=3]
        self.assertIsNone(A.paired_ranking_bootstrap(missing, expected_strata=list(range(4)),
                                                  draws=20, seed=17)['width_minus_chance']['interval'])
        self.assertFalse(A.paired_ranking_bootstrap(records[:8], expected_strata=list(range(4)),
                                                  draws=20, seed=17)['inference_available'])

    def test_skipped_actor_steps_are_not_observations(self):
        self.assertEqual(A.actor_update_mean([2,0,4,0], [True,False,True,False]), 3.)
        self.assertEqual(A.actor_update_mean([0,0,4,0], [True,False,True,False]), 2.)
        with self.assertRaises(ValueError):
            A.actor_update_mean([2,0], [True])

    def test_common_learning_curve_and_final_bank_are_separate(self):
        steps=np.arange(5000,1000001,5000)
        r=A.policy_summary(steps,np.ones((5,200))*2,np.ones((5,200))*3,
                           np.ones((5,20))*4,np.ones((5,20))*8)
        np.testing.assert_equal(r['curve_difference'], np.ones(5))
        np.testing.assert_equal(r['final_difference'], np.ones(5)*4)
        with self.assertRaises(ValueError):
            A.policy_summary(steps[:-1],np.ones((5,199)),np.ones((5,199)),np.ones((5,20)),np.ones((5,20)))


class FigureTests(unittest.TestCase):
    def test_numeric_export_and_missing_asset_refusal(self):
        from datetime import datetime, timezone
        import os
        import shutil
        from experiments.ood import report as R
        root=Path(os.environ.get('BCA_OOD_ANALYSIS_TEST_OUTPUT',str(Path(__file__).resolve().parents[2]
            /'runs/ood'/('analysis-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))))
        root.mkdir(parents=True,exist_ok=False)
        path=root/'reader'
        receipt=R.build(path)
        self.assertEqual(receipt['verified_figures'],10)
        self.assertEqual(receipt['simulator_steps'],0)
        R.verify_report(path)
        missing=root/'intentional-missing-asset'
        shutil.copytree(path,missing)
        (missing/'07-ranking.svg').unlink()
        with self.assertRaises(ValueError):
            R.verify_report(missing)
        tampered=root/'intentional-changed-table'
        shutil.copytree(path,tampered)
        table=tampered/'07-ranking.csv'
        table.write_text(table.read_text().replace('Width AUROC,0.0,1.0','Width AUROC,0.0,0.9'))
        with self.assertRaises(ValueError):
            R.verify_report(tampered)
        # Both adverse artifacts remain saved; the accepted reader is untouched.
        R.verify_report(path)
        with self.assertRaises(FileExistsError):
            R.build(path)


if __name__ == '__main__':
    unittest.main()
