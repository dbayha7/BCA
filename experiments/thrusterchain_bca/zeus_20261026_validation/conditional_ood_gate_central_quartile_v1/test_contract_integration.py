"""Contract-to-interface checks; no real datasets, models or dynamics."""
import json
from pathlib import Path
import unittest

import numpy as np

import analysis
import geometry
import launch
import reference


class ContractIntegrationTests(unittest.TestCase):
    def test_fixed_contract_matches_runtime_constants_and_budget(self):
        c = launch.decision_contract()
        self.assertEqual(c['region']['center'], geometry.CENTER)
        self.assertEqual(c['region']['half_width'], geometry.HALF_WIDTH)
        self.assertEqual(c['region']['high_load'], geometry.LOAD_CUTOFF)
        self.assertEqual(c['support']['max_original_neighbors'], geometry.ROW_CAP)
        self.assertEqual(c['support']['reference_anchors_per_stratum'], geometry.REFERENCE_PER_STRATUM)
        self.assertEqual(c['support']['reference_quantile'], geometry.REFERENCE_QUANTILE)
        self.assertEqual(c['caps']['cpu_distance_pairs'], geometry.MAX_DISTANCE_PAIRS)
        self.assertEqual(c['controls']['random_seed'], geometry.CONTROL_SEED)
        self.assertEqual(c['calibration']['replacement_seed'], geometry.CALIBRATION_REPLACEMENT_SEED)
        self.assertEqual(c['caps']['total_simulator_transitions'], 64*sum([0,25,50,75])/4 + 6*16*sum([100,75,50,25]) + 16*100 + 104)
        self.assertEqual(c['reference']['gamma'], reference.GAMMA)
        self.assertEqual(c['reference']['horizon'], reference.HORIZON)

    def test_geometry_to_analysis_keeps_missing_reference_a_failure(self):
        # This fixture is too small to fill all reference strata. Strong value
        # performance elsewhere cannot repair this geometry decision.
        obs = np.zeros((80,14), np.float32)
        obs[:,10] = .6
        actions = np.zeros((80,4), np.float32)
        actions[:,0] = np.resize([-.5,0.,.5],80)
        episodes = np.arange(80,dtype=np.int64)
        times = np.zeros(80,dtype=np.int64)
        masks, control = geometry.control_masks(obs, actions, times, episode_ids=episodes)
        metric = geometry.fit_support_metric(obs)
        support = geometry.support_report(obs,actions,episodes,times,obs[:1],np.array([999]),times[:1],masks,metric)
        cal = geometry.calibration_plan(obs,actions,episodes,horizon=1)
        report = analysis.analyze_geometry(obs,actions,episodes,times,masks,control,metric,support,cal,obs[:1],contract=launch.decision_contract())
        self.assertFalse(report['passed'])
        self.assertFalse(report['criteria']['all_training_reference_strata_available'])
        self.assertEqual(len(report['descriptive_conditional_action_means']['hole']['high']),4)
        json.dumps(report,allow_nan=False)

    def test_old_failed_decision_is_preserved(self):
        self.assertEqual(launch.verify_source_lineage()['prior_decision'], 'failed_development_design_gate')


if __name__ == '__main__':
    unittest.main()
