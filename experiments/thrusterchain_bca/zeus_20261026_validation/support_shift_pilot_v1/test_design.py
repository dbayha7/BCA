"""Synthetic CPU checks only; no actual model, environment or fitting."""
import unittest
import numpy as np
import design


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.actions = np.zeros((3, 6, 4), dtype=np.float32)
        self.actions[0, :2, 0] = .8
        self.actions[1, :3, 0] = .8
        self.actions[2, :1, 0] = .8
        self.ids = np.asarray([12, 8, 20])

    def test_support_boundary_and_bounds(self):
        self.assertEqual(design.support_mask(np.array([[.5,0,0,0],[.50001,0,0,0]])).tolist(), [True, False])
        for invalid in ([[1.1,0,0,0]], [[np.nan,0,0,0]]):
            with self.assertRaises(ValueError): design.support_mask(invalid)

    def test_broad_count_matches_each_episode(self):
        rows = design.training_rows(self.actions, self.ids)
        for key in rows:
            np.testing.assert_array_equal(np.bincount(rows[key] // 6, minlength=3), [4,3,5])
            self.assertEqual(len(np.unique(rows[key])), 12)
        self.assertTrue(np.all(self.actions.reshape(-1,4)[rows["restricted"],0] <= .5))
        repeated = design.training_rows(self.actions, self.ids)
        np.testing.assert_array_equal(rows["broad_count_matched"], repeated["broad_count_matched"])

    def test_no_eligible_episode_refused(self):
        self.actions[0,:,0] = .8
        with self.assertRaises(ValueError): design.training_rows(self.actions, self.ids)

    def test_probe_has_one_eligible_row_per_source_episode(self):
        rows = design.training_rows(self.actions, self.ids)["restricted"]
        probes = design.one_per_episode(rows, self.ids, 6, 99)
        self.assertEqual(set(probes // 6), {0,1,2})
        self.assertTrue(set(probes).issubset(set(rows)))
        with self.assertRaises(ValueError): design.one_per_episode(rows[rows >= 6], self.ids, 6, 99)

    def test_intact_tuple_selection_and_source_immutability(self):
        bank = {key: np.arange(18) + offset for offset,key in enumerate(("obs","action","reward","next_obs","done"))}
        before = {key:value.copy() for key,value in bank.items()}
        rows = design.training_rows(self.actions, self.ids)["restricted"]
        selected = design.select_intact_transitions(bank, rows)
        for key in bank:
            np.testing.assert_array_equal(selected[key], before[key][rows])
            np.testing.assert_array_equal(bank[key], before[key])
        with self.assertRaises(ValueError): design.select_intact_transitions({"action": bank["action"]}, rows)

    def test_distance_gradient_matches_finite_difference(self):
        actions = np.array([[.2,-.3,.1,-.4]], dtype=np.float64)
        anchor = np.array([[-.1,.1,.2,.1]], dtype=np.float64)
        calculated = design.distance_action_gradient(actions, anchor)
        epsilon = 1e-6
        for j in range(4):
            plus,minus = actions.copy(),actions.copy()
            plus[0,j] += epsilon; minus[0,j] -= epsilon
            fd = design.FACTOR * (np.mean((plus-anchor)**2)-np.mean((minus-anchor)**2))/(2*epsilon)
            self.assertAlmostEqual(calculated[0,j], fd, places=8)
        np.testing.assert_array_equal(design.distance_action_gradient(anchor, anchor), np.zeros_like(anchor))

    def test_ratio_matches_rms_not_scalar_offsets(self):
        bca = np.array([[1.,2.,3.,4.],[4.,3.,2.,1.]])
        distance = bca * 2
        receipt = design.match_coefficient(bca, distance)
        self.assertEqual(receipt["distance_coefficient"], 1.5)
        self.assertAlmostEqual(receipt["distance_coefficient"]*receipt["distance_gradient_rms"], receipt["bca_gradient_rms"])
        self.assertEqual(design.match_coefficient(2*bca, distance)["distance_coefficient"], 3.)
        # Derivatives of f(a) and f(a)+100 have identical matching inputs.
        derivative = lambda a,offset: ((a+1e-5)**2+offset-((a-1e-5)**2+offset))/(2e-5)
        np.testing.assert_allclose(derivative(bca,0), derivative(bca,100), atol=1e-8, rtol=0)

    def test_degenerate_and_nonfinite_matching_stops(self):
        for bca,distance in ((np.zeros((2,4)),np.ones((2,4))),
                             (np.ones((2,4)),np.zeros((2,4))),
                             (np.full((2,4),np.nan),np.ones((2,4)))):
            with self.assertRaises(ValueError): design.match_coefficient(bca,distance)

    def test_exact_budgets(self):
        counts = design.maximum_counts()
        self.assertEqual([counts[k] for k in ("critic_updates","actor_updates","auxiliary_updates","calibrations","evaluation_transitions")], [64000,32000,64000,16,51200])
        self.assertEqual(counts["untrained_reference_policies"], 4)
        self.assertEqual(counts["evaluation_episodes"], 512)
        self.assertEqual(counts["replay_rows_drawn_with_replacement"], 8192000)


if __name__ == "__main__": unittest.main()
