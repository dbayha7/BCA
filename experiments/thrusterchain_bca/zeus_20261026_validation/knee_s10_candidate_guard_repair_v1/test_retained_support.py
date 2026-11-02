"""Pure synthetic fixtures only. Never invokes collection, physics or a learner."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "knee_edge_ood_v3_guard_repair_v1"))
sys.path.insert(0, str(HERE))
import numpy as np
import g2_feasibility as G
import retained_support as R


def fixture(n=70):
    ids = np.tile(np.arange(n), 3)
    times = np.repeat([0, 25, 50], n)
    rng = np.random.default_rng(901)
    obs = rng.normal(size=(len(ids), 14))
    obs[:, 13] = (100-times)/100
    rows = {"obs": obs, "episode_ids": ids, "times": times}
    mask = np.ones(len(ids), bool)
    return rows, mask, mask.copy(), obs[:3].copy(), obs[[1, n+1, 2*n+1]].copy(), np.array([0,25,50])


class RetainedSupportTests(unittest.TestCase):
    def test_60_row_group_is_inconclusive_without_metric_call(self):
        args = list(fixture())
        args[1][-10:] = False
        with patch.object(R.SM4, "l15", side_effect=AssertionError("must not compute")):
            result = R.retained_l15(*args)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertFalse(result["ok"])
        self.assertIsNone(result["anchor_share"])
        self.assertEqual(result["retained_deep_reference_groups"]["50"], {"rows":60,"distinct_episodes":60})

    def test_only_retained_arrays_reach_actual_metric(self):
        args = list(fixture())
        args[1][[3,73,143]] = False
        with patch.object(R.SM4, "l15", wraps=R.SM4.l15) as metric:
            result = R.retained_l15(*args)
        self.assertEqual(result["status"], "PASS")
        for actual, full in zip(metric.call_args.args[:4], [args[0]["obs"],args[0]["episode_ids"],args[0]["times"],args[2]]):
            np.testing.assert_array_equal(actual, full[args[1]])

    def test_removed_outlier_cannot_change_retained_normalization_or_result(self):
        args = list(fixture())
        args[1][[3,73,143]] = False
        expected = R.retained_l15(*args)
        args[0]["obs"][~args[1]] = 1e20
        self.assertEqual(expected, R.retained_l15(*args))

    def test_bars_and_actual_64_boundary_unchanged(self):
        args = fixture(64)
        out = R.retained_l15(*args)
        self.assertEqual(out["status"], "PASS")
        self.assertEqual((out["neighbors"],out["start_bar"],out["anchor_bar"]), (64,.95,.9))
        with patch.object(R.SM4,"l15",return_value={"pass":False,"start_share":.94,"anchor_share":.9,"thresholds_by_t":{0:{"threshold":1}}}):
            self.assertEqual(R.retained_l15(*args)["status"], "FAIL")

    def test_duplicate_rows_rejected(self):
        args = list(fixture())
        args[0]["episode_ids"][1] = 0
        self.assertEqual(R.retained_l15(*args)["status"], "INCONCLUSIVE")

    def test_nonfinite_retained_input_rejected(self):
        args = list(fixture()); args[0]["obs"][0,0] = np.nan
        self.assertEqual(R.retained_l15(*args)["status"], "INCONCLUSIVE")

    def test_non_boolean_mask_rejected(self):
        args = list(fixture()); args[1] = args[1].astype(int)
        self.assertEqual(R.retained_l15(*args)["status"], "INCONCLUSIVE")

    def test_bad_episode_or_time_identifiers_rejected(self):
        for field, value in [("episode_ids",np.nan),("episode_ids",.5),("episode_ids",-1),("times",np.inf),("times",100)]:
            with self.subTest(field=field,value=value):
                args=list(fixture()); args[0][field]=args[0][field].astype(float); args[0][field][0]=value
                self.assertEqual(R.retained_l15(*args)["status"],"INCONCLUSIVE")

    def test_bad_query_shapes_and_times_rejected(self):
        for which, value in [(3,np.zeros((3,13))),(4,np.zeros((3,15))),(5,np.array([0,25,100]))]:
            with self.subTest(which=which):
                args=list(fixture()); args[which]=value
                self.assertEqual(R.retained_l15(*args)["status"],"INCONCLUSIVE")

    def test_observation_time_must_match_group_time(self):
        for which in ("rows","starts","anchors"):
            with self.subTest(which=which):
                args=list(fixture())
                target=args[0]["obs"] if which=="rows" else args[3] if which=="starts" else args[4]
                target[0,13]=.123
                self.assertEqual(R.retained_l15(*args)["status"],"INCONCLUSIVE")

    def test_nonfinite_metric_threshold_cannot_pass(self):
        with patch.object(R.SM4,"l15",return_value={"pass":True,"start_share":1.,"anchor_share":1.,"thresholds_by_t":{0:{"threshold":float('inf')}}}):
            self.assertEqual(R.retained_l15(*fixture())["status"], "INCONCLUSIVE")

    def test_label_missing_support_stays_inconclusive(self):
        d={"stall3":{k:{"ok":True} for k in ("D1","D2","D3","D4","B1")}}
        d["stall3"]["D3"] = {"ok":False,"status":"INCONCLUSIVE"}
        labels = G.decision_labels(d,{"H1":{"ok":True},"H2":{"ok":True}})
        self.assertEqual(len(labels),1)
        self.assertTrue(labels[0].startswith("INCONCLUSIVE"))

    def test_scientific_entry_is_disabled_before_data_generation(self):
        with patch.object(G.D,"generate",side_effect=AssertionError("collection forbidden")):
            with self.assertRaisesRegex(SystemExit,"Repair-only"):
                G.main()

    def test_real_d_checks_passes_cutoff_mask_to_helper(self):
        rows, mask, deep, starts, anchors, times = fixture(100)
        rows.update(actions=np.zeros((len(mask),4),np.float32),bands=np.zeros(len(mask),int),episodes=np.array([0,1,2]))
        rows["actions"][:,0]=-.7
        rows["bands"][:70]=1
        arrays={"reset_component":np.ones(3),"observations":np.zeros((3,101,14),np.float32)}
        ds=types.SimpleNamespace(arrays=arrays,metadata={"recorder_sha256":G.sha(HERE/'recorder_wide.py'),"behavior":{"name":"knee_s10_wide_window_v1"}})
        with patch.object(G.M,"fitting_rows",return_value=rows), patch.object(G.M,"arm_masks",return_value=({"cutoff":mask},None)), patch.object(G,"retained_l15",return_value={"ok":False,"status":"INCONCLUSIVE"}) as check, patch.object(G.pi_ref,"act",return_value=np.zeros((297,4))), patch.object(G.pi_ref_v1_dev,"act",return_value=np.zeros((297,4))):
            out=G.d_checks({"stall3":ds},starts,[{"obs":o,"t":t} for o,t in zip(anchors,times)])
        np.testing.assert_array_equal(check.call_args.args[1],mask)
        self.assertEqual(out["stall3"]["D3"]["status"],"INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
