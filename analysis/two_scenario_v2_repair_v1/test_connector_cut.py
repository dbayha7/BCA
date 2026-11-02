"""Synthetic transition-array tests only; no physical plant or model execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location("_connector_cut_fixture_test", HERE / "connector_cut.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def state(q):
    x = np.zeros(13, np.float64); x[0] = q
    return x


def geometry():
    return {"A": state(-.1).tolist(), "B": state(0).tolist(), "C": state(1).tolist(), "D": state(2).tolist(),
            "goal": {"center": state(2)[:10].tolist(), "radius": .1},
            "danger": {"center": state(10)[:10].tolist(), "radius": .1}}


def fixture(fit=8, cal=128, reserved=4):
    D = C.authenticated_data_module()
    total = fit + cal + reserved
    qs = [-.1, 0, .1, .3, .6, .9, 1, 1.5, 2]
    rows = []
    for parent in range(total):
        role = 0 if parent < fit else 1 if parent < fit + cal else 2
        for j in range(8):
            x, y, t = state(qs[j]), state(qs[j + 1]), 24 + j
            rows.append((x, y, D.T.observe(x, t), D.T.observe(y, t + 1), np.zeros(4, np.float32),
                         99.0 if j == 7 else -1.0, j == 7, parent, len(rows), t, role, parent, parent % 4))
    types = (np.float64, np.float64, np.float32, np.float32, np.float32, np.float64, bool,
             np.int64, np.int64, np.int32, np.int8, np.int64, np.int8)
    arrays = {key: np.asarray(values, dtype=typ) for key, values, typ in zip(D.ARRAY_FIELDS, zip(*rows), types)}
    protocol = {"parent_episodes": total, "fit_parent_episodes": fit, "calibration_parent_episodes": cal,
                "reserved_parent_episodes": reserved, "geometry_recipe": {"connector_trim_fraction": .2},
                "collection": {"physical_start_jitter": .01},
                "setup_rules": {"minimum_approach_parents": 8, "minimum_fragment_parents": 8,
                                "calibration_eligible_parent_minimum": 128, "minimum_removed_parents": 8,
                                "approach_physical_l2_radius": .1, "approach_current_radius": .005,
                                "approach_wave_l2_radius": .02}}
    return arrays, geometry(), protocol


class ConnectorCutTests(unittest.TestCase):
    def test_projection_complete_state_and_physical_axis(self):
        x = state(.4); x[10:] = [123, -54, 99]
        self.assertEqual(C.progress(x, geometry()), .4)
        g = geometry(); g["C"] = g["B"]
        with self.assertRaises(ValueError):
            C.progress(x, g)

    def test_closed_boundaries_reverse_and_direct_jump(self):
        a = np.array([state(q) for q in (0, .2, .8, 1, -.2, .05, .9)])
        b = np.array([state(q) for q in (.2, .1, 1, 0, 1.2, .1, 1.2)])
        got = C.retained_mask({"states": a, "next_states": b}, geometry())
        np.testing.assert_array_equal(got, [False, False, False, False, False, True, True])
        np.testing.assert_array_equal(got, C.retained_mask({"states": b, "next_states": a}, geometry()))

    def test_mask_is_not_outcome_or_role_dependent(self):
        a, g, _ = fixture()
        before = C.retained_mask(a, g)
        a["rewards"] *= -999
        a["roles"][:] = 2
        np.testing.assert_array_equal(before, C.retained_mask(a, g))

    def test_success_paths_cut_genuine_fragments_and_labels_preserved(self):
        a, g, p = fixture()
        before = {k: v.tobytes() for k, v in a.items()}
        out = C.analyze(a, g, p)
        self.assertEqual(out["decision"], "saved_data_mask_checks_passed")
        self.assertEqual(out["by_role"]["fitting"]["intact_retained_success_parents"], 0)
        self.assertEqual(out["by_role"]["fitting"]["genuine_fragment_pair_parents"], 8)
        self.assertEqual(out["by_role"]["fitting"]["retained_next_goal_rows"], 8)
        self.assertEqual(out["support"]["B"]["distinct_parents"], 8)
        self.assertEqual(out["support"]["C"]["distinct_parents"], 8)
        parent = out["by_parent"][0]
        self.assertEqual(parent["upstream_fragment_row_ids"], [0, 1])
        self.assertEqual(parent["downstream_goal_fragment_row_ids"], [5, 6, 7])
        self.assertEqual(parent["original_first_time"], 24)
        self.assertEqual(parent["original_final_time"], 31)
        self.assertFalse(out["training_ready"])
        self.assertEqual(before, {k: v.tobytes() for k, v in a.items()})

    def test_fewer_than_eight_support_parents_fails(self):
        a, g, p = fixture(fit=7)
        out = C.analyze(a, g, p)
        self.assertIn("B_retained_full_state_time_support", out["failed_criteria"])
        self.assertIn("eight_genuine_fitting_fragment_pairs", out["failed_criteria"])

    def test_full_state_support_rejects_wrong_current(self):
        a, g, p = fixture()
        # The geometric query is moved only for the exogenous current coordinate;
        # physical cut and saved data remain unchanged.
        g["B"][10] = .1
        out = C.analyze(a, g, p)
        self.assertEqual(out["support"]["B"]["distinct_parents"], 0)
        self.assertIn("B_retained_full_state_time_support", out["failed_criteria"])

    def test_insufficient_calibration_parents_fails(self):
        a, g, p = fixture(cal=127)
        self.assertIn("128_eligible_calibration_parents", C.analyze(a, g, p)["failed_criteria"])

    def test_entire_goal_ball_separation_required(self):
        a, g, p = fixture()
        g["goal"]["radius"] = 1.3
        out = C.analyze(a, g, p)
        self.assertLess(out["separation"]["entire_goal_ball_q_lower"], .8)
        self.assertIn("entire_goal_ball_after_slab", out["failed_criteria"])

    def test_entire_initial_box_not_just_observed_starts(self):
        a, g, p = fixture()
        p["collection"]["physical_start_jitter"] = .5
        out = C.analyze(a, g, p)
        self.assertTrue(out["separation"]["all_observed_starts_before_slab"])
        self.assertIn("entire_original_start_box_before_slab", out["failed_criteria"])

    def test_hashes_are_explicit_and_deterministic(self):
        a, g, p = fixture()
        first, second = C.analyze(a, g, p), C.analyze(a, g, p)
        self.assertEqual(first["hashes"], second["hashes"])
        self.assertEqual(first["mask_definition"]["closed_band"], [.2, .8])
        self.assertEqual(len(first["hashes"]["retained_mask_uint8_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
