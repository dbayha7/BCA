"""Pure saved-array/fake-recorder tests. No physical simulator or learner runs."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("_two_scenario_v3_amendment_test", HERE / "amendment.py")
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


def state(x=0., y=0., z=0.):
    result = np.zeros(13)
    result[:3] = [x, y, z]
    return result


def fixture():
    route = [state(t / 10, t / 2) for t in range(101)]
    excursion = [route[25].copy()] + [state(2.5 + j / 10, 12.5 - j / 2) for j in range(1, 6)]
    trace = [{"phase": "nominal", "time": t, "state": route[t].tolist(),
              "action": [.5, 0, 0, 0], "next_state": route[t + 1].tolist()} for t in range(100)]
    trace += [{"phase": "excursion", "time": 25 + j, "state": excursion[j].tolist(),
               "action": [-.5, 0, 0, 0], "next_state": excursion[j + 1].tolist()} for j in range(5)]
    m = {"geometry": {"nominal_route": np.asarray(route).tolist(), "excursion": np.asarray(excursion).tolist()}}
    recipe = {"initial_state": state().tolist(), "b_time": 25, "c_time": 30, "goal_time": 75,
              "goal_radius": .1, "danger_radius_max": .1, "danger_radius_fraction": .25,
              "minimum_region_radius": .01, "connector_trim_fraction": .2,
              "connector_radius_fraction": .1}
    return m, trace, recipe


class AmendmentTests(unittest.TestCase):
    def test_geometry_uses_exact_first_bad_successor_and_fixed_width(self):
        m, trace, recipe = fixture()
        result = A.geometry_from_saved(m, trace, recipe)
        g = result["geometry"]
        self.assertEqual(g["danger"]["center"], trace[100]["next_state"][:10])
        self.assertEqual(g["excursion"], [trace[100]["state"], trace[100]["next_state"]])
        self.assertEqual(g["connector"]["radius"], .01)
        self.assertNotEqual(g["connector"]["radius"], .1 * result["values"]["connector_length"])
        self.assertEqual(g["B"], m["geometry"]["nominal_route"][25])
        self.assertEqual(g["C"], m["geometry"]["nominal_route"][30])
        self.assertEqual(g["D"], m["geometry"]["nominal_route"][75])
        self.assertTrue(result["checks"]["safe_first_edge_avoids_danger"])
        self.assertTrue(result["checks"]["bad_first_edge_enters_danger"])
        self.assertEqual(result["failed_criteria"], [])
        self.assertEqual(result["not_evaluable"], [])

    def test_width_is_independent_of_route_speed_fraction(self):
        m, trace, recipe = fixture()
        recipe["connector_radius_fraction"] = .001
        self.assertEqual(A.geometry_from_saved(m, trace, recipe)["geometry"]["connector"]["radius"], .01)

    def test_report_overlap_without_raising_or_searching(self):
        m, trace, recipe = fixture()
        recipe["goal_radius"] = 100.0
        result = A.geometry_from_saved(m, trace, recipe)
        self.assertFalse(result["checks"]["goal_danger_disjoint"])
        self.assertIn("goal_danger_disjoint", result["failed_criteria"])
        self.assertIn("geometry", result)

    def test_bad_source_and_original_trace_mismatch_fail_named_checks(self):
        m, trace, recipe = fixture()
        trace[100]["state"][0] += .1
        result = A.geometry_from_saved(m, trace, recipe)
        self.assertFalse(result["checks"]["hazard_source_is_exact_B"])
        self.assertFalse(result["checks"]["excursion_matches_saved_measurements"])
        m, trace, recipe = fixture()
        m["geometry"]["nominal_route"][2][0] += 1
        self.assertFalse(A.geometry_from_saved(m, trace, recipe)["checks"]["nominal_matches_saved_measurements"])

    def test_bad_action_phase_or_time_is_not_silently_substituted(self):
        m, trace, recipe = fixture()
        trace[100]["action"][0] = .5
        trace[100]["time"] = 26
        result = A.geometry_from_saved(m, trace, recipe)
        self.assertFalse(result["checks"]["first_opposing_action_unchanged"])
        self.assertFalse(result["checks"]["trace_phases_and_times_unchanged"])

    def test_nonfinite_evidence_returns_failed_or_unevaluable_checks(self):
        m, trace, recipe = fixture()
        trace[100]["next_state"][0] = float("nan")
        result = A.geometry_from_saved(m, trace, recipe)
        self.assertFalse(result["checks"]["captured_states_and_actions_finite"])
        self.assertFalse(result["checks"]["danger_separation_finite"])
        self.assertIn("safe_first_edge_avoids_danger", result["not_evaluable"])

    def test_fixed_times_and_complete_trace_required(self):
        m, trace, recipe = fixture()
        with self.assertRaises(ValueError):
            A.geometry_from_saved(m, trace[:-1], recipe)
        recipe["b_time"] = 24
        with self.assertRaises(ValueError):
            A.geometry_from_saved(m, trace, recipe)

    def test_geometry_does_not_mutate_saved_inputs(self):
        m, trace, recipe = fixture()
        before = copy.deepcopy((m, trace, recipe))
        A.geometry_from_saved(m, trace, recipe)
        self.assertEqual((m, trace, recipe), before)

    def test_recorder_reverses_only_danger_family_at_25(self):
        control = lambda x, t: np.array([.5, -.25, .125, -.5])
        config = {"action_noise": .03, "wide_action_noise": .15}
        geometry = {"danger_time": 25, "b_time": 25, "c_time": 30}
        for family in range(4):
            for time in (24, 25, 26, 29, 30):
                rng = mock.Mock(); rng.normal.return_value = np.zeros(4)
                got = A.recorder_action(control, state(), time, family, rng, geometry, config)
                expected = -control(None, time) if family == 2 and time == 25 else control(None, time)
                np.testing.assert_array_equal(got, expected.astype(np.float32))
                sigma = 0.0 if family == 3 else .15 if family == 1 else .03
                rng.normal.assert_called_once_with(0.0, sigma, 4)

    def test_collector_reuses_original_code_with_one_isolated_global(self):
        self.assertIs(A.collect.__code__, A.D.collect.__code__)
        self.assertIsNot(A.collect.__globals__, A.D.collect.__globals__)
        changed = [key for key in A.D.collect.__globals__
                   if A.collect.__globals__[key] is not A.D.collect.__globals__[key]]
        self.assertEqual(changed, ["recorder_action"])
        self.assertIs(A.collect.__globals__["recorder_action"], A.recorder_action)
        self.assertIsNot(A.D.collect.__globals__["recorder_action"], A.recorder_action)
        self.assertIs(A.collect.__globals__["episode_roles"], A.D.episode_roles)
        self.assertIs(A.collect.__globals__["ARRAY_FIELDS"], A.D.ARRAY_FIELDS)
        self.assertNotIn("next_actions", A.D.ARRAY_FIELDS)

    def test_original_sources_remain_sealed_and_unique(self):
        self.assertEqual(A.verify_original_sources(), A.ORIGINAL_SOURCE_HASHES)
        self.assertEqual(A.T.__name__, "experiments.thrusterchain_bca.two_scenario_v1.task")
        self.assertEqual(A.D.__name__, "experiments.thrusterchain_bca.two_scenario_v1.data")

    def test_callback_stops_adapted_collector_before_any_plant_work(self):
        callback = mock.Mock(side_effect=RuntimeError("Fixture stop before initialization"))
        fake_task, controller = mock.Mock(), mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "Fixture stop"):
            A.collect(fake_task, controller, {"A": state().tolist()}, {"parents": 1280, "split_seed": 23}, on_parent_start=callback)
        callback.assert_called_once_with(0)
        fake_task.step.assert_not_called()
        controller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
