"""Pure budget/startup/persistence fixtures; no physical plant or model imports."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("two_scenario_v3_setup_fixture", HERE / "setup.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class SetupTests(unittest.TestCase):
    def inputs(self):
        return SimpleNamespace(shape=(13,)), SimpleNamespace(shape=(4,))

    def test_cumulative_budget_preserves_prior_costs(self):
        a = S.read(HERE / "BUILD_ALLOCATION.json")
        self.assertEqual(a["prior_charged_seconds"] + a["new_phase_charged_seconds"] + a["later_study_reserved_seconds"], 14400)
        self.assertEqual(a["new_preparation_allowance_seconds"] + a["new_manual_setup_cap_seconds"], a["new_phase_charged_seconds"])
        self.assertEqual(sum(S.Counter.CAPS.values()), a["new_setup_transition_cap"])
        self.assertEqual(210 + a["new_setup_transition_cap"] + 237600, 498312)
        self.assertEqual(260590 - a["new_setup_transition_cap"], 88)
        self.assertEqual(4 + S.Counter.START_CAPS["geometry"], 50)
        self.assertFalse(a["training_authorized"])

    def test_cap_and_batch_rejection_precede_plant(self):
        c = S.Counter(lambda *args: self.fail("No plant call allowed"), lambda: None)
        with self.assertRaisesRegex(ValueError, "One full"):
            c(SimpleNamespace(shape=(10, 13)), self.inputs()[1])
        c.attempted["geometry"] = 4502
        with self.assertRaisesRegex(ValueError, "cap exhausted"):
            c(*self.inputs())

    def test_failed_call_preserves_unknown_counts(self):
        def fail(*args):
            raise RuntimeError("fixture failure")
        c = S.Counter(fail, lambda: None)
        with self.assertRaises(RuntimeError):
            c(*self.inputs())
        self.assertEqual(c.receipt()["attempted"]["geometry"], 1)
        self.assertEqual(c.receipt()["completed"]["geometry"], 0)
        self.assertFalse(c.receipt()["execution_counts_known"])

    def test_progress_failure_keeps_completed_work(self):
        def fail(unused):
            raise OSError("fixture progress write failure")
        c = S.Counter(lambda *args: None, lambda: None, fail)
        c.attempted["geometry"] = c.completed["geometry"] = 999
        with self.assertRaises(OSError):
            c(*self.inputs())
        self.assertEqual(c.completed["geometry"], 1000)
        self.assertTrue(c.receipt()["execution_counts_known"])

    def test_initializations_deadline_and_no_replay_reset(self):
        c = S.Counter(None, lambda: None)
        for _ in range(46):
            c.initialize()
        with self.assertRaises(ValueError):
            c.initialize()
        c.begin("replay")
        with self.assertRaises(ValueError):
            c.initialize()
        def expired():
            raise TimeoutError("fixture")
        c.deadline = expired
        with self.assertRaises(TimeoutError):
            c(*self.inputs())
        self.assertEqual(sum(c.attempted.values()), 0)

    def test_save_is_exclusive_and_preserves_nonfinite_diagnostics(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "diagnostics.json"
            S.save(p, {"radius": float("nan"), "passed": False})
            self.assertEqual(S.read(p), {"radius": "NaN", "passed": False})
            with self.assertRaises(FileExistsError):
                S.save(p, {"passed": True})

    def test_conflicting_v2_reservation_blocks(self):
        with TemporaryDirectory() as td, patch.object(S, "HERE", Path(td) / "two_scenario_v3"):
            S.no_overlap()
            (Path(td) / "two_scenario_v2/setup_attempt_1").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Concurrent alternative"):
                S.no_overlap()

    def test_windows_cannot_reserve_or_import_physics(self):
        with patch.object(S.os, "name", "nt"), patch.object(S, "check") as check:
            with self.assertRaisesRegex(ValueError, "Manual existing Linux"):
                S.execute_setup()
            check.assert_not_called()

    def test_future_scientific_work_remains_disabled_and_fixed(self):
        p = S.read(HERE / "protocol.json")
        self.assertFalse(p["training_authorized"])
        self.assertFalse(p["scientific_evaluation_authorized"])
        self.assertEqual((p["kappa"], p["native_td3_alpha"], p["bc_coefficient"]), (3., 2.5, 1.))
        self.assertEqual((p["geometry_recipe"]["b_time"], p["geometry_recipe"]["c_time"]), (25, 30))
        self.assertEqual(p["geometry_amendment"]["connector_radius"], .01)
        self.assertEqual(p["geometry_amendment"]["danger_terminal_time"], 26)


if __name__ == "__main__":
    unittest.main()
