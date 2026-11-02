"""Budget/startup fixtures only; never imports or executes the canonical plant."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("two_scenario_setup_fixture", HERE / "setup.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class SetupTests(unittest.TestCase):
    def inputs(self):
        return SimpleNamespace(shape=(13,)), SimpleNamespace(shape=(4,))

    def test_overall_allocation_and_training_arithmetic(self):
        a, p = S.read(HERE / "BUILD_ALLOCATION.json"), S.read(HERE / "protocol.json")
        self.assertEqual(a["preparation_charge_seconds"] + a["manual_setup_cap_seconds"] + a["later_study_reserved_seconds"], 14400)
        self.assertEqual(sum(a[k] for k in ("geometry_transition_cap", "collection_transition_cap", "single_replay_transition_cap")), 260800)
        self.assertEqual(a["setup_transition_cap"] + a["later_evaluation_transition_cap"], 498400)
        self.assertEqual(sum(p[k] for k in ("evaluation_return_steps", "evaluation_own_qpi_steps", "evaluation_common_reference_steps", "evaluation_prefix_parity_steps")), 237600)
        self.assertEqual(2 * 3 * (10000 + 2 * 10000), p["total_critic_updates"])
        self.assertEqual(2 * 3 * (1 + 2), p["total_calibrations"])
        self.assertFalse(p["training_authorized"])
        self.assertFalse(p["scientific_evaluation_authorized"])

    def test_physical_cap_rejects_before_the_call(self):
        calls = []
        c = S.Counter(lambda *args: calls.append(1), lambda: None)
        c.attempted["geometry"] = c.completed["geometry"] = 4800
        with self.assertRaisesRegex(ValueError, "Phase transition"):
            c(*self.inputs())
        self.assertEqual(calls, [])

    def test_failed_call_keeps_attempted_work_unknown(self):
        def fail(*args): raise RuntimeError("fixture plant failure")
        c = S.Counter(fail, lambda: None)
        with self.assertRaises(RuntimeError): c(*self.inputs())
        self.assertEqual(c.attempted["geometry"], 1)
        self.assertEqual(c.completed["geometry"], 0)
        self.assertFalse(c.receipt()["execution_counts_known"])

    def test_batches_cannot_be_miscounted_as_one(self):
        c = S.Counter(lambda *a: None, lambda: None)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            c(SimpleNamespace(shape=(10,13)), SimpleNamespace(shape=(10,4)))
        self.assertEqual(c.attempted["geometry"], 0)

    def test_initialization_caps_and_replay_has_no_reset(self):
        c = S.Counter(lambda *a: None, lambda: None)
        for _ in range(48): c.initialize()
        with self.assertRaises(ValueError): c.initialize()
        c.begin("collection")
        c.starts["collection"] = 1279
        c.initialize(1279)
        with self.assertRaises(ValueError): c.initialize(1280)
        c.begin("replay")
        with self.assertRaises(ValueError): c.initialize()
        self.assertEqual(c.receipt()["gym_reset_calls"], 0)

    def test_deadline_stops_before_physical_work(self):
        def expired(): raise TimeoutError("fixture deadline")
        c = S.Counter(lambda *a: self.fail("must not run"), expired)
        with self.assertRaises(TimeoutError): c(*self.inputs())
        self.assertEqual(sum(c.attempted.values()), 0)

    def test_output_is_exclusive_and_cannot_reset_attempt(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ATTEMPT.json"
            S.save(p, {"fixture": 1})
            with self.assertRaises(FileExistsError): S.save(p, {"fixture": 2})
            self.assertEqual(S.read(p), {"fixture": 1})

    def test_windows_entrypoint_refuses_before_reservation_or_import(self):
        with TemporaryDirectory() as td, patch.object(S, "HERE", Path(td)), patch.object(S, "check", return_value={}), \
                patch.object(S.os, "name", "nt"), patch.object(S.importlib.metadata, "version") as runtime:
            with self.assertRaisesRegex(ValueError, "Manual existing Linux"):
                S.execute_setup()
            runtime.assert_not_called()
            self.assertFalse((Path(td) / "setup_attempt_1").exists())

    def test_frozen_geometry_does_not_use_method_outcomes(self):
        p = S.read(HERE / "protocol.json")
        r = p["geometry_recipe"]
        self.assertEqual((r["b_time"], r["c_time"], r["goal_time"]), (25,30,75))
        self.assertEqual(p["setup_rules"]["witness_families"], 15)
        self.assertTrue(p["setup_rules"]["no_geometry_search"])
        self.assertEqual(p["native_td3_alpha"], 2.5)
        self.assertEqual(p["bc_coefficient"], 1.)

    def test_connector_evidence_requires_real_gap_and_both_fragments(self):
        import numpy as np
        arrays = {"episode_ids": np.repeat(np.arange(3), 8), "roles": np.zeros(24, np.int8),
                  "times": np.tile(np.arange(23,31), 3), "terminals": np.tile([False]*7+[True], 3),
                  "rewards": np.tile([-1.]*7+[99.], 3)}
        mask = np.ones(24, bool)
        self.assertEqual(S.joined_fragment_parent_ids(arrays, mask), [])
        mask[3:6] = False  # Parent0: true omitted block26..28, retained25and29.
        mask[11:16] = False  # Parent1: no retained suffix.
        mask[16:22] = False  # Parent2: no retained prefix.
        self.assertEqual(S.joined_fragment_parent_ids(arrays, mask), [0])
        mask[7] = False  # A goal that is not retained is not supported C->D.
        self.assertEqual(S.joined_fragment_parent_ids(arrays, mask), [])
        mask[7] = True
        mask[0] = False  # A second deleted block breaks the claimed intact prefix.
        self.assertEqual(S.joined_fragment_parent_ids(arrays, mask), [])
        mask[0] = True
        arrays["rewards"][7] = -101.
        self.assertEqual(S.joined_fragment_parent_ids(arrays, mask), [])


if __name__ == "__main__":
    unittest.main()
