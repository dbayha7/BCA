"""Pure fixtures: no canonical simulator/model imports or physical calls."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load("_danger_diagnostic_test", HERE / "danger_diagnostic.py")
T = load("_danger_diagnostic_fake_task", S.V2 / "task.py")


def state(t, y=0., family=0):
    x = np.zeros(13, np.float64)
    x[0], x[1], x[10] = t, y, family * .001
    return x


def anchors():
    return [{"family": i, "source_row": 29, "time": 29, "state": state(29, family=i).tolist(),
             "bad_action": [-1., 0., 0., 0.], "expected_next_state": state(30, -1., i).tolist(),
             "expected_reward": -101.01, "expected_terminated": True,
             "expected_terminal_reason": "danger", "expected_next_time": 30} for i in range(15)]


def fake_env(deadline=lambda: None, *, goal=True):
    g = {"danger": {"center": state(30, -1.)[:10].tolist(), "radius": .01},
         "goal": {"center": state(32 if goal else 1000, .3)[:10].tolist(), "radius": .01},
         "connector": {"center": state(500)[:10].tolist(), "radius": .01}}
    def plant(x, a):
        out = x.copy()
        out[0] += 1
        out[1] += -1. if a[0] < 0 else .1
        return out
    counter = S.Counter(plant, deadline)
    return T.Task(counter, g), counter


def control(x, t):
    return np.array([1., 0., 0., 0.], np.float32)


def fake_evidence():
    cases = []
    for a in anchors():
        rows = [{"time": t, "state": state(t, family=a["family"]).tolist(),
                 "next_state": state(t + 1, family=a["family"]).tolist(),
                 "action": [1., 0., 0., 0.], "reward": -1.01,
                 "terminated": False, "reason": None} for t in range(30)]
        rows[29].update(state=a["state"], next_state=a["expected_next_state"], action=a["bad_action"],
                        reward=a["expected_reward"], terminated=True, reason="danger")
        cases.append({"family": a["family"], "variants": {"danger": {"trace": rows, "terminal_time": 30}}})
    return {"families": 15, "cases": cases}


class DiagnosticFixtures(unittest.TestCase):
    def test_same_full_state_time_early_stop_and_discount(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            env, counter = fake_env()
            called = []
            def controller(x, t):
                called.append((x.copy(), t))
                return control(x, t)
            original = S.append
            def append_after_bank(path, event):
                self.assertTrue((root / "ANCHOR_BANK.json").exists())
                return original(path, event)
            with patch.object(S, "append", side_effect=append_after_bank):
                report = S.run_cases(env, controller, anchors(), counter, root)
            self.assertTrue(report["passed"], report)
            self.assertEqual((report["denominator"], report["completed_cases"]), (15, 15))
            self.assertEqual(counter.receipt()["plant_calls_completed"], 60)
            self.assertEqual(counter.starts, 30)
            case = S.read(root / "case_00.json")
            self.assertEqual(case["safe"]["terminal_time"], 32)
            self.assertAlmostEqual(case["safe"]["return"], -1.01 + .98 * -1.01 + .98**2 * 98.99, places=13)
            self.assertEqual(case["bad"]["trace"][0]["state"], case["safe"]["trace"][0]["state"])
            self.assertEqual([t for _, t in called[:3]], [29, 30, 31])
            self.assertTrue(np.array_equal(called[0][0], np.asarray(anchors()[0]["state"])))

    def test_all_scientific_failures_preserved_and_exact_maximum(self):
        with TemporaryDirectory() as td:
            env, counter = fake_env(goal=False)
            report = S.run_cases(env, control, anchors(), counter, Path(td))
            self.assertFalse(report["passed"])
            self.assertEqual((report["denominator"], report["completed_cases"]), (15, 15))
            self.assertEqual(report["safe_goal_count"], 0)
            self.assertEqual((counter.attempted, counter.completed, counter.starts), (1080, 1080, 30))
            self.assertEqual(len(list(Path(td).glob("case_*.json"))), 15)
            with self.assertRaisesRegex(ValueError, "transition cap"):
                counter(state(0), np.zeros(4, np.float32))
            self.assertEqual(counter.attempted, 1080)

    def test_fatal_replay_mismatch_persists_case_without_new_anchor(self):
        bank = anchors()
        bank[0]["expected_reward"] += .1
        with TemporaryDirectory() as td:
            env, counter = fake_env()
            with self.assertRaises(S.ReplayMismatch):
                S.run_cases(env, control, bank, counter, Path(td))
            case = S.read(Path(td) / "case_00.json")
            self.assertFalse(case["bad"]["exact_replay"])
            self.assertIsNone(case["safe"])
            self.assertEqual(counter.completed, 1)
            self.assertEqual(len(list(Path(td).glob("case_*.json"))), 1)
            self.assertEqual(len(S.read(Path(td) / "ANCHOR_BANK.json")), 15)

    def test_transition_write_failure_is_not_zero_work(self):
        with TemporaryDirectory() as td:
            env, counter = fake_env()
            with patch.object(S, "append", side_effect=OSError("fixture full disk")):
                with self.assertRaisesRegex(OSError, "full disk"):
                    S.run_cases(env, control, anchors(), counter, Path(td))
            self.assertEqual(counter.completed, 1)
            self.assertIn("full disk", S.read(Path(td) / "case_00.json")["error"])

    def test_case_write_failure_reaches_stdout_and_caller(self):
        with TemporaryDirectory() as td:
            env, counter = fake_env()
            bank = anchors(); bank[0]["expected_reward"] += .1
            original = S.save
            def fail_case(path, value):
                if path.name.startswith("case_"):
                    raise OSError("fixture case disk failure")
                original(path, value)
            output = io.StringIO()
            with patch.object(S, "save", side_effect=fail_case), redirect_stdout(output):
                with self.assertRaises(S.ReplayMismatch):
                    S.run_cases(env, control, bank, counter, Path(td))
            event = json.loads(output.getvalue())
            self.assertEqual(event["status"], "case_persistence_failed")
            self.assertEqual(event["counts"]["plant_calls_completed"], 1)

    def test_timeout_before_next_call_preserves_partial_case(self):
        holder = {}
        def deadline():
            if holder["counter"].completed >= 3:
                raise TimeoutError("fixture time exhausted")
        env, counter = fake_env(deadline)
        holder["counter"] = counter
        with TemporaryDirectory() as td:
            with self.assertRaises(TimeoutError):
                S.run_cases(env, control, anchors(), counter, Path(td))
            self.assertEqual((counter.attempted, counter.completed), (3, 3))
            self.assertIn("TimeoutError", S.read(Path(td) / "case_00.json")["error"])

    def test_failed_plant_call_keeps_uncertainty_and_no_retry(self):
        def fail(*args): raise RuntimeError("fixture plant failure")
        c = S.Counter(fail, lambda: None)
        with self.assertRaises(RuntimeError): c(state(29), np.zeros(4, np.float32))
        self.assertEqual((c.attempted, c.completed), (1, 0))
        self.assertFalse(c.receipt()["execution_counts_known"])

    def test_exclusive_bank_write_prevents_any_second_calls(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            S.save(root / "ANCHOR_BANK.json", {"existing": True})
            env, counter = fake_env()
            with self.assertRaises(FileExistsError):
                S.run_cases(env, control, anchors(), counter, root)
            self.assertEqual((counter.starts, counter.attempted), (0, 0))

    def test_anchor_bank_retains_all_coordinates_and_rejects_partial_lineage(self):
        evidence = fake_evidence()
        bank = S.anchor_bank(evidence)
        self.assertEqual(bank, anchors())
        evidence["cases"][7]["variants"]["danger"]["trace"][28]["next_state"][12] = 1.
        with self.assertRaisesRegex(ValueError, "lineage"):
            S.anchor_bank(evidence)
        with self.assertRaisesRegex(ValueError, "All 15"):
            S.anchor_bank({"families": 14, "cases": evidence["cases"][:-1]})

    def test_windows_refuses_before_checks_reservation_or_runtime(self):
        with TemporaryDirectory() as td, patch.object(S, "HERE", Path(td)), \
                patch.object(S, "os", SimpleNamespace(name="nt")), \
                patch.object(S, "check") as check, patch.object(S, "load_runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "Manual existing Linux"):
                S.execute()
            check.assert_not_called(); runtime.assert_not_called()
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_v3_reservation_blocks_shared_budget(self):
        with TemporaryDirectory() as td, patch.object(S, "V3", Path(td)):
            S.no_v3_attempt()
            (Path(td) / "setup_attempt_1").mkdir()
            with self.assertRaisesRegex(ValueError, "shared budget"):
                S.no_v3_attempt()

    def test_contract_detects_code_or_test_changes_and_cannot_reprepare(self):
        with TemporaryDirectory() as td, patch.object(S, "HERE", Path(td)), patch.object(S, "source_check", return_value={}):
            root = Path(td)
            (root / "danger_diagnostic.py").write_text("fixture source", encoding="utf-8")
            (root / "test_danger_diagnostic.py").write_text("fixture tests", encoding="utf-8")
            S.prepare(); S.check()
            with self.assertRaises(FileExistsError): S.prepare()
            (root / "test_danger_diagnostic.py").write_text("changed tests", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code, tests or contract"):
                S.check()

    def test_changed_bound_source_rejected_before_importing_checker(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            source = root / "v2"; source.mkdir()
            (source / "setup.py").write_text("fixture checker", encoding="utf-8")
            (source / "setup_attempt_1").mkdir()
            S.save(source / "MANIFEST.json", {"files": {"v2/setup.py": S.sha(source / "setup.py")}})
            S.save(source / "setup_attempt_1/geometry.json", {"fixture": True})
            S.save(source / "setup_attempt_1/RESULT.json", {
                "prepared_identity_unchanged": True, "execution_counts_known": True,
                "status": "setup_passed_pending_geometry_protocol_review",
                "counts": {"attempted": {"total": 188832}, "completed": {"total": 188832}}})
            bound = {name: S.sha(source / name) for name in
                     ("MANIFEST.json", "setup_attempt_1/geometry.json", "setup_attempt_1/RESULT.json")}
            protocol = {"gamma": .98, "horizon": 100,
                        "setup_rules": {"reference_return_advantage": 5., "required_safe_goal_families": 12}}
            with patch.object(S, "V2", source), patch.object(S, "REPO", root), \
                    patch.object(S, "BOUND", bound), patch.object(S, "no_v3_attempt"), \
                    patch.object(S, "load_module", return_value=SimpleNamespace(check=lambda: protocol)) as loader:
                self.assertEqual(S.source_check(), protocol)
                loader.reset_mock()
                (source / "setup_attempt_1/geometry.json").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    S.source_check()
                loader.assert_not_called()

    def test_budget_arithmetic_preserves_total_and_old_work(self):
        r = S.RULES
        self.assertEqual(15 * (1 + 100 - 29), r["plant_call_cap"])
        self.assertEqual(r["previous_charged_seconds"] + r["phase_charge_seconds"], 1890)
        self.assertEqual(1890 + r["remaining_reserved_seconds"], 14400)
        self.assertEqual(189042 + 1080 + 70678 + 237600, 498400)
        self.assertEqual(r["external_timeout_seconds"] + r["kill_grace_seconds"], r["manual_execution_charge_seconds"])


if __name__ == "__main__":
    unittest.main()
