"""Synthetic tasks/policies only; production physics and model calls are absent."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load("_smoke_evaluation_fixture", HERE / "smoke_evaluate.py")
E = load("_smoke_sealed_evaluation_fixture", HERE / "evaluate.py")
R, T = E._source_modules()


def point(t, y=0):
    x = np.zeros(13, np.float64); x[0], x[1] = t, y
    return x


def fixtures(*, goal=True):
    geometry = {"A": point(0).tolist(), "B": point(25).tolist(), "C": point(30, 10).tolist(),
                "danger": {"center": point(30, -1)[:10].tolist(), "radius": .15},
                "goal": {"center": point(40 if goal else 1000)[:10].tolist(), "radius": .2},
                "connector": {"center": point(1000)[:10].tolist(), "radius": .01}}
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    known = {"state": point(29).tolist(), "time": 29, "action": [-1., 0., 0., 0.],
             "next_state": point(30, -1).tolist(), "reward": -101.01, "terminated": True, "reason": "danger"}
    return geometry, protocol, known


def controller(state, time):
    return np.array([1., 0., 0., 0.], np.float32)


class Counter:
    def __init__(self):
        self.phase = None
        self.attempted = dict.fromkeys(S.PHASE_CAPS, 0)
        self.completed = dict.fromkeys(S.PHASE_CAPS, 0)

    def begin(self, phase):
        if phase not in S.PHASE_CAPS: raise ValueError("Fake phase mismatch")
        self.phase = phase

    def __call__(self, state, action):
        if self.attempted[self.phase] >= S.PHASE_CAPS[self.phase] or sum(self.attempted.values()) >= 3320:
            raise ValueError("Fake physical cap")
        self.attempted[self.phase] += 1
        x = state.copy(); x[0] += 1
        clock = round(float(state[0]))
        x[1] = 3. * float(action[0]) if clock == 25 else float(action[0]) if clock == 29 else 0.
        self.completed[self.phase] += 1
        return x

    def receipt(self):
        return {"attempted": dict(self.attempted), "completed": dict(self.completed),
                "execution_counts_known": self.attempted == self.completed}


def policies(calls):
    result = {}
    for arm in ("native", "bca"):
        def actor(obs, arm=arm):
            calls.append((arm, "actor", len(obs)))
            return np.tile(np.array([.2 if arm == "native" else .4, 0., 0., 0.], np.float32), (len(obs), 1))
        def q1(obs, actions, arm=arm):
            calls.append((arm, "q1", obs.copy(), actions.copy()))
            return np.full(len(obs), -1000. if arm == "bca" else 1000., np.float64)
        def width(obs, actions, arm=arm):
            calls.append((arm, "width", obs.copy(), actions.copy()))
            return np.full(len(obs), 50., np.float64)
        result[arm] = {"actor": actor, "q1": q1, "width": width}
    return result


class SmokeFixtures(unittest.TestCase):
    def run_smoke(self, *, goal=True, labels=None, known_mutator=None, persist_hook=None):
        geom, protocol, known = fixtures(goal=goal)
        if known_mutator: known_mutator(known)
        counter, records, calls = Counter(), {}, []
        def persist(name, value):
            self.assertNotIn(name, records)
            if persist_hook: persist_hook(name, value)
            records[name] = value
        with patch.object(S, "_sealed_inputs", return_value=(E, R, T, known)):
            if labels is None:
                result = S.check_and_evaluate(T.Task(counter, geom), controller, geom, protocol,
                                              policies(calls), counter, persist)
            else:
                with patch.object(E, "support_labels", side_effect=labels):
                    result = S.check_and_evaluate(T.Task(counter, geom), controller, geom, protocol,
                                                  policies(calls), counter, persist)
        return result, counter, records, calls

    def test_all_frozen_banks_actual_edges_and_inference_path(self):
        result, counter, records, calls = self.run_smoke()
        self.assertTrue(result["passed"], result)
        self.assertEqual((result["final_banks"], result["anchors"], result["common_edges"]), (6, 96, 288))
        self.assertEqual(counter.completed["prefix_parity"], 2688)
        self.assertEqual(counter.completed["bank_edges"], 288)
        self.assertEqual(counter.completed["forced_qpi"], 24)
        self.assertEqual(counter.completed["returns"], 80)
        self.assertEqual(sum(counter.completed.values()), 3080)
        self.assertEqual(len([k for k in records if "_edge_" in k]), 288)
        for bank in result["bank_coverage"]:
            self.assertEqual(bank["common_edges"], 48)
            self.assertGreaterEqual(bank["omitted_count"], 1)
            self.assertGreaterEqual(bank["retained_count"], 1)
        for arm in ("native", "bca"):
            report = records["smoke_" + arm + "_evaluation"]
            self.assertEqual(report["bad_reference"]["counts"]["transitions_completed"], 1)
            self.assertTrue(report["bad_reference"]["exact_saved_bad_replay"])
            self.assertEqual(report["bad_reference"]["events"][0]["state"], report["safe_reference"]["events"][0]["state"])
            self.assertEqual(report["safe_reference"]["events"][0]["time"], 29)
            self.assertFalse(report["efficacy_acceptance_applied"])
            self.assertEqual(sum(c[0] == arm and c[1] == "q1" for c in calls), 1)
            self.assertEqual(sum(c[0] == arm and c[1] == "width" for c in calls), 1)
            q_call = next(c for c in calls if c[0] == arm and c[1] == "q1")
            self.assertEqual(q_call[2].shape, (2, 14))
            np.testing.assert_array_equal(q_call[3], [[-1., 0., 0., 0.], [1., 0., 0., 0.]])
        self.assertTrue(result["policy_performance_is_not_a_smoke_pass_criterion"])

    def test_failed_coverage_keeps_every_bank_and_no_adaptive_queries(self):
        original, calls = E.support_labels, []
        def labels(*args):
            calls.append(1)
            return np.zeros((16, 3), bool) if len(calls) == 3 else original(*args)
        result, counter, records, _ = self.run_smoke(labels=labels)
        self.assertFalse(result["passed"])
        self.assertEqual(len(calls), 6)
        self.assertEqual(result["final_banks"], 6)
        self.assertEqual(result["completed_disposable_arms"], 2)
        self.assertEqual(result["bank_coverage"][2]["omitted_count"], 0)
        self.assertEqual(counter.completed["bank_edges"], 288)
        self.assertIn("smoke_honeypot_seed1_coverage", records)
        self.assertIn("smoke_evaluation_report", records)

    def test_horizon_outcomes_and_huge_q_errors_do_not_determine_pass(self):
        result, counter, records, _ = self.run_smoke(goal=False)
        self.assertTrue(result["passed"])
        self.assertEqual(counter.completed, S.PHASE_CAPS)
        self.assertEqual(sum(counter.completed.values()), 3320)
        for arm in ("native", "bca"):
            record = records["smoke_" + arm + "_evaluation"]
            self.assertEqual(record["free_final_time"], 100)
            self.assertEqual(record["safe_reference"]["final_times"], [100])
            self.assertEqual(record["safe_reference"]["events"][-1]["info"]["terminal_reason"], "horizon")
        with self.assertRaisesRegex(ValueError, "cap"):
            counter(point(0), np.zeros(4, np.float32))
        self.assertEqual(sum(counter.attempted.values()), 3320)

    def test_own_policy_returns_use_production_discount_and_early_termination(self):
        _, _, records, _ = self.run_smoke()
        for arm in ("native", "bca"):
            result = records["smoke_" + arm + "_evaluation"]
            expected = sum(.98 ** j * event["reward"] for j, event in enumerate(result["safe_reference"]["events"]))
            self.assertAlmostEqual(expected, result["own_policy_qpi"][1], places=12)
            self.assertEqual(result["own_policy_qpi"][0], -101.01)
            self.assertEqual(len(result["bad_reference"]["events"]), 1)

    def test_bad_replay_mismatch_is_saved_and_cannot_pass(self):
        geom, protocol, known = fixtures()
        known["reward"] += .1
        counter, records = Counter(), {}
        with patch.object(S, "_sealed_inputs", return_value=(E, R, T, known)):
            with self.assertRaisesRegex(ValueError, "exact replay"):
                S.check_and_evaluate(T.Task(counter, geom), controller, geom, protocol,
                                      policies([]), counter, records.__setitem__)
        self.assertFalse(records["smoke_native_bad_reference"]["exact_saved_bad_replay"])
        self.assertEqual(sum(counter.completed.values()), 2977)
        self.assertEqual(records["smoke_evaluation_failure"]["completed_disposable_arms"], 0)

    def test_persistence_failure_keeps_actual_physical_counts(self):
        geom, protocol, known = fixtures()
        counter, records = Counter(), {}
        def persist(name, value):
            if name.endswith("_edge_00_0"): raise OSError("fixture persistence failure")
            records[name] = value
        with patch.object(S, "_sealed_inputs", return_value=(E, R, T, known)):
            with self.assertRaisesRegex(OSError, "persistence"):
                S.check_and_evaluate(T.Task(counter, geom), controller, geom, protocol, policies([]), counter, persist)
        self.assertEqual(counter.completed["prefix_parity"], 480)
        self.assertEqual(counter.completed["bank_edges"], 1)
        self.assertEqual(records["smoke_evaluation_failure"]["counts"]["completed"], counter.completed)

    def test_callback_shape_failure_cannot_be_labelled_a_complete_smoke(self):
        geom, protocol, known = fixtures()
        counter, records, p = Counter(), {}, policies([])
        p["native"]["width"] = lambda obs, actions: np.zeros((2, 1), np.float64)
        with patch.object(S, "_sealed_inputs", return_value=(E, R, T, known)):
            with self.assertRaisesRegex(ValueError, "callback has invalid"):
                S.check_and_evaluate(T.Task(counter, geom), controller, geom, protocol, p, counter, records.__setitem__)
        self.assertEqual(sum(counter.completed.values()), 2976)
        self.assertFalse(records["smoke_evaluation_failure"]["passed"])

    def test_unknown_physical_accounting_cannot_pass(self):
        class UnknownCounter(Counter):
            def receipt(self):
                result = super().receipt()
                result["execution_counts_known"] = False
                return result
        geom, protocol, known = fixtures()
        counter, records = UnknownCounter(), {}
        with patch.object(S, "_sealed_inputs", return_value=(E, R, T, known)):
            with self.assertRaisesRegex(ValueError, "physical smoke accounting"):
                S.check_and_evaluate(T.Task(counter, geom), controller, geom, protocol,
                                      policies([]), counter, records.__setitem__)
        self.assertEqual(records["smoke_evaluation_failure"]["completed_disposable_arms"], 2)
        self.assertFalse(records["smoke_evaluation_failure"]["passed"])


if __name__ == "__main__":
    unittest.main()
