"""Fake policies and injected fake dynamics only; no canonical plant or JAX."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_path_a_evaluation_fixture", HERE / "evaluate.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
OLD, T = E._source_modules()


def protocol():
    p = json.loads((E.V2 / "protocol.json").read_text(encoding="utf-8"))
    p.update(condition_anchor_times=dict(E.ANCHOR_TIMES), condition_anchor_recipes=dict(E.ANCHOR_RECIPES),
             common_actions=list(E.COMMON_ACTIONS))
    return p


def point(x, y=0.):
    value = np.zeros(13, np.float64)
    value[0], value[1] = x, y
    return value


def geometry(goal=31.):
    return {"A": point(0).tolist(), "B": point(25).tolist(), "C": point(30).tolist(),
            "danger": {"center": point(1000)[:10].tolist(), "radius": .1},
            "goal": {"center": point(goal)[:10].tolist(), "radius": .2},
            "connector": {"kind": "capsule", "start": point(26)[:10].tolist(),
                          "end": point(29)[:10].tolist(), "radius": .01}}


def controller(state, t):
    return np.array([1., 0., 0., 0.], np.float32)


class FakeCounter:
    def __init__(self, plant=None):
        self.phase = "prefix_parity"
        self.attempted = dict.fromkeys(E.PHASE_CAPS, 0)
        self.completed = dict.fromkeys(E.PHASE_CAPS, 0)
        self.plant = plant or self.default_plant

    @staticmethod
    def default_plant(state, action):
        nxt = state.copy()
        nxt[0] += 1
        return nxt

    def begin(self, phase):
        if phase not in E.PHASE_CAPS: raise ValueError("Unknown fake-meter phase")
        self.phase = phase

    def __call__(self, state, action):
        if self.attempted[self.phase] >= E.PHASE_CAPS[self.phase] or sum(self.attempted.values()) >= 237600:
            raise ValueError("Fake meter cap reached")
        self.attempted[self.phase] += 1
        value = self.plant(state, action)
        self.completed[self.phase] += 1
        return value

    def receipt(self):
        return {"attempted": dict(self.attempted), "completed": dict(self.completed),
                "execution_counts_known": self.attempted == self.completed}


def policies():
    result = {}
    for seed in range(3):
        for condition in E.ANCHOR_TIMES:
            for arm in ("native", "bca"):
                value = .2 if arm == "native" else .4
                def actor(obs, value=value):
                    return np.tile(np.array([value, 0., 0., 0.], np.float32), (len(obs), 1))
                result[(seed, condition, arm)] = {
                    "actor": actor, "q1": lambda obs, actions: np.asarray(actions[:, 0], np.float64),
                    "width": lambda obs, actions: np.full(len(obs), .2, np.float64),
                    "identity": {"seed_index": seed, "condition": condition, "arm": arm,
                        "checkpoint_sha256": ("a" if arm == "native" else "b") * 64,
                        "normalization_sha256": "c" * 64, "dataset_sha256": "d" * 64,
                        "cumulative_critic_updates": 20000, "training_kappa": 0. if arm == "native" else 3.}}
    return result


class EvaluationFixtures(unittest.TestCase):
    def test_reserved_and_tight_arithmetic(self):
        p = E.plan(protocol())
        self.assertEqual(p["tight_phase_upper_bounds"], {"returns": 153600, "own_qpi": 56064,
                         "common_reference": 21024, "prefix_parity": 2688})
        self.assertEqual(p["tight_transition_upper_bound"], 233376)
        self.assertEqual(p["reserved_transition_cap"], 237600)
        self.assertEqual(p["own_qpi_references"], 768)
        bad = protocol(); bad["condition_anchor_times"]["honeypot"] = 25
        with self.assertRaisesRegex(ValueError, "setting differs"):
            E.plan(bad)
        bad = protocol(); bad["evaluation_seeds"][0]["returns"] = bad["collection"]["collection_seed"]
        with self.assertRaisesRegex(ValueError, "overlap collection"):
            E.plan(bad)

    def test_paired_independent_start_draws_preserve_full_state(self):
        p, g = protocol(), geometry()
        a = E.start_bank(g, p["collection"], 123, 128)
        self.assertEqual(a.dtype, np.float64)
        self.assertEqual(a.shape, (128, 13))
        self.assertTrue(np.array_equal(a, E.start_bank(g, p["collection"], 123, 128)))
        self.assertFalse(np.array_equal(a, E.start_bank(g, p["collection"], 124, 128)))
        self.assertTrue(np.all((a[:, 10] >= .015) & (a[:, 10] <= .025)))

    def test_honeypot_edges_and_slab_labels_do_not_use_capsule(self):
        g = geometry()
        g["danger"] = {"center": point(27)[:10].tolist(), "radius": .1}
        x = np.stack((point(25), point(28), point(27)))
        y = np.stack((point(29), point(29), point(28)))
        np.testing.assert_array_equal(E.support_labels(T, x, y, g, "honeypot"), [True, False, True])
        # A far-off-axis edge still crosses the separating slab; the old capsule does not.
        x, y = np.stack((point(25, 100), point(30, 100))), np.stack((point(30, 100), point(25, 100)))
        self.assertTrue(E.support_labels(T, x, y, g, "bottleneck").all())
        self.assertFalse(T.edge_intersects(x, y, g["connector"]).any())

    def test_prefix_recipe_exact_time_and_single_last_step_replay(self):
        counter, g = FakeCounter(), geometry()
        task = T.Task(counter, g)
        saved = {}
        states, times, traces = E.build_anchors(task, controller, np.stack([point(0), point(0)]), "honeypot", saved.__setitem__)
        self.assertTrue(np.array_equal(times, [29, 29]))
        self.assertEqual(counter.completed["prefix_parity"], 60)
        self.assertTrue(np.array_equal(states[:, 0], [29, 29]))
        for trace in traces:
            self.assertEqual([r["time"] for r in trace if r["action"][0] < 0], [25, 26, 27, 28])
        self.assertTrue(saved["anchor_00"]["last_prefix_step_exact_replay"])

    def test_early_terminal_anchor_is_not_replaced(self):
        counter = FakeCounter(); saved = {}
        with self.assertRaisesRegex(ValueError, "prefix terminated"):
            E.build_anchors(T.Task(counter, geometry(goal=20.)), controller,
                            np.stack([point(0), point(0)]), "honeypot", saved.__setitem__)
        self.assertEqual(counter.completed["prefix_parity"], 20)
        self.assertIn("invalid_anchor_00", saved)
        self.assertNotIn("anchor_01", saved)

    def test_prefix_parity_mismatch_fails(self):
        calls = []
        def changed(state, action):
            calls.append(1)
            nxt = FakeCounter.default_plant(state, action)
            if len(calls) == 26: nxt[2] += .01
            return nxt
        counter, saved = FakeCounter(changed), {}
        with self.assertRaisesRegex(ValueError, "parity failed"):
            E.build_anchors(T.Task(counter, geometry()), controller, np.stack([point(0)]), "bottleneck", saved.__setitem__)
        self.assertFalse(saved["anchor_00"]["last_prefix_step_exact_replay"])

    def test_common_reference_uses_exact_state_and_true_termination(self):
        g, counter = geometry(), FakeCounter()
        counter.begin("common_reference")
        start = point(29); start[4] = 1e-12
        observed = []
        def exact_controller(x, t):
            observed.append((x.copy(), t)); return controller(x, t)
        values, first, count = E.controller_qpi(T.Task(counter, g), exact_controller,
            np.stack([start]), np.array([29], np.int64), np.zeros((1, 1, 4), np.float32))
        self.assertEqual(count, 2)
        self.assertEqual(observed[0][1], 30)
        self.assertEqual(observed[0][0][4], 1e-12)
        self.assertAlmostEqual(values[0, 0], -1. + .98 * 98.99)
        self.assertEqual(first[0, 0, 0], 30.)

    def test_own_actor_slot_is_excluded_from_shared_ranking(self):
        q = np.array([[1., 2., 3., 1000.]])
        u = np.array([[0., 0., 1., 0.]])
        truth = np.array([[4., 5., 6., -1000.]])
        own = OLD.score_diagnostics(q, u, truth, error_tolerance=1., tie_tolerance=1e-10, common_slots=3)
        self.assertEqual(own["common_action_ranking"]["raw_indices"], [2])
        self.assertEqual(own["common_action_ranking"]["adjusted_indices"], [1])
        common = E._common_ranking(q, u, truth[:, :3])
        self.assertEqual(common["raw_indices"], [2])
        self.assertEqual(common["adjusted_indices"], [1])

    def test_signed_harms_and_threshold_boundaries(self):
        q = np.array([[3., -3., 1., -1.]])
        u = np.array([[2., 1., 1., 0.]])
        details = E._signed_detail(q, u, np.zeros_like(q), np.array([[True, True, False, False]]))
        self.assertEqual(details["omitted_edge"]["raw_over"], 1)
        self.assertEqual(details["omitted_edge"]["raw_under"], 1)
        self.assertEqual(details["retained_edge"]["raw_near"], 2)
        self.assertEqual(details["over_to_under"], 1)
        self.assertEqual(details["near_to_under"], 1)
        self.assertEqual(details["existing_underestimation_worsened"], 1)

    def test_unmatched_identity_rejected_before_rollouts(self):
        p = policies(); p[(0, "honeypot", "bca")]["identity"]["normalization_sha256"] = "e" * 64
        counter = FakeCounter()
        with self.assertRaisesRegex(ValueError, "Unmatched pair"):
            E.evaluate_study(T.Task(counter, geometry()), controller, geometry(), protocol(), p, counter, lambda *args: None)
        self.assertEqual(sum(counter.attempted.values()), 0)

    def test_persistence_failure_does_not_fabricate_completion_or_continue(self):
        counter = FakeCounter()
        calls = []
        def fail_bank(name, value):
            calls.append(name)
            if name.endswith("_banks"): raise OSError("fixture disk error")
        with self.assertRaisesRegex(OSError, "disk error"):
            E.evaluate_study(T.Task(counter, geometry()), controller, geometry(), protocol(), policies(), counter, fail_bank)
        self.assertEqual(sum(counter.attempted.values()), 0)
        self.assertIn("evaluation_failure", calls)

    def test_full_fake_matrix_completes_despite_negative_paired_returns(self):
        counter, records = FakeCounter(), {}
        def persist(name, value):
            self.assertNotIn(name, records)
            # Keep the raw probe/paired results and avoid retaining every long return trace twice.
            records[name] = value if name.endswith(("_qpi", "_paired", "_banks")) or name == "evaluation_report" else True
        report = E.evaluate_study(T.Task(counter, geometry()), controller, geometry(), protocol(), policies(), counter, persist)
        self.assertEqual(report["status"], "completed_frozen_evaluation")
        self.assertEqual((report["completed_policy_evaluations"], report["completed_paired_cells"], report["completed_return_episodes"]), (12, 6, 1536))
        self.assertTrue(all(c["paired_returns"]["paired_mean_difference"] < 0 for c in report["cells"]))
        self.assertEqual(report["logical_completed_transitions_by_phase"], counter.completed)
        self.assertEqual(counter.completed["prefix_parity"], 2688)
        self.assertLess(sum(counter.completed.values()), 237600)
        for condition in E.ANCHOR_TIMES:
            native = records[condition + "_seed0_native_qpi"]
            bca = records[condition + "_seed0_bca_qpi"]
            self.assertEqual(native["states"], bca["states"])
            self.assertEqual(native["times"], [E.ANCHOR_TIMES[condition]] * 16)
            np.testing.assert_array_equal(np.asarray(native["actions"])[:, :3], np.asarray(bca["actions"])[:, :3])
            self.assertTrue(native["native_adjusted_score_is_diagnostic_only"])
            self.assertFalse(bca["native_adjusted_score_is_diagnostic_only"])
            self.assertEqual(np.asarray(native["raw_q1"]).shape, (16, 4))
            self.assertEqual(np.asarray(native["omitted_edge"]).shape, (16, 4))


if __name__ == "__main__":
    unittest.main()
