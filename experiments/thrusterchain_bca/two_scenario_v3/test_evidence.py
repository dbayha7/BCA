"""Fake-plant and synthetic-array fixtures only; no canonical dynamics imports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[0] / "two_scenario_v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = load("_v3_evidence_fixture", HERE / "evidence.py")
T = load("_v3_evidence_task_fixture", OLD / "task.py")
D = load("_v3_evidence_data_fixture", OLD / "data.py")


def protocol():
    return json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def point(x, y=0.):
    p = np.zeros(13, np.float64)
    p[:2] = x, y
    return p


def geometry():
    return {"A": point(0).tolist(), "B": point(25).tolist(), "C": point(30, 5).tolist(),
            "goal": {"center": point(40, 15)[:10].tolist(), "radius": .2},
            "danger": {"center": point(26, -1)[:10].tolist(), "radius": .2},
            "connector": {"kind": "capsule", "start": point(26.2, 1.2)[:10].tolist(),
                          "end": point(28.8, 3.8)[:10].tolist(), "radius": .1}}


class FakeCounter:
    def __init__(self, plant):
        self.plant, self.calls, self.starts = plant, 0, 0

    def initialize(self):
        if self.starts >= 46:
            raise ValueError("Fixture initialization cap")
        self.starts += 1

    def __call__(self, state, action):
        if self.calls >= 4502:
            raise ValueError("Fixture physical-call cap")
        self.calls += 1
        return self.plant(state, action)


def fake_plant(state, action):
    x = state.copy()
    x[0] += 1
    if round(float(state[0])) >= 25:
        x[1] += action[0]
    return x


def control(state, t):
    return np.array([1., 0., 0., 0.], np.float32)


class GeometryTests(unittest.TestCase):
    def run_case(self, *, plant=fake_plant, geom=None, task_class=T.Task, controller=control):
        g = geometry() if geom is None else geom
        c = FakeCounter(plant)
        env = task_class(c, g)
        return E.geometry_evidence(T, env, controller, g, protocol(), c), c

    def test_exact_shared_prefix_single_opposed_action_and_parity(self):
        r, c = self.run_case()
        self.assertTrue(r["passed"], r["checks"])
        self.assertEqual(r["danger_entry_count"], 15)
        self.assertEqual(r["safe_goal_count"], 15)
        self.assertEqual(c.starts, 46)
        self.assertEqual(c.calls, sum(len(v["trace"]) for case in r["cases"] for v in case["variants"].values()) + 2)
        self.assertLessEqual(c.calls, 4502)
        self.assertGreaterEqual(r["nominal_from_B_gaps"]["danger"], 5.)
        self.assertGreaterEqual(r["nominal_from_B_gaps"]["wait"], 5.)
        for case in r["cases"]:
            variants = case["variants"]
            self.assertTrue(case["same_prefix_exact"])
            self.assertTrue(case["same_B_state_exact"])
            self.assertEqual(variants["danger"]["terminal_time"], 26)
            self.assertEqual(len(variants["danger"]["trace"]), 26)
            self.assertEqual(variants["danger"]["trace"][25]["action"], [-1., 0., 0., 0.])
            self.assertFalse(variants["safe"]["trace"][25]["terminated"])

    def test_later_danger_is_not_an_immediate_access_pass(self):
        g = geometry()
        # One opposite action at25 then nominal control reaches this at t28.
        g["danger"]["center"] = point(28, 1)[:10].tolist()
        r, _ = self.run_case(geom=g)
        self.assertFalse(r["passed"])
        self.assertEqual(r["danger_entry_count"], 0)
        self.assertFalse(r["checks"]["nominal_danger_first_action"])
        self.assertEqual(r["cases"][0]["variants"]["danger"]["terminal_time"], 28)
        trace = r["cases"][0]["variants"]["danger"]["trace"]
        self.assertEqual([row["time"] for row in trace if row["action"][0] < 0], [25])

    def test_unmatched_prefix_fails_even_if_B_states_happen_to_match(self):
        calls = {"t5": 0}
        def nondeterministic_controller(state, t):
            a = control(state, t)
            if t == 5:
                calls["t5"] += 1
                a[0] = 1. if calls["t5"] % 3 == 1 else 0.
            return a
        r, _ = self.run_case(controller=nondeterministic_controller)
        self.assertFalse(r["checks"]["all_families_identical_prefix_and_B"])
        self.assertTrue(r["cases"][0]["same_B_state_exact"])
        self.assertFalse(r["passed"])

    def test_wrong_geometry_B_does_not_pass_using_an_unmatched_restore(self):
        g = geometry()
        g["B"][2] = .1
        r, _ = self.run_case(geom=g)
        self.assertFalse(r["checks"]["nominal_B_equals_geometry_B"])
        self.assertFalse(r["passed"])

    def test_snapshot_mismatch_fails(self):
        class WrongRestore(T.Task):
            def step_snapshot(self, snapshot, action):
                restored, reward, info = super().step_snapshot(snapshot, action)
                return restored, reward + .01, info
        r, c = self.run_case(task_class=WrongRestore)
        self.assertFalse(r["checks"]["exact_snapshot_direct_parity"])
        self.assertFalse(r["passed"])
        self.assertEqual(c.starts, 46)

    def test_every_nonterminal_witness_stays_within_maximum(self):
        g = geometry()
        g["goal"]["center"] = point(1000)[:10].tolist()
        g["danger"]["center"] = point(1000, -1000)[:10].tolist()
        r, c = self.run_case(geom=g)
        self.assertFalse(r["passed"])
        self.assertEqual(c.calls, 4502)
        self.assertEqual(c.starts, 46)


def synthetic_arrays():
    """Explicit small raw-parent fixtures, not production collection evidence.

    Eight safe fitting parents have a retained prefix and goal suffix around a
    removed connector edge; eight others terminate in danger after the B row.
    128 separate calibration parents remain eligible under both masks.
    """
    rows = {key: [] for key in D.ARRAY_FIELDS}
    for parent in range(144):
        danger = 8 <= parent < 16
        role = D.FIT if parent < 16 else D.CALIBRATION
        times = range(23, 26 if danger else 32)
        for t in times:
            state, nxt = point(t), point(t + 1)
            done = t == (25 if danger else 31)
            if danger and t == 25:
                nxt[1] = 10.
            values = {"states": state, "next_states": nxt,
                "observations": T.observe(state, t), "next_observations": T.observe(nxt, t + 1),
                "actions": np.zeros(4, np.float32), "rewards": -101. if danger and done else 99. if done else -1.,
                "terminals": done, "episode_ids": parent, "row_ids": len(rows["row_ids"]),
                "times": t, "roles": role, "fragment_ids": parent, "families": parent % 4}
            for key, value in values.items():
                rows[key].append(value)
    arrays = {}
    for key in D.ARRAY_FIELDS:
        dtype = np.float64 if key in ("states", "next_states", "rewards") else np.float32 if key in ("observations", "next_observations", "actions") else bool if key == "terminals" else np.int64
        arrays[key] = np.asarray(rows[key], dtype=dtype)
    g = {"B": point(25).tolist(), "C": point(30).tolist(),
         "danger": {"center": point(26, 10)[:10].tolist(), "radius": .1},
         "connector": {"kind": "capsule", "start": point(27.2)[:10].tolist(),
                       "end": point(27.8)[:10].tolist(), "radius": .01}}
    D.validate_arrays(arrays, require_all_parents=False)
    return arrays, g


class SupportTests(unittest.TestCase):
    def test_retained_B25_only_and_real_connector_fragments(self):
        arrays, geom = synthetic_arrays()
        r = E.support_evidence(D, T, arrays, geom, protocol())
        self.assertTrue(all(x["passed"] for x in r.values()), r)
        self.assertEqual(r["honeypot"]["approach_query_times"], {"B": 25})
        self.assertEqual(r["honeypot"]["approach_distinct_fitting_parents"], {"B": 8})
        self.assertEqual(r["bottleneck"]["approach_query_times"], {"B": 25, "C": 30})
        self.assertEqual(r["bottleneck"]["retained_before_after_connector_goal_parent_ids"], list(range(8)))
        for result in r.values():
            self.assertEqual(result["calibration_eligible_parents"], 128)
            self.assertEqual(result["forbidden_retained_edges"], 0)
            self.assertEqual(result["removed_fitting_parents"], 8)

    def test_t29_or_removed_rows_cannot_rescue_insufficient_B25_support(self):
        arrays, geom = synthetic_arrays()
        row = np.flatnonzero((arrays["episode_ids"] == 0) & (arrays["times"] == 25))[0]
        arrays["states"][row, 2] = .2
        arrays["next_states"][row - 1] = arrays["states"][row]
        arrays["observations"][row] = T.observe(arrays["states"][row], 25)
        arrays["next_observations"][row - 1] = arrays["observations"][row]
        D.validate_arrays(arrays, require_all_parents=False)
        r = E.support_evidence(D, T, arrays, geom, protocol())
        self.assertEqual(r["honeypot"]["approach_distinct_fitting_parents"]["B"], 7)
        self.assertFalse(r["honeypot"]["passed"])
        self.assertFalse(r["honeypot"]["checks"]["retained_fitting_approach_support"])

    def test_calibration_parents_not_rows_and_no_fitting_backfill(self):
        arrays, geom = synthetic_arrays()
        arrays["roles"][arrays["episode_ids"] == 143] = D.RESERVED
        r = E.support_evidence(D, T, arrays, geom, protocol())
        self.assertTrue(all(x["calibration_eligible_parents"] == 127 for x in r.values()))
        self.assertTrue(all(not x["passed"] for x in r.values()))

    def test_forged_next_action_or_terminal_label_fails(self):
        arrays, geom = synthetic_arrays()
        for corruption in ("next_actions", "terminals", "next_states", "fragment_ids"):
            def bad_view(a, g, condition):
                view = {k: v.copy() for k, v in D.condition_arrays(a, g, condition).items()}
                if corruption == "next_actions":
                    view["next_actions"] = view["actions"].copy()
                elif corruption == "terminals":
                    view["terminals"][0] = not view["terminals"][0]
                else:
                    view[corruption][0] += 1
                return view
            fake_D = SimpleNamespace(**{name: getattr(D, name) for name in
                ("FIT", "CALIBRATION", "ARRAY_FIELDS", "retained_mask", "fragment_ids_for_mask")}, condition_arrays=bad_view)
            with self.subTest(corruption=corruption):
                r = E.support_evidence(fake_D, T, arrays, geom, protocol())
                self.assertTrue(all(not x["checks"]["original_retained_transition_lineage"] for x in r.values()))
                self.assertTrue(all(not x["passed"] for x in r.values()))

    def test_missing_goal_suffix_is_not_stitching_evidence(self):
        arrays, geom = synthetic_arrays()
        arrays["rewards"][(arrays["episode_ids"] == 0) & arrays["terminals"]] = -1.
        r = E.support_evidence(D, T, arrays, geom, protocol())
        self.assertEqual(len(r["bottleneck"]["retained_before_after_connector_goal_parent_ids"]), 7)
        self.assertFalse(r["bottleneck"]["passed"])


if __name__ == "__main__":
    unittest.main()
