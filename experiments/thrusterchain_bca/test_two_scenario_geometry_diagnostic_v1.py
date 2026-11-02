"""Explicit fake-only tests for the bounded geometry diagnostic.

No test imports canonical dynamics, constructs the real plant, or runs a model.
The only trajectories below are arithmetic toy fixtures with two coordinates.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = load_file("_two_scenario_geometry_diagnostic_test", HERE / "two_scenario_geometry_diagnostic_v1.py")
T = load_file("_two_scenario_geometry_task_test", HERE / "two_scenario_v1" / "task.py")


def recipe():
    return {"initial_state": [0.0] * 13, "b_time": 25, "c_time": 30,
            "goal_time": 75, "goal_radius": .1, "danger_radius_max": .1,
            "danger_radius_fraction": .25, "minimum_region_radius": .01,
            "connector_trim_fraction": .2, "connector_radius_fraction": .1}


def point(first, second, third=0.0):
    x = np.zeros(13, np.float64)
    x[:3] = [first, second, third]
    return x


def fake_route():
    return [point(t / 10, t / 2) for t in range(101)]


def trace_fixture(*, danger_only=False, connector_only=False, nonfinite=False):
    route = fake_route()
    if connector_only:
        route[30] = route[25].copy()
        route[30][0] += .001
    if nonfinite:
        route[30][0] = float("nan")
    witness = [route[25].copy()]
    for j in range(1, 6):
        x = point(2.5 + j / 10, 12.5 - j / 2)
        if danger_only:
            x = route[25 + j].copy()
        if connector_only:
            x[2] = j / 5
        witness.append(x)
    trace = []
    for t in range(100):
        trace.append({"phase": "nominal", "state": route[t].tolist(), "time": t,
                      "action": [.5, 0, 0, 0], "next_state": route[t + 1].tolist()})
    for j in range(5):
        trace.append({"phase": "excursion", "state": witness[j].tolist(), "time": 25 + j,
                      "action": [-.5, 0, 0, 0], "next_state": witness[j + 1].tolist()})
    return trace


class FakePlant:
    def __init__(self):
        self.calls = 0

    def __call__(self, state, action):
        self.calls += 1
        x = state.copy()
        x[0] += .1
        x[1] += float(action[0])
        return x, 0.0


def fake_controller(state, time):
    return np.array([.5, 0, 0, 0], np.float32)


class GeometryDiagnosticTests(unittest.TestCase):
    def test_danger_only_failure_names_its_criterion(self):
        got = G.measurements(T, trace_fixture(danger_only=True), recipe())
        self.assertEqual(got["values"]["danger_radius"], 0.0)
        self.assertFalse(got["checks"]["danger_radius_minimum"])
        self.assertTrue(got["checks"]["connector_radius_minimum"])
        self.assertIn("danger_radius_minimum", got["failed_criteria"])
        self.assertNotIn("connector_radius_minimum", got["failed_criteria"])

    def test_connector_only_failure_names_its_criterion(self):
        got = G.measurements(T, trace_fixture(connector_only=True), recipe())
        self.assertLess(got["values"]["connector_radius"], .01)
        self.assertFalse(got["checks"]["connector_radius_minimum"])
        self.assertTrue(got["checks"]["danger_radius_minimum"])
        self.assertIn("connector_radius_minimum", got["failed_criteria"])
        self.assertNotIn("danger_radius_minimum", got["failed_criteria"])

    def test_nonfinite_measurement_is_not_a_success(self):
        got = G.measurements(T, trace_fixture(nonfinite=True), recipe())
        self.assertFalse(got["checks"]["connector_radius_finite"])
        self.assertIn("connector_radius_finite", got["failed_criteria"])
        self.assertTrue(all(value is None or isinstance(value, (bool, np.bool_)) for value in got["checks"].values()))
        self.assertFalse(all(value is True for value in got["checks"].values()))

    def test_persist_is_exclusive_and_nonfinite_json_is_tagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            G.persist(path, {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")})
            raw = path.read_text()
            decoded = json.loads(raw, parse_constant=lambda value: self.fail("Non-standard JSON number " + value))
            self.assertTrue(all(isinstance(value, str) for value in decoded.values()))
            with self.assertRaises(FileExistsError):
                G.persist(path, {"replacement": True})
            self.assertEqual(path.read_text(), raw)

    def test_successful_fake_original_construction_measurement_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plant = Path(tmp), FakePlant()
            capture = G.Capture(T, plant, recipe(), root, lambda: None)
            original = T.construct_geometry(capture, fake_controller, recipe())
            evidence = json.loads((root / "measurements.json").read_text())
            self.assertEqual(plant.calls, 105)
            self.assertEqual(evidence["failed_criteria"], [])
            self.assertTrue(all(value is True for value in evidence["checks"].values()))
            for name in ("goal", "danger", "connector"):
                self.assertEqual(evidence["geometry"][name], original[name])
            counts = capture.counts()
            self.assertEqual(counts["attempted"], 105)
            self.assertEqual(counts["completed"], 105)
            self.assertEqual(counts["initializations"], 2)

    def test_measurements_saved_before_original_geometry_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plant = Path(tmp), FakePlant()
            frozen = recipe(); frozen["minimum_region_radius"] = 100.0
            capture = G.Capture(T, plant, frozen, root, lambda: None)
            with self.assertRaisesRegex(ValueError, "Frozen geometry is degenerate"):
                T.construct_geometry(capture, fake_controller, frozen)
            evidence = json.loads((root / "measurements.json").read_text())
            self.assertIn("danger_radius_minimum", evidence["failed_criteria"])
            self.assertIn("connector_radius_minimum", evidence["failed_criteria"])
            self.assertEqual(capture.counts()["completed"], 105)

    def test_measurement_write_failure_preserves_completed_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plant = Path(tmp), FakePlant()
            capture = G.Capture(T, plant, recipe(), root, lambda: None)
            real_persist = G.persist
            def failed_save(path, value):
                if Path(path).name == "measurements.json":
                    raise OSError("Explicit fixture disk failure")
                return real_persist(path, value)
            with mock.patch.object(G, "persist", side_effect=failed_save):
                with self.assertRaises(Exception):
                    T.construct_geometry(capture, fake_controller, recipe())
            self.assertEqual(capture.counts()["attempted"], 105)
            self.assertEqual(capture.counts()["completed"], 105)
            self.assertEqual(plant.calls, 105)
            self.assertIn("Explicit fixture disk failure", str(capture.measurements_persist_error))
            self.assertFalse((root / "measurements.json").exists())

    def test_call_106_is_rejected_before_plant(self):
        with tempfile.TemporaryDirectory() as tmp:
            plant = FakePlant()
            capture = G.Capture(T, plant, recipe(), Path(tmp), lambda: None)
            T.construct_geometry(capture, fake_controller, recipe())
            with self.assertRaises(Exception):
                capture(point(0, 0), np.zeros(4, np.float32))
            self.assertEqual(plant.calls, 105)
            self.assertEqual(capture.counts()["completed"], 105)

    def test_plant_failure_keeps_attempted_and_completed_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            plant = mock.Mock(side_effect=RuntimeError("Explicit fixture plant failure"))
            capture = G.Capture(T, plant, recipe(), Path(tmp), lambda: None)
            with self.assertRaisesRegex(RuntimeError, "Explicit fixture plant failure"):
                capture(point(0, 0), np.zeros(4, np.float32))
            self.assertEqual(capture.counts()["attempted"], 1)
            self.assertEqual(capture.counts()["completed"], 0)


if __name__ == "__main__":
    unittest.main()
