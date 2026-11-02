"""Cheap explicit fake-plant tests; never import or call canonical dynamics."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task as T
import data as D


def state(first=0.0):
    x = np.zeros(13, np.float64)
    x[0] = first
    return x


def sphere(first, radius=.25):
    return {"center": state(first)[:10].tolist(), "radius": radius}


def geometry():
    return {"goal": sphere(10), "danger": sphere(5),
            "connector": {"kind": "capsule", "start": state(2)[:10].tolist(),
                          "end": state(3)[:10].tolist(), "radius": .1}}


def fake_step(x, a):
    nxt = x.copy()
    nxt[0] += float(a[0])
    return nxt, -987654.0


def fixture_arrays():
    # One complete fake episode. A region centered at1.5 removes its middle
    # edge, leaving the later recorded rows as a separate genuine fragment.
    states = np.array([state(i) for i in range(4)])
    nxt = np.array([state(i + 1) for i in range(4)])
    times = np.arange(4, dtype=np.int32)
    return {
        "states": states, "next_states": nxt,
        "observations": np.array([T.observe(x, t) for x, t in zip(states, times)]),
        "next_observations": np.array([T.observe(x, t + 1) for x, t in zip(nxt, times)]),
        "actions": np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)),
        "rewards": np.array([-1.01, -1.01, -1.01, 98.99], np.float64),
        "terminals": np.array([False, False, False, True]),
        "episode_ids": np.zeros(4, np.int64), "row_ids": np.arange(4, dtype=np.int64),
        "times": times, "roles": np.zeros(4, np.int8),
        "fragment_ids": np.zeros(4, np.int64), "families": np.zeros(4, np.int8),
    }


class TaskDataTests(unittest.TestCase):
    def test_import_source_has_no_simulator_or_model_import(self):
        for name in ("task.py", "data.py"):
            tree = ast.parse((HERE / name).read_text())
            imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any(any(s in n for s in ("jax", "flax", "canonical", "worlds", "runtime")) for n in imported))

    def test_file_loader_uses_exact_sibling_task(self):
        spec = importlib.util.spec_from_file_location("fake_file_loaded_data_v1", HERE / "data.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"task": mock.Mock()}):
            spec.loader.exec_module(module)
        self.assertEqual(Path(module.T.__file__).resolve(), (HERE / "task.py").resolve())

    def test_full_state_and_absolute_time_restore(self):
        env = T.Task(fake_step, geometry())
        x = state(1)
        x[10:] = [.02, .01, -.03]
        snap = env.restore(x, 37, terminated=False)
        x[0] = 99
        self.assertEqual(snap.state[0], 1)
        self.assertEqual(snap.time, 37)
        np.testing.assert_array_equal(env.observe(snap.state, snap.time)[:13], snap.state.astype(np.float32))
        self.assertEqual(env.observe(snap.state, snap.time)[13], np.float32(.63))

    def test_goal_reward_and_true_terminal(self):
        env = T.Task(fake_step, geometry())
        nxt, reward, done, info = env.step(state(9), 60, [1, 0, 0, 0])
        self.assertEqual(reward, 98.99)
        self.assertTrue(done)
        self.assertTrue(info["goal"])
        self.assertEqual(nxt[0], 10)
        with self.assertRaises(ValueError):
            env.step(nxt, 61, [0, 0, 0, 0])

    def test_hazard_crossing_with_both_endpoints_outside(self):
        g = geometry()
        g["danger"] = sphere(.5, .1)
        env = T.Task(fake_step, g)
        self.assertFalse(T.contains(state(0), g["danger"]))
        self.assertFalse(T.contains(state(1), g["danger"]))
        _, reward, done, info = env.step(state(0), 0, [1, 0, 0, 0])
        self.assertEqual(reward, -101.01)
        self.assertTrue(done and info["danger"])

    def test_danger_priority_over_goal(self):
        g = geometry(); g["danger"] = sphere(9.5, .1)
        _, reward, done, info = T.Task(fake_step, g).step(state(9), 0, [1, 0, 0, 0])
        self.assertTrue(done and info["danger"])
        self.assertFalse(info["goal"])
        self.assertLess(reward, -100)

    def test_crossing_snapshot_preserves_absorbing_status(self):
        g = geometry(); g["danger"] = sphere(.5, .1)
        env = T.Task(fake_step, g)
        live = env.restore(state(0), 20, terminated=False)
        stopped, _, _ = env.step_snapshot(live, [1, 0, 0, 0])
        self.assertFalse(T.contains(stopped.state, g["danger"]))
        self.assertTrue(stopped.terminated)
        restored = env.restore(stopped.state, stopped.time, terminated=stopped.terminated,
                               terminal_reason=stopped.terminal_reason)
        with self.assertRaises(ValueError):
            env.step_snapshot(restored, [0, 0, 0, 0])
        with self.assertRaises(TypeError):
            env.restore(stopped.state, stopped.time)

    def test_no_early_time_only_goal_and_real_horizon(self):
        env = T.Task(fake_step, geometry())
        for t in (0, 25, 30, 75, 98):
            _, _, done, info = env.step(state(0), t, [0, 0, 0, 0])
            self.assertFalse(done)
        self.assertTrue(env.step(state(0), 99, [0, 0, 0, 0])[2])
        with self.assertRaises(ValueError):
            env.step(state(0), 100, [0, 0, 0, 0])

    def test_goal_danger_independent_of_wave_and_time(self):
        g = geometry()
        for c, w in ((0.0, 0.0), (.5, .9)):
            x = state(10); x[10:] = [c, w, -w]
            self.assertTrue(T.contains(x, g["goal"]))

    def test_action_cast_and_clipping_exact(self):
        calls = []
        def fake(x, a):
            calls.append(a.copy()); return fake_step(x, a)
        T.Task(fake, geometry()).step(state(), 0, [2, -2, .123456789, 0])
        self.assertEqual(calls[0].dtype, np.float32)
        np.testing.assert_array_equal(calls[0], np.array([1, -1, .123456789, 0], np.float32))

    def test_sphere_tangency_and_degenerate_edges(self):
        region = sphere(0, 1)
        a, b = state(-2), state(2); a[1] = b[1] = 1
        self.assertTrue(T.edge_intersects(a, b, region))
        self.assertTrue(T.edge_intersects(state(1), state(1), region))
        self.assertFalse(T.edge_intersects(state(2), state(2), region))

    def test_connector_capsule_crossing_and_boundary(self):
        region = geometry()["connector"]
        self.assertTrue(T.edge_intersects(state(1), state(4), region))
        self.assertTrue(T.contains(state(2), region))
        a, b = state(1), state(4); a[1] = b[1] = .2
        self.assertFalse(T.edge_intersects(a, b, region))
        self.assertFalse(T.contains(state(0), region))

    def test_masks_all_roles_and_no_label_leakage(self):
        a = fixture_arrays()
        g = geometry(); g["danger"] = sphere(1.5, .1)
        baseline = D.retained_mask(a, g, "honeypot")
        np.testing.assert_array_equal(baseline, [True, False, True, True])
        for role in (D.FIT, D.CALIBRATION, D.RESERVED):
            a["roles"][:] = role
            np.testing.assert_array_equal(D.retained_mask(a, g, "honeypot"), baseline)

    def test_fragments_never_join_or_relabel_terminal(self):
        a = fixture_arrays(); g = geometry(); g["danger"] = sphere(1.5, .1)
        got = D.condition_arrays(a, g, "honeypot")
        np.testing.assert_array_equal(got["row_ids"], [0, 2, 3])
        np.testing.assert_array_equal(got["times"], [0, 2, 3])
        np.testing.assert_array_equal(got["fragment_ids"], [0, 1, 1])
        np.testing.assert_array_equal(got["terminals"], [False, False, True])
        self.assertNotIn("next_actions", got)
        self.assertFalse(got["states"].flags.writeable)

    def test_balanced_disjoint_parent_roles_before_outcomes(self):
        roles = D.episode_roles(1234)
        self.assertEqual(tuple(np.bincount(roles)), (768, 256, 256))
        for f in range(4):
            self.assertEqual(tuple(np.bincount(roles[f::4])), (192, 64, 64))
        np.testing.assert_array_equal(roles, D.episode_roles(1234))

    def test_raw_validation_rejects_false_join_and_time_mismatch(self):
        a = fixture_arrays()
        self.assertEqual(D.validate_arrays(a, require_all_parents=False)["rows"], 4)
        a["states"][2, 0] += .01
        with self.assertRaises(ValueError):
            D.validate_arrays(a, require_all_parents=False)

    def test_hash_binds_shape_dtype_metadata_and_labels(self):
        a = fixture_arrays(); original = D.digest(a, {"name": "fixture"})
        self.assertNotEqual(original, D.digest(a, {"name": "changed"}))
        a["rewards"][1] += 1
        self.assertNotEqual(original, D.digest(a, {"name": "fixture"}))

    def test_invalid_state_action_time_fail_before_plant(self):
        fake = mock.Mock(side_effect=AssertionError("Plant must not run"))
        env = T.Task(fake, geometry())
        for x, t, a in ((np.zeros(12), 0, np.zeros(4)), (state(), .5, np.zeros(4)),
                        (state(), 1, [np.nan] * 4), (state(), 1, np.zeros(3))):
            with self.assertRaises(ValueError):
                env.step(x, t, a)
        fake.assert_not_called()

    def test_single_geometry_recipe_fake_plant_exact_call_count(self):
        calls = []
        def fake(x, a):
            calls.append(1)
            y = x.copy(); y[0] += .1; y[1] += float(a[0])
            return y, 0.0
        recipe = {"initial_state": state().tolist(), "b_time": 25, "c_time": 30,
                  "goal_time": 75, "goal_radius": .1, "danger_radius_max": .1,
                  "danger_radius_fraction": .25, "minimum_region_radius": .01,
                  "connector_trim_fraction": .2, "connector_radius_fraction": .1}
        g = T.construct_geometry(fake, lambda x, t: np.array([.5, 0, 0, 0]), recipe)
        self.assertEqual(len(calls), 105)
        np.testing.assert_array_equal(g["goal"]["center"], np.asarray(g["nominal_route"])[75, :10])
        self.assertFalse(T.contains(np.asarray(g["B"]), g["connector"]))
        self.assertFalse(T.contains(np.asarray(g["C"]), g["connector"]))
        self.assertTrue(T.edge_intersects(np.asarray(g["B"]), np.asarray(g["C"]), g["connector"]))
        self.assertLessEqual(g["danger"]["radius"], .1)

    def test_invalid_geometry_recipe_fails_without_plant_calls(self):
        fake = mock.Mock(side_effect=AssertionError("No plant calls"))
        with self.assertRaises(ValueError):
            T.construct_geometry(fake, None, {"b_time": 30, "c_time": 25, "goal_time": 75})
        fake.assert_not_called()

    def test_recover_before_danger_family_fixed_rule(self):
        controller = lambda x, t: np.array([.5, -.25, .125, -.5])
        g, config = {"b_time": 25, "c_time": 30}, {"action_noise": .03, "wide_action_noise": .15}
        for time in (24, 25, 28, 29, 30):
            rng = mock.Mock()
            rng.normal.return_value = np.zeros(4)
            recovery = D.recorder_action(controller, state(), time, 3, rng, g, config)
            expected = -controller(None, time) if 25 <= time <= 28 else controller(None, time)
            np.testing.assert_array_equal(recovery, expected.astype(np.float32))
            rng.normal.assert_called_once_with(0.0, 0.0, 4)
            danger = D.recorder_action(controller, state(), time, 2, rng, g, config)
            expected = -controller(None, time) if 25 <= time <= 29 else controller(None, time)
            np.testing.assert_array_equal(danger, expected.astype(np.float32))

    def test_parent_callback_stops_before_initialization_or_plant(self):
        fake_task, controller = mock.Mock(), mock.Mock()
        callback = mock.Mock(side_effect=RuntimeError("Stop before parent initialization"))
        with self.assertRaisesRegex(RuntimeError, "Stop before"):
            D.collect(fake_task, controller, {"A": state().tolist()},
                      {"parents": 1280, "split_seed": 22}, on_parent_start=callback)
        callback.assert_called_once_with(0)
        fake_task.step.assert_not_called()
        controller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
