"""Pure fixtures and static checks only. No real environment imports or resets."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PATH = Path(__file__).with_name("diagnose.py")
spec = importlib.util.spec_from_file_location("reset_diagnostic", PATH)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


class DiagnosticTests(unittest.TestCase):
    def test_import_has_no_environment_or_accelerator_side_effect(self):
        self.assertFalse(any(name.split(".")[0] in
                             {"thrusterchain", "worlds", "jax", "jaxlib", "torch", "tensorflow"}
                             for name in sys.modules))

    def test_equal_vector(self):
        a = np.arange(13, dtype=np.float64)
        result = D.describe_values(a, a.copy(), np)
        self.assertTrue(result["original_array_equal"])
        self.assertTrue(result["byte_equal"])
        self.assertEqual(result["differing_coordinates"], [])

    def test_one_ulp_is_detected_not_tolerated(self):
        for value in (1., -1.):
            a = np.full(13, value, np.float64)
            b = a.copy()
            b[11] = np.nextafter(value, math.inf)
            result = D.describe_values(a, b, np)
            self.assertFalse(result["original_array_equal"])
            self.assertEqual(result["differing_coordinates"], [11])
            self.assertEqual(result["coordinates"][11]["ulp_distance"], 1)

    def test_zero_and_cross_zero_ulp(self):
        tiny = np.nextafter(0., 1.)
        self.assertEqual(D.ulps(-tiny, tiny), 2)
        self.assertEqual(D.ulps(-0., 0.), 0)
        a, b = np.zeros(13), np.zeros(13)
        b[0] = -0.
        result = D.describe_values(a, b, np)
        self.assertTrue(result["original_array_equal"])
        self.assertFalse(result["byte_equal"])

    def test_structure_and_dtype_are_not_silently_cast(self):
        a = np.zeros(13, np.float64)
        self.assertFalse(D.describe_values(a, a.astype(np.float32), np)["same_dtype"])
        result = D.describe_values(a, a.reshape(1, 13), np)
        self.assertFalse(result["same_shape"])
        self.assertFalse(result["original_array_equal"])

    def test_nonfinite_differences_are_explicit(self):
        a = np.zeros(13)
        b = a.copy()
        b[2] = np.nan
        result = D.describe_values(a, b, np)
        self.assertEqual(result["differing_coordinates"], [2])
        self.assertIsNone(result["coordinates"][2]["ulp_distance"])

    def test_one_reset_and_close_success_or_error(self):
        for fail in (False, True):
            calls = []
            class FakeEnv:
                def reset(self, **kwargs):
                    calls.append(("reset", kwargs))
                    if fail:
                        raise ValueError("fixture failure")
                    return "observation", {"state": "fixture"}
                def close(self):
                    calls.append(("close", None))
                def step(self, *args):
                    self.fail("No step allowed")
            counts = {"reset_calls_attempted": 0, "reset_calls_completed": 0}
            if fail:
                with self.assertRaises(ValueError):
                    D.one_reset(FakeEnv(), 123, {"start": {}}, counts)
            else:
                D.one_reset(FakeEnv(), 123, {"start": {}}, counts)
            self.assertEqual(calls, [("reset", {"seed": 123, "options": {"start": {}}}),
                                     ("close", None)])
            self.assertEqual(counts, {"reset_calls_attempted": 1,
                                      "reset_calls_completed": 0 if fail else 1})

    def test_static_calls_have_no_steps_full_loader_or_training(self):
        tree = ast.parse(PATH.read_text())
        call_names = [node.func.attr if isinstance(node.func, ast.Attribute) else
                      node.func.id if isinstance(node.func, ast.Name) else ""
                      for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertEqual(call_names.count("reset"), 1)
        self.assertFalse(set(call_names) & {"step", "dynamics", "dynamics_jax", "train", "generate",
                                            "calibrate", "allclose", "isclose"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({n.name.split(".")[0] for n in node.names} &
                                 {"dataset_knee", "stage1_worker", "screen_worker", "jax", "torch"})

    def test_step_guard_always_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            D.forbidden_step()

    def test_frozen_manifest_and_sources_unchanged(self):
        self.assertEqual(len(D.verify_seal()["files"]), 154)


if __name__ == "__main__":
    unittest.main()
