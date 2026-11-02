"""Pure saved-array tests; no real dataset/environment module is imported."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).with_name("immutable_data.py")
spec = importlib.util.spec_from_file_location("immutable_data_fixture", PATH)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class FakeKneeDataset:
    __slots__ = ("_arrays", "_metadata_json", "_digest")

    def __init__(self, *args, **kwargs):
        raise AssertionError("The replaying constructor must never be called")

    def __setattr__(self, name, value):
        raise AttributeError("immutable")

    arrays = property(lambda self: self._arrays)
    metadata = property(lambda self: json.loads(self._metadata_json))
    digest = property(lambda self: self._digest)


def fake_module():
    return SimpleNamespace(__file__=str(M.DATASET_MODULE), SCHEMA=M.SCHEMA,
                           DTYPES=dict(M.DTYPES), KneeDataset=FakeKneeDataset)


class ImmutableDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(M.DATASET, allow_pickle=False) as archive:
            cls.envelope = json.loads(str(archive["manifest"]))
            cls.arrays = {name: archive[name] for name in M.DTYPES}

    def test_exact_authorized_dataset_and_episode_pairing(self):
        result, receipt = M.load_verified(M.DATASET, fake_module())
        self.assertIs(type(result), FakeKneeDataset)
        self.assertEqual(result.digest, M.CONTENT_SHA)
        self.assertEqual(result.metadata, self.envelope["metadata"])
        for name, expected in self.arrays.items():
            self.assertEqual(result.arrays[name].dtype, expected.dtype)
            self.assertEqual(result.arrays[name].shape, expected.shape)
            self.assertEqual(result.arrays[name].tobytes(), expected.tobytes(), name)
            with self.assertRaises(ValueError):
                result.arrays[name].setflags(write=True)
        with self.assertRaises(TypeError):
            result.arrays["states"] = self.arrays["states"]
        metadata = result.metadata
        metadata["seed"] = 1
        self.assertEqual(result.metadata["seed"], M.ENTROPY)
        self.assertEqual(receipt["producer"]["verified_source_files"], 44)
        self.assertEqual(receipt["producer"]["producer_validated_transition_rows"], 128000)
        self.assertEqual(receipt["simulator_steps"], 0)
        self.assertTrue(receipt["historical_validation_work_reused_not_reexecuted"])
        json.dumps(receipt, allow_nan=False)

    def test_changed_archive_bytes_rejected(self):
        with TemporaryDirectory() as tmp:
            altered = Path(tmp) / M.DATASET.name
            shutil.copyfile(M.DATASET, altered)
            with altered.open("ab") as stream:
                stream.write(b"changed archive container")
            with patch.object(M, "DATASET", altered):
                with self.assertRaisesRegex(ValueError, "archive hash differs"):
                    M.load_verified(altered, fake_module())

    def test_other_path_rejected_even_with_same_name(self):
        with self.assertRaisesRegex(ValueError, "authorized dataset path"):
            M.load_verified(Path("elsewhere") / M.DATASET.name, fake_module())

    def test_one_ulp_content_change_rejected(self):
        changed = dict(self.arrays)
        changed["rewards"] = changed["rewards"].copy()
        changed["rewards"][0, 0] = np.nextafter(changed["rewards"][0, 0], np.inf)
        with self.assertRaisesRegex(ValueError, "content hash differs"):
            M.check_content(changed, self.envelope)

    def test_changed_provenance_and_metadata_rejected(self):
        changed = copy.deepcopy(self.envelope)
        changed["metadata"]["provenance"] = "smoke"
        with self.assertRaisesRegex(ValueError, "metadata identity differs"):
            M.check_content(self.arrays, changed)
        changed = copy.deepcopy(self.envelope)
        changed["metadata"]["generator"]["numpy_version"] = "different"
        with self.assertRaisesRegex(ValueError, "content hash differs"):
            M.check_content(self.arrays, changed)

    def test_envelope_digest_is_not_trusted_without_recomputation(self):
        changed = copy.deepcopy(self.envelope)
        changed["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash differs"):
            M.check_content(self.arrays, changed)

    def test_shape_dtype_schema_and_nonfinite_rejected(self):
        for value in (self.arrays["rewards"][:, :-1], self.arrays["rewards"].astype(np.float32)):
            changed = dict(self.arrays, rewards=value)
            with self.assertRaisesRegex(ValueError, "shape/dtype differs"):
                M.check_content(changed, self.envelope)
        changed = dict(self.arrays)
        changed.pop("trajectory_ids")
        with self.assertRaisesRegex(ValueError, "schema differs"):
            M.check_content(changed, self.envelope)
        changed = dict(self.arrays, rewards=self.arrays["rewards"].copy())
        changed["rewards"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            M.check_content(changed, self.envelope)

    def test_changed_producer_receipt_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            content = json.loads((M.RUN / "report.json").read_text())
            content["collection"]["stall3"]["sha256"] = "0" * 64
            path.write_text(json.dumps(content))
            with patch.object(M, "RUN", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "Producer evidence changed"):
                    M.authenticate_producer()

    def test_changed_dataset_class_schema_rejected(self):
        module = fake_module()
        module.DTYPES["states"] = "<f4"
        with self.assertRaisesRegex(ValueError, "class/source/schema differs"):
            M.load_verified(M.DATASET, module)

    def test_imports_and_calls_are_data_only(self):
        self.assertFalse(any(name.split(".")[0] in {"jax", "jaxlib", "thrusterchain", "worlds", "dataset_knee"}
                             for name in sys.modules))
        tree = ast.parse(PATH.read_text())
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else
                 n.func.id if isinstance(n.func, ast.Name) else ""
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertFalse(calls & {"reset", "step", "dynamics", "validate", "generate", "train", "calibrate", "allclose"})


if __name__ == "__main__":
    unittest.main()
