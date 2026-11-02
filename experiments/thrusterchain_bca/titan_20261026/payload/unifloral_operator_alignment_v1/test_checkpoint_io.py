"""Synthetic/stdlib loader guard tests; never restore an actual model."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_operator_checkpoint_unit", ROOT / "checkpoint_io.py")
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


def fixture():
    split = {"scale_bank_training_row_indices": list(range(256)), "scale_bank_seed": 2026102363,
             "train_episodes": 768, "train_rows": 76800, "calibration_rows": 128}
    bindings = {"config": deepcopy(io.CONFIG), "datasetidentity": {"sha256": io.DATA_SHA,
        "digest": io.DATA_DIGEST, "split_receipt": split},
        "normalization": {"mean": [0.0] * 14, "std": [1.0] * 14}}
    metadata = {"schema": "unifloral.full_training_checkpoint.v1", "counts": deepcopy(io.COUNTS),
        "bindings": deepcopy(bindings), "bindings_sha256": io.sha(io.encoded(bindings) + b"\n"),
        "state_sha256": io.STATE_SHA, "state_bytes": 10328701}
    return metadata, bindings


class LoaderContractTests(unittest.TestCase):
    def test_complete_terminal_contract(self):
        io._terminal(*fixture())

    def test_refuse_earlier_refresh(self):
        meta, bindings = fixture()
        meta["counts"]["last_refresh_critic"] = 15000
        with self.assertRaises(ValueError):
            io._terminal(meta, bindings)

    def test_refuse_boolean_counter(self):
        meta, bindings = fixture()
        meta["counts"]["phase"] = False
        with self.assertRaises(ValueError):
            io._terminal(meta, bindings)

    def test_refuse_wrong_payload(self):
        meta, bindings = fixture()
        meta["state_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            io._terminal(meta, bindings)

    def test_refuse_changed_training_config(self):
        meta, bindings = fixture()
        bindings["config"]["gamma"] = .99
        with self.assertRaises(ValueError):
            io._terminal(meta, bindings)

    def test_refuse_changed_bindings_digest(self):
        meta, bindings = fixture()
        meta["bindings_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            io._terminal(meta, bindings)

    def test_refuse_invalid_normalization(self):
        meta, bindings = fixture()
        bindings["normalization"]["std"][0] = -1
        meta["bindings"] = deepcopy(bindings)
        meta["bindings_sha256"] = io.sha(io.encoded(bindings) + b"\n")
        with self.assertRaises(ValueError):
            io._terminal(meta, bindings)

    def test_saved_indices(self):
        _, bindings = fixture()
        self.assertEqual(io._indices(bindings["datasetidentity"]["split_receipt"]), list(range(256)))

    def test_refuse_duplicate_indices(self):
        _, bindings = fixture()
        split = bindings["datasetidentity"]["split_receipt"]
        split["scale_bank_training_row_indices"][1] = 0
        with self.assertRaises(ValueError):
            io._indices(split)

    def test_refuse_out_of_range_or_boolean_index(self):
        for value in (-1, 76800, True, 1.5):
            _, bindings = fixture()
            split = bindings["datasetidentity"]["split_receipt"]
            split["scale_bank_training_row_indices"][0] = value
            with self.assertRaises(ValueError):
                io._indices(split)

    def test_refuse_wrong_bank_role(self):
        _, bindings = fixture()
        split = bindings["datasetidentity"]["split_receipt"]
        split["scale_bank_seed"] += 1
        with self.assertRaises(ValueError):
            io._indices(split)

    def test_protocol_exact(self):
        protocol = {"gamma": .98, "horizon": 100, "fitting_rows": 256,
                    "checkpoint_manifest_sha256": io.EXPORT_SHA}
        io._protocol(protocol)
        for key, value in (("gamma", .99), ("horizon", 100.0), ("fitting_rows", 128),
                           ("checkpoint_manifest_sha256", "0" * 64)):
            wrong = {**protocol, key: value}
            with self.assertRaises(ValueError):
                io._protocol(wrong)

    def test_json_duplicates_and_nonfinite_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                io.read_json(raw)

    def test_module_import_uses_no_scientific_packages(self):
        # Modules may already exist in an external test host; this import must add none.
        before = set(sys.modules)
        isolated = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isolated)
        added = set(sys.modules) - before
        self.assertFalse(any(name.split('.')[0] in {"jax", "flax", "numpy", "optax", "distrax"} for name in added))


class CommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_checkpoint_test_", dir=ROOT)
        self.root = Path(self.temp.name)
        self.files = {"metadata.json": b'{"synthetic_test_only":true}', "state.msgpack": b"not-a-model"}
        for name, value in self.files.items():
            (self.root / name).write_bytes(value)
        self.marker = {"schema": "unifloral.committed_artifact.v1", "kind": "training_checkpoint",
                      "files": {name: {"sha256": io.sha(value), "bytes": len(value)} for name, value in self.files.items()}}
        raw = io.encoded(self.marker)
        (self.root / "COMMITTED.json").write_bytes(raw)
        self.marker_sha = io.sha(raw)

    def tearDown(self):
        self.temp.cleanup()

    def check(self):
        return io._commit(self.root, "training_checkpoint", self.marker_sha, set(self.files))

    def test_complete_commit(self):
        self.assertEqual(self.check(), {name: io.sha(raw) for name, raw in self.files.items()})

    def test_truncated_payload_rejected(self):
        (self.root / "state.msgpack").write_bytes(b"not")
        with self.assertRaises(ValueError):
            self.check()

    def test_extra_file_rejected(self):
        (self.root / "extra.txt").write_text("synthetic")
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_file_rejected(self):
        (self.root / "state.msgpack").unlink()
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_marker_rejected(self):
        (self.root / "COMMITTED.json").write_bytes(b"{}")
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_artifact_kind_rejected(self):
        with self.assertRaises(ValueError):
            io._commit(self.root, "inference_export", self.marker_sha, set(self.files))

    def test_path_escape_rejected(self):
        for relative in ("../outside", str(self.root / "state.msgpack"), "C:/elsewhere/file"):
            with self.assertRaises(ValueError):
                io.checked_file(self.root, relative)


if __name__ == "__main__":
    unittest.main()
