"""Typed generator contract regressions against the unchanged v2 artifact.

The same tests run under actual NumPy 2.2.6 and 2.3.5; provenance edits are
synthetic fixtures, not a substitute for those distinct runtime executions.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from thrusterchain.datasets import (
    DTYPES, DatasetError, TrajectoryDataset, canonical_json, content_digest, load_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/smoke_v2/offline_dataset.npz"
SCRATCH = ROOT / "audits/revision_v3_metadata/v3.verification"
ORIGINAL_FILE_SHA256 = "62d82777d30ca7a0c2fc2529a9f5fde4bd63ad87a527b7b34a81bfd1ecba550a"
ORIGINAL_CONTENT_SHA256 = "7086f0a0657db70f0caabf045cd9372597dd9a384a1365bac1df63cfa76617fc"


def integer_leaves(value, path=()):
    if type(value) is int:
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from integer_leaves(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from integer_leaves(child, (*path, index))


def replace_at(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def reverse_object_order(value):
    if isinstance(value, dict):
        return {key: reverse_object_order(child) for key, child in reversed(list(value.items()))}
    if isinstance(value, list):
        return [reverse_object_order(child) for child in value]
    return value


class TypedGeneratorMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ARTIFACT, allow_pickle=False) as archive:
            cls.arrays = {name: archive[name] for name in DTYPES}
            cls.metadata = json.loads(str(archive["manifest"]))["metadata"]
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def assert_rehashed_metadata_refused(self, metadata):
        refreshed = content_digest(self.arrays, metadata)
        self.assertNotEqual(refreshed, ORIGINAL_CONTENT_SHA256)
        with self.assertRaisesRegex(DatasetError, "random-stream hierarchy"):
            TrajectoryDataset(self.arrays, metadata, expected_digest=refreshed)

    def test_all_five_confirmed_aliases_are_refused_in_rehashed_npz(self):
        cases = (
            (("episodes", 0, "trajectory_id"), False),
            (("episodes", 0, "reset_spawn_key", 1), False),
            (("episodes", 1, "action_spawn_key", 1), True),
            (("seed_sequence", "pool_size"), 4.0),
            (("named_streams", "actions", "spawn_key_prefix"), [1703.0]),
        )
        with tempfile.TemporaryDirectory(prefix="typed_metadata_", dir=SCRATCH) as directory:
            for index, (path, replacement) in enumerate(cases):
                with self.subTest(path=path):
                    metadata = copy.deepcopy(self.metadata)
                    replace_at(metadata["generator"], path, replacement)
                    # Demonstrate the precise Python-equality alias being closed.
                    self.assertEqual(metadata["generator"], self.metadata["generator"])
                    refreshed = content_digest(self.arrays, metadata)
                    self.assertNotEqual(refreshed, ORIGINAL_CONTENT_SHA256)
                    fixture = Path(directory) / (str(index) + ".npz")
                    np.savez_compressed(fixture, **self.arrays, manifest=np.asarray(canonical_json(
                        {"metadata": metadata, "sha256": refreshed})))
                    with np.load(fixture, allow_pickle=False) as archive:
                        written = json.loads(str(archive["manifest"]))
                        self.assertEqual(written["sha256"], content_digest(self.arrays, written["metadata"]))
                    with self.assertRaisesRegex(DatasetError, "random-stream hierarchy"):
                        load_dataset(fixture)

    def test_every_nested_generator_integer_refuses_a_float(self):
        leaves = list(integer_leaves(self.metadata["generator"]))
        # Every episode contributes ID, reset seed, and both two-part addresses.
        self.assertEqual(len(leaves), 6 * len(self.arrays["trajectory_ids"]) + 5)
        for path, value in leaves:
            with self.subTest(path=path):
                metadata = copy.deepcopy(self.metadata)
                replace_at(metadata["generator"], path, float(value))
                self.assert_rehashed_metadata_refused(metadata)

    def test_every_zero_or_one_episode_address_refuses_a_boolean(self):
        aliases = [(path, value) for path, value in integer_leaves(self.metadata["generator"])
                   if path[0] == "episodes" and value in (0, 1)]
        self.assertEqual(len(aliases), 6)
        for path, value in aliases:
            with self.subTest(path=path):
                metadata = copy.deepcopy(self.metadata)
                replace_at(metadata["generator"], path, bool(value))
                self.assertEqual(metadata["generator"], self.metadata["generator"])
                self.assert_rehashed_metadata_refused(metadata)

    def test_nested_container_shapes_and_fields_remain_exact(self):
        changes = (
            lambda g: g["seed_sequence"].update(root_spawn_key={}),
            lambda g: g["named_streams"]["reset"].update(spawn_key_prefix=1701),
            lambda g: g["named_streams"]["split"].update(spawn_key=[1702, 0]),
            lambda g: g["episodes"][-1].update(action_spawn_key={"0": 1703, "1": 39}),
            lambda g: g["episodes"][-1].update(reset_spawn_key=[1701]),
            lambda g: g["episodes"][-1].pop("reset_seed"),
            lambda g: g["episodes"][-1].update(unrecognized_field=0),
            lambda g: g["named_streams"]["actions"].update(child=None),
            lambda g: g.update(episodes={}),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                metadata = copy.deepcopy(self.metadata)
                change(metadata["generator"])
                self.assert_rehashed_metadata_refused(metadata)

    def test_json_object_order_does_not_change_the_contract_or_digest(self):
        metadata = reverse_object_order(self.metadata)
        self.assertNotEqual(list(metadata["generator"]), list(self.metadata["generator"]))
        dataset = TrajectoryDataset(self.arrays, metadata, expected_digest=ORIGINAL_CONTENT_SHA256)
        self.assertEqual(dataset.digest, ORIGINAL_CONTENT_SHA256)
        self.assertEqual(canonical_json(dataset.metadata), canonical_json(self.metadata))

    def test_untouched_artifact_preserves_arrays_provenance_and_original_hashes(self):
        self.assertEqual(hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(), ORIGINAL_FILE_SHA256)
        dataset = load_dataset(ARTIFACT)
        self.assertEqual(dataset.metadata["generator"]["numpy_version"], "2.2.6")
        self.assertEqual(canonical_json(dataset.metadata), canonical_json(self.metadata))
        self.assertEqual(dataset.digest, ORIGINAL_CONTENT_SHA256)
        for name, expected in self.arrays.items():
            self.assertEqual(dataset.arrays[name].dtype.str, expected.dtype.str)
            self.assertEqual(dataset.arrays[name].tobytes(), expected.tobytes(), name)
        self.assertEqual(hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(), ORIGINAL_FILE_SHA256)

    def test_valid_recorded_numpy_versions_remain_provenance(self):
        runtime_version = np.__version__
        for version in ("2.2.6", "2.3.5", "7.8.9rc1.post2.dev3+qa.1"):
            with self.subTest(version=version):
                metadata = copy.deepcopy(self.metadata)
                metadata["generator"]["numpy_version"] = version
                refreshed = content_digest(self.arrays, metadata)
                dataset = TrajectoryDataset(self.arrays, metadata, expected_digest=refreshed)
                self.assertEqual(dataset.metadata["generator"]["numpy_version"], version)
                self.assertEqual(canonical_json(dataset.metadata), canonical_json(metadata))
                self.assertEqual(dataset.digest, refreshed)
        self.assertEqual(np.__version__, runtime_version)

    def test_provenance_change_still_requires_a_new_digest(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["generator"]["numpy_version"] = "2.3.5"
        with self.assertRaisesRegex(DatasetError, "SHA-256 mismatch"):
            TrajectoryDataset(self.arrays, metadata, expected_digest=ORIGINAL_CONTENT_SHA256)


if __name__ == "__main__":
    unittest.main()
