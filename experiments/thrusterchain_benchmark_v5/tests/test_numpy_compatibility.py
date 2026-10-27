"""Dataset provenance portability and seeded-collector lineage regressions.

Changing fixture metadata tests validation policy, not an actual NumPy runtime
matrix. The untouched packaged-artifact test also runs in each real runtime.
"""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thrusterchain.datasets import (
    DTYPES, DatasetError, SPLITS, TrajectoryDataset, canonical_json,
    content_digest, load_dataset,
)


ARTIFACT = Path(__file__).resolve().parents[1] / "results" / "smoke_v2" / "offline_dataset.npz"
ORIGINAL_CONTENT_SHA256 = "7086f0a0657db70f0caabf045cd9372597dd9a384a1365bac1df63cfa76617fc"


class NumpyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_sha256 = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        with np.load(ARTIFACT, allow_pickle=False) as archive:
            cls.arrays = {name: archive[name] for name in DTYPES}
            cls.envelope = json.loads(str(archive["manifest"]))

    def metadata_copy(self):
        return copy.deepcopy(self.envelope["metadata"])

    def arrays_copy(self):
        return {name: value.copy() for name, value in self.arrays.items()}

    def write_fixture(self, directory, *, metadata=None, arrays=None, digest=None):
        metadata = self.metadata_copy() if metadata is None else metadata
        arrays = self.arrays if arrays is None else arrays
        envelope = {"metadata": metadata,
                    "sha256": content_digest(arrays, metadata) if digest is None else digest}
        path = Path(directory) / "fixture.npz"
        np.savez_compressed(path, **arrays, manifest=np.asarray(canonical_json(envelope)))
        return path

    def test_untouched_packaged_artifact_load_preserves_provenance_and_hash(self):
        dataset = load_dataset(ARTIFACT)
        self.assertEqual(dataset.metadata["generator"]["numpy_version"], "2.2.6")
        self.assertEqual(dataset.metadata, self.envelope["metadata"])
        self.assertEqual(dataset.digest, ORIGINAL_CONTENT_SHA256)
        self.assertEqual(content_digest(self.arrays, self.envelope["metadata"]), dataset.digest)
        self.assertEqual(hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(), self.file_sha256)

    def test_written_npz_accepts_a_different_recorded_version(self):
        # This is a synthetic provenance fixture; np.__version__ is never patched.
        runtime_version = np.__version__
        recorded_version = "2.3.5" if runtime_version != "2.3.5" else "2.2.6"
        metadata = self.metadata_copy()
        metadata["generator"]["numpy_version"] = recorded_version
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory, metadata=metadata)
            dataset = load_dataset(path)
        self.assertNotEqual(recorded_version, runtime_version)
        self.assertEqual(np.__version__, runtime_version)
        self.assertEqual(dataset.metadata, metadata)
        self.assertEqual(dataset.digest, content_digest(self.arrays, metadata))
        # A version different from the runtime may still be the original recorded
        # version. Only a change to stored content must change the content digest.
        if metadata == self.envelope["metadata"]:
            self.assertEqual(dataset.digest, ORIGINAL_CONTENT_SHA256)
        else:
            self.assertNotEqual(dataset.digest, ORIGINAL_CONTENT_SHA256)

    def test_release_candidate_development_and_local_versions_remain_provenance(self):
        for version in ("2.3.0rc1", "2.4.0.dev0+git.abcdef", "2.3.5.post1", "2.3.5+vendor.1"):
            with self.subTest(version=version):
                metadata = self.metadata_copy()
                metadata["generator"]["numpy_version"] = version
                dataset = TrajectoryDataset(self.arrays, metadata)
                self.assertEqual(dataset.metadata["generator"]["numpy_version"], version)

    def test_malformed_or_missing_numpy_version_is_refused_after_rehash(self):
        malformed = (None, True, 235, [], {}, "", "2.3", "v2.3.5", " 2.3.5",
                     "2.3.5\n", "2.3.5garbage", "2.3.5+", "2.3.5..dev0")
        with tempfile.TemporaryDirectory() as directory:
            for version in malformed:
                with self.subTest(version=version):
                    metadata = self.metadata_copy()
                    metadata["generator"]["numpy_version"] = version
                    path = self.write_fixture(directory, metadata=metadata)
                    with self.assertRaisesRegex(DatasetError, "NumPy version provenance"):
                        load_dataset(path)
            metadata = self.metadata_copy()
            del metadata["generator"]["numpy_version"]
            path = self.write_fixture(directory, metadata=metadata)
            with self.assertRaisesRegex(DatasetError, "NumPy version provenance"):
                load_dataset(path)

    def test_valid_version_change_cannot_retain_the_original_digest(self):
        metadata = self.metadata_copy()
        metadata["generator"]["numpy_version"] = "2.3.5"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory, metadata=metadata, digest=ORIGINAL_CONTENT_SHA256)
            with self.assertRaisesRegex(DatasetError, "SHA-256 mismatch"):
                load_dataset(path)

    def test_generator_structure_tampering_is_refused_after_rehash(self):
        changes = (
            lambda g: g.update(version="continuous_box_sampler_v999"),
            lambda g: g.update(bit_generator="Philox"),
            lambda g: g["seed_sequence"].update(entropy=g["seed_sequence"]["entropy"] + 1),
            lambda g: g["named_streams"]["actions"].update(spawn_key_prefix=[999]),
            lambda g: g["episodes"][0].update(action_spawn_key=[1703, 999]),
            lambda g: g["episodes"][0].update(reset_seed=g["episodes"][0]["reset_seed"] ^ 1),
            lambda g: g.update(unrecognized_field="not allowed"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, change in enumerate(changes):
                with self.subTest(change=index):
                    metadata = self.metadata_copy()
                    metadata["generator"]["numpy_version"] = "2.3.5"
                    change(metadata["generator"])
                    path = self.write_fixture(directory, metadata=metadata)
                    with self.assertRaisesRegex(DatasetError, "random-stream hierarchy"):
                        load_dataset(path)

    def test_source_model_hashes_remain_exact_after_rehash(self):
        for field in ("canonical_provenance_sha256", "adapter_source_sha256", "canonical_loader_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                metadata = self.metadata_copy()
                metadata["generator"]["numpy_version"] = "2.3.5"
                metadata["source_model_metadata"][field] = "0" * 64
                path = self.write_fixture(directory, metadata=metadata)
                with self.assertRaisesRegex(DatasetError, "environment revision"):
                    load_dataset(path)

    def test_episode_seed_array_tampering_is_refused_after_rehash(self):
        arrays = self.arrays_copy()
        arrays["episode_seeds"][0] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory, arrays=arrays)
            with self.assertRaisesRegex(DatasetError, "reset seeds disagree"):
                load_dataset(path)

    def test_copied_training_episode_with_distinct_calibration_labels_is_refused(self):
        arrays = self.arrays_copy()
        source = np.flatnonzero(arrays["split_codes"] == SPLITS.index("train"))[0]
        target = np.flatnonzero(arrays["split_codes"] == SPLITS.index("final_calibration"))[0]
        identity_fields = {"trajectory_ids", "group_ids", "episode_seeds", "split_codes"}
        for name in arrays:
            if name not in identity_fields:
                arrays[name][target] = arrays[name][source]
        self.assertNotEqual(arrays["episode_seeds"][source], arrays["episode_seeds"][target])
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory, arrays=arrays)
            with self.assertRaisesRegex(DatasetError, "seeded collector reset lineage"):
                load_dataset(path)

    def test_requested_action_stream_tampering_is_refused(self):
        arrays = self.arrays_copy()
        # A tiny edit keeps the same quantized action, region, density and physics.
        # It exceeds lineage tolerance while remaining inside this float32 cell.
        value = arrays["requested_actions"][0, 0, 1]
        quantized = float(np.float32(value))
        arrays["requested_actions"][0, 0, 1] = (value + quantized) / 2
        self.assertGreater(abs(arrays["requested_actions"][0, 0, 1] - value), 1e-12)
        np.testing.assert_array_equal(arrays["requested_actions"].astype(np.float32), arrays["actions"])
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory, arrays=arrays)
            with self.assertRaisesRegex(DatasetError, "seeded collector action lineage"):
                load_dataset(path)

    def test_seeded_lineage_permits_float64_rounding_noise(self):
        arrays = self.arrays_copy()
        arrays["states"][0, 0, 11] = np.nextafter(arrays["states"][0, 0, 11], np.inf)
        arrays["requested_actions"][0, 0, 1] = np.nextafter(arrays["requested_actions"][0, 0, 1], np.inf)
        dataset = TrajectoryDataset(arrays, self.metadata_copy())
        self.assertNotEqual(dataset.digest, ORIGINAL_CONTENT_SHA256)

    def test_validation_under_inference_guard_does_not_collect_or_step(self):
        from thrusterchain.inference_guard import InferenceGuard
        with ExitStack() as stack:
            for target in ("thrusterchain.environment.ThrusterChainEnv.__init__",
                           "thrusterchain.environment.ThrusterChainEnv.reset",
                           "thrusterchain.environment.ThrusterChainEnv.step",
                           "thrusterchain.datasets.generate_dataset"):
                stack.enter_context(patch(target, side_effect=AssertionError("validation collected data")))
            with InferenceGuard() as guard:
                dataset = load_dataset(ARTIFACT)
        self.assertEqual(dataset.digest, ORIGINAL_CONTENT_SHA256)
        self.assertEqual(guard.ledger["forbidden_api_calls"], 0)
        self.assertEqual(guard.events, dict(training_calls=0, dataset_collection_calls=0, write_attempts=0))


if __name__ == "__main__":
    unittest.main()
