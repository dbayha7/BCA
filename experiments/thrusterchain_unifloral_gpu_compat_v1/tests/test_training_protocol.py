"""Public contract regressions: counts, immutable state and safe transport."""
import base64
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from thrusterchain.datasets import DatasetError, inference_evaluation_guard
from thrusterchain.training_protocol import (
    AdapterCapabilities, DEFAULT_OFFLINE_UPDATES, EvaluationConfig,
    LearnerAdapter, LearnerCheckpoint, MAX_OFFLINE_UPDATES, OfflineBudget,
    OnlineBudget, OnlineLearnerAdapter, REQUIRED_CHECKPOINT_COMPONENTS,
    TransitionBatch,
)


def checkpoint(steps=2, **overrides):
    values = {
        "adapter_id": "test-host-v1",
        "components": {name: ("state:" + name).encode() for name in REQUIRED_CHECKPOINT_COMPONENTS},
        "optimizer_steps": steps,
        "optimizer_step_counts": {"actor": steps // 2, "critic": steps - steps // 2},
        "metadata": {
            "primary_optimizer": "critic",
            "training_config": {"seed": 7, "settings": {"learning_rate": 0.01}},
            "dataset_lineage": {"training_split_sha256": "test-digest"},
        },
    }
    values.update(overrides)
    return LearnerCheckpoint(**values)


def batch(**overrides):
    values = {
        "observations": np.arange(12, dtype=np.float32).reshape(3, 4),
        "actions": np.arange(6, dtype=np.float32).reshape(3, 2),
        "rewards": np.array([1., 2., 3.]),
        "next_observations": np.arange(12, dtype=np.float32).reshape(3, 4) + 1,
        "terminated": np.array([False, True, False]),
        "truncated": np.array([False, False, True]),
        "bootstrap_mask": np.array([1., 0., 1.]),
        "origin": np.array([0, 0, 1], dtype=np.uint8),
        "group_ids": np.array([10, 10, 20], dtype=np.int64),
    }
    values.update(overrides)
    return TransitionBatch(**values)


def rehash_envelope(document):
    raw = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    document["sha256"] = hashlib.sha256(raw).hexdigest()
    return json.dumps(document).encode()


class TrainingBudgetTests(unittest.TestCase):
    def test_defaults_maximum_and_explicit_zero(self):
        self.assertEqual(DEFAULT_OFFLINE_UPDATES, 1_000_000)
        self.assertEqual(MAX_OFFLINE_UPDATES, 3_000_000)
        self.assertEqual(OfflineBudget().to_dict(), {"updates": 1_000_000, "batch_size": 256})
        self.assertEqual(OfflineBudget(MAX_OFFLINE_UPDATES).updates, 3_000_000)
        self.assertEqual(OfflineBudget(0).updates, 0)
        with self.assertRaises(ValueError):
            OfflineBudget(3_000_001)

    def test_python_integer_validation_for_all_counts(self):
        for bad in (True, False, 1., "1", np.int64(1), None):
            constructors = (
                lambda: OfflineBudget(updates=bad),
                lambda: OfflineBudget(batch_size=bad),
                lambda: OnlineBudget(bad, 0, .5, 2, {}),
                lambda: OnlineBudget(1, bad, .5, 2, {}),
                lambda: OnlineBudget(1, 1, .5, bad, {}),
                lambda: EvaluationConfig(episodes=bad),
                lambda: EvaluationConfig(seed=bad),
            )
            for constructor in constructors:
                with self.subTest(bad=repr(bad), constructor=constructor):
                    with self.assertRaises(TypeError):
                        constructor()

    def test_negative_counts_and_nonpositive_batch_or_episodes(self):
        for constructor in (
            lambda: OfflineBudget(-1), lambda: OfflineBudget(batch_size=0),
            lambda: OnlineBudget(-1, 0, 0., 1, {}),
            lambda: OnlineBudget(1, -1, 0., 1, {}),
            lambda: OnlineBudget(1, 0, 0., 0, {}),
            lambda: EvaluationConfig(episodes=0), lambda: EvaluationConfig(seed=-1),
        ):
            with self.assertRaises(ValueError):
                constructor()

    def test_online_has_no_implicit_fields(self):
        with self.assertRaises(TypeError):
            OnlineBudget()
        with self.assertRaises(TypeError):
            OnlineBudget(interactions=3, updates=2, offline_fraction=.5, batch_size=4)
        with self.assertRaises(ValueError):
            OnlineBudget(0, 1, .5, 2, {})
        self.assertEqual(OnlineBudget(0, 0, 0., 1, {}).updates, 0)
        self.assertEqual(OnlineBudget(2, 0, 1., 1, {}).interactions, 2)

    def test_online_fraction_and_settings_validation(self):
        for invalid in (-.01, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                OnlineBudget(1, 1, invalid, 2, {})
        for invalid in (True, "0.5", None):
            with self.assertRaises(TypeError):
                OnlineBudget(1, 1, invalid, 2, {})
        for invalid in (None, [], ""):
            with self.assertRaises(TypeError):
                OnlineBudget(1, 1, .5, 2, invalid)
        for invalid in ({"lr": float("nan")}, {"lr": float("inf")}):
            with self.assertRaises(ValueError):
                OnlineBudget(1, 1, .5, 2, invalid)
        with self.assertRaises(TypeError):
            OnlineBudget(1, 1, .5, 2, {1: "bad key"})

    def test_budget_settings_are_deeply_detached(self):
        settings = {"schedule": {"milestones": [2, 4]}, "learning_rate": .1}
        budget = OnlineBudget(2, 2, .5, 2, settings)
        settings["schedule"]["milestones"].append(9)
        exposed = budget.learning_settings
        exposed["schedule"]["milestones"].append(8)
        exported = budget.to_dict()
        exported["learning_settings"]["schedule"]["milestones"].append(7)
        self.assertEqual(budget.learning_settings["schedule"]["milestones"], [2, 4])
        with self.assertRaises(FrozenInstanceError):
            budget.updates = 9

    def test_evaluation_frozen_default_and_explicit_toggle(self):
        config = EvaluationConfig()
        self.assertTrue(config.frozen_policy)
        self.assertIsNone(config.online)
        self.assertEqual(config.to_dict(), {"frozen_policy": True, "episodes": 2,
                                           "seed": 20361022, "online": None})
        zero = OnlineBudget(0, 0, .5, 2, {})
        with self.assertRaises(ValueError):
            EvaluationConfig(online=zero)
        with self.assertRaises(ValueError):
            EvaluationConfig(frozen_policy=False)
        explicit = EvaluationConfig(frozen_policy=False, online=zero)
        self.assertEqual(explicit.to_dict()["online"], zero.to_dict())
        for invalid in (0, 1, "false", None):
            with self.assertRaises(TypeError):
                EvaluationConfig(frozen_policy=invalid)

    def test_capability_validation_and_lightweight_protocol_properties(self):
        self.assertEqual(AdapterCapabilities("closed-form", "closed_form", False).fit_semantics,
                         "closed_form")
        for kwargs in ({"adapter_id": ""}, {"fit_semantics": "epochs"}, {"supports_online": 1}):
            values = {"adapter_id": "host", "fit_semantics": "optimizer_steps", "supports_online": True,
                      "primary_optimizer": "critic"}
            values.update(kwargs)
            with self.assertRaises((TypeError, ValueError)):
                AdapterCapabilities(**values)
        self.assertTrue(isinstance(LearnerAdapter.optimizer_steps, property))
        self.assertTrue(isinstance(LearnerAdapter.optimizer_step_counts, property))
        self.assertTrue(callable(OnlineLearnerAdapter.configure_learning_settings))
        with self.assertRaises(ValueError):
            AdapterCapabilities("missing-primary", "optimizer_steps", True)
        with self.assertRaises(ValueError):
            AdapterCapabilities("closed", "closed_form", False, "critic")
        self.assertEqual(AdapterCapabilities("host", "optimizer_steps", True, "critic").primary_optimizer,
                         "critic")


class LearnerCheckpointTests(unittest.TestCase):
    def test_round_trip_is_deterministic_and_component_complete(self):
        original = checkpoint()
        restored = LearnerCheckpoint.from_bytes(original.to_bytes())
        self.assertEqual(restored.digest, original.digest)
        self.assertEqual(restored.to_bytes(), original.to_bytes())
        self.assertEqual(restored.metadata, original.metadata)
        self.assertEqual(dict(restored.components), dict(original.components))
        self.assertEqual(dict(restored.optimizer_step_counts), {"actor": 1, "critic": 1})
        self.assertEqual(restored.optimizer_steps, 2)
        self.assertEqual(restored.primary_optimizer, "critic")
        self.assertEqual(restored.primary_optimizer_steps, 1)
        self.assertEqual(set(restored.components), REQUIRED_CHECKPOINT_COMPONENTS)

    def test_component_order_does_not_change_digest(self):
        first = checkpoint()
        second = checkpoint(components=dict(reversed(list(first.components.items()))),
                            optimizer_step_counts={"critic": 1, "actor": 1})
        self.assertEqual(first.to_bytes(), second.to_bytes())

    def test_constructor_copies_and_properties_do_not_expose_mutation(self):
        components = {name: b"state" for name in REQUIRED_CHECKPOINT_COMPONENTS}
        counts = {"optimizer": 2}
        metadata = {"nested": {"values": [1, 2]}}
        item = checkpoint(components=components, optimizer_step_counts=counts, metadata=metadata)
        digest = item.digest
        components["model"] = b"changed"
        counts["optimizer"] = 400
        metadata["nested"]["values"].append(3)
        item.metadata["nested"]["values"].append(4)
        item.component_hashes["model"] = "changed"
        self.assertEqual(item.digest, digest)
        self.assertEqual(item.metadata, {"nested": {"values": [1, 2]}})
        with self.assertRaises(TypeError):
            item.components["model"] = b"changed"
        with self.assertRaises(TypeError):
            item.optimizer_step_counts["optimizer"] = 3
        with self.assertRaises(FrozenInstanceError):
            item.optimizer_steps = 5

    def test_component_presence_bytes_and_na_declarations(self):
        values = dict(checkpoint().components)
        values.pop("optimizer")
        with self.assertRaisesRegex(ValueError, "missing checkpoint"):
            checkpoint(components=values)
        values = dict(checkpoint().components)
        values["model"] = bytearray(b"mutable")
        with self.assertRaises(TypeError):
            checkpoint(components=values)
        for invalid in ({"not_applicable_components": "optimizer"},
                        {"not_applicable_components": ["missing"]},
                        {"not_applicable_components": ["rng", "rng"]}):
            with self.assertRaises((TypeError, ValueError)):
                checkpoint(metadata=invalid)
        values = dict(checkpoint().components)
        values["target_networks"] = b'{"applicable":false,"reason":"no targets"}'
        item = checkpoint(components=values, metadata={"not_applicable_components": ["target_networks"]})
        self.assertIn("target_networks", item.components)

    def test_total_and_individual_step_counts_are_strict(self):
        for invalid in (True, 2., np.int64(2), "2"):
            with self.assertRaises(TypeError):
                checkpoint(steps=invalid, optimizer_step_counts={"optimizer": 2})
            with self.assertRaises(TypeError):
                checkpoint(optimizer_step_counts={"optimizer": invalid})
        with self.assertRaises(ValueError):
            checkpoint(optimizer_step_counts={"optimizer": 1})
        with self.assertRaises(ValueError):
            checkpoint(optimizer_step_counts={"optimizer": -1, "other": 3})
        self.assertEqual(checkpoint(steps=0, optimizer_step_counts={}, metadata={}).optimizer_steps, 0)

    def test_primary_count_is_distinct_from_total_and_requires_declared_key(self):
        item = checkpoint(steps=7, optimizer_step_counts={"critic": 4, "actor": 2, "temperature": 1})
        self.assertEqual(item.optimizer_steps, 7)
        self.assertEqual(item.primary_optimizer_steps, 4)
        self.assertEqual(checkpoint(metadata={}).primary_optimizer_steps, None)
        with self.assertRaisesRegex(ValueError, "must name an optimizer"):
            checkpoint(metadata={"primary_optimizer": "missing"})

    def test_payload_tamper_and_rehashed_component_tamper_are_refused(self):
        document = json.loads(checkpoint().to_bytes())
        document["payload"]["optimizer_steps"] = 99
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            LearnerCheckpoint.from_bytes(json.dumps(document).encode())
        document = json.loads(checkpoint().to_bytes())
        document["payload"]["components"]["model"] = base64.b64encode(b"changed").decode()
        with self.assertRaisesRegex(ValueError, "component hash mismatch"):
            LearnerCheckpoint.from_bytes(rehash_envelope(document))

    def test_rehashed_bad_base64_schema_and_counts_are_refused(self):
        document = json.loads(checkpoint().to_bytes())
        document["payload"]["components"]["model"] = "!not base64!"
        with self.assertRaises(ValueError):
            LearnerCheckpoint.from_bytes(rehash_envelope(document))
        document = json.loads(checkpoint().to_bytes())
        document["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema version"):
            LearnerCheckpoint.from_bytes(json.dumps(document).encode())
        document = json.loads(checkpoint().to_bytes())
        document["payload"]["optimizer_step_counts"]["actor"] = True
        with self.assertRaises(ValueError):
            LearnerCheckpoint.from_bytes(rehash_envelope(document))

    def test_duplicate_json_keys_and_nonfinite_values_are_refused(self):
        raw = checkpoint().to_bytes()
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            LearnerCheckpoint.from_bytes(raw.replace(b'"schema_version":1',
                                                     b'"schema_version":1,"schema_version":1'))
        with self.assertRaises(ValueError):
            checkpoint(metadata={"bad": float("nan")})
        with self.assertRaises(TypeError):
            checkpoint(metadata={"bad": object()})
        with self.assertRaises(ValueError):
            LearnerCheckpoint.from_bytes(b"\x80pickle is not a supported format")

    def test_exclusive_file_creation_and_guarded_saved_alias(self):
        item = checkpoint()
        saved_alias = item.save
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            self.assertEqual(item.save(path), path)
            self.assertEqual(LearnerCheckpoint.load(path).digest, item.digest)
            old = path.read_bytes()
            with self.assertRaises(FileExistsError):
                checkpoint(steps=4).save(path)
            self.assertEqual(path.read_bytes(), old)
            forbidden = Path(temporary) / "forbidden.json"
            with inference_evaluation_guard() as ledger:
                with self.assertRaises(DatasetError):
                    saved_alias(forbidden)
                self.assertEqual(ledger["forbidden_api_calls"], 1)
            self.assertFalse(forbidden.exists())


class TransitionBatchTests(unittest.TestCase):
    def test_arrays_are_copied_and_cannot_be_made_writable(self):
        observations = np.ones((3, 4), dtype=np.float32)
        item = batch(observations=observations)
        observations[0, 0] = 999
        self.assertEqual(item.observations[0, 0], 1)
        for name in item.__dataclass_fields__:
            array = getattr(item, name)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(ValueError):
            item.actions[0, 0] = 99
        retrieved = item.observations
        retrieved.shape = (12,)
        self.assertEqual(item.observations.shape, (3, 4))
        retrieved.dtype = np.uint8
        self.assertEqual(item.observations.dtype, np.dtype(np.float32))

    def test_select_and_concat_preserve_executed_actions_origins_and_groups(self):
        item = batch()
        selected = item.select([2, 0])
        np.testing.assert_array_equal(selected.actions, item.actions[[2, 0]])
        np.testing.assert_array_equal(selected.origin, [1, 0])
        np.testing.assert_array_equal(selected.group_ids, [20, 10])
        combined = selected.concat(item.select([1]))
        self.assertEqual(len(combined), 3)
        np.testing.assert_array_equal(combined.group_ids, [20, 10, 10])
        self.assertEqual(len(item.select(0)), 1)
        self.assertEqual(len(item.select(slice(0, 0))), 0)
        self.assertEqual(len(item.select([True, False, True])), 2)

    def test_true_termination_and_time_limit_have_different_bootstrap_masks(self):
        item = batch()
        self.assertEqual(item.bootstrap_mask[1], 0)
        self.assertTrue(item.terminated[1])
        self.assertEqual(item.bootstrap_mask[2], 1)
        self.assertTrue(item.truncated[2])
        with self.assertRaisesRegex(ValueError, "truncation still bootstraps"):
            batch(bootstrap_mask=np.array([1., 0., 0.]))

    def test_shapes_flags_origins_object_arrays_and_nonfinite_values_are_refused(self):
        invalid = (
            {"rewards": np.zeros((3, 1))},
            {"actions": np.zeros((2, 2))},
            {"next_observations": np.zeros((3, 5))},
            {"terminated": np.array([0, 1, 0])},
            {"origin": np.array([0, 2, 1])},
            {"origin": np.array([0., 0., 1.])},
            {"group_ids": np.array([1., 2., 3.])},
            {"group_ids": np.array([object(), object(), object()], dtype=object)},
            {"rewards": np.array([1., np.nan, 3.])},
        )
        for fields in invalid:
            with self.subTest(fields=list(fields)):
                with self.assertRaises((TypeError, ValueError)):
                    batch(**fields)


class MachineContractTests(unittest.TestCase):
    def test_machine_contract_agrees_with_public_defaults(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads((root / "contracts/training_modes_v1.json").read_text())
        self.assertEqual(contract["offline_budget"]["default_optimizer_steps"], OfflineBudget().updates)
        self.assertEqual(contract["offline_budget"]["maximum_optimizer_steps"], MAX_OFFLINE_UPDATES)
        self.assertTrue(contract["evaluation"]["frozen_policy_default"])
        self.assertEqual(set(contract["checkpoint"]["required_components"]), REQUIRED_CHECKPOINT_COMPONENTS)
        self.assertFalse(contract["frozen_boundary"]["arbitrary_code_purity_proven"])


if __name__ == "__main__":
    unittest.main()
