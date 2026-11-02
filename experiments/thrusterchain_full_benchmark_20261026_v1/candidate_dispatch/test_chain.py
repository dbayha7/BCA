"""CPU-only draft checks against actual on-disk mixed-engine receipts.

The study runtime gate is stubbed to a fixed CPU test identity. No GPU or full
benchmark is started. GPU parity and production integration remain pending.
"""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import study
from candidate_dispatch import chain, dispatcher
from host_integration.cql_adapter import CQLAdapter
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_protocol import LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import _rng, _with_runner_state, train_offline


class ChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="thruster-mixed-chain-")
        cls.root = Path(cls.tmp.name) / "study"
        cls.root.mkdir()
        cls.spec = study.read_json(Path("validation/smoke_spec_20261027_v3.json"))
        cls.case = cls.spec["cases"][0]
        cls.runtime = {"scope": "CPU-only deterministic test stub"}
        cls.binding = dispatcher.binding()
        cls.manifest = {"schema": chain.SCHEMA, "spec": cls.spec,
                        "sources": study.source_identity(), "runtime": cls.runtime,
                        "dispatch_binding": cls.binding, "chain_verifier_sha256": chain.source_sha256()}
        study.write_new(cls.root / "manifest.json", cls.manifest)
        cls.digest = study.sha(cls.root / "manifest.json")
        (cls.root / "manifest.sha256").write_text(cls.digest + "\n", encoding="ascii")
        folder = cls.root / "datasets" / cls.case["id"]
        folder.mkdir(parents=True)
        cls.dataset = generate_behavior_dataset(folder / "data.npz", behavior=cls.case["behavior"],
            episodes=cls.spec["episodes_per_dataset"], seed=cls.case["data_seed"], provenance="smoke")
        study.write_new(folder / "receipt.json", {"case": cls.case, "dataset_sha256": cls.dataset.digest,
            "file_sha256": study.sha(folder / "data.npz"), "train_split_sha256": cls.dataset.split_digest("train"),
            "episodes": cls.spec["episodes_per_dataset"], "schema": cls.dataset.metadata["schema"]})
        for method in dispatcher.ROUTES:
            prev, prior_receipt = None, None
            for step in (2, 4):
                cp, native = dispatcher.train_chunk(cls.dataset, method, cls.case["train_seed"], 8, 2,
                                                    cls.binding, prev, prior_receipt)
                out = cls.root / "training" / cls.case["id"] / method / f"round_{step:07d}"
                out.mkdir(parents=True)
                cp.save(out / "checkpoint.json")
                study.write_new(out / "receipt.json", {"schema": "thrusterchain.study_chunk.v1", "case": cls.case,
                    "method": method, "critic_rounds": step, "chunk_critic_rounds": 2,
                    "primary_steps_total": dispatcher._steps(method, step),
                    "optimizer_counts": dict(cp.optimizer_step_counts), "checkpoint_sha256": cp.digest,
                    "checkpoint_file_sha256": study.sha(out / "checkpoint.json"), "wall_seconds": .001,
                    "native_receipt": native})
                prev, prior_receipt = cp, native

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, method="td3bc", terminal=True, digest=None):
        with patch.object(study, "environment", return_value=self.runtime):
            return chain.verify(self.root, self.digest if digest is None else digest,
                                self.case["id"], method, terminal=terminal)

    def round(self, method, number):
        return self.root / "training" / self.case["id"] / method / f"round_{number:07d}"

    def change(self, path, content, callback):
        original = path.read_bytes()
        try:
            path.write_bytes(content)
            callback()
        finally:
            path.write_bytes(original)

    def test_complete_all_routes_and_explicit_nonterminal(self):
        for method in dispatcher.ROUTES:
            with self.subTest(method=method):
                report = self.check(method)
                self.assertEqual(report["chunks_verified"], 2)
                self.assertEqual(report["critic_rounds_verified"], 4)
        second = self.round("td3bc", 4)
        hidden = second.with_name("hidden_second")
        second.rename(hidden)
        try:
            self.assertEqual(self.check(terminal=False)["critic_rounds_verified"], 2)
            with self.assertRaises(RuntimeError): self.check(terminal=True)
        finally:
            hidden.rename(second)
        with self.assertRaises(ValueError): self.check(terminal=None)

    def test_holes_invalid_rounds_and_no_manifest_self_attestation(self):
        first = self.round("td3bc", 2)
        hidden = first.with_name("hidden_first")
        first.rename(hidden)
        try:
            for terminal in (True, False):
                with self.assertRaises(RuntimeError): self.check(terminal=terminal)
        finally:
            hidden.rename(first)
        extra = first.parent / "round_0000003"
        extra.mkdir()
        try:
            with self.assertRaises(RuntimeError): self.check(terminal=False)
        finally:
            extra.rmdir()
        with self.assertRaises(RuntimeError): self.check(digest="0" * 64)
        with patch.object(study, "environment", return_value={"other": "drift"}):
            with self.assertRaises(RuntimeError):
                chain.verify(self.root, self.digest, self.case["id"], "td3bc", terminal=True)

    def test_failure_and_pending_never_become_resumable(self):
        parent = self.round("td3bc", 2).parent
        marker = parent / "failure.json"
        marker.write_text('{"terminal":true}', encoding="utf8")
        try:
            for terminal in (True, False):
                with self.assertRaisesRegex(RuntimeError, "terminal training failure"):
                    self.check(terminal=terminal)
            last = self.round("td3bc", 4)
            hidden = last.with_name("hidden_second")
            last.rename(hidden)
            try:
                with self.assertRaisesRegex(RuntimeError, "terminal training failure"):
                    self.check(terminal=False)
            finally:
                hidden.rename(last)
        finally:
            marker.unlink()
        pending = parent / ".pending-interrupted"
        pending.mkdir()
        try:
            for terminal in (True, False):
                with self.assertRaisesRegex(RuntimeError, "pending training publication"):
                    self.check(terminal=terminal)
        finally:
            pending.rmdir()

    def test_changes_during_verification_are_detected(self):
        verify_receipt = dispatcher.verify_published_receipt
        receipt = self.round("td3bc", 2) / "receipt.json"
        dataset = self.root / "datasets" / self.case["id"] / "data.npz"
        dataset_receipt = dataset.parent / "receipt.json"
        for target in (receipt, dataset, dataset_receipt):
            original = target.read_bytes()
            changed = [False]
            def mutate(*args, **kwargs):
                value = verify_receipt(*args, **kwargs)
                if not changed[0]:
                    target.write_bytes(original + b" ")
                    changed[0] = True
                return value
            try:
                with patch.object(dispatcher, "verify_published_receipt", side_effect=mutate):
                    with self.assertRaises(RuntimeError): self.check()
                self.assertTrue(changed[0])
            finally:
                target.write_bytes(original)

    def test_valid_but_wrong_dataset_is_rejected(self):
        folder = self.root / "datasets" / self.case["id"]
        data_path, receipt_path = folder / "data.npz", folder / "receipt.json"
        old_data, old_receipt = data_path.read_bytes(), receipt_path.read_bytes()
        alternatives = [
            {"behavior": "feedback_weak"},
            {"seed": self.case["data_seed"] + 1},
            {"episodes": 11},
            {"stall": False},
            {"task_semantics": "continuing_time_limit"},
        ]
        try:
            for index, overrides in enumerate(alternatives):
                with self.subTest(overrides=overrides):
                    args = {"behavior": self.case["behavior"], "seed": self.case["data_seed"],
                            "episodes": self.spec["episodes_per_dataset"], "provenance": "smoke"}
                    args.update(overrides)
                    alt_path = Path(self.tmp.name) / f"wrong-{index}.npz"
                    wrong = generate_behavior_dataset(alt_path, **args)
                    shutil.copyfile(alt_path, data_path)
                    receipt = {"case": self.case, "dataset_sha256": wrong.digest,
                        "file_sha256": study.sha(data_path), "train_split_sha256": wrong.split_digest("train"),
                        "episodes": self.spec["episodes_per_dataset"], "schema": wrong.metadata["schema"]}
                    receipt_path.write_bytes(study.encoded(receipt))
                    # Even a self-consistent receipt cannot substitute a valid
                    # dataset whose declared behavior, seed or task differs.
                    with self.assertRaisesRegex(RuntimeError, "dataset metadata"):
                        study.data_for(self.root, self.case)
                    with self.assertRaisesRegex(RuntimeError, "dataset metadata"):
                        self.check()
        finally:
            data_path.write_bytes(old_data); receipt_path.write_bytes(old_receipt)

    def test_dataset_and_checkpoint_bytes_are_bound(self):
        dataset = self.root / "datasets" / self.case["id"] / "data.npz"
        self.change(dataset, dataset.read_bytes() + b"changed", lambda: self.assert_bad())
        checkpoint = self.round("td3bc", 2) / "checkpoint.json"
        self.change(checkpoint, checkpoint.read_bytes() + b"changed", lambda: self.assert_bad())

    def assert_bad(self):
        with self.assertRaises(Exception): self.check()

    def test_rehashed_predecessor_receipt_is_rejected(self):
        receipt_path = self.round("td3bc", 4) / "receipt.json"
        receipt = study.read_json(receipt_path)
        native = receipt["native_receipt"]
        native["resume_from_sha256"] = "0" * 64
        native["receipt_sha256"] = dispatcher._hash({k: v for k, v in native.items() if k != "receipt_sha256"})
        self.change(receipt_path, study.encoded(receipt), lambda: self.assert_bad())

    def test_foreign_engine_cannot_enter_a_chain(self):
        target = self.round("td3bc", 2)
        source = self.round("cql", 2)
        cp_path, receipt_path = target / "checkpoint.json", target / "receipt.json"
        old_cp, old_receipt = cp_path.read_bytes(), receipt_path.read_bytes()
        try:
            shutil.copyfile(source / "checkpoint.json", cp_path)
            shutil.copyfile(source / "receipt.json", receipt_path)
            with self.assertRaises(RuntimeError): self.check()
        finally:
            cp_path.write_bytes(old_cp); receipt_path.write_bytes(old_receipt)

    def test_cql_native_training_cannot_enter_candidate_chain_on_disk(self):
        first = LearnerCheckpoint.load(self.round("cql", 2) / "checkpoint.json")
        mixed, native = train_offline(self.dataset, CQLAdapter, budget=OfflineBudget(2, 8),
                                      seed=self.case["train_seed"], resume_from=first)
        folder = self.round("cql", 4)
        cp_path, receipt_path = folder / "checkpoint.json", folder / "receipt.json"
        old_cp, old_receipt = cp_path.read_bytes(), receipt_path.read_bytes()
        try:
            cp_path.write_bytes(mixed.to_bytes())
            receipt = study.read_json(receipt_path)
            receipt.update(checkpoint_sha256=mixed.digest, checkpoint_file_sha256=study.sha(cp_path),
                           optimizer_counts=dict(mixed.optimizer_step_counts), native_receipt=native)
            receipt_path.write_bytes(study.encoded(receipt))
            with self.assertRaisesRegex(RuntimeError, "frozen CQL candidate route"):
                self.check("cql")
        finally:
            cp_path.write_bytes(old_cp); receipt_path.write_bytes(old_receipt)

    def test_pinned_adapter_configuration_and_all_optimizer_schedules(self):
        learner_seed = int(_rng(self.case["train_seed"], "learner_initialization").integers(0, 2**63 - 1))
        obs_dim = self.dataset.arrays["observations"].shape[-1]
        for method in dispatcher.ROUTES:
            cp = LearnerCheckpoint.load(self.round(method, 2) / "checkpoint.json")
            chain._validate_configuration(cp, method, learner_seed, obs_dim, 2)
            wrong_id = LearnerCheckpoint("wrong-adapter", cp.components, cp.optimizer_steps,
                                         cp.optimizer_step_counts, cp.metadata)
            with self.assertRaisesRegex(RuntimeError, "adapter ID"):
                chain._validate_configuration(wrong_id, method, learner_seed, obs_dim, 2)
            counts = dict(cp.optimizer_step_counts)
            non_primary = next(k for k in counts if k != cp.primary_optimizer)
            counts[non_primary] += 1
            wrong_schedule = LearnerCheckpoint(cp.adapter_id, cp.components, sum(counts.values()), counts, cp.metadata)
            with self.assertRaisesRegex(RuntimeError, "optimizer counters"):
                chain._validate_configuration(wrong_schedule, method, learner_seed, obs_dim, 2)
            metadata = cp.metadata
            metadata["adapter"]["args"]["seed"] += 1
            wrong_args = LearnerCheckpoint(cp.adapter_id, cp.components, cp.optimizer_steps,
                                           cp.optimizer_step_counts, metadata)
            with self.assertRaisesRegex(RuntimeError, "resolved configuration"):
                chain._validate_configuration(wrong_args, method, learner_seed, obs_dim, 2)

    def test_independent_sampler_replay_detects_rehashed_cursor_change(self):
        folder = self.round("td3bc", 2)
        cp_path, receipt_path = folder / "checkpoint.json", folder / "receipt.json"
        old_cp, old_receipt = cp_path.read_bytes(), receipt_path.read_bytes()
        cp = LearnerCheckpoint.load(cp_path)
        state = copy.deepcopy(cp.metadata["runner_state"])
        state["sampler_rng_state"] = _rng(self.case["train_seed"] + 1, "offline_batches").bit_generator.state
        forged = _with_runner_state(cp, state)
        receipt = study.read_json(receipt_path)
        native = receipt["native_receipt"]
        native["checkpoint_sha256"] = forged.digest
        native["receipt_sha256"] = dispatcher._hash({k: v for k, v in native.items() if k != "receipt_sha256"})
        try:
            cp_path.write_bytes(forged.to_bytes())
            receipt["checkpoint_sha256"] = forged.digest
            receipt["checkpoint_file_sha256"] = study.sha(cp_path)
            receipt_path.write_bytes(study.encoded(receipt))
            with self.assertRaisesRegex(RuntimeError, "PCG64"): self.check()
        finally:
            cp_path.write_bytes(old_cp); receipt_path.write_bytes(old_receipt)


if __name__ == "__main__":
    unittest.main()
