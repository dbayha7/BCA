import copy
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import sys
import unittest
from unittest.mock import patch

import study


ROOT = Path(__file__).resolve().parent


class StudyContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = study.read_json(ROOT / "spec.full.json")

    def test_authorized_matrix_and_primary_accounting(self):
        study.validate_spec(self.spec)
        self.assertEqual(len(self.spec["cases"]) * len(self.spec["methods"]), 60)
        self.assertEqual(len(self.spec["starts"]) * len(self.spec["shifts"]) * len(self.spec["evaluation_seeds"]), 80)
        self.assertEqual(study.primary_steps("unifloral-unified", 1_000_000), 500_000)
        self.assertEqual(study.primary_steps("td3bc", 1_000_000), 1_000_000)
        self.assertEqual(study.primary_steps("unifloral-unified", 10_000), 5_000)

    def test_bad_budgets_seed_pairing_and_ids_refused(self):
        for change in (lambda s: s.update(critic_rounds=999_999),
                       lambda s: s["cases"][1].update(data_seed=s["cases"][0]["data_seed"]),
                       lambda s: s["cases"][3].update(train_seed=1),
                       lambda s: s["cases"][0].update(id="../escape"),
                       lambda s: s["evaluation_seeds"].append(s["evaluation_seeds"][0])):
            x = copy.deepcopy(self.spec); change(x)
            with self.assertRaises(ValueError): study.validate_spec(x)

    def test_twin_critic_and_unified_counts_are_checked(self):
        def cp(primary, **counts): return types.SimpleNamespace(primary_optimizer=primary, optimizer_step_counts=counts)
        study.verify_counts("unifloral-unified", cp("actor", actor=5, critic=10), 10)
        study.verify_counts("cql", cp("critic1", critic1=10, critic2=10), 10)
        with self.assertRaises(RuntimeError): study.verify_counts("unifloral-unified", cp("actor", actor=5, critic=9), 10)
        with self.assertRaises(RuntimeError): study.verify_counts("cql", cp("critic1", critic1=10, critic2=9), 10)

    def test_dataset_loader_rejects_valid_data_from_wrong_behavior_or_seed(self):
        from thrusterchain.quality_datasets import generate_behavior_dataset
        case = {"id": "fixture", "behavior": "random", "data_seed": 641,
                "train_seed": 642}
        spec = {"profile": "smoke", "cases": [case], "episodes_per_dataset": 10}
        with TemporaryDirectory() as temp, patch.object(study, "read_manifest", return_value={"spec": spec}):
            root = Path(temp)
            for index, (behavior, seed, accepted) in enumerate((
                    ("random", 641, True), ("feedback_strong", 641, False),
                    ("random", 643, False))):
                folder = root / str(index) / "datasets" / case["id"]
                folder.mkdir(parents=True)
                data = generate_behavior_dataset(folder / "data.npz", behavior=behavior, episodes=10,
                                                 seed=seed, provenance="research")
                study.write_new(folder / "receipt.json", {"case": case, "dataset_sha256": data.digest,
                    "file_sha256": study.sha(folder / "data.npz"),
                    "train_split_sha256": data.split_digest("train"), "episodes": 10,
                    "schema": data.metadata["schema"]})
                if accepted:
                    self.assertEqual(study.data_for(root / str(index), case).digest, data.digest)
                    study.collect(root / str(index), spec)
                else:
                    with self.assertRaisesRegex(RuntimeError, "frozen case and study"):
                        study.data_for(root / str(index), case)
                    with self.assertRaisesRegex(RuntimeError, "frozen case and study"):
                        study.collect(root / str(index), spec)

    def test_dataset_metadata_and_receipt_match_pinned_study_contract(self):
        from thrusterchain.quality_datasets import generate_behavior_dataset
        case = {"id": "fixture", "behavior": "random", "data_seed": 641,
                "train_seed": 642}
        data = generate_behavior_dataset(behavior="random", episodes=10, seed=641, provenance="research")
        spec = {"profile": "full", "cases": [case], "episodes_per_dataset": 10}
        receipt = {"case": case, "dataset_sha256": data.digest, "file_sha256": "fixture",
                   "train_split_sha256": data.split_digest("train"), "episodes": 10,
                   "schema": data.metadata["schema"]}
        study.validate_case_dataset(data, receipt, case, spec)
        changes = (
            lambda m: m.update(episodes=11), lambda m: m.update(horizon=99),
            lambda m: m.update(gamma=.95), lambda m: m.update(task_semantics="continuing_time_limit"),
            lambda m: m.update(observation_mode="physical_padded"), lambda m: m.update(start_family="big_waves"),
            lambda m: m.update(physics={"stall": False, "stall_slope": 1.5}),
            lambda m: m.update(provenance="smoke"), lambda m: m.update(schema="other"),
        )
        for mutate in changes:
            with self.subTest(mutation=mutate):
                metadata = copy.deepcopy(data.metadata)
                mutate(metadata)
                other = types.SimpleNamespace(metadata=metadata, digest=data.digest,
                                              split_digest=data.split_digest)
                with self.assertRaisesRegex(RuntimeError, "frozen case and study"):
                    study.validate_case_dataset(other, receipt, case, spec)
        for bad in (dict(receipt, episodes=11), dict(receipt, schema="other"),
                    dict(receipt, train_split_sha256="other"), dict(receipt, extra="unbound")):
            with self.assertRaisesRegex(RuntimeError, "frozen case and study"):
                study.validate_case_dataset(data, bad, case, spec)
        smoke = dict(spec, profile="smoke")
        metadata = copy.deepcopy(data.metadata); metadata["provenance"] = "smoke"
        other = types.SimpleNamespace(metadata=metadata, digest=data.digest, split_digest=data.split_digest)
        study.validate_case_dataset(other, receipt, case, smoke)

    def test_native_receipt_hash_and_checkpoint_lineage(self):
        from thrusterchain.training_runner import _hash
        engine = study.sha(ROOT / "compiled" / "engine.py")
        def cp(digest, critic):
            return types.SimpleNamespace(digest=digest, adapter_id="adapter", primary_optimizer="critic",
                optimizer_step_counts={"critic": critic}, metadata={"runner_state": {
                    "engine": "compiled-native-scan-v1", "engine_sha256": engine}})
        first, second = cp("first", 10), cp("second", 20)
        data = {"dataset_sha256": "data", "train_split_sha256": "split"}
        spec = {"chunk_critic_rounds": 10, "batch_size": 3}
        native = {"schema": "thrusterchain.compiled_chunk.v1", "method": "td3bc",
            "engine": "compiled-native-scan-v1", "engine_sha256": engine, "dataset_sha256": "data",
            "split_sha256": {"train": "split"}, "preprocessing_source": "train_only",
            "learner_visible_split": "train", "adapter_id": "adapter", "primary_optimizer": "critic",
            "executed_primary_optimizer_updates": 10, "optimizer_step_counts": {"critic": 20},
            "optimizer_step_deltas": {"critic": 10}, "executed_optimizer_steps_total": 10,
            "batch_size": 3, "dataset_draws": 30, "environment_interactions": 0,
            "checkpoint_sha256": "second", "resume_from_sha256": "first"}
        native["receipt_sha256"] = _hash(native)
        study.verify_native_receipt(native, second, first, data, "td3bc", spec)
        tampered = dict(native, dataset_draws=31)
        with self.assertRaisesRegex(RuntimeError, "lineage"):
            study.verify_native_receipt(tampered, second, first, data, "td3bc", spec)
        spliced = dict(native, resume_from_sha256="different_checkpoint")
        spliced["receipt_sha256"] = _hash({k: v for k, v in spliced.items() if k != "receipt_sha256"})
        with self.assertRaisesRegex(RuntimeError, "lineage"):
            study.verify_native_receipt(spliced, second, first, data, "td3bc", spec)
        with self.assertRaisesRegex(RuntimeError, "lineage"):
            study.verify_native_receipt(native, second, None, data, "td3bc", spec)

    def test_atomic_publication_refuses_overwrite_and_preserves_partial(self):
        with TemporaryDirectory() as root:
            parent = Path(root); target = parent / "completed"
            with study.publish_directory(target) as temp:
                study.write_new(temp / "payload.json", {"value": 1})
            self.assertEqual(study.read_json(target / "payload.json"), {"value": 1})
            with self.assertRaises(FileExistsError):
                with study.publish_directory(target): pass
            with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                with study.publish_directory(parent / "failed") as temp:
                    study.write_new(temp / "partial.json", {"value": 2})
                    raise RuntimeError("controlled failure")
            self.assertEqual(len(list(parent.glob(".pending-*"))), 1)
            with self.assertRaisesRegex(RuntimeError, "requires inspection"):
                with study.publish_directory(parent / "failed"): pass

    def test_global_barrier_allows_recorded_failures_but_not_missing(self):
        spec = {"cases": [{"id": "c1"}], "methods": ["td3bc", "iql"], "critic_rounds": 10}
        def checkpoints(_study, _case, method, _spec):
            return [(10, object())] if method == "td3bc" else []
        with TemporaryDirectory() as root, patch.object(study, "checkpoints", side_effect=checkpoints):
            root = Path(root)
            with self.assertRaisesRegex(RuntimeError, "evaluation blocked"):
                study.verify_barrier(root, spec)
            failure = root / "training/c1/iql/failure.json"; failure.parent.mkdir(parents=True)
            study.write_new(failure, {"case": spec["cases"][0], "method": "iql", "terminal": True})
            study.verify_barrier(root, spec)

    def test_json_duplicate_and_nonfinite_values_rejected(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "x.json"
            path.write_text('{"x":1,"x":2}', encoding="utf8")
            with self.assertRaises(ValueError): study.read_json(path)
            path.write_text('{"x":NaN}', encoding="utf8")
            with self.assertRaises(ValueError): study.read_json(path)

    def test_runtime_identity_blocks_compiler_and_jax_config_drift(self):
        config = types.SimpleNamespace(jax_disable_jit=False, jax_enable_x64=False,
            jax_default_matmul_precision="highest", jax_default_prng_impl="threefry2x32",
            jax_threefry_partitionable=True)
        jax = types.SimpleNamespace(config=config)
        with patch.dict(os.environ, {"XLA_FLAGS": "--first", "JAX_PLATFORMS": "cuda"}, clear=True):
            original = study.execution_settings(jax)
            manifest = {"sources": {"source": "bound"}, "runtime": {"execution_settings": original}}
            with patch.object(study, "read_manifest", return_value=manifest), \
                 patch.object(study, "source_identity", return_value={"source": "bound"}), \
                 patch.object(study, "environment", side_effect=lambda: {"execution_settings": study.execution_settings(jax)}):
                self.assertEqual(study.load_manifest(Path("unused")), manifest)
                os.environ["XLA_FLAGS"] = "--different"
                with self.assertRaisesRegex(RuntimeError, "runtime changed"):
                    study.load_manifest(Path("unused"))
                os.environ["XLA_FLAGS"] = "--first"
                config.jax_disable_jit = True
                with self.assertRaisesRegex(RuntimeError, "runtime changed"):
                    study.load_manifest(Path("unused"))
                config.jax_disable_jit = False
                config.jax_enable_x64 = True
                with self.assertRaisesRegex(RuntimeError, "runtime changed"):
                    study.load_manifest(Path("unused"))

    def test_one_failed_cell_does_not_skip_other_cells_or_retry(self):
        case = {"id": "a", "behavior": "random", "data_seed": 1, "train_seed": 2}
        spec = {"cases": [case], "methods": ["td3bc", "iql"], "critic_rounds": 2,
                "chunk_critic_rounds": 2, "batch_size": 3}
        calls = []
        class FakeCheckpoint:
            primary_optimizer = "critic"
            optimizer_step_counts = {"critic": 2}
            digest = "fixture"
            def save(self, path): Path(path).write_text("checkpoint", encoding="utf8")
        def train_chunk(data, method, seed, batch, count, checkpoint):
            calls.append(method)
            if method == "td3bc": raise FloatingPointError("controlled nonfinite")
            return FakeCheckpoint(), {"sampled_draws": batch * count}
        package = types.ModuleType("compiled"); package.__path__ = []
        engine = types.ModuleType("compiled.engine"); engine.train_chunk = train_chunk
        with TemporaryDirectory() as root, patch.dict(sys.modules, {"compiled": package, "compiled.engine": engine}), \
             patch.object(study, "data_for", return_value=object()), patch.object(study, "verify_data_seeds"), \
             patch.object(study, "checkpoints", return_value=[]), patch.object(study, "load_manifest"), \
             patch.object(study, "verify_native_receipt"), \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(root)
            receipt = root / "datasets/a/receipt.json"; receipt.parent.mkdir(parents=True)
            study.write_new(receipt, {})
            study.train_all(root, spec, resume=False)
            failure = study.read_json(root / "training/a/td3bc/failure.json")
            self.assertEqual(failure["exception_type"], "FloatingPointError")
            self.assertTrue((root / "training/a/iql/round_0000002/checkpoint.json").is_file())
            self.assertEqual(calls, ["td3bc", "iql"])
            spec["methods"] = ["td3bc"]
            study.train_all(root, spec, resume=True)
            self.assertEqual(calls, ["td3bc", "iql"])

    def test_report_distinguishes_failed_attempt_from_checkpointed_costs(self):
        random = {"id": "r", "behavior": "random", "data_seed": 1, "train_seed": 2}
        strong = {"id": "s", "behavior": "feedback_strong", "data_seed": 1, "train_seed": 2}
        spec = {"cases": [random, strong], "methods": ["td3bc"], "starts": ["recorded", "strong_current"],
                "shifts": ["baseline"], "evaluation_seeds": [4, 5], "batch_size": 3, "critic_rounds": 2}
        cp = types.SimpleNamespace(digest="model", optimizer_step_counts={"critic": 2})
        dataset = types.SimpleNamespace(digest="dataset")
        def checkpoints(_root, case, _method, _spec):
            return [] if case["id"] == "r" else [(2, cp)]
        with TemporaryDirectory() as temp, patch.object(study, "verify_barrier"), \
             patch.object(study, "data_for", return_value=dataset), patch.object(study, "checkpoints", side_effect=checkpoints):
            root = Path(temp)
            fail = root / "training/r/td3bc/failure.json"; fail.parent.mkdir(parents=True)
            study.write_new(fail, {"wall_seconds": 4.5, "exception_type": "FloatingPointError"})
            chunk = root / "training/s/td3bc/round_0000002"; chunk.mkdir(parents=True)
            study.write_new(chunk / "receipt.json", {"wall_seconds": 1.25})
            evaluation = root / "evaluation/s/td3bc"; evaluation.mkdir(parents=True)
            episodes = [{"episode_seed": seed, "steps": 100, "return": 1., "discounted_return": 1.} for seed in (4, 5)]
            block = {"policy_sha256_before": "x", "policy_sha256_after": "x", "scenarios": {"baseline": {"episodes": episodes}}}
            study.write_evaluation_record(evaluation, "result.json", {"case": strong, "method": "td3bc", "checkpoint_sha256": "model",
                "dataset_sha256": "dataset", "starts": {"recorded": block, "strong_current": block}})
            study.report(root, spec)
            result = study.read_json(root / "report.json")
            failed = next(x for x in result["training_costs"] if x["case"] == "r")
            self.assertEqual(failed["sampled_transition_draws_checkpointed"], 0)
            self.assertEqual(failed["failed_attempt_updates_and_draws"], "unknown")
            self.assertEqual(failed["failed_attempt_wall_seconds"], 4.5)
            completed = next(x for x in result["training_costs"] if x["case"] == "s")
            self.assertEqual(completed["sampled_transition_draws_checkpointed"], 6)
            self.assertIsNone(result["hypotheses"]["H1"]["expected_direction_met"])

    def test_evaluation_digest_detects_changed_record(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            study.write_evaluation_record(root, "result.json", {"return": 1})
            self.assertEqual(study.evaluation_record(root, "result.json"), {"return": 1})
            (root / "result.json").write_text('{"return": 2}', encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                study.evaluation_record(root, "result.json")

    @unittest.skipUnless(os.name == "posix", "advisory flock is a POSIX runtime feature")
    def test_read_only_status_works_while_writer_holds_lock(self):
        with TemporaryDirectory() as temp, patch.object(study, "read_manifest", return_value={"spec": self.spec}), \
             patch.object(study, "status", return_value={"training_complete": 1}), \
             patch.object(sys, "argv", ["study.py", "status", "--study", temp]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with study.locked(Path(temp)):
                study.main()
            self.assertEqual(json.loads(output.getvalue())["training_complete"], 1)


if __name__ == "__main__": unittest.main()
