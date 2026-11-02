"""Small-budget executable checks for the v4 adapter runner."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from thrusterchain.datasets import (TrajectoryDataset, generate_dataset,
                                   inference_evaluation_guard)
from thrusterchain.sgd_adapter import AffineSGDBCAdapter
from thrusterchain.training_protocol import (AdapterCapabilities, EvaluationConfig,
                                            LearnerCheckpoint, OfflineBudget, OnlineBudget)
from thrusterchain.training_runner import (RunnerError, evaluate_frozen, run_mode,
                                          train_offline, train_transitions)


class RecordingAdapter(AffineSGDBCAdapter):
    def __init__(self):
        super().__init__()
        self.batches = []
        self.actions_seen = 0
        self.configurations = 0
        self.checkpoint_calls = 0

    def initialize(self, batch, seed, learning_settings):
        self.initial_groups = set(batch.group_ids.tolist())
        super().initialize(batch, seed, learning_settings)

    def update(self, batch):
        self.batches.append((batch, self.actions_seen))
        return super().update(batch)

    def act(self, observation, explore=False):
        self.actions_seen += 1
        return super().act(observation, explore)

    def configure_learning_settings(self, settings):
        self.configurations += 1
        super().configure_learning_settings(settings)

    def checkpoint(self):
        self.checkpoint_calls += 1
        return super().checkpoint()


class TrainingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=7601, provenance="smoke")
        cls.pretrained, cls.training_receipt = train_offline(
            cls.dataset, AffineSGDBCAdapter, budget=OfflineBudget(updates=3, batch_size=7), seed=8601)

    def factory(self):
        learners = []
        def create():
            learner = RecordingAdapter()
            learners.append(learner)
            return learner
        return create, learners

    def test_offline_never_constructs_environment_and_only_exposes_train(self):
        create, learners = self.factory()
        original_split = TrajectoryDataset.split
        def only_train(dataset, name):
            self.assertEqual(name, "train")
            return original_split(dataset, name)
        with patch.object(TrajectoryDataset, "split", only_train), patch(
                "thrusterchain.environment.ThrusterChainEnv", side_effect=AssertionError("offline env access")):
            checkpoint, receipt = train_offline(self.dataset, create, budget=OfflineBudget(5, 7), seed=42)
        train_groups = set(self.dataset.split("train")["group_ids"].tolist())
        self.assertEqual(learners[0].initial_groups, train_groups)
        for batch, actions_seen in learners[0].batches:
            self.assertLessEqual(set(batch.group_ids.tolist()), train_groups)
            self.assertFalse(batch.origin.any())
            self.assertEqual(actions_seen, 0)
            with self.assertRaises(ValueError):
                batch.observations.setflags(write=True)
        self.assertEqual(checkpoint.optimizer_steps, 5)
        self.assertEqual(receipt["budget"]["executed"]["updates"], 5)
        self.assertEqual(receipt["environment_interactions"], 0)
        self.assertEqual(receipt["dataset_sha256"], self.dataset.digest)
        # Full state serialization belongs only at run boundaries.
        self.assertLessEqual(learners[0].checkpoint_calls, 3)

    def test_resume_matches_uninterrupted_complete_checkpoint(self):
        whole, _ = train_offline(self.dataset, AffineSGDBCAdapter, budget=OfflineBudget(9, 11), seed=765)
        first, _ = train_offline(self.dataset, AffineSGDBCAdapter, budget=OfflineBudget(4, 11), seed=765)
        serialized = LearnerCheckpoint.from_bytes(first.to_bytes())
        second, receipt = train_offline(self.dataset, AffineSGDBCAdapter,
                budget=OfflineBudget(5, 11), seed=765, resume_from=serialized)
        self.assertEqual(second.to_bytes(), whole.to_bytes())
        self.assertEqual(receipt["resumed_from_sha256"], first.digest)
        self.assertEqual(receipt["optimizer_steps_before"], 4)
        self.assertEqual(receipt["budget"]["executed"]["updates"], 5)
        for kwargs in ({"seed": 99}, {"budget": OfflineBudget(1, 99)},
                       {"learning_settings": {"learning_rate": .1}}):
            request = {"budget": OfflineBudget(1, 11), "seed": 765, "resume_from": first}
            request.update(kwargs)
            with self.assertRaisesRegex(RunnerError, "resume requires matching"):
                train_offline(self.dataset, AffineSGDBCAdapter, **request)

    def test_overshooting_or_missing_updates_are_rejected(self):
        class DoubleUpdate(AffineSGDBCAdapter):
            def update(self, batch):
                super().update(batch)
                return super().update(batch)
        class MissingUpdate(AffineSGDBCAdapter):
            def update(self, batch):
                return {}
        for adapter in (DoubleUpdate, MissingUpdate):
            with self.assertRaisesRegex(RunnerError, "exactly one primary optimizer step"):
                train_offline(self.dataset, adapter, budget=OfflineBudget(1, 5))

    def test_primary_budget_counts_critic_and_delayed_actor_separately(self):
        class CriticActor:
            capabilities = AdapterCapabilities("critic_actor_fixture", "optimizer_steps", False,
                                                primary_optimizer="critic")
            def __init__(self):
                self.critic = AffineSGDBCAdapter()
                self.actor = 0.0
                self.actor_steps = 0
            @property
            def optimizer_steps(self):
                return self.critic.optimizer_steps + self.actor_steps
            @property
            def optimizer_step_counts(self):
                return {"critic": self.critic.optimizer_steps, "actor": self.actor_steps}
            def initialize(self, batch, seed, learning_settings):
                self.critic.initialize(batch, seed, learning_settings)
            def update(self, batch):
                metrics = self.critic.update(batch)
                if self.critic.optimizer_steps % 2:
                    self.actor -= .01 * 2 * (self.actor - float(np.mean(batch.actions)))
                    self.actor_steps += 1
                return metrics
            def checkpoint(self):
                checkpoint = self.critic.checkpoint()
                metadata = checkpoint.metadata
                metadata["primary_optimizer"] = "critic"
                components = dict(checkpoint.components)
                components["actor"] = str(self.actor).encode("ascii")
                return LearnerCheckpoint(self.capabilities.adapter_id, components, self.optimizer_steps,
                                         self.optimizer_step_counts, metadata)
        checkpoint, receipt = train_offline(self.dataset, CriticActor, budget=OfflineBudget(3, 5))
        self.assertEqual(checkpoint.optimizer_step_counts, {"critic": 3, "actor": 2})
        self.assertEqual(checkpoint.optimizer_steps, 5)
        self.assertEqual(receipt["primary_optimizer"], "critic")
        self.assertEqual(receipt["executed_primary_optimizer_updates"], 3)
        self.assertEqual(receipt["executed_optimizer_steps_total"], 5)

    def test_resume_cursor_consistency_and_cap_cannot_be_bypassed(self):
        primary = self.pretrained.primary_optimizer
        for cursor in (False, -1, 3.0, 0):
            metadata = self.pretrained.metadata
            metadata["runner_state"]["total_runner_updates"] = cursor
            bad = LearnerCheckpoint(self.pretrained.adapter_id, dict(self.pretrained.components),
                    3_000_000, {primary: 3_000_000}, metadata)
            with self.assertRaisesRegex(RunnerError, "cursor must equal"):
                train_offline(self.dataset, AffineSGDBCAdapter, budget=OfflineBudget(1, 7), seed=8601, resume_from=bad)
        metadata = self.pretrained.metadata
        metadata["runner_state"]["total_runner_updates"] = 3_000_000
        capped = LearnerCheckpoint(self.pretrained.adapter_id, dict(self.pretrained.components),
                                  3_000_000, {primary: 3_000_000}, metadata)
        with self.assertRaisesRegex(RunnerError, "cannot exceed"):
            train_offline(self.dataset, AffineSGDBCAdapter, budget=OfflineBudget(1, 7), seed=8601, resume_from=capped)

    def test_closed_form_gradient_budget_rejected_before_initialize(self):
        class ClosedForm:
            capabilities = AdapterCapabilities("closed-form-example", "closed_form", False)
            def initialize(self, *args, **kwargs):
                raise AssertionError("unsupported adapter was initialized")
        with self.assertRaisesRegex(RunnerError, "closed-form"):
            train_offline(self.dataset, ClosedForm, budget=OfflineBudget(1, 5))

    def test_frozen_default_has_no_updates_and_complete_clean_episodes(self):
        create, learners = self.factory()
        result = run_mode(self.dataset, self.pretrained, create, EvaluationConfig(episodes=1), families=("recorded",))
        self.assertEqual(result["receipt"]["status"], "frozen_policy")
        self.assertEqual(result["checkpoints"]["recorded"].digest, self.pretrained.digest)
        self.assertEqual(learners[0].actions_seen, 0)
        self.assertFalse(learners[0].batches)
        evaluation = result["frozen_evaluation"]["recorded"]
        self.assertEqual(evaluation["inference_calls"], 100)
        self.assertEqual(evaluation["parameter_updates"], 0)
        record = evaluation["episode_records"][0]
        self.assertEqual(record["bootstrap_masks"], [1.0]*99 + [0.0])
        self.assertEqual(record["terminated"], [False]*99 + [True])
        self.assertEqual(record["truncated"], [False]*100)
        self.assertEqual(evaluation["policy_sha256_before"], evaluation["policy_sha256_after"])
        self.assertTrue(set(evaluation["episode_seeds"]).isdisjoint(self.dataset.arrays["episode_seeds"]))

    def test_zero_online_budget_is_explicitly_no_online_learning(self):
        for interactions in (0, 4):
            create, learners = self.factory()
            config = EvaluationConfig(False, 1, 100, OnlineBudget(interactions, 0, .5, 5, {"learning_rate": .02}))
            result = run_mode(self.dataset, self.pretrained, create, config, families=("recorded",))
            self.assertEqual(result["receipt"]["status"], "no_online_learning")
            self.assertEqual(result["receipt"]["adaptation_interactions"], interactions)
            self.assertEqual(learners[0].configurations, 0)
            if interactions == 0:
                self.assertEqual(result["checkpoints"]["recorded"].digest, self.pretrained.digest)
            else:
                self.assertTrue(result["receipt"]["collection_only"])
                for name, value in self.pretrained.components.items():
                    if name != "rng":
                        self.assertEqual(result["checkpoints"]["recorded"].components[name], value)
            self.assertEqual(result["frozen_evaluation"]["recorded"]["environment_steps"], 100)

    def test_mixed_replay_exact_counts_schedule_partial_and_independent_scenarios(self):
        create, learners = self.factory()
        config = EvaluationConfig(False, 1, 777, OnlineBudget(3, 5, .5, 5, {"learning_rate": .002}))
        result = run_mode(self.dataset, self.pretrained, create, config,
                          families=("recorded", "strong_current"))
        self.assertIsNot(learners[0], learners[1])
        for learner, family in zip(learners, ("recorded", "strong_current")):
            raw = result["raw_adaptation"][family]
            self.assertEqual([seen for _, seen in learner.batches], [1, 2, 2, 3, 3])
            self.assertEqual(raw["interactions"], 3)
            self.assertEqual(raw["updates"], 5)
            self.assertEqual(raw["replay"]["sampled_offline"], 10)
            self.assertEqual(raw["replay"]["sampled_online"], 15)
            self.assertEqual(raw["restored_checkpoint_sha256"], self.pretrained.digest)
            self.assertEqual(raw["partial_episode"]["steps"], 3)
            self.assertFalse(raw["partial_episode"]["complete"])
            self.assertEqual(raw["complete_episodes"], [])
            self.assertIsNone(raw["complete_episode_mean_return"])
            self.assertEqual(raw["learning_settings_after"]["learning_rate"], .002)
            self.assertNotEqual(raw["learning_settings_before_sha256"], raw["learning_settings_after_sha256"])
            for batch, seen in learner.batches:
                self.assertEqual(int(np.sum(batch.origin == 0)), 2)
                self.assertEqual(int(np.sum(batch.origin == 1)), 3)
                self.assertTrue(np.all(batch.group_ids[batch.origin == 1] == -1))
            frozen = result["frozen_evaluation"][family]
            self.assertEqual(frozen["episodes"], 1)
            self.assertEqual(frozen["environment_steps"], 100)
            self.assertTrue(set(frozen["episode_seeds"]).isdisjoint(raw["episode_seeds"]))
            self.assertEqual(result["checkpoints"][family].optimizer_steps, self.pretrained.optimizer_steps + 5)
        self.assertFalse(result["receipt"]["adaptation_returns_in_frozen_results"])

    def test_online_end_cannot_masquerade_as_exact_offline_resume(self):
        config = EvaluationConfig(False, 1, 700, OnlineBudget(1, 1, 0., 2, {}))
        result = run_mode(self.dataset, self.pretrained, AffineSGDBCAdapter, config, families=("recorded",))
        end = result["checkpoints"]["recorded"]
        self.assertEqual(end.metadata["runner_state"]["phase"], "online_end_checkpoint")
        with self.assertRaisesRegex(RunnerError, "resume requires matching"):
            train_offline(self.dataset, AffineSGDBCAdapter, budget=OfflineBudget(1, 7), seed=8601, resume_from=end)

    def test_online_full_and_partial_episode_accounting(self):
        config = EvaluationConfig(False, 1, 700, OnlineBudget(101, 1, 0., 2, {}))
        result = run_mode(self.dataset, self.pretrained, AffineSGDBCAdapter, config, families=("recorded",))
        raw = result["raw_adaptation"]["recorded"]
        self.assertEqual(len(raw["complete_episodes"]), 1)
        self.assertEqual(raw["complete_episodes"][0]["steps"], 100)
        self.assertEqual(raw["partial_episode"]["steps"], 1)
        self.assertEqual(raw["transition_records"][99]["bootstrap_mask"], 0.0)
        self.assertEqual(raw["transition_records"][100]["bootstrap_mask"], 1.0)
        self.assertNotEqual(raw["transition_records"][99]["group_ids"], raw["transition_records"][100]["group_ids"])

    def test_requested_and_executed_actions_are_separate(self):
        class Unbounded(RecordingAdapter):
            def act(self, observation, explore=False):
                self.actions_seen += 1
                return np.array([1.5, -2., .1234567890123, 0.0])
        config = EvaluationConfig(False, 1, 99, OnlineBudget(1, 1, 0., 2, {}))
        result = run_mode(self.dataset, self.pretrained, Unbounded, config, families=("recorded",))
        row = result["raw_adaptation"]["recorded"]["transition_records"][0]
        self.assertEqual(row["requested_actions"][0], 1.5)
        np.testing.assert_array_equal(row["actions"], np.clip(np.asarray(row["requested_actions"], dtype=np.float32), -1, 1))

    def test_frozen_mutation_and_swallowed_write_attempt_are_rejected(self):
        class Mutating:
            value = 0
            def state_bytes(self):
                return str(self.value).encode()
            def act(self, observation):
                self.value += 1
                return np.zeros(4)
        class Writing:
            def state_bytes(self):
                return b"constant"
            def act(self, observation):
                try:
                    Path("must-not-be-created-by-frozen-evaluation.txt").write_text("bad")
                except RuntimeError:
                    pass
                return np.zeros(4)
        request = dict(family="recorded", episodes=1, episode_seeds=[991],
                       physics=self.dataset.metadata["physics"], task_semantics="finite_horizon")
        with self.assertRaisesRegex(RunnerError, "state changed during action"):
            evaluate_frozen(Mutating(), **request)
        with self.assertRaisesRegex(RunnerError, "forbidden operation attempted"):
            evaluate_frozen(Writing(), **request)
        self.assertFalse(Path("must-not-be-created-by-frozen-evaluation.txt").exists())

    def test_frozen_evaluation_rejects_lossy_or_duplicate_seeds(self):
        learner = AffineSGDBCAdapter()
        learner.restore(self.pretrained)
        for seeds in ([1, 1.1], [True, 2], [-1, 2], [2, 2]):
            with self.assertRaises(RunnerError):
                evaluate_frozen(learner.freeze(), family="recorded", episodes=2,
                    episode_seeds=seeds, physics=self.dataset.metadata["physics"], task_semantics="finite_horizon")

    def test_same_factory_instance_and_nonmatching_dataset_are_rejected(self):
        shared = AffineSGDBCAdapter()
        with self.assertRaisesRegex(RunnerError, "new learner instance"):
            run_mode(self.dataset, self.pretrained, lambda: shared, EvaluationConfig(episodes=1),
                     families=("recorded", "strong_current"))
        other = generate_dataset(episodes=10, seed=7602)
        with self.assertRaisesRegex(RunnerError, "not bound to this dataset"):
            run_mode(other, self.pretrained, AffineSGDBCAdapter, EvaluationConfig(episodes=1), families=("recorded",))

    def test_repeat_run_reproduces_artifacts_and_runner_alias_is_guarded(self):
        config = EvaluationConfig(False, 1, 700, OnlineBudget(2, 2, .5, 3, {}))
        first = run_mode(self.dataset, self.pretrained, AffineSGDBCAdapter, config, families=("recorded",))
        second = run_mode(self.dataset, self.pretrained, AffineSGDBCAdapter, config, families=("recorded",))
        for key in ("raw_adaptation", "frozen_evaluation", "receipt"):
            self.assertEqual(json.dumps(first[key], sort_keys=True), json.dumps(second[key], sort_keys=True))
        with inference_evaluation_guard(), self.assertRaisesRegex(ValueError, "disabled during frozen evaluation"):
            train_offline(self.dataset, AffineSGDBCAdapter, budget=OfflineBudget(1, 1))

    def test_every_checkpoint_is_saved_reloaded_and_restored_before_first_frozen_evaluation(self):
        from thrusterchain import training_runner as runner
        families = ("recorded", "strong_current")
        loaded_ids, restored_ids = set(), set()
        original_load, original_evaluate = LearnerCheckpoint.load, runner.evaluate_frozen
        class RestoreRecording(RecordingAdapter):
            def restore(self, checkpoint):
                if id(checkpoint) in loaded_ids:
                    restored_ids.add(id(checkpoint))
                super().restore(checkpoint)
        with tempfile.TemporaryDirectory(prefix=".runner-checkpoint-tests-", dir=Path(__file__).resolve().parents[1]) as temporary:
            directory = Path(temporary) / "checkpoints"
            def observed_load(path):
                self.assertTrue(all((directory / f"{family}.json").is_file() for family in families))
                checkpoint = original_load(path)
                loaded_ids.add(id(checkpoint))
                return checkpoint
            def observed_evaluate(*args, **kwargs):
                self.assertEqual(len(loaded_ids), len(families))
                self.assertEqual(restored_ids, loaded_ids)
                for family in families:
                    self.assertEqual(original_load(directory / f"{family}.json").primary_optimizer_steps, 4)
                return original_evaluate(*args, **kwargs)
            config = EvaluationConfig(False, 1, 811, OnlineBudget(2, 1, .5, 4, {}))
            with patch.object(LearnerCheckpoint, "load", side_effect=observed_load), patch.object(
                    runner, "evaluate_frozen", side_effect=observed_evaluate) as evaluation:
                result = run_mode(self.dataset, self.pretrained, RestoreRecording, config,
                                  families=families, checkpoint_directory=directory)
            self.assertEqual(evaluation.call_count, len(families))
            receipt = result["receipt"]
            self.assertEqual(receipt["checkpoint_persistence"], "saved_and_reloaded")
            self.assertTrue(receipt["persisted_before_frozen_evaluation"])
            self.assertEqual(receipt["evaluation_checkpoint_source"], "reloaded_verified_checkpoint")
            for family in families:
                artifact = receipt["checkpoint_artifacts"][family]
                self.assertEqual(Path(artifact["path"]), directory / f"{family}.json")
                self.assertEqual(original_load(artifact["path"]).digest, artifact["sha256"])
                self.assertEqual(result["checkpoints"][family].digest, artifact["sha256"])

    def test_tampered_checkpoint_reload_refused_before_frozen_evaluation(self):
        from thrusterchain import training_runner as runner
        original_load = LearnerCheckpoint.load
        with tempfile.TemporaryDirectory(prefix=".runner-checkpoint-tests-", dir=Path(__file__).resolve().parents[1]) as temporary:
            directory = Path(temporary) / "checkpoints"
            def replaced_load(path):
                checkpoint = original_load(path)
                metadata = checkpoint.metadata
                metadata["tampered_after_save"] = True
                return LearnerCheckpoint(checkpoint.adapter_id, dict(checkpoint.components),
                       checkpoint.optimizer_steps, dict(checkpoint.optimizer_step_counts), metadata)
            with patch.object(LearnerCheckpoint, "load", side_effect=replaced_load), patch.object(
                    runner, "evaluate_frozen") as evaluation, self.assertRaisesRegex(RunnerError, "saved checkpoint changed"):
                run_mode(self.dataset, self.pretrained, AffineSGDBCAdapter, EvaluationConfig(episodes=1),
                         families=("recorded",), checkpoint_directory=directory)
            evaluation.assert_not_called()
            self.assertEqual(original_load(directory / "recorded.json").digest, self.pretrained.digest)

    def test_checkpoint_destinations_preflight_before_collection_and_never_overwrite(self):
        with tempfile.TemporaryDirectory(prefix=".runner-checkpoint-tests-", dir=Path(__file__).resolve().parents[1]) as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            blocker = root / "file"
            blocker.write_text("preserve me", encoding="utf8")
            config = EvaluationConfig(False, 1, 811, OnlineBudget(2, 1, .5, 4, {}))
            for path, error in ((existing, FileExistsError), (blocker, FileExistsError),
                                (blocker / "nested" / "checkpoints", NotADirectoryError)):
                with self.subTest(path=path), patch("thrusterchain.environment.ThrusterChainEnv") as environment:
                    with self.assertRaises(error):
                        run_mode(self.dataset, self.pretrained, AffineSGDBCAdapter, config,
                                 families=("recorded",), checkpoint_directory=path)
                    environment.assert_not_called()
            self.assertEqual(blocker.read_text(encoding="utf8"), "preserve me")
            self.assertEqual(list(existing.iterdir()), [])

    def test_in_memory_evaluation_does_not_claim_saved_checkpoint_evidence(self):
        result = run_mode(self.dataset, self.pretrained, AffineSGDBCAdapter,
                          EvaluationConfig(episodes=1), families=("recorded",))
        receipt = result["receipt"]
        self.assertEqual(receipt["checkpoint_persistence"], "in_memory_snapshot")
        self.assertFalse(receipt["persisted_before_frozen_evaluation"])
        self.assertEqual(receipt["checkpoint_artifacts"], {})
        self.assertEqual(result["frozen_evaluation"]["recorded"]["evaluation_checkpoint_source"], "in_memory_snapshot")
        self.assertIsNone(result["frozen_evaluation"]["recorded"]["checkpoint_path"])


if __name__ == "__main__":
    unittest.main()
