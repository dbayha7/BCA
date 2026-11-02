"""Provenance regression and explicit cooperative-guard boundary probes."""
import builtins
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thrusterchain.environment import Physics, matrices
from thrusterchain.inference_guard import InferenceGuard
from thrusterchain.planning import BoundedPlanner, identify_from_training


class PlannerProvenanceRevisionTests(unittest.TestCase):
    def test_implicit_oracle_cannot_be_labeled_as_training(self):
        with patch("thrusterchain.planning.matrices",
                   side_effect=AssertionError("must reject before oracle access")):
            for label in ("train_only_identified_model", "caller_supplied_model", "custom"):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "provenance"):
                        BoundedPlanner(provenance=label)

    def test_implicit_oracle_provenance_is_derived(self):
        for label in (None, "privileged_true_model"):
            planner = BoundedPlanner(provenance=label, planning_horizon=1)
            receipt = planner.receipt()
            self.assertEqual(receipt["provenance"], "privileged_true_model")
            self.assertEqual(receipt["matrix_source"], "environment.matrices")
            self.assertFalse(receipt["training_provenance_verified"])
            planner.plan(np.zeros(13), 1)
            self.assertEqual(planner.last_certificate["provenance"], receipt["provenance"])

    def test_explicit_matrices_do_not_prove_training_or_privilege(self):
        A, B = matrices(Physics())
        for label in (None, "train_only_identified_model", "privileged_true_model", "custom"):
            with self.subTest(label=label):
                with patch("thrusterchain.planning.matrices",
                           side_effect=AssertionError("explicit matrices must not load oracle")):
                    planner = BoundedPlanner(A, B, provenance=label, planning_horizon=1)
                receipt = planner.receipt()
                self.assertEqual(receipt["provenance"], "caller_supplied_model")
                self.assertEqual(receipt["declared_provenance"], label)
                self.assertEqual(receipt["matrix_source"], "explicit_A_B_arguments")
                self.assertFalse(receipt["training_provenance_verified"])
                planner.plan(np.zeros(13), 1)
                self.assertEqual(planner.last_certificate["provenance"], "caller_supplied_model")
                self.assertEqual(planner.last_certificate["declared_provenance"], label)

    def test_real_identification_and_existing_runner_call_remain_supported(self):
        rng = np.random.default_rng(72)
        A = np.eye(13) * .8
        B = rng.normal(scale=.1, size=(13, 4))
        states = rng.normal(size=(80, 13))
        actions = rng.uniform(-1, .4, size=(80, 4))
        observations = np.zeros((80, 2, 14), dtype=np.float32)
        observations[:, 0, :13] = states
        observations[:, 1, :13] = states @ A.T + actions @ B.T

        class TrainOnlyFixture:
            metadata = {"physics": {"stall": False}}
            digest = "provenance_revision_fixture"

            def split(self, name):
                if name != "train":
                    raise AssertionError("identification accessed a non-training split")
                return dict(observations=observations, actions=actions[:, None],
                            group_ids=np.arange(80))

        with patch("thrusterchain.planning.matrices", side_effect=AssertionError("oracle access")):
            learned = identify_from_training(TrainOnlyFixture())
            planner = BoundedPlanner(learned.A, learned.B, Physics(stall=False), 2,
                                     provenance="train_only_identified_model")
            planner.plan(np.zeros(13), 2)
        self.assertEqual(planner.receipt()["model_hash"], learned.receipt["matrix_hash"])
        self.assertEqual(learned.receipt["fit_uses"], "train split only")
        self.assertFalse(planner.receipt()["training_provenance_verified"])


class CooperativeGuardRevisionTests(unittest.TestCase):
    def test_supported_write_lookups_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="thrusterchain_guard_") as directory:
            path = Path(directory) / "supported_write.txt"
            with InferenceGuard() as guard:
                for operation in (lambda: builtins.open(path, "w"),
                                  lambda: io.open(path, "w"),
                                  lambda: os.open(path, os.O_WRONLY | os.O_CREAT),
                                  lambda: path.open("w")):
                    with self.assertRaisesRegex(RuntimeError, "write_attempts"):
                        operation()
            self.assertEqual(guard.events["write_attempts"], 4)
            self.assertFalse(path.exists())

    def test_preimported_builtin_open_write_is_an_explicit_expected_hole(self):
        saved_open = builtins.open
        with tempfile.TemporaryDirectory(prefix="thrusterchain_guard_") as directory:
            path = Path(directory) / "alias_write.txt"
            with InferenceGuard() as guard:
                with saved_open(path, "w", encoding="utf8") as stream:
                    stream.write("harmless regression fixture")
            self.assertEqual(path.read_text(encoding="utf8"), "harmless regression fixture")
            self.assertEqual(guard.receipt()["write_attempts"], 0)
            self.assertFalse(guard.receipt()["all_file_writes_monitored"])
            self.assertIn("preimported_file_io_aliases", guard.receipt()["unsupported_monitoring"])

    def test_final_canary_read_is_unmonitored_and_allowed(self):
        with tempfile.TemporaryDirectory(prefix="thrusterchain_guard_") as directory:
            path = Path(directory) / "final_canary.txt"
            path.write_text("synthetic final-only canary", encoding="utf8")
            with InferenceGuard() as guard:
                observed = path.read_text(encoding="utf8")
            self.assertEqual(observed, "synthetic final-only canary")
            self.assertEqual(sum(guard.events.values()), 0)
            self.assertEqual(guard.ledger["forbidden_api_calls"], 0)
            self.assertFalse(guard.receipt()["read_io_monitored"])

    def test_transient_adaptation_is_not_measured_by_guard_counters(self):
        state = {"parameter": 1}
        with InferenceGuard() as guard:
            state["parameter"] = 2
            state["parameter"] = 1
        self.assertEqual(state, {"parameter": 1})
        self.assertEqual(sum(guard.events.values()), 0)
        self.assertIn("transient_state_adaptation", guard.receipt()["unsupported_monitoring"])

    def test_receipt_distinguishes_context_activation_and_counter_scope(self):
        for active in (False, True):
            with InferenceGuard(activate_context=active) as guard:
                pass
            receipt = guard.receipt()
            self.assertEqual(receipt["context_guard_activated"], active)
            self.assertFalse(receipt["arbitrary_code_inference_purity_verified"])
            self.assertIn("native_calls", receipt["unsupported_monitoring"])
            self.assertIn("threads", receipt["unsupported_monitoring"])
            self.assertIn("instrumented", receipt["zero_counter_interpretation"])


if __name__ == "__main__":
    unittest.main()
