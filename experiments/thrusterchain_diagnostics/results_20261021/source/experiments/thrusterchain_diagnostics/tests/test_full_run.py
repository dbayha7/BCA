"""Driver lifecycle contracts; mocked optimizers never launch a full experiment."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.thrusterchain_diagnostics import full_run as F
from experiments.thrusterchain_diagnostics import training as T, training_data as D


class FullRunContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config = F.default_config()
        F._write(self.directory / "config.json", self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def test_frozen_budget_and_disjoint_worker_partition(self):
        jobs = F.job_matrix(F.validate_config(self.config))
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({job["job_id"] for job in jobs}), 30)
        partitions = [{job["job_id"] for job in jobs[index::3]} for index in range(3)]
        self.assertTrue(all(len(part) == 10 for part in partitions))
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
        self.config["settings"]["updates"] = 10
        with self.assertRaises(ValueError):
            F.validate_config(self.config)

    def test_duplicate_live_lock_and_clean_release(self):
        with F.run_lock(self.directory):
            with self.assertRaises(RuntimeError):
                with F.run_lock(self.directory):
                    self.fail("duplicate lock admitted")
        self.assertFalse(F._read(self.directory / ".driver.lock")["active"])
        with F.run_lock(self.directory):
            pass
        self.assertFalse(F._read(self.directory / ".driver.lock")["active"])

    def test_stale_identity_requires_positive_evidence(self):
        old = dict(active=True, host=socket.gethostname(), pid=123, creation_token="old")
        with patch.object(F, "_process_identity", return_value=("unknown", None)):
            self.assertFalse(F._proven_stale(old))
        with patch.object(F, "_process_identity", return_value=("dead", None)):
            self.assertTrue(F._proven_stale(old))
            self.assertFalse(F._proven_stale(dict(old, host="another-machine")))
        with patch.object(F, "_process_identity", return_value=("alive", "new")):
            self.assertTrue(F._proven_stale(old))
            self.assertFalse(F._proven_stale(dict(old, creation_token=None)))

    def test_lock_refuses_unprovable_old_owner_without_mutation(self):
        path = self.directory / ".driver.lock"
        old = dict(active=True, host=socket.gethostname(), pid=os.getpid(), creation_token=None)
        F._write(path, old)
        with self.assertRaises(RuntimeError):
            with F.run_lock(self.directory):
                pass
        self.assertEqual(F._read(path), old)

    def test_freeze_detects_data_or_source_change(self):
        F._write(self.directory / "job_matrix.json", F.job_matrix(self.config))
        F._write(self.directory / "provenance.json", {"source_sha256": {"source.py": "initial"}})
        names = ("config.json", "job_matrix.json", "provenance.json")
        manifest = dict(schema=F.SCHEMA, ready=True, files={name: F._sha(self.directory / name) for name in names})
        F._write(self.directory / "prepared_manifest.json", manifest)
        with patch.object(F, "_source_hashes", return_value={"source.py": "initial"}):
            self.assertEqual(F.verify_run(self.directory), self.config)
        with patch.object(F, "_source_hashes", return_value={"source.py": "changed"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                F.verify_run(self.directory)
        (self.directory / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen artifact changed"):
            F.verify_run(self.directory)

    def _mock_training(self, train):
        prepared = SimpleNamespace(manifest={"frozen": True}, train="training", bank="bank")
        for job in F.job_matrix(self.config):
            F._write(F._dataset_path(self.directory, job) / "preparation.json",
                     dict(manifest=prepared.manifest, training_transition_sha256="hash", calibration_transition_sha256="hash"))
        stack = ExitStack()
        stack.enter_context(patch.object(F, "verify_run", return_value=self.config))
        stack.enter_context(patch.object(F.socket, "gethostname", return_value="zeus"))
        stack.enter_context(patch.object(F, "_event"))
        stack.enter_context(patch.object(D, "load_dataset", return_value=SimpleNamespace()))
        stack.enter_context(patch.object(F, "_prepared", return_value=prepared))
        stack.enter_context(patch.object(T, "tree_sha256", return_value="hash"))
        stack.enter_context(patch.object(T, "train", side_effect=train))
        stack.enter_context(patch.object(T, "load_trained", return_value=SimpleNamespace(record={"checkpoint": {"update": 100000}})))
        return stack

    @staticmethod
    def _success(*args, **kwargs):
        return SimpleNamespace(record=dict(status="completed", final_update=100000,
            checkpoint=dict(file="checkpoint.msgpack", update=100000), training_seconds=1.0))

    def test_failed_job_is_terminal_continues_and_never_drops_denominator(self):
        calls = []
        def train(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise FloatingPointError("invalid native update")
            return self._success(*args, **kwargs)
        with self._mock_training(train):
            result = F.run_jobs(self.directory, max_jobs=2)
            self.assertEqual(result["statuses"], dict(failed=1, completed=1, pending=28))
            self.assertEqual(result["planned_jobs"], 30)
            self.assertFalse(result["complete"])
            F.run_jobs(self.directory, max_jobs=1)
        self.assertEqual(len(calls), 3)
        self.assertNotEqual(calls[0]["out_dir"], calls[2]["out_dir"])
        self.assertTrue(all(call["resume"] for call in calls))

    def test_interrupted_job_resumes_same_directory(self):
        calls = []
        def train(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise KeyboardInterrupt()
            return self._success(*args, **kwargs)
        with self._mock_training(train):
            with self.assertRaises(KeyboardInterrupt):
                F.run_jobs(self.directory, max_jobs=1)
            self.assertEqual(F.status(self.directory)["statuses"], dict(interrupted=1, pending=29))
            result = F.run_jobs(self.directory, max_jobs=1)
        self.assertEqual(calls[0]["out_dir"], calls[1]["out_dir"])
        self.assertEqual(result["statuses"], dict(completed=1, pending=29))

    def test_refusal_does_not_claim_completed_checkpoint(self):
        def train(*args, **kwargs):
            return SimpleNamespace(record=dict(status="refused", reason="insufficient bank", final_update=0))
        with self._mock_training(train):
            result = F.run_jobs(self.directory, max_jobs=1, worker_index=1)
        self.assertEqual(result["statuses"], dict(pending=29, refused=1))
        self.assertIsNone(result["comparisons"][0]["mean_difference"])

    def test_recovery_requires_explicit_flag_and_rejects_invalid_updates(self):
        calls = []
        def train(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise OSError("temporary checkpoint write failure")
            return self._success(*args, **kwargs)
        with self._mock_training(train):
            F.run_jobs(self.directory, max_jobs=1)
            F.run_jobs(self.directory, max_jobs=1, retry_failed=True)
        self.assertEqual(calls[0]["out_dir"], calls[1]["out_dir"])
        self.assertTrue(F._recoverable_failure("OSError: temporary checkpoint write failure"))
        self.assertFalse(F._recoverable_failure("FloatingPointError: invalid native update"))
        self.assertFalse(F._recoverable_failure("ValueError: source identity changed"))

    def test_partial_diagnostic_evaluation_is_preserved_and_retryable(self):
        calls = []
        def evaluate():
            calls.append(1)
            return ({"critic_diagnostics": {"status": "failed" if len(calls) == 1 else "completed"}},
                    {"return_value": np.array([1.0])})
        with patch.object(F, "_event"):
            first = F._evaluation_task(self.directory, "jobs/example/final", evaluate)
            self.assertTrue(F._diagnostic_failed(first))
            second = F._evaluation_task(self.directory, "jobs/example/final", evaluate)
            self.assertFalse(F._diagnostic_failed(second))
            F._evaluation_task(self.directory, "jobs/example/final", evaluate)
        self.assertEqual(len(calls), 2)
        history = list((self.directory / "evaluation" / "jobs" / "example").glob("final.previous-*"))
        self.assertEqual(len(history), 1)
        self.assertTrue((history[0] / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
