"""Synthetic metadata only: no models, checkpoints, or simulator outcomes."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ood.standard_receipt import bind_receipt, write_receipt

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "experiments/standard_bca/manifest.json.gz"


class StandardReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(gzip.decompress(MANIFEST.read_bytes()))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.lane = self.root / "queue"
        self.item = deepcopy(self.manifest["runs"][0])
        self.reset(self.item)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")

    def reset(self, item):
        self.item = item
        self.run_id = item["row"]["run_id"]
        self.run = self.lane / "runs" / self.run_id
        self.attempt = self.lane / "attempts" / self.run_id
        self.dispatch = dict(command=["python", str(self.source / "train.py"),
            "--algorithm", item["algorithm"], "--dataset", item["dataset"],
            "--method", item["method"], "--seed", str(item["seed"]),
            "--output", str(self.run), "--device", "cuda", "--lock", "gpu.lock",
            "--data-dir", "data"], started="2026-09-27T10:00:00+00:00", pid=123)
        self.actual = dict(started=self.dispatch["started"],
            ended="2026-09-27T11:00:00+00:00", actual_returncode=0,
            timeout=False, interruption_signal=None)
        self.result = dict(completed=True, run_id=self.run_id, steps_completed=1000000,
            expected_counts=item["row"]["expected_counts"], events_sha256="a" * 64,
            checkpoints=[dict(step=n, path=f"checkpoint_{n}.msgpack", sha256="b" * 64)
                         for n in [10000, 50000, 1000000]])
        self.put(self.attempt / "dispatch.json", self.dispatch)
        self.put(self.attempt / "actual_exit.json", self.actual)
        self.put(self.run / "resolved.json", item["row"])
        self.put(self.run / "source.json", self.manifest["source_sha256"])
        self.put(self.run / "exit.json", dict(completed=True, exit_code=0))
        self.put(self.run / "result.json", self.result)

    def bind(self):
        return bind_receipt(MANIFEST, self.lane, self.source, self.run_id)

    def test_host_and_bca_bind_but_do_not_accept_science(self):
        for item in self.manifest["runs"][:2]:
            self.reset(item)
            receipt = self.bind()
            self.assertEqual(receipt["schema"], "ood-process-exit-v1")
            self.assertEqual(receipt["exit_code"], 0)
            self.assertEqual(receipt["run_id"], self.run_id)
            for flag in ["ready_for_collection", "checkpoint_decoded", "events_verified",
                         "data_contents_verified", "source_contents_verified"]:
                self.assertIs(receipt[flag], False)
            self.assertEqual(receipt["status"], "pending")

    def test_failed_or_stale_actual_exits_rejected(self):
        for field, value in [("actual_returncode", 1), ("actual_returncode", None),
                             ("actual_returncode", False), ("timeout", True),
                             ("interruption_signal", 15),
                             ("started", "2026-09-27T09:00:00+00:00"),
                             ("ended", "2026-09-27T09:00:00+00:00"),
                             ("ended", "2026-09-27T11:00:00")]:
            with self.subTest(field=field, value=value):
                self.put(self.attempt / "actual_exit.json", {**self.actual, field: value})
                with self.assertRaises(ValueError): self.bind()

    def test_missing_actual_exit_cannot_be_replaced_by_learner_receipt(self):
        (self.attempt / "actual_exit.json").unlink()
        with self.assertRaises(ValueError): self.bind()

    def test_dispatch_source_and_scientific_arguments_rejected(self):
        for index, value in [(1, "other/train.py"), (3, "rebrac"), (5, "walker2d"),
                             (7, "bca"), (9, "42"), (11, str(self.root / "other")),
                             (13, "cpu")]:
            with self.subTest(index=index):
                changed = deepcopy(self.dispatch); changed["command"][index] = value
                self.put(self.attempt / "dispatch.json", changed)
                with self.assertRaises(ValueError): self.bind()
        changed = deepcopy(self.dispatch)
        changed["command"] += ["--method", "host"]
        self.put(self.attempt / "dispatch.json", changed)
        with self.assertRaises(ValueError): self.bind()

    def test_resolved_source_and_learner_completion_rejected(self):
        for name, value in [("resolved", {**self.item["row"], "calibration_weighting": "iw"}),
                            ("resolved", {**self.item["row"], "action_bound": True}),
                            ("source", {**self.manifest["source_sha256"], "train.py": "c" * 64}),
                            ("exit", {"completed": True, "exit_code": False})]:
            with self.subTest(name=name):
                self.reset(self.item); self.put(self.run / (name + ".json"), value)
                with self.assertRaises(ValueError): self.bind()

    def test_incomplete_result_or_checkpoint_schedule_rejected(self):
        for key, value in [("completed", False), ("run_id", "other"),
                           ("steps_completed", 50000), ("expected_counts", {}),
                           ("checkpoints", self.result["checkpoints"][:-1]),
                           ("events_sha256", "")]:
            with self.subTest(key=key):
                self.put(self.run / "result.json", {**self.result, key: value})
                with self.assertRaises(ValueError): self.bind()

    def test_unknown_run_and_unsupported_adapter_scope_rejected(self):
        with self.assertRaises(ValueError):
            bind_receipt(MANIFEST, self.lane, self.source, "../other")
        for item in self.manifest["runs"]:
            if item["algorithm"] in ["cql", "iql"] or item["dataset"] == "halfcheetah":
                with self.subTest(host=item["algorithm"], dataset=item["dataset"]):
                    with self.assertRaises(ValueError):
                        bind_receipt(MANIFEST, self.lane, self.source, item["row"]["run_id"])

    def test_changed_manifest_and_exclusive_output(self):
        bad = self.root / "manifest.json"
        bad.write_bytes(b"{}")
        with self.assertRaises(ValueError):
            bind_receipt(bad, self.lane, self.source, self.run_id)
        path = self.root / "receipt.json"
        receipt = self.bind(); write_receipt(path, receipt)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError): write_receipt(path, receipt)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
