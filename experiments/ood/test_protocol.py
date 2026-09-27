"""CPU-only contracts. All checkpoint evidence here is explicitly synthetic."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ood import protocol as P


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = P.build_manifest()

    def test_complete_pending_matrix_and_budget(self):
        m = self.manifest
        self.assertEqual(m, P.build_manifest())
        self.assertEqual(len(m["pairs"]), 20)
        self.assertEqual(len(m["checkpoints"]), 40)
        self.assertEqual({r["status"] for r in m["checkpoints"]}, {"pending"})
        self.assertFalse(m["ready_for_collection"])
        self.assertEqual(m["budget"]["per_pair"], {
            "collection": 12800, "outcomes": 1280000,
            "repeat_checks": 5120, "coverage": 700000})
        self.assertEqual(m["budget"]["science_total"], 39958400)
        self.assertEqual(m["budget"]["grand_total"], 39998400)
        self.assertEqual(m["budget"]["training_updates_to_execute"], 0)
        P.validate_manifest(m)

    def test_existing_training_declarations_are_reused(self):
        for row in self.manifest["checkpoints"]:
            with self.subTest(run=row["run_id"]):
                r = row["resolved"]
                self.assertEqual(r["expected_counts"]["host_updates"], 1000000)
                self.assertEqual(r["expected_counts"]["actor_updates"], 500000)
                events = r["protocol"]["evaluation_events"]
                self.assertEqual(len(events), 201)
                self.assertEqual(len(events[-1]["episode_seeds"]), 20)
                self.assertEqual(len(r["protocol"]["refresh_events"]),
                                 198 if r["method"] == "bca" else 0)

    def test_streams_are_unique_and_exclude_training_streams(self):
        values = list(P.stream_seeds(self.manifest))
        self.assertEqual(len(values), len(set(values)))
        self.assertFalse(set(values) & P.training_seeds(self.manifest["checkpoints"]))
        for pair in self.manifest["pairs"]:
            s = pair["streams"]
            self.assertEqual(len(s["collection"]), 64)  # shared across collectors
            self.assertEqual(len(s["continuation"]), 256)  # shared across actions
            P.validate_episode_ids(s["fresh_calibration"], s["test"])

    def test_scientific_edits_and_unknown_settings_are_rejected(self):
        for key, value in (("horizon", 251), ("hosts", ["cql"]),
                           ("continuations", ["best"]), ("new_setting", True)):
            changed = P.load_config()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                P.validate_config(changed)
        changed = P.load_config()
        changed["coverage"]["adaptive_feedback"] = True
        with self.assertRaises(ValueError):
            P.validate_config(changed)

    def test_rehashed_wrong_manifest_is_still_rejected(self):
        for change in (lambda m: m["checkpoints"].pop(),
                       lambda m: m["pairs"][0].update(seed=42),
                       lambda m: m.update(ready_for_collection=True),
                       lambda m: m["checkpoints"][0].update(status="verified")):
            m = deepcopy(self.manifest)
            change(m)
            m["manifest_sha256"] = P.digest({k: v for k, v in m.items()
                                           if k != "manifest_sha256"})
            with self.assertRaises(ValueError):
                P.validate_manifest(m)

    def test_calibration_test_overlap_duplicates_and_adaptive_feedback(self):
        cal, test = list(range(200)), list(range(200, 700))
        P.validate_episode_ids(cal, test)
        for a, b, adaptive in ((cal, [0] + test[1:], False),
                               (cal, [test[0]] * 500, False),
                               (cal[:-1], test, False), (cal, test, True)):
            with self.assertRaises(ValueError):
                P.validate_episode_ids(a, b, adaptive_feedback=adaptive)

    def test_request_cannot_change_estimand(self):
        pair = self.manifest["pairs"][0]
        request = dict(pair_id=pair["pair_id"], seed=pair["seed"],
                       continuation="bca", horizon=250)
        P.validate_request(self.manifest, request)
        for key, value in (("seed", 42), ("continuation", "best"),
                           ("horizon", 251), ("pair_id", "missing")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                P.validate_request(self.manifest, {**request, key: value})

    def test_resource_caps_include_collection_repeats_and_engineering(self):
        pair_id = self.manifest["pairs"][0]["pair_id"]
        caps = self.manifest["budget"]["per_pair"]
        usage = {"science": {pair_id: dict(caps)}, "engineering": {}}
        P.validate_usage(self.manifest, usage)
        for name in caps:
            bad = deepcopy(usage)
            bad["science"][pair_id][name] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                P.validate_usage(self.manifest, bad)
        for count in (10001, -1, True, 1.2):
            with self.assertRaises(ValueError):
                P.validate_usage(self.manifest, {
                    "science": {}, "engineering": {"td3_bc/hopper": count}})
        with self.assertRaises(ValueError):
            P.validate_usage(self.manifest, {"science": {"extra": dict(caps)},
                                             "engineering": {}})

    def test_exclusive_output_cannot_replace_a_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            P.write_manifest(path, self.manifest)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                P.write_manifest(path, self.manifest)
            self.assertEqual(before, path.read_bytes())
            self.assertNotIn(str(path), before.decode())


class EvidenceTests(unittest.TestCase):
    """Reported metadata is checked; dummy bytes are never decoded as a model."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = P.build_manifest()
        cls.row = next(r for r in cls.manifest["checkpoints"]
                       if r["resolved"]["method"] == "bca")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = {}
        r = self.row["resolved"]
        self.put("resolved", r)
        self.put("source", self.manifest["training_source_sha256"])
        self.put("preparation", {"accepted": True, "metadata": {
            "raw_identity": {"sha256": r["cache"]["sha256"]},
            "training_converted_ids": [0, 1], "heldout_converted_ids": [3],
            "normalization_fit_converted_ids": [0, 1],
            "normalization_fit": "training_complement",
            "obs_mean": [0.0], "obs_std": [1.0]}})
        self.put("checkpoint", b"SYNTHETIC NOT A CHECKPOINT")
        self.put("events", b"SYNTHETIC NOT AN EVENT JOURNAL")
        self.put("result", {
            "completed": True, "run_id": self.row["run_id"],
            "steps_completed": 1000000, "expected_counts": r["expected_counts"],
            "events_sha256": self.files["events"]["sha256"],
            "checkpoints": [{"step": 1000000,
                "path": self.files["checkpoint"]["path"],
                "sha256": self.files["checkpoint"]["sha256"],
                "counters": {"actor": {"/step": 500000, "/adam/count": 500000},
                             "critic": {"/step": 1000000, "/adam/count": 1000000},
                             "accepted_scale_fits": 987654}}]})
        self.put("actual_exit", {"schema": "ood-process-exit-v1",
            "run_id": self.row["run_id"], "exit_code": 0,
            "result_sha256": self.files["result"]["sha256"]})

    def put(self, role, value):
        path = self.root / role
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(raw)
        self.files[role] = {"path": path.name, "sha256": P.file_hash(path)}

    def read(self, role):
        return json.loads((self.root / role).read_text())

    def check(self):
        return P.validate_evidence(self.manifest, self.row["run_id"], self.files,
                                   self.root)

    def test_metadata_acceptance_is_not_scientific_verification(self):
        result = self.check()
        self.assertTrue(result["metadata_checked"])
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["checkpoint_decoded"])
        self.assertFalse(result["events_verified"])
        self.assertFalse(result["ready_for_collection"])

    def test_missing_checkpoint_or_actual_exit_is_rejected(self):
        for role in ("checkpoint", "actual_exit"):
            saved = self.files.pop(role)
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.check()
            self.files[role] = saved

    def test_modified_checkpoint_is_rejected(self):
        (self.root / "checkpoint").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_file_and_learner_exit_are_insufficient(self):
        self.put("actual_exit", {"completed": True, "exit_code": 0})
        with self.assertRaises(ValueError):
            self.check()
        (self.root / "checkpoint").unlink()
        with self.assertRaises(ValueError):
            self.check()

    def test_zero_accepted_scale_fits_is_valid_metadata(self):
        result = self.read("result")
        result["checkpoints"][0]["counters"]["accepted_scale_fits"] = 0
        self.put("result", result)
        actual = self.read("actual_exit")
        actual["result_sha256"] = self.files["result"]["sha256"]
        self.put("actual_exit", actual)
        self.assertFalse(self.check()["ready_for_collection"])

    def test_return_scores_do_not_select_or_promote_checkpoints(self):
        for score in (-10000.0, 10000.0):
            result = self.read("result")
            result["final_score"] = score
            self.put("result", result)
            actual = self.read("actual_exit")
            actual["result_sha256"] = self.files["result"]["sha256"]
            self.put("actual_exit", actual)
            checked = self.check()
            self.assertEqual(checked["run_id"], self.row["run_id"])
            self.assertEqual(checked["status"], "pending")

    def test_wrong_configuration_or_source_is_rejected(self):
        for role in ("resolved", "source"):
            original = self.read(role)
            self.put(role, {**original, "seed" if role == "resolved" else "train.py": 42})
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.check()
            self.put(role, original)

    def test_counts_and_completion_must_be_present_and_correct(self):
        original = self.read("result")
        for edit in (lambda r: r["checkpoints"][0].pop("counters"),
                     lambda r: r["checkpoints"][0]["counters"]["actor"].update({"/step": 499999}),
                     lambda r: r["checkpoints"][0]["counters"].update(accepted_scale_fits=1000001),
                     lambda r: r.update(completed=False),
                     lambda r: r.update(steps_completed=100000)):
            result = deepcopy(original)
            edit(result)
            self.put("result", result)
            exit_record = self.read("actual_exit")
            exit_record["result_sha256"] = self.files["result"]["sha256"]
            self.put("actual_exit", exit_record)
            with self.assertRaises(ValueError):
                self.check()

    def test_failure_and_stale_exit_cannot_be_accepted(self):
        original = self.read("actual_exit")
        for patch in ({"exit_code": 1}, {"exit_code": False},
                      {"result_sha256": "0" * 64}, {"run_id": "another-run"}):
            self.put("actual_exit", {**original, **patch})
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.check()

    def test_wrong_data_or_leaked_preparation_is_rejected(self):
        original = self.read("preparation")
        for key, value in (("raw_identity", {"sha256": "0" * 64}),
                           ("heldout_converted_ids", [1]),
                           ("normalization_fit_converted_ids", [0, 1, 3])):
            changed = deepcopy(original)
            changed["metadata"][key] = value
            self.put("preparation", changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()


if __name__ == "__main__":
    unittest.main()
