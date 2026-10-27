"""Synthetic metadata fixtures only; no native model, GPU or experiment imports."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_handoff as handoff


class HandoffTests(unittest.TestCase):
    def test_safe_member_rejects_escape(self):
        for name in ("../outside", "/absolute", "a/../b", "a\\b", "C:/file", "a//b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                handoff.member(handoff.HERE, name)
        self.assertEqual(handoff.member(handoff.HERE, "snapshot/a.json"), handoff.HERE / "snapshot/a.json")

    def archive_fixture(self, root):
        snapshot = root / "snapshot"
        snapshot.mkdir()
        item = snapshot / "a.json"
        item.write_bytes(b'{"synthetic": true}')
        manifest = root / "ARCHIVE.json"
        manifest.write_text(json.dumps({"schema": "bca.interrupted_attempt_archive.v1", "attempt_ordinal": 2,
            "files": [{"path": "a.json", "bytes": item.stat().st_size, "sha256": handoff.sha(item)}]}))
        return manifest, item

    def test_archive_checks_hash_bytes_and_membership(self):
        with tempfile.TemporaryDirectory(dir=handoff.HERE, prefix="fixture_") as directory:
            manifest, item = self.archive_fixture(Path(directory))
            expected = handoff.sha(manifest)
            handoff.verify_archive(manifest, expected)
            item.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                handoff.verify_archive(manifest, expected)
            item.write_bytes(b'{"synthetic": true}')
            extra = item.parent / "extra.json"
            extra.write_text("{}")
            with self.assertRaises(ValueError):
                handoff.verify_archive(manifest, expected)
            with self.assertRaises(ValueError):
                handoff.verify_archive(manifest, "wrong_manifest_hash")

    def matching_fixture(self):
        counts = {"critic": 10000, "actor": 5000, "auxiliary": 10000, "actor_target": 5000,
                  "critic_target": 5000, "outer_steps": 5000, "aux_fit_attempts": 10000,
                  "refresh_attempts": 1, "refresh_accepted": 1}
        match = {"distance_coefficient": handoff.COEFFICIENTS[handoff.CELLS[0]], "probe_rows": 768,
                 "no_rematch": True, "optimizer_updates": 0, "calibration_calls": 0}
        return {"additional_updates": counts}, match

    def test_exact_warmup_counts_and_fixed_coefficient(self):
        report, matching = self.matching_fixture()
        handoff.validate_counts_and_match(report, matching, handoff.CELLS[0])
        self.assertEqual(report["additional_updates"]["critic"], 10000)
        self.assertEqual(matching["distance_coefficient"], 0.07585831620304931)

    def test_incomplete_or_rematched_inputs_refused(self):
        report, matching = self.matching_fixture()
        report["additional_updates"]["critic"] = 9999
        with self.assertRaises(ValueError):
            handoff.validate_counts_and_match(report, matching, handoff.CELLS[0])
        report, matching = self.matching_fixture()
        matching["distance_coefficient"] *= 2
        with self.assertRaises(ValueError):
            handoff.validate_counts_and_match(report, matching, handoff.CELLS[0])

    def test_cli_cannot_execute(self):
        with patch.object(handoff, "verify") as verify, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                handoff.main(["--execute"])
            self.assertEqual(result.exception.code, 2)
            verify.assert_not_called()
        with patch.object(handoff, "verify", return_value={"training_started": False}) as verify, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(handoff.main(["--preflight"]), 0)
            verify.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
