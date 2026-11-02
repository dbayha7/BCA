"""CPU fixtures only; no runtime initialization, model load, or experiment."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import titan


class HandoffTests(unittest.TestCase):
    def invoke(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = titan.main(argv)
        return code, json.loads(stream.getvalue())

    def test_execute_always_blocked_with_missing_hold(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(titan, "HERE", Path(directory)), \
                patch.object(titan, "load_support") as loader:
            code, result = self.invoke(["--execute"])
            self.assertEqual(code, 3)
            self.assertFalse(result["execution_allowed"])
            self.assertIsNotNone(result["snapshot_error"])
            loader.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_execute_always_blocked_even_if_snapshot_claims_released(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(titan, "HERE", Path(directory)):
            path = Path(directory) / "ATTEMPT_HOLD.json"
            path.write_text('{"execution_allowed": true, "status": "released"}', encoding="utf-8")
            code, result = self.invoke(["--execute"])
            self.assertEqual(code, 3)
            self.assertFalse(result["execution_allowed"])
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_status_is_saved_snapshot_not_live_status(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(titan, "HERE", Path(directory)):
            snapshot = {"status": "uncertain", "last_known_critic_updates": 200}
            (Path(directory) / "ATTEMPT_HOLD.json").write_text(json.dumps(snapshot), encoding="utf-8")
            code, result = self.invoke(["--status"])
            self.assertEqual(code, 0)
            self.assertEqual(result["origin_attempt_snapshot"], snapshot)
            self.assertFalse(result["live_process_status_checked"])
            self.assertFalse(result["final_execution_counts_known"])

    def test_preflight_calls_frozen_checks_only(self):
        caps = {key: index for index, key in enumerate(("total_critic_updates", "total_actor_updates",
            "total_auxiliary_updates", "total_calibrations", "evaluation_episodes",
            "simulator_transition_cap", "total_timeout_seconds"))}
        from unittest.mock import Mock
        helper = SimpleNamespace(verify_prepared=Mock(return_value=(caps, {"files": {"a": "x"},
            "dependencies": {"b": "y"}})), verify_inputs=Mock())
        with patch.object(titan, "ORIGIN_HASHES", {"PREPARED.json": "p", "protocol.json": "q"}), \
                patch.object(Path, "read_bytes", side_effect=[b"p", b"q"]), \
                patch.object(titan.hashlib, "sha256", side_effect=[SimpleNamespace(hexdigest=lambda: "p"),
                    SimpleNamespace(hexdigest=lambda: "q")]), \
                patch.object(titan, "load_support", return_value=helper), \
                patch.object(titan, "origin_status", return_value={"execution_allowed": False}):
            code, result = self.invoke(["--preflight"])
        self.assertEqual(code, 0)
        helper.verify_prepared.assert_called_once_with()
        helper.verify_inputs.assert_called_once_with()
        self.assertFalse(result["attempt_reserved"])
        self.assertFalse(result["runtime_packages_or_cuda_checked"])
        self.assertEqual(result["caps"], caps)

    def test_preflight_refuses_changed_origin_before_import(self):
        with patch.object(Path, "read_bytes", return_value=b"changed"), \
                patch.object(titan, "load_support") as loader:
            code, result = self.invoke(["--preflight"])
        self.assertEqual(code, 1)
        self.assertFalse(result["execution_allowed"])
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
