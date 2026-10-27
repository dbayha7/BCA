"""Synthetic reservation and timeout fixtures; no model/fit/calibration."""
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import launch


class LauncherFixtures(unittest.TestCase):
    def test_reservation_is_exclusive_and_preserved(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            root = Path(temp)
            launch.reserve(root, "fixture")
            with self.assertRaises((ValueError, FileExistsError)):
                launch.reserve(root, "duplicate")
            self.assertTrue((root / "ATTEMPT.json").is_file())

    def test_successful_stub(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            result = launch.supervise([sys.executable, "-c", "print('synthetic success')"], temp, 5)
            self.assertEqual(result, {"exit_code": 0, "timed_out": False})

    def test_failed_stub(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            result = launch.supervise([sys.executable, "-c", "raise SystemExit(7)"], temp, 5)
            self.assertEqual(result, {"exit_code": 7, "timed_out": False})

    def test_timeout_stops_stub(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            result = launch.supervise([sys.executable, "-c", "import time; time.sleep(3)"], temp, .1)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)

    def test_interrupt_stops_owned_child(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            child = MagicMock()
            child.wait.side_effect = [KeyboardInterrupt(), -9]
            child.poll.return_value = None
            child.returncode = -9
            with patch.object(launch.subprocess, "Popen", return_value=child):
                with patch.object(launch.os, "name", "nt"):
                    result = launch.supervise(["synthetic"], temp, 5)
            self.assertTrue(result["interrupted"])
            child.kill.assert_called_once()
            self.assertEqual(child.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
