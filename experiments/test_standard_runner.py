"""Harmless lifecycle fixtures: no JAX, data, or simulator."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from experiments.standard_runner import exclusive, run_child


class Lifecycle(unittest.TestCase):
    def test_success_and_explicit_failure(self):
        with tempfile.TemporaryDirectory() as d:
            for code in (0, 7):
                out = Path(d)/str(code)
                self.assertEqual(run_child([sys.executable,"-c",f"raise SystemExit({code})"],out,5),code)
                receipt=json.loads((out/"actual_exit.json").read_text())
                self.assertEqual(receipt["actual_returncode"],code)
                self.assertFalse(receipt["timeout"])
                with self.assertRaises(FileExistsError):
                    run_child([sys.executable,"-c","pass"],out,5)

    def test_timeout_records_real_exit(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"timeout"
            self.assertNotEqual(run_child([sys.executable,"-c","import time; time.sleep(60)"],out,.2),0)
            r=json.loads((out/"actual_exit.json").read_text())
            self.assertTrue(r["timeout"])
            self.assertLess(r["actual_returncode"],0)

    def test_lock_rejects_second_owner(self):
        with tempfile.TemporaryDirectory() as d:
            lock=Path(d)/"lane.lock"
            with exclusive(lock):
                code="from experiments.standard_runner import exclusive; from pathlib import Path; "
                code+=f"c=exclusive(Path({str(lock)!r})); c.__enter__()"
                p=subprocess.run([sys.executable,"-c",code],capture_output=True)
                self.assertNotEqual(p.returncode,0)
                self.assertIn(b"BlockingIOError",p.stderr)
            with exclusive(lock):
                pass

if __name__=="__main__":
    unittest.main()
