"""Regression tests for the two defects that stopped gate v1 attempt 1 in its source-identity phase.

1. reference.authenticate_core read core.canonical.__file__, but the pinned loader builds that module in
   memory (types.ModuleType), so it has no __file__.
2. worker.run's crash handler called type(exc).__name(), which raised a second AttributeError and hid
   the first error.

Neither test decodes data, steps the simulator or loads a model.
"""
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1] / "titan_20261026/payload/transfer_bundle"


class CanonicalSourceIdentity(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("gymnasium") is not None, "real core needs gymnasium")
    def test_real_core_authenticates_without_module_file(self):
        import worker
        import reference
        core, files = worker._core()
        self.assertFalse(hasattr(core.canonical, "__file__"), "pinned canonical module is expected to be in-memory")
        out = reference.authenticate_core(core)
        self.assertEqual(Path(out["canonical_source"]).name, "thrusterchain.py")
        self.assertEqual(Path(out["canonical_source"]).resolve(),
                         (BUNDLE / "core_v4/thrusterchain/canonical/thrusterchain.py").resolve())
        self.assertEqual(out["canonical_sha256"], reference.CANONICAL_SHA256)


class CrashHandlerRecordsErrorType(unittest.TestCase):
    def test_failure_is_recorded_not_masked(self):
        import worker
        with tempfile.TemporaryDirectory() as tmp:
            # An empty contract fails inside the guarded block (prepared-files check or contract
            # comparison), which must reach the handler and be recorded with its real type.
            result = worker.run(tmp, {}, None, time.monotonic() + 60)
            self.assertEqual(result["status"], "failed_or_interrupted_design_gate")
            self.assertIsInstance(result["error"]["type"], str)
            self.assertTrue(result["error"]["type"].endswith("Error"), result["error"])
            self.assertTrue((Path(tmp) / "report.json").is_file())


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()
