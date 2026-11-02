"""Run v6 tests while explicitly excluding unavailable v4 archival receipts.

The excluded classes are still present in tests/. They test original byte hashes
and/or v4 datasets using v4 validation; they must be run on the original snapshot
with the corresponding historical files, not on synthetic replacements.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVAL_CLASSES = {
    "test_numpy_compatibility.NumpyCompatibilityTests",
    "test_reporting_integrity.ReportingIntegrityTests",
    "test_revision_claims.ArchivedTerminalClaimTests",
    "test_revision_v3_csv.StrictControlCSVTests",
    "test_revision_v3_identification.IdentificationReceiptRevisionTests",
    "test_revision_v3_metadata.TypedGeneratorMetadataTests",
}


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    names = set()
    for test in cases(suite):
        cls = type(test)
        name = cls.__module__ + "." + cls.__name__
        if name in ARCHIVAL_CLASSES:
            cls.__unittest_skip__ = True
            cls.__unittest_skip_why__ = "v4 historical artifact and v4 loader required"
            names.add(name)
    if names != ARCHIVAL_CLASSES:
        raise RuntimeError("unexpected archival test inventory: " + repr(names ^ ARCHIVAL_CLASSES))
    print("Archived v4 classes excluded from v6 validation:", *sorted(names), sep="\n  ", flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
