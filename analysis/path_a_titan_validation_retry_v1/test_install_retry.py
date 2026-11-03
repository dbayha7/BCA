"""Pure temporary-file tests; no experiment imports or scientific execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("install_retry", Path(__file__).with_name("install_retry.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo, self.source = self.base / "repo", self.base / "patch"
        self.study = self.repo / M.STUDY
        self.study.mkdir(parents=True)
        self.source.mkdir()
        (self.repo / "original.py").write_bytes(b"unchanged original\r\n")
        write(self.study / "TITAN_MANIFEST.json", {"files": {"original.py": M.sha(self.repo / "original.py")}})
        write(self.study / "TITAN_APPROVAL.json", {"fixture": "original approval"})
        self.mhash, self.ahash = M.sha(self.study / "TITAN_MANIFEST.json"), M.sha(self.study / "TITAN_APPROVAL.json")
        self.patches = patch.multiple(M, MANIFEST_SHA=self.mhash, APPROVAL_SHA=self.ahash)
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.prior = self.study / "titan_validation_1"
        write(self.prior / "ATTEMPT.json", {"pid": 34106, "process_cap_seconds": 120,
              "titan_manifest_sha256": self.mhash, "titan_approval_sha256": self.ahash,
              "target_identity": {"repository": "/mnt/c/Users/David Bayha/BCA-PathA",
                                  "executable": "/home/dbayha/bca-gpu/bin/python"}})
        (self.prior / "progress.jsonl").write_bytes(b'{"phase":"before_template_initialization"}\n')
        for name in ("titan_retry.py", "install_retry.py", "README.md"):
            (self.source / name).write_bytes(("fixture " + name).encode())
        self.manifest = {"schema": "path_a.titan_validation_retry.v1", "titan_manifest_sha256": self.mhash,
            "files": {M.PREFIX + p.name: M.sha(p) for p in self.source.iterdir()}}
        self.seal()

    def seal(self):
        write(self.source / "RETRY_MANIFEST.json", self.manifest)
        self.digest = M.sha(self.source / "RETRY_MANIFEST.json")
        write(self.source / "RETRY_APPROVAL.json", {"retry_manifest_sha256": self.digest,
              "validation_attempt": 2, "prior_pid": 34106,
              "zero_update_retry_only": True, "conditional_existing_recovery": True})

    def install(self, confirmation=True):
        return M.install(self.source, self.repo, self.digest, confirmation)

    def test_additive_install_preserves_original_bytes_and_receipts(self):
        before = {p.relative_to(self.repo).as_posix(): p.read_bytes() for p in self.repo.rglob("*") if p.is_file()}
        receipt = self.install()
        self.assertTrue(receipt["no_scientific_work"])
        self.assertGreaterEqual(receipt["installation_elapsed_seconds"], 0)
        for name, content in before.items():
            self.assertEqual((self.repo / name).read_bytes(), content)
        target = self.repo / M.PREFIX
        self.assertEqual(M.read(target / "INSTALL_RECEIPT.json"), receipt)
        self.assertEqual(receipt["first_validation_files"], {n: M.sha(self.prior / n) for n in ("ATTEMPT.json", "progress.jsonl")})

    def test_second_install_refused_without_overwrite(self):
        self.install()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.install()

    def test_helper_requires_explicit_answer_and_absence(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.install(False)
        (self.repo / "path_a_a1.py").write_text("# do not execute", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be reviewed"):
            self.install()

    def test_changed_source_rejected(self):
        (self.repo / "original.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "pinned file changed"):
            self.install()

    def test_changed_patch_rejected(self):
        (self.source / "titan_retry.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Patch file changed"):
            self.install()

    def test_terminal_first_attempt_requires_reconciliation(self):
        write(self.prior / "RESULT.json", {"status": "pass"})
        with self.assertRaisesRegex(ValueError, "terminal evidence"):
            self.install()

    def test_wrong_first_attempt_refused(self):
        obj = M.read(self.prior / "ATTEMPT.json")
        obj["pid"] = 1
        write(self.prior / "ATTEMPT.json", obj)
        with self.assertRaisesRegex(ValueError, "approved interrupted"):
            self.install()

    def test_existing_production_refused(self):
        (self.study / "run_titan_recovery_1").mkdir()
        with self.assertRaisesRegex(ValueError, "reservation already exists"):
            self.install()

    def test_escape_path_refused_before_write(self):
        self.manifest["files"] = {M.PREFIX + "../original.py": "0" * 64}
        self.seal()
        with self.assertRaisesRegex(ValueError, "Unsafe patch filename"):
            self.install()
        self.assertFalse((self.repo / M.PREFIX).exists())

    def test_original_paths_cannot_be_patched(self):
        self.manifest["files"] = {"original.py": "0" * 64}
        self.seal()
        with self.assertRaisesRegex(ValueError, "frozen or unrelated"):
            self.install()

    def test_wrong_manifest_digest_refused(self):
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            M.install(self.source, self.repo, "0" * 64, True)

    def test_redirected_analysis_directory_refused_before_write(self):
        original = Path.resolve
        raw_target = self.repo / M.PREFIX
        def resolve(path, *args, **kwargs):
            if path == raw_target:
                return self.base / "outside"
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.install()
        self.assertFalse(raw_target.exists())
        self.assertFalse((self.base / "outside").exists())


if __name__ == "__main__":
    unittest.main()
