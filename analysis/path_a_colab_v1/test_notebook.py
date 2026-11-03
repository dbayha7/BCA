"""Notebook structure and pure byte/guard fixtures only; never run setup or science."""
import base64
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
BUNDLE = Path(r"C:\Users\David\AppData\Local\Temp\BCA_PathA_Titan_20261102")
NB = json.loads((HERE / "Path_A_Colab.ipynb").read_text(encoding="utf-8"))
CELLS = {c["id"]: "".join(c["source"]) for c in NB["cells"]}
NAMESPACE = {}
# The first cell contains only standard-library imports and decoded source bytes.
exec(compile(CELLS["embedded-source"], "embedded-source", "exec"), NAMESPACE)
PAYLOAD = NAMESPACE["PAYLOAD"]


class NotebookTests(unittest.TestCase):
    def test_structure_and_all_python_cell_syntax(self):
        self.assertEqual((NB["nbformat"], NB["nbformat_minor"]), (4, 5))
        ids = [c["id"] for c in NB["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in NB["cells"]:
            self.assertRegex(cell["id"], r"^[a-zA-Z0-9_-]{1,64}$")
            self.assertIn(cell["cell_type"], ("code", "markdown"))
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_default_run_all_stops_before_any_side_effect(self):
        state = dict(NAMESPACE)
        with self.assertRaisesRegex(RuntimeError, "status remains unknown"):
            exec(compile(CELLS["ownership"], "ownership", "exec"), state)
        for cell_id, marker in (("install", "INSTALL_EXACT_RUNTIME"), ("validate", "RUN_ZERO_UPDATE_VALIDATION"), ("run", "RUN_PATH_A")):
            self.assertIn(marker + " = False", CELLS[cell_id])
            with self.assertRaises(RuntimeError):
                exec(compile(CELLS[cell_id], cell_id, "exec"), {})

    def test_exact_texts_binaries_and_formatting_from_frozen_archive(self):
        self.assertEqual(len(PAYLOAD["text_files"]), 657)
        self.assertEqual(len(PAYLOAD["binary_files"]), 14)
        with zipfile.ZipFile(BUNDLE / "path_a_titan_recovery.zip") as z:
            manifest = json.loads(z.read("TRANSFER_MANIFEST.json"))
            self.assertEqual(set(manifest["files"]), set(PAYLOAD["text_files"]) | set(PAYLOAD["binary_files"]))
            for name, text in PAYLOAD["text_files"].items():
                self.assertEqual(base64.b64decode(text), z.read(name), name)
            special = "experiments/thrusterchain_bca/two_scenario_v2/SETUP_APPROVAL.json"
            self.assertEqual(hashlib.sha256(base64.b64decode(PAYLOAD["text_files"][special])).hexdigest(),
                             "35fac7db435090e49c7bdc2da2586274736d8191848d2d8ab641f4a23262814b")

    def test_exact_additive_profile_and_full_hash_lock(self):
        self.assertEqual(base64.b64decode(PAYLOAD["extras"]["colab_profile.py"]), (HERE / "colab_profile.py").read_bytes())
        lock = base64.b64decode(PAYLOAD["extras"]["resolved-requirements.txt"])
        self.assertEqual(hashlib.sha256(lock).hexdigest(), NAMESPACE["LOCK_SHA256"])
        rows = [x for x in lock.decode().splitlines() if x.strip() and not x.startswith("#")]
        self.assertEqual(len(rows), 58)
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}", x) for x in rows))

    def test_real_extraction_then_idempotent_verification_and_conflict_refusal(self):
        unpacker = {"__name__": "test_embedded_unpacker"}
        exec(compile(base64.b64decode(PAYLOAD["extras"]["unpack_path_a.py"]), "unpacker", "exec"), unpacker)
        archive = BUNDLE / "path_a_titan_recovery.zip"
        # A short workspace-local fixture root avoids Windows MAX_PATH for the
        # unchanged nested payload; Colab itself uses POSIX paths.
        with tempfile.TemporaryDirectory(prefix="ct_", dir=HERE.parents[1]) as directory:
            root = Path(directory) / "isolated"
            first = unpacker["unpack"](archive, NAMESPACE["ARCHIVE_SHA256"], root)
            self.assertEqual((first["copied"], first["overwritten"]), (671, 0))
            second = unpacker["unpack"](archive, NAMESPACE["ARCHIVE_SHA256"], root, verify_only=True)
            self.assertEqual((second["already_matching"], second["copied"]), (671, 0))
            target = root / next(iter(PAYLOAD["text_files"]))
            target.write_bytes(b"explicit fixture corruption")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                unpacker["unpack"](archive, NAMESPACE["ARCHIVE_SHA256"], root)
            self.assertEqual(target.read_bytes(), b"explicit fixture corruption")

    def test_boundaries_and_no_implicit_launch(self):
        self.assertIn("exist_ok=False", CELLS["ledger"])
        self.assertLess(CELLS["ledger"].index("time.monotonic()"), CELLS["ledger"].index("drive.mount"))
        self.assertLess(CELLS["ledger"].index("exclusive_json(LEDGER"), len(CELLS["ledger"]))
        self.assertIn('"--require-hashes"', CELLS["install"])
        self.assertIn('"--only-binary=:all:"', CELLS["install"])
        self.assertIn('"--no-deps"', CELLS["install"])
        self.assertIn('"https://pypi.org/simple"', CELLS["install"])
        self.assertIn('shutil.which("timeout")', CELLS["install"])
        self.assertIn('"--kill-after=5s"', CELLS["install"])
        self.assertIn('subprocess.Popen(command', CELLS["install"])
        self.assertIn("115, env=runtime_env", CELLS["validate"])
        self.assertIn("cap-5, env=runtime_env", CELLS["run"])
        self.assertNotIn("p.proof()", CELLS["run"])  # Kernel is not the isolated interpreter.
        self.assertIn("p.check(approval=True)", CELLS["run"])
        self.assertIn("APPROVE COLAB MIGRATION", CELLS["approve"])
        self.assertIn("completed_return_episodes\": 1536", CELLS["results"])


if __name__ == "__main__": unittest.main()
