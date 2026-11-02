"""Build and install a real wheel offline; check retained attribution artifacts."""
import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTICE_HASHES = {
    "third_party/BCA_LICENSE.txt":
        "af878c708defb49f47c78cd9293386ad09fbbae07f20f24cf9e26ac98a834c95",
    "third_party/BCA_NOTICE.txt":
        "f08019e8e763a4e9d78845664216ba5bf7b05bd7cdbf3dc8f65ac7019fd94588",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_and_install(source_root, run_root):
    """Keep durable build sources, outputs, install target and logs in run_root."""
    source_root, run_root = Path(source_root).resolve(), Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    stage = run_root / "source"
    stage.mkdir()
    shutil.copy2(source_root / "pyproject.toml", stage / "pyproject.toml")
    for name in ("thrusterchain", "third_party"):
        shutil.copytree(source_root / name, stage / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    input_hashes = {
        path.relative_to(stage).as_posix(): sha256(path.read_bytes())
        for path in sorted(stage.rglob("*")) if path.is_file()
    }
    wheel_dir = run_root / "wheels"
    install_dir = run_root / "installed"
    build_command = [sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
                     "--no-build-isolation", "--no-cache-dir", "--wheel-dir",
                     str(wheel_dir), str(stage)]
    # Use a fresh short scratch path for pip's deeply nested ephemeral cache.
    # Windows extended paths are incompatible with setuptools 78's license-copy
    # join of a backslash prefix and the portable third_party/... relative path.
    # LOCALAPPDATA can be redirected into a managed sandbox with incompatible
    # rename semantics. Select the explicit user-local temporary root instead.
    scratch_base = Path.home() / "AppData" / "Local" / "Temp" if os.name == "nt" else None
    with tempfile.TemporaryDirectory(prefix="tcw-", dir=scratch_base) as temporary:
        env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INDEX="1",
                   TEMP=temporary, TMP=temporary)
        result = subprocess.run(build_command, cwd=run_root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run_root / "build.log").write_text(result.stdout, encoding="utf8")
        if result.returncode:
            raise RuntimeError(f"Offline wheel build failed:\n{result.stdout}")
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"Expected one wheel, found {wheels}")
        wheel = wheels[0]
        install_command = [sys.executable, "-m", "pip", "install", "--no-index",
                           "--no-deps", "--no-cache-dir", "--no-compile",
                           "--target", str(install_dir), str(wheel)]
        result = subprocess.run(install_command, cwd=run_root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run_root / "install.log").write_text(result.stdout, encoding="utf8")
        if result.returncode:
            raise RuntimeError(f"Offline target install failed:\n{result.stdout}")
    return dict(source_root=str(source_root), run_root=str(run_root),
                wheel=str(wheel), install_dir=str(install_dir),
                commands=[build_command, install_command], input_sha256=input_hashes,
                pip_temporary_directory=temporary,
                pip_temporary_cleaned=not Path(temporary).exists())


def inspect_distribution(build):
    """Describe actual wheel and installed contents, including missing notices."""
    wheel, install_dir = Path(build["wheel"]), Path(build["install_dir"])
    with zipfile.ZipFile(wheel) as archive:
        metadata_path, = [name for name in archive.namelist()
                         if name.endswith(".dist-info/METADATA")]
        dist_info = metadata_path.rsplit("/", 1)[0]
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        wheel_members = {name: sha256(archive.read(name))
                         for name in archive.namelist() if not name.endswith("/")}
        records = {row[0]: row[1:] for row in csv.reader(io.StringIO(
            archive.read(f"{dist_info}/RECORD").decode("utf8")))}
        notices = {}
        for relative_path, expected in NOTICE_HASHES.items():
            path = f"{dist_info}/licenses/{relative_path}"
            installed_path = install_dir / path
            data = archive.read(path) if path in wheel_members else None
            record = records.get(path)
            encoded = (base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                       .rstrip(b"=").decode("ascii")) if data is not None else None
            notices[relative_path] = dict(
                expected_sha256=expected, wheel_path=path,
                source_sha256=build["input_sha256"].get(relative_path),
                wheel_sha256=sha256(data) if data is not None else None,
                installed_sha256=sha256(installed_path.read_bytes())
                    if installed_path.is_file() else None,
                wheel_record_matches=record == [f"sha256={encoded}", str(len(data))]
                    if data is not None else False,
            )
    distribution = importlib.metadata.Distribution.at(install_dir / dist_info)
    installed_files = {str(item).replace("\\", "/")
                       for item in distribution.files or []}
    for notice in notices.values():
        notice["installed_record_lists_file"] = notice["wheel_path"] in installed_files
    source_package = {name: digest for name, digest in build["input_sha256"].items()
                      if name.startswith("thrusterchain/")}
    package_matches = {name: wheel_members.get(name) == digest
                       for name, digest in source_package.items()}
    receipt = dict(
        schema="thrusterchain.wheel-notices.v1", **build,
        python=sys.version, setuptools=importlib.metadata.version("setuptools"),
        pip=importlib.metadata.version("pip"), wheel_sha256=sha256(wheel.read_bytes()),
        wheel_bytes=wheel.stat().st_size, name=metadata["Name"],
        version=metadata["Version"], metadata_version=metadata["Metadata-Version"],
        license_files=metadata.get_all("License-File", []),
        license_expression=metadata["License-Expression"], license=metadata["License"],
        installed_license_files=distribution.metadata.get_all("License-File", []),
        notices=notices, packaged_source_matches=package_matches,
        packaged_source_all_match=all(package_matches.values()),
        scope="Artifact retention and local offline installation checks only; "
              "no new project license or release authorization is asserted.",
    )
    return receipt


class WheelNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verification = ROOT / ".verification" / "packaging"
        verification.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="tests-", dir=verification)
        cls.addClassCleanup(cls.temporary.cleanup)
        build = build_and_install(ROOT, Path(cls.temporary.name) / "run")
        cls.receipt = inspect_distribution(build)

    def test_license_file_metadata_is_explicit_without_new_license_claim(self):
        self.assertEqual(self.receipt["version"], "0.4.1")
        self.assertEqual(self.receipt["metadata_version"], "2.4")
        self.assertCountEqual(self.receipt["license_files"], NOTICE_HASHES)
        self.assertCountEqual(self.receipt["installed_license_files"], NOTICE_HASHES)
        self.assertIsNone(self.receipt["license_expression"])
        self.assertIsNone(self.receipt["license"])

    def test_existing_license_and_notice_are_byte_identical_in_wheel(self):
        for path, expected in NOTICE_HASHES.items():
            with self.subTest(path=path):
                notice = self.receipt["notices"][path]
                self.assertEqual(notice["source_sha256"], expected)
                self.assertEqual(notice["wheel_sha256"], expected)
                self.assertTrue(notice["wheel_record_matches"])

    def test_target_install_retains_notices_and_record_entries(self):
        for path, expected in NOTICE_HASHES.items():
            with self.subTest(path=path):
                notice = self.receipt["notices"][path]
                self.assertEqual(notice["installed_sha256"], expected)
                self.assertTrue(notice["installed_record_lists_file"])

    def test_wheel_contains_current_package_and_canonical_json(self):
        self.assertTrue(self.receipt["packaged_source_all_match"],
                        self.receipt["packaged_source_matches"])
        self.assertIn("thrusterchain/canonical/plant.json",
                      self.receipt["packaged_source_matches"])


if __name__ == "__main__":
    unittest.main()
