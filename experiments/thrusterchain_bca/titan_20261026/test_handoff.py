"""Only synthetic archive/launcher checks; no scientific imports or execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import restore_assets as assets
import launch


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='handoff_fixture_', dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, members=None):
        members = members or [('payload/small.npz', b'numeric-fixture')]
        archive = self.root / 'assets.zip'
        with zipfile.ZipFile(archive, 'w') as package:
            for name, raw in members:
                package.writestr(name, raw)
        files = [{'path': name, 'bytes': len(raw), 'sha256': digest(raw), 'storage': 'runtime_archive'} for name, raw in members]
        manifest = {'schema': 'bca.portable_handoff.v1', 'package_status': 'ready_gpu_execution_pending',
                    'files': files, 'runtime_archive': {'bytes': archive.stat().st_size, 'sha256': assets.sha_file(archive)}}
        (self.root / 'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest))
        return archive, manifest

    def test_restore_and_idempotent_matching(self):
        archive, _ = self.fixture()
        self.assertEqual(assets.restore(archive, self.root)['restored_files'], 1)
        self.assertEqual(assets.restore(archive, self.root)['already_matching_files'], 1)
        self.assertEqual(assets.verify_package(self.root)['files_verified'], 1)

    def test_refuse_unequal_existing_file(self):
        archive, _ = self.fixture()
        target = self.root / 'payload/small.npz'
        target.parent.mkdir()
        target.write_bytes(b'user content')
        with self.assertRaises(FileExistsError):
            assets.restore(archive, self.root)
        self.assertEqual(target.read_bytes(), b'user content')

    def test_unsafe_paths(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'payload\\escape', 'payload/../escape', 'payload/.git/config', 'payload//empty', 'payload/file. ', 'payload/NUL.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                assets.safe_target(self.root, name)

    def test_duplicate_archive_rejected(self):
        archive, manifest = self.fixture()
        with zipfile.ZipFile(archive, 'a') as package:
            package.writestr('payload/small.npz', b'numeric-fixture')
        manifest['runtime_archive'] = {'bytes': archive.stat().st_size, 'sha256': assets.sha_file(archive)}
        (self.root / 'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            assets.restore(archive, self.root)

    def test_wrong_content_hash_rejected_before_write(self):
        archive, manifest = self.fixture()
        manifest['files'][0]['sha256'] = '0' * 64
        (self.root / 'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            assets.restore(archive, self.root)
        self.assertFalse((self.root / 'payload').exists())

    def test_symlink_member_rejected(self):
        archive, manifest = self.fixture()
        info = zipfile.ZipInfo('payload/small.npz')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, 'w') as package:
            package.writestr(info, b'numeric-fixture')
        manifest['runtime_archive'] = {'bytes': archive.stat().st_size, 'sha256': assets.sha_file(archive)}
        (self.root / 'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            assets.restore(archive, self.root)

    def test_missing_asset_reported(self):
        self.fixture()
        with self.assertRaises(FileNotFoundError):
            assets.verify_package(self.root)

    def test_git_only_verification(self):
        _, manifest = self.fixture()
        source = self.root / 'source.py'
        source.write_bytes(b'print(1)\n')
        manifest['files'].append({'path': 'source.py', 'bytes': source.stat().st_size, 'sha256': assets.sha_file(source), 'storage': 'git'})
        (self.root / 'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest))
        self.assertEqual(assets.verify_package(self.root, require_assets=False)['files_verified'], 1)

    def test_pending_package_blocks_launch(self):
        _, manifest = self.fixture()
        manifest['package_status'] = 'pending_operator_seal'
        (self.root / 'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            assets.verify_package(self.root, require_assets=False)

    def test_launcher_preflight_uses_current_python_and_bundle_override(self):
        runner = self.root / 'payload/unifloral_operator_alignment_v1/run_checks.py'
        runner.parent.mkdir(parents=True)
        runner.write_text('')
        manifest = {'files': [{'path': 'payload/unifloral_operator_alignment_v1/run_checks.py', 'storage': 'git'}]}
        with patch.object(launch, 'verify_package', return_value={'status': 'verified'}), patch.object(launch, 'read_manifest', return_value=manifest), patch.object(launch.subprocess, 'run') as run:
            run.return_value.returncode = 0
            self.assertEqual(launch.launch(root=self.root), 0)
            args, kwargs = run.call_args
            self.assertEqual(args[0][0], sys.executable)
            self.assertEqual(args[0][-1], '--preflight')
            self.assertEqual(kwargs['env']['BCA_TRANSFER_BUNDLE_ROOT'], str(self.root / 'payload/transfer_bundle'))

    def test_manifest_case_aliases_rejected(self):
        self.fixture([('payload/data.npz', b'a'), ('payload/DATA.npz', b'b')])
        with self.assertRaises(ValueError):
            assets.read_manifest(self.root)

    def test_destination_changed_between_passes_rejected(self):
        archive, _ = self.fixture()
        target = self.root / 'payload/small.npz'
        original = assets.safe_target
        calls = 0
        def change_on_second_pass(root, name):
            nonlocal calls
            calls += 1
            resolved = original(root, name)
            # First call validates the manifest; second validates archive content.
            if calls == 3:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_bytes(b'concurrent content')
            return resolved
        with patch.object(assets, 'safe_target', side_effect=change_on_second_pass):
            with self.assertRaises(FileExistsError):
                assets.restore(archive, self.root)
        self.assertEqual(target.read_bytes(), b'concurrent content')

    def test_unlisted_runner_never_executed(self):
        with patch.object(launch, 'verify_package', return_value={'status': 'verified'}), patch.object(launch, 'read_manifest', return_value={'files': []}), patch.object(launch.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                launch.launch(root=self.root)
            run.assert_not_called()

    def test_execute_requires_explicit_runtime_acknowledgement(self):
        with self.assertRaises(ValueError):
            launch.launch(execute=True, root=self.root)


if __name__ == '__main__':
    unittest.main()
