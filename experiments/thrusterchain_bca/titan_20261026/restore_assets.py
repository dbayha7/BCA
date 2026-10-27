"""Verify and restore hash-bound runtime assets without overwriting differing files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parent


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_target(root, name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Unsafe package path')
    parts = name.split('/')
    if any(part in ('', '.', '..', '.git') for part in parts):
        raise ValueError('Unsafe package path components')
    reserved = {'con', 'prn', 'aux', 'nul'} | {'com' + str(i) for i in range(1, 10)} | {'lpt' + str(i) for i in range(1, 10)}
    if any(part != part.rstrip(' .') or part.split('.')[0].casefold() in reserved for part in parts):
        raise ValueError('Platform-ambiguous package path')
    path = PurePosixPath(name)
    if path.is_absolute():
        raise ValueError('Absolute package path')
    root = Path(root).resolve()
    target = root.joinpath(*parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError('Package path escapes destination')
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError('Symlinked package path')
        current = current.parent
    return target


def read_manifest(root=ROOT):
    root = Path(root).resolve()
    value = json.loads((root / 'HANDOFF_MANIFEST.json').read_text(encoding='utf-8'))
    if value.get('schema') != 'bca.portable_handoff.v1':
        raise ValueError('Unsupported handoff manifest')
    files = value.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('Empty file manifest')
    seen = set()
    for item in files:
        name = item.get('path')
        safe_target(root, name)
        if name.casefold() in seen:
            raise ValueError('Duplicate manifest path')
        seen.add(name.casefold())
        if item.get('storage') not in ('git', 'runtime_archive'):
            raise ValueError('Unknown file storage role')
        if type(item.get('bytes')) is not int or item['bytes'] < 0:
            raise ValueError('Invalid file byte count')
        digest = item.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid file hash')
    return value


def verify_package(root=ROOT, *, require_assets=True, require_ready=True):
    manifest = read_manifest(root)
    if require_ready and manifest.get('package_status') != 'ready_gpu_execution_pending':
        raise ValueError('Package is not sealed and ready for the bounded audit')
    missing, checked = [], 0
    for item in manifest['files']:
        if not require_assets and item['storage'] == 'runtime_archive':
            continue
        target = safe_target(root, item['path'])
        if not target.is_file():
            missing.append(item['path'])
        elif target.stat().st_size != item['bytes'] or sha_file(target) != item['sha256']:
            raise ValueError('Package file differs: ' + item['path'])
        else:
            checked += 1
    if missing:
        raise FileNotFoundError('Missing package files: ' + ', '.join(missing))
    return {'status': 'verified', 'files_verified': checked, 'assets_required': require_assets,
            'manifest_sha256': sha_file(Path(root) / 'HANDOFF_MANIFEST.json')}


def restore(archive, root=ROOT):
    root = Path(root).resolve()
    manifest = read_manifest(root)
    expected = {item['path']: item for item in manifest['files'] if item['storage'] == 'runtime_archive'}
    declaration = manifest.get('runtime_archive', {})
    archive = Path(archive)
    if archive.stat().st_size != declaration.get('bytes') or sha_file(archive) != declaration.get('sha256'):
        raise ValueError('Runtime archive differs from the manifest')
    restored, matching = 0, 0
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        names = [info.filename for info in entries]
        if len(names) != len({name.casefold() for name in names}):
            raise ValueError('Duplicate archive members')
        if set(names) != set(expected):
            raise ValueError('Archive member set differs from manifest')
        # Verify the entire archive and all pre-existing destinations before any write.
        for info in entries:
            target = safe_target(root, info.filename)
            item = expected[info.filename]
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 1:
                raise ValueError('Unsupported archive member type')
            if info.file_size != item['bytes']:
                raise ValueError('Archive member size differs')
            digest = hashlib.sha256()
            actual = 0
            with package.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    actual += len(block)
                    if actual > item['bytes']:
                        raise ValueError('Archive expansion exceeds declared size')
                    digest.update(block)
            if actual != item['bytes'] or digest.hexdigest() != item['sha256']:
                raise ValueError('Archive member content differs: ' + info.filename)
            if target.exists():
                if not target.is_file() or target.stat().st_size != item['bytes'] or sha_file(target) != item['sha256']:
                    raise FileExistsError('Refusing to overwrite differing asset: ' + info.filename)
        for info in entries:
            target = safe_target(root, info.filename)
            item = expected[info.filename]
            if target.exists():
                if not target.is_file() or target.stat().st_size != item['bytes'] or sha_file(target) != item['sha256']:
                    raise FileExistsError('Destination changed after verification: ' + info.filename)
                matching += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            safe_target(root, info.filename)
            # Exclusive creation intentionally preserves unexpected/racing files.
            with package.open(info) as source, target.open('xb') as destination:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    destination.write(block)
            if target.stat().st_size != item['bytes'] or sha_file(target) != item['sha256']:
                raise ValueError('Written asset verification failed: ' + info.filename)
            restored += 1
    return {'status': 'assets_restored', 'restored_files': restored,
            'already_matching_files': matching, 'overwritten_files': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', nargs='?', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--git-files-only', action='store_true')
    args = parser.parse_args()
    if args.archive and (args.verify_only or args.git_files_only):
        parser.error('Choose restoration or verification')
    if args.archive:
        print(json.dumps(restore(args.archive)))
    elif args.verify_only or args.git_files_only:
        print(json.dumps(verify_package(require_assets=not args.git_files_only)))
    else:
        parser.error('Provide the runtime archive or --verify-only')


if __name__ == '__main__':
    main()
