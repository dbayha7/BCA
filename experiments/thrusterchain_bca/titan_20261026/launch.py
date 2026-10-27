"""Portable entrypoint for the bounded operator audit; no installation or training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from restore_assets import read_manifest, verify_package

ROOT = Path(__file__).resolve().parent


def launch(execute=False, acknowledge=False, root=ROOT):
    if execute and not acknowledge:
        raise ValueError('Execution requires --allow-outside-upstream-constraints')
    receipt = verify_package(root)
    manifest = read_manifest(root)
    runner_relative = 'payload/unifloral_operator_alignment_v1/run_checks.py'
    runner_entries = [item for item in manifest['files'] if item['path'] == runner_relative]
    if len(runner_entries) != 1 or runner_entries[0]['storage'] != 'git':
        raise ValueError('Audit runner must be a verified Git manifest member')
    print(json.dumps({'event': 'portable_payload_verified', **receipt}), flush=True)
    root = Path(root).resolve()
    runner = root / runner_relative
    if not runner.is_file():
        raise FileNotFoundError('Missing operator audit runner')
    env = dict(os.environ, BCA_TRANSFER_BUNDLE_ROOT=str(root / 'payload' / 'transfer_bundle'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-B', '-u', str(runner), '--execute' if execute else '--preflight']
    if acknowledge:
        command.append('--allow-outside-upstream-constraints')
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--allow-outside-upstream-constraints', action='store_true')
    args = parser.parse_args()
    return launch(args.execute, args.allow_outside_upstream_constraints)


if __name__ == '__main__':
    raise SystemExit(main())
