"""Seal the saved-development audit after synthetic tests, without audit execution."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    for name in ('prepared_files.json', 'PREPARATION_RECEIPT.json', 'unit_test_output.txt'):
        if (ROOT / name).exists():
            raise FileExistsError(str(ROOT / name))
    names = ['analyze_saved.py', 'metrics.py', 'test_inputs.py', 'test_metrics.py', 'prepare.py',
             'protocol.json', 'DEFINITIONS_FROZEN.json', 'PROTOCOL_AMENDMENT_01.json',
             'AMENDMENT_FROZEN.json', 'README.md']
    before = {name: sha(ROOT / name) for name in names}
    for name in names:
        raw = (ROOT / name).read_bytes()
        if name.endswith('.py'):
            tree = ast.parse(raw, filename=name)
            for node in ast.walk(tree):
                modules = ([item.name.split('.')[0] for item in node.names] if isinstance(node, ast.Import)
                           else [(node.module or '').split('.')[0]] if isinstance(node, ast.ImportFrom) else [])
                if set(modules) & {'jax', 'flax', 'optax', 'gymnasium', 'thrusterchain'}:
                    raise ValueError('Model or simulator import in saved-data audit: ' + name)
        elif name.endswith('.json'):
            json.loads(raw)
    modules = [importlib.import_module(name) for name in ('test_inputs', 'test_metrics')]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
    expected = suite.countTestCases()
    if expected < 20:
        raise AssertionError('Expected complete synthetic preparation tests')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != expected or result.skipped:
        raise AssertionError(stream.getvalue())
    if {'jax', 'flax', 'optax', 'gymnasium', 'thrusterchain'} & set(sys.modules):
        raise AssertionError('Preparation imported a model or simulator')
    if before != {name: sha(ROOT / name) for name in names}:
        raise AssertionError('Sources changed during preparation tests')
    with (ROOT / 'unit_test_output.txt').open('x', encoding='utf-8') as stream_out:
        stream_out.write(stream.getvalue())
    receipt = {
        'schema': 'unifloral.development_mechanism.preparation.v1',
        'status': 'static_and_synthetic_tests_pass',
        'utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'executable': sys.executable,
        'tests_passed': result.testsRun, 'failures': 0, 'errors': 0, 'skipped': 0,
        'source_hashes': before,
        'unit_test_output_sha256': sha(ROOT / 'unit_test_output.txt'),
        'fixture_scope': 'Synthetic unit fixtures only; no real inference or new data generation',
        'confirmation_files_read': 0, 'new_simulator_calls': 0, 'new_gpu_work': 0,
        'optimizer_updates': 0, 'calibrator_updates': 0,
        'full_training_hold': True
    }
    write(ROOT / 'PREPARATION_RECEIPT.json', receipt)
    names += ['PREPARATION_RECEIPT.json', 'unit_test_output.txt']
    write(ROOT / 'prepared_files.json', {'schema': 'prepared_files.v1',
                                        'files': {name: sha(ROOT / name) for name in names}})
    print(json.dumps({'status': receipt['status'], 'tests_passed': result.testsRun,
                      'manifest_sha256': sha(ROOT / 'prepared_files.json'),
                      'confirmation_files_read': 0, 'new_gpu_or_simulator_work': False}))


if __name__ == '__main__':
    main()
