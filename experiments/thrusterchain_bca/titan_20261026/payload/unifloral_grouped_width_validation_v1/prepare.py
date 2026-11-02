"""Seal grouped diagnostic source after synthetic-only tests."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
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
    import numpy as np
    assert platform.system() == 'Windows' and platform.python_version() == '3.12.14' and np.__version__ == '2.3.5'
    for name in ('prepared_files.json', 'PREPARATION_RECEIPT.json', 'unit_test_output.txt'):
        if (ROOT / name).exists():
            raise FileExistsError(str(ROOT / name))
    names = ['run_checks.py', 'predictive.py', 'test_runner.py', 'test_predictive.py',
             'prepare.py', 'protocol.json', 'DEFINITIONS_FROZEN.json', 'README.md']
    before = {name: sha(ROOT / name) for name in names}
    banned = {'jax', 'flax', 'optax', 'gymnasium', 'thrusterchain', 'torch', 'sklearn'}
    for name in names:
        raw = (ROOT / name).read_bytes()
        if name.endswith('.py'):
            tree = ast.parse(raw, filename=name)
            for node in ast.walk(tree):
                modules = ([item.name.split('.')[0] for item in node.names] if isinstance(node, ast.Import)
                           else [(node.module or '').split('.')[0]] if isinstance(node, ast.ImportFrom) else [])
                if set(modules) & banned:
                    raise ValueError('Out-of-scope package import: ' + name)
        elif name.endswith('.json'):
            json.loads(raw)
    protocol = json.loads((ROOT / 'protocol.json').read_bytes())
    freeze = json.loads((ROOT / 'DEFINITIONS_FROZEN.json').read_bytes())
    assert freeze['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert freeze['diagnostic_fits_started'] is False
    assert protocol['full_training_hold'] is True
    modules = [importlib.import_module(name) for name in ('test_runner', 'test_predictive')]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
    expected = suite.countTestCases()
    assert expected >= 15, 'Expected complete synthetic tests'
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != expected or result.skipped:
        raise AssertionError(output.getvalue())
    assert not banned & set(sys.modules), 'Model or simulator imported'
    assert before == {name: sha(ROOT / name) for name in names}
    with (ROOT / 'unit_test_output.txt').open('x', encoding='utf-8') as stream:
        stream.write(output.getvalue())
    receipt = {'schema': 'unifloral.grouped_width.preparation.v1', 'status': 'static_and_synthetic_tests_pass',
               'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'executable': sys.executable,
               'numpy': np.__version__, 'tests_passed': result.testsRun, 'failures': 0, 'errors': 0, 'skipped': 0,
               'source_hashes': before, 'unit_test_output_sha256': sha(ROOT / 'unit_test_output.txt'),
               'fixture_scope': 'Synthetic diagnostics only; no actual development fits or outcome analysis',
               'confirmation_files_read': 0, 'new_simulator_calls': 0, 'new_gpu_work': 0,
               'policy_optimizer_updates': 0, 'calibrator_updates': 0, 'full_training_hold': True}
    write(ROOT / 'PREPARATION_RECEIPT.json', receipt)
    names += ['PREPARATION_RECEIPT.json', 'unit_test_output.txt']
    write(ROOT / 'prepared_files.json', {'schema': 'prepared_files.v1', 'files': {name: sha(ROOT / name) for name in names}})
    print(json.dumps({'status': receipt['status'], 'tests_passed': result.testsRun,
                      'prepared_manifest_sha256': sha(ROOT / 'prepared_files.json')}))


if __name__ == '__main__':
    main()
