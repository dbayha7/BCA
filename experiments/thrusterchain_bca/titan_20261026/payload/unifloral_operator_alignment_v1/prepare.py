"""Synthetic-only preparation of the operator alignment package."""
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
    import numpy as np
    for name in ('prepared_files.json', 'PREPARATION_RECEIPT.json', 'unit_test_output.txt'):
        if (ROOT / name).exists():
            raise FileExistsError(str(ROOT / name))
    names = ['checkpoint_io.py', 'gpu_execution.py', 'operator_audit.py', 'run_checks.py',
             'test_checkpoint_io.py', 'test_gpu_execution.py', 'test_operator_audit.py', 'test_runner.py',
             'prepare.py', 'protocol.json', 'DEFINITIONS_FROZEN.json', 'README.md', 'CHECKPOINT_DEPENDENCIES.json']
    before = {name: sha(ROOT / name) for name in names}
    prohibited_top = {'jax', 'flax', 'optax', 'gymnasium', 'thrusterchain', 'torch', 'scipy'}
    for name in names:
        raw = (ROOT / name).read_bytes()
        if name.endswith('.py'):
            tree = ast.parse(raw, filename=name)
            for node in tree.body:
                modules = ([item.name.split('.')[0] for item in node.names] if isinstance(node, ast.Import)
                           else [(node.module or '').split('.')[0]] if isinstance(node, ast.ImportFrom) else [])
                if set(modules) & prohibited_top:
                    raise ValueError('Scientific dependency imported at module scope: ' + name)
        elif name.endswith('.json'):
            json.loads(raw)
    freeze = json.loads((ROOT / 'DEFINITIONS_FROZEN.json').read_bytes())
    assert freeze['protocol_sha256'] == sha(ROOT / 'protocol.json') and freeze['new_evaluator_calls'] == 0
    modules = [importlib.import_module(name) for name in
               ('test_checkpoint_io', 'test_gpu_execution', 'test_operator_audit', 'test_runner')]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
    expected = suite.countTestCases()
    assert expected >= 35, 'Expected complete bounded-fixture suite'
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != expected or result.skipped:
        raise AssertionError(output.getvalue())
    assert not prohibited_top & set(sys.modules), 'Scientific runtime imported by local fixture preparation'
    assert before == {name: sha(ROOT / name) for name in names}, 'Source changed during preparation'
    with (ROOT / 'unit_test_output.txt').open('x', encoding='utf-8') as stream:
        stream.write(output.getvalue())
    receipt = {'schema': 'unifloral.operator_alignment.preparation.v1',
               'status': 'static_and_cpu_fixture_tests_pass', 'utc': datetime.now(timezone.utc).isoformat(),
               'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
               'tests_passed': result.testsRun, 'failures': 0, 'errors': 0, 'skipped': 0,
               'source_hashes': before, 'unit_test_output_sha256': sha(ROOT / 'unit_test_output.txt'),
               'fixture_scope': 'Synthetic/mock CPU tests only; no actual model restore, GPU or simulator calls',
               'actual_gpu_audit_executed': False, 'new_evaluator_calls': 0, 'confirmation_files_read': 0,
               'policy_optimizer_updates': 0, 'calibrator_updates': 0, 'full_training_hold': True,
               'zeus_runtime_tested': False}
    write(ROOT / 'PREPARATION_RECEIPT.json', receipt)
    names += ['PREPARATION_RECEIPT.json', 'unit_test_output.txt']
    write(ROOT / 'prepared_files.json', {'schema': 'prepared_files.v1', 'files': {name: sha(ROOT / name) for name in names}})
    print(json.dumps({'status': receipt['status'], 'tests_passed': result.testsRun,
                      'prepared_manifest_sha256': sha(ROOT / 'prepared_files.json'), 'gpu_execution_pending': True}))


if __name__ == '__main__':
    main()
