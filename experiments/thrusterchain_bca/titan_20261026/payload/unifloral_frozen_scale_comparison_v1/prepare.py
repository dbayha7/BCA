"""Freeze this development protocol after standard-library preparation tests."""
import ast
from datetime import datetime, timezone
import hashlib
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
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')

def main():
    for name in ('prepared_files.json','PREPARATION_RECEIPT.json','unit_test_output.txt'):
        if (ROOT/name).exists(): raise FileExistsError(str(ROOT/name))
    names=['score_contract.py','test_contract.py','scoring.py','simulation.py','run_diagnostic.py','prepare.py',
           'protocol.json','SIMULATION_DIFF.json','README.md','PREREGISTERED_CHECKS.md']
    for name in names:
        raw=(ROOT/name).read_bytes()
        if name.endswith('.py'):
            tree=ast.parse(raw,filename=name)
            for node in tree.body:
                modules=[]
                if isinstance(node,ast.Import): modules=[x.name.split('.')[0] for x in node.names]
                elif isinstance(node,ast.ImportFrom): modules=[(node.module or '').split('.')[0]]
                if set(modules)&{'numpy','jax','flax','optax','scipy','gymnasium'}:
                    raise ValueError('Scientific top-level import: '+name)
        elif name.endswith('.json'): json.loads(raw)
    import test_contract
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_contract)
    stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun!=18 or result.skipped: raise AssertionError(stream.getvalue())
    with (ROOT/'unit_test_output.txt').open('x',encoding='utf-8') as out: out.write(stream.getvalue())
    receipt={'schema':'unifloral.frozen_scale_comparison.preparation.v1','status':'static_and_stdlib_tests_pass',
             'utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,
             'unit_tests_passed':result.testsRun,'failures':0,'errors':0,'skipped':0,
             'python_asts':[n for n in names if n.endswith('.py')], 'protocol_sha256':sha(ROOT/'protocol.json'),
             'unit_test_output_sha256':sha(ROOT/'unit_test_output.txt'),
             'scientific_imports_or_execution':False,'optimizer_updates':0,'simulator_steps':0,
             'gpu_execution':'pending_manual_user_command','scientific_validation_status':'incomplete_design_review_required'}
    write(ROOT/'PREPARATION_RECEIPT.json',receipt)
    names+=['PREPARATION_RECEIPT.json','unit_test_output.txt']
    write(ROOT/'prepared_files.json',{'schema':'prepared_files.v1','files':{n:sha(ROOT/n) for n in names}})
    print(json.dumps({'status':receipt['status'],'passed_tests':result.testsRun,'manifest_sha256':sha(ROOT/'prepared_files.json')}))

if __name__=='__main__': main()
