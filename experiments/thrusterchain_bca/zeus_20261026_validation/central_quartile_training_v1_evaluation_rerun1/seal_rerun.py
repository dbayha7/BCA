"""Freeze this explicitly approved evaluation-only rerun once; never execute models."""
import ast
from datetime import datetime, timezone, timedelta
import json
from common import HERE, read, sha, require
import authorization
import recovery_inputs as recovery
import rerun_inputs as rerun

CHANGED = {'evaluate_worker.py', 'launch.py', 'authorization.py', 'README.md', 'test_evaluate_worker.py'}
NEW = {'lock_tolerant.py', 'rerun_inputs.py', 'prepare_rerun.py', 'seal_rerun.py', 'RERUN_PROTOCOL.md',
       'RERUN_INPUTS.json', 'AUTHORIZATION.json', 'RERUN_CHECKS.json', 'RERUN_REVIEW.md',
       'test_rerun_writes.py', 'test_rerun_gate.py', 'test_rerun_routing.py', 'test_rerun_worker_flow.py',
       'test_rerun_launch.py', 'test_rerun_real_inputs.py'}


def seal():
    for name in ('PREPARED.json', 'ATTEMPT.json', 'run', 'RUN_RESULT.json'):
        require(not (HERE/name).exists(), 'Existing seal/attempt preserved: '+name)
    folders = [p.name for p in HERE.iterdir() if p.is_dir()]
    require(not folders, 'Remove test or cache folders before sealing: '+', '.join(folders))
    present = {p.name for p in HERE.iterdir() if p.is_file()}
    require(present == authorization.UNCHANGED | CHANGED | NEW,
            'Unexpected or missing package files: '+str(sorted(present ^ (authorization.UNCHANGED | CHANGED | NEW))))
    approval = authorization.verify(HERE, read(HERE/'protocol.json'))
    recovery.verify_prior(all_artifacts=True)
    rerun.verify_continuation(all_artifacts=True)
    croot = rerun.continuation_root()
    parent = read(croot/'PREPARED.json')
    repo = HERE.parents[3]
    inputs = dict(parent['inputs'])
    for name in list(parent['files'])+['PREPARED.json']:
        path = croot/name
        inputs[path.relative_to(repo).as_posix()] = sha(path)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    files = {path.name: sha(path) for path in sorted(HERE.iterdir())
             if path.is_file() and path.suffix in ('.py', '.md', '.json')}
    record = {'schema': 'unifloral.central_quartile.prepared.v1.evaluation_rerun1',
        'recorded_at_est': datetime.now(timezone(timedelta(hours=-5))).isoformat(),
        'scientific_execution_during_preparation': False, 'production_attempt_started': False,
        'execution_authorized': True, 'training_authorized': False, 'authorization': approval,
        'target_runtime_rerun_entry_tested': False,
        'files': files, 'dependencies': parent.get('dependencies', {}), 'inputs': inputs,
        'rerun_inputs_sha256': sha(HERE/'RERUN_INPUTS.json'), 'recovery_sha256': sha(HERE/'RECOVERY.json')}
    with (HERE/'PREPARED.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n')
    return {'status': 'sealed_evaluation_rerun_not_started', 'prepared_sha256': sha(HERE/'PREPARED.json'),
            'package_files': len(files), 'pinned_inputs': len(inputs),
            'original_artifacts': len(recovery.recovery()['prior_files_sha256']),
            'continuation_artifacts': len(rerun.plan()['continuation_files_sha256'])}


if __name__ == '__main__': print(json.dumps(seal()))
