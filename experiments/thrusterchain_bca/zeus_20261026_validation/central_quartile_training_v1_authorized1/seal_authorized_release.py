"""Seal the approved release once; never reserve, initialize models or dispatch."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
from common import HERE, read, sha, require
import authorization


def seal():
    for name in ('PREPARED.json','ATTEMPT.json','run','RUN_RESULT.json','runtime_validation_v1'):
        require(not (HERE/name).exists(), 'Existing seal/attempt preserved: '+name)
    protocol=read(HERE/'protocol.json')
    approval=authorization.verify(HERE,protocol)
    prior=HERE.parent/authorization.VALIDATED_PACKAGE
    parent=read(prior/'PREPARED.json')
    repo=HERE.parents[3]
    inputs=dict(parent['inputs'])
    for name in list(parent['files'])+['PREPARED.json','VALIDATION_ATTEMPT_3.json',
             'VALIDATION_RESULT_3.json','runtime_validation_v1/report.json','runtime_validation_v1/RESULT.json',
             'runtime_validation_v1/CHILD_OUTCOME.json','runtime_validation_v1/RESERVATION.json',
             'runtime_validation_v1/WORKER_ENTRY.json']:
        path=prior/name
        inputs[path.relative_to(repo).as_posix()]=sha(path)
    for name,expected in inputs.items():
        path=(repo/name).resolve()
        require(path.is_relative_to(repo.resolve()) and sha(path)==expected,'Pinned input changed: '+name)
    for path in HERE.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    files={path.name:sha(path) for path in sorted(HERE.iterdir())
           if path.is_file() and path.suffix in ('.py','.md','.json')}
    result={'schema':'unifloral.central_quartile.prepared.v1.authorized1',
        'utc':datetime.now(timezone.utc).isoformat(), 'status':'approved_for_one_attempt_not_started',
        'scientific_execution_during_preparation':False,'production_attempt_started':False,
        'execution_authorized':True,'authorization_sha256':approval['authorization_sha256'],
        'validated_prepared_sha256':authorization.VALIDATED_SEAL,
        'validation_result_sha256':authorization.VALIDATION_RESULT,
        'scientific_protocol_sha256':authorization.PROTOCOL,
        'target_runtime_verified_for_unchanged_computational_files':True,
        'files':files,'dependencies':{},'inputs':inputs,'budget':read(HERE/'AUTHORIZATION.json')['budget']}
    with (HERE/'PREPARED.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    return {'status':result['status'],'prepared_sha256':sha(HERE/'PREPARED.json'),
        'authorization_sha256':approval['authorization_sha256'],'protocol_sha256':authorization.PROTOCOL,
        'package_files':len(files),'pinned_inputs':len(inputs),'attempt_reserved':False}


if __name__=='__main__': print(json.dumps(seal()))
