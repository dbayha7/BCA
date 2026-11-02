"""Freeze this explicitly approved continuation once; never execute models."""
import ast
from datetime import datetime,timezone,timedelta
import json
from common import HERE,read,sha,require
import authorization
import recovery_inputs as recovery

def seal():
    for name in ('PREPARED.json','ATTEMPT.json','run','RUN_RESULT.json'):
        require(not (HERE/name).exists(),'Existing seal/attempt preserved: '+name)
    approval=authorization.verify(HERE,read(HERE/'protocol.json'))
    recovery.verify_prior(all_artifacts=True)
    parent=read(recovery.prior_root()/'PREPARED.json')
    repo=HERE.parents[3]
    inputs=dict(parent['inputs'])
    for name in list(parent['files'])+['PREPARED.json']:
        path=recovery.prior_root()/name
        inputs[path.relative_to(repo).as_posix()]=sha(path)
    for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    files={path.name:sha(path) for path in sorted(HERE.iterdir())
           if path.is_file() and path.suffix in ('.py','.md','.json')}
    record={'schema':'unifloral.central_quartile.prepared.v1.continuation1',
        'recorded_at_est':datetime.now(timezone(timedelta(hours=-5))).isoformat(),
        'scientific_execution_during_preparation':False,'production_attempt_started':False,
        'execution_authorized':True,'authorization':approval,
        'target_runtime_continuation_entry_tested':False,
        'target_exact_restore_before_updates_required':True,
        'files':files,'dependencies':parent.get('dependencies',{}),'inputs':inputs,
        'prior_artifact_manifest_sha256':sha(HERE/'RECOVERY.json')}
    with (HERE/'PREPARED.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    return {'status':'sealed_continuation_not_started','prepared_sha256':sha(HERE/'PREPARED.json'),
        'package_files':len(files),'pinned_inputs':len(inputs),'prior_artifacts':len(recovery.recovery()['prior_files_sha256'])}

if __name__=='__main__':print(json.dumps(seal()))
