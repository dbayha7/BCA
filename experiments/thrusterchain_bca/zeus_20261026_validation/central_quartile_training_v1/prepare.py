"""Seal a new reviewed package once; never dispatch models or reset a reservation."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    for name in ('PREPARED.json','ATTEMPT.json','RUN_RESULT.json','run','runtime_validation_v1'):
        if (HERE/name).exists():
            raise FileExistsError('Existing seal or attempt preserved; no automatic re-freeze: '+name)
    p=json.loads((HERE/'protocol.json').read_text())
    if p['execution_authorized'] or p['execution_status']!='hold_pending_budget_review':
        raise ValueError('This preparation must remain held for review')
    if p['data_plan_sha256']!=sha(HERE/'DATA_PLAN.json'):
        raise ValueError('Data plan changed')
    p['protocol_text_sha256']=sha(HERE/'PROTOCOL.md')
    (HERE/'protocol.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    gate=HERE.parent/'conditional_ood_gate_central_quartile_v1'
    if sha(gate/'run/report.json')!=p['gate_report_sha256']:
        raise ValueError('Passed gate evidence changed')
    inputs=dict(json.loads((gate/'PREPARED.json').read_text())['files'])
    paths={gate/'run/report.json',gate/'run/RESULT.json',gate/'PREPARED.json'}
    for name in ('matched_policy_pilot_v1','support_shift_pilot_v1','environment_audit_v1'):
        folder=HERE.parent/name
        paths.update(folder.glob('*.py'))
        paths.update(folder.glob('*.json'))
        paths.update(folder.glob('*.md'))
    for path in paths:
        relative=path.resolve().relative_to(REPO.resolve()).as_posix()
        inputs[relative]=sha(path)
    for name,expected in inputs.items():
        path=(REPO/name).resolve()
        if not path.is_relative_to(REPO.resolve()) or sha(path)!=expected:
            raise ValueError('Input identity differs: '+name)
    files={path.name:sha(path) for path in sorted(HERE.iterdir())
           if path.is_file() and path.suffix in ('.py','.md','.json')}
    seal=dict(schema='unifloral.central_quartile.prepared.v1',utc=datetime.now(timezone.utc).isoformat(),
        status='frozen_for_target_validation_not_production_launch',scientific_execution_during_preparation=False,
        files=files,dependencies={},inputs=inputs,production_attempt_started=False,
        target_runtime_verified=False,execution_authorized=False,
        budget=dict(critic_updates=p['total_critic_updates'],actor_updates=p['total_actor_updates'],
            auxiliary_updates=p['total_auxiliary_updates'],calibrations=p['total_calibrations'],
            evaluation_transitions=p['simulator_transition_cap'],seconds=p['total_timeout_seconds']))
    with (HERE/'PREPARED.json').open('x',encoding='utf-8',newline='\n') as out:
        json.dump(seal,out,indent=2,sort_keys=True);out.write('\n')
    return dict(status=seal['status'],package_files=len(files),source_inputs=len(inputs),
                prepared_sha256=sha(HERE/'PREPARED.json'),protocol_sha256=sha(HERE/'protocol.json'))


if __name__=='__main__': print(json.dumps(freeze()))
