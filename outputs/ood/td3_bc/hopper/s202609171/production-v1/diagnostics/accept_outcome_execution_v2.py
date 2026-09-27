"""Separate execution preflight; no models/simulator/training calls."""
import json,sys,shutil,datetime
from pathlib import Path
repo=Path('/mnt/c/Users/David Bayha/Documents/GitHub/BCA');sys.path.insert(0,str(repo))
from experiments.ood.collect import file_hash,read_artifact,write_artifact
from experiments.ood.production import inputs,LEDGER
from experiments.ood.candidate_stage import accepted_states
from experiments.ood.outcome_stage import accepted_candidates
from experiments.ood.resources import ResourceLedger,scope_caps
p=Path('/home/dbayha/bca-work/ood-execution-v1/td3-hopper-s202609171-outcomes-v1')
monitor=Path('/mnt/c/Users/David Bayha/Documents/Codex/2026-09-09/this-is-my-phd-research-im/work/standard_bca_noiw_campaign_v1/monitor_20260927T223452Z')
declaration=read_artifact(p/'declaration.json',file_hash(p/'declaration.json'))
binding,full,pair,pre=inputs(repo)
sr,sb=accepted_states(repo,Path(declaration['states']),Path(declaration['states_actual_exit']))
cr,cb=accepted_candidates(repo,Path(declaration['candidates']),Path(declaration['candidates_actual_exit']))
assert sb==declaration['state_binding'] and cb==declaration['candidate_binding']
assert all(file_hash(repo/f)==h for f,h in declaration['source_sha256'].items())
assert declaration['continuations']==['host','bca'] and declaration['primary_continuation']=='bca'
assert declaration['maximum_outcome_transitions']==full['budget']['per_pair']['outcomes']==256*10*2*250
assert declaration['maximum_repeat_transitions']==full['budget']['per_pair']['repeat_checks']==256*10*2
assert declaration['maximum_environment_transitions']==1285121 and declaration['maximum_physics_steps']==5140484
assert declaration['action_absolute_tolerance']==1e-6 and declaration['reward_absolute_tolerance']==1e-7
assert declaration['first_transition_and_full_state_repeat_tolerance']==0.
assert declaration['initialization_reset_seed']==full['engineering_streams']['td3_bc/hopper'][6]
assert declaration['gpu_allocation']==0 and declaration['device']=='cpu' and declaration['maximum_worker_seconds']==86400
assert not declaration['coverage_accepted'] and not declaration['global_ready'] and declaration['required_training_seeds']==5
assert shutil.disk_usage(p).free>declaration['minimum_free_disk_bytes']
receipts={}
for name in ('regression_v1.actual_exit.json','stream_gate_launch_v1.actual_exit.json','candidates_wsl_v1_actual_exit.json','rollouts_v1_actual_exit.json',
             'audit_stream_gate_v1_actual_exit.json','audit_states_v1_actual_exit.json','audit_candidates_v1_actual_exit.json',
             'state_launch_v1_actual_exit.json','candidate_launch_v1_actual_exit.json'):
    path=monitor/name;record=json.loads(path.read_text());assert record.get('actual_returncode',record.get('actual_exit'))==0,(name,record)
    receipts[name]=dict(sha256=file_hash(path),record=record)
with ResourceLedger(LEDGER,scope_caps([x['pair_id'] for x in full['pairs']],list(full['engineering_streams'])),pre['declaration_sha256']) as ledger:
    resources=ledger.audit()
assert resources['reserved']['global']==[12889,51556]
assert resources['reserved']['outcomes/'+declaration['pair_id']]==resources['reserved']['repeat_checks/'+declaration['pair_id']]==[0,0]
out=dict(schema='ood-first-pair-behavioral-execution-acceptance-v1',accepted=True,
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),declaration_sha256=file_hash(p/'declaration.json'),
    source_sha256=declaration['source_sha256'],verified_receipts=receipts,distinct_regression_tests=103,
    resources_before=resources,free_disk_bytes=shutil.disk_usage(p).free,
    review=['Original accepted CPU checkpoint, full-state and streaming gates reused.',
            'All 256 states and 2560 first-action slots independently checked before outcomes.',
            'Full frozen candidate/support/actual JAX key banks and both continuations bound.',
            'Each first transition and full end state repeats exactly before continuation.',
            'Horizon includes the first step; original termination/time limit ends rollout without learned tail.',
            'Every constructor/outcome/repeat call reserved in original cumulative ledger before physics.',
            'One worker takes shared local lock; no GPU, training change or duplicate controller.',
            'Any failure or 24-hour timeout stops and preserves original partial evidence; no retry path.'],
    accepted_scope='First TD3+BC Hopper seed202609171 action-harm execution only, both continuations.',
    scientific_results_accepted=False,coverage_accepted=False,global_ready=False)
write_artifact(p/'execution-acceptance.json',out)
print(json.dumps(dict(accepted=True,declaration_sha256=out['declaration_sha256'],regression_tests=103,
                     outcome_transitions_not_started=True)))
