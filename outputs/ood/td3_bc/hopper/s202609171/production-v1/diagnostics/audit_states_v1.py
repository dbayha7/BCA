"""Inspect original saved production states/transitions only; no simulator/model calls."""
import io,json,sys,sqlite3,collections
from pathlib import Path
import numpy as np
repo=Path('/mnt/c/Users/David Bayha/Documents/GitHub/BCA');sys.path.insert(0,str(repo))
from experiments.ood.collect import file_hash,read_artifact,sha
from experiments.ood.streaming import ArtifactArchive
from experiments.ood.simulator import DATA_FIELDS,MODEL_FIELDS,tree_hash
from experiments.ood.resources import ResourceLedger,scope_caps
p=Path('/home/dbayha/bca-work/ood-execution-v1/td3-hopper-s202609171-states-v1')
proc=Path(str(p).replace('states-v1','states-process-v1'))
def read(n):return read_artifact(p/n,file_hash(p/n))
result=read('acceptance.json');declaration=read('declaration.json')
ex=json.loads((proc/'actual_exit.json').read_text());cmd=json.loads((proc/'dispatch.json').read_text())['command']
assert ex['actual_returncode']==0 and ex['timeout'] is False and ex['interruption_signal'] is None
assert cmd[cmd.index('--phase')+1]=='states' and cmd[cmd.index('--attempt')+1]==str(p)
assert declaration['pair_id']=='td3_bc/hopper/s202609171' and file_hash(p/'declaration.json')==result['declaration_sha256']
assert all(file_hash(repo/f)==h for f,h in declaration['source_sha256'].items())
assert result['state_bank_completed'] is True and result['paired_reset_blocks']==64 and len(result['rows'])==256
accepted=repo/'runs/ood/td3-hopper-s202609171-connection-v2'
pre=read_artifact(accepted/'trained-connection-v1/precommit.json',file_hash(accepted/'trained-connection-v1/precommit.json'))
full=read_artifact(accepted/'full-declaration.json',pre['declaration_sha256'])
binding=read_artifact(accepted/'training-binding.json',pre['binding_sha256'])
assert all(file_hash(repo/f)==h for f,h in binding['frozen_source_sha256'].items())
pair,=[x for x in full['pairs'] if x['pair_id']==declaration['pair_id']]
assert declaration['paired_reset_seeds']==pair['streams']['collection']
keys=read_artifact(p/'continuation-keys.json',result['continuation_keys_sha256'])
assert keys['keys'].shape==(256,250,2) and keys['keys'].dtype==np.uint32
assert keys['seeds']==pair['streams']['continuation'] and keys['horizon']==250 and not keys['used_in_state_collection']
caps=scope_caps([x['pair_id'] for x in full['pairs']],list(full['engineering_streams']))
ledger_path=Path('/home/dbayha/bca-work/ood-resources-v2/ledger.sqlite')
with ResourceLedger(ledger_path,caps,pre['declaration_sha256']) as ledger:
    ledger_audit=ledger.audit();assert ledger_audit==result['resources_after']
    entries=[json.loads(x[0]) for x in ledger.db.execute('SELECT payload FROM entries WHERE token LIKE ? ORDER BY id',(str(p)+'/%',))]
assert entries[0]['token']==str(p)+'/constructor' and entries[0]['scopes']==['engineering/td3_bc/hopper']
assert all(e['environment']==1 and e['physics']==4 for e in entries)
calls=entries[1:];assert len(calls)==result['explicit_transitions']<=12800
assert result['environment_transitions']==len(calls)+1 and result['physics_steps']==4*(len(calls)+1)
before=read('resources-before.json');assert before['reserved']['global']==[88,352]
assert ledger_audit['reserved']['global']==[89+len(calls),356+4*len(calls)]
assert ledger_audit['reserved']['collection/'+declaration['pair_id']]==[len(calls),4*len(calls)]
assert ledger_audit['reserved']['engineering/td3_bc/hopper']==[53,212]
assert all(v==[0,0] for k,v in ledger_audit['reserved'].items() if k.startswith(('outcomes/','repeat_checks/')))
rows={};strata=collections.Counter();schema=None
for i,item in enumerate(result['rows']):
    row=read_artifact(p/'states'/f'state-{i:03d}.json',item['sha256'])
    collector=('host','bca')[i//128];episode=(i%128)//2;t=(0,100)[i%2]
    assert row['index']==i and row['collector']==collector and row['episode']==episode and row['capture_step']==t
    assert row['reset_seed']==pair['streams']['collection'][episode] and row['state_id']==f'{collector}/{episode}/{t}'
    assert row['status']==item['status'];strata[f"{collector}/{t}/{row['status']}"]+=1
    if row['status']=='captured':
        s=row['snapshot'];assert sha(s)==row['snapshot_sha256'] and s['elapsed_steps']==t
        assert all(k in s for k in (*DATA_FIELDS,*MODEL_FIELDS,'rng','action_rng','native_action_rng','time','has_reset','udd_state'))
        assert not s['terminated'] and not s['truncated'] and s['has_reset'] is True
        current={k:None if s[k] is None else (str(s[k].dtype),s[k].shape) for k in (*DATA_FIELDS,*MODEL_FIELDS)}
        if schema is None:schema=current
        assert schema==current
        obs=np.concatenate([s['qpos'][1:],np.clip(s['qvel'],-10,10)])
        assert np.array_equal(obs,row['observation'])
    else:assert t==100 and row['missing_reason']=='episode_ended_before_capture'
    rows[(collector,episode,t)]=row
for episode in range(64):assert tree_hash(rows[('host',episode,0)]['snapshot'])==tree_hash(rows[('bca',episode,0)]['snapshot'])
assert sum(v for k,v in strata.items() if k.endswith('/captured'))==result['captured']
bounds=read('live-native-bounds.json');lo=bounds['native_low'];hi=bounds['native_high']
def decode_sim(v):
    if isinstance(v,dict):
        if set(v)=={'dtype','shape','hex'}:return np.frombuffer(bytes.fromhex(v['hex']),dtype=np.dtype(v['dtype'])).reshape(v['shape']).copy()
        return {k:decode_sim(x) for k,x in v.items()}
    if isinstance(v,list):return [decode_sim(x) for x in v]
    return v
counts=collections.Counter();last={};max_error=0.;first_fields=None
with ArtifactArchive(p/'transitions.sqlite',read_only=True) as archive:
    audit=archive.audit();assert audit==result['archive_audit'] and audit['records']==3*len(calls)
    for number,entry in enumerate(calls,1):
        owner=entry['token'][len(str(p))+1:];collector,ep,t=owner.split('/');ep=int(ep);t=int(t)
        assert entry['scopes']==['collection/'+declaration['pair_id']] and t==counts[(collector,ep)] and t<100
        counts[(collector,ep)]+=1
        with np.load(io.BytesIO(archive.read(f'transition-{number:05d}-input.npz')),allow_pickle=False) as z:
            action=z['proposed_action'].copy();state=decode_sim(json.loads(str(z['state_json'])))
        with np.load(io.BytesIO(archive.read(f'transition-{number:05d}-applied.npz')),allow_pickle=False) as z:applied=z['applied_action'].copy()
        with np.load(io.BytesIO(archive.read(f'transition-{number:05d}.npz')),allow_pickle=False) as z:rec={k:z[k].copy() for k in z.files}
        assert set(state)==set(rows[(collector,ep,0)]['snapshot']) and state['elapsed_steps']==t
        assert not state['terminated'] and not state['truncated']
        if t==0:assert tree_hash(state)==tree_hash(rows[(collector,ep,0)]['snapshot'])
        else:
            prior=last[(collector,ep)]
            assert not bool(prior['terminated'] or prior['truncated'])
            assert np.array_equal(prior['observation'],np.concatenate([state['qpos'][1:],np.clip(state['qvel'],-10,10)]))
        assert action.dtype==applied.dtype==np.float32 and np.array_equal(action,rec['proposed_action'])
        assert np.array_equal(np.clip(lo+(action+1)*.5*(hi-lo),lo,hi),applied)
        assert np.array_equal(applied,rec['applied_action']) and np.array_equal(applied,rec['sim_ctrl'])
        reconstructed=float(rec['forward'])+1.-.001*float(np.square(applied.astype(np.float64)).sum())
        error=abs(float(rec['reward'])-reconstructed);assert error<=1e-7 and error==float(rec['reward_error'])
        max_error=max(max_error,error);assert int(rec['physics_steps'])==4 and int(rec['elapsed_steps'])==t+1
        last[(collector,ep)]=rec
    for collector in ('host','bca'):
        for ep in range(64):
            row=rows[(collector,ep,100)];rec=last[(collector,ep)]
            if row['status']=='captured':
                assert counts[(collector,ep)]==100 and np.array_equal(rec['observation'],row['observation'])
                assert not bool(rec['terminated'] or rec['truncated'])
            else:assert bool(rec['terminated'] or rec['truncated'])
assert file_hash(p/'transitions.sqlite')==result['archive_sha256']
out=dict(schema='ood-production-state-independent-check-v1',accepted=True,pair_id=declaration['pair_id'],
    acceptance_sha256=file_hash(p/'acceptance.json'),worker_actual_exit_sha256=file_hash(proc/'actual_exit.json'),
    archive_sha256=file_hash(p/'transitions.sqlite'),worker_actual_exit=ex,archive_audit=audit,
    state_rows=256,captured=result['captured'],missing=result['missing'],strata=dict(strata),paired_reset_full_states_exact=64,
    all_restore_fields_present=True,native_applied_dtype='float32',all_applied_controls_exact=True,
    reward_reconstruction_max_error=max_error,action_queries=len(calls),environment_transitions=len(calls)+1,
    explicit_transitions=len(calls),constructor_transitions=1,physics_steps=4*(len(calls)+1),
    cumulative_global_reserved=ledger_audit['reserved']['global'],state_bank_accepted=True,
    scientific_action_outcomes_accepted=False,coverage_accepted=False,all_108_frozen_files_unchanged=True,
    additional_model_queries=0,additional_simulator_steps=0)
with (p/'independent-check.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
