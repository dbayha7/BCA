"""Read-only saved-array audit; no model imports, queries or simulator calls."""
import io,json,sys
from pathlib import Path
import numpy as np
repo=Path('/mnt/c/Users/David Bayha/Documents/GitHub/BCA');sys.path.insert(0,str(repo))
from experiments.ood.collect import read_artifact,file_hash,sha
from experiments.ood.streaming import ArtifactArchive
p=Path('/home/dbayha/bca-work/ood-execution-v1/td3-hopper-s202609171-stream-gate-v1')
proc=Path('/home/dbayha/bca-work/ood-execution-v1/td3-hopper-s202609171-stream-gate-process-v1')
def read(n):return read_artifact(p/n,file_hash(p/n))
result=read('acceptance.json');declaration=read('declaration.json')
ex=json.loads((proc/'actual_exit.json').read_text());dispatch=json.loads((proc/'dispatch.json').read_text())
assert ex['actual_returncode']==0 and not ex['timeout'] and ex['interruption_signal'] is None
assert dispatch['command'][-2:]==['--phase','gate']
assert dispatch['command'][dispatch['command'].index('--attempt')+1]==str(p)
assert file_hash(p/'declaration.json')==result['declaration_sha256']
assert all(file_hash(repo/f)==h for f,h in declaration['source_sha256'].items())
closed=repo/'runs/ood/td3-hopper-s202609171-connection-v2/training-binding.json'
binding=read_artifact(closed,file_hash(closed))
assert len(binding['frozen_source_sha256'])==108
assert all(file_hash(repo/f)==h for f,h in binding['frozen_source_sha256'].items())
before=read('resources-before.json');after=result['resources_after']
assert before['reserved']['global']==[81,324] and after['reserved']['global']==[88,352]
assert after['reserved']['engineering/td3_bc/hopper']==[52,208] and after['entries']-before['entries']==7
assert result['environment_transitions']==7 and result['physics_steps']==28
assert all(v==[0,0] for k,v in after['reserved'].items() if k.startswith(('collection/','outcomes/','repeat_checks/')))
bounds=read('live-native-bounds.json')
assert bounds['native_low'].dtype==np.float32 and np.array_equal(bounds['native_low'],[-1,-1,-1])
assert bounds['native_high'].dtype==np.float32 and np.array_equal(bounds['native_high'],[1,1,1])
assert np.array_equal(bounds['actuator_ctrlrange'],np.array([[-1.,1.]]*3))
comparisons=[];arrays=0
with ArtifactArchive(p/'transitions.sqlite',read_only=True) as archive:
    audited=archive.audit();assert audited==result['archive_audit'] and audited['records']==12
    for method,ref_index,stream_indices in [('host',1,[2,3]),('bca',4,[5,6])]:
        query=read(method+'-query-comparison.json');ref=read(method+'-reference-transition.json')
        assert query['direct'].dtype==query['fast'].dtype==np.float32
        assert query['direct'].tobytes()==query['fast'].tobytes() and np.max(query['error'])==0
        errors=[]
        for repeat,index in enumerate(stream_indices):
            saved=read(f'{method}-stream-{repeat}.json');assert sha(saved)==sha(ref)
            rec=saved['record'];a=rec['proposed_action'];lo=bounds['native_low'];hi=bounds['native_high']
            expected=np.clip(lo+(a+1)*.5*(hi-lo),lo,hi)
            assert rec['applied_action'].dtype==expected.dtype==np.float32
            assert np.array_equal(expected,rec['applied_action']) and np.array_equal(rec['sim_ctrl'],expected)
            # Native hopper cost is computed in applied float32, retained here.
            reconstructed=rec['forward']+1.-.001*np.sum(expected**2)
            err=abs(float(reconstructed)-rec['reward']);assert err<=1e-7
            errors.append(err)
            for suffix in ('-input','-applied',''):
                name=f'transition-{index:05d}{suffix}.npz'
                with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as actual, np.load(p/'reference-npz'/f'transition-{ref_index:05d}{suffix}.npz',allow_pickle=False) as expected_npz:
                    assert set(actual.files)==set(expected_npz.files)
                    for key in actual.files:
                        x,y=actual[key],expected_npz[key]
                        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),(name,key)
                        arrays+=1
        comparisons.append(dict(method=method,max_action_error=0.,repeated_records_and_full_end_states_exact=True,
                                reward_reconstruction_max_error=max(errors)))
assert file_hash(p/'transitions.sqlite')==result['archive_sha256']
out=dict(schema='ood-streaming-independent-check-v1',accepted=True,gate_acceptance_sha256=file_hash(p/'acceptance.json'),
    worker_actual_exit_sha256=file_hash(proc/'actual_exit.json'),archive_sha256=file_hash(p/'transitions.sqlite'),
    declaration_sha256=file_hash(p/'declaration.json'),worker_exit=ex,archive_audit=audited,npz_arrays_exact=arrays,
    comparisons=comparisons,all_108_frozen_files_unchanged=True,environment_transitions=7,physics_steps=28,
    global_cumulative_reserved=[88,352],native_bounds_captured=True,scientific_outcomes=False,coverage=False,
    additional_model_queries=0,additional_simulator_steps=0)
with (p/'independent-check.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
