"""Saved-array candidate/support checks. No model, simulator or learner calls."""
import json,sys
from pathlib import Path
import numpy as np
repo=Path('/mnt/c/Users/David Bayha/Documents/GitHub/BCA');sys.path.insert(0,str(repo))
from experiments.ood.collect import file_hash,read_artifact,sha
p=Path('/home/dbayha/bca-work/ood-execution-v1/td3-hopper-s202609171-candidates-v1')
states=Path(str(p).replace('candidates-v1','states-v1'));proc=Path(str(p).replace('candidates-v1','candidates-process-v1'))
def read(n):return read_artifact(p/n,file_hash(p/n))
r=read('acceptance.json');decl=read_artifact(p/'declaration.json',r['declaration_sha256'])
ex=json.loads((proc/'actual_exit.json').read_text());cmd=json.loads((proc/'dispatch.json').read_text())['command']
assert ex['actual_returncode']==0 and not ex['timeout'] and ex['interruption_signal'] is None
assert cmd[cmd.index('--attempt')+1]==str(p) and 'experiments.ood.candidate_stage' in cmd
assert all(file_hash(repo/f)==h for f,h in decl['source_sha256'].items())
assert r['precommit_completed'] is True and r['environment_transitions']==r['physics_steps']==0
assert r['captured']==256 and r['missing']==0 and len(r['rows'])==256
assert file_hash(states/'acceptance.json')==decl['state_binding']['acceptance_sha256']
state_result=read_artifact(states/'acceptance.json',decl['state_binding']['acceptance_sha256'])
assert file_hash(states/'continuation-keys.json')==decl['state_binding']['continuation_keys_sha256']
bank=read_artifact(p/'support-bank.json',r['support_bank_sha256'])
assert bank['training_rows']==998895 and bank['heldout_rows']==1103 and len(bank['training_ids'])==32768 and bank['neighbors']==32
assert bank['seed']==decl['support_seed'] and set(bank['reference_episodes']).isdisjoint(bank['validation_episodes'])
episodes=np.array(sorted(bank['reference_episodes']+bank['validation_episodes']))
rng=np.random.default_rng(bank['seed']);permuted=rng.permutation(episodes);cut=int(.8*len(episodes))
assert permuted[:cut].tolist()==bank['reference_episodes'] and permuted[cut:].tolist()==bank['validation_episodes']
assert len(bank['validation_indices'])==len(bank['validation_episodes'])
assert np.array_equal(bank['validation_episode_ids'],bank['validation_episodes'])
assert set(bank['reference_episode_ids']).issubset(bank['reference_episodes'])
assert not set(bank['training_ids'])&set(bank['validation_training_ids'])
def distance(obs,actions):
    sq=np.mean((bank['observations'].astype(np.float64)-obs)**2,axis=1)
    near=np.lexsort((np.arange(len(sq)),sq))[:32]
    recorded=bank['actions'][near]
    return np.sqrt(np.mean((actions.astype(np.float64)[:,None,:]-recorded[None,:,:])**2,axis=2)).min(axis=1),recorded[0]
dist=[float(distance(obs,a[None])[0][0]) for obs,a in zip(bank['validation_observations'],bank['validation_actions'])]
assert np.array_equal(dist,bank['validation_distances'])
assert float(np.quantile(dist,.95,method='linear'))==bank['threshold'] and bank['thresholded_labels_available'] is True
actors=read('actor-actions-before-candidates.json');assert actors['state_indices']==list(range(256))
bounds=read_artifact(states/'live-native-bounds.json',file_hash(states/'live-native-bounds.json'))
lo,hi=bounds['native_low'],bounds['native_high']
aliases=0;distant=0;max_dose_error=0.;clipped=0
for i,item in enumerate(r['rows']):
    state=read_artifact(states/'states'/f'state-{i:03d}.json',state_result['rows'][i]['sha256'])
    row=read_artifact(p/'rows'/f'candidate-{i:03d}.json',item['sha256'])
    actions=read_artifact(p/'rows'/f'actions-{i:03d}.json',row['action_file_sha256'])
    scores=row['scores'];host=actors['actions']['host'][i];bca=actors['actions']['bca'][i]
    assert row['snapshot_sha256']==state['snapshot_sha256'] and row['state_id']==state['state_id']
    assert np.array_equal(actors['observations'][i],state['observation'])
    assert actions['seed']==decl['candidate_seeds'][i] and scores['random_seed']==decl['random_score_seeds'][i]
    normalized=(state['observation'].astype(np.float32)-bank['obs_mean'])/bank['obs_std']
    distances,nearest=distance(normalized,actions['applied'])
    # The nearest observation is independent of which candidate action is queried.
    gen=np.random.default_rng(actions['seed']);uniform=gen.uniform(-1,1,host.shape).astype(np.float32)
    direction=gen.normal(size=host.shape);direction/=np.sqrt(np.mean(direction**2))
    proposed=np.stack([host,bca,nearest,uniform]+[host.astype(np.float64)+sign*rho*direction for rho in (.05,.15,.30) for sign in (-1,1)])
    assert np.array_equal(actions['direction'],direction) and np.array_equal(actions['proposed'],proposed)
    sent=np.clip(proposed,-1,1).astype(np.float32);applied=np.clip(lo+(sent+1)*.5*(hi-lo),lo,hi)
    assert np.array_equal(actions['sent'],sent) and np.array_equal(actions['applied'],applied)
    assert np.array_equal(scores['support'],distances) and scores['support_threshold']==bank['threshold']
    assert np.array_equal(scores['support_distant'],distances>bank['threshold'])
    expected_alias=[next((j for j in range(k) if np.array_equal(applied[k],applied[j])),k) for k in range(10)]
    assert actions['alias']==expected_alias and scores['action_bank_sha256']==sha(actions)
    random=np.random.default_rng(scores['random_seed']).random(10)
    for k,j in enumerate(expected_alias):
        if k!=j:random[k]=random[j];aliases+=1
    assert np.array_equal(random,scores['random']) and scores['native_width'] is None
    radius=scores['radius'];unit=scores['unit'];scale=scores['scale']
    assert float(radius)==max(float(scores['bayesian_radius']),float(scores['conformal_radius']))
    expected_width=radius*(np.maximum(scale,np.float32(1e-6))*unit)
    assert np.array_equal(expected_width,scores['width']) and scores['usable'].all()
    normalizer=np.maximum(expected_width,unit)
    a=expected_width/normalizer;b=unit/normalizer
    expected_dose=np.float32(1.)+np.float32(.5)*(a/(a+b))
    error=float(np.max(np.abs(expected_dose-scores['dose'])));max_dose_error=max(error,max_dose_error)
    assert np.isfinite(scores['dose']).all() and (scores['dose']>=1).all() and (scores['dose']<=1.5).all()
    assert error<=float(np.finfo(np.float32).eps)*2  # numerical readout only; action/reward gates unchanged
    distant+=int(scores['support_distant'].sum());clipped+=int((actions['clipping_fraction']>0).sum())
out=dict(schema='ood-candidates-independent-check-v1',accepted=True,pair_id=decl['pair_id'],
    acceptance_sha256=file_hash(p/'acceptance.json'),worker_actual_exit_sha256=file_hash(proc/'actual_exit.json'),
    declaration_sha256=file_hash(p/'declaration.json'),support_bank_sha256=r['support_bank_sha256'],
    actual_worker_exit=ex,states=256,candidate_slots=2560,duplicate_aliases=aliases,clipped_slots=clipped,
    support_reference_rows=32768,support_reference_episodes=len(bank['reference_episodes']),
    support_validation_episodes=len(bank['validation_episodes']),support_threshold=bank['threshold'],support_distant_slots=distant,
    actions_and_support_recomputed_exactly=True,both_radii_retained=True,width_arithmetic_exact=True,
    dose_arithmetic_max_abs_error=max_dose_error,continuation_keys_sha256=decl['state_binding']['continuation_keys_sha256'],
    environment_transitions=0,physics_steps=0,additional_model_queries=0,scientific_outcomes_accepted=False,coverage_accepted=False)
with (p/'independent-check.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
