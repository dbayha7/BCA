"""Recover the saved first CQL host result after its post-run schema failure.

No learner, policy, simulator, or evaluation is executed. Original files and the
original process exit 1 stay unchanged; all new artifacts go to a new directory.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import collections
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
import sys

os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
import h5py
import numpy as np
from flax import serialization

RUN = 'cql-hopper-host-s202609171'
MANIFEST = '13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe'

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def ah(v):
    a = np.asarray(v)
    return hashlib.sha256(json.dumps([a.dtype.str, a.shape]).encode() + np.ascontiguousarray(a).tobytes()).hexdigest()
def module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def save(p, v):
    with p.open('x') as f:
        json.dump(v, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())

def audit(root, validator, output):
    output.mkdir(parents=True, exist_ok=False)
    source = root/'source'; lane = root/'queue-local-v1'; run = lane/'runs'/RUN
    require(sha(source/'manifest.json') == MANIFEST, 'Original manifest mismatch')
    manifest = read(source/'manifest.json'); item, = [x for x in manifest['runs'] if x['row']['run_id'] == RUN]
    row = item['row']; require(read(run/'resolved.json') == row, 'Resolved row changed')
    require(not (run/'result.json').exists() and not (run/'exit.json').exists(), 'Original disposition changed')
    failure = read(run/'failure.json')
    require(failure['type'] == 'KeyError' and failure['message'] == "'normalized_score'", 'Different failure')
    actual_path = lane/'attempts'/RUN/'actual_exit.json'; actual = read(actual_path)
    require(actual['actual_returncode'] == 1 and not actual['timeout'] and actual['interruption_signal'] is None,
            'Unexpected original process disposition')
    protected = [run/n for n in ('resolved.json','source.json','preparation.json','events.jsonl.gz','failure.json')]
    protected += [run/f'checkpoint_{s}.msgpack' for s in row['expected_counts']['checkpoint_steps']]
    protected += [actual_path, lane/'queue_status.json', lane/'controller_exit.json',
                  root/'controller-process-local-v1/actual_exit.json']
    before = {str(p): sha(p) for p in protected}; save(output/'original_hashes.json', before)
    identity = read(run/'source.json'); prep = read(run/'preparation.json'); m = prep['metadata']
    require(prep['accepted'] and prep['learner_updates'] == 0, 'Preparation failed')
    for name, h in manifest['source_sha256'].items():
        require(sha(source/name) == sha(run/'source'/name) == identity[name] == m['source_files'][name] == h,
                'Original source mismatch: '+name)
    require(m['source_files'] == identity, 'Runtime provenance differs')
    require(m['settings']['args'] == {**row['native_args'], 'wandb_project':'unifloral','wandb_team':'flair','wandb_group':'debug'}, 'Native settings differ')
    require(m['settings']['attachment'] == row['config'] and m['settings']['protocol'] == row['protocol'], 'Method or protocol differs')
    require(digest(m['settings']) == m['settings_sha256'], 'Settings digest differs')
    cache = Path.home()/'.d4rl/datasets'/row['cache']['filename']
    require(sha(cache) == row['cache']['sha256'] and cache.stat().st_size == row['cache']['bytes'], 'Cache differs')
    with h5py.File(cache, 'r') as f: raw = {k:f[k][()] for k in m['raw_array_hashes']}
    require({k:ah(v) for k,v in raw.items()} == m['raw_array_hashes'], 'Raw arrays differ')
    require(digest(m['raw_array_hashes']) == m['raw_dataset_sha256'], 'Raw digest differs')
    ids = np.flatnonzero(~raw['timeouts'][:-1].astype(bool))
    ep = np.r_[0, np.cumsum((raw['terminals'] | raw['timeouts'])[:-1], dtype=np.int64)][ids]
    require(np.array_equal(ids, m['conversion_raw_indices']) and np.array_equal(ep,m['converted_episode_ids']), 'Conversion differs')
    n = len(ids); train = np.asarray(m['training_converted_indices']); hold = np.asarray(m['heldout_converted_indices'])
    boundaries = np.r_[0, np.flatnonzero(np.r_[ep[1:] != ep[:-1], True])+1]
    order = np.random.default_rng(row['protocol']['calibration_seed']).permutation(len(boundaries)-1)
    count = np.searchsorted(np.cumsum(np.diff(boundaries)[order]), row['protocol']['calibration_target_size'])+1
    expected_hold = np.sort(np.concatenate([np.arange(boundaries[i],boundaries[i+1]) for i in order[:count]]))
    require(np.array_equal(hold, expected_hold) and np.array_equal(train,np.flatnonzero(~np.isin(np.arange(n),hold))), 'Reservation differs')
    require(len(hold) <= n*row['protocol']['calibration_max_fraction'] and len(train)==998895 and len(hold)==1103, 'Partition sizes differ')
    require(np.array_equal(ids[train],m['training_raw_indices']) and np.array_equal(ids[hold],m['heldout_raw_indices']) and not m['reference_raw_indices'], 'Raw partition mapping differs')
    require(hashlib.sha256(hold.astype('<i8').tobytes()).hexdigest()==m['reservation']['calibration_indices_sha256'], 'Reservation hash differs')
    obs=raw['observations'][ids].astype(np.float32); nxt=raw['observations'][ids+1].astype(np.float32)
    action=raw['actions'][ids].astype(np.float32); done=raw['terminals'][ids].astype(bool)
    mean=obs[train].mean(0); std=obs[train].std(0)+.001
    require(np.array_equal(mean,np.asarray(m['obs_mean'],np.float32)) and np.array_equal(std,np.asarray(m['obs_std'],np.float32)), 'Normalization differs')
    reward_module=module(source/'runtime/reward_range.py','original_reward_range')
    reward, rm=reward_module.normalize_training_rewards(raw['rewards'][ids].astype(np.float32).astype(np.float64),done,train,
        max_episode_steps=1000,reward_scale=1.,reward_bias=0.)
    rm['fit_raw_indices']=ids[rm['fit_converted_indices']].tolist()
    require(rm==m['reward_normalization'], 'Reward fit differs')
    # Hopper bypasses the antmaze-only transform, retaining the fitted CQL reward units.
    arrays=dict(obs=(obs-mean)/std,action=action,reward=reward.astype(np.float32),next_obs=(nxt-mean)/std,done=done.astype(np.float32))
    require({k:ah(v) for k,v in arrays.items()}==m['prepared_array_hashes'], 'Prepared arrays differ')
    for name, indices in [('training',train),('heldout',hold)]:
        require({k:ah(v[indices]) for k,v in arrays.items()}==m['run_input_hashes'][name], 'Pool differs: '+name)
    require(m['run_input_hashes']['reference'] is None and ah(mean)==m['run_input_hashes']['obs_mean'] and ah(std)==m['run_input_hashes']['obs_std'], 'Reference/statistics differ')
    del raw, obs, nxt, arrays, reward
    check=module(validator,'corrected_validation'); checkpoints=[]
    def finite(v):
        if isinstance(v,dict): return all(finite(x) for x in v.values())
        if isinstance(v,(list,tuple)): return all(finite(x) for x in v)
        return v is None or bool(np.isfinite(np.asarray(v)).all())
    for step in row['expected_counts']['checkpoint_steps']:
        path=run/f'checkpoint_{step}.msgpack'; tree=serialization.msgpack_restore(path.read_bytes())
        counts=check.checkpoint_counts(tree,'cql',step)
        require(finite(tree), 'Nonfinite saved checkpoint')
        require(all(tree['state'][k] is None for k in ['calibrator','posterior','residual_scale']), 'Host has BCA state')
        extra={k:check.counters(tree['state']['native'][k]) for k in ['critic1_target','critic2_target','log_alpha','log_alpha_prime']}
        # CQL increments target step on Polyak updates but never their optimizer.
        for name in ('critic1_target','critic2_target'):
            require(extra[name]['/step']==step and all(v==0 for k,v in extra[name].items() if k.endswith('/count')),
                    'Target counter differs: '+name)
        # The disabled Lagrange branch still applies a zero gradient, advancing Adam.
        for name in ('log_alpha','log_alpha_prime'):
            require(all(x==step for x in extra[name].values()), 'Temperature counter differs: '+name)
        checkpoints.append(dict(step=step,path=path.name,sha256=sha(path),counters=counts,extra_counters=extra))
    records=[]; evaluations=[]; metrics=[]; kinds=collections.Counter()
    with gzip.open(run/'events.jsonl.gz','rt') as f:
        for line in f:
            e=json.loads(line); kinds[e['kind']]+=1
            if e['kind']=='prepared': require(e['step']==0 and e['metadata']==m,'Prepared journal differs')
            elif e['kind']=='accepted_scan':
                require(finite(e['metrics_last']), 'Nonfinite logged metrics'); metrics.append(e)
            elif e['kind'] in ('periodic','final'): evaluations.append(e)
            elif e['kind']!='completed': raise ValueError('Unexpected journal event: '+e['kind'])
            records.append({k:v for k,v in e.items() if k!='metadata'})
    require(kinds==dict(prepared=1,accepted_scan=1000,periodic=200,final=1,completed=1), 'Incomplete journal')
    require(records[-1]==dict(kind='completed',step=1000000,final_episodes_requested=20,final_episodes_actual=20,scored_final=True), 'Missing terminal completion')
    protocol=dict(row['protocol']); protocol['evaluation_events']=[SimpleNamespace(**x) for x in protocol['evaluation_events']]
    check.verify_events(SimpleNamespace(**protocol),records,evaluations,host='cql')
    require(digest(row['protocol']['evaluation_events'])==m['evaluation_events_sha256'], 'Bank declaration digest differs')
    max_score_error=0.
    for bank,event in zip(evaluations,row['protocol']['evaluation_events']):
        require(bank['event_sha256']==digest(event) and bank['score_transform']==m['evaluation_score_transform'], 'Bank binding differs')
        require(bank['returns']==[x['return'] for x in bank['episodes']] and bank['scores']==[x['score'] for x in bank['episodes']], 'Episode arrays differ')
        for i,e in enumerate(bank['episodes']):
            require(e['seed']==event['episode_seeds'][i] and e['episode_index']==i and e['worker']==i%10 and e['batch_start']==i//10*10 and e['completed'] is True and 0<e['length']<=1000,'Episode identity/completion differs')
            t=bank['score_transform']; score=100*(e['return']-t['reference_min'])/(t['reference_max']-t['reference_min'])
            max_score_error=max(max_score_error,abs(score-e['score']))
    require(max_score_error<1e-10,'Score arithmetic differs')
    curve=[float(np.mean(e['scores'])) for e in evaluations[:-1]]; final=evaluations[-1]['scores']
    summary=dict(final_mean=float(np.mean(final)),final_median=float(np.median(final)),periodic_curve_mean=float(np.mean(curve)),last_periodic_mean=curve[-1],training_seeds=1,bca_comparison=None)
    require({str(p):sha(p) for p in protected}==before,'Original evidence changed')
    result=dict(completed=True,run_id=RUN,steps_completed=1000000,evaluations=evaluations,checkpoints=checkpoints,
                events_sha256=sha(run/'events.jsonl.gz'),expected_counts=row['expected_counts'],
                recovery=dict(kind='saved_only_post_run_validation',original_run=str(run),original_worker_exit=actual,
                              original_failure=failure,training_rerun=False,simulator_steps_added=0,model_queries_added=0))
    save(output/'recovered_result.json',result); save(output/'evaluations.json',evaluations); save(output/'metric_snapshots.json',metrics)
    audit=dict(schema='standard-cql-saved-result-recovery-v1',accepted_saved_training_result=True,
       original_process_success=False,original_worker_exit=actual,run_id=RUN,manifest_sha256=MANIFEST,
       source_files_checked=len(manifest['source_sha256']),original_files_unchanged=True,host_updates=1000000,
       actor_updates=1000000,updates_per_critic=1000000,journal_counts=dict(kinds),evaluation_episodes=2020,
       metrics_are_sparse_last_rows=True,checkpoint_counters=checkpoints,training_rows=len(train),holdout_rows=len(hold),
       run_input_hashes=m['run_input_hashes'],max_score_arithmetic_error=max_score_error,summary=summary,
       model_queries_added=0,simulator_steps_added=0,training_updates_added=0,ood_ready=False,
       recovered_result_sha256=sha(output/'recovered_result.json'),validator_sha256=sha(validator),
       audit_source_sha256=sha(__file__),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(output/'audit.json',audit); print(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--validator',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.validator,a.output)
