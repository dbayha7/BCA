"""Real ReBRAC checkpoint queries on CPU, zero simulator/training transitions."""
from pathlib import Path
import datetime,hashlib,json,os,sys,traceback

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/users/dbayha/bca-standard-noiw-v1/campaign/source')
LANE=SOURCE.parent/'queue-cluster-v1'
sys.path[:0]=[str(ROOT),str(SOURCE)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
    with (ROOT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']==''
    pins=json.loads((ROOT/'input-pins.json').read_bytes())
    for name,h in pins.items():assert sha(ROOT/name)==h,('Staged input changed',name)
    manifest=json.loads((SOURCE/'manifest.json').read_bytes())
    assert sha(SOURCE/'manifest.json')=='13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe'
    for name,h in manifest['source_sha256'].items():assert sha(SOURCE/name)==h,('Original training source changed',name)
    from runtime.environment import setup
    setup()
    import jax,jax.numpy as jnp,numpy as np,h5py
    from flax import serialization as S
    from runtime.config import typed
    from runtime.td3_bc import _tree_hash
    from runtime.validation import checkpoint_counts
    from experiments.ood.adapters import CheckpointAdapter,_schema_match
    from calibration.dose import frozen_level_dose
    assert jax.default_backend()=='cpu'
    from unittest.mock import patch
    receipts=[]
    for method in ('host','bca'):
        audit=json.loads((ROOT/(method+'-audit.json')).read_bytes())
        assert audit['accepted'] is True and audit['host_updates']==1000000 and audit['actor_updates']==500000
        run=LANE/'runs'/audit['run_id'];preparation=json.loads((run/'preparation.json').read_bytes())
        for name in ('preparation.json','result.json'):
            assert sha(run/name)==audit['evidence_sha256'][name]
        actual=json.loads((LANE/'attempts'/audit['run_id']/'actual_exit.json').read_bytes())
        assert all(actual[k]==audit['actual_worker_exit'][k] for k in audit['actual_worker_exit'])
        assert actual['actual_returncode']==0 and not actual['timeout'] and actual['interruption_signal'] is None
        item,=[r for r in manifest['runs'] if r['row']['run_id']==audit['run_id']]
        runner,args,spec,protocol=typed(item['row']);cfg=spec.config()
        metadata=preparation['metadata'];cache=Path(metadata['raw_identity']['path'])
        assert sha(cache)==audit['data_cache_sha256']
        with h5py.File(cache,'r') as f:raw={k:f[k][()] for k in metadata['raw_array_hashes']}
        # This prepare entry has no constructor. The patch catches accidental
        # load_and_prepare/gym.make use before even a native constructor step.
        with patch('gym.make',side_effect=RuntimeError('No environment in query gate')):
            prepared=runner.prepare(raw,args,spec,protocol,max_action=1.,max_episode_steps=1000,raw_identity=metadata['raw_identity'])
        expected=audit['run_input_hashes'];recomputed=prepared.metadata['run_input_hashes']
        changed=[k for k in expected if expected[k]!=recomputed[k]]
        assert set(changed)<=({'cal_obs_mean','cal_obs_std'} if method=='bca' else set()),changed
        final,=[c for c in audit['checkpoints'] if c['step']==1000000]
        checkpoint=run/'checkpoint_1000000.msgpack';assert sha(checkpoint)==final['sha256']
        payload=checkpoint.read_bytes();decoded=S.msgpack_restore(payload)
        import algorithms.rebrac_bca as P
        rng,template,models=P.initialize(args,cfg,prepared.training)
        expected_tree=dict(state=template,training_rng=rng,step=jnp.int32(0))
        _schema_match(S.to_state_dict(expected_tree),decoded)
        counts=checkpoint_counts(decoded,'rebrac',1000000,args.policy_freq)
        restored=S.from_bytes(expected_tree,payload)
        state=restored['state']
        if method=='bca':
            assert _tree_hash(state.cal_obs_mean)==expected['cal_obs_mean']
            assert _tree_hash(state.cal_obs_std)==expected['cal_obs_std']
            cal=P.Calibrator(jnp.asarray(state.cal_obs_mean),jnp.asarray(state.cal_obs_std),state_dep=True)
            models=(models[0],models[1],cal)
            assert bool(P.calibration_storage_valid(models,state))
        # Bind the existing read-only query methods to the fully schema-checked
        # restored state; only fixed, checkpoint-saved normalization is restored.
        adapter=CheckpointAdapter.__new__(CheckpointAdapter)
        adapter.path=checkpoint;adapter.host='rebrac';adapter.args=args;adapter.config=cfg;adapter.P=P
        adapter.obs_dim=prepared.training.obs.shape[1];adapter.action_dim=prepared.training.action.shape[1]
        adapter.mean=jnp.asarray(prepared.obs_mean);adapter.std=jnp.asarray(prepared.obs_std);adapter.bound=1.
        adapter.models=models;adapter.state=state;adapter.counts=counts
        adapter._restored=restored;adapter._serialized=S.to_bytes(restored);adapter._file_sha=final['sha256']
        # Fixed first 16 training rows: no score-based selection or new RNG bank.
        batch=jax.tree_util.tree_map(lambda x:x[:16],prepared.training)
        reference=np.asarray(models[0].apply(state.native.actor.params,batch.obs))
        actual_action=adapter.actions(batch.obs, ) if np.array_equal(prepared.obs_mean,np.zeros(adapter.obs_dim)) and np.array_equal(prepared.obs_std,np.ones(adapter.obs_dim)) else None
        assert actual_action is not None,'This fixed raw-observation gate requires the declared unnormalized ReBRAC host'
        error=np.abs(actual_action.astype(np.float64)-reference.astype(np.float64))
        with (ROOT/(method+'-action-parity.npz')).open('xb') as f:
            np.savez(f,observations=np.asarray(batch.obs),actual=actual_action,reference=reference,error=error)
        assert np.max(error)<=1e-6
        target_key=np.asarray(restored['training_rng'],np.uint32)
        target=adapter.targets(batch,target_key)
        _,next_q=adapter.target_components(batch,target_key)
        independent=np.asarray(batch.reward+args.gamma*(1-batch.done)*next_q)
        with (ROOT/(method+'-target-parity.npz')).open('xb') as f:
            np.savez(f,target=target,independent=independent,key=target_key)
        assert np.array_equal(target,independent),'Recorded-next-action target formula differs'
        score=adapter.score(batch.obs,actual_action)
        if method=='bca':
            scale=models[2].apply(state.posterior.cal_params,batch.obs,jnp.asarray(actual_action))
            direct=frozen_level_dose(state.posterior,scale,'full',cfg.blend)
            assert np.array_equal(np.asarray(direct.width),score['width']) and np.array_equal(np.asarray(direct.dose),score['dose'])
            with (ROOT/(method+'-frozen-score.npz')).open('xb') as f:np.savez(f,**score)
        else:assert score is None
        adapter.assert_unchanged()
        receipts.append(dict(method=method,run_id=audit['run_id'],checkpoint_sha256=final['sha256'],counts=counts,
            action_max_error=float(np.max(error)),target_arithmetic_exact=True,native_width_na=score is None,
            saved_calibrator_statistics_restored=method=='bca',cpu_preparation_hash_differences=changed,
            posterior_score_exact=method=='bca',checkpoint_unchanged=True))
    save('query-acceptance.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        query_gate_passed=True,pair=['rebrac','hopper',202609171],backend=jax.default_backend(),receipts=receipts,
        simulator_steps=0,training_updates=0,simulator_gate_pending=True,ood_outcomes_complete=False))
    print(json.dumps(dict(query_gate_passed=True,receipts=receipts,simulator_steps=0)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
