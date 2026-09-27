"""First-pair support/candidate precommit. No simulator construction or outcomes."""
from copy import deepcopy
import gzip,importlib.metadata,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.ood.collect import require,file_hash,read_artifact,write_artifact,sha,support_bank,support_query
from experiments.ood.production import inputs,worker_lease,PAIR,LOCK
from experiments.ood.real_connection import _load_pair,linux_path
from experiments.ood.streaming import FrozenPolicy
from experiments.ood.candidates import action_bank,score_bank,SLOTS


def accepted_states(repo,states,actual_exit):
    ex=json.loads(actual_exit.read_text())
    require(ex['actual_returncode']==0 and ex['timeout'] is False and ex['interruption_signal'] is None,
            'No accepted state-worker actual exit.')
    dispatch=json.loads((actual_exit.parent/'dispatch.json').read_text())['command']
    require(dispatch[dispatch.index('--phase')+1]=='states' and dispatch[dispatch.index('--attempt')+1]==str(states),
            'Wrong state worker command.')
    independent=json.loads((states/'independent-check.json').read_text())
    require(independent['accepted'] is True
            and independent['acceptance_sha256']==file_hash(states/'acceptance.json')
            and independent['worker_actual_exit_sha256']==file_hash(actual_exit)
            and independent['archive_sha256']==file_hash(states/'transitions.sqlite'),
            'Missing or changed independent state acceptance.')
    result=read_artifact(states/'acceptance.json',independent['acceptance_sha256'])
    require(result['state_bank_completed'] is True and len(result['rows'])==256
            and result['paired_reset_blocks']==64 and result['captured']+result['missing']==256,
            'Incomplete declared state layout.')
    declaration=read_artifact(states/'declaration.json',result['declaration_sha256'])
    require(declaration['pair_id']==PAIR and declaration['phase']=='states','Wrong state pair.')
    for rel,h in declaration['source_sha256'].items():require(file_hash(repo/rel)==h,'Changed state execution source.')
    for row in result['rows']:
        require(file_hash(states/'states'/f"state-{row['index']:03d}.json")==row['sha256'],'Changed saved state row.')
    require(file_hash(states/'continuation-keys.json')==result['continuation_keys_sha256'],'Changed actual continuation keys.')
    return result,dict(acceptance_sha256=file_hash(states/'acceptance.json'),independent_sha256=file_hash(states/'independent-check.json'),
                       actual_exit_sha256=file_hash(actual_exit),archive_sha256=file_hash(states/'transitions.sqlite'),
                       continuation_keys_sha256=result['continuation_keys_sha256'])


def prepare_support(repo,lane,binding,seed):
    """Reuse exact frozen preparation; derive support only from accepted training rows."""
    import h5py
    from runtime.config import typed
    cp=binding['checkpoints']['host'];run=lane/'runs'/cp['run_id']
    manifest=json.loads(gzip.decompress((repo/'experiments/standard_bca/manifest.json.gz').read_bytes()))
    row,=[r for r in manifest['runs'] if r['row']['run_id']==cp['run_id']]
    runner,args,spec,protocol=typed(deepcopy(row['row']))
    audit=json.loads(linux_path(binding['inputs']['host/accepted_audit']['path']).read_text())
    require(file_hash(run/'preparation.json')==audit['evidence_sha256']['preparation.json'],'Changed accepted preparation.')
    metadata=json.loads((run/'preparation.json').read_text())['metadata']
    with h5py.File(metadata['raw_identity']['path'],'r') as source:raw={k:source[k][()] for k in metadata['raw_array_hashes']}
    with patch('gym.make',side_effect=RuntimeError('No candidate-stage simulator constructor allowed')):
        prepared=runner.prepare(raw,args,spec,protocol,max_action=1.,max_episode_steps=1000,raw_identity=metadata['raw_identity'])
    require(prepared.metadata['run_input_hashes']==cp['preparation_input_hashes'],'Changed support preparation inputs.')
    ids=np.asarray(prepared.training_ids)
    require(np.array_equal(ids,np.asarray(metadata['training_converted_ids'])),'Training ID mismatch.')
    episodes=np.asarray(prepared.metadata['original_terminal_timeout_episode_ids'],np.int64)[ids]
    obs=np.asarray(prepared.training.obs);actions=np.asarray(prepared.training.action)
    bank=support_bank(obs,actions,ids,episodes,seed=seed)
    selected=bank['reference_indices'];validation=np.asarray(bank['validation_indices'],np.int64)
    require(not set(bank['reference_episodes'])&set(bank['validation_episodes']),'Support episode overlap.')
    bank.update(validation_observations=obs[validation].copy(),validation_actions=actions[validation].copy(),
        validation_training_ids=ids[validation].copy(),reference_episode_ids=episodes[selected].copy(),
        validation_episode_ids=episodes[validation].copy(),obs_mean=np.asarray(prepared.obs_mean),obs_std=np.asarray(prepared.obs_std),
        training_rows=len(ids),heldout_rows=len(prepared.heldout_ids),all_training_ids_sha256=sha(ids),
        accepted_input_hashes=cp['preparation_input_hashes'],preparation_sha256=file_hash(run/'preparation.json'),
        normalization='Exact native float32 training-complement transform, std plus .001.',
        episode_contract='Original terminal/timeout episode labels among accepted training rows; not independent transition samples.')
    return bank


def run(repo,lane,states,states_exit,attempt):
    require(os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU required before startup.')
    binding,full,pair,pre=inputs(repo)
    state_result,state_binding=accepted_states(repo,states,states_exit)
    paths=['experiments/ood/candidate_stage.py','experiments/ood/candidates.py','experiments/ood/production.py',
           'experiments/ood/streaming.py','experiments/ood/adapters.py','experiments/ood/collect.py','experiments/ood/real_connection.py']
    attempt.mkdir(parents=True,exist_ok=False)
    declaration=dict(schema='ood-candidate-precommit-v1',pair_id=PAIR,state_binding=state_binding,
        original_full_declaration_sha256=pre['declaration_sha256'],training_binding_sha256=pre['binding_sha256'],
        checkpoint_sha256=pre['checkpoint_sha256'],source_sha256={p:file_hash(repo/p) for p in paths},
        support_seed=pair['streams']['preparation'][0],candidate_seeds=pair['streams']['candidates'],
        random_score_seeds=pair['streams']['random_score'],slot_names=SLOTS,shared_lock=str(LOCK),device='cpu',gpu_allocation=0,
        environment_transitions=0,physics_steps=0,scientific_outcomes=False,coverage=False,
        coordinate_contract='Proposed float64 offsets, clipped then cast to float32; native transform once. Score applied native unit-bound coordinates.')
    declaration_sha=write_artifact(attempt/'declaration.json',declaration)
    with worker_lease():
        try:
            from runtime.environment import setup
            setup()
            import jax,jax.numpy as jnp
            require(jax.default_backend()=='cpu','Unexpected backend.')
            state_runtime=read_artifact(states/'runtime.json',file_hash(states/'runtime.json'))
            packages={p:importlib.metadata.version(p) for p in state_runtime['packages']}
            require(packages==state_runtime['packages'],'Changed accepted runtime.')
            write_artifact(attempt/'runtime.json',dict(packages=packages,backend=jax.default_backend(),devices=[str(d) for d in jax.devices()]))
            bank=prepare_support(repo,lane,binding,declaration['support_seed'])
            support_sha=write_artifact(attempt/'support-bank.json',bank)
            models=_load_pair(repo,lane,binding);policies={k:FrozenPolicy(v) for k,v in models.items()}
            require(np.array_equal(bank['obs_mean'],models['host'].mean) and np.array_equal(bank['obs_std'],models['host'].std),
                    'Support and actor normalizations differ.')
            bounds=read_artifact(states/'live-native-bounds.json',file_hash(states/'live-native-bounds.json'))
            rows=[read_artifact(states/'states'/f"state-{r['index']:03d}.json",r['sha256']) for r in state_result['rows']]
            captures=[r for r in rows if r['status']=='captured']
            observations=np.stack([r['observation'] for r in captures])
            actor={k:p.actions(observations) for k,p in policies.items()}
            write_artifact(attempt/'actor-actions-before-candidates.json',dict(state_indices=[r['index'] for r in captures],observations=observations,actions=actor))
            out=[];position=0
            (attempt/'rows').mkdir()
            for row in rows:
                index=row['index']
                if row['status']=='missing':saved=dict(index=index,state_id=row['state_id'],status='missing',reason=row['missing_reason'])
                else:
                    obs=row['observation'];host=actor['host'][position];bca=actor['bca'][position];position+=1
                    normalized=np.asarray((jnp.asarray(obs)-models['host'].mean)/models['host'].std)
                    _,nearest=support_query(bank,normalized,host[None,:])
                    actions=action_bank(host,bca,nearest,declaration['candidate_seeds'][index],bounds)
                    # Save all actions BEFORE widths/support warnings; no score-based selection.
                    action_sha=write_artifact(attempt/'rows'/f'actions-{index:03d}.json',actions)
                    support,_=support_query(bank,normalized,actions['applied'])
                    scores=models['bca'].score(np.repeat(obs[None,:],10,axis=0),actions['applied'])
                    scores=score_bank(actions,scores,support,bank['threshold'],declaration['random_score_seeds'][index])
                    saved=dict(index=index,state_id=row['state_id'],status='captured',snapshot_sha256=row['snapshot_sha256'],
                        action_file_sha256=action_sha,continuation_seed=pair['streams']['continuation'][index],
                        continuation_bank_sha256=state_binding['continuation_keys_sha256'],scores=scores)
                h=write_artifact(attempt/'rows'/f'candidate-{index:03d}.json',saved)
                out.append(dict(index=index,state_id=row['state_id'],status=row['status'],sha256=h))
            for p in policies.values():p.seal()
            require(all(file_hash(repo/p)==h for p,h in declaration['source_sha256'].items()),'Candidate source changed.')
            result=dict(schema='ood-candidate-completion-v1',precommit_completed=True,pair_id=PAIR,rows=out,
                captured=len(captures),missing=256-len(captures),support_bank_sha256=support_sha,declaration_sha256=declaration_sha,
                actual_exit_required_separately=True,independent_audit_required=True,scientific_outcomes_accepted=False,
                coverage_accepted=False,environment_transitions=0,physics_steps=0,actor_query_batches=2,width_query_batches=len(captures))
            write_artifact(attempt/'acceptance.json',result);print(json.dumps({k:result[k] for k in ['captured','missing','environment_transitions']}))
        except BaseException as exc:
            write_artifact(attempt/'failure.json',dict(error=repr(exc),no_retry=True));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for name in ('repo','lane','states','states-exit','attempt'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.repo,a.lane,a.states,a.states_exit,a.attempt)
