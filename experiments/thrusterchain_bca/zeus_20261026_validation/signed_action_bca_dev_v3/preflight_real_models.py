"""Read-only six-checkpoint scale integration preflight; no simulator or fitting.

Temporarily recalibrates on *previously inspected development episodes*, never
on prospective v3 evaluation data. Printed coefficients are plumbing checks,
not prospective results or threshold selection. Saved inputs are not changed.
"""
import json
import numpy as np
import runner

HASHED = (runner.HERE/'runner.py', runner.HERE/'score_controls.py', runner.HERE/'fresh_data.py',
          runner.HERE/'bank_v3.py', runner.HERE/'PROTOCOL.md', runner.V1/'q1_bca.py',
          runner.V2/'signed_bca.py', runner.V2/'revised_compat.py')
SOURCE_HASHES = {str(path):runner.digest(path) for path in HASHED}


def main():
    old, runtime = runner.old, runner.runtime
    native = runtime.load_native(); receipt=old.cpu_receipt(native)
    terminal, _, _ = old.completed(); raw=(runner.RERUN/'protocol.json').read_bytes()
    protocol=json.loads(raw); assert old.sha(raw)==terminal['protocol_sha256']
    _, previous=runner.input_pins()
    prior=json.loads((runner.V1/'diagnostic_run_01/RESULT.json').read_bytes())
    archive=runner.V2/'diagnostic_run_01/fresh_behavior_episodes.npz'
    with np.load(archive,allow_pickle=False) as bank:
        t=np.asarray(bank['selected_times'],np.int64); ix=np.arange(128)
        o=np.asarray(bank['observations']); a=np.asarray(bank['actions'])
        selected=native.Transition(obs=o[ix,t],action=a[ix,t],reward=np.asarray(bank['rewards'])[ix,t].astype(np.float32),
            next_obs=o[ix,t+1],done=np.asarray(bank['terminated'])[ix,t].astype(np.float32))
        cal=old.sliced(native.jax,selected,np.arange(64))
        check=old.sliced(native.jax,selected,np.arange(64,128))
        row_ids=np.asarray(bank['selected_row_ids'],np.int64)[:64]
        episode_ids=np.asarray(bank['episode_seeds'],np.int64)[:64]
    compat_ref=cross=None; tested=0; cases=[]
    for seed in old.SEEDS:
        data=runtime.module('_expanded_match_smoke_data',runner.RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt);ctx=adapter.context
        rows=np.asarray(data.source_row_ids,np.int64);e=rows//100;t=rows%100
        x=np.asarray(data.train.obs,np.float32);a=np.asarray(data.train.action,np.float32)
        source=np.asarray(json.loads((runner.RERUN/'DATA_PLAN.json').read_bytes())['conditions']['hole']['gradient_probe_source_row_ids'],np.int64)
        lookup={int(v):i for i,v in enumerate(rows)};ix=np.array([lookup[int(v)] for v in source],np.int64)
        if compat_ref is None:
            compat_ref,cross,_=runner.revised_compat.fit_crossvalidated(x,a,e,t,ix,row_ids=rows)
        for arm in old.ARMS:
            state,_,norm,branch=old.checkpoint(seed,arm,protocol,old.sha(raw),adapter,template)
            before=runtime.tree_digest(state);agent=state.microstep.agent_state
            policy=runner.frozen.FrozenPolicy(runtime,branch,state)
            oldcase=next(v for v in prior['cases'] if v['seed']==seed and v['arm']==arm)
            v2case=next(v for v in previous['cases'] if v['seed']==seed and v['arm']==arm)
            init=oldcase['auxiliary_init_seed']
            qm,qa,qc,qargs=runner.v2.restore_aux(ctx,agent,norm,init,oldcase,'q1')
            tm,ta,tc,targs=runner.v2.restore_aux(ctx,agent,norm,init,oldcase,'twin')
            sm,sa,sc,sargs=runner.signed_restore(ctx,agent,norm,init+10000,v2case['signed_artifact'])
            key=native.jax.random.PRNGKey(2026122400+seed*2+(arm=='bca'))
            qp=runner.q1_bca.make_q1_pipeline(ctx.pipeline)
            qa,flags,_=qp.refresh(qargs,qc,qm,qa,cal,key,heldout_ids=row_ids)
            ta,tflags,_=ctx.pipeline.refresh(targs,tc,tm,ta,cal,key,heldout_ids=row_ids)
            assert bool(flags['posterior_certified']) and bool(np.asarray(ctx.pipeline.reference_valid(qa.posterior)))
            assert bool(tflags['posterior_certified']) and bool(np.asarray(ctx.pipeline.reference_valid(ta.posterior)))
            sr=runner.signed_bca.refresh(ctx.pipeline,sargs,sm,sa,cal,key,heldout_ids=episode_ids)
            width=runner.v2.width_function(native.jax,qp,qm,qa.posterior)
            twidth=runner.v2.width_function(native.jax,ctx.pipeline,tm,ta.posterior)
            signed=runner.v2.signed_function(native.jax,sm,sa,sr['q_bayes'])
            obs=native.jnp.asarray(x[ix]); act=policy._act(obs)
            m=runner.score_controls.match_controls(obs,act,lambda o,z:policy._q(o,z)[:,0],width,signed,
                runner.revised_compat.compatibility_cost_jax,cross['actions'],cross['bandwidths'],cross['reliable'])
            noise=native.jax.random.fold_in(key,1357911)
            q1,target=runner.signed_bca.q1_and_target(ctx.pipeline,sargs,sm,sa,check,noise)
            heads=np.asarray(sm[1].apply(sa.native.critic.params,check.obs,check.action))
            qw,qu,qv=width(check.obs,check.action); tw,tu,tv=twidth(check.obs,check.action)
            bound,mu,sigma=signed(check.obs,check.action)
            q1,target,qw,tw,bound,mu,sigma=map(np.asarray,(q1,target,qw,tw,bound,mu,sigma))
            assert heads.shape==(64,2) and all(v.shape==(64,) for v in (q1,target,qw,tw,bound,mu,sigma))
            assert all(np.isfinite(v).all() for v in (heads,q1,target,qw,tw,bound,mu,sigma))
            assert all(bool(np.asarray(v).all()) for v in (qu,qv,tu,tv))
            assert np.array_equal(q1,heads[:,0])
            after=runtime.tree_digest(state);assert before==after
            result={'seed':seed,'arm':arm,'state_digest_before':before,'state_digest_after':after,
                'lambda_mean':m['lambda_mean'],'lambda_gradient':m['lambda_gradient'],'eta':m['eta'],
                'hybrid_matched_kappa':m['hybrid_matched_kappa'],'probe_rows':m['probe_count'],
                'check_noise_key':np.asarray(noise).tolist(), 'finite_check_rows':64,
                'q1_centered_td_abs_coverage_count':int(np.sum(abs(target-q1)<=qw)),
                'twin_min_centered_td_abs_coverage_count':int(np.sum(abs(target-heads.min(axis=1))<=tw)),
                'signed_td_upper_coverage_count':int(np.sum(q1-target<=mu+sr['q_bayes']*sigma))}
            cases.append(result);print(json.dumps(result),flush=True)
            tested+=1
    assert tested==6
    assert SOURCE_HASHES=={str(path):runner.digest(path) for path in HASHED}, 'sources changed during preflight'
    output={'status':'passed_no_simulator_or_updates','cases':cases,'probe_rows_per_checkpoint':768,
            'source_hashes':SOURCE_HASHES,
            'source_archive_sha256':runner.digest(archive),'source_role':'previously inspected development episodes; no new evaluation',
            'simulator_transitions':0,'actor_critic_target_auxiliary_updates':0,
            'interpretation':'plumbing test with old data; not prospective calibration or shifted-coverage evidence'}
    runner.write_json(runner.HERE/'PREFLIGHT_REAL_MODELS_FINAL.json',output)
    print('PASS: six restored checkpoints; 768 allowed probes each; no simulator or optimizer updates',flush=True)


if __name__=='__main__': main()
