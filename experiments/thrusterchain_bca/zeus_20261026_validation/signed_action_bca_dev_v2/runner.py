"""One-shot, bounded fresh-data frozen-checkpoint diagnostic. See PROTOCOL.md."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'action_sensitive_bca_dev_v1'
RERUN = HERE.parent / 'central_quartile_training_v1_evaluation_rerun1'
sys.path.insert(0, str(RERUN))
sys.path.insert(0, str(PREVIOUS))
sys.path.insert(0, str(HERE))
import diagnostic_runner as old
import evaluation
import evaluate_worker as frozen
import runtime
import compatibility
import q1_bca
import revised_compat
import signed_bca
import fresh_data

CAP = 197704
SECONDS = 7200
SOURCES = ('PROTOCOL.md', 'runner.py', 'signed_bca.py', 'revised_compat.py', 'fresh_data.py')
BLOCKS = {'ordinary': (2026121000, 2026121100, 2026121401, 'recorded'),
          'strong_current': (2026121200, 2026121300, 2026121402, 'strong_current')}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def seal():
    target = HERE / 'SOURCE_SEAL.json'
    require(not target.exists(), 'Source seal already exists')
    terminal, report, refs = old.completed()
    old.check_seal(refs)
    require((PREVIOUS/'diagnostic_run_01/RESULT.json').is_file(), 'Missing previous diagnostic')
    write_json(target, {'schema': 'signed_frozen_diagnostic_source_seal_v1',
        'sources': {name:digest(HERE/name) for name in SOURCES},
        'previous_development_seal_sha256': digest(PREVIOUS/'DEV_SEAL.json'),
        'previous_result_sha256': digest(PREVIOUS/'diagnostic_run_01/RESULT.json'),
        'reference_hashes': refs})
    return digest(target)


def check_seal():
    raw=(HERE/'SOURCE_SEAL.json').read_bytes(); data=json.loads(raw)
    require(data['schema']=='signed_frozen_diagnostic_source_seal_v1'
            and set(data['sources'])==set(SOURCES), 'Invalid source seal')
    require(all(digest(HERE/k)==v for k,v in data['sources'].items()), 'Source changed after seal')
    require(digest(PREVIOUS/'DEV_SEAL.json')==data['previous_development_seal_sha256']
            and digest(PREVIOUS/'diagnostic_run_01/RESULT.json')==data['previous_result_sha256'],
            'Previous diagnostic identity differs')
    _, _, refs=old.completed(); old.check_seal(refs)
    require(data['reference_hashes']==refs,'Reference identity differs')
    return data, hashlib.sha256(raw).hexdigest()


def restore_aux(ctx, agent, norm, seed, case, which):
    from flax import serialization
    models, aux, config, args=q1_bca.fresh_auxiliary_exact_normalization(ctx,agent,seed,
        expected_mean=norm['saved_mean'],expected_std=norm['saved_std'])
    desc=case['q1_auxiliary_artifact' if which=='q1' else 'fresh_twin_min_auxiliary_artifact']
    raw=old.artifact(PREVIOUS/'diagnostic_run_01',desc)
    template={'calibrator':aux.calibrator,'posterior':aux.posterior,'residual_scale':aux.residual_scale,
              'aux_rng':ctx.native.jax.random.PRNGKey(seed)}
    restored=serialization.from_bytes(template,raw)
    aux=aux._replace(calibrator=restored['calibrator'],posterior=restored['posterior'],
                     residual_scale=restored['residual_scale'])
    require(int(np.asarray(aux.calibrator.step))==2000,'Saved auxiliary update count differs')
    return models,aux,config,args


def rms(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x,dtype=np.float64)**2,axis=1))))


def width_function(jax, pipeline, models, ref):
    def width(o,a):
        w, usable, valid=pipeline.live_width(models,ref,o,a)
        return w,usable,valid
    return jax.jit(width)


def signed_function(jax, models, state, threshold):
    # Function-local captures prevent later checkpoint iterations from replacing
    # the model/state when JAX traces an unseen evaluation batch shape.
    def predict(o,a):
        return signed_bca.predict(models,state,o,a,threshold)
    return jax.jit(predict)


def screen(cases):
    results={}
    for context in old.CONTEXTS:
        groups={}; opportunities={}; cells={}
        for arm in old.ARMS:
            count=0; eligible=0
            for seed in old.SEEDS:
                case=next(x for x in cases if x['seed']==seed and x['arm']==arm)
                block=case['contexts'][context]; p=block['per_probe']; high=np.asarray(p['high_load'],bool)
                values=np.asarray(p['qpi']); raw=np.asarray(p['q1']); scores={k:np.asarray(v) for k,v in p['scores'].items()}
                def choices(x): return old.selected(x)
                def regret(x):
                    c=choices(x); return values.max(axis=1)-values[np.arange(32),c]
                raw_regret=regret(raw); combined=regret(scores['combined']);
                opportunity=bool(high.sum()>=8 and (raw_regret[high]>.1).sum()>=8)
                eligible+=opportunity
                diffs={k:float((regret(scores[k])[high]-combined[high]).mean()) if high.any() else None
                       for k in ('signed_bayes','revised_compatibility','strength_matched_q1')}
                b=choices(raw); a=choices(scores['combined']); gain=values[np.arange(32),a]-values[np.arange(32),b]
                improved=int((gain[high]>.1).sum()); harmed=int((gain[high]<-.1).sum())
                qmae=block['scores']['q1_aligned']['error']['high_band_fixed']['mae']
                cmae=block['scores']['combined']['error']['high_band_fixed']['mae']
                mae_gain=None if qmae is None or cmae is None else qmae-cmae
                qualifies=bool(opportunity and mae_gain is not None and mae_gain>=1. and
                               all(v is not None and v>=.1 for v in diffs.values()) and harmed<=improved)
                count+=qualifies
                cells[f'{arm}/seed{seed}']={'opportunity':opportunity,'high_anchors':int(high.sum()),
                    'raw_regret_over_point_one':int((raw_regret[high]>.1).sum()),'qualifies':qualifies,
                    'high_band_mae_improvement_vs_q1':mae_gain,'regret_improvement':diffs,
                    'improved_switches':improved,'harmed_switches':harmed}
            groups[arm]=count
            opportunities[arm]=eligible
        passed=all(v>=2 for v in groups.values())
        status='passed' if passed else ('inconclusive' if any(v<2 for v in opportunities.values()) else 'failed')
        results[context]={'qualifying_seeds':groups,'eligible_seeds':opportunities,'passed':passed,'status':status,'cells':cells}
    overall='passed' if any(v['passed'] for v in results.values()) else (
        'inconclusive' if any(v['status']=='inconclusive' for v in results.values()) else 'failed')
    return {'passed':overall=='passed','status':overall,'contexts':results,
            'scope':'exploratory forced-action mechanism; no policy training or policy-efficacy claim'}


def run(output):
    started=time.monotonic(); deadline=started+SECONDS
    seal_data,seal_sha=check_seal()
    terminal,report,refs=old.completed()
    proto_raw=(RERUN/'protocol.json').read_bytes(); protocol=json.loads(proto_raw)
    require(old.sha(proto_raw)==terminal['protocol_sha256'],'Training protocol differs')
    prev=json.loads((PREVIOUS/'diagnostic_run_01/RESULT.json').read_bytes())
    require(prev['status']=='completed_development_diagnostic','Prior diagnostic incomplete')
    native=runtime.load_native(); receipt=old.cpu_receipt(native); jax,jnp=native.jax,native.jnp
    core,reference,core_receipt=frozen.load_simulator()
    ledger=reference.new_ledger(CAP, 190000)
    ledger.update(max_resets=257,max_critic_rows=3840,auxiliary_updates=0,calibration_refreshes=0,
                  actor_updates=0,critic_updates=0,actor_target_updates=0,critic_target_updates=0)
    def progress(event, **extra):
        old.max_time(started)
        with (output/'progress.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps({'event':event,'elapsed_seconds':time.monotonic()-started,
                'ledger':ledger,**extra},sort_keys=True,allow_nan=False)+'\n');f.flush()
        print(event,flush=True)
    progress('initialized')
    fresh_data.preflight()
    banks={}
    for name,(qseed,fseed,pseed,law) in BLOCKS.items():
        banks[name]=evaluation.build_context_bank(core,reference,context=name,
            q_seeds=np.arange(qseed,qseed+32,dtype=np.int64),free_seeds=np.arange(fseed,fseed+32,dtype=np.int64),
            prefix_seed=pseed,start_law=core.canonical.START_LAWS[law],ledger=ledger,deadline=deadline)
        bank=banks[name]
        np.savez_compressed(output/(name+'_bank.npz'),states=bank['states'],times=bank['times'],
            observations=bank['observations'],free_starts=bank['free_starts'],q_seeds=bank['q_seeds'],free_seeds=bank['free_seeds'])
        if name=='ordinary':
            # The passed adapter hard-codes a reset bound of 81 for parity.
            parity=reference.runtime_parity(core,bank['states'][0],ledger,deadline=deadline)
    progress('evaluation_banks_created')
    cal,check,fresh_receipt=fresh_data.collect(native,core,reference,ledger,deadline,output)
    progress('fresh_calibration_collected')
    require(len(cal.obs)==len(check.obs)==64,'Fresh calibration/check sizes differ')
    prepared=[]; source_hashes={}; compat_ref=None; cross=None; cref=None
    for seed in old.SEEDS:
        data=runtime.module('_signed_diagnostic_data',RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt);ctx=adapter.context
        x=np.asarray(data.train.obs,np.float32); a=np.asarray(data.train.action,np.float32)
        rows=np.asarray(data.source_row_ids,np.int64); episodes=rows//100; times=rows%100
        require(len(rows)==67108 and len(np.unique(episodes))==768,'Training identity differs')
        probe=np.asarray(json.loads((RERUN/'DATA_PLAN.json').read_bytes())['conditions']['hole']['gradient_probe_source_row_ids'],np.int64)
        lookup={int(v):i for i,v in enumerate(rows)}; ix=np.array([lookup[int(v)] for v in probe],np.int64)
        if compat_ref is None:
            compat_ref,cross,cref=revised_compat.fit_crossvalidated(x,a,episodes,times,ix,row_ids=rows)
            orig_ref=compatibility.fit_reference(x,a,episodes,times,row_ids=rows,k=64,time_bin=10)
            orig_cross=compatibility.crossfit_query(x,a,episodes,times,ix,row_ids=rows,k=64,time_bin=10)
            row_digest=old.sha(rows.astype('<i8').tobytes())
        else: require(old.sha(rows.astype('<i8').tobytes())==row_digest,'Training data differs across checkpoints')
        for arm in old.ARMS:
            progress('checkpoint_fitting_started',seed=seed,arm=arm)
            state, hashes,norm,branch=old.checkpoint(seed,arm,protocol,old.sha(proto_raw),adapter,template)
            source_hashes.update(hashes);before=runtime.tree_digest(state);agent=state.microstep.agent_state
            policy=frozen.FrozenPolicy(runtime,branch,state)
            previous=next(c for c in prev['cases'] if c['seed']==seed and c['arm']==arm)
            cached,_=old.load_cached(report,seed,arm,'ordinary'); oldbank=old.cached_banks(report)['ordinary']
            prior_actions=np.asarray(cached['q']['per_probe']['actions'],np.float32)
            prior_obs=np.repeat(oldbank['observations'],6,axis=0)
            old.parity(policy.act(oldbank['observations']),prior_actions[:,0,:], 'Saved actor parity')
            old.parity(policy.q(prior_obs,prior_actions.reshape(-1,4))[:,0].reshape(32,6),
                       cached['q']['per_probe']['q1'],'Saved Q1 parity')
            native_model=agent.actor.apply_fn.__self__
            original_cal=ctx.pipeline.Calibrator(native_model.obs_mean,native_model.obs_std,state_dep=True)
            original_models=(ctx.models[0],ctx.models[1],original_cal)
            original_width=width_function(jax,ctx.pipeline,original_models,state.aux.posterior)
            ow,ou,ov=original_width(jnp.asarray(prior_obs),jnp.asarray(prior_actions.reshape(-1,4)))
            require(bool(np.asarray(ou).all()) and bool(np.asarray(ov).all()),'Original width invalid')
            old.parity(np.asarray(ow).reshape(32,6),
                (np.asarray(cached['q']['per_probe']['q1'])-np.asarray(cached['q']['per_probe']['adjusted']))/3.,
                'Saved original width parity')
            init_seed=previous['auxiliary_init_seed']
            qm,qa,qc,qargs=restore_aux(ctx,agent,norm,init_seed,previous,'q1')
            tm,ta,tc,targs=restore_aux(ctx,agent,norm,init_seed,previous,'twin')
            sm,sa,sc,sargs=signed_bca.initialize(ctx,agent,init_seed+10000,
                expected_mean=norm['saved_mean'],expected_std=norm['saved_std'])
            train=jax.jit(lambda st,b,k:signed_bca.fit_step(ctx.pipeline,sargs,sm,st,b,k))
            rng=np.random.Generator(np.random.PCG64(init_seed+10000));key=jax.random.PRNGKey(init_seed+10000)
            for step in range(2000):
                old.max_time(started)
                batch=old.sliced(jax,ctx.train_transition,rng.integers(0,len(rows),256))
                key,sub=jax.random.split(key);sa,metrics=train(sa,batch,sub)
                require(bool(np.asarray(metrics['signed_fit_accepted'])),'Signed fit rejected')
                ledger['auxiliary_updates']+=1
            require(int(np.asarray(sa.calibrator.step))==2000,'Signed optimizer count differs')
            key,sub=jax.random.split(key); pipeline=q1_bca.make_q1_pipeline(ctx.pipeline)
            ids=np.asarray(fresh_receipt['calibration_row_ids'],np.int64)
            episode_ids=np.asarray(fresh_receipt['episode_seeds'][:64],np.int64)
            qa,qflags,qd=pipeline.refresh(qargs,qc,qm,qa,cal,sub,heldout_ids=ids)
            ta,tflags,td=ctx.pipeline.refresh(targs,tc,tm,ta,cal,sub,heldout_ids=ids)
            sr=signed_bca.refresh(ctx.pipeline,sargs,sm,sa,cal,sub,heldout_ids=episode_ids)
            ledger['calibration_refreshes']+=3
            require(bool(qflags['posterior_certified']) and bool(tflags['posterior_certified'])
                    and bool(np.asarray(ctx.pipeline.reference_valid(qa.posterior)))
                    and bool(np.asarray(ctx.pipeline.reference_valid(ta.posterior))), 'Symmetric calibration rejected')
            qwidth=width_function(jax,pipeline,qm,qa.posterior); twidth=width_function(jax,ctx.pipeline,tm,ta.posterior)
            signed=signed_function(jax,sm,sa,sr['q_bayes'])
            split=signed_function(jax,sm,sa,sr['q_split'])
            obs=jnp.asarray(x[ix]); acts=policy._act(obs); n=jnp.asarray(cross['actions']);h=jnp.asarray(cross['bandwidths'],jnp.float32)
            guard=jnp.asarray(cross['reliable'],jnp.float32)
            qg=jax.grad(lambda z:jnp.sum(policy._q(obs,z)[:,0]))(acts)
            ug=jax.grad(lambda z:jnp.sum(qwidth(obs,z)[0]))(acts)
            sg=jax.grad(lambda z:jnp.sum(signed(obs,z)[0]))(acts)
            dg=jax.grad(lambda z:jnp.sum(revised_compat.compatibility_cost_jax(z,n,h)*guard))(acts)
            on=jnp.asarray(orig_cross['actions']);oh=jnp.asarray(orig_cross['bandwidths'],jnp.float32)
            odg=jax.grad(lambda z:jnp.sum(compatibility.compatibility_cost_jax(z,on,oh)))(acts)
            qr,ur,dr,odr=map(rms,(qg,ug,dg,odg))
            require(all(np.isfinite(v) and v>1e-6 for v in (qr,ur,dr,odr)),'Invalid gradient matching')
            eta=.25*qr/dr; orig_eta=.25*qr/odr; match=rms(np.asarray(sg)+eta*np.asarray(dg))/ur
            require(0<eta<=30 and 0<orig_eta<=30 and 0<match<=30,'Gradient coefficient invalid')
            cq,cy=signed_bca.q1_and_target(ctx.pipeline,sargs,sm,sa,check,jax.random.fold_in(key,1357911))
            _,mu,sigma=signed(check.obs,check.action);err=np.asarray(cq)-np.asarray(cy);mu=np.asarray(mu);sigma=np.asarray(sigma)
            checked={'td_residual':err.tolist(),'mu':mu.tolist(),'sigma':sigma.tolist(),
                     'signed_mae':float(np.mean(abs(err-mu))),'zero_mae':float(np.mean(abs(err))),
                     'upper_td_coverage':float(np.mean(err<=mu+sr['q_bayes']*sigma))}
            from flax import serialization
            signed_raw=serialization.to_bytes(sa.calibrator)
            filename=f'seed{seed}_{arm}_signed.msgpack';(output/filename).write_bytes(signed_raw)
            prepared.append(dict(seed=seed,arm=arm,policy=policy,state=state,before=before,ctx=ctx,
                qwidth=qwidth,twidth=twidth,original_width=original_width,signed=signed,split=split,eta=eta,orig_eta=orig_eta,match=match,
                qthreshold=float(np.asarray(qa.posterior.threshold)),tthreshold=float(np.asarray(ta.posterior.threshold)),
                sr=sr,check=checked,hashes=hashes,norm=norm,signed_artifact={'path':filename,'sha256':old.sha(signed_raw)},
                gradient={'q_rms':qr,'symmetric_width_rms':ur,'signed_rms':rms(sg),'revised_cost_rms':dr,
                          'original_cost_rms':odr,'eta':eta,'original_eta':orig_eta,'matched_kappa':match}))
            require(runtime.tree_digest(state)==before,'Frozen training state changed during fitting')
            progress('checkpoint_fitted',seed=seed,arm=arm)
    require(ledger['auxiliary_updates']==12000 and ledger['calibration_refreshes']==18,'Fit budget differs')
    progress('all_scores_frozen')
    cases=[]
    for p in prepared:
        case={k:p[k] for k in ('seed','arm','qthreshold','tthreshold','sr','check','hashes','norm','signed_artifact','gradient')}
        case['contexts']={};policy=p['policy']
        for context in old.CONTEXTS:
            bank=banks[context]; out=evaluation.evaluate_snapshot(core,reference,policy,bank,ledger,trained=True,deadline=deadline)
            panel=out['q']['per_probe']; obs=np.asarray(bank['observations'],np.float32);acts=np.asarray(panel['actions'],np.float32)
            flat=acts.reshape(-1,4); repeated=np.repeat(obs,6,axis=0);raw=np.asarray(panel['q1'],np.float64);values=np.asarray(panel['qpi'],np.float64)
            def score_all(o,actions,q):
                n=len(o); ti=old.time_index(o); query=revised_compat.query_neighbors(compat_ref,o,ti)
                oq=compatibility.query_neighbors(orig_ref,o,ti);nbs=query['actions'];guard=np.asarray(query['reliable'],np.float32)
                c=np.asarray(revised_compat.compatibility_cost_jax(jnp.asarray(actions),jnp.asarray(nbs),
                    jnp.asarray(compat_ref.base.action_bandwidth,jnp.float32)))*guard
                oc=np.asarray(compatibility.compatibility_cost_jax(jnp.asarray(actions),jnp.asarray(oq['actions']),
                    jnp.asarray(orig_ref.action_bandwidth,jnp.float32)))
                # Permute at the state-family level for the bank; it breaks the
                # state-neighbor match but preserves actions and all score labels.
                if n==192:
                    perm=np.random.Generator(np.random.PCG64(2026121510)).permutation(32)
                    shuffled=np.repeat(nbs[::6][perm],6,axis=0)
                else:
                    perm=np.random.Generator(np.random.PCG64(2026121510)).permutation(n)
                    shuffled=nbs[perm]
                sh=np.asarray(revised_compat.compatibility_cost_jax(jnp.asarray(actions),jnp.asarray(shuffled),
                    jnp.asarray(compat_ref.base.action_bandwidth,jnp.float32)))*guard
                w,wu,wv=p['qwidth'](jnp.asarray(o),jnp.asarray(actions));tw,tu,tv=p['twidth'](jnp.asarray(o),jnp.asarray(actions))
                ow,ou,ov=p['original_width'](jnp.asarray(o),jnp.asarray(actions))
                require(all(bool(np.asarray(v).all()) for v in (wu,wv,tu,tv,ou,ov)),'Width invalid')
                sp,mu,sigma=p['signed'](jnp.asarray(o),jnp.asarray(actions));split,_,_=p['split'](jnp.asarray(o),jnp.asarray(actions))
                w,tw,ow,sp,split,mu,sigma=map(lambda z:np.asarray(z,np.float64),(w,tw,ow,sp,split,mu,sigma))
                scores={'raw_q1':q,'original_bca':q-3*ow,'fresh_twin_min':q-3*tw,'q1_aligned':q-3*w,
                    'original_compatibility':q-p['orig_eta']*oc,'revised_compatibility':q-p['eta']*c,
                    'signed_bayes':q-sp,'signed_split':q-split,'combined':q-sp-p['eta']*c,
                    'strength_matched_q1':q-p['match']*w,'shuffled_compatibility':q-p['eta']*sh}
                require(all(np.isfinite(v).all() for v in scores.values()),'Nonfinite score')
                return scores,{'reliable':guard.tolist(),'cost':c.tolist(),'original_cost':oc.tolist(),
                    'shuffled_cost':sh.tolist(),'mu':mu.tolist(),'sigma':sigma.tolist(),
                    'signed_bound':sp.tolist(),'aligned_width':w.tolist(),'twin_width':tw.tolist(),
                    'shuffle_permutation':perm.tolist(),
                    'shuffle_identical_neighbor_sets':int(np.all(shuffled==nbs,axis=(1,2)).sum())}
            scores,extra=score_all(repeated,flat,raw.ravel());scores={k:v.reshape(32,6) for k,v in scores.items()}
            block=old.summarize_bank(raw,values,scores,obs,acts)
            vp=out['visited']['per_row']; vo=np.asarray(vp['observations'],np.float32);va=np.asarray(vp['actions'],np.float32)
            vq=np.asarray(vp['q1'],np.float64);vtruth=np.asarray(vp['qpi_suffix'],np.float64)
            vs,ve=score_all(vo,va,vq)
            vh,vhole=evaluation.hole_mask(vo,va);masks={'all':np.ones(128,bool),'high_hole':vhole,'high_nonhole':vh&~vhole,
                'low_band':~vh&(np.abs(va[:,0])<=.25),'low_nonband':~vh&(np.abs(va[:,0])>.25)}
            block['visited']={k:{m:evaluation.error_metrics(v[z],vtruth[z]) for m,z in masks.items()} for k,v in vs.items()}
            block['per_probe']={'observations':obs.tolist(),'actions':acts.tolist(),'high_load':panel['high_load'],
                'q1':raw.tolist(),'qpi':values.tolist(),'scores':{k:v.tolist() for k,v in scores.items()},'components':extra,
                'bank_id':bank['bank_id']}
            block['visited_per_row']={'observations':vo.tolist(),'actions':va.tolist(),'q1':vq.tolist(),
                'qpi_suffix':vtruth.tolist(),'scores':{k:v.tolist() for k,v in vs.items()},'components':ve}
            block['free']=out['free'];case['contexts'][context]=block
            progress('context_evaluated',seed=p['seed'],arm=p['arm'],context=context)
        require(runtime.tree_digest(p['state'])==p['before'],'Frozen checkpoint changed')
        policy.assert_unchanged();cases.append(case)
        write_json(output/f"seed{p['seed']}_{p['arm']}_case.json",case)
    require(ledger['transition_steps_completed']==ledger['transition_steps_attempted']==CAP,
            'Simulator count differs from exact bounded budget')
    require(ledger['auxiliary_updates']==12000 and ledger['calibration_refreshes']==18 and
            all(ledger[k]==0 for k in ('actor_updates','critic_updates','actor_target_updates','critic_target_updates')),
            'Optimizer ledger differs')
    for path,h in source_hashes.items():require(digest(path)==h,'Checkpoint source changed')
    require(check_seal()[1]==seal_sha,'Source seal changed')
    result={'status':'completed_frozen_checkpoint_diagnostic','source_seal_sha256':seal_sha,
        'source_hashes':source_hashes,'runtime':receipt,'simulator':core_receipt,'parity':parity,
        'fresh_data':fresh_receipt,'compatibility_fit':cref,'training_row_ids_sha256':row_digest,
        'banks':{k:{'bank_id':v['bank_id'],'npz_sha256':digest(output/(k+'_bank.npz'))} for k,v in banks.items()},
        'ledger':ledger,'cases':cases,'mechanism_screen':screen(cases),'elapsed_seconds':time.monotonic()-started,
        'limitations':['Fresh evaluation is after prior development exploration','Signed residual is one-step TD, not Qpi error',
           'Compatibility distance guard is not a support certificate','Frozen action screen does not establish trained-policy performance']}
    write_json(output/'RESULT.json',result);progress('complete');return result


if __name__=='__main__':
    if sys.argv[1:]==['--seal']:
        print(seal());sys.exit(0)
    require(sys.argv[1:]==['--run'],'Use --seal once, then --run once')
    require(os.environ.get('JAX_PLATFORMS')=='cpu','Explicit CPU required')
    check_seal();output=HERE/'diagnostic_run_01';output.mkdir(exist_ok=False)
    try:
        print(json.dumps({'status':run(output)['status']}),flush=True)
    except BaseException as exc:
        write_json(output/'FAILURE.json',{'status':'failed_no_automatic_retry','error':str(exc),'traceback':traceback.format_exc()})
        raise
