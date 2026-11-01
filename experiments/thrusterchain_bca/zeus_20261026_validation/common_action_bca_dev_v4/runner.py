"""One-shot common-action, common-continuation frozen diagnostic."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V3 = BASE/'signed_action_bca_dev_v3'
for path in (V3, HERE):
    sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location('_common_v3_runner', V3/'runner.py')
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
old, runtime, frozen, evaluation = v3.old, v3.runtime, v3.frozen, v3.evaluation
bank_spec = importlib.util.spec_from_file_location('_common_bank_v4', HERE/'common_bank.py')
common_bank = importlib.util.module_from_spec(bank_spec); bank_spec.loader.exec_module(common_bank)
starts_spec = importlib.util.spec_from_file_location('_fresh_starts_v4', HERE/'fresh_starts.py')
fresh_starts = importlib.util.module_from_spec(starts_spec); starts_spec.loader.exec_module(fresh_starts)

CAP = 364904
SECONDS = 7200
SOURCES = ('PROTOCOL.md','runner.py','common_bank.py','fresh_starts.py')
SOURCES_ORDER = ('random_prefix', 'native_visited', 'bca_visited')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def input_pins():
    seal,_ = v3.check_seal()
    path = V3/'diagnostic_run_01/RESULT.json'; result = json.loads(path.read_bytes())
    require(result.get('status')=='completed_expanded_frozen_diagnostic' and
            result.get('source_seal_sha256')==seal, 'v3 diagnostic identity differs')
    archive = V3/'diagnostic_run_01'/result['fresh_data']['artifact']
    require(archive.is_file() and not archive.is_symlink() and
            digest(archive)==result['fresh_data']['artifact_sha256'],'v3 calibration archive differs')
    return {'v3_source_seal_sha256':seal,'v3_result_sha256':digest(path),
            'v3_calibration_archive_sha256':digest(archive)},result,archive


def seal():
    target=HERE/'SOURCE_SEAL.json'; require(not target.exists(),'source seal already exists')
    pins,_,_=input_pins(); audit=fresh_starts.seed_audit()
    write_json(target,{'schema':'bca.common_action_continuation.v4','inputs':pins,'seed_audit':audit,
                       'sources':{x:digest(HERE/x) for x in SOURCES}})
    return digest(target)


def check_seal():
    target=HERE/'SOURCE_SEAL.json'; obj=json.loads(target.read_bytes())
    require(obj['schema']=='bca.common_action_continuation.v4' and set(obj['sources'])==set(SOURCES)
            and all(digest(HERE/k)==v for k,v in obj['sources'].items()),'source seal differs')
    pins,result,archive=input_pins()
    require(obj['inputs']==pins and fresh_starts.seed_audit()==obj['seed_audit'], 'input or seed audit differs')
    return digest(target),result,archive


def archived_calibration(native, archive):
    """Return v3's 64 already-consumed intact calibration tuples and episode identities."""
    with np.load(archive,allow_pickle=False) as bank:
        t=np.asarray(bank['selected_times'],np.int64); ix=np.arange(128)
        obs=np.asarray(bank['observations']); action=np.asarray(bank['actions'])
        require(t.shape==(128,) and obs.shape==(128,101,14) and action.shape==(128,100,4),
                'v3 calibration archive shapes differ')
        all_rows=native.Transition(obs=obs[ix,t],action=action[ix,t],
            reward=np.asarray(bank['rewards'])[ix,t].astype(np.float32),
            next_obs=obs[ix,t+1],done=np.asarray(bank['terminated'])[ix,t].astype(np.float32))
        row_ids=np.asarray(bank['selected_row_ids'],np.int64)[:64]
        episode_ids=np.asarray(bank['episode_seeds'],np.int64)[:64]
    require(len(np.unique(row_ids))==len(np.unique(episode_ids))==64,'duplicate v3 calibration episode')
    return old.sliced(native.jax,all_rows,np.arange(64)),row_ids,episode_ids


def reconstruct(ctx,agent,norm,oldcase,v2case,v3case,cal,row_ids,episode_ids):
    """Replay v3 calibrations; no parameter fitting and no outcome selection."""
    jax=ctx.native.jax; init=oldcase['auxiliary_init_seed']
    qm,qa,qc,qargs=v3.v2.restore_aux(ctx,agent,norm,init,oldcase,'q1')
    tm,ta,tc,targs=v3.v2.restore_aux(ctx,agent,norm,init,oldcase,'twin')
    sm,sa,sc,sargs=v3.signed_restore(ctx,agent,norm,init+10000,v2case['signed_artifact'])
    key=jax.numpy.asarray(v3case['calibration_key'],dtype=jax.numpy.uint32)
    pipeline=v3.q1_bca.make_q1_pipeline(ctx.pipeline)
    qa,qf,_=pipeline.refresh(qargs,qc,qm,qa,cal,key,heldout_ids=row_ids)
    ta,tf,_=ctx.pipeline.refresh(targs,tc,tm,ta,cal,key,heldout_ids=row_ids)
    sr=v3.signed_bca.refresh(ctx.pipeline,sargs,sm,sa,cal,key,heldout_ids=episode_ids)
    require(bool(qf['posterior_certified']) and bool(tf['posterior_certified']) and
            bool(np.asarray(ctx.pipeline.reference_valid(qa.posterior))) and
            bool(np.asarray(ctx.pipeline.reference_valid(ta.posterior))), 'replayed reference invalid')
    values=(float(np.asarray(qa.posterior.threshold)),float(np.asarray(ta.posterior.threshold)),
            float(sr['q_bayes']),float(sr['q_split']))
    expected=(v3case['qthreshold'],v3case['tthreshold'],v3case['sr']['q_bayes'],v3case['sr']['q_split'])
    require(values==expected,'reconstructed v3 thresholds not exact')
    qwidth=v3.v2.width_function(jax,pipeline,qm,qa.posterior)
    twidth=v3.v2.width_function(jax,ctx.pipeline,tm,ta.posterior)
    signed=v3.v2.signed_function(jax,sm,sa,sr['q_bayes'])
    split=v3.v2.signed_function(jax,sm,sa,sr['q_split'])
    return qwidth,twidth,signed,split,values


def choices(score):
    x=np.asarray(score,np.float64)
    require(x.ndim==2 and x.shape[1]==15 and np.isfinite(x).all(),'bad common-bank scores')
    return np.argmax(x.max(axis=1,keepdims=True)-x<=1e-6,axis=1)


def summarize(values,scores,sources,guard):
    values=np.asarray(values,np.float64); sources=np.asarray(sources);guard=np.asarray(guard,bool)
    require(values.shape==(32,15) and sources.shape==(32,) and guard.shape==(32,),'summary shape differs')
    idx=np.arange(32); chosen={k:choices(x) for k,x in scores.items()}
    out={}
    for src in SOURCES_ORDER:
        group=sources==src;require(group.sum()==(16 if src=='random_prefix' else 8),'source size differs')
        groups={'all':group,'guard_on':group&guard,'guard_off':group&~guard}
        cells={}
        for sub,mask in groups.items():
            methods={}
            for name,x in scores.items():
                c=chosen[name]; val=values[idx,c];reg=values.max(axis=1)-val
                methods[name]={'mean_regret':float(reg[mask].mean()) if mask.any() else None,
                    'mean_score_error':float((np.asarray(x)[mask]-values[mask]).mean()) if mask.any() else None,
                    'mae':float(np.abs(np.asarray(x)[mask]-values[mask]).mean()) if mask.any() else None,
                    'selected_slot':c[mask].tolist()}
            compares={}
            for left,right,label in (('raw_q1','q1_aligned','raw_to_symmetric'),
                  ('q1_aligned','sym_compat_guarded','symmetric_to_guarded'),
                  ('sym_compat_guarded','sym_compat_unguarded','guarded_to_unguarded')):
                a,b=chosen[left],chosen[right];gain=values[idx,b]-values[idx,a];changed=(a!=b)&mask
                compares[label]={'mean_gain':float(gain[mask].mean()) if mask.any() else None,
                    'changed':int(changed.sum()), 'improved':int((changed&(gain>.1)).sum()),
                    'harmed':int((changed&(gain<-.1)).sum()),
                    'near_tied':int((changed&(abs(gain)<=.1)).sum()),
                    'per_state_gain':gain[mask].tolist()}
            cells[sub]={'states':int(mask.sum()),'methods':methods,'comparisons':compares}
        out[src]=cells
    return out


def screen(cases):
    out={}
    for context in old.CONTEXTS:
        groups={}; eligible={}; cells={}
        for arm in old.ARMS:
            qualified=available=0
            for seed in old.SEEDS:
                case=next(c for c in cases if c['seed']==seed)
                z=case['contexts'][context]['summaries'][arm][arm][arm+'_visited']['all']['comparisons']['symmetric_to_guarded']
                informative=z['improved']+z['harmed']; ok=informative>=4
                good=bool(ok and z['mean_gain']>=.05 and z['harmed']<=z['improved'])
                available+=ok;qualified+=good
                cells[f'{arm}/seed{seed}']={'eligible':ok,'qualifies':good,**z}
            groups[arm]=qualified;eligible[arm]=available
        passed=all(v>=2 for v in groups.values())
        status='passed' if passed else ('inconclusive' if any(v<2 for v in eligible.values()) else 'failed')
        out[context]={'status':status,'qualifying_seeds':groups,'eligible_seeds':eligible,'cells':cells}
    return out


def run(output):
    started=time.monotonic();deadline=started+SECONDS
    seal_sha,prior,archive=check_seal()
    terminal,report,_=old.completed(); proto_raw=(v3.RERUN/'protocol.json').read_bytes();protocol=json.loads(proto_raw)
    require(old.sha(proto_raw)==terminal['protocol_sha256'],'prior training protocol differs')
    v1prior=json.loads((v3.V1/'diagnostic_run_01/RESULT.json').read_bytes())
    _,v2prior=v3.input_pins()
    native=runtime.load_native();receipt=old.cpu_receipt(native);jax,jnp=native.jax,native.jnp
    core,reference,core_receipt=frozen.load_simulator()
    ledger=reference.new_ledger(CAP,360000)
    ledger.update(max_resets=129,max_critic_rows=5760,reconstruction_refreshes=0,
        actor_updates=0,critic_updates=0,actor_target_updates=0,critic_target_updates=0,auxiliary_updates=0,
        candidate_scoring_rows_attempted=0,candidate_scoring_rows_completed=0)
    def progress(event,**extra):
        old.max_time(started)
        with (output/'progress.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps({'event':event,'elapsed_seconds':time.monotonic()-started,'ledger':ledger,**extra},
                               sort_keys=True,allow_nan=False)+'\n');f.flush()
        print(event,flush=True)
    progress('initialized')
    random_pools={}
    for context in old.CONTEXTS:
        pool=fresh_starts.random_pool(core,reference,evaluation,context,ledger,deadline)
        random_pools[context]=pool
        if context=='ordinary':
            parity=reference.runtime_parity(core,pool['states'][0],ledger,deadline=deadline)
        progress('random_pool_created',context=context)
    cal,row_ids,episode_ids=archived_calibration(native,archive)
    prepared={};source_hashes={};compat_ref=None
    for seed in old.SEEDS:
        data=runtime.module('_common_diagnostic_data',v3.RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt);ctx=adapter.context
        rows=np.asarray(data.source_row_ids,np.int64); x=np.asarray(data.train.obs,np.float32);a=np.asarray(data.train.action,np.float32)
        require(len(rows)==67108 and len(np.unique(rows//100))==768,'training identity differs')
        row_digest=old.sha(rows.astype('<i8').tobytes())
        require(row_digest==prior['training_row_ids_sha256'],'v3 training rows differ')
        if compat_ref is None:
            probe=np.asarray(json.loads((v3.RERUN/'DATA_PLAN.json').read_bytes())['conditions']['hole']['gradient_probe_source_row_ids'],np.int64)
            lookup={int(v):i for i,v in enumerate(rows)};ix=np.asarray([lookup[int(v)] for v in probe],np.int64)
            compat_ref,_,cref=v3.revised_compat.fit_crossvalidated(x,a,rows//100,rows%100,ix,row_ids=rows)
            require(cref['reference_sha256']==prior['compatibility_fit']['reference_sha256'] and
                    cref['state_radius_p95']==prior['compatibility_fit']['state_radius_p95'] and
                    cref['action_bandwidth']==prior['compatibility_fit']['action_bandwidth'],
                    'compatibility reference differs from v3')
        for arm in old.ARMS:
            state,hashes,norm,branch=old.checkpoint(seed,arm,protocol,old.sha(proto_raw),adapter,template)
            source_hashes.update(hashes);before=runtime.tree_digest(state);agent=state.microstep.agent_state
            v1case=next(c for c in v1prior['cases'] if c['seed']==seed and c['arm']==arm)
            v2case=next(c for c in v2prior['cases'] if c['seed']==seed and c['arm']==arm)
            v3case=next(c for c in prior['cases'] if c['seed']==seed and c['arm']==arm)
            require(before==v3case['before'],'checkpoint differs from v3')
            policy=frozen.FrozenPolicy(runtime,branch,state)
            cache,_=old.load_cached(report,seed,arm,'ordinary');cachebank=old.cached_banks(report)['ordinary']
            acts=np.asarray(cache['q']['per_probe']['actions'],np.float32)
            old.parity(policy.act(cachebank['observations']),acts[:,0,:],'saved actor parity')
            old.parity(policy.q(np.repeat(cachebank['observations'],6,axis=0),acts.reshape(-1,4))[:,0].reshape(32,6),
                       cache['q']['per_probe']['q1'],'saved Q1 parity')
            width,twidth,signed,split,thresholds=reconstruct(ctx,agent,norm,v1case,v2case,v3case,cal,row_ids,episode_ids)
            ledger['reconstruction_refreshes']+=3
            require(runtime.tree_digest(state)==before,'restoration changed checkpoint')
            prepared[(seed,arm)]={'seed':seed,'arm':arm,'state':state,'before':before,'policy':policy,
                'qwidth':width,'twidth':twidth,'signed':signed,'split':split,'match':v3case['match'],
                'thresholds':thresholds,'hashes':hashes,'norm':norm}
            progress('checkpoint_reconstructed',seed=seed,arm=arm)
    require(len(prepared)==6 and ledger['reconstruction_refreshes']==18,'checkpoint reconstruction count differs')
    progress('all_scores_frozen')
    plans={}
    for seed in old.SEEDS:
        for context in old.CONTEXTS:
            first=random_pools[context]
            nv=fresh_starts.visited_pool(core,reference,evaluation,prepared[(seed,'native')]['policy'],context,seed,'native',ledger,deadline)
            bv=fresh_starts.visited_pool(core,reference,evaluation,prepared[(seed,'bca')]['policy'],context,seed,'bca',ledger,deadline)
            states=np.concatenate([first['states'],nv['states'],bv['states']]);times=np.concatenate([first['times'],nv['times'],bv['times']])
            obs=np.concatenate([first['observations'],nv['observations'],bv['observations']]);seeds=np.concatenate([first['seeds'],nv['seeds'],bv['seeds']])
            sources=np.asarray(['random_prefix']*16+['native_visited']*8+['bca_visited']*8)
            require(states.shape==(32,13) and np.array_equal(obs,reference.observation(states,times)), 'pooled states invalid')
            actor={arm:reference.actor_actions(prepared[(seed,arm)]['policy'],obs,ledger,deadline=deadline) for arm in old.ARMS}
            bank=common_bank.candidate_bank(actor['native'],actor['bca'])
            require(bank.shape==(32,15,4),'common action shape differs')
            repeated=np.repeat(obs,15,axis=0);flat=bank.reshape(-1,4)
            query=v3.revised_compat.query_neighbors(compat_ref,obs,times)
            neighbors=np.repeat(query['actions'],15,axis=0);guard=np.asarray(query['reliable'],bool)
            d=np.asarray(v3.revised_compat.compatibility_cost_jax(jnp.asarray(flat),jnp.asarray(neighbors),
                         jnp.asarray(compat_ref.base.action_bandwidth,jnp.float32)))
            scores={};components={}
            for arm in old.ARMS:
                p=prepared[(seed,arm)];ledger['candidate_scoring_rows_attempted']+=len(flat)
                q=evaluation.queried_q(p['policy'],repeated,flat,ledger,reference,deadline,None)[:,0]
                width,usable,valid=p['qwidth'](jnp.asarray(repeated),jnp.asarray(flat))
                tw,tu,tv=p['twidth'](jnp.asarray(repeated),jnp.asarray(flat))
                require(all(bool(np.asarray(x).all()) for x in (usable,valid,tu,tv)),'width invalid')
                bound,mu,sigma=p['signed'](jnp.asarray(repeated),jnp.asarray(flat))
                sp,_,_=p['split'](jnp.asarray(repeated),jnp.asarray(flat))
                computed=v3.score_controls.apply_scores(q,np.asarray(width),np.asarray(bound),d,np.repeat(guard,15),p['match'])
                computed['fresh_twin_min']=q-3*np.asarray(tw,np.float64)
                computed['signed_split']=q-np.asarray(sp,np.float64)
                scores[arm]={k:np.asarray(val,np.float64).reshape(32,15) for k,val in computed.items()}
                components[arm]={'width':np.asarray(width).reshape(32,15),'signed_bound':np.asarray(bound).reshape(32,15),
                    'mu':np.asarray(mu).reshape(32,15),'sigma':np.asarray(sigma).reshape(32,15)}
                ledger['candidate_scoring_rows_completed']+=len(flat)
            path=output/f'seed{seed}_{context}_frozen_plan.npz'
            with path.open('xb') as f:
                np.savez_compressed(f,states=states,times=times,observations=obs,reset_seeds=seeds,sources=sources,
                    actions=bank,guard=guard,cost=d.reshape(32,15),nearest_row_ids=np.asarray(query['row_ids'][:,0]),
                    **{f'{arm}__{name}':val for arm in old.ARMS for name,val in scores[arm].items()})
            plans[(seed,context)]={'states':states,'times':times,'observations':obs,'reset_seeds':seeds,
                'sources':sources,'actions':bank,'guard':guard,'cost':d.reshape(32,15),'scores':scores,
                'components':components,'plan_sha256':digest(path)}
            progress('common_bank_and_scores_frozen',seed=seed,context=context,sha256=digest(path))
    require(ledger['resets_attempted']==ledger['resets_completed']==129 and
            ledger.get('random_prefix_steps_completed')==1200 and ledger.get('actor_prefix_steps_completed')==3600 and
            ledger['candidate_scoring_rows_attempted']==ledger['candidate_scoring_rows_completed']==5760,
            'pool or scoring budget differs')
    progress('all_plans_frozen_before_qpi')
    cases=[]
    for seed in old.SEEDS:
        case={'seed':seed,'contexts':{},'checkpoints':{arm:{k:prepared[(seed,arm)][k] for k in ('before','hashes','norm','thresholds','match')} for arm in old.ARMS}}
        for context in old.CONTEXTS:
            p=plans[(seed,context)];truth={}
            for continuation in old.ARMS:
                truth[continuation]=v3.continuation15(core,reference,prepared[(seed,continuation)]['policy'],
                    p['states'],p['times'],p['actions'],ledger,deadline=deadline)
            summaries={critic:{cont:summarize(truth[cont],p['scores'][critic],p['sources'],p['guard'])
                               for cont in old.ARMS} for critic in old.ARMS}
            case['contexts'][context]={'summaries':summaries,'per_probe':{
                'states':p['states'].tolist(),'times':p['times'].tolist(),'observations':p['observations'].tolist(),
                'reset_seeds':p['reset_seeds'].tolist(),'sources':p['sources'].tolist(),'actions':p['actions'].tolist(),
                'guard':p['guard'].tolist(),'compatibility_cost':p['cost'].tolist(),
                'scores':{arm:{k:v.tolist() for k,v in p['scores'][arm].items()} for arm in old.ARMS},
                'components':{arm:{k:v.tolist() for k,v in p['components'][arm].items()} for arm in old.ARMS},
                'qpi':{arm:truth[arm].tolist() for arm in old.ARMS},'frozen_plan_sha256':p['plan_sha256']}}
            progress('context_cross_evaluated',seed=seed,context=context)
        for arm in old.ARMS:
            pp=prepared[(seed,arm)];require(runtime.tree_digest(pp['state'])==pp['before'],'checkpoint changed')
            pp['policy'].assert_unchanged()
        cases.append(case);write_json(output/f'seed{seed}_case.json',case)
    require(ledger['transition_steps_attempted']==ledger['transition_steps_completed']==CAP and
            ledger['continuation_rows_completed']==5760 and ledger['critic_rows_completed']==5760 and
            ledger['reconstruction_refreshes']==18 and all(ledger[k]==0 for k in
            ('actor_updates','critic_updates','actor_target_updates','critic_target_updates','auxiliary_updates')),
            'final bounded accounting differs')
    for path,h in source_hashes.items():require(digest(path)==h,'checkpoint source changed')
    require(check_seal()[0]==seal_sha,'source seal changed')
    result={'status':'completed_common_action_continuation_diagnostic','source_seal_sha256':seal_sha,
        'source_hashes':source_hashes,'runtime':receipt,'simulator':core_receipt,'parity':parity,
        'compatibility_fit':cref,'prior_calibration_archive_sha256':digest(archive),
        'state_sources':SOURCES_ORDER,'seed_audit':fresh_starts.seed_audit(),'ledger':ledger,'cases':cases,
        'screen':screen(cases),'elapsed_seconds':time.monotonic()-started,
        'limitations':['V3 calibration and model development already consumed; no new fitting',
          'Same-action cross-policy scores do not measure own-policy critic accuracy',
          'Guard is not a support certificate; forced first actions do not measure policy return',
          'Three training seeds and one dataset; states and actions are dependent']}
    write_json(output/'RESULT.json',result);progress('complete');return result


if __name__=='__main__':
    if sys.argv[1:]==['--seal']:
        print(seal());sys.exit(0)
    require(sys.argv[1:]==['--run'],'Use --seal then --run')
    require(os.environ.get('JAX_PLATFORMS')=='cpu','explicit CPU required')
    check_seal(); output=HERE/'diagnostic_run_01';output.mkdir(exist_ok=False)
    try:
        print(json.dumps({'status':run(output)['status']}),flush=True)
    except BaseException as exc:
        write_json(output/'FAILURE.json',{'status':'failed_no_automatic_retry','error':str(exc),'traceback':traceback.format_exc(),
            'counts_note':'See progress.jsonl; counts after the last saved event may be unknown'})
        raise
