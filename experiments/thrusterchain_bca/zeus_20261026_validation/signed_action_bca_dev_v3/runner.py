"""One-shot expanded frozen-checkpoint screen; see PROTOCOL.md."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V2 = BASE / 'signed_action_bca_dev_v2'
V1 = BASE / 'action_sensitive_bca_dev_v1'
RERUN = BASE / 'central_quartile_training_v1_evaluation_rerun1'
for path in (RERUN, V1, V2, HERE):
    sys.path.insert(0, str(path))
import diagnostic_runner as old
import evaluation
import evaluate_worker as frozen
import runtime
import compatibility
import q1_bca
import revised_compat
import signed_bca
import bank_v3
import score_controls
import fresh_data
import importlib.util
_v2spec = importlib.util.spec_from_file_location('_v2_frozen_runner', V2/'runner.py')
v2 = importlib.util.module_from_spec(_v2spec)
_v2spec.loader.exec_module(v2)

CAP = 377704
SECONDS = 7200
SOURCES = ('PROTOCOL.md', 'runner.py', 'bank_v3.py', 'score_controls.py', 'fresh_data.py')
BLOCKS = {'ordinary': (2026122000, 2026122200, 'recorded'),
          'strong_current': (2026122100, 2026122201, 'strong_current')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def input_pins():
    _, v2seal = v2.check_seal()
    p = V2/'diagnostic_run_01/RESULT.json'
    result = json.loads(p.read_bytes())
    require(result['status'] == 'completed_frozen_checkpoint_diagnostic', 'V2 diagnostic incomplete')
    artifacts = {}
    for case in result['cases']:
        desc = case['signed_artifact']
        name = desc['path']
        require(Path(name).name == name and ':' not in name, 'unsafe signed artifact')
        target = V2/'diagnostic_run_01'/name
        require(target.is_file() and not target.is_symlink() and digest(target) == desc['sha256'],
                'signed model artifact hash mismatch')
        artifacts[name] = desc['sha256']
    require(len(artifacts) == 6, 'six signed artifacts required')
    return {'v2_source_seal_sha256': v2seal, 'v2_result_sha256': digest(p),
            'signed_artifacts': artifacts}, result


def seal():
    path = HERE/'SOURCE_SEAL.json'
    require(not path.exists(), 'source seal already exists')
    pins, _ = input_pins()
    fresh_data.preflight()
    write_json(path, {'schema':'signed_expanded_v3_source_seal', 'inputs':pins,
                      'sources':{x:digest(HERE/x) for x in SOURCES}})
    return digest(path)


def check_seal():
    path = HERE/'SOURCE_SEAL.json'
    obj = json.loads(path.read_bytes())
    require(obj['schema'] == 'signed_expanded_v3_source_seal' and
            set(obj['sources']) == set(SOURCES), 'invalid source seal')
    require(all(digest(HERE/k) == v for k,v in obj['sources'].items()), 'source changed after seal')
    pins, result = input_pins()
    require(pins == obj['inputs'], 'prior inputs changed after seal')
    return digest(path), result


def signed_restore(ctx, agent, norm, seed, artifact):
    from flax import serialization
    models, state, config, args = signed_bca.initialize(ctx, agent, seed,
        expected_mean=norm['saved_mean'], expected_std=norm['saved_std'])
    path = V2/'diagnostic_run_01'/artifact['path']
    require(digest(path) == artifact['sha256'], 'saved signed model differs')
    state = state._replace(calibrator=serialization.from_bytes(state.calibrator, path.read_bytes()))
    require(int(np.asarray(state.calibrator.step)) == 2000, 'signed optimizer count differs')
    return models, state, config, args


def continuation_general(core, reference, policy, states, times, actions, ledger, *,
                         deadline=None, progress=None):
    """Exact own-policy finite-horizon values for [N,K,4] with arbitrary K."""
    states, times, actions = np.asarray(states), np.asarray(times), np.asarray(actions)
    n, k = len(states), actions.shape[1]
    require(states.shape == (n,13) and states.dtype == np.float64 and np.isfinite(states).all()
            and times.shape == (n,) and times.dtype.kind in 'iu' and np.all((times >= 0)&(times < 100))
            and actions.shape == (n,k,4) and actions.dtype == np.float32 and k > 0
            and np.isfinite(actions).all() and np.all(np.abs(actions) <= 1), 'bad continuation inputs')
    remaining = np.repeat(100-times, k)
    total = int(remaining.sum())
    require(ledger['transition_steps_attempted'] + total <= ledger['max_transition_steps'],
            'continuation cannot fit budget')
    current = np.repeat(states, k, axis=0).copy()
    initial = actions.reshape(-1,4)
    values = np.zeros(n*k, np.float64)
    policy.assert_unchanged()
    for t in range(int(remaining.max())):
        active = np.flatnonzero(t < remaining)
        if t == 0:
            a = initial[active]
        else:
            obs = reference.observation(current[active], 100-remaining[active]+t)
            a = reference.actor_actions(policy,obs,ledger,deadline=deadline,progress=progress)
        nxt, reward = reference._dynamics(core,current[active],a,ledger,'continuation',deadline,progress)
        values[active] += (.98**t)*reward
        current[active] = nxt
    policy.assert_unchanged()
    require(np.isfinite(values).all(), 'invalid Qpi continuation')
    ledger['continuation_rows_completed'] = ledger.get('continuation_rows_completed',0)+n*k
    return values.reshape(n,k)


def continuation15(core, reference, policy, states, times, actions, ledger, *, deadline=None, progress=None):
    """Use authenticated six-action evaluator for first six; general path for nine."""
    require(actions.shape == (len(states),15,4), '15 candidate actions required')
    first = reference.continuation_values(core,policy,states,times,actions[:,:6,:],ledger,
                                         deadline=deadline,progress=progress)
    rest = continuation_general(core,reference,policy,states,times,actions[:,6:,:],ledger,
                                deadline=deadline,progress=progress)
    return np.concatenate((first,rest),axis=1)


def choices(scores):
    x = np.asarray(scores, np.float64)
    require(x.ndim == 2 and x.shape[1] == 15 and np.isfinite(x).all(), 'invalid score bank')
    return np.argmax(x.max(axis=1,keepdims=True)-x <= 1e-6,axis=1)


def summarize(actions, values, scores, obs, strata):
    actions, values = np.asarray(actions),np.asarray(values,np.float64)
    require(actions.shape == (32,15,4) and values.shape == (32,15), 'bad bank summary')
    high,_ = evaluation.hole_mask(obs,np.zeros((32,4),np.float32))
    raw = np.asarray(scores['raw_q1']); raw_c = choices(raw); idx=np.arange(32)
    true_gap = values[:,1:9]-values[:,:1]
    unique = np.any(actions[:,1:9] != actions[:,:1],axis=2)
    info = unique & (np.abs(true_gap) > .1)
    groups = {'natural':np.asarray(strata)=='natural','ambiguous_q1':np.asarray(strata)=='ambiguous_q1'}
    result = {}
    for group,g in groups.items():
        require(g.sum()==16, 'stratum size differs')
        per={}
        for name,score in scores.items():
            score=np.asarray(score,np.float64); choice=choices(score)
            reward=values[idx,choice]; gain=reward-values[idx,raw_c]; regret=values.max(axis=1)-reward
            pair_delta=score[:,1:9]-score[:,:1]; mask=info & g[:,None]
            score_tie=np.abs(pair_delta)<=1e-6
            correct=((pair_delta*true_gap)>0) & ~score_tie
            per[name]={'mean_regret':float(regret[g].mean()),'regret':regret[g].tolist(),
                'changed':int(((choice!=raw_c)&g).sum()), 'improved':int(((gain>.1)&g).sum()),
                'harmed':int(((gain<-.1)&g).sum()), 'changed_near_tie':int(((abs(gain)<=.1)&(choice!=raw_c)&g).sum()),
                'selected_slot':choice[g].tolist(),
                'ranking_informative_pairs':int(mask.sum()),'ranking_correct':int((correct&mask).sum()),
                'ranking_tied_score':int((score_tie&mask).sum()),
                'ranking_accuracy':float((correct&mask).sum()/mask.sum()) if mask.any() else None,
                'all_action_error':evaluation.error_metrics(score[g],values[g]),
                'high_load_fixed_band_error':evaluation.error_metrics(score[g&high,11:14],values[g&high,11:14]),
                'high_load_mean_regret':float(regret[g&high].mean()) if (g&high).any() else None}
        raw_regret=values.max(axis=1)-values[idx,raw_c]
        result[group]={'states':int(g.sum()), 'high_load_states':int((g&high).sum()),
            'informative_families':int((info.any(axis=1)&g).sum()),
            'raw_mistake_families':int(((raw_regret>.1)&g).sum()), 'scores':per}
    return result, high


def screen(cases):
    output={}
    for context in old.CONTEXTS:
        output[context]={}
        for stratum in ('natural','ambiguous_q1'):
            groups={}; eligible={}; cells={}
            for arm in old.ARMS:
                count=avail=0
                for seed in old.SEEDS:
                    c=next(x for x in cases if x['seed']==seed and x['arm']==arm)
                    z=c['contexts'][context]['summary'][stratum]; s=z['scores']; p=s['sym_compat_guarded']
                    ok=z['informative_families']>=8 and z['raw_mistake_families']>=1
                    raw_gain=s['raw_q1']['mean_regret']-p['mean_regret']
                    matched_gain=s['sym_hybrid_strength_matched']['mean_regret']-p['mean_regret']
                    sym_gain=s['q1_aligned']['mean_regret']-p['mean_regret']
                    good=bool(ok and raw_gain>=.05 and matched_gain>=.05 and sym_gain>=0
                              and p['harmed']<=p['improved'])
                    count+=good; avail+=ok
                    cells[f'{arm}/seed{seed}']={'eligible':ok,'qualifies':good,
                        'informative_families':z['informative_families'],'raw_mistake_families':z['raw_mistake_families'],
                        'regret_improvement_vs_raw':raw_gain,'regret_improvement_vs_matched':matched_gain,
                        'regret_improvement_vs_symmetric':sym_gain,'improved':p['improved'],'harmed':p['harmed']}
                groups[arm]=count; eligible[arm]=avail
            passed=all(x>=2 for x in groups.values())
            status='passed' if passed else ('inconclusive' if any(x<2 for x in eligible.values()) else 'failed')
            output[context][stratum]={'status':status,'qualifying_seeds':groups,'eligible_seeds':eligible,'cells':cells}
    return output


def run(output):
    started=time.monotonic(); deadline=started+SECONDS
    seal_sha,previous=check_seal()
    terminal,report,_=old.completed();proto_raw=(RERUN/'protocol.json').read_bytes();protocol=json.loads(proto_raw)
    require(old.sha(proto_raw)==terminal['protocol_sha256'],'training protocol differs')
    prior_v1=json.loads((V1/'diagnostic_run_01/RESULT.json').read_bytes())
    native=runtime.load_native();receipt=old.cpu_receipt(native);jax,jnp=native.jax,native.jnp
    core,reference,core_receipt=frozen.load_simulator()
    ledger=reference.new_ledger(CAP,360000)
    ledger.update(max_resets=257,max_critic_rows=11520,auxiliary_updates=0,calibration_refreshes=0,
                  actor_updates=0,critic_updates=0,actor_target_updates=0,critic_target_updates=0,
                  matching_probe_rows_attempted=0,matching_probe_rows_completed=0,
                  candidate_scoring_rows_attempted=0,candidate_scoring_rows_completed=0)
    def progress(event,**extra):
        old.max_time(started)
        with (output/'progress.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps({'event':event,'elapsed_seconds':time.monotonic()-started,'ledger':ledger,**extra},
                               sort_keys=True,allow_nan=False)+'\n'); f.flush()
        print(event,flush=True)
    progress('initialized');fresh_data.preflight();pools={}
    for name,(seed,prefix,law) in BLOCKS.items():
        pool=bank_v3.build_pool(core,reference,evaluation,context=name,
            seeds=np.arange(seed,seed+64,dtype=np.int64),prefix_seed=prefix,
            start_law=core.canonical.START_LAWS[law],ledger=ledger,deadline=deadline)
        pools[name]=pool
        if name=='ordinary': parity=reference.runtime_parity(core,pool['states'][0],ledger,deadline=deadline)
    progress('fresh_pools_created')
    cal,check,fresh_receipt=fresh_data.collect(native,core,reference,ledger,deadline,output)
    progress('fresh_calibration_collected')
    prepared=[];source_hashes={}; compat_ref=cross=cref=None
    for seed in old.SEEDS:
        data=runtime.module('_expanded_diagnostic_data',RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt);ctx=adapter.context
        x=np.asarray(data.train.obs,np.float32);a=np.asarray(data.train.action,np.float32)
        rows=np.asarray(data.source_row_ids,np.int64); episodes=rows//100;times=rows%100
        require(len(rows)==67108 and len(np.unique(episodes))==768,'training identity differs')
        probe=np.asarray(json.loads((RERUN/'DATA_PLAN.json').read_bytes())['conditions']['hole']['gradient_probe_source_row_ids'],np.int64)
        lookup={int(v):i for i,v in enumerate(rows)};ix=np.array([lookup[int(v)] for v in probe],np.int64)
        if compat_ref is None:
            compat_ref,cross,cref=revised_compat.fit_crossvalidated(x,a,episodes,times,ix,row_ids=rows)
            row_digest=old.sha(rows.astype('<i8').tobytes())
        else: require(old.sha(rows.astype('<i8').tobytes())==row_digest,'training data differs')
        for arm in old.ARMS:
            progress('checkpoint_preparing',seed=seed,arm=arm)
            state,hashes,norm,branch=old.checkpoint(seed,arm,protocol,old.sha(proto_raw),adapter,template)
            source_hashes.update(hashes);before=runtime.tree_digest(state);agent=state.microstep.agent_state
            policy=frozen.FrozenPolicy(runtime,branch,state)
            oldcase=next(c for c in prior_v1['cases'] if c['seed']==seed and c['arm']==arm)
            v2case=next(c for c in previous['cases'] if c['seed']==seed and c['arm']==arm)
            cache,_=old.load_cached(report,seed,arm,'ordinary');oldbank=old.cached_banks(report)['ordinary']
            cacheacts=np.asarray(cache['q']['per_probe']['actions'],np.float32)
            old.parity(policy.act(oldbank['observations']),cacheacts[:,0,:],'saved actor parity')
            old.parity(policy.q(np.repeat(oldbank['observations'],6,axis=0),cacheacts.reshape(-1,4))[:,0].reshape(32,6),
                       cache['q']['per_probe']['q1'],'saved Q1 parity')
            init_seed=oldcase['auxiliary_init_seed']
            qm,qa,qc,qargs=v2.restore_aux(ctx,agent,norm,init_seed,oldcase,'q1')
            tm,ta,tc,targs=v2.restore_aux(ctx,agent,norm,init_seed,oldcase,'twin')
            sm,sa,sc,sargs=signed_restore(ctx,agent,norm,init_seed+10000,v2case['signed_artifact'])
            key=jax.random.PRNGKey(2026122400+seed*2+(arm=='bca'))
            pipeline=q1_bca.make_q1_pipeline(ctx.pipeline); ids=np.asarray(fresh_receipt['calibration_row_ids'],np.int64)
            qa,qf,_=pipeline.refresh(qargs,qc,qm,qa,cal,key,heldout_ids=ids)
            ta,tf,_=ctx.pipeline.refresh(targs,tc,tm,ta,cal,key,heldout_ids=ids)
            sr=signed_bca.refresh(ctx.pipeline,sargs,sm,sa,cal,key,
                heldout_ids=np.asarray(fresh_receipt['episode_seeds'][:64],np.int64))
            ledger['calibration_refreshes']+=3
            require(bool(qf['posterior_certified']) and bool(tf['posterior_certified']) and
                    bool(np.asarray(ctx.pipeline.reference_valid(qa.posterior))) and
                    bool(np.asarray(ctx.pipeline.reference_valid(ta.posterior))), 'calibration rejected')
            qwidth=v2.width_function(jax,pipeline,qm,qa.posterior);twidth=v2.width_function(jax,ctx.pipeline,tm,ta.posterior)
            signed=v2.signed_function(jax,sm,sa,sr['q_bayes']);split=v2.signed_function(jax,sm,sa,sr['q_split'])
            ledger['matching_probe_rows_attempted']+=len(ix)
            obs=jnp.asarray(x[ix]);acts=policy._act(obs)
            match=score_controls.match_controls(obs,acts,lambda o,z:policy._q(o,z)[:,0],qwidth,signed,
                revised_compat.compatibility_cost_jax,cross['actions'],cross['bandwidths'],cross['reliable'])
            ledger['matching_probe_rows_completed']+=len(ix)
            check_noise_key=jax.random.fold_in(key,1357911)
            cq,cy=signed_bca.q1_and_target(ctx.pipeline,sargs,sm,sa,check,check_noise_key)
            _,mu,sigma=signed(check.obs,check.action);err=np.asarray(cq)-np.asarray(cy);mu=np.asarray(mu);sigma=np.asarray(sigma)
            heads=tm[1].apply(ta.native.critic.params,check.obs,check.action)
            require(heads.shape==(64,2) and np.asarray(cq).shape==np.asarray(cy).shape==mu.shape==sigma.shape==(64,)
                    and all(np.isfinite(np.asarray(v)).all() for v in (heads,cq,cy,mu,sigma))
                    and np.all(sigma>0) and np.array_equal(np.asarray(heads)[:,0],np.asarray(cq)),
                    'held-out critic shape, critic identity or finite values differ')
            twin_center=np.asarray(heads).min(axis=-1)
            qw,qu,qv=qwidth(check.obs,check.action);tw,tu,tv=twidth(check.obs,check.action)
            require(all(bool(np.asarray(v).all()) for v in (qu,qv,tu,tv)), 'held-out width invalid')
            qw,tw=np.asarray(qw),np.asarray(tw)
            require(qw.shape==tw.shape==(64,) and np.isfinite(qw).all() and np.isfinite(tw).all(),
                    'held-out width shape or finite values invalid')
            checked={'signed_mae':float(np.mean(abs(err-mu))),'zero_mae':float(np.mean(abs(err))),
                'upper_td_coverage':float(np.mean(err<=mu+sr['q_bayes']*sigma)),
                'q1_symmetric_abs_td_coverage':float(np.mean(abs(err)<=qw)),
                'twin_own_symmetric_abs_td_coverage':float(np.mean(abs(np.asarray(cy)-twin_center)<=tw)),
                'twin_symmetric_q1_abs_td_coverage':float(np.mean(abs(err)<=tw)),
                'noise_key':np.asarray(check_noise_key).tolist(),
                'residual':err.tolist(),'mu':mu.tolist(),'sigma':sigma.tolist(),
                'q1_width':qw.tolist(),'twin_width':tw.tolist(),'twin_center':twin_center.tolist()}
            require(runtime.tree_digest(state)==before,'frozen learner changed')
            prepared.append({'seed':seed,'arm':arm,'state':state,'before':before,'policy':policy,'ctx':ctx,
                'qwidth':qwidth,'twidth':twidth,'signed':signed,'split':split,'match':match,'sr':sr,'check':checked,
                'norm':norm,'hashes':hashes,'qthreshold':float(np.asarray(qa.posterior.threshold)),
                'tthreshold':float(np.asarray(ta.posterior.threshold)), 'calibration_key':np.asarray(key).tolist()})
            progress('checkpoint_prepared',seed=seed,arm=arm)
    require(len(prepared)==6 and ledger['calibration_refreshes']==18,'preparation count differs')
    progress('all_models_and_controls_frozen')
    selections={};all_banks={};all_q={}
    for context in old.CONTEXTS:
        pool=pools[context];obs=pool['observations'];query=revised_compat.query_neighbors(compat_ref,obs,pool['times'])
        nearest=np.asarray(query['actions'][:,0,:],np.float32)
        banks=[];qs=[]
        for p in prepared:
            acts=reference.actor_actions(p['policy'],obs,ledger,deadline=deadline)
            bank=bank_v3.candidate_bank(acts,nearest)
            q=evaluation.queried_q(p['policy'],np.repeat(obs,15,axis=0),bank.reshape(-1,4),
                                   ledger,reference,deadline,None)[:,0].reshape(64,15)
            banks.append(bank);qs.append(q)
        banks=np.asarray(banks,np.float32);qs=np.asarray(qs,np.float64)
        selection=bank_v3.select_indices(pool['times'],qs,banks)
        ix=selection['indices']; selections[context]=selection;all_banks[context]=banks;all_q[context]=qs
        filename=output/(context+'_pool_and_selection.npz')
        with filename.open('xb') as f:
            np.savez_compressed(f,states=pool['states'],observations=obs,times=pool['times'],seeds=pool['seeds'],
                banks=banks,q1=qs,selected_indices=ix,strata=np.asarray(selection['strata']),
                nearest_row_ids=np.asarray(query['row_ids'][:,0]),
                nearest_distances=np.asarray(query['distances'][:,0]),
                median_normalized_gap=selection['median_normalized_gap'],
                per_checkpoint_normalized_gap=selection['per_checkpoint_normalized_gap'],
                duplicate_action_slots=selection['duplicate_action_slots'])
        progress('pool_selection_frozen',context=context,sha256=digest(filename))
    progress('all_selections_frozen')
    cases=[]
    for i,p in enumerate(prepared):
        case={k:p[k] for k in ('seed','arm','before','match','sr','check','norm','hashes','qthreshold','tthreshold','calibration_key')}
        case['contexts']={}
        for context in old.CONTEXTS:
            pool=pools[context];sel=selections[context];ix=sel['indices'];obs=pool['observations'][ix]
            actions=all_banks[context][i,ix]; raw=all_q[context][i,ix]
            repeat=np.repeat(obs,15,axis=0);flat=actions.reshape(-1,4)
            ledger['candidate_scoring_rows_attempted']+=len(flat)
            query=revised_compat.query_neighbors(compat_ref,obs,pool['times'][ix])
            neighbors=np.repeat(query['actions'],15,axis=0);guard=np.repeat(query['reliable'],15)
            d=np.asarray(revised_compat.compatibility_cost_jax(jnp.asarray(flat),jnp.asarray(neighbors),
                         jnp.asarray(compat_ref.base.action_bandwidth,jnp.float32)))
            w,usable,valid=p['qwidth'](jnp.asarray(repeat),jnp.asarray(flat))
            tw,tu,tv=p['twidth'](jnp.asarray(repeat),jnp.asarray(flat))
            require(all(bool(np.asarray(z).all()) for z in (usable,valid,tu,tv)),'width invalid')
            b,mu,sigma=p['signed'](jnp.asarray(repeat),jnp.asarray(flat));spl,_,_=p['split'](jnp.asarray(repeat),jnp.asarray(flat))
            scores=score_controls.apply_scores(raw.ravel(),np.asarray(w),np.asarray(b),d,np.asarray(guard,bool),p['match'])
            scores['fresh_twin_min']=raw.ravel()-3*np.asarray(tw,np.float64)
            scores['signed_split']=raw.ravel()-np.asarray(spl,np.float64)
            scores={k:np.asarray(v).reshape(32,15) for k,v in scores.items()}
            ledger['candidate_scoring_rows_completed']+=len(flat)
            # No reference outcomes have been accessed until this point.
            values=continuation15(core,reference,p['policy'],pool['states'][ix],pool['times'][ix],actions,ledger,deadline=deadline)
            summary,high=summarize(actions,values,scores,obs,sel['strata'])
            case['contexts'][context]={'summary':summary,'per_probe':{
                'pool_indices':ix.tolist(),'times':pool['times'][ix].tolist(),'strata':sel['strata'],
                'observations':obs.tolist(),'actions':actions.tolist(),'high_load':high.tolist(),
                'q1':raw.tolist(),'qpi':values.tolist(),'scores':{k:v.tolist() for k,v in scores.items()},
                'components':{'reliable':np.asarray(guard).reshape(32,15).tolist(),
                    'cost':d.reshape(32,15).tolist(),'width':np.asarray(w).reshape(32,15).tolist(),
                    'signed_bound':np.asarray(b).reshape(32,15).tolist(),
                    'mu':np.asarray(mu).reshape(32,15).tolist(),'sigma':np.asarray(sigma).reshape(32,15).tolist()}}}
            progress('context_evaluated',seed=p['seed'],arm=p['arm'],context=context)
        require(runtime.tree_digest(p['state'])==p['before'],'checkpoint changed')
        p['policy'].assert_unchanged();cases.append(case);write_json(output/f"seed{p['seed']}_{p['arm']}_case.json",case)
    require(ledger['transition_steps_attempted']==ledger['transition_steps_completed']==CAP and
            ledger['continuation_rows_completed']==5760 and ledger['resets_completed']==257 and
            ledger['matching_probe_rows_attempted']==ledger['matching_probe_rows_completed']==4608 and
            ledger['candidate_scoring_rows_attempted']==ledger['candidate_scoring_rows_completed']==5760,
            'simulation counts differ from bounded plan')
    require(ledger['calibration_refreshes']==18 and all(ledger[k]==0 for k in
        ('auxiliary_updates','actor_updates','critic_updates','actor_target_updates','critic_target_updates')),
        'optimizer or calibration ledger differs')
    for path,h in source_hashes.items():require(digest(path)==h,'checkpoint source changed')
    require(check_seal()[0]==seal_sha,'source seal changed')
    result={'status':'completed_expanded_frozen_diagnostic','source_seal_sha256':seal_sha,
        'source_hashes':source_hashes,'runtime':receipt,'simulator':core_receipt,'parity':parity,
        'fresh_data':fresh_receipt,'compatibility_fit':cref,'training_row_ids_sha256':row_digest,
        'pools':{k:{'pool_id':v['pool_id'],'npz_sha256':digest(output/(k+'_pool_and_selection.npz'))} for k,v in pools.items()},
        'ledger':ledger,'cases':cases,'screen':screen(cases),'elapsed_seconds':time.monotonic()-started,
        'limitations':['Exploratory follow-up after inspected prior results',
            'Scaled signed scores do not retain a calibration coverage claim',
            'Action banks and frozen own-policy references do not establish policy improvement',
            'Q1-ambiguity-selected stream is conditional and not a natural-start estimate',
            'State-distance guard is not a support certificate']}
    write_json(output/'RESULT.json',result);progress('complete');return result


if __name__=='__main__':
    if sys.argv[1:]==['--seal']:
        print(seal());sys.exit(0)
    require(sys.argv[1:]==['--run'],'Use --seal then --run')
    require(os.environ.get('JAX_PLATFORMS')=='cpu','explicit CPU required')
    check_seal();output=HERE/'diagnostic_run_01';output.mkdir(exist_ok=False)
    try:
        print(json.dumps({'status':run(output)['status']}),flush=True)
    except BaseException as exc:
        write_json(output/'FAILURE.json',{'status':'failed_no_automatic_retry','error':str(exc),'traceback':traceback.format_exc(),
            'counts_note':'See progress.jsonl; counts after the last saved event may be unknown'})
        raise
