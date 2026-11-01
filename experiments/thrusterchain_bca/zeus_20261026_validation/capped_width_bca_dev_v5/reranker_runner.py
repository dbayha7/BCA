"""Bounded v5 frozen-controller evaluation; no policy or auxiliary updates.

``--prepare`` computes training-probe caps; ``--seal`` binds those caps and
source inputs; ``--run`` evaluates one fixed, fresh-start attempt.  Nothing
executes on import.  See PROTOCOL.md for all scientific and budget contracts.
"""
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
V4 = BASE / 'common_action_bca_dev_v4'
spec = importlib.util.spec_from_file_location('_v5_v4', V4 / 'runner.py')
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
v3, old, runtime, frozen, evaluation = v4.v3, v4.old, v4.runtime, v4.frozen, v4.evaluation

METHODS = ('actor', 'raw_q1', 'symmetric', 'capped', 'guarded')
STARTS = {'ordinary': 2026130000, 'strong_current': 2026130100}
MAX_STEPS = 192104
MAX_SCORE_ROWS = 1152000
MAX_CAP_ROWS = 69120
SECONDS = 7200
FILES = ('PROTOCOL.md', 'reranker_runner.py')
LIVE_LEDGER = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def npz_write(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)


def seed_audit():
    """Replay v4 audit and check every new seed against all consumed streams."""
    previous = v4.fresh_starts.seed_audit()
    logged = v4.fresh_starts.module('_v5_logged', BASE/'bca_one_sided_v1/logged_data.py')
    src = logged.provenance()
    used = set(map(int, np.asarray(src['vectors']['episode_seeds']).ravel()))
    used.add(int(src['metadata']['seed']))
    protocol = json.loads((v3.RERUN/'protocol.json').read_bytes())
    used.add(int(protocol['evaluation']['parity_seed']))
    used.add(2026102803)  # fixed explicit-state parity reset
    for context in old.CONTEXTS:
        p = protocol['evaluation'][context]
        used.update(map(int, p['q_seeds'])); used.update(map(int, p['free_seeds']))
        used.add(int(p['prefix_seed']))
    for start in (2026102400, 2026102500, 2026102700):
        used.update(range(start, start+64))
    used.update((2026121401, 2026121402, 2026121500, 2026121501, 2026121510,
                 2026122200, 2026122201, 2026122300, 2026122301))
    used.update(range(2026122400, 2026122406))
    for previous_dir in (v3.V2, v3.HERE):
        root = previous_dir/'diagnostic_run_01'
        result = json.loads((root/'RESULT.json').read_bytes())
        behavior = root/result['fresh_data']['artifact']
        require(sha(behavior) == result['fresh_data']['artifact_sha256'], 'prior behavior changed')
        with np.load(behavior, allow_pickle=False) as z:
            used.update(map(int, z['episode_seeds']))
        for context in old.CONTEXTS:
            if previous_dir == v3.V2:
                path, desc, keys = root/(context+'_bank.npz'), result['banks'][context], ('q_seeds', 'free_seeds')
            else:
                path, desc, keys = root/(context+'_pool_and_selection.npz'), result['pools'][context], ('seeds',)
            require(sha(path) == desc['npz_sha256'], 'prior pool changed')
            with np.load(path, allow_pickle=False) as z:
                for key in keys:
                    used.update(map(int, z[key]))
    for context in old.CONTEXTS:
        used.update(map(int, v4.fresh_starts.random_seeds(context)))
        used.add(int(v4.fresh_starts.PREFIX_SEEDS[context]))
        for seed in old.SEEDS:
            for arm in old.ARMS:
                used.update(map(int, v4.fresh_starts.visited_seeds(context, seed, arm)))
    chosen = [s for start in STARTS.values() for s in range(start, start+16)]
    require(len(set(chosen)) == 32 and not used.intersection(chosen), 'fresh seed collision')
    return {'fresh_seeds': {k: list(range(s, s+16)) for k, s in STARTS.items()},
            'prior_seed_count': len(used), 'v4_audit': previous,
            'fixed_explicit_state_parity_reset_seed': 2026102803,
            'protected_transition_rows_read': 0}


def distinct_top_gap(actions, values):
    """Exact float32 full-vector deduplication; earliest slot wins any tie."""
    a, q = np.asarray(actions), np.asarray(values, np.float64)
    require(a.shape == (15, 4) and a.dtype == np.float32 and q.shape == (15,)
            and np.isfinite(a).all() and np.isfinite(q).all(), 'bad probe bank')
    unique = []
    for i in range(15):
        if not any(np.array_equal(a[i], a[j]) for j in unique):
            unique.append(i)
    require(len(unique) >= 2, 'fewer than two distinct probe actions')
    order = sorted(unique, key=lambda j: (-q[j], j))
    return float(q[order[0]] - q[order[1]])


def caps_from_probes(banks, q, row_ids):
    banks, q, ids = np.asarray(banks), np.asarray(q), np.asarray(row_ids, np.int64)
    require(banks.shape == (768, 15, 4) and q.shape == (768, 15)
            and ids.shape == (768,) and len(np.unique(ids//100)) == 768,
            'need one retained probe per episode')
    gaps = np.array([distinct_top_gap(a, s) for a, s in zip(banks, q)], np.float64)
    bins = (ids % 100)//25
    caps, counts = [], []
    for b in range(4):
        take = (bins == b) & np.isfinite(gaps) & (gaps > 0)
        count = int(len(np.unique(ids[take]//100)))
        require(count >= 50, f'time bin {b}: fewer than 50 positive distinct-episode gaps')
        caps.append(float(np.median(gaps[take]))); counts.append(count)
    return np.asarray(caps, np.float64), np.asarray(counts, np.int64), gaps


def score_banks(q, width, cost, guard, eta, cap):
    """Return the four predeclared score matrices, no hidden score scaling."""
    q, u, cost = map(lambda x: np.asarray(x, np.float64), (q, width, cost))
    guard, cap = np.asarray(guard, bool), np.asarray(cap, np.float64)
    n = len(q)
    require(q.shape == u.shape == cost.shape == (n,15) and guard.shape == cap.shape == (n,)
            and all(np.isfinite(x).all() for x in (q,u,cost,cap))
            and np.all(u >= 0) and np.all(cost >= -1e-6) and np.all(cap > 0)
            and np.isfinite(eta) and eta > 0, 'invalid scoring inputs')
    contrast = 3.0 * (u - u[:, :1])
    clipped = np.clip(contrast, -cap[:, None], cap[:, None])
    return {'raw_q1': q, 'symmetric': q-3*u, 'capped': q-clipped,
            'guarded': q-3*u-float(eta)*np.where(guard[:,None],cost,0.)}, contrast, clipped


def initialize(*, reconstruct):
    """Restore checkpoints and the exact v3 compatibility reference."""
    _, prior, archive = v4.input_pins()
    terminal, report, _ = old.completed()
    raw = (v3.RERUN/'protocol.json').read_bytes(); protocol = json.loads(raw)
    require(old.sha(raw) == terminal['protocol_sha256'], 'training protocol differs')
    native = runtime.load_native(); receipt = old.cpu_receipt(native)
    cal, row_ids, eps = v4.archived_calibration(native, archive)
    v1prior = json.loads((v3.V1/'diagnostic_run_01/RESULT.json').read_bytes())
    _, v2prior = v3.input_pins()
    items = {}; source_hashes = {}; compat = cross = cref = None
    for seed in old.SEEDS:
        data = runtime.module('_v5_data',v3.RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt);ctx=adapter.context
        rows=np.asarray(data.source_row_ids,np.int64); x=np.asarray(data.train.obs,np.float32)
        a=np.asarray(data.train.action,np.float32)
        require(len(rows)==67108 and len(np.unique(rows//100))==768
                and old.sha(rows.astype('<i8').tobytes())==prior['training_row_ids_sha256'], 'training identity differs')
        probe=np.asarray(json.loads((v3.RERUN/'DATA_PLAN.json').read_bytes())['conditions']['hole']['gradient_probe_source_row_ids'],np.int64)
        lookup={int(v):i for i,v in enumerate(rows)}
        ix=np.asarray([lookup[int(v)] for v in probe],np.int64)
        if compat is None:
            compat,cross,cref=v3.revised_compat.fit_crossvalidated(x,a,rows//100,rows%100,ix,row_ids=rows)
            require(cref['reference_sha256']==prior['compatibility_fit']['reference_sha256']
                    and cref['state_radius_p95']==prior['compatibility_fit']['state_radius_p95']
                    and cref['action_bandwidth']==prior['compatibility_fit']['action_bandwidth'],
                    'v3 compatibility reference differs')
        for arm in old.ARMS:
            state, hashes, norm, branch = old.checkpoint(seed,arm,protocol,old.sha(raw),adapter,template)
            source_hashes.update(hashes); before=runtime.tree_digest(state);agent=state.microstep.agent_state
            prior3=next(c for c in prior['cases'] if c['seed']==seed and c['arm']==arm)
            require(before==prior3['before'], 'checkpoint differs from v3')
            policy=frozen.FrozenPolicy(runtime,branch,state)
            cache,_=old.load_cached(report,seed,arm,'ordinary'); oldbank=old.cached_banks(report)['ordinary']
            acts=np.asarray(cache['q']['per_probe']['actions'],np.float32)
            old.parity(policy.act(oldbank['observations']),acts[:,0,:],'saved actor parity')
            old.parity(policy.q(np.repeat(oldbank['observations'],6,axis=0),acts.reshape(-1,4))[:,0].reshape(32,6),
                       cache['q']['per_probe']['q1'],'saved Q1 parity')
            qwidth=thresholds=None
            if reconstruct:
                p1=next(c for c in v1prior['cases'] if c['seed']==seed and c['arm']==arm)
                p2=next(c for c in v2prior['cases'] if c['seed']==seed and c['arm']==arm)
                qwidth,_,_,_,thresholds=v4.reconstruct(ctx,agent,norm,p1,p2,prior3,cal,row_ids,eps)
            require(runtime.tree_digest(state)==before,'checkpoint changed on restoration')
            items[(seed,arm)]={'state':state,'before':before,'policy':policy,'qwidth':qwidth,
                'match':prior3['match'],'thresholds':thresholds,'hashes':hashes}
    return {'native':native,'items':items,'compat':compat,'cross':cross,'compat_receipt':cref,
            'obs':x[ix], 'ids':probe, 'source_hashes':source_hashes,
            'runtime':receipt,'prior':prior,'calibration_archive_sha256':sha(archive)}


def prepare():
    """Only retained training data and frozen actor/Q inference; no simulator."""
    start=time.monotonic()
    require(not (HERE/'CAP_PROBES.json').exists() and not (HERE/'CAP_PROBES.npz').exists(),
            'cap preparation already exists; do not overwrite')
    seed_audit(); c=initialize(reconstruct=False)
    require(time.monotonic()-start < SECONDS, 'cap preparation deadline exceeded')
    arrays={'row_ids':c['ids']}; cases=[]
    for seed in old.SEEDS:
        for arm in old.ARMS:
            require(time.monotonic()-start < SECONDS, 'cap preparation deadline exceeded')
            p=c['items'][(seed,arm)]; policy=p['policy']; obs=c['obs']
            actor=np.asarray(policy.act(obs),np.float32)
            bank=v3.bank_v3.candidate_bank(actor,np.asarray(c['cross']['actions'][:,0],np.float32))
            q=np.asarray(policy.q(np.repeat(obs,15,axis=0),bank.reshape(-1,4)))[:,0].reshape(768,15)
            cap,count,gaps=caps_from_probes(bank,q,c['ids'])
            key=f'seed{seed}_{arm}'; arrays[key+'_q']=q; arrays[key+'_actions']=bank
            arrays[key+'_gaps']=gaps; arrays[key+'_caps']=cap
            cases.append({'seed':seed,'arm':arm,'caps':cap.tolist(),'positive_unique_episode_counts':count.tolist(),
                          'before':p['before'],'after':runtime.tree_digest(p['state'])})
            require(cases[-1]['before']==cases[-1]['after'], 'checkpoint changed during cap preparation')
            print('cap_prepared',seed,arm,flush=True)
    require(time.monotonic()-start < SECONDS, 'cap preparation deadline exceeded')
    npz_write(HERE/'CAP_PROBES.npz', **arrays)
    manifest={'schema':'bca.capped_width.cap_probes.v5','cases':cases,'probe_count':768,
              'training_candidate_rows':MAX_CAP_ROWS,'training_probe_episode_independent':False,
              'bank_neighbor':'v3 episode-fold-excluded crossfit nearest logged action',
              'caps_rule':'median finite positive distinct-vector raw-Q1 top-two gap by 25-step bin',
              'compatibility_reference_sha256':c['compat_receipt']['reference_sha256'],
              'source_hashes':c['source_hashes'],'npz_sha256':sha(HERE/'CAP_PROBES.npz'),
              'prepare_script_sha256':sha(HERE/'reranker_runner.py'),
              'simulation_transitions':0,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-start}
    json_write(HERE/'CAP_PROBES.json',manifest)
    return manifest


def input_pins():
    v4_seal, _, _ = v4.check_seal()
    previous, _, _ = v4.input_pins()
    cap=json.loads((HERE/'CAP_PROBES.json').read_bytes())
    require(cap['schema']=='bca.capped_width.cap_probes.v5' and cap['simulation_transitions']==0
            and cap['optimizer_updates']==0 and len(cap['cases'])==6
            and sha(HERE/'CAP_PROBES.npz')==cap['npz_sha256']
            and sha(HERE/'reranker_runner.py')==cap['prepare_script_sha256'], 'cap receipt differs')
    return {'prior':previous,'v4_source_seal_sha256':v4_seal,
            'v4_result_sha256':sha(V4/'diagnostic_run_01/RESULT.json'),
            'cap_json_sha256':sha(HERE/'CAP_PROBES.json'),
            'cap_npz_sha256':sha(HERE/'CAP_PROBES.npz')},cap


def seal():
    target=HERE/'RERANKER_SOURCE_SEAL.json';require(not target.exists(),'source seal exists')
    pins,_=input_pins()
    json_write(target,{'schema':'bca.capped_width.reranker.v5','inputs':pins,'seed_audit':seed_audit(),
                       'sources':{f:sha(HERE/f) for f in FILES}})
    return sha(target)


def check_seal():
    target=HERE/'RERANKER_SOURCE_SEAL.json';obj=json.loads(target.read_bytes());pins,cap=input_pins()
    require(obj['schema']=='bca.capped_width.reranker.v5' and obj['inputs']==pins
            and obj['seed_audit']==seed_audit() and set(obj['sources'])==set(FILES)
            and all(sha(HERE/k)==v for k,v in obj['sources'].items()),'seal or input changed')
    return sha(target),cap


def select_step(c, item, reference, states, t, caps, ledger, deadline):
    """Choose all five controllers on their own 16 physical states."""
    require(states.shape == (5,16,13) and states.dtype == np.float64 and 0 <= t < 100,
            'invalid controller states')
    obs=reference.observation(states.reshape(-1,13),np.full(80,t,np.int64))
    actor=reference.actor_actions(item['policy'],obs,ledger,deadline=deadline).reshape(5,16,4)
    robs=obs.reshape(5,16,14)[1:].reshape(64,14)
    times=np.full(64,t,np.int64)
    found=v3.revised_compat.query_neighbors(c['compat'],robs,times)
    bank=v3.bank_v3.candidate_bank(actor[1:].reshape(64,4),np.asarray(found['actions'][:,0],np.float32))
    repeated=np.repeat(robs,15,axis=0);flat=bank.reshape(-1,4)
    n=len(flat)
    require(ledger['candidate_scoring_rows_attempted']+n<=MAX_SCORE_ROWS,'score row cap exceeded')
    ledger['candidate_scoring_rows_attempted']+=n
    q=evaluation.queried_q(item['policy'],repeated,flat,ledger,reference,deadline,None)[:,0].reshape(64,15)
    jnp=c['native'].jnp
    u,usable,valid=item['qwidth'](jnp.asarray(repeated),jnp.asarray(flat))
    require(bool(np.asarray(usable).all()) and bool(np.asarray(valid).all()),'invalid calibrated width')
    u=np.asarray(u,np.float64).reshape(64,15)
    d=np.asarray(v3.revised_compat.compatibility_cost_jax(jnp.asarray(flat),
        jnp.asarray(np.repeat(found['actions'],15,axis=0)),
        jnp.asarray(c['compat'].base.action_bandwidth,jnp.float32)),np.float64).reshape(64,15)
    cap=np.full(64,float(caps[t//25]),np.float64)
    scores,contrast,clipped=score_banks(q,u,d,found['reliable'],item['match']['eta'],cap)
    indices=np.arange(64); names=np.repeat(np.asarray(METHODS[1:]),16)
    slots=np.empty(64,np.int64)
    for name in METHODS[1:]:
        where=names==name
        # NumPy returns first exact maximal slot, including preserved duplicates.
        slots[where]=np.argmax(scores[name][where],axis=1)
    selected=bank[indices,slots].reshape(4,16,4)
    actions=np.concatenate((actor[:1],selected),axis=0)
    require(actions.shape==(5,16,4) and actions.dtype==np.float32 and np.isfinite(actions).all()
            and np.all(np.abs(actions)<=1),'invalid selected actions')
    ledger['candidate_scoring_rows_completed']+=n
    info={'slots':slots.reshape(4,16),'guards':found['reliable'].reshape(4,16),
          'selected_q':q[indices,slots].reshape(4,16),'selected_width':u[indices,slots].reshape(4,16),
          'selected_contrast':contrast[indices,slots].reshape(4,16),
          'selected_clipped':clipped[indices,slots].reshape(4,16),
          'clip_saturated':(np.abs(contrast[indices,slots])>cap).reshape(4,16),
          'cap':cap.reshape(4,16),'selected_cost':d[indices,slots].reshape(4,16)}
    return actions,info


def fresh_states(core, reference, ledger, deadline):
    pools={}
    for context,base in STARTS.items():
        law=core.canonical.START_LAWS['recorded' if context=='ordinary' else 'strong_current']
        states=[]
        for seed in range(base,base+16):
            env=evaluation._reset(core,reference,seed,law,ledger,deadline,None)
            try: states.append(np.asarray(env.state,np.float64).copy())
            finally: env.close()
        pools[context]=np.asarray(states,np.float64)
        require(pools[context].shape==(16,13) and np.isfinite(pools[context]).all(), 'invalid fresh states')
    return pools


def controller_rollouts(core,reference,c,item,initial,initial_actions,initial_info,caps,ledger,deadline,progress):
    state=np.repeat(initial[None,:,:],5,axis=0).copy()
    returns=np.zeros((5,16),np.float64)
    actions=np.empty((5,16,100,4),np.float32); rewards=np.empty((5,16,100),np.float64)
    observations=np.empty((5,16,100,14),np.float32)
    info={k:np.empty((4,16,100),dtype=np.asarray(v).dtype) for k,v in initial_info.items()}
    for t in range(100):
        observations[:,:,t]=reference.observation(state,np.full((5,16),t,np.int64))
        if t == 0: action,stepinfo=initial_actions,initial_info
        else: action,stepinfo=select_step(c,item,reference,state,t,caps,ledger,deadline)
        actions[:,:,t]=action
        for k,v in stepinfo.items(): info[k][:,:,t]=v
        nxt,reward=reference._dynamics(core,state.reshape(80,13),action.reshape(80,4),
                                       ledger,'controller',deadline,None)
        state=nxt.reshape(5,16,13);reward=reward.reshape(5,16)
        rewards[:,:,t]=reward;returns+=(.98**t)*reward
        if t in (24,49,74,99): progress('controller_steps',t=t+1)
    require(np.isfinite(returns).all(), 'nonfinite controller return')
    return {'returns':returns,'actions':actions,'rewards':rewards,'observations':observations,**info}


def screen(cases):
    cells={};qual={'native':0,'bca':0};ordinary_ok=True
    for arm in old.ARMS:
        for seed in old.SEEDS:
            case=next(c for c in cases if c['seed']==seed and c['arm']==arm)
            strong=np.asarray(case['contexts']['strong_current']['returns'],np.float64)
            ordinary=np.asarray(case['contexts']['ordinary']['returns'],np.float64)
            require(strong.shape==ordinary.shape==(5,16),'missing screen values')
            raw=float(np.mean(strong[3]-strong[1])); sym=float(np.mean(strong[3]-strong[2]))
            vs_actor=float(np.mean(ordinary[3]-ordinary[0]))
            ok=bool(raw>=1 and sym>=1); guard=bool(vs_actor>=-1)
            qual[arm]+=ok;ordinary_ok &= guard
            cells[f'{arm}/seed{seed}']={'strong_capped_minus_raw':raw,'strong_capped_minus_symmetric':sym,
                'ordinary_capped_minus_actor':vs_actor,'strong_qualifies':ok,'ordinary_nonharm':guard}
    passed=all(n>=2 for n in qual.values()) and ordinary_ok
    return {'status':'passed' if passed else 'failed','qualifying_strong_seeds':qual,
            'ordinary_all_seeds_nonharm':ordinary_ok,'cells':cells,
            'scope':'engineering screen on three paired training seeds; no significance claim'}


def run(output):
    global LIVE_LEDGER
    start=time.monotonic();deadline=start+SECONDS
    seal_sha,cap_manifest=check_seal()
    core,reference,core_receipt=frozen.load_simulator()
    ledger=reference.new_ledger(MAX_STEPS,200000)
    LIVE_LEDGER=ledger
    ledger.update(max_resets=33,max_critic_rows=MAX_SCORE_ROWS,reconstruction_refreshes=0,
        candidate_scoring_rows_attempted=0,candidate_scoring_rows_completed=0,
        actor_updates=0,critic_updates=0,actor_target_updates=0,critic_target_updates=0,auxiliary_updates=0)
    def progress(event,**data):
        reference._check_time(deadline)
        with (output/'progress.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps({'event':event,'elapsed_seconds':time.monotonic()-start,
                                'ledger':ledger,**data},sort_keys=True,allow_nan=False)+'\n');f.flush()
        print(event,flush=True)
    progress('initialized')
    c=initialize(reconstruct=True);ledger['reconstruction_refreshes']=18
    for case in cap_manifest['cases']:
        p=c['items'][(case['seed'],case['arm'])]
        require(p['before']==case['before']==case['after'],'caps and checkpoint identity differ')
    require(c['source_hashes']==cap_manifest['source_hashes'], 'source identities differ from cap preparation')
    require(c['compat_receipt']['reference_sha256']==cap_manifest['compatibility_reference_sha256'],
            'compatibility differs from cap preparation')
    progress('frozen_models_reconstructed')
    pools=fresh_states(core,reference,ledger,deadline)
    parity=reference.runtime_parity(core,pools['ordinary'][0],ledger,deadline=deadline)
    require(parity['passed'], 'simulator parity failed')
    npz_write(output/'FRESH_STARTS.npz',**{k:v for k,v in pools.items()},
              **{k+'_seeds':np.arange(v,v+16,dtype=np.int64) for k,v in STARTS.items()})
    progress('fresh_starts_and_parity_frozen')
    caps={(z['seed'],z['arm']):np.asarray(z['caps'],np.float64) for z in cap_manifest['cases']}
    plans={};plan_arrays={}
    for seed in old.SEEDS:
        for arm in old.ARMS:
            item=c['items'][(seed,arm)]
            for context in old.CONTEXTS:
                initial=np.repeat(pools[context][None,:,:],5,axis=0)
                action,info=select_step(c,item,reference,initial,0,caps[(seed,arm)],ledger,deadline)
                plans[(seed,arm,context)]=(action,info)
                key=f'seed{seed}_{arm}_{context}'
                plan_arrays[key+'_actions']=action
                for name,value in info.items():plan_arrays[key+'_'+name]=value
    npz_write(output/'FIRST_ACTION_PLANS.npz',**plan_arrays)
    progress('all_first_actions_frozen_before_outcomes',plan_sha256=sha(output/'FIRST_ACTION_PLANS.npz'))
    cases=[]
    for seed in old.SEEDS:
        for arm in old.ARMS:
            item=c['items'][(seed,arm)]
            case={'seed':seed,'arm':arm,'before':item['before'],'thresholds':item['thresholds'],
                  'caps':caps[(seed,arm)].tolist(),'contexts':{}}
            for context in old.CONTEXTS:
                first,info=plans[(seed,arm,context)]
                def cell_progress(event,**kw): progress(event,seed=seed,arm=arm,context=context,**kw)
                roll=controller_rollouts(core,reference,c,item,pools[context],first,info,caps[(seed,arm)],
                                         ledger,deadline,cell_progress)
                # Each row is one start and five fixed, already-frozen first actions.
                qpi=v3.continuation_general(core,reference,item['policy'],pools[context],
                    np.zeros(16,np.int64),first.transpose(1,0,2),ledger,deadline=deadline)
                require(qpi.shape==(16,5),'first-action reference shape differs')
                path=output/f'seed{seed}_{arm}_{context}.npz'
                npz_write(path,**roll,first_qpi=qpi,first_actions=first)
                cell={'returns':roll['returns'].tolist(),'first_qpi':qpi.T.tolist(),
                      'clip_saturated_count_by_controller':roll['clip_saturated'].sum(axis=(1,2)).astype(int).tolist(),
                      'guard_count_by_controller':roll['guards'].sum(axis=(1,2)).astype(int).tolist(),
                      'telemetry_artifact':path.name,'telemetry_sha256':sha(path)}
                case['contexts'][context]=cell
                progress('cell_completed',seed=seed,arm=arm,context=context)
            item['policy'].assert_unchanged();case['after']=runtime.tree_digest(item['state'])
            require(case['after']==case['before'],'checkpoint changed during evaluation')
            cases.append(case);json_write(output/f'seed{seed}_{arm}_case.json',case)
    require(ledger['transition_steps_attempted']==ledger['transition_steps_completed']==MAX_STEPS
            and ledger['controller_steps_completed']==96000 and ledger['continuation_steps_completed']==96000
            and ledger['continuation_rows_completed']==960
            and ledger['candidate_scoring_rows_attempted']==ledger['candidate_scoring_rows_completed']==MAX_SCORE_ROWS
            and ledger['critic_rows_attempted']==ledger['critic_rows_completed']==MAX_SCORE_ROWS
            and ledger['resets_attempted']==ledger['resets_completed']==33
            and all(ledger[k]==0 for k in ('actor_updates','critic_updates','actor_target_updates',
                                           'critic_target_updates','auxiliary_updates')),'final bounded accounting differs')
    for path,h in c['source_hashes'].items():require(sha(path)==h,'checkpoint input changed')
    require(check_seal()[0]==seal_sha,'source seal changed')
    result={'schema':'bca.capped_width.reranker.result.v5','status':'completed_frozen_controller_evaluation',
        'source_seal_sha256':seal_sha,'source_hashes':c['source_hashes'],'runtime':c['runtime'],
        'simulator':core_receipt,'parity':parity,'compatibility_fit':c['compat_receipt'],
        'calibration_archive_sha256':c['calibration_archive_sha256'],'seed_audit':seed_audit(),
        'fresh_starts_sha256':sha(output/'FRESH_STARTS.npz'),
        'first_action_plans_sha256':sha(output/'FIRST_ACTION_PLANS.npz'),
        'methods':METHODS,'ledger':ledger,'cases':cases,'screen':screen(cases),
        'elapsed_seconds':time.monotonic()-start,
        'limitations':['online evaluation of fixed rerankers, no policy updates',
            'first-action Qpi follows saved actor, not reranker',
            'original v3 auxiliary and calibration consumed in development',
            'caps use in-sample retained training probes and are not calibrated bounds',
            'three paired training seeds, one dataset, shared starts']}
    json_write(output/'RERANKER_RESULT.json',result);progress('complete');return result


if __name__=='__main__':
    require(os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest',
            'explicit CPU and highest matmul precision required')
    if sys.argv[1:]==['--prepare']:
        print(json.dumps({'caps':len(prepare()['cases'])}),flush=True)
    elif sys.argv[1:]==['--seal']:
        print(seal(),flush=True)
    elif sys.argv[1:]==['--run']:
        check_seal();output=HERE/'reranker_run_01';output.mkdir(exist_ok=False)
        try: print(json.dumps({'status':run(output)['status']}),flush=True)
        except BaseException as exc:
            json_write(output/'FAILURE.json',{'status':'failed_no_automatic_retry','error':str(exc),
                'traceback':traceback.format_exc(),'ledger_at_exception':LIVE_LEDGER,
                'counts_note':'Progress is durable at logged boundaries; hard interruption can leave later counts unknown'})
            raise
    else: raise ValueError('Use --prepare, --seal, or --run')
