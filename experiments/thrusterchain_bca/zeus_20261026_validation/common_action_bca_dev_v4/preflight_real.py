"""Read-only replay of six checkpoints on logged states; no simulation or updates."""
import json
import numpy as np
import runner


HASHED=[runner.HERE/x for x in runner.SOURCES]
HASHED += [runner.V3/'runner.py',runner.v3.V1/'q1_bca.py',runner.v3.V2/'signed_bca.py',
           runner.v3.V2/'revised_compat.py',runner.V3/'score_controls.py']


def main():
    before_sources={str(p):runner.digest(p) for p in HASHED}
    pins,prior,archive=runner.input_pins();old=runner.old;runtime=runner.runtime
    native=runtime.load_native();receipt=old.cpu_receipt(native);jax,jnp=native.jax,native.jnp
    terminal,report,_=old.completed(); raw=(runner.v3.RERUN/'protocol.json').read_bytes();protocol=json.loads(raw)
    assert old.sha(raw)==terminal['protocol_sha256']
    v1prior=json.loads((runner.v3.V1/'diagnostic_run_01/RESULT.json').read_bytes());_,v2prior=runner.v3.input_pins()
    cal,ids,episodes=runner.archived_calibration(native,archive)
    compat=None;cases=[]
    for seed in old.SEEDS:
        data=runtime.module('_common_preflight_data',runner.v3.RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt);ctx=adapter.context
        x=np.asarray(data.train.obs,np.float32);a=np.asarray(data.train.action,np.float32);rows=np.asarray(data.source_row_ids,np.int64)
        assert old.sha(rows.astype('<i8').tobytes())==prior['training_row_ids_sha256']
        if compat is None:
            probe=np.asarray(json.loads((runner.v3.RERUN/'DATA_PLAN.json').read_bytes())['conditions']['hole']['gradient_probe_source_row_ids'],np.int64)
            lookup={int(v):i for i,v in enumerate(rows)};ix=np.asarray([lookup[int(v)] for v in probe],np.int64)
            compat,_,cref=runner.v3.revised_compat.fit_crossvalidated(x,a,rows//100,rows%100,ix,row_ids=rows)
            assert cref['reference_sha256']==prior['compatibility_fit']['reference_sha256']
        policies={}; models={}
        for arm in old.ARMS:
            state,_,norm,branch=old.checkpoint(seed,arm,protocol,old.sha(raw),adapter,template)
            before=runtime.tree_digest(state);agent=state.microstep.agent_state
            v1case=next(c for c in v1prior['cases'] if c['seed']==seed and c['arm']==arm)
            v2case=next(c for c in v2prior['cases'] if c['seed']==seed and c['arm']==arm)
            v3case=next(c for c in prior['cases'] if c['seed']==seed and c['arm']==arm)
            assert before==v3case['before']
            qwidth,twidth,signed,split,thresholds=runner.reconstruct(ctx,agent,norm,v1case,v2case,v3case,cal,ids,episodes)
            policies[arm]=runner.frozen.FrozenPolicy(runtime,branch,state)
            cache,_=old.load_cached(report,seed,arm,'ordinary');cachebank=old.cached_banks(report)['ordinary']
            old_actions=np.asarray(cache['q']['per_probe']['actions'],np.float32)
            old.parity(policies[arm].act(cachebank['observations']),old_actions[:,0,:],'saved actor parity')
            old.parity(policies[arm].q(np.repeat(cachebank['observations'],6,axis=0),old_actions.reshape(-1,4))[:,0].reshape(32,6),
                       cache['q']['per_probe']['q1'],'saved Q1 parity')
            saved=v3case['contexts']['ordinary']['per_probe'];so=np.asarray(saved['observations'],np.float32)
            sa=np.asarray(saved['actions'],np.float32).reshape(-1,4);repeat=np.repeat(so,15,axis=0)
            sq=np.asarray(policies[arm].q(repeat,sa))[:,0]
            sw,_,_=qwidth(jnp.asarray(repeat),jnp.asarray(sa));sb,_,_=signed(jnp.asarray(repeat),jnp.asarray(sa))
            nq=runner.v3.revised_compat.query_neighbors(compat,so,np.asarray(saved['times'],np.int64))
            sd=np.asarray(runner.v3.revised_compat.compatibility_cost_jax(jnp.asarray(sa),
                jnp.asarray(np.repeat(nq['actions'],15,axis=0)),jnp.asarray(compat.base.action_bandwidth,jnp.float32)))
            old.parity(sq.reshape(32,15),saved['q1'],'v3 saved q1 parity')
            old.parity(np.asarray(sw).reshape(32,15),saved['components']['width'],'v3 saved width parity')
            old.parity(np.asarray(sb).reshape(32,15),saved['components']['signed_bound'],'v3 saved signed parity')
            assert np.array_equal(np.repeat(nq['reliable'],15).reshape(32,15),np.asarray(saved['components']['reliable']))
            old.parity(sd.reshape(32,15),saved['components']['cost'],'v3 saved compatibility parity')
            replay=runner.v3.score_controls.apply_scores(sq,np.asarray(sw),np.asarray(sb),sd,
                np.repeat(nq['reliable'],15),v3case['match'])
            for name,value in replay.items():
                old.parity(value.reshape(32,15),saved['scores'][name],'v3 saved score parity: '+name)
            models[arm]=(state,before,qwidth,twidth,signed,split,v3case['match'],thresholds)
        obs=x[:32];t=rows[:32]%100
        acts={arm:np.asarray(policies[arm].act(obs),np.float32) for arm in old.ARMS}
        bank=runner.common_bank.candidate_bank(acts['native'],acts['bca']);flat=bank.reshape(-1,4);repeated=np.repeat(obs,15,axis=0)
        query=runner.v3.revised_compat.query_neighbors(compat,obs,t);neighbors=np.repeat(query['actions'],15,axis=0)
        gate=np.repeat(query['reliable'],15)
        cost=np.asarray(runner.v3.revised_compat.compatibility_cost_jax(jnp.asarray(flat),jnp.asarray(neighbors),jnp.asarray(compat.base.action_bandwidth,jnp.float32)))
        for arm in old.ARMS:
            state,before,qwidth,twidth,signed,split,match,thresholds=models[arm]
            q=np.asarray(policies[arm].q(repeated,flat))[:,0]
            width,usable,valid=qwidth(jnp.asarray(repeated),jnp.asarray(flat));tw,tu,tv=twidth(jnp.asarray(repeated),jnp.asarray(flat))
            b,mu,sigma=signed(jnp.asarray(repeated),jnp.asarray(flat));sp,_,_=split(jnp.asarray(repeated),jnp.asarray(flat))
            assert all(bool(np.asarray(z).all()) for z in (usable,valid,tu,tv))
            scores=runner.v3.score_controls.apply_scores(q,np.asarray(width),np.asarray(b),cost,gate,match)
            assert all(np.isfinite(z).all() for z in (q,width,tw,b,mu,sigma,sp,*scores.values()))
            assert runtime.tree_digest(state)==before
            cases.append({'seed':seed,'arm':arm,'common_rows':32,'candidates':480,'before':before,
                'after':runtime.tree_digest(state),'thresholds':thresholds,'finite_scores':True})
            print(json.dumps(cases[-1]),flush=True)
    assert len(cases)==6 and before_sources=={str(p):runner.digest(p) for p in HASHED}
    runner.write_json(runner.HERE/'PREFLIGHT_REAL.json',{'status':'passed_no_simulator_or_updates',
        'cases':cases,'source_hashes':before_sources,'inputs':pins,'simulation_transitions':0,'optimizer_updates':0,
        'scope':'previously inspected training states used for plumbing only, no new outcomes'})
    print('PASS: 6 checkpoint threshold reconstructions and shared-bank scoring; no simulation',flush=True)


if __name__=='__main__': main()
