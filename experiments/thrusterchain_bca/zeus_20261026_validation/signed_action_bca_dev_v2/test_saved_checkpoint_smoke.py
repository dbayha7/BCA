"""Read-only real-checkpoint restore and calibration plumbing, no simulator/updates."""
import json
import numpy as np
import runner


def main():
    old=runner.old;runtime=runner.runtime; native=runtime.load_native();receipt=old.cpu_receipt(native)
    terminal,report,refs=old.completed();old.check_seal(refs)
    raw=(runner.RERUN/'protocol.json').read_bytes();protocol=json.loads(raw)
    prior=json.loads((runner.PREVIOUS/'diagnostic_run_01/RESULT.json').read_bytes())
    bank=old.cached_banks(report)['ordinary']; tested=0
    for seed in old.SEEDS:
        data=runtime.module('_signed_smoke_data',runner.RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt)
        ctx=adapter.context
        for arm in old.ARMS:
            state,_,norm,branch=old.checkpoint(seed,arm,protocol,old.sha(raw),adapter,template)
            before=runtime.tree_digest(state);agent=state.microstep.agent_state
            policy=runner.frozen.FrozenPolicy(runtime,branch,state)
            saved,_=old.load_cached(report,seed,arm,'ordinary')
            acts=np.asarray(saved['q']['per_probe']['actions'],np.float32)
            obs=np.repeat(bank['observations'],6,axis=0)
            old.parity(policy.act(bank['observations']),acts[:,0,:],'smoke actor')
            old.parity(policy.q(obs,acts.reshape(-1,4))[:,0].reshape(32,6),saved['q']['per_probe']['q1'],'smoke Q1')
            case=next(c for c in prior['cases'] if c['seed']==seed and c['arm']==arm)
            for kind in ('q1','twin'):
                models,aux,config,args=runner.restore_aux(ctx,agent,norm,case['auxiliary_init_seed'],case,kind)
                cal=old.sliced(native.jax,ctx.heldout_transition,np.arange(64))
                pipeline=runner.q1_bca.make_q1_pipeline(ctx.pipeline) if kind=='q1' else ctx.pipeline
                aux,flags,_=pipeline.refresh(args,config,models,aux,cal,native.jax.random.PRNGKey(123),
                    heldout_ids=np.asarray(ctx.heldout_ids['row_ids'])[:64])
                assert flags['posterior_certified'] and bool(np.asarray(ctx.pipeline.reference_valid(aux.posterior)))
                w,usable,valid=pipeline.live_width(models,aux.posterior,cal.obs,cal.action)
                assert np.isfinite(w).all() and bool(np.asarray(usable).all()) and bool(np.asarray(valid).all())
                assert int(np.asarray(aux.calibrator.step))==2000
            sm,sa,sc,sargs=runner.signed_bca.initialize(ctx,agent,case['auxiliary_init_seed']+10000,
                expected_mean=norm['saved_mean'],expected_std=norm['saved_std'])
            test_batch=old.sliced(native.jax,ctx.train_transition,np.arange(32))
            fit=native.jax.jit(lambda st,b,k:runner.signed_bca.fit_step(ctx.pipeline,sargs,sm,st,b,k))
            temp,metrics=fit(sa,test_batch,native.jax.random.PRNGKey(123))
            assert bool(np.asarray(metrics['signed_fit_accepted'])) and np.isfinite(np.asarray(metrics['signed_loss']))
            assert int(np.asarray(temp.calibrator.step))==1 and int(np.asarray(sa.calibrator.step))==0
            function=runner.signed_function(native.jax,sm,temp,1.0)
            action=native.jnp.asarray(test_batch.action); obs=native.jnp.asarray(test_batch.obs)
            grad=native.jax.grad(lambda x:native.jnp.sum(function(obs,x)[0]))(action)
            assert np.isfinite(np.asarray(grad)).all()
            assert runtime.tree_digest(state)==before
            tested+=1
            print(f'{seed}/{arm}: restored, parity, both widths, signed update/gradient passed',flush=True)
    print(f'{tested} checkpoints; 6 discarded auxiliary test updates; no simulator or actor/critic updates')


if __name__=='__main__': main()
