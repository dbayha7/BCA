"""Read-only CPU restore/parity feasibility check; no fitting or simulation."""
import hashlib
import json

import diagnostic_runner as d


def main():
    terminal,report,_=d.completed()
    raw=(d.RERUN/'protocol.json').read_bytes(); protocol=json.loads(raw)
    d.require(hashlib.sha256(raw).hexdigest()==terminal['protocol_sha256'],'Protocol changed')
    native=d.runtime.load_native(); receipt=d.cpu_receipt(native)
    data=d.runtime.module('_dev_cpu_preflight_data',d.RERUN/'data.py').load_condition(native,'hole')
    adapter,template=d.runtime._initialize_data(protocol,0,'hole',native,data,receipt)
    state,_,normalization,branch_adapter=d.checkpoint(0,'native',protocol,hashlib.sha256(raw).hexdigest(),adapter,template)
    policy=d.frozen.FrozenPolicy(d.runtime,branch_adapter,state)
    ctx=branch_adapter.context; model=state.microstep.agent_state.actor.apply_fn.__self__
    cal=ctx.pipeline.Calibrator(model.obs_mean,model.obs_std,state_dep=True)
    models=(ctx.models[0],ctx.models[1],cal)
    policy._width=ctx.native.jax.jit(lambda o,a:ctx.pipeline.live_width(models,state.aux.posterior,o,a))
    banks=d.cached_banks(report);x,_=d.load_cached(report,0,'native','ordinary')
    obs=banks['ordinary']['observations']; acts=d.np.asarray(x['q']['per_probe']['actions'],d.np.float32)
    repeated=d.np.repeat(obs,6,axis=0)
    actor=d.parity(policy.act(obs),acts[:,0,:],'Actor')
    q=d.parity(policy.q(repeated,acts.reshape(192,4))[:,0].reshape(32,6),x['q']['per_probe']['q1'],'Q1')
    raw=d.np.asarray(x['q']['per_probe']['q1']); adj=d.np.asarray(x['q']['per_probe']['adjusted'])
    width=d.parity(policy.width(repeated,acts.reshape(192,4)).reshape(32,6),(raw-adj)/3.,'Original width')
    before=d.runtime.tree_digest(state)
    aligned_models,aligned_aux,aligned_config,aligned_args=d.q1_bca.fresh_auxiliary_exact_normalization(
        ctx,state.microstep.agent_state,2026102900,
        expected_mean=normalization['saved_mean'],expected_std=normalization['saved_std'])
    d.require(int(d.np.asarray(aligned_aux.calibrator.step))==0 and d.runtime.tree_digest(state)==before,
              'Fresh aligned auxiliary initialization changed native state')
    policy.assert_unchanged()
    print(json.dumps({'status':'read_only_cpu_restore_and_q_parity_passed','normalization':normalization,'actor_parity':actor,'q1_parity':q,'width_parity':width,
        'fresh_q1_auxiliary_step':int(d.np.asarray(aligned_aux.calibrator.step)),
        'actor_updates':0,'critic_updates':0,'auxiliary_updates':0,'simulator_transitions':0}))


if __name__=='__main__': main()
