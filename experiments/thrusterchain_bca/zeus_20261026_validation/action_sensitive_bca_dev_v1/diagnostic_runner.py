"""Development-only frozen-policy diagnostic; see DIAGNOSTIC_PROTOCOL.md.

Explicit CPU, one-shot execution only. Never invokes policy/critic optimizers or
the simulator. All cached reference values are held out from fitting/matching.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RERUN = BASE / 'central_quartile_training_v1_evaluation_rerun1'
sys.path.insert(0, str(RERUN))
import evaluation
import evaluate_worker as frozen
import rerun_inputs
import runtime
sys.path.insert(0, str(HERE))
import compatibility
import q1_bca

CAP_SECONDS = 7200
STEPS = 2000
BATCH = 256
K = 64
TOL = 1e-5
SEEDS = (0, 1, 2)
ARMS = ('native', 'bca')
CONTEXTS = ('ordinary', 'strong_current')
SEALED_FILES = ('DIAGNOSTIC_PROTOCOL.md','diagnostic_runner.py','q1_bca.py','compatibility.py',
                'actor_penalty.py','preflight_cpu.py','test_q1_bca.py','test_compatibility.py',
                'test_actor_penalty.py','test_diagnostic_runner.py','seal_dev.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def check_seal(ref_hash):
    seal_raw=(HERE/'DEV_SEAL.json').read_bytes(); seal=json.loads(seal_raw)
    require(seal.get('schema')=='bca.action_sensitive.development_integrity.v1'
            and set(seal['files'])==set(SEALED_FILES), 'Invalid development seal')
    require(seal['reference_hashes']==ref_hash,'Reference identity differs from development seal')
    for name,expected in seal['files'].items():
        require(sha((HERE/name).read_bytes())==expected,'Development source differs from seal: '+name)
    return seal,sha(seal_raw)


def artifact(base, desc):
    require(set(desc) == {'path', 'bytes', 'sha256'}, 'Bad cached descriptor')
    rel = Path(desc['path'])
    require(not rel.is_absolute() and len(rel.parts) == 1 and ':' not in str(rel), 'Unsafe artifact')
    path = Path(base) / rel
    require(path.is_file() and not path.is_symlink(), 'Missing or linked artifact')
    raw = path.read_bytes()
    require(len(raw) == desc['bytes'] and sha(raw) == desc['sha256'], 'Cached hash mismatch')
    return raw


def completed():
    raw = (RERUN/'RUN_RESULT.json').read_bytes(); terminal = json.loads(raw)
    require(terminal.get('status') == 'completed_evaluation_rerun' and terminal.get('error') is None
            and terminal.get('prepared_identity_unchanged') and terminal.get('execution_counts_known')
            and terminal.get('all_declared_jobs_completed'), 'No completed reference evaluation')
    rep = (RERUN/'run/evaluation/report.json').read_bytes()
    require(sha(rep) == terminal['jobs'][0]['report_sha256'], 'Evaluation report hash mismatch')
    report = json.loads(rep); repro = report['reproduction']
    require(repro == terminal['reproduction'] and repro['contexts_checked'] == 144
            and repro['tier1_exact'] == 143 and repro['tier2_near_exact'] == 0
            and sorted(repro['banks_exact']) == ['ordinary', 'strong_current']
            and len([k for k in report['snapshot_results'] if '/' in k]) == 144,
            'Reference reproduction gate differs')
    return terminal, report, {'terminal': sha(raw), 'report': sha(rep)}


def cpu_receipt(native):
    import importlib.metadata as md
    jax = native.jax
    require(os.environ.get('JAX_PLATFORMS') == 'cpu' and jax.default_backend() == 'cpu',
            'Development runner requires explicit CPU execution')
    require(str(jax.config.jax_default_matmul_precision) == 'highest' and not jax.config.jax_enable_x64
            and not jax.config.jax_disable_jit, 'Float32, highest-precision JIT required')
    packages = {}
    for name in ('jax','jaxlib','numpy','flax','optax','distrax','chex','tensorflow-probability','scipy'):
        packages[name] = md.version(name)
    return {'backend': 'cpu', 'executable': sys.executable, 'python': sys.version,
            'devices': [str(v) for v in jax.devices()], 'packages': packages,
            'jax_default_matmul_precision':str(jax.config.jax_default_matmul_precision),
            'jax_enable_x64':bool(jax.config.jax_enable_x64), 'native_source':native.__file__,
            'source_receipt':native._source_receipt,
            'deviation':'CPU development diagnostic; frozen GPU study unchanged'}


def pair_error(before, after, reference):
    return evaluation.paired_error_change(before, after, reference)


def selected(scores):
    scores = np.asarray(scores, np.float64)
    require(scores.shape == (32,6) and np.isfinite(scores).all(), 'Selection shape')
    return np.argmax(scores.max(axis=1, keepdims=True)-scores <= 1e-6, axis=1)


def summarize_bank(raw, values, scores, observations, actions):
    raw, values = np.asarray(raw, np.float64), np.asarray(values, np.float64)
    require(raw.shape == values.shape == (32,6), 'Reference bank shape')
    high,_ = evaluation.hole_mask(np.asarray(observations,np.float32),np.zeros((32,4),np.float32))
    slots=np.arange(6)[None,:]
    masks = {'all': np.ones((32,6),bool), 'high_band_fixed':high[:,None] & np.isin(slots,[2,3,4]),
             'high_flanks_fixed':high[:,None] & np.isin(slots,[1,5]),
             'low_band_fixed':~high[:,None] & np.isin(slots,[2,3,4]),
             'low_flanks_fixed':~high[:,None] & np.isin(slots,[1,5]),
             'own_actor':np.broadcast_to(slots==0,(32,6))}
    original_choice=selected(raw); idx=np.arange(32)
    q_order=np.argsort(-raw,axis=1,kind='stable'); best=q_order[:,0]; second=q_order[:,1]
    gap=raw[idx,best]-raw[idx,second]
    out={}
    for name, s in scores.items():
        s=np.asarray(s,np.float64); require(s.shape==(32,6) and np.isfinite(s).all(),'Invalid candidate score')
        choice=selected(s); gain=values[idx,choice]-values[idx,original_choice]
        penalty=raw-s
        out[name]={'error': {k:evaluation.error_metrics(s[m],values[m]) for k,m in masks.items()},
            'error_change_vs_raw': {k:pair_error(raw[m],s[m],values[m]) for k,m in masks.items()},
            'decision':evaluation.decision_metrics(s,values),
            'choice_change_vs_raw':{'changed':int((choice!=original_choice).sum()),
                'improved':int((gain>.1).sum()),'harmed':int((gain<-.1).sum()),
                'near_tied':int((abs(gain)<=.1).sum()),'per_state_gain':gain.tolist()},
            'penalty_mean':float(penalty.mean()), 'penalty_range_per_state':np.ptp(penalty,axis=1).tolist(),
            'top_two_q_margin':gap.tolist(),
            'top_two_penalty_difference':(penalty[idx,best]-penalty[idx,second]).tolist()}
    return {'high_load_anchor_families':int(high.sum()),'scores':out}


def time_index(obs):
    x=np.asarray(obs,np.float64); raw=(1.-x[:,-1])*100
    t=np.rint(raw).astype(np.int64)
    require(np.max(np.abs(raw-t)) < 1e-4 and np.all((0<=t)&(t<100)), 'Invalid time coordinate')
    return t


def cached_banks(report):
    import io
    output=RERUN/'run/evaluation'; result={}
    for name in CONTEXTS:
        meta=json.loads(artifact(output,report['snapshot_results'][name+'_bank']))
        raw=artifact(output,meta['arrays'])
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            result[name]={'observations':z['observations'].copy(),'times':z['times'].copy(),
                          'bank_id':meta['bank_id'],'sha256':sha(raw)}
        require(result[name]['observations'].shape==(32,14)
                and np.array_equal(time_index(result[name]['observations']),result[name]['times']), 'Bank shape/time mismatch')
    return result


def load_cached(report, seed, arm, context):
    output=RERUN/'run/evaluation'
    key=f'seed{seed}_hole__{arm}__20000/{context}'
    raw=artifact(output,report['snapshot_results'][key]); x=json.loads(raw)
    require((x['seed'],x['condition'],x['arm'],x['checkpoint'],x['context']) ==
            (seed,'hole',arm,20000,context) and x['complete_state_unchanged'], 'Cached identity mismatch')
    return x,sha(raw)


def parity(actual, cached, label):
    a,b=np.asarray(actual,np.float64),np.asarray(cached,np.float64)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' invalid')
    absolute=np.abs(a-b); scaled=absolute/(1+np.abs(b))
    require(np.max(scaled)<=TOL,label+' CPU/GPU parity exceeds 1e-5 scaled tolerance')
    return {'max_abs':float(absolute.max()),'max_scaled':float(scaled.max()),'tolerance':TOL}


def mechanism_screen(cases):
    """Frozen exploratory screen; this is not a policy-efficacy test."""
    cells={}; passes={}
    for context in CONTEXTS:
        groups={}
        for arm in ARMS:
            eligible=[]
            for seed in SEEDS:
                matches=[c for c in cases if c['seed']==seed and c['arm']==arm]
                require(len(matches)==1,'Screen checkpoint coverage differs')
                s=matches[0]['contexts'][context]['scores']; combined=s['combined']
                changes=combined['choice_change_vs_raw']; regret=combined['decision']['mean_regret']
                valid=(changes['changed']>=1 and changes['harmed']<=changes['improved']
                       and regret<s['compatibility_only']['decision']['mean_regret']
                       and regret<s['strength_matched_q1']['decision']['mean_regret'])
                cells[f'{context}/{arm}/seed{seed}']={'qualifies':bool(valid),'changed':changes['changed'],
                    'improved':changes['improved'],'harmed':changes['harmed'],'combined_mean_regret':regret,
                    'compatibility_mean_regret':s['compatibility_only']['decision']['mean_regret'],
                    'matched_q1_mean_regret':s['strength_matched_q1']['decision']['mean_regret']}
                eligible.append(valid)
            groups[arm]=int(sum(eligible))
        passes[context]={'qualifying_seeds':groups,'passed':all(n>=2 for n in groups.values())}
    return {'passed':any(x['passed'] for x in passes.values()),'contexts':passes,'cells':cells,
            'scope':'Development-only frozen-action mechanism screen, not learned policy efficacy'}


def rng_split_calibration(ctx):
    ids=ctx.data.heldout_ids
    order=np.argsort(ids['episode_ids'],kind='stable')
    fit,check=order[::2],order[1::2]
    require(len(fit)==len(check)==64 and len(np.unique(ids['episode_ids']))==128, 'Calibration split identity')
    require(not set(ids['episode_ids'][fit]) & set(ids['episode_ids'][check]), 'Calibration split overlap')
    return fit,check


def sliced(jax, batch, ix):
    return jax.tree_util.tree_map(lambda v:v[ix], batch)


def calibration_values(jax, q1_pipeline, original_pipeline, models, args, aux, heldout, refresh_key,
                       center='q1'):
    """Recompute exact threshold inputs using the refresh's identical noise key."""
    noise_key=jax.random.fold_in(refresh_key,1213156420)
    q1,target=q1_pipeline.q1_and_target(args,models,aux.native,heldout,noise_key)
    require(center in ('q1','twin_min'), 'Unknown calibration center')
    if center=='twin_min':
        heads=models[1].apply(aux.native.critic.params,heldout.obs,heldout.action)
        require(heads.shape==(64,2),'Twin-min calibration heads have wrong shape')
        q=heads.min(axis=-1)
    else:
        q=q1
    prediction=models[2].apply(aux.calibrator.params,heldout.obs,heldout.action)
    # Match freeze_reference: float32 residual arithmetic and positive_scale,
    # then float64 absolute normalized scores.
    residual=np.asarray(target-q,np.float64)
    scale=np.asarray(original_pipeline.positive_scale(prediction,aux.residual_scale),np.float64)
    require(q.shape==target.shape==prediction.shape==(64,) and residual.shape==scale.shape==(64,)
            and np.isfinite(residual).all() and np.isfinite(scale).all() and np.all(scale>0),
            'Calibration input reconstruction invalid')
    return {'center':center,'online_q1':np.asarray(q1).tolist(),'center_value':np.asarray(q).tolist(),
            'target':np.asarray(target).tolist(),
            'residual_target_minus_center':residual.tolist(),'scale_prediction':np.asarray(prediction).tolist(),
            'positive_scale_q_units':scale.tolist(),'normalized_absolute_score':(abs(residual)/scale).tolist(),
            'refresh_key':np.asarray(refresh_key).tolist(),'target_noise_key':np.asarray(noise_key).tolist()}


def finite_difference_checks(jax, jnp, policy, pipeline, models, ref, obs, actions, neighbors, bandwidth,
                             qg, ug, dg):
    """Check live action gradients on three fixed probes, no model updates."""
    step=1e-3; tolerance=.02; indices=(0,1,2); checks={}
    for name, expected in (('q1',qg),('width',ug),('compatibility',dg)):
        errors=[]
        for i in indices:
            o=obs[i:i+1]; n=neighbors[i:i+1]; h=bandwidth[i:i+1]
            def f(a):
                if name=='q1': return policy._q(o,a)[0,0]
                if name=='width': return pipeline.live_width(models,ref,o,a)[0][0]
                return compatibility.compatibility_cost_jax(a,n,h)[0]
            for dim in range(4):
                delta=np.zeros((1,4),np.float32); delta[0,dim]=step
                a=actions[i:i+1]; d=jnp.asarray(delta)
                estimate=float((np.asarray(f(a+d))-np.asarray(f(a-d)))/(2*step))
                true=float(expected[i,dim]); scaled=abs(estimate-true)/(1+abs(true))
                require(np.isfinite(scaled) and scaled<=tolerance, name+' finite difference failed')
                errors.append(scaled)
        checks[name]={'probe_rows':len(indices),'action_coordinates':4,'step':step,
                      'max_scaled_error':float(max(errors)),'tolerance':tolerance}
    return checks


def checkpoint(seed,arm,protocol,protocol_sha,adapter,template):
    cell=f'seed{seed}_hole'; directory=rerun_inputs.job_directory(cell,arm)
    report, matching, hashes=frozen.training_seal(directory,cell,arm,protocol_sha)
    spec={'cell':cell,'seed':seed,'condition':'hole','stage':arm,'arm':arm,'checkpoint':20000,
          'id':f'{cell}__{arm}__20000'}
    export,more=frozen.checkpoint_export(directory,report,spec,protocol_sha); hashes.update(more)
    export.update(matching=matching, training_report=report)
    # CPU reductions of training observations can differ from the GPU's saved
    # statistics. Restore the checkpoint's actual, authenticated static model
    # bindings, rather than normalizing with newly computed CPU statistics.
    # This is diagnostic-only and does not modify any frozen source.
    ctx=adapter.context; jnp=ctx.native.jnp
    n=export['manifest']['normalization']; mean=jnp.asarray(n['mean'],jnp.float32); std=jnp.asarray(n['std'],jnp.float32)
    cpu=ctx.normalization
    norm_diff={'mean_max_abs':float(np.max(np.abs(np.asarray(cpu['mean'])-np.asarray(n['mean'])))),
               'std_max_abs':float(np.max(np.abs(np.asarray(cpu['std'])-np.asarray(n['std'])))),
               'saved_mean':n['mean'],'saved_std':n['std']}
    require(mean.shape==(14,) and std.shape==(14,) and np.isfinite(mean).all() and np.isfinite(std).all(),
            'Invalid authenticated checkpoint normalization')
    actor=ctx.native.DeterministicTanhActor(4,mean,std)
    critic=ctx.native.DualQNetwork(mean,std)
    initial=template.microstep.agent_state
    agent=initial._replace(actor=initial.actor.replace(apply_fn=actor.apply),
        actor_target=initial.actor_target.replace(apply_fn=actor.apply),
        dual_q=initial.dual_q.replace(apply_fn=critic.apply),
        dual_q_target=initial.dual_q_target.replace(apply_fn=critic.apply))
    template=template._replace(microstep=template.microstep._replace(agent_state=agent),
        aux=template.aux._replace(native=ctx.pipeline.native_view(agent)),
        refresh_at=jnp.array([10000,15000],jnp.int32),critic_budget=jnp.int32(20000))
    branch_adapter=runtime.make_adapter(ctx,template,arm,3. if arm=='bca' else 0.,
                                      agent.actor.params,[10000,15000],20000)
    state_raw=frozen.artifact(export['base'],export['state_descriptor'])
    state=runtime.restore_exact(branch_adapter,template,state_raw)
    from state_exact import assert_exact_state
    assert_exact_state(frozen.artifact(export['base'],export['actor_descriptor']),
                       state.microstep.agent_state.actor.params,label='Full-state actor')
    require(runtime.validate(branch_adapter,state)['counts']==export['manifest']['counts'], 'Restored counts differ')
    return state,hashes,norm_diff,branch_adapter


def max_time(start):
    require(time.monotonic()-start<CAP_SECONDS, 'Diagnostic deadline exceeded')


def run(output):
    started=time.monotonic(); terminal,report,ref_hash=completed()
    seal,seal_sha=check_seal(ref_hash); code_hashes=seal['files']
    protocol_raw=(RERUN/'protocol.json').read_bytes(); protocol=json.loads(protocol_raw)
    require(sha(protocol_raw)==terminal['protocol_sha256'],'Protocol hash mismatch')
    native=runtime.load_native(); receipt=cpu_receipt(native)
    jax,jnp=native.jax,native.jnp
    banks=cached_banks(report); pool=q1_bca.make_q1_pipeline
    ledger={'actor_updates':0,'critic_updates':0,'actor_target_updates':0,'critic_target_updates':0,
            'simulator_transitions':0,'auxiliary_updates':0,'q1_auxiliary_updates':0,
            'twin_min_auxiliary_updates':0,'calibration_refreshes':0,
            'processed_checkpoints':0,'processed_contexts':0,'compatibility_evaluation_rows':0,
            'crossfit_training_probe_rows':0}
    def progress(event,**metadata):
        entry={'event':event,'elapsed_seconds':time.monotonic()-started,'committed_ledger':dict(ledger),
               'physical_counts_exact_only_after_success':True,**metadata}
        with (output/'progress.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(entry,sort_keys=True,allow_nan=False)+'\n'); f.flush()
    progress('initialized')
    cases=[]; all_hashes={}; reference=None; cross=None; data_hash=None
    for seed in SEEDS:
        max_time(started)
        data=runtime.module('_dev_q1_data', RERUN/'data.py').load_condition(native,'hole')
        adapter,template=runtime._initialize_data(protocol,seed,'hole',native,data,receipt)
        max_time(started)
        ctx=adapter.context
        train_obs=np.asarray(data.train.obs,np.float32); train_actions=np.asarray(data.train.action,np.float32)
        train_ids=np.asarray(data.source_row_ids,np.int64); episode_ids=train_ids//100; timesteps=train_ids%100
        require(len(train_ids)==67108 and len(np.unique(episode_ids))==768, 'Training identity mismatch')
        probe_ids=np.asarray(json.loads((RERUN/'DATA_PLAN.json').read_bytes())['conditions']['hole']['gradient_probe_source_row_ids'],np.int64)
        lookup={int(v):i for i,v in enumerate(train_ids)}
        indices=np.array([lookup[int(v)] for v in probe_ids],np.int64)
        require(len(indices)==768 and np.array_equal(train_obs[indices],np.asarray(data.probe.obs)), 'Probe identity mismatch')
        if reference is None:
            reference=compatibility.fit_reference(train_obs,train_actions,episode_ids,timesteps,row_ids=train_ids,k=K,time_bin=10)
            cross=compatibility.crossfit_query(train_obs,train_actions,episode_ids,timesteps,indices,row_ids=train_ids,k=K,time_bin=10)
            max_time(started)
            ledger['crossfit_training_probe_rows']+=len(indices)
            data_hash=sha(train_ids.astype('<i8').tobytes())
            for b in banks.values():
                b['neighbors']=compatibility.query_neighbors(reference,b['observations'],b['times'])
        else:
            require(sha(train_ids.astype('<i8').tobytes())==data_hash,'Training rows differ across seeds')
        cal_ix,check_ix=rng_split_calibration(ctx)
        for arm in ARMS:
            max_time(started)
            progress('checkpoint_started',seed=seed,arm=arm)
            state, hashes,norm_diff,branch_adapter=checkpoint(seed,arm,protocol,sha(protocol_raw),adapter,template); all_hashes.update(hashes)
            before=runtime.tree_digest(state); agent=state.microstep.agent_state
            policy=frozen.FrozenPolicy(runtime,branch_adapter,state)
            native_model=agent.actor.apply_fn.__self__
            original_cal=ctx.pipeline.Calibrator(native_model.obs_mean,native_model.obs_std,state_dep=True)
            original_models=(ctx.models[0],ctx.models[1],original_cal)
            policy._width=jax.jit(lambda o,a:ctx.pipeline.live_width(original_models,state.aux.posterior,o,a))
            init_seed=2026102900+seed*2+(arm=='bca')
            models,aux,config,args=q1_bca.fresh_auxiliary_exact_normalization(ctx,agent,init_seed,
                expected_mean=norm_diff['saved_mean'],expected_std=norm_diff['saved_std'])
            twin_models,twin_aux,twin_config,twin_args=q1_bca.fresh_auxiliary_exact_normalization(ctx,agent,init_seed,
                expected_mean=norm_diff['saved_mean'],expected_std=norm_diff['saved_std'])
            initial_digest=runtime.tree_digest((aux.calibrator,aux.posterior,aux.residual_scale))
            require(initial_digest==runtime.tree_digest((twin_aux.calibrator,twin_aux.posterior,twin_aux.residual_scale)),
                    'Fresh auxiliary initial states differ')
            pipeline=pool(ctx.pipeline)
            fit=jax.jit(lambda st,b,k:pipeline.fit_scale(args,config,models,st,b,k))
            twin_fit=jax.jit(lambda st,b,k:ctx.pipeline.fit_scale(twin_args,twin_config,twin_models,st,b,k))
            rng=np.random.Generator(np.random.PCG64(init_seed)); key=jax.random.PRNGKey(init_seed)
            for step in range(STEPS):
                max_time(started)
                batch=sliced(jax,ctx.train_transition,rng.integers(0,len(train_ids),BATCH))
                key,sub=jax.random.split(key); aux,metrics=fit(aux,batch,sub)
                require(bool(np.asarray(metrics['scale_fit_accepted'])),'Auxiliary update rejected')
                ledger['auxiliary_updates']+=1;ledger['q1_auxiliary_updates']+=1
                max_time(started)
                twin_aux,twin_metrics=twin_fit(twin_aux,batch,sub)
                require(bool(np.asarray(twin_metrics['scale_fit_accepted'])),'Twin-min auxiliary update rejected')
                ledger['auxiliary_updates']+=1;ledger['twin_min_auxiliary_updates']+=1
                max_time(started)
                if (step+1)%250==0:
                    progress('auxiliary_committed',seed=seed,arm=arm,checkpoint_auxiliary_updates=step+1)
                    print(json.dumps({'seed':seed,'arm':arm,'auxiliary_updates':step+1}),flush=True)
            require(int(np.asarray(aux.calibrator.step))==int(np.asarray(twin_aux.calibrator.step))==STEPS,
                    'Auxiliary optimizer count mismatch')
            key,sub=jax.random.split(key)
            calibration_batch=sliced(jax,ctx.heldout_transition,cal_ix)
            aux,flags,detail=pipeline.refresh(args,config,models,aux,calibration_batch,sub,
                                              heldout_ids=np.asarray(ctx.heldout_ids['row_ids'])[cal_ix])
            max_time(started)
            require(bool(flags['posterior_certified']) and bool(np.asarray(aux.posterior.ready))
                    and bool(np.asarray(ctx.pipeline.reference_valid(aux.posterior))), 'Q1 calibration rejected')
            ledger['calibration_refreshes']+=1
            twin_aux,twin_flags,twin_detail=ctx.pipeline.refresh(twin_args,twin_config,twin_models,twin_aux,
                calibration_batch,sub,heldout_ids=np.asarray(ctx.heldout_ids['row_ids'])[cal_ix])
            max_time(started)
            require(bool(twin_flags['posterior_certified']) and bool(np.asarray(twin_aux.posterior.ready))
                    and bool(np.asarray(ctx.pipeline.reference_valid(twin_aux.posterior))), 'Twin-min calibration rejected')
            ledger['calibration_refreshes']+=1
            ref=aux.posterior
            cal_values=calibration_values(jax,pipeline,ctx.pipeline,models,args,aux,calibration_batch,sub)
            twin_cal_values=calibration_values(jax,pipeline,ctx.pipeline,twin_models,twin_args,twin_aux,
                                               calibration_batch,sub,center='twin_min')
            require(detail['scores']==len(cal_values['normalized_absolute_score']), 'Calibration score count differs')
            require(twin_detail['scores']==len(twin_cal_values['normalized_absolute_score'])
                    and np.array_equal(cal_values['target'],twin_cal_values['target']),
                    'Matched twin-min calibration target or count differs')
            max_time(started)
            from flax import serialization
            saved=serialization.to_bytes({'calibrator':aux.calibrator,'posterior':ref,
                'residual_scale':aux.residual_scale,'aux_rng':key})
            filename=f'seed{seed}_hole__{arm}__q1_aux.msgpack'
            with (output/filename).open('xb') as stream: stream.write(saved)
            aux_artifact={'path':filename,'sha256':sha(saved),'bytes':len(saved)}
            twin_ref=twin_aux.posterior
            twin_saved=serialization.to_bytes({'calibrator':twin_aux.calibrator,'posterior':twin_ref,
                'residual_scale':twin_aux.residual_scale,'aux_rng':key})
            twin_filename=f'seed{seed}_hole__{arm}__twin_min_aux.msgpack'
            with (output/twin_filename).open('xb') as stream: stream.write(twin_saved)
            twin_artifact={'path':twin_filename,'sha256':sha(twin_saved),'bytes':len(twin_saved)}
            width_fn=jax.jit(lambda o,a: pipeline.live_width(models,ref,o,a))
            twin_width_fn=jax.jit(lambda o,a:ctx.pipeline.live_width(twin_models,twin_ref,o,a))
            obs=jnp.asarray(train_obs[indices]); actions=policy._act(obs)
            neighbors=jnp.asarray(cross['actions']); widths=jnp.asarray(cross['bandwidths'],jnp.float32)
            gradq=jax.jit(jax.grad(lambda a:jnp.sum(policy._q(obs,a)[:,0])))(actions)
            gradu=jax.jit(jax.grad(lambda a:jnp.sum(pipeline.live_width(models,ref,obs,a)[0])))(actions)
            gradd=jax.jit(jax.grad(lambda a:jnp.sum(compatibility.compatibility_cost_jax(a,neighbors,widths))))(actions)
            qg,ug,dg=map(lambda x:np.asarray(x,np.float64),(gradq,gradu,gradd))
            fd=finite_difference_checks(jax,jnp,policy,pipeline,models,ref,obs,actions,neighbors,widths,qg,ug,dg)
            max_time(started)
            rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
            qr,ur,dr=map(rms,(qg,ug,dg))
            require(all(np.isfinite(x) and x>1e-6 for x in (qr,ur,dr)),'Gradient matching invalid')
            eta=.25*qr/dr; combined=rms(3*ug+eta*dg); matched=combined/ur
            require(0<eta<=30 and 0<matched<=30 and np.isfinite(combined),'Matching coefficient invalid or exceeds bound')
            check=sliced(jax,ctx.heldout_transition,check_ix)
            cq,cy=pipeline.q1_and_target(args,models,aux.native,check,jax.random.fold_in(key,1357911))
            w,usable,valid=width_fn(check.obs,check.action)
            require(bool(np.asarray(usable).all()) and bool(np.asarray(valid).all()),'Heldout width invalid')
            residual=np.asarray(cy)-np.asarray(cq); w=np.asarray(w)
            twin_heads=twin_models[1].apply(twin_aux.native.critic.params,check.obs,check.action)
            require(twin_heads.shape==(64,2),'Heldout twin-min shape invalid')
            twin_center=np.asarray(twin_heads).min(axis=-1)
            twin_residual=np.asarray(cy)-twin_center
            twin_w,twin_usable,twin_valid=twin_width_fn(check.obs,check.action)
            require(bool(np.asarray(twin_usable).all()) and bool(np.asarray(twin_valid).all()),'Twin-min heldout width invalid')
            twin_w=np.asarray(twin_w)
            case={'seed':seed,'arm':arm,'checkpoint':20000,'source_hashes':hashes,'cpu_vs_saved_normalization':norm_diff,
                'auxiliary_init_seed':init_seed,'updates':STEPS,'calibration_rows':64,'check_rows':64,
                'q1_auxiliary_artifact':aux_artifact,'fresh_twin_min_auxiliary_artifact':twin_artifact,
                'auxiliary_initial_digest':initial_digest,
                'sampler_rng_state':rng.bit_generator.state,'aux_rng_after_refresh':np.asarray(key).tolist(),
                'calibration_episode_ids':np.asarray(ctx.heldout_ids['episode_ids'])[cal_ix].tolist(),
                'check_episode_ids':np.asarray(ctx.heldout_ids['episode_ids'])[check_ix].tolist(),
                'calibration_row_ids':np.asarray(ctx.heldout_ids['row_ids'])[cal_ix].tolist(),
                'check_row_ids':np.asarray(ctx.heldout_ids['row_ids'])[check_ix].tolist(),
                'calibration_flags':runtime.host(flags),'calibration_details':runtime.host(detail),
                'calibration_input_rows':cal_values,
                'fresh_twin_min_calibration_flags':runtime.host(twin_flags),
                'fresh_twin_min_calibration_details':runtime.host(twin_detail),
                'fresh_twin_min_calibration_input_rows':twin_cal_values,
                'residual_unit':float(np.asarray(aux.residual_scale)),'threshold':float(np.asarray(ref.threshold)),
                'fresh_twin_min_residual_unit':float(np.asarray(twin_aux.residual_scale)),
                'fresh_twin_min_threshold':float(np.asarray(twin_ref.threshold)),
                'heldout_td_abs_coverage':float(np.mean(abs(residual)<=w)),
                'heldout_td_abs_mean':float(np.mean(abs(residual))),'heldout_width_mean':float(w.mean()),
                'fresh_twin_min_own_center_td_abs_coverage':float(np.mean(abs(twin_residual)<=twin_w)),
                'fresh_twin_min_q1_center_td_abs_coverage':float(np.mean(abs(residual)<=twin_w)),
                'q1_width_twin_min_center_td_abs_coverage':float(np.mean(abs(twin_residual)<=w)),
                'heldout_td_rows':{'online_q1':np.asarray(cq).tolist(),'target':np.asarray(cy).tolist(),
                    'online_twin_min':twin_center.tolist(),'residual_target_minus_q1':residual.tolist(),
                    'residual_target_minus_twin_min':twin_residual.tolist(),
                    'aligned_width':w.tolist(),'fresh_twin_min_width':twin_w.tolist()},
                'matching':{'probe_rows':768,'q_action_rms':qr,'u_action_rms':ur,'d_action_rms':dr,
                            'eta':eta,'combined_action_rms':combined,'matched_kappa':matched,
                            'probe_row_ids':probe_ids.tolist(),'probe_actor_actions':np.asarray(actions).tolist(),
                            'q_action_gradients':qg.tolist(),'u_action_gradients':ug.tolist(),'d_action_gradients':dg.tolist(),
                            'folds':cross['query_folds'].tolist(),'fold_bandwidths':cross['bandwidths'].tolist(),
                            'fold_references':cross['reference_sha256_by_fold'],'finite_difference_checks':fd}, 'contexts':{}}
            for context in CONTEXTS:
                max_time(started)
                x,cached_hash=load_cached(report,seed,arm,context); bank=banks[context]
                require(x['bank_id']==bank['bank_id'],'Cached context bank mismatch')
                obs=np.asarray(bank['observations'],np.float32); acts=np.asarray(x['q']['per_probe']['actions'],np.float32)
                raw=np.asarray(x['q']['per_probe']['q1'],np.float64)
                require(acts.shape==(32,6,4) and raw.shape==(32,6),'Cached bank dimensions')
                actor_parity=parity(policy.act(obs),acts[:,0,:],'Actor')
                repeated=np.repeat(obs,6,axis=0); flat=acts.reshape(192,4)
                qparity=parity(policy.q(repeated,flat)[:,0].reshape(32,6),raw,'Online Q1')
                original=np.asarray(x['q']['per_probe']['adjusted'],np.float64)
                original_width_parity=parity(policy.width(repeated,flat).reshape(32,6),
                                             (raw-original)/3.,'Original bank width')
                newwidth,usable,valid=width_fn(jnp.asarray(repeated),jnp.asarray(flat))
                require(bool(np.asarray(usable).all()) and bool(np.asarray(valid).all()),'Bank width invalid')
                newwidth=np.asarray(newwidth,np.float64).reshape(32,6)
                twinwidth,twin_usable,twin_valid=twin_width_fn(jnp.asarray(repeated),jnp.asarray(flat))
                require(bool(np.asarray(twin_usable).all()) and bool(np.asarray(twin_valid).all()), 'Twin-min bank width invalid')
                twinwidth=np.asarray(twinwidth,np.float64).reshape(32,6)
                nb=np.repeat(bank['neighbors']['actions'],6,axis=0)
                cost=np.asarray(compatibility.compatibility_cost_jax(jnp.asarray(flat),jnp.asarray(nb),jnp.asarray(reference.action_bandwidth,jnp.float32)),np.float64).reshape(32,6)
                ledger['compatibility_evaluation_rows']+=192
                # These references are accessed only after fitting, calibration and both matches are frozen.
                qpi=np.asarray(x['q']['per_probe']['qpi'],np.float64)
                scores={'raw_q1':raw,'original_bca':original,'fresh_twin_min':raw-3*twinwidth,'q1_aligned':raw-3*newwidth,
                        'compatibility_only':raw-eta*cost,'combined':raw-3*newwidth-eta*cost,
                        'strength_matched_q1':raw-matched*newwidth}
                result=summarize_bank(raw,qpi,scores,obs,acts)
                visited=x['visited']['per_row']; vo=np.asarray(visited['observations'],np.float32)
                va=np.asarray(visited['actions'],np.float32); vq=np.asarray(visited['q1'],np.float64)
                vqpi=np.asarray(visited['qpi_suffix'],np.float64)
                require(vo.shape==(128,14) and va.shape==(128,4),'Visited shape')
                vp=parity(policy.q(vo,va)[:,0],vq,'Visited Q1')
                v_original=np.asarray(visited['adjusted'],np.float64)
                v_width_parity=parity(policy.width(vo,va),(vq-v_original)/3.,'Original visited width')
                vn=compatibility.query_neighbors(reference,vo,time_index(vo))
                vw,vu,vv=width_fn(jnp.asarray(vo),jnp.asarray(va))
                require(bool(np.asarray(vu).all()) and bool(np.asarray(vv).all()),'Visited width invalid')
                tvw,tvu,tvv=twin_width_fn(jnp.asarray(vo),jnp.asarray(va))
                require(bool(np.asarray(tvu).all()) and bool(np.asarray(tvv).all()),'Twin-min visited width invalid')
                tvw=np.asarray(tvw,np.float64)
                vw=np.asarray(vw,np.float64); vd=np.asarray(compatibility.compatibility_cost_jax(jnp.asarray(va),jnp.asarray(vn['actions']),jnp.asarray(reference.action_bandwidth,jnp.float32)),np.float64)
                ledger['compatibility_evaluation_rows']+=128
                vs={'raw_q1':vq,'original_bca':v_original,'fresh_twin_min':vq-3*tvw,
                    'q1_aligned':vq-3*vw,'compatibility_only':vq-eta*vd,
                    'combined':vq-3*vw-eta*vd,'strength_matched_q1':vq-matched*vw}
                vh,vhole=evaluation.hole_mask(vo,va)
                vmasks={'all':np.ones(128,bool),'high_hole':vhole,'high_nonhole':vh & ~vhole,
                        'low_band':~vh & (np.abs(va[:,0])<=.25),'low_nonband':~vh & (np.abs(va[:,0])>.25)}
                result['visited']={k:{'error':{m:evaluation.error_metrics(v[z],vqpi[z]) for m,z in vmasks.items()},
                    'change_vs_raw':{m:pair_error(vq[z],v[z],vqpi[z]) for m,z in vmasks.items()}} for k,v in vs.items()}
                result['per_probe']={'observations':obs.tolist(),'actions':acts.tolist(),'q1':raw.tolist(),
                    'qpi':qpi.tolist(),'q1_aligned_width':newwidth.tolist(),'fresh_twin_min_width':twinwidth.tolist(),
                    'compatibility_cost':cost.tolist(),
                    'scores':{k:v.tolist() for k,v in scores.items()},'bank_id':bank['bank_id']}
                result['visited_per_row']={'observations':vo.tolist(),'actions':va.tolist(),'q1':vq.tolist(),
                    'qpi_suffix':vqpi.tolist(),'q1_aligned_width':vw.tolist(),'fresh_twin_min_width':tvw.tolist(),
                    'compatibility_cost':vd.tolist(),
                    'scores':{k:v.tolist() for k,v in vs.items()}}
                result.update(cached_sha256=cached_hash, actor_parity=actor_parity,q1_parity=qparity,
                              original_width_parity=original_width_parity,visited_q1_parity=vp,
                              visited_original_width_parity=v_width_parity)
                case['contexts'][context]=result; ledger['processed_contexts']+=1
                max_time(started)
            policy.assert_unchanged(); require(runtime.tree_digest(state)==before,'Frozen state changed')
            cases.append(case); ledger['processed_checkpoints']+=1
            encoded=(json.dumps(case,sort_keys=True,allow_nan=False)+'\n').encode()
            with (output/f'seed{seed}_hole__{arm}__case.json').open('xb') as f: f.write(encoded)
            progress('checkpoint_completed',seed=seed,arm=arm,case_sha256=sha(encoded))
            print(json.dumps({'completed':f'seed{seed}_hole/{arm}','ledger':ledger}),flush=True)
    require(ledger['auxiliary_updates']==24000 and ledger['q1_auxiliary_updates']==12000
            and ledger['twin_min_auxiliary_updates']==12000 and ledger['calibration_refreshes']==12
            and ledger['processed_checkpoints']==6 and ledger['processed_contexts']==12
            and ledger['compatibility_evaluation_rows']==3840 and ledger['crossfit_training_probe_rows']==768
            and not any(ledger[k] for k in ('actor_updates','critic_updates','actor_target_updates','critic_target_updates','simulator_transitions')),
            'Final ledger mismatch')
    for path,expected in all_hashes.items():require(sha(Path(path).read_bytes())==expected,'Checkpoint source changed')
    for name,expected in code_hashes.items():require(sha((HERE/name).read_bytes())==expected,'Diagnostic source changed')
    require(sha((RERUN/'RUN_RESULT.json').read_bytes())==ref_hash['terminal']
            and sha((RERUN/'run/evaluation/report.json').read_bytes())==ref_hash['report'], 'Reference changed')
    require(sha((HERE/'DEV_SEAL.json').read_bytes())==seal_sha,'Development integrity seal changed')
    max_time(started)
    result={'status':'completed_development_diagnostic','protocol_sha256':sha((HERE/'DIAGNOSTIC_PROTOCOL.md').read_bytes()),
        'reference_hashes':ref_hash,'runtime':receipt,'ledger':ledger,'source_hashes':all_hashes,
        'diagnostic_code_sha256':code_hashes,'development_seal_sha256':seal_sha,
        'reference_data_sha256':reference.fingerprint_sha256,'training_row_ids_sha256':data_hash,
        'reference_action_bandwidth':reference.action_bandwidth.tolist(),
        'elapsed_seconds':time.monotonic()-started,'cases':cases,'mechanism_screen':mechanism_screen(cases),
        'limitations':['Previously consumed development episodes and Q references',
                       'No trained policy updates or new return estimates',
                       'Compatibility is not a verified support boundary',
                       'One-step Q1 residual calibration is not a true-Q or shifted-action guarantee']}
    encoded=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    max_time(started)
    temporary=output/'RESULT.json.tmp'
    with temporary.open('x',encoding='utf8',newline='\n') as f: f.write(encoded)
    max_time(started)
    temporary.replace(output/'RESULT.json')
    progress('complete')
    return {'status':result['status'],'ledger':ledger,'elapsed_seconds':result['elapsed_seconds']}


if __name__=='__main__':
    require(os.environ.get('JAX_PLATFORMS')=='cpu','Set JAX_PLATFORMS=cpu before running')
    output=HERE/'diagnostic_run_01'
    output.mkdir(exist_ok=False)
    try:
        print(json.dumps(run(output)),flush=True)
    except BaseException as exc:
        (output/'FAILURE.json').write_text(json.dumps({'status':'failed_no_automatic_retry','error':str(exc),
            'traceback':traceback.format_exc()},indent=2)+'\n',encoding='utf8')
        raise
