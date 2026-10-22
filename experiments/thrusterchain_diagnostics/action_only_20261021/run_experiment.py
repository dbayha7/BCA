"""Bounded, resumable frozen-model action-shift evaluation. No training entry point."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import time

from engine import configure_runtime
RUNTIME = configure_runtime()
import numpy as np
from engine import FrozenScoreEngine, sha256
from calibration_methods import calibrate_queries, evaluate_queries, paired_metrics, json_ready, load_native_wbcp, advisor_source_map

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE.parent / 'results_20261021/release/ThrusterChain_results_20261021'
DEFAULT_RUN = Path(r'C:\Users\David\dev\BCA\experiments\thrusterchain_diagnostics\training_runs\core_20261020T2250Z')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def array_hash(arrays):
    h = hashlib.sha256()
    for key, val in sorted(arrays.items()):
        val = np.ascontiguousarray(val)
        h.update(key.encode()); h.update(str(val.dtype).encode()); h.update(str(val.shape).encode()); h.update(val.tobytes())
    return h.hexdigest()


def rng(protocol, rep, stream):
    return np.random.default_rng(np.random.SeedSequence([protocol['rng']['base_seed'], rep, stream]))


def sample_actions(probabilities, states, uniforms):
    cumulative = np.cumsum(probabilities[states], axis=1)
    cumulative[:, -1] = 1.0
    return np.sum(uniforms[:, None] >= cumulative, axis=1).astype(np.int64)


def distribution_stats(values):
    values = np.asarray(values, float)
    return dict(min=float(values.min()), median=float(np.median(values)), p90=float(np.quantile(values,.9)),
                p99=float(np.quantile(values,.99)), max=float(values.max()), mean=float(values.mean()), sum=float(values.sum()))


def grid_audit(engine):
    membership = np.all(np.any(engine.actor_actions[:, :, None].astype(float) == engine.atoms[None, None, :], axis=2), axis=1)
    actor_beta = engine.beta[np.arange(64), engine.actor_action_index]
    return dict(states=64, supported_actor_states=int(membership.sum()), off_grid_actor_states=int((~membership).sum()),
                exact_membership=membership.tolist(), continuous_actor=engine.actor_actions.tolist(),
                projected_action=engine.grid[engine.actor_action_index].tolist(),
                projection_l2=np.linalg.norm(engine.actor_actions-engine.grid[engine.actor_action_index],axis=1).tolist(),
                projected_actor_beta=actor_beta.tolist(), expected_positive_bank_hits_128=float(128*actor_beta.mean()),
                zero_hit_probability_128=float((1-actor_beta.mean())**128),
                conclusion='Projected actor target is supported on625 atoms. Off-grid deployed actor mass is structurally unsupported; no finite ratio or coverage is assigned to it.')


def source_inventory():
    return {p.relative_to(HERE).as_posix():sha256(p) for p in sorted(HERE.glob('*.py'))} | {'protocol.json':sha256(HERE/'protocol.json')}


def load_engine(args, protocol):
    return FrozenScoreEngine.load(bundle_root=args.bundle, canonical_run_dir=args.run,
                                  state_seed=protocol['state_law']['selection_seed'])


def prepare(args, protocol):
    if (HERE/'freeze.json').exists():
        raise RuntimeError('Already frozen; do not overwrite preflight provenance')
    started=time.perf_counter()
    engine=load_engine(args,protocol)
    catalog=engine.prepare_catalog()
    # Fixed shape/label fixture, unrelated to the registered calibration/evaluation draws.
    states=np.arange(4096,dtype=np.int64)%64
    actions=np.arange(4096,dtype=np.int64)%625
    noise=np.random.default_rng(2026102199).normal(size=(4096,4))
    engine.score(states,actions,noise)
    start=time.perf_counter()
    scored=engine.score(states,actions,noise)
    score_seconds=time.perf_counter()-start
    if not all(np.isfinite(x).all() for x in (scored.scores,scored.scale,scored.target)):
        raise AssertionError('invalid scorer fixture')
    for level in protocol['action_law']['mixtures']:
        q=engine.target_probabilities(level)
        assert np.allclose(q.sum(1),1,rtol=0,atol=2e-14)
        assert not np.any((q>0)&(engine.beta==0))
        assert np.allclose((engine.beta*(q/engine.beta)).sum(1),1,rtol=0,atol=2e-14)
    assert np.array_equal(engine.target_probabilities(0)/engine.beta,np.ones((64,625)))
    np.savez_compressed(HERE/'state_bank.npz',**engine.selected,grid=engine.grid,beta=engine.beta,
                        actor_actions=engine.actor_actions,actor_action_index=engine.actor_action_index)
    native=load_native_wbcp(Path(args.bundle)/'source/calibration/wbcp.py')
    write_json(HERE/'engine_identity.json',engine.metadata)
    write_json(HERE/'off_grid_support_audit.json',grid_audit(engine))
    write_json(HERE/'source_map_calibration.json',advisor_source_map(native))
    write_json(HERE/'preflight.json',dict(created_utc=utc(),runtime=RUNTIME,platform=platform.platform(),
        python=platform.python_version(),processor=platform.processor(),checkpoint=engine.metadata['checkpoint'],
        catalog_shapes=catalog,scoring_fixture_identity=array_hash(dict(states=states,actions=actions,noise=noise)),
        synchronized_score_seconds_4096=score_seconds,fixture_scores_finite=True,
        setup_seconds=time.perf_counter()-started,law_normalization=True,ratio_expectation_one=True,
        equal_weight_identity=True,beta_min=float(engine.beta.min()),beta_max=float(engine.beta.max()),
        expected_scoring_seconds_main=score_seconds*protocol['calibration']['replications']*(3+128/4096),
        note='Throughput fixture only: no registered replication or coverage result evaluated. Calibration and I/O add time.'))
    print(json.dumps(json_ready(dict(event='prepared',score_seconds_4096=score_seconds,setup_seconds=time.perf_counter()-started,
        off_grid=grid_audit(engine)['off_grid_actor_states'],expected_scoring_minutes=score_seconds*606.25/60))),flush=True)


def freeze(args, protocol):
    if (HERE/'freeze.json').exists():
        raise RuntimeError('Freeze already exists; never replace it')
    required=['preflight.json','engine_identity.json','state_bank.npz','off_grid_support_audit.json','review_preflight.md']
    for name in required:
        if not (HERE/name).is_file(): raise RuntimeError('Missing preflight artifact '+name)
    review=(HERE/'review_preflight.md').read_text(encoding='utf-8')
    if 'PREFLIGHT APPROVED' not in review: raise RuntimeError('Independent review has not approved execution')
    write_json(HERE/'freeze.json',dict(schema='action-only-freeze-v1',frozen_utc=utc(),protocol=protocol,
        source_sha256=source_inventory(),preflight_sha256={n:sha256(HERE/n) for n in required},
        bundle_root=str(Path(args.bundle).resolve()),canonical_run_dir=str(Path(args.run).resolve()),
        post_hoc=True,no_main_results_observed=True,replication_ids=list(range(protocol['calibration']['replications']))))
    print(json.dumps(dict(event='frozen',sha256=sha256(HERE/'freeze.json'),utc=utc())),flush=True)


def verify_freeze(args):
    record=json.loads((HERE/'freeze.json').read_text())
    for name,digest in record['source_sha256'].items():
        if sha256(HERE/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    for name,digest in record['preflight_sha256'].items():
        if sha256(HERE/name)!=digest:raise RuntimeError('Frozen preflight artifact changed: '+name)
    if str(Path(args.bundle).resolve())!=record['bundle_root'] or str(Path(args.run).resolve())!=record['canonical_run_dir']:
        raise RuntimeError('Source roots differ from freeze')
    return record


def counts(evaluation, selection=None):
    selection=np.ones(evaluation.covered.shape,dtype=bool) if selection is None else selection
    finite=~evaluation.refused & selection
    return dict(coverage_numerator=int(np.sum(evaluation.covered & selection)),coverage_denominator=int(selection.sum()),
        finite_coverage_numerator=int(np.sum(evaluation.covered & finite)),finite_coverage_denominator=int(finite.sum()),
        usable_coverage_numerator=int(np.sum(evaluation.usable & selection)),usable_coverage_denominator=int(selection.sum()),
        threshold_usable_count=int(finite.sum()),fallback_count=int(np.sum(evaluation.refused & selection)),
        vacuous_count=int(np.sum(evaluation.refused & selection)))


def run_rep(engine,native,protocol,rep,outdir):
    started=time.perf_counter(); n=protocol['calibration']['size']; m=protocol['evaluation']['queries_per_replication_per_mixture']
    cal_z=rng(protocol,rep,1).integers(0,64,n); cal_a=sample_actions(engine.beta,cal_z,rng(protocol,rep,2).random(n))
    cal_noise=rng(protocol,rep,3).normal(size=(n,4)); cal=engine.score(cal_z,cal_a,cal_noise)
    cal_arrays=dict(state_id=cal_z,action_id=cal_a,label_noise=cal_noise,scores=cal.scores,scale=cal.scale,target=cal.target,q=cal.q)
    bank_id=array_hash(cal_arrays)
    z=rng(protocol,rep,5).integers(0,64,m); base=sample_actions(engine.beta,z,rng(protocol,rep,6).random(m))
    mix=rng(protocol,rep,7).random(m); noise=rng(protocol,rep,8).normal(size=(m,4))
    posterior_seed=[protocol['rng']['base_seed'],rep,4]
    kwargs=dict(posterior_seed=posterior_seed,native=native,alpha=protocol['calibration']['miscoverage'],
                beta=protocol['calibration']['credibility'],draws=protocol['calibration']['posterior_draws'])
    plain=calibrate_queries(cal.scores,np.ones(n),np.ones(m),**kwargs)
    records=[]; raw=dict(calibration_state_id=cal_z,calibration_action_id=cal_a,calibration_label_noise=cal_noise,
                       calibration_scores=cal.scores,calibration_scale=cal.scale,calibration_target=cal.target,calibration_q=cal.q,
                       evaluation_state_id=z,evaluation_base_action=base,evaluation_mixture_uniform=mix,evaluation_label_noise=noise)
    for index,level in enumerate(protocol['action_law']['mixtures']):
        a=np.where(mix<level,engine.actor_action_index[z],base)
        scores=engine.score(z,a,noise)
        q=engine.target_probabilities(level)
        weight=q[cal_z,cal_a]/engine.beta[cal_z,cal_a]; mass=q[z,a]/engine.beta[z,a]
        weighted=calibrate_queries(cal.scores,weight,mass,**kwargs)
        uniform_eval=evaluate_queries(plain,scores.scores,scores.scale,unit=engine.unit,blend=.5,absolute_error=scores.absolute_error)
        weighted_eval=evaluate_queries(weighted,scores.scores,scores.scale,unit=engine.unit,blend=.5,absolute_error=scores.absolute_error)
        if level==0:
            if not (np.array_equal(plain.threshold,weighted.threshold) and np.array_equal(uniform_eval.dose,weighted_eval.dose)):
                raise AssertionError('lambda0 paired identity failed')
        query_id=array_hash(dict(state_id=z,action_id=a,label_noise=noise,scores=scores.scores,scale=scores.scale))
        row=dict(replication=rep,mixture=level,calibration_bank_id=bank_id,calibration_size=n,
            evaluation_query_id=query_id,evaluation_size=m,checkpoint_sha256=protocol['checkpoint_sha256'],
            frozen_model_update=100000,frozen_scale_update=95000,posterior_seed=posterior_seed,
            state_bank_sha256=sha256(HERE/'state_bank.npz'),off_grid_audit_sha256=sha256(HERE/'off_grid_support_audit.json'),
            calibration_weights=distribution_stats(weight),query_mass=distribution_stats(mass),
            n_eff=weighted.n_eff,positive_weight_count=weighted.positive_weights,no_positive_weight_bank=weighted.positive_weights==0,
            projected_actor_hits=int(np.sum(cal_a==engine.actor_action_index[cal_z])),
            lambda0_exact_identity=level==0,methods={},paired=paired_metrics(uniform_eval,weighted_eval))
        for method,fit,evaluation in [('uniform',plain,uniform_eval),('exact_categorical_iw_per_query',weighted,weighted_eval)]:
            metrics=dict(evaluation.summary,**counts(evaluation),calibration=fit.summary())
            metrics['mean_finite_full_width']=2*metrics['mean_finite_half_width'] if metrics['mean_finite_half_width'] is not None else None
            metrics['subgroups']={}
            for context in range(4):
                selected=engine.selected['context'][z]==context
                cal_selected=engine.selected['context'][cal_z]==context
                ww=(np.ones(n) if method=='uniform' else weight)[cal_selected]
                metrics['subgroups']['context_'+str(context)]=dict(**counts(evaluation,selected),
                    calibration_rows=int(cal_selected.sum()),calibration_positive_weights=int(np.count_nonzero(ww)),
                    calibration_ess=float(ww.sum()**2/np.dot(ww,ww)) if np.any(ww) else 0.,
                    structural_support='all625commands; empirical counts are not a subgroup certificate')
            for t in (0,50,99):
                metrics['subgroups']['time_'+str(t)]=counts(evaluation,engine.selected['time'][z]==t)
            row['methods'][method]=metrics
            raw[f'lambda{index}_{method}_threshold']=fit.threshold
            raw[f'lambda{index}_{method}_refusal']=fit.refused
        records.append(row)
        for key,values in dict(action_id=a,scores=scores.scores,scale=scores.scale,target=scores.target,q=scores.q,
                               calibration_weights=weight,query_mass=mass).items():raw[f'lambda{index}_{key}']=values
    rawpath=outdir/f'replicate_{rep:03d}.npz'
    pending=rawpath.with_suffix('.pending.npz');np.savez_compressed(pending,**raw);os.replace(pending,rawpath)
    record=dict(replication=rep,status='completed',freeze_sha256=sha256(HERE/'freeze.json'),completed_utc=utc(),seconds=time.perf_counter()-started,
                raw_file=rawpath.name,raw_sha256=sha256(rawpath),records=records)
    write_json(outdir/f'replicate_{rep:03d}.json',record)
    return record


def execute(args,protocol):
    frozen=verify_freeze(args); root=HERE/'results'; root.mkdir(exist_ok=True)
    lock=HERE/'execution.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Execution lock exists; verify its owner before any resume')
    os.write(fd,json.dumps(dict(pid=os.getpid(),started_utc=utc())).encode());os.close(fd)
    began=time.perf_counter();completed=[];failures=[]
    try:
        engine=load_engine(args,protocol);engine.prepare_catalog()
        native=load_native_wbcp(Path(args.bundle)/'source/calibration/wbcp.py')
        print(json.dumps(dict(event='started',pid=os.getpid(),utc=utc(),replications=200,
            checkpoint=protocol['checkpoint_sha256'],freeze_sha256=sha256(HERE/'freeze.json'))),flush=True)
        for rep in frozen['replication_ids']:
            file=root/f'replicate_{rep:03d}.json'
            if file.exists():
                old=json.loads(file.read_text());raw=root/old['raw_file']
                valid=(old['status']=='completed' and old['replication']==rep and
                    old['freeze_sha256']==sha256(HERE/'freeze.json') and old['raw_file']==f'replicate_{rep:03d}.npz' and
                    sha256(raw)==old['raw_sha256'] and len(old['records'])==3 and
                    [r['mixture'] for r in old['records']]==protocol['action_law']['mixtures'] and
                    all(r['replication']==rep and r['checkpoint_sha256']==protocol['checkpoint_sha256'] and
                        r['state_bank_sha256']==sha256(HERE/'state_bank.npz') and
                        r['calibration_size']==protocol['calibration']['size'] and
                        r['evaluation_size']==protocol['evaluation']['queries_per_replication_per_mixture'] for r in old['records']))
                if not valid:raise RuntimeError('Invalid existing replication '+str(rep))
                completed.append(rep);continue
            try:
                result=run_rep(engine,native,protocol,rep,root);completed.append(rep)
                print(json.dumps(dict(event='replication_completed',replication=rep,seconds=result['seconds'],
                    completed=len(completed),total=200)),flush=True)
            except Exception as error:
                import traceback
                failure=dict(replication=rep,status='failed',utc=utc(),error=repr(error),traceback=traceback.format_exc())
                write_json(root/f'replicate_{rep:03d}.failure.json',failure);failures.append(rep)
                print(json.dumps(dict(event='replication_failed',replication=rep,error=repr(error))),flush=True)
            write_json(HERE/'execution_status.json',dict(updated_utc=utc(),pid=os.getpid(),status='running',
                completed=completed,failed=failures,total=200,elapsed_seconds=time.perf_counter()-began))
        write_json(HERE/'execution_status.json',dict(updated_utc=utc(),pid=os.getpid(),status='completed' if not failures else 'completed_with_failures',
            completed=completed,failed=failures,total=200,elapsed_seconds=time.perf_counter()-began))
        print(json.dumps(dict(event='finished',completed=len(completed),failed=failures,seconds=time.perf_counter()-began)),flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','freeze','run'])
    parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    protocol=json.loads((HERE/'protocol.json').read_text())
    {'prepare':prepare,'freeze':freeze,'run':execute}[args.command](args,protocol)
