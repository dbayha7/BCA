"""Reserved fixed-checkpoint evaluator. Import performs no scientific work.

All full-state checkpoints are authenticated before model creation. Evaluation
restores static templates, never calls learner step, fit, refresh or match.
"""
from datetime import datetime, timezone
from pathlib import Path
import io
import json
import math
import sys
import time

import numpy as np

from common import HERE, read, sha, digest, atomic_bytes, atomic_json, module, require
import evaluation
import recovery_inputs


def checked(base, relative):
    base = Path(base).resolve(); rel = Path(relative); path = base / rel
    require(not rel.is_absolute() and '..' not in rel.parts and ':' not in str(relative)
            and not path.is_symlink() and path.resolve().is_relative_to(base)
            and path.is_file(), 'Unsafe or missing sealed artifact: '+str(relative))
    if path.resolve().is_relative_to(recovery_inputs.prior_root().resolve()):
        return recovery_inputs.authenticate_prior_file(
            path.resolve().relative_to(recovery_inputs.prior_root().resolve()).as_posix())
    return path.read_bytes()


def artifact(base, descriptor):
    require(set(descriptor) == {'path','sha256','bytes'}, 'Malformed artifact descriptor')
    raw = checked(base, descriptor['path'])
    require(digest(raw) == descriptor['sha256'] and len(raw) == descriptor['bytes'],
            'Artifact bytes/hash mismatch: '+descriptor['path'])
    return raw


def snapshot_plan(protocol):
    expected = {f'seed{s}_{c}' for s in range(3) for c in evaluation.CONDITIONS}
    require(len(protocol['cells']) == 9 and set(protocol['cells']) == expected, 'Nine distinct cells required')
    rows=[]
    for cell in protocol['cells']:
        seed, condition = int(cell[4]), cell[6:]
        for stage, step in [('warmup',0),('warmup',10000)]+[(a,t) for a in evaluation.ARMS for t in (15000,20000)]:
            rows.append({'cell':cell,'seed':seed,'condition':condition,'stage':stage,
                         'arm':'native' if stage=='warmup' else stage,'checkpoint':step,
                         'id':f'{cell}__{stage}__{step:05d}'})
    require(len(rows)==72 and sum(x['checkpoint']>0 for x in rows)==63, 'Snapshot arithmetic mismatch')
    return rows


def validate_protocol(p):
    fixed={'schema':'unifloral.central_quartile_training.v1','gamma':.98,'horizon':100,
           'bca_kappa':3.,'actor_distance_factor':19.29619572569095,
           'evaluation_policies':72,'q_evaluation_policies':63,'evaluation_episodes':4608,
           'simulator_transition_cap':1975304,'evaluation_actor_row_cap':1953792,
           'evaluation_critic_row_cap':40320,'evaluation_width_row_cap':40320,
           'error_tolerance':.1,'score_tie_tolerance':1e-6,'improvement_tolerance':.1,
           'minimum_high_load_anchor_families':8,'evaluation_timeout_seconds':7200,
           'learning_practical_effect':1.,'primary_accuracy_practical_effect':1.,
           'primary_policy_practical_effect':1.,'evaluation_bootstrap_draws':2000,
           'native_hole_return_harm_threshold':1.,'native_hole_mae_harm_threshold':1.}
    require(all(type(p.get(k)) is type(v) and p[k]==v for k,v in fixed.items()), 'Evaluation constants differ')
    cfg=p['evaluation']
    require(cfg['contexts']==list(evaluation.CONTEXTS)
            and cfg['anchor_times']==[0,25,50,75]*8
            and cfg['fixed_actions']==[[a,0,0,0] for a in evaluation.FIXED_A0]
            and cfg['anchors_per_context']==cfg['free_episodes_per_context']==32, 'Bank configuration differs')
    seeds=[]
    for context,law in [('ordinary','recorded'),('strong_current','strong_current')]:
        c=cfg[context]
        require(c['start_law']==law, 'Context reset law differs')
        for key in ('q_seeds','free_seeds'):
            require(len(c[key])==32 and all(type(s) is int for s in c[key]), 'Invalid seed list')
            seeds.extend(c[key])
        seeds.append(c['prefix_seed'])
    require(len(seeds)==len(set(seeds)), 'Evaluation seed collision')
    require(cfg['parity_seed']==2026102803, 'Pinned supplied-state parity seed differs')
    snapshot_plan(p)


def training_seal(directory, cell, stage, protocol_sha):
    directory=Path(directory)
    sr, rr=checked(directory,'SUPERVISOR_RESULT.json'),checked(directory,'report.json')
    supervisor, report=json.loads(sr),json.loads(rr)
    kind='warmup' if stage=='warmup' else 'branch'
    require(supervisor.get('job')==cell+'/'+stage and supervisor.get('completed') is True
            and supervisor.get('exit_code')==0 and supervisor.get('timed_out') is False
            and supervisor.get('interrupted') is False and supervisor.get('report_sha256')==digest(rr),
            'Incomplete supervisor or changed report: '+cell+'/'+stage)
    require(report.get('status')=='completed_central_quartile_'+kind
            and report.get('protocol_sha256')==protocol_sha and report.get('cell')==cell
            and report.get('condition')==cell[6:] and report.get('seed_index')==int(cell[4])
            and report.get('arm')==('native' if stage=='warmup' else stage)
            and report.get('execution_counts_known') is True, 'Training identity/counts differ')
    hashes={str(directory/'SUPERVISOR_RESULT.json'):digest(sr),str(directory/'report.json'):digest(rr)}
    mr=checked(directory,'matching_receipt.json'); matching=json.loads(mr)
    require(digest(mr)==report['matching_receipt_sha256'], 'Matching seal differs')
    require(matching.get('cell')==cell and matching.get('protocol_sha256')==protocol_sha
            and math.isfinite(matching['distance_coefficient']) and matching['distance_coefficient']>0,
            'Invalid fixed matching receipt')
    hashes[str(directory/'matching_receipt.json')]=digest(mr)
    if (cell,stage)==('seed2_hole','bca'):
        hashes.update(continuation_seal(directory,report,mr))
    else:
        require(report.get('continuation_lineage') is None, 'Unexpected continuation lineage')
    return report,matching,hashes


def continuation_seal(directory,report,matching_raw):
    """A completed new segment authenticates an old prefix, never an old success."""
    require(Path(directory).resolve()==(HERE/'run/seed2_hole/bca').resolve(),
            'Resumed branch must use its new job receipt')
    lineage=report.get('continuation_lineage') or {}
    source='run/seed2_hole/bca/checkpoints/step_16000/manifest.json'
    raw=recovery_inputs.authenticate_prior_file(source); manifest=json.loads(raw)
    require(manifest.get('cell')=='seed2_hole' and manifest.get('arm')=='bca'
            and manifest.get('protocol_sha256')==report['protocol_sha256']
            and manifest.get('absolute_critic_step')==16000
            and manifest.get('matching_receipt_sha256')==report['matching_receipt_sha256']
            and manifest.get('complete_optimizer_target_rng_state') is True,
            'Continuation source checkpoint identity differs')
    require(lineage.get('source_job')=='seed2_hole/bca' and lineage.get('resume_from_step')==16000
            and lineage.get('source_checkpoint')=={'path':source,'sha256':digest(raw)},
            'Missing or changed continuation source lineage')
    require(report.get('segment_initial_counts')==manifest['counts'],
            'Continuation segment initial counters differ')
    restore=report.get('continuation_restore_receipt',{})
    exact=restore.get('exact_state',{})
    require(restore.get('status')=='passed_exact_continuation_restore'
            and restore.get('source_checkpoint')==lineage['source_checkpoint']
            and restore.get('counts')==manifest['counts']
            and all(restore.get(k)==0 for k in ('optimizer_updates','calibration_updates','simulator_transitions'))
            and exact.get('comparison')=='exact_decoded_state'
            and exact.get('numeric_tolerances_used') is False
            and exact.get('checkpoint_sha256')==manifest['artifact']['sha256'],
            'Continuation exact runtime restore receipt differs')
    logical=dict(critic=10000,actor=5000,actor_target=5000,critic_target=5000,outer_steps=5000,
                 auxiliary=10000,aux_fit_attempts=10000,refresh_attempts=1,refresh_accepted=1)
    physical=dict(critic=4000,actor=2000,actor_target=2000,critic_target=2000,outer_steps=2000,
                  auxiliary=4000,aux_fit_attempts=4000,refresh_attempts=0,refresh_accepted=0)
    require(report.get('additional_updates')==logical
            and report.get('segment_additional_updates')==physical
            and report.get('ledger',{}).get('committed_additional_updates')==physical
            and report.get('logical_branch_counts_known') is True
            and report.get('prior_physical_execution_counts_known') is False,
            'Continuation logical/physical update accounting differs')
    for initial,expected in [('initial_counts',logical),('segment_initial_counts',physical)]:
        require(all(report['final_counts'][k]-report[initial][k]==v for k,v in expected.items()),
                'Continuation initial/final counters differ')
    require(report['final_counts']['critic']==report['final_counts']['auxiliary']==20000
            and report['final_counts']['actor']==10000 and report['final_counts']['phase']==0,
            'Continuation final checkpoint boundary differs')
    old_matching='run/seed2_hole/bca/matching_receipt.json'
    require(matching_raw==recovery_inputs.authenticate_prior_file(old_matching),
            'Continuation fixed matching receipt changed')
    return {str(recovery_inputs.prior_root()/source):digest(raw),
            str(recovery_inputs.prior_root()/old_matching):digest(matching_raw)}


def checkpoint_export(directory, report, spec, protocol_sha):
    descriptor=report['checkpoint_manifests'][str(spec['checkpoint'])]
    raw=artifact(directory,descriptor); manifest=json.loads(raw)
    base=Path(directory)/Path(descriptor['path']).parent
    expected={'schema':'unifloral.central_quartile.checkpoint.v1','protocol_sha256':protocol_sha,
              'cell':spec['cell'],'condition':spec['condition'],'seed_index':spec['seed'],
              'kind':'warmup' if spec['stage']=='warmup' else 'branch','arm':spec['arm'],
              'absolute_critic_step':spec['checkpoint'],'complete_optimizer_target_rng_state':True,
              'resume_authorized':False}
    require(all(manifest.get(k)==v for k,v in expected.items()), 'Checkpoint identity differs')
    n=spec['checkpoint']; counts=manifest['counts']; refresh=0 if n==0 else 1 if n==10000 else 2
    require(counts['critic']==counts['auxiliary']==n and counts['actor']==n//2
            and counts['phase']==0 and counts['refresh_accepted']==refresh,
            'Checkpoint counters/calibration differ')
    require(manifest.get('matching_receipt_sha256')==(None if n==0 else report['matching_receipt_sha256']),
            'Checkpoint matching receipt binding differs')
    state=artifact(base,manifest['artifact']); actor=artifact(base,manifest['actor_params'])
    hashes={str(Path(directory)/descriptor['path']):digest(raw),
            str(base/manifest['artifact']['path']):digest(state),
            str(base/manifest['actor_params']['path']):digest(actor)}
    if (spec['cell'],spec['stage'])==('seed2_hole','bca'):
        source=json.loads(recovery_inputs.authenticate_prior_file(
            'run/seed2_hole/bca/checkpoints/step_16000/manifest.json'))
        require(manifest['normalization']==source['normalization'],
                'Continuation normalization differs from restored checkpoint')
        if spec['checkpoint']==20000:
            require(manifest['counts']==report['final_counts'],
                    'Continuation report/final checkpoint counters differ')
    if (spec['cell'],spec['stage'],spec['checkpoint'])==('seed2_hole','bca',15000):
        prefix='run/seed2_hole/bca/checkpoints/step_15000/'
        require(descriptor['path']=='checkpoints/step_15000/manifest.json',
                'Imported checkpoint location differs')
        for name,value in [('manifest.json',raw),(manifest['artifact']['path'],state),
                           (manifest['actor_params']['path'],actor)]:
            source=prefix+name
            require(value==recovery_inputs.authenticate_prior_file(source),
                    'Imported 15k checkpoint differs from authenticated prefix: '+name)
            hashes[str(recovery_inputs.prior_root()/source)]=digest(value)
    return {'spec':spec,'manifest':manifest,'base':base,'state_descriptor':manifest['artifact'],
            'actor_descriptor':manifest['actor_params']},hashes


def assemble_exports(protocol,jobs_root,protocol_sha):
    jobs_root=Path(jobs_root).resolve()
    require(jobs_root==(HERE/'run').resolve(), 'Undeclared training root')
    exports, jobs, hashes={}, {}, {}
    for spec in snapshot_plan(protocol):
        key=(spec['cell'],spec['stage']); directory=recovery_inputs.job_directory(*key)
        if key not in jobs:
            report,matching,sealed=training_seal(directory,*key,protocol_sha)
            jobs[key]=(report,matching); hashes.update(sealed)
        export,sealed=checkpoint_export(directory,jobs[key][0],spec,protocol_sha)
        export['matching']=jobs[key][1]; export['training_report']=jobs[key][0]
        exports[spec['id']]=export; hashes.update(sealed)
    normalizations=[x['manifest']['normalization'] for x in exports.values()]
    require(all(x==normalizations[0] for x in normalizations), 'Common normalization differs')
    for seed in range(3):
        initial=[jobs[(f'seed{seed}_{c}','warmup')][0]['initial_parameter_hashes'] for c in evaluation.CONDITIONS]
        require(all(x==initial[0] for x in initial), 'Paired initialization differs across conditions')
    require(len({jobs[(f'seed{s}_full','warmup')][0]['initial_parameter_hashes']['actor'] for s in range(3)})==3,
            'Independent actor initializations identical')
    return exports,hashes


def restore_snapshot(runtime,adapter,template,export):
    require(adapter.context.normalization==export['manifest']['normalization'], 'Runtime normalization differs')
    require(adapter.context.initial_parameter_hashes==export['training_report']['initial_parameter_hashes'],
            'Static template initialization differs')
    state=runtime.restore_exact(adapter,template,artifact(export['base'],export['state_descriptor']))
    from state_exact import assert_exact_state
    assert_exact_state(artifact(export['base'],export['actor_descriptor']),
                       state.microstep.agent_state.actor.params, label='Full-state actor')
    require(runtime.validate(adapter,state)['counts']==export['manifest']['counts'], 'Restored counts differ')
    return state


class FrozenPolicy:
    """Actual pinned native apply functions and original calibrated width."""
    def __init__(self,runtime,adapter,state):
        self.runtime,self.adapter,self.state=runtime,adapter,state
        self.before=runtime.tree_digest(state)
        context=adapter.context; self.jnp=context.native.jnp; jax=context.native.jax
        actor=state.microstep.agent_state.actor; critic=state.microstep.agent_state.dual_q
        self._act=jax.jit(lambda obs:actor.apply_fn(actor.params,obs).sample(seed=None))
        self._q=jax.jit(lambda obs,actions:critic.apply_fn(critic.params,obs,actions))
        self._width=jax.jit(lambda obs,actions:context.pipeline.live_width(context.models,state.aux.posterior,obs,actions))

    def act(self,observations):
        value=np.asarray(self._act(self.jnp.asarray(observations,self.jnp.float32)))
        require(value.shape==(len(observations),4) and value.dtype==np.float32
                and np.isfinite(value).all() and (np.abs(value)<=1).all(), 'Invalid deployed native action')
        return value

    def q(self,observations,actions):
        return np.asarray(self._q(self.jnp.asarray(observations,self.jnp.float32),self.jnp.asarray(actions,self.jnp.float32)))

    def width(self,observations,actions):
        width,usable,valid=self._width(self.jnp.asarray(observations,self.jnp.float32),self.jnp.asarray(actions,self.jnp.float32))
        width=np.asarray(width)
        require(bool(np.all(usable)) and bool(np.all(valid)) and width.shape==(len(observations),)
                and np.isfinite(width).all() and (width>=0).all(), 'Calibrated width invalid; no fallback')
        return width

    def assert_unchanged(self):
        require(self.runtime.tree_digest(self.state)==self.before, 'Evaluation mutated complete model state')


def calibrated_score(policy,ledger,progress):
    def score(obs,actions,q1):
        n=len(obs)
        require(ledger['width_rows_attempted']+n<=ledger['max_width_rows'], 'Width row cap exceeded')
        ledger['width_rows_attempted']+=n; progress(ledger)
        width=policy.width(obs,actions)
        ledger['width_rows_completed']+=n
        return np.asarray(q1,np.float64)-3.*np.asarray(width,np.float64)
    return score


def load_simulator():
    prior=module('_quartile_prior_core_loader',HERE.parent/'matched_policy_pilot_v1/evaluate_worker.py')
    core,receipt=prior.load_core()
    gate=HERE.parent/'conditional_ood_gate_central_quartile_v1'
    geometry=module('_quartile_passed_geometry',gate/'geometry.py')
    previous=sys.modules.get('geometry')
    try:
        sys.modules['geometry']=geometry
        reference=module('_quartile_passed_reference',gate/'reference.py')
    finally:
        if previous is None: sys.modules.pop('geometry',None)
        else: sys.modules['geometry']=previous
    receipt['reference_authentication']=reference.authenticate_core(core)
    receipt['reference_sha256']=sha(gate/'reference.py')
    return core,reference,receipt


def paired_summary(values,protocol):
    matrix=np.asarray(values,np.float64)
    require(matrix.shape==(3,32) and np.isfinite(matrix).all(), 'Paired seed/start matrix differs')
    rng=np.random.Generator(np.random.PCG64(protocol['evaluation_bootstrap_seed']))
    seed_ix=rng.integers(0,3,(protocol['evaluation_bootstrap_draws'],3))
    start_ix=rng.integers(0,32,(protocol['evaluation_bootstrap_draws'],32))
    sampled=matrix[seed_ix[:,:,None],start_ix[:,None,:]].mean(axis=(1,2))
    return {'seed_means':matrix.mean(axis=1).tolist(),'mean':float(matrix.mean()),
            'paired_values_by_seed':matrix.tolist(),
            'hierarchical_seed_start_descriptive_95pct':np.quantile(sampled,[.025,.975]).tolist(),
            'uncertainty':'Three seed blocks and shared start families; coarse descriptive interval, no significance claim'}


def common_reference_panels(results):
    """Cached fixed-action score deviations against full/native/20k pi, per seed.

    This reference was fixed by design, not selected by return. Every actor may
    estimate a different own-policy Q, so this panel is not an accuracy claim
    about that actor's intended value function. Slot0 is excluded entirely.
    """
    references={(r['seed'],r['context']):r for r in results
        if r['condition']=='full' and r['arm']=='native' and r['checkpoint']==20000}
    require(len(references)==6, 'Six complete common-reference policies required')
    output={}
    for row in results:
        if row['checkpoint']==0: continue
        ref=references[(row['seed'],row['context'])]
        require(row['bank_id']==ref['bank_id'], 'Common-reference state/time bank differs')
        source=row['q']['per_probe']; target=ref['q']['per_probe']
        actions=np.asarray(source['actions'],np.float32)
        require(actions.shape==(32,6,4) and np.array_equal(actions[:,1:],np.asarray(target['actions'],np.float32)[:,1:]),
                'Common-reference complete fixed actions differ')
        high=np.asarray(source['high_load'],bool)
        require(np.array_equal(high,np.asarray(target['high_load'],bool)), 'Common-reference strata differ')
        values=np.asarray(target['qpi'],np.float64)[:,1:]
        raw=np.asarray(source['q1'],np.float64)[:,1:]
        adjusted=None if source['adjusted'] is None else np.asarray(source['adjusted'],np.float64)[:,1:]
        band=np.zeros((32,5),bool);band[:,1:4]=True
        masks={'all_fixed':np.ones((32,5),bool),'high_band_fixed':band & high[:,None],
               'low_band_fixed':band & ~high[:,None],'high_retained_flanks':~band & high[:,None],
               'low_retained_flanks':~band & ~high[:,None]}
        panels={}
        for name,m in masks.items():
            panels[name]={'raw_score_deviation':evaluation.error_metrics(raw[m],values[m]),
                'state_count':int(np.any(m,axis=1).sum())}
            if adjusted is not None: panels[name].update(
                adjusted_score_deviation=evaluation.error_metrics(adjusted[m],values[m]),
                same_critic_adjustment=evaluation.paired_error_change(raw[m],adjusted[m],values[m]))
        key=f"seed{row['seed']}_{row['condition']}__{row['arm']}__{row['checkpoint']}__{row['context']}"
        output[key]={'reference_policy':f"seed{row['seed']}_full/native/20000",'bank_id':row['bank_id'],
            'panels':panels,'common_qpi_fixed_actions':values.tolist(),
            'own_minus_common_qpi_fixed_actions':(np.asarray(source['qpi'])[:,1:]-values).tolist(),
            'scope':'Score deviation from a fixed full-data native reference policy, not each critic own-policy accuracy',
            'additional_simulator_transitions':0,'own_actor_slot_excluded':True}
    require(len(output)==126,'All trained snapshot/context common-reference panels required')
    return output


def summarize(results,protocol):
    table={(r['seed'],r['condition'],r['arm'],r['checkpoint'],r['context']):r for r in results}
    require(len(results)==len(table)==144, 'All snapshot/context results required')
    def get(seed,condition,arm,step,context): return table[(seed,condition,arm,step,context)]
    def gains(condition,control,step,context):
        return np.asarray([np.asarray(get(s,condition,'bca',step,context)['free']['returns'])-
                           np.asarray(get(s,condition,control,step,context)['free']['returns']) for s in range(3)])
    comparisons={}; interactions={}; native_harm={}
    for context in evaluation.CONTEXTS:
        for step in (15000,20000):
            native_harm[f'{context}/{step}/native_hole_minus_random']=paired_summary(np.asarray([
                np.asarray(get(s,'hole','native',step,context)['free']['returns'])-
                np.asarray(get(s,'random','native',step,context)['free']['returns']) for s in range(3)]),protocol)
            for control in ('native','distance'):
                for condition in evaluation.CONDITIONS:
                    comparisons[f'{context}/{step}/{condition}/bca_minus_{control}']=paired_summary(gains(condition,control,step,context),protocol)
                interactions[f'{context}/{step}/bca_minus_{control}/hole_minus_random']=paired_summary(
                    gains('hole',control,step,context)-gains('random',control,step,context),protocol)
    learning=paired_summary(np.asarray([np.asarray(get(s,'full','native',20000,'ordinary')['free']['returns'])-
        np.asarray(get(s,'full','native',0,'ordinary')['free']['returns']) for s in range(3)]),protocol)
    accuracy={}; accuracy_interactions={}; condition_counts={}; native_mae_harm={}
    for context in evaluation.CONTEXTS:
        effects={}
        for condition in evaluation.CONDITIONS:
            rows=[]
            for s in range(3):
                n=get(s,condition,'native',20000,context)['q']['panels']['high_band_fixed_primary']
                b=get(s,condition,'bca',20000,context)['q']['panels']['high_band_fixed_primary']
                condition_counts[context]=n['state_count']
                rows.append(None if n['raw']['mae'] is None or b['adjusted']['mae'] is None else n['raw']['mae']-b['adjusted']['mae'])
            effects[condition]=rows
        accuracy[context]=effects
        accuracy_interactions[context]=[None if effects['hole'][s] is None or effects['random'][s] is None
                                         else effects['hole'][s]-effects['random'][s] for s in range(3)]
        native_mae_harm[context]=[]
        for s in range(3):
            h=get(s,'hole','native',20000,context)['q']['panels']['high_band_fixed_primary']['raw']['mae']
            r=get(s,'random','native',20000,context)['q']['panels']['high_band_fixed_primary']['raw']['mae']
            native_mae_harm[context].append(None if h is None or r is None else h-r)
    learn=all(x>=protocol['learning_practical_effect'] for x in learning['seed_means'])
    policy=all(x>=protocol['primary_policy_practical_effect'] for x in comparisons['ordinary/20000/hole/bca_minus_native']['seed_means']) and all(
        x>=0 for x in comparisons['ordinary/20000/hole/bca_minus_distance']['seed_means'])
    support=condition_counts['ordinary']>=protocol['minimum_high_load_anchor_families']
    accuracy_pass=support and all(x is not None and x>=protocol['primary_accuracy_practical_effect'] for x in accuracy['ordinary']['hole'])
    harmed_return=all(x<=-protocol['native_hole_return_harm_threshold'] for x in native_harm['ordinary/20000/native_hole_minus_random']['seed_means'])
    harmed_mae=support and all(x is not None and x>=protocol['native_hole_mae_harm_threshold'] for x in native_mae_harm['ordinary'])
    positive_return_interaction=all(x>0 for x in interactions['ordinary/20000/bca_minus_native/hole_minus_random']['seed_means'])
    positive_accuracy_interaction=support and all(x is not None and x>0 for x in accuracy_interactions['ordinary'])
    return {'decision':'completed_fixed_development_study_no_automatic_successor',
            'learning_adequacy_passed':learn,'policy_screen_passed':learn and policy,
            'accuracy_screen_passed':learn and accuracy_pass,'conditional_accuracy_testable':support,
            'learning':learning,'policy_comparisons':comparisons,'return_interactions':interactions,
            'accuracy_native_q1_minus_bca_adjusted_mae':accuracy,'accuracy_hole_minus_random':accuracy_interactions,
            'native_hole_minus_random_return':native_harm,'native_hole_minus_random_mae':native_mae_harm,
            'native_hole_return_harm_each_seed':harmed_return,'native_hole_mae_harm_each_seed':harmed_mae,
            'policy_rescue_screen_passed':learn and harmed_return and policy and positive_return_interaction,
            'accuracy_rescue_screen_passed':learn and harmed_mae and accuracy_pass and positive_accuracy_interaction,
            'high_load_anchor_families':condition_counts,
            'missing_label_policy_interaction_positive_each_seed':positive_return_interaction,
            'missing_label_accuracy_interaction_positive_each_seed':positive_accuracy_interaction,
            'interpretation':'Separate policy/accuracy and stricter rescue screens. Rescue requires measured native Hole-vs-Random harm; positive BCA gains alone do not establish repair. Different trained policies have different Qpi. No confirmation or Bayesian-specific claim.'}


def run(protocol,output_dir,job):
    import launch
    launch.require_execution_authorization(protocol)
    validate_protocol(protocol)
    require(read(HERE/'protocol.json')==protocol,'Protocol differs from saved file')
    output=Path(output_dir).resolve()
    require(output==(HERE/'run/evaluation').resolve() and output.is_dir() and not Path(output_dir).is_symlink(),
            'Reserved evaluation output required')
    require(not any((output/n).exists() for n in ('progress.json','report.json','EVALUATION_ATTEMPT.json')),
            'Existing evaluation evidence; no retry')
    with (output/'EVALUATION_ATTEMPT.json').open('x') as stream:
        json.dump({'automatic_retry':False,'protocol_sha256':sha(HERE/'protocol.json')},stream)
    started=time.monotonic(); cap=min(protocol['evaluation_timeout_seconds'],job.get('timeout_seconds',0))
    deadline=min(started+cap,job.get('deadline_monotonic',0))
    require(math.isfinite(deadline) and deadline>started and cap>0,'Active evaluation supervisor deadline required')
    ledger={'optimizer_updates':0,'calibration_updates':0,'source_decode_attempts':0,
            'source_episodes_decoded':0,'source_transition_rows_decoded':0,'protected_transition_rows_decoded':0}
    phase='authenticate_exports'; results=[]; files={}; stage_id=None; protocol_sha=sha(HERE/'protocol.json')
    def progress(_=None):
        atomic_json(output/'progress.json',{'status':'running','phase':phase,'snapshot':stage_id,
            'completed_snapshot_contexts':len(results),'ledger':ledger,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.monotonic()-started})
    def check_deadline(): require(time.monotonic()<deadline,'Evaluation deadline reached')
    try:
        progress(); check_deadline()
        exports,input_hashes=assemble_exports(protocol,job['jobs_root'],protocol_sha)
        require(sha(HERE/'DATA_PLAN.json')==protocol['data_plan_sha256'],'Data plan differs')
        core,reference,core_receipt=load_simulator()
        ledger.update(reference.new_ledger(protocol['simulator_transition_cap'],protocol['evaluation_actor_row_cap']))
        ledger.update(max_resets=129,max_critic_rows=protocol['evaluation_critic_row_cap'],
                      max_width_rows=protocol['evaluation_width_row_cap'],width_rows_attempted=0,width_rows_completed=0)
        banks={}; phase='shared_banks'
        for context in evaluation.CONTEXTS:
            c=protocol['evaluation'][context]
            bank=evaluation.build_context_bank(core,reference,context=context,q_seeds=c['q_seeds'],
                free_seeds=c['free_seeds'],prefix_seed=c['prefix_seed'],start_law=core.canonical.START_LAWS[c['start_law']],
                ledger=ledger,deadline=deadline,progress=progress)
            require(bank['times'].tolist()==protocol['evaluation']['anchor_times'],'Anchor ordering differs')
            arrays={k:v for k,v in bank.items() if isinstance(v,np.ndarray)}
            buf=io.BytesIO();np.savez(buf,**arrays)
            bank_file=atomic_bytes(output/(context+'_bank.npz'),buf.getvalue())
            meta={k:v for k,v in bank.items() if not isinstance(v,np.ndarray)}
            files[context+'_bank']=atomic_json(output/(context+'_bank.json'),{**meta,'arrays':bank_file})
            banks[context]=bank
            if context=='ordinary':
                parity=reference.runtime_parity(core,bank['states'][0],ledger,deadline=deadline,progress=progress)
        runtime=module('_quartile_runtime',HERE/'runtime.py'); phase='snapshot_inference'; runtime_receipts={}
        for cell in protocol['cells']:
            check_deadline(); seed,condition=int(cell[4]),cell[6:]
            ledger['source_decode_attempts']+=1;progress()
            adapter,template=runtime.create(protocol,seed,condition)
            ledger['source_episodes_decoded']+=896;ledger['source_transition_rows_decoded']+=89600
            require(adapter.context.data.receipt['protected_transition_rows_decoded']==0,'Protected data decoded')
            runtime_receipts[cell]=runtime.host(adapter.context.runtime)
            warm_export=exports[f'{cell}__warmup__10000']
            warm=restore_snapshot(runtime,adapter,template,warm_export); matching=warm_export['matching']
            require(runtime.tree_digest(warm.aux.posterior)==matching['reference_digest'] and
                    runtime.tree_digest(warm.microstep.agent_state.actor.params)==matching['anchor_actor_digest'],
                    'Frozen reference/anchor differs from matching receipt')
            anchor=FrozenPolicy(runtime,adapter,warm)
            for spec in (x for x in snapshot_plan(protocol) if x['cell']==cell):
                check_deadline();stage_id=spec['id']; export=exports[stage_id]
                require(export['matching']==matching,'Branch rematched its distance coefficient')
                aa,tt=(adapter,template) if spec['stage']=='warmup' else runtime.branch(adapter,warm,protocol,seed,spec['arm'],matching['distance_coefficient'])
                state=restore_snapshot(runtime,aa,tt,export); policy=FrozenPolicy(runtime,aa,state)
                for context,bank in banks.items():
                    progress();check_deadline()
                    result=evaluation.evaluate_snapshot(core,reference,policy,bank,ledger,trained=spec['checkpoint']>0,
                        adjusted_score=calibrated_score(policy,ledger,progress) if spec['checkpoint']>0 else None,
                        deadline=deadline,progress=progress)
                    result.update({k:spec[k] for k in ('cell','seed','condition','arm','checkpoint')})
                    if spec['stage']=='distance':
                        anchor_actions=reference.actor_actions(anchor,bank['observations'],ledger,deadline=deadline,progress=progress)
                        q=result['q']['per_probe']; actions=np.asarray(q['actions']);values=np.asarray(q['qpi']);raw=np.asarray(q['q1'])
                        distance=protocol['actor_distance_factor']*np.mean((actions-anchor_actions[:,None,:])**2,axis=-1)
                        score=raw-matching['distance_coefficient']*distance
                        result['distance_q']=evaluation.q_panels(bank['observations'],actions,raw,values,score)
                        result['distance_coefficient']=matching['distance_coefficient'];anchor.assert_unchanged()
                    result['complete_state_unchanged']=True;policy.assert_unchanged()
                    files[stage_id+'/'+context]=atomic_json(output/(stage_id+'__'+context+'.json'),runtime.host(result))
                    results.append(result)
            anchor.assert_unchanged()
        phase='final_accounting';check_deadline()
        require(ledger['transition_steps_completed']==ledger['transition_steps_attempted']==1975304
                and ledger['resets_completed']==ledger['resets_attempted']==129
                and ledger['actor_rows_completed']==ledger['actor_rows_attempted']==1953792
                and ledger['critic_rows_completed']==ledger['critic_rows_attempted']==40320
                and ledger['width_rows_completed']==ledger['width_rows_attempted']==40320
                and ledger['source_episodes_decoded']==8064 and ledger['source_transition_rows_decoded']==806400,
                'Final evaluation accounting differs')
        for path,expected in input_hashes.items():require(sha(path)==expected,'Sealed input changed: '+path)
        require(sha(HERE/'protocol.json')==protocol_sha and sha(HERE/'DATA_PLAN.json')==protocol['data_plan_sha256'],
                'Protocol/data plan changed during evaluation')
        reference.authenticate_core(core)
        decision=summarize(results,protocol)
        common_reference=common_reference_panels(results)
        files['common_reference']=atomic_json(output/'COMMON_REFERENCE.json',common_reference)
        report={'status':'completed_central_quartile_evaluation','protocol_sha256':protocol_sha,
            'data_plan_sha256':protocol['data_plan_sha256'],**decision,'ledger':ledger,'execution_counts_known':True,
            'snapshot_results':files,'sealed_input_hashes':input_hashes,'core':core_receipt,'parity':parity,
            'runtime_receipts':runtime_receipts,'elapsed_seconds':time.monotonic()-started,'error':None,
            'automatic_retry':False,'full_training_hold':True,
            'data_read_scope':'Nine static template initializations each decode the same768 fitting+128 development calibration episodes; repeated reads counted, no updates'}
        atomic_json(output/'report.json',report);phase=report['status'];progress();return report
    except BaseException as exc:
        pairs=[('transition_steps_attempted','transition_steps_completed'),('actor_rows_attempted','actor_rows_completed'),
               ('critic_rows_attempted','critic_rows_completed'),('width_rows_attempted','width_rows_completed'),
               ('resets_attempted','resets_completed')]
        known=all(ledger.get(a,0)==ledger.get(b,0) for a,b in pairs) and ledger['source_decode_attempts']*896==ledger['source_episodes_decoded']
        report={'status':'failed_central_quartile_evaluation','protocol_sha256':protocol_sha,'phase':phase,
                'error':type(exc).__name__+': '+str(exc),'ledger':ledger,'execution_counts_known':known,
                'completed_snapshot_contexts':len(results),'snapshot_results':files,'decision':'incomplete_no_utility_conclusion',
                'elapsed_seconds':time.monotonic()-started,'automatic_retry':False,'full_training_hold':True}
        atomic_json(output/'report.json',report);phase=report['status'];progress();return report
