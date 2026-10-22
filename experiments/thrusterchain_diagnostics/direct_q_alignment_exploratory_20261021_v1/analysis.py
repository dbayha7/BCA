"""Exploratory analysis of saved records only. No simulator or model imports/calls."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
TOL = 1e-6
CONTEXT = ('stall_low', 'stall_high', 'monotone_low', 'monotone_high')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def summarize(x):
    x = np.asarray(x, float).ravel(); x = x[np.isfinite(x)]
    if not len(x): return {'n': 0, 'mean': None, 'min': None, 'median': None, 'max': None}
    return {'n': len(x), 'mean': float(x.mean()), 'min': float(x.min()),
            'median': float(np.median(x)), 'max': float(x.max())}

def corr(x, y):
    x,y = np.asarray(x,float).ravel(),np.asarray(y,float).ravel()
    good = np.isfinite(x)&np.isfinite(y); x,y=x[good],y[good]
    reason = None
    if len(x)<2: reason='insufficient_finite_pairs'
    elif np.ptp(x)==0 and np.ptp(y)==0: reason='constant_both'
    elif np.ptp(x)==0: reason='constant_width_or_predictor'
    elif np.ptp(y)==0: reason='constant_target'
    if reason:
        return dict(n=int(len(x)), pearson=None, spearman=None, undefined_reason=reason)
    return dict(n=int(len(x)), pearson=float(np.corrcoef(x,y)[0,1]),
                spearman=float(np.corrcoef(rankdata(x),rankdata(y))[0,1]), undefined_reason=None)

def centered(x):
    return x-x.mean(axis=1,keepdims=True)

def within_rank_corr(x,y):
    xr=np.array([rankdata(a)-np.mean(rankdata(a)) for a in x])
    yr=np.array([rankdata(a)-np.mean(rankdata(a)) for a in y])
    denom=np.linalg.norm(xr)*np.linalg.norm(yr)
    return None if denom==0 else float(np.sum(xr*yr)/denom)

def variance_partition(x):
    return dict(total=float(np.var(x)), between_state_means=float(np.var(x.mean(axis=1))),
                mean_within_state=float(np.mean(np.var(x,axis=1))),
                fraction_between_states=float(np.var(x.mean(axis=1))/np.var(x)) if np.var(x)>0 else None)

def mean_or_none(x):
    return float(np.mean(x)) if len(x) else None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE.parent/'direct_q_diagnostic_20261021_v2')
    parser.add_argument('--output',type=Path,default=HERE/'results_final')
    args=parser.parse_args(); source=args.input.resolve(); out=args.output.resolve()
    if out.exists(): raise SystemExit('Output exists; use --output with a new directory to preserve evidence.')
    inputs=['results/raw_results.npz','results/candidate_records.csv','results/within_state_correlations.csv',
            'results/fixture_results.json','prepared/engine_identity.json','prepared/coefficient_receipt.json',
            'prepared/preparation_manifest.json','protocol.json','diagnostic.py','run_diagnostic.py']
    hashes={name:sha(source/name) for name in inputs}
    with np.load(source/'results/raw_results.npz',allow_pickle=False) as a:
        raw={key:a[key] for key in a.files}
    identity=json.loads((source/'prepared/engine_identity.json').read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    q,truth,width,eta=(raw[k].astype(float) for k in ('q1','true_policy_q','width','eta'))
    n,k=q.shape
    if (n,k)!=(32,12): raise ValueError('Unexpected source bank dimensions')
    if not all(np.isfinite(x).all() for x in (q,truth,width,eta,raw['q2'])):
        raise ValueError('Nonfinite records require explicit expanded handling; refusing silent deletion')
    if not (raw['rollout_completed'].all() and raw['rollout_finite'].all()):
        raise ValueError('Incomplete saved oracle records')
    err=q-truth; absolute=abs(err); positive=np.maximum(err,0)
    np.testing.assert_array_equal(err,raw['signed_q1_error'])
    np.testing.assert_array_equal(positive,raw['positive_q1_overestimation'])
    qmin=np.minimum(q,raw['q2']); head_gap=q-qmin
    targets={'signed_error':err,'absolute_error':absolute,'positive_overestimation':positive,
             'signed_twin_min_error':qmin-truth,'absolute_twin_min_error':abs(qmin-truth),
             'positive_twin_min_overestimation':np.maximum(qmin-truth,0)}
    actor_distance=np.mean((raw['actions'].astype(float)-raw['actor_actions'][:,None,:])**2,axis=-1)
    behavior_distance=np.mean((raw['actions'].astype(float)-raw['behavior_mode'][:,None,:])**2,axis=-1)
    np.testing.assert_allclose(behavior_distance,raw['distance'],atol=1e-12,rtol=1e-12)
    distances={'behavior_mode_distance':behavior_distance,'actor_distance':actor_distance}
    groups={'all':np.ones(n,bool)}
    for t in np.unique(raw['time']): groups[f'time_{t}']=raw['time']==t
    for c in range(4):
        groups[CONTEXT[c]]=raw['context']==c
        for t in np.unique(raw['time']): groups[f'{CONTEXT[c]}_time_{t}']=(raw['context']==c)&(raw['time']==t)
    association_rows=[]
    for group,ix in groups.items():
        for name,y in {**targets,**distances}.items():
            for aggregation,xv,yv in (
                ('pooled_rows',width[ix],y[ix]),
                ('state_means',width[ix].mean(axis=1),y[ix].mean(axis=1)),
                ('state_centered',centered(width[ix]),centered(y[ix]))):
                association_rows.append(dict(group=group,states=int(ix.sum()),target=name,
                                             aggregation=aggregation,**corr(xv,yv)))
    state_rows=[]; correlation_rows=[]
    for i in range(n):
        row=dict(state_id=i,context=CONTEXT[int(raw['context'][i])],time=int(raw['time'][i]),
                 horizon_remaining=100-int(raw['time'][i]),candidates=k,
                 width_min=float(width[i].min()),width_max=float(width[i].max()),
                 width_range=float(np.ptp(width[i])),width_mean=float(width[i].mean()),
                 width_unique=int(len(np.unique(width[i]))),eta_min=float(eta[i].min()),eta_max=float(eta[i].max()),
                 eta_floor_count=int(np.sum(eta[i]<=1e-6)),width_refusal_count=int(raw['width_refusal'][i].sum()),
                 signed_error_mean=float(err[i].mean()),signed_error_min=float(err[i].min()),signed_error_max=float(err[i].max()),
                 signed_error_range=float(np.ptp(err[i])),absolute_error_mean=float(absolute[i].mean()),
                 overestimated_candidates=int(np.sum(err[i]>0)),all_nonpositive_errors=bool(np.all(err[i]<=0)),
                 signed_error_constant=bool(np.ptp(err[i])==0),width_constant=bool(np.ptp(width[i])==0),
                 action_boundary_candidates=int(np.any(abs(raw['actions'][i])>=1,axis=-1).sum()),
                 actor_at_boundary=bool(np.any(abs(raw['actor_actions'][i])>=1)))
        for name,y in targets.items():
            result=corr(width[i],y[i]); reason=result['undefined_reason']
            if reason=='constant_target' and name.startswith('positive'):
                reason='all_zero_positive_part' if np.all(y[i]==0) else 'constant_nonzero_positive_part'
            result['undefined_reason']=reason
            row[name+'_spearman']=result['spearman']
            correlation_rows.append(dict(state_id=i,context=row['context'],time=row['time'],
                target=name,**result))
        for name,y in distances.items(): row[name+'_spearman']=corr(width[i],y[i])['spearman']
        state_rows.append(row)
    counts={name:dict(Counter(r['undefined_reason'] or 'defined' for r in correlation_rows if r['target']==name)) for name in targets}
    original_corr=list(csv.DictReader((source/'results/within_state_correlations.csv').open(encoding='utf-8')))
    all_original=[r for r in original_corr if r['support']=='all']
    assert len(all_original)==n
    for r in all_original:
        i=int(r['state_id']); new=state_rows[i]['positive_overestimation_spearman']
        assert (r['spearman']=='')==(new is None)
        if new is not None: np.testing.assert_allclose(new,float(r['spearman']),rtol=1e-12,atol=1e-12)
    pair_rows=[]
    for i in range(n):
        for a in range(k):
            for b in range(a+1,k):
                du=float(width[i,a]-width[i,b]); de=float(err[i,a]-err[i,b])
                dq=float(q[i,a]-q[i,b]); dt=float(truth[i,a]-truth[i,b]); ds=dq-du
                sign=lambda z:0 if abs(z)<=TOL else int(np.sign(z))
                truth_sign,raw_sign,new_sign=map(sign,(dt,dq,ds))
                raw_loss=None if truth_sign==0 else (.5 if raw_sign==0 else float(raw_sign!=truth_sign))
                new_loss=None if truth_sign==0 else (.5 if new_sign==0 else float(new_sign!=truth_sign))
                pair_rows.append(dict(state_id=i,context=CONTEXT[int(raw['context'][i])],time=int(raw['time'][i]),
                    candidate_a=a,candidate_b=b,width_difference=du,advantage_error_difference=de,
                    raw_q_difference=dq,true_advantage=dt,penalized_score_difference=ds,
                    absolute_advantage_error_before=abs(de),absolute_advantage_error_after=abs(de-du),
                    width_direction_alignment=None if sign(de)==0 else (.5 if sign(du)==0 else float(sign(du)==sign(de))),
                    absolute_advantage_error_change=abs(de-du)-abs(de),
                    ordering_changed=raw_sign!=new_sign,truth_near_tie=truth_sign==0,
                    raw_ranking_loss=raw_loss,width_ranking_loss=new_loss))
    pair_summaries=[]
    for group,ix in groups.items():
        rows=[r for r in pair_rows if ix[r['state_id']]]
        du=np.array([r['width_difference'] for r in rows]);de=np.array([r['advantage_error_difference'] for r in rows])
        # Add both orientations so association is invariant to arbitrary candidate-pair ordering.
        pc=corr(np.r_[du,-du],np.r_[de,-de])
        valid=[r for r in rows if r['raw_ranking_loss'] is not None]
        per_state_losses=[]
        for state_id in np.flatnonzero(ix):
            sr=[r for r in valid if r['state_id']==state_id]
            if sr:
                per_state_losses.append((np.mean([r['raw_ranking_loss'] for r in sr]),
                                         np.mean([r['width_ranking_loss'] for r in sr])))
        changes=np.array([r['absolute_advantage_error_change'] for r in rows])
        pair_summaries.append(dict(group=group,states=int(ix.sum()),unordered_pairs=len(rows),
            signed_difference_pearson=pc['pearson'],signed_difference_spearman=pc['spearman'],
            oriented_entries_for_correlation=int(2*len(rows)),
            direction_alignment_mean=mean_or_none([r['width_direction_alignment'] for r in rows if r['width_direction_alignment'] is not None]),
            absolute_error_improved=int(np.sum(changes < -TOL)),absolute_error_worsened=int(np.sum(changes > TOL)),
            absolute_error_tied=int(np.sum(abs(changes)<=TOL)),
            mean_absolute_advantage_error_before=mean_or_none([r['absolute_advantage_error_before'] for r in rows]),
            mean_absolute_advantage_error_after=mean_or_none([r['absolute_advantage_error_after'] for r in rows]),
            raw_ranking_loss=mean_or_none([r['raw_ranking_loss'] for r in valid]),
            width_ranking_loss=mean_or_none([r['width_ranking_loss'] for r in valid]),
            comparable_unordered_pairs=len(valid),
            equal_state_mean_raw_concordance=mean_or_none([1-r[0] for r in per_state_losses]),
            equal_state_mean_width_concordance=mean_or_none([1-r[1] for r in per_state_losses]),
            truth_near_tie_pairs=len(rows)-len(valid),
            ordering_changes=int(sum(r['ordering_changed'] for r in rows)),
            improved_ranking_pairs=sum(r['width_ranking_loss']<r['raw_ranking_loss'] for r in valid),
            worsened_ranking_pairs=sum(r['width_ranking_loss']>r['raw_ranking_loss'] for r in valid)))
    actor_match=np.all(raw['actions']==raw['actor_actions'][:,None,:],axis=-1)
    assert actor_match[:,6].all(),'Saved original-actor slot mismatch'
    choices={name:np.argmax(raw['score_'+name],axis=1) for name in
             ('raw_q','live_width','matched_distance','constant_offset','detached_width')}
    choices['original_actor']=np.full(n,6)
    choices['best_saved_candidate']=np.argmax(truth,axis=1)
    choice_rows=[]; choice_summary=[]
    truth_best=truth.max(axis=1)
    raw_value=truth[np.arange(n),choices['raw_q']]; actor_value=truth[:,6]
    for name,selected in choices.items():
        values=truth[np.arange(n),selected]
        for i,j in enumerate(selected):
            choice_rows.append(dict(method=name,state_id=i,context=CONTEXT[int(raw['context'][i])],time=int(raw['time'][i]),
                candidate_id=int(j),original_actor_action=bool(actor_match[i,j]),support=str(raw['support_class'][i,j]),
                q1=float(q[i,j]),true_policy_q=float(truth[i,j]),width=float(width[i,j]),signed_error=float(err[i,j]),
                delta_vs_raw=float(values[i]-raw_value[i]),delta_vs_original_actor=float(values[i]-actor_value[i]),
                opportunity_loss_to_best_saved_candidate=float(truth_best[i]-values[i])))
        for group,ix in groups.items():
            delta=values[ix]-raw_value[ix]; da=values[ix]-actor_value[ix]
            choice_summary.append(dict(method=name,group=group,states=int(ix.sum()),
                mean_true_policy_q=float(values[ix].mean()),mean_delta_vs_raw=float(delta.mean()),
                better_than_raw=int(np.sum(delta>TOL)),worse_than_raw=int(np.sum(delta < -TOL)),tied_with_raw=int(np.sum(abs(delta)<=TOL)),
                mean_delta_vs_original_actor=float(da.mean()),better_than_actor=int(np.sum(da>TOL)),worse_than_actor=int(np.sum(da < -TOL)),
                actor_action_chosen=int(actor_match[np.flatnonzero(ix),selected[ix]].sum()),
                mean_opportunity_loss=float(np.mean(truth_best[ix]-values[ix]))))
    switches=[]
    for i in range(n):
        a,b=int(choices['raw_q'][i]),int(choices['live_width'][i])
        if a!=b:
            switches.append(dict(state_id=i,context=CONTEXT[int(raw['context'][i])],time=int(raw['time'][i]),
                raw_candidate=a,width_candidate=b,width_choice_is_actor=bool(actor_match[i,b]),
                q_gap_raw_minus_width=float(q[i,a]-q[i,b]),width_gap_raw_minus_width=float(width[i,a]-width[i,b]),
                advantage_error_gap=float(err[i,a]-err[i,b]),true_gain=float(truth[i,b]-truth[i,a]),
                signed_error_raw=float(err[i,a]),signed_error_width=float(err[i,b])))
    ref=identity['frozen_reference'];factor=float(ref['threshold']*ref['unit'])
    floor_eta=np.maximum(eta,1e-6)
    head_summary=[]
    for group,ix in groups.items():
        for name,pred in [('Q1',q),('Q2',raw['q2']),('twin_min',qmin)]:
            e=pred[ix]-truth[ix]
            head_summary.append(dict(group=group,head=name,states=int(ix.sum()),candidates=int(e.size),
                mean_signed_error=float(e.mean()),mean_absolute_error=float(abs(e).mean()),
                rmse=float(np.sqrt(np.mean(e**2))),overestimated_candidates=int(np.sum(e>TOL)),
                underestimated_candidates=int(np.sum(e < -TOL)),near_zero_candidates=int(np.sum(abs(e)<=TOL)),
                width_signed_spearman=corr(width[ix],e)['spearman'],width_abs_spearman=corr(width[ix],abs(e))['spearman']))
    context_time_rows=[]
    for group,ix in groups.items():
        context_time_rows.append(dict(group=group,states=int(ix.sum()),width_mean=float(width[ix].mean()),
            width_within_range_mean=float(np.ptp(width[ix],axis=1).mean()),
            error_mean=float(err[ix].mean()),absolute_error_mean=float(absolute[ix].mean()),
            positive_error_mean=float(positive[ix].mean()),positive_error_states=int(np.any(positive[ix]>0,axis=1).sum()),
            mean_q1_minus_twin_min=float(head_gap[ix].mean()),
            mean_width_behavior_distance_spearman=mean_or_none([state_rows[i]['behavior_mode_distance_spearman'] for i in np.flatnonzero(ix)])))
    summary=dict(schema='saved-record-value-error-alignment-exploratory-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        status='exploratory_already_inspected_bank_no_new_model_calls',states=n,candidates_per_state=k,rows=n*k,
        unique_training_models=1,statistical_interpretation='descriptive; no p-values or confidence intervals; pairs are dependent',
        choice_scope='Unrestricted full candidate bank only. Original actor is not inserted into restricted support-class comparisons.',
        near_tie_tolerance=TOL,all_widths_usable=bool(raw['width_usable'].all()),
        correlation_definition_counts=counts,
        error_counts=dict(strict_positive=int((err>0).sum()),strict_negative=int((err<0).sum()),exact_zero=int((err==0).sum())),
        pooled_associations=[r for r in association_rows if r['group']=='all'],
        within_state_spearman_summary={name:summarize([r[name+'_spearman'] for r in state_rows if r[name+'_spearman'] is not None]) for name in targets},
        within_state_centered_rank_similarity={name:within_rank_corr(width,y) for name,y in targets.items()},
        variance_partition={name:variance_partition(y) for name,y in {'width':width,**targets}.items()},
        width_structure=dict(reference=ref,global_factor=factor,width_range=summarize(np.ptp(width,axis=1)),
            width_over_eta=summarize(width/floor_eta),max_float64_formula_difference=float(np.max(abs(width-factor*floor_eta))),
            max_saved_native_formula_difference=float(np.max(abs(width-raw['width_formula']))),eta=summarize(eta),
            eta_floor_count=int((eta<=1e-6).sum()),constant_width_states=int(np.sum(np.ptp(width,axis=1)==0)),
            architectural_eta_floor=0.01,architectural_floor_count=int((eta<=float(np.float32(.01))).sum()),
            identical_width_eta_rank_states=int(sum(np.array_equal(rankdata(width[i]),rankdata(eta[i])) for i in range(n))),
            exact_actor_action_boundary_states=int(np.any(abs(raw['actor_actions'])>=1,axis=1).sum()),
            maximum_abs_original_actor_action=float(np.max(abs(raw['actor_actions']))),
            exact_candidate_action_boundary_rows=int(np.any(abs(raw['actions'])>=1,axis=-1).sum()),
            network_activation_or_jacobian_saturation='unavailable: saved outputs do not identify derivatives/activation states'),
        head_gap=dict(q1_exceeds_twin_min_rows=int(np.sum(head_gap>0)),gap=summarize(head_gap),
            q1_positive_to_min_nonpositive_rows=int(np.sum((err>0)&((qmin-truth)<=0)))),
        all_pair_summary=pair_summaries[0],selection_switches=switches,
        all_choice_summaries=[r for r in choice_summary if r['group']=='all'],
        lag_and_target=dict(checkpoint_update=protocol['checkpoint']['update'],reference_update=protocol['checkpoint']['reference_update'],
            reference_update_lag=protocol['checkpoint']['update']-protocol['checkpoint']['reference_update'],
            target_parameter_age='EMA target parameters saved at100k; no single exact age or matched counterfactual is identifiable here',
            native_label='noisy smoothed target-network one-step TD label and current twin-min prediction',
            oracle_label='one supplied action then original unsmoothed frozen actor to finite horizon',
            repeated_saved_one_step_noise_labels='unavailable',saved_successor_readouts_for_matched_lag_contrast='unavailable',
            epistemic_vs_representation_bias='not identifiable from one fitted critic/reference and these deterministic Q labels'),
        validation=dict(original_positive_error_correlations_reproduced=True,
            arithmetic_errors_match_saved=True,behavior_distances_match_saved=True,actor_slot_verified=True),
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        input_sha256=hashes,analysis_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.md'),
        unavailable=['reference95k Q/eta counterfactual on this bank','new calibration or learning trajectory',
                     'per-query stochastic target-noise variance','network activation/Jacobian saturation','epistemic uncertainty decomposition'])
    out.mkdir(parents=True)
    for filename,rows in [('state_diagnostics.csv',state_rows),('within_state_correlations.csv',correlation_rows),
        ('associations.csv',association_rows),('pairwise_alignment.csv',pair_rows),('pairwise_summaries.csv',pair_summaries),
        ('choice_records.csv',choice_rows),('choice_summaries.csv',choice_summary),
        ('head_comparison.csv',head_summary),('context_time_summary.csv',context_time_rows)]:
        write_csv(out/filename,rows)
    dump(out/'summary.json',summary)
    dump(out/'manifest.json',{'input_sha256':hashes,'analysis_sha256':sha(__file__),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
        'outputs':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},'new_model_calls':0,'new_simulator_transitions':0,'training_updates':0})
    print(json.dumps({k:summary[k] for k in ['states','rows','correlation_definition_counts','within_state_spearman_summary',
        'all_pair_summary','selection_switches','all_choice_summaries','head_gap']},indent=2,allow_nan=False))

if __name__=='__main__': main()
