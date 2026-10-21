"""Descriptive penalty/error alignment on available frozen Zeus query arrays.

No rollouts, fitting, selection, or statistical independence of queries assumed.
The missing four Titan full/BCA raw arrays are explicitly excluded, not imputed.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
CONTEXTS=['stall_low','stall_high','monotone_low','monotone_high']
NAMES=['zero','negative_knee','positive_knee','negative_one','positive_one','actor']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rho(x,y):
    x,y=np.asarray(x).ravel(),np.asarray(y).ravel()
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)
def ordering(x,y):
    i,j=np.triu_indices(6,1)
    dx=x[:,i]-x[:,j]; dy=y[:,i]-y[:,j]
    non=np.abs(dy)>1e-9
    return dict(comparisons=int(non.sum()), concordant=float(np.mean((dx[non]*dy[non])>0)) if non.any() else None,
                discordant=float(np.mean((dx[non]*dy[non])<0)) if non.any() else None,
                penalty_tied=float(np.mean(dx[non]==0)) if non.any() else None)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    jobs=json.loads((args.run_dir/'job_matrix.json').read_text());rows=[];cases=[];files=[];missing=[]
    for job in jobs:
        if job['arm']!='bca':continue
        path=args.run_dir/'evaluation/jobs'/job['job_id']/'final/per_start.npz'
        if not path.exists():missing.append(job['job_id']);continue
        z=np.load(path,allow_pickle=False)
        assert np.array_equal(z['q_actions'],z['calibration_actions'])
        assert np.array_equal(z['calibration_context'],np.repeat(z['q_context'][:,None],6,axis=1))
        truth=z['q_truth'];error=np.abs(z['q_predictions'].min(axis=2)-truth)
        loss=truth.max(axis=1,keepdims=True)-truth
        dose=z['calibration_dose'];width=z['calibration_width'];residual=z['calibration_absolute_residual']
        assert truth.shape==error.shape==loss.shape==dose.shape==width.shape==(48,6)
        assert all(np.isfinite(x).all() for x in (truth,error,loss,dose,width,residual))
        files.append(dict(path=str(path),sha256=sha(path)))
        for index,name in [(-1,'pooled')]+list(enumerate(CONTEXTS)):
            mask=np.ones(48,bool) if index<0 else z['q_context']==index
            rows.append(dict(job_id=job['job_id'],retention=job['retention'],replicate=job['replicate'],context=name,states=int(mask.sum()),
                dose_vs_abs_Q_error_spearman=rho(dose[mask],error[mask]),dose_vs_candidate_opportunity_loss_spearman=rho(dose[mask],loss[mask]),
                width_vs_native_residual_spearman=rho(width[mask],residual[mask]),
                within_state_error_ordering=ordering(dose[mask],error[mask]),within_state_opportunity_loss_ordering=ordering(dose[mask],loss[mask])))
        best=truth.argmax(axis=1);worst=truth.argmin(axis=1)
        for s in range(48):
            b,w=int(best[s]),int(worst[s]); gap=float(truth[s,b]-truth[s,w])
            if gap>1e-9 and dose[s,b]>dose[s,w]:
                cases.append(dict(job_id=job['job_id'],query_state_index=s,context=CONTEXTS[int(z['q_context'][s])],time=int(z['q_time'][s]),
                    best_candidate=NAMES[b],worst_candidate=NAMES[w],best_command=z['q_actions'][s,b].tolist(),worst_command=z['q_actions'][s,w].tolist(),
                    best_true_Q=float(truth[s,b]),worst_true_Q=float(truth[s,w]),candidate_opportunity_loss_gap=gap,
                    best_dose=float(dose[s,b]),worst_dose=float(dose[s,w]),best_abs_Q_error=float(error[s,b]),worst_abs_Q_error=float(error[s,w]),
                    description='Larger BC multiplier on the better candidate than on the worse candidate; descriptive counterexample to universal multiplier/harm ordering, not a failed uncertainty theorem or evidence about the full loss/gradient.'))
    assert len(files)==11 and len(missing)==4
    pooled=[r for r in rows if r['context']=='pooled']
    summary={}
    for retention in ('full','rare','zero'):
        rr=[r for r in pooled if r['retention']==retention]
        summary[retention]=dict(available_seeds=len(rr),planned_seeds=5,
            error_correlations=[r['dose_vs_abs_Q_error_spearman'] for r in rr],
            opportunity_loss_correlations=[r['dose_vs_candidate_opportunity_loss_spearman'] for r in rr],
            within_state_error_concordance=[r['within_state_error_ordering']['concordant'] for r in rr],
            within_state_loss_concordance=[r['within_state_opportunity_loss_ordering']['concordant'] for r in rr])
    report=dict(schema='thrusterchain-penalty-alignment-descriptive-v1',source='11 available Zeus BCA terminal raw arrays',
        missing_raw_jobs=missing,summary=summary,per_seed_context=rows,
        inversion_count=len(cases),total_states=11*48,reproduction_cases=cases,files=files,
        definitions=dict(dose='native bounded multiplier on actor behavior-cloning term',error='absolute twin-min Q minus deterministic same-policy rollout Q',opportunity_loss='best of six same-state true Q values minus candidate Q; not physical damage or full action-space regret',
                         correlation='Spearman rank correlation over designed query pairs; descriptive, no p-value or iid inference',within_state='pairwise concordance excluding true target ties; avoids between-state difficulty dominating this statistic'),
        limitations=['Only 1/5 full BCA raw runs available; do not extrapolate that seed to full condition.',
          'All five rare and zero BCA raw runs are available; each learned policy defines its own continuation Q.',
          'Residual scale is not trained to rank true Q error or harmful actions. A weak/negative association is a utility finding for that proposed use, not proof its formula is wrong.',
          'This measures the BC multiplier only. The actual BC loss also contains squared distance to the logged action, and its gradient depends on the actor; neither actual loss nor gradient ordering follows from multiplier ordering.',
          'Native BCA evaluates the multiplier at recorded actions before weighting each logged row cloning loss. These six-candidate values are diagnostic readouts of the frozen function, not actual training weights or direct penalties imposed on those candidate commands.',
          'The recorded dose at 100k is a diagnostic query evaluation, not the full history of training intervention values.',
          'Counterexamples enumerate all qualifying states in deterministic job/state order, without outcome-based omission.'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'penalty_alignment.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=['# Penalty alignment on frozen query pairs','', 'The multiplier is behavior-cloning regularization, not an estimated physical cost. These diagnostics test its proposed usefulness as an error/harm signal, not a mathematical guarantee. No training or rollout was added.','', '| Retention | Available BCA seeds | Dose vs absolute Q-error rank correlations | Dose vs candidate opportunity-loss rank correlations |','|---|---:|---|---|']
    for k,v in summary.items():text.append('| '+k+' | '+str(v['available_seeds'])+'/5 | '+', '.join(f'{x:.3f}' for x in v['error_correlations'])+' | '+', '.join(f'{x:.3f}' for x in v['opportunity_loss_correlations'])+' |')
    text+=['',f'Across {11*48} designed states, {len(cases)} have a larger BC multiplier on the best-Q candidate than on the worst-Q candidate. Every qualifying example is saved with seed, state index, time, commands, Q values and doses. This is a counterexample to an unconditional harm-ranking interpretation, not proof that uncertainty should always be smaller for good actions.','',*report['limitations'],'','Reproduce with `python analysis_penalty_alignment.py --run-dir <core_20261020T2250Z> --output <new-directory>`. The same command works on bundled raw evidence; no checkpoint bytes are needed.']
    (args.output/'PENALTY_ALIGNMENT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,inversions=len(cases),states=528),indent=2))
if __name__=='__main__': main()
