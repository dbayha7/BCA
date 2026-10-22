"""Read-only report construction after the independently frozen V2 run completes.

No imports of diagnostic/model code, no new labels, no optimizer or simulator calls.
Writes only inside this report_assets directory.
"""
from __future__ import annotations
import argparse
import ast
import collections
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
METHODS=('raw_q','live_width','matched_distance','constant_offset','detached_width')
LABELS={'raw_q':'Raw Q1','live_width':'Q1 minus width','matched_distance':'Matched constant-coefficient distance',
        'constant_offset':'State-constant width offset','detached_width':'Detached width (same forward score)'}
CONTEXTS=('stall_low','stall_high','monotone_low','monotone_high')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def esc(value):return html.escape(str(value),quote=True)
def fmt(value,digits=4):return 'unavailable' if value is None else f'{value:.{digits}f}'
def ci(value):return 'unavailable' if value is None else f'[{value[0]:.4f}, {value[1]:.4f}]'
def table(headers,data):
    return '<div class="dq-table"><table><thead><tr>'+''.join('<th scope="col">'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table></div>'


def source(path,function):
    text=Path(path).read_text(encoding='utf-8');tree=ast.parse(text)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==function)
    return dict(path=str(path),file=Path(path).name,function=function,first_line=node.lineno,last_line=node.end_lineno,
        sha256=sha(path),text='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno]))


def sources():
    mapping={}
    for key,file,func in [('deployment','diagnostic.py','single_row_policy'),('state-law','diagnostic.py','generate_states'),
        ('candidates','diagnostic.py','build_candidates'),('support','diagnostic.py','classify_actions'),
        ('width','diagnostic.py','read_width_and_q'),('matching','diagnostic.py','matching_coefficient'),
        ('scores','diagnostic.py','score_arrays'),('oracle','diagnostic.py','truth_rollouts'),
        ('fixtures','diagnostic.py','validate_truth'),('ranking','diagnostic.py','analyze'),
        ('bootstrap','diagnostic.py','bootstrap_interval'),('freeze','run_diagnostic.py','authenticate_preparation')]:
        mapping[key]=source(ROOT/file,func)
    mechanical=ROOT.parent/'direct_q_plan_20261021_v1/mechanics/objective.py'
    tree=ast.parse(mechanical.read_text(encoding='utf-8'))
    functions=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)]
    mapping['mechanical-objective']=source(mechanical,functions[-1])
    return mapping


def verify():
    status=read(ROOT/'results/status.json')
    if status.get('status')!='complete':raise RuntimeError('Main run is not complete; no partial-result report is produced')
    for filename,digest in status['output_sha256'].items():
        if sha(ROOT/'results'/filename)!=digest:raise ValueError('Recorded output hash mismatch: '+filename)
    if status['completed_main_returns']!=384 or status['main_transitions']!=21696 or status['fixture_transitions']!=608:
        raise ValueError('Budget accounting differs from the frozen protocol')
    if status['parameters_before']!=status['parameters_after']:raise ValueError('Frozen parameters changed')
    freeze=read(ROOT/'freeze.json');manifest=read(ROOT/'prepared/preparation_manifest.json')
    if freeze['preparation_manifest_sha256']!=sha(ROOT/'prepared/preparation_manifest.json'):
        raise ValueError('Preparation/freeze identity mismatch')
    for filename,digest in freeze['source_sha256'].items():
        if sha(ROOT/filename)!=digest:raise ValueError('Frozen source mismatch: '+filename)
    analysis=read(ROOT/'results/paired_results.json')
    candidates=rows(ROOT/'results/candidate_records.csv');decisions=rows(ROOT/'results/paired_decisions.csv')
    if len(candidates)!=384:raise ValueError('Not all candidate rows are present')
    if len({(int(r['state_id']),int(r['candidate_id'])) for r in candidates})!=384:raise ValueError('Duplicated candidate identity')
    if any(not math.isfinite(float(r['true_policy_q'])) for r in candidates):raise ValueError('Nonfinite candidate label')
    return status,manifest,analysis,candidates,decisions


def group(analysis,name='pooled',support='all'):
    return next(x for x in analysis['groups'] if x['group']==name and x['support']==support)


def summaries(analysis,decisions):
    result=[]
    for method in METHODS:
        m=group(analysis)['methods'][method]
        records=[r for r in decisions if r['group']=='pooled' and r['support']=='all' and r['method']==method]
        result.append(dict(method=method,label=LABELS[method],**m,
            changed_candidate_choices=sum(r['chosen_candidate']!=r['raw_candidate'] for r in records),
            states_in_decision_export=len(records)))
    return result


def correlation_summary(analysis):
    out={}
    for support in ('all','common_on_grid','rare_on_grid','unsupported'):
        rr=[r for r in analysis['within_state_width_positive_error_spearman'] if r['support']==support]
        vals=[r['spearman'] for r in rr if r['spearman'] is not None]
        out[support]=dict(states=len(rr),defined=len(vals),undefined=len(rr)-len(vals),
            mean=statistics.mean(vals) if vals else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,
            positive=sum(v>0 for v in vals),negative=sum(v<0 for v in vals),zero=sum(v==0 for v in vals))
    return out


def saved_error_comparison():
    """Descriptive score errors from saved arrays; no new simulator/model calls."""
    import numpy as np
    with np.load(ROOT/'results/raw_results.npz',allow_pickle=False) as data:
        truth=data['true_policy_q'];raw=data['q1'];penalized=data['score_live_width'];classes=data['support_class']
        result=[]
        for support in ('all','common_on_grid','rare_on_grid','unsupported'):
            mask=np.ones(truth.shape,bool) if support=='all' else classes==support
            for name,score in [('raw_q',raw),('live_width',penalized)]:
                error=(score-truth)[mask]
                result.append(dict(support=support,method=name,rows=len(error),mean_signed_error=float(error.mean()),
                    mean_positive_overestimation=float(np.maximum(error,0).mean()),
                    mean_underestimation_magnitude=float(np.maximum(-error,0).mean()),
                    rmse=float(np.sqrt(np.mean(error**2))),positive_error_rows=int(np.sum(error>0))))
        if np.any(penalized>raw):raise ValueError('nonnegative-width subtraction unexpectedly increased a score')
    return result


def diagnostic_cases(candidates,decisions):
    index={(int(r['state_id']),int(r['candidate_id'])):r for r in candidates}
    result=[]
    for row in decisions:
        if row['group']=='pooled' and row['method']=='live_width' and float(row['delta_vs_raw'])!=0:
            state=int(row['state_id']);selected=index[state,int(row['chosen_candidate'])];raw=index[state,int(row['raw_candidate'])]
            result.append(dict(support=row['support'],state_id=state,context=selected['context'],time=int(selected['time']),
                delta_vs_raw=float(row['delta_vs_raw']),chosen_candidate=int(row['chosen_candidate']),raw_candidate=int(row['raw_candidate']),
                chosen_is_original_actor=int(row['chosen_candidate'])==6,chosen_true_q=float(selected['true_policy_q']),raw_true_q=float(raw['true_policy_q']),
                chosen_width=float(selected['width']),raw_width=float(raw['width']),
                chosen_action=[float(selected['action_'+str(k)]) for k in range(4)],raw_action=[float(raw['action_'+str(k)]) for k in range(4)]))
    return result


def effect_plot(analysis):
    entries=[('All candidates · width',group(analysis)['methods']['live_width'],'#167d73'),
             ('All candidates · distance',group(analysis)['methods']['matched_distance'],'#3f648d'),
             ('Common grid only · width',group(analysis,support='common_on_grid')['methods']['live_width'],'#a35649'),
             ('Unsupported only · width',group(analysis,support='unsupported')['methods']['live_width'],'#7762a3')]
    lo,hi=-.1,.9;x0,x1=325,940
    def x(value):return x0+(value-lo)/(hi-lo)*(x1-x0)
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 410" role="img" aria-labelledby="dq-plot-title dq-plot-desc">',
        '<title id="dq-plot-title">Sparse selection gains do not establish adaptive benefit</title>',
        '<desc id="dq-plot-desc">Mean selected true-Q changes against raw Q. Intervals are paired state bootstrap intervals. The common-grid interval is unavailable because some eligible strata contain one state.</desc>',
        '<rect width="1000" height="410" fill="#fbfaf7"/><g font-family="Arial,sans-serif" fill="#182d36">',
        '<text x="24" y="32" font-size="23" font-weight="bold">The width changes one full-bank choice</text>',
        '<text x="24" y="59" font-size="15">Mean true-Q change versus raw Q; one action, then the original actor</text>',
        f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="82" y2="326" stroke="#536571" stroke-dasharray="4 4"/>']
    for i,(label,m,color) in enumerate(entries):
        y=100+62*i;mean=m['mean_delta_vs_raw'];bounds=m['paired_state_bootstrap_ci95']
        out.append(f'<text x="24" y="{y+5}" font-size="15">{esc(label)}</text>')
        if bounds is not None:
            out.append(f'<line x1="{x(bounds[0]):.2f}" x2="{x(bounds[1]):.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        out.append(f'<circle cx="{x(mean):.2f}" cy="{y}" r="6" fill="{color}"/>')
        text=f"{mean:+.4f}; {m['improved_states']} better / {m['harmed_states']} worse / {m['unchanged_states']} unchanged"
        if bounds is None:text+=' · interval unavailable'
        out.append(f'<text x="325" y="{y+25}" font-size="13">{esc(text)}</text>')
    for value in (0,.2,.4,.6,.8):
        out.append(f'<text x="{x(value):.2f}" y="348" font-size="13" text-anchor="middle">{value:.1f}</text>')
    out.extend(['<text x="325" y="374" font-size="13">Distance comparison is descriptive; no direct superiority test against width was planned.</text>',
                '<text x="325" y="395" font-size="13">Conditional evidence from 32 states, tiny strata and one trained checkpoint.</text></g></svg>'])
    return '\n'.join(out)


def main_table(summary,markup=False):
    data=[]
    for m in summary:
        if m['status']!='evaluated':
            data.append([m['label'],m['status']]+['unavailable']*6);continue
        data.append([m['label'],fmt(m['mean_delta_vs_raw']),ci(m['paired_state_bootstrap_ci95']),
            f"{m['improved_states']} / {m['harmed_states']} / {m['unchanged_states']}",
            str(m['changed_candidate_choices']),fmt(m['mean_opportunity_loss']),
            fmt(m['mean_advantage_vs_original_actor']),fmt(m['mean_ranking_concordance'])])
    return data


def markdown(status,analysis,summary,correlations,error_comparison,cases):
    live=next(x for x in summary if x['method']=='live_width')
    lines=['# One choice improves; adaptive uncertainty benefit remains unproven','',
        f"Completed 384/384 candidate continuations on 32 designed states: 21,696 main plant transitions and 608 validation transitions. New training updates: {status['new_training_updates']}.",'',
        f"For Q1 minus width versus raw Q1, the mean selected-action true-Q difference is **{fmt(live.get('mean_delta_vs_raw'))}**, "
        f"with the predeclared paired state-bootstrap interval **{ci(live.get('paired_state_bootstrap_ci95'))}**. "
        f"There are {live.get('improved_states')} improved, {live.get('harmed_states')} harmed and {live.get('unchanged_states')} unchanged state-level values.",'',
        'These are one-action deviations followed by the original actor. They are not returns of a new policy that repeatedly chooses Q-minus-width actions. No actor or critic was updated.', '',
        'The sole full-bank improvement is state 6 (stall/low load, time 0): width changes candidate 2 to candidate 6, the original actor action, gaining 4.922153. This restores the original actor choice rather than demonstrating a new learned-policy improvement. '
        'The distance control has a larger descriptive mean gain (0.405389), but no direct distance-versus-width superiority test was planned.', '',
        'The common-grid restriction exposes a negative case: state 18 (monotone/low load, time 99) loses 0.3. Across its 13 eligible states, width changes no state for the better and one for the worse; the mean is −0.023077 and its interval is unavailable because some strata contain only one eligible state. '
        'On unsupported commands, one selection improves while mean pairwise concordance worsens from 0.65625 to 0.65.', '',
        '## Placement: what was previously tested, and what changes here','',
        'Native BCA multiplies the behavior-cloning loss by a detached width-derived dose. That changes the strength of copying logged actions; it does not directly subtract width from the actor-action Q1 score. '
        'The prior mechanical stage passed 18/18 tests and showed the proposed live width has the intended action derivative. Its actual-checkpoint audit found a nonzero live-minus-host actor-gradient norm, while a detached additive width reproduced the host gradient. '
        'This proves a mechanical path, not that the width ranks harmful actions correctly. The present diagnostic checks that separate ranking question without gradients or learning.', '',
        '## Paired candidate decisions','',
        '| Score | Mean true-Q change vs raw Q1 | Paired 95% interval | Improved / harmed / unchanged | Candidate choice changed | Mean candidate opportunity loss | Mean advantage vs original actor | Mean pairwise concordance |',
        '|---|---|---|---|---|---|---|---|']
    lines += ['| '+' | '.join(r)+' |' for r in main_table(summary)]
    lines += ['', 'Opportunity loss is the best true Q among this state’s eligible fixed candidates minus the selected candidate’s true Q. It is not regret to a continuous-action optimum. '
        'Improvement/harm counts use the saved exact signed differences. Pairwise ranking ignores true near-ties within 1e−6 and gives predicted near-ties half credit. '
        'The original actor action is candidate 6 in every state.', '',
        'The detached-width and live-width forward scores are identical. Their identical rankings are an algebraic control, not evidence that detaching leaves actor optimization unchanged. '
        'A state-constant subtraction preserves raw-Q ranking. The distance comparator uses one coefficient matched to within-state score dispersion on eight adaptation states before any oracle labels; it does not match gradient strength.', '',
        '## Source → target law and score target','',
        '- Historical band source: 128 pooled rows, with one uniformly selected transition from each reserved behavior episode. Under full retention, time is uniform over 0..99 and physical states follow the behavior trajectories. Commands follow the 625-action product categorical behavior law: each coordinate assigns probability 0.84 to its mode and 0.04 to each other grid value. The calibration rows were adaptively reused, with a reference last refreshed at 95k. The calibrated quantity was a noisy one-step target residual for the current twin-min critic.',
        '- New evaluator law: 32 independent fresh physical-state draws, balanced across four actuator-regime/load contexts; 8 states per context with times 0/50/99 represented 3/3/2. All ten position/velocity coordinates are uniform on [−0.5,0.5], current is 0.3 or 1.1, and wave amplitude is 0.2 with uniform phase.',
        '- Each state has six distinct categorical grid actions, then the original continuous actor and five uniform continuous commands. These are designed candidates, not samples of policy occupancy or the historical calibration law.',
        '- Scoring uses terminal100k Q1 and the historical frozen width. Truth takes the candidate once, then follows the unchanged unsmoothed actor through horizon 100 with discount 1. The target and distribution both differ from the band’s calibration target.',
        '- No importance-ratio correction, Q-coverage certificate or support guarantee is inferred from a finite width. This is the simple Q-minus-width candidate, not Option C.', '',
        '## Support, refusal and error alignment','',
        f"Candidate counts: {json.dumps(analysis['support_counts'],sort_keys=True)}. Native width refusals: {analysis['width_refusals']}/384. "
        f"Failure events: {status['failure_events']}/384. All continuations remain in the saved records.", '',
        'Common means categorical mass at least 0.01; rare means positive mass below 0.01; unsupported means zero categorical mass. Classification uses actual float32 atom equality, not the candidate slot. Conditional action support does not establish state or calibration support.', '',
        '| Candidate class | Defined width/positive-error correlations | Mean within-state Spearman | Positive / negative / zero |',
        '|---|---|---|---|']
    for support,c in correlations.items():
        lines.append(f"| {support} | {c['defined']}/{c['states']} ({c['undefined']} undefined) | {fmt(c['mean'])} | {c['positive']} / {c['negative']} / {c['zero']} |")
    lines += ['', 'These correlations compare width with max(Q1−true Q,0), within a state. Constant or insufficient ranks remain undefined. Only 10/32 full-bank correlations are defined ; 2 are positive and 8 negative (mean −0.154882). The other 22 are not discarded or assigned zero. A correlation does not establish calibration or policy benefit.', '',
        '## Lower overestimation can conceal worse underestimation','',
        '| Score, all 384 candidates | Mean signed error | Mean positive overestimation | Mean underestimation magnitude | RMSE | Positive-error rows |',
        '|---|---|---|---|---|---|']
    for e in error_comparison:
        if e['support']=='all':lines.append('| '+' | '.join([LABELS[e['method']],fmt(e['mean_signed_error']),fmt(e['mean_positive_overestimation']),fmt(e['mean_underestimation_magnitude']),fmt(e['rmse']),f"{e['positive_error_rows']}/384"] )+' |')
    lines += ['', 'Subtracting a nonnegative width necessarily cannot increase positive overestimation. That arithmetic fact is not evidence of calibrated uncertainty. The raw critic mostly underestimates on this bank (only 61/384 positive errors); the penalized score can deepen those underestimates and worsen RMSE. These descriptive score errors use saved labels only and are not an interval-coverage test.', '',
        '## Numerical deployment correction and failure accounting','',
        'V1 stopped before the main bank because the same observation produced batch-size-dependent native float32 actor outputs. A 2.38e−7 action difference amplified through feedback to a 0.01839038 return discrepancy. '
        'V1 used 304 fixture transitions; its preserved failure trace added 304. These failed checks remain evidence.',
        'V2 calls the unchanged native actor on exactly one observation for candidate generation, every continuation and replay. No parameter, precision, scientific seed, budget, coefficient rule or tolerance was changed. '
        'This is a specified numerical deployment convention, not a claim of bitwise equivalence to V1’s batch-dependent policy. V2 records completed fixture phases immediately.', '',
        f"V2 numerical fixture results: `{json.dumps(analysis['numerical_validation'],sort_keys=True)}`.",'',
        '## Uncertainty and negative findings that remain in scope','',
        '- The independent inference unit is the state, not each action. The fixed 10,000 bootstrap draws resample states within context×time strata and keep all actions at a sampled state together.',
        '- Full-bank strata contain only 2–3 states. Support-class analyses rerank within the class and require at least two candidates per state. Intervals are refused if any nonempty eligible stratum contains fewer than two states; absent strata are not invented.',
        '- Intervals are conditional on one trained checkpoint, the designed state law and the fixed candidate realization. They do not cover variation across training seeds. Deterministic numerical labels have no Monte Carlo rollout uncertainty; replay tolerance is still a numerical limitation.',
        '- Finite native width usability is not Q coverage. The width was calibrated to a different one-step residual target, and the historical bank was adaptively reused.',
        '- The matched-distance control matches score dispersion, not gradient strength. This comparison cannot attribute a later learning benefit to adaptive uncertainty.',
        '- No actor update, critic-update isolation experiment, learned-policy benefit, full policy rollout under the new ranking rule, or Option C test occurred.', '',
        'See `report_summary.json` for every pooled and subgroup result, undefined-correlation count, source hash and numerical receipt. All original candidate and decision rows remain in the untouched results folder.', '',
        'Reproduce this report and SVG from saved records only: `python report_assets/build_report.py`. It requires NumPy to read the saved NPZ, validates output/freeze hashes, and performs no model inference or extra rollout.', '']
    return '\n'.join(lines)


def fragment(status,analysis,summary,correlations,refs,error_comparison,cases,plot):
    def link(key):
        r=refs[key];return f'<a href="#dq-code-{key}">{r["file"]}:{r["first_line"]}–{r["last_line"]}</a>'
    live=next(x for x in summary if x['method']=='live_width')
    coefficient=analysis['coefficient_receipt']['result']
    stages=[
        ('D1','Freeze the experiment','One 100k model, a 95k residual-band reference, seeds, candidate law and unchanged parameters.','Hash authentication before oracle execution.',link('freeze')),
        ('D2','Define the numerical policy','Use unchanged native actor inference on one observation at a time.','Replay must agree at atol 1e−4, rtol 1e−6; terminal 99 equals immediate reward.',link('deployment')+'; '+link('fixtures')),
        ('D3','Create independent evaluator inputs','32 new designed states and 12 candidates each; separate 8-state adaptation bank.','No oracle labels for coefficient matching or input selection.',link('state-law')+'; '+link('candidates')+'; '+link('matching')),
        ('D4','Export the actual width','Read Q1 and native width U=R×unit×max(eta, floor).','Check formula/native agreement and retain native usability/refusals.',link('width')),
        ('D5','Compare score placement','Raw Q1; Q1−U; matched distance; state-constant offset; detached U.','Same forward scores for detached/live U; state-constant offset keeps raw ranking.',link('scores')),
        ('D6','Measure the one-action consequence','Execute candidate once, then the original fixed actor until time 99.','All 384 deterministic continuations retained; no learned-policy claim.',link('oracle')),
        ('D7','Pair evidence at the state level','Compare selected true Q, harm, opportunity loss and pairwise ranking.','Bootstrap states within context×time; refuse underpopulated subgroup intervals.',link('ranking')+'; '+link('bootstrap'))]
    out=[f'''<section id="direct-q-ranking" aria-labelledby="dq-heading">
<h2 id="dq-heading">One choice improves; adaptive uncertainty benefit remains unproven</h2>
<p>The earlier native BCA comparison strengthened <b>behavior cloning</b>—the pull toward recorded commands. The proposed direct penalty instead lowers a candidate action’s value score: <b>Q1 − uncertainty width</b>. Those are different intervention locations.</p>
<div class="callout"><b>Completed frozen diagnostic:</b> 32 fresh designed states × 12 candidates = 384 one-action continuations. No actor or critic was trained or updated. The mean selected-action true-Q change for Q1 minus width versus raw Q1 is <b>{fmt(live.get('mean_delta_vs_raw'))}</b>, with paired state-bootstrap 95% interval <b>{ci(live.get('paired_state_bootstrap_ci95'))}</b>.</div>
<figure class="dq-figure">{plot}<figcaption>The distance control has a larger descriptive mean gain. No direct superiority test against width was planned, so this does not establish that one method is generally better.</figcaption></figure>
<h3 id="dq-mechanics">Mechanical correctness and useful uncertainty are separate tests</h3>
<p>The preceding mechanical stage passed 18/18 deterministic tests. It showed that recomputing the same frozen width while retaining its action derivative changes the actor gradient. The native exported width is detached; adding that detached value to Q cannot supply the intended uncertainty derivative. Native BCA still changes optimization because its detached multiplier multiplies an action-dependent cloning loss.</p>
<pre class="dq-pseudo"><span><b>Native BCA</b> actor loss = −c × Q1(actor action) + detached dose(logged action) × cloning distance</span>
<span><b>Proposed direct path</b> actor loss = −c × [Q1(actor action) − live width(actor action)] + cloning distance</span>
<span><b>This diagnostic</b> select one of 12 fixed candidates by score; execute it once; then follow the original actor</span></pre>
<p>The first stage establishes a gradient path. The new ranking diagnostic tests whether that width helps identify better actions. Neither stage alone establishes learned-policy benefit. This is the simple Q-minus-width candidate; Option C was not run.</p>
<h3 id="dq-laws">The historical calibration source and new query target differ</h3>
{table(['Role','State/time and action law','Quantity'],[
    ['Historical band source','128 pooled rows: one uniformly selected transition per reserved behavior episode. Full retention gives time uniform over 0..99 and behavior-trajectory physical states. The 625-command product categorical law assigns each coordinate 0.84 to its mode and 0.04 to each other grid value. Rows adaptively reused; frozen 95k reference.','Noisy one-step target residual using the current twin-min critic.'],
    ['New evaluator bank','32 fresh uniform physical-state draws, 8 per regime/load context; times 0/50/99 appear 3/3/2 per context. Six distinct grid candidates, the original actor, and five uniform continuous commands.','Terminal 100k Q1 and its frozen-width score at each candidate.'],
    ['Deterministic oracle','Candidate action once, then the original unsmoothed actor using single-row native inference through horizon 100, discount 1.','Finite-horizon Q of a one-action deviation; no Qstar approximation or repeatedly applied new controller.'],
    ['Coefficient adaptation','Eight separate states, two per context at times0/50 ; 12 candidates per state. No return labels.','Match pooled within-state centered RMS of width and squared distance to behavior mode.']])}
<p>The fresh positions/velocities are uniform on [−0.5,0.5]; load is 0.3 or 1.1; wave amplitude is 0.2 with uniform phase. These are designed states at assigned times, not policy occupancy samples. A late-time row does not mean that physical state arose from a rollout.</p>
<p><b>No Q-coverage claim follows.</b> The historical one-step residual target, the current Q1 score and the deterministic long-return label differ. Finite width or conditional grid-action support does not certify state/calibration support.</p>
<h3 id="dq-stages">Trace the implemented test</h3>
<pre class="dq-pseudo"><span><b>01 · D1</b> Authenticate fixed checkpoint, reference, inputs, protocol and source hashes.</span>
<span><b>02 · D2</b> Call the unchanged actor with one observation; pass fixed numerical replay checks.</span>
<span><b>03 · D3</b> Use the frozen 32-state, 12-candidate bank and outcome-free distance coefficient.</span>
<span><b>04 · D4</b> Read Q1 and native U; record every unusable width.</span>
<span><b>05 · D5</b> Compare Q1, Q1−U, Q1−kD, Q1−state_mean(U), and detached Q1−U.</span>
<span><b>06 · D6</b> Measure each candidate once, then continue with the original actor.</span>
<span><b>07 · D7</b> Pair selected true-Q differences by state; retain harmful and unavailable results.</span></pre>
{table(['Stage','Purpose','Inputs / operation','Invariant / limit','Actual source'],stages)}
<h3 id="dq-decisions">Do the selected actions improve?</h3>
{table(['Score','Mean true-Q change vs raw','Paired95% interval','Improved / harmed / unchanged','Changed choices','Candidate opportunity loss','Advantage vs original actor','Pairwise concordance'],main_table(summary))}
<p>Opportunity loss is relative to the best of the same eligible candidates, not a continuous-action optimum. Each score chooses once and then hands control back to the original actor. The detached and live widths have the same forward score and therefore the same ranking here; their different gradients matter only in a separate optimizer experiment. The state-constant offset is a ranking control.</p>
<p>The distance coefficient is <b>{fmt(coefficient.get('coefficient'),8)}</b>, status <b>{esc(coefficient['status'])}</b>. It matches variation in scores before observing returns. It does not match the strength of actor gradients.</p>
<div class="callout"><b>The only full-bank improvement returns to the original actor.</b> At state 6 (stall/low load, time 0), the width changes candidate 2 to candidate 6—the original actor action—and gains 4.922153. The other 31 full-bank choices are unchanged. This is not evidence of a newly learned policy improving.</div>
<p><b>Negative case retained:</b> restricting the comparison to common grid commands harms state 18 (monotone/low load, time 99) by 0.3. Across 13 eligible states the mean change is −0.023077, with no improved states and one harmed state. Its interval is refused because some eligible strata have only one state. For unsupported commands, one chosen action improves, yet mean ranking concordance declines from 0.65625 to 0.65.</p>
<h3 id="dq-support">Support, errors and refused conclusions remain visible</h3>
<p>Candidate counts: <code>{esc(json.dumps(analysis['support_counts'],sort_keys=True))}</code>. Native width refusals: <b>{analysis['width_refusals']}/384</b>. Saved failure events: <b>{status['failure_events']}/384</b>. Failure means an absolute physical position exceeds 3 at a pre/post state, or an incomplete/nonfinite continuation; threshold events do not truncate the rollout.</p>
<p>Common grid actions have categorical mass at least 0.01; rare grid actions have positive mass below 0.01; unsupported commands have mass 0. Classification uses actual float32 equality to the action atoms. A continuous candidate slot is not automatically unsupported, and an on-grid action does not establish joint state/calibration overlap.</p>
{table(['Candidate class','Defined within-state correlations','Mean width / positive-error Spearman','Positive / negative / zero'],[[esc(k),f"{v['defined']}/{v['states']} ({v['undefined']} undefined)",fmt(v['mean']),f"{v['positive']} / {v['negative']} / {v['zero']}"] for k,v in correlations.items()])}
<p>Positive error is max(Q1−true Q,0). Undefined correlations from constant ranks or too few actions are retained. These are descriptive error-alignment checks, not confidence-band coverage or causal evidence of policy improvement.</p>
<p>Only 10 of 32 full-bank width/error correlations are defined: 2 positive and 8 negative, with mean −0.154882. The other 22 are undefined, including states with constant zero positive error. They are not evidence of successful alignment.</p>
<h3 id="dq-error-tradeoff">Lower overestimation can accompany worse overall error</h3>
{table(['Score, all 384 candidates','Mean signed error','Positive overestimation, mean','Underestimation magnitude, mean','RMSE','Positive-error rows'],[[esc(LABELS[e['method']]),fmt(e['mean_signed_error']),fmt(e['mean_positive_overestimation']),fmt(e['mean_underestimation_magnitude']),fmt(e['rmse']),f"{e['positive_error_rows']}/384"] for e in error_comparison if e['support']=='all'])}
<p>Subtracting a nonnegative width mechanically cannot increase positive overestimation. The raw critic mostly underestimates on this bank; only 61 of 384 errors are positive. A smaller positive-error metric alone therefore cannot validate the penalty, especially when subtraction deepens underestimation or increases RMSE. The table compares scores against saved deterministic labels; it is not a Q-coverage claim.</p>
<h3 id="dq-numerics">A failed replay check caused a documented numerical repair</h3>
<p>V1 stopped before labeling the main bank. Native float32 actor inference changed by 2.38×10⁻⁷ when the same observation was evaluated in a batch of two rather than one. Feedback amplified that into a 0.01839038 return discrepancy. The failed execution used 304 fixture transitions; the separate failure trace added 304. Both records are preserved.</p>
<p>V2 fixes one-row actor inference for every diagnostic path. It changes neither learned parameters nor precision, seeds, scientific budgets, coefficient rule or tolerances. It defines a reproducible numerical deployment, without claiming bitwise equivalence to V1’s batch-dependent policy. The 608-transition V2 fixtures passed with batch/scalar maximum discrepancy <b>{analysis['numerical_validation']['batch_scalar_max_error']:.2e}</b>, policy-action Q=V discrepancy <b>{analysis['numerical_validation']['policy_action_q_equals_v_max_error']:.2e}</b>, and terminal immediate-reward discrepancy <b>{analysis['numerical_validation']['terminal99_max_error']:.2e}</b>.</p>
<h3 id="dq-limits">What this evidence can—and cannot—decide</h3>
<p>The independent inference unit is the state. The 10,000 paired bootstrap draws resample within regime/load/time strata while retaining all actions for a state. Full-bank strata contain only 2–3 states. Support-subgroup intervals are unavailable if any nonempty eligible stratum has fewer than two states; missing strata are not manufactured. The intervals concern this designed law, fixed candidate realization and one checkpoint, not training-seed variation.</p>
<p>All harmful changes, unchanged choices, weak or negative error alignment, and unavailable intervals are retained below. A working gradient and a better frozen candidate ranking would still leave multi-step learning, true-Q calibration and policy benefit untested. No new actor step, broad training comparison, coverage-level sensitivity sweep or Option C evaluation occurred.</p>
<p><b>Reproduce without more compute:</b> <code>python report_assets/build_report.py</code> rebuilds this report and the SVG from saved JSON/CSV/NPZ records after verifying their hashes. It performs no model inference, actor update or extra rollout.</p>
<details><summary>All planned subgroup outcomes and unavailable intervals</summary>{subgroup_table(analysis)}</details>
<details><summary>Actual line-numbered source behind the stages</summary>{source_panels(refs)}</details>
<p class="dq-meta">The report reads saved outputs only. It executes no model or simulator. Protocol SHA256 <code>{sha(ROOT/'protocol.json')}</code>; reviewed freeze SHA256 <code>{sha(ROOT/'freeze.json')}</code>; result JSON SHA256 <code>{sha(ROOT/'results/paired_results.json')}</code>.</p>
</section>''']
    return CSS+'\n'+'\n'.join(out)


def subgroup_table(analysis):
    data=[]
    for g in analysis['groups']:
        for method,m in g['methods'].items():
            data.append([esc(g['group']),esc(g['support']),str(g['eligible_states']),esc(LABELS[method]),esc(m['status']),
                fmt(m.get('mean_delta_vs_raw')),ci(m.get('paired_state_bootstrap_ci95')),
                esc(m.get('bootstrap_refusal_reason') or m.get('reason') or 'available'),
                esc(json.dumps(g['bootstrap_context_time_counts'],sort_keys=True))])
    return table(['State group','Action class','Eligible states','Score','Status','True-Q change vs raw','Paired95% interval','Interval/refusal reason','Eligible context×time counts'],data)


def source_panels(refs):
    parts=[]
    for key,r in refs.items():
        code='\n'.join(f'<span><b>{r["first_line"]+i:04d}</b> {esc(line)}</span>' for i,line in enumerate(r['text'].splitlines()))
        parts.append(f'<details id="dq-code-{key}"><summary>{esc(r["file"])}:{r["first_line"]}–{r["last_line"]} · {esc(r["function"])}</summary><p class="dq-meta">{esc(r["path"])}<br>SHA256 <code>{r["sha256"]}</code></p><pre class="dq-source">{code}</pre></details>')
    return ''.join(parts)


CSS='''<style>
#direct-q-ranking .dq-table{overflow-x:auto;margin:18px 0}
#direct-q-ranking table{min-width:760px;width:100%}
#direct-q-ranking td{vertical-align:top;overflow-wrap:anywhere}
#direct-q-ranking .dq-pseudo,#direct-q-ranking .dq-source{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;background:var(--surface);padding:18px}
#direct-q-ranking .dq-pseudo span,#direct-q-ranking .dq-source span{display:block}
#direct-q-ranking .dq-pseudo b,#direct-q-ranking .dq-source b{color:#167d73;display:inline-block;min-width:7em}
#direct-q-ranking .dq-source{font-size:12px}
#direct-q-ranking .dq-meta{font-size:12px;overflow-wrap:anywhere}
#direct-q-ranking details{margin:16px 0}
#direct-q-ranking .dq-figure{margin:24px 0;padding:12px;background:#fbfaf7;border:1px solid var(--rule);border-radius:10px}
#direct-q-ranking .dq-figure svg{width:100%;height:auto;display:block}
#direct-q-ranking .dq-figure figcaption{padding:10px;line-height:1.6}
</style>'''


class Check(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if a.get('id'):self.ids.append(a['id'])
        if tag=='a' and a.get('href','').startswith('#'):self.links.append(a['href'][1:])


def main():
    status,manifest,analysis,candidates,decisions=verify()
    refs=sources();summary=summaries(analysis,decisions);correlations=correlation_summary(analysis)
    error_comparison=saved_error_comparison();cases=diagnostic_cases(candidates,decisions);plot=effect_plot(analysis)
    output=fragment(status,analysis,summary,correlations,refs,error_comparison,cases,plot)
    check=Check();check.feed(output)
    if any(v>1 for v in collections.Counter(check.ids).values()):raise ValueError('duplicate report IDs')
    if set(check.links)-set(check.ids):raise ValueError('broken report source links')
    if '\ufffd' in output:raise ValueError('Unicode replacement character in new report')
    (HERE/'RESULTS.md').write_text(markdown(status,analysis,summary,correlations,error_comparison,cases),encoding='utf-8')
    (HERE/'section.html').write_text(output,encoding='utf-8')
    (HERE/'selected_action_effects.svg').write_text(plot,encoding='utf-8')
    report=dict(schema='direct-q-report-v2',status=status,pooled_summary=summary,correlations=correlations,
        groups=analysis['groups'],error_summaries=analysis['error_summaries'],support_counts=analysis['support_counts'],
        saved_array_score_errors=error_comparison,nonzero_width_decision_cases=cases,
        source_map=refs,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'protocol.json',ROOT/'freeze.json',ROOT/'prepared/preparation_manifest.json',ROOT/'results/status.json',ROOT/'results/paired_results.json',ROOT/'results/candidate_records.csv',ROOT/'results/paired_decisions.csv']},
        output_sha256={name:sha(HERE/name) for name in ('RESULTS.md','section.html','selected_action_effects.svg')},
        validation=dict(all_recorded_output_hashes_checked=True,source_freeze_checked=True,immutable_parameter_hashes_checked=True,
            candidates=384,source_links_resolved=True,new_duplicate_ids=0,new_unicode_replacement_characters=0,
            new_model_or_simulator_calls=0,rendered_browser_tested=False),report_source_sha256=sha(__file__))
    (HERE/'report_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(HERE/'RESULTS.md'),fragment=str(HERE/'section.html'),output_sha256=report['output_sha256']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();main()
