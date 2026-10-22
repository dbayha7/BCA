"""Analysis-only summary of all registered action-only calibration replications.

Does not import, fit, or execute the model or calibration procedure. Run only after
execution_status.json is terminal. Undefined finite-band summaries remain visible.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHODS = ('uniform', 'exact_categorical_iw_per_query')
LABELS = {'uniform':'Uniform calibration', 'exact_categorical_iw_per_query':'Exact importance weighting'}
METRICS = ('coverage_including_infinite','usable_coverage','finite_conditional_coverage',
           'refusal_rate','mean_finite_half_width','mean_finite_full_width',
           'mean_dose_including_refusal','mean_accepted_dose',
           'absolute_error_dose_spearman_all','absolute_error_width_spearman_accepted')
COUNT_KEYS = ('coverage_numerator','coverage_denominator','finite_coverage_numerator',
              'finite_coverage_denominator','usable_coverage_numerator','usable_coverage_denominator',
              'threshold_usable_count','fallback_count','vacuous_count')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def numeric(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def estimate(values, expected):
    valid=[float(v) for v in values if numeric(v)]
    n=len(valid)
    if not n:
        return dict(mean=None,ci95=None,n_defined=0,n_undefined=len(values),n_missing=expected-len(values),
                    sd=None,se=None,min=None,max=None,unit='independent calibration replication')
    sd=statistics.stdev(valid) if n>1 else None
    se=sd/math.sqrt(n) if sd is not None else None
    half=float(student_t.ppf(.975,n-1))*se if n>1 else None
    mean=statistics.mean(valid)
    return dict(mean=mean,ci95=[mean-half,mean+half] if half is not None else None,
        n_defined=n,n_undefined=len(values)-n,n_missing=expected-len(values),sd=sd,se=se,
        min=min(valid),max=max(valid),unit='independent calibration replication',
        interpretation='conditional mean among replications where defined' if n<len(values) else 'mean across all completed registered replications')


def quantiles(values):
    a=sorted(float(x) for x in values if numeric(x))
    if not a:return dict(n=0,min=None,median=None,p90=None,p99=None,max=None)
    def q(p):
        pos=(len(a)-1)*p;i=int(pos);fraction=pos-i
        return a[i]*(1-fraction)+a[min(i+1,len(a)-1)]*fraction
    return dict(n=len(a),min=a[0],median=q(.5),p90=q(.9),p99=q(.99),max=a[-1])


def flatten(value,prefix=''):
    out={}
    for k,v in value.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flatten(v,key+'.'))
        elif isinstance(v,list):out[key]=json.dumps(v,separators=(',',':'))
        else:out[key]=v
    return out


def csv_file(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def verify_record(item,protocol,root,*,check_raw=True):
    rid=item['replication']
    if item.get('status')!='completed':raise ValueError(f'replicate{rid} is not complete')
    raw=root/'results'/item['raw_file']
    if check_raw and (not raw.is_file() or sha(raw)!=item['raw_sha256']):raise ValueError(f'raw integrity failed for replication{rid}')
    levels=protocol['action_law']['mixtures']
    if [x['mixture'] for x in item['records']]!=levels:raise ValueError(f'mixture matrix mismatch for replication{rid}')
    bank_ids=set()
    for row in item['records']:
        if row['replication']!=rid or row['checkpoint_sha256']!=protocol['checkpoint_sha256']:
            raise ValueError('replication/model identity mismatch')
        if row['calibration_size']!=protocol['calibration']['size'] or row['evaluation_size']!=protocol['evaluation']['queries_per_replication_per_mixture']:
            raise ValueError('registered sample budget changed')
        if row['state_bank_sha256']!=sha(root/'state_bank.npz') or row['off_grid_audit_sha256']!=sha(root/'off_grid_support_audit.json'):
            raise ValueError('state/support artifact changed')
        if set(row['methods'])!=set(METHODS):raise ValueError('method set mismatch')
        bank_ids.add(row['calibration_bank_id'])
        for method,data in row['methods'].items():
            n=data['coverage_denominator']
            if n!=row['evaluation_size']:raise ValueError('query count mismatch')
            if data['threshold_usable_count']+data['fallback_count']!=n:raise ValueError('query accounting mismatch')
            if data['coverage_numerator']!=data['finite_coverage_numerator']+data['vacuous_count']:
                raise ValueError('vacuous coverage not separated')
            if abs(data['coverage_including_infinite']-data['coverage_numerator']/n)>1e-12:raise ValueError('coverage ratio mismatch')
            if data['finite_coverage_denominator']==0 and data['finite_conditional_coverage'] is not None:
                raise ValueError('undefined finite coverage was fabricated')
        if row['mixture']==0:
            if row['methods']['uniform']!=row['methods']['exact_categorical_iw_per_query']:
                # Per-method metadata may differ in otherwise irrelevant descriptive keys.
                for key in METRICS+COUNT_KEYS:
                    if row['methods']['uniform'].get(key)!=row['methods']['exact_categorical_iw_per_query'].get(key):
                        raise ValueError('equal-weight identity failed: '+key)
    if len(bank_ids)!=1:raise ValueError('mixtures do not share a calibration bank')


def summarize_group(rows,expected):
    group=dict(completed_replications=len(rows),methods={},paired={},
        n_eff=estimate([r['n_eff'] for r in rows],expected),
        n_eff_distribution=quantiles([r['n_eff'] for r in rows]),
        positive_weight_count=estimate([r['positive_weight_count'] for r in rows],expected),
        projected_actor_hits=quantiles([r['projected_actor_hits'] for r in rows]),
        no_positive_weight_banks=sum(r['no_positive_weight_bank'] for r in rows),
        no_target_action_hit_banks=sum(r['projected_actor_hits']==0 for r in rows),
        weight_tail_summary={k:estimate([r['calibration_weights'][k] for r in rows],expected) for k in ('mean','max','p99')},
        query_mass_summary={k:estimate([r['query_mass'][k] for r in rows],expected) for k in ('mean','max','p99')})
    for method in METHODS:
        vals=[r['methods'][method] for r in rows]
        summary=dict(metrics={key:estimate([x.get(key) for x in vals],expected) for key in METRICS},
                     counts={key:sum(x[key] for x in vals) for key in COUNT_KEYS},refusal_reasons={})
        reasons=collections.Counter()
        for val in vals:reasons.update(val.get('refusal_counts',{}))
        summary['refusal_reasons']=dict(reasons)
        summary['replications_all_queries_refused']=sum(x['threshold_usable_count']==0 for x in vals)
        summary['replications_any_refusal']=sum(x['fallback_count']>0 for x in vals)
        summary['subgroups']={}
        subkeys=sorted({k for val in vals for k in val['subgroups']})
        for key in subkeys:
            sub=[val['subgroups'][key] for val in vals]
            summary['subgroups'][key]=dict(counts={k:sum(x[k] for x in sub) for k in COUNT_KEYS},
                coverage=estimate([x['coverage_numerator']/x['coverage_denominator'] if x['coverage_denominator'] else None for x in sub],expected),
                refusal=estimate([x['fallback_count']/x['coverage_denominator'] if x['coverage_denominator'] else None for x in sub],expected),
                usable_coverage=estimate([x['usable_coverage_numerator']/x['usable_coverage_denominator'] if x['usable_coverage_denominator'] else None for x in sub],expected),
                no_positive_calibration_weight_banks=sum(x.get('calibration_positive_weights',1)==0 for x in sub),
                calibration_ess=estimate([x.get('calibration_ess') for x in sub],expected))
        group['methods'][method]=summary
    for key in sorted({k for r in rows for k in r['paired']}):
        if key=='n_queries':continue
        group['paired'][key]=estimate([r['paired'].get(key) for r in rows],expected)
    # Same-named metric differences remain paired; undefined pairs are explicit.
    group['paired_same_metric']={key:estimate([
        r['methods'][METHODS[1]].get(key)-r['methods'][METHODS[0]].get(key)
        if numeric(r['methods'][METHODS[1]].get(key)) and numeric(r['methods'][METHODS[0]].get(key)) else None
        for r in rows],expected) for key in METRICS}
    return group


def pct(x):return 'undefined' if x is None else f'{100*x:.2f}%'
def fmt(x,n=4):return 'undefined' if x is None else f'{x:.{n}f}'
def interval(x,percentage=False):
    f=pct if percentage else fmt
    return f(x['mean'])+(f" [{f(x['ci95'][0])}, {f(x['ci95'][1])}]" if x['ci95'] else '')


def pp_interval(x):
    if x['mean'] is None:return 'undefined'
    center=f"{100*x['mean']:+.2f} pp"
    return center+(f" [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}] pp" if x['ci95'] else '')


def result_markdown(aggregate):
    a=aggregate
    lines=['# Controlled action-only calibration evaluation','',
        f"Completed **{a['completion']['completed']}/{a['completion']['planned']}** registered independent calibration replications. "
        f"Missing: {a['completion']['missing_ids']}; terminal failed: {a['completion']['failed_ids']}; "
        f"historical failure records: {a['completion']['failure_record_ids']}.",'',
        'Uniform calibration already exceeds the nominal 90% residual-coverage target at all three mixture levels. Exact weighting increases total coverage under action shift, '
        'but much of that increase comes with infinite-band refusal and lower finite-and-covered coverage. This is a coverage/usefulness tradeoff, not evidence of policy benefit.',
        '', 'The model, physical state/time/context law, target networks and label-noise law are fixed. Only the current conditional action law changes. '
        'Both methods see the same fresh calibration rows and the same evaluation rows. Importance weighting uses exact categorical target/source probabilities.',
        '', '## Coverage is not enough: keep infinite bands visible','',
        '| Mixture λ | Procedure | Coverage including infinity (95% interval) | Finite and covered / all | Refusal / fallback | Finite conditional coverage |',
        '|---|---|---|---|---|---|']
    for level,g in a['mixtures'].items():
        for method in METHODS:
            m=g['methods'][method]['metrics']
            lines.append(f"| {level} | {LABELS[method]} | {interval(m['coverage_including_infinite'],True)} | {interval(m['usable_coverage'],True)} | {interval(m['refusal_rate'],True)} | {interval(m['finite_conditional_coverage'],True)} |")
    lines+=['', 'Nominal residual coverage is 90%. An infinite band covers every residual mathematically but provides no finite uncertainty bound. '
        '“Finite and covered / all” counts only useful covered queries and keeps refused queries in the denominator. '
        'Finite conditional coverage conditions on accepted queries and can describe a selected subset.', '',
        '## Paired weighting contrasts','',
        '| Mixture λ | Δ coverage including infinity | Δ finite-and-covered fraction | Δ refusal | Δ multiplier | Jointly finite width difference |',
        '|---|---|---|---|---|---|']
    for level,g in a['mixtures'].items():
        p=g['paired']
        lines.append(f"| {level} | {pp_interval(p['delta_coverage_including_infinite'])} | {pp_interval(p['delta_usable_coverage'])} | {pp_interval(p['delta_refusal_rate'])} | {interval(p['delta_mean_dose'])} | {interval(p['delta_mean_half_width_jointly_accepted'])} |")
    lines+=['','Differences are exact importance weighting minus uniform on the same queries. Percentage quantities above are differences in percentage points. '
        'Width contrasts condition on queries where both bands are finite; they are not unconditional expected widths. '
        'Separate method means of finite width and finite conditional coverage use method-specific accepted subsets; only the jointly-finite contrast compares the same subset.', '',
        '## Calibration mass and refusal causes','',
        '| Mixture λ | Mean effective sample size [95% interval] | Positive-weight rows, mean | Banks without target-action hits | Banks with all weights zero |',
        '|---|---|---|---|---|']
    for level,g in a['mixtures'].items():
        lines.append(f"| {level} | {interval(g['n_eff'])} | {fmt(g['positive_weight_count']['mean'],2)} | {g['no_target_action_hit_banks']}/{g['completed_replications']} | {g['no_positive_weight_banks']}/{g['completed_replications']} |")
    lines+=['', 'At λ=1 only the projected actor command has positive target mass at each state. '
        f"Observed no-hit/all-zero banks: {a['mixtures']['1.0']['no_positive_weight_banks']}/{a['completion']['completed']}. "
        'All observed weighted refusals in this completed experiment are posterior-infinite thresholds from nonempty positive-weight banks. '
        'The separately tested all-zero-bank refusal remains a negative fixture, not an observed main-run failure. '
        'A nonempty bank can still produce an infinite posterior threshold because query mass is too large relative to calibration mass.', '',
        '## Finite widths and multiplier readout','',
        '| Mixture λ | Procedure | Finite half-width mean [95% interval] | Replications with a defined width | Mean multiplier [95% interval] |',
        '|---|---|---|---|---|']
    for level,g in a['mixtures'].items():
        for method in METHODS:
            m=g['methods'][method]['metrics'];w=m['mean_finite_half_width']
            lines.append(f"| {level} | {LABELS[method]} | {interval(w)} | {w['n_defined']}/{a['completion']['planned']} ({w['n_undefined']} undefined; {w['n_missing']} missing) | {interval(m['mean_dose_including_refusal'])} |")
    off=a['off_grid_support']
    lines+=['',f"The deployed continuous actor is off-grid at **{off['off_grid_actor_states']}/{off['states']}** fixed states. "
        'Its actions are structurally unsupported by this discrete source law. The supported grid projection is a different target; rounding cannot certify the continuous actor.', '',
        '## Scope and interpretation','',
        '- The replication unit is the independently drawn calibration bank and its paired evaluation random streams. The 4,096 queries within each replication are not treated as 4,096 independent replications of the calibration procedure.',
        '- Intervals are unadjusted two-sided Student-t intervals across independent calibration replications, conditional on this one trained checkpoint and fixed finite state bank. Subgroup intervals are descriptive; no simultaneous or per-state certificate is claimed.',
        '- Undefined finite-band metrics remain undefined and their counts are shown. All refusals remain in overall coverage, useful-coverage and fallback denominators.',
        '- This is the BCA-specific Bayesian threshold procedure with an exact per-query importance-mass adapter. It is not a weighted-conformal reference and no borrowed frequentist coverage guarantee is asserted.',
        '- Scores measure a one-step Bellman-target residual, not error against true long-horizon policy value. The original trained 95k scale is frozen while both calibration and evaluation residuals use the same 100k predictor and target.',
        '- The multiplier is a diagnostic readout only. No actor update, policy improvement, state-visitation robustness, critical-region certification or general conclusion about every IWBCA version follows.',
        '- The finite-state law contains already observed source episodes and the model choice was availability-based after the original training study. This is a frozen post-hoc control, not retroactive preregistration or new-model generalization.', '',
        '## Reproducibility and evidence','',
        f"- Checkpoint SHA256: `{a['identities']['checkpoint_sha256']}`.",
        f"- Protocol freeze SHA256: `{a['identities']['freeze_sha256']}`.",
        f"- State-bank SHA256: `{a['identities']['state_bank_sha256']}`.",
        '- `paired_replicates.csv` retains each mixture and paired method result with bank/query identities, numerator/denominator counts, width, usability, fallback, effective sample size, no-hit and support-audit identity.',
        '- `method_replicates.csv`, `subgroup_replicates.csv` and `paired_records.json` retain additional detail; `aggregate.json` records all undefined/missing counts and source hashes.',
        '- Local `results/replicate_*.npz` files retain draw-level evidence. Analysis does not run calibration or inference again.', '',
        'Command: `python analysis/aggregate_results.py` (requires SciPy for Student-t quantiles).', '']
    return '\n'.join(lines)


def stack_svg(aggregate):
    rows=[]
    for level,g in aggregate['mixtures'].items():
        for method in METHODS:
            m=g['methods'][method]['metrics']
            accepted=m['usable_coverage']['mean']; refused=m['refusal_rate']['mean']
            if accepted is not None and refused is not None:
                rows.append((level,LABELS[method],accepted,max(0,1-accepted-refused),refused))
    width=1050;height=160+70*len(rows)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="ao-stack-title ao-stack-desc">',
         '<title id="ao-stack-title">Finite coverage versus vacuous coverage under action shift</title>',
         '<desc id="ao-stack-desc">Each bar partitions all queries into finite and covered, finite and missed, and infinite-band refusal. A large refusal fraction can inflate total mathematical coverage.</desc>',
         f'<rect width="{width}" height="{height}" fill="#fbfaf7"/>',
         '<g font-family="Arial,sans-serif" fill="#182d36">',
         '<text x="24" y="34" font-size="23" font-weight="bold">A covered query can still have no finite band</text>',
         '<text x="24" y="60" font-size="15">Same model and states; λ increases the probability of the grid actor command</text>']
    colors=('#167d73','#d5a242','#7762a3')
    for i,(level,method,covered,missed,refused) in enumerate(rows):
        y=95+i*70
        out+=[f'<text x="24" y="{y+15}" font-size="15">λ={level} · {html.escape(method)}</text>']
        x=335
        for value,color in zip((covered,missed,refused),colors):
            out.append(f'<rect x="{x:.3f}" y="{y}" width="{650*value:.3f}" height="25" fill="{color}"/>');x+=650*value
        out.append(f'<text x="335" y="{y+44}" font-size="13">Finite covered {100*covered:.2f}% · finite missed {100*missed:.2f}% · refused {100*refused:.2f}%</text>')
    y=height-25
    for x,color,label in [(24,colors[0],'Finite and covered'),(285,colors[1],'Finite and missed'),(545,colors[2],'Infinite band / refusal')]:
        out+=[f'<rect x="{x}" y="{y-14}" width="16" height="16" fill="{color}"/>',f'<text x="{x+25}" y="{y}" font-size="15">{label}</text>']
    out.append('</g></svg>')
    return '\n'.join(out)


def main(root=ROOT,output_dir=HERE,records_only=False):
    root=Path(root).resolve();output_dir=Path(output_dir).resolve()
    if records_only and output_dir==(root/'analysis').resolve():
        raise ValueError('Records-only analysis requires a separate output directory; preserve the original fully verified analysis')
    expected_record_hashes=None
    if records_only:
        original=read(root/'analysis/aggregate.json')
        expected_record_hashes=original['identities']['record_sha256']
    status=read(root/'execution_status.json')
    if status.get('status') not in ('completed','completed_with_failures'):
        raise RuntimeError('Main evaluation is not terminal; do not analyze a partial favorable prefix')
    freeze=read(root/'freeze.json');protocol=freeze['protocol'];expected=protocol['calibration']['replications']
    expected_ids=list(range(expected));items=[];record_files={};missing=[];failed=[]
    historical_failures=[]
    for rep in expected_ids:
        p=root/'results'/f'replicate_{rep:03d}.json'
        fp=root/'results'/f'replicate_{rep:03d}.failure.json'
        if fp.exists():historical_failures.append(rep)
        if not p.exists():
            (failed if fp.exists() else missing).append(rep);continue
        digest=sha(p)
        if records_only and expected_record_hashes.get(p.name)!=digest:
            raise ValueError('Portable JSON record differs from the original full-analysis manifest: '+p.name)
        item=read(p);verify_record(item,protocol,root,check_raw=not records_only);items.append(item);record_files[p.name]=digest
    if sorted(status['completed'])!=sorted(x['replication'] for x in items):raise ValueError('terminal status disagrees with saved records')
    support_audit=read(root/'off_grid_support_audit.json')
    records=[dict(r,off_grid_actor_states=support_audit['off_grid_actor_states'],
        actor_support_audit_states=support_audit['states'],
        supported_actor_states=support_audit['supported_actor_states'],
        off_grid_support_outcome='off-grid deployed actor queries unsupported; no finite ratio or coverage assigned')
        for x in items for r in x['records']]
    if len({x['records'][0]['calibration_bank_id'] for x in items})!=len(items):raise ValueError('duplicate independent calibration bank identity')
    levels=protocol['action_law']['mixtures']
    groups={str(level):summarize_group([r for r in records if r['mixture']==level],expected) for level in levels}
    aggregate=dict(schema='action-only-analysis-v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        completion=dict(planned=expected,completed=len(items),missing_ids=missing,failed_ids=failed,failure_record_ids=historical_failures,
            recovered_failure_ids=sorted(set(historical_failures)-set(failed)),replication_ids=[x['replication'] for x in items]),
        identities=dict(checkpoint_sha256=protocol['checkpoint_sha256'],freeze_sha256=sha(root/'freeze.json'),
            state_bank_sha256=sha(root/'state_bank.npz'),off_grid_audit_sha256=sha(root/'off_grid_support_audit.json'),
            protocol_sha256=sha(root/'protocol.json'),record_sha256=record_files),
        design=dict(replication_unit='independent fresh calibration bank plus paired evaluation streams',calibration_size=protocol['calibration']['size'],
            evaluation_queries_per_mixture_replication=protocol['evaluation']['queries_per_replication_per_mixture'],
            nominal_coverage=.9,posterior_draws=1000,trained_checkpoints=1,state_rows=64,independent_source_episodes=64,
            method_interpretation='native Bayesian procedure; uniform versus exact categorical per-query importance-mass variant'),
        inference=dict(interval='unadjusted two-sided Student-t, df=n_defined-1',conditional_on='one fixed trained checkpoint and64-row state bank',
            undefined_rule='finite-band summaries show n_defined and n_undefined; no imputation or silent dropping',
            paired_rule='within-replication weighted-minus-uniform differences on the same evaluation queries'),
        verification=dict(raw_checksums_checked=not records_only,
            record_hashes_checked_against_original_full_analysis=records_only,
            scope='saved-record reaggregation only; raw NPZ files not present or checked' if records_only else 'full local analysis with every raw NPZ checksum verified'),
        off_grid_support=read(root/'off_grid_support_audit.json'),mixtures=groups)
    output_dir.mkdir(parents=True,exist_ok=True)
    write(output_dir/'aggregate.json',aggregate);write(output_dir/'paired_records.json',records)
    paired=[];long=[];sub=[]
    for r in records:
        base={k:v for k,v in r.items() if k not in ('methods',)}
        pair=flatten(base)
        for method,m in r['methods'].items():
            slim={k:v for k,v in m.items() if k not in ('subgroups','calibration')}
            pair.update(flatten(slim,method+'.'))
            long.append(dict(flatten({k:v for k,v in base.items() if k!='paired'}),method=method,**flatten(slim),
                calibration_summary_json=json.dumps(m['calibration'],separators=(',',':'))))
            for name,vals in m['subgroups'].items():
                sub.append(dict(replication=r['replication'],mixture=r['mixture'],method=method,subgroup=name,
                    calibration_bank_id=r['calibration_bank_id'],evaluation_query_id=r['evaluation_query_id'],**flatten(vals)))
        paired.append(pair)
    csv_file(output_dir/'paired_replicates.csv',paired);csv_file(output_dir/'method_replicates.csv',long);csv_file(output_dir/'subgroup_replicates.csv',sub)
    (output_dir/'RESULTS.md').write_text(result_markdown(aggregate),encoding='utf-8')
    (output_dir/'coverage_partition.svg').write_text(stack_svg(aggregate),encoding='utf-8')
    write(output_dir/'analysis_validation.json',dict(completed=len(items),paired_rows=len(paired),method_rows=len(long),subgroup_rows=len(sub),
        equal_weight_identity_checked=True,coverage_accounting_checked=True,raw_checksums_checked=not records_only,
        record_hashes_checked_against_original_full_analysis=records_only,unique_calibration_banks_checked=True,
        missing_ids=missing,failed_ids=failed,historical_failure_records=historical_failures,
        aggregate_sha256=sha(output_dir/'aggregate.json'),analysis_source_sha256=sha(__file__)))
    print(json.dumps(dict(completed=len(items),missing=missing,failed=failed,aggregate=str(output_dir/'aggregate.json'),records_only=records_only)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,default=HERE)
    parser.add_argument('--records-only',action='store_true',help='Verify saved JSON against original analysis record hashes; do not claim raw NPZ verification. Requires a separate output directory.')
    args=parser.parse_args();main(args.root,args.output_dir,args.records_only)
