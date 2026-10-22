"""Append a verified, source-mapped action-only result section to the v2 guide.

No model/calibration execution and no modification of the input HTML. The existing
scripts, figures, historical text and navigation behavior are retained bytewise.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREVIOUS=ROOT.parent/'results_20261021/presentation/ThrusterChain_diagnostics_results_v2.html'
sys.path.insert(0,str(ROOT/'analysis'))
from aggregate_results import METHODS,LABELS,interval,pct,fmt,pp_interval


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(v):return html.escape(str(v),quote=True)
def table(head,rows):
    return '<div class="ao-table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def function_source(path,name):
    source=Path(path).read_text(encoding='utf-8');tree=ast.parse(source)
    node=next((n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),None)
    if node is None:raise ValueError(f'Missing source function {name} in {path}')
    return dict(path=str(path),file=Path(path).name,function=name,first_line=node.lineno,last_line=node.end_lineno,
                sha256=sha(path),text='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))


def map_sources(root):
    identity=read(root/'engine_identity.json')
    native=Path(identity['source_closure'])/'calibration/wbcp.py'
    refs={}
    for key,path,name in [
        ('state-bank',root/'engine.py','select_states'),('freeze',root/'run_experiment.py','freeze'),
        ('target-law',root/'engine.py','target_probabilities'),('score',root/'engine.py','score'),
        ('sampling',root/'run_experiment.py','run_rep'),
        ('query-adapter',root/'calibration_methods.py','calibrate_queries'),
        ('posterior',native,'crossings'),('native-threshold',native,'calibrate'),
        ('band-dose',root/'calibration_methods.py','evaluate_queries'),
        ('replicate-inference',root/'analysis/aggregate_results.py','estimate')]:
        refs[key]=function_source(path,name)
    return refs


def ref_link(refs,key):
    r=refs[key]
    return f'<a href="#ao-source-{key}">{esc(r["file"])}:{r["first_line"]}–{r["last_line"]}</a>'


def sources_html(refs):
    out=[]
    for key,r in refs.items():
        lines=r['text'].splitlines()
        code='\n'.join(f'<span><b>{r["first_line"]+i:04d}</b> {esc(line)}</span>' for i,line in enumerate(lines))
        out.append(f'<details id="ao-source-{key}"><summary>{esc(r["file"])}:{r["first_line"]}–{r["last_line"]} · {esc(r["function"])}</summary><p class="ao-meta">SHA256 <code>{r["sha256"]}</code><br>{esc(r["path"])}</p><pre class="ao-code">{code}</pre></details>')
    return ''.join(out)


def stage_table(refs,a):
    stages=[
        ('A1','Freeze the question','One terminal model, its target networks, η95/unit95, and 64 fixed source states.','The same state/time/context law and score kernel feed both methods.','Checkpoint/source hashes and scored fixtures; no fitting.','PASS: provenance and fixed-law contract',ref_link(refs,'state-bank')+'; '+ref_link(refs,'freeze')),
        ('A2','Specify the action law','625 commands; β is the exact product of four categorical laws. qλ moves probability to the projected actor command.','Every target grid command has source mass; Σβ=Σq=1 and Σβ(q/β)=1.','Exhaustive grid normalization and exact-ratio invariants.','PASS on the declared discrete law',ref_link(refs,'target-law')),
        ('A3','Draw paired evidence','128 new calibration draws from μβ; 4,096 independent evaluation draws from μqλ.','Both methods share each calibration/query bank. Replications have independent streams.','Saved IDs, sizes, seeds and raw-array checksums; all 200 registered replications accounted for.','PASS: paired source/target construction',ref_link(refs,'sampling')),
        ('A4','Compute the named error','The same 100k critic, 100k smoothed one-step target, and frozen 95k scale.','S=|Y−Q|/σ. No label/predictor-age difference between calibration and evaluation.','Native scoring and terminal-mask numerical fixtures.','PASS: residual target; true-Q coverage UNTESTED',ref_link(refs,'score')),
        ('A5','Weight the calibration mass','Uniform uses weights 1 / test mass 1. Exact IW uses q/β and each query’s own mass.','Rescale bank and query mass together; retain zero weights; no clipping.','λ=0 must reproduce uniform exactly; all-zero banks refuse.','PASS: arithmetic; finite-sample usefulness measured below',ref_link(refs,'query-adapter')+'; '+ref_link(refs,'native-threshold')),
        ('A6','Choose the posterior threshold','1,000 shared exponential draws; nominal 90% residual mass; 95% posterior credibility.','Query mass sits at infinity. Insufficient finite mass yields R=∞.','Separate no-hit banks from posterior-infinite thresholds.','Empirical evaluation, no frequentist theorem asserted',ref_link(refs,'posterior')),
        ('A7','Convert to band and dose','Half-width U=Rσ; finite multiplier 1+0.5U/(U+unit95); refused multiplier 1.','Fallback is identical across methods. Band refusal is not action refusal or support certification.','Report covered/finite/refused counts and diagnostic multiplier.','PASS: matched readout; intervention benefit UNTESTED',ref_link(refs,'band-dose')),
        ('A8','Interpret the paired outcome','Compare within-replication differences over 200 independent calibration replications.','Queries are not independent replications of the calibration procedure.','Conditional t intervals; every undefined/missing metric remains visible.','Conditional evidence only; policy/occupancy claims UNTESTED',ref_link(refs,'replicate-inference'))]
    rows=[]
    for sid,purpose,inputs,premise,test,status,source in stages:
        rows.append([f'<b>{sid}</b><br>{esc(purpose)}',esc(inputs),esc(premise),esc(test),esc(status),source])
    return table(['Stage / purpose','Input → output','Premise / expected behavior','Isolated evidence','Status / takeaway','Actual source'],rows)


def section(a,refs,root):
    off=a['off_grid_support']; planned=a['completion']['planned']; completed=a['completion']['completed']
    curve=(root/'analysis/coverage_partition.svg').read_text(encoding='utf-8')
    rows=[]
    for level,g in a['mixtures'].items():
        for method in METHODS:
            m=g['methods'][method]['metrics'];c=g['methods'][method]['counts']
            rows.append([esc(level),esc(LABELS[method]),esc(interval(m['coverage_including_infinite'],True)),
                esc(interval(m['usable_coverage'],True)),esc(interval(m['refusal_rate'],True)),
                f'{c["coverage_numerator"]:,}/{c["coverage_denominator"]:,}',f'{c["finite_coverage_numerator"]:,}/{c["finite_coverage_denominator"]:,}'])
    primary=table(['λ','Procedure','Coverage incl. infinity [95% interval]','Finite and covered / all [95% interval]','Refusal [95% interval]','Covered / all queries','Covered / finite queries'],rows)
    paired=[]
    mass=[]
    widths=[]
    causes=[]
    subgroup=[]
    for level,g in a['mixtures'].items():
        p=g['paired']
        paired.append([esc(level),esc(pp_interval(p['delta_coverage_including_infinite'])),esc(pp_interval(p['delta_usable_coverage'])),esc(pp_interval(p['delta_refusal_rate'])),esc(interval(p['delta_mean_dose'])),esc(interval(p['delta_mean_half_width_jointly_accepted']))])
        mass.append([esc(level),esc(interval(g['n_eff'])),esc(fmt(g['positive_weight_count']['mean'],2)),f'{g["no_target_action_hit_banks"]}/{completed}',f'{g["no_positive_weight_banks"]}/{completed}',esc(interval(g['query_mass_summary']['max']))])
        for method in METHODS:
            summary=g['methods'][method];m=summary['metrics'];w=m['mean_finite_half_width']
            widths.append([esc(level),esc(LABELS[method]),esc(interval(w)),f'{w["n_defined"]}/{planned}',str(w['n_undefined']),esc(interval(m['mean_dose_including_refusal']))])
            causes.append([esc(level),esc(LABELS[method]),str(summary['replications_all_queries_refused']),'<code>'+esc(json.dumps(summary['refusal_reasons'],sort_keys=True))+'</code>'])
            for name,sg in summary['subgroups'].items():
                subgroup.append([esc(level),esc(LABELS[method]),esc(name),esc(interval(sg['coverage'],True)),esc(interval(sg['usable_coverage'],True)),esc(interval(sg['refusal'],True)),str(sg['no_positive_calibration_weight_banks']) if name.startswith('context') else 'not a separate bank'])
    full=a['mixtures']['1.0'];iw=full['methods'][METHODS[1]]['metrics'];uniform=full['methods'][METHODS[0]]['metrics']
    interpretation=(f'At the fully shifted grid-actor target, uniform calibration covers {pct(uniform["coverage_including_infinite"]["mean"])} of residuals. '
        f'Exact importance weighting reports {pct(iw["coverage_including_infinite"]["mean"])} total coverage, with {pct(iw["refusal_rate"]["mean"])} infinite-band refusals '
        f'and {pct(iw["usable_coverage"]["mean"])} finite-and-covered queries. These quantities must be read together.')
    out=[f'''<section id="action-only-results" aria-labelledby="ao-heading">
<h2 id="ao-heading">Action-only shift: does exact weighting provide a useful band?</h2>
<p class="ao-lead">A controller can ask about commands that are rare in its recorded data. Before asking whether uncertainty improves a policy, we can test a narrower link: when only command probabilities change, does reweighting the calibration evidence preserve a useful uncertainty band?</p>
<div class="callout"><b>{completed}/{planned} registered calibration replications completed.</b> This is a new frozen-model evaluation, conducted after the original 30-job training study. No actor or critic was trained again. The prior training study and historical guide remain below. The previously proposed action-only protocol has now been executed as this separately frozen control.</div>
<p><b>Observed result:</b> {esc(interpretation)}</p>
<p>Uniform calibration already exceeds nominal 90% residual coverage at every mixture level. Exact weighting adds coverage and refusal together; these results do not show that uniform calibration failed the nominal target or that weighting improved a policy.</p>
<figure class="ao-figure">{curve}<figcaption>Total mathematical coverage combines the green and purple portions. Purple queries are refused infinite bands, which give no finite error bound. Each bar averages the same 200 independent calibration replications; the tables show the paired uncertainty.</figcaption></figure>
<h3 id="ao-laws">Only the current action law changes</h3>
<div class="ao-laws"><div><h4>Fresh calibration source</h4><p><b>State:</b> z∼μ, uniform over 64 fixed state/time/context rows.</p><p><b>Command:</b> a∼β(·|z), the known 625-command categorical behavior law.</p><p><b>Score:</b> one fixed predictor, target kernel and scale.</p></div><div class="ao-arrow" aria-hidden="true">→</div><div><h4>Evaluation target</h4><p><b>Same state law:</b> z∼μ.</p><p><b>Changed command law:</b> q<sub>λ</sub>=(1−λ)β+λδ<sub>grid actor</sub>, λ∈{{0,0.5,1}}.</p><p><b>Same score kernel:</b> the current command changes the next state through the same plant.</p></div></div>
<p>The 64 rows come from 64 source episodes, 16 per actuator-regime/load context. Each context has six rows at time 0, five at 50 and five at 99. Both methods repeatedly draw state IDs independently from that same finite law. Each coordinate of β assigns 0.84 to its nearest controller atom and 0.04 to each other atom; the four-coordinate joint probability is their product. The minimum supported mass is 0.04⁴=0.00000256. Grid projection uses the nearest atom, with lower-atom ties.</p>
<p><b>This is action-only shift conditional on a fixed finite state bank.</b> It is not state-visitation shift, a rollout comparison, or a relabeling of the historical retained dataset. The old 128-row episode-sampled calibration bank is not reused. At λ=0 the new calibration and evaluation laws coincide; “full retention” alone did not imply that property in the earlier stress-query evaluation.</p>
<h3 id="ao-definitions">What the band promises—and what it does not</h3>
<p><b>BCA</b> here uses a Bayesian calibration threshold to scale a predicted uncertainty band. <b>Importance weighting</b> changes the relative mass assigned to calibration scores using the exact target/source ratio q<sub>λ</sub>(a|z)/β(a|z). <b>Nominal 90% coverage</b> is the intended frequency with which the named one-step target lies in the band. It is a target to measure, not a result to assume.</p>
<pre class="ao-equation">Y = reward + (1 − terminal) × min Q_target(next_state, noisy_target_actor_command)
S = |Y − min Q_current(state, command)| / [max(η95(state, command), 10⁻⁶) × unit95]
half-width U = threshold × [max(η95, 10⁻⁶) × unit95]</pre>
<p>The current critic and target networks are fixed at 100,000 updates; scale parameters and residual unit are fixed from the 95,000-update reference. Both fresh calibration and evaluation scores use exactly this same convention. Horizon 100, discount 1 and the remaining-time observation agree; time 99 has no bootstrap. Clipped next-action Gaussian noise is the shared conditional label kernel. No importance ratio is assigned to that noise.</p>
<p><b>A one-step residual is not true long-horizon value error.</b> A finite threshold is not a support certificate. An infinite threshold is a band refusal; it does not instruct the actor to refuse an action. The original native fallback multiplier 1 is kept identical in both procedures.</p>
<h3 id="ao-stage-map">Verify the chain one stage at a time</h3>
<p>These A1–A8 labels describe the boundaries of this isolated control. They are not a claim that the advisor’s illustrative module numbers were implementation names. A correct ratio formula does not prove calibrated coverage; calibrated residual coverage does not prove sensible action penalties or policy improvement.</p>
<div class="ao-panel"><h4>Screenshot-ready annotated pseudocode</h4><pre class="ao-code">
<span><b>01 · A1</b> Freeze checkpoint 100k, target networks, η95/unit95, and state law μ.</span>
<span><b>02 · A2</b> Enumerate 625 commands; verify β&gt;0, Σβ=Σq=1, and Eβ[q/β]=1.</span>
<span><b>03 · A3</b> For each independent replication, draw 128 new source pairs (z,a)∼μβ.</span>
<span><b>04 · A4</b> Compute the same frozen residual score S for calibration and evaluation.</span>
<span><b>05 · A5</b> Uniform: weights 1, query mass 1. IW: weights q/β, query mass q(query)/β(query).</span>
<span><b>06 · A6</b> Use the same 1,000 exponential draws; test mass at infinity can force R=∞.</span>
<span><b>07 · A7</b> Evaluate both bands on the same 4,096 target queries; count every refusal.</span>
<span><b>08 · A7</b> Read out finite multiplier 1+0.5U/(U+unit95); refused multiplier stays 1.</span>
<span><b>09 · A8</b> Pair method differences within each replication, then summarize all 200.</span>
</pre></div>
{stage_table(refs,a)}
<h3 id="ao-coverage">Coverage, usefulness and refusal must be reported together</h3>
{primary}
<p><b>Finite and covered / all</b> counts covered queries with a finite band and retains refusals in the denominator. <b>Covered / finite</b> conditions on the accepted subset, which can be selected by query mass. All counts remain visible. The confidence intervals use independent calibration replications, not individual correlated queries.</p>
<h3 id="ao-paired">The paired comparison isolates weighting</h3>
{table(['λ','Δ coverage incl. infinity','Δ finite and covered / all','Δ refusal','Δ mean multiplier','Δ half-width, jointly finite'],paired)}
<p>Differences are importance weighting minus uniform, evaluated on the same queries. Percentage differences are percentage points. The λ=0 equal-weight identity is checked exactly in every replication; a difference there would indicate an implementation problem. A high IID uncertainty band can still be appropriate if residuals are large.</p>
<h3 id="ao-mass">No hits, low effective sample size and formula failure are different</h3>
{table(['λ','Effective sample size, mean [95% interval]','Positive-weight rows, mean','Banks with no projected-actor hits','Banks with all weights zero','Maximum query mass, mean [95% interval]'],mass)}
<p>Effective sample size is (Σw)²/Σw², with zero assigned to an all-zero bank. At λ=1, only the projected actor command has positive target mass for a given state. If no calibration draw hits that command, the bank contains no target-weighted evidence. The implementation records this case as a refusal, without flooring weights, adding data, or discarding the replication. A nonempty bank can also refuse when its posterior finite mass cannot support the query.</p>
<p><b>Observed here:</b> all 200 calibration banks contain projected-actor hits. No main-run bank has all weights zero. The weighted refusals actually observed are posterior-infinite thresholds from nonempty banks; the all-zero case is a separate tested negative fixture.</p>
{table(['λ','Procedure','Replications with every query refused','Query-level status counts'],causes)}
<h3 id="ao-width-dose">A finite width and a neutral fallback are visible downstream</h3>
{table(['λ','Procedure','Finite half-width mean [95% interval]','Replications with defined finite width','Undefined width replications','Mean multiplier [95% interval]'],widths)}
<p>Widths in the table summarize finite bands only. Each method can accept a different subset of queries, so its separate finite-width mean is not a matched-subset comparison. The paired jointly-finite contrast above compares the same subset. Undefined-replication counts prevent refusal from disappearing through averaging. Any query with an infinite band receives multiplier 1 in this matched diagnostic. No optimizer consumes these readouts in this experiment, so these values establish neither a helpful training intervention nor a harmful one.</p>
<h3 id="ao-off-grid">The continuous actor remains a separate support question</h3>
<div class="callout"><b>{off['off_grid_actor_states']}/{off['states']} deployed actor outputs are off the categorical grid.</b> The source law gives those exact commands zero mass. Their target/source ratio is unsupported, and this test assigns them no finite importance weight or coverage claim. The nearest-grid actor is supported, but it is a different specified target.</div>
<details><summary>Context and time subgroup accounting</summary><p>Contexts 0–3 are stall/low load, stall/high load, monotone/low load and monotone/high load. These are descriptive subgroup diagnostics, not separate subgroup certificates. State/time probabilities are the same in source and target by design; realized counts differ across their independent draws.</p>{table(['λ','Procedure','Subgroup','Coverage incl. infinity','Finite and covered / all','Refusal','Banks with zero positive subgroup weight'],subgroup)}</details>
<h3 id="ao-decision">What can move forward from this evidence?</h3>
<p>The exact categorical law and equal-weight invariants support advancing the <b>controlled weighting test harness</b>. A useful uncertainty procedure must also supply finite bands under its intended query law; total coverage inflated by refusal is not sufficient. This experiment can identify that limitation without confusing it with a ratio-estimation defect.</p>
<p><b>Still untested:</b> new trained seeds, rare/zero retained-source laws, estimated ratios, state-visitation shifts, true policy-value coverage, actual actor optimization and policy benefit. No result here validates every importance-weighted BCA variant or repairs the separately documented historical clipped-density defect. An equal-strength constant-intervention control remains necessary before attributing any future policy benefit to adaptive calibration.</p>
<h3 id="ao-reproduce">Reproduce a consistent or exceptional case</h3>
<p>Each replication has a saved calibration bank, shared query bank, score arrays, threshold arrays, source/target mass, no-hit flag and identity hashes. The paired CSV contains 600 rows, one per replication and mixture. It preserves both method outcomes rather than only aggregate means. Replication IDs 0–199 all belong to the frozen matrix; missing and failed outcomes are reported explicitly.</p>
<pre class="ao-equation">python analysis/aggregate_results.py
# Analysis only: reads saved evidence and validates raw checksums.
# It does not run the model, recalibrate bands, or train a policy.</pre>
<p class="ao-meta">Checkpoint SHA256 <code>{a['identities']['checkpoint_sha256']}</code><br>Freeze SHA256 <code>{a['identities']['freeze_sha256']}</code><br>State-bank SHA256 <code>{a['identities']['state_bank_sha256']}</code></p>
<details><summary>Actual line-numbered source behind the stages</summary><p>The following excerpts are taken from the implemented files and the preserved native training-source closure. Links above lead to these exact lines; the hashes identify their bytes. Pseudocode is explanatory; these excerpts are actual code.</p>{sources_html(refs)}</details>
<details><summary>Statistical scope and artifact verification</summary><p>All intervals are unadjusted two-sided Student-t intervals across the 200 independent calibration replications, conditional on one trained checkpoint and one fixed state bank. Subgroup intervals are descriptive. Queries within a replication do not increase the number of independent calibration replications. Undefined finite metrics retain explicit counts. This control was planned after the training results, then frozen before its own main draws; it is not retroactive preregistration.</p><p>Static HTML structure, internal anchors and unchanged historical scripts pass the builder checks. No new JavaScript is needed for this section. Rendered-browser, navigation and JavaScript-runtime QA were blocked: the browser tool rejected the local file URL under its security policy. No workaround was attempted and no browser check is claimed.</p></details>
</section>''']
    return '\n'.join(out)


CSS='''<style>
#action-only-results .ao-lead{font-size:1.15em;max-width:76ch}
#action-only-results .ao-figure{margin:24px 0;background:#fbfaf7;border:1px solid var(--rule);border-radius:10px;padding:10px}
#action-only-results .ao-figure svg{display:block;width:100%;height:auto}
#action-only-results .ao-figure figcaption{padding:10px;line-height:1.55}
#action-only-results .ao-table-wrap{overflow-x:auto;margin:18px 0}
#action-only-results table{width:100%;min-width:760px}
#action-only-results td{vertical-align:top;overflow-wrap:anywhere}
#action-only-results .ao-laws{display:grid;grid-template-columns:1fr auto 1fr;gap:18px;padding:20px;background:var(--surface);border:1px solid var(--rule);border-radius:10px}
#action-only-results .ao-laws h4{margin-top:0}
#action-only-results .ao-arrow{align-self:center;font-size:32px}
#action-only-results .ao-panel{border-left:5px solid #167d73;background:var(--surface);padding:18px;margin:20px 0}
#action-only-results .ao-code{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7;font-size:13px}
#action-only-results .ao-code span{display:block}
#action-only-results .ao-code b{color:#167d73;display:inline-block;min-width:7em}
#action-only-results .ao-equation{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:var(--surface);line-height:1.7}
#action-only-results .ao-meta{font-size:12px;overflow-wrap:anywhere}
#action-only-results details{margin:15px 0}
@media(max-width:650px){#action-only-results .ao-laws{grid-template-columns:1fr}#action-only-results .ao-arrow{transform:rotate(90deg);justify-self:center}#action-only-results .ao-code{font-size:12px}}
</style>'''


class Structure(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[];self.sections=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if a.get('id'):self.ids.append(a['id'])
        if tag=='a' and a.get('href','').startswith('#'):self.links.append(a['href'][1:])
        if tag=='section' and a.get('id'):self.sections.append(a['id'])


def main(root=ROOT,previous=PREVIOUS,output=None):
    aggregate=read(root/'analysis/aggregate.json')
    if aggregate['completion']['completed']!=aggregate['completion']['planned']:
        raise RuntimeError('This presentation template requires the complete matrix; report incomplete outcomes before adapting it')
    prior=Path(previous).read_text(encoding='utf-8')
    if 'id="action-only-results"' in prior:raise ValueError('Input already contains action-only results')
    refs=map_sources(root);addition=CSS+section(aggregate,refs,root)
    if '\ufffd' in addition:raise AssertionError('New section contains a Unicode replacement character')
    marker='<section id="training-results"'
    if prior.count(marker)!=1:raise ValueError('Expected one prior training-results section')
    revised=prior.replace(marker,addition+'\n'+marker,1)
    historical_notice='<div class="callout ao-historical-plan-note"><b>Historical planning text, preserved from the previous guide.</b> The “proposed / not run” status below describes that earlier document version. The separately frozen action-only evaluation has now completed all 200 replications. <a href="#action-only-results">Read the completed action-only results and their limits.</a></div>'
    plan_heading='<h3 id="tr-action-only-plan">Advisor follow-up: isolate the action law first</h3>'
    if revised.count(plan_heading)!=1:raise ValueError('Expected one historical action-only plan heading')
    revised=revised.replace(plan_heading,plan_heading+historical_notice,1)
    archived_plan='<details><summary>Complete advisor action-only post-hoc plan — proposed, not run</summary>'
    if revised.count(archived_plan)!=1:raise ValueError('Expected one archived action-only plan')
    revised=revised.replace(archived_plan,historical_notice+archived_plan,1)
    output=Path(output or HERE/'ThrusterChain_diagnostics_results_v3.html')
    if output.exists() and output.read_text(encoding='utf-8')!=revised:
        raise FileExistsError('Preserve previous output; specify a new filename for a revised build')
    HERE.mkdir(exist_ok=True)
    if not output.exists():output.write_text(revised,encoding='utf-8')
    old=Structure();old.feed(prior);new=Structure();new.feed(revised)
    olddup={k for k,v in collections.Counter(old.ids).items() if v>1}
    newdup={k for k,v in collections.Counter(new.ids).items() if v>1}
    if newdup-olddup:raise AssertionError('New duplicate IDs: '+repr(newdup-olddup))
    missing=[x for x in new.links if x.startswith('ao-') and x not in set(new.ids)]
    if missing:raise AssertionError('New broken anchors: '+repr(missing))
    script_pattern=r'<script\b[^>]*>.*?</script\s*>'
    if re.findall(script_pattern,prior,re.S)!=re.findall(script_pattern,revised,re.S):
        raise AssertionError('Historical scripts changed')
    if revised.replace(addition+'\n','',1).replace(historical_notice,'')!=prior:raise AssertionError('Historical input was not preserved exactly')
    receipt=dict(input=str(previous),input_sha256=sha(previous),output=str(output),output_sha256=sha(output),
        output_bytes=output.stat().st_size,aggregate_sha256=sha(root/'analysis/aggregate.json'),
        historical_content_preserved_exactly=True,historical_script_blocks_unchanged=True,
        historical_plan_notices_added=2,
        new_sections=['action-only-results'],new_duplicate_ids=[],new_broken_internal_anchors=[],
        browser_rendering_tested=False,browser_navigation_tested=False,javascript_runtime_tested=False,
        browser_qa_blocker='Browser tool denied local file URL under security policy; no workaround attempted',
        source_map=refs)
    (HERE/'build_v3_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_map'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--previous',type=Path,default=PREVIOUS);parser.add_argument('--output',type=Path)
    args=parser.parse_args();main(args.root,args.previous,args.output)
