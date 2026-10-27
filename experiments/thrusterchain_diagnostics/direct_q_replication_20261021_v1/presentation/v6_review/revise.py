"""Five incremental presentation passes. No experiment execution or data mutation."""
from pathlib import Path
import hashlib, json, re, sys

ROOT = Path(__file__).resolve().parent
PRESENTATION = ROOT.parent
SOURCE = PRESENTATION / 'ThrusterChain_diagnostics_results_v5_fresh_replication.html'
OUTPUT = PRESENTATION / 'ThrusterChain_diagnostics_results_v6_reviewed.html'

def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)

def insert_after(text, marker, addition):
    return replace_once(text, marker, marker + addition)

def read_current():
    return OUTPUT.read_text(encoding='utf-8')

def save(text, number, title, changes):
    OUTPUT.write_text(text, encoding='utf-8', newline='\n')
    snapshot = ROOT / f'pass_{number:02d}.html'
    snapshot.write_bytes(OUTPUT.read_bytes())
    log_path = ROOT / 'passes.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    assert len(log) == number - 1, 'Passes must be applied in order.'
    log.append(dict(pass_number=number, title=title, changes=changes,
                    bytes=snapshot.stat().st_size, sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()))
    log_path.write_text(json.dumps(log, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(log[-1], indent=2))

def pass_one():
    s = SOURCE.read_text(encoding='utf-8')
    s = replace_once(s, '<title>ThrusterChain</title>', '')
    s = replace_once(s, '<html><head>', '<html lang="en"><head><title>ThrusterChain diagnostics — results and fresh-state replication | V6</title>')
    start = s.index('    <p class="note"><b>20 October addition:</b>')
    end = s.index('    <div class="start-bar">', start)
    s = s[:start] + '''    <p class="note"><b>21 October 2026 · V6 reading revision of V5 evidence.</b> This page brings together the historical study, the completed 30-job training comparison, the action-only test, and two direct-Q diagnostic banks. The fresh-state replication reuses one trained model; it adds no training runs.</p>
    <div class="v6-intro" id="evidence-overview" aria-labelledby="v6-overview-heading">
      <h2 id="v6-overview-heading">Does the uncertainty width help choose better actions?</h2>
      <p><b>The fresh-state bank does not demonstrate an added benefit from the adaptive width.</b> Subtracting the width barely changes average pair ordering, and the mean gain in the chosen action’s continuation return is close to zero. These results describe one fixed model and one designed bank of 32 states.</p>
      <ol class="v6-route">
        <li><a href="#training-results"><b>1 · Establish the training result.</b></a> Compare the original learner, native BCA’s stronger pull toward recorded actions, and the most-caution control. This is the completed 30-job comparison.</li>
        <li><a href="#action-only-results"><b>2 · Isolate the calibration question.</b></a> Change only the current action distribution on a finite grid. Check coverage, width and refusal together.</li>
        <li><a href="#direct-q-ranking"><b>3 · Test a direct value penalty.</b></a> Keep the trained actor and critic fixed. Subtract width from the critic’s score and inspect which candidate wins.</li>
        <li><a href="#direct-q-replication"><b>4 · Repeat on fresh states.</b></a> Freeze the new inputs before obtaining their returns. Test error alignment, pair ordering and the selected action separately.</li>
      </ol>
      <p class="note">The newest result is shown first below for returning readers. Follow the numbered links above for the experimental sequence. The <a href="#learn">step-by-step lesson</a> explains the earlier study; its settings and outcomes should not be pooled with the later diagnostics.</p>
    </div>
''' + s[end:]
    s = replace_once(s, '<div class="start-bar">', '<div class="start-bar">\n      <a class="btn" href="#direct-q-replication">Read the fresh-state result</a>')
    s = replace_once(s, 'The process, step by step, as it runs in ThrusterChain.', 'The historical study, step by step. The later diagnostics above use their own documented data and evaluation settings.')
    for target, label in [('direct-q-replication', '4 · Fresh-state replication'), ('direct-q-ranking', '3 · First direct-Q test'), ('action-only-results', '2 · Action-only calibration test'), ('training-results', '1 · Native BCA training comparison')]:
        pattern = rf'(<a href="#{target}" data-unit="{target}">)[^<]+(</a>)'
        s, count = re.subn(pattern, lambda m: m[1] + label + m[2], s)
        assert count == 1
    css = '''
<style id="v6-reading-style">
.v6-intro{display:grid;gap:16px;padding:clamp(16px,3vw,28px);background:var(--surface);border:1px solid var(--rule-strong);border-left:4px solid var(--accent);border-radius:10px;min-width:0}
.v6-intro h2{font-size:clamp(24px,3.4vw,34px);line-height:1.18}
.v6-route{list-style:none;padding:0;display:grid;gap:12px;max-width:76ch}
.v6-route li{padding-block:9px;border-bottom:1px solid var(--rule);line-height:1.55}
.v6-route a{display:block;margin-bottom:3px}
.v6-note{padding:14px 16px;border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:4px;max-width:76ch}
</style>
'''
    s = replace_once(s, '</head>', css + '</head>')
    save(s, 1, 'Orientation and study sequence', ['Current dated introduction', 'Four-stage evidence reading path', 'Latest-result entry point', 'Explicit historical-study boundary', 'Language and document title'])

def pass_two():
    s = read_current()
    s = replace_once(s, 'BCA is the existing Bayesian calibration procedure.', 'BCA means Bayesian Conformal Aggregation.')
    s = replace_once(s, 'earlier results remain below, unchanged and unpooled.', 'earlier measurements remain below and are not pooled with this bank.')
    s = replace_once(s, 'Points are state means; lines are descriptive95percent bootstrap intervals. Zero marks no paired difference.', 'Points are equal-state means; lines are descriptive 95% state-bootstrap intervals. Zero marks no paired difference. Correlation contrasts and concordance change are different quantities.')
    s = insert_after(s, '<h3 id="fr-premise">Define the premise before the mechanism</h3>', '''
<dl class="v6-terms">
  <dt>Actor, π</dt><dd>The already-trained controller. It supplies the command after the candidate’s first step.</dd>
  <dt>Critic score, Q1</dt><dd>The first learned critic’s predicted value for a state and candidate action. A larger score wins.</dd>
  <dt>Continuation return, Qπ</dt><dd>Apply one candidate action, then follow the unchanged actor to step 100. Add the simulator rewards with discount 1. “True Q” in the evidence tables means this numerical reference, not the best possible return.</dd>
  <dt>Width, U</dt><dd>BCA’s native uncertainty half-width, fitted to historical one-step residuals. Its nominal 90% setting does not establish coverage of Qπ.</dd>
  <dt>Signed error, e</dt><dd><code>e = Q1 − Qπ</code>. Positive means overestimation; negative means underestimation. “Relative” compares the signed errors of two actions at the same state; it does not mean percentage error.</dd>
  <dt>Distance, D</dt><dd>The distance term used by the frozen control, subtracted with the inherited coefficient 0.519506430786272. This coefficient was not fitted again on the fresh bank.</dd>
</dl>
<p class="v6-note"><b>How a width could help:</b> for two candidates A and B, the corrected score gap is <code>(Q1(A) − Q1(B)) − (U(A) − U(B))</code>. Only the difference in their widths can change their order. Subtracting the same number from every action leaves the winner unchanged.</p>
''')
    s = replace_once(s, 'The primary error is <b>e = Q1 − true Q of the original actor continuation</b>.', 'The primary error is <b>e = Q1 − Qπ</b>, as defined above.')
    s = replace_once(s, 'Retain comparable-pair counts, selected trueQ, raw/control/actor contrasts and every invalid state. <a href="#fr-code-concordance">replication.py:215-226</a>', 'Retain comparable-pair counts, selected Qπ, raw/control/actor contrasts and every invalid state. <a href="#fr-code-concordance">concordance:215-226</a>; <a href="#fr-code-state-metrics">state_metrics:229-267</a>; <a href="#fr-code-analysis">analyze:298-334</a>')
    s = replace_once(s, 'Permute the12 widths once', 'Permute the 12 widths once')
    s = insert_after(s, '<h3 id="results-status">Where things stand, and what is still pending, as of 20 Oct 2026</h3>', '<p class="v6-note"><b>Historical status, through 20 October.</b> The 21 October <a href="#training-results">training comparison</a>, <a href="#direct-q-ranking">direct-Q test</a> and <a href="#direct-q-replication">fresh-state replication</a> are reported separately. Read those sections for the later evidence.</p>')
    s = insert_after(s, '<h3 id="bu-whole">The whole picture</h3>', '<p class="v6-note"><b>Scope: the historical uses cataloged in this section.</b> The later direct-Q tests explicitly change where the penalty is applied. See the <a href="#direct-q-ranking">first bank</a> and <a href="#direct-q-replication">fresh-state bank</a> before treating an open question here as current.</p>')
    example = '''
<h3 id="fr-worked-example">One changed choice explains the small average</h3>
<p>In fresh state 24 (monotone world, high load, time 0), raw Q1 chooses candidate 0. Subtracting width chooses candidate 7. The calculation below uses rounded display values; the saved decision uses full precision.</p>
<div class="v6-scroll" tabindex="0" role="region" aria-label="Worked action-choice example"><table class="v6-compact">
<caption>Fresh state 24 · larger scores and continuation returns are better</caption>
<thead><tr><th scope="col">Quantity</th><th scope="col">Candidate 0</th><th scope="col">Candidate 7</th></tr></thead>
<tbody><tr><th scope="row">Raw Q1</th><td>−271.535248</td><td>−271.582428</td></tr>
<tr><th scope="row">Width U</th><td>3.464158</td><td>3.371878</td></tr>
<tr><th scope="row">Corrected Q1 − U</th><td>−274.999406</td><td>−274.954306</td></tr>
<tr><th scope="row">Measured Qπ</th><td>−216.149217</td><td>−222.155682</td></tr>
<tr><th scope="row">Command support</th><td>Common grid command</td><td>Outside the behavior grid</td></tr></tbody></table></div>
<p>The raw-score advantage of candidate 0 is <b>0.047180</b>. Candidate 7 receives a <b>0.092280</b> smaller penalty, enough to reverse that order. Its continuation return is <b>6.006465 lower</b>. Finite width therefore does not certify that the new choice is better or supported.</p>
<p>Across all 32 states, width improves 4 choices, harms 1 and leaves 27 unchanged. The mean gain is only <b>+0.044886 return units</b>. Counting improvements alone hides their sizes; the mean and the individual outcomes answer different questions.</p>
'''
    s = replace_once(s, '<h3 id="fr-controls">', example + '<h3 id="fr-controls">')
    s = replace_once(s, '</style>\n</head>', '.v6-terms{display:grid;grid-template-columns:minmax(100px,150px) minmax(0,1fr);gap:12px 18px;margin:0;max-width:80ch}.v6-terms dt{font-weight:700}.v6-terms dd{margin:0}.v6-scroll{overflow-x:auto;max-width:100%;min-width:0}.v6-compact{width:100%;border-collapse:collapse}.v6-compact th,.v6-compact td{padding:10px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}.v6-compact caption{text-align:left;color:var(--muted);padding-block:8px;font-size:15px}@media(max-width:560px){.v6-terms{grid-template-columns:1fr;gap:4px}.v6-terms dd{margin-bottom:12px}}\n</style>\n</head>')
    save(s, 2, 'Definitions, scope and a worked decision', ['Define actor, Q1, Qpi, width, signed error and distance', 'Explain relative error without percentage ambiguity', 'Show verified harmful action reversal', 'Mark old status and synthesis as historical', 'Repair source links and text spacing'])

def pass_three():
    import csv
    rows = {r['metric']: r for r in csv.DictReader((PRESENTATION.parent / 'results/metric_summary.csv').open())}
    s = read_current()
    s = replace_once(s, 'Fresh-bank mean alignment is negative; ranking scarcely changes', 'Fresh states: little ranking change, no demonstrated benefit from adaptive width')
    s = replace_once(s, 'tracks <b>relative signed value error</b>', 'tracks <b>differences in signed Q error across actions at the same state</b>')
    s = replace_once(s, '32 fresh designed states × 12 candidates = 384 one-action continuations.', 'The first designed bank: 32 states × 12 candidates = 384 one-action continuations.')
    s = replace_once(s, 'so the data no longer show the stall.', 'so examples of the stall become very sparse.')
    s = replace_once(s, 'The distance term used by the frozen control, subtracted with the inherited coefficient 0.519506430786272. This coefficient was not fitted again on the fresh bank.', 'The mean squared difference between the candidate’s four command coordinates and the behavior controller’s modal command. The coefficient 0.519506430786272 matches within-state score variation using eight separate adaptation states from the first diagnostic; it does not match gradients and was not retuned on this bank.')
    start = s.index('<div class="callout"><b>Completed:</b>', s.index('<section id="direct-q-replication"'))
    end = s.index('</div>', start) + len('</div>')
    s = s[:start] + '<div class="v6-note"><b>Completed:</b> 32 valid states × 12 candidates = 384 diagnostic continuations, using 21,696 simulator transitions. The networks stayed fixed. This added no training or separate validation run; the earlier replay-validation receipt was reused.</div>' + s[end:]
    start = s.index('<figure><svg', s.index('<section id="direct-q-replication"'))
    end = s.index('</figure>', start) + len('</figure>')
    cards = '''<div class="v6-result-cards" aria-label="Three different questions on the fresh-state bank">
<article><h3>1 · Does width track signed error?</h3><p class="v6-metric">−0.111290</p><p>Mean within-state correlation</p><p class="v6-interval">95% interval: [−0.231189, 0.002985]</p><p>The observed mean is negative, opposite to the intended direction; its interval includes zero.</p></article>
<article><h3>2 · Does it order pairs better?</h3><p class="v6-metric">71.58% → 71.63%</p><p>Correct pair ordering, raw → width</p><p class="v6-interval">Change: +0.0524 percentage points<br>95% interval: [−0.5653, +0.6214] points</p><p>Average pair ordering changes very little. This is not a relative percentage gain.</p></article>
<article><h3>3 · Does its chosen action improve?</h3><p class="v6-metric">+0.044886</p><p>Mean continuation-return change versus raw Q1</p><p class="v6-interval">95% interval: [−0.367432, 0.421841]</p><p>Four gains and one loss nearly cancel; 27 choices stay the same.</p></article>
</div><p class="note">Each interval describes variation across states for this one frozen model and designed candidate bank. A correlation, a fraction of correctly ordered pairs, and a return difference have different meanings and units. <a href="#fr-premise">Definitions</a> · <a href="#fr-uncertainty">Interval construction and limits</a>.</p>'''
    s = s[:start] + cards + s[end:]
    start = s.index('<p>On this fresh bank,')
    end = s.index('</p>',start)+4
    s = s[:start] + '''<p><b>No incremental benefit from adaptive width is demonstrated on this bank.</b> This is a descriptive result for one trained model, not a claim that all BCA variants must fail. The <a href="#fr-worked-example">worked choice below</a> shows why a penalty can change an action without improving its return.</p>
<p>The shuffled-width control gains +0.370231 selected return on average while its concordance changes by −0.004298. A score can worsen many unchosen pairs yet improve the one candidate it picks. The single fixed shuffle tests width placement descriptively; it is not a randomization test.</p>''' + s[end:]
    # Preserve all 33 original metric rows verbatim inside expandable evidence tables.
    for anchor, title in [('fr-alignment', 'All 5 alignment endpoints with exact intervals and denominators'), ('fr-ordering', 'All 10 ranking endpoints with exact intervals and denominators'), ('fr-choice', 'All 18 selected-return endpoints with exact intervals and denominators')]:
        a=s.index(f'<h3 id="{anchor}"')
        t=s.index('<div class="fr-table">',a)
        e=s.index('</table></div>',t)+len('</table></div>')
        s=s[:t]+f'<details class="v6-evidence"><summary>{title}</summary>'+s[t:e]+'</details>'+s[e:]
    s = insert_after(s, '<h3 id="fr-alignment">Does the width follow relative error?</h3>', '<p>A useful relative penalty would be larger for actions whose signed error is larger. Here the width’s mean correlation is −0.111290; the distance control’s is +0.014273. Their paired difference is −0.125563, with descriptive interval [−0.242948, −0.002111]. The width’s own interval, shown above, includes zero.</p>')
    s = insert_after(s, '<h3 id="fr-ordering">Does it order the actions better?</h3>', '<p>Raw Q1 orders 71.58% of comparable pairs correctly; subtracting width gives 71.63%. The distance control gives 72.59%, and shuffled width gives 71.15%. These are equal-state averages, not percentages from pooling every pair across the bank.</p>')
    body=[]
    for key,label in [('raw_q','Raw Q1'),('live_width','Q1 − width'),('matched_distance','Q1 − matched distance'),('shuffled_width','Q1 − shuffled width'),('constant_offset','Q1 − state-constant offset')]:
        r=rows['secondary_delta_vs_raw_'+key];lo,hi=json.loads(r['ci95'])
        body.append(f'<tr><th scope="row">{label}</th><td>{float(r["mean"]):+.6f}</td><td>[{lo:+.6f}, {hi:+.6f}]</td></tr>')
    summary = '<p><b>Higher return is better, including when returns are negative.</b> This compact table shows the paired change from the raw-Q1 choice, in simulator return units. Every row uses all 32 states.</p><div class="v6-scroll" tabindex="0" role="region" aria-label="Fresh-bank selected-action return changes"><table class="v6-compact"><caption>One candidate action, then the unchanged actor · all 32 states</caption><thead><tr><th scope="col">Scoring rule</th><th scope="col">Mean change</th><th scope="col">Descriptive 95% interval</th></tr></thead><tbody>'+''.join(body)+'</tbody></table></div><p>Width minus distance is −0.283109 return units, interval [−0.652308, −0.009318]. This paired contrast describes the fresh bank; it does not establish general superiority across trained models.</p>'
    s = insert_after(s, '<h3 id="fr-choice">What happens when a score chooses?</h3>', summary)
    # A simple invariant is easier to read once; exact accounting stays in each disclosure.
    s = insert_after(s, '<h3 id="fr-uncertainty">What the intervals can describe</h3>', '<p><b>Accounting in this bank:</b> all 32 planned states contribute to every displayed endpoint and every one of the 10,000 bootstrap draws; there are zero undefined draws. Exact counts remain in the full tables. The general missing-value rules below still apply.</p>')
    comparison = '''<h3 id="fr-bank-comparison">What changed between the two direct-Q banks?</h3>
<div class="v6-scroll" tabindex="0" role="region" aria-label="First and fresh direct-Q bank comparison"><table class="v6-compact"><caption>Separate banks, one inherited model · results are not pooled</caption><thead><tr><th scope="col">Item</th><th scope="col">First bank</th><th scope="col">Fresh-state bank</th></tr></thead><tbody>
<tr><th scope="row">Inputs</th><td>32 designed states × 12 candidates</td><td>New 32 states × 12 candidates</td></tr>
<tr><th scope="row">Training</th><td>One frozen checkpoint</td><td>The same checkpoint; no retraining</td></tr>
<tr><th scope="row">Alignment endpoint</th><td>Spearman with positive-part error; 10/32 states defined</td><td>Pearson with signed error; 32/32 states defined</td></tr>
<tr><th scope="row">Width: improved / harmed / unchanged choices</th><td>1 / 0 / 31</td><td>4 / 1 / 27</td></tr>
<tr><th scope="row">Width: mean return change versus raw</th><td>+0.153817</td><td>+0.044886</td></tr>
</tbody></table></div><p>The state inputs are new, but the trained model is shared and the primary alignment question was clarified after the first bank. The correlation statistics are therefore different endpoints, not two estimates of an unchanged test. The fresh bank’s rules were frozen before its own return labels.</p>'''
    s = replace_once(s, '<h3 id="fr-stages">', comparison + '<h3 id="fr-stages">')
    s = replace_once(s, '</style>\n</head>', '.v6-result-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,250px),1fr));gap:14px}.v6-result-cards article{background:var(--surface);border:1px solid var(--rule);border-top:3px solid var(--accent);border-radius:8px;padding:18px;display:grid;align-content:start;gap:10px;min-width:0}.v6-result-cards h3{font-size:18px;margin:0}.v6-metric{font-size:clamp(22px,2.5vw,30px);font-weight:700;line-height:1.2;font-variant-numeric:tabular-nums}.v6-interval{font-size:15px;color:var(--muted)}.v6-evidence summary{font-weight:600;cursor:pointer;padding:12px 0}.v6-evidence{border-top:1px solid var(--rule)}\n</style>\n</head>')
    save(s, 3, 'Results hierarchy and quantitative interpretation', ['Three readable endpoint cards replace mixed-scale chart', 'Compact selected-return comparison from saved CSV', 'All original metric rows retained in disclosures', 'Concordance expressed in percentage points', 'Cross-bank endpoint and training distinction', 'Correct sparse-stall wording and define distance matching'])

def pass_four():
    import html
    s=read_current()
    captions={
      'direct-q-replication':['Fresh bank: signed-error alignment', 'Fresh bank: pair-ordering concordance', None, 'Fresh bank: selected continuation returns and paired contrasts', 'Fresh bank: changed choices and actor restorations', None, None, 'Fresh bank: comparable pair counts and ties'],
      'direct-q-ranking':['First direct-Q bank: source and target contracts', 'First direct-Q bank: implementation stages', 'First direct-Q bank: selected-return changes', 'First direct-Q bank: support and defined error correlations', 'First direct-Q bank: signed-error tradeoffs', 'First direct-Q bank: complete state and action subgroup results'],
      'action-only-results':['Action-only test: stage-by-stage checks', 'Action-only test: coverage, finite coverage and refusal', 'Action-only test: paired weighting effects', 'Action-only test: effective sample size and positive-weight rows', 'Action-only test: refusal statuses', 'Action-only test: finite widths and downstream dose', 'Action-only test: complete subgroup results']}
    for section,names in captions.items():
        a=s.index(f'<section id="{section}"');b=s.index('</section>',a)+len('</section>')
        block=s[a:b]
        tables=list(re.finditer(r'<table\b[^>]*>.*?</table>',block,re.S))
        assert len(tables)==len(names),(section,len(tables))
        for i,m in reversed(list(enumerate(tables))):
            t=m.group(0)
            if names[i]:
                cap_id=f'v6-{section}-table-{i+1}'
                t=re.sub(r'(<table\b[^>]*>)',lambda x:x[1]+f'<caption id="{cap_id}">{names[i]}</caption>',t,count=1)
                # First-column labels are row headings, not unlabelled data cells.
                head,body=t.split('<tbody>',1)
                body=re.sub(r'<tr>\s*<td>(.*?)</td>',r'<tr><th scope="row">\1</th>',body,flags=re.S)
                t=head+'<tbody>'+body
                before=block[:m.start()]
                wrapper=re.search(r'<div class="(?:fr|dq|ao)-table">$',before)
                if wrapper:
                    before=before[:wrapper.start()]+wrapper[0][:-1]+f' tabindex="0" role="region" aria-labelledby="{cap_id}">'
                block=before+t+block[m.end():]
        # Keep fixed-coordinate chart text readable; captions remain outside the scrolling region.
        figure_count=0
        def figure(match):
            nonlocal figure_count
            figure_count+=1
            f=match.group(0)
            if '<svg' not in f:return f
            f=f.replace('<svg',f'<div class="v6-chart-scroll" tabindex="0" role="region" aria-label="{html.escape(section)} chart {figure_count}: scroll horizontally on a narrow screen"><svg',1)
            f=f.replace('</svg>','</svg></div>',1)
            return f
        block=re.sub(r'<figure\b[^>]*>.*?</figure>',figure,block,flags=re.S)
        s=s[:a]+block+s[b:]
    css='''
<style id="v6-accessibility-style">
/* Scoped improvements for the three diagnostics addenda. */
#direct-q-replication figure,#direct-q-ranking figure,#action-only-results figure{background:var(--surface);color:var(--ink)}
#direct-q-replication figcaption,#direct-q-ranking figcaption,#action-only-results figcaption{color:var(--ink-2);padding-top:10px}
.v6-chart-scroll{overflow-x:auto;max-width:100%;min-width:0;background:#fbfaf7;color:#1a201e;border-radius:6px}
#direct-q-ranking .v6-chart-scroll svg,#action-only-results .v6-chart-scroll svg{min-width:900px;width:100%;display:block}
#direct-q-replication table.v6-compact{min-width:0;width:100%;table-layout:fixed;font-variant-numeric:tabular-nums}
#direct-q-replication table.v6-compact th,#direct-q-replication table.v6-compact td{overflow-wrap:anywhere;white-space:normal;text-align:left}
#direct-q-replication table.v6-compact th:first-child{width:38%}
#direct-q-replication .fr-table,#direct-q-ranking .dq-table,#action-only-results .ao-table{max-width:100%;min-width:0;overscroll-behavior-inline:contain}
#direct-q-replication th[scope="row"],#direct-q-ranking th[scope="row"],#action-only-results th[scope="row"]{font-weight:600;text-align:left;vertical-align:top;white-space:normal}
#direct-q-replication .fr-source b,#direct-q-replication .fr-pseudo b{color:var(--accent-ink)}
:is(.v6-scroll,.v6-chart-scroll,.fr-table,.dq-table,.ao-table):focus-visible,.v6-evidence summary:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
.v6-evidence summary{min-height:44px;line-height:1.5}
#direct-q-replication .v6-result-cards h3{border:0;padding:0}
@media(max-width:560px){#direct-q-replication table.v6-compact th,#direct-q-replication table.v6-compact td{padding:8px 5px;font-size:14px;line-height:1.45}.v6-result-cards{grid-template-columns:1fr}.v6-intro{padding:16px}.v6-route li{padding-block:10px}.v6-intro p,.v6-result-cards p{text-align:left}}
@media print{.v6-result-cards{display:block}.v6-result-cards article{break-inside:avoid;margin-block:12px}.v6-chart-scroll{overflow:visible}#direct-q-ranking .v6-chart-scroll svg,#action-only-results .v6-chart-scroll svg{min-width:0}}
</style>
'''
    # A late style block wins over earlier addendum rules without global overrides.
    s=replace_once(s,'</body>',css+'</body>')
    save(s,4,'Accessible tables and resilient narrow-screen styling',['Captions and row headings across 18 evidence tables', 'Keyboard-focusable named horizontal scroll regions', 'Dark-mode figure captions use theme surfaces', 'Chart text keeps readable dimensions with horizontal scrolling', 'Compact tables override inherited wide minimums', 'Visible focus states and readable touch summaries'])

def pass_five():
    s=read_current()
    s=replace_once(s, 'all 32 planned states contribute to every displayed endpoint and every one of the 10,000 bootstrap draws; there are zero undefined draws.', 'all 32 planned states are eligible for every displayed endpoint. Each of the 10,000 bootstrap draws contains 32 defined state contributions, counting repeated sampled states; no draw is undefined.')
    s=replace_once(s, 'Larger relative error should receive a larger penalty if the penalty is to correct ordering.', 'For the penalty to correct ordering, it should be larger on the relatively overvalued action: the one with the larger signed error at that same state.')
    old='Pearson over all 12 indexed candidates equals the normalized dot product of every i&lt;j difference, without recentering the differences. That equality is a code invariant, not a second independent piece of evidence.'
    s=replace_once(s,old,'</p><details class="v6-evidence"><summary>Why the pair-difference calculation gives the same correlation</summary><p>'+old+'</p></details><p class="note">The main result is one within-state correlation per defined state, averaged equally across states.')
    s=replace_once(s,'<h3 id="fr-alignment">Does the width follow relative error?</h3>','<h3 id="fr-alignment">Does a larger width accompany a larger signed error?</h3>')
    s=replace_once(s,'<b>Source:</b> The historical band uses 128 pooled calibration rows:', '<b>Source:</b> The historical band pools 128 calibration rows with equal weight:')
    # Make boundaries between sampling contracts visible while preserving every condition.
    a=s.index('<h3 id="fr-laws">');b=s.index('<h3 id="fr-alignment">',a)
    block=s[a:b].replace(' <b>Target inputs:</b>','</p><p><b>Target inputs:</b>').replace(' <b>Truth target:</b>','</p><p><b>Truth target:</b>')
    s=s[:a]+block+s[b:]
    start=s.index('<p>Conditional-action support counts:')
    end=s.index('<h3 id="fr-uncertainty">',start)
    s=s[:start]+'''<ul class="v6-support">
<li><b>47 common grid commands:</b> behavior probability at least 0.01.</li>
<li><b>145 rare grid commands:</b> behavior probability greater than 0 and less than 0.01.</li>
<li><b>192 unsupported commands:</b> behavior probability is zero under exact float32 grid equality.</li>
</ul>
<p>These 384 counts describe conditional action support on the fresh bank. They do not certify state coverage or a valid uncertainty band. There were <b>0 width refusals</b>, <b>0 invalid states</b>, <b>0 physical/continuation failures</b> and <b>0 nonfinite returns</b>.</p>
<details class="v6-evidence"><summary>How refusals and invalid records are handled</summary><p>A finite unusable width falls back to the raw score, so refusing a band does not refuse the action. A numeric invalidity excludes the whole state from paired endpoint contributions, preserves every record and never triggers a replacement. Physical threshold events remain valid measured outcomes. No invalid-state reasons were recorded in this bank.</p></details>
''' +s[end:]
    s=replace_once(s,'<p>No hypothesis test, independent-seed generalization, new learned-policy benefit, long-horizon Q coverage or additional penalty variant is claimed.</p>', '<p class="v6-note"><b>What remains unresolved:</b> whether the signal helps across independently trained models, whether repeatedly selecting a corrected-score action improves a deployed controller, and whether a band covers long-horizon Qπ. This experiment measures none of those outcomes. Its intervals are descriptive, and it supplies no hypothesis-test or automatic training decision.</p>')
    # Correct a historical introduction that phrased a nominal coverage target as achieved coverage.
    s=s.replace('sized on recorded rows kept out of training so that the critic&#x27;s one-step error falls inside it on 90% of them. A wide band marks a place where the critic may be wrong.', 'calibrated on recorded rows kept out of training with a nominal 90% one-step residual-coverage target. This setting does not establish achieved coverage on shifted queries or for long-horizon value. A wide band is intended to signal greater uncertainty.')
    s=replace_once(s, 'sized on recorded rows kept out of training so that it\n            contains the critic\'s', 'calibrated on recorded rows kept out of training with a nominal 90% coverage target for the critic\'s')
    s=replace_once(s, 'own value of the next state) 90% of the time. Where the band is wide it adds caution:', 'own value of the next state). That target does not establish achieved coverage after distribution shift. Where the band is wide it adds caution:')
    s=s.replace('<th scope="col">Paired95% interval</th>','<th scope="col">Paired 95% interval</th>')
    s=insert_after(s, '<h3 id="fr-bank-comparison">What changed between the two direct-Q banks?</h3>', '<p>In the first bank, the one improved choice returns to the original actor’s own command. In the fresh bank, five commands change and one is worse. Both tests evaluate one action followed by the original actor; neither trains or deploys a new policy.</p>')
    s=insert_after(s,'<div class="v6-intro" id="evidence-overview" aria-labelledby="v6-overview-heading">', '<p class="note">Reading guide: <a href="#direct-q-replication">latest result</a> · <a href="#fr-worked-example">worked action choice</a> · <a href="#fr-bank-comparison">what changed between banks</a> · <a href="#fr-uncertainty">limits of the evidence</a></p>')
    save(s,5,'Independent review corrections and final clarity',['Distinguish bootstrap draws from unique states', 'Move mathematical identity into optional detail', 'Separate source, inputs and truth contracts', 'Replace raw support JSON with labeled counts', 'Correct nominal versus achieved coverage wording', 'State unresolved questions beside the evidence', 'Add direct reading-guide links'])

def pass_six():
    s=read_current()
    s=replace_once(s, 'Compare the original learner, native BCA’s stronger pull toward recorded actions, and the most-caution control. This is the completed 30-job comparison.', 'Compare TD3+BC with native TD3+BC+BCA across three data-retention settings and five training replicates: 2 × 3 × 5 = 30 completed jobs.')
    s=replace_once(s, '<p>The width changes 5 commands: 4 improve, 1 is harmed and 27 are unchanged versus raw Q. Mean selected-Q change is 0.044886, interval [-0.367432, 0.421841]. The direct width-minus-distance selected-Q difference is -0.283109, interval [-0.652308, -0.009318].</p>', '')
    s=replace_once(s, '<p>The shuffled-width control has mean selected-Q change 0.370231 versus raw, despite mean concordance change -0.004298. Average pair ordering and the value of the single maximum-score choice are different endpoints. One fixed permutation is a descriptive placement control, not a randomization test.</p>', '')
    save(s,6,'Final independent reading and correction',['Correct the 30-job matrix to two arms by three retention settings by five replicates', 'Remove repeated width-choice and shuffled-control summaries while retaining every value in evidence tables'])

if __name__ == '__main__':
    globals()['pass_' + {'1':'one','2':'two','3':'three','4':'four','5':'five','6':'six'}[sys.argv[1]]]()
