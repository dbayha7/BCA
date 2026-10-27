"""Build fresh-bank replication reports from authenticated saved records only.

This script does not import model/simulator code, produce labels or retrain anything.
Run after the root has confirmed completed execution: python report_assets/build_report.py
"""
from collections import Counter
import ast
import csv
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report_assets'
PRESENTATION = ROOT / 'presentation'
V4 = ROOT.parent / 'direct_q_diagnostic_20261021_v2/presentation_reviewed/ThrusterChain_diagnostics_results_v4_direct_Q.html'
V4_SHA = '4e7bc666b8df074d576599c3d1dd1c66620fc3c537702f63504bd23fa1aef6c0'
METHODS = ('raw_q', 'live_width', 'matched_distance', 'shuffled_width', 'constant_offset')
LABELS = {'raw_q': 'Raw Q1', 'live_width': 'Q1 minus width', 'matched_distance': 'Q1 minus matched distance',
          'shuffled_width': 'Q1 minus shuffled width', 'constant_offset': 'Q1 minus state-constant offset'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def fmt(value, places=6):
    if value is None or not isinstance(value, (float, int)) or not math.isfinite(value):
        return 'undefined'
    return f'{value:.{places}f}'


def interval(record):
    bounds = record.get('ci95')
    return '['+', '.join(fmt(x) for x in bounds)+']' if bounds else 'unavailable'


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def html_table(headers, rows):
    return '<div class="fr-table"><table><thead><tr>'+''.join('<th scope="col">'+escape(x)+'</th>' for x in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def authenticate():
    protocol, freeze = read(ROOT/'protocol.json'), read(ROOT/'freeze.json')
    preparation = read(ROOT/'prepared/preparation_manifest.json')
    result, status = read(ROOT/'results/paired_results.json'), read(ROOT/'results/status.json')
    assert status['status'] in ('complete', 'complete_with_invalid_states'), 'execution is not terminal'
    assert status['parameters_before'] == status['parameters_after'] == preparation['parameters_before'] == preparation['parameters_after']
    assert status['new_training_updates'] == status['new_validation_transitions'] == 0
    assert status['main_transitions'] <= protocol['truth']['planned_transitions']
    assert freeze['preparation_manifest_sha256'] == sha(ROOT/'prepared/preparation_manifest.json')
    for name, expected in preparation['source_sha256'].items():
        assert sha(ROOT/name) == expected == freeze['source_sha256'][name], name
    for name, expected in preparation['artifact_sha256'].items():
        assert sha(ROOT/'prepared'/name) == expected, name
    for name, expected in status['output_sha256'].items():
        assert sha(ROOT/'results'/name) == expected, name
    assert sha(V4) == V4_SHA, 'reviewed V4 baseline changed'
    with (ROOT/'results/candidate_records.csv').open(encoding='utf-8', newline='') as file:
        candidates = list(csv.DictReader(file))
    assert len(candidates) == 384 and len({(x['state_id'], x['candidate_id']) for x in candidates}) == 384
    assert len(result['state_records']) == 32
    assert {x['state_id'] for x in result['state_records']} == set(range(32))
    return protocol, freeze, preparation, result, status, candidates


def source_map():
    plan = [('authenticate', 'run_replication.py', 'authenticate'),
            ('states', 'replication.py', 'generate_states'), ('bank', 'replication.py', 'prepare_bank'),
            ('width', 'replication.py', 'read_native_outputs'), ('scores', 'replication.py', 'score_arrays'),
            ('truth', 'run_replication.py', 'rollout_saved_inputs'),
            ('pearson', 'replication.py', 'pearson'), ('pairs', 'replication.py', 'pairwise_pearson'),
            ('concordance', 'replication.py', 'concordance'), ('state-metrics', 'replication.py', 'state_metrics'),
            ('draws', 'replication.py', 'bootstrap_plan'), ('bootstrap', 'replication.py', 'bootstrap_statistic'),
            ('analysis', 'replication.py', 'analyze')]
    refs = {}
    for key, name, function in plan:
        text = (ROOT/name).read_text(encoding='utf-8')
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == function)
        refs[key] = {'file': name, 'function': function, 'first_line': node.lineno, 'last_line': node.end_lineno,
                     'source_sha256': sha(ROOT/name), 'text': '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])}
    return refs


def rows_for_metrics(result, names):
    output = []
    for label, name in names:
        value = result['metrics'][name]
        output.append([label, fmt(value['mean']), interval(value), f"{value['defined_states']}/{value['planned_states']}",
                       f"{value['bootstrap_defined_denominator_min']}..{value['bootstrap_defined_denominator_max']}",
                       value['undefined_bootstrap_draws']])
    return output


def tables(result):
    alignment = [('Width versus signed error', 'primary_rho_width'), ('Distance versus signed error', 'primary_rho_distance'),
                 ('Shuffled width versus signed error', 'primary_rho_shuffled'),
                 ('Width minus distance correlation', 'primary_rho_width_minus_distance'),
                 ('Width minus shuffled correlation', 'primary_rho_width_minus_shuffled')]
    decisions = []
    for method in METHODS:
        decisions.extend([(LABELS[method]+' concordance', 'concordance_'+method),
                          (LABELS[method]+' concordance change', 'primary_concordance_change_'+method)])
    selection = []
    for method in METHODS:
        selection.extend([(LABELS[method]+' selected true Q', 'secondary_chosen_true_q_'+method),
                          (LABELS[method]+' change versus raw', 'secondary_delta_vs_raw_'+method),
                          (LABELS[method]+' advantage versus actor', 'secondary_advantage_vs_actor_'+method)])
    selection.extend([('Original actor true Q', 'secondary_original_actor_true_q'),
                      ('Width minus distance: selected true Q', 'secondary_selected_q_live_width_minus_matched_distance'),
                      ('Width minus shuffled: selected true Q', 'secondary_selected_q_live_width_minus_shuffled_width')])
    return {key: rows_for_metrics(result, value) for key, value in [('alignment', alignment), ('concordance', decisions), ('selection', selection)]}


HEADERS = ['Endpoint', 'Mean', '95% interval', 'Defined / planned states', 'Defined states per bootstrap draw', 'Undefined draws / 10000']
SOURCE_LAW = ('The historical band uses 128 pooled calibration rows: one uniformly selected transition per reserved behavior episode. '
              'Under full retention, time is uniform over 0..99 and physical states follow behavior trajectories. '
              'The 625-command product categorical behavior law gives each coordinate probability 0.84 at its mode and 0.04 at each other atom. '
              'These rows were adaptively reused; the frozen reference was last refreshed at update 95000. '
              'Its target is a noisy one-step residual for the current twin-min critic.')
TARGET_LAW = ('The new bank has 32 fresh independent physical-state draws, eight per stall/monotone and low/high-load context. '
              'Times 0, 50, 99 have counts 3, 3, 2 in each context. Ten position/velocity coordinates are uniform on [-0.5, 0.5]; '
              'load is 0.3 or 1.1 and wave amplitude 0.2 has uniform phase. These reset-law states crossed with assigned times '
              'are designed queries, not states visited by the behavior or learned policy. '
              'Each state has six distinct grid commands: include the nearest-grid actor command and behavior mode when distinct, then fill by uniform grid IDs without replacement. '
              'Append the original continuous actor command and five Uniform[-1,1]^4 commands. All 12 indices remain, including cross-slot duplicates.')
TRUTH = ('A true-Q label executes the candidate once, then follows the unchanged unsmoothed original actor to horizon 100 with discount 1 and no terminal bootstrap. '
         'It is the deterministic numerical return of this one-action deviation, not Q-star or a new policy that repeatedly selects corrected-score actions.')
UNCERTAINTY = ('One common saved set of 10,000 bootstrap draws resamples whole states within the original context/time cells, retaining the 3/3/2 counts. '
               'Actions and action pairs are not independent replications. Each mean correlation uses its own defined-state mask; '
               'each correlation contrast uses its pair-specific intersection. The same eligibility mask is applied after drawing from all 32 original states; '
               'missing states are not replaced or resampled away. Every draw denominator is retained. Any draw with no defined contributions makes that endpoint interval unavailable. '
               'These percentile intervals are descriptive and conditional on this one checkpoint, designed state law and candidate realization; '
               'there are only 2-3 states per context/time cell. They do not quantify variation across independently trained models.')


def count_details(result, candidates):
    support = dict(Counter(row['support_class'] for row in candidates))
    pairs = {method: {'comparable_pairs': sum(x['methods'].get(method, {}).get('comparable_pairs', 0) for x in result['state_records']),
                      'true_near_ties': sum(x['methods'].get(method, {}).get('true_near_ties', 0) for x in result['state_records']),
                      'predicted_near_ties': sum(x['methods'].get(method, {}).get('predicted_near_ties', 0) for x in result['state_records'])}
             for method in METHODS}
    signed = {}
    for method in METHODS:
        values = [x['methods'][method]['delta_vs_raw'] for x in result['state_records'] if x['valid']]
        signed[method] = {'improved': sum(x > 0 for x in values), 'harmed': sum(x < 0 for x in values), 'unchanged': sum(x == 0 for x in values)}
    return {'support': support, 'pair_counts': pairs, 'selected_value_change_counts': signed}


def chart(result):
    names = [('Width - distance correlation', 'primary_rho_width_minus_distance'),
             ('Width - shuffled correlation', 'primary_rho_width_minus_shuffled'),
             ('Width concordance change', 'primary_concordance_change_live_width')]
    bounds = [0.0]
    for _, name in names:
        m = result['metrics'][name]
        bounds.extend(x for x in [m['mean'], *(m.get('ci95') or [])] if x is not None)
    lo, hi = min(bounds), max(bounds)
    span = max(hi-lo, .1); lo -= .12*span; hi += .12*span
    x = lambda v: 310+(v-lo)/(hi-lo)*510
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 245" role="img" aria-labelledby="fr-chart-title fr-chart-desc">',
           '<title id="fr-chart-title">Fresh-bank alignment contrasts and ranking change</title>',
           '<desc id="fr-chart-desc">Points are state means; lines are descriptive95percent bootstrap intervals. Zero marks no paired difference.</desc>',
           '<rect width="900" height="245" fill="#fbfaf7"/>',
           f'<line x1="{x(0):.2f}" y1="24" x2="{x(0):.2f}" y2="183" stroke="#777" stroke-dasharray="4 4"/>']
    for i, (label, name) in enumerate(names):
        m = result['metrics'][name]; y = 45+55*i
        svg.append(f'<text x="15" y="{y+5}" font-family="sans-serif" font-size="16" fill="#222">{escape(label)}</text>')
        if m.get('ci95'):
            a, b = m['ci95']; svg.append(f'<line x1="{x(a):.2f}" y1="{y}" x2="{x(b):.2f}" y2="{y}" stroke="#147d73" stroke-width="3"/>')
        if m['mean'] is not None:
            svg.append(f'<circle cx="{x(m["mean"]):.2f}" cy="{y}" r="6" fill="#147d73"/>')
    svg += [f'<text x="{x(lo):.2f}" y="210" font-family="sans-serif" font-size="14">{lo:.3f}</text>',
            f'<text x="{x(0):.2f}" y="210" text-anchor="middle" font-family="sans-serif" font-size="14">0</text>',
            f'<text x="{x(hi):.2f}" y="210" text-anchor="end" font-family="sans-serif" font-size="14">{hi:.3f}</text>', '</svg>']
    return '\n'.join(svg)


def outcome_paragraphs(result, details):
    m = result['metrics']; changes = details['selected_value_change_counts']['live_width']
    return [
        'On this fresh bank, the mean width/error relationship runs opposite to the intended relative-error penalty direction, while the change in action-ranking concordance is close to zero. No incremental benefit from adaptive width is demonstrated on this bank. This is not an impossibility result or a test of all importance-weighted BCA variants; no broad training follows from this report.',
        f"The width changes {result['counts']['live_width']['changed_command']} commands: {changes['improved']} improve, {changes['harmed']} is harmed and {changes['unchanged']} are unchanged versus raw Q. Mean selected-Q change is {fmt(m['secondary_delta_vs_raw_live_width']['mean'])}, interval {interval(m['secondary_delta_vs_raw_live_width'])}. The direct width-minus-distance selected-Q difference is {fmt(m['secondary_selected_q_live_width_minus_matched_distance']['mean'])}, interval {interval(m['secondary_selected_q_live_width_minus_matched_distance'])}.",
        f"The shuffled-width control has mean selected-Q change {fmt(m['secondary_delta_vs_raw_shuffled_width']['mean'])} versus raw, despite mean concordance change {fmt(m['primary_concordance_change_shuffled_width']['mean'])}. Average pair ordering and the value of the single maximum-score choice are different endpoints. One fixed permutation is a descriptive placement control, not a randomization test."
    ]


def markdown(protocol, result, status, rows, details):
    m = result['metrics']
    lead = (f"Fresh-bank mean width/signed-error correlation: {fmt(m['primary_rho_width']['mean'])}; "
            f"width-minus-distance correlation: {fmt(m['primary_rho_width_minus_distance']['mean'])} "
            f"{interval(m['primary_rho_width_minus_distance'])}; width-minus-shuffled correlation: "
            f"{fmt(m['primary_rho_width_minus_shuffled']['mean'])} {interval(m['primary_rho_width_minus_shuffled'])}.")
    content = ['# Fresh-bank mean alignment is negative; ranking scarcely changes', '', *[item for paragraph in outcome_paragraphs(result,details) for item in (paragraph,'')], lead, '',
        f"Completed {status['completed_main_returns']}/384 candidate continuations, {status['main_transitions']}/21696 planned plant transitions. "
        f"Valid states: {result['valid_states']}/32; invalid states: {result['invalid_states']}. New training updates and new validation transitions: 0.", '',
        'This separate fresh bank was specified after the earlier 32-state bank was inspected, then frozen before its own return labels. '
        'It replicates a frozen-model diagnostic, not independent training seeds or a confirmatory learner comparison. Earlier results remain unchanged and are not pooled.', '',
        '## What is being checked', '',
        'BCA means the existing Bayesian calibration procedure. Its native use changes behavior-cloning strength. '
        'This diagnostic instead ranks fixed commands by Q1 minus the exported uncertainty width. '
        'The earlier mechanical tests verified the proposed live-width gradient path; the present run performs no gradient step.', '',
        'The primary error is signed e=Q1-Qpi, so a larger penalty should track larger relative overestimation within a state. '
        'This differs from the earlier positive-error Spearman summary. Pearson correlation of all indexed candidates equals the normalized dot product of all i<j differences, without recentering those differences. '
        'The equivalent calculation is an identity check, not a second independent result.', '',
        '## Source and target laws', '', SOURCE_LAW, '', TARGET_LAW, '', TRUTH, '',
        'The historical target, evaluator inputs and long-return labels differ. A finite width is not a long-horizon Q-coverage certificate. '
        'The nominal 90% one-step calibration setting is frozen metadata, not a coverage result of this replication.', '',
        '## Signed-error alignment', '', markdown_table(HEADERS, rows['alignment']), '',
        'Each standalone mean uses its own defined states. Paired differences use the pair-specific common intersection; paired arm means and all masks/denominators remain in paired_results.json.', '',
        '## Ranking concordance', '',
        'Concordance is the fraction of comparable true action pairs ordered correctly. True differences at most 1e-6 are excluded; predicted differences at most 1e-6 earn half credit. '
        'The aggregate is an equal-state mean, not a pooled-pair estimate.', '', markdown_table(HEADERS, rows['concordance']), '',
        '## Selected action consequences', '', markdown_table(HEADERS, rows['selection']), '',
        'Each chosen command is executed once and then control returns to the original actor. Changes of candidate index and actual command are counted separately. '
        'Selecting the original actor after raw Q chose another command is restoration, not evidence that a newly trained policy improved.', '',
        markdown_table(['Score', 'Changed indices', 'Changed commands', 'Selected actor command', 'Actor restorations', 'Improved / harmed / unchanged versus raw'],
            [[LABELS[key], result['counts'][key]['changed_index'], result['counts'][key]['changed_command'], result['counts'][key]['selected_actor_command'],
              result['counts'][key]['actor_restoration'], ' / '.join(str(details['selected_value_change_counts'][key][x]) for x in ('improved','harmed','unchanged'))] for key in METHODS]), '',
        '## Controls, support and refusal', '',
        f"Width coefficient remains 1; inherited distance coefficient remains {protocol['frozen']['distance_coefficient']}. Neither is retuned. "
        'One fixed within-state permutation preserves the 12 indexed widths while changing their placement; it is a control, not a permutation test. '
        'The state-constant offset must preserve raw-Q ordering.', '',
        f"Conditional-action support counts: {json.dumps(details['support'],sort_keys=True)}. Native width refusals: {result['native_width_refusals']}/384. "
        f"Evaluated failure events: {status['failure_events_evaluated']}/{status['evaluated_main_returns']}; nonfinite evaluated returns: {status['nonfinite_evaluated_returns']}/{status['evaluated_main_returns']}.", '',
        'Common grid actions have beta>=0.01; rare grid actions have 0<beta<0.01; unsupported means beta=0 under exact float32 grid equality. '
        'These labels do not certify state/calibration overlap. A finite but unusable native width falls back to raw scoring; refusal does not reject an action. '
        'Invalid numeric input or return invalidates the whole state across arms without replacement. Physical threshold events remain measured outcomes.', '',
        'Invalid-state reasons: '+json.dumps(result['invalid_reasons'],sort_keys=True)+'.', '',
        '## What the intervals mean', '', UNCERTAINTY, '',
        'No hypothesis test, policy-benefit conclusion, Q coverage test, checkpoint selection or extra variants were introduced.', '',
        'Rebuild this report and V5 without model calls: `python report_assets/build_report.py`. '
        'Reproduce the numerical analysis from saved arrays with the frozen `reproduce_analysis.py` command documented in README.md. '
        'The report checks source/input/result hashes before writing. V5 preserves V4 apart from this new section and its navigation links. '
        'Static HTML/source-link checks passed; rendered browser, navigation and runtime JavaScript QA was not performed for V5.', '']
    return '\n'.join(content)


def source_panels(refs):
    panels = []
    for key, r in refs.items():
        code = '\n'.join('<span><b>'+str(r['first_line']+i).zfill(4)+'</b> '+escape(line)+'</span>' for i,line in enumerate(r['text'].splitlines()))
        panels.append(f'<h4 id="fr-code-{key}">{escape(r["file"])}:{r["first_line"]}-{r["last_line"]} — {escape(r["function"])}</h4><pre class="fr-source">{code}</pre>')
    return '\n'.join(panels)


def fragment(protocol, result, status, rows, details, refs, plot):
    m = result['metrics']
    def link(key):
        r = refs[key]
        return f'<a href="#fr-code-{key}">{r["file"]}:{r["first_line"]}-{r["last_line"]}</a>'
    numbered = [
        ('01', 'Authenticate', 'Lock the existing 100k model, 95k residual reference, inherited coefficient, source and fresh inputs before labels.', 'authenticate'),
        ('02', 'Design the fresh bank', 'Draw 32 new physical states; assign the same context/time law and 12 indexed candidates.', 'states'),
        ('03', 'Fix the placement control', 'Permute the12 widths once within each state; preserve their usability flags and full produced precision.', 'bank'),
        ('04', 'Score every command', 'Compare raw Q1, Q1-U, Q1-kD, Q1-shuffled(U), and Q1-state_mean(U).', 'scores'),
        ('05', 'Measure the consequence', 'Execute each candidate once, then follow the unchanged single-row actor to horizon 100.', 'truth'),
        ('06', 'Check relative error', 'Compute within-state Pearson(width, signed Q1 error); verify the equivalent all-pairs difference formula.', 'state-metrics'),
        ('07', 'Check decisions', 'Retain comparable-pair counts, selected trueQ, raw/control/actor contrasts and every invalid state.', 'concordance'),
        ('08', 'Quantify conditional variation', 'Use one saved whole-state bootstrap plan; retain endpoint masks and undefined draws.', 'analysis')]
    pseudo = '\n'.join(f'<span><b>{num} · {escape(title)}</b> {escape(text)} {link(key)}</span>' for num,title,text,key in numbered)
    counts = [[LABELS[k], result['counts'][k]['changed_index'], result['counts'][k]['changed_command'], result['counts'][k]['selected_actor_command'],
              result['counts'][k]['actor_restoration'], ' / '.join(str(details['selected_value_change_counts'][k][x]) for x in ('improved','harmed','unchanged'))] for k in METHODS]
    return f'''<style>
#direct-q-replication .fr-table{{overflow-x:auto;margin:18px 0}}#direct-q-replication table{{min-width:760px;width:100%}}
#direct-q-replication td{{vertical-align:top;overflow-wrap:anywhere}}#direct-q-replication .fr-pseudo,#direct-q-replication .fr-source{{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;padding:18px;background:var(--surface)}}
#direct-q-replication .fr-pseudo span,#direct-q-replication .fr-source span{{display:block}}#direct-q-replication .fr-pseudo b,#direct-q-replication .fr-source b{{color:#167d73;display:inline-block;margin-right:1em}}#direct-q-replication .fr-source{{font-size:12px}}
#direct-q-replication figure{{padding:12px;background:#fbfaf7;border:1px solid var(--rule);border-radius:10px}}#direct-q-replication svg{{width:100%;height:auto}}#direct-q-replication details{{margin:18px 0}}#direct-q-replication .fr-meta{{font-size:12px;overflow-wrap:anywhere}}
</style>
<section id="direct-q-replication" aria-labelledby="fr-heading">
<h2 id="fr-heading">Fresh-bank mean alignment is negative; ranking scarcely changes</h2>
<p>This new bank tests whether the frozen uncertainty width tracks <b>relative signed value error</b>, then whether subtracting it improves ordering or action choice. It was specified after the earlier bank was inspected and frozen before its own labels. This is a descriptive frozen-model replication; earlier results remain below, unchanged and unpooled.</p>
<div class="callout"><b>Completed:</b> {status['completed_main_returns']}/384 candidate continuations; {status['main_transitions']}/21696 planned transitions; {result['valid_states']}/32 valid states. No new training or validation rollouts. Mean width/error correlation <b>{fmt(m['primary_rho_width']['mean'])}</b>; width-minus-distance correlation <b>{fmt(m['primary_rho_width_minus_distance']['mean'])}</b>, interval <b>{interval(m['primary_rho_width_minus_distance'])}</b>.</div>
<figure>{plot}<figcaption>These endpoints answer different questions. The first two compare alignment with signed error; the third measures ranking change. Intervals are conditional state-bootstrap descriptions, not independent-training-seed evidence.</figcaption></figure>
{''.join('<p>'+escape(paragraph)+'</p>' for paragraph in outcome_paragraphs(result,details))}
<h3 id="fr-premise">Define the premise before the mechanism</h3>
<p>BCA is the existing Bayesian calibration procedure. Its native intervention changes behavior-cloning strength. Here we study the proposed direct score <b>Q1 minus uncertainty width</b>, without updating either network. Earlier tests established the proposed live-width derivative mechanically; a derivative alone does not establish useful uncertainty.</p>
<p>The primary error is <b>e = Q1 − true Q of the original actor continuation</b>. Larger relative error should receive a larger penalty if the penalty is to correct ordering. This is signed-error <b>Pearson</b> alignment; it is a different endpoint from the earlier positive-error Spearman analysis. Pearson over all 12 indexed candidates equals the normalized dot product of every i&lt;j difference, without recentering the differences. That equality is a code invariant, not a second independent piece of evidence.</p>
<h3 id="fr-laws">Historical calibration source → new evaluator target</h3>
<p><b>Source:</b> {escape(SOURCE_LAW)}</p><p><b>Target inputs:</b> {escape(TARGET_LAW)}</p><p><b>Truth target:</b> {escape(TRUTH)}</p>
<p>The frozen nominal 90% setting concerns the historical one-step residual target. This replication does not test interval coverage for long-horizon Q. Conditional command support and finite width do not establish joint state/calibration coverage.</p>
<h3 id="fr-alignment">Does the width follow relative error?</h3>{html_table(HEADERS, rows['alignment'])}
<p>Each mean uses its own defined states; each paired correlation contrast uses its pair-specific intersection. A zero centered predictor or error norm gives an undefined correlation, never an artificial zero. Paired arm means on the same intersection remain in the saved numerical results.</p>
<h3 id="fr-ordering">Does it order the actions better?</h3>
<p>Concordance is the fraction of comparable true action pairs ordered correctly. True differences at most 10<sup>−6</sup> are excluded; predicted differences at most 10<sup>−6</sup> earn half credit. Means give equal weight to each defined state, not to each pooled pair.</p>{html_table(HEADERS, rows['concordance'])}
<h3 id="fr-choice">What happens when a score chooses?</h3>{html_table(HEADERS, rows['selection'])}
<p>The chosen command runs once, then control returns to the original actor. The table reports both selected-value levels and paired differences against raw, distance, shuffled-width and original-actor controls. These are not returns from a new controller that repeatedly chooses the corrected score.</p>
{html_table(['Score','Changed indices','Changed commands','Selected actor command','Actor restorations','Improved / harmed / unchanged versus raw'], counts)}
<p>An index change can select the same command when candidates duplicate. Returning to the original actor after raw Q chose another command is restoration, not new learned-policy improvement.</p>
<h3 id="fr-controls">Frozen controls, support and refusals</h3>
<p>The width coefficient stays 1 and the inherited distance coefficient stays {protocol['frozen']['distance_coefficient']}; neither is retuned. One saved within-state permutation changes width placement while preserving its indexed values. It is a placement control, not a permutation test. The state-constant subtraction preserves raw ordering.</p>
<p>Conditional-action support counts: <code>{escape(json.dumps(details['support'],sort_keys=True))}</code>. Native width refusals: <b>{result['native_width_refusals']}/384</b>. Invalid states: <b>{result['invalid_states']}/32</b>. Evaluated physical/continuation failures: <b>{status['failure_events_evaluated']}/{status['evaluated_main_returns']}</b>; nonfinite evaluated returns: <b>{status['nonfinite_evaluated_returns']}/{status['evaluated_main_returns']}</b>.</p>
<p>Common commands have β≥0.01; rare commands have 0&lt;β&lt;0.01; unsupported commands have β=0 under exact float32 atom equality. These are conditional action labels. A finite unusable width falls back to raw scoring; band refusal is not action refusal. A numeric invalidity removes the whole state from paired endpoint contributions, preserves every record and never triggers a replacement. Physical threshold events remain valid measured outcomes.</p>
<p>Invalid-state reasons: <code>{escape(json.dumps(result['invalid_reasons'],sort_keys=True))}</code>.</p>
<h3 id="fr-uncertainty">What the intervals can describe</h3><p>{escape(UNCERTAINTY)}</p>
<p>No hypothesis test, independent-seed generalization, new learned-policy benefit, long-horizon Q coverage or additional penalty variant is claimed.</p>
<h3 id="fr-stages">Trace the actual implementation</h3><pre class="fr-pseudo">{pseudo}</pre>
<details><summary>Exact pair denominators and undefined records</summary>{html_table(['Score','Comparable pairs across states','True near-ties','Predicted near-ties among comparable pairs'], [[LABELS[k],v['comparable_pairs'],v['true_near_ties'],v['predicted_near_ties']] for k,v in details['pair_counts'].items()])}<pre>{escape(json.dumps({'individual_rho_defined':result['individually_defined_rho_states'],'paired_rho_defined':result['pair_common_defined_rho_states'],'invalid_reasons':result['invalid_reasons']},indent=2))}</pre></details>
<details><summary>Actual numbered source for each stage</summary>{source_panels(refs)}</details>
<p><b>Reproduce from saved records:</b> <code>python report_assets/build_report.py</code> rebuilds this report, plot and V5 without model inference or new labels. The separate frozen <code>reproduce_analysis.py</code> reproduces statistics from saved arrays.</p>
<p class="fr-meta">Checkpoint SHA256 <code>{protocol['frozen']['checkpoint_sha256']}</code>. Protocol SHA256 <code>{sha(ROOT/'protocol.json')}</code>. Result SHA256 <code>{sha(ROOT/'results/paired_results.json')}</code>. Static HTML/source-link checks passed. Rendered browser, navigation and runtime JavaScript QA was not performed for V5.</p>
</section>'''


class Tags(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.hrefs=[]; self.scripts=0
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if 'href' in a:self.hrefs.append(a['href'])
        if tag=='script':self.scripts+=1


def build_v5(section):
    original=V4.read_bytes(); text=original.decode('utf-8')
    needle='<section id="direct-q-ranking"'
    assert text.count(needle)==1
    marker='<!-- BEGIN FRESH REPLICATION ADDENDUM -->\n'+section+'\n<!-- END FRESH REPLICATION ADDENDUM -->\n'
    text=text.replace(needle,marker+needle,1)
    old='<a href="#direct-q-ranking" data-unit="direct-q-ranking">'
    assert text.count(old)==1
    nav='<li class="nv-sec"><a href="#direct-q-replication" data-unit="direct-q-replication">Fresh-bank value-penalty replication</a></li>\n'
    anchor='<li class="nv-sec">'+old
    assert text.count(anchor)==1
    text=text.replace(anchor,nav+anchor,1)
    assert text.replace(marker,'',1).replace(nav,'',1).encode('utf-8')==original
    new=Tags();new.feed(section); full=Tags();full.feed(text); prior=Tags();prior.feed(original.decode('utf-8'))
    assert len(new.ids)==len(set(new.ids))
    assert not set(new.ids).intersection(prior.ids)
    assert all(x[1:] in set(full.ids) for x in new.hrefs if x.startswith('#'))
    assert new.scripts==0 and section.count(chr(65533))==0
    assert full.scripts==prior.scripts
    PRESENTATION.mkdir(exist_ok=True)
    target=PRESENTATION/'ThrusterChain_diagnostics_results_v5_fresh_replication.html'
    target.write_bytes(text.encode('utf-8'))
    return target, {'v4_byte_preservation_after_removing_additions':True,'new_ids':len(new.ids),'new_source_links':len(new.hrefs),
                    'new_script_tags':new.scripts,'existing_scripts_unchanged_count':prior.scripts,'new_replacement_characters':0,
                    'all_new_internal_links_resolve':True,'rendered_browser_tested':False}


def main():
    protocol,freeze,prepared,result,status,candidates=authenticate()
    refs=source_map(); rows=tables(result); details=count_details(result,candidates); plot=chart(result)
    report=markdown(protocol,result,status,rows,details)
    section=fragment(protocol,result,status,rows,details,refs,plot)
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    (OUT/'section.html').write_text(section,encoding='utf-8')
    (OUT/'alignment_and_ranking.svg').write_text(plot,encoding='utf-8')
    guide,qa=build_v5(section)
    receipt={'schema':'fresh-replication-report-v1','report_source_sha256':sha(__file__),'model_or_simulator_calls':0,
        'source_freeze_verified':True,'all_prepared_and_result_artifact_hashes_verified':True,
        'v4_source':str(V4),'v4_sha256':sha(V4),'v5_sha256':sha(guide),'static_qa':qa,
        'input_sha256':{name:sha(ROOT/name) for name in ('protocol.json','freeze.json','prepared/preparation_manifest.json','results/status.json','results/paired_results.json','results/candidate_records.csv')},
        'output_sha256':{name:sha(OUT/name) for name in ('RESULTS.md','section.html','alignment_and_ranking.svg')},
        'source_map':refs,'details':details,'all_metrics':result['metrics']}
    write_json(OUT/'report_receipt.json',receipt)
    write_json(PRESENTATION/'static_qa.json',qa)
    print(json.dumps({'guide':str(guide),'guide_sha256':sha(guide),'output_sha256':receipt['output_sha256'],'static_qa':qa},indent=2))


if __name__=='__main__':
    main()
