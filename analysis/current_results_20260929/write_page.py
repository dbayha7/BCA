from pathlib import Path
import json,html,csv,statistics,hashlib,argparse
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path('C:/Users/David Bayha/Documents/GitHub/BCA'));parser.add_argument('--output',type=Path);args=parser.parse_args()
R=args.repo;O=args.output or R/'outputs/current_results/2026-09-29'
x=json.loads((O/'plot_data.json').read_text());figs=json.loads((O/'figure_manifest.json').read_text());ps=x['paired_summaries'];rr=x['records']
H={'td3_bc':'TD3+BC','cql':'CQL','rebrac':'ReBRAC','iql':'IQL'}
E={'hopper':'Hopper','walker2d':'Walker','halfcheetah':'HalfCheetah','maze2d':'Maze2D','pen-human':'Pen human','pen-cloned':'Pen cloned','pen-expert':'Pen expert'}
total=len({r['run_id'] for r in rr});reviewed=sum(p['independently_reviewed'] for p in ps)
intro=f'''# BCA experiment update — 29 September 2026

Standard BCA has not yet shown a consistent performance benefit. The new CQL endpoints are weaker; TD3 Walker improves its final mean but is weaker on average during learning. We now have clear training comparisons, but the revised OOD study has no completed pair yet.

This report contains {total} completed or recovered physical training runs, {len(rr)} saved actor trajectories, and {len(ps)} available within-seed host/BCA comparisons. Only {reviewed} training pairs have completed the separate independent review. Additional extracted comparisons are preliminary. These counts describe different units and must not be added together.

Training snapshot: {x['training_captured_utc']} through {x['training_extraction_finished_utc']}; IQL saved-array extraction finished {x['iql_extraction_finished_utc']}. OOD worker status: {x['ood_captured_utc']}. These are timestamped snapshots, not a live page.

## What has changed

- CQL now supplies completed 1M-update comparisons on Hopper and Walker. The original CQL host output-schema failure was recovered from saved evidence without training again; its original process exit 1 remains recorded.
- TD3 Walker's first training pair passed independent saved-data review. TD3 HalfCheetah's host finished during extraction; its BCA partner was running and no completed comparison is reported for that cell.
- All available completed curves are organized into four host atlases covering all seven declared datasets. Unfinished cells are visible. IQL's primary comparison uses the two actors sharing nuisance Q/V, with its standalone host shown separately.

## New comparisons: same training seed, 1M host updates

| Comparison | Host final | BCA final | Final difference | Difference in mean periodic curve | Evidence |
|---|---:|---:|---:|---:|---|
'''
for h,e in [('cql','hopper'),('cql','walker2d'),('td3_bc','walker2d')]:
 p=next(p for p in ps if p['host']==h and p['environment']==e and p['seed']==202609171)
 intro+=f'| {H[h]} {E[e]} | {p["host_final"]:.2f} | {p["bca_final"]:.2f} | {p["final_difference"]:+.2f} | {p["curve_difference"]:+.2f} | '+('Independent review passed' if p['independently_reviewed'] else 'Independent review pending')+' |\n'
intro+='''
Final return averages 20 reserved final episodes. Mean periodic curve averages all 200 equally spaced evaluation-bank means from 5k to 1M updates; it is not best-checkpoint performance. These selected hosts use 10 episodes per periodic bank. One training seed per new pair does not provide a training-seed uncertainty estimate. CQL Hopper used different hardware for host and BCA (local RTX 5070 Ti versus cluster A100).

TD3 Walker wins 8/20 paired final resets despite its positive final mean. CQL Walker wins 11/20 despite its negative final mean. Large gains or losses on a subset of resets can drive an average; the final-reset plot shows all of them.

## Answers for the advisor

**Why are we doing this?** Offline policies can propose state–action combinations with little relevant dataset support. We need to test whether BCA's warning signal helps identify actions whose actual continuation outcomes are worse, and whether the host uses that signal beneficially.

**What does the current comparison change?** It adds standard BCA with equal calibration-fitting weights while retaining Bayesian and conformal components. The base host is the comparison; importance weighting is not yet the treatment in these plots. IQL's paired actors share nuisance Q/V, so the comparison is about their actor treatment, not separately improved critics.

**What do the training results accomplish?** They show whether the complete learned policy performs better and how performance changes during learning. They do not tell us whether BCA correctly ranks the danger of individual unfamiliar actions.

**Why is the evidence of benefit still weak?** The new pairs have one training seed, endpoints and learning-time averages differ, and seed effects can change sign. In the reviewed ReBRAC Walker trio, the BCA-minus-host final differences are +4.98, −2.99 and +33.64: the average +11.88 is strongly affected by one seed. Reviewed ReBRAC Hopper is nearly tied on average (−0.03). The OOD mechanism test is still incomplete. This supports a cautious conclusion, not a claim that BCA universally helps or universally fails.

**What would a good OOD result mean?** Within the same simulator state, larger BCA warnings would consistently track actions with worse measured continuation outcomes across matched seeds. If the policy also chooses better actions with lower candidate-set regret, that would support useful decision guidance in the tested panel. Neither finding alone is a universal OOD detector or proof of conformal coverage under arbitrary shift.

**What would a bad OOD result mean?** If widths fail to rank harmful actions, the signal is not useful for that panel and continuation policy. If ranking is useful but choices or returns worsen, the place or strength of BCA's influence on the host needs investigation. If nominal coverage misses its target, assess the calibration assumptions and target being covered separately. These are different failure modes; return or Q loss cannot diagnose all of them.

**What happens next, and why?** Finish and verify the five active OOD pairs, then report within-state harm ranking, actual action outcomes and candidate-set regret with the protocol's fixed horizons, seeds and gates. Continue the declared remaining pairs and independent training reviews. Keep current coverage/settings fixed so the baseline remains interpretable. Only after establishing the standard-BCA limitation should a separately specified weighted-conformal treatment be compared. Weighted scale fitting by itself is not weighted conformal prediction; the target distribution, ratio estimator, weighted quantile and test-point mass require their own validated implementation. We should not assume importance weighting will repair every failure mode.

## OOD execution and the remaining gap

Five revised pairs are active: local TD3 Hopper seed171, plus cluster ReBRAC Hopper171 and Walker171/172/173. Zero of the 20 pairs in this tranche is complete; 15 have not started. Both cluster A100 allocations were also active on the training queues at the recorded check. OOD simulation runs on CPU workers, so the GPU training speed does not determine how quickly these interventions finish.

The progress figure counts explicit simulator transitions, including setup and collection. It is not a completion percentage, not a scientific comparison, and not an ETA. The TD3 count includes its preserved pre-interruption prefix and continuation. No active scientific SQLite database was opened by this plotting update. Final OOD harm, coverage and regret charts remain unavailable until the pairs close and pass verification; old pilot outcomes are not relabeled as new evidence.

## How to read the plots

'''
for f in figs:intro+=f'### {f["title"]}\n\n{f["caption"]}\n\n![{f["title"]}]({f["name"]}.png)\n\n'
intro+='''## Definitions and evidence boundaries

- Training runs, actor trajectories, paired training comparisons and completed OOD comparisons are different counts. IQL accounts for the extra actor trajectories.
- The declared training matrix remains 280 physical runs / 315 actor trajectories; this report contains 94 produced results, including one recovered original failure. Remaining results are not assumed successful.
- Atlas shading is ±1 sample standard deviation across the available matched training seeds. It is descriptive dispersion, not a confidence interval. Periodic IQL banks have two episodes; other hosts have ten. Every final bank has 20.
- The paired table reports the exact same completed seed set for host and BCA in each cell. An unmatched finished host is plotted as gray context and excluded from paired means.
- CQL scalar diagnostics are 1,000 sparse last-row observations, not all one million updates. TD3 diagnostic curves use audited block means with actor skips handled by the exporter.
- All figures retain raw plotted values. No smoothing, score-based filtering, best-seed selection, training change, simulator run or checkpoint query was performed to make this report.
- Source/configuration/result hashes and exact bank values accompany the report. Hash and bank checks on preliminary outputs do not substitute for the full independent source/data/checkpoint audit.
- All comparisons are current local controls. External published means and older campaigns are not substituted into these pairs. No main research PDF was changed.

## Calculations

For periodic bank j in seed s, score m(s,j) is the arithmetic mean of that bank's episode scores. The learning-time summary is (1/200) × sum over j of m(s,j). The final result f(s) is the mean of the separate 20 final episodes. The paired effects are f_BCA(s) − f_host(s) and curve_BCA(s) − curve_host(s). Across seeds, average the seed-level effects equally; do not treat evaluation episodes as additional independent training seeds. Final-reset dots subtract the two policies' returns under the same reset seed.

Candidate-set regret for the forthcoming OOD panel is max over tested candidate actions of measured return minus the selected action's measured return, within a fixed state and continuation setup. It measures regret among tested candidates, not regret against the unknown optimal action.
'''
(O/'ANALYSIS.md').write_text(intro,encoding='utf-8')
# Build a single offline page; external scripts/fonts are unnecessary.
def esc(v):return html.escape(str(v))
cards=''.join(f'<a href="#{f["name"]}">{esc(f["title"])}</a>' for f in figs)
sections=''.join(f'<section id="{f["name"]}"><div class="kicker">FIGURE {i+1:02d}</div><h2>{esc(f["title"])}</h2><p>{esc(f["caption"])}</p><a href="{f["name"]}.png"><img loading="lazy" src="{f["name"]}.png" alt="{esc(f["title"])}"></a><div class="downloads"><a href="{f["name"]}.png">PNG</a><a href="{f["name"]}.pdf">PDF</a><a href="{f["name"]}.svg">SVG</a></div></section>' for i,f in enumerate(figs))
rows=''
for p in ps:
 label='Reviewed' if p['independently_reviewed'] else 'Review pending'
 rows+=f'<tr data-host="{p["host"]}"><td>{H[p["host"]]}</td><td>{E[p["environment"]]}</td><td>{p["seed"]}</td><td>{p["host_final"]:.2f}</td><td>{p["bca_final"]:.2f}</td><td class="num">{p["final_difference"]:+.2f}</td><td>{p["curve_difference"]:+.2f}</td><td>{label}</td></tr>'
datasetrows=''.join(f'<li><strong>{esc(e)}:</strong> {esc(ds)}</li>' for e,ds in sorted({(E[r['environment']],r['dataset_id']) for r in rr}))
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BCA · Current results · 29 Sep 2026</title><style>
:root{--ink:#183047;--muted:#536778;--blue:#2563a6;--orange:#d05a27}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f8;color:var(--ink);font:17px/1.6 system-ui,Arial}header{background:#132d45;color:white;padding:48px max(24px,calc((100vw - 1220px)/2))}h1{font-size:clamp(30px,4vw,49px);line-height:1.14;margin:12px 0 20px;max-width:1000px}header p{max-width:1080px;color:#d9e6ef}main{max-width:1268px;margin:auto;padding:28px 24px}section,.panel{background:white;border:1px solid #dce4ea;border-radius:12px;padding:28px;margin:22px 0;scroll-margin-top:16px}h2{font-size:27px;line-height:1.25;margin:8px 0 14px}h3{margin:24px 0 8px}p{max-width:1100px}img{width:100%;height:auto;display:block;margin-top:18px}a{color:var(--blue)}.kicker{font-size:12px;font-weight:750;letter-spacing:.14em;color:#70889a}header .kicker{color:#9ac8e8}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{padding:18px;background:white;border-radius:10px;border-top:4px solid var(--blue)}.stat b{display:block;font-size:34px;line-height:1.3}.stat span{font-size:14px;color:var(--muted)}.alert{border-left:5px solid var(--orange);background:#fff7ef;padding:16px 20px}.nav,.downloads{display:flex;flex-wrap:wrap;gap:10px}.nav a,.downloads a{padding:7px 12px;background:#eef4f9;border-radius:5px;text-decoration:none;font-size:14px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #dce4ea;white-space:nowrap}th{background:#edf3f7}th:first-child,td:first-child,td:nth-child(2),td:last-child{text-align:left}.num{font-weight:bold}select{font:inherit;padding:6px 12px;margin-bottom:16px}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}.two h3{margin-top:0}.small{font-size:14px;color:var(--muted)}li{margin-bottom:9px}footer{padding:24px;color:var(--muted);font-size:14px}@media(max-width:750px){.stats{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}section,.panel{padding:18px}main{padding:14px}header{padding:28px 20px}}@media print{body{background:white}section{break-before:page;border:0}.nav,select,.downloads{display:none}header{background:white;color:black}header p{color:black}}
</style><header><div class="kicker">STANDARD BCA · NO IMPORTANCE WEIGHTING · 1M UPDATES</div><h1>Performance is mixed.<br>The OOD mechanism test is still running.</h1><p>New CQL and TD3 results, all available learning curves, calibration diagnostics and a precise view of what is—and is not—finished.</p><p class="small" style="color:#bdd0df">29 September 2026 · Saved training snapshot 12:59–13:03 UTC · OOD status 12:52 UTC · Timestamped report, not live status</p></header><main>'''
page+=f'<div class="stats"><div class="stat"><b>{total}/280</b><span>physical training results produced</span></div><div class="stat"><b>{len(ps)}</b><span>available within-seed comparisons</span></div><div class="stat"><b>{reviewed}</b><span>independently reviewed training pairs</span></div><div class="stat"><b>0/20</b><span>completed revised OOD pairs; 5 active</span></div></div>'
page+='''<div class="panel"><h2>What to tell your advisor</h2><div class="two"><div><h3>What we know</h3><ul><li><b>CQL Hopper:</b> 62.98 → 56.79 final; average learning curve nearly tied.</li><li><b>CQL Walker:</b> 80.13 → 63.88 final; average learning curve also slightly lower.</li><li><b>TD3 Walker:</b> 77.25 → 82.90 final, but lower average during learning and only 8/20 final-reset wins.</li><li><b>Reviewed ReBRAC:</b> Hopper nearly tied across 3 seeds; Walker mean improves, but the seed effects are +4.98, −2.99 and +33.64.</li></ul></div><div><h3>What this means</h3><p>There is no consistent benefit established yet. The new CQL/TD3 pairs have one training seed, and endpoint gains can differ from performance during learning.</p><p>The OOD experiment asks a different question: <b>does a larger BCA warning identify actions that actually lead to worse outcomes?</b> Return curves and fitting losses cannot answer that on their own.</p></div></div><div class="alert"><b>Evidence status:</b> TD3 Walker is independently reviewed. The new CQL comparisons await that review; CQL Hopper also uses different host/BCA hardware. Preliminary atlas rows remain explicitly labeled.</div><p><a href="ANALYSIS.md">Full explanation and advisor answers</a> · <a href="paired_summary.csv">Per-seed results CSV</a> · <a href="plot_data.json">Exact plotted data and provenance</a></p></div>'''
page+='<nav class="nav">'+cards+'</nav>'+sections
page+='''<section><h2>All available paired results</h2><p>Positive differences favor BCA. Final and learning-time differences answer different questions. IQL uses the two actors within the same shared-Q/V run. Review-pending rows are descriptive saved outputs, not accepted OOD evidence.</p><label for="host">Host: </label><select id="host"><option value="all">All hosts</option><option value="cql">CQL</option><option value="td3_bc">TD3+BC</option><option value="rebrac">ReBRAC</option><option value="iql">IQL</option></select><div class="table-wrap"><table><thead><tr><th>Host</th><th>Dataset</th><th>Training seed</th><th>Host final</th><th>BCA final</th><th>Final difference</th><th>Curve difference</th><th>Evidence</th></tr></thead><tbody>'''+rows+'''</tbody></table></div></section>'''
page+='''<section><h2>What we do next—and what each outcome means</h2><ol><li><b>Finish and verify the active OOD pairs.</b> Retain matched simulator states, fixed continuation streams and the declared numerical gates.</li><li><b>Compare warnings with measured harm within each state.</b> Good ranking supports a useful warning signal in that panel. Poor ranking points to a signal–harm mismatch, even if the scale fitting succeeds.</li><li><b>Compare action choices and candidate-set regret.</b> Good ranking with worse decisions points toward how the host uses the signal. Regret is relative to tested candidate actions, not the unknown optimal policy.</li><li><b>Test the calibration target separately.</b> Nominal coverage concerns residual bands under stated assumptions. It is not the fraction of reliance on the old policy, and it is not automatically coverage of true return.</li><li><b>Then test the specified importance-weighted extension.</b> First establish the standard-BCA limitation. Correct weighted conformal prediction needs the declared distribution ratio, weighted quantile and test-point mass; weighted scale fitting alone is insufficient. Do not tune coverage to improve this baseline after seeing its outcomes.</li></ol></section><section><h2>Scope and reproducibility</h2><p>The report uses all available completed saved curves, no score-based selection, no smoothing and no new training or simulator queries. Older campaigns and the main PDF are unchanged. The one recovered original CQL exit 1 remains preserved. IQL produces two actors in each BCA run, explaining why there are 115 trajectories from 94 physical runs.</p><ul>'''+datasetrows+'''</ul><p>All hosts target 1M host updates. TD3+BC and ReBRAC have 500k delayed actor updates. IQL periodic banks use 2 episodes; other hosts use 10. Final banks use 20. Shading is descriptive ±1 sample SD across matched training seeds, not a confidence interval.</p><p><a href="VERIFICATION.json">Numerical and artifact verification</a> · <a href="ANALYSIS.md">Complete analysis</a> · <a href="paired_summary.csv">Paired results</a></p></section></main><footer>Bayesian Conformal Aggregation · Standard/no-IW study · No new OOD scientific completion claimed.</footer><script>document.getElementById('host').addEventListener('change',function(){document.querySelectorAll('tbody tr').forEach(r=>r.hidden=this.value!=='all'&&r.dataset.host!==this.value)})</script></html>'''
page=page.replace('Saved training snapshot 12:59–13:03 UTC',f'Saved training snapshot {x["training_captured_utc"][11:16]}–{x["training_extraction_finished_utc"][11:16]} UTC')
(O/'index.html').write_text(page,encoding='utf-8')
print('Wrote analysis and dashboard',O)
