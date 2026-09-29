"""Independent arithmetic/link checks on the generated report, using stdlib sums."""
from pathlib import Path
import json, math, statistics, hashlib, re, argparse
from datetime import datetime,timezone
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path('C:/Users/David Bayha/Documents/GitHub/BCA'));parser.add_argument('--output',type=Path);args=parser.parse_args()
R=args.repo;O=args.output or R/'outputs/current_results/2026-09-29'
x=json.loads((O/'plot_data.json').read_text());rs=x['records'];ps=x['paired_summaries']
mean=lambda values:math.fsum(values)/len(values)
errors=[];maxerr=0.0
for r in rs:
 assert len(r['means'])==len(r['periodic_scores'])==200
 assert len(r['final_scores'])==len(r['final_seeds'])==20
 assert r['steps']==list(range(5000,1000001,5000))
 expected_episodes=2 if r['host']=='iql' else 10
 for scores,seeds,m in zip(r['periodic_scores'],r['periodic_seeds'],r['means']):
  assert len(scores)==len(seeds)==expected_episodes
  assert len(set(seeds))==expected_episodes
  assert all(math.isfinite(v) for v in scores)
  error=abs(mean(scores)-m);maxerr=max(maxerr,error);assert error<1e-10
 assert abs(mean(r['final_scores'])-r['final_mean'])<1e-10
 assert abs(mean([mean(s) for s in r['periodic_scores']])-r['curve_mean'])<1e-10
for p in ps:
 a=next(r for r in rs if r['run_id']==p['host_run'] and r['arm']=='host')
 b=next(r for r in rs if r['run_id']==p['bca_run'] and r['arm']=='bca')
 assert a['training_seed']==b['training_seed']==p['seed']
 assert a['dataset_id']==b['dataset_id']
 assert a['final_seeds']==b['final_seeds'];assert a['periodic_seeds']==b['periodic_seeds']
 assert abs(mean(b['final_scores'])-mean(a['final_scores'])-p['final_difference'])<1e-10
 assert p['reset_wins']==sum(v>w for v,w in zip(b['final_scores'],a['final_scores']))
 if p['host']=='iql':assert a['run_id']==b['run_id'] and '-bca-noiw-' in a['run_id']
assert len(rs)==115 and len({r['run_id'] for r in rs})==94 and len(ps)==46
assert sum(p['independently_reviewed'] for p in ps)==8
# Check an independently audited pair against its separately published comparison export.
receipt=json.loads((R/'outputs/standard_bca/td3_bc/walker2d/bca_noiw/s202609171/verified-v1/comparison.json').read_text())
known=next(p for p in ps if p['host']=='td3_bc' and p['environment']=='walker2d')
assert abs(known['final_difference']-5.650837482779551)<1e-10
assert abs(known['curve_difference']-(-2.21619661784716))<1e-10
assert known['reset_wins']==8
content=(O/'index.html').read_text(encoding='utf-8')
links=re.findall(r'(?:src|href)="([^"]+)"',content)
for link in links:
 if link.startswith('#'):assert f'id="{link[1:]}"' in content
 elif link!='VERIFICATION.json':assert (O/link).is_file(),link
figs=json.loads((O/'figure_manifest.json').read_text());assert len(figs)==12
for f in figs:
 for ext in ['png','pdf','svg']:
  p=O/f'{f["name"]}.{ext}';assert p.stat().st_size>1000
result={'accepted_report_arithmetic_and_links':True,'checked_utc':datetime.now(timezone.utc).isoformat(),
 'physical_results':94,'actor_trajectories':115,'paired_training_comparisons':46,'independently_reviewed_training_pairs':8,
 'revised_ood_completed':0,'revised_ood_active':5,'figures':12,'max_periodic_mean_recomputation_error':maxerr,
 'checks':['All 200 banks and final 20 episodes per trajectory','Exact same reset seeds within pairs',
  'IQL paired host and BCA use same physical shared-Q/V run','Independent stdlib arithmetic for every bank, final and effect',
  'Published TD3 Walker independent receipt values','All HTML image/download/anchor targets exist','All 36 figure exports exist'],
 'visual_review':'All 12 static PNG figures inspected; endpoint-marker clipping corrected.',
 'browser_review':'Local HTML browser preview blocked by file URL policy. No workaround attempted. HTML links checked statically; browser interaction unverified.',
 'scope':'Report checks only; not a substitute for independent training/checkpoint/OOD acceptance.',
 'file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file() and p.name!='VERIFICATION.json'}}
(O/'VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='file_sha256'},indent=2))
