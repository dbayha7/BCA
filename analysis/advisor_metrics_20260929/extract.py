"""Read only already-exported, closed evidence; never open live simulator archives."""
import json, gzip, hashlib
from pathlib import Path
import numpy as np
R=Path('C:/Users/David Bayha/Documents/GitHub/BCA')
W=Path(__file__).resolve().parent
sources={}
def load(p):
 p=R/p; raw=p.read_bytes();sources[str(p.relative_to(R))]=hashlib.sha256(raw).hexdigest()
 return json.loads(gzip.decompress(raw).decode('utf-8-sig') if p.suffix=='.gz' else raw.decode('utf-8-sig'))
base='outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/'
summary=load(base+'summary.json');obs=load(base+'observations.json');panels=load(base+'panels.json')
primary=[r for r in obs if r['continuation']=='bca'];alt=[r for r in primary if not r['reference']]
assert len(primary)==2560 and len(alt)==2304 and sum(r['harmful'] for r in alt)==276
assert sum(r['support_distant'] for r in alt)==233
for r in primary:
 assert abs(r['tested_bank_regret']-(r['tested_bank_maximum']-r['raw_return']))<1e-9
 assert r['harmful']==(r['reference_return']-r['raw_return']>1)
assert abs(np.mean([r['tested_bank_regret'] for r in primary if r['slot']==0])-summary['bca']['overall']['mean_host_first_tested_bank_regret'])<1e-10
assert abs(np.mean([r['tested_bank_regret'] for r in primary if r['slot']==1])-summary['bca']['overall']['mean_bca_first_tested_bank_regret'])<1e-10
local=load('analysis/current_results_20260929/inputs/local_exports.json')
metrics={}
for host in ['td3_bc','rebrac']:
 for arm in ['host','bca']:
  rid=f'{host}-walker2d-'+('host' if arm=='host' else 'bca-noiw')+'-s202609171'
  directory=Path(local[rid]['path']).parent
  blocks=load(directory/'metric_blocks.json');audit=load(directory/'audit.json')
  assert audit['accepted'] and len(blocks)==1000 and blocks[-1]['step']==1000000
  keyq='q_mean' if host=='td3_bc' else 'q_min'
  keys=[keyq,'critic_loss','scale_loss','actor_bc_multiplier_mean']
  metric={k:[r[k]['mean'] for r in blocks] for k in keys if k in blocks[0]}
  metric['steps']=[r['step'] for r in blocks]
  if arm=='bca':
   refresh=load(directory/'refreshes.json');assert len(refresh)==198
   metric['refresh']=[{'step':r['step'],'radii':{k:r['radii'][k][0] for k in ['radius','bayesian_radius','conformal_radius']},'unit':r.get('residual_unit'),'ess':r['radii'].get('effective_sample_size'),'floor_dose_change':r.get('component_diagnostics',{}).get('level_floor_dose_absdiff')} for r in refresh]
   assert all(r['radii']['radius']==max(r['radii']['conformal_radius'],r['radii']['bayesian_radius']) for r in metric['refresh'])
   metric['conformal_nonbinding_count']=sum(r['radii']['conformal_radius']<r['radii']['bayesian_radius'] for r in metric['refresh'])
   metric['final_counter']=audit['checkpoints'][-1]['decoded_counters']
  metrics[rid]=metric
cluster=load('analysis/current_results_20260929/inputs/cluster_snapshot.json.gz')
for run in cluster['runs']:
 rid=run['run_id']
 if rid.startswith('cql-') and run.get('scalar_snapshots'):
  snaps=run['scalar_snapshots']
  metrics[rid]={'steps':[r['step'] for r in snaps],**{k:[r['metrics_last'].get(k) for r in snaps] for k in ['average_qf1','qf1_loss','cql_min_qf1_loss','critic_dose_mean','scale_loss','scale_fit_accepted']},'refresh_count':len(run['refreshes']),'sampling':'sparse last rows; full checkpoint review pending'}
wid=np.array([r['width'] for r in primary]);pt=[p for p in panels if p['continuation']=='bca']
width={'min':float(wid.min()),'median':float(np.median(wid)),'max':float(wid.max()),'all_alternative_widths_tied_panels':sum(len(set(p['metrics']['scores']['width']))==1 for p in pt)}
report={'pilot':summary,'pilot_width':width,'metrics':metrics,'sources':sources,'status_snapshot':'2026-09-29 approximately 09:00 EDT','fresh_coverage_verified':False,'revised_ood_completed':0,'revised_ood_active':5,'report_only':True}
(W/'metrics.json').write_text(json.dumps(report,separators=(',',':')),encoding='utf-8')
print(json.dumps({'pilot_AP':summary['bca']['overall']['balanced_stratum_within_width_ap'],'width':width,'training':{k:{'last':{n:v[-1] for n,v in m.items() if isinstance(v,list) and n not in ['refresh']},'nonbinding':m.get('conformal_nonbinding_count'),'counter':m.get('final_counter')} for k,m in metrics.items() if k.startswith(('td3','rebrac'))}},indent=2))
