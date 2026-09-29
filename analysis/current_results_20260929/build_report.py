"""Build figures from frozen saved-data extracts; no training or simulator imports."""
from pathlib import Path
import json, hashlib, csv, shutil, html, gzip, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--repo',type=Path,default=Path('C:/Users/David Bayha/Documents/GitHub/BCA'))
parser.add_argument('--output',type=Path);args=parser.parse_args()
W=args.inputs;R=args.repo;OUT=args.output or R/'outputs/current_results/2026-09-29'
OUT.mkdir(parents=True,exist_ok=True)
def load(p):
 p=Path(p)
 if not p.exists() and p.with_suffix(p.suffix+'.gz').exists():p=p.with_suffix(p.suffix+'.gz')
 return json.loads(gzip.decompress(p.read_bytes()).decode('utf-8-sig') if p.suffix=='.gz' else p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cluster=load(W/'cluster_snapshot.json'); assert not cluster['errors'],cluster['errors']
local=load(W/'local_exports.json')
status=load(W/'ood_status.json') if (W/'ood_status.json').exists() else load(W.parent/'cql_validation_recovery_v1/update_20260929T1252.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,'axes.labelsize':11,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
BLUE='#2563a6'; ORANGE='#d05a27'; GRAY='#707a8b'; GREEN='#21866e'
hosts={'td3_bc':'TD3+BC','cql':'CQL','rebrac':'ReBRAC','iql':'IQL'}
envs={'hopper':'Hopper','walker2d':'Walker','halfcheetah':'HalfCheetah','maze2d':'Maze2D',
      'pen-human':'Pen human','pen-cloned':'Pen cloned','pen-expert':'Pen expert'}
records=[]; proven=[]; diagnostics={}
remote={r['run_id']:r for r in cluster['runs']}
def parse_id(rid):
    base,seed=rid.rsplit('-s',1)
    if '-bca-noiw' in base: base=base.removesuffix('-bca-noiw');method='bca'
    else: base=base.removesuffix('-host');method='host'
    host,env=base.split('-',1)
    return host,env,method,int(seed)
def add(rid,arm,banks,evidence,source):
    h,e,m,s=parse_id(rid)
    pp=sorted([b for b in banks if b['kind']=='periodic'],key=lambda b:b['step'])
    ff=[b for b in banks if b['kind']=='final'];assert len(pp)==200 and len(ff)==1
    assert [b['step'] for b in pp]==list(range(5000,1000001,5000))
    assert len(ff[0]['scores'])==20
    dataset=next((v['environment'] for k,v in remote.items() if parse_id(k)[:2]==(h,e)),banks[0].get('dataset_id'))
    assert dataset
    rec={'run_id':rid,'host':h,'environment':e,'dataset_id':dataset,'training_seed':s,'arm':arm,'evidence':evidence,'source':source,
         'steps':[b['step'] for b in pp],'periodic_scores':[b['scores'] for b in pp],
         'periodic_seeds':[b['seeds'] for b in pp], 'means':[float(np.mean(b['scores'])) for b in pp],
         'final_scores':ff[0]['scores'],'final_seeds':ff[0]['seeds']}
    rec['final_mean']=float(np.mean(rec['final_scores']));rec['curve_mean']=float(np.mean(rec['means']))
    records.append(rec)
for rid,entry in local.items():
    p=Path(entry['path']);p=p if p.is_absolute() else R/p
    assert sha(p)==entry['sha256'];h,e,m,s=parse_id(rid)
    ev=load(p);sk,rk=('score','return') if h=='cql' else ('normalized_score','raw_return')
    banks=[{'kind':b['kind'],'step':b['step'],'dataset_id':b.get('score_transform',{}).get('dataset'),'scores':[x[sk] for x in b['episodes']],
            'seeds':[x['seed'] for x in b['episodes']]} for b in ev]
    audit=load(p.parent/'audit.json')
    assert audit.get('accepted_saved_training_result') is True if h=='cql' else audit.get('accepted') is True
    evidence='Independent saved-data review' if h!='cql' else 'Saved recovery reviewed; original exit 1 retained'
    add(rid,m,banks,evidence,str(p.relative_to(R)))
    proven.append({'run_id':rid,'evaluations':str(p.relative_to(R)),'sha256':sha(p),'audit_sha256':sha(p.parent/'audit.json')})
    diag={}
    for name in ['metric_blocks','metric_snapshots','refreshes']:
        f=p.parent/(name+'.json')
        if f.exists():diag[name]=load(f);proven.append({'run_id':rid,'file':str(f.relative_to(R)),'sha256':sha(f)})
    diagnostics[rid]=diag
for run in cluster['runs']:
    if run['reused_local']:assert run['run_id'] in local
    else:
        for t in run['trajectories']:add(run['run_id'],t['arm'],t['banks'],'Completed; independent review pending',run['root'])
    if 'scalar_snapshots' in run:diagnostics[run['run_id']]={'metric_snapshots':run['scalar_snapshots'],'refreshes':run['refreshes']}
    proven.append({k:v for k,v in run.items() if k not in ['trajectories','scalar_snapshots','refreshes']})
ids=[(r['run_id'],r['arm']) for r in records];assert len(ids)==len(set(ids))
pairs=[]
for h in hosts:
 for e in envs:
  rr=[r for r in records if r['host']==h and r['environment']==e]
  for s in sorted({r['training_seed'] for r in rr}):
   aa={r['arm']:r for r in rr if r['training_seed']==s}
   if 'host' in aa and 'bca' in aa:
    a,b=aa['host'],aa['bca'];assert a['final_seeds']==b['final_seeds'];assert a['periodic_seeds']==b['periodic_seeds']
    dif=np.array(b['final_scores'])-a['final_scores']
    pairs.append({'host':h,'environment':e,'seed':s,'host_final':a['final_mean'],'bca_final':b['final_mean'],
      'final_difference':b['final_mean']-a['final_mean'],'host_curve':a['curve_mean'],'bca_curve':b['curve_mean'],
      'curve_difference':b['curve_mean']-a['curve_mean'],'reset_wins':int((dif>0).sum()),'reset_ties':int((dif==0).sum()),
      'host_run':a['run_id'],'bca_run':b['run_id'],'independently_reviewed':h!='iql' and a['run_id'] in local and b['run_id'] in local})
snap={'training_captured_utc':cluster['captured_utc'],'training_extraction_finished_utc':cluster['finished_utc'],
      'iql_extraction_finished_utc':cluster.get('iql_extracted_utc'),'ood_captured_utc':status['ood_cluster']['utc'],
      'records':records,'paired_summaries':pairs,'queues':cluster['queues'],'ood':status,'provenance':proven}
(OUT/'plot_data.json').write_text(json.dumps(snap,separators=(',',':')),encoding='utf-8')
with (OUT/'paired_summary.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(pairs[0]),lineterminator='\n');writer.writeheader();writer.writerows(pairs)
figures=[]
def finish(fig,name,title,caption):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    svg=OUT/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig);figures.append({'name':name,'title':title,'caption':caption})
def xaxis(ax):
    ax.set_xlim(0,1.025);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Host updates (millions)')
def find(h,e,arm,s=202609171):return next(r for r in records if r['host']==h and r['environment']==e and r['arm']==arm and r['training_seed']==s)
selected=[('cql','hopper'),('cql','walker2d'),('td3_bc','walker2d')]
fig,axs=plt.subplots(1,3,figsize=(16,5))
for ax,(h,e) in zip(axs,selected):
 a,b=find(h,e,'host'),find(h,e,'bca')
 for r,c,label in [(a,BLUE,'Host'),(b,ORANGE,'Host + BCA')]:
  ax.plot(np.array(r['steps'])/1e6,r['means'],color=c,lw=1.4,label=label)
  ax.scatter([1],[r['final_mean']],s=85,color=c,marker='D',edgecolor='white',zorder=5)
 ax.set_title(f'{hosts[h]} · {envs[e]}\nFinal: {a["final_mean"]:.2f} → {b["final_mean"]:.2f}')
 ax.set_ylabel('Normalized return');xaxis(ax);ax.legend(loc='best')
fig.suptitle('New completed comparisons: BCA does not consistently improve return',fontsize=19,fontweight='bold',y=1.06)
fig.text(.5,-.06,'One training seed each · Lines: 200 ten-episode periodic banks · Diamonds: separate 20-episode final bank\nCQL: independent review pending; Hopper uses mixed hardware · TD3 Walker: independent review passed',ha='center',fontsize=10)
fig.tight_layout()
finish(fig,'01_new_learning_curves','New completed learning curves',
 'Read left to right through 1M updates. Every periodic mean is shown without smoothing. The diamonds use a separate final reset bank. CQL training results are preliminary pending independent review; TD3 Walker is reviewed. CQL Hopper compares a local RTX 5070 Ti host run with an A100 BCA run. These are policy-performance results, not action-level OOD results.')
fig,axs=plt.subplots(1,2,figsize=(13,5.3));labels=[];delta=[];curve=[]
for h,e in selected:
 p=next(p for p in pairs if p['host']==h and p['environment']==e and p['seed']==202609171)
 labels.append(f'{hosts[h]}\n{envs[e]}');delta.append(p['final_difference']);curve.append(p['curve_difference'])
for ax,vals,title in zip(axs,[delta,curve],['Separate final bank','Average of all 200 periodic banks']):
 bars=ax.bar(np.arange(3),vals,color=[GREEN if v>=0 else ORANGE for v in vals],width=.58)
 ax.axhline(0,color='#24344b',lw=1);ax.set_xticks(np.arange(3),labels);ax.set_ylabel('BCA − host (normalized return)');ax.set_title(title)
 lo=min(0,min(vals));hi=max(0,max(vals));pad=(hi-lo)*.3+1;ax.set_ylim(lo-pad,hi+pad)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+(.5 if v>=0 else -.5),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontweight='bold',fontsize=14)
fig.suptitle('A better endpoint can coexist with a weaker learning curve',fontsize=18,fontweight='bold');fig.tight_layout(rect=(0,0,1,.93))
finish(fig,'02_endpoint_vs_curve','Endpoint versus performance during learning',
 'Positive values favor BCA. TD3 Walker improves the final mean but has a lower average periodic score; it wins only 8 of 20 paired final resets. These are distinct questions, not conflicting calculations. One training seed per pair cannot estimate training-seed uncertainty. CQL figures await independent review.')
fig,axs=plt.subplots(1,3,figsize=(15,5))
for ax,(h,e) in zip(axs,selected):
 a,b=find(h,e,'host'),find(h,e,'bca');delta=np.array(b['final_scores'])-a['final_scores']
 ax.axhline(0,color=GRAY,lw=1);ax.scatter(range(1,21),delta,c=[GREEN if v>0 else ORANGE for v in delta],s=35)
 ax.axhline(delta.mean(),color='#24344b',ls='--',label=f'Mean {delta.mean():+.2f}')
 ax.set_title(f'{hosts[h]} {envs[e]}\nBCA wins {sum(delta>0)}/20 resets');ax.set_xlabel('Paired final reset index');ax.set_ylabel('BCA − host normalized return');ax.set_xticks([1,5,10,15,20]);ax.legend()
fig.suptitle('Why the final average alone can hide inconsistent outcomes',fontsize=18,fontweight='bold');fig.tight_layout(rect=(0,0,1,.92))
finish(fig,'02b_final_reset_differences','Final-reset differences',
 'Each dot compares host and BCA from the same final reset seed, with fixed trained policies. All 20 resets appear in their recorded order. TD3 Walker has a positive mean despite only 8 positive resets; CQL Walker has a negative mean despite 11 positive resets. This is variation over environment resets, not 20 independent training seeds, and does not isolate action-level harm.')
for h in hosts:
 fig,axs=plt.subplots(4,2,figsize=(13,15));axs=axs.ravel()
 for ax,(e,en) in zip(axs,envs.items()):
  matched=[p['seed'] for p in pairs if p['host']==h and p['environment']==e]
  rr=[r for r in records if r['host']==h and r['environment']==e]
  n=len(matched);ax.set_title(f'{en} · {n}/5 paired seeds')
  if n:
   for arm,c,label in [('host',BLUE,'Host'),('bca',ORANGE,'Host + BCA')]:
    rows=[r for r in rr if r['arm']==arm and r['training_seed'] in matched];ys=np.array([r['means'] for r in rows]);x=np.array(rows[0]['steps'])/1e6
    for y in ys:ax.plot(x,y,color=c,lw=.55,alpha=.28)
    ax.plot(x,ys.mean(0),color=c,lw=1.7,label=label)
    if n>1:ax.fill_between(x,ys.mean(0)-ys.std(0,ddof=1),ys.mean(0)+ys.std(0,ddof=1),color=c,alpha=.12)
   if h=='iql':
    rows=[r for r in rr if r['arm']=='standalone_host' and r['training_seed'] in matched]
    if rows:ax.plot(x,np.mean([r['means'] for r in rows],axis=0),color=GRAY,ls=':',lw=1,label='Standalone host context')
  else:ax.text(.5,.5,'No completed paired comparison',ha='center',va='center',transform=ax.transAxes,color=GRAY)
  unmatched=[r for r in rr if r['training_seed'] not in matched and r['arm']!='standalone_host']
  for r in unmatched:ax.plot(np.array(r['steps'])/1e6,r['means'],color=GRAY,ls='--',lw=1,label='Completed host; partner pending')
  ax.set_ylabel('Normalized return');xaxis(ax)
  if n or unmatched:ax.legend(fontsize=8,loc='best')
 axs[-1].axis('off')
 note='Matched training seeds only\nThin lines: individual seeds\nThick lines: seed mean\nShade: ±1 sample SD across seeds\nSD is descriptive, not a confidence interval.\n\nPending cells remain visible.\nIndependent review status is in the table.'
 if h=='iql':note+='\n\nPrimary pair shares nuisance Q/V.\nGray standalone host is context only.\nIQL periodic banks have 2 episodes.'
 else:note+='\n\nPeriodic banks have 10 episodes.'
 axs[-1].text(.03,.98,note,va='top',linespacing=1.6,fontsize=12)
 fig.suptitle(f'{hosts[h]} · all seven datasets · standard BCA without IW',fontsize=20,fontweight='bold',y=.997)
 fig.tight_layout(rect=(0,0,1,.975))
 finish(fig,f'03_atlas_{h}',f'{hosts[h]} learning-curve atlas',
  'The same completed training seeds are used for both methods in each colored comparison. Shading describes variation among the available seeds; it is not a confidence interval or final five-seed conclusion. Unpaired completed runs are gray dashed context. '+('IQL compares its paired actors sharing the same learned Q/V; the separately trained host is shown only as dotted gray context. ' if h=='iql' else '')+'Most atlas rows await the complete independent source/data/checkpoint audit; exact status appears in the per-seed table. No best-seed or best-checkpoint selection.')
fig,axs=plt.subplots(1,2,figsize=(12,5))
for ax,e in zip(axs,['hopper','walker2d']):
 pp=[p for p in pairs if p['host']=='rebrac' and p['environment']==e and p['independently_reviewed']]
 for j,p in enumerate(pp):ax.plot([0,1],[p['host_final'],p['bca_final']],marker='o',label=f'Seed {p["seed"]}',lw=2)
 ax.set_xticks([0,1],['Host','Host + BCA']);ax.set_ylabel('Final normalized return');ax.set_title(f'ReBRAC {envs[e]} · {len(pp)} reviewed seeds');ax.legend(fontsize=9)
fig.suptitle('Keep the individual training seeds visible',fontsize=18,fontweight='bold');fig.tight_layout(rect=(0,0,1,.94))
finish(fig,'04_reviewed_seed_pairs','Reviewed ReBRAC training-seed comparisons',
 'Each line connects the two final 20-episode means for one paired training seed. Agreement across seeds is stronger evidence than a single endpoint; three seeds remain short of the planned five. The lines describe whole-policy performance, not whether BCA correctly identifies harmful actions.')
# Accepted TD3 Walker calibration and loss traces: actor-only block means were computed in the audited export.
rid='td3_bc-walker2d-bca-noiw-s202609171';d=diagnostics[rid];blocks=d['metric_blocks'];ref=d['refreshes']
fig,axs=plt.subplots(2,2,figsize=(13,9));x=np.array([r['step'] for r in blocks])/1e6
for key,c,label in [('bayesian_radius',ORANGE,'Bayesian radius'),('conformal_radius',BLUE,'Conformal radius')]:
 axs[0,0].plot(np.array([r['step'] for r in ref])/1e6,[r['radii'][key][0] for r in ref],color=c,label=label,lw=1.5)
axs[0,0].set_title('Both radius components remain active');axs[0,0].set_ylabel('Radius (normalized residual units)');axs[0,0].legend()
for ax,key,title,ylabel in [(axs[0,1],'actor_bc_multiplier_mean','BCA changes the actor’s BC strength','Multiplier'),(axs[1,0],'scale_loss','Calibrator fitting objective','Block mean objective'),(axs[1,1],'critic_loss','Critic fitting objective','Block mean loss')]:
 ax.plot(x,[r[key]['mean'] for r in blocks],color=ORANGE,lw=1.3);ax.set_title(title);ax.set_ylabel(ylabel)
for ax in axs.ravel():xaxis(ax)
fig.suptitle('TD3+BC Walker: what BCA was doing during training',fontsize=19,fontweight='bold');fig.tight_layout(rect=(0,0,1,.95))
finish(fig,'05_td3_calibration','TD3 Walker calibration during training',
 '198 recorded refreshes retain both Bayesian and conformal radii; the effective radius is their maximum. The actor’s behavior-cloning penalty is multiplied by the BCA signal. The scale and critic plots use audited 1,000-update block means; actor statistics exclude skipped actor updates. Successful fitting does not show that width ranks OOD harm correctly.')
# CQL sparse last-row traces: never label these as full scan means or per-update logs.
fig,axs=plt.subplots(2,3,figsize=(16,8))
for row,e in enumerate(['hopper','walker2d']):
 for arm,c,label in [('host',BLUE,'Host'),('bca',ORANGE,'Host + BCA')]:
  rid=f'cql-{e}-'+('bca-noiw' if arm=='bca' else 'host')+'-s202609171'
  snaps=diagnostics[rid]['metric_snapshots'];xx=np.array([s['step'] for s in snaps])/1e6
  for col,key,title in [(0,'average_qf1','Q1 on recorded actions'),(1,'qf1_loss','Critic 1 Bellman loss'),(2,'cql_min_qf1_loss','Critic 1 conservative objective')]:
   vals=[s['metrics_last'][key] for s in snaps];axs[row,col].plot(xx,vals,color=c,label=label,lw=1)
   axs[row,col].set_title(f'{envs[e]} · {title}');xaxis(axs[row,col]);axs[row,col].set_ylabel('Logged value');axs[row,col].legend(fontsize=9)
fig.suptitle('CQL diagnostics: model objectives are not measured OOD harm',fontsize=19,fontweight='bold');fig.tight_layout(rect=(0,0,1,.95))
finish(fig,'06_cql_diagnostics','CQL Q and loss diagnostics',
 'These are 1,000 sparse last-row snapshots per completed run, not means over all updates. Q1 is evaluated on recorded dataset actions. Different Q/loss trajectories can help locate a mechanism to investigate, but cannot establish which proposed actions cause harm. Original CQL host failure/recovery remains preserved; the new CQL runs still await independent review.')
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for e,c in [('hopper',BLUE),('walker2d',ORANGE)]:
 snaps=diagnostics[f'cql-{e}-bca-noiw-s202609171']['metric_snapshots'];xx=np.array([s['step'] for s in snaps])/1e6
 for ax,key,title in [(axs[0],'critic_dose_mean','Multiplier on CQL’s conservative penalty'),(axs[1],'scale_loss','Calibrator fitting objective')]:
  ax.plot(xx,[s['metrics_last'][key] for s in snaps],color=c,label=envs[e],lw=1.2);ax.set_title(title);xaxis(ax);ax.legend()
axs[0].set_ylabel('Logged multiplier');axs[1].set_ylabel('Logged scale objective')
fig.suptitle('CQL: the calibration signal is active; usefulness is still a question',fontsize=17,fontweight='bold');fig.tight_layout(rect=(0,0,1,.92))
finish(fig,'06b_cql_calibration','CQL calibration signal',
 'Sparse last-row snapshots show the multiplier used on CQL’s conservative objective and the fitting objective. Both runs record 198 successful posterior refreshes. Historical numerical radii are not in these CQL refresh records, so no radius history is fabricated. Sampled fit acceptance is not a complete per-update acceptance log; independent final checkpoint-counter review is still pending. A nonzero multiplier alone does not demonstrate useful harm ranking.')
oodrows=[]
localood=status['ood_local'];s=localood['status'];oodrows.append(('TD3+BC Hopper · 171',s['transitions']+localood['old_explicit_transitions']))
for r in status['ood_cluster']['ood']:
 s=r['status'];h,e,seed=s['pair'];oodrows.append((f'{hosts[h]} {envs[e]} · {str(seed)[-3:]}',s['transitions']))
fig,ax=plt.subplots(figsize=(12,5.7));names,values=zip(*oodrows);bars=ax.barh(names,values,color=BLUE,height=.6)
ax.invert_yaxis();ax.set_xlim(0,max(values)*1.23);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
for bar,v in zip(bars,values):ax.text(v+max(values)*.012,bar.get_y()+bar.get_height()/2,f'{v:,}',va='center',fontsize=12)
ax.set_xlabel('Recorded explicit simulator transitions (including setup/collection)')
ax.set_title('OOD execution: 5 active pairs · 0 completed pairs · 15 unstarted',fontsize=17,fontweight='bold',pad=18)
fig.text(.5,-.012,'Status snapshot: 29 Sep 2026, 12:52 UTC · Counts are workload, not percentage complete or scientific results.',ha='center',fontsize=11)
fig.tight_layout()
finish(fig,'07_ood_progress','OOD execution progress',
 'Four cluster CPU workers and one local continuation worker are running the five pairs. Bar length counts recorded explicit transitions, including engineering checks and collection; different episode lengths prevent direct conversion to percent complete. TD3 includes its preserved original prefix plus its continuation. No revised pair is complete, so final harm-ranking, coverage and regret findings are unavailable. No live scientific database was read for this report.')

(OUT/'figure_manifest.json').write_text(json.dumps(figures,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'records':len(records),'physical_results':len(set(r['run_id'] for r in records)),
 'pairs':len(pairs),'figures':len(figures),'selected':[p for p in pairs if (p['host'],p['environment']) in selected],
 'ood_counts':status['counts']},indent=2))
