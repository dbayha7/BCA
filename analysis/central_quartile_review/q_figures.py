"""Rebuild the authenticated central-quartile Q figures; no simulation or model calls.

Uses extract.py to verify the completed terminal receipt, reproduction, 144
context hashes and arithmetic before plotting. Outputs only in figures_q/.
"""
from pathlib import Path
import subprocess, sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RERUN = ROOT / 'experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1'
OUT = HERE / 'figures_q'
result = subprocess.run([sys.executable, '-B', str(HERE/'extract.py'), str(RERUN)], capture_output=True, text=True, check=True)
data = json.loads(result.stdout)
snaps = data['snapshots']
OUT.mkdir(exist_ok=True)
RAW = '#334155'; TEAL = '#087e8b'; RED = '#b85050'; GOLD = '#c36b3b'
BG = '#fbfaf7'; GRID = '#d9dedf'; TEXT = '#182832'; MUTED = '#52636a'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':13,
                     'axes.labelsize':10, 'xtick.labelsize':9, 'ytick.labelsize':10,
                     'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
                     'figure.facecolor':BG, 'axes.facecolor':BG, 'savefig.facecolor':BG,
                     'text.color':TEXT, 'axes.labelcolor':TEXT, 'xtick.color':TEXT, 'ytick.color':TEXT})
contexts = [('ordinary','Ordinary'), ('strong_current','Strong current')]
conditions = [('full','Full data'), ('hole','Conditional hole'), ('random','Random control')]

def snap(ctx, cond, arm, seed):
    return snaps[f'{ctx}/{cond}/{arm}/20000/seed{seed}']
def header(fig, title, subtitle):
    fig.text(.055,.972,title,fontsize=20,weight='bold',va='top')
    fig.text(.055,.920,subtitle,fontsize=10.2,color=MUTED,va='top',linespacing=1.35)
def footer(fig, text):
    fig.text(.055,.035,text,fontsize=8.5,color=MUTED,va='bottom',linespacing=1.4)
def panel(ax):
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis='x',color=GRID,linewidth=.8,zorder=0)
    ax.tick_params(length=0,pad=5)
    ax.set_axisbelow(True)
def save(fig, stem):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{stem}.{ext}',dpi=300,facecolor=BG)
    plt.close(fig)

def pairs_plot(stem, title, subtitle, getter, left_label, right_label, foot):
    vals=[getter(ctx,cond,s) for ctx,_ in contexts for cond,_ in conditions for s in range(3)]
    maxx=max(v for pair in vals for v in pair)
    fig,axs=plt.subplots(2,3,figsize=(14.4,8.6),sharex=True)
    header(fig,title,subtitle)
    for r,(ctx,ctxlabel) in enumerate(contexts):
        for c,(cond,clabel) in enumerate(conditions):
            ax=axs[r,c];panel(ax)
            for s in range(3):
                a,b=getter(ctx,cond,s); y=2-s
                ax.plot([a,b],[y,y],color='#a8b4b7',lw=2.0,zorder=2)
                ax.scatter([a],[y],s=78,color=RAW,zorder=3)
                ax.scatter([b],[y],s=78,color=TEAL,zorder=4)
                # numeric values sit below paired dots, preserving the exact pairing
                ax.annotate(f'{a:.1f}',(a,y),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=8.5,color=RAW)
                ax.annotate(f'{b:.1f}',(b,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8.5,color=TEAL)
            ax.set_yticks([2,1,0],['Seed 0','Seed 1','Seed 2']);ax.set_ylim(-.5,2.55)
            ax.set_xlim(0,maxx*1.18);ax.set_title(f'{ctxlabel}  /  {clabel}',loc='left',fontweight='bold',pad=11,fontsize=11)
            if r == 1: ax.set_xlabel('MAE (return units)  ·  lower is better')
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=RAW,markeredgecolor=RAW,label=left_label,markersize=8),
                        Line2D([0],[0],marker='o',color='none',markerfacecolor=TEAL,markeredgecolor=TEAL,label=right_label,markersize=8)],
               loc='upper right',bbox_to_anchor=(.96,.886),frameon=False,ncol=2,fontsize=10)
    footer(fig,foot+'\nMAE = mean absolute error in discounted-return units; U = BCA uncertainty score. Source: authenticated report 55cae667d369.')
    fig.subplots_adjust(left=.09,right=.97,top=.79,bottom=.16,wspace=.26,hspace=.43)
    save(fig,stem)

pairs_plot('01_cross_model_accuracy',
    'Q accuracy: BCA scores versus native Q1',
    'High-load anchors, fixed actions inside the withheld band  |  Final checkpoint  |  One line per training seed',
    lambda ctx,cond,s:(snap(ctx,cond,'native',s)['q_panels']['high_band_fixed_primary']['raw']['mae'],
                       snap(ctx,cond,'bca',s)['q_panels']['high_band_fixed_primary']['adjusted']['mae']),
    'Native Q1','BCA Q1 − 3U',
    'Each model is compared with its own policy continuation: these are different Qπ references. Ordinary hole is the prespecified accuracy screen.\n21 ordinary / 32 strong-current anchor families per seed; three fixed actions per anchor are correlated. Three independent training seeds, one dataset and mask.')

pairs_plot('02_same_critic_accuracy',
    'Does the BCA adjustment improve the same critic?',
    'BCA-trained critics only  |  Same states, actions and own-policy reference for each pair  |  Final checkpoint',
    lambda ctx,cond,s:tuple(snap(ctx,cond,'bca',s)['q_panels']['high_band_fixed_primary'][k]['mae'] for k in ('raw','adjusted')),
    'Raw Q1','Adjusted Q1 − 3U',
    'High-load fixed-band actions; smaller MAE is better. The strong-current, full-data seed 0 is a counterexample: 16.3 → 20.8.\n21 ordinary / 32 strong-current anchor families per seed, with three correlated actions per anchor. This tests score adjustment, not policy-return causation.')

pairs_plot('04_visited_accuracy',
    'Accuracy on states each policy actually visits',
    'BCA-trained critics, free-policy trajectories  |  Same visited action and discounted reward suffix in each pair  |  Final checkpoint',
    lambda ctx,cond,s:tuple(snap(ctx,cond,'bca',s)['visited_panels']['all'][k]['mae'] for k in ('raw','adjusted')),
    'Raw Q1','Adjusted Q1 − 3U',
    '32 episodes per seed and context, sampled at t = 0, 25, 50 and 75 (128 dependent rows). Smaller MAE is better.\nThese are freely visited actions, not the forced-action anchor bank; the observed removal-mask label does not establish local joint support.')

# Per-action changes and sign reversals are separate quantities: a crossing can
# improve or worsen absolute error. The bars are counts, not replicate counts.
fig,axs=plt.subplots(2,2,figsize=(14.4,8.8))
header(fig,'Pessimism helps many scores — and harms others',
       'BCA-trained critics in the conditional-hole condition  |  High-load fixed-band forced actions  |  Final checkpoint')
for c,(ctx,label) in enumerate(contexts):
    vals=[snap(ctx,'hole','bca',s)['q_panels']['high_band_fixed_primary'] for s in range(3)]
    n=vals[0]['raw']['rows']; y=np.arange(3)
    ax=axs[0,c];panel(ax)
    imp=np.array([v['same_critic_change']['improved'] for v in vals]); bad=np.array([v['same_critic_change']['worsened'] for v in vals]); tie=np.array([v['same_critic_change']['near_tied'] for v in vals])
    ax.barh(y,imp,color=TEAL,height=.54,label='Improved',zorder=2);ax.barh(y,bad,left=imp,color=RED,height=.54,label='Worsened',zorder=2)
    if np.any(tie): ax.barh(y,tie,left=imp+bad,color='#c3cdce',height=.54,zorder=2)
    for i in range(3):
        ax.text(imp[i]/2,i,str(imp[i]),ha='center',va='center',color='white',weight='bold',fontsize=10)
        if bad[i]: ax.text(imp[i]+bad[i]/2,i,str(bad[i]),ha='center',va='center',color='white',weight='bold',fontsize=10)
        crossing=vals[i]['same_critic_change']['class_transitions']['over_to_under']
        ax.text(n+2,i,f'{crossing} over → under',ha='left',va='center',color=TEXT,fontsize=9.5)
        assert imp[i]+bad[i]+tie[i]==n
    ax.set_yticks(y,[f'Seed {i}' for i in range(3)]);ax.invert_yaxis();ax.set_xlim(0,n*1.53)
    ax.set_title(f'{label}  ·  {n} actions per seed',loc='left',weight='bold',fontsize=12)
    ax.set_xlabel('Number of action errors  ·  improvement / worsening > 0.1')
    ax=axs[1,c];panel(ax)
    for i,v in enumerate(vals):
        for dy,key in [(-.19,'raw'),(.19,'adjusted')]:
            yy=i+dy; p=v[key]; left=0
            for count,color in [(p['over'],GOLD),(p['near_zero'],'#c3cdce'),(p['under'],RAW)]:
                ax.barh(yy,count,left=left,height=.30,color=color,zorder=2);left+=count
            assert left==n
            ax.text(n+2,yy,('raw' if key=='raw' else 'adjusted')+f"  {p['over']} over / {p['under']} under",va='center',fontsize=8.6,color=TEXT)
    ax.set_yticks(y,[f'Seed {i}' for i in range(3)]);ax.invert_yaxis();ax.set_xlim(0,n*1.70)
    ax.set_title('Error sign before and after adjustment',loc='left',weight='bold',fontsize=12)
    ax.set_xlabel('Number of action errors  ·  over > +0.1; under < −0.1')
fig.legend(handles=[Patch(facecolor=TEAL,label='Absolute error improved'),Patch(facecolor=RED,label='Absolute error worsened'),
                    Patch(facecolor=GOLD,label='Overestimated'),Patch(facecolor=RAW,label='Underestimated'),Patch(facecolor='#c3cdce',label='Within ±0.1')],
           loc='upper right',bbox_to_anchor=(.965,.885),ncol=3,frameon=False,fontsize=9)
footer(fig,'Signed error = score − own-policy Qπ; the same BCA-trained policy is used before and after adjustment. A sign crossing is not necessarily a worsening.\nOrdinary: 21 anchor families × 3 actions; strong current: 32 × 3. Correlated action counts are descriptive, not independent seed replicates.\nSource: authenticated evaluation report 55cae667d369; see figure manifest for provenance.')
fig.subplots_adjust(left=.09,right=.96,top=.79,bottom=.16,wspace=.23,hspace=.58)
save(fig,'03_adjustment_harm')

# Accuracy rescue separates measured baseline damage from the BCA-specific interaction.
fig,axs=plt.subplots(2,2,figsize=(14.4,8.6),sharex=True)
header(fig,'Did BCA repair accuracy damage from the hole?',
       'Final checkpoint  |  High-load fixed-band actions  |  Each point is one independently trained seed')
screens=data['screens_and_comparisons']
for r,(ctx,label) in enumerate(contexts):
    for c,(key,title,detail) in enumerate([
        ('native_hole_minus_random_mae','Was the native critic harmed?','Hole MAE − random MAE'),
        ('accuracy_hole_minus_random','Was BCA’s benefit specific to the hole?','BCA accuracy benefit in hole − benefit in random')]):
        ax=axs[r,c];panel(ax); v=screens[key][ctx]
        ax.axvline(0,color=RAW,lw=1.1,zorder=1)
        for i,value in enumerate(v):
            y=2-i;ax.plot([0,value],[y,y],color='#a8b4b7',lw=2,zorder=2)
            ax.scatter(value,y,s=90,color=TEAL if value>0 else RED,zorder=3)
            ax.annotate(f'{value:+.2f}',(value,y),xytext=(8 if value>=0 else -8,0),
                        textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=10,weight='bold')
        ax.set_yticks([2,1,0],['Seed 0','Seed 1','Seed 2']);ax.set_ylim(-.5,2.5);ax.set_xlim(-5,22)
        ax.set_title(f'{label} / {title}\n{detail}',loc='left',fontsize=11,weight='bold',pad=10)
        if r==1:ax.set_xlabel('MAE difference in discounted-return units  ·  positive is to the right')
footer(fig,'The ordinary-current accuracy-rescue screen required native damage ≥ 1, a passing accuracy screen, and a positive interaction in every seed; it failed.\nCross-arm accuracy compares each trained policy against its own Qπ; it is not a same-reference treatment effect. Three seeds, one fixed dataset and mask.\nSource: authenticated evaluation report 55cae667d369; see figure manifest for provenance.')
fig.subplots_adjust(left=.095,right=.97,top=.78,bottom=.17,wspace=.28,hspace=.52)
save(fig,'05_accuracy_rescue')

# Different load and action strata are controls, not causal comparisons of load.
strata=[('high_band_fixed_primary','High load / band'),('high_flanks_fixed','High load / flanks'),
        ('low_band_fixed_control','Low load / band'),('low_flanks_fixed','Low load / flanks')]
fig,axs=plt.subplots(2,4,figsize=(16,8.5),sharex=True)
header(fig,'Where does the score adjustment change accuracy?',
       'Conditional-hole BCA critic  |  Fixed complete actions  |  Same-critic raw and adjusted MAE, by training seed')
for r,(ctx,label) in enumerate(contexts):
    for c,(name,stitle) in enumerate(strata):
        ax=axs[r,c];panel(ax)
        ps=[snap(ctx,'hole','bca',i)['q_panels'][name] for i in range(3)]
        n=ps[0]['raw']['rows'];states=ps[0]['state_count']
        ax.set_title(f'{label} / {stitle}\n{states} anchors · {n} actions',loc='left',weight='bold',fontsize=10.5,pad=10)
        if not n:
            ax.text(.5,.5,'No anchors in this stratum',transform=ax.transAxes,ha='center',va='center',fontsize=10,color=MUTED)
        else:
            for i,panel_data in enumerate(ps):
                a=panel_data['raw']['mae'];b=panel_data['adjusted']['mae'];y=2-i
                ax.plot([a,b],[y,y],color='#a8b4b7',lw=1.8,zorder=2)
                ax.scatter([a],[y],s=62,color=RAW,zorder=3);ax.scatter([b],[y],s=62,color=TEAL,zorder=4)
                ax.annotate(f'{a:.1f}',(a,y),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=8,color=RAW)
                ax.annotate(f'{b:.1f}',(b,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8,color=TEAL)
        ax.set_yticks([2,1,0],['Seed 0','Seed 1','Seed 2']);ax.set_ylim(-.5,2.55);ax.set_xlim(0,60)
        if r==1:ax.set_xlabel('MAE (return units) · lower is better')
fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=RAW,markeredgecolor=RAW,label='Raw Q1',markersize=8),
                    Line2D([0],[0],marker='o',color='none',markerfacecolor=TEAL,markeredgecolor=TEAL,label='Adjusted Q1 − 3U',markersize=8)],
           loc='upper right',bbox_to_anchor=(.965,.884),frameon=False,ncol=2,fontsize=10)
footer(fig,'Band actions: a₀ = −0.25, 0, +0.25. Flanks: a₀ = −0.5, +0.5; remaining commands are zero. High/low load also changes the state distribution.\nStrong-current anchor bank has no low-load anchors. These action rows share states; no joint support or causal effect of load is established.\nSource: authenticated evaluation report 55cae667d369; see figure manifest for provenance.')
fig.subplots_adjust(left=.075,right=.97,top=.78,bottom=.17,wspace=.34,hspace=.51)
save(fig,'06_accuracy_by_stratum')

# Re-read and authenticate the common-reference artifact at use time.
report_path=RERUN/'run/evaluation/report.json'
report_bytes=report_path.read_bytes()
assert hashlib.sha256(report_bytes).hexdigest()==data['source_report_sha256']
ref_desc=json.loads(report_bytes)['snapshot_results']['common_reference']
ref_bytes=(RERUN/'run/evaluation'/ref_desc['path']).read_bytes()
assert len(ref_bytes)==ref_desc['bytes'] and hashlib.sha256(ref_bytes).hexdigest()==ref_desc['sha256']
common=json.loads(ref_bytes)
def common_pair(ctx,cond,seed):
    prefix=f'seed{seed}_{cond}'
    native=common[f'{prefix}__native__20000__{ctx}']
    bca=common[f'{prefix}__bca__20000__{ctx}']
    assert native['reference_policy']==bca['reference_policy']==f'seed{seed}_full/native/20000'
    assert native['bank_id']==bca['bank_id'] and native['own_actor_slot_excluded'] and bca['own_actor_slot_excluded']
    return (native['panels']['high_band_fixed']['raw_score_deviation']['mae'],
            bca['panels']['high_band_fixed']['adjusted_score_deviation']['mae'])
pairs_plot('07_common_reference',
    'How do scores compare against one shared policy reference?',
    'High-load fixed-band actions  |  Same states, actions and full-data native 20k reference per seed  |  Final checkpoint',
    common_pair,'Native Q1','BCA Q1 − 3U',
    'Reference is the full-data native 20k continuation for the same seed and context, not each model’s own policy. This is deviation from a common reference,\nnot operational accuracy for every policy. 21 ordinary / 32 strong-current anchors per seed; three correlated actions per anchor.')

manifest={'source_report_sha256':data['source_report_sha256'],'snapshot_count':144,'seed_count':3,'checkpoint_critic_updates':20000,
          'authentication':'extract.py verifies terminal, reproduction, full coverage and all referenced hashes',
          'outputs':['01_cross_model_accuracy','02_same_critic_accuracy','03_adjustment_harm','04_visited_accuracy','05_accuracy_rescue','06_accuracy_by_stratum','07_common_reference'],
          'formats':['png 300 dpi','pdf','svg'],'scientific_limits':'three training seeds; one dataset/mask; anchor action counts dependent; own-policy Q references'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest,indent=2))

