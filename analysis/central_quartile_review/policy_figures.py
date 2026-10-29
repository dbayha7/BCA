"""Recreate policy figures from the authenticated completed evaluation, without simulation.

Run: .venv/Scripts/python.exe -B analysis/central_quartile_review/policy_figures.py
The extractor checks the terminal receipt and all source-artifact hashes first.
"""
from pathlib import Path
import json
import csv
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PACKAGE = ROOT/'experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1'
OUT = BASE/'figures_policy'
OUT.mkdir(exist_ok=True)
raw = subprocess.run([sys.executable, '-B', str(BASE/'extract.py'), str(PACKAGE)], capture_output=True, text=True, check=True)
data = json.loads(raw.stdout)
rows = data['snapshots']; summaries = data['screens_and_comparisons']
COLORS = {'native':'#334155','bca':'#087e8b','distance':'#c36b3b'}
LABELS = {'native':'Native','bca':'BCA','distance':'Distance control'}
CONDITIONS = [('full','Full data'),('hole','Conditional hole'),('random','Random removal')]
CONTEXTS = [('ordinary','Ordinary current'),('strong_current','Strong current')]
ARMS = ['native','bca','distance']
MARKERS = ['o','s','^']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a9b6c0','text.color':'#172b3b','axes.labelcolor':'#172b3b','xtick.color':'#425466','ytick.color':'#425466','figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})

def vals(ctx,cond,arm,t=20000):
    return np.array([rows[f'{ctx}/{cond}/{arm}/{t}/seed{s}']['discounted_return'] for s in range(3)])
def title(fig, head, sub):
    fig.text(.075,.975,head,ha='left',va='top',fontsize=20,fontweight='bold',color='#142d3d')
    fig.text(.075,.93,sub,ha='left',va='top',fontsize=11,color='#526678')
def note(fig, line):
    fig.text(.075,.018,line,ha='left',va='bottom',fontsize=8.5,color='#526678')
def save(fig,name):
    for ext in ('png','svg','pdf'): fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

def trim(ax):
    ax.grid(axis='x',color='#e6edf1',lw=.8,zorder=0)
    ax.set_axisbelow(True)

def marker_legend():
    return [Line2D([],[],ls='',marker=m,color='#526678',markersize=6,label=f'Seed {i}') for i,m in enumerate(MARKERS)]

def figure_returns():
    fig, axes = plt.subplots(1,2,figsize=(13,6.4))
    title(fig,'Final policy returns under both shifts','Higher / less-negative discounted return is better • 20k checkpoint • 32 paired evaluation starts per seed')
    for ax,(ctx,context_label) in zip(axes,CONTEXTS):
        for i,(cond,label) in enumerate(CONDITIONS):
            for j,arm in enumerate(ARMS):
                y=2-i + (.23-j*.23)
                v=vals(ctx,cond,arm)
                ax.plot([min(v),max(v)],[y,y],color=COLORS[arm],lw=1.4,alpha=.38,zorder=1)
                for s in range(3): ax.scatter(v[s],y,marker=MARKERS[s],s=42,facecolor='white',edgecolor=COLORS[arm],linewidth=1.5,zorder=3)
                ax.scatter(v.mean(),y,marker='D',s=38,color=COLORS[arm],zorder=4)
        ax.set_title(context_label,loc='left',fontweight='bold',pad=12)
        ax.set_yticks([2,1,0],['Full data','Conditional hole','Random removal'])
        ax.set_ylim(-.48,2.48);ax.set_xlabel('Discounted return (higher is better)')
        trim(ax)
    legend=[Line2D([],[],ls='',marker='D',color=COLORS[a],markersize=7,label=LABELS[a]+' mean') for a in ARMS]+marker_legend()
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.10),ncol=6,frameon=False,fontsize=9.5)
    note(fig,'Each open marker is one training-seed mean. Diamonds average the three seeds equally. X-axis ranges differ across contexts.\nSource: authenticated rerun; one fixed dataset and removal mask. Strong current changes the observed initial-current law.')
    fig.subplots_adjust(left=.13,right=.97,top=.82,bottom=.27,wspace=.33)
    save(fig,'01_final_returns')

def figure_contrasts():
    fig,axes=plt.subplots(2,2,figsize=(12.6,8),sharex=False)
    title(fig,'Does BCA improve policy return?','Final checkpoint • paired difference within training seed and evaluation starts; values to the right of zero favor BCA')
    for row,(ctx,context_label) in enumerate(CONTEXTS):
      for col,other in enumerate(['native','distance']):
        ax=axes[row,col]
        for i,(cond,label) in enumerate(CONDITIONS):
            v=vals(ctx,cond,'bca')-vals(ctx,cond,other)
            key=f'{ctx}/20000/{cond}/bca_minus_{other}'
            assert np.allclose(v,summaries['policy_comparisons'][key]['seed_means'],rtol=0,atol=1e-9)
            y=2-i
            ax.plot([v.min(),v.max()],[y,y],color='#c4d7d8',lw=3,zorder=1)
            for s in range(3): ax.scatter(v[s],y+(s-1)*.12,s=54,marker=MARKERS[s],color=COLORS['bca'],edgecolor='white',linewidth=.6,zorder=3)
            ax.scatter(v.mean(),y,marker='D',s=35,color='#143d43',zorder=4)
        ax.axvline(0,color='#526678',lw=1.1)
        if ctx=='ordinary' and other=='native': ax.axvline(1,color='#a8b9c1',lw=1,ls='--')
        ax.set_title(f'{context_label}  |  BCA − {LABELS[other]}',loc='left',fontweight='bold')
        ax.set_yticks([2,1,0],[v[1] for v in CONDITIONS]);ax.set_ylim(-.45,2.45)
        ax.set_xlabel('Difference in discounted return')
        trim(ax)
    fig.legend(handles=marker_legend()+[Line2D([],[],ls='',marker='D',color='#143d43',markersize=6,label='Mean of seeds')],loc='lower center',bbox_to_anchor=(.5,.094),ncol=4,frameon=False,fontsize=9.5)
    note(fig,'The primary screen required ordinary-hole BCA − native ≥ +1 in every seed and BCA − distance ≥ 0 in every seed; it failed.\nDotted line marks +1 in the primary panel. Three seed means are independent training replications; no significance claim.')
    fig.subplots_adjust(left=.15,right=.97,top=.82,bottom=.24,wspace=.36,hspace=.48)
    save(fig,'02_policy_contrasts')

def figure_interaction():
    fig,axes=plt.subplots(1,2,figsize=(12.6,6.1),sharey=True)
    title(fig,'Did BCA specifically rescue the missing-label hole?','Final checkpoint • paired seed-level differences • the horizontal zero line means no effect')
    for ax,(ctx,label) in zip(axes,CONTEXTS):
       for j,s in enumerate(range(3)):
        damage=vals(ctx,'hole','native')[s]-vals(ctx,'random','native')[s]
        inter=(vals(ctx,'hole','bca')[s]-vals(ctx,'hole','native')[s])-(vals(ctx,'random','bca')[s]-vals(ctx,'random','native')[s])
        expected=summaries['return_interactions'][f'{ctx}/20000/bca_minus_native/hole_minus_random']['seed_means'][s]
        assert abs(inter-expected)<1e-9
        ax.scatter(j-.12,damage,marker=MARKERS[s],s=90,color=COLORS['native'],edgecolor='white',lw=.8,zorder=3)
        ax.scatter(j+.12,inter,marker=MARKERS[s],s=90,color=COLORS['bca'],edgecolor='white',lw=.8,zorder=3)
       ax.axhline(0,color='#526678',lw=1.1)
       ax.grid(axis='y',color='#e6edf1',lw=.8);ax.set_axisbelow(True)
       ax.set_title(label,loc='left',fontweight='bold',pad=12)
       ax.set_xticks(range(3),['Seed 0','Seed 1','Seed 2']);ax.set_xlabel('Independent training seed')
       ax.set_ylabel('Difference in discounted return')
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS['native'],label='Native: hole − random'),Line2D([],[],marker='o',ls='',color=COLORS['bca'],label='BCA effect: hole − random')],loc='lower center',bbox_to_anchor=(.5,.14),ncol=2,frameon=False)
    note(fig,'BCA effect = (BCA − native in hole) − (BCA − native in random). The prespecified rescue screen failed.\nThe random control matches removal counts and coarse strata, not every state or reward distribution.')
    fig.subplots_adjust(left=.11,right=.97,top=.80,bottom=.29,wspace=.16)
    save(fig,'03_hole_interaction')

def figure_curves():
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    title(fig,'How returns changed after the shared 10k fork','Fixed snapshots at 10k, 15k and 20k critic updates • lines connect observations; no checkpoint selection')
    for r,(ctx,context) in enumerate(CONTEXTS):
      for c,(cond,label) in enumerate(CONDITIONS):
       ax=axes[r,c]
       for arm in ARMS:
        history=np.array([vals(ctx,cond,'native',10000),vals(ctx,cond,arm,15000),vals(ctx,cond,arm,20000)])
        for s in range(3): ax.plot([10,15,20],history[:,s],color=COLORS[arm],lw=.8,alpha=.18)
        ax.plot([10,15,20],history.mean(axis=1),color=COLORS[arm],lw=2.2,marker='o',markersize=4,zorder=3)
       ax.set_title(f'{context} · {label}',loc='left',fontweight='bold',fontsize=11)
       ax.grid(axis='y',color='#e6edf1',lw=.8);ax.set_axisbelow(True)
       ax.set_xticks([10,15,20]);ax.set_xlabel('Critic updates (thousands)')
       if c==0: ax.set_ylabel('Discounted return\n(higher is better)')
    fig.legend(handles=[Line2D([],[],color=COLORS[a],lw=2,label=LABELS[a]) for a in ARMS],loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False)
    note(fig,'Thick lines are equal-seed means; faint lines are individual seeds. The 10k point is shared by the three branches within each cell.\nAxis ranges differ across panels; untrained 0k is omitted to show post-fork changes. The frozen comparison is 20k.')
    fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.23,wspace=.30,hspace=.52)
    save(fig,'04_post_fork_learning')

if __name__=='__main__':
    assert data['reproduction']['tier1_exact']==143 and data['reproduction']['tier2_near_exact']==0
    for fn in [figure_returns,figure_contrasts,figure_interaction,figure_curves]: fn()
    manifest={'source_report_sha256':data['source_report_sha256'],'source':'144 authenticated rerun contexts via extract.py','condition_count':3,'training_seeds':3,'episodes_per_seed_context':32,'discount':0.98,'checkpoint':20000,'figures':{'01_final_returns':'Absolute free-policy discounted returns at 20k','02_policy_contrasts':'BCA minus native and distance per seed at 20k','03_hole_interaction':'Native hole minus random and BCA vs native hole-minus-random interaction at 20k','04_post_fork_learning':'10k, 15k, 20k free-policy return curves'}}
    with (OUT/'policy_plot_data.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['evaluation_context','training_data','arm','critic_updates','seed','mean_discounted_return_32_episodes'])
        for ctx,_ in CONTEXTS:
            for condition,_ in CONDITIONS:
                for arm,step in [('native',0),('native',10000)]+[(a,t) for a in ARMS for t in (15000,20000)]:
                    for seed,value in enumerate(vals(ctx,condition,arm,step)):
                        w.writerow([ctx,condition,arm,step,seed,repr(float(value))])
    manifest['data_csv']='policy_plot_data.csv'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))
