"""Forced-bank and policy-exposure figures from a fully verified evaluation.

Re-runs extract.py first, so incomplete or changed inputs stop before plotting.
"""
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1'
OUT = HERE / 'figures_decisions'
DATA = json.loads(subprocess.run([sys.executable, '-B', str(HERE/'extract.py'), str(RUN)],
    check=True, capture_output=True, text=True).stdout)['snapshots']
COL = {'native':'#334155', 'bca':'#087e8b', 'distance':'#c36b3b'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white', 'axes.labelcolor':'#273444','text.color':'#172534',
    'xtick.color':'#273444','ytick.color':'#273444'})

def row(ctx, cond, arm, seed):
    return DATA[f'{ctx}/{cond}/{arm}/20000/seed{seed}']

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def decisions():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), sharex=True, sharey=True)
    fig.suptitle('Does the pessimistic score change forced-action decisions?', x=.06, ha='left',
                 fontsize=20, fontweight='bold', y=.98)
    fig.text(.06, .93, 'BCA-trained policy at 20k updates · 32 shared anchor states per context and seed', fontsize=12)
    maxreg=0
    for i, context in enumerate(('ordinary','strong_current')):
        for j, cond in enumerate(('full','hole','random')):
            ax=axes[i,j]
            for seed in range(3):
                d=row(context,cond,'bca',seed)
                raw=d['raw_decisions']; adj=d['adjusted_decisions']
                x,y=raw['mean_regret'],adj['mean_regret']
                maxreg=max(maxreg,x,y)
                ax.plot([x,y],[seed,seed], color='#a9b3bc',lw=2,zorder=1)
                ax.scatter(x,seed,s=56,color=COL['native'],zorder=2)
                ax.scatter(y,seed,s=56,color=COL['bca'],zorder=3)
                changed=sum(a != b for a,b in zip(raw['selected_slot'],adj['selected_slot']))
                change=d['signed_candidate_margins']['all']['adjustment_choice_change']
                assert changed == change['changed_states']
                ax.text(.99,seed, f"{changed} changed  ·  {change['improved_states']} improved  ·  {change['harmed_states']} harmed",
                        transform=ax.get_yaxis_transform(), ha='right', va='center',fontsize=9, color='#405268')
            ax.set_yticks(range(3),['Seed 0','Seed 1','Seed 2'])
            ax.invert_yaxis() if i==0 and j==0 else None
            ax.set_title(f"{'Ordinary' if i==0 else 'Strong current'} · {cond.title()}",loc='left',fontsize=12,pad=9)
            ax.grid(axis='x',color='#e2e8ed',lw=.8); ax.set_axisbelow(True)
    for ax in axes[-1]: ax.set_xlabel('Mean oracle-bank regret (discounted return units)')
    # The explanatory annotations use axes coordinates on the right; a generous
    # x-range keeps them from obscuring the data and preserves a common scale.
    for ax in axes.flat: ax.set_xlim(0,max(1,maxreg*2.9))
    fig.text(.06,.155,'●  Raw Q1',color=COL['native'],fontsize=11,weight='bold')
    fig.text(.17,.155,'●  Q1 − 3U',color=COL['bca'],fontsize=11,weight='bold')
    fig.text(.06,.115,'One forced action from a six-action bank, then that checkpoint’s own actor. Regret is relative to the best action in this bank only.',fontsize=10)
    fig.text(.06,.08,'Improved / harmed: change in true bank-choice Qπ greater than +0.1 / less than −0.1. Changed choices with near-tied value are not counted as either.',fontsize=10)
    fig.text(.06,.04,'Source: authenticated central-quartile rerun · fixed 20k snapshots · 3 independent training seeds · not learned-policy return',fontsize=9,color='#596b79')
    fig.subplots_adjust(left=.08,right=.98,top=.87,bottom=.24,wspace=.20,hspace=.42)
    save(fig,'forced_bank_decisions')

def exposure():
    fig, axes=plt.subplots(2,3,figsize=(15,8.7),sharex=True,sharey=True)
    fig.suptitle('How often do the free policies enter the withheld region?',x=.06,ha='left',y=.98,fontsize=20,fontweight='bold')
    fig.text(.06,.93,'Share of high-load rollout steps where the first action is in the withheld band · 32 episodes per seed',fontsize=12)
    for i,ctx in enumerate(('ordinary','strong_current')):
        for j,cond in enumerate(('full','hole','random')):
            ax=axes[i,j]
            for arm,y in zip(('native','bca','distance'),(2,1,0)):
                values=[row(ctx,cond,arm,s)['high_step_hole_fraction']*100 for s in range(3)]
                ax.plot([min(values),max(values)],[y,y],color=COL[arm],alpha=.36,lw=2)
                for s,value in enumerate(values):
                    ax.scatter(value,y+(s-1)*.13,color=COL[arm],s=45,marker=('o','s','^')[s],zorder=3)
                ax.text(np.mean(values),y+.29,f'{np.mean(values):.1f}%',ha='center',fontsize=9,color=COL[arm])
            ax.set_yticks([2,1,0],['Native','BCA','Distance'])
            ax.set_title(f"{'Ordinary' if i==0 else 'Strong current'} · {cond.title()}",loc='left',fontsize=12,pad=9)
            ax.set_xlim(0,100);ax.set_ylim(-.55,2.6);ax.grid(axis='x',color='#e2e8ed');ax.set_axisbelow(True)
    for ax in axes[-1]:ax.set_xlabel('High-load steps in withheld band (%)')
    fig.text(.06,.15,'● Seed 0     ■ Seed 1     ▲ Seed 2     Label = equal-seed mean',fontsize=10)
    fig.text(.06,.105,'Exposure is a mask label on visited state–action pairs; it does not establish local state density or joint-action support.',fontsize=10)
    fig.text(.06,.06,'Source: authenticated central-quartile rerun · free policies · final 20k snapshots · 3 independent training seeds',fontsize=9,color='#596b79')
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.24,wspace=.20,hspace=.42)
    save(fig,'free_policy_hole_exposure')

def signed_margins():
    fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
    fig.suptitle('Is the center action better than either retained flank?',x=.06,ha='left',y=.98,fontsize=20,fontweight='bold')
    fig.text(.06,.93,'BCA-trained final policies · high-load forced-action anchors · each point is a state family',fontsize=12)
    values=[]
    for i,ctx in enumerate(('ordinary','strong_current')):
        for j,cond in enumerate(('full','hole','random')):
            ax=axes[i,j]; ax.axvline(0,color='#536477',lw=1.3)
            for seed in range(3):
                panel=row(ctx,cond,'bca',seed)['signed_candidate_margins']['high_load_hole']
                v=np.asarray(panel['center_minus_best_retained_qpi'],dtype=float)
                assert len(v)==(21 if ctx=='ordinary' else 32)
                assert (v>.1).sum()==panel['center_better_than_retained']
                assert (v<-.1).sum()==panel['center_worse_than_retained']
                values.extend(v)
                jitter=np.linspace(-.18,.18,len(v))
                ax.scatter(v,seed+jitter,s=15,color='#087e8b',alpha=.5,linewidth=0,zorder=2)
                ax.scatter(v.mean(),seed,s=65,marker='D',color='#273b50',zorder=3)
                ax.text(.99,seed,f"{(v>.1).sum()} better / {(v<-.1).sum()} worse",transform=ax.get_yaxis_transform(),
                        ha='right',va='center',fontsize=9,color='#405268')
            ax.set_yticks(range(3),['Seed 0','Seed 1','Seed 2'])
            if i==0 and j==0: ax.invert_yaxis()
            ax.set_title(f"{'Ordinary' if i==0 else 'Strong current'} · {cond.title()}",loc='left',fontsize=12,pad=9)
            ax.grid(axis='x',color='#e2e8ed'); ax.set_axisbelow(True)
    low=min(values);high=max(values); width=high-low
    for ax in axes.flat: ax.set_xlim(low-.05*width, high+.5*width)
    for ax in axes[-1]:ax.set_xlabel('Qπ(center) − max Qπ(flanks), discounted return units')
    fig.text(.06,.155,'● State family      ◆ Mean across state families',fontsize=10)
    fig.text(.06,.115,'Center: a₀ = 0; flanks: a₀ = −0.5 or +0.5; remaining action coordinates are zero. Each value follows the same checkpoint’s actor.',fontsize=10)
    fig.text(.06,.08,'Positive means center is better. Better / worse counts use a ±0.1 tolerance; the remaining states are near tied.',fontsize=10)
    fig.text(.06,.04,'Source: authenticated rerun · 21 ordinary and 32 strong-current high-load anchors per seed · no claim about global optimal actions',fontsize=9,color='#596b79')
    fig.subplots_adjust(left=.08,right=.98,top=.87,bottom=.24,wspace=.20,hspace=.42)
    save(fig,'center_vs_retained_flanks')

if __name__ == '__main__':
    decisions(); exposure(); signed_margins()
    print(f'Wrote forced-bank and exposure figures to {OUT}')
