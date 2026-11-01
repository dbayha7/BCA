"""Render source-bound, per-seed figures for the common-action diagnostic.

This is post-run analysis only. It reads a completed immutable RESULT.json,
does not query the simulator, select models, or change diagnostic inputs.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE/'diagnostic_run_01'/'RESULT.json'
CONTEXTS = ('ordinary', 'strong_current')
CTX_LABELS = ('Ordinary current', 'Strong current')
ARMS = ('native', 'bca')
SOURCES = ('random_prefix', 'native_visited', 'bca_visited')
SOURCE_LABELS = ('Random-prefix states', 'Native-actor visited', 'BCA-actor visited')
COMPARISONS = ('raw_to_symmetric', 'symmetric_to_guarded', 'guarded_to_unguarded')
COLORS = ('#0072B2', '#D55E00', '#7851A9')
MARKERS = ('o', 's', '^')
SEED_OFFSETS = (-.13, 0., .13)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':12,
    'axes.labelsize':10, 'axes.spines.top':False, 'axes.spines.right':False,
    'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white', 'figure.facecolor':'white'})


def require(ok, message):
    if not ok: raise ValueError(message)


def load_result(path=RESULT):
    raw=Path(path).read_bytes(); result=json.loads(raw)
    require(result.get('status')=='completed_common_action_continuation_diagnostic', 'completed result required')
    ledger=result['ledger']; require(ledger['transition_steps_completed']==ledger['transition_steps_attempted']==364904
        and ledger['continuation_rows_completed']==5760 and ledger['reconstruction_refreshes']==18
        and all(ledger[k]==0 for k in ('actor_updates','critic_updates','actor_target_updates',
                                        'critic_target_updates','auxiliary_updates')), 'incomplete or nonfrozen ledger')
    cases=result['cases']; require(len(cases)==3 and sorted(c['seed'] for c in cases)==[0,1,2],
                                  'three distinct paired seed cases required')
    return result, hashlib.sha256(raw).hexdigest()


def get(cases, seed, context, critic, continuation, source, group='all'):
    case=next(c for c in cases if c['seed']==seed)
    return case['contexts'][context]['summaries'][critic][continuation][source][group]


def clean_axes(ax):
    ax.set_axisbelow(True); ax.grid(axis='x',color='#dfe6eb',linewidth=.8)
    ax.tick_params(axis='both',length=0,pad=5)


def seed_handles():
    return [Line2D([0],[0],marker=MARKERS[i],color='none',markerfacecolor=COLORS[i],
                   markeredgecolor=COLORS[i],markersize=7,label=f'Training seed {i}') for i in range(3)]


def save(fig, stem, output):
    fig.savefig(output/(stem+'.png'),dpi=210,bbox_inches='tight')
    fig.savefig(output/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def figure_cross(cases, output):
    combos=[('native','native'),('native','bca'),('bca','native'),('bca','bca')]
    labels=('Native critic → native policy','Native critic → BCA policy',
            'BCA critic → native policy','BCA critic → BCA policy')
    vals=[]
    for cx in CONTEXTS:
        for src in SOURCES:
            for critic,cont in combos:
                for seed in range(3):
                    val=get(cases,seed,cx,critic,cont,src)['comparisons']['symmetric_to_guarded']['mean_gain']
                    if val is not None: vals.append(abs(val))
    limit=max(.1,max(vals,default=.1))*1.2
    fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True)
    fig.suptitle('Common action bank: does compatibility improve on symmetric BCA?',fontweight='bold',fontsize=18,y=.995)
    fig.legend(handles=seed_handles(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
    for row,cx in enumerate(CONTEXTS):
        for col,src in enumerate(SOURCES):
            ax=axes[row,col];clean_axes(ax); ax.axvline(0,color='#546678',linewidth=1)
            for j,(critic,cont) in enumerate(combos):
                for seed in range(3):
                    x=get(cases,seed,cx,critic,cont,src)['comparisons']['symmetric_to_guarded']['mean_gain']
                    if x is not None: ax.scatter(x,j+SEED_OFFSETS[seed],s=40,c=COLORS[seed],marker=MARKERS[seed],zorder=3)
            ax.set_yticks(range(4),labels if col==0 else ['']*4);ax.invert_yaxis();ax.set_xlim(-limit,limit)
            ax.set_title(CTX_LABELS[row]+' · '+SOURCE_LABELS[col],loc='left',fontweight='bold')
            ax.set_xlabel('Mean Qπ gain: guarded hybrid − symmetric (return units)')
    fig.subplots_adjust(left=.21,right=.985,top=.90,bottom=.08,hspace=.48,wspace=.12)
    save(fig,'01_common_action_gain',output)


def figure_decomposition(cases, output):
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
    values=[]
    for cx in CONTEXTS:
        for arm in ARMS:
            for seed in range(3):
                comparisons=get(cases,seed,cx,arm,arm,arm+'_visited')['comparisons']
                values.extend(abs(comparisons[k]['mean_gain']) for k in ('raw_to_symmetric','symmetric_to_guarded'))
    limit=max(.1,max(values,default=.1))*1.2
    fig.suptitle('Own-policy visited states: where action changes come from',fontsize=18,fontweight='bold',y=.99)
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#0072B2',markeredgecolor='#0072B2',label='Raw Q1 → symmetric BCA'),
             Line2D([0],[0],marker='s',color='none',markerfacecolor='#D55E00',markeredgecolor='#D55E00',label='Symmetric → guarded hybrid')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=2,frameon=False)
    for row,cx in enumerate(CONTEXTS):
        for col,arm in enumerate(ARMS):
            ax=axes[row,col];clean_axes(ax);ax.axvline(0,color='#546678',lw=1)
            for seed in range(3):
                comp=get(cases,seed,cx,arm,arm,arm+'_visited')['comparisons']
                for off,key,color,marker in ((-.12,'raw_to_symmetric','#0072B2','o'),(.12,'symmetric_to_guarded','#D55E00','s')):
                    ax.scatter(comp[key]['mean_gain'],seed+off,s=58,color=color,marker=marker,zorder=3)
            ax.set_yticks(range(3),[f'Seed {i}' for i in range(3)]);ax.invert_yaxis();ax.set_xlim(-limit,limit)
            ax.set_title(CTX_LABELS[row]+' · '+arm.upper()+' checkpoint',loc='left',fontweight='bold')
            ax.set_xlabel('Mean selected Qπ change (return units)')
    fig.subplots_adjust(left=.10,right=.985,top=.88,bottom=.09,hspace=.42,wspace=.20)
    save(fig,'02_width_and_compatibility',output)


def figure_sensitivity(cases, output):
    fig,axes=plt.subplots(2,3,figsize=(17,8),sharex=True)
    vals=[]
    for cx in CONTEXTS:
        for src in SOURCES:
            for arm in ARMS:
                for seed in range(3):
                    n=get(cases,seed,cx,arm,'native',src)['comparisons']['symmetric_to_guarded']['mean_gain']
                    b=get(cases,seed,cx,arm,'bca',src)['comparisons']['symmetric_to_guarded']['mean_gain']
                    vals.append(b-n)
    limit=max(.1,max(map(abs,vals),default=.1))*1.2
    fig.suptitle('Does the same score change look different under another continuation?',fontsize=18,fontweight='bold',y=.99)
    fig.legend(handles=seed_handles(),loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    for row,cx in enumerate(CONTEXTS):
        for col,src in enumerate(SOURCES):
            ax=axes[row,col];clean_axes(ax);ax.axvline(0,color='#546678',lw=1)
            for j,arm in enumerate(ARMS):
                for seed in range(3):
                    n=get(cases,seed,cx,arm,'native',src)['comparisons']['symmetric_to_guarded']['mean_gain']
                    b=get(cases,seed,cx,arm,'bca',src)['comparisons']['symmetric_to_guarded']['mean_gain']
                    ax.scatter(b-n,j+SEED_OFFSETS[seed],s=45,c=COLORS[seed],marker=MARKERS[seed],zorder=3)
            ax.set_yticks(range(2),['Native critic','BCA critic'] if col==0 else ['','']);ax.invert_yaxis();ax.set_xlim(-limit,limit)
            ax.set_title(CTX_LABELS[row]+' · '+SOURCE_LABELS[col],loc='left',fontweight='bold')
            ax.set_xlabel('Hybrid gain under BCA continuation − native continuation')
    fig.subplots_adjust(left=.13,right=.985,top=.87,bottom=.09,hspace=.48,wspace=.12)
    save(fig,'03_continuation_sensitivity',output)


def figure_switches(cases, output):
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharex=True,sharey=True)
    fig.suptitle('Own-policy visited states: improved and harmed choices',fontsize=18,fontweight='bold',y=.99)
    for col,cx in enumerate(CONTEXTS):
        ax=axes[col];clean_axes(ax);ax.axvline(0,color='#546678',lw=1)
        labels=[]
        for j,(arm,seed) in enumerate((a,s) for a in ARMS for s in range(3)):
            z=get(cases,seed,cx,arm,arm,arm+'_visited')['comparisons']['symmetric_to_guarded']
            ax.barh(j,-z['harmed'],color='#C64D4D',height=.58);ax.barh(j,z['improved'],color='#008766',height=.58)
            ax.text(-z['harmed']-.12,j,str(z['harmed']),ha='right',va='center',fontsize=9)
            ax.text(z['improved']+.12,j,str(z['improved']),ha='left',va='center',fontsize=9)
            ax.text(7.5,j,f"{z['near_tied']} near-tied",ha='right',va='center',fontsize=9,color='#526578')
            labels.append(f'{arm.upper()} · seed {seed}')
        ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(-8.2,8.2)
        ax.set_title(CTX_LABELS[col],loc='left',fontweight='bold')
        ax.set_xlabel('Harmed  ←  changed states  →  improved')
    handles=[Line2D([0],[0],color='#C64D4D',lw=8,label='Harmed by more than 0.1'),
             Line2D([0],[0],color='#008766',lw=8,label='Improved by more than 0.1')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
    fig.subplots_adjust(left=.15,right=.98,top=.88,bottom=.15,wspace=.12)
    save(fig,'04_choice_changes',output)


def figure_guard(cases, output):
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharex=True,sharey=True)
    fig.suptitle('Where the state-distance guard applies the added penalty',fontsize=17,fontweight='bold',y=.99)
    fig.legend(handles=seed_handles(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    for col,cx in enumerate(CONTEXTS):
        ax=axes[col];clean_axes(ax)
        for j,src in enumerate(SOURCES):
            for seed in range(3):
                # Same states and reference determine the guard for both critics/continuations.
                total=get(cases,seed,cx,'native','native',src,'all')['states']
                active=get(cases,seed,cx,'native','native',src,'guard_on')['states']
                require(total>0 and 0<=active<=total,'invalid guard counts')
                ax.scatter(100*active/total,j+SEED_OFFSETS[seed],s=50,c=COLORS[seed],marker=MARKERS[seed],zorder=3)
        ax.set_yticks(range(3),SOURCE_LABELS);ax.invert_yaxis();ax.set_xlim(-3,103)
        ax.set_title(CTX_LABELS[col],loc='left',fontweight='bold')
        ax.set_xlabel('States with guard on (%)')
    fig.subplots_adjust(left=.18,right=.985,top=.76,bottom=.16,wspace=.14)
    save(fig,'05_guard_activation',output)


def write_csv(cases, output):
    names=['context','seed','critic','continuation','state_source','guard_group','states','comparison',
           'mean_gain','changed','improved','harmed','near_tied']
    with (output/'paired_comparisons.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for cx in CONTEXTS:
            for seed in range(3):
                for arm in ARMS:
                    for cont in ARMS:
                        for src in SOURCES:
                            for group in ('all','guard_on','guard_off'):
                                cell=get(cases,seed,cx,arm,cont,src,group)
                                for name in COMPARISONS:
                                    z=cell['comparisons'][name]
                                    writer.writerow({'context':cx,'seed':seed,'critic':arm,'continuation':cont,
                                        'state_source':src,'guard_group':group,'states':cell['states'],'comparison':name,
                                        **{k:z[k] for k in ('mean_gain','changed','improved','harmed','near_tied')}})


def render(path=RESULT, output=None):
    result, sha=load_result(path);cases=result['cases']
    output=HERE/'plots_v1' if output is None else Path(output)
    output.mkdir(parents=False,exist_ok=False)
    figure_cross(cases,output);figure_decomposition(cases,output)
    figure_sensitivity(cases,output);figure_switches(cases,output);figure_guard(cases,output);write_csv(cases,output)
    guide=['# Common-action / common-continuation figure guide','',f'Source: `diagnostic_run_01/RESULT.json`, SHA-256 `{sha}`.','',
      'Actors, critics and auxiliary model parameters remain frozen. Values compare forced first actions followed by a specified saved actor; they are not trained-policy returns or Q-star.',
      'The off-diagonal critic/continuation combinations are cross-policy diagnostics, not that critic’s own-policy accuracy. Three markers are paired training seeds, not independent state/action rows.','',
      '1. **Common-action gain:** change in selected Qπ from symmetric Q1 BCA to guarded hybrid, for each critic/continuation pair; positive is better.',
      '2. **Width and compatibility:** paired raw-to-symmetric and symmetric-to-hybrid changes on each critic’s own visited states with its own continuation.',
      '3. **Continuation sensitivity:** for the same critic, states, actions and selected alternatives, hybrid gain under BCA continuation minus gain under native continuation.',
      '4. **Choice changes:** counts of gains or losses greater than 0.1 on eight own-policy visited states per seed; near ties are labeled.',
      '5. **Guard activation:** proportion of states where the extra compatibility cost is active; shared-state results are repeated for each paired seed.','',
      'Random-prefix, native-visited and BCA-visited sources are reported separately. A state-distance gate is not a support certificate. Each seed has 16 random-prefix and eight states from each actor; states and candidate actions are dependent.',
      'The full all/guard-on/guard-off comparison data are in `paired_comparisons.csv`. The underlying RESULT retains all action-level values and scores.','',
      'Prespecified own-visited screen:']
    for cx in CONTEXTS:
        s=result['screen'][cx]
        guide.append(f"- {cx}: **{s['status']}**; qualifying native {s['qualifying_seeds']['native']}/3, BCA {s['qualifying_seeds']['bca']}/3; eligible native {s['eligible_seeds']['native']}/3, BCA {s['eligible_seeds']['bca']}/3.")
    (output/'FIGURE_GUIDE.md').write_text('\n'.join(guide)+'\n',encoding='utf8')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'PLOT_RECEIPT.json').write_text(json.dumps({'source_sha256':sha,'files':files},indent=2,sort_keys=True)+'\n',encoding='utf8')
    return output


if __name__=='__main__':
    require(len(sys.argv)==1,'Usage: python plot_results.py')
    print(render())
