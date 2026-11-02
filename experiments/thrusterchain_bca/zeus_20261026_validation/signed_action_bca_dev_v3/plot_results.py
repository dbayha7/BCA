"""Render completed expanded-bank diagnostics, keeping training seeds visible.

This reader never alters or selects experiment methods. An output directory is
created only for a completed source with accounted frozen-checkpoint updates.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'diagnostic_run_01/RESULT.json'
CONTEXTS = ('ordinary', 'strong_current')
ARMS = ('native', 'bca')
STRATA = ('natural', 'ambiguous_q1')
METHODS = ('raw_q1', 'q1_aligned', 'signed_bayes', 'signed_mean_matched',
           'signed_gradient_matched', 'revised_compatibility', 'sym_compat_guarded',
           'sym_compat_unguarded', 'sym_hybrid_strength_matched')
LABELS = {'raw_q1':'Raw Q1', 'q1_aligned':'Symmetric Q1 BCA',
          'signed_bayes':'Signed BCA', 'signed_mean_matched':'Signed · matched mean',
          'signed_gradient_matched':'Signed · matched gradient',
          'revised_compatibility':'Compatibility only',
          'sym_compat_guarded':'Symmetric + compatibility',
          'sym_compat_unguarded':'Symmetric + unguarded cost',
          'sym_hybrid_strength_matched':'Matched symmetric width'}
COLORS = ('#165f86', '#bf5a32', '#72509b')
MARKERS = ('o', 's', '^')


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.axisbelow':True,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'savefig.dpi':220,'pdf.fonttype':42,'figure.facecolor':'white'})


def draw(fig, out, name, sha):
    fig.savefig(out/(name+'.png'), dpi=220, bbox_inches='tight',
                metadata={'Description':'Frozen diagnostic source SHA256 '+sha})
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight',
                metadata={'Subject':'Frozen diagnostic source SHA256 '+sha})
    plt.close(fig)


def finite_or_none(value):
    return value is None or np.isfinite(float(value))


def extract(result):
    """Validate and index all score metrics, retaining native inference data."""
    assert result['status']=='completed_expanded_frozen_diagnostic'
    ledger=result['ledger']
    assert ledger['transition_steps_completed']==ledger['transition_steps_attempted']==377704
    assert ledger['continuation_rows_completed']==5760 and ledger['calibration_refreshes']==18
    assert all(ledger[k]==0 for k in ('auxiliary_updates','actor_updates','critic_updates',
                                      'actor_target_updates','critic_target_updates'))
    cases=result['cases']; assert len(cases)==6
    index={(c['arm'],c['seed']):c for c in cases}
    assert len(index)==6 and set(index)=={(a,s) for a in ARMS for s in range(3)}
    for case in cases:
        for context in CONTEXTS:
            block=case['contexts'][context]; probe=block['per_probe']
            truth=np.asarray(probe['qpi'],float); actions=np.asarray(probe['actions'],float)
            assert truth.shape==(32,15) and actions.shape==(32,15,4) and np.isfinite(truth).all()
            guard=np.asarray(probe['components']['reliable'],bool)
            assert guard.shape==(32,15) and np.all(guard==guard[:,:1])
            for stratum in STRATA:
                data=block['summary'][stratum]; assert data['states']==16
                for key in METHODS:
                    m=data['scores'][key]
                    assert finite_or_none(m['ranking_accuracy']) and np.isfinite(m['mean_regret'])
                    assert m['changed']==m['improved']+m['harmed']+m['changed_near_tie']
    return index


def scatter_plot(index, out, sha, *, metric, title, axis_label, name):
    fig, axes=plt.subplots(4,2,figsize=(15,16),sharey=True,constrained_layout=True)
    fig.suptitle(title,fontsize=17,fontweight='bold')
    y=np.arange(len(METHODS))[::-1]
    for row,(context,arm) in enumerate((c,a) for c in CONTEXTS for a in ARMS):
        for col,stratum in enumerate(STRATA):
            ax=axes[row,col]
            for seed in range(3):
                block=index[(arm,seed)]['contexts'][context]['summary'][stratum]
                for yy, method in zip(y,METHODS):
                    value=block['scores'][method][metric]
                    if value is None: continue
                    value=float(value)*(100 if metric=='ranking_accuracy' else 1)
                    ax.scatter(value, yy+(seed-1)*.14, color=COLORS[seed],marker=MARKERS[seed],s=32,zorder=3)
            ax.set_yticks(y,[LABELS[m] for m in METHODS]);ax.grid(axis='x',color='#e4e9ef')
            context_label=context.replace('_',' ').title();group='Natural' if stratum=='natural' else 'Q1-ambiguity selected'
            ax.set_title(f'{context_label} · {arm.upper()} — {group}',loc='left',weight='bold')
            ax.set_xlabel(axis_label)
            if metric=='ranking_accuracy': ax.set_xlim(-4,104)
            if col==0: ax.tick_params(labelleft=True)
    handles=[plt.Line2D([0],[0],marker=MARKERS[i],color='none',markerfacecolor=COLORS[i],
                        markeredgecolor=COLORS[i],label=f'Seed {i}') for i in range(3)]
    fig.legend(handles=handles,loc='outside upper right',ncol=3,frameon=False)
    draw(fig,out,name,sha)


def choice_plot(index, out, sha):
    fig,ax=plt.subplots(figsize=(14,14),constrained_layout=True)
    rows=[(c,a,s,t) for c in CONTEXTS for a in ARMS for t in STRATA for s in range(3)]
    palette=['#168162','#c44b47','#c89132','#e1e7ec']
    names=['Improved switch','Harmful switch','Near-tied switch','Unchanged']
    for i,(context,arm,seed,stratum) in enumerate(rows):
        block=index[(arm,seed)]['contexts'][context]['summary'][stratum]
        score=block['scores']['sym_compat_guarded']; n=block['states']
        counts=[score['improved'],score['harmed'],score['changed_near_tie'],n-score['changed']]
        assert sum(counts)==n and min(counts)>=0
        left=0
        for j,count in enumerate(counts):
            ax.barh(i,count,left=left,color=palette[j],height=.68,label=names[j] if i==0 else None)
            if count: ax.text(left+count/2,i,str(count),ha='center',va='center',
                              color='white' if j<3 else '#41505d',weight='bold',fontsize=8.5)
            left+=count
        ax.text(n+.15,i,f'{n} states',va='center',fontsize=8,color='#526473')
    ax.set_yticks(range(len(rows)),[f"{c.replace('_',' ').title()} · {a.upper()} · {('Natural' if t=='natural' else 'Selected')} · {s}" for c,a,s,t in rows])
    ax.invert_yaxis();ax.set_xlim(0,19);ax.set_xlabel('Anchor families')
    ax.set_title('Symmetric BCA + guarded compatibility: changes versus raw Q1',loc='left',weight='bold',fontsize=15,pad=14)
    ax.grid(axis='x',color='#e4e9ef');fig.legend(ncol=4,loc='outside lower center',frameon=False)
    draw(fig,out,'03_choice_changes',sha)


def calibration_plot(index,out,sha):
    fig,axes=plt.subplots(1,3,figsize=(16,6.2),constrained_layout=True)
    fig.suptitle('Fresh TD checks, scale matching and state-distance gate',fontsize=16,weight='bold')
    cases=[index[(arm,seed)] for arm in ARMS for seed in range(3)]
    labels=[f'{x["arm"].upper()} {x["seed"]}' for x in cases]
    for i,c in enumerate(cases):
        check=c['check']; axes[0].plot([check['zero_mae'],check['signed_mae']],[i,i],color='#9aa9b5',lw=1.5)
        axes[0].scatter(check['zero_mae'],i,color='#aebac3',s=38);axes[0].scatter(check['signed_mae'],i,color='#165f86',s=38)
        m=c['match'];axes[1].scatter(m['lambda_mean'],i,color='#bd5830',marker='s',s=38)
        axes[1].scatter(m['lambda_gradient'],i,color='#72509b',marker='^',s=38)
    for ax in axes[:2]: ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.grid(axis='x',color='#e4e9ef')
    axes[0].set_title('Held-out signed one-step TD',loc='left',weight='bold');axes[0].set_xlabel('Mean absolute residual prediction error')
    axes[0].legend(handles=[plt.Line2D([0],[0],marker='o',color='none',markerfacecolor='#aebac3',markeredgecolor='#aebac3',label='Predict zero'),
        plt.Line2D([0],[0],marker='o',color='none',markerfacecolor='#165f86',markeredgecolor='#165f86',label='Signed model')],frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)
    axes[1].set_title('Training-probe signed scale controls',loc='left',weight='bold');axes[1].set_xlabel('Multiplier applied to signed bound')
    axes[1].legend(handles=[plt.Line2D([0],[0],marker='s',color='none',markerfacecolor='#bd5830',markeredgecolor='#bd5830',label='Match mean penalty'),
        plt.Line2D([0],[0],marker='^',color='none',markerfacecolor='#72509b',markeredgecolor='#72509b',label='Match gradient RMS')],frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=1)
    rows=[(c,t) for c in CONTEXTS for t in STRATA]
    for i,(context,stratum) in enumerate(rows):
        vals=[]
        for c in cases:
            p=c['contexts'][context]['per_probe'];mask=np.asarray(p['strata'])==stratum
            vals.append(int(np.asarray(p['components']['reliable'],bool)[mask,0].sum()))
        assert len(set(vals))==1
        axes[2].barh(i,vals[0],color='#468b9f',height=.55);axes[2].text(vals[0]+.2,i,f'{vals[0]} / 16',va='center',fontsize=9)
    axes[2].set_yticks(range(4),[f"{c.replace('_',' ').title()} · {('Natural' if t=='natural' else 'Selected')}" for c,t in rows]);axes[2].invert_yaxis()
    axes[2].set_xlim(0,19);axes[2].set_title('State-distance heuristic',loc='left',weight='bold');axes[2].set_xlabel('States within training-distance threshold')
    axes[2].grid(axis='x',color='#e4e9ef')
    draw(fig,out,'04_calibration_and_matching',sha)


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='plots_v1',help='new sibling output folder')
    options=parser.parse_args(); assert Path(options.out).name==options.out and options.out not in ('','.','..')
    source_raw=SOURCE.read_bytes();result=json.loads(source_raw);sha=hashlib.sha256(source_raw).hexdigest()
    index=extract(result);out=HERE/options.out;out.mkdir(exist_ok=False)
    style()
    scatter_plot(index,out,sha,metric='mean_regret',title='Expanded action bank: regret to the best of 15 candidates',axis_label='Mean regret, return units · lower is better',name='01_regret')
    scatter_plot(index,out,sha,metric='ranking_accuracy',title='Actor versus local perturbations: pairwise ranking',axis_label='Correct ranking on informative pairs (%)',name='02_pairwise_accuracy')
    choice_plot(index,out,sha);calibration_plot(index,out,sha)
    with (out/'all_metrics.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['context','arm','seed','stratum','method','states','high_states','informative_families','raw_mistake_families','mean_regret','ranking_accuracy','informative_pairs','changed','improved','harmed','near_tied','all_action_mae','high_band_mae'])
        for context in CONTEXTS:
            for arm in ARMS:
                for seed in range(3):
                    for stratum in STRATA:
                        block=index[(arm,seed)]['contexts'][context]['summary'][stratum]
                        for method,score in block['scores'].items():
                            writer.writerow([context,arm,seed,stratum,method,block['states'],block['high_load_states'],block['informative_families'],block['raw_mistake_families'],score['mean_regret'],score['ranking_accuracy'],score['ranking_informative_pairs'],score['changed'],score['improved'],score['harmed'],score['changed_near_tie'],score['all_action_error']['mae'],score['high_load_fixed_band_error']['mae']])
    lines=['# Expanded action-bank diagnostic: figure guide','',f'Source: `diagnostic_run_01/RESULT.json`, SHA-256 `{sha}`.','',
        'The actor, critic and target are frozen. Points are three checkpoint seeds. Action pairs and shared anchor states are dependent; results do not establish trained-policy improvement.','',
        '1. **Regret:** each selected action is compared with the best of the 15 candidate vectors, not a global optimum.',
        '2. **Pairwise accuracy:** actor versus coordinate perturbations with true Qπ gap >0.1; ties in scores are counted incorrect. Each point may have a different number of informative dependent action pairs.',
        '3. **Choice changes:** compares symmetric Q1 BCA plus guarded compatibility with raw Q1; improvements and harms require Qπ difference beyond ±0.1.',
        '4. **Fresh check and matching:** one-step TD prediction on 64 new behavior episodes; coefficients matched on training probes; state-distance is a heuristic, not a support certificate. Scaled signed scores have no original coverage claim.','',
        'Each context has 16 natural and 16 separately selected states. The ambiguity-selected stream is conditional on a Q1 rule and is not a natural-start estimate. Full per-method counts and metrics are in `all_metrics.csv`.','',
        'Prespecified exploratory screen:']
    for context in CONTEXTS:
        for stratum in STRATA:
            s=result['screen'][context][stratum]
            lines.append(f'- {context}, {stratum}: **{s["status"]}**; qualifying native {s["qualifying_seeds"]["native"]}/3, BCA {s["qualifying_seeds"]["bca"]}/3.')
    lines+=['','No actor, critic, actor-target, critic-target, or auxiliary optimizer updates were performed.','']
    (out/'FIGURE_GUIDE.md').write_text('\n'.join(lines),encoding='utf8')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'PLOT_RECEIPT.json').write_text(json.dumps({'result_sha256':sha,'files':files},indent=2,sort_keys=True)+'\n',encoding='utf8')
    print(out)


if __name__=='__main__':
    run()
