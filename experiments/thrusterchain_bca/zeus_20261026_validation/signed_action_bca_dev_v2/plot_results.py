"""Plot only completed frozen-diagnostic evidence, with source SHA provenance.

Run after RESULT.json exists. Nothing here changes the experiment outputs or
selects a method. Each marker is a checkpoint seed, not an independent state.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'diagnostic_run_01' / 'RESULT.json'
OUT = HERE / 'plots'
CONTEXTS = ('ordinary', 'strong_current')
ARMS = ('native', 'bca')
METHODS = ('raw_q1', 'fresh_twin_min', 'q1_aligned', 'original_compatibility',
           'revised_compatibility', 'signed_bayes', 'signed_split', 'combined',
           'strength_matched_q1', 'shuffled_compatibility')
LABELS = {'raw_q1':'Raw Q1', 'fresh_twin_min':'Twin-min width',
          'q1_aligned':'Q1 width', 'original_compatibility':'Original compatibility',
          'revised_compatibility':'Revised compatibility', 'signed_bayes':'Signed Bayesian',
          'signed_split':'Signed split', 'combined':'Signed + revised',
          'strength_matched_q1':'Matched Q1 width', 'shuffled_compatibility':'Shuffled neighbors'}
MAIN = ('raw_q1', 'q1_aligned', 'signed_bayes', 'revised_compatibility',
        'combined', 'strength_matched_q1')
COLORS = ('#165f86', '#bb5830', '#6d4f9b')
MARKERS = ('o', 's', '^')


def chosen(scores):
    x = np.asarray(scores, np.float64)
    return np.argmax(x.max(axis=1, keepdims=True) - x <= 1e-6, axis=1)


def summarise(case, context):
    b = case['contexts'][context]; p = b['per_probe']
    high = np.asarray(p['high_load'], bool)
    qpi = np.asarray(p['qpi'], np.float64)
    assert high.shape == (32,) and qpi.shape == (32, 6)
    assert np.isfinite(qpi).all()
    raw_choices = chosen(p['q1'])
    results = {}
    for name in METHODS:
        scores = np.asarray(p['scores'][name], np.float64)
        assert scores.shape == (32, 6) and np.isfinite(scores).all()
        ix = chosen(scores)
        regret = qpi.max(axis=1) - qpi[np.arange(32), ix]
        gain = qpi[np.arange(32), ix] - qpi[np.arange(32), raw_choices]
        changed = ix != raw_choices
        results[name] = {'high_band_mae': b['scores'][name]['error']['high_band_fixed']['mae'],
                         'high_load_regret': float(regret[high].mean()) if high.any() else None,
                         'changed': int((changed & high).sum()),
                         'improved': int((changed & high & (gain > .1)).sum()),
                         'harmed': int((changed & high & (gain < -.1)).sum()),
                         'near_tied': int((changed & high & (abs(gain) <= .1)).sum())}
    reliable = np.asarray(p['components']['reliable'], bool)
    assert reliable.shape == (192,)
    reliable = reliable.reshape(32, 6)
    assert np.all(reliable == reliable[:, :1])
    return {'high_count': int(high.sum()), 'reliable': int(reliable[:, 0].sum()), 'methods': results}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.titlesize':11, 'axes.labelsize':9,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.axisbelow':True, 'xtick.labelsize':8.5,
                         'ytick.labelsize':8.5, 'savefig.dpi':220,
                         'pdf.fonttype':42, 'figure.facecolor':'white'})


def save(fig, name, source_sha):
    fig.savefig(OUT/(name+'.png'), dpi=220, bbox_inches='tight',
                metadata={'Description': 'Frozen-checkpoint diagnostic source SHA256 '+source_sha})
    fig.savefig(OUT/(name+'.pdf'), bbox_inches='tight',
                metadata={'Subject': 'Frozen-checkpoint diagnostic source SHA256 '+source_sha})
    plt.close(fig)


def figure_outcomes(summaries, source_sha):
    fig, axes = plt.subplots(4, 2, figsize=(13.8, 14), sharey=True, constrained_layout=True)
    fig.suptitle('Frozen action scores: accuracy and decision regret', fontsize=17, fontweight='bold')
    rows = [(c,a) for c in CONTEXTS for a in ARMS]
    y = np.arange(len(MAIN))[::-1]
    for row, (context,arm) in enumerate(rows):
        cells = [summaries[(context,arm,s)] for s in range(3)]
        for col, (field, title) in enumerate((('high_band_mae','High-load, fixed-band MAE'),
                                               ('high_load_regret','High-load six-action regret'))):
            ax=axes[row,col]
            for seed, item in enumerate(cells):
                values=[item['methods'][k][field] for k in MAIN]
                for yy,val in zip(y,values):
                    if val is not None:
                        ax.scatter(val, yy+(seed-1)*.13, color=COLORS[seed], marker=MARKERS[seed],
                                   s=32, zorder=3, label=f'Seed {seed}' if yy==y[0] else None)
            ax.set_yticks(y, [LABELS[k] for k in MAIN]);ax.grid(axis='x',color='#e4e9ef')
            ax.set_title(f"{context.replace('_',' ').title()} · {arm.upper()} — {title}", loc='left', weight='bold')
            ax.set_xlabel('Absolute Q error, return units' if col==0 else 'Regret, return units  ·  lower is better')
            if col==0: ax.tick_params(labelleft=True)
    handles=[plt.Line2D([0],[0],marker=MARKERS[i],color='none',markerfacecolor=COLORS[i],
                        markeredgecolor=COLORS[i],label=f'Seed {i}') for i in range(3)]
    fig.legend(handles=handles,loc='outside upper right',ncol=3,frameon=False)
    save(fig,'01_accuracy_and_regret',source_sha)


def figure_choices(summaries, source_sha):
    fig,ax=plt.subplots(figsize=(12,8.4),constrained_layout=True)
    rows=[(c,a,s) for c in CONTEXTS for a in ARMS for s in range(3)]
    palette=['#168162','#c44b47','#c89132','#e1e7ec']
    names=['Improved switch','Harmful switch','Near-tied switch','Unchanged']
    for i,key in enumerate(rows):
        item=summaries[key];v=item['methods']['combined'];total=item['high_count']
        values=[v['improved'],v['harmed'],v['near_tied'],total-v['changed']]
        assert sum(values)==total and min(values)>=0
        left=0
        for k,value in enumerate(values):
            ax.barh(i,value,left=left,color=palette[k],height=.64,label=names[k] if i==0 else None)
            if value: ax.text(left+value/2,i,str(value),ha='center',va='center',
                              color='white' if k<3 else '#41505d',fontweight='bold',fontsize=9)
            left+=value
        ax.text(total+.4,i,f'{total} high-load states',va='center',fontsize=8,color='#526473')
    ax.set_yticks(range(len(rows)),[f"{c.replace('_',' ').title()}  ·  {a.upper()}  ·  seed {s}" for c,a,s in rows])
    ax.invert_yaxis();ax.set_xlim(0,39);ax.set_xlabel('High-load anchor families')
    ax.set_title('How the combined score changes action selection',loc='left',weight='bold',fontsize=15,pad=14)
    ax.grid(axis='x',color='#e4e9ef');fig.legend(ncol=4,loc='outside lower center',frameon=False)
    save(fig,'02_choice_changes',source_sha)


def figure_calibration(cases, summaries, source_sha):
    fig, axes=plt.subplots(1,3,figsize=(14.6,6.1),constrained_layout=True)
    fig.suptitle('What the fresh checks do—and do not—establish',fontsize=16,fontweight='bold')
    ordered=sorted(cases,key=lambda c:(c['arm']!='native',c['seed']))
    labels=[f"{x['arm'].upper()} {x['seed']}" for x in ordered]
    for i,c in enumerate(ordered):
        check=c['check']; axes[0].plot([check['zero_mae'],check['signed_mae']],[i,i],color='#9aa9b5',lw=1.5,zorder=1)
        axes[0].scatter(check['zero_mae'],i,marker='o',s=38,color='#aebac3',zorder=2)
        axes[0].scatter(check['signed_mae'],i,marker='o',s=38,color='#165f86',zorder=3)
        axes[1].scatter(100*check['upper_td_coverage'],i,color='#165f86',s=38)
    axes[0].set_title('Held-out signed TD prediction',loc='left',weight='bold')
    axes[0].set_xlabel('Mean absolute one-step residual error');axes[0].set_yticks(range(6),labels);axes[0].invert_yaxis()
    axes[0].legend(handles=[plt.Line2D([0],[0],marker='o',color='none',markerfacecolor='#aebac3',markeredgecolor='#aebac3',label='Predict zero'),
                            plt.Line2D([0],[0],marker='o',color='none',markerfacecolor='#165f86',markeredgecolor='#165f86',label='Signed model')],frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)
    axes[1].axvline(90,color='#9ba9b3',linestyle='--',lw=1)
    axes[1].set_xlim(0,102);axes[1].set_yticks(range(6),labels);axes[1].invert_yaxis()
    axes[1].set_title('Fresh behavior-row check',loc='left',weight='bold')
    axes[1].set_xlabel('Observed one-sided TD coverage (%)')
    axes[2].set_title('State-distance heuristic',loc='left',weight='bold')
    for i,(context,arm) in enumerate((c,a) for c in CONTEXTS for a in ARMS):
        nums=[summaries[(context,arm,s)]['reliable'] for s in range(3)]
        assert nums[0]==nums[1]==nums[2]
        axes[2].barh(i,nums[0],color='#468b9f',height=.55)
        axes[2].text(nums[0]+.35,i,f'{nums[0]} / 32',va='center',fontsize=9)
    axes[2].set_yticks(range(4),[f"{c.replace('_',' ').title()} · {a.upper()}" for c,a in ((c,a) for c in CONTEXTS for a in ARMS)])
    axes[2].invert_yaxis();axes[2].set_xlim(0,38);axes[2].set_xlabel('Anchor states within training-distance threshold')
    for ax in axes: ax.grid(axis='x',color='#e4e9ef')
    save(fig,'03_calibration_and_distance',source_sha)


def main():
    raw=SOURCE.read_bytes(); source_sha=hashlib.sha256(raw).hexdigest();data=json.loads(raw)
    assert data['status']=='completed_frozen_checkpoint_diagnostic'
    cases=data['cases'];assert len(cases)==6
    assert all(data['ledger'][k]==0 for k in ('actor_updates','critic_updates','actor_target_updates','critic_target_updates'))
    assert {(c['arm'],c['seed']) for c in cases}=={(a,s) for a in ARMS for s in range(3)}
    summaries={(context,c['arm'],c['seed']):summarise(c,context) for c in cases for context in CONTEXTS}
    OUT.mkdir(exist_ok=False)
    style();figure_outcomes(summaries,source_sha);figure_choices(summaries,source_sha);figure_calibration(cases,summaries,source_sha)
    fields=['context','arm','seed','method','high_load_count','high_band_mae','high_load_regret','changed','improved','harmed','near_tied','reliable_states']
    with (OUT/'paired_metrics.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for (context,arm,seed),item in sorted(summaries.items()):
            for method,v in item['methods'].items():
                writer.writerow(dict(context=context,arm=arm,seed=seed,method=method,high_load_count=item['high_count'],
                                     reliable_states=item['reliable'],**v))
    lines=['# Frozen-checkpoint diagnostic: figure guide', '',
           f'Source: `diagnostic_run_01/RESULT.json`, SHA-256 `{source_sha}`.', '',
           'The actor, critic, and target models were frozen. Each plotted marker is one of three checkpoint seeds; '
           'the contexts and candidate scores use paired anchors. These results do not measure policy improvement.', '',
           '1. **Accuracy and regret:** high-load fixed-band Q MAE is compared to same-policy simulator return; '
           'high-load regret compares the selected action with the best of only six candidates. '
           'These high-load-only regret values differ from the all-anchor regret in the raw results report.',
           '2. **Choice changes:** improvements and harms compare the combined score with raw Q1 on the same state, '
           'using a 0.1-unit tolerance. Unchanged choices are shown separately.',
           '3. **Calibration and distance:** one-step TD checks use 64 new behavior episodes; '
           'the state-distance gate is a heuristic, not a support certificate or shifted-policy coverage guarantee.', '',
           'The full ten-method per-cell numbers are in `paired_metrics.csv`. The prespecified exploratory screen:']
    for context in CONTEXTS:
        x=data['mechanism_screen']['contexts'][context]
        suffix = ' (insufficient prespecified action-ranking opportunity)' if x.get('status') == 'inconclusive' else ''
        lines.append(f"- {context.replace('_',' ').title()}: {x.get('status', 'passed' if x['passed'] else 'did not pass')}{suffix}; "
                     f"qualifying native seeds {x['qualifying_seeds']['native']}/3 and "
                     f"BCA seeds {x['qualifying_seeds']['bca']}/3.")
    lines.extend(['','No actor, critic, actor-target, or critic-target update was performed. '
                  'This is a development screen after prior exploratory analysis.',''])
    (OUT/'FIGURE_GUIDE.md').write_text('\n'.join(lines),encoding='utf8')
    report={'schema':'signed_diagnostic_plot_receipt_v1','result_sha256':source_sha,
            'status':data['status'],'mechanism_screen':data['mechanism_screen'],
            'ledger':data['ledger'],'plot_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()},
            'scope':'Paired frozen-action screen; three training seeds; no policy updates or policy-efficacy claim.'}
    (OUT/'PLOT_RECEIPT.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'plot_directory':str(OUT),'source_sha256':source_sha,'files':list(report['plot_files'])},indent=2))


if __name__=='__main__': main()
