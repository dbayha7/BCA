"""Render provenance-bound figures from completed v5 results only.

No simulator, optimizer, or model inference is imported or run. The generated
episode CSV is a convenience view; the hashed NPZ files remain the source.
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
RERANKER = HERE/'reranker_run_01'/'RERANKER_RESULT.json'
REFITS = HERE/'paired_refits_run_01'/'RESULT.json'
V4_RESULT = HERE.parent/'common_action_bca_dev_v4'/'diagnostic_run_01'/'RESULT.json'
METHODS = ('actor', 'raw_q1', 'symmetric', 'capped', 'guarded')
LABELS = ('Original actor', 'Raw Q1', 'Symmetric BCA', 'Capped width', 'Guarded hybrid')
CONTEXTS = ('ordinary', 'strong_current')
CONTEXT_LABELS = ('Ordinary current', 'Strong current')
ARMS = ('native', 'bca')
SEED_COLORS = ('#1877a6', '#bc5b24', '#75519c')
SEED_MARKERS = ('o', 's', '^')
COMPARISON_COLORS = ('#82726d', '#0072b2', '#d55e00', '#8063a0')

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10.5,
    'axes.titlesize':12, 'axes.labelsize':10.5, 'axes.spines.top':False,
    'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42,
    'savefig.facecolor':'white', 'figure.facecolor':'white'})


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def individual_refit_sd(widths, sources):
    """RMS refit SD of each checkpoint's 3U contrast to bank slot zero."""
    w=np.asarray(widths,np.float64);source=np.asarray(sources)
    require(w.shape==(4,2,32,15) and source.shape==(32,) and np.isfinite(w).all(),
            'bad refit arrays or state-source labels')
    mask=source!='random_prefix'
    require(int(mask.sum())==16, 'expected 16 pooled actor-visited states')
    selected=w[:,:,mask,:]
    centered=3.0*(selected-selected[:,:,:,:1])
    sd=np.sqrt(np.mean(np.var(centered,axis=0,ddof=1),axis=(1,2)))
    return {'rms_native_refit_sd':float(sd[0]),'rms_bca_refit_sd':float(sd[1])}


def load_result(reranker=RERANKER, refits=REFITS):
    """Authenticate completed ledgers and join to their exact NPZ telemetry."""
    rp, fp = Path(reranker), Path(refits)
    r, f = json.loads(rp.read_bytes()), json.loads(fp.read_bytes())
    require(r.get('status') == 'completed_frozen_controller_evaluation', 'completed reranker result required')
    require(f.get('status') == 'completed_paired_auxiliary_refits', 'completed paired refits required')
    require(tuple(r['methods']) == METHODS, 'controller order differs')
    rl, fl = r['ledger'], f['ledger']
    require(rl['transition_steps_attempted']==rl['transition_steps_completed']==192104
            and rl['continuation_rows_completed']==960 and rl['candidate_scoring_rows_completed']==1152000
            and all(rl[k]==0 for k in ('actor_updates','critic_updates','actor_target_updates',
                                        'critic_target_updates','auxiliary_updates')), 'incomplete or nonfrozen reranker ledger')
    require(fl['auxiliary_updates']==48000 and fl['calibration_refreshes']==24
            and fl['candidate_width_rows']==23040 and fl['simulator_transitions']==0
            and all(fl[k]==0 for k in ('actor_updates','critic_updates','actor_target_updates',
                                        'critic_target_updates')), 'incomplete or nonfrozen refit ledger')
    require(len(r['cases'])==6 and len(f['cases'])==3, 'wrong seed/case count')
    starts = rp.parent/'FRESH_STARTS.npz'
    require(digest(starts)==r['fresh_starts_sha256'], 'fresh-start artifact differs')
    data={}
    with np.load(starts,allow_pickle=False) as z:
        seed_ids={cx:np.asarray(z[cx+'_seeds'],np.int64) for cx in CONTEXTS}
    for cx in CONTEXTS:
        require(seed_ids[cx].shape==(16,) and len(np.unique(seed_ids[cx]))==16, 'bad reset ids')
    for seed in range(3):
        for arm in ARMS:
            case = next(c for c in r['cases'] if c['seed']==seed and c['arm']==arm)
            require(case['before']==case['after'], 'checkpoint digest changed')
            for cx in CONTEXTS:
                cell=case['contexts'][cx]; path=rp.parent/cell['telemetry_artifact']
                require(digest(path)==cell['telemetry_sha256'], 'controller telemetry digest differs')
                with np.load(path,allow_pickle=False) as z:
                    row={name:np.asarray(z[name]) for name in
                         ('returns','first_qpi','first_actions','clip_saturated','guards')}
                require(row['returns'].shape==(5,16) and row['first_qpi'].shape==(16,5)
                        and row['first_actions'].shape==(5,16,4)
                        and row['clip_saturated'].shape==row['guards'].shape==(4,16,100)
                        and np.isfinite(row['returns']).all() and np.isfinite(row['first_qpi']).all()
                        and np.isfinite(row['first_actions']).all(), 'bad telemetry array')
                require(np.array_equal(row['returns'],np.asarray(cell['returns']))
                        and np.array_equal(row['first_qpi'].T,np.asarray(cell['first_qpi'])), 'JSON/NPZ values differ')
                require(np.array_equal(row['clip_saturated'].sum((1,2)),
                                       np.asarray(cell['clip_saturated_count_by_controller']))
                        and np.array_equal(row['guards'].sum((1,2)),
                                           np.asarray(cell['guard_count_by_controller'])), 'telemetry totals differ')
                row['reset_seeds']=seed_ids[cx]; data[(cx,arm,seed)]=row
    require(digest(V4_RESULT)==f['v4_result_sha256'], 'v4 state-source result differs')
    prior=json.loads(V4_RESULT.read_bytes())
    for seed in range(3):
        fc=next(c for c in f['cases'] if c['seed']==seed)
        pc=next(c for c in prior['cases'] if c['seed']==seed)
        for cx in CONTEXTS:
            a=fc['contexts'][cx]['width_artifact']
            require(digest(fp.parent/a['path'])==a['sha256'], 'paired refit width artifact differs')
            with np.load(fp.parent/a['path'],allow_pickle=False) as z:
                extra=individual_refit_sd(z['widths'],pc['contexts'][cx]['per_probe']['sources'])
            fc['contexts'][cx]['summary']['actor_visited'].update(extra)
    return r, f, data, {'reranker_result_sha256':digest(rp), 'paired_refits_result_sha256':digest(fp),
                         'fresh_starts_sha256':digest(starts), 'v4_source_sha256':digest(V4_RESULT),
                         'plot_script_sha256':digest(__file__)}


def first_action_counts(row, baseline, tolerance=.1):
    """Count changes to capped action against baseline on the same starts."""
    require(baseline in (1,2), 'baseline must be raw or symmetric')
    a=np.asarray(row['first_actions'])
    delta=np.asarray(row['first_qpi'])[:,3]-np.asarray(row['first_qpi'])[:,baseline]
    changed=np.any(a[3] != a[baseline],axis=1)
    require(delta.shape==(16,) and changed.shape==(16,) and np.isfinite(delta).all(), 'bad first-action data')
    return {'unchanged':int((~changed).sum()), 'harmed':int((changed & (delta < -tolerance)).sum()),
            'near_tied':int((changed & (np.abs(delta) <= tolerance)).sum()),
            'improved':int((changed & (delta > tolerance)).sum()), 'mean_gain':float(delta.mean())}


def clean(ax):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#e1e7ea',linewidth=.8)
    ax.tick_params(length=0,pad=5)


def seeds_legend():
    return [Line2D([0],[0],marker=SEED_MARKERS[s],color='none',markerfacecolor=SEED_COLORS[s],
                   markeredgecolor=SEED_COLORS[s],label=f'Training seed {s}',markersize=7) for s in range(3)]


def save(fig, output, stem):
    fig.savefig(output/(stem+'.png'),dpi=210,bbox_inches='tight')
    fig.savefig(output/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def fig_paired_returns(data, output):
    comp=(1,2,4); labs=('vs raw Q1','vs symmetric','vs guarded hybrid')
    fig, axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
    fig.suptitle('Does capped width improve complete-episode return?',fontsize=18,fontweight='bold',y=.99)
    fig.legend([Line2D([0],[0],marker='o',color='none',markerfacecolor=COMPARISON_COLORS[i+1],
                       markeredgecolor=COMPARISON_COLORS[i+1],markersize=7) for i in range(3)],labs,
               loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    all_values=[]
    for cx in CONTEXTS:
        for arm in ARMS:
            for seed in range(3):
                v=data[(cx,arm,seed)]['returns'];all_values.extend(float(np.mean(v[3]-v[i])) for i in comp)
    lim=max(1.,max(map(abs,all_values),default=1.))*1.2
    for i,cx in enumerate(CONTEXTS):
        for j,arm in enumerate(ARMS):
            ax=axes[i,j];clean(ax);ax.axvline(0,color='#606d75',lw=1)
            for seed in range(3):
                v=data[(cx,arm,seed)]['returns']
                for k,control in enumerate(comp):
                    ax.scatter(np.mean(v[3]-v[control]),seed+(k-1)*.18,s=48,
                               color=COMPARISON_COLORS[k+1],zorder=3)
            ax.set_title(f'{CONTEXT_LABELS[i]} · {arm.upper()} checkpoint',loc='left',fontweight='bold')
            ax.set_yticks(range(3),[f'Seed {k}' for k in range(3)]);ax.invert_yaxis();ax.set_xlim(-lim,lim)
            ax.set_xlabel('Capped minus comparison · mean discounted return (16 paired starts)')
    fig.subplots_adjust(top=.86,bottom=.09,left=.10,right=.98,hspace=.40,wspace=.24)
    save(fig,output,'01_paired_return_gains')


def fig_actor_gain(data, output):
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharex=True,sharey=True)
    fig.suptitle('How does the capped controller compare with its saved actor?',
                 fontsize=17,fontweight='bold',y=.99)
    fig.legend(handles=seeds_legend(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    vals=[float(np.mean(data[(cx,arm,s)]['returns'][3]-data[(cx,arm,s)]['returns'][0]))
          for cx in CONTEXTS for arm in ARMS for s in range(3)]
    lim=max(1.,max(map(abs,vals),default=1.))*1.2
    for i,cx in enumerate(CONTEXTS):
        ax=axes[i];clean(ax);ax.axvline(0,color='#606d75',lw=1)
        for j,arm in enumerate(ARMS):
            for seed in range(3):
                row=data[(cx,arm,seed)]['returns']
                ax.scatter(np.mean(row[3]-row[0]),j+(seed-1)*.15,s=55,
                           marker=SEED_MARKERS[seed],color=SEED_COLORS[seed],zorder=3)
        ax.set_xlim(-lim,lim);ax.set_yticks(range(2),['Native checkpoint','BCA checkpoint']);ax.set_ylim(1.5,-.5)
        ax.set_title(CONTEXT_LABELS[i],loc='left',fontweight='bold')
        ax.set_xlabel('Capped minus actor · mean discounted return (16 paired starts)')
    fig.subplots_adjust(top=.78,bottom=.15,left=.16,right=.98,wspace=.16)
    save(fig,output,'06_capped_vs_actor')


def fig_absolute_returns(data, output):
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
    fig.suptitle('Complete-episode returns for all fixed controllers',fontsize=18,fontweight='bold',y=.99)
    fig.legend(handles=seeds_legend(),loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    for i,cx in enumerate(CONTEXTS):
        for j,arm in enumerate(ARMS):
            ax=axes[i,j];clean(ax)
            for m in range(5):
                for seed in range(3):
                    v=data[(cx,arm,seed)]['returns'][m]
                    ax.scatter(np.mean(v),m+(seed-1)*.16,s=45,marker=SEED_MARKERS[seed],color=SEED_COLORS[seed],zorder=3)
            ax.set_yticks(range(5),LABELS);ax.invert_yaxis()
            ax.set_title(f'{CONTEXT_LABELS[i]} · {arm.upper()} checkpoint',loc='left',fontweight='bold')
            ax.set_xlabel('Mean discounted episode return; higher is better')
    fig.subplots_adjust(top=.86,bottom=.09,left=.16,right=.98,hspace=.40,wspace=.38)
    save(fig,output,'02_complete_returns')


def fig_first_actions(data, output):
    fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=True,sharey=True)
    fig.suptitle('First action only: where capped width helps or hurts',fontsize=18,fontweight='bold',y=.99)
    keys=('harmed','near_tied','improved','unchanged')
    colors=('#c54d4d','#e8b966','#218b73','#c9d3d8')
    desc=('Harmed > 0.1','Changed, within ±0.1','Improved > 0.1','Same action')
    fig.legend([Line2D([0],[0],color=c,lw=8) for c in colors],desc,
               loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False)
    for i,cx in enumerate(CONTEXTS):
        for j,base in enumerate((1,2)):
            ax=axes[i,j];clean(ax)
            for n,(arm,seed) in enumerate((a,s) for a in ARMS for s in range(3)):
                row=first_action_counts(data[(cx,arm,seed)],base);left=0
                for k,color in zip(keys,colors):
                    value=row[k];ax.barh(n,value,left=left,color=color,height=.62)
                    if value>=2:ax.text(left+value/2,n,str(value),ha='center',va='center',fontsize=9,color='#182c36')
                    left+=value
                require(left==16,'first-action counts do not sum to 16')
            ax.set_xlim(0,16);ax.set_xticks([0,4,8,12,16]);ax.set_yticks(range(6),
                [f'{a.upper()} · seed {s}' for a in ARMS for s in range(3)]);ax.set_ylim(5.5,-.5)
            ax.set_title(f'{CONTEXT_LABELS[i]} · capped vs {LABELS[base]}',loc='left',fontweight='bold')
            ax.set_xlabel('Paired starts (16 per seed and checkpoint)')
    fig.subplots_adjust(top=.86,bottom=.09,left=.15,right=.98,hspace=.38,wspace=.18)
    save(fig,output,'03_first_action_changes')


def fig_telemetry(data, output):
    fig,axes=plt.subplots(2,2,figsize=(15,8),sharex=True,sharey=True)
    fig.suptitle('How often were the cap and guard active?',fontsize=18,fontweight='bold',y=.99)
    fig.legend(handles=seeds_legend(),loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    for i,cx in enumerate(CONTEXTS):
        for j,kind in enumerate(('cap','guard')):
            ax=axes[i,j];clean(ax)
            for y,arm in enumerate(ARMS):
                for seed in range(3):
                    row=data[(cx,arm,seed)]
                    # Index 2 is the capped controller; index 3 is guarded.
                    fraction=np.mean(row['clip_saturated'][2] if kind=='cap' else row['guards'][3])*100
                    ax.scatter(fraction,y+(seed-1)*.15,s=50,marker=SEED_MARKERS[seed],color=SEED_COLORS[seed],zorder=3)
            ax.set_xlim(-2,102);ax.set_xticks([0,20,40,60,80,100]);ax.set_yticks(range(2),['Native checkpoint','BCA checkpoint']);ax.set_ylim(1.5,-.5)
            ax.set_title(f'{CONTEXT_LABELS[i]} · '+('capped controller' if kind=='cap' else 'guarded controller'),
                         loc='left',fontweight='bold')
            ax.set_xlabel('Selected-action cap saturation (%)' if kind=='cap' else 'Guard active at decision (%)')
    fig.subplots_adjust(top=.84,bottom=.09,left=.16,right=.98,hspace=.40,wspace=.20)
    save(fig,output,'04_cap_and_guard_activity')


def fig_refits(refits, output):
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharex=True,sharey=True)
    fig.suptitle('Paired auxiliary refits: contrast size versus refit variability',fontsize=17,fontweight='bold',y=.99)
    colors=('#0072b2','#d55e00','#218b73','#9c5795')
    fig.legend([Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markeredgecolor=c) for c in colors],
               ['RMS paired mean contrast','RMS SD of paired difference',
                'RMS native within-checkpoint SD','RMS BCA within-checkpoint SD'],loc='upper center',
               bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
    vals=[]
    for cx in CONTEXTS:
        for case in refits['cases']:
            s=case['contexts'][cx]['summary']['actor_visited']
            vals += [s[k] for k in ('rms_paired_mean_contrast','rms_across_refit_sd',
                                    'rms_native_refit_sd','rms_bca_refit_sd')]
    lim=max(.05,max(vals,default=.05))*1.2
    for i,cx in enumerate(CONTEXTS):
        ax=axes[i];clean(ax)
        for seed in range(3):
            case=next(c for c in refits['cases'] if c['seed']==seed)
            s=case['contexts'][cx]['summary']['actor_visited']
            for j,k in enumerate(('rms_paired_mean_contrast','rms_across_refit_sd',
                                  'rms_native_refit_sd','rms_bca_refit_sd')):
                ax.scatter(s[k],seed+(j-1.5)*.15,s=55,color=colors[j],zorder=3)
        ax.set_xlim(0,lim);ax.set_yticks(range(3),[f'Seed {k}' for k in range(3)]);ax.set_ylim(2.5,-.5)
        ax.set_title(CONTEXT_LABELS[i]+' · actor-visited development states',loc='left',fontweight='bold')
        ax.set_xlabel('RMS action-contrast quantity (Q units)')
    fig.subplots_adjust(top=.73,bottom=.15,left=.10,right=.98,wspace=.22)
    save(fig,output,'05_paired_refit_variability')


def write_csv(data, refits, output):
    names=('context','checkpoint','training_seed','reset_seed','controller','discounted_return',
           'first_action_qpi','action_0','action_1','action_2','action_3')
    with (output/'episode_values.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for cx in CONTEXTS:
            for arm in ARMS:
                for seed in range(3):
                    row=data[(cx,arm,seed)]
                    for m,method in enumerate(METHODS):
                        for i in range(16):
                            a=row['first_actions'][m,i]
                            w.writerow(dict(context=cx,checkpoint=arm,training_seed=seed,
                                reset_seed=int(row['reset_seeds'][i]),controller=method,
                                discounted_return=float(row['returns'][m,i]),first_action_qpi=float(row['first_qpi'][i,m]),
                                action_0=float(a[0]),action_1=float(a[1]),action_2=float(a[2]),action_3=float(a[3])))
    names=('context','checkpoint','training_seed','comparison','unchanged','harmed','near_tied','improved','mean_gain')
    with (output/'first_action_comparisons.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for cx in CONTEXTS:
            for arm in ARMS:
                for seed in range(3):
                    for b in (1,2):
                        w.writerow(dict(context=cx,checkpoint=arm,training_seed=seed,comparison=METHODS[b],
                                        **first_action_counts(data[(cx,arm,seed)],b)))
    names=('context','training_seed','rms_paired_mean_contrast','rms_across_refit_sd',
           'rms_native_refit_sd','rms_bca_refit_sd')
    with (output/'refit_contrast_variability.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for cx in CONTEXTS:
            for seed in range(3):
                case=next(c for c in refits['cases'] if c['seed']==seed)
                s=case['contexts'][cx]['summary']['actor_visited']
                w.writerow(dict(context=cx,training_seed=seed,**{k:s[k] for k in names[2:]}))


def render(reranker=RERANKER, refits=REFITS, output=None):
    r,f,data,pins=load_result(reranker,refits)
    output=HERE/'plots_v2' if output is None else Path(output)
    output.mkdir(parents=False,exist_ok=False)
    fig_paired_returns(data,output);fig_absolute_returns(data,output);fig_first_actions(data,output)
    fig_telemetry(data,output);fig_refits(f,output);fig_actor_gain(data,output);write_csv(data,f,output)
    screen=r['screen']
    guide=['# Frozen-controller evaluation: figure guide','',
        f"Controller result SHA-256: `{pins['reranker_result_sha256']}`.",
        f"Paired-refit result SHA-256: `{pins['paired_refits_result_sha256']}`.",
        f"Fresh-start artifact SHA-256: `{pins['fresh_starts_sha256']}`.",'',
        f"The prespecified engineering screen **{screen['status']}**. "
        f"Qualifying strong-current seeds: native {screen['qualifying_strong_seeds']['native']}/3, "
        f"BCA {screen['qualifying_strong_seeds']['bca']}/3; ordinary-current nonharm for all seeds: "
        f"{screen['ordinary_all_seeds_nonharm']}.",'',
        '1. **Paired return gains:** per checkpoint and training seed, the mean over 16 matched fresh starts of capped-controller discounted full-episode return minus each reranking comparator. Positive is better. Each controller acts repeatedly on its own evolving trajectory.',
        '2. **Complete returns:** each point is a checkpoint-seed mean across the same 16 starts. Higher is better, including when all returns are negative.',
        '3. **First-action changes:** for the same initial state, compare capped to raw or symmetric. Each selected first action is followed by that checkpoint’s original actor. A changed action is improved/harmed only when the Qπ difference exceeds +0.1/−0.1; same actions are separate. This is not a full-controller return.',
        '4. **Cap and guard activity:** fraction of 1,600 selected decisions per seed on the capped and guarded controllers, respectively. The two controllers follow different state distributions, so the two percentages do not describe the same states.',
        '5. **Paired refit variability:** on actor-visited v4 development states, show RMS of the four-refit mean native/BCA contrast, RMS of the across-refit SD of that paired difference, and RMS across-refit SDs of native and BCA contrasts separately. Every 3U contrast is anchored at the native actor action; rows include duplicates and are dependent. These are not confidence intervals or a causal estimate.',
        '6. **Capped versus actor:** the same mean paired return gain relative to the original actor, shown on its own scale so the smaller differences among rerankers remain readable in Figure 1.','',
        'All five controllers and six checkpoint actors and critics are frozen. The score cap is computed from in-sample retained training probes and is not a calibrated or safety bound. Qπ uses the saved actor continuation, not the repeated reranker. The 16 starts are shared across checkpoints and methods; only three paired training seeds and one dataset are available. No uncertainty bars or statistical-significance claims are implied.',
        'Each underlying row, matched reset seed, first action, episode return and first-action Qπ is in `episode_values.csv`; `first_action_comparisons.csv` contains the count and mean-gain summary. `refit_contrast_variability.csv` contains all four plotted refit quantities. Full step telemetry and model artifacts are retained in the source run directories.', '']
    (output/'FIGURE_GUIDE.md').write_text('\n'.join(guide),encoding='utf8')
    files={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    (output/'PLOT_RECEIPT.json').write_text(json.dumps({'inputs':pins,'files':files},sort_keys=True,indent=2)+'\n',encoding='utf8')
    return output


if __name__=='__main__':
    require(len(sys.argv)==1,'Usage: python plot_results.py')
    print(render())
