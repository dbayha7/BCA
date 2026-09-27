"""Generate and verify a synthetic OOD reader. Never load research results.

Every plotted numeric value is read back from its exported CSV. PNG/SVG/PDF,
input fixtures, hashes and plot-data receipts share one immutable output folder.
"""

import argparse
import csv
import hashlib
import html
import json
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.ood import analyze as A
from experiments.ood.protocol import digest, file_hash, load_config

FIXTURE_SEED = 1900927201
STRATA = ['host/reset', 'host/100', 'bca/reset', 'bca/100']


def json_value(value):
    if isinstance(value, dict):
        return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        A.require(not np.isnan(value), 'Do not silently turn NaN into missing results.')
        return 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(json_value(value), f, indent=2, allow_nan=False)
        f.write('\n')


def fixture_records(case='complete'):
    """Construct exact state panels; both collectors share each reset-time state."""
    rows=[]
    for seed in range(5):
        for block in range(12 if case != 'sparse' else 1):
            for si,stratum in enumerate(STRATA):
                if case == 'sparse' and si != 0:
                    continue
                # Reset duplicates have identical inputs and outcomes.
                state = 'reset' if stratum.endswith('reset') else stratum
                offset = seed*.2 + block*.1 + (0 if state == 'reset' else si)
                returns = [10+offset, 8+offset, 11+offset, 6+offset, 10+offset]
                widths = [1, 3, 1, 4, 1]
                if case == 'tied':
                    widths = [2]*5
                if case == 'no_harm':
                    returns = [10+offset, 11+offset, 12+offset, 13+offset, 10+offset]
                p=dict(actions=[[0],[.1],[.2],[.3],[0]], returns=returns, width=widths,
                       support=[0,.1,.8,.2,0], random=[0,.7,.3,.2,0])
                row=A.state_metrics(p)
                rows.append(dict(seed=seed,block=seed*100+block,stratum=stratum,
                                 state_id=state,continuation='bca',panel=p,**row))
    return rows


def series(rows, panel, name, x, y, *, kind='line', lower=None, upper=None, note=''):
    for i,(xx,yy) in enumerate(zip(x,y)):
        rows.append(dict(panel=panel,series=name,x=float(xx),y=None if yy is None else float(yy),
                         lower=None if lower is None else float(lower[i]),
                         upper=None if upper is None else float(upper[i]),kind=kind,note=note))


def make_fixtures():
    rng=np.random.default_rng(FIXTURE_SEED)
    steps=np.arange(5000,1000001,5000)
    host=np.array([-6+18*(1-np.exp(-steps/270000))+i*.7+np.sin(steps/90000+i) for i in range(5)])
    bca=host+np.array([1.2+.5*np.sin(steps/150000+i)-i*.25 for i in range(5)])
    hf=np.array([rng.normal(13+i*.7,3,20) for i in range(5)])
    bf=hf+np.array([np.linspace(-1,2,20)+i*.15 for i in range(5)])
    scale_scores=np.arange(1,201,dtype=float)/100
    bb=rng.exponential(size=(128,200))
    test_errors=np.arange(1,501,dtype=float)/250
    return dict(steps=steps,host=host,bca=bca,host_final=hf,bca_final=bf,
                calibration_scores=scale_scores,bootstrap_weights=bb,test_errors=test_errors,
                cases={name:fixture_records(name) for name in ['complete','tied','no_harm','sparse']})


def plot_specs(data):
    """Expected arrays, independent of the renderer/CSV reader."""
    steps=np.asarray(data['steps'])
    host,bca=np.asarray(data['host']),np.asarray(data['bca'])
    policy=A.policy_summary(steps,host,bca,data['host_final'],data['bca_final'])
    specs=[]
    def add(slug,title,question,panels,rows,counts):
        specs.append(dict(slug=slug,title=title,question=question,panels=panels,rows=rows,counts=counts))
    def panel(title,xlabel,ylabel,**kwargs):
        return dict(title=title,xlabel=xlabel,ylabel=ylabel,**kwargs)

    rows=[]
    draws=np.random.default_rng(FIXTURE_SEED+1).integers(0,5,(10000,5))
    for name,curves in [('Host',host),('Host + BCA',bca)]:
        for s,curve in enumerate(curves):
            series(rows,0,f'{name} seed {s}',steps,curve,kind='faint')
        low,high=np.quantile(curves[draws].mean(1),[.025,.975],axis=0)
        series(rows,0,name+' mean',steps,curves.mean(0),lower=low,upper=high)
    add('01-learning','Learning curves','Policy performance',
        [panel('All seeds; approximate seed-bootstrap 95% bands','Host/critic updates','Illustrative normalized return')],rows,
        '5 paired seeds • 200 periodic banks • actor updates = host updates / 2')

    rows=[]
    for j,(name,bank) in enumerate([('Host',data['host_final']),('Host + BCA',data['bca_final'])]):
        for s,values in enumerate(bank):
            series(rows,0,f'{name} seed {s}',np.arange(20)/60+j+ s*.025,values,kind='scatter')
    series(rows,1,'Curve average',range(5),policy['curve_difference'],kind='scatter')
    series(rows,1,'Reserved final mean',range(5),policy['final_difference'],kind='scatter')
    add('02-final','Final performance is separate','Policy performance',
        [panel('Reserved bank: episode spread, not seed confidence','Method / episode jitter','Illustrative normalized return',ticks={.2:'Host',1.2:'Host + BCA'}),
         panel('Paired difference: BCA minus host','Training seed','Illustrative normalized return')],rows,
        '5 paired seeds • 20 final episodes per method/seed • 200 common periodic banks')

    rows=[]
    series(rows,0,'Qmin on recorded actions',steps,2+np.log1p(steps/5000))
    series(rows,1,'Critic MSE',steps,3*np.exp(-steps/180000)+.2)
    series(rows,2,'Actor BC MSE (unweighted)',steps,.7*np.exp(-steps/300000)+.1)
    add('03-loss','Training quantities have different meanings','Training diagnostics',
        [panel('Prediction, not observed return','Host updates','Synthetic Q units'),
         panel('Prediction loss, not OOD harm','Host updates','Synthetic squared target units'),
         panel('Actor-update rows only; skipped rows excluded','Host updates','Synthetic action-square units')],rows,
        '1 constructed journal • 200 displayed entries • no optimizer was run')

    rows=[]
    series(rows,0,'Accepted',steps,.8*steps)
    series(rows,0,'Abstained',steps,.2*steps)
    series(rows,1,'ESS fraction',steps,.32+.06*np.sin(steps/100000))
    series(rows,1,'Acceptance threshold',steps,np.full(200,.25),kind='reference')
    series(rows,2,'Consumed multiplier',steps,1.27+.04*np.sin(steps/150000))
    series(rows,3,'Accepted fit objective',steps,.02+.01*np.exp(-steps/200000))
    add('04-fitting','Fitting activity and applied strength','Training diagnostics',
        [panel('Constructed counters','Host updates','Number of scale fits'),
         panel('ESS and its declared threshold','Host updates','Fraction'),
         panel('Strength actually consumed by the host','Host updates','Dimensionless multiplier'),
         panel('Objective on accepted fits; not proof of useful ranking','Host updates','Dimensionless objective')],rows,
        '1 constructed journal • accepted + abstained = host updates • no actual fits')

    rows=[]
    bayes=1.2+.08*np.sin(steps/140000); floor=1.18+.07*np.cos(steps/170000)
    for name,y in [('Bayesian',bayes),('Conformal',floor),('Applied maximum',np.maximum(bayes,floor))]:
        series(rows,0,name,steps,y,kind='reference' if name=='Applied maximum' else 'line')
    series(rows,1,'Frozen residual unit',steps,2+.2*np.sin(steps/130000))
    add('05-radius','Both radius components remain visible','Training diagnostics',
        [panel('R = max(Bayesian, conformal)','Host updates','Dimensionless radius'),
         panel('Width = radius × frozen unit × scale','Host updates','Synthetic residual units')],rows,
        '1 constructed journal • 200 example snapshots • no historical values inferred')

    rows=[]
    for seed in range(5):
        severity=np.array([0,.05,.15,.30])
        series(rows,0,f'Seed {seed}',severity,1+severity*(seed+1))
        series(rows,1,f'Seed {seed}',severity,.02+severity*(.6+seed*.1))
    add('06-shift','Unfamiliarity and width','Support sensitivity',
        [panel('Width response to a constructed action shift','Pre-clipping perturbation RMS','Synthetic width'),
         panel('Distance is a proxy, not true OOD status','Pre-clipping perturbation RMS','Synthetic support distance')],rows,
        '5 synthetic seeds • 4 severities • no dataset-neighbor query')

    scenarios=[('Correct',[0,1],[0,1]),('Reversed',[0,1],[1,0]),('Tied',[0,1],[2,2]),
               ('No harm',[0,0],[0,1]),('No safe alternative',[1,1],[0,1])]
    rows=[]
    for i,(name,y,s) in enumerate(scenarios):
        series(rows,0,'Width AUROC',[i],[A.auc(y,s)],kind='scatter',note=name)
    series(rows,0,'Tie reference',[-.25,4.25],[.5,.5],kind='reference')
    series(rows,1,'Within each state',[0,1],[1.,1.],kind='scatter')
    series(rows,1,'Pooled across the two states',[.5],[A.auc([0,1,0,1],[0,1,10,11])],kind='scatter')
    add('07-ranking','Does the score identify harmful actions?','Harm ranking',
        [panel('Single-class states are N/A','Known-answer case','AUROC',ticks={i:n for i,(n,_,_) in enumerate(scenarios)},ylim=[-.08,1.12]),
         panel('Same ordering within states; shifted score levels','State / pooled comparison','AUROC',ylim=[0,1.12])],rows,
        '5 known-answer cases • 2 alternatives each • separate pooled example has 2 states')

    rows=[]
    for name,scores in [('Correct score',[0,2,4]),('Reversed score',[4,2,0]),('Tied score',[1,1,1])]:
        curve=A.risk_retention([0,2,4],scores)
        series(rows,0,name,[r['retained_fraction'] for r in curve],[r['mean_harm'] for r in curve])
    series(rows,1,'Reward contribution',[0,1,2,3],[6,2,-.1,7.9],kind='scatter')
    series(rows,2,'Terminated',[0,1],[8,6],kind='scatter')
    series(rows,2,'Time limit',[0,1],[2,4],kind='scatter')
    add('08-risk','Selection, reward components and endings','Harm ranking',
        [panel('Remove highest scores; keep tie groups intact','Retained fraction','Mean raw reward loss'),
         panel('Constructed difference: 6 + 2 − 0.1 = 7.9','Component','Raw reward difference',ticks={0:'Forward',1:'Alive',2:'Cost',3:'Total'}),
         panel('Termination and truncation stay separate','Continuation','Episodes',ticks={0:'Host',1:'BCA'})],rows,
        '3 alternatives for retention • 1 reward arithmetic example • 10 constructed episodes per continuation')

    radius=A.radius_reference(data['calibration_scores'],data['bootstrap_weights'])
    rules={'Training example':1.6,'Fresh conformal':radius['conformal'],'Fresh max':radius['radius'],
           'IID tolerance reference':A.tolerance_radius(data['calibration_scores']),
           'Too few calibration scores':A.conformal_radius(np.arange(1,9))}
    coverage={name:A.coverage(data['test_errors'],np.full(500,r)) for name,r in rules.items()}
    rows=[]
    for i,(name,r) in enumerate(rules.items()):
        result=coverage[name]
        lo,hi=result['miscoverage_interval']
        series(rows,0,'Observed coverage',[i],[result['coverage']],kind='scatter',lower=[1-hi],upper=[1-lo],note=name)
        series(rows,1,'Residual width',[i],[r],kind='scatter',note='Infinite width is vacuous' if np.isinf(r) else name)
    series(rows,0,'Nominal 90%',[-.2,4.2],[.9,.9],kind='reference')
    ticks={i:name for i,name in enumerate(rules)}
    add('09-coverage','Coverage and band width must be read together','Residual coverage',
        [panel('Exact binomial episode intervals; conditional on fixed band','Radius rule','Fraction covered',ticks=ticks,ylim=[0,1.12]),
         panel('Infinity is shown, never clipped into a finite result','Radius rule','Synthetic residual width',ticks=ticks)],rows,
        '200 calibration scores (8 for the last rule) • 500 disjoint constructed test scores • 128 saved Bayesian draws')

    rows=[]
    returns=np.array([-3.,-12.1,-.1]); widths=np.array([1.,3.,2.])
    series(rows,0,'Measured return',[0,1,2],returns,kind='scatter')
    series(rows,0,'Harm = reference − candidate',[0,1,2],A.harm(-3,returns),kind='scatter')
    series(rows,1,'Width',[0,1,2],widths,kind='scatter')
    series(rows,2,'Prediction Q',[0],[6.],kind='scatter')
    series(rows,2,'Learning target: 1 + 0.9 × 8',[1],[8.2],kind='scatter')
    series(rows,2,'Absolute residual: |8.2 − 6|',[2],[2.2],kind='scatter')
    series(rows,2,'Band: radius 1.5 × unit 1 × scale 2',[3],[3.],kind='scatter')
    add('10-arithmetic','A small arithmetic walkthrough','Worked example',
        [panel('Three-step exact toy return','First action','Reward / loss',ticks={0:'Reference 0',1:'Familiar −1',2:'Unfamiliar +1'}),
         panel('Larger width should warn of greater harm','First action','Illustrative BCA width',ticks={0:'Reference 0',1:'Familiar −1',2:'Unfamiliar +1'}),
         panel('2.2 ≤ 3: this error is covered','Quantity','Illustrative value',ticks={0:'Q',1:'Target',2:'Error',3:'Width'})],rows,
        '1 toy state • 3 first actions • separate one-critic residual illustration; target is an estimate, not true Q')
    return specs, dict(policy=policy,radii=radius,coverage=coverage)


FIELDS=['panel','series','x','y','lower','upper','kind','note']


def read_csv(path):
    with path.open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        r['panel']=int(r['panel'])
        for k in ['x','y','lower','upper']:
            r[k]=None if r[k]=='' else float(r[k])
    return rows


def render(spec,out,manifest_hash,pdf_sink=None):
    stem=spec['slug']
    with (out/(stem+'.csv')).open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,FIELDS,lineterminator='\n'); writer.writeheader(); writer.writerows(spec['rows'])
    rows=read_csv(out/(stem+'.csv'))
    A.require(rows==spec['rows'],'CSV round trip changed plotted data.')
    n=len(spec['panels'])
    cols=min(n,3) if n<=3 else 2
    nr=(n+cols-1)//cols
    fig,axes=plt.subplots(nr,cols,figsize=(max(8,5.8*cols),5.8 if nr==1 else 10.2),squeeze=False)
    axes=axes.ravel()
    trace=[]
    for pi,meta in enumerate(spec['panels']):
        ax=axes[pi]
        for name in dict.fromkeys(r['series'] for r in rows if r['panel']==pi):
            group=[r for r in rows if r['panel']==pi and r['series']==name]
            finite=[r for r in group if r['y'] is not None and np.isfinite(r['y'])]
            x=np.array([r['x'] for r in finite]); y=np.array([r['y'] for r in finite])
            kind=group[0]['kind']
            color=('#ce6320' if name.startswith('Host + BCA') else '#236fa1') if name.startswith('Host') else None
            if kind=='scatter':
                artist=ax.scatter(x,y,label=name,s=30,zorder=3,color=color)
                np.testing.assert_array_equal(np.asarray(artist.get_offsets()),np.column_stack([x,y]))
            else:
                artist,=ax.plot(x,y,label=name,alpha=.28 if kind=='faint' else 1,
                    lw=.8 if kind=='faint' else 2,ls='--' if kind=='reference' else '-',color=color,
                    marker='o' if len(x)<=5 and kind=='line' else None)
                np.testing.assert_array_equal(artist.get_xdata(),x)
                np.testing.assert_array_equal(artist.get_ydata(),y)
            if finite and all(r['lower'] is not None for r in finite):
                lo=np.array([r['lower'] for r in finite]); hi=np.array([r['upper'] for r in finite])
                A.require(np.isfinite([lo,hi]).all() and np.all(lo<=hi),'Invalid uncertainty envelope.')
                if kind=='scatter':
                    bars=ax.vlines(x,lo,hi,color='#78839b',lw=1.5)
                    np.testing.assert_array_equal(bars.get_segments(),np.array([[[xx,l],[xx,u]] for xx,l,u in zip(x,lo,hi)]))
                else:
                    band=ax.fill_between(x,lo,hi,alpha=.15,color=color)
                    vertices={tuple(p) for path in band.get_paths() for p in path.vertices}
                    A.require(vertices==set(zip(x,lo))|set(zip(x,hi)),'Uncertainty band differs from CSV bounds.')
            for r in group:
                if r['y'] is None or np.isinf(r['y']):
                    ax.text(r['x'],.93,'N/A' if r['y'] is None else '∞',
                            transform=ax.get_xaxis_transform(),ha='center',fontsize=13,fontweight='bold')
                    ax.update_datalim([[r['x'],0]])
            trace.extend(group)
        ax.set(title=textwrap.fill(meta['title'],48),xlabel=meta['xlabel'],ylabel=meta['ylabel'])
        ax.grid(alpha=.18)
        if 'ticks' in meta:
            ax.set_xticks(list(meta['ticks']),list(meta['ticks'].values()),rotation=18,ha='right')
        if 'ylim' in meta:
            ax.set_ylim(meta['ylim'])
        handles,labels=ax.get_legend_handles_labels()
        chosen=[(h,l) for h,l in zip(handles,labels) if ' seed ' not in l]
        if chosen:
            ax.legend(*zip(*chosen),fontsize=8,loc='best')
        ax.tick_params(labelsize=9)
    fig.suptitle('SYNTHETIC TEST • '+spec['title'],fontsize=16,fontweight='bold',y=.99)
    fig.text(.02,.025,spec['counts']+'\nFixture manifest '+manifest_hash[:16]+' • NO RESEARCH RESULTS',fontsize=8)
    fig.tight_layout(rect=[0,.10,1,.94],pad=2)
    for ext in ['png','svg','pdf']:
        fig.savefig(out/(stem+'.'+ext),dpi=150,facecolor='white')
    if pdf_sink is not None:
        pdf_sink.savefig(fig,facecolor='white')
    plt.close(fig)
    # Row order can change when grouped into artists; compare canonical multisets.
    key=lambda r:(r['panel'],r['series'],r['x'],str(r['y']))
    A.require(sorted(trace,key=key)==sorted(rows,key=key),'Some CSV rows were not represented in the figure.')
    write_json(out/(stem+'-plot-data.json'),trace)


def build(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    sources={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    manifest=dict(kind='synthetic_calculation_and_figure_tests',fixture_seed=FIXTURE_SEED,
        bootstrap_draws=10000,scientific_results=False,training_updates=0,model_queries=0,
        simulator_steps=0,config_semantic_sha256=digest(load_config()),source_sha256=sources)
    manifest['sha256']=digest(manifest)
    write_json(out/'manifest.json',manifest)
    data=make_fixtures(); write_json(out/'inputs.json',data)
    specs,calculations=plot_specs(data)
    summaries={name:{s:A.ranking_summary([r for r in rows if r['seed']==s],STRATA)
                     for s in range(5)} for name,rows in data['cases'].items()}
    bootstrap=A.paired_ranking_bootstrap(data['cases']['complete'],expected_strata=STRATA,
        draws=10000,seed=FIXTURE_SEED+2,expected_seeds=list(range(5)))
    calculations.update(scenarios=summaries,paired_bootstrap=bootstrap)
    write_json(out/'calculations.json',calculations)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out/'all-figures.pdf') as pages:
        for spec in specs:
            render(spec,out,manifest['sha256'],pages)
    write_html(out,specs,manifest,summaries)
    verify_report(out, pending_receipt=True)
    receipt=dict(kind='synthetic_only',verified_figures=len(specs),csv_artist_equality=True,
                 browser_layout_verified=False,
                 training_updates=0,model_queries=0,simulator_steps=0,
                 assets={p.name:file_hash(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'verification.json',receipt)
    verify_report(out)
    return receipt


def verify_report(out, *, pending_receipt=False):
    """Recompute expected numeric tables; reject missing/tampered assets and traces."""
    import re
    out=Path(out)
    data=json.loads((out/'inputs.json').read_text())
    specs,_=plot_specs(data)
    for spec in specs:
        rows=read_csv(out/(spec['slug']+'.csv'))
        A.require(rows==spec['rows'],'Figure CSV differs from recomputed fixture: '+spec['slug'])
        trace=json.loads((out/(spec['slug']+'-plot-data.json')).read_text())
        key=lambda r:(r['panel'],r['series'],r['x'],str(r['y']))
        A.require(sorted(trace,key=key)==sorted(json_value(rows),key=key),'Plot-data trace mismatch.')
        for ext in ['png','svg','pdf']:
            p=out/(spec['slug']+'.'+ext)
            A.require(p.is_file() and p.stat().st_size>100, 'Missing/empty figure export.')
    doc=(out/'index.html').read_text(encoding='utf8')
    for target in re.findall(r'(?:href|src)="([^"]+)"',doc):
        if target == 'verification.json' and pending_receipt:
            continue  # The one self-referenced receipt is written after these checks.
        if target.startswith('#'):
            A.require('id="'+target[1:]+'"' in doc, 'Broken section link.')
        else:
            p=(out/target).resolve()
            A.require(p.is_relative_to(out.resolve()) and p.is_file(),'Broken/external report asset: '+target)
    for status in ['COMPLETE FIXTURE','MISSING','FAILED','SPARSE','NO HARMFUL ACTIONS','TIED SCORES']:
        A.require(status in doc,'Missing visible completeness state: '+status)
    if (out/'verification.json').exists():
        for name,sha in json.loads((out/'verification.json').read_text())['assets'].items():
            A.require(file_hash(out/name)==sha,'Changed published report asset: '+name)


def write_html(out,specs,manifest,summaries):
    h=html.escape
    value=lambda v:'N/A' if v is None else f'{v:.2f}'
    full,tied,zero,sparse=(summaries[k][0] for k in ['complete','tied','no_harm','sparse'])
    cases=[('COMPLETE FIXTURE','Known correct ranking',value(full['within_all_strata']),f"All four strata; {full['unique_valid_states']} unique valid states per seed."),
           ('TIED SCORES','Constant width',value(tied['within_all_strata']),'Every harmful/nonharmful pair ties; no ranking information.'),
           ('NO HARMFUL ACTIONS','No valid two-class states',value(zero['within_all_strata']),f"{zero['harmful_alternatives']} harmful alternatives. This is not a safety claim."),
           ('SPARSE','One state and missing strata',value(sparse['within_available_strata'])+' available / '+value(sparse['within_all_strata'])+' complete','One valid state per seed; withhold broad claims.'),
           ('MISSING','Checkpoint unavailable','N/A','No outcome or metric is fabricated.'),
           ('FAILED','Restore gate failed','N/A','Preserve the attempt; no automatic outcome retry.')]
    table=''.join(f'<tr><td><b>{s}</b></td><td>{n}</td><td>{v}</td><td>{d}</td></tr>' for s,n,v,d in cases)
    cards=''
    for spec in specs:
        slug=spec['slug']
        links=' · '.join(f'<a href="{slug}.{ext}">{ext.upper()}</a>' for ext in ['png','svg','pdf','csv'])
        cards+=f'<section class="figure" data-question="{h(spec["question"])}" id="{slug}"><h2>{h(spec["title"])}</h2><p>{h(spec["counts"])}</p><div class="chart"><img src="{slug}.svg" alt="Synthetic test: {h(spec["title"])}"></div><p>{links} · Scroll a wide chart or open SVG for full size.</p></section>'
    options=''.join(f'<option>{h(q)}</option>' for q in dict.fromkeys(s['question'] for s in specs))
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BCA — calculation and figure checks</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f9;color:#18263b;font:16px/1.55 system-ui,sans-serif}}header{{background:#20375e;color:white;padding:32px max(24px,calc((100vw - 1250px)/2));}}h1{{font-size:32px;line-height:1.2;margin:8px 0}}h2{{font-size:23px;line-height:1.3}}.tag{{font-weight:800;letter-spacing:.09em;color:#ffe293}}main{{max-width:1300px;margin:auto;padding:24px}}.intro,.figure,.status{{background:white;border:1px solid #d7dfeb;border-radius:12px;padding:24px;margin:0 0 24px}}.notice{{border-left:5px solid #daab38;padding:10px 16px;background:#fffae8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:20px}}.grid b{{display:block;font-size:19px}}a{{color:#155799}}.scroll,.chart{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:780px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #d7dfeb;vertical-align:top}}th{{background:#edf2f8}}img{{display:block;width:100%;min-width:850px;height:auto}}select{{font:inherit;padding:9px;max-width:100%}}label{{font-weight:700}}code{{overflow-wrap:anywhere}}.muted{{color:#596a7f}}[hidden]{{display:none!important}}@media(max-width:600px){{header{{padding:22px}}main{{padding:12px}}.intro,.figure,.status{{padding:16px}}h1{{font-size:26px}}h2{{font-size:21px}}}}
</style></head><body><header><div class="tag">SYNTHETIC TESTS · NO RESEARCH RESULTS</div><h1>Can we trust the calculations and figures?</h1><p>Known answers first. Trained-model conclusions come later.</p></header><main>
<div class="intro"><p class="notice"><b>These are deliberately constructed examples.</b> No policy was trained, checkpoint accepted or simulator run. A correct chart is evidence about our measurement tools, not evidence that BCA works.</p><div class="grid">
<div><b>Harm: what happened?</b>Reference return − candidate return. Example: −3 − (−12.1) = <strong>9.1 lost reward</strong>. Both use the same starting state and later policy.</div>
<div><b>Ranking: was the warning useful?</b>Compare harmful and nonharmful actions in the same state. AUROC 1 = correct order, 0.5 = ties/chance-level order, 0 = reversed order. A missing class means N/A.</div>
<div><b>Coverage: did the band contain the error?</b>Learning target 1 + 0.9 × 8 = 8.2; predicted Q = 6; error = 2.2. Radius 1.5 × unit 1 × scale 2 = width 3, so 2.2 ≤ 3. The target is an estimate, not known true Q.</div>
<div><b>Uncertainty: how repeatable is it?</b>Resample paired training seeds, then whole reset episodes. Keep related actions and scores together. Five seeds give limited precision.</div></div><p><b>Keep the questions separate:</b> covering residual errors, recognizing harmful actions, and improving policy return are three different results.</p></div>
<section class="status" id="completeness"><h2>Completeness and failure examples</h2><p>Every intended result has a visible status. Blank results are never filled with zero.</p><div class="scroll"><table><thead><tr><th>Status</th><th>Case</th><th>Within-state width AUROC</th><th>Meaning</th></tr></thead><tbody>{table}</tbody></table></div></section>
<div class="intro"><label for="question">Show calculations for </label><select id="question"><option>All questions</option>{options}</select><p class="muted">Ten figures, each with its original numeric CSV and PNG/SVG/PDF exports. All lines are unsmoothed. The training curves and journals below are constructed examples. Numeric exports are checked automatically; browser/mobile layout requires separate review.</p><p><a href="all-figures.pdf">All figures as one PDF</a> · <a href="manifest.json">Fixture identity</a> · <a href="calculations.json">Calculation readout</a> · <a href="inputs.json">Exact inputs</a> · <a href="verification.json">Verification receipt</a></p><small>Manifest <code>{manifest['sha256']}</code></small></div>{cards}
</main><script>document.getElementById('question').addEventListener('change',e=>{{for(const s of document.querySelectorAll('.figure'))s.hidden=e.target.value!=='All questions'&&s.dataset.question!==e.target.value;}});</script></body></html>'''
    (out/'index.html').write_text(text,encoding='utf8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        verify_report(args.output)
        print('All report numeric tables, traces and local assets verified.')
    else:
        receipt=build(args.output)
        print(json.dumps(dict(output=str(args.output),verified_figures=receipt['verified_figures'],synthetic_only=True)))
