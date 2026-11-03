"""Render the checked saved-pair analysis. No scientific code is imported."""
from pathlib import Path
import argparse, csv, hashlib, json, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
COLORS={'stall3':'#0072B2','nostall':'#D55E00'}
MARKERS=['o','s','^']; DOSES=[0,21,63,189]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=HERE/'output/run2');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    source=args.input.resolve();out=args.output.resolve()
    if not out.is_relative_to(HERE) or out.exists():raise ValueError('Use a new output directory inside this analysis')
    receipt=json.loads((source/'RECEIPT.json').read_text());assert all(sha(source/n)==h for n,h in receipt['outputs'].items())
    data=json.loads((source/'metrics.json').read_text());policies=data['policies'];lookup={(p['world'],p['dose'],p['seed']):p for p in policies}
    with (source/'pairs.csv').open() as f: rows=list(csv.DictReader(f))
    def pairs(w,k,s):return {name:np.array([float(r[name]) for r in rows if r['world']==w and int(r['dose'])==k and int(r['seed'])==s]) for name in ['g','m','du','e','min_error','a0']}
    def models(w,k):return [lookup[w,k,s] for s in range(3)]
    def exact(p):return p['sensitivity'][0]
    def meanmae(w,k,component):return np.mean([exact(p)['relative_error'][component]['mae'] for p in models(w,k)])
    def counts(w,k,key):return [exact(p)['counts'][key] for p in models(w,k)]
    def fmtseq(values,d=3):return ' / '.join(f'{v:.{d}f}' if isinstance(v,(float,np.floating)) else str(v) for v in values)
    out.mkdir(parents=True,exist_ok=False)
    figures=[]
    fig,axs=plt.subplots(2,2,figsize=(11,8.3));fig.subplots_adjust(left=.09,right=.97,bottom=.14,top=.87,hspace=.45,wspace=.25)
    fig.suptitle('Native Q: a shared bias does not explain the wrong ordering',fontsize=18,fontweight='bold',y=.97)
    for j,w in enumerate(['stall3','nostall']):
        ax=axs[0,j];xs=[];ys=[]
        for s in range(3):
            a=pairs(w,0,s);xs.extend(a['g']);ys.extend(a['m']);ax.scatter(a['g'],a['m'],s=11,alpha=.25,c=COLORS[w],marker=MARKERS[s],label=f'Seed {s}')
        lo=min(min(xs),min(ys));hi=max(max(xs),max(ys));ax.plot([lo,hi],[lo,hi],color='#60717C',lw=1,ls='--',label='Correct margin')
        ax.axhline(0,color='#444',lw=.8);ax.axvline(0,color='#444',lw=.8);ax.set_title('Stall world' if w=='stall3' else 'No-stall world',color=COLORS[w],fontweight='bold');ax.set_xlabel('True advantage over logged action, g');ax.set_ylabel('Raw Q1 margin, m');ax.grid(alpha=.13)
        ax=axs[1,j]
        for s in range(3):
            p=lookup[w,0,s];xx=np.array([.85,.9,1.0])+(s-1)*.0012
            y=[p['fixed_strata'][f'magnitude_{i}']['e']['mae'] for i in [1,2,3]]
            ax.plot(xx,y,marker=MARKERS[s],color=COLORS[w],alpha=.75,label=f'Seed {s}')
        ax.set_xticks([.85,.9,1.0]);ax.set_xlabel('Withheld first-thruster magnitude');ax.set_ylabel('MAE of relative error m - g');ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.legend(fontsize=9,frameon=False)
    fig.text(.04,.045,'Top: 576 pairs per seed; points share 192 states. Upper-left quadrant prefers a harmful action; lower-right misses a useful one.\nBottom: all three fixed candidate magnitudes, no selected threshold. Axes differ by world; all raw margins and gaps are shown.',fontsize=10,color='#40515D')
    p=out/'01_native_relative_ranking.png';fig.savefig(p,dpi=220);figures.append(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8.3));fig.subplots_adjust(left=.09,right=.97,bottom=.15,top=.86,hspace=.42,wspace=.26)
    fig.suptitle('The penalty changes rankings; it does not become an accurate Q estimate',fontsize=16,fontweight='bold',y=.97)
    styles=[('q1','Raw Q1','#64727C','--'),('twin_min','Twin minimum','#009E73',':'),('adjusted','Q1 - kappa U','#151F29','-')]
    for j,w in enumerate(['stall3','nostall']):
        ax=axs[0,j]
        for off,(key,label,color,ls) in enumerate(styles):
            xx=np.arange(4)+(off-1)*.055
            for s in range(3):ax.scatter(xx+(s-1)*.012,[exact(lookup[w,k,s])['relative_error'][key]['mae'] for k in DOSES],marker=MARKERS[s],s=28,color=color,alpha=.65)
            ax.plot(xx,[meanmae(w,k,key) for k in DOSES],ls=ls,color=color,lw=2,label=label)
        ax.set_title('Stall world' if w=='stall3' else 'No-stall world',color=COLORS[w],fontweight='bold');ax.set_ylabel('Relative margin MAE');ax.set_ylim(bottom=0);ax.set_xticks(range(4),['Native','21','63*','189']);ax.grid(alpha=.15)
        if j==0:ax.legend(fontsize=9,frameon=False)
        ax=axs[1,j]
        for off,(key,label,color,ls) in enumerate([('corrected_harmful','Harmful preference corrected','#0072B2','-'),('suppressed_useful','Useful preference suppressed','#D55E00','--'),('new_harmful_preference','New harmful preference','#882255',':')]):
            xx=np.arange(3)+(off-1)*.07
            for s in range(3):ax.scatter(xx+(s-1)*.015,[exact(lookup[w,k,s])['counts'][key] for k in [21,63,189]],marker=MARKERS[s],s=24,color=color,alpha=.65)
            ax.plot(xx,[np.mean(counts(w,k,key)) for k in [21,63,189]],ls=ls,color=color,lw=2,label=label)
        ax.set_xticks(range(3),['21','63*','189']);ax.set_xlabel('Final model / actual kappa');ax.set_ylabel('Pair count per model (of 576)');ax.set_ylim(bottom=0);ax.grid(alpha=.15)
        if j==0:ax.legend(fontsize=8,frameon=False)
    fig.text(.04,.045,'Every dot is a training seed; lines are three-seed means. Each row compares scores on that model\'s own-policy values.\nPair flips against logged action are not bank argmax choices or deployed behavior. *63 was prespecified; this analysis is retrospective.',fontsize=10,color='#40515D')
    fig.savefig(out/'02_penalty_margin_tradeoff.png',dpi=220);figures.append(fig)

    native_table=[['Native metric','Stall: seeds 0 / 1 / 2','No-stall: seeds 0 / 1 / 2']]
    for label,key in [('Harmful pairs raw Q prefers','raw_false_preference'),('Useful pairs raw Q misses','raw_missed_useful')]:native_table.append([label,fmtseq(counts('stall3',0,key)),fmtseq(counts('nostall',0,key))])
    native_table.append(['Relative Q1-gap MAE',fmtseq([exact(p)['relative_error']['q1']['mae'] for p in models('stall3',0)]),fmtseq([exact(p)['relative_error']['q1']['mae'] for p in models('nostall',0)])])
    signal_table=[['Risk signal (native)','Stall: seeds 0 / 1 / 2','No-stall: seeds 0 / 1 / 2']]
    for key,label in [('negative_raw_margin','Negative Q1 margin'),('delta_u','U minus logged U'),('cutoff_distance','Distance beyond cutoff'),('head_excess_contrast','Q1 - twin-min contrast')]:signal_table.append([label,*[fmtseq([exact(p)['signals'][key]['pooled_auc'] for p in models(w,0)]) for w in ['stall3','nostall']]])
    eps=data['thresholds'][1];near=sum(p['sensitivity'][2]['counts']['near'] for p in policies)
    pages=[
      ('Which saved signals distinguish good from bad unsupported actions?',[
        ('p','Completed knee/stall study, not Path A. This is a retrospective follow-up using 24 saved models and 13,824 withheld-versus-logged pairs. Three training seeds are independent replications; actions and states are nested and reused. No model inference, simulator calls, training or recalibration ran.'),
        ('h','1. Native Q gets the relative decision wrong in stall'),
        ('p','For each state, g is the true own-policy return advantage of a withheld action over the logged action; m is the Q1 difference for that same pair. Relative error is e = m - g. A shared additive Q bias cancels. Harm means g < 0; useful means g > 0. These labels do not depend on the sign of either action\'s absolute Q error.'),
        ('table',native_table),
        ('p','Each model has 576 pairs: 192 states x three withheld alternatives. Native stall falsely prefers 510 of 1,212 harmful pairs and misses 353 of 516 useful pairs across the three models. No-stall falsely prefers 75 of 1,119 and misses 109 of 609. These counts differ from the previous bank-argmax counts because each bank contributes three pair comparisons.'),
        ('p','Among the 510 harmful stall preferences, 117 have no positive absolute Q1 error. No-stall has 38 such cases out of 75. Conversely, 203 of the 353 useful stall actions that Q misses have positive absolute Q error. Detecting absolute overestimation alone therefore cannot identify every ranking mistake.'),
        ('p','Native stall harmful false preferences have mean true advantage -3.747 but predicted advantage +0.725. Missed useful pairs have mean true advantage +0.707 but predicted advantage -0.973. The limitation is not only a common downward or upward offset.')]),
      ('Native ranking evidence', [('figure',figures[0])]),
      ('What U captures and what it learns',[
        ('h','2. Relative U reveals context that absolute U hid'),
        ('p','The table uses pooled harm AUROC across state-relative pairs within each model: higher signal predicts a lower-valued withheld action. This is descriptive discrimination, not a calibrated decision threshold. No AUROC direction was flipped.'),
        ('table',signal_table),
        ('p','In no-stall, delta U contains useful cross-state information (AUROC 0.751-0.813 for native), refining the earlier near-chance result for absolute U. Subtracting logged U removes a large state-level offset. Within a bank the subtraction is constant and cannot change candidate ordering. Only 2/6/4 native no-stall banks have both harm labels, so within-state harm discrimination remains weakly sampled.'),
        ('p','In stall, neither raw margin nor delta U reliably separates harmful from useful pairs. For native harmful false preferences mean delta U is only 0.00934, versus 0.04225 for missed useful pairs. A larger global multiplier does not supply the missing sign. These are associations, not a proof of why training produced the policy.'),
        ('h','3. The actual residual objective is a different estimand'),
        ('p','The local scale minimizes squared error between a Bayesian-weighted minibatch average of soft coverage and 0.9, plus a width regularizer. Its residual is a one-step, smoothed-target twin-min TD target minus current twin-min Q. Calibration uses an absolute residual divided by scale. Neither target is the signed full-return pair error m - g.'),
        ('p','The actor uses Q1 - kappa U with unchanged BC strength and native critic loss. Q1-to-twin-min mismatch is real, but changing the saved margin to twin-min barely changes native MAE: stall 2.12546 -> 2.12064; no-stall 0.33645 -> 0.34421. It is not a sufficient explanation of the observed ranking problem.')]),
      ('The penalty helps some decisions and harms others', [('figure',figures[1])]),
      ('Interpretation and the next measurement',[
        ('h','4. Preserve the benefit while locating the cost'),
        ('p','At kappa 189, stall corrects 68/192/68 harmful pair preferences, suppresses 32/54/33 useful preferences, and creates 42/6/27 new harmful preferences. No-stall corrects 10/14/42, suppresses 72/159/126, and creates 22/12/18. A positive U may still have a negative difference from logged U, so a downward score can increase relative preference for some actions.'),
        ('p','Relative score MAE increases after adjustment for every BCA model. That does not erase improved argmax regret or policy returns: a conservative score is not an unbiased Q estimate. The actual kappa-189 stall return gains remain +11.245/+121.479/+9.760, with no-stall changes -41.421/-98.321/-20.525. Candidate pairs cannot establish the cause of those policy effects.'),
        ('h','5. Numerical near-ties do not explain the broad pattern'),
        ('p',f'Exact original labels are preserved. Sensitivity excludes |g| <= {eps:.12f} and <= {2*eps:.12f}, anchored to the largest saved I1 discrepancy and twice that value. The larger level marks {near}/13,824 pairs near-tied, at most three per model. This is an empirical stress scale, not a proven error bound; the two original I1 failures and formal INCONCLUSIVE result remain unchanged.'),
        ('h','6. One next measurement, before choosing an estimator change'),
        ('p','Saved candidates contain Q1, Q2, U and own-policy Qpi, but no aligned signed TD targets, pointwise residuals, eta components or candidate gradients. Existing fitting-probe gradient files have different actions and no matched candidate oracle values. These cannot prove whether information was lost in the residual proxy, scale fit, calibration or penalty mapping.'),
        ('p','The smallest justified prospective test is a residual-alignment measurement on the existing native checkpoints and these same banks, not another training matrix. Use authenticated 15k states to examine the reference used by final U; 20k residuals answer a different current-model question. Up to 1,152 unique new one-step withheld successors would cover two worlds x 192 states x three alternatives; logged successors can be reused. Six models require 4,608 target-row evaluations per checkpoint level. No long rollouts or optimizer updates are needed. This would require separate execution approval and remain development analysis.'),
        ('p','Do not select a new algorithm from these labels. The evidence argues against simply increasing kappa, switching Q1 to twin-min, or subtracting a logged-state constant as a cure. First determine whether signed action-relative residuals contain the missing distinction. BC and Path A remain unchanged.')]),
      ('Evidence boundaries and reproducibility',[
        ('h','What is available when deciding?'),
        ('p','Given a state and candidate actions, Q1, Q2 and U are model outputs. Margins and delta U also require a comparator; this analysis has the saved logged action, which a deployed policy at a new state does not automatically possess. The cutoff label is benchmark knowledge. It is not a universal support estimator.'),
        ('p','Reward and successor state arrive after acting. One-step TD targets also need target-policy/critic queries and a specified smoothing-noise realization. Full-return Qpi, true gap g, relative error e and harm labels are evaluator-only. They were not fed into learning or calibration.'),
        ('h','What is and is not paired'),
        ('p','Within a world, complete candidate inputs are shared across models, but each model has its own continuation policy. Between worlds, identical row IDs do not mean identical physical states: the preceding analysis verified different retained states, rewards, successors and normalization. World identification is not the same problem as choosing the best action within one world.'),
        ('h','Source and numerical provenance'),
        ('p','Source review binds the actual actor, bridge, scale, calibration and evaluator code to recorded job integrity pins. All 24 complete original evaluation reports and previous analysis outputs are hash checked before and after analysis. The independent review is saved in SOURCE_REVIEW.md. Raw per-pair values and all seed/strength/sensitivity statistics are included; no fitted classifier, predictor search or new threshold claim is used.'),
        ('p','Reproduce from the authorized BCA checkout with the existing NumPy/matplotlib environment: python -B analysis/knee_decision_signal_v1/analyze.py --output-dir analysis/knee_decision_signal_v1/output/new_run. Then run build_report.py --input <new_run> --output <new_report_directory>. Both output directories must be new. Original saved data/reports are required; the ZIP contains analysis code and outputs, not large original datasets or checkpoints.'),
        ('p','Original CF_RESULT SHA256: b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197. The report is an exploratory explanation, not a replacement of the frozen verdict, a conditional-coverage guarantee, a Bayesian-superiority claim, or a finding about Honeypot/Bottleneck.')])
    ]
    markdown=[]
    def textpage(title,blocks,index):
        fig=plt.figure(figsize=(8.5,11));fig.patch.set_facecolor('white');y=.945
        titlelines=textwrap.wrap(title,44);fig.text(.07,y,'\n'.join(titlelines),fontsize=18,fontweight='bold',va='top',color='#142B3B');y-=len(titlelines)*.027+.024
        for kind,body in blocks:
            if kind=='h':
                lines=textwrap.wrap(body,80);fig.text(.07,y,'\n'.join(lines),fontsize=11.5,fontweight='bold',va='top',color='#0072B2');y-=len(lines)*.018+.012
            elif kind=='p':
                lines=textwrap.wrap(body,101);fig.text(.07,y,'\n'.join(lines),fontsize=9.4,linespacing=1.3,va='top',color='#23343F');y-=len(lines)*.0155+.016
            elif kind=='table':
                height=len(body)*.031;ax=fig.add_axes([.065,y-height,.87,height]);ax.axis('off');table=ax.table(cellText=body[1:],colLabels=body[0],loc='center',cellLoc='left',colWidths=[.36,.32,.32]);table.auto_set_font_size(False);table.set_fontsize(8.4);table.scale(1,1.4)
                for (row,col),cell in table.get_celld().items():
                    cell.set_edgecolor('#D4DEE4');cell.set_facecolor('#E8F1F7' if row==0 else ('#F7F9FA' if row%2 else 'white'))
                    if row==0:cell.set_text_props(fontweight='bold')
                y-=height+.026
        if y<.065:raise ValueError(f'Text page overflow: {index}, y={y}')
        fig.text(.07,.035,'Knee study | Saved-evidence analysis | No scientific execution',fontsize=8,color='#596A75');fig.text(.93,.035,str(index),ha='right',fontsize=8,color='#596A75');return fig
    pdfpath=out/'Knee_Decision_Signal_Report.pdf'
    with PdfPages(pdfpath) as pdf:
        for index,(title,blocks) in enumerate(pages,1):
            markdown.append('# '+title+'\n')
            if blocks[0][0]=='figure':
                pdf.savefig(blocks[0][1]);markdown.append('See figure '+str((index//2))+'.\n');continue
            f=textpage(title,blocks,index);pdf.savefig(f);plt.close(f)
            for kind,body in blocks:
                if kind=='table':markdown.append('\n'.join([' | '.join(row) for row in body])+'\n')
                else:markdown.append(('## ' if kind=='h' else '')+body+'\n')
    (out/'REPORT.md').write_text('\n'.join(markdown),encoding='utf-8')
    assert all(sha(source/n)==h for n,h in receipt['outputs'].items())
    (out/'REPORT_RECEIPT.json').write_text(json.dumps({'status':'completed_report','input_receipt_sha256':sha(source/'RECEIPT.json'),'author_source_sha256':sha(__file__),'pages':len(pages),'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'renderer':'matplotlib PdfPages; reportlab unavailable; rendered Poppler review required','scientific_calls':0},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(pdfpath),'pages':len(pages),'sha256':sha(pdfpath)}))

if __name__=='__main__':main()
