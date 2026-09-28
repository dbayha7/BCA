"""Advisor brief from saved, bound evidence; no model or simulator execution."""
import csv,datetime,gzip,hashlib,html,json,os,re,shutil,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PACKAGED=(HERE/'inputs').is_dir()
WORK=Path(os.environ.get('BCA_ADVISOR_WORK',str(HERE/'inputs' if PACKAGED else HERE)))
REPO=Path(os.environ.get('BCA_REPO',str(HERE.parents[2] if PACKAGED else Path('C:/Users/David Bayha/Documents/GitHub/BCA'))))
OUT=REPO/'outputs/advisor_progress/2026-09-29'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs={}
def bound(p):
    inputs[str(p)]=sha(p);return read(p)
if (WORK/'preliminary_cluster.json').exists():
    data=bound(WORK/'preliminary_cluster.json')
else:
    packed=WORK.parent/'preliminary_cluster_evaluations.json.gz'
    inputs[str(packed)]=sha(packed);data=json.loads(gzip.decompress(packed.read_bytes()))
snapshot=bound(WORK/'cluster_status.json')
assert bound(WORK/'extract_actual_exit.json')['actual_exit']==0
assert bound(WORK/'probe_actual_exit.json')['actual_exit']==0
assert data['physical_runs']==56 and data['actor_trajectories']==70
assert snapshot['completed_worker_exits_all_zero'] and snapshot['completed_learner_exits_all_success']
for who in ('controller_identity','worker_identity'):
    assert all(snapshot[who][k] for k in ('present','command_matches','start_ticks_match','group_session_match'))
groups={}
for r in data['records']:
    for e in r['curve']+[r['final']]:
        assert abs(statistics.mean(e['scores'])-e['mean'])<1e-10
    if r['pairing']=='standalone_host_context_only':continue
    key=(r['algorithm'],r['dataset'],r['training_seed'])
    groups.setdefault(key,{})[r['actor']]=r
assert len(groups)==28
rows=[]
for (host,env,seed),v in groups.items():
    assert set(v)=={'host','bca'}
    for x,y in zip(v['host']['curve']+[v['host']['final']],v['bca']['curve']+[v['bca']['final']]):
        assert x['step']==y['step'] and x['episode_seeds']==y['episode_seeds']
    a,b=v['host'],v['bca']
    rows.append(dict(host=host,dataset=env,seed=seed,host_final=a['final_mean'],bca_final=b['final_mean'],delta=b['final_mean']-a['final_mean'],host_curve=statistics.mean(e['mean'] for e in a['curve']),bca_curve=statistics.mean(e['mean'] for e in b['curve']),pairing=a['pairing'],full_training_audit=False))

accepted=[]
for host,env,seed in [('td3_bc','hopper',202609171),('rebrac','hopper',202609171),('rebrac','walker2d',202609171),('rebrac','hopper',202609172)]:
    p=REPO/f'outputs/standard_bca/{host}/{env}/bca_noiw/s{seed}/verified-v1/comparison.json'
    c=bound(p)
    a,b=c['host_summary'],c['bca_summary']
    assert abs(b['final_mean']-a['final_mean']-c['final_delta'])<1e-10
    if host=='rebrac':
        match=next(r for r in rows if (r['host'],r['dataset'],r['seed'])==(host,env,seed))
        assert abs(match['host_final']-a['final_mean'])<1e-10 and abs(match['bca_final']-b['final_mean'])<1e-10
        assert abs(match['host_curve']-a['periodic_curve_mean'])<1e-10
        assert abs(match['bca_curve']-b['periodic_curve_mean'])<1e-10
        match['full_training_audit']=True
    accepted.append(dict(host=host,dataset=env,seed=seed,host_final=a['final_mean'],bca_final=b['final_mean'],delta=c['final_delta'],curve_delta=c['curve_delta'],wins=c['paired_episode_wins'],source=str(p.relative_to(REPO))))

for name in ('standard-first-td3-pair.json','standard-first-rebrac-pair.json','standard-first-rebrac-walker-pair.json','standard-second-rebrac-hopper-pair.json','ood-real-harm-report.json'):
    bound(REPO/'docs/validation'/name)
oodroot=REPO/'outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1'
ood=bound(oodroot/'summary.json')
for name in ('01_ranking.png','02_harm_and_support.png','03_effects_and_regret.png'):
    p=oodroot/name;inputs[str(p)]=sha(p);shutil.copyfile(p,OUT/name)
inputs[str(REPO/'outputs/ood/robustness-v2/preparation-v1/README.md')]=sha(REPO/'outputs/ood/robustness-v2/preparation-v1/README.md')
amendment=bound(WORK/'amended_stream_gate.json');assert amendment['passed'] and amendment['amendment_applied']

with (OUT/'cluster_pair_results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if (WORK/'preliminary_cluster.json').exists():
    with gzip.GzipFile(filename=str(OUT/'preliminary_cluster_evaluations.json.gz'),mode='wb',mtime=0) as f:f.write((WORK/'preliminary_cluster.json').read_bytes())
for name in ('probe_actual_exit.json','extract_actual_exit.json','amended_stream_gate.json','amendment_application.json','amendment_tests_actual_exit.json','amendment_tests.stderr'):
    shutil.copyfile(WORK/name,OUT/name)
# Save compact live metadata; full process receipts remain in the research workspace.
status={k:v for k,v in snapshot.items() if k not in ('queue_status','completed_receipts','sample_schemas')}
status['current_run']=snapshot['queue_status']['current']
(OUT/'cluster_live_status.json').write_text(json.dumps(status,indent=2)+'\n')

labels={'hopper':'Hopper','walker2d':'Walker2d','halfcheetah':'HalfCheetah','maze2d':'Maze2d','pen-human':'Pen human','pen-cloned':'Pen cloned','pen-expert':'Pen expert'}
hosts={'td3_bc':'TD3+BC','rebrac':'ReBRAC','iql':'IQL'}
envs=list(labels)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
for host in ('rebrac','iql'):
    fig,axs=plt.subplots(2,4,figsize=(16,8.5));axs=axs.ravel()
    for ax,env in zip(axs,envs):
        for method,color in [('host','#375a96'),('bca','#b94f24')]:
            ys=[]
            for seed in (202609171,202609172):
                r=groups[(host,env,seed)][method]
                x=np.array([e['step'] for e in r['curve']])/1e6;y=np.array([e['mean'] for e in r['curve']]);ys.append(y)
                ax.plot(x,y,color=color,alpha=.30,lw=.8)
            ax.plot(x,np.mean(ys,axis=0),color=color,lw=1.8,label='Host' if method=='host' else 'BCA, no IW')
        ax.set_title(labels[env],weight='bold');ax.set_xlabel('Host updates (millions)');ax.set_ylabel('Normalized score');ax.grid(alpha=.15)
    axs[-1].axis('off');axs[-1].text(0,.96,'PRELIMINARY',weight='bold',fontsize=15,color='#9b471a',va='top')
    axs[-1].text(0,.83,'Thin lines: each of two seeds\nThick lines: two-seed mean\n\nFive seeds are required.\nNo confidence band shown.\nNo smoothing or selected checkpoints.\n\n'+('Primary IQL pair shares Q/V.\n2 episodes per periodic bank.' if host=='iql' else 'Separate matched training runs.\n10 episodes per periodic bank.')+'\nFinal 20-episode bank is separate.',va='top',linespacing=1.5)
    axs[0].legend(frameon=False);fig.suptitle(hosts[host]+' | all seven training datasets | first two seeds',fontsize=18,weight='bold',y=.985)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/(host+'_learning_curves.png'),dpi=150,bbox_inches='tight',pad_inches=.25);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,6.8))
for ax,host in zip(axs,('rebrac','iql')):
    for j,env in enumerate(envs):
        rr=[next(r for r in rows if (r['host'],r['dataset'],r['seed'])==(host,env,s)) for s in (202609171,202609172)]
        ax.plot([r['delta'] for r in rr],[j,j],c='#aab4c3',lw=1)
        for r,marker,c,dy in zip(rr,('o','s'),('#375a96','#b94f24'),(-.10,.10)):
            ax.scatter(r['delta'],j+dy,c=c,marker=marker,s=55,label='Seed 1' if j==0 and marker=='o' else 'Seed 2' if j==0 else None)
    ax.axvline(0,color='#566273',lw=.8);ax.set_yticks(range(7),[labels[e] for e in envs]);ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_title(hosts[host],weight='bold');ax.set_xlabel('BCA − host final normalized score (points)');ax.legend(frameon=False)
fig.suptitle('Preliminary training results: show both seeds, including reversals',fontsize=16,weight='bold')
fig.text(.5,.025,'20 final episodes per actor; only 2/5 training seeds. Saved-result checks passed; full audits remain incomplete.\nIQL compares actors sharing Q/V. Different horizontal scales; no pooled score across datasets.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(OUT/'cluster_seed_differences.png',dpi=150,bbox_inches='tight',pad_inches=.25);plt.close(fig)

accepted_table='| Host / dataset | Seed | Host final | BCA final | Difference | Periodic-mean difference |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(f"| {hosts[r['host']]} / {labels[r['dataset']]} | {r['seed']} | {r['host_final']:.3f} | {r['bca_final']:.3f} | {r['delta']:+.3f} | {r['curve_delta']:+.3f} |" for r in accepted)
cluster_table='| Host / dataset | Seed 1: host → BCA | Seed 1 difference | Seed 2: host → BCA | Seed 2 difference |\n|---|---:|---:|---:|---:|\n'
for host in ('rebrac','iql'):
    for env in envs:
        a,b=[next(r for r in rows if (r['host'],r['dataset'],r['seed'])==(host,env,s)) for s in (202609171,202609172)]
        cluster_table+=f"| {hosts[host]} / {labels[env]} | {a['host_final']:.2f} → {a['bca_final']:.2f} | {a['delta']:+.2f} | {b['host_final']:.2f} → {b['bca_final']:.2f} | {b['delta']:+.2f} |\n"

md=f'''# BCA: advisor meeting brief — September 29, 2026

**Snapshot:** September 28, 6:35 p.m. EDT (22:35 UTC); saved evaluations extracted at 22:37 UTC. Standard BCA means both Bayesian and conformal components, without importance weighting of calibration fitting. Coverage and penalty settings are unchanged.

## What I would tell the advisor first

“The experiment asks whether BCA's prediction-error bands are useful for decisions involving poorly represented actions. We are testing two separate things: whether the bands warn about actions that actually hurt return, and whether the policy trained with BCA behaves better after those actions. The first TD3+BC Hopper seed does not yet show that benefit. Meanwhile, the cluster has completed two training seeds across all seven datasets for ReBRAC and IQL. Those training results vary by dataset and seed, so they cannot substitute for the OOD test. We are preserving the unfavorable results and completing the controlled follow-up before deciding whether importance weighting addresses a specific limitation.”

## 1. Why do this OOD experiment?

- **Research objective:** Determine whether standard BCA improves behavior when an action is poorly represented at the current state, and whether its width provides useful warning of harmful actions.
- **Why training curves are insufficient:** A higher average episode return can arise without better OOD handling. A lower Bellman residual can also coexist with poor action choices because its future-value target is itself a network estimate.
- **Why BCA might help:** Its learned residual scale can give different training examples different levels of caution. If that variation aligns with harmful action choices, the resulting training adjustment may improve the policy. This alignment is a hypothesis to test.
- **What this accomplishes:** It separates useful warning, better first-action choice, and better subsequent behavior. That tells us where an apparent benefit or limitation occurs and makes the next change defensible.

## 2. What are we doing, in plain language?

1. Train a plain host and its standard-BCA counterpart with the declared data, budget and seeds. BCA is active in the training objective after its first usable reference.
2. Freeze the resulting policies. Capture the complete simulator state from both policies' visits, so we can restore the same situation exactly.
3. Propose legal actions and measure how far each is from actions recorded in similar training states. This is an empirical support measure; unfamiliar does not automatically mean bad.
4. Save the candidate actions, randomness and BCA warning scores before measuring their consequences.
5. Restore the same state, impose one candidate action, then let the frozen host or BCA policy continue for up to 250 total steps. These simulator interactions are **evaluation**, not calibration fitting or further training.
6. Compare outcomes with a familiar reference action and compare both policies after the exact same imposed action. Repeat across states, support bands and independently trained seeds.

The completed v1 bank has ten actions per state and uses a host-action harm reference. The prospective v2 bank deliberately allocates near, moderately distant and strongly distant actions, uses the same recorded-action anchor for both policies, and separates action shift at familiar states from joint state/action shift. V2 is a declared follow-up after inspecting v1, not a retroactive change to v1.

## 3. Where does BCA enter, and what does calibration know?

| Host | BCA changes during training | What remains distinct |
|---|---|---|
| TD3+BC | Width-dependent multiplier on actor behavior cloning | Bellman critic targets retain the host rule |
| ReBRAC | Width-dependent multiplier on actor behavior cloning | Critic BC remains unchanged |
| CQL | Per-transition multiplier on the conservative critic gap | This is not a separate uncertainty penalty for every candidate action |
| IQL | Shrinks the capped actor advantage weight's excess above one | Primary actors share the same Q/V system |

BCA fits a positive scale to detached errors between Q predictions and the host's bootstrapped target. At reference refreshes, it uses reserved offline transitions to calculate Bayesian and conformal radii and keeps their maximum. Width combines the frozen scale, residual unit and radius. The calibrator does **not** know the true long-run Q-value and does **not** try actions in an environment during this procedure. The Bayesian component distributes mass over residual scores; it is not a collection of additional acting agents.

Ordinary policy evaluation uses frozen learned weights; no new penalty is added only at evaluation. The OOD study can query the frozen width as a diagnostic, but its rollouts do not update the policy or calibrator. Nominal 90% residual coverage is not 90% reliance on the old policy, nor 90% safe actions. Fresh OOD residual coverage has not been established by the current behavioral results.

## 4. What do good or bad results mean?

| Observation | What it supports | What it does not establish |
|---|---|---|
| Wider bands consistently rank harmful actions higher | Useful warning for the measured panel and continuation | Better policy return by itself, or universal OOD detection |
| BCA has higher return after the same unfamiliar action | Better subsequent behavior in that comparison | A better initial action, or the reason training improved |
| BCA loses less relative to the same familiar action AND has better absolute stressed return | Stronger evidence of useful action-stress robustness | A theorem or a guarantee for untested actions |
| BCA chooses a better first action with later policy fixed | Better local initial decisions | Broad whole-policy superiority |
| Width ranking is near chance | Little demonstrated separation of harmful and harmless actions | That conformal residual coverage is mathematically false |
| BCA underperforms on stressed actions | No benefit in that measured comparison | Proof of excessive conservatism or proof that IW will fix it |

For each outcome we report all declared seeds, missing captures and eligible states. Thousands of related actions are not thousands of independent training seeds. A nominally higher number from one seed is exploratory. The v2 protocol requires all five seeds and its predeclared uncertainty checks before a replicated directional claim.

## 5. Results we can present now

**Completed action study: TD3+BC Hopper, one training seed.** 5,120 recorded rollouts, 1,280,000 outcome steps, 256 captured rows from 64 shared reset blocks. Original interruption/recovery evidence is preserved.

| OOD diagnostic | Result | Meaning |
|---|---:|---|
| Width AUROC, equal-stratum mean of within-state rankings | 0.529 | Only slightly above chance descriptively |
| Support-distance AUROC, same aggregation | 0.566 | Higher than width in this seed |
| Fixed random / constant warning | 0.513 / 0.500 | Baseline context; no significance claim |
| Harmful alternatives under BCA continuation | 276 / 2,304 | Loss exceeds 1 raw reward unit versus host first action |
| Panels with both harm classes | 111 / 256 | Other panels' AUROC is undefined, not zero |
| BCA minus host first-action return, same BCA continuation | −0.120 | No average initial-action gain in this panel |
| Post-hoc distant-action return, BCA minus host continuation | −5.046 | BCA lower after the same imposed action |
| Post-hoc distant-action degradation advantage | −0.019 | Essentially no advantage relative to the common anchor |

The last two rows reuse 233 support-distant actions across 193 states from v1; they are a **post-hoc, unbalanced, one-seed bridge**, not the prospective v2 result. Distant actions improved average return relative to the chosen familiar anchor under both policies. Novelty and harm are different. These results do not yet demonstrate improved OOD handling by BCA.

![Accepted first-seed warning results](01_ranking.png)

**Independently accepted training comparisons: eight physical runs, four paired results.** These are whole-policy evaluations, not new OOD tests. Final scores use a separate twenty-episode bank. Periodic differences use the arithmetic mean of 200 periodic-bank means, not the final checkpoint's periodic evaluation.

{accepted_table}

ReBRAC Hopper is now independently accepted for **two of five seeds**. Its final differences are tiny, and its second-seed periodic mean is lower with BCA. ReBRAC Walker's first-seed higher mean coexists with a lower median and only 7/20 paired episode wins. The full five-seed conclusions remain pending.

## 6. What does the cluster show?

**56/140 cluster runs have successful process and learner exit receipts:** 28 ReBRAC and 28 IQL; both methods on all seven datasets for seeds 202609171 and 202609172. The live third-seed ReBRAC Hopper host was at 475,000 updates when inspected. Its original controller and worker identities matched. The cluster continued through the desktop interruptions.

The following is a **preliminary saved-evaluation readout**, not full acceptance of all training results. Result hashes match the queue; actual exits, budgets, evaluation artifact hashes/counts, finite scores and paired reset-bank equality were checked. Source/data reconstruction, full journals and checkpoint counters for the remaining runs still require their independent audits. Six cluster physical runs plus two local runs have those full accepted audits; **50 cluster closures remain pending full audit** at this snapshot.

IQL compares its two actors within the same BCA run, sharing Q/V. Its separately trained host is context only and is retained in the data export. No standalone IQL host is silently substituted into the primary pair. These scores are local candidates, not the published Unifloral/CORL means.

{cluster_table}

The two-seed pattern is mixed: ReBRAC HalfCheetah and Pen-human are lower with BCA in both seeds; ReBRAC Pen-expert is higher in both. IQL Hopper and Pen-human are lower in both, while Pen-expert is higher. Several other cells reverse sign. ReBRAC Maze's −82.42 then +93.21 illustrates why a favorable seed cannot be selected as the conclusion. No confidence interval or cross-dataset aggregate is claimed from this partial grid.

![Both seed differences](cluster_seed_differences.png)

![ReBRAC learning curves](rebrac_learning_curves.png)

![IQL learning curves](iql_learning_curves.png)

## 7. Why has the OOD study only finished one seed?

- The first action study was interrupted by two desktop restarts. Completed trajectories were preserved; partial attempts and recovery overhead remain documented. Recovery and independent saved-outcome checks consumed time.
- The subsequent request for a more direct OOD-action test led to a prospective support-balanced follow-up. Future unstarted v1 collection is held while that version is integrated; no v2 scientific collection is running yet.
- Runtime, storage, shared-resource and simulator-adapter acceptance are still pending. Many synthetic component tests have passed, but those are engineering progress, not additional scientific results. This execution/integration and audit backlog is the bottleneck.
- The local training lane is separately stopped on a CQL evaluation-schema failure after final evaluation. Its original failure and outputs remain preserved; it has not been silently retried. Cluster training is still active.
- The one v2 candidate-seed overlap is now corrected under explicit approval. The full 16,640-stream check passes with exactly one changed assignment. This removes that blocker but does not pass the remaining execution gates.

It would be inaccurate to say only one training seed exists, or to promise that all five OOD seeds will finish before the meeting. The immediate deliverable is the verified first study, transparent preliminary cluster results, and an explicit path to replication.

## 8. Justify each design choice or change

| Choice / change | Why it makes sense | Expectation and current check |
|---|---|---|
| Start with BCA without fitting IW | Isolates the basic calibration-to-host mechanism | Useful width/behavior is not yet demonstrated by v1 |
| Same data reservations, host settings and training budget | Reduces explanations based on unequal data or training | Pair audits verify them; remaining cluster audits pending |
| Freeze policies and restore full simulator state | Keeps comparisons tied to the same learned agents and situation | v1 first-repeat gates passed; v2 must archive complete final states |
| Hold later policy fixed when testing first actions | Separates initial action quality from continuation quality | v1 BCA-first effect is about −0.120 under BCA continuation |
| Impose identical actions under both continuations | Tests what happens after the same unfamiliar move | Post-hoc v1 bridge does not show a BCA advantage |
| V2 near/moderate/strong support bands | Ensures the test deliberately includes different degrees of action novelty | Prepared using training data only; v2 outcome results pending |
| Same familiar anchor, plus absolute stressed return | Prevents a weak familiar baseline from making degradation look artificially good | Report both contrasts; never choose each policy's own best action as the main reference |
| Separate state novelty and action novelty | Avoids attributing joint shift entirely to unfamiliar actions | State-near primary and joint-shift sensitivity are predeclared |
| Five training seeds | Tests whether findings survive training randomness | Two cluster seeds exist; OOD replication is incomplete |
| One approved candidate-seed correction | Removes a declared identifier collision before v2 outcomes | Exactly one change; 16,640 unique streams, no predecessor overlap |
| Hold coverage/strength tuning | Avoids changing several mechanisms after seeing a weak result | No coverage or penalty-strength setting changed |

## 9. What do we do next, and when is IW justified?

1. Finish the real v2 execution checks using the already accepted trained pairs; do not treat synthetic fixture success as simulator acceptance. Keep full state/action/reward archives and the cumulative step cap.
2. Continue fixed-order audits of newly completed cluster runs. Reuse accepted exports and present all declared seeds, including losses and reversals.
3. Execute the support-balanced OOD protocol on accepted eligible pairs, then read out initial-action quality, continuation robustness and warning ranking separately. First tranche is TD3+BC/ReBRAC × Hopper/Walker × five seeds; IQL/CQL training does not automatically mean those OOD studies are finished or ready.
4. Diagnose the limitation before proposing the next weighting rule. If the scale fits commonly seen transitions well but fails on the state–action regions relevant to the intended policy, a separately declared fitting-weight change is a plausible hypothesis. If width ranks harm well but behavior remains poor, the width-to-host adjustment may instead be the issue. If the residual target is weakly related to outcome harm, changing weights alone may not fix it.
5. For a future IW comparison, state the target population, exact weights and assumptions, compare with no-IW under matching budgets/seeds, monitor effective sample size, and retain both Bayesian and conformal components. Stronger weights can emphasize the wrong examples or reduce effective data. The current unweighted radius stage is not automatically a weighted-conformal covariate-shift guarantee.

**Suggested wording:** “We start with standard BCA to locate its strengths and limitations. We will test importance weighting if we identify a mismatch it is designed to address. Better weighting is an empirical hypothesis; weak vanilla results alone do not establish its necessity or benefit.”

## Evidence and access

- [Per-seed cluster results](cluster_pair_results.csv); [saved episode/curve arrays](preliminary_cluster_evaluations.json.gz); [live cluster snapshot](cluster_live_status.json).
- [Original accepted OOD report](../../ood/td3_bc/hopper/s202609171/real-harm-report-v1/report.html); [post-hoc bridge and limitations](../../ood/robustness-v2/preparation-v1/README.md).
- [BCA pseudocode](../../../ALGORITHMS.md); [exact integration points](../../../INTEGRATION.md); [approved one-stream amendment](../../../docs/OOD_V2_STREAM_AMENDMENT_20260928.md).
- Historical OOD limitations remain: unavailable 5,121 final following full-state contents, unknown ancestor exits and one input-only historical outcome, with all reservations retained. No missing evidence is invented. Fresh residual coverage and global OOD readiness remain unaccepted.
- This brief performs no training, model query or simulator step. It changes no scientific setting, old result, frozen source or thesis PDF. All original failures and negative outcomes remain available.
'''
(OUT/'ADVISOR_BRIEF.md').write_text(md,encoding='utf8')

# Small explicit renderer for this document's paragraphs, tables and figures.
def inline(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    return s
parts=[];lines=md.splitlines();i=0
while i<len(lines):
    line=lines[i]
    if not line: i+=1;continue
    if line.startswith('# '):parts.append('<h1>'+inline(line[2:])+'</h1>')
    elif line.startswith('## '):parts.append('<h2>'+inline(line[3:])+'</h2>')
    elif line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[s.strip() for s in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',s) for s in cells):table.append(cells)
            i+=1
        parts.append('<div class="table"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in table[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in table[1:])+'</tbody></table></div>');continue
    elif line.startswith('!['):
        m=re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)',line);assert m
        parts.append(f'<figure><img src="{m[2]}" alt="{html.escape(m[1])}"><figcaption>{html.escape(m[1])}</figcaption></figure>')
    elif line.startswith('- ') or re.match(r'^\d+\. ',line):
        numbered=not line.startswith('- ');items=[]
        while i<len(lines) and (re.match(r'^\d+\. ',lines[i]) if numbered else lines[i].startswith('- ')):
            items.append(re.sub(r'^(?:- |\d+\. )','',lines[i]));i+=1
        tag='ol' if numbered else 'ul';parts.append('<'+tag+'>'+''.join('<li>'+inline(v)+'</li>' for v in items)+'</'+tag+'>');continue
    else:parts.append('<p>'+inline(line)+'</p>')
    i+=1
page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>BCA advisor brief · September 29</title><style>body{margin:0;background:#edf1f6;color:#172339;font:18px/1.6 system-ui,sans-serif}main{max-width:1140px;margin:32px auto;background:white;padding:40px 52px;box-shadow:0 4px 22px #17233912}h1{font-size:36px;line-height:1.2;color:#18345b}h2{font-size:25px;border-top:3px solid #dce5f0;padding-top:26px;margin-top:42px;color:#18345b}p,li{max-width:100ch}li{margin:9px 0}strong{color:#173353}a{color:#1d5fa1}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.45}th{background:#203959;color:white;text-align:left}th,td{padding:11px 12px;border-bottom:1px solid #dce3eb;vertical-align:top}tr:nth-child(even){background:#f2f5f9}.table{overflow:auto}figure{margin:28px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#536178}footer{margin-top:40px;font-size:14px;color:#536178}@media(max-width:700px){main{margin:0;padding:24px}h1{font-size:29px}body{font-size:16px}}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0}h2{break-after:avoid}table,figure{break-inside:avoid}a{color:inherit}}</style><main>'+''.join(parts)+'<footer>Prepared from preserved evidence. Preliminary status is explicitly labeled; five-seed and OOD acceptance remain separate.</footer></main></html>'
(OUT/'ADVISOR_BRIEF.html').write_text(page,encoding='utf8')
report=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_utc=snapshot['observed_utc'],physical_cluster_closures=56,cluster_actor_trajectories=70,complete_cluster_training_seeds=2,fully_accepted_physical_runs=8,fully_accepted_pairs=4,completed_ood_training_seeds=1,accepted_training_comparisons=accepted,preliminary_pairs=rows,source_sha256=inputs,model_queries=0,simulator_steps=0)
(OUT/'evidence.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(sha(Path(p))==h for p,h in inputs.items())
print(json.dumps(dict(output=str(OUT),cluster_pairs=len(rows),accepted_pairs=len(accepted),source_files=len(inputs),figures=6),indent=2))
