"""Declare comparison accounting; unknown development history remains unknown."""
import copy, hashlib, importlib.util, json
from pathlib import Path
base=Path(__file__).resolve().parents[1]
skill=Path(r'C:\Users\David\.codex\skills\offline-rl-experiments')
a=json.loads((base/'analysis/aggregate.json').read_text(encoding='utf-8'))
template=json.loads((skill/'assets/comparison.json').read_text(encoding='utf-8'))
out=copy.deepcopy(template); out['experiments']=[]
out['comparison_goal']='Within this frozen continuous ThrusterChain task, compare native TD3+BC and BCA under critical-region retention; five matched trained replicates per arm/retention.'
out['accounting']='Each experiment entry combines five independent trained replicas. Update/draw/wall-time counts are totals over the five. Sampled transitions counts critic minibatch draws, including repeats; actor/calibrator reuse does not create new source records. BCA additionally reuses its reserved 128-row bank at 18 refreshes per seed, 11520 score evaluations per five-seed condition. Final evaluation steps exclude separately recorded midpoint and designed-query diagnostics.'
for retention in ('full','rare','zero'):
    for arm in ('td3_bc','bca'):
        jobs=[j for j in a['jobs'] if j['job']['retention']==retention and j['job']['arm']==arm]
        assert len(jobs)==5
        e=copy.deepcopy(template['experiments'][0]); e['id']=retention+'/'+arm; e['algorithm']='native TD3+BC'+(' plus residual-calibrated BC multiplier' if arm=='bca' else '')
        hashes=[j['preparation']['learner_rows_sha256'] for j in jobs]
        e['source']=dict(repository='https://github.com/dbayha7/BCA.git',commit='b007052a2494dc651ac00469fa9e922b287cd1e5 + isolated uncommitted suite, source hashes recorded',entrypoint='experiments/thrusterchain_diagnostics/full_run.py; authorized Titan transfer wrapper for eight assigned jobs',resolved_config_sha256='b26924fa127cdb001ae1e1233056e8fc6a871921f29bb58986b8f967660f017e',composition='Deterministic TD3+BC actor, twin critic/target networks, delayed actor, raw reward and finite-horizon terminal mask; '+('native detached bounded BCA multiplier on BC loss, no value-credit refusal' if arm=='bca' else 'unweighted BC loss'))
        e['dataset']=dict(id='ThrusterChain observed regime H100 '+retention,version='core_20261020T2250Z, five data seeds 1101-1105',sha256=hashlib.sha256(json.dumps(hashes).encode()).hexdigest(),transitions=sum(j['preparation']['training_rows'] for j in jobs),per_replica_learner_hashes=hashes)
        e['training']=dict(actor_updates=sum(j['actor_updates'] for j in jobs),critic_updates=sum(j['updates'] for j in jobs),batch_size=256,sampled_transitions=sum(j['updates'] for j in jobs)*256,wall_seconds=sum(j['training_seconds'] for j in jobs),device='CPU; '+', '.join(j['host']+' seed '+str(j['job']['optimization_seed']) for j in jobs),online_update_env_steps=0,synthetic_transitions=0,model_based=False,dynamics_protocol=None)
        e['selection']=dict(regime='strict_offline',rule='Frozen 100000-update terminal policy; no online selection. 50000 midpoint evaluated descriptively, never used for checkpoint or hyperparameter selection. Strict offline describes this frozen training/selection run; development history is incompletely counted.',search_space_sha256='b26924fa127cdb001ae1e1233056e8fc6a871921f29bb58986b8f967660f017e',candidates_trained=5,online_eval_episodes=0,online_eval_env_steps=0,prior_development_env_steps=None,curve_episode_budgets=[0])
        e['splits']=dict(tuning_tasks=None,report_tasks=['ThrusterChain observed regime H100'],tuning_datasets=None,report_datasets=[str(j['job']['data_seed'])+'/'+retention for j in jobs],training_seeds=[j['job']['optimization_seed'] for j in jobs],selection_eval_seeds=[],report_eval_seeds=[82001],holdout_policy='Final bank fixed before training, no checkpoint selection; validation72001 descriptive. Model-specific Q queries92001 and calibration evaluation102001 separately recorded. Prior development feedback not fully enumerated.')
        e['evaluation']=dict(action_mode='deterministic',checkpoint_rule='final 100000 critic updates',episodes_per_trained_seed=400,total_env_steps=200000,holdout_protected=True,uncertainty_unit='trained_seed',normalization=dict(kind='raw',ref_min=None,ref_max=None,source='finite H100 undiscounted native task reward',clip=False))
        out['experiments'].append(e)
(base/'audit/comparison.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
spec=importlib.util.spec_from_file_location('comparison_audit',skill/'scripts/audit_comparison.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
report=mod.audit(out)
(base/'audit/comparison_metadata_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],missing=report['missing'],findings=report['findings'])))
