"""Create an audit-helper manifest from the completed frozen-controller run.

Training counters describe *this* evaluation, not the earlier checkpoint
training. Unknown prior selection budgets remain null. This file is post-run
documentation and does not modify the sealed protocol or run any simulator.
"""
import copy
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
template = Path(r'C:\Users\David\.agents\skills\offline-rl-experiments\assets\comparison.json')
base = json.loads(template.read_text(encoding='utf-8'))
result_path = root / 'reranker_run_01' / 'RERANKER_RESULT.json'
result = json.loads(result_path.read_text(encoding='utf-8'))
assert result['status'] == 'completed_frozen_controller_evaluation'
seed_sets = result['seed_audit']['fresh_seeds']
row = base['experiments'][0]
base['comparison_goal'] = 'Compare five fixed controllers and capped versus raw and symmetric action selection on fresh starts.'
base['generalization_claim'] = 'within_tasks'
base['experiments'] = []
for method in result['methods']:
    x = copy.deepcopy(row)
    x['id'] = method
    x['algorithm'] = 'frozen TD3+BC checkpoint; fixed ' + method + ' controller'
    x['source'].update(repository='BCA', commit='84202ebd1d089534180b454e8fa1f1e80d866718',
        entrypoint='capped_width_bca_dev_v5/reranker_runner.py',
        resolved_config_sha256=hashlib.sha256((root/'RERANKER_SOURCE_SEAL.json').read_bytes()).hexdigest(),
        composition='Uncommitted v5 source and fixed 20k native or BCA checkpoint; see source seal and terminal result')
    x['dataset'].update(id='ThrusterChain retained hole', version='core_v4', transitions=67108)
    x['training'].update(actor_updates=0, critic_updates=0, online_update_env_steps=0,
                         synthetic_transitions=0, model_based=False, device='CPU')
    x['selection'].update(regime='online_selection',
        rule='Fixed methods and cap from training probes before fresh outcomes; prior online development was used',
        search_space_sha256=hashlib.sha256((root/'PROTOCOL.md').read_bytes()).hexdigest())
    x['splits'].update(tuning_tasks=['ordinary','strong_current'], report_tasks=['ordinary','strong_current'],
        tuning_datasets=['retained hole'], report_datasets=['retained hole'], training_seeds=[0,1,2],
        report_eval_seeds=seed_sets['ordinary']+seed_sets['strong_current'],
        holdout_policy='Fresh resets; prior development on same dataset and contexts, no independent dataset holdout')
    x['evaluation'].update(action_mode='deterministic', checkpoint_rule='final 20k hole checkpoints, fixed before outcomes',
        episodes_per_trained_seed=32, total_env_steps=38400, holdout_protected=True,
        uncertainty_unit='trained_seed', normalization={'kind':'raw','ref_min':None,'ref_max':None,'source':None,'clip':False})
    base['experiments'].append(x)
base['accounting_note'] = ('Per controller: 19,200 complete-rollout and 19,200 first-action-continuation transitions; '
                           '104 shared simulator-parity steps excluded from per-controller total. '
                           'Prior development selection interactions are unknown and intentionally null.')
(root/'comparison.json').write_text(json.dumps(base,indent=2)+'\n',encoding='utf-8')
