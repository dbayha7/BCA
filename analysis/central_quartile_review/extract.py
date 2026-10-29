"""Read-only extraction of the *completed* central-quartile evaluation.

Run: py -3.10 analysis/central_quartile_review/extract.py <rerun-package>
Nothing is read from individual result files until the terminal receipt, report,
coverage and every artifact hash have been checked. Output is JSON on stdout.
"""
import hashlib
import json
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def artifact(root, desc):
    require(set(desc) == {'path', 'bytes', 'sha256'}, 'bad artifact descriptor')
    rel = Path(desc['path'])
    require(not rel.is_absolute() and len(rel.parts) == 1 and ':' not in str(rel), 'unsafe artifact path')
    path = root / rel
    require(not path.is_symlink() and path.is_file(), 'missing or unsafe artifact')
    raw = path.read_bytes()
    require(len(raw) == desc['bytes'] and sha(raw) == desc['sha256'], 'artifact hash mismatch: ' + str(rel))
    return raw


def main(root):
    root = Path(root)
    terminal = json.loads((root / 'RUN_RESULT.json').read_bytes())
    require(terminal.get('status') == 'completed_evaluation_rerun', 'terminal status is not successful')
    require(terminal.get('prepared_identity_unchanged') is True and terminal.get('all_declared_jobs_completed') is True
            and terminal.get('execution_counts_known') is True and terminal.get('error') is None,
            'terminal integrity or accounting check failed')
    jobs = terminal['jobs']
    require(len(jobs) == 1 and jobs[0].get('completed') is True and jobs[0].get('execution_counts_known') is True,
            'evaluation job is incomplete')
    out = root / 'run' / 'evaluation'
    raw = (out / 'report.json').read_bytes()
    require(sha(raw) == jobs[0]['report_sha256'], 'report hash mismatch')
    report = json.loads(raw)
    require(report.get('status') == 'completed_central_quartile_evaluation'
            and report.get('execution_counts_known') is True and report.get('error') is None
            and report.get('study_evaluation_attempt') == 2, 'report is incomplete')
    reproduction = report['reproduction']
    require(reproduction == terminal['reproduction'] and reproduction['recorded_contexts'] == 143
            and reproduction['contexts_checked'] == 144
            and reproduction['tier1_exact'] + reproduction['tier2_near_exact'] == 143
            and reproduction['new_contexts'] == ['seed2_random__distance__20000/strong_current']
            and set(reproduction['banks_exact']) == {'ordinary', 'strong_current'}, 'reproduction gate incomplete')
    descriptors = report['snapshot_results']
    contexts = {k for k in descriptors if '/' in k}
    expected = {f'seed{s}_{c}__{stage}__{step:05d}/{context}'
                for s in range(3) for c in ('full', 'hole', 'random')
                for stage, step in [('warmup', 0), ('warmup', 10000)]
                + [(a, t) for a in ('native', 'bca', 'distance') for t in (15000, 20000)]
                for context in ('ordinary', 'strong_current')}
    require(contexts == expected and set(descriptors) == expected | {'ordinary_bank', 'strong_current_bank', 'common_reference'},
            'snapshot coverage differs')
    # Verify every file before parsing any result data. Bank array descriptors
    # are nested in their verified metadata and are verified too.
    verified = {k: artifact(out, d) for k, d in descriptors.items()}
    for context in ('ordinary', 'strong_current'):
        bank = json.loads(verified[context + '_bank'])
        artifact(out, bank['arrays'])
    rows = {}
    for key in sorted(contexts):
        x = json.loads(verified[key])
        identity = (x['seed'], x['condition'], x['arm'], x['checkpoint'], x['context'])
        stage_id, key_context = key.split('/')
        cell, stage, step = stage_id.split('__')
        require(cell == f"seed{x['seed']}_{x['condition']}" and int(step) == x['checkpoint']
                and key_context == x['context'] and x['arm'] == ('native' if stage == 'warmup' else stage),
                'snapshot descriptor and identity differ')
        require(identity not in rows and x['complete_state_unchanged'] is True, 'duplicate/changed model state')
        rows[identity] = x
    require(len(rows) == 144, 'identity coverage differs')
    extracted = {}
    for key, x in sorted(rows.items()):
        seed, condition, arm, step, context = key
        label = f'{context}/{condition}/{arm}/{step}/seed{seed}'
        free = x['free']
        value = {'discounted_return': free['mean_return'], 'undiscounted_return': free['mean_undiscounted_return'],
                 'high_step_hole_fraction': free['high_step_hole_fraction'], 'episodes_entering_hole': free['episodes_entering_hole']}
        if step > 0:
            value['q_panels'] = x['q']['panels']
            value['visited_panels'] = x['visited']['panels']
            value['raw_decisions'] = x['q']['raw_decisions']
            value['adjusted_decisions'] = x['q']['adjusted_decisions']
            value['signed_candidate_margins'] = x['q']['signed_candidate_margins']
            if 'distance_q' in x:
                value['distance_panels'] = x['distance_q']['panels']
        extracted[label] = value
    # Verify report's primary and other BCA-control paired matrices directly.
    for name, summary in report['policy_comparisons'].items():
        context, step, condition, comparison = name.split('/')
        control = comparison.removeprefix('bca_minus_')
        for seed in range(3):
            bca = rows[(seed, condition, 'bca', int(step), context)]['free']['returns']
            baseline = rows[(seed, condition, control, int(step), context)]['free']['returns']
            recorded = summary['paired_values_by_seed'][seed]
            require(len(bca) == len(baseline) == len(recorded) == 32, 'return row count differs')
            require(all(abs((a-b)-v) < 1e-9 for a,b,v in zip(bca, baseline, recorded)), 'return comparison differs')
    result = {'source_report_sha256': sha(raw), 'reproduction': reproduction,
              'disclosure': terminal['disclosure'], 'screens_and_comparisons': {
                  k: report[k] for k in ('learning_adequacy_passed', 'policy_screen_passed',
                     'accuracy_screen_passed', 'conditional_accuracy_testable', 'learning',
                     'policy_comparisons', 'return_interactions', 'accuracy_native_q1_minus_bca_adjusted_mae',
                     'accuracy_hole_minus_random', 'native_hole_minus_random_return', 'native_hole_minus_random_mae',
                     'policy_rescue_screen_passed', 'accuracy_rescue_screen_passed', 'high_load_anchor_families')},
              'snapshots': extracted}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        require(len(sys.argv) == 2, 'usage: extract.py <rerun-package>')
        main(sys.argv[1])
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('Refusing extraction: ' + str(exc), file=sys.stderr)
        sys.exit(1)
