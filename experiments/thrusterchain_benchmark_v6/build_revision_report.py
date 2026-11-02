"""Generate the v3 meeting brief from validated immutable run records."""
import argparse
import json
from pathlib import Path
from thrusterchain.reporting import validate_control_records

ROOT=Path(__file__).resolve().parent

def build(run,report):
    run=Path(run)
    results=json.loads((run/'benchmark_results.json').read_text())
    control=validate_control_records(results,run/'control_episode_records.csv')
    diagnostic=json.loads((run/'diagnostic_results.json').read_text())
    review=json.loads((ROOT/'audits/ten_pass_revision_v3/ten_pass_review.json').read_text())
    suite=json.loads((ROOT/'audits/revision_v3_final/full_suite.json').read_text())
    source=json.loads((run/'source_manifest_after.json').read_text())['source_sha256']
    if (not review['all_passed'] or review['source_sha256']!=source or not suite['successful']
            or suite['source_sha256']!=source):
        raise RuntimeError('passing current review and full suite required')
    total=zero=0
    for cell in diagnostic['cells']:
        for repetition in cell['repetitions']:
            for arm in repetition['raw']['paired_action_bank']['arms'].values():
                total+=len(arm['selected_index'])
                zero+=sum(index==0 for index in arm['selected_index'])
    labels={'behavior_cloning':'Offline BC','random':'Uniform random',
            'true_model_mpc':'True-model MPC','identified_model_mpc':'Train-only ID + MPC'}
    families=('recorded','strong_current','big_waves','big_kicks')
    table=['| Controller | Recorded | Strong current | Big waves | Big kicks |',
           '|---|---:|---:|---:|---:|']
    for method,label in labels.items():
        table.append('| '+label+' | '+' | '.join(f"{results['baselines'][method]['families'][name]['mean_return']:.2f}" for name in families)+' |')
    recipe=dict(profile='smoke',independent_training_seeds=1,control_validation=control,
        conditional_evaluation_episodes_per_family=results['evaluation_episodes_per_family'],
        evaluation_pairing='same reset seeds across methods and families; families are correlated',
        return_undiscounted='sum t=0..99 reward[t]',return_discounted='sum t=0..99 0.98**t * reward[t]',
        means='arithmetic mean over raw returns',sample_sd='numpy.std(returns,ddof=1); null for n=1',
        confidence_intervals='not claimed from this tiny smoke',training_seed_uncertainty='not estimated',
        terminal_choice_count=total,terminal_zero_action_choices=zero,
        diagnostic_recipe='diagnostic_results.json statistical_recipe and per-repetition raw records',
        decision_benefit_claim=False)
    with (run/'statistical_recipe.json').open('x',encoding='utf8') as handle:
        json.dump(recipe,handle,indent=2,allow_nan=False)
    lines=['# ThrusterChain: revision v3 meeting brief','',
        'ThrusterChain tests offline control and explicitly controlled state/action coverage in a small deterministic simulator. Regular mode trains from a fixed log, freezes the selected policy and preprocessing, then runs live simulator episodes. BCA is not required to use the benchmark.','',
        '**Evidence level: verified software integration and diagnostic smoke. A research-scale neural-policy comparison is still pending.**','',
        f"The new run used {results['training_episodes']} offline episodes, one dataset/training seed, four controllers and four scenario families, with {results['evaluation_episodes_per_family']} paired evaluation seeds per family. All {control['rows']} complete episodes are retained as raw records. Bounded MPC uses planning horizon {results['planner_horizon']}; this is a receding-horizon reference, not the full-episode optimum.",'',
        f"The full pinned-runtime suite passed {suite['tests_run']-suite['skipped']} tests with {suite['skipped']} skips. Ten topical acceptance passes passed against the exact source snapshot below. These software passes are not independent scientific replications.",'',
        '## What the smoke shows','',
        'Raw mean return is shown below; higher is better. Two evaluation seeds describe one frozen policy and do not establish algorithm superiority or uncertainty across training seeds.','',*table,'',
        'The identified-model controller fits only allowed training observations and executed actions. It assumes the public actuator map and reward structure. Its matrices and separate identification receipt are retained; the true-model planner remains evaluator-privileged.','',
        '## What the terminal diagnostic can establish','',
        f"The nine coverage cells and four weighting/fallback arms produced {total:,} action choices; {zero:,} selected the zero action. The terminal reward contains only present-state cost and command cost, making zero the optimal terminal command. These runs test weighting, coverage and refusal mechanics; they cannot demonstrate uncertainty-guided decision benefit. No replacement fixture was tuned to improve a method's score.",'',
        'The calibrated target is the exact deterministic final-step reward after the canonical float32 action map. Density ratios use continuous requested commands. Neither a general neural-Q guarantee, a posterior over true Q, nor adaptive policy-improvement coverage is claimed. Missing support, infinite width, ESS and finite-width coverage remain distinct.','',
        '## Corrections verified in this revision','',
        '- Generator metadata now distinguishes integer fields from Boolean and floating-point aliases. Recorded NumPy version provenance and seeded-lineage checks remain intact.',
        '- Strict CSV parsing refuses malformed quoting before numeric consistency checks. Valid quoted exports and reordered rows remain supported.',
        '- Train-only identification receipts require essential fitting facts that agree with the actual saved matrices and dataset. Matching duplicate receipts alone is insufficient. This checks internal consistency, not authenticity of execution.',
        '- The original results remain unchanged. Earlier corrections to NumPy compatibility, planner labels, controllability wording and installed attribution remain covered by regression checks.','',
        '## Use and extend','',
        '[Regular quickstart](README.md) | [Environment](docs/environment.md) | [Coverage](docs/coverage_and_calibration.md) | [Research comparisons](docs/research_positioning.md) | [Revision receipts](REVISION_V3.md)','',
        'The guards verify supported cooperative adapters. Integrated stages share a process; CLI stages share filesystem permissions. Reads, saved I/O aliases and arbitrary native or transient adaptation are outside comprehensive interception. Zero observed intercepted calls is not a security-isolation claim.','',
        'Five independent training/data seeds, full neural-host budgets and fidelity, large final banks, untouched scenario-family evaluation and external release review remain future gates. The shifted families in this smoke are known development families. Historical 45-run results are preserved as development evidence; their exact actor source and normalization audit remain separate.','',
        f'Completed run: `{run.as_posix()}`. Source SHA-256: `{source}`.','']
    with Path(report).open('x',encoding='utf8') as handle: handle.write('\n'.join(lines))
    print(json.dumps(dict(report=str(Path(report).resolve()),control_rows=control['rows'],terminal_choices=total,terminal_zero_choices=zero,source_sha256=source)))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'results/revision_v3_smoke')
    parser.add_argument('--report',type=Path,default=ROOT/'MEETING_REPORT_V3.md')
    args=parser.parse_args(); build(args.run,args.report)
