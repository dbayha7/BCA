"""Analyze terminal saved JSON only; cannot train, simulate, or write outside this package."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from analysis_math import InvalidData, aggregate, analyze_seed, require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PACKAGE = ROOT / 'experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2'
DEFAULT_ATTEMPT = PACKAGE / 'stage1_exploratory_run/attempt_1'
EST = timezone(timedelta(hours=-5))


class WaitingForData(Exception):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, records, allowed_root):
    path = path.resolve()
    require(path.is_relative_to(allowed_root.resolve()), 'Input escapes the approved attempt: ' + str(path))
    if not path.is_file():
        raise WaitingForData(str(path))
    raw = path.read_bytes()
    def invalid_constant(value):
        raise InvalidData('Nonstandard JSON numeric constant: ' + value)
    result = json.loads(raw, parse_constant=invalid_constant)
    require(isinstance(result, dict), 'JSON object required: ' + str(path))
    records[str(path.resolve())] = digest(raw)
    return result


def check_sources(contract):
    for section in ('source_sha256', 'lineage_sha256'):
        for rel, expected in contract.get(section, {}).items():
            path = (ROOT / rel).resolve()
            require(path.is_relative_to(ROOT.resolve()), 'Pinned path escapes repository: ' + rel)
            require(path.is_file() and digest(path.read_bytes()) == expected,
                    'Pinned source or lineage changed: ' + rel)


def inspect_attempt(attempt, contract):
    """Fail closed on missing terminal evidence; never read model files or raw datasets."""
    records = {}
    allowed = (ROOT / contract['allowed_attempt']).resolve()
    require(allowed.is_relative_to(ROOT.resolve()), 'Approved attempt path escapes repository')
    require(attempt.resolve() == allowed, 'Unapproved input attempt')
    terminal = read_json(attempt / 'report.json', records, attempt)
    require(terminal.get('EXPLORATORY') and terminal.get('protocol_label') is None,
            'Expected explicitly exploratory terminal receipt')
    require(terminal.get('finished_utc') and terminal.get('failed_unit') is None, 'Attempt has no clean terminal completion')
    reservation = read_json(attempt / 'ATTEMPT.json', records, attempt)
    require(reservation.get('manifest_sha256') == contract['a4_manifest_sha256'], 'Different A4 package identity')
    results = []
    for seed in range(3):
        directory = attempt / f'evaluate_s{seed}'
        require(not (directory / 'NOT_REPRODUCED.json').exists(), 'Evaluation has nonreproduction evidence')
        job = read_json(directory / 'job.json', records, attempt)
        report = read_json(directory / 'report.json', records, attempt)
        supervisor = read_json(directory / 'SUPERVISOR_RESULT.json', records, attempt)
        require(job.get('id') == f'evaluate_s{seed}' and job.get('kind') == 'evaluate'
                and type(job.get('seed_index')) is int and job['seed_index'] == seed, 'Evaluation job identity mismatch')
        require(job.get('provenance') == 'research' and all(job.get(k) is None for k in
                ('test_seeds', 'training_seeds', 'visited_seed', 'smoke')), 'Test/smoke/overridden evaluation is ineligible')
        require(report.get('status') == 'completed' and report.get('unit_label') == 'COMPLETE'
                and report.get('kind') == 'evaluate' and report.get('job') == f'evaluate_s{seed}'
                and report.get('seed_index') == seed and not report.get('smoke') and not report.get('error'),
                'Evaluation worker did not complete normally')
        require(supervisor.get('completed') is True and supervisor.get('exit_code') == 0
                and supervisor.get('timed_out') is False and supervisor.get('interrupted') is False
                and not supervisor.get('error'), 'Evaluation supervisor did not confirm completion')
        require(isinstance(report.get('producer_sha256'), str) and len(report['producer_sha256']) == 64
                and all(c in '0123456789abcdef' for c in report['producer_sha256']),
                'Missing producer fingerprint')
        integrity = job.get('integrity')
        require(isinstance(integrity, dict), 'Missing integrity mapping')
        recorded = integrity.get('files')
        require(isinstance(recorded, dict) and all(isinstance(p, str) for p in recorded), 'Invalid integrity file mapping')
        for name, expected in contract['evaluation_source_sha256'].items():
            matches = [value for path, value in recorded.items()
                       if path.replace('\\', '/').endswith('/knee_edge_ood_v2/' + name)]
            require(matches == [expected], 'Job integrity lacks expected evaluation source: ' + name)
        runs = report.get('runs', {})
        try:
            cut = runs[f's{seed}_cutoff_stall3']['020000']['anchors_e2']
            random = runs[f's{seed}_random_stall3']['020000']['anchors_e2']
        except (KeyError, TypeError) as exc:
            raise InvalidData('Missing paired stall3/20k anchors') from exc
        results.append(analyze_seed(seed, cut, random))
    for path, expected in records.items():
        require(digest(Path(path).read_bytes()) == expected, 'Input changed during analysis: ' + path)
    output = aggregate(results)
    output.update(status='completed_saved_array_analysis', empirical=True,
                  original_exploratory_label=terminal.get('exploratory_label', terminal.get('label')),
                  original_protocol_label=terminal.get('protocol_label'), input_sha256=records,
                  pairing_evidence='Same source-pinned evaluation job builds shared anchors; matching fixed slots/times/order. '
                                   'Independent anchor identifiers are not stored in these reports.',
                  producer_fingerprint_independently_authenticated=False,
                  limitations=['Native-policy comparison only; no BCA detection or policy-improvement conclusion.',
                               'Recorded source hashes are checked against this contract; the AST producer fingerprint is recorded but not independently recomputed. This is not a full transitive seal audit.',
                               'Own-policy Q-pi differs across arms and from the native smoothed TD target.',
                               'Medians are reported separately; their additivity is not assumed.',
                               'V0(f) is one check, not a complete repaired Stage 1 decision.'])
    return output


def run(attempt, out):
    attempt, out = Path(attempt).resolve(), Path(out).resolve()
    require(out.is_relative_to(HERE / 'outputs'), 'Outputs must stay inside this isolated package outputs directory')
    require(not out.exists(), 'Output directory already exists; choose a new analysis identity')
    contract_raw = (HERE / 'SOURCE_CONTRACT.json').read_bytes()
    contract = json.loads(contract_raw)
    check_sources(contract)
    try:
        result = inspect_attempt(attempt, contract)
    except WaitingForData as exc:
        result = {'status': 'waiting_for_terminal_saved_arrays', 'empirical': False,
                  'missing': str(exc), 'scientific_result': None, 'protocol_decision': None,
                  'can_unlock_stage2': False}
    check_sources(contract)
    result.update(observed_est=datetime.now(EST).isoformat(), attempt=str(attempt),
                  analysis_contract_sha256=digest(contract_raw), numpy_version=np.__version__,
                  scientific_model_updates=0, simulator_transitions=0, calibration_updates=0,
                  analysis_source_sha256={p.name: digest(p.read_bytes()) for p in (HERE / 'analysis_math.py', HERE / 'analyze_saved.py')})
    out.mkdir(parents=True, exist_ok=False)
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.attempt, args.out)
    except (InvalidData, ValueError, OSError) as exc:
        print(json.dumps({'status': 'invalid_source_or_request', 'error': str(exc), 'scientific_result': None}))
        return 2
    print(json.dumps({k: result[k] for k in ('status', 'empirical', 'protocol_decision', 'can_unlock_stage2')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
