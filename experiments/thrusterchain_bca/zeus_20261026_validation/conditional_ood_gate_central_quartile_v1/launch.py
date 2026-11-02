"""One separately reserved design gate. No imports of scientific code at top level."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PAYLOAD = HERE.parents[1] / 'titan_20261026/payload'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, exclusive=False):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with Path(path).open('x', encoding='utf-8', newline='\n') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    else:
        temporary = Path(str(path) + '.tmp.' + str(os.getpid()))
        temporary.write_text(raw, encoding='utf-8')
        os.replace(temporary, path)


def decision_contract():
    frozen = read(HERE / 'DECISION_FROZEN.json')
    if sha(HERE / 'contract.json') != frozen['contract_sha256']:
        raise ValueError('Decision contract changed after freeze')
    return read(HERE / 'contract.json')


def verify_source_lineage():
    lineage = read(HERE / 'SOURCE_LINEAGE.json')
    prior = HERE.parent / 'conditional_ood_gate_v1_retry_20261028'
    if lineage['prior_directory'] != prior.name or not lineage['new_experiment_not_retry']:
        raise ValueError('Replacement experiment lineage differs')
    if (sha(prior / 'run/report.json') != lineage['prior_report_sha256']
            or sha(prior / 'run/RESULT.json') != lineage['prior_result_sha256']):
        raise ValueError('Prior failed gate evidence changed')
    report, result = read(prior / 'run/report.json'), read(prior / 'run/RESULT.json')
    if (report['decision'] != 'failed_development_design_gate'
            or result['exit_code'] != 0 or not result['execution_counts_known']):
        raise ValueError('Prior gate is not the completed failed design screen')
    return {'prior_decision': report['decision'], 'new_experiment_not_retry': True}


def prepare():
    decision_contract()
    verify_source_lineage()
    if (HERE / 'run').exists() or (HERE / 'PREPARED.json').exists():
        raise FileExistsError('Existing preparation or attempt is preserved; no automatic re-freeze')
    paths = set(HERE.glob('*.py')) | {HERE / x for x in ('contract.json', 'DECISION_FROZEN.json', 'PROTOCOL.md', 'SOURCE_LINEAGE.json')}
    folders = ['transfer_bundle/core_v4', 'unifloral_td3bc_validation_v1',
               'unifloral_bca_pipeline_validation_v1', 'unifloral_frozen_usefulness_v1',
               'unifloral_frozen_usefulness_v1_seedfix',
               'unifloral_operator_alignment_v1', 'unifloral_checkpoint_v1',
               'unifloral_export_recovery_v2']
    for folder in folders:
        base = PAYLOAD / folder
        paths.update(p for p in base.rglob('*.py') if '__pycache__' not in p.parts)
        paths.update(base.glob('*.json'))
    paths.update(p for p in (PAYLOAD / 'transfer_bundle/core_v4').rglob('*')
                 if p.is_file() and p.suffix in ('.json', '.npy', '.npz'))
    paths.add(HERE.parent / 'bca_one_sided_v1/logged_data.py')
    paths.add(HERE.parent / 'support_shift_pilot_v1/runtime.py')
    paths.add(PAYLOAD / 'transfer_bundle/BUNDLE_MANIFEST.json')
    paths.add(PAYLOAD / 'transfer_bundle/research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz')
    checkpoint = PAYLOAD / 'unifloral_checkpoint_v1/train_20261023T183248_552877Z'
    paths.add(checkpoint / 'bindings.json')
    # The authenticated checkpoint/export verifier checks legacy manifest linkage.
    # Freeze the exact model components independently as well.
    paths.update(checkpoint.rglob('state.msgpack'))
    export = PAYLOAD / 'unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/inference_export'
    paths.update(p for p in export.iterdir() if p.is_file())
    records = {}
    for p in sorted(paths):
        if not p.is_file() or not p.resolve().is_relative_to(REPO.resolve()):
            raise ValueError('Missing or unsafe input: ' + str(p))
        records[p.relative_to(REPO).as_posix()] = sha(p)
    result = {'schema': 'bca.conditional_gate_prepared.v1', 'created_utc': utc(),
              'contract_sha256': sha(HERE / 'contract.json'), 'files': records,
              'gate_outcomes_read': False, 'attempt_started': False}
    write(HERE / 'PREPARED.json', result, exclusive=True)
    return {'status': 'prepared_not_executed', 'files': len(records),
            'prepared_sha256': sha(HERE / 'PREPARED.json'), 'contract_sha256': result['contract_sha256']}


def verify_prepared():
    contract = decision_contract()
    verify_source_lineage()
    prepared = read(HERE / 'PREPARED.json')
    if prepared['contract_sha256'] != sha(HERE / 'contract.json'):
        raise ValueError('Prepared contract identity differs')
    for relative, expected in prepared['files'].items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO.resolve()) or sha(path) != expected:
            raise ValueError('Frozen source/input changed: ' + relative)
    return {'prepared_sha256': sha(HERE / 'PREPARED.json'),
            'contract_sha256': prepared['contract_sha256'],
            'verified_files': len(prepared['files']), 'global_seconds': contract['caps']['global_seconds']}


def runtime_versions(contract):
    actual, mismatches = {}, {}
    for package, expected in contract['runtime_packages'].items():
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual[package] = None
        if actual[package] != expected:
            mismatches[package] = {'required': expected, 'actual': actual[package]}
    if sys.version_info[:2] != (3, 12):
        mismatches['python_minor'] = {'required': '3.12', 'actual': '.'.join(map(str, sys.version_info[:2]))}
    return {'python': sys.version, 'executable': sys.executable, 'packages': actual,
            'compatible': not mismatches, 'mismatches': mismatches,
            'scope': 'Package metadata only; device and model readiness checked inside capped worker'}


def reserve(directory, identity):
    directory.mkdir(exist_ok=False)
    write(directory / 'ATTEMPT.json', {'reserved_utc': utc(), 'pid': os.getpid(),
          'attempt': 1, 'identity': identity, 'no_automatic_restart': True,
          'old_training_holds_unchanged': True}, exclusive=True)


def optional_receipt(path):
    if not path.exists():
        return None, None
    try:
        return read(path), None
    except Exception as exc:
        return None, type(exc).__name__ + ': ' + str(exc)


def require_supervised_worker(seconds, attempt):
    if not math.isfinite(seconds) or not 0 < seconds <= 900:
        raise ValueError('Worker needs a finite positive remaining budget no larger than900seconds')
    if os.getppid() != attempt['pid']:
        raise ValueError('Worker must be the direct child of its reserving supervisor')


def execute():
    started = time.monotonic()
    contract = decision_contract()
    identity = verify_prepared()
    runtime = runtime_versions(contract)
    if not runtime['compatible']:
        raise ValueError('Runtime mismatch; nothing reserved: ' + json.dumps(runtime['mismatches']))
    out = HERE / 'run'
    reserve(out, identity)
    write(out / 'RUNTIME_METADATA.json', runtime)
    env = os.environ.copy()
    env.update({'JAX_PLATFORMS': 'cuda', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest',
                'XLA_FLAGS': '--xla_gpu_triton_gemm_any=True', 'PYTHONUNBUFFERED': '1',
                'PYTHONDONTWRITEBYTECODE': '1'})
    remaining = contract['caps']['global_seconds'] - (time.monotonic() - started)
    child = None
    interrupted, timed_out, exit_code, error = False, False, None, None
    try:
        if remaining <= 0:
            raise TimeoutError('Global budget exhausted during preflight')
        with (out / 'worker.stdout.txt').open('x', encoding='utf-8') as stdout, (out / 'worker.stderr.txt').open('x', encoding='utf-8') as stderr:
            child = subprocess.Popen([sys.executable, '-B', str(HERE / 'launch.py'), '--worker', '--seconds', str(remaining)],
                                     cwd=HERE, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=(os.name != 'nt'))
            write(out / 'DISPATCH.json', {'utc': utc(), 'pid': child.pid, 'remaining_seconds': remaining})
            last = None
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= contract['caps']['global_seconds']:
                    timed_out = True
                    break
                progress = out / 'progress.json'
                if progress.exists():
                    current = read(progress)
                    tag = (current.get('phase'), current.get('saved_utc', current.get('utc')))
                    if tag != last:
                        print(json.dumps(current), flush=True)
                        last = tag
                time.sleep(min(2, max(0.05, contract['caps']['global_seconds'] - elapsed)))
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        if child is not None:
            if child.poll() is None:
                # Only this newly created, separately reserved worker is affected.
                child.kill()
            exit_code = child.wait()
        report, report_error = optional_receipt(out / 'report.json')
        last_progress, progress_error = optional_receipt(out / 'progress.json')
        unchanged = None
        try:
            unchanged = verify_prepared() == identity
        except BaseException as exc:
            error = error or ('Post-run identity check: ' + str(exc))
        result = {'utc': utc(), 'status': 'interrupted' if interrupted else 'timed_out' if timed_out else 'worker_finished',
                  'exit_code': exit_code, 'elapsed_seconds': time.monotonic()-started,
                  'timed_out': timed_out, 'interrupted': interrupted, 'prepared_identity_unchanged': unchanged,
                  'worker_status': report.get('status') if report else None,
                  'decision': report.get('decision') if report else None,
                  'counts': report.get('ledger') if report else last_progress.get('ledger') if last_progress else None,
                  'execution_counts_known': bool(report and report.get('execution_counts_known') and not timed_out and not interrupted),
                  'report_sha256': sha(out / 'report.json') if report else None,
                  'report_read_error': report_error, 'progress_read_error': progress_error,
                  'error': error, 'no_automatic_retry': True, 'training_hold_remains': True}
        write(out / 'RESULT.json', result)
        print(json.dumps(result), flush=True)
    return 0 if exit_code == 0 and not timed_out and not interrupted and unchanged else 1


def worker_main(seconds):
    # A worker cannot be invoked directly to create an unreserved attempt.
    identity = verify_prepared()
    out = HERE / 'run'
    attempt = read(out / 'ATTEMPT.json')
    require_supervised_worker(float(seconds), attempt)
    if attempt['identity'] != identity:
        raise ValueError('Worker reservation identity mismatch')
    with (out / 'WORKER_ENTRY.json').open('x', encoding='utf-8') as f:
        json.dump({'utc': utc(), 'pid': os.getpid()}, f)
    deadline = time.monotonic() + min(float(seconds), decision_contract()['caps']['global_seconds'])
    import worker
    result = worker.run(out, decision_contract(), None, deadline)
    return 0 if result['status'].startswith('completed') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'preflight', 'execute', 'worker'):
        group.add_argument('--'+flag, action='store_true', help=argparse.SUPPRESS if flag == 'worker' else None)
    parser.add_argument('--seconds', type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare()))
    elif args.preflight:
        print(json.dumps({'identity': verify_prepared(), 'runtime': runtime_versions(decision_contract()),
                          'attempt_exists': (HERE / 'run').exists()}))
    elif args.worker:
        return worker_main(args.seconds)
    else:
        return execute()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'error': type(exc).__name__+': '+str(exc)}), flush=True)
        raise SystemExit(1)
