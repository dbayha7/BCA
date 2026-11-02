"""Explicit amendment validation, lineage attempt 3. No production execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(ROOT))
import launch
from common import atomic_json, read, require, sha

CAP_SECONDS = 900


class Signals:
    """Record-only handlers cannot interrupt child creation, cleanup or receipts."""
    def __init__(self):
        self.received = []
        self.previous = {}

    def record(self, number, frame=None):
        self.received.append(int(number))

    def __enter__(self):
        for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
            if hasattr(signal, name):
                number = getattr(signal, name)
                self.previous[number] = signal.getsignal(number)
                signal.signal(number, self.record)
        return self

    def __exit__(self, *args):
        for number, handler in self.previous.items():
            signal.signal(number, handler)


def verify_retry(*, verify_sources=True, digest_fn=sha):
    if verify_sources:
        launch.verify_prepared()
    lineage = read(HERE / 'AMENDMENT_LINEAGE.json')
    require(lineage['validation_attempt_ordinal'] == 3 and lineage['cap_seconds'] == CAP_SECONDS,
            'Amendment validation identity/budget differs')
    require(lineage['prior_final_counts_known'] is False, 'Prior uncertainty must remain recorded')
    for name, expected in lineage['prior_files_sha256'].items():
        path = (launch.REPO / name).resolve()
        require(path.is_relative_to(launch.REPO.resolve()) and digest_fn(path) == expected,
                'Prior evidence changed: ' + name)
    manifest = dict(attempt_ordinal=3, cap_seconds=CAP_SECONDS,
        original_prepared_sha256=digest_fn(HERE / 'PREPARED.json'),
        original_protocol_sha256=digest_fn(HERE / 'protocol.json'),
        archive_directory=lineage['prior_package'],
        lineage_sha256=digest_fn(HERE / 'AMENDMENT_LINEAGE.json'))
    return manifest, lineage


def extra_process_check(proc_root=Path('/proc')):
    """Same-user exact validator/retry paths; a denied inspection stops launch."""
    prior = ROOT.parent / 'central_quartile_training_v1'
    targets = {(ROOT / 'validate_runtime.py').resolve(), Path(__file__).resolve(),
        (prior / 'validate_runtime.py').resolve(),
        (prior / 'validation_retry_20261028_v1/retry.py').resolve()}
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid() or not (process / 'comm').read_text().strip().startswith(('python', 'pypy')):
                continue
            for raw in (process / 'cmdline').read_bytes().split(b'\0')[1:]:
                path = Path(os.fsdecode(raw))
                if path.name not in ('validate_runtime.py', 'retry.py', 'validation_supervisor.py'):
                    continue
                if not path.is_absolute():
                    path = (process / 'cwd').resolve(strict=True) / path
                if path.resolve() in targets:
                    found.append(int(process.name))
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError('Scoped validation process inspection denied; no launch') from exc
    require(not found, 'Existing validator/retry process: ' + str(found))


def reserve(root, retry_root, manifest, receipt):
    output = Path(root) / 'runtime_validation_v1'
    require(not output.exists(), 'Validation output already exists; inspect it, do not retry')
    record = dict(attempt_ordinal=3, supervisor_pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=manifest['original_protocol_sha256'], amendment_prepared_sha256=manifest['original_prepared_sha256'],
        prior_archive=manifest['archive_directory'], prior_final_counts_known=False,
        prior_elapsed_seconds=None, prior_elapsed_seconds_lower_bound=receipt['prior_supervisor_elapsed_seconds_lower_bound'],
        worker_cap_seconds=CAP_SECONDS, production_attempt_reserved=False, automatic_retry=False)
    launch.save_new(Path(retry_root) / 'VALIDATION_ATTEMPT_3.json', record)
    output.mkdir(exist_ok=False)
    atomic_json(output / 'RESERVATION.json', record)
    return output


def supervise(command, output, deadline, signals, env=None, popen=subprocess.Popen):
    started = time.monotonic()
    outcome = dict(exit_code=None, timed_out=False, interrupted=False, error=None)
    process = None
    last_progress = None
    try:
        with (output / 'worker.log').open('x', encoding='utf-8') as log:
            if not signals.received and time.monotonic() < deadline:
                process = popen(command, cwd=output, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=(os.name != 'nt'))
            while process is not None and process.poll() is None:
                if signals.received or time.monotonic() >= deadline:
                    break
                try:
                    process.wait(timeout=min(.25, max(.001, deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
                progress, _ = launch.read_report(output / 'progress.json')
                if progress and progress.get('utc') != last_progress:
                    last_progress = progress.get('utc')
                    print(json.dumps({k: progress.get(k) for k in ('utc', 'stage', 'elapsed_seconds')}), flush=True)
    except BaseException as exc:
        outcome['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        outcome['interrupted'] = bool(signals.received)
        outcome['timed_out'] = time.monotonic() >= deadline
        if process is not None:
            try:
                # Only this supervisor's owned child group. Record-only handlers
                # remain installed during kill/wait, including repeated Ctrl-C.
                launch.prior.terminate_owned(process)
                outcome['exit_code'] = process.wait(timeout=10)
            except BaseException as exc:
                outcome['cleanup_error'] = type(exc).__name__ + ': ' + str(exc)
        outcome['elapsed_seconds'] = time.monotonic() - started
        outcome['signals'] = list(signals.received)
        atomic_json(output / 'CHILD_OUTCOME.json', outcome)
    return outcome


def postcheck(manifest, deadline, signals):
    """Chunked identity check, cancellable before a full repeated archive read."""
    def digest(path):
        if signals.received:
            raise InterruptedError('Postcheck interrupted; identity remains unverified')
        if time.monotonic() >= deadline:
            raise TimeoutError('Validation deadline reached during postcheck')
        value = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while True:
                if signals.received:
                    raise InterruptedError('Postcheck interrupted; identity remains unverified')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Validation deadline reached during postcheck')
                chunk = stream.read(1024*1024)
                if not chunk:
                    return value.hexdigest()
                value.update(chunk)
    require(digest(ROOT / 'PREPARED.json') == manifest['original_prepared_sha256'], 'Prepared seal changed')
    frozen = read(ROOT / 'PREPARED.json')
    for base, entries in ((ROOT, frozen['files']), (ROOT.parent, frozen['dependencies']),
                          (launch.REPO, frozen.get('inputs', {}))):
        for name, expected in entries.items():
            path = (base / name).resolve()
            require(path.is_relative_to(base.resolve()) and digest(path) == expected, 'Frozen identity changed: ' + name)
    fresh, _ = verify_retry(verify_sources=False, digest_fn=digest)
    require(fresh == manifest, 'Retry manifest changed since preflight/reservation')
    return True


def finalize(output, outcome, manifest, receipt, deadline, signals, check=postcheck):
    report, report_error = launch.read_report(output / 'report.json')
    progress, progress_error = launch.read_report(output / 'progress.json')
    result = dict(**outcome, status='child_finished_postcheck_pending', attempt_ordinal=3,
        protocol_sha256=manifest['original_protocol_sha256'], prior_archive=manifest['archive_directory'],
        prepared_identity_unchanged=None, passed=False, training_hold_remains=True,
        production_attempt_reserved=False, no_automatic_retry=True, worker_status=report.get('status') if report else None,
        report_sha256=sha(output / 'report.json') if report else None, report_read_error=report_error,
        progress_read_error=progress_error, accounting=report if report else progress,
        accounting_scope='terminal_report' if report else 'last_progress_lower_bound' if progress else 'unavailable',
        execution_counts_known=bool(report and report.get('execution_counts_known')
            and not outcome['interrupted'] and not outcome['timed_out'] and not outcome.get('cleanup_error')),
        cumulative_execution_counts_known=False, cumulative_elapsed_seconds=None,
        prior_accounting=receipt['prior_attempts'], prior_accounting_scope='attempt_1_unknown_attempt_2_terminal_known',
        cumulative_elapsed_seconds_lower_bound=receipt['prior_supervisor_elapsed_seconds_lower_bound'] + outcome['elapsed_seconds'])
    # Save the owned-child outcome BEFORE any potentially expensive postcheck.
    atomic_json(output / 'RESULT.json', result)
    try:
        result['prepared_identity_unchanged'] = check(manifest, deadline, signals)
    except BaseException as exc:
        result['postcheck_error'] = type(exc).__name__ + ': ' + str(exc)
    result.update(status='child_cleanup_failed_liveness_uncertain' if outcome.get('cleanup_error') else 'worker_finished',
        signals=list(signals.received),
        interrupted=outcome['interrupted'] or bool(signals.received),
        timed_out=outcome['timed_out'] or time.monotonic() >= deadline)
    result['passed'] = bool(outcome['exit_code'] == 0 and result['prepared_identity_unchanged'] is True
        and not result['interrupted'] and not result['timed_out'] and not outcome.get('cleanup_error') and not outcome.get('error')
        and report and report.get('status') == 'passed_synthetic_runtime_validation')
    atomic_json(output / 'RESULT.json', result)
    return result


def main(protocol=None, prepared=None, *, execute=True):
    manifest, receipt = verify_retry()
    current_protocol, current_prepared = launch.verify_prepared()
    if protocol is not None or prepared is not None:
        require(protocol == current_protocol and prepared == current_prepared, 'Caller preparation differs')
    launch.verify_inputs()
    require(not launch.EXECUTION_ENABLED and not (ROOT / 'ATTEMPT.json').exists(), 'Production hold/reservation differs')
    if not execute:
        print(json.dumps(dict(status='retry_preflight_passed', attempt_reserved=False,
            target_runtime_tested=False, training_hold_remains=True, cap_seconds=CAP_SECONDS)))
        return 0
    require(sys.platform == 'linux' and Path(sys.prefix).resolve() == Path(launch.EXPECTED_ENV).resolve(),
            'Use the existing isolated Ubuntu runtime; nothing reserved')
    require('include-system-site-packages = false' in (Path(sys.prefix) / 'pyvenv.cfg').read_text(),
            'Runtime must not inherit packages')
    launch.scoped_process_check()
    extra_process_check()
    with Signals() as signals:
        output = reserve(ROOT, HERE, manifest, receipt)
        started = time.monotonic()
        deadline = started + CAP_SECONDS
        outcome = supervise([sys.executable, '-B', str(ROOT / 'validate_runtime.py'), '--worker', str(output)],
            output, deadline, signals, launch.prior.child_environment())
        result = finalize(output, outcome, manifest, receipt, deadline, signals)
        result['elapsed_seconds_including_postcheck'] = time.monotonic() - started
        atomic_json(output / 'RESULT.json', result)
        atomic_json(HERE / 'VALIDATION_RESULT_3.json', result)
        print(json.dumps({k: result.get(k) for k in ('status', 'attempt_ordinal', 'passed', 'worker_status',
            'elapsed_seconds_including_postcheck', 'interrupted', 'timed_out', 'postcheck_error',
            'execution_counts_known', 'cumulative_execution_counts_known', 'training_hold_remains')}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--execute-validation', action='store_true')
    args = parser.parse_args()
    raise SystemExit(main(execute=args.execute_validation))
