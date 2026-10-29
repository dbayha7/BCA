"""One manually requested, capped synthetic target-runtime check; no production run."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import warnings

from common import HERE, read, sha, atomic_json, module, require


def reserve_validation(root, protocol_sha, supervisor_pid):
    output = Path(root) / 'runtime_validation_v1'
    output.mkdir(exist_ok=False)
    atomic_json(output / 'RESERVATION.json', dict(protocol_sha256=protocol_sha,
        supervisor_pid=supervisor_pid, utc=datetime.now(timezone.utc).isoformat(),
        production_attempt_reserved=False, automatic_retry=False))
    return output


def counted_environment(base, ledger, transition_cap=512, reset_cap=128, progress=None):
    """Instrument the real subclass without changing physics/reset semantics."""
    class CountedEnvironment(base):
        def reset(self, *args, **kwargs):
            require(ledger['resets_attempted'] < reset_cap, 'Checker reset cap reached')
            ledger['resets_attempted'] += 1
            if progress is not None: progress()
            result = super().reset(*args, **kwargs)
            ledger['resets_completed'] += 1
            if progress is not None: progress()
            return result
        def step(self, *args, **kwargs):
            require(ledger['transitions_attempted'] < transition_cap, 'Checker transition cap reached')
            ledger['transitions_attempted'] += 1
            if progress is not None: progress()
            result = super().step(*args, **kwargs)
            ledger['transitions_completed'] += 1
            if progress is not None: progress()
            return result
    return CountedEnvironment


def worker(output):
    import launch
    p, _ = launch.verify_prepared()
    launch.verify_inputs()
    output = Path(output).resolve()
    require(output == (HERE / 'runtime_validation_v1').resolve(), 'Unexpected validation output')
    reservation = read(output / 'RESERVATION.json')
    require(reservation['supervisor_pid'] == os.getppid()
            and reservation['protocol_sha256'] == sha(HERE / 'protocol.json')
            and not (output / 'report.json').exists(), 'Validation worker is not owned or is already complete')
    launch.save_new(output/'WORKER_ENTRY.json', dict(pid=os.getpid(), parent_pid=os.getppid(),
        protocol_sha256=reservation['protocol_sha256'], utc=datetime.now(timezone.utc).isoformat()))
    started = time.monotonic()
    ledger = dict(resets_attempted=0, resets_completed=0, transitions_attempted=0, transitions_completed=0)
    result = dict(status='running_synthetic_runtime_validation', production_data_rows=0,
        pretrained_checkpoint_restores=0, production_attempt_reserved=False,
        protocol_sha256=sha(HERE/'protocol.json'), gymnasium_ledger=ledger,
        scientific_execution_authorized=False, automatic_retry=False)
    model_started = False
    initialization_started = False
    stage = 'runtime_setup'
    def journal(snapshot=None):
        if snapshot is not None:
            result['initialization_progress' if stage == 'read_only_source_initialization' else 'model_progress'] = snapshot
        atomic_json(output/'progress.json', dict(status='running', stage=stage,
            elapsed_seconds=time.monotonic()-started, utc=datetime.now(timezone.utc).isoformat(),
            model_progress=result.get('model_progress'), gymnasium_ledger=dict(ledger),
            initialization_progress=result.get('initialization_progress'),
            production_data_rows=result.get('initialization_check',{}).get('counts',{}).get('source_tuples_completed',0),
            real_data_optimizer_updates=0, production_attempt_reserved=False,
            counts_scope='Last durable snapshot; a killed worker may have later unrecorded completions'))
    try:
        journal()
        require(sys.version_info[:2] == (3,12), 'Expected frozen Python 3.12 runtime')
        require(importlib.metadata.version('gymnasium') == '1.0.0', 'Expected Gymnasium 1.0.0')
        stage = 'real_model_synthetic_check'
        journal()
        runtime = module('_central_validation_runtime', HERE/'runtime.py')
        model_started = True
        check = runtime.synthetic_model_check(p, progress=journal)
        result['model_check'] = check
        require(check['status'] == 'passed_synthetic_runtime_check', 'Synthetic model check failed; see model_check')
        stage = 'read_only_source_initialization'
        journal()
        initialization_started = True
        initialization=runtime.real_source_initialization_check(p,deadline=started+900,progress=journal)
        result['initialization_check']=initialization
        result['production_data_rows']=initialization['counts']['source_tuples_completed']
        result['real_data_optimizer_updates']=0
        require(initialization['status']=='passed_real_source_initialization_check',
                'Real-source initialization check failed; see initialization_check')
        stage = 'full_gymnasium_checker'
        journal(check)
        loader = module('_central_validation_core_loader', HERE.parent/'conditional_ood_gate_central_quartile_v1/worker.py')
        core, source = loader._core()
        from gymnasium.utils.env_checker import check_env
        env = counted_environment(core.ThrusterChainEnv, ledger, progress=journal)(stall=True, stall_slope=1.5,
            start='recorded', observation_mode='time_aware', task_semantics='finite_horizon')
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                check_env(env)  # No skip flags. Warnings are evidence, not silently discarded.
            result['gymnasium'] = dict(passed=True, warnings=[str(w.message) for w in recorded], source=source)
        finally:
            env.close()
        require(time.monotonic()-started < 900, 'Synthetic validation exceeded global deadline')
        launch.verify_prepared()
        result['status'] = 'passed_synthetic_runtime_validation'
    except BaseException as exc:
        result.update(status='failed_synthetic_runtime_validation', error=type(exc).__name__+': '+str(exc))
    result.update(elapsed_seconds=time.monotonic()-started,
        execution_counts_known=ledger['transitions_attempted']==ledger['transitions_completed']
            and ledger['resets_attempted']==ledger['resets_completed']
            and (not model_started or bool(result.get('model_check',{}).get('execution_counts_known',
                result.get('model_check',{}).get('status')=='passed_synthetic_runtime_check')))
            and (not initialization_started or bool(result.get('initialization_check',{}).get('execution_counts_known',False))))
    result['stage'] = stage
    atomic_json(output/'report.json', result)
    return 0 if result['status']=='passed_synthetic_runtime_validation' else 1


def main(protocol, manifest):
    import launch
    require(sys.platform == 'linux' and Path(sys.prefix).resolve()==Path(launch.EXPECTED_ENV).resolve(),
            'Use the existing isolated Ubuntu runtime; nothing reserved')
    require('include-system-site-packages = false' in (Path(sys.prefix)/'pyvenv.cfg').read_text(),
            'Synthetic runtime must not inherit packages')
    current_protocol, current_manifest = launch.verify_prepared()
    launch.verify_inputs()
    require(protocol == current_protocol and manifest == current_manifest,
            'Validation arguments differ from fresh prepared inputs; nothing reserved')
    launch.scoped_process_check()
    output = reserve_validation(HERE, sha(HERE/'protocol.json'), os.getpid())
    started = time.monotonic()
    outcome = dict(exit_code=None, timed_out=False, interrupted=False, error=None, elapsed_seconds=0.)
    old_handlers = {}
    def stop(signum, frame):
        raise KeyboardInterrupt('Validation supervisor signal '+str(signum))
    try:
        for name in ('SIGINT','SIGTERM','SIGHUP'):
            if hasattr(signal,name):
                number=getattr(signal,name)
                old_handlers[number]=signal.getsignal(number)
                signal.signal(number,stop)
        outcome = launch.supervise([sys.executable,'-B',str(HERE/'validate_runtime.py'),'--worker',str(output)],
            output,900,launch.prior.child_environment(),'synthetic_target_validation',deadline=started+900)
    except BaseException as exc:
        outcome.update(error=type(exc).__name__+': '+str(exc), interrupted=isinstance(exc,KeyboardInterrupt),
            elapsed_seconds=time.monotonic()-started)
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number,handler)
    report, error = launch.read_report(output/'report.json')
    progress, progress_error = launch.read_report(output/'progress.json')
    unchanged = False
    try:
        _, current = launch.verify_prepared()
        unchanged = current == manifest
    except Exception as exc:
        error = error or str(exc)
    result = dict(**outcome, protocol_sha256=sha(HERE/'protocol.json'), prepared_identity_unchanged=unchanged,
        report_sha256=sha(output/'report.json') if report else None, report_read_error=error,
        worker_status=report.get('status') if report else None, production_attempt_reserved=False,
        last_progress=progress, progress_read_error=progress_error,
        accounting=report if report else progress,
        accounting_scope='terminal_report' if report else 'last_progress_lower_bound' if progress else 'unavailable',
        execution_counts_known=bool(report and report.get('execution_counts_known') and not outcome['timed_out']
                                    and not outcome['interrupted']), training_hold_remains=True,
        no_automatic_retry=True)
    result['passed'] = bool(outcome['exit_code']==0 and unchanged and not outcome['timed_out']
        and not outcome['interrupted'] and report and report.get('status')=='passed_synthetic_runtime_validation')
    atomic_json(output/'RESULT.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',required=True)
    raise SystemExit(worker(parser.parse_args().worker))
