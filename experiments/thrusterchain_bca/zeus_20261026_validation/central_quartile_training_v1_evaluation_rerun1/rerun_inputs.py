"""Read-only, hash-authenticated access to continuation1 for the evaluation-only rerun."""
import json
from pathlib import Path
from common import HERE, read, digest, require, sha
import recovery_inputs

CONTINUATION = 'central_quartile_training_v1_continuation1'
RELEASE = 'central_quartile_training_v1_evaluation_rerun1'
CONTINUATION_JOBS = ['seed2_hole/bca', 'seed2_random/distance', 'seed2_random/native', 'seed2_random/bca']
NEW_CONTEXT = 'seed2_random__distance__20000/strong_current'
BANK_KEYS = ['ordinary_bank', 'strong_current_bank']
ORIGINAL_ALLOWANCE_SECONDS = 43200
RERUN_SECONDS_CAP = 7200
KEY_FILES = ('PREPARED.json', 'RUN_RESULT.json', 'ATTEMPT.json', 'AUTHORIZATION.json', 'RUN_AUTHORIZATION.json',
             'DEADLINE.json', 'protocol.json', 'RECOVERY.json', 'run/evaluation/report.json',
             'run/evaluation/SUPERVISOR_RESULT.json')


def plan():
    r = read(HERE/'RERUN_INPUTS.json')
    require(r['schema'] == 'bca.central_quartile.evaluation_rerun_inputs.v1'
            and r['continuation_package'] == CONTINUATION and r['release_directory'] == RELEASE
            and r['continuation_jobs'] == CONTINUATION_JOBS == recovery_inputs.NEW_JOBS
            and r['new_context'] == NEW_CONTEXT and r['recorded_contexts'] == 143,
            'Rerun lineage or recorded evaluation scope changed')
    require(r['original_elapsed_seconds'] == recovery_inputs.recovery()['prior_elapsed_seconds'] == 16451.765985543
            and r['continuation_elapsed_seconds'] == 2452.264564082
            and r['original_allowance_seconds'] == ORIGINAL_ALLOWANCE_SECONDS
            and r['rerun_seconds_cap'] == RERUN_SECONDS_CAP, 'Rerun time allowance changed')
    pins = r['continuation_files_sha256']
    require(all(name in pins for name in KEY_FILES)
            and all(isinstance(v, str) and len(v) == 64 for v in pins.values()), 'Continuation pins incomplete')
    return r


def continuation_root():
    return (HERE.parent/CONTINUATION).resolve()


def authenticate_continuation_file(relative):
    name = Path(relative)
    root = continuation_root()
    path = (root/name).resolve()
    require(not name.is_absolute() and path.is_relative_to(root), 'Continuation artifact path escaped')
    key = name.as_posix()
    expected = plan()['continuation_files_sha256'].get(key)
    require(isinstance(expected, str) and len(expected) == 64, 'Unpinned continuation artifact: '+key)
    raw = path.read_bytes()
    require(digest(raw) == expected, 'Continuation artifact changed: '+key)
    return raw


def job_directory(cell, stage):
    job = cell+'/'+stage
    require(job in recovery_inputs.recovery()['prior_completed_jobs'] or job in CONTINUATION_JOBS,
            'Unknown evaluation job: '+job)
    return (continuation_root()/'run'/job if job in CONTINUATION_JOBS
            else recovery_inputs.prior_root()/'run'/job)


def recorded_evaluation():
    """The failed first evaluation's receipt; result contents are compared only by the gate."""
    report = json.loads(authenticate_continuation_file('run/evaluation/report.json'))
    files = report['snapshot_results']
    contexts = sorted(k for k in files if '/' in k)
    require(report['status'] == 'failed_central_quartile_evaluation'
            and report['decision'] == 'incomplete_no_utility_conclusion'
            and report['phase'] == 'snapshot_inference' and report['completed_snapshot_contexts'] == 143
            and report['error'].startswith('PermissionError: [Errno 13]') and 'progress.json' in report['error'],
            'Recorded first evaluation failure differs')
    require(len(files) == 145 and sorted(k for k in files if '/' not in k) == BANK_KEYS
            and len(contexts) == 143 and NEW_CONTEXT not in files, 'Recorded evaluation inventory differs')
    return report


def recorded_bytes(descriptor):
    require(set(descriptor) == {'path', 'sha256', 'bytes'} and Path(descriptor['path']).name == descriptor['path'],
            'Malformed recorded result descriptor')
    raw = authenticate_continuation_file('run/evaluation/'+descriptor['path'])
    require(digest(raw) == descriptor['sha256'] and len(raw) == descriptor['bytes'],
            'Recorded result bytes differ: '+descriptor['path'])
    return raw


def verify_continuation(all_artifacts=False):
    r = plan()
    for name in (r['continuation_files_sha256'] if all_artifacts else KEY_FILES):
        authenticate_continuation_file(name)
    terminal = json.loads(authenticate_continuation_file('RUN_RESULT.json'))
    jobs = terminal['jobs']
    require(terminal['status'] == 'stopped_after_incomplete_job' and terminal['prepared_identity_unchanged'] is True
            and terminal['decision'] == 'incomplete_no_utility_conclusion'
            and terminal['all_declared_jobs_completed'] is False
            and terminal['cumulative_execution_counts_known'] is False
            and terminal['elapsed_seconds'] == r['continuation_elapsed_seconds'],
            'Continuation terminal receipt differs')
    require([j['job'] for j in jobs] == CONTINUATION_JOBS+['evaluation']
            and all(j['completed'] is True and j['exit_code'] == 0 and j['execution_counts_known'] is True
                    for j in jobs[:4]), 'Continuation training jobs are not all complete')
    evaluation = jobs[4]
    require(evaluation['completed'] is False and evaluation['exit_code'] == 1
            and evaluation['timed_out'] is False and evaluation['interrupted'] is False
            and evaluation['worker_status'] == 'failed_central_quartile_evaluation'
            and evaluation['report_sha256'] == r['continuation_files_sha256']['run/evaluation/report.json'],
            'Continuation evaluation failure receipt differs')
    recorded_evaluation()
    require(terminal['protocol_sha256'] == sha(HERE/'protocol.json') == r['protocol_sha256'],
            'Scientific protocol differs')
    return {'continuation_prepared_sha256': r['continuation_files_sha256']['PREPARED.json'],
            'continuation_terminal_sha256': r['continuation_files_sha256']['RUN_RESULT.json'],
            'continuation_evaluation_report_sha256': r['continuation_files_sha256']['run/evaluation/report.json'],
            'training_physical_accounting': terminal['physical_accounting'],
            'protocol_sha256': r['protocol_sha256']}


def rerun_seconds():
    r = plan()
    remaining = r['original_allowance_seconds']-r['original_elapsed_seconds']-r['continuation_elapsed_seconds']
    seconds = min(r['rerun_seconds_cap'], remaining)
    require(seconds > 0, 'No unspent study allowance remains; no rerun authorized')
    return seconds
