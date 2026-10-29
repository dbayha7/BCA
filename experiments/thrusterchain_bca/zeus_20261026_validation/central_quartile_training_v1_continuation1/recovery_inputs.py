"""Read-only, hash-authenticated mapping of original and continuation artifacts."""
from datetime import datetime
from pathlib import Path
from common import HERE, read, digest, require, sha

PRIOR = 'central_quartile_training_v1_authorized1'
NEW_JOBS = ['seed2_hole/bca', 'seed2_random/distance', 'seed2_random/native', 'seed2_random/bca']

def recovery():
    r = read(HERE/'RECOVERY.json')
    require(r['schema'] == 'bca.central_quartile.continuation_plan.v1'
            and r['prior_package'] == PRIOR and r['new_training_jobs'] == NEW_JOBS,
            'Continuation lineage or remaining jobs changed')
    require(len(r['prior_completed_jobs']) == len(set(r['prior_completed_jobs'])) == 32
            and not set(r['prior_completed_jobs']).intersection(NEW_JOBS), 'Completed/pending jobs overlap')
    require(r['maximum_recovery_seconds'] == 14400
            and r['original_deadline_est'] == '2026-10-29T05:31:39.070521-05:00'
            and r['prior_elapsed_seconds'] == 16451.765985543
            and r['remaining_consumed_time_allowance_seconds'] == 26748.234014457,
            'Approved deadline or consumed-time allowance changed')
    require(r['resume_checkpoint_step'] == 16000 and r['resume_job'] == NEW_JOBS[0]
            and r['prior_execution_counts_known'] is False and r['no_automatic_retry'] is True,
            'Continuation checkpoint/accounting rules changed')
    require(r['new_work_caps'] == {'critic':34000,'actor':17000,'auxiliary':34000,
            'calibrations':3,'posterior_draws':3000,'gradient_snapshots':7,'gradient_probe_rows':5376,
            'evaluation_episodes':4608,'simulator_transitions':1975304}, 'Continuation resource caps changed')
    return r

def prior_root():
    return (HERE.parent/PRIOR).resolve()

def authenticate_prior_file(relative):
    name = Path(relative)
    root = prior_root()
    path = (root/name).resolve()
    require(not name.is_absolute() and path.is_relative_to(root), 'Prior artifact path escaped')
    key = name.as_posix()
    expected = recovery()['prior_files_sha256'].get(key)
    require(isinstance(expected,str) and len(expected)==64, 'Unpinned prior artifact: '+key)
    raw = path.read_bytes()
    require(digest(raw) == expected, 'Prior artifact changed: '+key)
    return raw

def job_directory(cell, stage):
    job = cell+'/'+stage
    r = recovery()
    require(job in r['prior_completed_jobs'] or job in NEW_JOBS, 'Unknown continuation job: '+job)
    return (HERE/'run'/job if job in NEW_JOBS else prior_root()/'run'/job)

def verify_prior(all_artifacts=False):
    import json
    r = recovery()
    names = r['prior_files_sha256'] if all_artifacts else (
        'PREPARED.json','RUN_RESULT.json','ATTEMPT.json','AUTHORIZATION.json',
        'protocol.json','run/PAIRED_INITIALIZATION.json',r['resume_manifest'])
    for name in names:
        authenticate_prior_file(name)
    terminal = json.loads(authenticate_prior_file('RUN_RESULT.json'))
    require(terminal['status'] == 'stopped_after_incomplete_job'
            and terminal['prepared_identity_unchanged'] is True
            and terminal['execution_counts_known'] is False
            and terminal['elapsed_seconds'] == r['prior_elapsed_seconds'], 'Original terminal receipt differs')
    completed = [j['job'] for j in terminal['jobs'] if j.get('completed')]
    require(completed == r['prior_completed_jobs'], 'Original completed-job inventory differs')
    require(len(terminal['jobs']) == 33 and terminal['jobs'][-1]['job'] == r['resume_job']
            and terminal['jobs'][-1]['interrupted'] is True
            and terminal['jobs'][-1]['exit_code'] == -9, 'Interrupted owned-worker termination differs')
    require(sha(HERE/'protocol.json') == r['protocol_sha256'], 'Scientific protocol differs')
    return r

def recovery_seconds(now_epoch):
    r = recovery()
    seconds = min(r['maximum_recovery_seconds'], r['remaining_consumed_time_allowance_seconds'],
                  datetime.fromisoformat(r['original_deadline_est']).timestamp()-now_epoch)
    require(seconds > 0, 'Original study deadline has expired; no continuation authorized')
    return seconds
