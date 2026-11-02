"""Read-only authorization of one separately sealed production release."""
from pathlib import Path
from common import read, sha, require

VALIDATED_PACKAGE = 'central_quartile_training_v1_amendment1'
VALIDATED_SEAL = '122503422dd51b0717c34597311c9f1bc0a0065f2332480ef6eeadd439198763'
VALIDATION_RESULT = 'e7d3f03d5770d17ea5109d1c39c2ae55927782aa1d7dbccb39577f3d46c6b48f'
VALIDATION_REPORT = '490cc4e6f63ded5e1e8d6e1448d4f06e93733d64d0f5d1d42c381b2e04fb3553'
PROTOCOL = '95a9dbe042bbe82e868cceb02810e7f1ffa71c0949925996ea17ea3f54d4cdfd'
# Only release entry/approval fixtures and the current README differ. Every
# model, data, evaluator, exact-state comparator and scientific input is copied
# byte-for-byte from the validated package.
ALLOWED_RELEASE_CHANGES = {'launch.py', 'README.md', 'test_launch.py', 'test_amendment.py'}


def verify(root, protocol):
    root = Path(root).resolve()
    a = read(root/'AUTHORIZATION.json')
    require(a['schema'] == 'bca.central_quartile.production_authorization.v1'
        and a['release_directory'] == root.name == 'central_quartile_training_v1_authorized1',
        'Authorization release identity differs')
    require(a['execution_authorized'] is True and type(a['approved_attempts']) is int
        and a['approved_attempts'] == 1 and a['automatic_restart'] is False
        and a['resume_authorized'] is False, 'One-attempt authorization differs')
    require(a['approval']['user_message_id'] == 'Sentinel_2f0ccb63d1d88191ad138e242854c996'
        and a['approval']['utc'] == '2026-10-28T21:49:53+00:00'
        and a['approval']['response'] == 'Yes', 'Explicit user approval differs')
    require(a['validated_package'] == VALIDATED_PACKAGE, 'Validated package differs')
    prior = root.parent/VALIDATED_PACKAGE
    for field, name, expected in (
        ('validated_prepared_sha256', 'PREPARED.json', VALIDATED_SEAL),
        ('validation_result_sha256', 'VALIDATION_RESULT_3.json', VALIDATION_RESULT),
        ('validation_report_sha256', 'runtime_validation_v1/report.json', VALIDATION_REPORT)):
        require(a[field] == expected == sha(prior/name), 'Validated evidence changed: '+name)
    require(a['protocol_sha256'] == PROTOCOL == sha(root/'protocol.json')
        and protocol == read(root/'protocol.json'), 'Authorized scientific protocol changed')
    require(a['job_counts'] == {'warmups':9,'branches':27,'evaluations':1}, 'Approved job matrix differs')
    require(all(type(value) is int and protocol.get(key) == value for key,value in a['budget'].items())
        and a['budget'] == {'total_timeout_seconds':43200,'total_critic_updates':360000,
            'total_actor_updates':180000,'total_auxiliary_updates':360000,'total_calibrations':36,
            'total_posterior_draws':36000,'evaluation_episodes':4608,'simulator_transition_cap':1975304},
        'Approved budget differs')
    result = read(prior/'VALIDATION_RESULT_3.json')
    require(result['passed'] is True and result['exit_code'] == 0 and result['attempt_ordinal'] == 3
        and result['prepared_identity_unchanged'] is True and result['execution_counts_known'] is True
        and not result['timed_out'] and not result['interrupted'] and result['error'] is None
        and result['report_sha256'] == VALIDATION_REPORT, 'Target validation is not eligible')
    seal = read(prior/'PREPARED.json')
    for name, expected in seal['files'].items():
        require(sha(prior/name) == expected, 'Validated package changed: '+name)
        if name not in ALLOWED_RELEASE_CHANGES:
            require(sha(root/name) == expected, 'Validated scientific implementation changed: '+name)
    for folder in (root.parent/'central_quartile_training_v1', prior):
        require(not any((folder/name).exists() for name in ('ATTEMPT.json','run','RUN_RESULT.json')),
            'Historical central-quartile production evidence exists; inspect before launch')
    return {'execution_authorized':True,'approved_attempts':1,
        'authorization_sha256':sha(root/'AUTHORIZATION.json'),
        'validated_prepared_sha256':VALIDATED_SEAL,'validation_report_sha256':VALIDATION_REPORT,
        'scientific_protocol_sha256':PROTOCOL,'total_seconds_cap':43200,
        'automatic_restart':False,'resume_authorized':False}
