"""One approved continuation; the original scientific protocol stays sealed."""
from pathlib import Path
from common import read, sha, require, module
import recovery_inputs as recovery

ALLOWED_CHANGES = {'AUTHORIZATION.json','authorization.py','launch.py','train_worker.py','evaluate_worker.py','README.md',
                   'test_train_worker.py','test_evaluate_worker.py'}

def verify(root, protocol):
    root = Path(root).resolve()
    a = read(root/'AUTHORIZATION.json')
    r = recovery.verify_prior()
    require(root.name == a['release_directory'] == 'central_quartile_training_v1_continuation1'
        and a['schema'] == 'bca.central_quartile.continuation_authorization.v1', 'Continuation authorization identity differs')
    require(a['execution_authorized'] is True and type(a['approved_attempts']) is int
        and a['approved_attempts'] == 1 and a['automatic_restart'] is False, 'One continuation only is authorized')
    require(a['approval']['user_message_id'] == 'Sentinel_431aa4e2b3ec819187e224f632c9e820'
        and a['approval']['response'] == 'yes be quick and lets get it finished', 'Explicit continuation approval differs')
    require(a['recovery_sha256'] == sha(root/'RECOVERY.json')
        and a['prior_prepared_sha256'] == r['prior_prepared_sha256']
        and a['prior_terminal_sha256'] == r['prior_terminal_sha256']
        and a['protocol_sha256'] == r['protocol_sha256'] == sha(root/'protocol.json')
        and protocol == read(root/'protocol.json'), 'Authorization evidence hash differs')
    old = recovery.prior_root()
    validated = module('_continuation_original_authorization',old/'authorization.py').verify(old,protocol)
    for name, expected in read(old/'PREPARED.json')['files'].items():
        require(sha(old/name) == expected, 'Frozen original source changed: '+name)
        if name not in ALLOWED_CHANGES:
            require(sha(root/name) == expected, 'Unapproved continuation source change: '+name)
    return {'execution_authorized':True,'approved_attempts':1,'automatic_restart':False,
        'authorization_sha256':sha(root/'AUTHORIZATION.json'),'recovery_sha256':sha(root/'RECOVERY.json'),
        'prior_prepared_sha256':r['prior_prepared_sha256'],'prior_terminal_sha256':r['prior_terminal_sha256'],
        'validated_runtime':validated,'new_work_caps':r['new_work_caps'],
        'maximum_recovery_seconds':r['maximum_recovery_seconds'],
        'original_deadline_est':r['original_deadline_est'],
        'prior_counts_known':False,'repeat_work_lower_bound':r['repeat_work_lower_bound']}
