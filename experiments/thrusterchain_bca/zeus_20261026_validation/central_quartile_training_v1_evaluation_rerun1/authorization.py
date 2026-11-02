"""One approved evaluation-only rerun; the scientific protocol and all earlier evidence stay sealed."""
from pathlib import Path
from common import read, sha, require, module
import rerun_inputs as rerun

SCHEMA = 'bca.central_quartile.evaluation_rerun_authorization.v1'
MATCH_RULE = 'exact_or_near_exact_1e-6'
# Byte-identical to the sealed continuation1 package: every model, data, evaluator,
# exact-state comparator, scientific protocol, original-study pin and their tests.
UNCHANGED = {'common.py', 'recovery_inputs.py', 'evaluation.py', 'runtime.py', 'data.py', 'design.py',
             'state_exact.py', 'gradient_audit.py', 'train_worker.py', 'protocol.json', 'PROTOCOL.md',
             'DATA_PLAN.json', 'RECOVERY.json', 'test_evaluation.py', 'test_state_exact.py', 'test_runtime.py',
             'test_data.py', 'test_gradient_audit.py'}
APPROVAL = {
    'initial_request': 'Rerun eval',
    'questions': [
        {'question': 'Approve one evaluation-only rerun? No training: it re-evaluates the 72 existing snapshots '
                     'with the frozen evaluation code, plus a fix so a briefly locked file can\'t crash it. One '
                     'attempt, launched by you, capped at 2 hours. It runs past the original Oct 29, 5:31 a.m. EST '
                     'deadline, using unspent time from the 12-hour budget, and is disclosed as a second continuation.',
         'answer': 'Yes, approve (Recommended)'},
        {'question': 'The GPU may not reproduce the 143 earlier results down to the last bit. What should count as '
                     'reproduced?',
         'answer': 'Exact or near-exact (Recommended)',
         'answer_description': 'Byte-identical; or, failing that, every number within |new-old| <= 1e-6*(1+|old|) '
                               'and every count, label and string identical. Anything else stops the run as '
                               'incomplete. Which one applied is disclosed.'}],
    'answered_by_utc': '2026-10-29T17:00:19Z',
    'channel': 'Claude Code desktop chat; the user answered two structured questions'}


def verify(root, protocol):
    root = Path(root).resolve()
    a = read(root/'AUTHORIZATION.json')
    require(a['schema'] == SCHEMA and a['release_directory'] == root.name == rerun.RELEASE,
            'Rerun authorization identity differs')
    require(a['execution_authorized'] is True and type(a['approved_attempts']) is int and a['approved_attempts'] == 1
            and a['automatic_restart'] is False and a['training_authorized'] is False
            and a['original_calendar_deadline_waived'] is True and a['seconds_cap'] == rerun.RERUN_SECONDS_CAP
            and a['match_rule'] == MATCH_RULE, 'Only one evaluation-only rerun is authorized')
    require(a['approval'] == APPROVAL, 'Explicit rerun approval differs')
    lineage = rerun.verify_continuation()
    require(a['rerun_inputs_sha256'] == sha(root/'RERUN_INPUTS.json')
            and a['continuation_prepared_sha256'] == lineage['continuation_prepared_sha256']
            and a['continuation_terminal_sha256'] == lineage['continuation_terminal_sha256']
            and a['protocol_sha256'] == lineage['protocol_sha256'] == sha(root/'protocol.json')
            and protocol == read(root/'protocol.json'), 'Rerun authorization evidence hash differs')
    croot = rerun.continuation_root()
    continuation = module('_rerun_continuation_authorization', croot/'authorization.py').verify(croot, protocol)
    sealed = read(croot/'PREPARED.json')['files']
    for name, expected in sealed.items():
        require(sha(croot/name) == expected, 'Sealed continuation source changed: '+name)
    for name in sorted(UNCHANGED):
        require(name in sealed and sha(root/name) == sealed[name], 'Unapproved rerun source change: '+name)
    return {'execution_authorized': True, 'approved_attempts': 1, 'automatic_restart': False,
            'training_authorized': False, 'original_calendar_deadline_waived': True,
            'seconds_cap': rerun.RERUN_SECONDS_CAP, 'match_rule': MATCH_RULE,
            'authorization_sha256': sha(root/'AUTHORIZATION.json'),
            'rerun_inputs_sha256': sha(root/'RERUN_INPUTS.json'),
            'continuation_prepared_sha256': lineage['continuation_prepared_sha256'],
            'continuation_terminal_sha256': lineage['continuation_terminal_sha256'],
            'continuation_authorization': continuation}
