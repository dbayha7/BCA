"""Pin every continuation1 file by hash once; reads bytes only, parses no result contents."""
from datetime import datetime, timezone, timedelta
import json
from common import HERE, read, sha, require
import recovery_inputs
import rerun_inputs as rerun


def prepare():
    for name in ('RERUN_INPUTS.json', 'PREPARED.json', 'ATTEMPT.json', 'run', 'RUN_RESULT.json'):
        require(not (HERE/name).exists(), 'Existing rerun preparation preserved: '+name)
    root = rerun.continuation_root()
    pins = {path.relative_to(root).as_posix(): sha(path)
            for path in sorted(root.rglob('*')) if path.is_file()}
    require(not any('.tmp.' in name or '__pycache__' in name for name in pins),
            'Continuation contains temporary or cache files; inspect before pinning')
    terminal = read(root/'RUN_RESULT.json')
    record = {'schema': 'bca.central_quartile.evaluation_rerun_inputs.v1',
        'release_directory': rerun.RELEASE, 'continuation_package': rerun.CONTINUATION,
        'continuation_jobs': rerun.CONTINUATION_JOBS, 'new_context': rerun.NEW_CONTEXT,
        'recorded_contexts': 143,
        'original_elapsed_seconds': recovery_inputs.recovery()['prior_elapsed_seconds'],
        'continuation_elapsed_seconds': terminal['elapsed_seconds'],
        'original_allowance_seconds': rerun.ORIGINAL_ALLOWANCE_SECONDS,
        'rerun_seconds_cap': rerun.RERUN_SECONDS_CAP,
        'protocol_sha256': sha(HERE/'protocol.json'),
        'continuation_files_sha256': pins,
        'recorded_at_est': datetime.now(timezone(timedelta(hours=-5))).isoformat(),
        'scientific_execution_during_preparation': False,
        'recorded_result_contents_parsed': False}
    with (HERE/'RERUN_INPUTS.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n')
    rerun.verify_continuation(all_artifacts=True)
    return {'status': 'continuation_pinned', 'pinned_files': len(pins),
            'rerun_inputs_sha256': sha(HERE/'RERUN_INPUTS.json')}


if __name__ == '__main__': print(json.dumps(prepare()))
