"""Read-only zero-simulation parity audit of the completed v5 evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np


def main():
    root=Path(__file__).resolve().parent/'reranker_run_01'
    result=json.loads((root/'RERANKER_RESULT.json').read_bytes())
    assert result['status']=='completed_frozen_controller_evaluation'
    max_abs=0.;checks=0;exact=0
    for case in result['cases']:
        for context,cell in case['contexts'].items():
            path=root/cell['telemetry_artifact']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==cell['telemetry_sha256']
            with np.load(path,allow_pickle=False) as z:
                returns=np.asarray(z['returns']);qpi=np.asarray(z['first_qpi'])
                first=np.asarray(z['first_actions']);actual=np.asarray(z['actions'])[:,:,0,:]
                assert returns.shape==(5,16) and qpi.shape==(16,5) and first.shape==actual.shape==(5,16,4)
                assert np.array_equal(first,actual),'planned and executed initial actions differ'
                difference=np.abs(qpi[:,0]-returns[0])
                max_abs=max(max_abs,float(difference.max()))
                checks+=len(difference);exact+=int(np.count_nonzero(difference==0))
    assert checks==192 and max_abs<=1e-10, f'original actor Qpi/return mismatch: {max_abs}'
    print(json.dumps({'status':'passed','episode_checks':checks,'exact_equal':exact,
                      'maximum_absolute_difference':max_abs,'absolute_tolerance':1e-10,
                      'planned_actions_match_executed':True,'simulation_transitions':0},sort_keys=True))


if __name__=='__main__':main()
