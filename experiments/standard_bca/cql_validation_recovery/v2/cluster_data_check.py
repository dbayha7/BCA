"""Native CQL preparation checks; reuse seven byte-pinned accepted TD3 cells."""
from pathlib import Path
from dataclasses import asdict
import contextlib,gc,hashlib,io,json,os,sys
root=Path('/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v2')
old=Path('/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v1')
source=root/'source'
sys.path.insert(0,str(source));os.environ['D4RL_DATASET_DIR']='/users/dbayha/.d4rl/datasets'
from runtime.environment import setup
setup()
from runtime.config import resolve,typed
from train import prepare
import jax
assert jax.default_backend()=='cpu'
pins=json.loads((root/'prior_attempt.json').read_text())
assert all(hashlib.sha256((old/n).read_bytes()).hexdigest()==h for n,h in pins['files'].items())
assert json.loads((old/'data-check-process-v1/actual_exit.json').read_text())['actual_returncode']==1
assert not (old/'queue-v1').exists() and not (old/'controller-process-v1').exists()
expected=json.loads((root/'expected_preparation.json').read_text())['cells']
records=json.loads((old/'cluster_data_progress.json').read_text())
assert len(records)==7 and {r['host'] for r in records}=={'td3_bc'}
for r in records:
    assert r['accepted']
    e=next(x for x in expected if (x['host'],x['dataset'])==(r['host'],r['dataset']))
    assert r['paired_data_fingerprints']==e['paired_data_fingerprints']
for original in [x for x in expected if x['host']=='cql']:
    dataset=original['dataset'];identities=[]
    for method in ('host','bca'):
        row=resolve(source/'configs/cql.yaml',method,202609171,root/'unused-data-output',dataset)
        with contextlib.redirect_stdout(io.StringIO()):
            objects=typed(row);prepared=prepare(row,objects)
        R,args,spec,protocol=objects
        # These are the same pre-initialization contracts in CQL.run_prepared.
        R.validate_protocol(args,spec,protocol)
        assert R._digest(prepared.metadata)==prepared.metadata_sha256
        assert R._digest(R._settings(args,spec,protocol))==prepared.metadata['settings_sha256']
        assert R.source_identity()==prepared.metadata['source_files']
        hashes=R._prepared_hashes(prepared.training,prepared.heldout,prepared.reference,
            prepared.obs_mean,prepared.obs_std,prepared.max_action,prepared.max_episode_steps)
        assert hashes==prepared.metadata['run_input_hashes']
        assert R._digest([asdict(e) for e in protocol.evaluation_events])==prepared.metadata['evaluation_events_sha256']
        # Use the exact CQL fingerprint definition in the original preparation audit.
        identity=[R._digest({k:R._array_hash(v) for k,v in prepared.training._asdict().items()}),
                  R._array_hash(prepared.obs_mean),R._array_hash(prepared.obs_std)]
        assert identity==original['paired_data_fingerprints'],('cql',dataset,method,'data mismatch')
        identities.append(identity);del prepared;gc.collect()
    assert identities[0]==identities[1]
    records.append(dict(host='cql',dataset=dataset,accepted=True,paired_data_fingerprints=identities[0]))
    (root/'cluster_data_progress.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[-1]),flush=True)
assert len(records)==14
(root/'cluster_data_acceptance.json').write_text(json.dumps(dict(accepted=True,cells=records,
    reused_td3_cells=7,new_cql_cells=7,learner_updates=0,explicit_outcome_steps=0),indent=2))
