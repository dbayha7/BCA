"""Cluster CPU data gate against the accepted local preparation fingerprints."""
from pathlib import Path
import os,sys,json,gc,contextlib,io
root=Path('/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v1');source=root/'source'
sys.path.insert(0,str(source));os.environ['D4RL_DATASET_DIR']='/users/dbayha/.d4rl/datasets'
from runtime.environment import setup
setup()
from runtime.config import resolve,typed
from train import prepare
import jax
assert jax.default_backend()=='cpu'
expected=json.loads((root/'expected_preparation.json').read_text())['cells']
records=[]
for host in ('td3_bc','cql'):
 for original in [x for x in expected if x['host']==host]:
  dataset=original['dataset'];identities=[]
  for method in ('host','bca'):
   row=resolve(source/'configs'/(host+'.yaml'),method,202609171,root/'unused-data-output',dataset)
   with contextlib.redirect_stdout(io.StringIO()):
    objects=typed(row);prepared=prepare(row,objects)
   if host=='iql':
    D,m,o=objects;P=m.P
    identity=[P.fingerprint(prepared.train),P.fingerprint(prepared.calibration),P.fingerprint(prepared.obs_mean),P.fingerprint(prepared.obs_std)]
   else:
    R,args,spec,protocol=objects;R.validate_prepared(args,spec,protocol,prepared)
    identity=[R._tree_hash(prepared.training),R._array_hash(prepared.obs_mean),R._array_hash(prepared.obs_std)]
   assert identity==original['paired_data_fingerprints'],(host,dataset,method,'prepared data mismatch')
   identities.append(identity);del prepared;gc.collect()
  assert identities[0]==identities[1]
  records.append(dict(host=host,dataset=dataset,accepted=True,paired_data_fingerprints=identities[0]))
  (root/'cluster_data_progress.json').write_text(json.dumps(records,indent=2))
  print(json.dumps(records[-1]),flush=True)
(root/'cluster_data_acceptance.json').write_text(json.dumps(dict(accepted=True,cells=records,learner_updates=0,explicit_outcome_steps=0),indent=2))
