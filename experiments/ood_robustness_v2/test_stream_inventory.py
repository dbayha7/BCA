import copy
import hashlib
import json
import unittest
from unittest.mock import patch
from stream_inventory import parse_bound,seed_fields,inventory,check_v2,PAIRS,STREAM_SIZES,digest
from candidate_design import stream_seed

def fixtures():
    training=dict(runs=[dict(seed=i,row=dict(protocol=dict(evaluation_seed=100+i))) for i in range(280)])
    counter=1600000000
    def allocate(n):
        nonlocal counter
        out=list(range(counter,counter+n));counter+=n;return out
    v1=dict(schema='bca-ood-declaration-v1',checkpoints=[dict(resolved=dict(seed=i)) for i in range(40)],
        pairs=[dict(pair_id=f'{h}/{e}/s{s}',streams={p:allocate(n) for p,n in STREAM_SIZES.items()}) for h,e,s in PAIRS],
        engineering_streams={f'{h}/{e}':allocate(1024) for h in ('td3_bc','rebrac') for e in ('hopper','walker2d')},
        bootstrap_seeds=allocate(10000),plotting_fixture_seeds=allocate(1))
    return training,v1

def build(t,v):
    v=copy.deepcopy(v);v['manifest_sha256']=digest({k:x for k,x in v.items() if k!='manifest_sha256'})
    tb=json.dumps(t).encode();vb=json.dumps(v).encode()
    return inventory(tb,hashlib.sha256(tb).hexdigest(),vb,hashlib.sha256(vb).hexdigest())

class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.t,cls.v=fixtures()
    def test_inherited_seed_fields_and_paths(self):
        result=seed_fields(dict(seed_block={'a':[17,{'b':18}]},unrelated=19,seed_flag=True))
        self.assertEqual(result,[dict(seed=17,path=['seed_block','a',0]),dict(seed=18,path=['seed_block','a',1,'b'])])
    def test_numeric_seed_bounds(self):
        for x in (-1,2**32):
            with self.assertRaises(ValueError):seed_fields(dict(seed=x))
    def test_external_hash_required(self):
        with self.assertRaises(ValueError):parse_bound(b'{}','0'*64)
    def test_duplicate_json_keys_refused(self):
        b=b'{"a":1,"a":2}'
        with self.assertRaises(ValueError):parse_bound(b,hashlib.sha256(b).hexdigest())
    def test_nonfinite_json_refused(self):
        b=b'{"a":NaN}'
        with self.assertRaises(ValueError):parse_bound(b,hashlib.sha256(b).hexdigest())
    def test_complete_inventory_counts(self):
        x=build(self.t,self.v);self.assertEqual(sum(r['source']=='v1_ood' for r in x['records']),72777)
        self.assertEqual(len(x['records']),72777+560+40)
    def test_missing_training_run_refused(self):
        t=copy.deepcopy(self.t);t['runs'].pop()
        with self.assertRaises(ValueError):build(t,self.v)
    def test_duplicate_pair_refused(self):
        v=copy.deepcopy(self.v);v['pairs'][1]=copy.deepcopy(v['pairs'][0])
        with self.assertRaises(ValueError):build(self.t,v)
    def test_missing_original_purpose_refused(self):
        v=copy.deepcopy(self.v);v['pairs'][0]['streams'].pop('test_target_noise')
        with self.assertRaises(ValueError):build(self.t,v)
    def test_missing_original_stream_element_refused(self):
        v=copy.deepcopy(self.v);v['pairs'][0]['streams']['continuation'].pop()
        with self.assertRaises(ValueError):build(self.t,v)
    def test_missing_engineering_cell_refused(self):
        v=copy.deepcopy(self.v);v['engineering_streams'].pop('rebrac/walker2d')
        with self.assertRaises(ValueError):build(self.t,v)
    def test_known_training_collision_stops_without_replacement(self):
        t=copy.deepcopy(self.t);target=stream_seed('td3_bc','walker2d',202609171,'candidate',54)
        self.assertEqual(target,404735174);t['runs'][0]['row']['protocol']['evaluation_seed']=target
        r=check_v2(build(t,self.v));self.assertFalse(r['passed']);self.assertEqual(r['predecessor_collisions'][0]['seed'],target)
        self.assertFalse(r['automatic_repair']);self.assertFalse(r['jax_keys_generated'])
    def test_within_new_duplicates_refused(self):
        with patch('stream_inventory.stream_seed',return_value=4000000000):r=check_v2(build(self.t,self.v))
        self.assertFalse(r['passed']);self.assertEqual(len(r['within_v2_duplicates']),16639)
    def test_no_collision_pass_remains_not_execution_acceptance(self):
        r=check_v2(build(self.t,self.v));self.assertTrue(r['passed']);self.assertFalse(r['stream_execution_accepted'])
    def test_changed_unique_inventory_refused(self):
        x=build(self.t,self.v);x['unique_seeds'].pop()
        with self.assertRaises(ValueError):check_v2(x)

if __name__=='__main__':unittest.main(verbosity=2)
