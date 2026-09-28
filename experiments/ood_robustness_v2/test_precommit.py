import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from precommit_bank import (PIN_NAMES, capture_schedule, planned_streams, bind_keys,
                            build_precommit, check_precommit)
from precommit_file import canonical, write_once, read_bound


class Preparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair=('td3_bc','hopper',202609171)
        cls.keys=np.arange(256*250*2,dtype=np.uint32).reshape(256,250,2)
        cls.key_receipt=bind_keys(cls.pair,cls.keys,np.asarray([[999999,999998]],np.uint32),np.empty((0,2),np.uint32))
        cls.inputs=dict(pair=cls.pair,state_index=0,pins={n:hashlib.sha256(n.encode()).hexdigest() for n in PIN_NAMES},
                        key_receipt=cls.key_receipt,keys=cls.keys[0],anchor_sent=np.zeros(3,np.float32),
                        neighboring_actions=np.zeros((32,3),np.float32),host_sent=np.zeros(3,np.float32),
                        bca_sent=np.asarray([.8,.8,.8],np.float32),q95=.1,q99=.5,
                        state_distance=.2,state_q95=.3)

    def build(self,**updates):
        args=copy.deepcopy(self.inputs);args.update(updates);return build_precommit(**args)

    def test_fixed_capture_schedule(self):
        rows=capture_schedule(*self.pair)
        self.assertEqual(len(rows),256)
        self.assertEqual([rows[i]['capture_step'] for i in (0,1,128,129)],[100,300,100,300])
        for i in range(128):
            self.assertEqual(rows[i]['reset_seed'],rows[i+128]['reset_seed'])
            self.assertNotEqual(rows[i]['candidate_seed'],rows[i+128]['candidate_seed'])
        self.assertEqual(len({r['reset_seed'] for r in rows}),64)
        self.assertEqual(rows[-1]['reset_block'],63)

    def test_invalid_pair(self):
        for pair in [('cql','hopper',202609171),('td3_bc','ant',202609171),('td3_bc','hopper',0)]:
            with self.assertRaises(ValueError): capture_schedule(*pair)

    def test_seed_registry_no_old_collision(self):
        r=planned_streams([1,2])
        self.assertEqual(r['count'],20*(64+3*256))
        with self.assertRaises(ValueError): planned_streams([r['seeds'][0]])
        with self.assertRaises(ValueError): planned_streams([True])
        with self.assertRaises(ValueError): planned_streams([])

    def test_key_table_shape_dtype(self):
        for keys in (self.keys.astype(np.int64),self.keys[:255],self.keys[:,:,:1]):
            with self.assertRaises(ValueError): bind_keys(self.pair,keys,np.asarray([[999999,999998]],np.uint32),np.empty((0,2),np.uint32))

    def test_key_collisions_old_and_new(self):
        old=np.asarray([[999999,999998]],np.uint32)
        for old_arg,prior in [(self.keys[0,:1],np.empty((0,2),np.uint32)),(old,self.keys[1,:1])]:
            with self.assertRaises(ValueError): bind_keys(self.pair,self.keys,old_arg,prior)
        keys=self.keys.copy();keys[1,0]=keys[0,0]
        with self.assertRaises(ValueError): bind_keys(self.pair,keys,old,np.empty((0,2),np.uint32))
        with self.assertRaises(ValueError): bind_keys(self.pair,self.keys,np.empty((0,2),np.uint32),np.empty((0,2),np.uint32))

    def test_fourteen_fixed_slots_and_alias(self):
        b=self.build();self.assertEqual([r['slot'] for r in b['slots']],list(range(14)))
        self.assertEqual([r['role'] for r in b['slots']],[*['near']*4,*['moderate']*4,*['strong']*4,'host_policy','bca_policy'])
        self.assertEqual(b['slots'][12]['alias_of'],0)
        self.assertEqual(b['host_reference_slot'],12);self.assertEqual(b['anchor_slot'],0)
        self.assertTrue(b['support_quotas_complete']);self.assertFalse(b['execution_accepted'])

    def test_quota_failure_retains_native_slot_numbers(self):
        b=self.build(q95=1.1,q99=1.2)
        self.assertFalse(b['support_quotas_complete'])
        self.assertEqual([r['slot'] for r in b['slots'] if r['status']=='missing'],list(range(4,12)))
        self.assertEqual(b['slots'][12]['alias_of'],0)
        self.assertEqual(b['slots'][13]['role'],'bca_policy')
        self.assertEqual(b['unique_trajectory_slots'],[0,1,2,3,13])

    def test_both_native_aliases_reuse(self):
        b=self.build(bca_sent=np.zeros(3,np.float32))
        self.assertEqual([r['alias_of'] for r in b['slots'][12:]],[0,0])
        self.assertEqual(len(b['unique_trajectory_slots']),12)

    def test_state_band_boundary_and_no_reweight(self):
        self.assertEqual(self.build(state_distance=.3)['state_support'],'near')
        self.assertEqual(self.build(state_distance=.30001)['state_support'],'distant')
        with self.assertRaises(ValueError): self.build(state_distance=-.1)

    def test_proposed_sent_applied_and_clipping(self):
        b=self.build()
        for row in b['slots']:
            self.assertIn('proposed',row);self.assertIn('sent',row);self.assertIn('applied',row)
            self.assertIn('clipped',row);self.assertIn('distance',row);self.assertIn('band',row)
        self.assertEqual(b['slots'][0]['proposal_index'],-1)
        self.assertEqual(b['pool_size'],8192)
        self.assertEqual(b['slots'][13]['band'],'strong')

    def test_external_pins_exact(self):
        for pins in ({},dict(self.inputs['pins'],surplus='a'*64),dict(self.inputs['pins'],state_snapshot='z'*64)):
            with self.assertRaises(ValueError): self.build(pins=pins)

    def test_wrong_key_row_or_receipt_refused(self):
        with self.assertRaises(ValueError): self.build(keys=self.keys[1])
        r=copy.deepcopy(self.key_receipt);r['pair']=['rebrac','hopper',202609171]
        with self.assertRaises(ValueError): self.build(key_receipt=r)
        with self.assertRaises(ValueError): self.build(state_index=256)

    def test_native_actions_finite_and_exact_dtype(self):
        for a in (np.zeros(3,np.float64),np.asarray([0,np.nan,0],np.float32),np.ones(2,np.float32),np.full(3,1.01,np.float32)):
            with self.assertRaises(ValueError): self.build(host_sent=a)

    def test_rebuilt_binding_detects_mutation(self):
        b=self.build();check_precommit(b,**copy.deepcopy(self.inputs))
        b['slots'][4]['alias_of']=0
        with self.assertRaises(ValueError): check_precommit(b,**copy.deepcopy(self.inputs))

    def test_rebuilt_binding_uses_external_input(self):
        b=self.build();args=copy.deepcopy(self.inputs);args['pins']['state_snapshot']='f'*64
        with self.assertRaises(ValueError): check_precommit(b,**args)

    def test_no_outcome_or_width_input(self):
        with self.assertRaises(TypeError): self.build(widths=np.arange(14))
        with self.assertRaises(TypeError): self.build(returns=np.arange(14))

    def test_signed_zero_bytes_preserved(self):
        b=self.build(host_sent=np.full(3,-0.,np.float32))
        self.assertEqual(b['slots'][12]['alias_of'],0)
        self.assertNotEqual(b['slots'][12]['sent'],b['slots'][0]['sent'])

    def test_anchor_must_be_first_neighbor(self):
        with self.assertRaises(ValueError): self.build(anchor_sent=np.full(3,.01,np.float32))

    def test_key_receipt_cannot_promote_acceptance(self):
        r=copy.deepcopy(self.key_receipt);r['execution_accepted']=True
        with self.assertRaises(ValueError): self.build(key_receipt=r)
        r=copy.deepcopy(self.key_receipt);r['extra']='unbound'
        with self.assertRaises(ValueError): self.build(key_receipt=r)

    def test_walker_native_dimensions(self):
        pair=('rebrac','walker2d',202609175)
        receipt=bind_keys(pair,self.keys,np.asarray([[999999,999998]],np.uint32),np.empty((0,2),np.uint32))
        b=self.build(pair=pair,key_receipt=receipt,anchor_sent=np.zeros(6,np.float32),
                     neighboring_actions=np.zeros((32,6),np.float32),host_sent=np.zeros(6,np.float32),bca_sent=np.ones(6,np.float32))
        self.assertEqual(b['slots'][13]['sent']['shape'],[6])


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'bank.json'

    def test_exclusive_bound_roundtrip(self):
        obj={'schema':'fixture','slots':[1,2],'minus_zero':-0.}
        h=write_once(self.path,obj);self.assertEqual(read_bound(self.path,h),obj)
        self.assertEqual(h,hashlib.sha256(self.path.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError): write_once(self.path,obj)

    def test_wrong_hash_and_changed_content_refused(self):
        h=write_once(self.path,{'a':1})
        with self.assertRaises(ValueError): read_bound(self.path,'0'*64)
        self.path.write_bytes(b'{"a":2}\n')
        with self.assertRaises(ValueError): read_bound(self.path,h)

    def test_duplicate_keys_and_nonfinite_refused(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'):
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError): read_bound(self.path,hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError): canonical({'a':float('nan')})

    def test_symlink_refused(self):
        target=self.path.with_name('target');target.write_bytes(b'{}')
        self.path.symlink_to(target)
        with self.assertRaises(ValueError): read_bound(self.path,hashlib.sha256(b'{}').hexdigest())
        with self.assertRaises(ValueError): write_once(self.path,{})

    def test_size_depth_types_refused(self):
        with self.assertRaises(ValueError): canonical({'x':'a'*2_000_001})
        obj={};p=obj
        for _ in range(40): p['x']={};p=p['x']
        with self.assertRaises(ValueError): canonical(obj)
        with self.assertRaises(ValueError): canonical({1:'bad'})
        with self.assertRaises(ValueError): canonical({'x':np.zeros(3)})

    def test_durability_failure_preserves_file(self):
        with mock.patch('precommit_file.os.fsync',side_effect=OSError('synthetic fsync failure')):
            with self.assertRaises(OSError): write_once(self.path,{'a':1})
        self.assertTrue(self.path.exists())
        with self.assertRaises(FileExistsError): write_once(self.path,{'a':1})

    def test_large_read_refused(self):
        raw=b' '*2_000_001;self.path.write_bytes(raw)
        with self.assertRaises(ValueError): read_bound(self.path,hashlib.sha256(raw).hexdigest())

    def test_symlink_parent_refused(self):
        actual=self.path.with_name('actual');actual.mkdir()
        link=self.path.with_name('linked');link.symlink_to(actual,target_is_directory=True)
        with self.assertRaises(ValueError): write_once(link/'bank.json',{})


if __name__=='__main__': unittest.main(verbosity=2)
