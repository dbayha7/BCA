"""Checks only the separately versioned, explicitly approved amendment."""
import hashlib,json,unittest
from pathlib import Path
import numpy as np
import candidate_design as original
import candidate_design_amended_v1 as amended
import precommit_bank_amended_v1 as precommit
import collector_driver_amended_v1 as collector
import outcome_driver_amended_v1 as outcome
import stream_inventory_amended_v1 as inventory

class AmendmentTests(unittest.TestCase):
    def test_independent_full_map_single_difference(self):
        changes=[];seen=set()
        for h,e,s in inventory.PAIRS:
            for purpose in ('reset','candidate','continuation','random_score'):
                for i in range(64 if purpose=='reset' else 256):
                    identity=(h,e,s,purpose,i)
                    raw=json.dumps(['bca-action-support-robustness-v2/2026-09-28',*identity],separators=(',',':')).encode()
                    old=int(hashlib.sha256(raw).hexdigest()[:8],16)
                    new=amended.stream_seed(*identity)
                    self.assertEqual(old,original.stream_seed(*identity))
                    if new!=old:changes.append((*identity,old,new))
                    self.assertNotIn(new,seen);seen.add(new)
        self.assertEqual(len(seen),16640)
        self.assertEqual(changes,[('td3_bc','walker2d',202609171,'candidate',54,404735174,404735181)])

    def test_bound_document_and_bindings(self):
        root=Path(__file__).resolve().parents[2]
        pin=hashlib.sha256((root/'docs/OOD_V2_STREAM_AMENDMENT_20260928.md').read_bytes()).hexdigest()
        self.assertEqual(pin,amended.AMENDMENT_SHA256)
        self.assertEqual(pin,precommit.AMENDMENT_SHA256)
        self.assertEqual(pin,outcome.AMENDMENT_SHA256)
        self.assertIs(precommit.stream_seed,amended.stream_seed)
        self.assertIs(inventory.stream_seed,amended.stream_seed)
        self.assertIs(collector.capture_schedule,precommit.capture_schedule)
        self.assertIs(outcome.capture_schedule,precommit.capture_schedule)
        row=precommit.capture_schedule('td3_bc','walker2d',202609171)[54]
        self.assertEqual((row['collector'],row['reset_block'],row['capture_step'],row['candidate_seed']),('host',27,100,404735181))

    def test_future_collision_refuses_without_repair(self):
        for value in (404735181,amended.stream_seed('rebrac','hopper',202609173,'reset',1)):
            with self.assertRaisesRegex(ValueError,'collision'):precommit.planned_streams([value])
        self.assertEqual(amended.stream_seed('td3_bc','walker2d',202609171,'candidate',54),404735181)

    def test_new_precommit_binds_amendment_and_rejects_mutation(self):
        pair=('td3_bc','walker2d',202609171)
        keys=np.arange(256*250*2,dtype=np.uint32).reshape(256,250,2)
        key_receipt=precommit.bind_keys(pair,keys,np.array([[999999,999998]],np.uint32),np.empty((0,2),np.uint32))
        kwargs=dict(pair=pair,state_index=54,pins={n:hashlib.sha256(n.encode()).hexdigest() for n in precommit.PIN_NAMES},key_receipt=key_receipt,keys=keys[54],anchor_sent=np.zeros(6,np.float32),neighboring_actions=np.zeros((32,6),np.float32),host_sent=np.zeros(6,np.float32),bca_sent=np.full(6,.8,np.float32),q95=.1,q99=.5,state_distance=.2,state_q95=.3)
        bank=precommit.build_precommit(**kwargs)
        self.assertEqual(bank['amendment_sha256'],amended.AMENDMENT_SHA256)
        self.assertEqual(bank['capture']['candidate_seed'],404735181)
        self.assertFalse(bank['execution_accepted'])
        bank['amendment_sha256']='0'*64
        with self.assertRaises(ValueError):precommit.check_precommit(bank,**kwargs)

if __name__=='__main__':unittest.main(verbosity=2)
