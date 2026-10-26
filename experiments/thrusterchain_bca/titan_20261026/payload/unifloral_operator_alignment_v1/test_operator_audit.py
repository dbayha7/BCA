"""Synthetic-only tests; never import models or load scientific result banks."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('operator_math_under_test',ROOT/'operator_audit.py')
AUDIT=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
PROTOCOL=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def bank(development):
    n=1152 if development else 256
    row=np.arange(n);c=row%9
    qmin=(2.+.2*c).astype(np.float32)
    current=np.column_stack((qmin+.1*(row%3),qmin)).astype(np.float32)
    rewards=np.full(n,.25,dtype=np.float32)
    done=(row%17==0).astype(np.float32)
    actor=np.tile(np.array([.95,-.95,.2,-.2],np.float32),(n,1))
    noise=np.stack([np.tile(np.array([.7,-.7,.1,-.1],np.float32)+np.float32(j*.01),(n,1)) for j in range(4)])
    clipped=np.clip(noise,-.5,.5)
    next_actions=np.clip(actor[None]+clipped,-1.,1.)
    qnext=np.stack([np.column_stack((1.+.05*j+.03*c,1.2+.05*j+.03*c)) for j in range(4)]).astype(np.float32)
    target=rewards[None]+(np.float32(1)-done[None])*np.float32(.98)*qnext.min(axis=-1)
    value={'observations':np.zeros((n,14),np.float32),'actions':np.zeros((n,4),np.float32),
           'q_current':current,'u':(1.+.1*c).astype(np.float32),'u_usable':True,'u_valid':True,
           'rewards':rewards,'done':done,'next_obs':np.zeros((n,14),np.float32),
           'target_actor_next':actor,'raw_noise':noise,'clipped_noise':clipped,
           'next_actions':next_actions,'target_q':qnext,'native_y':target,
           'snapshot_receipt':{'final_refresh_verified':True,'current_equals_reference':True}}
    if development:
        value['saved_values']=current[:,0].astype(np.float64)-.3*(c-4)
        value['physics_rewards']=rewards.astype(np.float64)
        value['metadata']=[{'group_id':int(i//9),'candidate_id':int(i%9),
                            'batch':PROTOCOL['batch_names'][i//9//64],
                            'time':PROTOCOL['query_times'][(i//9)%4],
                            'episode_seed':1000+int(i//9)} for i in row]
    else:
        value['metadata']=[{'group_id':int(i//2),'row_index':int(i)} for i in row]
    return value


class OperatorAuditTests(unittest.TestCase):
    def test_accounting_orientation_and_tight_identity(self):
        source=bank(True)
        q=source['q_current'].astype(np.float64);y=source['native_y'].astype(np.float64)
        terms,receipt=AUDIT.accounting(q,y,source['saved_values'],PROTOCOL)
        np.testing.assert_array_equal(terms['R'],q.min(axis=1)[None]-y)
        np.testing.assert_array_equal(terms['H'],q[:,0]-q.min(axis=1))
        np.testing.assert_array_equal(terms['G'],y-source['saved_values'][None])
        np.testing.assert_allclose(np.broadcast_to(terms['E'][None],terms['R'].shape),terms['R']+terms['H'][None]+terms['G'],rtol=0,atol=2e-15)
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['evaluated_row_draw_pairs'],4608)
        self.assertLessEqual(receipt['max_fraction_of_allowed_roundoff'],1)

    def test_terminal_mask_has_no_bootstrap(self):
        source=bank(False);terminal=source['done']==1
        source['target_q'][:,terminal,:]=1e8
        _,_,receipt=AUDIT.validate_bank(source,PROTOCOL,False)
        self.assertTrue(receipt['native_y']['passed'])
        np.testing.assert_array_equal(source['native_y'][:,terminal],np.broadcast_to(source['rewards'][terminal],(4,int(terminal.sum()))))

    def test_operator_replay_rejects_bad_clipping_action_and_target(self):
        for key,index in [('clipped_noise',(0,0,0)),('next_actions',(0,0,0)),('native_y',(0,1))]:
            with self.subTest(key=key):
                source=bank(False);source[key][index]-=.1
                with self.assertRaises(ValueError):AUDIT.validate_bank(source,PROTOCOL,False)

    def test_target_replay_uses_native_float32_under_cancellation(self):
        source=bank(False)
        source['rewards'][:]=-980000.;source['done'][:]=0.
        source['target_q'][:]=1000000.
        source['native_y'][:]=source['rewards'][None]+(np.float32(1.)-source['done'][None])*np.float32(.98)*source['target_q'].min(axis=-1)
        _,_,receipt=AUDIT.validate_bank(source,PROTOCOL,False)
        self.assertEqual(receipt['native_y']['max_abs_difference'],0.)
        wrong=source['rewards'].astype(np.float64)[None]+(1.-source['done'].astype(np.float64)[None])*float(np.float32(.98))*source['target_q'].astype(np.float64).min(axis=-1)
        self.assertGreater(float(np.max(np.abs(wrong-source['native_y']))),PROTOCOL['replay_atol'])

    def test_radius_is_u_not_half_u(self):
        source=bank(False)
        source['done'][:]=1;source['rewards'][:]=.25;source['native_y'][:]=.25
        source['q_current'][:]=1.;source['u'][:]=1.
        result=AUDIT.fitting_audit(source,PROTOCOL)
        self.assertEqual(result['draws'][0]['row_weighted_radius_containment'],1.)
        self.assertTrue(np.all(np.abs(result['draws'][0]['residual'])>.5*source['u']))

    def test_undefined_correlations_are_not_zero(self):
        self.assertEqual(AUDIT.pearson([.1]*9,range(9))['reason'],'constant_predictor')
        self.assertEqual(AUDIT.pearson(range(9),[0]*9)['reason'],'constant_outcome')
        self.assertIsNone(AUDIT.pearson([1],[2])['value'])
        self.assertAlmostEqual(AUDIT.pearson([0,1,2],[4,6,8])['value'],1.)
        source=bank(True);source['u'][:]=.1
        source['saved_values']=source['q_current'][:,0].astype(np.float64)+1.
        result=AUDIT.development_audit(source,PROTOCOL)
        relation=result['draws'][0]['overall']['relationships']['u_signed_error']
        self.assertEqual(relation['undefined_states'],128)
        self.assertIsNone(relation['equal_state_mean'])
        self.assertEqual(result['draws'][0]['overall']['relationships']['abs_residual_positive_error']['undefined_reasons'],{'constant_outcome':128})

    def test_reference_and_finite_guards_fail_closed(self):
        changes=[('u_usable',False),('u_valid',False),('u',np.full(256,-1.)),('u',np.full(256,np.nan)),
                 ('snapshot_receipt',{'final_refresh_verified':False,'current_equals_reference':True})]
        for key,value in changes:
            with self.subTest(key=key):
                source=bank(False);source[key]=value
                with self.assertRaises(ValueError):AUDIT.validate_bank(source,PROTOCOL,False)

    def test_fitting_row_and_episode_weights_distinct(self):
        source=bank(False)
        source['metadata']=[{'group_id':0 if i==0 else 1,'row_index':i} for i in range(256)]
        source['u'][:]=0.;source['u'][0]=100.
        result=AUDIT.fitting_audit(source,PROTOCOL)
        self.assertEqual(result['unique_episode_groups'],2)
        for draw in result['draws']:
            self.assertEqual(draw['row_weighted_radius_containment'],1/256)
            self.assertEqual(draw['equal_episode_group_radius_containment'],.5)
        self.assertFalse(result['heldout_coverage_claimed'])

    def test_preserve_noise_draws_before_absolute_operation(self):
        q=np.zeros((2,2),np.float64);y=np.array([[1.,1.],[-1.,-1.],[2.,2.],[-2.,-2.]])
        terms,_=AUDIT.accounting(q,y,np.zeros(2),PROTOCOL)
        self.assertEqual(terms['R'].shape,(4,2))
        self.assertEqual(float(np.mean(np.abs(terms['R']))),1.5)
        self.assertEqual(float(np.mean(np.abs(np.mean(terms['R'],axis=0)))),0.)
        spread=AUDIT._draw_spread([1.,2.,None,4.])
        self.assertEqual(spread['defined_draws'],3)
        self.assertEqual(spread['per_draw'],[1.,2.,None,4.])

    def test_full_synthetic_scope_and_equal_state_strata(self):
        result=AUDIT.analyze(bank(True),bank(False),PROTOCOL)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['development']['independent_states'],128)
        self.assertEqual(len(result['development']['draws']),4)
        for draw in result['development']['draws']:
            self.assertEqual(len(draw['states']),128)
            self.assertEqual(draw['overall']['states'],128)
            for batch in PROTOCOL['batch_names']:
                self.assertEqual(draw['by_batch'][batch]['states'],64)
                for time in PROTOCOL['query_times']:
                    self.assertEqual(draw['by_time'][str(time)]['states'],32)
                    self.assertEqual(draw['by_batch_time'][batch][str(time)]['states'],16)
            self.assertEqual(set(draw['overall']['relationships']),set(AUDIT.RELATIONSHIPS))
        self.assertTrue(result['full_training_hold'])
        for flag in ['automatic_training_approval','new_policy_training','method_or_strength_selected','expected_operator_claimed','new_confidence_intervals']:
            self.assertIs(result[flag],False)
        for flag in ['new_diagnostic_fits','policy_optimizer_updates','calibrator_updates','confirmation_files_read','model_or_simulator_calls_by_audit_module']:
            self.assertEqual(result[flag],0)
        json.dumps(result,allow_nan=False)

    def test_metadata_and_reward_cast_mismatch_rejected(self):
        source=bank(True);source['metadata'][1]['candidate_id']=0
        with self.assertRaises(ValueError):AUDIT.validate_bank(source,PROTOCOL,True)
        source=bank(True);source['physics_rewards'][0]+=.01
        with self.assertRaises(ValueError):AUDIT.validate_bank(source,PROTOCOL,True)

    def test_frozen_protocol_rejected_if_changed(self):
        protocol=copy.deepcopy(PROTOCOL);protocol['gamma']=.99
        with self.assertRaises(ValueError):AUDIT.analyze(bank(True),bank(False),protocol)


if __name__=='__main__':unittest.main()
