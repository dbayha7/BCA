"""Synthetic file/API fixtures only; no production model, data or simulator calls."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_worker as worker
import train_worker


def protocol():
    return worker.read(worker.HERE / 'protocol.json')


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True) + '\n').encode()
    Path(path).write_bytes(raw)
    return worker.digest(raw)


def counts(step):
    return dict(critic=step, auxiliary=step, actor=step//2, phase=0,
                refresh_accepted=0 if step==0 else 1 if step==10000 else 2)


def training_receipt(directory, cell='seed0_full', stage='warmup', protocol_sha='synthetic'):
    directory = Path(directory)
    matching = dict(cell=cell, protocol_sha256=protocol_sha, distance_coefficient=2.5)
    matching_sha = write_json(directory/'matching_receipt.json', matching)
    report = dict(status='completed_central_quartile_'+('warmup' if stage=='warmup' else 'branch'),
        protocol_sha256=protocol_sha, cell=cell, condition=cell[6:], seed_index=int(cell[4]),
        arm='native' if stage=='warmup' else stage, execution_counts_known=True,
        matching_receipt_sha256=matching_sha)
    report_sha = write_json(directory/'report.json', report)
    supervisor = dict(job=cell+'/'+stage, completed=True, exit_code=0, timed_out=False,
                      interrupted=False, report_sha256=report_sha)
    write_json(directory/'SUPERVISOR_RESULT.json', supervisor)
    return report, matching, supervisor


def summary_fixture(native_harm=True):
    """144 declared records; changing harm leaves BCA gains/interaction fixed."""
    result=[]
    for spec in worker.snapshot_plan(protocol()):
        for context in ('ordinary','strong_current'):
            condition,arm,step=spec['condition'],spec['arm'],spec['checkpoint']
            base=-20. if step==0 else -15. if step==10000 else -10.
            if native_harm and condition=='hole': base-=2.
            gain=(3. if condition=='hole' else 1.) if arm=='bca' else 1. if arm=='distance' and condition=='hole' else 0.
            native_mae=8. if native_harm and condition=='hole' else 5.
            adjusted=native_mae-(3. if condition=='hole' else 1.)
            result.append({**{k:spec[k] for k in ('seed','condition','arm','checkpoint')},
                'context':context,'free':{'returns':(base+gain+np.arange(32)*.01).tolist()},
                'q':{'panels':{'high_band_fixed_primary':{'state_count':24,
                    'raw':{'mae':native_mae},'adjusted':{'mae':adjusted}}}}})
    return result


class EvaluationWorkerTests(unittest.TestCase):
    def test_complete_fixed_snapshot_plan(self):
        rows = worker.snapshot_plan(protocol())
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({r['id'] for r in rows}), 72)
        self.assertEqual(sum(r['checkpoint'] > 0 for r in rows), 63)
        self.assertEqual({r['checkpoint'] for r in rows}, {0,10000,15000,20000})
        self.assertEqual(sum(r['checkpoint']==10000 for r in rows), 9)
        self.assertEqual(sum(r['checkpoint']==20000 for r in rows), 27)
        bad=protocol();bad['cells'][0]=bad['cells'][1]
        with self.assertRaisesRegex(ValueError, 'Nine distinct'): worker.snapshot_plan(bad)

    def test_protocol_rejects_seed_collision_and_bank_change(self):
        p=protocol();worker.validate_protocol(p)
        bad=copy.deepcopy(p)
        bad['evaluation']['ordinary']['free_seeds'][0]=bad['evaluation']['ordinary']['q_seeds'][0]
        with self.assertRaisesRegex(ValueError, 'collision'): worker.validate_protocol(bad)
        bad=copy.deepcopy(p);bad['evaluation']['fixed_actions'][0][1]=.1
        with self.assertRaisesRegex(ValueError, 'Bank configuration'): worker.validate_protocol(bad)
        bad=copy.deepcopy(p);bad['simulator_transition_cap']+=1
        with self.assertRaisesRegex(ValueError, 'constants'): worker.validate_protocol(bad)

    def test_artifact_paths_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=b'fixture bytes';(root/'state.bin').write_bytes(raw)
            desc=dict(path='state.bin',sha256=worker.digest(raw),bytes=len(raw))
            self.assertEqual(worker.artifact(root,desc),raw)
            for path in ['../state.bin','C:/state.bin',str(root/'state.bin')]:
                with self.assertRaisesRegex(ValueError,'Unsafe'): worker.checked(root,path)
            for key,value in [('bytes',len(raw)+1),('sha256','wrong')]:
                bad={**desc,key:value}
                with self.assertRaisesRegex(ValueError,'hash mismatch'): worker.artifact(root,bad)
            with self.assertRaisesRegex(ValueError,'Malformed'):
                worker.artifact(root,{**desc,'extra':True})

    def test_training_seal_rejects_incomplete_changed_or_wrong_cell(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report,matching,supervisor=training_receipt(root)
            actual,match,seals=worker.training_seal(root,'seed0_full','warmup','synthetic')
            self.assertEqual(actual,report);self.assertEqual(match,matching);self.assertEqual(len(seals),3)
            with self.assertRaisesRegex(ValueError,'Incomplete supervisor'):
                worker.training_seal(root,'seed1_full','warmup','synthetic')
            supervisor['interrupted']=True;write_json(root/'SUPERVISOR_RESULT.json',supervisor)
            with self.assertRaisesRegex(ValueError,'Incomplete supervisor'):
                worker.training_seal(root,'seed0_full','warmup','synthetic')
            report,_,_=training_receipt(root)
            (root/'report.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'changed report'):
                worker.training_seal(root,'seed0_full','warmup','synthetic')

    def test_actual_training_checkpoint_export_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            job=dict(cell='seed0_hole',condition='hole',seed_index=0,kind='branch',arm='bca')
            desc=train_worker.save_checkpoint(root,b'full state',b'actor',counts=counts(15000),
                job=job,protocol_sha='synthetic',normalization={'fixture':'common'},
                matching_receipt_sha256='matching')
            report=dict(checkpoint_manifests={'15000':desc},matching_receipt_sha256='matching')
            spec=dict(cell='seed0_hole',condition='hole',seed=0,stage='bca',arm='bca',checkpoint=15000)
            export,seals=worker.checkpoint_export(root,report,spec,'synthetic')
            self.assertEqual(len(seals),3)
            self.assertFalse(export['manifest']['resume_authorized'])
            wrong={**spec,'cell':'seed1_hole','seed':1}
            with self.assertRaisesRegex(ValueError,'Checkpoint identity'):
                worker.checkpoint_export(root,report,wrong,'synthetic')
            (export['base']/'full_state.msgpack').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                worker.checkpoint_export(root,report,spec,'synthetic')

    def test_checkpoint_counter_and_matching_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job=dict(cell='seed0_full',condition='full',seed_index=0,kind='warmup')
            desc=train_worker.save_checkpoint(root,b'state',b'actor',counts=counts(10000),
                job=job,protocol_sha='synthetic',normalization={},matching_receipt_sha256='matching')
            report=dict(checkpoint_manifests={'10000':desc},matching_receipt_sha256='wrong')
            spec=dict(cell='seed0_full',condition='full',seed=0,stage='warmup',arm='native',checkpoint=10000)
            with self.assertRaisesRegex(ValueError,'matching receipt binding'):
                worker.checkpoint_export(root,report,spec,'synthetic')
            report['matching_receipt_sha256']='matching'
            manifest_path=root/desc['path'];manifest=json.loads(manifest_path.read_text())
            manifest['counts']['refresh_accepted']=0
            raw=(json.dumps(manifest)+'\n').encode();manifest_path.write_bytes(raw)
            report['checkpoint_manifests']['10000']={**desc,'sha256':worker.digest(raw),'bytes':len(raw)}
            with self.assertRaisesRegex(ValueError,'counters/calibration'):
                worker.checkpoint_export(root,report,spec,'synthetic')

    def test_common_normalization_enforced_across_all_exports(self):
        p=protocol()
        def seal(directory,cell,stage,protocol_sha):
            return {'initial_parameter_hashes':{'actor':cell[4],'critic':'critic'+cell[4]}},{},{}
        def export(directory,report,spec,protocol_sha):
            return {'manifest':{'normalization':{'same':True}},'spec':spec},{}
        with tempfile.TemporaryDirectory() as d,patch.object(worker,'HERE',Path(d)),\
                patch.object(worker,'training_seal',side_effect=seal),\
                patch.object(worker,'checkpoint_export',side_effect=export):
            exports,_=worker.assemble_exports(p,Path(d)/'run','synthetic')
            self.assertEqual(len(exports),72)
            def mismatch(directory,report,spec,protocol_sha):
                return {'manifest':{'normalization':{'condition':spec['condition']}},'spec':spec},{}
            with patch.object(worker,'checkpoint_export',side_effect=mismatch):
                with self.assertRaisesRegex(ValueError,'Common normalization'):
                    worker.assemble_exports(p,Path(d)/'run','synthetic')

    def test_static_restore_requires_matching_normalization_actor_and_counts(self):
        from flax import serialization
        params={'w':np.array([1.,2.],np.float32),'b':np.array([0.],np.float32)}
        state=SimpleNamespace(microstep=SimpleNamespace(agent_state=SimpleNamespace(actor=SimpleNamespace(params=params))))
        context=SimpleNamespace(normalization={'shared':True},initial_parameter_hashes={'actor':'seed0'})
        adapter=SimpleNamespace(context=context)
        runtime=SimpleNamespace(restore_exact=lambda a,t,r:state,validate=lambda a,s:{'counts':counts(10000)})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'state.bin').write_bytes(b'fixture')
            actorraw=serialization.to_bytes(dict(reversed(list(params.items()))))
            self.assertNotEqual(actorraw,serialization.to_bytes(params))
            (root/'actor.bin').write_bytes(actorraw)
            def desc(name):
                raw=(root/name).read_bytes();return dict(path=name,bytes=len(raw),sha256=worker.digest(raw))
            export=dict(base=root,state_descriptor=desc('state.bin'),actor_descriptor=desc('actor.bin'),
                manifest={'normalization':context.normalization,'counts':counts(10000)},
                training_report={'initial_parameter_hashes':context.initial_parameter_hashes})
            self.assertIs(worker.restore_snapshot(runtime,adapter,object(),export),state)
            bad=copy.deepcopy(export);bad['manifest']['normalization']={'shared':False}
            with self.assertRaisesRegex(ValueError,'normalization'):worker.restore_snapshot(runtime,adapter,None,bad)
            bad=copy.deepcopy(export);bad['manifest']['counts']=counts(20000)
            with self.assertRaisesRegex(ValueError,'Restored counts'):worker.restore_snapshot(runtime,adapter,None,bad)
            (root/'actor.bin').write_bytes(serialization.to_bytes({'w':np.array([9.,9.],np.float32),'b':params['b']}))
            bad=copy.deepcopy(export);bad['actor_descriptor']=desc('actor.bin')
            with self.assertRaisesRegex(ValueError,'Full-state actor'):worker.restore_snapshot(runtime,adapter,None,bad)

    def test_paired_summary_preserves_seed_and_start_units(self):
        values=np.stack([np.arange(32)*.01+s for s in (1,2,3)])
        result=worker.paired_summary(values,protocol())
        np.testing.assert_allclose(result['seed_means'],[1.155,2.155,3.155])
        self.assertAlmostEqual(result['mean'],2.155)
        self.assertEqual(result['paired_values_by_seed'],values.tolist())
        self.assertEqual(result,worker.paired_summary(values,protocol()))
        with self.assertRaisesRegex(ValueError,'matrix'):worker.paired_summary(values.ravel(),protocol())

    def test_complete_144_result_matrix_measures_native_harm_before_rescue(self):
        p=protocol();p['native_hole_return_harm_threshold']=1.;p['native_hole_mae_harm_threshold']=1.
        rows=summary_fixture(native_harm=True)
        self.assertEqual(len(rows),144)
        result=worker.summarize(rows,p)
        self.assertTrue(result['learning_adequacy_passed'])
        self.assertTrue(result['policy_screen_passed'])
        self.assertTrue(result['accuracy_screen_passed'])
        self.assertTrue(result['policy_rescue_screen_passed'])
        self.assertTrue(result['accuracy_rescue_screen_passed'])
        with self.assertRaisesRegex(ValueError,'All snapshot'):
            worker.summarize(rows[:-1],p)
        with self.assertRaisesRegex(ValueError,'All snapshot'):
            worker.summarize(rows+[rows[0]],p)

    def test_positive_interaction_without_native_harm_is_not_rescue(self):
        p=protocol();p['native_hole_return_harm_threshold']=1.;p['native_hole_mae_harm_threshold']=1.
        result=worker.summarize(summary_fixture(native_harm=False),p)
        self.assertTrue(result['policy_screen_passed'])
        self.assertTrue(result['accuracy_screen_passed'])
        self.assertTrue(result['missing_label_policy_interaction_positive_each_seed'])
        self.assertTrue(result['missing_label_accuracy_interaction_positive_each_seed'])
        self.assertFalse(result['policy_rescue_screen_passed'])
        self.assertFalse(result['accuracy_rescue_screen_passed'])

    def test_common_reference_reuses_fixed_policy_values_and_excludes_actor_slot(self):
        rows=summary_fixture()
        for row in rows:
            row['bank_id']='shared_states_and_times_'+row['context']
            actions=np.zeros((32,6,4),np.float32)
            actions[:,1:,0]=[-.5,-.25,0,.25,.5]
            actions[:,0]=.123 if row['arm']=='bca' else -.456
            is_reference=row['condition']=='full' and row['arm']=='native' and row['checkpoint']==20000
            values=np.full((32,6),5. if is_reference else 9.)
            values[:,0]=1000000. if is_reference else -1000000.
            raw=np.full((32,6),7.);raw[:,0]=-999999.
            adjusted=np.full((32,6),6.);adjusted[:,0]=999999.
            row['q']['per_probe']={'actions':actions.tolist(),'high_load':[True]*24+[False]*8,
                'qpi':values.tolist(),'q1':raw.tolist(),'adjusted':adjusted.tolist()}
        result=worker.common_reference_panels(rows)
        self.assertEqual(len(result),126)
        key='seed0_hole__bca__20000__ordinary'
        panel=result[key]
        self.assertTrue(panel['own_actor_slot_excluded'])
        self.assertEqual(panel['additional_simulator_transitions'],0)
        self.assertEqual(panel['reference_policy'],'seed0_full/native/20000')
        self.assertEqual(panel['panels']['all_fixed']['raw_score_deviation']['mae'],2.)
        self.assertEqual(panel['panels']['all_fixed']['adjusted_score_deviation']['mae'],1.)
        self.assertEqual(panel['panels']['high_band_fixed']['state_count'],24)
        self.assertEqual(np.asarray(panel['common_qpi_fixed_actions']).shape,(32,5))
        np.testing.assert_array_equal(panel['own_minus_common_qpi_fixed_actions'],4.)
        bad=copy.deepcopy(rows)
        target=next(r for r in bad if r['seed']==0 and r['condition']=='hole' and r['arm']=='bca' and r['checkpoint']==20000)
        target['q']['per_probe']['actions'][0][1][2]=.2
        with self.assertRaisesRegex(ValueError,'complete fixed actions'):worker.common_reference_panels(bad)
        bad=copy.deepcopy(rows)
        next(r for r in bad if r['checkpoint']==10000)['bank_id']='changed state/time bank'
        with self.assertRaisesRegex(ValueError,'state/time bank'):worker.common_reference_panels(bad)

    def test_direct_evaluator_rejects_execution_hold_before_any_scientific_import(self):
        with patch.object(worker,'module') as loader:
            with self.assertRaisesRegex(ValueError,'Approved protocol changed: schema'):
                worker.run({'execution_authorized':True},Path('never_created'),{})
            loader.assert_not_called()

    def test_actual_worker_startup_failure_preserves_receipt_and_blocks_retry(self):
        """Control-flow fixture; atomic filesystem replacement is not exercised.

        This sandbox refused os.replace for the earlier temporary progress file.
        Preserve that limitation rather than retry the denied filesystem action.
        The actual exclusive attempt marker still protects the duplicate path.
        """
        import launch
        p=protocol()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'run/evaluation';output.mkdir(parents=True)
            write_json(root/'protocol.json',p)
            receipts={}
            def record(path,value):
                receipts[Path(path).name]=copy.deepcopy(value)
                return {'path':Path(path).name,'sha256':'mocked-control-flow-only','bytes':0}
            with patch.object(worker,'HERE',root),\
                    patch.object(launch,'require_execution_authorization'),\
                    patch.object(worker,'assemble_exports',side_effect=ValueError('synthetic missing export')) as exports,\
                    patch.object(worker,'load_simulator') as simulator,\
                    patch.object(worker,'module') as scientific_loader,\
                    patch.object(worker,'atomic_json',side_effect=record):
                report=worker.run(p,output,dict(timeout_seconds=30,deadline_monotonic=time.monotonic()+30,
                                               jobs_root=str(root/'run')))
                self.assertEqual(report['status'],'failed_central_quartile_evaluation')
                self.assertEqual(report['decision'],'incomplete_no_utility_conclusion')
                self.assertEqual(report['error'],'ValueError: synthetic missing export')
                self.assertTrue(report['execution_counts_known'])
                self.assertEqual(report['ledger']['optimizer_updates'],0)
                self.assertEqual(report['ledger']['source_transition_rows_decoded'],0)
                self.assertEqual(report['completed_snapshot_contexts'],0)
                self.assertEqual(receipts['report.json'],report)
                self.assertEqual(receipts['progress.json']['status'],'running')
                self.assertTrue((output/'EVALUATION_ATTEMPT.json').exists())
                simulator.assert_not_called();scientific_loader.assert_not_called()
                with self.assertRaisesRegex(ValueError,'Existing evaluation evidence'):
                    worker.run(p,output,dict(timeout_seconds=30,deadline_monotonic=time.monotonic()+30,
                                             jobs_root=str(root/'run')))
                self.assertEqual(exports.call_count,1)


if __name__=='__main__': unittest.main()
