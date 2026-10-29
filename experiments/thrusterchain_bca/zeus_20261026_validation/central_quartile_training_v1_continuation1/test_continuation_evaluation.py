"""Synthetic lineage/file fixtures; no model, learner, simulator or WSL calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_worker as worker


PENDING={('seed2_hole','bca'), *(('seed2_random',a) for a in ('native','bca','distance'))}


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True)+'\n').encode();path.write_bytes(raw)
    return dict(path=path.name,sha256=worker.digest(raw),bytes=len(raw))


def counts(step):
    return dict(critic=step,actor=step//2,actor_target=step//2,critic_target=step//2,
                outer_steps=step//2,auxiliary=step,aux_fit_attempts=step,
                refresh_attempts=0 if step==0 else 1 if step==10000 else 2,
                refresh_accepted=0 if step==0 else 1 if step==10000 else 2,phase=0)


def delta(final,initial):
    return {k:final[k]-initial[k] for k in final if k!='phase'}


def receipt(directory,cell,stage,complete=True):
    directory.mkdir(parents=True,exist_ok=True)
    matching=dict(cell=cell,protocol_sha256='synthetic',distance_coefficient=2.5)
    desc=write_json(directory/'matching_receipt.json',matching)
    report=dict(status='completed_central_quartile_'+('warmup' if stage=='warmup' else 'branch'),
        protocol_sha256='synthetic',cell=cell,condition=cell[6:],seed_index=int(cell[4]),
        arm='native' if stage=='warmup' else stage,execution_counts_known=True,
        matching_receipt_sha256=desc['sha256'])
    seal_receipt(directory,report,complete)
    return report


def seal_receipt(directory,report,complete=True):
    desc=write_json(directory/'report.json',report)
    stage='warmup' if report['status'].endswith('warmup') else report['arm']
    write_json(directory/'SUPERVISOR_RESULT.json',dict(job=report['cell']+'/'+stage,
        completed=complete,exit_code=0 if complete else -9,timed_out=False,
        interrupted=not complete,report_sha256=desc['sha256']))


def checkpoint(directory,step,matching_sha):
    folder=directory/'checkpoints'/f'step_{step:05d}';folder.mkdir(parents=True,exist_ok=True)
    def binary(name,raw):
        (folder/name).write_bytes(raw)
        return dict(path=name,sha256=worker.digest(raw),bytes=len(raw))
    manifest=dict(schema='unifloral.central_quartile.checkpoint.v1',protocol_sha256='synthetic',
        cell='seed2_hole',condition='hole',seed_index=2,kind='branch',arm='bca',
        absolute_critic_step=step,complete_optimizer_target_rng_state=True,resume_authorized=False,
        counts=counts(step),matching_receipt_sha256=matching_sha,normalization={'fixture':'common'},
        artifact=binary('full_state.msgpack',f'full state {step}'.encode()),
        actor_params=binary('actor_params.msgpack',f'actor {step}'.encode()) if step!=16000 else None)
    desc=write_json(folder/'manifest.json',manifest)
    return {**desc,'path':f'checkpoints/step_{step:05d}/manifest.json'}


class ContinuationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=worker.HERE)
        self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name);self.current=base/'current';self.prior=base/'prior'
        self.current.mkdir();self.prior.mkdir();self.pins={}
        def authenticate(relative):
            raw=(self.prior/relative).read_bytes()
            worker.require(self.pins.get(relative)==worker.digest(raw),'Prior lineage hash mismatch: '+relative)
            return raw
        def job_directory(cell,stage):
            return (self.current if (cell,stage) in PENDING else self.prior)/'run'/cell/stage
        for obj,name,value in [(worker,'HERE',self.current),
                (worker.recovery_inputs,'prior_root',lambda:self.prior),
                (worker.recovery_inputs,'authenticate_prior_file',authenticate),
                (worker.recovery_inputs,'job_directory',job_directory)]:
            patcher=patch.object(obj,name,value);patcher.start();self.addCleanup(patcher.stop)

    def pin(self):
        self.pins.update({p.relative_to(self.prior).as_posix():worker.sha(p)
                          for p in self.prior.rglob('*') if p.is_file()})

    def resumed(self):
        old=self.prior/'run/seed2_hole/bca';new=self.current/'run/seed2_hole/bca'
        old_report=receipt(old,'seed2_hole','bca',False)
        descriptor15=checkpoint(old,15000,old_report['matching_receipt_sha256'])
        descriptor16=checkpoint(old,16000,old_report['matching_receipt_sha256'])
        report=receipt(new,'seed2_hole','bca')
        for path in (old/'checkpoints/step_15000').iterdir():
            target=new/'checkpoints/step_15000'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
        descriptor20=checkpoint(new,20000,report['matching_receipt_sha256'])
        report.update(initial_counts=counts(10000),segment_initial_counts=counts(16000),final_counts=counts(20000),
            logical_branch_counts_known=True,prior_physical_execution_counts_known=False,
            additional_updates=delta(counts(20000),counts(10000)),
            segment_additional_updates=delta(counts(20000),counts(16000)),
            ledger={'committed_additional_updates':delta(counts(20000),counts(16000))},
            continuation_lineage={'source_job':'seed2_hole/bca','resume_from_step':16000,
                'source_checkpoint':{'path':'run/seed2_hole/bca/'+descriptor16['path'],
                                     'sha256':descriptor16['sha256']},
                'prefix_unique_replay_source_rows':None},
            checkpoint_manifests={'15000':descriptor15,'20000':descriptor20})
        source16=json.loads((old/descriptor16['path']).read_bytes())
        report['continuation_restore_receipt']=dict(status='passed_exact_continuation_restore',
            source_checkpoint=report['continuation_lineage']['source_checkpoint'],counts=counts(16000),
            optimizer_updates=0,calibration_updates=0,simulator_transitions=0,
            exact_state={'comparison':'exact_decoded_state','numeric_tolerances_used':False,
                         'checkpoint_sha256':source16['artifact']['sha256']})
        seal_receipt(new,report);self.pin()
        return old,new,report

    def test_mixed_roots_cover_exactly_32_old_and_four_new_jobs(self):
        seen={}
        def training_seal(directory,cell,stage,protocol_sha):
            seen[(cell,stage)]=Path(directory)
            return {'initial_parameter_hashes':{'actor':cell[4]}},{},{}
        def checkpoint_export(directory,report,spec,protocol_sha):
            return {'manifest':{'normalization':{'shared':True}}},{}
        protocol=worker.read(Path(__file__).parent/'protocol.json')
        with patch.object(worker,'training_seal',side_effect=training_seal),\
                patch.object(worker,'checkpoint_export',side_effect=checkpoint_export):
            exports,_=worker.assemble_exports(protocol,self.current/'run','synthetic')
            self.assertEqual(len(exports),72)
            self.assertEqual(sum(path.is_relative_to(self.prior) for path in seen.values()),32)
            self.assertEqual({key for key,path in seen.items() if path.is_relative_to(self.current)},PENDING)
            with self.assertRaisesRegex(ValueError,'Undeclared training root'):
                worker.assemble_exports(protocol,self.prior/'run','synthetic')

    def test_original_receipts_authenticated_even_when_self_consistently_resealed(self):
        directory=self.prior/'run/seed0_full/warmup'
        report=receipt(directory,'seed0_full','warmup');self.pin()
        worker.training_seal(directory,'seed0_full','warmup','synthetic')
        report['foreign_replacement']=True;seal_receipt(directory,report)
        with self.assertRaisesRegex(ValueError,'Prior lineage hash mismatch'):
            worker.training_seal(directory,'seed0_full','warmup','synthetic')

    def test_foreign_prior_receipt_not_in_lineage_is_rejected(self):
        directory=self.prior/'run/seed0_full/warmup'
        receipt(directory,'seed0_full','warmup')
        with self.assertRaisesRegex(ValueError,'Prior lineage hash mismatch'):
            worker.training_seal(directory,'seed0_full','warmup','synthetic')

    def test_fresh_remaining_branch_accepts_explicit_null_resume_lineage(self):
        directory=self.current/'run/seed2_random/native'
        report=receipt(directory,'seed2_random','native');report['continuation_lineage']=None
        seal_receipt(directory,report)
        worker.training_seal(directory,'seed2_random','native','synthetic')
        report['continuation_lineage']={'source_job':'seed2_hole/bca'};seal_receipt(directory,report)
        with self.assertRaisesRegex(ValueError,'Unexpected continuation lineage'):
            worker.training_seal(directory,'seed2_random','native','synthetic')

    def test_original_incomplete_branch_cannot_be_counted_as_completed(self):
        old,_,_=self.resumed()
        with self.assertRaisesRegex(ValueError,'Incomplete supervisor'):
            worker.training_seal(old,'seed2_hole','bca','synthetic')

    def test_completed_composite_has_exact_segment_counts_and_no_prefix_row_claim(self):
        old,new,report=self.resumed();original=(old/'SUPERVISOR_RESULT.json').read_bytes()
        actual,_,_=worker.training_seal(new,'seed2_hole','bca','synthetic')
        self.assertEqual(actual['additional_updates']['critic'],10000)
        self.assertEqual(actual['segment_additional_updates']['critic'],4000)
        self.assertIsNone(actual['continuation_lineage']['prefix_unique_replay_source_rows'])
        self.assertEqual(original,(old/'SUPERVISOR_RESULT.json').read_bytes())
        bad=copy.deepcopy(report);bad['segment_additional_updates']['critic']=10000;seal_receipt(new,bad)
        with self.assertRaisesRegex(ValueError,'logical/physical update accounting'):
            worker.training_seal(new,'seed2_hole','bca','synthetic')

    def test_missing_or_foreign_resume_lineage_is_rejected(self):
        _,new,report=self.resumed()
        for change in ({},{**report['continuation_lineage'],'source_job':'seed1_hole/bca'},
                       {**report['continuation_lineage'],'source_checkpoint':{'path':'elsewhere','sha256':'wrong'}}):
            bad=copy.deepcopy(report);bad['continuation_lineage']=change;seal_receipt(new,bad)
            with self.assertRaisesRegex(ValueError,'continuation source lineage'):
                worker.training_seal(new,'seed2_hole','bca','synthetic')

    def test_runtime_restore_must_be_exact_and_bound_to_source_state(self):
        _,new,report=self.resumed()
        for key,value in [('numeric_tolerances_used',True),('checkpoint_sha256','foreign')]:
            bad=copy.deepcopy(report);bad['continuation_restore_receipt']['exact_state'][key]=value
            seal_receipt(new,bad)
            with self.assertRaisesRegex(ValueError,'exact runtime restore receipt'):
                worker.training_seal(new,'seed2_hole','bca','synthetic')

    def test_imported_15k_export_requires_original_source_bytes(self):
        _,new,report=self.resumed()
        spec=dict(cell='seed2_hole',condition='hole',seed=2,stage='bca',arm='bca',checkpoint=15000)
        export,_=worker.checkpoint_export(new,report,spec,'synthetic')
        actor_path=export['base']/export['actor_descriptor']['path'];actor_path.write_bytes(b'foreign actor')
        manifest=export['manifest'];manifest['actor_params'].update(sha256=worker.sha(actor_path),bytes=actor_path.stat().st_size)
        desc=write_json(export['base']/'manifest.json',manifest)
        report['checkpoint_manifests']['15000'].update(sha256=desc['sha256'],bytes=desc['bytes'])
        with self.assertRaisesRegex(ValueError,'Imported 15k checkpoint differs'):
            worker.checkpoint_export(new,report,spec,'synthetic')

    def test_final_counts_and_normalization_still_bound_to_restored_source(self):
        _,new,report=self.resumed()
        spec=dict(cell='seed2_hole',condition='hole',seed=2,stage='bca',arm='bca',checkpoint=20000)
        worker.checkpoint_export(new,report,spec,'synthetic')
        bad=copy.deepcopy(report);bad['final_counts']['actor_target']-=1
        with self.assertRaisesRegex(ValueError,'report/final checkpoint counters'):
            worker.checkpoint_export(new,bad,spec,'synthetic')
        path=new/report['checkpoint_manifests']['20000']['path'];manifest=json.loads(path.read_bytes())
        manifest['normalization']={'changed':True};desc=write_json(path,manifest)
        report['checkpoint_manifests']['20000'].update(sha256=desc['sha256'],bytes=desc['bytes'])
        with self.assertRaisesRegex(ValueError,'normalization differs'):
            worker.checkpoint_export(new,report,spec,'synthetic')


if __name__=='__main__':
    unittest.main()
