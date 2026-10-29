"""CPU toy/IO recovery checks; no source dataset or production optimizer calls."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import train_worker as worker


def resumed_job():
    return dict(kind='branch',cell='seed2_hole',condition='hole',seed_index=2,arm='bca',
                resume_from_step=16000,resume_source_job='seed2_hole/bca')


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


class ContinuationBoundaryTests(unittest.TestCase):
    def test_only_exact_approved_resume_has_reduced_budget(self):
        protocol=worker.read(worker.HERE/'protocol.json')
        job=resumed_job()
        self.assertEqual(worker.validate_job(protocol,job),4000)
        ordinary=dict(kind='branch',cell='seed2_random',condition='random',seed_index=2,arm='native')
        self.assertEqual(worker.validate_job(protocol,ordinary),10000)
        for key,value in [('resume_from_step',15000),('resume_from_step',16000.),
                          ('resume_source_job','seed2_random/bca'),('arm','native'),
                          ('seed_index',1),('condition','random')]:
            bad=copy.deepcopy(job);bad[key]=value
            with self.assertRaises(ValueError):worker.validate_job(protocol,bad)
        for key in ('resume_source_job','resume_from_step'):
            bad=copy.deepcopy(job);del bad[key]
            with self.assertRaises(ValueError):worker.validate_job(protocol,bad)

    def test_physical_segment_counts_do_not_repeat_calibration_or_imported_work(self):
        keys=('critic','actor','actor_target','critic_target','outer_steps','auxiliary',
              'aux_fit_attempts','refresh_attempts','refresh_accepted')
        segment=worker.expected_segment_updates(keys,resumed_job(),4000)
        self.assertEqual(tuple(segment.values()),(4000,2000,2000,2000,2000,4000,4000,0,0))
        ordinary=dict(kind='branch',cell='seed2_random',condition='random',seed_index=2,arm='native')
        remaining=worker.expected_segment_updates(keys,ordinary,10000)
        totals={key:segment[key]+3*remaining[key] for key in keys}
        self.assertEqual(totals['critic'],34000)
        self.assertEqual(totals['actor'],17000)
        self.assertEqual(totals['auxiliary'],34000)
        self.assertEqual(totals['refresh_attempts'],3)

    def prefix_fixture(self,output):
        job=resumed_job();prefix='run/seed2_hole/bca/'
        matching=encoded({'distance_coefficient':.25,'anchor_actor_digest':'toy'})
        worker.immutable_bytes(output/'matching_receipt.json',matching)
        matching_sha=worker.digest(matching)
        files={prefix+'matching_receipt.json':matching}
        checkpoints={}
        for step in range(11000,16001,1000):
            path=f'checkpoints/step_{step:05d}/'
            state=f'toy full state {step}'.encode()
            art=dict(path='full_state.msgpack',bytes=len(state),sha256=worker.digest(state))
            actor=b'toy actor' if step==15000 else None
            actor_art=dict(path='actor.msgpack',bytes=len(actor),sha256=worker.digest(actor)) if actor else None
            manifest=dict(protocol_sha256='protocol',cell=job['cell'],condition='hole',seed_index=2,
                kind='branch',arm='bca',absolute_critic_step=step,counts={'critic':step,'phase':0},
                normalization={'fixture':True},matching_receipt_sha256=matching_sha,
                complete_optimizer_target_rng_state=True,artifact=art,actor_params=actor_art,
                resume_authorized=False)
            raw=encoded(manifest)
            files[prefix+path+'manifest.json']=raw
            files[prefix+path+'full_state.msgpack']=state
            if actor:files[prefix+path+'actor.msgpack']=actor
            checkpoints[str(step)]=dict(path=path+'manifest.json',bytes=len(raw),sha256=worker.digest(raw))
        files[prefix+'progress.json']=encoded(dict(job=job,checkpoint_manifests=checkpoints))
        files[prefix+'gradient_audit_015000.json']=encoded(dict(absolute_critic_step=15000,
            protocol_sha256='protocol',cell='seed2_hole',condition='hole',seed_index=2,arm='bca'))
        files[prefix+'refresh_receipt.json']=encoded({'absolute_critic':15000})
        pins={key:worker.digest(raw) for key,raw in files.items()}
        def authenticate(path):
            raw=files[path]
            worker.require(worker.digest(raw)==pins[path],'fixture authentication failure')
            return raw
        return job,files,pins,matching_sha,SimpleNamespace(authenticate_prior_file=authenticate)

    def test_imported_prefix_is_exact_exclusive_and_not_claimed_as_new_work(self):
        with tempfile.TemporaryDirectory(dir=worker.HERE,prefix='fixture_resume_') as tmp:
            output=Path(tmp)
            job,files,pins,matching_sha,api=self.prefix_fixture(output)
            with patch.dict(sys.modules,recovery_inputs=api):
                raw,manifest,index,gradients,lineage=worker.import_resume_prefix(
                    output,job,'protocol',{'fixture':True},matching_sha)
                self.assertEqual(raw,files['run/seed2_hole/bca/checkpoints/step_16000/full_state.msgpack'])
                self.assertEqual(manifest['absolute_critic_step'],16000)
                self.assertEqual(set(index),{'11000','12000','13000','14000','15000','16000'})
                self.assertEqual(set(gradients),{'15000'})
                self.assertIsNone(lineage['prefix_unique_replay_source_rows'])
                self.assertFalse(lineage['prior_interruption_tail_counts_known'])
                self.assertEqual(lineage['source_checkpoint']['path'],
                                 'run/seed2_hole/bca/checkpoints/step_16000/manifest.json')
                for key,raw in files.items():
                    relative=key.removeprefix('run/seed2_hole/bca/')
                    if relative!='progress.json':self.assertEqual((output/relative).read_bytes(),raw)
                self.assertEqual(pins,{key:worker.digest(raw) for key,raw in files.items()})
                with self.assertRaises(FileExistsError):
                    worker.import_resume_prefix(output,job,'protocol',{'fixture':True},matching_sha)

    def test_changed_prior_bytes_or_identity_fail_before_resume(self):
        for corruption in ('bytes','identity'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory(dir=worker.HERE,prefix='fixture_resume_') as tmp:
                output=Path(tmp)
                job,files,pins,matching_sha,api=self.prefix_fixture(output)
                key='run/seed2_hole/bca/checkpoints/step_11000/manifest.json'
                if corruption=='bytes':files[key]+=b' '
                with patch.dict(sys.modules,recovery_inputs=api):
                    with self.assertRaises(ValueError):
                        worker.import_resume_prefix(output,job,'wrong' if corruption=='identity' else 'protocol',
                                                    {'fixture':True},matching_sha)
                self.assertFalse((output/'checkpoints').exists())

    def test_source_orders_only_one_branch_before_restore_and_updates(self):
        tree=ast.parse(Path(worker.__file__).read_text())
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        calls=[node for node in ast.walk(run) if isinstance(node,ast.Call)]
        branches=[node for node in calls if isinstance(node.func,ast.Attribute)
                  and isinstance(node.func.value,ast.Name) and node.func.value.id=='runtime' and node.func.attr=='branch']
        restore=[node for node in calls if isinstance(node.func,ast.Name) and node.func.id=='restore_resume_state']
        steps=[node for node in calls if isinstance(node.func,ast.Attribute)
               and isinstance(node.func.value,ast.Name) and node.func.value.id=='runtime' and node.func.attr=='step']
        self.assertEqual(len(branches),1)
        self.assertEqual(len(restore),1)
        self.assertLess(branches[0].lineno,restore[0].lineno)
        self.assertLess(restore[0].lineno,steps[0].lineno)


class ToyStateRestorationTests(unittest.TestCase):
    def test_post_calibration_restore_is_exact_and_continuation_matches_uninterrupted(self):
        # Existing tiny algebraic model fixture. This never loads source data or
        # the real actor/critic and cannot dispatch production training.
        import test_runtime as toy
        import runtime
        import jax.numpy as jnp
        from flax import serialization
        adapter,state,events=toy.fixture()
        for _ in range(4):state,_=runtime.step(adapter,state)
        anchor=state.microstep.agent_state.actor.params
        template=state._replace(refresh_at=jnp.array([4,6],jnp.int32),critic_budget=jnp.int32(8))
        adapter=runtime.make_adapter(adapter.context,template,'bca',3.,anchor,[4,6],8)
        state=template
        for _ in range(2):state,_=runtime.step(adapter,state)
        before_events=list(events['refresh_counters'])
        raw=serialization.to_bytes(state)
        manifest={'counts':runtime.host(runtime.validate(adapter,state)['counts'])}
        with patch.object(runtime,'branch',side_effect=AssertionError('must not rebranch restored state')):
            restored,receipt=worker.restore_resume_state(runtime,adapter,template,raw,manifest)
        toy.equal_tree(self,state,restored)
        self.assertEqual(receipt['exact_state']['comparison'],'exact_decoded_state')
        self.assertFalse(receipt['exact_state']['numeric_tolerances_used'])
        self.assertEqual(receipt['optimizer_updates'],0)
        self.assertEqual(events['refresh_counters'],before_events)
        for _ in range(2):
            state,left=runtime.step(adapter,state)
            restored,right=runtime.step(adapter,restored)
            toy.equal_tree(self,state,restored)
            self.assertFalse(left['refresh_due'])
            self.assertEqual(left['counts'],right['counts'])
        self.assertEqual(events['refresh_counters'],before_events)
        self.assertEqual(runtime.validate(adapter,restored)['counts']['critic'],8)

    def test_manifest_counter_corruption_rejected_before_any_step(self):
        runtime=SimpleNamespace(restore_exact=lambda a,t,r,diagnostic:t,
            validate=lambda a,s:{'counts':{'critic':16000},'checks':{}},host=lambda x:x)
        with self.assertRaisesRegex(ValueError,'counts differ'):
            worker.restore_resume_state(runtime,object(),object(),b'toy',{'counts':{'critic':16001}})


if __name__=='__main__':unittest.main()
