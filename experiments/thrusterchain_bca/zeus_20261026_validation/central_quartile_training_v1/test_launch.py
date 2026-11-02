"""Synthetic launcher wiring; no scientific jobs or WSL calls."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import launch


class LaunchTests(unittest.TestCase):
    def protocol(self): return launch.read(launch.HERE / "protocol.json")

    def test_repository_inputs_cover_payload_without_escape(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_') as temp:
            repo=Path(temp); package=repo/'experiments/validation/package'; package.mkdir(parents=True)
            payload=repo/'experiments/payload'; payload.mkdir()
            (payload/'model.bin').write_bytes(b'synthetic fixture')
            (package/'PROTOCOL.md').write_text('synthetic protocol')
            p={'protocol_text_sha256':launch.sha(package/'PROTOCOL.md')}
            launch.save_new(package/'protocol.json',p)
            m={'scientific_execution_during_preparation':False,'files':{},'dependencies':{},
               'inputs':{'experiments/payload/model.bin':launch.sha(payload/'model.bin')}}
            launch.save_new(package/'PREPARED.json',m)
            with patch.object(launch,'REPO',repo), patch.object(launch,'validate_protocol',side_effect=lambda v:v):
                self.assertEqual(launch.verify_prepared(package),(p,m))
                (payload/'model.bin').write_bytes(b'changed fixture')
                with self.assertRaisesRegex(ValueError,'Repository input changed'):
                    launch.verify_prepared(package)

    def test_repository_input_absolute_or_parent_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_') as temp:
            repo=Path(temp); package=repo/'package'; package.mkdir()
            m={'scientific_execution_during_preparation':False,'files':{},'dependencies':{},
               'inputs':{'../outside.bin':'fake'}}
            launch.save_new(package/'PREPARED.json',m)
            with patch.object(launch,'REPO',repo):
                with self.assertRaisesRegex(ValueError,'escaped'): launch.verify_prepared(package)
                m['inputs']={str((repo/'outside.bin').resolve()):'fake'}
                launch.atomic_json(package/'PREPARED.json',m)
                with self.assertRaisesRegex(ValueError,'escaped'): launch.verify_prepared(package)

    def test_review_hold_cannot_be_bypassed_by_flags_or_worker_token(self):
        forged={'execution_authorized':True,'execution_status':'approved_for_one_attempt'}
        with patch.object(launch,'reserve') as reserve, patch.object(launch,'verify_prepared') as verify:
            with self.assertRaisesRegex(ValueError,'disabled'):
                launch.execute(forged,{})
            with self.assertRaisesRegex(ValueError,'disabled'):
                launch.worker_entry('invented-token','seed0_full/warmup')
            reserve.assert_not_called();verify.assert_not_called()

    def test_fixed_protocol_and_job_budget(self):
        p = self.protocol()
        self.assertIs(launch.validate_protocol(p),p)
        jobs = launch.job_specs(p)
        self.assertEqual([j["kind"] for j in jobs], ["warmup"]*9 + ["branch"]*27 + ["evaluation"])
        self.assertEqual(len({j["id"] for j in jobs}),37)
        self.assertEqual(sum(10000 if j["kind"]=="warmup" else 10000 for j in jobs[:-1]),360000)
        for key,value in (("total_timeout_seconds",43201),("bca_kappa",2.),("automatic_restart",True),
                          ("total_critic_updates",360001),("checkpoint_every",500)):
            changed = copy.deepcopy(p); changed[key]=value
            with self.assertRaises(ValueError): launch.validate_protocol(changed)

    def test_reservation_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            root = Path(temp)
            launch.reserve(root,"once")
            launch.save_new(root/"run/failed.json",{"status":"synthetic"})
            with self.assertRaises((ValueError,FileExistsError)): launch.reserve(root,"again")
            self.assertTrue((root/"run/failed.json").exists())

    def test_partial_reservation_blocks_reentry(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            root = Path(temp)
            launch.save_new(root/"ATTEMPT.json",{"token":"old"})
            with self.assertRaises(FileExistsError): launch.reserve(root,"new")

    def warmup_report(self,job):
        i = job["seed_index"]
        return {"status":"completed_central_quartile_warmup", "initial_parameter_hashes":
            {"actor":"actor"+str(i),"critic":"critic"+str(i),"auxiliary_scale":"scale"+str(i)},
            'cell':job['cell'],'condition':job['condition'],'seed_index':i,
            "initial_optimizer_steps":{"actor":0,"critic":0,"scale":0},
            "initial_optimizer_hashes":{"actor":"opt_actor","critic":"opt_critic","scale":"opt_scale"},
            "execution_counts_known":True}

    def sequence(self,root,fail_id=None,wrong_pair=False,corrupt_eval=False,interrupt_id=None):
        p = self.protocol()
        launch.save_new(root/"protocol.json",p)
        env = root/"synthetic_env"; env.mkdir()
        (env/"pyvenv.cfg").write_text("include-system-site-packages = false\n")
        called=[]
        def fake_supervise(command,output,cap,*args,**kwargs):
            dispatch=launch.read(Path(output)/"DISPATCH.json")
            job=dispatch["job"]; called.append(job["id"])
            self.assertLessEqual(cap,p[job["kind"]+"_timeout_seconds"])
            self.assertEqual(kwargs["deadline"],dispatch["deadline_monotonic"])
            failed=job["id"]==fail_id
            interrupted=job['id']==interrupt_id
            report=self.warmup_report(job) if job["kind"]=="warmup" else {
                "status":launch.expected_status(job),"execution_counts_known":True}
            if job['kind']=='warmup':
                report['protocol_sha256'] = launch.sha(root/'protocol.json')
                for name,value in (
                    ('actor_manifest',{'cell':job['cell'],'normalization':{'mean':[0.]*14,'std':[1.]*14,'fixture_digest':'fixture','rows':60924,'source':'common'}}),
                    ('data_receipt',{'condition':job['condition'],'common_normalization':{'source_row_ids_sha256':'rows','observation_bytes_sha256':'observations','rows':60924}})):
                    launch.save_new(Path(output)/(name+'.json'),value)
                    report[name+'_sha256']=launch.sha(Path(output)/(name+'.json'))
            if wrong_pair and job["id"]=="seed0_hole/warmup":
                report["initial_parameter_hashes"]["actor"]="mismatch"
            if failed: report["status"]="synthetic_failure"
            if interrupted:
                launch.save_new(Path(output)/'progress.json',{'ledger':{'critic':20},'execution_counts_known':False})
            elif job["kind"]=="evaluation" and corrupt_eval:
                (Path(output)/"report.json").write_text('{"status":')
            else: launch.save_new(Path(output)/"report.json",report)
            return {"exit_code":-9 if interrupted else 1 if failed else 0,"timed_out":False,
                    "interrupted":interrupted,"error":'synthetic interruption' if interrupted else None,"elapsed_seconds":.001}
        with patch.object(launch,"HERE",root), patch.object(launch,"EXPECTED_ENV",str(env)), \
             patch.object(launch.sys,"platform","linux"), patch.object(launch.sys,"prefix",str(env)), \
             patch.object(launch,"require_execution_authorization"), patch.object(launch,"scoped_process_check"), patch.object(launch,"verify_prepared",return_value=(p,{})), \
             patch.object(launch,"verify_inputs",return_value={}), \
             patch.object(launch,"supervise",side_effect=fake_supervise), \
             patch.object(launch,"verify_warmup_pairs",side_effect=lambda p: self.real_pair_check(p,root)), \
             contextlib.redirect_stdout(io.StringIO()):
            code=launch.execute(p,{})
        return code,called,launch.read(root/"RUN_RESULT.json")

    real_pair_check=staticmethod(launch.verify_warmup_pairs)

    def test_all_success_and_terminal_progress(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            root=Path(temp)
            code,called,report=self.sequence(root)
            self.assertEqual(code,0)
            self.assertEqual(len(called),37)
            self.assertEqual(called[-1],"evaluation")
            self.assertTrue(report["all_declared_jobs_completed"])
            self.assertEqual(launch.read(root/"PROGRESS.json")["status"],report["status"])
            self.assertEqual(launch.read_status(root)["status"],report["status"])

    def test_failed_warmup_never_branches_or_evaluates(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            code,called,report=self.sequence(Path(temp),fail_id="seed0_hole/warmup")
            self.assertEqual(code,1); self.assertEqual(len(called),2)
            self.assertFalse(report["all_declared_jobs_completed"])
            self.assertEqual(report["status"],"stopped_after_incomplete_job")

    def test_pair_mismatch_stops_before_any_branch(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            code,called,report=self.sequence(Path(temp),wrong_pair=True)
            self.assertEqual(code,1); self.assertEqual(len(called),9)
            self.assertIn("Initial networks differ",report["error"])

    def test_partial_evaluation_report_retains_terminal_evidence(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            code,called,report=self.sequence(Path(temp),corrupt_eval=True)
            self.assertEqual(code,1); self.assertEqual(len(called),37)
            self.assertIn("JSONDecodeError",report["evaluation_report_read_error"])
            self.assertFalse(report["execution_counts_known"])

    def test_interruption_keeps_lower_bound_and_stops_without_retry(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_') as temp:
            code,called,report=self.sequence(Path(temp),interrupt_id='seed0_full/warmup')
            self.assertEqual(code,1);self.assertEqual(called,['seed0_full/warmup'])
            self.assertFalse(report['execution_counts_known'])
            self.assertFalse(report['resume_supported'])
            self.assertTrue(report['no_automatic_restart'])
            self.assertEqual(report['jobs'][0]['ledger'],{'critic':20})
            self.assertEqual(report['jobs'][0]['ledger_source'],'last_progress_lower_bound')

    def test_status_is_readonly_and_tolerates_partial_progress(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            root=Path(temp)
            self.assertEqual(launch.read_status(root)["status"],"not_started")
            launch.reserve(root,"fixture")
            (root/"run/cell/branch").mkdir(parents=True)
            launch.atomic_json(root/"PROGRESS.json",{"status":"running","current_job":"cell/branch"})
            (root/"run/cell/branch/progress.json").write_text('{"status":')
            result=launch.read_status(root)
            self.assertIn("JSONDecodeError",result["progress_read_error"])
            self.assertIn("uncertain",result["note"])


if __name__=="__main__": unittest.main()
