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
    def setUp(self):
        # Original sequence fixtures isolate supervision, never actual /proc.
        for name, value in (("require_clearance", {"matching_processes": []}),
                            ("accounting", {"attempt_ordinal": 2, "prior_attempt_final_counts_known": False})):
            fixture = patch.object(launch.retry_gate, name, return_value=value)
            fixture.start()
            self.addCleanup(fixture.stop)

    def protocol(self): return launch.read(launch.HERE / "protocol.json")

    def test_fixed_protocol_and_job_budget(self):
        p = self.protocol()
        self.assertIs(launch.validate_protocol(p),p)
        jobs = launch.job_specs(p)
        self.assertEqual([j["kind"] for j in jobs], ["warmup"]*4 + ["branch"]*12 + ["evaluation"])
        self.assertEqual(len({j["id"] for j in jobs}),17)
        self.assertEqual(sum(10000 if j["kind"]=="warmup" else 2000 for j in jobs[:-1]),64000)
        for key,value in (("total_timeout_seconds",9001),("bca_kappa",2.),("automatic_restart",True),
                          ("total_critic_updates",64001),("evaluation_episodes",384)):
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
        return {"status":"completed_support_warmup", "initial_parameter_hashes":
            {"actor":"actor"+str(i),"critic":"critic"+str(i),"auxiliary_scale":"scale"+str(i)},
            "initial_optimizer_steps":{"actor":0,"critic":0,"scale":0},"execution_counts_known":True}

    def sequence(self,root,fail_id=None,wrong_pair=False,corrupt_eval=False):
        p = self.protocol()
        launch.save_new(root/"protocol.json",p)
        env = root/"synthetic_env"; env.mkdir()
        (env/"pyvenv.cfg").write_text("include-system-site-packages = false\n")
        called=[]
        def fake_supervise(command,output,cap,*args,**kwargs):
            self.assertEqual(Path(command[3]), root / "launch.py")
            dispatch=launch.read(Path(output)/"DISPATCH.json")
            job=dispatch["job"]; called.append(job["id"])
            self.assertLessEqual(cap,p[job["kind"]+"_timeout_seconds"])
            self.assertEqual(kwargs["deadline"],dispatch["deadline_monotonic"])
            failed=job["id"]==fail_id
            report=self.warmup_report(job) if job["kind"]=="warmup" else {
                "status":launch.expected_status(job),"execution_counts_known":True}
            if wrong_pair and job["id"]=="seed0_restricted/warmup":
                report["initial_parameter_hashes"]["actor"]="mismatch"
            if failed: report["status"]="synthetic_failure"
            if job["kind"]=="evaluation" and corrupt_eval:
                (Path(output)/"report.json").write_text('{"status":')
            else: launch.save_new(Path(output)/"report.json",report)
            return {"exit_code":1 if failed else 0,"timed_out":False,"interrupted":False,"error":None,"elapsed_seconds":.001}
        with patch.object(launch,"HERE",root), patch.object(launch,"EXPECTED_ENV",str(env)), \
             patch.object(launch.sys,"platform","linux"), patch.object(launch.sys,"prefix",str(env)), \
             patch.object(launch,"scoped_process_check"), patch.object(launch,"verify_prepared",return_value=(p,{})), \
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
            self.assertEqual(len(called),17)
            self.assertEqual(called[-1],"evaluation")
            self.assertTrue(report["all_declared_jobs_completed"])
            self.assertEqual(launch.read(root/"PROGRESS.json")["status"],report["status"])
            self.assertEqual(launch.read_status(root)["status"],report["status"])

    def test_failed_warmup_never_branches_or_evaluates(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            code,called,report=self.sequence(Path(temp),fail_id="seed0_restricted/warmup")
            self.assertEqual(code,1); self.assertEqual(len(called),2)
            self.assertFalse(report["all_declared_jobs_completed"])
            self.assertEqual(report["status"],"stopped_after_incomplete_job")

    def test_pair_mismatch_stops_before_any_branch(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            code,called,report=self.sequence(Path(temp),wrong_pair=True)
            self.assertEqual(code,1); self.assertEqual(len(called),4)
            self.assertIn("Initial networks differ",report["error"])

    def test_partial_evaluation_report_retains_terminal_evidence(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix="fixture_") as temp:
            code,called,report=self.sequence(Path(temp),corrupt_eval=True)
            self.assertEqual(code,1); self.assertEqual(len(called),17)
            self.assertIn("JSONDecodeError",report["evaluation_report_read_error"])
            self.assertFalse(report["execution_counts_known"])

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
