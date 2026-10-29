"""Execute/ownership fixtures with fake supervisors; no subprocesses or models."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch


class ContinuationLaunchTests(unittest.TestCase):
    def protocol(self):
        return launch.read(Path(launch.__file__).parent/'protocol.json')

    def recovery(self):
        return launch.read(Path(launch.__file__).parent/'RECOVERY.json')

    def setup_files(self,root):
        p=self.protocol()
        launch.save_new(root/'protocol.json',p)
        launch.save_new(root/'AUTHORIZATION.json',{'fixture_only':True})
        launch.save_new(root/'RECOVERY.json',self.recovery())
        env=root/'synthetic_env';env.mkdir()
        (env/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        return p,env

    def sequence(self,root,*,interrupted=None,cap=14400,hash_overrun_at=None):
        p,env=self.setup_files(root)
        r=self.recovery()
        clock=[1000.]
        calls=[];verify_calls=[]
        def verify():
            verify_calls.append(clock[0])
            if len(verify_calls)==hash_overrun_at:clock[0]+=cap+1
            return p,{}
        def supervise(command,output,worker_cap,envvars,job_id,**kwargs):
            self.assertIn('--worker-token',command)
            self.assertEqual(command[-1],job_id)
            dispatch=launch.read(output/'DISPATCH.json')
            job=dispatch['job'];calls.append(job_id)
            self.assertEqual(kwargs['deadline'],dispatch['deadline_monotonic'])
            self.assertLessEqual(worker_cap,p[job['kind']+'_timeout_seconds'])
            self.assertLessEqual(worker_cap,1000+cap-clock[0])
            self.assertLessEqual(dispatch['deadline_monotonic'],1000+cap)
            clock[0]+=1
            if job_id==interrupted:
                launch.save_new(output/'progress.json',{'ledger':{'committed_additional_updates':
                    {'critic':20,'actor':10,'auxiliary':20,'refresh_accepted':0}}})
                return dict(exit_code=-9,timed_out=False,interrupted=True,error='synthetic stop',elapsed_seconds=1)
            if job['kind']=='branch':
                resumed=job.get('resume_from_step')==16000
                n=4000 if resumed else 10000
                ledger={'committed_additional_updates':dict(critic=n,actor=n//2,auxiliary=n,
                    refresh_accepted=0 if resumed else 1)}
                # Deliberately expose logical 10k totals: the supervisor must
                # account from physical segment ledger, never these fields.
                report=dict(status=launch.expected_status(job),execution_counts_known=True,
                    additional_updates=dict(critic=10000,actor=5000,auxiliary=10000,refresh_accepted=1),ledger=ledger)
            else:
                report=dict(status=launch.expected_status(job),execution_counts_known=True,
                    decision='synthetic_fixture_only',counts={'episodes':4608,'transitions':1975304})
            launch.save_new(output/'report.json',report)
            return dict(exit_code=0,timed_out=False,interrupted=False,error=None,elapsed_seconds=1)
        with contextlib.ExitStack() as stack:
            for target,name,value in (
                (launch,'HERE',root),(launch,'EXPECTED_ENV',str(env)),
                (launch.sys,'platform','linux'),(launch.sys,'prefix',str(env))):
                stack.enter_context(patch.object(target,name,value))
            for target,name,kwargs in (
                (launch,'require_execution_authorization',{'return_value':{'fixture_only':True}}),
                (launch,'scoped_process_check',{'return_value':None}),
                (launch,'verify_prepared',{'side_effect':verify}),
                (launch,'verify_inputs',{'return_value':{}}),
                (launch,'verify_warmup_pairs',{'return_value':{'paired':True}}),
                (launch,'supervise',{'side_effect':supervise}),
                (launch.recovery,'verify_prior',{'return_value':r}),
                (launch.recovery,'recovery',{'return_value':r}),
                (launch.recovery,'recovery_seconds',{'return_value':cap}),
                (launch.shutil,'disk_usage',{'return_value':SimpleNamespace(free=10**15)}),
                (launch.time,'monotonic',{'side_effect':lambda:clock[0]}),
                (launch.time,'time',{'side_effect':lambda:clock[0]})):
                stack.enter_context(patch.object(target,name,**kwargs))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code=launch.execute(p,{})
        return code,calls,launch.read(root/'RUN_RESULT.json'),verify_calls

    def test_success_four_jobs_then_evaluation_and_physical_lower_bound(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_cont_launch_') as temp:
            root=Path(temp);code,calls,result,_=self.sequence(root)
            self.assertEqual(code,0)
            self.assertEqual(calls,['seed2_hole/bca','seed2_random/distance','seed2_random/native','seed2_random/bca','evaluation'])
            self.assertTrue(result['all_declared_jobs_completed'])
            self.assertTrue(result['prepared_identity_unchanged'])
            self.assertTrue(result['execution_counts_known'])
            self.assertFalse(result['cumulative_execution_counts_known'])
            counts=result['physical_accounting']
            self.assertEqual(counts['new_committed_counts'],dict(critic=34000,actor=17000,auxiliary=34000,calibrations=3))
            self.assertEqual(counts['cumulative_recorded_lower_bound'],
                             dict(critic=360600,actor=180300,auxiliary=360600,calibrations=36))
            self.assertFalse(counts['cumulative_counts_exact'])
            self.assertTrue(counts['new_counts_exact'])
            self.assertEqual(launch.read(root/'PROGRESS.json')['status'],result['status'])
            self.assertEqual(launch.read(root/'DEADLINE.json')['seconds_cap'],14400)

    def test_interrupted_first_worker_stops_without_retry_and_keeps_lower_bound(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_cont_launch_') as temp:
            code,calls,result,_=self.sequence(Path(temp),interrupted='seed2_hole/bca')
            self.assertEqual(code,1)
            self.assertEqual(calls,['seed2_hole/bca'])
            self.assertEqual(result['status'],'stopped_after_incomplete_job')
            self.assertTrue(result['no_automatic_restart'])
            self.assertFalse(result['execution_counts_known'])
            self.assertFalse(result['resume_supported'])
            self.assertEqual(result['jobs'][0]['ledger_source'],'last_progress_lower_bound')
            self.assertEqual(result['physical_accounting']['new_committed_counts']['critic'],20)
            self.assertEqual(result['physical_accounting']['cumulative_recorded_lower_bound']['critic'],326620)

    def test_identity_hash_time_counts_before_dispatch(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_cont_launch_') as temp:
            code,calls,result,_=self.sequence(Path(temp),cap=20,hash_overrun_at=1)
            self.assertEqual(code,1)
            self.assertEqual(calls,[])
            self.assertFalse(result['all_declared_jobs_completed'])
            self.assertIn('deadline',result['error'].lower())

    def test_final_identity_hash_time_cannot_report_success_after_cap(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_cont_launch_') as temp:
            code,calls,result,_=self.sequence(Path(temp),cap=20,hash_overrun_at=6)
            self.assertEqual(len(calls),5)
            self.assertGreater(result['elapsed_seconds'],result['total_seconds_cap'])
            self.assertEqual(code,1)
            self.assertEqual(result['status'],'completed_work_postcheck_exceeded_deadline')
            self.assertTrue(result['deadline_exceeded_during_readonly_postcheck'])

    def test_physical_accounting_rejects_overrun_and_noninteger_counts(self):
        for value in (34001,True,-1,3.5):
            jobs=[dict(job='seed2_hole/bca',ledger={'committed_additional_updates':{'critic':value}})]
            with self.subTest(value=value),self.assertRaises(ValueError):launch.physical_accounting(jobs)

    def test_completed_job_requires_its_exact_physical_segment_ledger(self):
        for ledger in ({},{'committed_additional_updates':{'critic':4000}},
                       {'committed_additional_updates':dict(critic=10000,actor=5000,auxiliary=10000,refresh_accepted=1)}):
            with self.subTest(ledger=ledger),self.assertRaisesRegex(ValueError,'physical segment ledger'):
                launch.physical_accounting([dict(job='seed2_hole/bca',completed=True,
                    execution_counts_known=True,ledger=ledger)])

    def test_reservation_is_exclusive_and_preserves_interruption_evidence(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_cont_launch_') as temp:
            root=Path(temp)
            launch.reserve(root,'once',1000.)
            launch.save_new(root/'run/interruption.json',{'unknown_tail':True})
            original=(root/'ATTEMPT.json').read_bytes()
            with self.assertRaises((ValueError,FileExistsError)):launch.reserve(root,'twice',2000.)
            self.assertEqual((root/'ATTEMPT.json').read_bytes(),original)
            self.assertTrue((root/'run/interruption.json').is_file())

    def test_worker_ownership_prevents_model_loading(self):
        for wrong in ('token','parent','terminal','dispatch'):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_cont_launch_') as temp:
                root=Path(temp);p,env=self.setup_files(root)
                launch.save_new(root/'ATTEMPT.json',{'token':'owned','supervisor_pid':123})
                output=root/'run/seed2_hole/bca';output.mkdir(parents=True)
                launch.save_new(output/'DISPATCH.json',{'token':'other' if wrong=='dispatch' else 'owned',
                    'timeout_seconds':100,'deadline_monotonic':200})
                if wrong=='terminal':launch.save_new(root/'RUN_RESULT.json',{'status':'ended'})
                with patch.object(launch,'HERE',root),patch.object(launch,'EXPECTED_ENV',str(env)), \
                    patch.object(launch.sys,'platform','linux'),patch.object(launch.sys,'prefix',str(env)), \
                    patch.object(launch,'verify_prepared',return_value=(p,{})), \
                    patch.object(launch,'require_execution_authorization',return_value={}), \
                    patch.object(launch.os,'getppid',return_value=999 if wrong=='parent' else 123), \
                    patch.object(launch,'module') as loader:
                    with self.assertRaises(ValueError):
                        launch.worker_entry('bad' if wrong=='token' else 'owned','seed2_hole/bca')
                    loader.assert_not_called()
                self.assertFalse((output/'WORKER_ENTRY.json').exists())


if __name__=='__main__':unittest.main()
