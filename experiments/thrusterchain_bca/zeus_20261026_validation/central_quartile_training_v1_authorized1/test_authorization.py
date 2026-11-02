"""Release permission, lineage and dispatch fixtures; no real launch/model calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import authorization
import launch


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.root=launch.HERE
        self.p=launch.read(self.root/'protocol.json')

    def test_actual_evidence_and_unchanged_computation_are_eligible(self):
        r=authorization.verify(self.root,self.p)
        self.assertTrue(r['execution_authorized'])
        self.assertEqual(r['total_seconds_cap'],43200)
        self.assertEqual(r['approved_attempts'],1)

    def changed_record(self,relative,mutate):
        real=authorization.read
        def replacement(path):
            value=real(path)
            if str(path).replace('\\','/').endswith(relative): mutate(value)
            return value
        return patch.object(authorization,'read',side_effect=replacement)

    def test_changed_approval_budget_or_scope_rejected(self):
        mutations=[lambda a:a['approval'].update(response='No'),
            lambda a:a.update(approved_attempts=2),lambda a:a.update(automatic_restart=True),
            lambda a:a.update(resume_authorized=True),lambda a:a.update(release_directory='other'),
            lambda a:a['budget'].update(total_timeout_seconds=43201),
            lambda a:a['budget'].update(total_calibrations=63)]
        for mutate in mutations:
            with self.subTest(mutation=mutate),self.changed_record('/AUTHORIZATION.json',mutate):
                with self.assertRaises(ValueError): authorization.verify(self.root,self.p)

    def test_failed_or_unknown_validation_rejected(self):
        for field,value in [('passed',False),('timed_out',True),('execution_counts_known',False),
                            ('prepared_identity_unchanged',False),('exit_code',1),('error','bad')]:
            with self.subTest(field=field),self.changed_record('/VALIDATION_RESULT_3.json',lambda a:a.update({field:value})):
                with self.assertRaisesRegex(ValueError,'not eligible'): authorization.verify(self.root,self.p)

    def test_modified_validation_hash_or_scientific_worker_rejected(self):
        real=authorization.sha
        for filename in ('VALIDATION_RESULT_3.json','runtime_validation_v1/report.json','train_worker.py','runtime.py'):
            with self.subTest(filename=filename),patch.object(authorization,'sha',side_effect=lambda p:'changed' if str(p).replace('\\','/').endswith(filename) else real(p)):
                with self.assertRaises(ValueError): authorization.verify(self.root,self.p)

    def test_missing_authorization_stops_before_reservation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(launch,'HERE',Path(temp)),patch.object(launch,'reserve') as reserve:
            with self.assertRaises(FileNotFoundError): launch.execute(self.p,{})
            reserve.assert_not_called()

    def test_public_preflight_does_not_reserve_or_dispatch(self):
        with patch.object(launch.sys,'argv',['launch.py','--preflight']),patch.object(launch,'verify_prepared',return_value=(self.p,{})), \
             patch.object(launch,'verify_inputs'),patch.object(launch,'reserve') as reserve,patch.object(launch,'execute') as execute, \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(launch.main(),0)
            self.assertTrue(json.loads(out.getvalue())['execution_authorized'])
            reserve.assert_not_called(); execute.assert_not_called()

    def test_worker_token_cannot_create_an_attempt(self):
        with patch.object(launch,'verify_prepared',return_value=(self.p,{})),patch.object(launch,'require_execution_authorization'), \
             patch.object(launch.sys,'platform','linux'),patch.object(launch.sys,'prefix',launch.EXPECTED_ENV),patch.object(launch,'reserve') as reserve:
            with self.assertRaises(FileNotFoundError): launch.worker_entry('forged','seed0_full/warmup')
            reserve.assert_not_called()

    def test_public_execute_routes_to_single_authorized_dispatch(self):
        with patch.object(launch.sys,'argv',['launch.py','--execute']), \
             patch.object(launch,'verify_prepared',return_value=(self.p,{'fixture':True})), \
             patch.object(launch,'verify_inputs'),patch.object(launch.signal,'signal'), \
             patch.object(launch,'execute',return_value=19) as execute:
            self.assertEqual(launch.main(),19)
            execute.assert_called_once_with(self.p,{'fixture':True})

    def test_public_execute_invalid_approval_never_reserves(self):
        with patch.object(launch.sys,'argv',['launch.py','--execute']), \
             patch.object(launch,'verify_prepared',return_value=(self.p,{})), \
             patch.object(launch,'verify_inputs'),patch.object(launch.signal,'signal'), \
             patch.object(launch,'module',return_value=authorization), \
             self.changed_record('/AUTHORIZATION.json',lambda a:a.update(approved_attempts=2)), \
             patch.object(launch,'reserve') as reserve,patch.object(launch,'supervise') as supervise:
            with self.assertRaisesRegex(ValueError,'One-attempt authorization differs'):launch.main()
            reserve.assert_not_called();supervise.assert_not_called()

    def test_known_validator_overlap_stops_launch(self):
        realpath=Path
        for name in ('validate_runtime.py','validation_supervisor.py'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                proc=Path(temp); child=proc/'99999999';child.mkdir()
                (child/'comm').write_text('python\n')
                target=self.root.parent/'central_quartile_training_v1_amendment1'/name
                (child/'cmdline').write_bytes(b'python\0'+str(target).encode()+b'\0--worker\0')
                def path(value): return proc if str(value)=='/proc' else realpath(value)
                with patch.object(launch,'Path',side_effect=path),patch.object(launch.os,'getuid',return_value=child.stat().st_uid,create=True):
                    with self.assertRaisesRegex(ValueError,'Existing scoped BCA process'): launch.scoped_process_check()

    def test_deadline_expiring_during_hash_check_never_dispatches(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='fixture_') as temp:
            root=Path(temp); env=root/'env';env.mkdir();(env/'pyvenv.cfg').write_text('include-system-site-packages = false')
            launch.save_new(root/'protocol.json',self.p);launch.save_new(root/'AUTHORIZATION.json',{'fixture':True})
            clock=[100.]
            def verify(): clock[0]=43301.;return self.p,{}
            with patch.object(launch,'HERE',root),patch.object(launch,'EXPECTED_ENV',str(env)), \
                 patch.object(launch.sys,'platform','linux'),patch.object(launch.sys,'prefix',str(env)), \
                 patch.object(launch,'require_execution_authorization',return_value={'fixture':True}), \
                 patch.object(launch,'scoped_process_check'),patch.object(launch,'verify_inputs'), \
                 patch.object(launch,'verify_prepared',side_effect=verify),patch.object(launch.time,'monotonic',side_effect=lambda:clock[0]), \
                 patch.object(launch,'supervise') as supervise,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.execute(self.p,{}),1)
                supervise.assert_not_called()
            result=launch.read(root/'RUN_RESULT.json')
            self.assertIn('deadline exhausted during identity verification',result['error'])
            self.assertEqual(result['jobs'],[])


if __name__=='__main__':unittest.main()
