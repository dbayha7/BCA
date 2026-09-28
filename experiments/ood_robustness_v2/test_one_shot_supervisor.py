"""Synthetic files/processes only; invented reviews do not accept real execution.

The success fixture explicitly removes CPython's automatically added LC_CTYPE
from its synthetic environment. A separate unmodified-interpreter case MUST
refuse it. This is not an accepted real environment fix or native compatibility.
"""
import copy,io,json,os,signal,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_execution_capsule import Fixture,pin,save
from one_shot_supervisor import (OneShotSupervisor,parse,process_identity,process_argv,
                                process_environment,evidence_path,protocol_file)
from engineering_worker_route import read_line

CODE=Path(__file__).resolve().parent
EVIDENCE=None
if len(sys.argv)==3 and sys.argv[1]=='--evidence-root':
    EVIDENCE=Path(sys.argv[2]);EVIDENCE.mkdir(exist_ok=False);sys.argv=sys.argv[:1]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)
    def tearDown(self):
        # Keep raw small synthetic child/process receipts; never real artifacts.
        root=Path(self.f.spec['paths']['run_root']+'.supervision-v1')
        if EVIDENCE is not None and root.is_dir():
            out=EVIDENCE/self._testMethodName;out.mkdir()
            for p in root.iterdir():
                if p.is_file() and not p.is_symlink():
                    raw=p.read_bytes();self.assertLess(len(raw),2_000_000);(out/p.name).write_bytes(raw)
            entry=Path(self.f.spec['assets']['entrypoint']['path'])
            if entry.is_file():(out/'synthetic_entrypoint.py').write_bytes(entry.read_bytes())
    def worker(self,body="pass",*,normalize_fixture=True):
        entry=Path(self.f.assets['entrypoint']['path'])
        text='import sys,os\nsys.path.insert(0,'+repr(str(CODE))+')\n'
        if normalize_fixture:text+="os.environ.pop('LC_CTYPE',None) # synthetic fixture only\n"
        text+='from engineering_worker_route import WorkerRoute\n'
        text+='def callback(binding,check):\n    '+body.replace('\n','\n    ')+'\n'
        text+='route=WorkerRoute()\nroute.run(callback)\n'
        entry.write_bytes(text.encode());self.f.spec['assets']['entrypoint']=pin(entry)
        interpreter=Path(sys.executable).resolve()
        self.f.assets['interpreter']=pin(interpreter);self.f.spec['assets']['interpreter']=pin(interpreter)
        self.f.spec['command']=[str(interpreter),'-I',str(entry),'--execution-declaration',str(self.f.declaration)]
        self.f.rebuild()
        return entry
    def supervisor(self,**kwargs):
        return OneShotSupervisor(declaration_pin=self.f.declaration_pin,trusted_review=self.f.trusted,timeout_seconds=kwargs.pop('timeout_seconds',10),**kwargs)
    def evidence(self,name='actual_exit.json'):
        return protocol_file(evidence_path(self.f.spec)/name)
    def failure(self,**kwargs):
        s=self.supervisor(**kwargs);self.assertRaises(RuntimeError,s.run);self.assertTrue(s.failed);return self.evidence()
    def test_actual_one_process_handshake(self):
        self.worker("assert binding['permitted_scope']=='engineering/td3_bc/hopper'\ncheck()\nprint('synthetic callback complete')")
        r=self.supervisor().run();self.assertEqual(r['actual_returncode'],0);self.assertTrue(r['route_completed']);self.assertTrue(r['release_sent'])
        self.assertFalse(r['science_accepted']);self.assertFalse(self.f.output.exists())
        self.assertEqual(self.f.ancestor.read_bytes(),b'synthetic; not SQLite');self.assertEqual(self.f.lock.read_bytes(),b'not acquired')
        self.assertIn(b'synthetic callback complete',(evidence_path(self.f.spec)/'stdout').read_bytes())
        observed=self.evidence('observed_ready.json');self.assertEqual(observed['command'],self.f.spec['command'])
    def test_actual_nonzero_23_not_invented_zero(self):
        self.worker('os._exit(23)');r=self.failure();self.assertEqual(r['actual_returncode'],23);self.assertTrue(r['release_sent']);self.assertFalse(r['route_completed'])
        self.assertFalse((evidence_path(self.f.spec)/'route_result.json').exists())
    def test_callback_exception(self):
        self.worker("raise ValueError('synthetic failure')");r=self.failure();self.assertEqual(r['actual_returncode'],1);self.assertTrue(r['release_sent'])
    def test_runtime_timeout_actual_signal(self):
        self.worker('import time\ntime.sleep(20)');r=self.failure(timeout_seconds=2)
        self.assertTrue(r['timeout']);self.assertEqual(r['actual_returncode'],-signal.SIGTERM);self.assertEqual(r['signals_sent'],[signal.SIGTERM])
    def test_effective_locale_addition_refused(self):
        self.worker(normalize_fixture=False);r=self.failure();self.assertEqual(r['actual_returncode'],1);self.assertFalse(r['release_sent'])
        self.assertIn(b'Effective Python environment changed',(evidence_path(self.f.spec)/'stderr').read_bytes())
    def test_held_stream_prevents_spawn(self):
        self.worker();self.f.review['gates']['fresh_streams']=False;self.f.close_review()
        with patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(ValueError,self.supervisor);popen.assert_not_called()
        self.assertFalse(evidence_path(self.f.spec).exists())
    def test_missing_review_prevents_spawn(self):
        self.worker();self.f.actual.unlink()
        with patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(FileNotFoundError,self.supervisor);popen.assert_not_called()
    def test_second_consumption(self):
        self.worker();s=self.supervisor();s.run();self.assertRaises(ValueError,s.run)
    def test_new_supervisor_cannot_reopen_claim(self):
        self.worker();self.supervisor().run();old=(evidence_path(self.f.spec)/'actual_exit.json').read_bytes()
        with patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(RuntimeError,self.supervisor().run);popen.assert_not_called()
        self.assertEqual((evidence_path(self.f.spec)/'actual_exit.json').read_bytes(),old)
    def test_precreated_claim_preserved(self):
        self.worker();root=evidence_path(self.f.spec);root.mkdir();(root/'pending').write_bytes(b'unknown')
        with patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(RuntimeError,self.supervisor().run);popen.assert_not_called()
        self.assertEqual((root/'pending').read_bytes(),b'unknown')
    def test_changed_source_before_launch(self):
        entry=self.worker();s=self.supervisor();entry.write_bytes(b'changed')
        with patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(RuntimeError,s.run);popen.assert_not_called()
    def test_failed_spawn_actual_exit_unknown(self):
        self.worker()
        with patch('one_shot_supervisor.subprocess.Popen',side_effect=OSError('synthetic launch failure')):r=self.failure()
        self.assertIsNone(r['actual_returncode']);self.assertIsNone(r['launched_pid']);self.assertFalse(r['release_sent'])
    def test_durability_failure_before_spawn(self):
        self.worker()
        with patch('one_shot_supervisor.write_once',side_effect=OSError('synthetic fsync failure')),patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(OSError,self.supervisor().run);popen.assert_not_called()
        self.assertTrue(evidence_path(self.f.spec).exists());self.assertFalse(self.f.output.exists())
    def test_changed_input_in_callback(self):
        self.worker("from pathlib import Path\nPath("+repr(self.f.assets['support_bank']['path'])+").write_bytes(b'changed')")
        r=self.failure();self.assertEqual(r['actual_returncode'],1);self.assertFalse(r['route_completed'])
    def test_swallowed_reentry_poisoned(self):
        self.worker('try: route.run(callback)\nexcept ValueError: pass')
        r=self.failure();self.assertEqual(r['actual_returncode'],1);self.assertFalse((evidence_path(self.f.spec)/'route_result.json').exists())
    def test_effective_environment_mutation(self):
        self.worker("os.environ['CUDA_VISIBLE_DEVICES']='0'");r=self.failure();self.assertEqual(r['actual_returncode'],1)
    def test_log_ceiling(self):
        self.worker("print('x'*5000,flush=True)");r=self.failure(max_log_bytes=1000)
        self.assertIn('log ceiling',r['failure']);self.assertFalse(r['route_completed'])
    def test_child_without_handshake(self):
        entry=self.worker();entry.write_bytes(b'pass\n');self.f.spec['assets']['entrypoint']=pin(entry);self.f.rebuild()
        r=self.failure();self.assertEqual(r['actual_returncode'],0);self.assertFalse(r['route_completed']);self.assertFalse(r['release_sent'])
    def test_corrupt_handshake(self):
        entry=self.worker();raw=entry.read_text().replace('route.run(callback)',"route.root.joinpath('route_result.json').write_text('{}')")
        entry.write_text(raw);self.f.spec['assets']['entrypoint']=pin(entry);self.f.rebuild()
        r=self.failure();self.assertEqual(r['actual_returncode'],0);self.assertFalse(r['route_completed'])
    def test_duplicate_protocol_keys(self):self.assertRaises(ValueError,parse,b'{"a":1,"a":2}')
    def test_nonfinite_protocol(self):self.assertRaises(ValueError,parse,b'{"x":NaN}')
    def test_oversized_protocol(self):self.assertRaises(ValueError,parse,b' '*2_000_001)
    def test_truncated_envelope(self):self.assertRaises(ValueError,read_line,io.BytesIO(b'{}'))
    def test_invalid_timeout(self):
        self.worker()
        for v in (0,-1,True,float('nan'),86401):self.assertRaises(ValueError,self.supervisor,timeout_seconds=v)
    def test_invalid_log_bound(self):
        self.worker()
        for v in (0,True,64_000_001):self.assertRaises(ValueError,self.supervisor,max_log_bytes=v)
    def test_supervisor_contract_mutation(self):
        self.worker();s=self.supervisor();s.timeout_seconds=999
        with patch('one_shot_supervisor.subprocess.Popen') as popen:self.assertRaises(RuntimeError,s.run);popen.assert_not_called()
    def test_supervisor_fork_refused(self):
        self.worker();s=self.supervisor()
        with patch('one_shot_supervisor.os.getpid',return_value=s.pid+1):self.assertRaises(ValueError,s.run)
    def test_evidence_cannot_replace_shared_lock(self):
        self.worker();self.f.spec['paths']['lock']=self.f.spec['paths']['run_root']+'.supervision-v1';self.f.rebuild()
        with patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(RuntimeError,self.supervisor().run);popen.assert_not_called()
        self.assertFalse(Path(self.f.spec['paths']['lock']).exists())
    def test_evidence_cannot_contain_shared_lock(self):
        self.worker();self.f.spec['paths']['lock']=self.f.spec['paths']['run_root']+'.supervision-v1/lock';self.f.rebuild()
        with patch('one_shot_supervisor.subprocess.Popen') as popen:
            self.assertRaises(RuntimeError,self.supervisor().run);popen.assert_not_called()
    def test_changed_capsule_contract_before_launch(self):
        self.worker();s=self.supervisor();s.capsule.pins['asset/support_bank']['sha256']='a'*64
        with patch('one_shot_supervisor.subprocess.Popen') as popen:self.assertRaises(RuntimeError,s.run);popen.assert_not_called()
    def test_worker_capsule_contract_mutation(self):
        self.worker("route.capsule.pins['asset/support_bank']['sha256']='a'*64")
        r=self.failure();self.assertEqual(r['actual_returncode'],1);self.assertFalse(r['route_completed'])
    def test_worker_parent_identity_mutation(self):
        self.worker("route.parent['start_ticks']+=1")
        r=self.failure();self.assertEqual(r['actual_returncode'],1);self.assertFalse(r['route_completed'])
    def test_launch_intent_precedes_process(self):
        self.worker();real=__import__('subprocess').Popen
        def observed(*args,**kw):
            intent=self.evidence('intent.json');self.assertEqual(intent['command'],args[0])
            self.assertEqual(kw['env'],self.f.spec['environment']);self.assertTrue(kw['start_new_session']);return real(*args,**kw)
        with patch('one_shot_supervisor.subprocess.Popen',side_effect=observed):self.supervisor().run()
    def test_actual_proc_fields(self):
        p=process_identity(os.getpid());self.assertEqual(p['ppid'],os.getppid());self.assertEqual(p['pgrp'],os.getpgrp());self.assertEqual(p['sid'],os.getsid(0))
        self.assertGreater(p['start_ticks'],0);self.assertTrue(process_argv(os.getpid()));self.assertIsInstance(process_environment(os.getpid()),dict)

if __name__=='__main__':unittest.main()
