import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_import_guard_v2 import NativeImportGuard,ImportRefusal,environment,pin


class Fixture:
    def __init__(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for n,b in [('builder.py',b'def build():\n    raise RuntimeError("BODY EXECUTED")\ndef cached():\n    return 7\n'),('cache.so',b'fake cache'),('lock',b'')]:
            (self.root/n).write_bytes(b)
        self.bindings={role:pin(self.root/n) for role,n in [('builder','builder.py'),('cache','cache.so'),('lock','lock')]}
        self.e=environment();self.pid=[100]
    def guard(self):return NativeImportGuard(self.bindings,effective_reader=lambda:dict(self.e),kernel_reader=lambda:environment(),pid_reader=lambda:self.pid[0])


class PureTests(unittest.TestCase):
    def setUp(self):self.f=Fixture();self.addCleanup(self.f.temp.cleanup)
    def test_exact_profile(self):self.assertEqual(len(environment()),14);self.f.guard().check()
    def test_startup_fields_explicit(self):
        self.assertEqual({k:environment()[k] for k in ('OPENBLAS_MAIN_FREE','GOTOBLAS_MAIN_FREE')},
                         {'OPENBLAS_MAIN_FREE':'1','GOTOBLAS_MAIN_FREE':'1'})
    def test_missing_openblas_startup_field(self):
        del self.f.e['OPENBLAS_MAIN_FREE']
        with self.assertRaises(ImportRefusal):self.f.guard()
    def test_changed_gotoblas_startup_field(self):
        self.f.e['GOTOBLAS_MAIN_FREE']='0'
        with self.assertRaises(ImportRefusal):self.f.guard()
    def test_first_reason_retained(self):
        g=self.f.guard();g.active=True
        with self.assertRaisesRegex(ImportRefusal,'Environment mutation refused'):
            g.audit('os.putenv',(b'NEW_FIELD',b'1'))
        first=dict(g.first_refusal)
        for _ in range(2):
            with self.assertRaisesRegex(ImportRefusal,'Environment mutation refused'):g.check()
        self.assertEqual(first,g.first_refusal)
        self.assertEqual(first['details'],dict(event='os.putenv',key='NEW_FIELD',value='1'))
    def test_extra_environment(self):
        self.f.e['HOME']='/tmp'
        with self.assertRaises(ImportRefusal):self.f.guard()
    def test_missing_environment(self):
        del self.f.e['LC_CTYPE']
        with self.assertRaises(ImportRefusal):self.f.guard()
    def test_changed_environment(self):
        self.f.e['MUJOCO_PY_FORCE_CPU']='0'
        with self.assertRaises(ImportRefusal):self.f.guard()
    def test_fork_poison(self):
        g=self.f.guard();self.f.pid[0]+=1
        with self.assertRaises(ImportRefusal):g.check()
        self.f.pid[0]-=1
        with self.assertRaises(ImportRefusal):g.check()
    def test_stale_source(self):
        (self.f.root/'builder.py').write_text('pass')
        with self.assertRaises(ImportRefusal):self.f.guard()
    def test_changed_cache(self):
        g=self.f.guard();(self.f.root/'cache.so').write_bytes(b'changed')
        with self.assertRaises(ImportRefusal):g.check()
    def test_changed_lock(self):
        g=self.f.guard();(self.f.root/'lock').write_text('changed')
        with self.assertRaises(ImportRefusal):g.check()
    def test_policy_mutation(self):
        g=self.f.guard();g.bindings['cache']['sha256']='0'*64
        with self.assertRaises(ImportRefusal):g.check()
    def test_role_alias(self):
        self.f.bindings['cache']=self.f.bindings['lock']
        with self.assertRaises(ImportRefusal):self.f.guard()
    def test_symlink(self):
        link=self.f.root/'link';link.symlink_to(self.f.root/'cache.so')
        with self.assertRaises(ImportRefusal):pin(link)
    def test_hardlink(self):
        os.link(self.f.root/'cache.so',self.f.root/'link')
        with self.assertRaises(ImportRefusal):pin(self.f.root/'cache.so')
    def test_phase_skip(self):
        with self.assertRaises(ImportRefusal):self.f.guard().advance('numpy')
    def test_phase_repeat(self):
        g=self.f.guard();g.advance('python')
        with self.assertRaises(ImportRefusal):g.advance('python')
    def test_premature_complete(self):
        with self.assertRaises(ImportRefusal):self.f.guard().complete()


class HookTests(unittest.TestCase):
    def run_case(self,case):
        cmd=[sys.executable,'-I',str(Path(__file__).resolve()),'fixture',case]
        p=subprocess.run(cmd,env=environment(),capture_output=True,timeout=20)
        print(json.dumps(dict(command=cmd,actual_exit=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode())),file=sys.stderr)
        self.assertEqual(p.returncode,0,(p.stdout+p.stderr).decode())
        report=json.loads(p.stdout);self.assertEqual(report['case'],case);self.assertTrue(report['verified'])
    def test_rebuild_before_body(self):self.run_case('rebuild')
    def test_subprocess_before_launch(self):self.run_case('subprocess')
    def test_file_write_before_creation(self):self.run_case('write')
    def test_remove_before_mutation(self):self.run_case('remove')
    def test_environment_change(self):self.run_case('environment')
    def test_startup_field_removal(self):self.run_case('startup_remove')
    def test_startup_field_change(self):self.run_case('startup_change')
    def test_package_lock_only(self):self.run_case('lock')
    def test_profile_loss(self):self.run_case('profile_loss')
    def test_changed_live_code(self):self.run_case('changed_code')
    def test_complete_once(self):self.run_case('complete')


def fixture(case):
    import fcntl
    f=Fixture();g=NativeImportGuard(f.bindings);g.advance('python')
    namespace={};exec(compile((f.root/'builder.py').read_bytes(),str(f.root/'builder.py'),'exec'),namespace)
    if case=='changed_code':exec(compile('def cached():\n    return 99\n',str(f.root/'builder.py'),'exec'),namespace)
    refused=False;result=None;exception_message=None;g.install()
    try:
        if case=='rebuild':namespace['build']()
        elif case=='changed_code':namespace['cached']()
        elif case=='subprocess':subprocess.run([sys.executable,'-c','raise RuntimeError("CHILD EXECUTED")'])
        elif case=='write':(f.root/'forbidden').write_text('wrong')
        elif case=='remove':(f.root/'cache.so').unlink()
        elif case=='environment':os.environ['UNDECLARED']='1'
        elif case=='startup_remove':del os.environ['GOTOBLAS_MAIN_FREE']
        elif case=='startup_change':os.environ['OPENBLAS_MAIN_FREE']='0'
        elif case=='lock':
            try:os.makedirs(f.root)
            except FileExistsError:pass
            with open(f.root/'lock','a') as handle:
                fcntl.lockf(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.lockf(handle,fcntl.LOCK_UN)
            assert namespace['cached']()==7
        elif case=='profile_loss':sys.setprofile(None)
        if case in ('lock','complete','profile_loss'):
            for p in ('numpy','jax','cpu_backend','mujoco_py'):g.advance(p)
            result=g.complete()
    except ImportRefusal as e:refused=True;exception_message=str(e)
    finally:g.active=False;sys.setprofile(None)
    if case in ('lock','complete'):
        assert not refused and result['native_import_route_completed']
        try:g.complete()
        except ImportRefusal:pass
        else:raise AssertionError('Second completion allowed')
    else:assert refused and g.failed
    if case in ('environment','startup_remove','startup_change'):
        key={'environment':'UNDECLARED','startup_remove':'GOTOBLAS_MAIN_FREE','startup_change':'OPENBLAS_MAIN_FREE'}[case]
        expected='Environment removal refused.' if case=='startup_remove' else 'Environment mutation refused.'
        assert exception_message==expected and g.first_refusal['message']==expected
        assert g.first_refusal['details']['key']==key
        first=dict(g.first_refusal)
        for _ in range(2):
            try:g.check()
            except ImportRefusal as e:assert str(e)==expected
            else:raise AssertionError('Poison did not persist')
        assert g.first_refusal==first and dict(os.environ)==environment()
    assert not (f.root/'forbidden').exists() and (f.root/'cache.so').read_bytes()==b'fake cache'
    assert (f.root/'lock').read_bytes()==b'' and 'UNDECLARED' not in os.environ
    f.temp.cleanup();print(json.dumps(dict(case=case,verified=True,first_refusal=g.first_refusal)))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':fixture(sys.argv[2])
    else:unittest.main()
