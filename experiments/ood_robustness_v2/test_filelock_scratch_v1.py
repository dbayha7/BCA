import hashlib,json,os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from filelock_scratch_v1 import FilelockScratch,ScratchRefusal,codes,source_pin

STRICT=Path('/home/dbayha/miniconda3/envs/corl-orig-local/lib/python3.10/site-packages/filelock/_strict.py')
EXPECTED_STRICT='379d37ee7580cf6005e7d2e5c8f32c3a0e08c062f5dc3eb1eb88614049f7bd04'


def fixture(case,base):
    base=Path(base);root=base/'scratch';root.mkdir(mode=0o700)
    assert os.environ['TMPDIR']==str(root) and tempfile.tempdir is None
    lib=Path(os.__file__).parent
    strict=STRICT
    if case not in ('complete','repeat','changed_source','changed_callback','changed_global','changed_api','replaced_root','wrong_mode','nonempty','wrong_tmpdir','cached_tempdir','hardlink_callback','owner_loss','source_missing','source_bytes_changed','helper_change','link_escape'):
        bodies={
            'outside_write':"Path(root+'/../outside').write_text('x')",
            'extra_dir':"os.mkdir(root+'/unexpected')",
            'environment':"os.environ['UNDECLARED']='1'",
            'subprocess':"__import__('subprocess').run(['/bin/true'])",
            'symlink':"os.symlink('/tmp',root+'/alias')",
            'rename':"os.rename(root,root+'-moved')",
            'empty_return':"return True",
            'false_return':"return False",
            'extra_probe':"tempfile.gettempdir();os.open(root+'/more',os.O_WRONLY|os.O_CREAT)",
            'partial_death':"with tempfile.TemporaryDirectory():\n        os._exit(23)",
            'reentry':"guard.run()",
            'swallowed':"try:\n        Path(root+'/../outside').write_text('x')\n    except BaseException:\n        pass\n    return True",
        }
        strict=base/'fixture.py';strict.write_text('def _probe_link_follow_symlinks():\n    '+bodies[case]+'\n')
    else:assert hashlib.sha256(strict.read_bytes()).hexdigest()==EXPECTED_STRICT
    if case in ('source_missing','source_bytes_changed','link_escape'):
        copy=base/'strict-copy.py';text=strict.read_text()
        if case=='link_escape':text=text.replace('Path(directory, "probe-link")','Path(root, "escape")')
        copy.write_text(text);strict=copy
    ns=dict(os=os,tempfile=tempfile,Path=Path,_HAS_LINK=hasattr(os,'link'),root=str(root))
    code=next(c for c in codes(compile(strict.read_bytes(),str(strict),'exec',dont_inherit=True)) if c.co_name=='_probe_link_follow_symlinks')
    callback=types.FunctionType(code,ns)
    bindings={role:source_pin(path) for role,path in dict(strict=strict,tempfile=lib/'tempfile.py',shutil=lib/'shutil.py',pathlib=lib/'pathlib.py',os=lib/'os.py').items()}
    if case=='wrong_mode':root.chmod(0o755)
    if case=='nonempty':(root/'prior').write_text('retained')
    if case=='wrong_tmpdir':os.environ['TMPDIR']=str(base)
    if case=='cached_tempdir':tempfile.tempdir=str(root)
    if case=='hardlink_callback':
        copy=base/'strict.py';copy.write_bytes(strict.read_bytes());os.link(copy,base/'strict-alias.py')
        bindings['strict']=source_pin(copy)
    g=None;failure=None;report=None
    try:
        g=FilelockScratch(str(root),callback,bindings)
        ns['guard']=g
        if case=='changed_source':g.bindings['strict']['sha256']='0'*64
        if case=='changed_callback':callback.__code__=(lambda:True).__code__
        if case=='changed_global':ns['_HAS_LINK']=False
        if case=='changed_api':os.link=lambda *a,**k:None
        if case=='owner_loss':g.pid+=1
        if case=='source_missing':strict.unlink()
        if case=='source_bytes_changed':strict.write_text('changed fixture source')
        if case=='helper_change':tempfile.TemporaryDirectory=lambda:None
        if case=='replaced_root':root.rename(base/'retained-old-root');root.mkdir(mode=0o700)
        report=g.run()
        if case=='repeat':g.run()
    except ScratchRefusal as e:failure=str(e)
    if case=='complete':assert failure is None and report['result'] and report['root_retained_empty']
    else:assert failure is not None,case
    if case=='repeat':assert failure=='Repeated scratch callback.' and report['root_retained_empty']
    if g is not None and failure is not None:
        first=g.first_refusal
        try:g.run()
        except ScratchRefusal as e:assert str(e)==failure and g.first_refusal==first
        else:raise AssertionError('Refusal not persistent')
    value=dict(case=case,verified=True,source_kind='exact installed function code with injected stdlib globals' if strict==STRICT else 'synthetic callback',
        failure=failure,first_refusal=None if g is None else g.first_refusal,report=report,
        events=[] if g is None else g.events,retained_fixture_directory=str(base),native_target_imports_performed=False)
    print(json.dumps(value),flush=True)
    # Refused fixtures keep their hooks and partial files; no finalizer cleanup.
    os._exit(0)


class Tests(unittest.TestCase):
    def run_case(self,case,expected_exit=0):
        fixture_root=Path('/home/dbayha/bca-work/ood-import-scratch-fixtures-v1/monitor_20260929T004802Z')
        fixture_root.mkdir(mode=0o700,parents=True,exist_ok=True)
        base=Path(tempfile.mkdtemp(prefix='fixture-',dir=fixture_root))
        env=dict(os.environ,TMPDIR=str(base/'scratch'))
        cmd=[sys.executable,'-I',str(Path(__file__).resolve()),'fixture',case,str(base)]
        p=subprocess.run(cmd,env=env,capture_output=True,timeout=20)
        print(json.dumps(dict(command=cmd,actual_exit=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode(),retained_fixture_directory=str(base))),file=sys.stderr)
        self.assertEqual(p.returncode,expected_exit,(p.stdout+p.stderr).decode())
        if expected_exit==0:
            r=json.loads(p.stdout);self.assertTrue(r['verified'])
            if case=='complete':
                self.assertEqual([e['event'] for e in r['events']],['open','os.remove','os.mkdir','os.utime','open','os.link','shutil.rmtree','open','os.scandir','os.remove','os.remove','os.rmdir'])
                self.assertEqual(r['events'][1]['readback_hex'],'626c6174')
                self.assertTrue(r['events'][6]['verified_empty_hard_link_pair'])
        else:
            self.assertTrue((base/'scratch').is_dir())
            self.assertEqual(len(list((base/'scratch').iterdir())),1)
    def test_exact_installed_callback(self):self.run_case('complete')
    def test_repeat(self):self.run_case('repeat')
    def test_changed_source(self):self.run_case('changed_source')
    def test_changed_callback(self):self.run_case('changed_callback')
    def test_changed_global(self):self.run_case('changed_global')
    def test_changed_api(self):self.run_case('changed_api')
    def test_replaced_root(self):self.run_case('replaced_root')
    def test_wrong_mode(self):self.run_case('wrong_mode')
    def test_nonempty(self):self.run_case('nonempty')
    def test_wrong_tmpdir(self):self.run_case('wrong_tmpdir')
    def test_cached_tempdir(self):self.run_case('cached_tempdir')
    def test_hardlink_callback(self):self.run_case('hardlink_callback')
    def test_owner_loss(self):self.run_case('owner_loss')
    def test_outside_write(self):self.run_case('outside_write')
    def test_extra_dir(self):self.run_case('extra_dir')
    def test_environment(self):self.run_case('environment')
    def test_subprocess(self):self.run_case('subprocess')
    def test_symlink(self):self.run_case('symlink')
    def test_rename(self):self.run_case('rename')
    def test_empty_return(self):self.run_case('empty_return')
    def test_false_return(self):self.run_case('false_return')
    def test_extra_probe(self):self.run_case('extra_probe')
    def test_partial_death(self):self.run_case('partial_death',23)
    def test_source_missing_poison(self):self.run_case('source_missing')
    def test_source_bytes_changed(self):self.run_case('source_bytes_changed')
    def test_helper_change(self):self.run_case('helper_change')
    def test_reentry(self):self.run_case('reentry')
    def test_swallowed_refusal(self):self.run_case('swallowed')
    def test_link_escape(self):self.run_case('link_escape')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':fixture(sys.argv[2],sys.argv[3])
    else:unittest.main()
