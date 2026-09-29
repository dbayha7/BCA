"""New isolated module-entry fixtures, never a full filelock package import."""
from pathlib import Path
import ast,base64,datetime,hashlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
MONITOR=HERE.parent/'standard_bca_noiw_campaign_v1/monitor_20260929T012803Z'
SAVED=MONITOR.parent/'monitor_20260929T004802Z/scratch_source_inspection.json'
ROOT=Path('/home/dbayha/bca-work/ood-import-bridge-fixtures-v1/monitor_20260929T012803Z')
CASES=('complete','outside_write','extra_probe','false_return','no_probe','repeat_call','repeat_module','repeat_complete','changed_result','changed_module','changed_function','changed_globals','wrong_callsite','environment','process','lock','audit_hook','fork_hook','source_changed','swallowed','partial_death','rebound_function','callback_defaults','parent_live_method','contract_change')


def child(case):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(HERE))
    import filelock_import_bridge_v1 as bridge
    import filelock_scratch_v1 as life
    ROOT.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='fixture-',dir=ROOT));root=directory/'scratch';root.mkdir(mode=0o700)
    os.environ['TMPDIR']=str(root)
    source=json.loads(SAVED.read_bytes())['sources']
    strict=next(v for k,v in source.items() if k.endswith('/filelock/_strict.py'))
    text=base64.b64decode(strict['base64']).decode();tree=ast.parse(text)
    function=ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_probe_link_follow_symlinks'))
    exact=function
    pre='from __future__ import annotations\nimport os,tempfile,sys\nfrom pathlib import Path\n_HAS_LINK=hasattr(os,"link")\n'
    call='_LINK_HONORS_FOLLOW_SYMLINKS = _probe_link_follow_symlinks()\n'
    post=''
    if case=='outside_write':pre+='Path(__file__).with_name("outside").write_text("forbidden")\n'
    if case=='extra_probe':function=function.replace('source.touch()', 'source.touch()\n            source.touch()')
    if case=='false_return':function=function.replace('    return True','    return False')
    if case=='no_probe':function='def _probe_link_follow_symlinks():\n    return True'
    if case=='repeat_call':post='_SECOND = _probe_link_follow_symlinks()\n'
    if case=='wrong_callsite':pre+='def _wrapper(fn):\n    return fn()\n';call='_EARLY = _wrapper(_probe_link_follow_symlinks)\n'+call
    if case=='changed_globals':pre+='_HAS_LINK=False\n'
    if case=='environment':function=function.replace('    if not _HAS_LINK:', '    os.environ["BRIDGE_UNDECLARED"]="1"\n    if not _HAS_LINK:')
    if case=='process':function=function.replace('    if not _HAS_LINK:', '    os.system("true")\n    if not _HAS_LINK:')
    if case=='lock':function=function.replace('    if not _HAS_LINK:', '    import fcntl\n    fcntl.flock(1,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    if not _HAS_LINK:')
    if case=='audit_hook':post='sys.addaudithook(lambda *args: None)\n'
    if case=='fork_hook':post='os.register_at_fork(after_in_child=lambda: None)\n'
    if case=='swallowed':
        function=function.replace('    if not _HAS_LINK:', '    try:\n        Path(__file__).with_name("outside").write_text("forbidden")\n    except BaseException:\n        pass\n    if not _HAS_LINK:')
    if case=='partial_death':function=function.replace('            source =', '            os.write(1,("PARTIAL_ROOT="+str(Path(directory).parent)+"\\n").encode())\n            os._exit(23)\n            source =')
    p=directory/'fixture_module.py';p.write_bytes((pre+function+'\n'+call+post).encode())
    bindings={role:life.source_pin(next(k for k in source if k.endswith('/'+name))) for role,name in [('tempfile','tempfile.py'),('shutil','shutil.py'),('pathlib','pathlib.py'),('os','os.py')]}
    bindings['strict']=life.source_pin(p)
    b=bridge.ImportScratchBridge(root,'_bridge_fixture',bindings)
    spec=importlib.util.spec_from_file_location('_bridge_fixture',p);module=importlib.util.module_from_spec(spec)
    sys.modules['_bridge_fixture']=module
    report=None;failure=None
    try:
        if case=='source_changed':p.write_bytes(p.read_bytes()+b'\n# changed after binding\n')
        b.install();spec.loader.exec_module(module)
        if case=='repeat_module':spec.loader.exec_module(module)
        if case=='changed_result':module._LINK_HONORS_FOLLOW_SYMLINKS=False
        if case=='changed_module':sys.modules['_bridge_fixture']=type(module)('_bridge_fixture')
        if case=='changed_function':module._probe_link_follow_symlinks.__code__=(lambda:True).__code__
        if case=='rebound_function':module._probe_link_follow_symlinks=lambda:True
        if case=='callback_defaults':module._probe_link_follow_symlinks.__defaults__=(1,)
        if case=='parent_live_method':life.FilelockScratch.check=lambda self:None
        if case=='contract_change':b.result_name='_ANOTHER_RESULT'
        report=b.complete()
        if case=='repeat_complete':b.complete()
    except BaseException as e:failure=type(e).__name__+': '+str(e)
    if case=='complete':
        assert failure is None and report['callback_invocations_observed']==1 and report['callback_invocations_by_bridge']==0
        assert len(report['boundary_events'])==4 and len(report['scratch_events'])==12
        assert list(root.iterdir())==[] and report['capability_result'] is True
    else:
        assert failure is not None and b.failed and b.first_refusal is not None,(case,failure,report)
        first=json.dumps(b.first_refusal,sort_keys=True)
        try:b.check()
        except bridge.BridgeRefusal:pass
        else:raise AssertionError('Poisoned bridge was reusable.')
        assert json.dumps(b.first_refusal,sort_keys=True)==first
        if case=='changed_function':assert b.first_refusal['message']=='Callback changed.'
    result=dict(case=case,verified=True,fixture_directory=str(directory),failure=failure,first_refusal=b.first_refusal,
                exact_installed_callback_text=function==exact,report=report,boundary_events=b.boundary_events,
                scratch_events=[] if b.scratch is None else b.scratch.events,
                full_filelock_package_imported=False,fixture_module_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    os.write(1,(json.dumps(result)+'\n').encode())
    # Keep failure hooks installed and preserve partial files; do not run later finalizers.
    os._exit(0)


class Tests(unittest.TestCase):
    def run_case(self,case):
        command=[sys.executable,'-I',str(Path(__file__).resolve()),'child',case]
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        p=subprocess.run(command,capture_output=True,timeout=40)
        receipt=dict(case=case,started=start,ended=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,actual_exit=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode())
        sys.stderr.write(json.dumps(receipt)+'\n')
        if case=='partial_death':
            self.assertEqual(p.returncode,23)
            root=Path(p.stdout.decode().strip().split('PARTIAL_ROOT=')[1]);children=list(root.iterdir())
            self.assertEqual(len(children),1);self.assertTrue(children[0].is_dir());self.assertEqual(list(children[0].iterdir()),[])
        else:
            self.assertEqual(p.returncode,0,p.stderr.decode())
            data=json.loads(p.stdout);self.assertTrue(data['verified'])
            if case in ('outside_write','environment','process','lock','audit_hook','swallowed'):
                self.assertIsNotNone(data['first_refusal']['event']);self.assertIsNotNone(data['first_refusal']['args'])


for case in CASES:setattr(Tests,'test_'+case,lambda self,case=case:self.run_case(case))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':child(sys.argv[2])
    else:unittest.main()
