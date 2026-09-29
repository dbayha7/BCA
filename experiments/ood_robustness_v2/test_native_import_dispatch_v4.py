"""New unified bootstrap/capability/finalizer fixtures, never full filelock."""
import ast,base64,hashlib,importlib.util,json,os,select,stat,subprocess,sys,time,weakref
from pathlib import Path
HERE=Path(__file__).resolve().parent
MONITOR=HERE.parent/'standard_bca_noiw_campaign_v1/monitor_20260929T041506Z'
BASE=Path('/home/dbayha/bca-work/ood-native-v4-capability-fixtures/monitor_20260929T041506Z')
CASES=('complete_first_finalizer','complete_existing_finalizer','wrong_definition_hash','outside_before_callback',
       'changed_td_method','changed_finalize_method','repeat_capability_import','repeat_binding','full_route_held',
       'registry_replaced','registration_flag_changed','detached_entry_restored','unrelated_entry_changed','exit_wrapper_changed',
       'no_bytecode_changed','cleanup_defaults_changed','direct_global_changed','td_owner_changed',
       'extra_atexit','native_atexit_alias','prebound_defaults_changed','weakref_native_changed')

def emit(v):print(json.dumps(v,sort_keys=True),flush=True)

def never_run():
    raise AssertionError('Sentinel finalizer callback must never execute')

class Target:pass

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def child(case,parent):
    sys.dont_write_bytecode=True
    p=Path(parent);b=json.loads((p/'binding.json').read_bytes())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==b['fixture_source_sha256']
    module=load('fixture_dispatcher_v4',p/'dispatcher.py')
    target=Target();sentinel=None
    if case in ('complete_existing_finalizer','unrelated_entry_changed'):sentinel=weakref.finalize(target,never_run)
    assert not [n for n in sys.modules if n.split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl')]
    g=module.NativeImportDispatcherV4((p/'declaration.json').read_bytes(),(p/'binding.json').read_bytes(),hashlib.sha256((p/'dispatcher.py').read_bytes()).hexdigest())
    expected=not case.startswith('complete_');caught=False;result=None;post_refusal_diagnostic=False
    try:
        g.install();emit(dict(stage='ready',identity=g.identity,environment=g.environment))
        g.release(json.loads(sys.stdin.readline()));g.create()
        g.write_diagnostic('events.jsonl',b'{"event":"bootstrap_complete"}\n')
        config=json.loads((p/'capability_config.json').read_bytes())
        if case=='prebound_defaults_changed':module.tempfile.TemporaryDirectory.__dict__['_cleanup'].__func__.__defaults__=(True,)
        g.bind_capability(str(p/'capability.py'),'v4_fixture_capability',config['source_sha256'],config['definition_sha256'])
        if case=='changed_td_method':module.tempfile.TemporaryDirectory.cleanup=lambda self:None
        if case=='changed_finalize_method':weakref.finalize.detach=lambda self:None
        if case=='no_bytecode_changed':sys.dont_write_bytecode=False;g.check()
        if case=='cleanup_defaults_changed':module.tempfile.TemporaryDirectory.__dict__['_cleanup'].__func__.__defaults__=(True,);g.check()
        if case=='direct_global_changed':module.tempfile._weakref=None;g.check()
        if case=='td_owner_changed':module.tempfile.TemporaryDirectory=type('SyntheticReplacedOwner',(),{});g.check()
        if case=='extra_atexit':module.atexit.register(never_run)
        if case=='native_atexit_alias':g.exit_original(never_run)
        if case=='weakref_native_changed':weakref.ref=lambda *args:None;g.check()
        if case=='repeat_binding':g.bind_capability(str(p/'capability.py'),'v4_fixture_capability',config['source_sha256'],config['definition_sha256'])
        observed=load('v4_fixture_capability',p/'capability.py')
        result=g.capability_result()
        if case=='repeat_capability_import':
            observed.__spec__.loader.exec_module(observed)
        elif case=='full_route_held':g.require_import_phase()
        elif case=='registry_replaced':weakref.finalize._registry=dict(weakref.finalize._registry);g.check()
        elif case=='registration_flag_changed':weakref.finalize._registered_with_atexit=False;g.check()
        elif case=='detached_entry_restored':weakref.finalize._registry[g.finalizer_object]=g.finalizer_info;g.check()
        elif case=='unrelated_entry_changed':weakref.finalize._registry[sentinel].atexit=False;g.check()
        elif case=='exit_wrapper_changed':module.atexit.register=lambda *args:None;g.check()
        if expected:raise AssertionError('Expected refusal was absent')
        assert result['root_retained_empty'] and len(result['scratch_events'])==12
        assert len(result['finalizer_events'])==4 and result['finalizer_events'][-1]['exact_entry_removed']
        assert result['registry_snapshots'][0]['entries']==result['registry_snapshots'][-1]['entries']
        assert len(result['atexit_registration_events'])==(3 if case=='complete_first_finalizer' else 0)
        if sentinel is not None:assert sentinel in weakref.finalize._registry and target is not None
        g.write_diagnostic('final.json',(json.dumps(result,sort_keys=True)+'\n').encode())
    except module.DispatchRefusal:
        caught=True;first=g.first_refusal
        if case=='native_atexit_alias':assert first['actual_positional_metadata'] is None and first['ordered_keyword_metadata'] is None
        try:g.check()
        except module.DispatchRefusal:pass
        assert g.first_refusal==first
        try:g.write_diagnostic('final.json',(json.dumps(dict(first_refusal=first))+'\n').encode());post_refusal_diagnostic=True
        except module.DispatchRefusal:assert g.first_refusal==first
    except BaseException as error:
        emit(dict(stage='unexpected',case=case,error_type=type(error).__name__,message=str(error),first_refusal=g.first_refusal,cap_events=g.cap_events,finalizer_events=g.finalizer_events,exit_events=g.exit_events))
        os._exit(2)
    if caught!=expected:
        emit(dict(stage='unexpected',case=case,caught=caught,expected=expected,first_refusal=g.first_refusal,cap_events=g.cap_events,finalizer_events=g.finalizer_events,exit_events=g.exit_events));os._exit(3)
    emit(dict(stage='result',case=case,expected_refusal=expected,caught=caught,first_refusal=g.first_refusal,post_refusal_diagnostic=post_refusal_diagnostic,
              capability_result=result,cap_events=g.cap_events,scratch_events=getattr(g,'cap_scratch_events',[]),finalizer_events=g.finalizer_events,exit_events=g.exit_events,
              registry_snapshots=getattr(g,'registry_events',[]),native_events=g.events,cap_native_events=g.cap_native_events,
              source_roles={k:dict(path=v[1],pin=v[2],aliases=v[3]) for k,v in g.source_roles.items()},
              profile_present=sys.getprofile() is g.profile_hook,target_modules=[n for n in sys.modules if n.split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl')],
              full_route_implemented=False,actual_production_profile_executed=False,scientific_execution_accepted=False))
    os._exit(0)

def inspect_process(pid):
    q=Path('/proc')/str(pid);tail=(q/'stat').read_bytes().rsplit(b') ',1)[1].split()
    raw=(q/'environ').read_bytes();assert raw.endswith(b'\0')
    pairs=[x.decode().split('=',1) for x in raw[:-1].split(b'\0')];assert len(pairs)==len(dict(pairs))
    return dict(identity=dict(pid=pid,parent_pid=int(tail[1]),session=int(tail[3]),group=int(tail[2]),start_ticks=int(tail[19]),argv_hex=(q/'cmdline').read_bytes().hex()),environment=dict(pairs))

def run_all(label):
    assert label in ('composition_tests_v1','composition_tests_v2','composition_tests_v3')
    root=BASE/label;root.mkdir(parents=True,exist_ok=False)
    saved=json.loads((MONITOR.parent/'monitor_20260929T004802Z/scratch_source_inspection.json').read_bytes())
    item=next(v for k,v in saved['sources'].items() if k.endswith('/filelock/_strict.py'))
    original=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(original).hexdigest()==item['sha256']=='379d37ee7580cf6005e7d2e5c8f32c3a0e08c062f5dc3eb1eb88614049f7bd04'
    tree=ast.parse(original);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_probe_link_follow_symlinks')
    definition=ast.get_source_segment(original.decode(),node);definition_sha=hashlib.sha256(definition.encode()).hexdigest()
    source=(HERE/'native_import_dispatch_v4.py').read_bytes();test=Path(__file__).read_bytes();declaration=(HERE/'native_import_contract_v3.json').read_bytes();d=json.loads(declaration)
    results=[]
    for case in CASES:
        parent=root/case;parent.mkdir(mode=0o700);s=parent.stat()
        b=dict(schema='native-import-v4-capability-fixture-v1',purpose='capability_fixture_only',declaration_sha256=hashlib.sha256(declaration).hexdigest(),parent=str(parent),parent_id=[s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)],scratch=str(parent/'scratch'),diagnostics=str(parent/'diagnostics'),environment=dict(d['environment'],TMPDIR=str(parent/'scratch')),fixture_source_sha256=hashlib.sha256(test).hexdigest())
        preface='import os\nimport tempfile\nfrom pathlib import Path\n_HAS_LINK = hasattr(os, "link")\n'
        if case=='outside_before_callback':preface+='os.mkdir('+repr(str(parent/'outside'))+', 448)\n'
        capability=(preface+'\n'+definition+'\n\n_LINK_HONORS_FOLLOW_SYMLINKS = _probe_link_follow_symlinks()\n').encode()
        config=dict(source_sha256=hashlib.sha256(capability).hexdigest(),definition_sha256='0'*64 if case=='wrong_definition_hash' else definition_sha)
        for n,data in (('binding.json',json.dumps(b,indent=2).encode()),('declaration.json',declaration),('dispatcher.py',source),('fixture.py',test),('capability.py',capability),('capability_config.json',json.dumps(config).encode())):(parent/n).write_bytes(data)
        cmd=[sys.executable,'-I',str(parent/'fixture.py'),'--child',case,str(parent)]
        start=time.time();p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=b['environment'],start_new_session=True,bufsize=0)
        observed=None;first=b''
        try:
            ready,unused,unused=select.select([p.stdout],[],[],25)
            if not ready:raise TimeoutError('Fixture readiness timeout')
            first=p.stdout.readline();item=json.loads(first)
            if item['stage']=='ready':
                observed=inspect_process(p.pid);assert observed==dict(identity=item['identity'],environment=b['environment'])
                assert observed['identity']['pid']==observed['identity']['session']==observed['identity']['group'] and observed['identity']['parent_pid']==os.getpid()
                p.stdin.write((json.dumps(observed)+'\n').encode());p.stdin.flush()
            tail,err=p.communicate(timeout=45);out=first+tail
        except BaseException:
            if p.poll() is None:os.killpg(p.pid,15)
            tail,err=p.communicate(timeout=5);out=first+tail
            (parent/'parent_failure.stdout').write_bytes(out);(parent/'parent_failure.stderr').write_bytes(err)
            raise
        rows=[json.loads(x) for x in out.splitlines()]
        r=dict(case=case,actual_exit=p.returncode,command=cmd,elapsed_seconds=time.time()-start,observed_before_release=observed,rows=rows,fixture_parent=str(parent),stdout_sha256=hashlib.sha256(out).hexdigest(),stderr_sha256=hashlib.sha256(err).hexdigest())
        (parent/'child.stdout').write_bytes(out);(parent/'child.stderr').write_bytes(err);(parent/'child_exit.json').write_text(json.dumps(r,indent=2))
        results.append(r);(MONITOR/(label+'_children.json')).write_text(json.dumps(results,indent=2))
        assert p.returncode==0,(case,p.returncode,out.decode(),err.decode())
        assert rows[-1]['stage']=='result' and not rows[-1]['target_modules'] and not (parent/'outside').exists()
        print(case,p.returncode,flush=True)
    print(json.dumps(dict(tests=len(results),skipped=0,expected_refusals=len(results)-2,successful_compositions=2,actual_production_profile_executed=False)))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--child':child(sys.argv[2],sys.argv[3])
    else:run_all(sys.argv[1])
