"""Isolated synthetic namespaces; actual registration APIs, no native target import."""
from pathlib import Path
import ast,atexit,base64,contextlib,datetime,hashlib,importlib.util,itertools,json,os,subprocess,sys,tempfile,threading,types,unittest,weakref
HERE=Path(__file__).resolve().parent
MONITOR=HERE.parent/'standard_bca_noiw_campaign_v1/monitor_20260929T023304Z'
PARENT=MONITOR.parent/'monitor_20260929T015303Z'
ROOT=Path('/home/dbayha/bca-work/ood-registration-args-fixtures-v1/monitor_20260929T023304Z')
CASES=('complete','wrong_callback','extra_audit_arg','extra_fork_keyword','missing_fork_keyword',
       'reordered_fork_keywords','extra_exit_arg','defaults_before','defaults_after','positional_defaults',
       'callback_code','callback_rebound','captured_global_rebound','direct_global_rebound','state_depth',
       'wrong_namespace','wrong_file','source_changed','original_api_changed','wrapper_changed','verifier_changed',
       'missing_registration','repeat_registration','repeat_complete','direct_bypass','swallowed',
       'environment','outside_write','complete_verifier_failure','observer_suppressed','target_audit_suppressed',
       'private_kwarg','wrapper_missing','contract_change','repeat_install','profile_change','observer_raised',
       'callback_metadata','module_file_after','kwdefault_object')


def state_verifier(ns,phase):
    state=ns['_FORK_STATE']
    assert type(state) is types.SimpleNamespace
    assert type(state.transition_context) is types.SimpleNamespace and type(state.transition_context.depth) is int and state.transition_context.depth==0
    assert state.pid==os.getpid() and state.fork_owner_depths=={}
    assert type(ns['_FORK_AUDIT_EVENTS']) is frozenset and ns['_FORK_AUDIT_EVENTS']==frozenset(('os.fork','os.forkpty'))
    assert ns['_SQLITE_TRANSITION_CONTEXT'].depth==0 and ns['_UNSAFE_FORK_EXIT_STATUS']==70
    assert type(ns['_ALL_INSTANCES']) is weakref.WeakValueDictionary and len(ns['_ALL_INSTANCES'])==0
    assert ns['_OWNED_DESCRIPTORS']=={} and ns['_INIT_PARAMETER_MODELS']=={}
    if ns['CASE']=='complete_verifier_failure' and phase=='complete':raise ValueError('Synthetic completion-state failure')


def fixture_source(case):
    saved=json.loads((PARENT/'local_import_graph_inventory.json').read_bytes())['sources']
    wanted={'_api.py':['_register_fork_hooks','_audit_fork_safety','_pin_fork_objects','_resume_parent_after_fork','_reset_child_after_fork'],
            '_read_write.py':['_abort_forked_sqlite_transition'],'_soft_rw/_sync.py':['_cleanup_all_instances']}
    selected={};pins={}
    for suffix,names in wanted.items():
        item=next(v for k,v in saved.items() if k.endswith('/filelock/'+suffix))
        raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
        text=raw.decode();tree=ast.parse(text);pins[suffix]=item['sha256']
        for name in names:
            node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
            selected[name]=ast.get_source_segment(text,node)
    header='''from __future__ import annotations
import os,sys,atexit,threading,weakref
from types import SimpleNamespace
from contextlib import suppress
from itertools import starmap
get_ident=threading.get_ident
_REGISTER_AT_FORK=os.register_at_fork
_FORK_STATE=SimpleNamespace(transition_context=SimpleNamespace(depth=0),fork_owner_depths={},pid=os.getpid())
_FORK_AUDIT_EVENTS=frozenset(('os.fork','os.forkpty'))
_SQLITE_TRANSITION_CONTEXT=SimpleNamespace(depth=0)
_UNSAFE_FORK_EXIT_STATUS=70
_ALL_INSTANCES=weakref.WeakValueDictionary()
_OWNED_DESCRIPTORS={}
_INIT_PARAMETER_MODELS={}
_FORK_OBJECTS=weakref.WeakValueDictionary()
_FORK_CLASSES=weakref.WeakValueDictionary()
def _fixture_helper(*args,**kwargs):
    raise AssertionError('Unexecuted synthetic dependency was called')
_ensure_current_process=_verify_unverified_descriptors=_snapshot_descriptor_for_fork=_detach_child_state=_fixture_helper
'''+f'CASE={case!r}\n'
    segments=dict(selected)
    if case=='extra_audit_arg':segments['_register_fork_hooks']=segments['_register_fork_hooks'].replace('sys.addaudithook(_audit_fork_safety)','sys.addaudithook(_audit_fork_safety,_abort_forked_sqlite_transition)')
    if case=='private_kwarg':segments['_register_fork_hooks']=segments['_register_fork_hooks'].replace('sys.addaudithook(_audit_fork_safety)',"sys.addaudithook(_audit_fork_safety,_kind='audit')")
    if case=='reordered_fork_keywords':
        segments['_register_fork_hooks']=segments['_register_fork_hooks'].replace('before=_pin_fork_objects,\n        after_in_parent=_resume_parent_after_fork,','after_in_parent=_resume_parent_after_fork,\n        before=_pin_fork_objects,')
    before='''
if CASE=='wrong_callback':_audit_fork_safety=_abort_forked_sqlite_transition
if CASE=='defaults_before':_audit_fork_safety.__kwdefaults__['_state']=SimpleNamespace(transition_context=SimpleNamespace(depth=0),fork_owner_depths={},pid=os.getpid())
if CASE=='positional_defaults':_audit_fork_safety.__defaults__=(None,)
if CASE=='swallowed':
    try:sys.addaudithook(lambda *args:None)
    except BaseException:pass
if CASE=='environment':os.environ['REGISTRATION_UNDECLARED']='1'
if CASE=='outside_write':open(__file__+'.outside','w')
_register_fork_hooks()
fork_arguments={'after_in_child':_abort_forked_sqlite_transition}
if CASE=='extra_fork_keyword':fork_arguments['before']=_pin_fork_objects
if CASE=='missing_fork_keyword':fork_arguments={}
os.register_at_fork(**fork_arguments)
extra_exit_arguments=[1] if CASE=='extra_exit_arg' else []
if CASE!='missing_registration':
    atexit.register(_cleanup_all_instances,*extra_exit_arguments)
'''
    text=header+'\n\n'.join(segments.values())+'\n'+before
    tree=ast.parse(text)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_register_fork_hooks')
    helper_calls=[n for n in ast.walk(helper) if isinstance(n,ast.Call)]
    audit_line=next(n.lineno for n in helper_calls if isinstance(n.func,ast.Attribute) and n.func.attr=='addaudithook')
    fork_line=next(n.lineno for n in helper_calls if isinstance(n.func,ast.Name) and n.func.id=='_REGISTER_AT_FORK')
    child_line=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='register_at_fork')
    exit_line=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='register')
    sites=[dict(kind='audit',caller='_register_fork_hooks',line=audit_line,args=['_audit_fork_safety'],kwargs={}),
           dict(kind='fork',caller='_register_fork_hooks',line=fork_line,args=[],kwargs=dict(before='_pin_fork_objects',after_in_parent='_resume_parent_after_fork',after_in_child='_reset_child_after_fork')),
           dict(kind='fork',caller='<module>',line=child_line,args=[],kwargs=dict(after_in_child='_abort_forked_sqlite_transition')),
           dict(kind='exit',caller='<module>',line=exit_line,args=['_cleanup_all_instances'],kwargs={})]
    return text,sites,pins,{n:segments[n]==original for n,original in selected.items()}


def child(case):
    sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE))
    import filelock_registration_args_v1 as guard
    ROOT.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='fixture-',dir=ROOT));path=directory/'fixture_module.py'
    source,sites,pins,exact=fixture_source(case);path.write_bytes(source.encode())
    spec=importlib.util.spec_from_file_location('_registration_fixture',path)
    module=importlib.util.module_from_spec(spec);sys.modules[module.__name__]=module
    native_fork=os.register_at_fork;native_audit=sys.addaudithook
    suppressed=[]
    if case in ('observer_suppressed','target_audit_suppressed','observer_raised'):
        class ExternalRefusal(BaseException):pass
        def suppress_registration(event,args):
            if event=='sys.addaudithook':
                suppressed.append(event)
                if case=='observer_raised':raise ExternalRefusal('Synthetic pre-existing hook refusal')
                if case=='observer_suppressed' or len(suppressed)>=2:raise RuntimeError('Synthetic suppression')
        native_audit(suppress_registration)
    b=guard.RegistrationArguments(module_name=module.__name__,source_pin=guard.pin(path),sites=sites,
         keyword_references={'_audit_fork_safety':dict(_fork_events='_FORK_AUDIT_EVENTS',_state='_FORK_STATE')},state_verifier=state_verifier)
    failure=None;report=None
    try:
        if case=='source_changed':path.write_bytes(path.read_bytes()+b'\n# changed\n')
        if case=='original_api_changed':b.original['fork']=lambda **kwargs:None
        if case=='wrapper_changed':b.wrappers['fork']=lambda *args,**kwargs:None
        if case=='wrapper_missing':del b.wrappers['fork']
        if case=='contract_change':b.sites[2]['kwargs']={}
        if case=='verifier_changed':b.verifier=lambda ns,phase:None
        b.install()
        if case=='repeat_install':b.install()
        if case=='profile_change':sys.setprofile(None)
        if case=='direct_bypass':native_fork(after_in_child=lambda:None)
        if case=='wrong_namespace':sys.modules[module.__name__]=types.ModuleType(module.__name__)
        if case=='wrong_file':module.__file__=str(path)+'.different'
        spec.loader.exec_module(module)
        if case=='defaults_after':module._audit_fork_safety.__kwdefaults__['_state']=types.SimpleNamespace()
        if case=='callback_code':module._abort_forked_sqlite_transition.__code__=(lambda:None).__code__
        if case=='callback_rebound':module._cleanup_all_instances=lambda:None
        if case=='callback_metadata':module._abort_forked_sqlite_transition.__name__='changed'
        if case=='module_file_after':module.__file__=str(path)+'.changed'
        if case=='kwdefault_object':module._audit_fork_safety.__kwdefaults__=dict(module._audit_fork_safety.__kwdefaults__)
        if case=='captured_global_rebound':module._FORK_STATE=types.SimpleNamespace(transition_context=types.SimpleNamespace(depth=0),fork_owner_depths={},pid=os.getpid())
        if case=='direct_global_rebound':module._SQLITE_TRANSITION_CONTEXT=types.SimpleNamespace(depth=0)
        if case=='state_depth':module._FORK_STATE.transition_context.depth=1
        if case=='repeat_registration':module._register_fork_hooks()
        report=b.complete()
        if case=='repeat_complete':b.complete()
    except BaseException as e:failure=type(e).__name__+': '+str(e)
    if case in ('complete','target_audit_suppressed'):
        assert failure is None,(case,failure,b.first_refusal)
        assert report['registration_requests']==4 and len(report['events'])==12
        assert [r['event'] for r in report['events']]==['request','native_entry','native_return']*4
        assert report['callback_invocations_by_observer']==0 and not report['audit_installation_effectiveness_accepted']
        if case=='target_audit_suppressed':assert len(suppressed)==2
    else:
        assert failure is not None and b.failed and b.first_refusal is not None,(case,failure,report)
        first=json.dumps(b.first_refusal,sort_keys=True)
        try:b.check()
        except guard.RegistrationRefusal:pass
        else:raise AssertionError('Poisoned observer reused.')
        assert json.dumps(b.first_refusal,sort_keys=True)==first
    result=dict(case=case,verified=True,fixture_directory=str(directory),fixture_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                saved_installed_source_sha256=pins,exact_selected_source_text=exact,failure=failure,first_refusal=b.first_refusal,
                events=b.events,report=report,full_filelock_imported=False,fork_executed=False,sqlite_connection_opened=False,
                shutdown_callbacks_executed=False,audit_delivery_count=None,observer_audit_canary_seen=b.install_seen,
                target_audit_suppression_fixture=case=='target_audit_suppressed',
                loaded_target_modules=[n for n in sys.modules if any(n==root or n.startswith(root+'.') for root in ('filelock','numpy','jax','mujoco_py','glfw'))])
    assert result['loaded_target_modules']==[]
    os.write(1,(json.dumps(result)+'\n').encode())
    # Keep observers active. No unguarded shutdown callbacks/finalizers are run.
    os._exit(0)


class Tests(unittest.TestCase):
    def run_case(self,case):
        command=[sys.executable,'-I',str(Path(__file__).resolve()),'child',case]
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()
        p=subprocess.run(command,capture_output=True,timeout=25)
        receipt=dict(case=case,started=started,ended=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,actual_exit=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode())
        sys.stderr.write(json.dumps(receipt)+'\n')
        self.assertEqual(p.returncode,0,p.stderr.decode())
        result=json.loads(p.stdout);self.assertTrue(result['verified'])
        if case=='complete':self.assertTrue(all(result['exact_selected_source_text'].values()))
        if case in ('wrong_callback','extra_fork_keyword','extra_exit_arg','extra_audit_arg'):
            self.assertIsNotNone(result['first_refusal']['caller'])
            self.assertIsNotNone(result['first_refusal']['args'])
            self.assertIsNotNone(result['first_refusal']['kwargs'])
        if case=='direct_bypass':
            self.assertIsNone(result['first_refusal']['args']);self.assertIsNone(result['first_refusal']['kwargs'])


for case in CASES:setattr(Tests,'test_'+case,lambda self,case=case:self.run_case(case))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':child(sys.argv[2])
    else:unittest.main()
