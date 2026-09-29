"""OPEN concrete unified fixture entry; production purpose remains held."""
import ast,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
# Known bootstrap imports precede the observation boundary. Their registration
# calls are not retrospectively observed, counted or accepted for production.
import concurrent.futures.thread as thread,logging,random,threading,weakref
import atexit
def shutdown_marker():print('UNEXPECTED_NORMAL_SHUTDOWN',flush=True)
atexit.register(shutdown_marker)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def setup(guard):
    rows={'api':guard.bind_filelock_api(api),'rw':guard.bind_filelock_readwrite(rw),'soft':guard.bind_filelock_soft(soft)}
    rows['random']=guard.bind_python(random,'Random.seed',random._inst.seed,('_inst',),defaults=(None,2),closure={'__class__':random.Random})
    rows['logging']=guard.bind_logging(logging)
    rows['weakref']=guard.bind_python(weakref,'finalize._exitfunc',weakref.finalize._exitfunc,('finalize',))
    rows['native']=[guard.bind_native_lock(thread,('_global_shutdown_lock',),name,getattr(thread._global_shutdown_lock,name)) for name in ('acquire','_at_fork_reinit','release')]
    rows['thread_exit']=guard.bind_python(thread,'_python_exit');rows['thread_callback']=guard.bind_python(api,'_thread_callback')
    return rows

def site(module,qualname,text):
    raw=Path(module.__file__).read_text();matches=[i for i,line in enumerate(raw.splitlines(),1) if text in line];assert len(matches)==1
    return module,qualname,matches[0]

p=Path(sys.argv[1]);control=json.loads((p/'fixture_control.json').read_bytes());case=control['case']
native_alias=sys.addaudithook
b=load('concrete_binder',p/'binder.py');d=load('unified_dispatcher',p/'dispatcher.py');m=load('unified_protocol',p/'protocol.py')
g=d.NativeImportDispatcherV5((p/'declaration.json').read_bytes(),(p/'binding.json').read_bytes(),m.DISPATCHER)
proto=m.WorkerProtocol(g,hashlib.sha256((p/'protocol.py').read_bytes()).hexdigest(),hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
g.bind_profile_failure(proto.profile_abort)
try:
    g.install();proto.ready_release();g.create();g.write_diagnostic('events.jsonl',b'{"event":"unified_bootstrap_complete"}\n')
    config=json.loads((p/'capability_config.json').read_bytes())
    g.bind_capability(str(p/'capability.py'),'v5_fixture_capability',config['source_sha256'],config['definition_sha256'])
    api=load('binding_fixture_api',p/'api.py');rw=load('binding_fixture_rw',p/'rw.py');soft=load('binding_fixture_soft',p/'soft.py');driver=load('binding_fixture_driver',p/'driver.py')
    specs=[]
    for mod in (threading,random,logging,weakref,thread):
        suffix=mod.__file__.split('/lib/python3.10/',1)[1];aliases=[f'/home/dbayha/miniconda3/{prefix}/lib/python3.10/{suffix}' for prefix in ('envs/corl-orig-local','envs/sdbca','pkgs/python-3.10.20-h741d88c_0')]
        specs.append((mod,control['stdlib_sha256'][mod.__name__],aliases))
    for mod in (api,rw,soft,driver):specs.append((mod,control['fixture_sha256'][Path(mod.__file__).name],[mod.__file__]))
    routes=[('audit',*site(api,'_register_fork_hooks','sys.addaudithook('),(api._audit_fork_safety,),{}),
      ('fork',*site(api,'_register_fork_hooks','    _REGISTER_AT_FORK('),(),dict(before=api._pin_fork_objects,after_in_parent=api._resume_parent_after_fork,after_in_child=api._reset_child_after_fork)),
      ('fork',*site(driver,'run_rest','# rw fork'),(),dict(after_in_child=rw._abort_forked_sqlite_transition)),
      ('exit',*site(driver,'run_rest','# soft exit'),(soft._cleanup_all_instances,),{}),
      ('fork',*site(driver,'run_rest','# random fork'),(),dict(after_in_child=random._inst.seed)),
      ('fork',*site(driver,'run_rest','# native fork'),(),dict(before=thread._global_shutdown_lock.acquire,after_in_child=thread._global_shutdown_lock._at_fork_reinit,after_in_parent=thread._global_shutdown_lock.release)),
      ('fork',*site(driver,'run_rest','# logging fork'),(),dict(before=logging._acquireLock,after_in_child=logging._after_at_fork_child_reinit_locks,after_in_parent=logging._releaseLock)),
      ('exit',*site(driver,'run_rest','# logging exit'),(logging.shutdown,),{}),
      ('threading',*site(driver,'run_rest','# internal threading'),(api._thread_callback,driver.TOKEN),dict(first=driver.FIRST,second=driver.SECOND))]
    if case=='wrong_order':routes[0],routes[1]=routes[1],routes[0]
    if case=='wrong_keyword_order':routes[1]=(*routes[1][:5],dict(after_in_child=api._reset_child_after_fork,after_in_parent=api._resume_parent_after_fork,before=api._pin_fork_objects))
    if case=='wrong_callback':routes[0]=(*routes[0][:4],(api._thread_callback,),{})
    g.bind_registration_routes(b,specs,routes,threading,setup)
    if case=='profile_internal_error':g.reg_originals=None
    if case=='production_route':g.require_import_phase()
    if case=='direct_alias':native_alias(api._audit_fork_safety)
    if case=='wrapper_changed':sys.addaudithook=native_alias;g.check()
    if case=='captured_default_changed':api._audit_fork_safety.__kwdefaults__['_state']=api._ForkState()
    if case=='native_owner_changed':thread._global_shutdown_lock=threading.Lock()
    if case=='premature_completion':proto.prepare_completion({})
    api._register_fork_hooks()
    if case=='repeat_registration':api._register_fork_hooks()
    load('v5_fixture_capability',p/'capability.py')
    if case=='callback_after_capability':random.Random.seed.__defaults__=(None,1)
    if case=='threading_prior_changed':threading._threading_atexits[0].keywords['changed']=True
    driver.run_rest()
    if case=='partial_after_return':threading._threading_atexits[-1].keywords['changed']=True
    if case=='registry_after_return':api._FORK_STATE.transition_context.depth=1
    if case=='extra_native_registration':os.register_at_fork(after_in_child=random._inst.seed)
    if case=='profile_removal':sys.setprofile(None)
    result=g.composed_result()
    if case=='wrong_combined_result':result['registrations']['requests']=0
    proto.prepare_completion(result)
    if case=='reject_completion_ack':
        packet=proto.receive();assert packet['stage']=='completion_ack';raise ValueError('Declared rejection after real parent acknowledgment')
    proto.accept_completion()
except BaseException as error:proto.abort(error)
