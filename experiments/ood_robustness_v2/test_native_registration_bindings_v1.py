"""Fresh child fixtures for concrete real callback forms; no filelock import."""
import ast,base64,datetime,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
W=HERE.parent/'standard_bca_noiw_campaign_v1/monitor_20260929T051207Z'
ROOT=Path('/home/dbayha/bca-work/ood-registration-bindings-fixtures/monitor_20260929T051207Z')
CASES=('complete','nonempty_registries','api_kw_state','api_kw_dictionary','api_global_state','api_depth','api_held','api_weak_entry','api_token','api_condition','api_registry_lock','rw_depth','rw_pypy','rw_poison','soft_entry','random_owner','random_closure','random_defaults','random_code','logging_captured_list','logging_global_list','logging_handler_method','logging_lock','native_owner','native_wrong_callback','native_wrong_name','threading_partial_keyword','threading_existing_change','threading_shutdown','repeat_internal','fixture_source_changed','binder_method_changed','production_held','callback_name','callback_qualname','callback_module','callback_annotation','wrong_code_filename','repeat_internal_return','threading_prior_nonpartial')
HASHES={'threading':'a21926e636bec8c2e5579f3e79cce144c36379ef93c4596252af29970983553d','random':'ce80a2471965e64ae93caea14490f16850432e65dce87f15edfe25b6f562f8d1','logging':'9069cd43c7a8aa8170b654b1ee21e0d35c15baebfb61e49ecb490df968dce07e','weakref':'dd8e03473ee5667c1a2caa43ede07797652bcb4035fabb60d60af10bb23a0886','concurrent.futures.thread':'b06f8899881193efc72cfc3ebf2836dce4e668b3856ad35f4016616d643a519e'}
CASES=(*CASES,'fixture_source_symlink')
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sources():
    saved=json.loads((W.parent/'monitor_20260929T015303Z/local_import_graph_inventory.json').read_bytes())['sources'];result={};definitions={}
    roles={'api':('_api.py',['_ThreadLocalRegistry','_ForkTransitionContext','_ForkState','_register_fork_hooks','_audit_fork_safety','_pin_fork_objects','_resume_parent_after_fork','_reset_child_after_fork','_ensure_current_process','_verify_unverified_descriptors','_snapshot_descriptor_for_fork','_detach_child_state']),
           'rw':('_read_write.py',['_SQLiteTransitionContext','_ForkedDatabaseRegistry','_abort_forked_sqlite_transition']),
           'soft':('_soft_rw/_sync.py',['_cleanup_all_instances'])}
    for role,(suffix,names) in roles.items():
        item=next(v for k,v in saved.items() if k.endswith('/filelock/'+suffix));raw=base64.b64decode(item['base64']);assert sha(raw)==item['sha256'];text=raw.decode();tree=ast.parse(text)
        selected={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names};assert set(selected)==set(names)
        definitions[role]=dict(original_sha256=item['sha256'],definitions=selected)
        header='from __future__ import annotations\nimport os,sys,threading,weakref,pathlib\nfrom threading import Condition,RLock,local,get_ident\nfrom weakref import WeakKeyDictionary,WeakValueDictionary\nfrom itertools import count,starmap\nfrom contextlib import suppress\n'
        if role=='api':
            body='\n\n'.join(selected[n] for n in names[:3])+'\n\n_registry=_ThreadLocalRegistry()\n_FORK_STATE=_ForkState()\n_FORK_OBJECTS=WeakValueDictionary()\n_FORK_CLASSES=WeakValueDictionary()\n_INIT_PARAMETER_MODELS=WeakKeyDictionary()\n_OWNED_DESCRIPTORS={}\n_TRANSITION_TOKENS=count()\n_FORK_AUDIT_EVENTS=frozenset(("os.fork","os.forkpty"))\n_REGISTER_AT_FORK=os.register_at_fork\n'+ '\n\n'.join(selected[n] for n in names[3:])
            body+='\n\ndef _thread_callback(*args,**kwargs):\n    raise AssertionError("Fixture callback must never execute")\n'
        elif role=='rw':body='\n\n'.join(selected.values())+'\n_SQLITE_TRANSITION_CONTEXT=_SQLiteTransitionContext()\n_FORKED_DATABASES=_ForkedDatabaseRegistry()\n_IS_PYPY=False\n_UNSAFE_FORK_EXIT_STATUS=70\n'
        else:body='\n\n'.join(selected.values())+'\n_ALL_INSTANCES=WeakValueDictionary()\n_ALL_INSTANCES_LOCK=threading.Lock()\n'
        result[role+'.py']=(header+body+'\n').encode()
    return result,definitions
def child(case,p):
    import concurrent.futures.thread as thread,logging,random,threading,weakref
    m=load('registration_bindings',HERE/'native_registration_bindings_v1.py')
    api=load('binding_fixture_api',p/'api.py');rw=load('binding_fixture_rw',p/'rw.py');soft=load('binding_fixture_soft',p/'soft.py')
    mods=[threading,random,logging,weakref,thread];specs=[]
    for mod in mods:
        suffix=mod.__file__.split('/lib/python3.10/',1)[1]
        aliases=[f'/home/dbayha/miniconda3/{prefix}/lib/python3.10/{suffix}' for prefix in ('envs/corl-orig-local','envs/sdbca','pkgs/python-3.10.20-h741d88c_0')]
        specs.append((mod,HASHES[mod.__name__],aliases))
    for mod in (api,rw,soft):specs.append((mod,sha(Path(mod.__file__).read_bytes()),[mod.__file__]))
    saved_native=thread._global_shutdown_lock.acquire
    rows={};binding=None;failure=None;report=None;assertions={}
    handler=logging.StreamHandler();handler.set_name('binding-fixture-handler')
    sentinel=None
    if case=='nonempty_registries':
        class Sentinel:
            def release(self,force=False):raise AssertionError('Sentinel release must never execute')
        sentinel=Sentinel();api._FORK_OBJECTS[17]=sentinel;api._FORK_CLASSES[18]=Sentinel;api._INIT_PARAMETER_MODELS[Sentinel]=('retained',1);api._registry.held['existing']=2;soft._ALL_INSTANCES[17]=sentinel;rw._FORKED_DATABASES._paths.add(Path('/declared-unused-fixture-path'))
    try:
        binding=m.RegistrationBindings(specs,purpose='production' if case=='production_held' else 'registration_binding_fixture_only')
        rows['api']=binding.bind_filelock_api(api);rows['rw']=binding.bind_filelock_readwrite(rw);rows['soft']=binding.bind_filelock_soft(soft)
        rows['random']=binding.bind_python(random,'Random.seed',random._inst.seed,('_inst',),defaults=(None,2),closure={'__class__':random.Random})
        rows['weakref_classmethod']=binding.bind_python(weakref,'finalize._exitfunc',weakref.finalize._exitfunc,('finalize',))
        rows['logging']=binding.bind_logging(logging);rows['thread_exit']=binding.bind_python(thread,'_python_exit')
        rows['native']=[binding.bind_native_lock(thread,('_global_shutdown_lock',),name,getattr(thread._global_shutdown_lock,name)) for name in ('acquire','_at_fork_reinit','release')]
        rows['thread_callback']=binding.bind_python(api,'_thread_callback')
        args=(object(),);kwargs={'first':object(),'second':object()}
        if case=='threading_prior_nonpartial':threading._threading_atexits.append(object())
        prior_thread_count=len(threading._threading_atexits)
        binding.begin_threading(threading,api._thread_callback,args,kwargs)
        returned=threading._register_atexit(api._thread_callback,*args,**kwargs)
        binding.end_threading(returned)
        assertions=dict(actual_internal_registration_once=True,prior_threading_entries=prior_thread_count,new_threading_entries=len(threading._threading_atexits),partial_is_original_callback=threading._threading_atexits[-1].func is api._thread_callback)
        if case=='api_kw_state':api._audit_fork_safety.__kwdefaults__['_state']=api._ForkState()
        if case=='api_kw_dictionary':api._audit_fork_safety.__kwdefaults__=dict(api._audit_fork_safety.__kwdefaults__)
        if case=='api_global_state':api._FORK_STATE=api._ForkState()
        if case=='api_depth':api._FORK_STATE.transition_context.depth=1
        if case=='api_held':api._registry.held['new']=1
        if case=='api_weak_entry':api._FORK_CLASSES[19]=type(handler)
        if case=='api_token':next(api._TRANSITION_TOKENS)
        if case=='api_condition':api._FORK_STATE.gate=threading.Condition(threading.RLock())
        if case=='api_registry_lock':api._FORK_STATE.registry_lock=threading.RLock()
        if case=='rw_depth':rw._SQLITE_TRANSITION_CONTEXT.depth=1
        if case=='rw_pypy':rw._IS_PYPY=True
        if case=='rw_poison':rw._FORKED_DATABASES._all_paths_poisoned=True
        if case=='soft_entry':soft._ALL_INSTANCES[22]=handler
        if case=='random_owner':random._inst=object()
        if case=='random_closure':random.Random.seed.__closure__[0].cell_contents=object
        if case=='random_defaults':random.Random.seed.__defaults__=(None,1)
        if case=='random_code':random.Random.seed.__code__=(lambda self:__class__).__code__ if False else api._ThreadLocalRegistry.__init__.__code__
        if case=='logging_captured_list':logging._handlerList.append(weakref.ref(handler))
        if case=='logging_global_list':logging._handlerList=list(logging._handlerList)
        if case=='logging_handler_method':logging.StreamHandler.flush.__code__=api._thread_callback.__code__
        if case=='logging_lock':logging._lock=threading.RLock()
        if case=='native_owner':thread._global_shutdown_lock=threading.Lock()
        if case=='native_wrong_callback':binding.bind_native_lock(thread,('_global_shutdown_lock',),'acquire',threading.Lock().acquire)
        if case=='native_wrong_name':binding.bind_native_lock(thread,('_global_shutdown_lock',),'release',saved_native)
        if case=='threading_partial_keyword':threading._threading_atexits[-1].keywords['third']=object()
        if case=='threading_existing_change':threading._threading_atexits[0].keywords['new']=object()
        if case=='threading_shutdown':threading._SHUTTING_DOWN=True
        if case=='repeat_internal':binding.begin_threading(threading,api._thread_callback,(),{})
        if case=='repeat_internal_return':binding.end_threading(None)
        if case=='callback_name':api._audit_fork_safety.__name__='altered'
        if case=='callback_qualname':api._audit_fork_safety.__qualname__='altered'
        if case=='callback_module':api._audit_fork_safety.__module__='altered'
        if case=='callback_annotation':api._audit_fork_safety.__annotations__['event']='altered'
        if case=='wrong_code_filename':api._audit_fork_safety.__code__=api._audit_fork_safety.__code__.replace(co_filename='unbound.py')
        if case=='fixture_source_changed':(p/'api.py').write_bytes((p/'api.py').read_bytes()+b'\n# changed fixture only\n')
        if case=='fixture_source_symlink':
            (p/'api.py').rename(p/'api.retained.py');(p/'api.py').symlink_to(p/'api.retained.py')
        if case=='binder_method_changed':binding.freeze=lambda value:None
        binding.check();report=binding.summary()
    except m.BindingRefusal as error:failure=str(error)
    except BaseException as error:
        print(json.dumps(dict(case=case,unexpected_exception=type(error).__name__,detail=str(error))),flush=True);os._exit(3)
    if case in ('complete','nonempty_registries'):
        if failure is not None:print(json.dumps(dict(case=case,unexpected_refusal=failure)),flush=True);os._exit(2)
    elif failure is None:print(json.dumps(dict(case=case,missing_refusal=True)),flush=True);os._exit(4)
    if binding is not None and failure is not None:
        first=binding.first
        try:binding.check()
        except m.BindingRefusal as error:assert str(error)==first==failure
        else:os._exit(5)
    inventory=sorted(n for n in sys.modules if n.split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl'))
    assert inventory==[]
    print(json.dumps(dict(case=case,report=report,bindings=rows,assertions=assertions,first_refusal=failure,first_refusal_preserved=failure is not None,environment=dict(os.environ),target_modules=inventory,snapshot_only=True,preboundary_registrations_observed=False,callback_invocations_by_binder=0,full_route_implemented=False,production_dispatch=False)),flush=True)
    os._exit(0)
def parent(label):
    assert label in ('bindings_tests_v1','bindings_tests_v2','bindings_tests_v3');root=ROOT/label;root.mkdir(parents=True,exist_ok=False)
    data,defs=sources();(W/(label+'_exact_definitions.json')).write_text(json.dumps(defs,indent=2))
    declaration=json.loads((HERE/'native_import_contract_v3.json').read_bytes());results=[]
    for case in CASES:
        p=root/case;p.mkdir(mode=0o700)
        for n,b in data.items():(p/n).write_bytes(b)
        env=dict(declaration['environment'],TMPDIR=str(p/'scratch'))
        cmd=[sys.executable,'-I',str(Path(__file__).resolve()),'--child',case,str(p)]
        started=datetime.datetime.now(datetime.timezone.utc).isoformat();r=subprocess.run(cmd,env=env,capture_output=True,timeout=45)
        for suffix,b in (('stdout',r.stdout),('stderr',r.stderr)):(p/suffix).write_bytes(b)
        receipt=dict(case=case,started=started,ended=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,environment=env,actual_child_exit=r.returncode,fixture_parent=str(p),source_sha256={n:sha(b) for n,b in data.items()},stdout_sha256=sha(r.stdout),stderr_sha256=sha(r.stderr),stdout_bytes=len(r.stdout),stderr_bytes=len(r.stderr))
        (p/'actual_exit.json').write_text(json.dumps(receipt,indent=2));results.append(receipt);(W/(label+'_children.json')).write_text(json.dumps(results,indent=2))
        print(case,r.returncode,flush=True)
        assert r.returncode==0,(case,r.stdout.decode(),r.stderr.decode())
    print(json.dumps(dict(tests=len(CASES),skipped=0,production_dispatch=False)))
if __name__=='__main__':
    if sys.argv[1]=='--child':child(sys.argv[2],Path(sys.argv[3]))
    else:parent(sys.argv[1])
