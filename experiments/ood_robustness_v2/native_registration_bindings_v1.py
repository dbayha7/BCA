"""Concrete callback/state bindings for the still-held native registration route.

No hook installer, callback invocation, import launcher or execution acceptance.
Bindings describe stable registration windows, not all future mutable states.
"""
import collections,dis,functools,hashlib,itertools,os,stat,sys,types,weakref
from pathlib import Path
PARTIAL=functools.partial

class BindingRefusal(BaseException):pass

def digest(raw):return hashlib.sha256(raw).hexdigest()
def source(path):
    p=Path(path);s=p.lstat();fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode))
    if not stat.S_ISREG(s.st_mode) or os.path.realpath(path)!=path:raise BindingRefusal('Canonical regular source required.')
    raw=p.read_bytes()
    if fields(p.lstat())!=fields(s):raise BindingRefusal('Source changed during read.')
    return (digest(raw),fields(s)),raw
def codes(root,path=()):
    for c in root.co_consts:
        if type(c) is types.CodeType:
            q=path+(c.co_name,);yield q,c;yield from codes(c,q)
def resolve(namespace,path):
    v=namespace
    for name in path:
        if type(v) is types.ModuleType:v=v.__dict__[name]
        elif isinstance(v,type):v=type.__getattribute__(v,'__dict__')[name]
        else:raise BindingRefusal('Only explicit module/class owner paths are supported.')
    return v
def function(descriptor):
    if type(descriptor) in (classmethod,staticmethod):return descriptor.__func__
    if type(descriptor) is types.FunctionType:return descriptor
    raise BindingRefusal('Python defining descriptor required.')

class RegistrationBindings:
    def __init__(self,module_specs,purpose='registration_binding_fixture_only'):
        self.first=None;self.pid=os.getpid();self.modules={};self.functions=[];self.natives=[];self.states=[];self.kept={};self.thread_pending=None;self.thread_events=[]
        if purpose!='registration_binding_fixture_only':self.fail('Full actual route requires separate production integration and prereview.')
        self.source_path=__file__;self.source_pin,self.raw=source(self.source_path)
        tree=compile(self.raw,self.source_path,'exec',dont_inherit=True);compiled=dict(codes(tree))
        self.methods={n:(v,v.__code__,v.__defaults__,v.__kwdefaults__) for n,v in type(self).__dict__.items() if type(v) is types.FunctionType}
        for n,(f,*_) in self.methods.items():
            if f.__code__!=compiled[('RegistrationBindings',n)]:self.fail('Binding method differs from full source.')
        self.method_names=tuple(self.methods)
        self.globals={n:globals()[n] for n in ('source','digest','codes','resolve','function','BindingRefusal','os','sys','stat','types','dis','weakref','functools','itertools','collections','Path','hashlib','PARTIAL')}
        self.helpers={n:(v,v.__code__,v.__defaults__) for n,v in self.globals.items() if type(v) is types.FunctionType}
        for n,(v,code,_) in self.helpers.items():
            if code!=compiled[(n,)]:self.fail('Binding helper differs from source.')
        for module,expected_sha,aliases in module_specs:
            if type(module) is not types.ModuleType or module.__name__.split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw'):self.fail('Actual target namespace is held; explicit fixture/stdlib modules only.')
            path=module.__dict__.get('__file__');pin,raw=source(path)
            if pin[0]!=expected_sha or not aliases or aliases[0]!=path:self.fail('External module source binding differs.')
            for alias in aliases:
                if source(alias)[0]!=pin:self.fail('Explicit shared source alias differs.')
            self.modules[module.__name__]=dict(module=module,path=path,pin=pin,aliases=tuple(aliases),namespace=module.__dict__,compiled=dict(codes(compile(raw,path,'exec',dont_inherit=True))))
        self.check()

    def fail(self,message):
        if self.first is None:self.first=message
        raise BindingRefusal(self.first)

    def freeze(self,value,seen=None):
        """Explicit containers/native forms only; opaque leaves record identity."""
        seen=set() if seen is None else seen;t=type(value)
        if value is None or t in (str,int,bool,bytes):return (id(t),value)
        if t is float:return (id(t),value.hex())
        if id(value) in seen:return ('cycle',id(value),id(t))
        seen.add(id(value))
        if t is dict:return ('dict',id(value),tuple((self.freeze(k,seen),self.freeze(v,seen)) for k,v in value.items()))
        if t in (list,tuple,collections.deque):return (t.__name__,id(value),tuple(self.freeze(v,seen) for v in value))
        if t in (set,frozenset):return (t.__name__,id(value),frozenset(self.freeze(v,seen) for v in value))
        if t is types.MethodType:return ('python_method',id(value.__self__),id(value.__func__),id(value.__func__.__code__))
        if t is types.BuiltinMethodType:
            return ('native_method',id(value.__self__),id(type(value.__self__)),value.__name__)
        if t is types.FunctionType:return ('function',id(value),id(value.__code__),id(value.__defaults__),id(value.__kwdefaults__))
        if t in (weakref.ReferenceType,weakref.KeyedRef):
            target=value()
            if target is not None:self.kept[id(target)]=target
            return ('weakref',id(value),id(target),id(t))
        if t is functools.partial:return ('partial',id(value),self.freeze(value.func,seen),self.freeze(value.args,seen),self.freeze(value.keywords,seen))
        return ('opaque_identity',id(value),id(t))

    def bind_python(self,module,qualname,callback=None,owner_path=None,defaults=None,kwdefaults=None,closure=None):
        self.check();info=self.modules[module.__name__];path=tuple(qualname.split('.'));descriptor=resolve(module,path);fn=function(descriptor)
        expected=info['compiled'].get(path)
        if expected is None or fn.__code__!=expected or fn.__code__.co_filename!=info['path'] or fn.__globals__ is not info['namespace'] or fn.__module__!=module.__name__ or fn.__name__!=path[-1] or fn.__qualname__!=qualname:self.fail('Python callback code/namespace/metadata differs.')
        if callback is None:callback=fn
        owner=None
        if type(callback) is types.MethodType:
            if owner_path is None:self.fail('Bound Python/class callback needs an explicit owning path.')
            owner=resolve(module,tuple(owner_path))
            if callback.__self__ is not owner or callback.__func__ is not fn:self.fail('Bound callback owner/function differs.')
            if type(descriptor) is classmethod and owner is not resolve(module,path[:-1]):self.fail('Classmethod defining owner differs.')
            if type(descriptor) is types.FunctionType and type(owner) is not resolve(module,path[:-1]):self.fail('Instance method defining class differs.')
        elif callback is not fn or owner_path is not None:self.fail('Plain callback or owning path differs.')
        expected_defaults=() if defaults is None else defaults;actual_defaults=() if fn.__defaults__ is None else fn.__defaults__
        if len(actual_defaults)!=len(expected_defaults) or any(self.freeze(a)!=self.freeze(b) for a,b in zip(actual_defaults,expected_defaults)):self.fail('Captured positional defaults differ.')
        if (fn.__defaults__ is None)!=(defaults is None):self.fail('Positional default tuple presence differs.')
        if (fn.__kwdefaults__ is None)!=(kwdefaults is None):self.fail('Keyword default dictionary presence differs.')
        if kwdefaults is not None:
            if list(fn.__kwdefaults__)!=list(kwdefaults) or any(fn.__kwdefaults__[k] is not v for k,v in kwdefaults.items()):self.fail('Captured keyword default references differ.')
        closure={} if closure is None else closure
        if tuple(closure)!=fn.__code__.co_freevars or len(fn.__closure__ or ())!=len(closure):self.fail('Exact declared closure required.')
        cells=tuple(fn.__closure__ or ())
        if any(c.cell_contents is not v for c,v in zip(cells,closure.values())):self.fail('Captured closure owner differs.')
        names={op.argval for c in (fn.__code__,*(c for _,c in codes(fn.__code__))) for op in dis.get_instructions(c) if op.opname=='LOAD_GLOBAL'}
        direct={n:fn.__globals__[n] if n in fn.__globals__ else fn.__builtins__[n] for n in names}
        pin=dict(module=module,path=path,descriptor=descriptor,fn=fn,code=fn.__code__,metadata=(fn.__name__,fn.__qualname__,fn.__module__),annotations=fn.__annotations__,annotation_value=self.freeze(fn.__annotations__),defaults=fn.__defaults__,default_value=self.freeze(fn.__defaults__),kw=fn.__kwdefaults__,kwvalue=self.freeze(fn.__kwdefaults__),cells=cells,cellvalues=tuple(closure.values()),direct=direct,owner=owner,owner_path=owner_path)
        self.functions.append(pin);self.check();return dict(form='bound_python' if type(callback) is types.MethodType else 'plain_python',function_id=id(fn),owner_id=None if owner is None else id(owner),source=fn.__code__.co_filename,line=fn.__code__.co_firstlineno,freevars=list(fn.__code__.co_freevars),positional_default_ids=[id(v) for v in actual_defaults],keyword_default_ids=None if fn.__kwdefaults__ is None else [[k,id(v)] for k,v in fn.__kwdefaults__.items()],direct_global_names=sorted(direct))

    def bind_native_lock(self,module,owner_path,name,callback):
        self.check();owner=resolve(module,tuple(owner_path));native_type=type(owner)
        if native_type.__module__!='_thread' or native_type.__name__ not in ('lock','RLock'):self.fail('Explicit native lock owner required.')
        descriptor=type.__getattribute__(native_type,'__dict__').get(name)
        if type(callback) is not types.BuiltinMethodType or callback.__self__ is not owner or callback.__name__!=name or descriptor is None or type(descriptor) is not types.MethodDescriptorType:self.fail('Native bound lock owner/descriptor differs.')
        if name not in ('acquire','release','_at_fork_reinit'):self.fail('Undeclared native lock registration method.')
        if descriptor.__get__(owner,native_type)!=callback:self.fail('Native descriptor does not resolve the supplied callback.')
        self.natives.append((module,tuple(owner_path),owner,native_type,name,descriptor,callback))
        self.check();return dict(form='native_bound_lock',owner_id=id(owner),type_id=id(native_type),descriptor_id=id(descriptor),name=name,args=None,kwargs=None)

    def _snapshot_api(self,module):
        ns=module.__dict__;state=ns['_FORK_STATE'];ctx=state.transition_context;registry=ns['_registry']
        if type(state) is not ns['_ForkState'] or type(ctx) is not ns['_ForkTransitionContext'] or type(registry) is not ns['_ThreadLocalRegistry']:self.fail('Exact filelock state classes required.')
        fields=('gate','registry_lock','parameter_models_lock','transition_context','transitions','active_transitions','admission_closed','fork_owner_depths','pinned_objects','pinned_classes','provisional_descriptor_tokens','pid')
        if set(vars(state))!=set(fields) or state.pid!=self.pid or type(state.active_transitions) is not int or type(state.admission_closed) is not bool:self.fail('Filelock fork-state fields differ.')
        if type(ctx.depth) is not int or type(registry.held) is not dict:self.fail('Filelock thread-local state differs.')
        if type(state.gate) is not ns['Condition'] or type(state.registry_lock).__module__!='_thread' or type(state.parameter_models_lock) is not type(state.registry_lock):self.fail('Filelock synchronization owners differ.')
        weak=[]
        for name,cls in (('_FORK_OBJECTS',weakref.WeakValueDictionary),('_FORK_CLASSES',weakref.WeakValueDictionary),('_INIT_PARAMETER_MODELS',weakref.WeakKeyDictionary)):
            value=ns[name]
            if type(value) is not cls:self.fail('Actual filelock weak registry required.')
            weak.append((name,id(value),self.freeze(vars(value))))
        token=ns['_TRANSITION_TOKENS']
        if type(token) is not itertools.count or type(ns['_OWNED_DESCRIPTORS']) is not dict:self.fail('Descriptor/token registry types differ.')
        # Native count.__reduce__ observes its current integer without advancing it.
        count_state=itertools.count.__reduce__(token)
        return (self.freeze(vars(state)),self.freeze(vars(state.gate)),self.freeze(vars(ctx)),ctx.depth,id(registry),self.freeze(registry.held),tuple(weak),self.freeze(ns['_OWNED_DESCRIPTORS']),id(token),count_state[1],id(ns['_FORK_AUDIT_EVENTS']),frozenset(ns['_FORK_AUDIT_EVENTS']))

    def bind_filelock_api(self,module):
        ns=module.__dict__;self.check()
        self.bind_python(module,'_ForkState.__init__');self.bind_python(module,'_ForkState.reset_synchronization')
        self.bind_python(module,'_ThreadLocalRegistry.__init__',closure={'__class__':ns['_ThreadLocalRegistry']})
        rows=[]
        for name in ('_audit_fork_safety','_pin_fork_objects','_resume_parent_after_fork','_reset_child_after_fork'):
            kw={'_fork_events':ns['_FORK_AUDIT_EVENTS'],'_state':ns['_FORK_STATE']} if name=='_audit_fork_safety' else None
            rows.append(self.bind_python(module,name,kwdefaults=kw))
        if ns['_FORK_AUDIT_EVENTS']!=frozenset(('os.fork','os.forkpty')):self.fail('Fork audit event set differs.')
        snap=self._snapshot_api(module);self.states.append(('api',module,snap));self.check()
        return dict(callbacks=rows,state_id=id(ns['_FORK_STATE']),context_id=id(ns['_FORK_STATE'].transition_context),gate_id=id(ns['_FORK_STATE'].gate),registry_held_id=id(ns['_registry'].held),weak_registry_sizes={n:len(vars(ns[n])['data']) for n in ('_FORK_OBJECTS','_FORK_CLASSES','_INIT_PARAMETER_MODELS')},transitive_helper_graph_accepted=False)

    def _snapshot_rw(self,module):
        ns=module.__dict__;ctx=ns['_SQLITE_TRANSITION_CONTEXT'];db=ns['_FORKED_DATABASES']
        if type(ctx) is not ns['_SQLiteTransitionContext'] or type(ctx.depth) is not int or type(db) is not ns['_ForkedDatabaseRegistry']:self.fail('Read/write state class differs.')
        if set(vars(db))!={'_lock','_paths','_identities','_sqlite_used','_all_paths_poisoned'} or type(db._paths) is not set or type(db._identities) is not set or type(db._sqlite_used) is not bool or type(db._all_paths_poisoned) is not bool:self.fail('Read/write database registry fields differ.')
        if ns['_IS_PYPY'] is not False or ns['_UNSAFE_FORK_EXIT_STATUS']!=70:self.fail('Pinned CPython read/write branch differs.')
        return (id(ctx),self.freeze(vars(ctx)),ctx.depth,id(db),self.freeze(vars(db)),id(ns['_SQLiteTransitionContext']),id(ns['_ForkedDatabaseRegistry']))

    def bind_filelock_readwrite(self,module):
        self.check();self.bind_python(module,'_ForkedDatabaseRegistry.__init__');row=self.bind_python(module,'_abort_forked_sqlite_transition')
        self.states.append(('rw',module,self._snapshot_rw(module)));self.check();return dict(callback=row,registry_id=id(module.__dict__['_FORKED_DATABASES']),sqlite_audit_registration_source_predicted_skipped=True,actual_registration_count=None,escrow_full_graph_accepted=False)

    def _snapshot_soft(self,module):
        ns=module.__dict__;value=ns['_ALL_INSTANCES'];lock=ns['_ALL_INSTANCES_LOCK']
        if type(value) is not weakref.WeakValueDictionary or type(lock).__module__!='_thread' or type(lock).__name__!='lock':self.fail('Soft read/write weak registry/native lock differs.')
        return (id(value),self.freeze(vars(value)),id(lock),id(type(lock)))

    def bind_filelock_soft(self,module):
        self.check();row=self.bind_python(module,'_cleanup_all_instances');self.states.append(('soft',module,self._snapshot_soft(module)));self.check()
        return dict(callback=row,registry_id=id(module.__dict__['_ALL_INSTANCES']),entries=len(module.__dict__['_ALL_INSTANCES'].data),future_release_graph_accepted=False)

    def _snapshot_logging(self,module):
        ns=module.__dict__;refs=ns['_handlerList'];handlers=ns['_handlers'];weakset=ns['_at_fork_reinit_lock_weakset']
        if type(refs) is not list or type(handlers) is not weakref.WeakValueDictionary or type(weakset) is not weakref.WeakSet:self.fail('Actual logging captured/weak registries required.')
        rows=[]
        for ref in refs:
            if type(ref) is not weakref.ReferenceType:self.fail('Logging weak handler reference differs.')
            handler=ref()
            if handler is not None:
                self.kept[id(handler)]=handler;methods=[]
                for name in ('acquire','flush','close','release','_at_fork_reinit'):
                    owner=next((c for c in type(handler).__mro__ if name in vars(c)),None)
                    if owner is None:self.fail('Handler method owner unavailable.')
                    fn=vars(owner)[name]
                    if type(fn) is not types.FunctionType:self.fail('Unsupported logging handler method form.')
                    info=self.modules[module.__name__]
                    if fn.__globals__ is not module.__dict__ or fn.__code__!=info['compiled'].get((owner.__name__,name)):self.fail('Handler code is outside the pinned logging source.')
                    methods.append((name,id(owner),id(fn),id(fn.__code__),id(fn.__defaults__),id(fn.__kwdefaults__)))
                rows.append((id(ref),id(handler),id(type(handler)),self.freeze(vars(handler)),tuple(methods)))
        return (id(refs),self.freeze(refs),id(handlers),self.freeze(vars(handlers)),id(ns['_lock']),id(type(ns['_lock'])),id(weakset),self.freeze(vars(weakset)),tuple(rows))

    def bind_logging(self,module):
        self.check();rows=[]
        for name in ('_acquireLock','_after_at_fork_child_reinit_locks','_releaseLock','shutdown'):
            rows.append(self.bind_python(module,name,defaults=(module.__dict__['_handlerList'],) if name=='shutdown' else None))
        self.states.append(('logging',module,self._snapshot_logging(module)));self.check();return dict(callbacks=rows,handler_list_id=id(module.__dict__['_handlerList']),live_handler_entries=len(module.__dict__['_handlerList']),preboundary_registration_count=None,full_handler_transitive_graph_accepted=False)

    def begin_threading(self,module,callback,args,kwargs):
        self.check()
        if self.thread_pending is not None or self.thread_events:self.fail('Internal threading registration is single-use.')
        self.bind_python(module,'_register_atexit')
        ns=module.__dict__
        if ns['_SHUTTING_DOWN'] is not False or type(ns['_threading_atexits']) is not list or type(args) is not tuple or type(kwargs) is not dict:self.fail('Internal threading state/arguments differ.')
        if any(type(v) is not PARTIAL for v in ns['_threading_atexits']):self.fail('Existing internal registration is not an actual partial.')
        if ns['functools'] is not functools or functools.partial is not PARTIAL:self.fail('Original partial constructor differs.')
        self.thread_pending=(module,ns['_threading_atexits'],tuple(ns['_threading_atexits']),self.freeze(ns['_threading_atexits']),callback,args,kwargs,functools.partial)
        self.thread_events.append(dict(event='internal_request',callback_id=id(callback),argument_ids=[id(v) for v in args],ordered_keyword_ids=[[k,id(v)] for k,v in kwargs.items()],prior_entries=len(ns['_threading_atexits'])))

    def end_threading(self,result):
        self.check()
        if self.thread_pending is None:self.fail('No pending internal registration.')
        module,items,old,oldprint,callback,args,kwargs,partial_type=self.thread_pending;ns=module.__dict__
        if result is not None or ns['_threading_atexits'] is not items or ns['_SHUTTING_DOWN'] is not False or len(items)!=len(old)+1 or any(a is not b for a,b in zip(items,old)):self.fail('Internal partial append/list/flag differs.')
        # Check the retained earlier partial objects without requiring a new list id.
        for a in old:
            if self.freeze(a)!=next(v for v in oldprint[2] if v[1]==id(a)):self.fail('Existing threading registration changed.')
        new=items[-1]
        if type(new) is not partial_type or new.func is not callback or len(new.args)!=len(args) or any(a is not b for a,b in zip(new.args,args)) or list(new.keywords)!=list(kwargs) or any(new.keywords[k] is not v for k,v in kwargs.items()):self.fail('Actual threading partial fields differ.')
        self.thread_events.append(dict(event='internal_return',partial_id=id(new),func_id=id(new.func),args_id=id(new.args),kwargs_id=id(new.keywords),ordered_keyword_ids=[[k,id(v)] for k,v in new.keywords.items()],retained_prior_entries=len(old)))
        self.states.append(('threading',module,(id(items),self.freeze(items),ns['_SHUTTING_DOWN'],partial_type)));self.thread_pending=None;self.check()

    def check(self):
        if self.first is not None:raise BindingRefusal(self.first)
        try:
            if os.getpid()!=self.pid or source(self.source_path)[0]!=self.source_pin: self.fail('Binding source/process changed.')
            if tuple(self.methods)!=self.method_names or any(globals()[n] is not v for n,v in self.globals.items()):self.fail('Binder method inventory/direct global changed.')
            for n,(f,code,defaults,kw) in self.methods.items():
                if n in self.__dict__ or type(self).__dict__.get(n) is not f or f.__code__ is not code or f.__defaults__ is not defaults or f.__kwdefaults__ is not kw:self.fail('Binder live method changed.')
            for f,code,defaults in self.helpers.values():
                if f.__code__ is not code or f.__defaults__ is not defaults:self.fail('Binder helper changed.')
            for name,info in self.modules.items():
                if sys.modules.get(name) is not info['module'] or info['module'].__dict__ is not info['namespace'] or info['namespace'].get('__file__')!=info['path']:self.fail('Registered source namespace changed.')
                for alias in info['aliases']:
                    if source(alias)[0]!=info['pin']:self.fail('Module/source alias changed.')
            for p in self.functions:
                fn=p['fn']
                if resolve(p['module'],p['path']) is not p['descriptor'] or function(p['descriptor']) is not fn or fn.__code__ is not p['code'] or fn.__defaults__ is not p['defaults'] or fn.__kwdefaults__ is not p['kw'] or self.freeze(fn.__defaults__)!=p['default_value'] or self.freeze(fn.__kwdefaults__)!=p['kwvalue']:self.fail('Python callback owner/code/default changed.')
                if (fn.__name__,fn.__qualname__,fn.__module__)!=p['metadata'] or fn.__annotations__ is not p['annotations'] or self.freeze(fn.__annotations__)!=p['annotation_value']:self.fail('Python callback metadata/annotations changed.')
                if len(fn.__closure__ or ())!=len(p['cells']) or any(a is not b or a.cell_contents is not v for a,b,v in zip(fn.__closure__ or (),p['cells'],p['cellvalues'])):self.fail('Python callback closure changed.')
                if p['owner_path'] is not None and resolve(p['module'],tuple(p['owner_path'])) is not p['owner']:self.fail('Python callback owning object changed.')
                for n,v in p['direct'].items():
                    if (fn.__globals__[n] if n in fn.__globals__ else fn.__builtins__[n]) is not v:self.fail('Callback direct global changed.')
            for module,path,owner,cls,name,desc,cb in self.natives:
                if resolve(module,path) is not owner or type(owner) is not cls or vars(cls).get(name) is not desc or cb.__self__ is not owner or cb.__name__!=name or desc.__get__(owner,cls)!=cb:self.fail('Native callback owner/descriptor changed.')
            for kind,module,expected in self.states:
                if kind=='threading':
                    ns=module.__dict__;actual=(id(ns['_threading_atexits']),self.freeze(ns['_threading_atexits']),ns['_SHUTTING_DOWN'],functools.partial)
                else:actual={'api':self._snapshot_api,'rw':self._snapshot_rw,'soft':self._snapshot_soft,'logging':self._snapshot_logging}[kind](module)
                if actual!=expected:self.fail('Bound '+kind+' registration state changed.')
            return True
        except BindingRefusal as error:self.fail(str(error))
        except BaseException as error:self.fail('Binding check failed: '+type(error).__name__)

    def summary(self):
        self.check();return dict(bound_python_functions=len(self.functions),bound_native_lock_methods=len(self.natives),bound_state_windows=[k for k,_,_ in self.states],internal_threading_events=list(self.thread_events),callback_invocations_by_binder=0,hook_installs_by_binder=0,first_refusal=self.first,preboundary_registrations_observed=False,full_transitive_graph_accepted=False,full_filelock_namespace_accepted=False,production_dispatch=False,scientific_execution_accepted=False)
