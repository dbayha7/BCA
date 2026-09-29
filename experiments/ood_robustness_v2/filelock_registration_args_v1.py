"""Source-bound observation of actual hook registration arguments.

Cooperative instrumentation, not a sandbox. Full live dependency/state acceptance,
audit-hook installation effectiveness and shutdown/fork execution are external.
Wrappers forward original callback objects; they never invoke/replay callbacks.
"""
import atexit
import dis
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types


class RegistrationRefusal(BaseException):
    pass


def codes(code):
    yield code
    for value in code.co_consts:
        if type(value) is types.CodeType:
            yield from codes(value)


def pin(path):
    p=Path(path)
    if not p.is_absolute() or p.resolve(strict=True)!=p or '..' in p.parts:
        raise ValueError('Canonical source required.')
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or not 0<s.st_size<2_000_000:
        raise ValueError('Bounded unaliased source required.')
    raw=p.read_bytes();after=p.lstat()
    fields=lambda x:[x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns]
    if fields(s)!=fields(after):raise ValueError('Source changed during read.')
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),stat=fields(s))


def metadata(value):
    if type(value) is types.FunctionType:
        return dict(type='function',id=id(value),module=value.__module__,name=value.__qualname__,
                    filename=value.__code__.co_filename,line=value.__code__.co_firstlineno)
    if type(value) in (str,int,bool) or value is None:
        return dict(type=type(value).__name__,value=value)
    return dict(type=type(value).__name__,id=id(value))


def globals_used(code):
    return {op.argval for c in codes(code) for op in dis.get_instructions(c) if op.opname=='LOAD_GLOBAL'}


class RegistrationArguments:
    """Validate declared callsites and callback bindings before native forwarding.

state_verifier(namespace, phase) must raise on invalid state. Component fixtures
use synthetic state; the verifier's transitive references are NOT self-certified.
The installed wrappers and poison remain active through process exit.
"""
    def __init__(self,*,module_name,source_pin,sites,keyword_references,state_verifier):
        self.pid=os.getpid();self.module_name=module_name
        self.source_pin=json.loads(json.dumps(source_pin));self.sites=json.loads(json.dumps(sites))
        self.keyword_references=json.loads(json.dumps(keyword_references))
        self.contract=json.dumps([self.module_name,self.source_pin,self.sites,self.keyword_references],sort_keys=True)
        self.first_refusal=None;self.failed=False;self.active=False;self.used=False;self.finished=False
        self.events=[];self.index=0;self.native_pending=None;self.native_seen=None
        self.install_token=object();self.install_seen=False
        self.module=None;self.namespace=None;self.bound={}
        self.original={'audit':sys.addaudithook,'fork':os.register_at_fork,'exit':atexit.register}
        self.original_pins=dict(self.original)
        if type(state_verifier) is not types.FunctionType or state_verifier.__closure__ is not None:
            self.refuse('Plain external state verifier required.')
        self.verifier=state_verifier;self.verifier_object=state_verifier
        self.verifier_pin=(state_verifier.__code__,state_verifier.__defaults__,state_verifier.__kwdefaults__)
        if pin(self.source_pin['path'])!=self.source_pin:self.refuse('Stale source pin.')
        code=compile(Path(self.source_pin['path']).read_bytes(),self.source_pin['path'],'exec',dont_inherit=True)
        self.compiled=list(codes(code));self.module_code=code
        self.wrappers={kind:self._wrapper(kind) for kind in self.original}
        self.wrapper_objects=dict(self.wrappers)
        self.wrapper_pins={k:(f.__code__,f.__defaults__,f.__kwdefaults__,tuple(c.cell_contents for c in f.__closure__)) for k,f in self.wrappers.items()}
        self.check()

    def _wrapper(self,kind):
        def wrapper(*args,**kwargs):
            return self._invoke(kind,sys._getframe(1),args,kwargs)
        return wrapper

    def refuse(self,message,kind=None,frame=None,args=None,kwargs=None):
        if self.first_refusal is None:
            self.first_refusal=dict(message=message,kind=kind,
                caller=None if frame is None else dict(filename=frame.f_code.co_filename,line=frame.f_lineno),
                args=None if args is None else [metadata(v) for v in args],
                kwargs=None if kwargs is None else [[k,metadata(v)] for k,v in kwargs.items()])
        self.failed=True
        raise RegistrationRefusal(self.first_refusal['message'])

    def _global(self,fn,name):
        if name in fn.__globals__:return fn.__globals__[name]
        return fn.__builtins__[name]

    def _bind_callback(self,name,value):
        if type(value) is not types.FunctionType or self.namespace.get(name) is not value:
            self.refuse('Callback object differs from declared namespace.')
        matches=[c for c in self.compiled if c.co_name==name]
        if len(matches)!=1 or value.__code__!=matches[0] or value.__code__.co_filename!=self.source_pin['path'] or value.__globals__ is not self.namespace or value.__module__!=self.module_name or value.__qualname__!=name:
            self.refuse('Callback defining code/namespace differs.')
        if value.__closure__ is not None or value.__defaults__ is not None:
            self.refuse('Unexpected callback closure or positional defaults.')
        refs=self.keyword_references.get(name,{})
        actual=value.__kwdefaults__ or {}
        if list(actual)!=list(refs) or any(actual[k] is not self.namespace[n] for k,n in refs.items()):
            self.refuse('Captured keyword default reference differs.')
        item=dict(fn=value,code=value.__code__,kw_object=value.__kwdefaults__,kw=dict(actual),
                  names=(value.__module__,value.__name__,value.__qualname__),
                  globals={n:self._global(value,n) for n in globals_used(value.__code__)})
        if name in self.bound and self.bound[name]['fn'] is not value:self.refuse('Callback rebound.')
        self.bound[name]=item

    def check(self):
        try:
            if self.failed:self.refuse('Previous registration refusal.')
            if self.pid!=os.getpid():self.refuse('Owner process changed.')
            if json.dumps([self.module_name,self.source_pin,self.sites,self.keyword_references],sort_keys=True)!=self.contract:
                self.refuse('Registration contract changed.')
            if pin(self.source_pin['path'])!=self.source_pin:self.refuse('Source bytes or identity changed.')
            if self.verifier is not self.verifier_object or (self.verifier.__code__,self.verifier.__defaults__,self.verifier.__kwdefaults__)!=self.verifier_pin:
                self.refuse('State verifier changed.')
            if set(self.original)!=set(self.original_pins) or any(self.original[k] is not v for k,v in self.original_pins.items()):
                self.refuse('Original native registration API changed.')
            if set(self.wrappers)!=set(self.wrapper_objects):self.refuse('Registration wrapper inventory changed.')
            for k,f in self.wrappers.items():
                if f is not self.wrapper_objects[k] or (f.__code__,f.__defaults__,f.__kwdefaults__,tuple(c.cell_contents for c in f.__closure__))!=self.wrapper_pins[k]:self.refuse('Registration wrapper changed.')
            if self.active:
                if sys.addaudithook is not self.wrappers['audit'] or os.register_at_fork is not self.wrappers['fork'] or atexit.register is not self.wrappers['exit'] or sys.getprofile()!=self.profile:
                    self.refuse('Registration observer API/profile changed.')
            if self.module is not None:
                if sys.modules.get(self.module_name) is not self.module or self.module.__dict__ is not self.namespace or self.namespace.get('__file__')!=self.source_pin['path'] or self.namespace.get('__name__')!=self.module_name:
                    self.refuse('Registered namespace replaced.')
                for name,b in self.bound.items():
                    f=b['fn']
                    if self.namespace.get(name) is not f or f.__code__!=b['code'] or (f.__module__,f.__name__,f.__qualname__)!=b['names'] or f.__globals__ is not self.namespace or f.__defaults__ is not None or f.__closure__ is not None or f.__kwdefaults__ is not b['kw_object']:
                        self.refuse('Bound callback changed.')
                    if list(f.__kwdefaults__ or {})!=list(b['kw']) or any(f.__kwdefaults__[k] is not v for k,v in b['kw'].items()):
                        self.refuse('Captured keyword defaults changed.')
                    if any(self._global(f,n) is not v for n,v in b['globals'].items()):self.refuse('Direct callback global reference changed.')
                    for k,n in self.keyword_references.get(name,{}).items():
                        if f.__kwdefaults__[k] is not self.namespace[n]:self.refuse('Captured state no longer matches module state.')
                self._verify_phase('check')
        except BaseException as error:
            if self.first_refusal is not None:raise RegistrationRefusal(self.first_refusal['message'])
            self.refuse('Identity/state check failed: '+type(error).__name__)

    def _verify_phase(self,phase):
        try:
            if self.verifier(self.namespace,phase) is not None:self.refuse('State verifier must return None after checks.')
        except BaseException as error:
            if self.first_refusal is not None:raise RegistrationRefusal(self.first_refusal['message'])
            self.refuse('State verifier refused '+phase+': '+type(error).__name__)

    def profile(self,frame,event,arg):
        if not self.active:return
        kind=next((k for k,v in self.original.items() if arg is v),None) if event.startswith('c_') else None
        if kind is None:return
        if self.failed:self.refuse('Native registration after refusal.',kind,frame)
        if self.native_pending!=kind or frame.f_code is not type(self)._invoke.__code__:
            self.refuse('Direct native registration bypass.',kind,frame)
        if event=='c_call':
            if self.native_seen is not None:self.refuse('Repeated native registration entry.',kind,frame)
            self.native_seen='entry';self.events.append(dict(event='native_entry',kind=kind,index=self.index))
        elif event=='c_return':
            if self.native_seen!='entry':self.refuse('Unmatched native registration return.',kind,frame)
            self.native_seen='return';self.events.append(dict(event='native_return',kind=kind,index=self.index))
        elif event=='c_exception':self.refuse('Native registration raised.',kind,frame)

    def audit(self,event,args):
        if event=='ood.registration_args.install' and len(args)==1 and args[0] is self.install_token:
            self.install_seen=True;return
        if not self.active:return
        if event=='sys.addaudithook':
            if self.native_pending!='audit' or self.native_seen!='entry':self.refuse('Unbound audit registration event.',event,args=args)
        elif event=='sys.setprofile':self.refuse('Profile replacement refused.',event,args=args)
        elif event.startswith(('subprocess.','socket.','fcntl.')) or event in ('os.system','os.fork','os.forkpty','os.exec','os.posix_spawn','os.spawn','os.putenv','os.unsetenv'):
            self.refuse('Process/network/environment/lock route refused.',event,args=args)
        elif event=='open' and isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            self.refuse('Write outside registration observation refused.',event,args=args)
        elif event in ('os.mkdir','os.remove','os.rmdir','os.link','os.symlink','os.rename','os.utime','os.chmod','os.chown','os.truncate','shutil.rmtree'):
            self.refuse('Mutation outside registration observation refused.',event,args=args)

    def _invoke(self,kind,frame,args,kwargs):
        try:
            self.check()
            if not self.active or self.finished or self.native_pending is not None or self.index>=len(self.sites):
                self.refuse('Repeated or incomplete registration request.',kind,frame,args,kwargs)
            site=self.sites[self.index]
            if site['kind']!=kind or frame.f_code.co_filename!=self.source_pin['path'] or frame.f_code.co_name!=site['caller'] or frame.f_lineno!=site['line'] or frame.f_code not in self.compiled:
                self.refuse('Registration order/callsite differs.',kind,frame,args,kwargs)
            module=sys.modules.get(self.module_name)
            if type(module) is not types.ModuleType or module.__dict__ is not frame.f_globals or frame.f_globals.get('__name__')!=self.module_name or frame.f_globals.get('__file__')!=self.source_pin['path']:
                self.refuse('Registered caller namespace required.',kind,frame,args,kwargs)
            if self.module is None:self.module=module;self.namespace=module.__dict__
            if len(args)!=len(site['args']) or list(kwargs)!=list(site['kwargs']):
                self.refuse('Registration argument shape differs.',kind,frame,args,kwargs)
            for name,value in zip(site['args'],args):self._bind_callback(name,value)
            for key,name in site['kwargs'].items():self._bind_callback(name,kwargs[key])
            self.check();self._verify_phase('before:'+str(self.index))
            self.events.append(dict(event='request',kind=kind,index=self.index,caller=site['caller'],line=frame.f_lineno,
                                    args=[metadata(v) for v in args],kwargs=[[k,metadata(v)] for k,v in kwargs.items()]))
            self.native_pending=kind;self.native_seen=None
            result=self.original[kind](*args,**kwargs)
            if self.native_seen!='return':self.refuse('Native registration entry/return not observed.')
            if result is not (args[0] if kind=='exit' else None):self.refuse('Unexpected native registration return value.')
            self.native_pending=None;self.native_seen=None
            self.check();self._verify_phase('after:'+str(self.index));self.index+=1
            return result
        except BaseException as error:
            if self.first_refusal is None:self.refuse('Registration binding failed: '+type(error).__name__,kind,frame,args,kwargs)
            self.failed=True
            # Retain actual request arguments even when an inner identity check refused first.
            if self.first_refusal['caller'] is None and self.first_refusal['kind'] is None:
                self.first_refusal.update(kind=kind,caller=dict(filename=frame.f_code.co_filename,line=frame.f_lineno),
                                         args=[metadata(v) for v in args],kwargs=[[k,metadata(v)] for k,v in kwargs.items()])
            raise RegistrationRefusal(self.first_refusal['message'])

    def install(self):
        self.check()
        if self.used or sys.getprofile() is not None:self.refuse('Single-use observer requires no prior profile.')
        self.used=True
        try:
            self.original['audit'](self.audit)
            sys.audit('ood.registration_args.install',self.install_token)
        except BaseException as error:self.refuse('Observer audit installation raised: '+type(error).__name__)
        if not self.install_seen:self.refuse('Observer audit hook installation was suppressed.')
        sys.setprofile(self.profile)
        sys.addaudithook=self.wrappers['audit'];os.register_at_fork=self.wrappers['fork'];atexit.register=self.wrappers['exit']
        self.active=True

    def complete(self):
        self.check()
        if not self.active or self.finished or self.index!=len(self.sites) or self.native_pending is not None:
            self.refuse('Missing or repeated completion.')
        self._verify_phase('complete');self.finished=True
        return dict(registration_requests=self.index,native_entry_returns_observed=self.index,
                    callback_invocations_by_observer=0,events=list(self.events),owner_pid=self.pid,
                    audit_installation_effectiveness_accepted=False,full_live_dependency_graph_accepted=False,
                    real_filelock_namespace_accepted=False,fork_shutdown_lifecycle_accepted=False,
                    native_environment_accepted=False,scientific_execution_accepted=False)
