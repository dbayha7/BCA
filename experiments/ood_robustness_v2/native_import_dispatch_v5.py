"""Concrete v3 bootstrap/diagnostic dispatcher; native import phases remain held.

One audit/profile owner observes exact worker-owned creation and diagnostic native
calls. This is cooperative instrumentation, not a security sandbox, native-C
syscall monitor, full dependency-graph acceptance or permission to import targets.
No cleanup/uninstall method exists. The enclosing pinned worker must os._exit.
"""
import ast
import atexit
import dis
import re
import shutil
import tempfile
import weakref
from pathlib import Path
import hashlib
import json
import os
import stat
import sys
import types

DECLARATION_SHA='5f922e3d0d27116139fd99293f537427f6d369f702b7fd685dbbe9de67567763'
FILES=('events.jsonl','final.json','modules.json','maps.txt')
MAX_FILE=8*1024*1024
MAX_TOTAL=32*1024*1024


class DispatchRefusal(BaseException):
    pass


def metadata(v):
    # Do not call an object's repr, properties or user-defined type methods.
    if type(v) in (str,int,bool,float) or v is None:return {'type':type(v).__name__,'value':v}
    if type(v) is bytes:return {'type':'bytes','bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}
    return {'type_id':id(type(v)),'object_id':id(v)}


def process_identity():
    with open('/proc/self/stat','rb') as f:raw=f.read()
    tail=raw.rsplit(b') ',1)[1].split()
    with open('/proc/self/cmdline','rb') as f:cmd=f.read()
    return dict(pid=os.getpid(),parent_pid=os.getppid(),session=os.getsid(0),group=os.getpgrp(),
                start_ticks=int(tail[19]),argv_hex=cmd.hex())


def kernel_environment():
    with open('/proc/self/environ','rb') as f:raw=f.read()
    if not raw.endswith(b'\0'):raise ValueError('Incomplete environment')
    pairs=[v.decode().split('=',1) for v in raw[:-1].split(b'\0')]
    if any(len(v)!=2 for v in pairs) or len({v[0] for v in pairs})!=len(pairs):raise ValueError('Duplicate environment')
    return dict(pairs)


def dir_id(s):
    if not stat.S_ISDIR(s.st_mode):raise ValueError('Directory required')
    return (s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode))


class NativeImportDispatcherV5:
    """OPEN unified integration draft. Fixture scope only; full native routes refuse.

This first implementation cannot consume a production launch acceptance. A later
version must bind and prereview the remaining declared routes without changing
the closed declaration. No caller-provided ready boolean can enable imports.
"""
    def __init__(self,declaration_bytes,binding_bytes,source_sha256):
        self._first=None;self.failed=False;self.used=False;self.active=False;self.released=False
        self.phase='uninstalled';self.ticket=None;self.native_seen=None;self.events=[]
        self.dirs={};self.descriptors={};self.written={};self.total=0;self.write_fd=None
        self.cap=None;self.cap_events=[];self.finalizer_events=[];self.cap_native_events=[];self.exit_pending=False;self.exit_seen=None;self.exit_events=[];self.exit_original=atexit.register;self.exit_wrapper=None;self.source_roles={};self.std_functions={};self.finalizer_info=None;self.finalizer_object=None;self.detach_frame=None;self.finalizer_frame=None;self.finalizer_target=None;self.registry_baseline=None;self.registry_pin=None;self.cap_callback=None;self.cap_active=False;self.cap_complete=False
        self.reg_guard=None;self.reg_routes=();self.reg_routes_pin=self.reg_routes;self.reg_pending=None;self.reg_index=0;self.reg_events=[];self.reg_kept=[];self.reg_thread_frame=None;self.reg_thread_code=None
        self.profile_failure=None
        self.reg_originals={'audit':sys.addaudithook,'fork':os.register_at_fork,'exit':atexit.register};self.reg_wrappers={};self.reg_wrapper_pins=[]
        self.reg_originals_pin=(self.reg_originals,tuple(self.reg_originals.items()))
        self.install_seen=False;self.install_token=object();self.installing_profile=False
        self.identity=process_identity();self.identity_bytes=json.dumps(self.identity,sort_keys=True)
        self.source_path=__file__;self.source_sha=source_sha256
        with open(self.source_path,'rb') as f:raw=f.read()
        if hashlib.sha256(raw).hexdigest()!=source_sha256:self.refuse('Dispatcher source differs.')
        if hashlib.sha256(declaration_bytes).hexdigest()!=DECLARATION_SHA:self.refuse('Closed v3 declaration differs.')
        self.declaration=json.loads(declaration_bytes);self.binding=json.loads(binding_bytes)
        self.binding_bytes=binding_bytes;self.binding_canonical=json.dumps(self.binding,sort_keys=True)
        b=self.binding
        if set(b)!={'schema','purpose','declaration_sha256','parent','parent_id','scratch','diagnostics','environment','fixture_source_sha256'}:
            self.refuse('Exact bootstrap fixture binding required.')
        if b['schema']!='native-import-v5-integration-fixture-v1' or b['purpose']!='integration_fixture_only' or b['declaration_sha256']!=DECLARATION_SHA:
            self.refuse('Production dispatch remains held.')
        self.parent=b['parent'];self.scratch=b['scratch'];self.diagnostics=b['diagnostics']
        # Explicit fixture-only roots cannot alias the proposed actual worker roots.
        if not self.parent.startswith('/home/dbayha/bca-work/ood-native-v5-integration-fixtures/'):
            self.refuse('Named persistent fixture parent required.')
        if [self.scratch,self.diagnostics]!=[self.parent+'/scratch',self.parent+'/diagnostics']:
            self.refuse('Distinct exact fixture child paths required.')
        if any(p in (self.declaration['scratch']['root'],self.declaration['scratch']['diagnostics_root']) for p in (self.scratch,self.diagnostics)):
            self.refuse('Actual v3 roots cannot be used by fixtures.')
        expected=dict(self.declaration['environment'],TMPDIR=self.scratch)
        if b['environment']!=expected or len(expected)!=15:self.refuse('Fixture environment differs from declared single path delta.')
        self.environment=expected;self.parent_id=tuple(b['parent_id'])
        self.api={n:getattr(os,n) for n in ('open','mkdir','write','fsync','close','unlink','remove','rmdir','rename','link','symlink','chmod','chown','truncate','ftruncate')}
        self.methods={n:(v,v.__code__,v.__defaults__,v.__kwdefaults__) for n,v in type(self).__dict__.items() if type(v) is types.FunctionType}
        compiled=compile(raw,self.source_path,'exec',dont_inherit=True)
        owner=next(v for v in compiled.co_consts if type(v) is types.CodeType and v.co_name=='NativeImportDispatcherV5')
        expected_codes={v.co_name:v for v in owner.co_consts if type(v) is types.CodeType}
        if any(fn.__code__!=expected_codes[n] or fn.__code__.co_filename!=self.source_path for n,(fn,_,_,_) in self.methods.items()):
            self.refuse('Dispatcher methods do not match compiled source.')
        self.method_names=tuple(self.methods)
        self.module_globals=globals()
        self.global_refs={n:globals()[n] for n in ('metadata','process_identity','kernel_environment','dir_id','DECLARATION_SHA','FILES','MAX_FILE','MAX_TOTAL','os','sys','stat','hashlib','json','types','ast','atexit','dis','re','shutil','tempfile','weakref','Path','__file__')}
        top_codes={v.co_name:v for v in compiled.co_consts if type(v) is types.CodeType}
        self.helper_pins={n:(globals()[n],globals()[n].__code__,globals()[n].__defaults__,globals()[n].__kwdefaults__) for n in ('metadata','process_identity','kernel_environment','dir_id')}
        if any(f.__code__!=top_codes[n] or f.__globals__ is not self.module_globals for n,(f,_,_,_) in self.helper_pins.items()):self.refuse('Dispatcher helper source differs.')
        self.audit_hook=self.audit;self.profile_hook=self.profile
        self._check_parent()
        self.parent_fd=os.open(self.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        if dir_id(os.fstat(self.parent_fd))!=self.parent_id:self.refuse('Parent descriptor differs.')
        self.check()

    @property
    def first_refusal(self):
        return None if self._first is None else json.loads(self._first)

    def refuse(self,reason,event=None,args=None,kwargs=None,caller=None):
        if self._first is None:
            f=sys._getframe(1) if caller is None else caller
            item=dict(sequence=len(self.events),phase=self.phase,reason=reason,event=event,
                      caller_source=f.f_code.co_filename,caller_line=f.f_lineno,
                      actual_positional_metadata=None if args is None else [metadata(v) for v in args],
                      ordered_keyword_metadata=None if kwargs is None else [[k,metadata(v)] for k,v in kwargs.items()],
                      availability='unavailable' if args is None else 'observed')
            self._first=json.dumps(item,sort_keys=True)
        self.failed=True
        raise DispatchRefusal(json.loads(self._first)['reason'])

    def _check_parent(self):
        p=self.parent
        if os.path.normpath(p)!=p or os.path.realpath(p)!=p:self.refuse('Canonical fixture parent required.')
        while p!='/':
            s=os.lstat(p)
            if not stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode):self.refuse('Parent ancestor replaced.')
            p=os.path.dirname(p)
        actual=dir_id(os.lstat(self.parent))
        if actual!=self.parent_id or actual[2:]!=(os.getuid(),os.getgid(),0o700):self.refuse('Parent ownership differs.')

    def check(self,diagnostic=False):
        try:
            if self.cap is not None:self._cap_check()
            self.reg_check()
            if self.failed and not diagnostic:self.refuse('Prior dispatcher refusal.')
            if any(self.module_globals.get(n) is not v for n,v in self.global_refs.items()):self.refuse('Dispatcher direct global reference changed.')
            for f,code,defaults,kw in self.helper_pins.values():
                if f.__code__ is not code or f.__defaults__ is not defaults or f.__kwdefaults__ is not kw or f.__globals__ is not self.module_globals:self.refuse('Dispatcher helper changed.')
            if json.dumps(self.identity,sort_keys=True)!=self.identity_bytes:self.refuse('Worker identity binding changed.')
            if json.dumps(process_identity(),sort_keys=True)!=self.identity_bytes:self.refuse('Worker process ownership changed.')
            if json.dumps(self.binding,sort_keys=True)!=self.binding_canonical:self.refuse('Bootstrap binding changed.')
            if dict(os.environ)!=self.environment or kernel_environment()!=self.environment:self.refuse('Exact fixture environment changed.')
            with open(self.source_path,'rb') as f:raw=f.read()
            if hashlib.sha256(raw).hexdigest()!=self.source_sha:self.refuse('Dispatcher source changed.')
            if tuple(self.methods)!=self.method_names:self.refuse('Dispatcher method inventory changed.')
            for n,(fn,code,defaults,kw) in self.methods.items():
                current=type(self).__dict__.get(n)
                if current is not fn or current.__code__ is not code or current.__defaults__ is not defaults or current.__kwdefaults__ is not kw:self.refuse('Dispatcher live method changed.')
            if any(getattr(os,n) is not v for n,v in self.api.items()):self.refuse('Native filesystem API changed.')
            if self.active and sys.getprofile() is not self.profile_hook:self.refuse('Dispatcher profile changed.')
            self._check_parent()
            if dir_id(os.fstat(self.parent_fd))!=self.parent_id:self.refuse('Parent descriptor replaced.')
            for name,(fd,identity) in self.dirs.items():
                if dir_id(os.stat(name,dir_fd=self.parent_fd,follow_symlinks=False))!=identity or dir_id(os.fstat(fd))!=identity:
                    self.refuse('Owned directory name/descriptor changed.')
            for name,(fd,identity) in self.descriptors.items():
                a=os.stat(name,dir_fd=self.dirs['diagnostics'][0],follow_symlinks=False);b=os.fstat(fd)
                for s in (a,b):
                    if not stat.S_ISREG(s.st_mode) or (s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink)!=identity or s.st_size!=self.written[name]:
                        self.refuse('Diagnostic name/descriptor/length changed.')
            return True
        except BaseException as e:
            self.refuse('Dispatcher identity check failed: '+type(e).__name__)

    def install(self):
        self.check()
        if self.profile_failure is None:self.refuse('Bound profile failure epilogue required.')
        if self.used or sys.getprofile() is not None:self.refuse('Dispatcher install is single-use.')
        self.used=True
        try:
            sys.addaudithook(self.audit_hook)
            sys.audit('ood.native_v3.bootstrap.canary',self.install_token)
            if not self.install_seen:self.refuse('Own audit hook installation suppressed.')
            self.installing_profile=True;sys.setprofile(self.profile_hook);self.installing_profile=False
            self.reg_wrappers={k:self._make_registration_wrapper(k) for k in ('audit','fork')}
            self.exit_wrapper=self._make_exit_wrapper()
            self.exit_wrapper_pin=(self.exit_wrapper,self.exit_wrapper.__code__,self.exit_wrapper.__closure__[0].cell_contents)
            self.reg_wrapper_pins=[(v,v.__code__,tuple(v.__closure__ or ()),tuple(c.cell_contents for c in v.__closure__ or ())) for v in (*self.reg_wrappers.values(),self.exit_wrapper)]
            sys.addaudithook=self.reg_wrappers['audit'];os.register_at_fork=self.reg_wrappers['fork'];atexit.register=self.exit_wrapper
            self.active=True;self.phase='awaiting_release';self.check()
        except BaseException as e:self.refuse('Dispatcher installation failed: '+type(e).__name__)

    def release(self,observed):
        self.check()
        if self.phase!='awaiting_release' or self.released:self.refuse('Repeated or premature release.')
        if observed!={'identity':self.identity,'environment':self.environment}:self.refuse('Parent release identity differs.')
        self.released=True;self.phase='released'

    def profile(self,frame,event,arg):
        try:self._profile_event(frame,event,arg)
        except BaseException as error:
            # Never unwind an exception through CPython's profile callback:
            # CPython would clear profiling and begin its error/shutdown path.
            if self._first is None:
                try:self.refuse('Profile observer failed: '+type(error).__name__,event,caller=frame)
                except DispatchRefusal:pass
            callback,owner,function,code,path,pin,exit_native=self.profile_failure
            try:
                if callback.__self__ is owner and callback.__func__ is function and function.__code__ is code and hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin:
                    callback(error)
            finally:exit_native(1)

    def bind_profile_failure(self,callback):
        self.check()
        if self.used or self.profile_failure is not None or type(callback) is not types.MethodType:self.refuse('Profile failure binding must precede installation.')
        function=callback.__func__;owner=callback.__self__;path=function.__code__.co_filename
        raw=Path(path).read_bytes();compiled=tuple(self._all_codes(compile(raw,path,'exec',dont_inherit=True)))
        if function.__code__ not in compiled or function.__name__!='profile_abort' or owner.g is not self or owner.exit_native is not os._exit:self.refuse('Profile failure epilogue source/owner differs.')
        self.profile_failure=(callback,owner,function,function.__code__,path,hashlib.sha256(raw).hexdigest(),os._exit)

    def _profile_event(self,frame,event,arg):
        if not self.active:return
        if self._registration_profile(frame,event,arg):return
        if self.cap is not None and self._cap_profile(frame,event,arg):return
        name=next((n for n,v in self.api.items() if arg is v),None) if event.startswith('c_') else None
        if name is None:return
        if self.ticket is None or self.ticket['name']!=name or frame.f_code is not self.methods['_native'][1]:
            self.refuse('Direct native filesystem alias refused.',name,caller=frame)
        if event=='c_call':
            if self.native_seen is not None:self.refuse('Repeated native operation entry.',name,caller=frame)
            self.native_seen='entry';self.events.append(dict(event='native_entry',operation=name,phase=self.phase,index=self.ticket['index']))
        elif event=='c_return':
            if self.native_seen!='entry':self.refuse('Unmatched native operation return.',name,caller=frame)
            self.native_seen='return';self.events.append(dict(event='native_return',operation=name,phase=self.phase,index=self.ticket['index']))
        elif event=='c_exception':self.refuse('Native filesystem operation raised.',name,self.ticket['args'],self.ticket['kwargs'],caller=frame)

    def audit(self,event,args):
        if event=='ood.native_v3.bootstrap.canary' and len(args)==1 and args[0] is self.install_token:
            self.install_seen=True;return
        if not self.active:return
        if event=='sys.setprofile' and self.installing_profile:return
        if self.cap_active and self._scratch_audit(event,args):return
        registration=event in ('sys.setprofile','sys.addaudithook')
        target=event=='import' and type(args[0]) is str and args[0].split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl')
        external=event.startswith(('subprocess.','socket.','fcntl.')) or event in ('os.system','os.fork','os.forkpty','os.exec','os.posix_spawn','os.spawn','os.putenv','os.unsetenv')
        writing=event=='open' and type(args[2]) is int and bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        mutation=event in ('os.mkdir','os.remove','os.rmdir','os.link','os.symlink','os.rename','os.utime','os.chmod','os.chown','os.truncate','shutil.rmtree')
        # Frame inspection itself emits an audit event. Classify first so benign
        # introspection cannot recurse into another frame inspection.
        if not (registration or target or external or writing or mutation):return
        origin=sys._getframe(1)
        if registration and event=='sys.addaudithook' and self.reg_pending is not None and self.reg_pending['kind']=='audit' and origin.f_code is self.methods['_register_native'][1]:return
        if registration:
            self.refuse('Additional observer registration refused.',event,args,caller=origin)
        if target:
            self.refuse('Native/package import phase is not implemented or accepted.',event,args,caller=origin)
        if external:
            self.refuse('Process/network/environment/lock route refused.',event,args,caller=origin)
        t=self.ticket
        if t is None:self.refuse('Mutation outside exact dispatcher ticket.',event,args,caller=origin)
        if event=='os.mkdir' and t['name']=='mkdir' and args==(t['args'][0],t['args'][1],t['kwargs']['dir_fd']):return
        if event=='open' and t['name']=='open' and args==(t['args'][0],None,t['args'][1]):return
        self.refuse('Mutation differs from dispatcher ticket.',event,args,caller=origin)

    def _native(self,name,*args,**kwargs):
        origin=sys._getframe(1);caller=origin.f_code
        allowed=(self.methods['create'][1],self.methods['_open_directory'][1],self.methods['write_diagnostic'][1])
        if caller not in allowed or self.ticket is not None:self.refuse('Unauthorized or nested dispatcher native call.',name,args,kwargs,caller=origin)
        self.check(diagnostic=caller is self.methods['write_diagnostic'][1])
        self._admit(name,args,kwargs,caller)
        index=len(self.events)
        self.ticket={'name':name,'args':args,'kwargs':kwargs,'index':index};self.native_seen=None
        self.events.append(dict(event='request',operation=name,phase=self.phase,index=index,
                                args=[metadata(v) for v in args],kwargs=[[k,metadata(v)] for k,v in kwargs.items()]))
        try:
            result=self.api[name](*args,**kwargs)
            if self.native_seen!='return':self.refuse('Native entry/return not observed.',name,args,kwargs)
            if self.events[-1]['event']!='native_return' or self.events[-1]['index']!=index:self.refuse('Native return evidence differs.')
            self.events[-1]['result']=metadata(result)
            return result
        except BaseException as e:
            self.refuse('Native operation failed: '+type(e).__name__,name,args,kwargs)
        finally:self.ticket=None;self.native_seen=None

    def _admit(self,name,args,kwargs,caller):
        if not self.active or not self.released:self.refuse('Native operation before active release.',name,args,kwargs)
        if name=='mkdir':
            expected='diagnostics' if self.phase=='create_diagnostics' else 'scratch' if self.phase=='create_scratch' else None
            if self.failed or caller is not self.methods['create'][1] or args!=(expected,0o700) or kwargs!={'dir_fd':self.parent_fd}:
                self.refuse('Root creation ticket differs.',name,args,kwargs)
            if os.path.lexists(self.parent+'/'+expected):self.refuse('Root already exists; no reuse.',name,args,kwargs)
        elif name=='open':
            dirflags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC
            flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC
            if caller is self.methods['_open_directory'][1]:
                expected='diagnostics' if self.phase=='create_diagnostics' else 'scratch' if self.phase=='create_scratch' else None
                if self.failed or args!=(expected,dirflags) or kwargs!={'dir_fd':self.parent_fd}:self.refuse('Owned root open differs.',name,args,kwargs)
            elif caller is self.methods['create'][1]:
                expected=FILES[len(self.descriptors)] if len(self.descriptors)<len(FILES) else None
                if self.failed or self.phase!='create_files' or args!=(expected,flags,0o600) or kwargs!={'dir_fd':self.dirs['diagnostics'][0]}:
                    self.refuse('Exclusive diagnostic open differs.',name,args,kwargs)
            else:self.refuse('Open caller differs.',name,args,kwargs)
        elif name=='write':
            if caller is not self.methods['write_diagnostic'][1] or len(args)!=2 or args[0]!=self.write_fd or type(args[1]) is not bytes or kwargs:
                self.refuse('Diagnostic descriptor write differs.',name,args,kwargs)
        elif name=='fsync':
            if kwargs or len(args)!=1:self.refuse('Fsync shape differs.',name,args,kwargs)
            if caller is self.methods['write_diagnostic'][1]:
                if args[0]!=self.write_fd:self.refuse('Fsync descriptor differs.',name,args,kwargs)
            elif caller is self.methods['create'][1]:
                if self.failed or args[0] not in (self.parent_fd,self.dirs.get('diagnostics',(None,))[0]):self.refuse('Directory fsync differs.',name,args,kwargs)
            else:self.refuse('Fsync caller differs.',name,args,kwargs)
        else:self.refuse('Native operation has no declared bootstrap route.',name,args,kwargs)

    def _open_directory(self,name):
        before=os.stat(name,dir_fd=self.parent_fd,follow_symlinks=False);identity=dir_id(before)
        if identity[2:]!=(os.getuid(),os.getgid(),0o700):self.refuse('Created root owner/mode differs.')
        fd=self._native('open',name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=self.parent_fd)
        if dir_id(os.fstat(fd))!=identity:self.refuse('Created root descriptor differs.')
        self.dirs[name]=(fd,identity)

    def create(self):
        self.check()
        if self.phase!='released' or not self.released:self.refuse('Worker creation is single-use after release.')
        if os.path.lexists(self.scratch) or os.path.lexists(self.diagnostics):self.refuse('Fresh child roots required before either creation.')
        try:
            for name in ('diagnostics','scratch'):
                self.check();self.phase='create_'+name
                self._native('mkdir',name,0o700,dir_fd=self.parent_fd);self._open_directory(name)
                self._native('fsync',self.parent_fd)
            if os.listdir(self.dirs['scratch'][0]):self.refuse('New scratch root not empty.')
            self.phase='create_files'
            for name in FILES:
                self.check()
                fd=self._native('open',name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self.dirs['diagnostics'][0])
                s=os.fstat(fd);identity=(s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink)
                if not stat.S_ISREG(s.st_mode) or identity[2:]!=(os.getuid(),os.getgid(),0o600,1) or s.st_size!=0:self.refuse('Created diagnostic file differs.')
                self.descriptors[name]=(fd,identity);self.written[name]=0;self._native('fsync',self.dirs['diagnostics'][0])
            self.phase='bootstrap_ready';self.check()
        except BaseException as e:self.refuse('Worker creation failed: '+type(e).__name__)

    def write_diagnostic(self,name,data):
        # This narrowly declared method is the only post-refusal native write route.
        self.check(diagnostic=True)
        if self.phase!='bootstrap_ready' or self.write_fd is not None or type(name) is not str or name not in self.descriptors or type(data) is not bytes:
            self.refuse('Exact bounded diagnostic call required.')
        if not data or self.written[name]+len(data)>MAX_FILE or self.total+len(data)>MAX_TOTAL:self.refuse('Diagnostic byte ceiling reached before write.')
        fd=self.descriptors[name][0];self.write_fd=fd
        try:
            offset=0
            while offset<len(data):
                n=self._native('write',fd,data[offset:])
                if type(n) is not int or n<=0 or n>len(data)-offset:self.refuse('Invalid diagnostic write count.')
                self.written[name]+=n;self.total+=n;offset+=n
            self._native('fsync',fd)
        except BaseException as e:self.refuse('Diagnostic write/fsync failed: '+type(e).__name__)
        finally:self.write_fd=None
        self.check(diagnostic=True)
        return dict(name=name,bytes_written=len(data),total_file_bytes=self.written[name],fsynced=True)

    def require_import_phase(self):
        self.refuse('Full registration/capability/native routes and independent prereview remain unimplemented.')

    def summary(self):
        self.check(diagnostic=True)
        return dict(phase=self.phase,pid=self.identity['pid'],owner_identity=self.identity,roots_created_by_worker=list(self.dirs),
                    diagnostic_bytes=dict(self.written),native_events=list(self.events),first_refusal=self.first_refusal,
                    capability_fixture_only=True,target_import_accepted=False,full_route_implemented=False,
                    target_audit_effectiveness_accepted=False,scientific_execution_accepted=False)

    def _all_codes(self,root):
        yield root
        for value in root.co_consts:
            if type(value) is types.CodeType:yield from self._all_codes(value)

    def _source_identity(self,path):
        s=os.lstat(path)
        if not stat.S_ISREG(s.st_mode) or os.path.realpath(path)!=path:self.refuse('Canonical regular capability source required.')
        with open(path,'rb') as f:raw=f.read()
        after=os.lstat(path)
        fields=lambda q:(q.st_dev,q.st_ino,q.st_size,q.st_mtime_ns,q.st_ctime_ns,q.st_nlink,q.st_uid,q.st_gid,stat.S_IMODE(q.st_mode))
        if fields(s)!=fields(after):self.refuse('Capability source changed during read.')
        return (hashlib.sha256(raw).hexdigest(),fields(s)),raw

    def bind_capability(self,source,module_name,source_sha,callback_source_sha):
        self.check()
        if sys.dont_write_bytecode is not True:self.refuse('Explicit no-bytecode-write import setting required.')
        if self.cap is not None or self.phase!='bootstrap_ready' or module_name in sys.modules:
            self.refuse('Capability binding is single-use after worker bootstrap.')
        if not source.startswith(self.parent+'/') or module_name!='v5_fixture_capability':
            self.refuse('Full filelock package import remains held; exact fixture namespace required.')
        identity,raw=self._source_identity(source)
        if identity[0]!=source_sha or identity[1][5]!=1:self.refuse('Fixture source identity differs.')
        module_code=compile(raw,source,'exec',dont_inherit=True);tree=ast.parse(raw)
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_probe_link_follow_symlinks']
        if len(nodes)!=1 or hashlib.sha256(ast.get_source_segment(raw.decode(),nodes[0]).encode()).hexdigest()!=callback_source_sha:
            self.refuse('Exact saved capability definition required.')
        callbacks=[c for c in self._all_codes(module_code) if c.co_name=='_probe_link_follow_symlinks']
        sites=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_LINK_HONORS_FOLLOW_SYMLINKS' for t in n.targets)]
        if len(callbacks)!=1 or len(sites)!=1 or ast.dump(sites[0].value)!=ast.dump(ast.Call(func=ast.Name(id='_probe_link_follow_symlinks',ctx=ast.Load()),args=[],keywords=[])):
            self.refuse('Exact ordinary module capability assignment required.')
        hashes={'os':'70e420e105d021d5ba2ec4d8a9b2131561db6d93e800e8580ddeb86cbd6959c3',
                'pathlib':'936f14a3ecc4ac5e896e4d673666d76b25a9d581f3b4c8d8d424e830fda2a69b',
                'shutil':'d96cf3b8b17c717f14dc67643cc8ad0943fb4b960d3e32981b2991613d39dde9',
                'tempfile':'676356b31756112053515fc1d550a99faf0f4bbaff6ccd4cc8a56310474f5c3a',
                'weakref':'dd8e03473ee5667c1a2caa43ede07797652bcb4035fabb60d60af10bb23a0886'}
        modules={'os':os,'pathlib':sys.modules['pathlib'],'shutil':shutil,'tempfile':tempfile,'weakref':weakref}
        allcodes={}
        for role,module in modules.items():
            path=module.__file__;pin,data=self._source_identity(path)
            if pin[0]!=hashes[role] or pin[1][5:]!=(3,1000,1000,0o644):self.refuse('Exact shared stdlib source pin differs.')
            suffix='/lib/python3.10/'+role+'.py'
            aliases=('/home/dbayha/miniconda3/envs/corl-orig-local'+suffix,
                     '/home/dbayha/miniconda3/envs/sdbca'+suffix,
                     '/home/dbayha/miniconda3/pkgs/python-3.10.20-h741d88c_0'+suffix)
            if path!=aliases[0]:self.refuse('Stdlib module source location differs.')
            for alias in aliases:
                a,unused=self._source_identity(alias)
                if a!=pin:self.refuse('Explicit current source alias differs.')
            self.source_roles[role]=(module,path,pin,aliases)
            allcodes[path]=tuple(self._all_codes(compile(data,path,'exec',dont_inherit=True)))
        functions={'td_init':tempfile.TemporaryDirectory.__init__,'td_cleanup':tempfile.TemporaryDirectory.cleanup,
                   'td_callback':tempfile.TemporaryDirectory.__dict__['_cleanup'].__func__,
                   'finalize_init':weakref.finalize.__init__,'detach':weakref.finalize.detach,
                   'exitfunc':weakref.finalize.__dict__['_exitfunc'].__func__,
                   'finalize_call':weakref.finalize.__call__}
        defaults_expected={'td_init':(None,None,None,False),'td_cleanup':None,'td_callback':(False,),
                           'finalize_init':None,'detach':None,'exitfunc':None,'finalize_call':(None,)}
        self.std_direct_globals={}
        for role,fn in functions.items():
            if type(fn) is not types.FunctionType or fn.__code__ not in allcodes.get(fn.__code__.co_filename,()) or fn.__globals__ is not sys.modules[fn.__module__].__dict__:
                self.refuse('Actual stdlib function code/namespace differs.')
            if fn.__defaults__!=defaults_expected[role] or fn.__kwdefaults__ is not None or fn.__closure__ is not None:self.refuse('Actual stdlib function defaults/closure differ from saved source.')
            names={op.argval for c in self._all_codes(fn.__code__) for op in dis.get_instructions(c) if op.opname=='LOAD_GLOBAL'}
            self.std_direct_globals[role]={n:fn.__globals__[n] if n in fn.__globals__ else fn.__builtins__[n] for n in names}
            self.std_functions[role]=(fn,fn.__code__,fn.__defaults__,fn.__kwdefaults__,fn.__globals__)
        self.std_codes=allcodes;self.std_function_objects=functions
        self.finalize_class=weakref.finalize;self.td_class=tempfile.TemporaryDirectory;self.info_class=weakref.finalize._Info
        self.weak_native_refs=(weakref.ref,weakref.ReferenceType)
        self.registry_pin=weakref.finalize._registry
        if type(self.registry_pin) is not dict or type(weakref.finalize._registered_with_atexit) is not bool or weakref.finalize._shutdown is not False:
            self.refuse('Unexpected live finalizer registry state.')
        self.registry_baseline=self._registry_snapshot();self.flag_before=weakref.finalize._registered_with_atexit
        self.dirty_before=weakref.finalize._dirty
        self.registry_events=[dict(event='baseline',registry_id=id(self.registry_pin),registered=self.flag_before,dirty=self.dirty_before,entries=self.registry_baseline)]
        self.cap_scratch_events=[];self.cap_stage=0;self.testfile=None;self.directory=None;self.directory_id=None;self.remaining=set();self.link_id=None;self.directory_fd=None
        self.cap=dict(source=source,identity=identity,module_name=module_name,module_code=module_code,callback_code=callbacks[0],line=sites[0].lineno,module=None,module_frame=None,callback_frame=None)
        self.cap_contract=(source,module_name,source_sha,callback_source_sha);self.cap_contract_pin=self.cap_contract
        if atexit.register is not self.exit_wrapper:self.refuse('Unified atexit registration changed before capability composition.')
        self.check()

    def _registry_snapshot(self,skip=None):
        result=[]
        for finalizer,info in self.registry_pin.items():
            if finalizer is skip:continue
            if type(finalizer) is not weakref.finalize or type(info) is not weakref.finalize._Info or type(info.weakref) is not weakref.ReferenceType or type(info.args) is not tuple:
                self.refuse('Unsupported actual finalizer registry entry.')
            result.append(dict(finalizer=id(finalizer),info=id(info),weakref=id(info.weakref),target=id(info.weakref()) if info.weakref() is not None else None,
                               callback=id(info.func),args=id(info.args),arg_values=[metadata(v) for v in info.args],
                               kwargs_id=id(info.kwargs),kwargs=None if info.kwargs is None else [[k,metadata(v)] for k,v in info.kwargs.items()],
                               atexit=info.atexit,index=info.index))
        return sorted(result,key=lambda x:x['index'])

    def _cap_check(self):
        c=self.cap
        if sys.dont_write_bytecode is not True:self.refuse('No-bytecode-write import setting changed.')
        if weakref.finalize is not self.finalize_class or tempfile.TemporaryDirectory is not self.td_class or weakref.finalize._Info is not self.info_class:self.refuse('Finalizer class owner changed.')
        if any(a is not b for a,b in zip((weakref.ref,weakref.ReferenceType),self.weak_native_refs)):self.refuse('Weak reference native symbol changed.')
        if self.cap_contract!=self.cap_contract_pin or self._source_identity(c['source'])[0]!=c['identity']:
            self.refuse('Capability source or contract changed.')
        for module,path,pin,aliases in self.source_roles.values():
            if module.__file__!=path or sys.modules.get(module.__name__) is not module:self.refuse('Stdlib module namespace changed.')
            for alias in aliases:
                if self._source_identity(alias)[0]!=pin:self.refuse('Stdlib source or shared alias changed.')
        for name,(fn,code,defaults,kw,namespace) in self.std_functions.items():
            if fn.__code__ is not code or fn.__defaults__ is not defaults or fn.__kwdefaults__ is not kw or fn.__globals__ is not namespace:
                self.refuse('Finalizer/TemporaryDirectory live code changed.')
            for n,value in self.std_direct_globals[name].items():
                if (fn.__globals__[n] if n in fn.__globals__ else fn.__builtins__[n]) is not value:self.refuse('Finalizer direct global reference changed.')
        current={'td_init':tempfile.TemporaryDirectory.__init__,'td_cleanup':tempfile.TemporaryDirectory.cleanup,
                 'td_callback':tempfile.TemporaryDirectory.__dict__['_cleanup'].__func__,
                 'finalize_init':weakref.finalize.__init__,'detach':weakref.finalize.detach,
                 'exitfunc':weakref.finalize.__dict__['_exitfunc'].__func__,'finalize_call':weakref.finalize.__call__}
        if any(current[n] is not f for n,f in self.std_function_objects.items()):self.refuse('Finalizer defining owner changed.')
        if weakref.finalize._registry is not self.registry_pin or weakref.finalize._shutdown is not False:self.refuse('Finalizer registry/ shutdown changed.')
        if self.finalizer_info is not None and weakref.finalize._registered_with_atexit is not True:self.refuse('Finalizer registration flag changed.')
        if self.cap_complete and self.finalizer_object in self.registry_pin:self.refuse('Detached finalizer entry was restored.')
        if self.registry_baseline!=self._registry_snapshot(self.finalizer_object):self.refuse('Other live finalizer registry entries changed.')
        if self.exit_wrapper is not None:
            obj,code,owner=self.exit_wrapper_pin
            if atexit.register is not obj or self.exit_wrapper is not obj or obj.__code__ is not code or obj.__closure__[0].cell_contents is not owner:self.refuse('Atexit wrapper changed.')
        if c['module'] is not None:
            ns=c['module'].__dict__
            if sys.modules.get(c['module_name']) is not c['module'] or ns is not c['module_frame'].f_globals or ns.get('__file__')!=c['source']:self.refuse('Capability module namespace changed.')
            if self.cap_callback is not None:
                fn=self.cap_callback
                if ns.get('_probe_link_follow_symlinks') is not fn or fn.__code__ is not c['callback_frame'].f_code or fn.__globals__ is not ns or fn.__defaults__ is not None or fn.__kwdefaults__ is not None or fn.__closure__ is not None:self.refuse('Capability callback binding changed.')
                if ns.get('os') is not os or ns.get('tempfile') is not tempfile or ns.get('Path') is not Path or ns.get('_HAS_LINK') is not True:self.refuse('Capability callback globals changed.')

    def _make_exit_wrapper(self):
        def registered(*args,**kwargs):
            caller=sys._getframe(1)
            if self.cap is not None and caller.f_code is self.std_functions['finalize_init'][1]:return self._register_finalizer_exit(caller,args,kwargs)
            return self._register_native('exit',caller,args,kwargs)
        return registered

    def _register_finalizer_exit(self,caller,args,kwargs):
        try:
            self.check()
            if not self.cap_active or self.flag_before or self.exit_events or self.exit_pending or caller is not self.finalizer_frame or caller.f_code is not self.std_functions['finalize_init'][1]:
                self.refuse('Only observed first weakref finalizer atexit call is declared.','atexit.register',args,kwargs,caller)
            if len(args)!=1 or kwargs or type(args[0]) is not types.MethodType or args[0].__self__ is not weakref.finalize or args[0].__func__ is not self.std_functions['exitfunc'][0]:
                self.refuse('Original weakref classmethod callback differs.','atexit.register',args,kwargs,caller)
            cb=args[0];self.exit_events.append(dict(event='request',callback_id=id(cb),owner_id=id(cb.__self__),function_id=id(cb.__func__),source=cb.__func__.__code__.co_filename,line=caller.f_lineno,args=[metadata(cb)],ordered_kwargs=[]))
            self.exit_pending=True;self.exit_seen=None
            result=self.exit_original(*args,**kwargs)
            if self.exit_seen!='return' or result is not cb:self.refuse('Finalizer registration native return differs.')
            self.exit_pending=False;self.check();return result
        except BaseException as e:
            self.refuse('Finalizer registration refused: '+type(e).__name__,'atexit.register',args,kwargs,caller)

    def _cap_profile(self,frame,event,arg):
        c=self.cap
        if event.startswith('c_') and self.ticket is not None:return False
        if event.startswith('c_') and arg is self.exit_original:
            if not self.exit_pending or frame.f_code is not self.methods['_register_finalizer_exit'][1]:
                self.refuse('Direct native atexit registration alias refused.','atexit.register',caller=frame)
            if event=='c_call':
                if self.exit_seen is not None:self.refuse('Repeated native finalizer registration.')
                self.exit_seen='entry';self.exit_events.append(dict(event='native_entry'))
            elif event=='c_return':
                if self.exit_seen!='entry':self.refuse('Unmatched native finalizer registration.')
                self.exit_seen='return';self.exit_events.append(dict(event='native_return'))
            else:self.refuse('Native finalizer registration exception.')
            return True
        if event=='c_call' and arg is os.register_at_fork:self.refuse('Remaining at-fork callback state graph is not implemented.','os.register_at_fork',caller=frame)
        if event=='call' and frame.f_code is self.std_functions['finalize_call'][1]:
            self.refuse('Finalizer callback execution is outside observed detach lifecycle.',caller=frame)
        if event=='call' and frame.f_code==c['module_code']:
            if c['module'] is not None:self.refuse('Repeated capability module execution.')
            module=sys.modules.get(c['module_name'])
            if type(module) is not types.ModuleType or module.__dict__ is not frame.f_globals or frame.f_globals.get('__file__')!=c['source']:self.refuse('Registered capability namespace required.')
            c['module']=module;c['module_frame']=frame;self.check();self.cap_events.append(dict(event='module_entry'));return True
        if event=='call' and frame.f_code==c['callback_code']:
            if c['callback_frame'] is not None or frame.f_back is not c['module_frame'] or frame.f_back.f_lineno!=c['line']:self.refuse('Capability must enter once from exact module callsite.')
            fn=frame.f_globals.get('_probe_link_follow_symlinks')
            if type(fn) is not types.FunctionType or fn.__code__ is not frame.f_code or fn.__globals__ is not frame.f_globals:self.refuse('Capability function object differs.')
            if tempfile.tempdir is not None or os.listdir(self.dirs['scratch'][0]):self.refuse('Fresh untouched tempfile scratch selection required.')
            self.cap_callback=fn;c['callback_frame']=frame;self.cap_active=True;self.check();self.cap_events.append(dict(event='callback_entry'));return True
        if self.cap_active and event=='call' and frame.f_code is self.std_functions['finalize_init'][1]:
            if self.finalizer_frame is not None or frame.f_back.f_code is not self.std_functions['td_init'][1]:self.refuse('Unexpected or repeated finalizer construction.')
            self.finalizer_frame=frame;self.finalizer_object=frame.f_locals['self'];self.finalizer_target=frame.f_locals['obj']
            if type(self.finalizer_target) is not tempfile.TemporaryDirectory:self.refuse('Actual TemporaryDirectory target required.')
            fn=frame.f_locals['func']
            if type(fn) is not types.MethodType or fn.__self__ is not tempfile.TemporaryDirectory or fn.__func__ is not self.std_functions['td_callback'][0]:self.refuse('TemporaryDirectory cleanup classmethod differs.')
            self.finalizer_events.append(dict(event='construct_entry',finalizer_id=id(self.finalizer_object),target_id=id(self.finalizer_target),callback_id=id(fn),owner_id=id(fn.__self__),function_id=id(fn.__func__)))
            self.check();return True
        if self.cap_active and event=='return' and frame is self.finalizer_frame:
            info=self.registry_pin.get(self.finalizer_object)
            if type(info) is not weakref.finalize._Info or type(info.weakref) is not weakref.ReferenceType or info.weakref() is not self.finalizer_target or info.func is not frame.f_locals['func'] or info.args is not frame.f_locals['args'] or info.kwargs is not frame.f_locals['kwargs']:
                self.refuse('Actual finalizer registry entry differs from constructor arguments.')
            if info.args!=(self.directory,) or list(info.kwargs)!=['warn_message','ignore_errors'] or type(info.kwargs['warn_message']) is not str or info.kwargs['ignore_errors'] is not False or info.atexit is not True or type(info.index) is not int or not weakref.finalize._registered_with_atexit:self.refuse('Actual finalizer entry fields differ.')
            self.finalizer_info=info;self.finalizer_entry_pin=(info.weakref,info.func,info.args,info.kwargs,dict(info.kwargs),info.atexit,info.index)
            self.finalizer_events.append(dict(event='registry_added',weakref_id=id(info.weakref),target_id=id(info.weakref()),callback_id=id(info.func),args=list(info.args),kwargs=[[k,v] for k,v in info.kwargs.items()],atexit=info.atexit,index=info.index,registered=weakref.finalize._registered_with_atexit,registry_size=len(self.registry_pin)))
            self.check();return True
        if self.cap_active and event=='call' and frame.f_code is self.std_functions['detach'][1]:
            if self.detach_frame is not None or frame.f_locals['self'] is not self.finalizer_object or frame.f_back.f_code is not self.std_functions['td_cleanup'][1]:self.refuse('Exact natural TemporaryDirectory detach required.')
            info=self.registry_pin.get(self.finalizer_object);pin=self.finalizer_entry_pin
            if info is not self.finalizer_info or info.weakref is not pin[0] or info.func is not pin[1] or info.args is not pin[2] or info.kwargs is not pin[3] or info.kwargs!=pin[4] or info.atexit is not pin[5] or info.index!=pin[6] or info.weakref() is not self.finalizer_target:self.refuse('Finalizer entry changed before detach.')
            self.detach_frame=frame;self.finalizer_events.append(dict(event='detach_entry',target_alive=True,registry_contains_exact_entry=True));self.check();return True
        if self.cap_active and event=='return' and frame is self.detach_frame:
            pin=self.finalizer_entry_pin
            if self.finalizer_object in self.registry_pin or type(arg) is not tuple or len(arg)!=4 or arg[0] is not self.finalizer_target or arg[1] is not pin[1] or arg[2] is not pin[2] or arg[3] is not pin[3]:self.refuse('Detach did not remove and return the exact live entry.')
            self.finalizer_events.append(dict(event='detach_return',exact_entry_removed=True,target_alive=True,other_entries_unchanged=self._registry_snapshot()==self.registry_baseline))
            self.check();return True
        if event=='return' and frame is c['callback_frame']:
            self.check()
            if arg is not True or self.cap_stage!=12 or os.path.lexists(self.directory) or os.path.lexists(self.testfile) or os.listdir(self.dirs['scratch'][0]) or len(self.finalizer_events)!=4 or self.finalizer_events[-1]['event']!='detach_return':self.refuse('Capability return lacks complete filesystem/finalizer lifecycle.')
            if len(self.exit_events)!=(0 if self.flag_before else 3):self.refuse('Conditional finalizer atexit route differs.')
            self.cap_active=False;self.cap_events.append(dict(event='callback_return',result=True));return True
        if event=='return' and frame is c['module_frame']:
            self.check()
            if self.cap_active or len(self.cap_events)!=3 or frame.f_globals.get('_LINK_HONORS_FOLLOW_SYMLINKS') is not True:self.refuse('Module result differs from observed capability.')
            self.cap_complete=True;self.cap_events.append(dict(event='module_return',result=True))
            self.registry_events.append(dict(event='after_detach',registry_id=id(self.registry_pin),registered=weakref.finalize._registered_with_atexit,dirty=weakref.finalize._dirty,entries=self._registry_snapshot()))
            return True
        if self.cap_active and event.startswith('c_'):
            name=next((n for n,v in self.api.items() if arg is v),None)
            if name is not None:
                if name not in ('open','mkdir','write','close','unlink','remove','rmdir','link') or frame.f_code not in self.std_codes.get(frame.f_code.co_filename,()) and frame is not c['callback_frame']:
                    self.refuse('Native operation outside bound capability source.',name,caller=frame)
                self.cap_native_events.append(dict(event=event,operation=name,source=frame.f_code.co_filename,line=frame.f_lineno,args=None,kwargs=None))
                if event=='c_exception' and name!='utime':self.refuse('Capability native operation raised.',name,caller=frame)
                return True
        return False

    def capability_result(self):
        self.check()
        if not self.cap_complete:self.refuse('Capability composition is incomplete.')
        return dict(boundaries=list(self.cap_events),scratch_events=list(self.cap_scratch_events),native_events=list(self.cap_native_events),
                    finalizer_events=list(self.finalizer_events),registry_snapshots=list(self.registry_events),atexit_registration_events=list(self.exit_events),
                    callback_invocations_by_observer=0,root_retained_empty=not os.listdir(self.dirs['scratch'][0]),
                    full_filelock_namespace_accepted=False,full_stdlib_native_graph_accepted=False,actual_production_profile_executed=False,scientific_execution_accepted=False)


    def _scratch_audit(self,event,args):
        mutation=event in ('os.mkdir','os.remove','os.rmdir','os.link','os.symlink','os.rename','os.utime','os.chmod','os.chown','os.truncate','shutil.rmtree')
        opening=event=='open' and ((self.cap_stage==8 and args[0]==self.directory) or bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if not (mutation or opening or event=='os.scandir'):return False
        self.check()
        f=sys._getframe(1)
        while f is not None and f is not self.cap['callback_frame']:f=f.f_back
        if f is None:self.refuse('Scratch operation outside observed capability.',event,args)
        if self.directory_id is not None and self.cap_stage<12:
            s=os.lstat(self.directory)
            if (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))!=self.directory_id or not stat.S_ISDIR(s.st_mode):self.refuse('Temporary directory changed.',event,args)
        self.cap_scratch_events.append(dict(stage=self.cap_stage,event=event,args=[metadata(v) for v in args]))
        if self.cap_stage==0 and event=='open':
            path,mode,flags=args;p=Path(path)
            expected=os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_CLOEXEC',0)
            if p.parent!=Path(self.scratch) or not re.fullmatch('[a-z0-9_]{8}',p.name) or p.exists() or mode is not None or flags!=expected:self.refuse('Default tempfile probe open mismatch.',event,args)
            self.testfile=str(p);self.cap_stage=1;return True
        if self.cap_stage==1 and event=='os.remove' and args==(self.testfile,-1):
            p=Path(self.testfile);s=p.lstat()
            if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.geteuid() or stat.S_IMODE(s.st_mode)!=0o600 or p.read_bytes()!=b'blat':self.refuse('Default tempfile probe readback mismatch.',event,args)
            self.cap_scratch_events[-1]['readback_hex']='626c6174';self.cap_stage=2;return True
        if self.cap_stage==2 and event=='os.mkdir':
            path,mode,dirfd=args;p=Path(path)
            if Path(self.testfile).exists() or p.parent!=Path(self.scratch) or not re.fullmatch('tmp[a-z0-9_]{8}',p.name) or p.exists() or mode!=0o700 or dirfd!=-1:self.refuse('Temporary directory create mismatch.',event,args)
            self.directory=str(p);self.cap_stage=3;return True
        if self.cap_stage==3:
            s=Path(self.directory).lstat();self.directory_id=(s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))
            if s.st_uid!=os.geteuid() or stat.S_IMODE(s.st_mode)!=0o700:self.refuse('Temporary directory permissions mismatch.',event,args)
            if event=='os.utime' and args==(str(Path(self.directory)/'probe-source'),None,None,-1):self.cap_stage=4;return True
        if self.cap_stage==4 and event=='open':
            path,mode,flags=args
            if path!=str(Path(self.directory)/'probe-source') or mode is not None or flags!=(os.O_WRONLY|os.O_CREAT|getattr(os,'O_CLOEXEC',0)) or Path(path).exists():self.refuse('Capability source open mismatch.',event,args)
            self.cap_stage=5;return True
        if self.cap_stage==5 and event=='os.link':
            source,target,srcfd,dstfd=args
            if (source,target,srcfd,dstfd)!=(str(Path(self.directory)/'probe-source'),str(Path(self.directory)/'probe-link'),-1,-1):self.refuse('Capability hard-link path mismatch.',event,args)
            p=Path(source);s=p.lstat()
            if s.st_nlink!=1 or not stat.S_ISREG(s.st_mode) or p.read_bytes()!=b'' or Path(target).exists():self.refuse('Capability source is not fresh empty file.',event,args)
            self.link_id=(s.st_dev,s.st_ino);self.cap_stage=6;return True
        if self.cap_stage==6 and event=='shutil.rmtree' and args==(self.directory,):
            for name in ('probe-source','probe-link'):
                p=Path(self.directory)/name;s=p.lstat()
                if (s.st_dev,s.st_ino)!=self.link_id or s.st_nlink!=2 or p.read_bytes()!=b'':self.refuse('Hard-link result readback mismatch.',event,args)
            self.cap_scratch_events[-1]['verified_empty_hard_link_pair']=True;self.cap_stage=8;return True
        if self.cap_stage==8 and event=='open' and args==(self.directory,None,getattr(os,'O_CLOEXEC',0)):
            self.cap_stage=9;return True
        if self.cap_stage==9 and event=='os.scandir' and type(args[0]) is int:
            fd=args[0]
            if os.readlink('/proc/self/fd/%d'%fd)!=self.directory:self.refuse('Cleanup descriptor mismatch.',event,args)
            self.directory_fd=fd;self.remaining={'probe-source','probe-link'};self.cap_stage=10;return True
        if self.cap_stage==10 and event=='os.remove':
            name,fd=args
            if name not in self.remaining or fd!=self.directory_fd or os.readlink('/proc/self/fd/%d'%fd)!=self.directory:self.refuse('Cleanup unlink outside exact pair.',event,args)
            p=Path(self.directory)/name;s=p.lstat()
            if (s.st_dev,s.st_ino)!=self.link_id or s.st_nlink!=len(self.remaining) or p.read_bytes()!=b'':self.refuse('Cleanup file identity changed.',event,args)
            self.remaining.remove(name)
            if not self.remaining:self.cap_stage=11
            return True
        if self.cap_stage==11 and event=='os.rmdir' and args==(self.directory,-1):
            if list(Path(self.directory).iterdir()):self.refuse('Cleanup directory not empty.',event,args)
            self.cap_stage=12;return True
        self.refuse('Unexpected scratch operation or order.',event,args)

    def _make_registration_wrapper(self,kind):
        def wrapper(*args,**kwargs):
            return self._register_native(kind,sys._getframe(1),args,kwargs)
        return wrapper

    def _callback_key(self,value):
        if type(value) is types.MethodType:return ('python_method',id(value.__self__),id(value.__func__))
        if type(value) is types.BuiltinMethodType:return ('native_method',id(value.__self__),id(type(value.__self__)),value.__name__)
        return ('identity',id(value))

    def bind_registration_routes(self,binder_module,module_specs,routes,threading_module,setup):
        self.check()
        if self.reg_guard is not None or self.phase!='bootstrap_ready' or self.cap is None:self.refuse('Registration composition is single-use after capability binding.')
        pin,unused=self._source_identity(binder_module.__file__)
        if pin[0]!='997fdc4653f204df3d799f47c604937075a15d001a2cb99b5200009cf384f251':self.refuse('Closed concrete binder source differs.')
        guard=binder_module.RegistrationBindings(module_specs)
        # The supplied setup is a source-bound fixture entrypoint, never a target
        # callback. Its full transitive graph is not production acceptance.
        if type(setup) is not types.FunctionType or setup.__code__ not in tuple(self._all_codes(compile(Path(setup.__code__.co_filename).read_bytes(),setup.__code__.co_filename,'exec',dont_inherit=True))):self.refuse('Binding setup source differs.')
        self.reg_guard=guard;self.reg_binder=binder_module;self.reg_binder_pin=(binder_module.__file__,pin)
        try:self.reg_bindings=setup(guard)
        except BaseException as error:self.refuse('Concrete registration binding refused: '+str(error))
        self.reg_threading=threading_module;self.reg_thread_code=threading_module._register_atexit.__code__
        self.reg_thread_native=threading_module._register_atexit
        self.reg_prior_thread_state=guard.freeze(threading_module._threading_atexits)
        if threading_module._SHUTTING_DOWN is not False:self.refuse('Threading shutdown already started.')
        accepted=[]
        for route in routes:
            kind,module,qualname,line,args,kwargs=route
            if kind not in ('audit','fork','exit','threading') or type(line) is not int:self.refuse('Declared registration route shape differs.')
            info=guard.modules.get(module.__name__)
            if info is None:self.refuse('Registration caller namespace is unbound.')
            code=info['compiled'].get(tuple(qualname.split('.')))
            if code is None:self.refuse('Compiled registration caller unavailable.')
            accepted.append((kind,module,code,line,tuple(self._callback_key(v) for v in args),tuple((k,self._callback_key(v)) for k,v in kwargs.items())))
        self.reg_routes=tuple(accepted);self.reg_routes_pin=self.reg_routes;self.reg_check()

    def reg_check(self):
        if self.reg_originals is not self.reg_originals_pin[0] or tuple(self.reg_originals.items())!=self.reg_originals_pin[1]:self.refuse('Original native registration bindings changed.')
        if self.active:
            if sys.addaudithook is not self.reg_wrappers['audit'] or os.register_at_fork is not self.reg_wrappers['fork'] or atexit.register is not self.exit_wrapper:self.refuse('Unified registration wrapper changed.')
            for wrapper,code,cells,values in self.reg_wrapper_pins:
                if wrapper.__code__ is not code or tuple(wrapper.__closure__ or ())!=cells or any(c.cell_contents is not v for c,v in zip(cells,values)):self.refuse('Unified wrapper code/closure changed.')
        if self.reg_guard is not None:
            try:
                if self._source_identity(self.reg_binder_pin[0])[0]!=self.reg_binder_pin[1]:self.refuse('Concrete binder source changed.')
                self.reg_guard.check()
                if not self.reg_guard.thread_events and (self.reg_guard.freeze(self.reg_threading._threading_atexits)!=self.reg_prior_thread_state or self.reg_threading._SHUTTING_DOWN is not False):self.refuse('Pre-existing internal threading state changed before request.')
                if self.reg_routes is not self.reg_routes_pin:self.refuse('Registration route declaration changed.')
                if self.reg_threading._register_atexit is not self.reg_thread_native or self.reg_thread_native.__code__ is not self.reg_thread_code:self.refuse('Original internal threading helper changed.')
            except BaseException as error:self.refuse('Concrete registration binding refused: '+str(error))

    def _registration_request(self,kind,caller,args,kwargs):
        if self.reg_guard is None or self.reg_pending is not None or self.reg_index>=len(self.reg_routes):self.refuse('Undeclared, repeated or nested registration.',kind,args,kwargs,caller)
        self.check();site=self.reg_routes[self.reg_index]
        expected=(kind,sys.modules.get(caller.f_globals.get('__name__')),caller.f_code,caller.f_lineno,tuple(self._callback_key(v) for v in args),tuple((k,self._callback_key(v)) for k,v in kwargs.items()))
        if expected!=site or caller.f_globals is not site[1].__dict__:self.refuse('Registration source/callsite/argument order differs.',kind,args,kwargs,caller)
        self.reg_kept.append((args,kwargs));index=self.reg_index
        self.reg_events.append(dict(event='request',kind=kind,index=index,caller_source=caller.f_code.co_filename,caller_line=caller.f_lineno,args=[metadata(v) for v in args],kwargs=[[k,metadata(v)] for k,v in kwargs.items()],argument_keys=[list(self._callback_key(v)) for v in args],keyword_keys=[[k,list(self._callback_key(v))] for k,v in kwargs.items()]))
        self.reg_pending=dict(kind=kind,index=index,args=args,kwargs=kwargs,seen=None)

    def _register_native(self,kind,caller,args,kwargs):
        self._registration_request(kind,caller,args,kwargs)
        try:
            result=self.reg_originals[kind](*args,**kwargs)
            if self.reg_pending['seen']!='return':self.refuse('Registration native entry/return unavailable.',kind,args,kwargs,caller)
            if result is not (args[0] if kind=='exit' else None):self.refuse('Original registration return differs.',kind,args,kwargs,caller)
            self.reg_pending=None;self.reg_index+=1;self.check();return result
        except BaseException as error:self.refuse('Original registration refused: '+type(error).__name__,kind,args,kwargs,caller)

    def _registration_profile(self,frame,event,arg):
        kind=next((k for k,v in self.reg_originals.items() if arg is v),None) if event.startswith('c_') else None
        if kind is not None:
            # The natural TemporaryDirectory finalizer uses the same original
            # atexit API, with its separate exact source/registry lifecycle.
            if kind=='exit' and self.exit_pending:return False
            pending=self.reg_pending
            if pending is None or pending['kind']!=kind or frame.f_code is not self.methods['_register_native'][1]:self.refuse('Direct native registration alias refused.',kind,caller=frame)
            self.reg_check()
            if event=='c_call' and pending['seen'] is None:pending['seen']='entry';name='native_entry'
            elif event=='c_return' and pending['seen']=='entry':pending['seen']='return';name='native_return'
            else:self.refuse('Registration native sequence/exception differs.',kind,pending['args'],pending['kwargs'],frame)
            self.reg_events.append(dict(event=name,kind=kind,index=pending['index'],caller_source=frame.f_code.co_filename,caller_line=frame.f_lineno));return True
        if self.reg_guard is not None and frame.f_code is self.reg_thread_code:
            if event=='call':
                callback=frame.f_locals['func'];args=frame.f_locals['arg'];kwargs=frame.f_locals['kwargs']
                self._registration_request('threading',frame.f_back,(callback,*args),kwargs)
                try:self.reg_guard.begin_threading(self.reg_threading,callback,args,kwargs)
                except BaseException as error:self.refuse('Internal threading entry binding refused: '+str(error),'threading',(callback,*args),kwargs,frame.f_back)
                self.reg_thread_frame=frame;return True
            if event=='return':
                if self.reg_thread_frame is not frame or self.reg_pending is None:self.refuse('Unmatched internal threading return.','threading',caller=frame)
                try:self.reg_guard.end_threading(arg)
                except BaseException as error:self.refuse('Internal threading return binding refused: '+str(error),'threading',caller=frame)
                self.reg_events.append(dict(event='internal_return',kind='threading',index=self.reg_index,partial_events=list(self.reg_guard.thread_events)))
                self.reg_pending=None;self.reg_thread_frame=None;self.reg_index+=1;self.check();return True
        return False

    def registration_result(self):
        self.check()
        if self.reg_guard is None or self.reg_index!=len(self.reg_routes) or self.reg_pending is not None:self.refuse('Incomplete unified registration sequence.')
        return dict(requests=self.reg_index,events=list(self.reg_events),bindings=self.reg_bindings,stable_state=self.reg_guard.summary(),callback_invocations_by_dispatcher=0,target_audit_installation_accepted=False,preboundary_registrations_observed=False,full_actual_route_accepted=False)

    def composed_result(self):
        return dict(capability=self.capability_result(),registrations=self.registration_result(),one_audit_profile_owner=True,full_actual_route_accepted=False)

