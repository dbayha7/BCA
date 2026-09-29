"""Concrete v3 bootstrap/diagnostic dispatcher; native import phases remain held.

One audit/profile owner observes exact worker-owned creation and diagnostic native
calls. This is cooperative instrumentation, not a security sandbox, native-C
syscall monitor, full dependency-graph acceptance or permission to import targets.
No cleanup/uninstall method exists. The enclosing pinned worker must os._exit.
"""
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


class NativeImportDispatcherV3:
    """Bootstrap phase only. Fixture scope is explicit; full native routes refuse.

This first implementation cannot consume a production launch acceptance. A later
version must bind and prereview the remaining declared routes without changing
the closed declaration. No caller-provided ready boolean can enable imports.
"""
    def __init__(self,declaration_bytes,binding_bytes,source_sha256):
        self._first=None;self.failed=False;self.used=False;self.active=False;self.released=False
        self.phase='uninstalled';self.ticket=None;self.native_seen=None;self.events=[]
        self.dirs={};self.descriptors={};self.written={};self.total=0;self.write_fd=None
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
        if b['schema']!='native-import-v3-bootstrap-fixture-v1' or b['purpose']!='bootstrap_fixture_only' or b['declaration_sha256']!=DECLARATION_SHA:
            self.refuse('Production dispatch remains held.')
        self.parent=b['parent'];self.scratch=b['scratch'];self.diagnostics=b['diagnostics']
        # Explicit fixture-only roots cannot alias the proposed actual worker roots.
        if not self.parent.startswith('/home/dbayha/bca-work/ood-native-v3-bootstrap-fixtures/'):
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
        owner=next(v for v in compiled.co_consts if type(v) is types.CodeType and v.co_name=='NativeImportDispatcherV3')
        expected_codes={v.co_name:v for v in owner.co_consts if type(v) is types.CodeType}
        if any(fn.__code__!=expected_codes[n] or fn.__code__.co_filename!=self.source_path for n,(fn,_,_,_) in self.methods.items()):
            self.refuse('Dispatcher methods do not match compiled source.')
        self.method_names=tuple(self.methods)
        self.module_globals=globals()
        self.global_refs={n:globals()[n] for n in ('metadata','process_identity','kernel_environment','dir_id','DECLARATION_SHA','FILES','MAX_FILE','MAX_TOTAL','os','sys','stat','hashlib','json','types','__file__')}
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
        if self.used or sys.getprofile() is not None:self.refuse('Dispatcher install is single-use.')
        self.used=True
        try:
            sys.addaudithook(self.audit_hook)
            sys.audit('ood.native_v3.bootstrap.canary',self.install_token)
            if not self.install_seen:self.refuse('Own audit hook installation suppressed.')
            self.installing_profile=True;sys.setprofile(self.profile_hook);self.installing_profile=False
            self.active=True;self.phase='awaiting_release';self.check()
        except BaseException as e:self.refuse('Dispatcher installation failed: '+type(e).__name__)

    def release(self,observed):
        self.check()
        if self.phase!='awaiting_release' or self.released:self.refuse('Repeated or premature release.')
        if observed!={'identity':self.identity,'environment':self.environment}:self.refuse('Parent release identity differs.')
        self.released=True;self.phase='released'

    def profile(self,frame,event,arg):
        if not self.active:return
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
        registration=event in ('sys.setprofile','sys.addaudithook')
        target=event=='import' and type(args[0]) is str and args[0].split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl')
        external=event.startswith(('subprocess.','socket.','fcntl.')) or event in ('os.system','os.fork','os.forkpty','os.exec','os.posix_spawn','os.spawn','os.putenv','os.unsetenv')
        writing=event=='open' and type(args[2]) is int and bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        mutation=event in ('os.mkdir','os.remove','os.rmdir','os.link','os.symlink','os.rename','os.utime','os.chmod','os.chown','os.truncate','shutil.rmtree')
        # Frame inspection itself emits an audit event. Classify first so benign
        # introspection cannot recurse into another frame inspection.
        if not (registration or target or external or writing or mutation):return
        origin=sys._getframe(1)
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
                    bootstrap_fixture_only=True,target_import_accepted=False,full_route_implemented=False,
                    target_audit_effectiveness_accepted=False,scientific_execution_accepted=False)
