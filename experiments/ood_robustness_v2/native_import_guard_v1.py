"""Cooperative cached-import instrumentation; no sandbox or science permission."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import sys
import types


class ImportRefusal(BaseException):
    pass


def environment():
    return {'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu',
        'XLA_PYTHON_CLIENT_PREALLOCATE':'false','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'LC_CTYPE':'C.UTF-8','TF_CPP_MIN_LOG_LEVEL':'1','TPU_SKIP_MDS_QUERY':'1',
        'MUJOCO_PY_MUJOCO_PATH':'/home/dbayha/.mujoco/mujoco210','MUJOCO_PY_FORCE_CPU':'1',
        'LD_LIBRARY_PATH':'/home/dbayha/.mujoco/mujoco210/bin',
        'PYGLFW_LIBRARY':'/home/dbayha/miniconda3/envs/corl-orig-local/lib/python3.10/site-packages/glfw/x11/libglfw.so'}


def pin(path):
    p=Path(path);s=p.stat()
    if not p.is_absolute() or str(p.resolve())!=str(p) or not p.is_file() or s.st_nlink!=1:
        raise ImportRefusal('Exact regular unaliased source/cache/lock path required.')
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns])


def kernel_environment():
    raw=Path('/proc/self/environ').read_bytes()
    if not raw.endswith(b'\0'):raise ImportRefusal('Incomplete kernel environment.')
    pairs=[s.decode().split('=',1) for s in raw[:-1].split(b'\0')]
    if any(len(p)!=2 for p in pairs) or len({p[0] for p in pairs})!=len(pairs):
        raise ImportRefusal('Malformed kernel environment.')
    return dict(pairs)


class NativeImportGuard:
    blocked_names=frozenset(('build','_build_impl','build_extensions','fix_shared_library',
        'manually_link_libraries','build_callback_fn','get_nvidia_lib_dir'))

    def __init__(self,bindings,*,effective_reader=None,kernel_reader=None,pid_reader=os.getpid):
        self.failed=False;self.installed=False;self.phase=-1;self.finished=False
        self.pid_reader=pid_reader;self.pid=pid_reader()
        self.effective_reader=effective_reader or (lambda:dict(os.environ))
        self.kernel_reader=kernel_reader or kernel_environment
        self.bindings=json.loads(json.dumps(bindings));self.binding_bytes=json.dumps(self.bindings,sort_keys=True)
        self.events=[];self.builder_calls={};self.active=False
        try:
            if set(bindings)!={'builder','cache','lock'}:raise ImportRefusal('Exactly three role bindings required.')
            if len({b['path'] for b in bindings.values()})!=3:raise ImportRefusal('Aliased role paths.')
            for b in bindings.values():
                if pin(b['path'])!=b:raise ImportRefusal('Stale binding.')
            self.builder=bindings['builder']['path'];self.lock=bindings['lock']['path']
            self.lock_parent=str(Path(self.lock).parent)
            self.codes=set()
            def gather(c):
                self.codes.add(c)
                for v in c.co_consts:
                    if isinstance(v,types.CodeType):gather(v)
            gather(compile(Path(self.builder).read_bytes(),self.builder,'exec',dont_inherit=True))
            self.check()
        except BaseException:self.failed=True;raise

    def refuse(self,message):
        self.failed=True
        raise ImportRefusal(message)

    def check(self):
        try:
            if self.failed or self.pid_reader()!=self.pid:self.refuse('Poisoned or forked route.')
            if json.dumps(self.bindings,sort_keys=True)!=self.binding_bytes:self.refuse('Bindings mutated.')
            if self.effective_reader()!=environment() or self.kernel_reader()!=environment():self.refuse('Exact native import environment required.')
            for b in self.bindings.values():
                s=Path(b['path']).stat()
                if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=b['stat']:
                    self.refuse('Source/cache/package-lock identity changed.')
            return True
        except BaseException:self.failed=True;raise

    def advance(self,name):
        self.check()
        phases=('python','numpy','jax','cpu_backend','mujoco_py')
        if self.finished or self.phase+1>=len(phases) or name!=phases[self.phase+1]:self.refuse('Import phase order/refusal.')
        self.phase+=1
        return dict(phase=name,pid=self.pid,effective_environment=self.effective_reader(),kernel_environment=self.kernel_reader())

    def profile(self,frame,event,arg):
        if not self.active:return
        if self.failed:self.refuse('Previous import refusal.')
        if event=='call' and frame.f_code.co_filename==self.builder:
            if frame.f_code not in self.codes:self.refuse('Changed live builder code.')
            name=frame.f_code.co_name
            if name in self.blocked_names:self.refuse('Rebuild/discovery route refused before function body: '+name)
            self.builder_calls[name]=self.builder_calls.get(name,0)+1

    def audit(self,event,args):
        if not self.active:return
        # These checks constrain the cooperative Python import route. Native C
        # syscalls and arbitrary interpreter tampering are outside this guard.
        if event.startswith('subprocess.') or event in ('os.system','os.fork','os.forkpty','os.exec','os.posix_spawn','os.spawn'):
            self.refuse('Child/process launch refused: '+event)
        if event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate','os.chmod','os.chown'):
            self.refuse('Filesystem mutation refused: '+event)
        if event=='open':
            path,mode,flags=args
            writing=isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                name=os.fsdecode(path) if isinstance(path,(str,bytes,os.PathLike)) else None
                expected=os.O_WRONLY|os.O_APPEND|os.O_CREAT
                if self.failed or name!=self.lock or (flags & ~getattr(os,'O_CLOEXEC',0))!=expected or mode!='a':
                    self.refuse('Only exact existing package-lock append-open allowed.')
                self.check();self.events.append(dict(event=event,path=name,mode=mode,flags=flags))
        elif event=='os.mkdir':
            if self.failed or os.fsdecode(args[0])!=self.lock_parent or not Path(self.lock_parent).is_dir():
                self.refuse('Only existing package-lock parent mkdir check allowed.')
            self.events.append(dict(event=event,path=self.lock_parent))
        elif event=='os.putenv':
            key,value=map(os.fsdecode,args)
            if self.failed or environment().get(key)!=value:self.refuse('Environment mutation refused.')
        elif event=='os.unsetenv':self.refuse('Environment removal refused.')
        elif event.startswith('fcntl.'):
            if event!='fcntl.lockf':self.refuse('Undeclared lock operation.')
            fd,cmd,length,start,whence=args
            if self.failed or os.readlink('/proc/self/fd/%d'%fd)!=self.lock or cmd not in (fcntl.LOCK_EX|fcntl.LOCK_NB,fcntl.LOCK_UN) or (length,start,whence)!=(None,None,0):
                self.refuse('Only existing package-lock whole-file lock/unlock allowed.')
            self.events.append(dict(event=event,path=self.lock,command=cmd))
        elif event.startswith('socket.'):self.refuse('Network/socket route refused.')

    def install(self):
        self.check()
        if self.installed or sys.getprofile() is not None:self.refuse('Reused or already profiled route.')
        self.installed=True;self.active=True
        sys.addaudithook(self.audit);sys.setprofile(self.profile)

    def complete(self):
        self.check()
        if not self.installed or self.finished or self.phase!=4:self.refuse('Incomplete or repeated native import route.')
        if sys.getprofile()!=self.profile:self.refuse('Profile hook lost.')
        # No import callback remains after this point; saving the final evidence
        # is the pinned probe's only remaining operation.
        self.finished=True;self.active=False;sys.setprofile(None)
        for b in self.bindings.values():
            if pin(b['path'])!=b:self.refuse('Final source/cache/package-lock bytes changed.')
        return dict(builder_calls=dict(self.builder_calls),allowed_write_events=list(self.events),
                    native_import_route_completed=True,scientific_execution_accepted=False)
