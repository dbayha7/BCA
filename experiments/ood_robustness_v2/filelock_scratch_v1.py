"""One source-bound filelock capability probe in a private scratch directory.

Cooperative Python instrumentation only. Not an import route, security sandbox,
scientific storage acceptance or permission to acquire the scientific lease.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import types


class ScratchRefusal(BaseException):
    pass


def source_pin(path):
    p=Path(path);s=p.stat()
    if not p.is_absolute() or str(p.resolve())!=str(p) or not p.is_file():
        raise ScratchRefusal('Exact regular source path required.')
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink])


def codes(root):
    result=[root]
    for item in root.co_consts:
        if isinstance(item,types.CodeType):result.extend(codes(item))
    return result


class FilelockScratch:
    def __init__(self,root,callback,bindings):
        self.first_refusal=None;self.failed=False;self.used=False;self.active=False
        self.pid=os.getpid();self.root=str(root);self.callback=callback
        self.bindings={k:dict(v,stat=list(v['stat'])) for k,v in bindings.items()}
        self.binding_bytes=json.dumps(self.bindings,sort_keys=True)
        self.events=[];self.stage=0;self.testfile=None;self.directory=None;self.directory_id=None
        self.remaining=set();self.link_id=None;self.directory_fd=None
        self.expected_api={n:getattr(os,n) for n in ('open','write','close','mkdir','rmdir','unlink','remove','link','utime','scandir')}
        self.expected_helpers=(tempfile.TemporaryDirectory,tempfile._get_default_tempdir,shutil.rmtree,Path.touch)
        if set(bindings)!={'strict','tempfile','shutil','pathlib','os'}:self.refuse('Five exact source roles required.')
        for b in self.bindings.values():
            if source_pin(b['path'])!=b:self.refuse('Stale source binding.')
        if self.bindings['strict']['stat'][-1]!=1:self.refuse('Callback source must be unaliased.')
        s=Path(root).lstat()
        if not Path(root).is_absolute() or str(Path(root).resolve())!=str(root) or not stat.S_ISDIR(s.st_mode) or stat.S_IMODE(s.st_mode)!=0o700 or s.st_uid!=os.geteuid() or list(Path(root).iterdir()):
            self.refuse('Fresh empty private owned scratch directory required.')
        self.root_id=(s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))
        if os.environ.get('TMPDIR')!=self.root or tempfile.tempdir is not None:self.refuse('Explicit fresh tempfile selection required.')
        tree=compile(Path(bindings['strict']['path']).read_bytes(),bindings['strict']['path'],'exec',dont_inherit=True)
        selected=[c for c in codes(tree) if c.co_name=='_probe_link_follow_symlinks']
        if len(selected)!=1 or callback.__code__!=selected[0] or callback.__closure__ or callback.__defaults__ or callback.__kwdefaults__:
            self.refuse('Exact callback code required.')
        self.callback_code=callback.__code__;self.callback_globals=callback.__globals__
        self.check()

    def refuse(self,message,event=None,args=None):
        if self.first_refusal is None:
            self.first_refusal=dict(message=message,event=event,args=None if args is None else [str(x) for x in args])
        self.failed=True
        raise ScratchRefusal(self.first_refusal['message'])

    def check(self):
        try:return self._check()
        except BaseException as e:self.refuse('Scratch identity check failed: '+type(e).__name__)

    def _check(self):
        if self.failed:self.refuse('Previously refused scratch route.')
        if os.getpid()!=self.pid:self.refuse('Scratch owner process changed.')
        if json.dumps(self.bindings,sort_keys=True)!=self.binding_bytes:self.refuse('Source bindings mutated.')
        if os.environ.get('TMPDIR')!=self.root:self.refuse('Scratch environment changed.')
        s=Path(self.root).lstat()
        if (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))!=self.root_id or not stat.S_ISDIR(s.st_mode):self.refuse('Scratch ownership or identity changed.')
        if self.callback.__code__!=self.callback_code or self.callback.__globals__ is not self.callback_globals:self.refuse('Callback changed.')
        g=self.callback_globals
        if g.get('os') is not os or g.get('tempfile') is not tempfile or g.get('Path') is not Path or g.get('_HAS_LINK') is not True:self.refuse('Callback globals changed.')
        if any(getattr(os,n) is not value for n,value in self.expected_api.items()):self.refuse('Filesystem API changed.')
        if (tempfile.TemporaryDirectory,tempfile._get_default_tempdir,shutil.rmtree,Path.touch)!=self.expected_helpers:self.refuse('Filesystem helper changed.')
        for b in self.bindings.values():
            if source_pin(b['path'])!=b:self.refuse('Source changed during scratch route.')
        if self.directory_id is not None and self.stage<12:
            s=Path(self.directory).lstat()
            if (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))!=self.directory_id or not stat.S_ISDIR(s.st_mode):self.refuse('Temporary directory changed.')

    def inside_callback(self):
        f=sys._getframe(1)
        while f:
            if f.f_code==self.callback_code and f.f_globals is self.callback_globals:return True
            f=f.f_back
        return False

    def audit(self,event,args):
        if not self.active:return
        mutation=event in ('os.mkdir','os.remove','os.rmdir','os.link','os.symlink','os.rename','os.utime','os.chmod','os.chown','os.truncate','shutil.rmtree')
        opening=event=='open' and ((self.stage==8 and args[0]==self.directory) or bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        extra=event.startswith(('subprocess.','socket.','fcntl.')) or event in ('os.system','os.fork','os.forkpty','os.exec','os.posix_spawn','os.putenv','os.unsetenv')
        if not (mutation or opening or extra or event=='os.scandir'):return
        if extra:self.refuse('Process/network/environment/lock operation refused.',event,args)
        self.check()
        if not self.inside_callback():self.refuse('Operation outside the bound callback.',event,args)
        self.events.append(dict(stage=self.stage,event=event,args=[str(x) for x in args]))
        if self.stage==0 and event=='open':
            path,mode,flags=args;p=Path(path)
            expected=os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_CLOEXEC',0)
            if p.parent!=Path(self.root) or not re.fullmatch('[a-z0-9_]{8}',p.name) or p.exists() or mode is not None or flags!=expected:self.refuse('Default tempfile probe open mismatch.',event,args)
            self.testfile=str(p);self.stage=1;return
        if self.stage==1 and event=='os.remove' and args==(self.testfile,-1):
            p=Path(self.testfile);s=p.lstat()
            if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.geteuid() or stat.S_IMODE(s.st_mode)!=0o600 or p.read_bytes()!=b'blat':self.refuse('Default tempfile probe readback mismatch.',event,args)
            self.events[-1]['readback_hex']='626c6174';self.stage=2;return
        if self.stage==2 and event=='os.mkdir':
            path,mode,dirfd=args;p=Path(path)
            if Path(self.testfile).exists() or p.parent!=Path(self.root) or not re.fullmatch('tmp[a-z0-9_]{8}',p.name) or p.exists() or mode!=0o700 or dirfd!=-1:self.refuse('Temporary directory create mismatch.',event,args)
            self.directory=str(p);self.stage=3;return
        if self.stage==3:
            s=Path(self.directory).lstat();self.directory_id=(s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))
            if s.st_uid!=os.geteuid() or stat.S_IMODE(s.st_mode)!=0o700:self.refuse('Temporary directory permissions mismatch.',event,args)
            if event=='os.utime' and args==(str(Path(self.directory)/'probe-source'),None,None,-1):self.stage=4;return
        if self.stage==4 and event=='open':
            path,mode,flags=args
            if path!=str(Path(self.directory)/'probe-source') or mode is not None or flags!=(os.O_WRONLY|os.O_CREAT|getattr(os,'O_CLOEXEC',0)) or Path(path).exists():self.refuse('Capability source open mismatch.',event,args)
            self.stage=5;return
        if self.stage==5 and event=='os.link':
            source,target,srcfd,dstfd=args
            if (source,target,srcfd,dstfd)!=(str(Path(self.directory)/'probe-source'),str(Path(self.directory)/'probe-link'),-1,-1):self.refuse('Capability hard-link path mismatch.',event,args)
            p=Path(source);s=p.lstat()
            if s.st_nlink!=1 or not stat.S_ISREG(s.st_mode) or p.read_bytes()!=b'' or Path(target).exists():self.refuse('Capability source is not fresh empty file.',event,args)
            self.link_id=(s.st_dev,s.st_ino);self.stage=6;return
        if self.stage==6 and event=='shutil.rmtree' and args==(self.directory,):
            for name in ('probe-source','probe-link'):
                p=Path(self.directory)/name;s=p.lstat()
                if (s.st_dev,s.st_ino)!=self.link_id or s.st_nlink!=2 or p.read_bytes()!=b'':self.refuse('Hard-link result readback mismatch.',event,args)
            self.events[-1]['verified_empty_hard_link_pair']=True;self.stage=8;return
        if self.stage==8 and event=='open' and args==(self.directory,None,getattr(os,'O_CLOEXEC',0)):
            self.stage=9;return
        if self.stage==9 and event=='os.scandir' and type(args[0]) is int:
            fd=args[0]
            if os.readlink('/proc/self/fd/%d'%fd)!=self.directory:self.refuse('Cleanup descriptor mismatch.',event,args)
            self.directory_fd=fd;self.remaining={'probe-source','probe-link'};self.stage=10;return
        if self.stage==10 and event=='os.remove':
            name,fd=args
            if name not in self.remaining or fd!=self.directory_fd or os.readlink('/proc/self/fd/%d'%fd)!=self.directory:self.refuse('Cleanup unlink outside exact pair.',event,args)
            p=Path(self.directory)/name;s=p.lstat()
            if (s.st_dev,s.st_ino)!=self.link_id or s.st_nlink!=len(self.remaining) or p.read_bytes()!=b'':self.refuse('Cleanup file identity changed.',event,args)
            self.remaining.remove(name)
            if not self.remaining:self.stage=11
            return
        if self.stage==11 and event=='os.rmdir' and args==(self.directory,-1):
            if list(Path(self.directory).iterdir()):self.refuse('Cleanup directory not empty.',event,args)
            self.stage=12;return
        self.refuse('Unexpected scratch operation or order.',event,args)

    def run(self):
        self.check()
        if self.used:self.refuse('Repeated scratch callback.')
        self.used=True;self.active=True;sys.addaudithook(self.audit)
        try:
            result=self.callback()
            self.check()
            if result is not True or self.stage!=12 or Path(self.directory).exists() or Path(self.testfile).exists() or list(Path(self.root).iterdir()):self.refuse('Incomplete scratch lifecycle.')
            self.active=False
            return dict(result=True,events=self.events,owner_pid=self.pid,scratch_root=self.root,root_retained_empty=True,
                        full_native_import_accepted=False,scientific_execution_accepted=False)
        except BaseException as e:
            self.refuse('Scratch callback failed: '+type(e).__name__)
