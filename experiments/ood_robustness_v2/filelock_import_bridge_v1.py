"""Observe a source-bound module capability call once, without calling it.

This bridge composes the closed scratch lifecycle. It is cooperative Python
instrumentation, not a security sandbox or acceptance of full filelock imports.
The actual stdlib graph and filelock hook registration gates remain external.
"""
import ast
import hashlib
import os
from pathlib import Path
import sys
import types
import filelock_scratch_v1 as lifecycle

PARENT_SHA256='9cff11889c93fb9d645c3640f8c188ac3d5d6005a4584616a1da367db9560548'


class BridgeRefusal(BaseException):
    pass


class ImportScratchBridge:
    def __init__(self,root,module_name,bindings,result_name='_LINK_HONORS_FOLLOW_SYMLINKS'):
        self.first_refusal=None;self.failed=False;self.active=False;self.used=False;self.finished=False
        self.pid=os.getpid();self.root=str(root);self.module_name=module_name;self.result_name=result_name
        self.bindings={k:dict(v,stat=list(v['stat'])) for k,v in bindings.items()}
        self.source=bindings['strict']['path'];self.module=None;self.module_frame=None;self.callback_frame=None
        self.scratch=None;self.module_seen=False;self.callback_seen=False;self.callback_returned=False;self.module_returned=False
        self.boundary_events=[];self.releasing=False
        self.parent_path=Path(lifecycle.__file__)
        if hashlib.sha256(self.parent_path.read_bytes()).hexdigest()!=PARENT_SHA256:self.refuse('Closed lifecycle source mismatch.')
        parent=compile(self.parent_path.read_bytes(),str(self.parent_path),'exec',dont_inherit=True)
        self.parent_codes={c.co_name:c for c in lifecycle.codes(parent) if c.co_name in lifecycle.FilelockScratch.__dict__}
        self.parent_functions={n:getattr(lifecycle.FilelockScratch,n) for n in self.parent_codes}
        raw=Path(self.source).read_bytes();self.module_code=compile(raw,self.source,'exec',dont_inherit=True)
        self.all_codes=set(lifecycle.codes(self.module_code))
        candidates=[c for c in self.all_codes if c.co_name=='_probe_link_follow_symlinks']
        if len(candidates)!=1:self.refuse('One capability function required.')
        self.callback_code=candidates[0]
        tree=ast.parse(raw);calls=[]
        for node in tree.body:
            targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,ast.AnnAssign) else []
            if any(isinstance(t,ast.Name) and t.id==result_name for t in targets):
                call=node.value
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='_probe_link_follow_symlinks' or call.args or call.keywords:
                    self.refuse('Exact module-level capability assignment required.')
                calls.append(node.lineno)
        if len(calls)!=1:self.refuse('One declared module callsite required.')
        self.call_line=calls[0]
        self.contract=(self.root,self.module_name,self.result_name,self.source,self.call_line)
        self.source_bindings_before={k:lifecycle.source_pin(v['path']) for k,v in self.bindings.items()}
        if self.source_bindings_before!=self.bindings:self.refuse('Stale bridge source bindings.')
        if module_name in sys.modules:self.refuse('Module must not already be imported.')
        self.check()

    def refuse(self,message,event=None,args=None):
        if self.first_refusal is None:
            self.first_refusal=dict(message=message,event=event,args=None if args is None else [str(x) for x in args])
        self.failed=True
        if self.used:self.active=True
        raise BridgeRefusal(self.first_refusal['message'])

    def check(self):
        try:
            if self.failed:self.refuse('Previously refused import bridge.')
            if os.getpid()!=self.pid:self.refuse('Bridge owner process changed.')
            if (self.root,self.module_name,self.result_name,self.source,self.call_line)!=self.contract:self.refuse('Bridge contract changed.')
            if os.environ.get('TMPDIR')!=self.root:self.refuse('Explicit scratch environment changed.')
            if hashlib.sha256(self.parent_path.read_bytes()).hexdigest()!=PARENT_SHA256:self.refuse('Closed lifecycle source changed.')
            for n,f in self.parent_functions.items():
                if getattr(lifecycle.FilelockScratch,n) is not f or f.__code__!=self.parent_codes[n]:self.refuse('Closed lifecycle live method changed.')
            for k,b in self.source_bindings_before.items():
                if self.bindings[k]!=b or lifecycle.source_pin(b['path'])!=b:self.refuse('Bridge source binding changed.')
            if set(self.bindings)!=set(self.source_bindings_before):self.refuse('Bridge source roles changed.')
            if self.module is not None:
                if sys.modules.get(self.module_name) is not self.module or self.module.__dict__ is not self.module_frame.f_globals or self.module.__dict__.get('__file__')!=self.source:
                    self.refuse('Live module namespace changed.')
            if self.scratch is not None:
                fn=self.scratch.callback
                if self.module.__dict__.get('_probe_link_follow_symlinks') is not fn or fn.__defaults__ is not None or fn.__kwdefaults__ is not None or fn.__closure__ is not None:
                    self.refuse('Live capability function or defaults changed.')
                self.scratch.check()
        except BaseException as e:
            detail=None if self.scratch is None else self.scratch.first_refusal
            if detail:self.refuse(detail['message'],detail['event'],detail['args'])
            self.refuse('Bridge identity check failed: '+type(e).__name__)

    def profile(self,frame,event,arg):
        if not self.active:return
        if event=='c_call' and arg is getattr(os,'register_at_fork',None):
            self.refuse('Filelock at-fork registration requires a separate state binding.','os.register_at_fork',(frame.f_code.co_filename,frame.f_lineno))
        if frame.f_code.co_filename!=self.source:return
        if self.failed:self.refuse('Previously refused import bridge.')
        if frame.f_code not in self.all_codes:self.refuse('Unexpected live module code.',event,(frame.f_code.co_name,))
        if event=='call' and frame.f_code==self.module_code:
            if self.module_seen:self.refuse('Repeated module execution.')
            module=sys.modules.get(self.module_name)
            if not isinstance(module,types.ModuleType) or module.__dict__ is not frame.f_globals or frame.f_globals.get('__name__')!=self.module_name or frame.f_globals.get('__file__')!=self.source:
                self.refuse('Registered module namespace required.')
            self.module=module;self.module_frame=frame;self.module_seen=True
            self.check();self.boundary_events.append(dict(event='module_entry',line=frame.f_lineno));return
        if event=='call' and frame.f_code==self.callback_code:
            if not self.module_seen or self.callback_seen or frame.f_back is not self.module_frame or frame.f_back.f_lineno!=self.call_line or frame.f_globals is not self.module.__dict__:
                self.refuse('Capability must enter once from exact module callsite.')
            callback=frame.f_globals.get('_probe_link_follow_symlinks')
            if not isinstance(callback,types.FunctionType) or callback.__code__!=frame.f_code or callback.__globals__ is not frame.f_globals:
                self.refuse('Live capability function binding differs.')
            self.check()
            self.scratch=lifecycle.FilelockScratch(self.root,callback,self.bindings)
            # Observe the function already entering. Never call scratch.run() or callback().
            self.scratch.used=True;self.scratch.active=True
            self.callback_frame=frame;self.callback_seen=True
            self.boundary_events.append(dict(event='callback_entry',line=frame.f_back.f_lineno));return
        if event=='return' and frame is self.callback_frame:
            self.check();s=self.scratch
            if arg is not True or s.stage!=12 or Path(s.directory).exists() or Path(s.testfile).exists() or list(Path(self.root).iterdir()):
                self.refuse('Capability returned without complete exact scratch lifecycle.','callback_return',(arg,s.stage))
            s.active=False;self.callback_returned=True
            self.boundary_events.append(dict(event='callback_return',value=True,stage=s.stage));return
        if event=='return' and frame is self.module_frame:
            self.check()
            if not self.callback_returned or frame.f_globals.get(self.result_name) is not True:self.refuse('Module returned without the observed capability result.')
            self.module_returned=True;self.boundary_events.append(dict(event='module_return',result=True));return

    def audit(self,event,args):
        if not self.active:return
        if event in ('sys.addaudithook','sys.setprofile') and not self.releasing:
            self.refuse('Additional audit/profile registration is unbound.',event,args)
        writing=event=='open' and isinstance(args[2],int) and bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        mutation=event in ('os.mkdir','os.remove','os.rmdir','os.link','os.symlink','os.rename','os.utime','os.chmod','os.chown','os.truncate','shutil.rmtree')
        forbidden=event.startswith(('subprocess.','socket.','fcntl.')) or event in ('os.system','os.fork','os.forkpty','os.exec','os.posix_spawn','os.spawn','os.putenv','os.unsetenv')
        if forbidden:self.refuse('Process/network/environment/lock route refused.',event,args)
        if self.failed and (writing or mutation):self.refuse('Mutation after bridge refusal.',event,args)
        if self.scratch is not None and self.scratch.active:
            try:self.scratch.audit(event,args)
            except BaseException as e:
                detail=self.scratch.first_refusal
                self.refuse(detail['message'] if detail else 'Scratch observation failed: '+type(e).__name__,event,args)
        elif writing or mutation:self.refuse('Write outside capability scope.',event,args)

    def install(self):
        self.check()
        if self.used or sys.getprofile() is not None:self.refuse('Bridge is single-use and requires no existing profile.')
        self.used=True
        sys.addaudithook(self.audit);sys.setprofile(self.profile);self.active=True

    def complete(self):
        self.check()
        if self.finished or not self.active or not self.module_returned or sys.getprofile()!=self.profile:self.refuse('Incomplete, repeated or detached bridge.')
        if self.module.__dict__.get(self.result_name) is not True:self.refuse('Observed module capability result changed.')
        self.finished=True;self.releasing=True
        sys.setprofile(None);self.active=False
        return dict(module=self.module_name,source=self.source,owner_pid=self.pid,scratch_root=self.root,
                    callback_invocations_observed=1,callback_invocations_by_bridge=0,capability_result=True,
                    boundary_events=list(self.boundary_events),scratch_events=list(self.scratch.events),root_retained_empty=True,
                    real_filelock_package_accepted=False,complete_stdlib_graph_accepted=False,native_environment_accepted=False,scientific_execution_accepted=False)
