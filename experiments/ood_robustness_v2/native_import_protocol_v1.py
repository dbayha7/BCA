"""Bounded native-import parent/completion protocol; full native dispatch held.

The current binding is explicitly fixture-only. It composes with the exact closed
v4 dispatcher without installing another hook/profile. No callback replay or
scientific authority is supplied. Failure never cleans/reuses worker roots.
"""
import datetime,hashlib,json,os,selectors,signal,stat,subprocess,sys,time,types
from pathlib import Path

CONTRACT='5f922e3d0d27116139fd99293f537427f6d369f702b7fd685dbbe9de67567763'
DISPATCHER='9380e8fceae651015a6207dbbc579a45bfb54d913f52d88a437403feba6a31db'
PREFIX='/home/dbayha/bca-work/ood-native-v4-capability-fixtures/monitor_20260929T044506Z/'
FILES=('events.jsonl','final.json','modules.json','maps.txt')
LOG_LIMIT=8*1024*1024
TIMEOUT=180
FRAME_LIMIT=65536
MARKER=b'OOD_NATIVE_PROTOCOL_V1 '

def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def parse(raw):
    require(type(raw) is bytes and 0<len(raw)<=FRAME_LIMIT,'Protocol frame bound exceeded.')
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'Duplicate protocol key.');result[k]=v
        return result
    result=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite protocol.')))
    canonical(result);return result
def identity(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_bytes();tail=raw.rsplit(b') ',1)[1].split()
    return dict(pid=pid,parent_pid=int(tail[1]),session=int(tail[3]),group=int(tail[2]),start_ticks=int(tail[19]),argv_hex=(p/'cmdline').read_bytes().hex())
def environment(pid):
    raw=(Path('/proc')/str(pid)/'environ').read_bytes();require(raw.endswith(b'\0'),'Incomplete kernel environment.')
    pairs=[x.decode().split('=',1) for x in raw[:-1].split(b'\0')];require(len(pairs)==len(dict(pairs)),'Duplicate kernel environment.')
    return dict(pairs)
def directory_id(s):
    require(stat.S_ISDIR(s.st_mode),'Directory required.')
    return [s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)]
def file_id(s):
    require(stat.S_ISREG(s.st_mode),'Regular file required.')
    return [s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def read_file(path,limit,expected=None):
    p=Path(path);before=file_id(p.lstat());require(before[5]==1 and 0<=before[6]<=limit,'Unaliased bounded file required.')
    fd=os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        require(file_id(os.fstat(fd))==before,'File descriptor identity differs.')
        chunks=[];n=0
        while True:
            b=os.read(fd,min(65536,limit+1-n))
            if not b:break
            chunks.append(b);n+=len(b);require(n<=limit,'File grew beyond bound.')
        require(file_id(os.fstat(fd))==before==file_id(p.lstat()),'File changed during read.')
    finally:os.close(fd)
    raw=b''.join(chunks);require(len(raw)==before[6],'File byte length differs.')
    if expected is not None:require(sha(raw)==expected,'Pinned source differs.')
    return dict(identity=before,bytes=len(raw),sha256=sha(raw)),raw
def ancestors(path):
    p=str(path);require(os.path.normpath(p)==p==os.path.realpath(p),'Canonical path required.')
    while p!='/':directory_id(os.lstat(p));p=os.path.dirname(p)
def fsync_directory(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:os.fsync(fd)
    finally:os.close(fd)
def save_once(path,value):
    raw=canonical(value);fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        n=0
        while n<len(raw):
            wrote=os.write(fd,raw[n:]);require(wrote>0,'Evidence write failed.');n+=wrote
        os.fsync(fd)
    finally:os.close(fd)
    fsync_directory(Path(path).parent)

def diagnostics(parent,root_ids):
    """Read-only exact name/fd checks; parent never creates worker roots."""
    parent=Path(parent);result={};opened=[]
    pfd=os.open(str(parent),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        for name in ('diagnostics','scratch'):
            before=directory_id(os.stat(name,dir_fd=pfd,follow_symlinks=False));require(before==root_ids[name] and before[2:]==[1000,1000,0o700],'Worker root identity differs.')
            fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=pfd);opened.append(fd)
            require(directory_id(os.fstat(fd))==before,'Root descriptor differs.')
            names=os.listdir(fd);require(set(names)==set(FILES) if name=='diagnostics' else names==[],'Unexpected diagnostic entries or nonempty scratch.')
        dfd=opened[0]
        for name in FILES:
            before=file_id(os.stat(name,dir_fd=dfd,follow_symlinks=False));require(before[2:6]==[1000,1000,0o600,1] and before[6]<=LOG_LIMIT,'Diagnostic owner/mode/size differs.')
            fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=dfd)
            try:
                require(file_id(os.fstat(fd))==before,'Diagnostic descriptor differs.');h=hashlib.sha256();n=0
                while True:
                    b=os.read(fd,min(65536,LOG_LIMIT+1-n))
                    if not b:break
                    n+=len(b);require(n<=LOG_LIMIT,'Diagnostic read ceiling.');h.update(b)
                require(file_id(os.fstat(fd))==before==file_id(os.stat(name,dir_fd=dfd,follow_symlinks=False)) and n==before[6],'Diagnostic changed during read.')
            finally:os.close(fd)
            result[name]=dict(identity=before,bytes=n,sha256=h.hexdigest())
        require(sum(v['bytes'] for v in result.values())<=4*LOG_LIMIT,'Total diagnostic ceiling.')
        for name,fd in zip(('diagnostics','scratch'),opened):require(directory_id(os.fstat(fd))==root_ids[name]==directory_id(os.stat(name,dir_fd=pfd,follow_symlinks=False)),'Root changed during diagnostic read.')
    finally:
        for fd in opened:os.close(fd)
        os.close(pfd)
    return result

class WorkerProtocol:
    def __init__(self,dispatcher,protocol_sha,entry_sha):
        self.g=dispatcher;self.protocol_sha=protocol_sha;self.entry_sha=entry_sha
        self.entry_path=sys.argv[0];self.source_path=__file__;self.exit_native=os._exit
        self.pid=os.getpid();self.parent=identity(os.getppid());self.owner=identity(self.pid)
        self.pending=None;self.finished=False;self.released=False
        raw=Path(self.source_path).read_bytes();require(sha(raw)==protocol_sha and sha(Path(self.entry_path).read_bytes())==entry_sha,'Worker protocol/entry source differs.')
        self.methods={n:(v,v.__code__,v.__defaults__,v.__kwdefaults__) for n,v in type(self).__dict__.items() if type(v) is types.FunctionType}
        compiled=compile(raw,self.source_path,'exec',dont_inherit=True);owner=next(c for c in compiled.co_consts if type(c) is types.CodeType and c.co_name=='WorkerProtocol')
        codes={c.co_name:c for c in owner.co_consts if type(c) is types.CodeType}
        require(all(fn.__code__==codes[n] for n,(fn,*unused) in self.methods.items()),'Worker method code differs from source.')
        self.globals_pin={n:globals()[n] for n in ('sha','require','canonical','parse','identity','environment','file_id','FILES','MARKER','FRAME_LIMIT','LOG_LIMIT','CONTRACT','DISPATCHER','os','sys','json','stat','Path')}
        top={c.co_name:c for c in compiled.co_consts if type(c) is types.CodeType}
        self.helpers={n:(v,v.__code__,v.__defaults__,v.__kwdefaults__) for n,v in self.globals_pin.items() if type(v) is types.FunctionType}
        require(all(fn.__code__==top[n] for n,(fn,*unused) in self.helpers.items()),'Worker protocol helper code differs.')
        require(self.g.source_sha==DISPATCHER and self.g.first_refusal is None,'Closed dispatcher binding differs.')
    def check(self,diagnostic=False):
        require(os._exit is self.exit_native and os.getpid()==self.pid and identity(self.pid)==self.owner and identity(self.parent['pid'])==self.parent,'Worker/parent/exit identity changed.')
        require(sha(Path(self.source_path).read_bytes())==self.protocol_sha and sha(Path(self.entry_path).read_bytes())==self.entry_sha,'Worker protocol/entry source changed.')
        require(all(globals().get(n) is v for n,v in self.globals_pin.items()),'Worker protocol direct global changed.')
        for n,(fn,code,defaults,kw) in self.methods.items():require(n not in self.__dict__ and type(self).__dict__.get(n) is fn and fn.__code__ is code and fn.__defaults__ is defaults and fn.__kwdefaults__ is kw,'Worker protocol method changed.')
        for fn,code,defaults,kw in self.helpers.values():require(fn.__code__ is code and fn.__defaults__ is defaults and fn.__kwdefaults__ is kw,'Worker protocol helper changed.')
        self.g.check(diagnostic=diagnostic)
    def emit(self,packet):
        raw=canonical(packet);require(len(raw)<=FRAME_LIMIT,'Worker protocol packet ceiling.');sys.stdout.buffer.write(MARKER+raw);sys.stdout.buffer.flush()
    def receive(self):
        raw=sys.stdin.buffer.readline(FRAME_LIMIT+1);require(raw.endswith(b'\n'),'Missing or oversized parent acknowledgment.');return parse(raw)
    def ready_release(self):
        self.check();require(not self.released and self.g.phase=='awaiting_release','Protocol release reused.')
        packet=dict(stage='ready',identity=self.g.identity,environment=self.g.environment,protocol_sha256=self.protocol_sha,entry_sha256=self.entry_sha,dispatcher_sha256=DISPATCHER)
        self.emit(packet);release=self.receive()
        require(release==dict(stage='release',identity=self.owner,environment=self.g.environment,parent=self.parent),'Independent parent release differs.')
        self.check();self.g.release(dict(identity=release['identity'],environment=release['environment']));self.released=True
    def prepare_completion(self,result):
        self.check();require(self.released and not self.finished and self.pending is None and self.g.cap_complete,'Completion before capability or repeated.')
        require(result==self.g.capability_result(),'Completion does not bind actual capability result.')
        modules=sorted(n for n in sys.modules if n.split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl'))
        require(modules==[],'Target import is outside fixture protocol.')
        self.g.write_diagnostic('modules.json',canonical(dict(target_modules=modules,snapshot_only=True)))
        maps=Path('/proc/self/maps').read_bytes();require(len(maps)<=LOG_LIMIT,'Maps diagnostic ceiling.');self.g.write_diagnostic('maps.txt',maps)
        final=dict(schema='native-import-completion-fixture-v1',worker=self.owner,capability=result,profile_present=sys.getprofile() is self.g.profile_hook,full_route_implemented=False,actual_production_profile_executed=False,scientific_execution_accepted=False)
        self.g.write_diagnostic('final.json',canonical(final));self.check()
        # Readback uses ordinary read APIs; no extra write/native operation route.
        manifests={}
        for name,(fd,pin) in self.g.descriptors.items():
            p=Path(self.g.diagnostics)/name;s=p.lstat();raw=p.read_bytes()
            require((s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink)==pin and s.st_size==self.g.written[name]==len(raw),'Worker completion diagnostic changed.')
            manifests[name]=dict(identity=file_id(s),bytes=len(raw),sha256=sha(raw))
        self.check();self.pending=dict(stage='completion_pending',identity=self.owner,environment=self.g.environment,diagnostics=manifests,root_ids={k:list(v[1]) for k,v in self.g.dirs.items()},protocol_sha256=self.protocol_sha,entry_sha256=self.entry_sha,dispatcher_sha256=DISPATCHER,profile_present=True,actual_production_profile_executed=False,scientific_execution_accepted=False)
        self.pending_hash=sha(canonical(self.pending));self.emit(self.pending);return self.pending
    def accept_completion(self):
        self.check();require(self.pending is not None and not self.finished,'No pending completion.')
        ack=self.receive();expected=dict(stage='completion_ack',identity=self.owner,parent=self.parent,completion_sha256=self.pending_hash,diagnostics=self.pending['diagnostics'])
        require(ack==expected and sha(canonical(self.pending))==self.pending_hash,'Parent completion acknowledgment differs.')
        self.check()
        for name,item in self.pending['diagnostics'].items():
            p=Path(self.g.diagnostics)/name;require(file_id(p.lstat())==item['identity'] and sha(p.read_bytes())==item['sha256'],'Diagnostic changed after completion acknowledgment.')
        self.check();self.finished=True
        self.emit(dict(stage='completion_ack_received',identity=self.owner,completion_sha256=self.pending_hash,profile_present=sys.getprofile() is self.g.profile_hook))
        self.check();self.exit_native(0)
    def abort(self,error):
        # Preserve first dispatcher refusal and attempt only its bounded diagnostic
        # route. A changed guard may refuse this too; no deactivation/cleanup.
        try:
            self.check(diagnostic=True)
            self.g.write_diagnostic('events.jsonl',canonical(dict(event='worker_failure',error_type=type(error).__name__,first_refusal=self.g.first_refusal)))
        except BaseException:pass
        try:self.emit(dict(stage='worker_failure',error_type=type(error).__name__,first_refusal=self.g.first_refusal,profile_present=sys.getprofile() is self.g.profile_hook))
        except BaseException:pass
        # Native exit was captured and source-bound before installation. Do not
        # execute registered shutdown/finalizer callbacks after a failed check.
        self.exit_native(1)

class NativeImportParent:
    def __init__(self,spec):
        self.spec=json.loads(canonical(spec));self.spec_pin=sha(canonical(self.spec));self.used=False
        self.owner=identity(os.getpid());self.source=Path(__file__);self.source_pin=sha(self.source.read_bytes())
        compiled=compile(self.source.read_bytes(),str(self.source),'exec',dont_inherit=True)
        owner=next(c for c in compiled.co_consts if type(c) is types.CodeType and c.co_name=='NativeImportParent');codes={c.co_name:c for c in owner.co_consts if type(c) is types.CodeType}
        self.methods={n:(v,v.__code__,v.__defaults__,v.__kwdefaults__) for n,v in type(self).__dict__.items() if type(v) is types.FunctionType}
        require(all(fn.__code__==codes[n] for n,(fn,*unused) in self.methods.items()),'Parent code differs from source.')
        self.globals_pin={n:globals()[n] for n in ('sha','utc','require','canonical','parse','identity','environment','directory_id','file_id','read_file','ancestors','fsync_directory','save_once','diagnostics','os','sys','stat','time','signal','subprocess','selectors','Path','LOG_LIMIT','TIMEOUT','FILES','MARKER','FRAME_LIMIT')}
        top={c.co_name:c for c in compiled.co_consts if type(c) is types.CodeType}
        self.helpers={n:(v,v.__code__,v.__defaults__,v.__kwdefaults__) for n,v in self.globals_pin.items() if type(v) is types.FunctionType}
        require(all(fn.__code__==top[n] for n,(fn,*unused) in self.helpers.items()),'Parent helper source differs.')
        s=self.spec
        require(s['purpose']=='native_import_completion_fixture_only' and s['parent'].startswith(PREFIX),'Full actual native route remains held before launch.')
        require(s['protocol_sha256']==self.source_pin and s['dispatcher_sha256']==DISPATCHER,'Protocol or closed dispatcher pin differs.')
        require(type(s['fixture_timeout_seconds']) in (int,float) and 0<s['fixture_timeout_seconds']<=TIMEOUT,'Fixture timeout must be within declared180 seconds.')
        require(s['stdout_limit']==s['stderr_limit']==LOG_LIMIT,'Exact declared parent log ceilings required.')
        self.parent=Path(s['parent']);ancestors(self.parent);self.parent_id=directory_id(self.parent.lstat())
        require(self.parent_id==s['parent_id'] and self.parent_id[2:]==[1000,1000,0o700],'Fixture parent identity differs.')
        _,decl=read_file(self.parent/'declaration.json',100000,CONTRACT);self.decl=json.loads(decl)
        require(s['environment']==dict(self.decl['environment'],TMPDIR=str(self.parent/'scratch')) and len(s['environment'])==15,'Exact fixture environment required.')
        require(s['command']==['/home/dbayha/miniconda3/envs/corl-orig-local/bin/python','-I',str(self.parent/'worker.py'),str(self.parent)],'Exact isolated command required.')
        self.inputs={'worker.py':s['entry_sha256'],'dispatcher.py':DISPATCHER,'protocol.py':self.source_pin,'binding.json':s['binding_sha256'],'declaration.json':CONTRACT,'capability.py':s['capability_sha256'],'capability_config.json':s['capability_config_sha256'],'fixture_control.json':s['control_sha256']}
        self.check_inputs()
    def check_inputs(self):
        require(identity(os.getpid())==self.owner and sha(canonical(self.spec))==self.spec_pin and sha(self.source.read_bytes())==self.source_pin,'Parent source/identity/spec changed.')
        require(all(globals().get(n) is v for n,v in self.globals_pin.items()),'Parent direct global changed.')
        for n,(fn,code,defaults,kw) in self.methods.items():require(n not in self.__dict__ and type(self).__dict__.get(n) is fn and fn.__code__ is code and fn.__defaults__ is defaults and fn.__kwdefaults__ is kw,'Parent live method changed.')
        for fn,code,defaults,kw in self.helpers.values():require(fn.__code__ is code and fn.__defaults__ is defaults and fn.__kwdefaults__ is kw,'Parent helper changed.')
        require(directory_id(self.parent.lstat())==self.parent_id,'Fixture parent replaced.')
        for name,h in self.inputs.items():read_file(self.parent/name,2_000_000,h)
    def observe(self,pid,first=None):
        self.check_inputs();observed=dict(identity=identity(pid),environment=environment(pid));i=observed['identity']
        require(i['pid']==i['session']==i['group']==pid and i['parent_pid']==self.owner['pid'],'Private worker relationship differs.')
        require(bytes.fromhex(i['argv_hex'])==b'\0'.join(x.encode() for x in self.spec['command'])+b'\0' and observed['environment']==self.spec['environment'],'Actual worker command/environment differs.')
        if first is not None:require(observed==first,'Worker identity changed before completion acknowledgment.')
        return observed
    def run(self):
        require(not self.used,'Parent dispatch is single-use.');self.used=True;self.check_inputs()
        require(not any(os.path.lexists(self.parent/n) for n in ('scratch','diagnostics')),'Worker roots must be absent before launch.')
        root=self.parent/'supervision';root.mkdir(mode=0o700);root_id=directory_id(root.lstat());fsync_directory(self.parent)
        started=utc();p=None;first=None;second=None;released=False;ack=False;acked=False;failure=None;signals=[];logs={};counts={'stdout':0,'stderr':0};packets=[];overflow=None;pending=None;checked=None;actual=None
        try:
            save_once(root/'launch_intent.json',dict(utc=started,spec=self.spec,source_sha256=self.source_pin,parent=self.owner,declared_timeout_seconds=TIMEOUT,actual_fixture_timeout_seconds=self.spec['fixture_timeout_seconds'],worker_roots_absent=True,production_dispatch=False))
            for name in counts:
                fd=os.open(str(root/name),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600);logs[name]=(fd,file_id(os.fstat(fd)))
            fsync_directory(root);p=subprocess.Popen(self.spec['command'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=self.spec['environment'],start_new_session=True,close_fds=True,bufsize=0)
            deadline=time.monotonic()+self.spec['fixture_timeout_seconds'];sel=selectors.DefaultSelector();buffer=b''
            for name,stream in (('stdout',p.stdout),('stderr',p.stderr)):sel.register(stream,selectors.EVENT_READ,name)
            while sel.get_map():
                require(time.monotonic()<deadline,'Worker protocol timeout.');self.check_inputs()
                require(directory_id(root.lstat())==root_id,'Supervisor root replaced.')
                for key,unused in sel.select(.05):
                    name=key.data;data=os.read(key.fileobj.fileno(),65536)
                    if not data:sel.unregister(key.fileobj);continue
                    remain=LOG_LIMIT-counts[name];keep=data[:remain];fd,pin=logs[name]
                    current=file_id(os.fstat(fd));named=file_id((root/name).lstat());require(current==named and current[:6]==pin[:6] and current[6]==counts[name],'Parent log identity/length changed.')
                    off=0
                    while off<len(keep):
                        n=os.write(fd,keep[off:]);require(n>0,'Parent log write failed.');off+=n
                    counts[name]+=len(keep)
                    if len(data)>remain:
                        overflow=dict(stream=name,retained_bytes=counts[name],overflow_chunk_bytes=len(data)-remain,overflow_chunk_sha256=sha(data[remain:]),unread_pipe_bytes_unavailable=True);raise ValueError('Worker log byte ceiling exceeded.')
                    if name!='stdout':continue
                    buffer+=data
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1)
                        if not line.startswith(MARKER):continue
                        packet=parse(line[len(MARKER):]+b'\n');packets.append(packet);stage=packet.get('stage')
                        if stage=='ready':
                            require(first is None and not released,'Repeated readiness.');first=self.observe(p.pid)
                            expected=dict(stage='ready',**first,protocol_sha256=self.source_pin,entry_sha256=self.spec['entry_sha256'],dispatcher_sha256=DISPATCHER)
                            require(packet==expected,'Readiness packet differs.')
                            save_once(root/'observed_before_release.json',dict(utc=utc(),observation=first))
                            release=dict(stage='release',**first,parent=self.owner);save_once(root/'release_intent.json',release)
                            p.stdin.write(canonical(release));p.stdin.flush();released=True
                        elif stage=='completion_pending':
                            require(released and not ack and pending is None,'Premature or repeated completion.');pending=packet
                            require(packet['identity']==first['identity'] and packet['environment']==self.spec['environment'] and packet['protocol_sha256']==self.source_pin and packet['entry_sha256']==self.spec['entry_sha256'] and packet['dispatcher_sha256']==DISPATCHER and packet['profile_present'] is True and packet['actual_production_profile_executed'] is packet['scientific_execution_accepted'] is False,'Completion binding differs.')
                            second=self.observe(p.pid,first);checked=diagnostics(self.parent,packet['root_ids']);require(checked==packet['diagnostics'],'Parent diagnostic readback differs.')
                            _,final_raw=read_file(self.parent/'diagnostics/final.json',LOG_LIMIT);final=json.loads(final_raw)
                            require(final['worker']==first['identity'] and final['profile_present'] is True and final['full_route_implemented'] is final['actual_production_profile_executed'] is final['scientific_execution_accepted'] is False,'Final diagnostic completion fields differ.')
                            save_once(root/'observed_before_completion_ack.json',dict(utc=utc(),observation=second,diagnostics=checked,completion_sha256=sha(canonical(packet))))
                            self.observe(p.pid,first)
                            response=dict(stage='completion_ack',identity=first['identity'],parent=self.owner,completion_sha256=sha(canonical(packet)),diagnostics=checked)
                            save_once(root/'completion_ack_intent.json',response);p.stdin.write(canonical(response));p.stdin.flush();ack=True
                        elif stage=='completion_ack_received':
                            require(ack and not acked and packet==dict(stage='completion_ack_received',identity=first['identity'],completion_sha256=sha(canonical(pending)),profile_present=True),'Worker acknowledgment receipt differs.');acked=True
                        elif stage=='worker_failure':raise ValueError('Worker reported retained failure.')
                        else:raise ValueError('Unknown protocol stage.')
                    if len(buffer)>FRAME_LIMIT:
                        require(not buffer.startswith(MARKER),'Oversized protocol line.');buffer=b''
            actual=p.wait(timeout=max(.01,deadline-time.monotonic()))
            require(actual==0 and released and ack and acked,'Worker exit lacks accepted completion handshake.')
            require(diagnostics(self.parent,pending['root_ids'])==checked,'Diagnostics changed after worker exit.');self.check_inputs()
        except BaseException as e:failure=type(e).__name__+': '+str(e)
        finally:
            # A reported failure is followed by the worker's pinned os._exit.
            # Reap that actual child before attempting an unnecessary signal.
            if p is not None and p.poll() is None and packets and packets[-1].get('stage')=='worker_failure':
                try:p.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
            if p is not None and p.poll() is None:
                try:
                    current=dict(identity=identity(p.pid),environment=environment(p.pid));i=current['identity']
                    require(i['pid']==i['session']==i['group']==p.pid and i['parent_pid']==self.owner['pid'] and (first is None or current==first),'Refusing stop of changed child identity.')
                    os.killpg(p.pid,signal.SIGTERM);signals.append(15)
                    try:p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        require(dict(identity=identity(p.pid),environment=environment(p.pid))==current,'Worker identity changed before forced stop.');os.killpg(p.pid,signal.SIGKILL);signals.append(9);p.wait(timeout=2)
                except (ProcessLookupError,FileNotFoundError):p.wait(timeout=2)
                except BaseException as e:
                    failure=(failure or '')+'; stop unavailable: '+type(e).__name__
            if p is not None:
                if p.poll() is None:
                    # Waiting on the Popen child is safe even when /proc access
                    # was unavailable; do not fabricate its exit or signal an
                    # identity which failed verification.
                    try:p.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
                actual=p.poll()
                for stream in (p.stdin,p.stdout,p.stderr):
                    if stream is not None:stream.close()
            for fd,unused in logs.values():os.fsync(fd);os.close(fd)
            fsync_directory(root)
            receipt=dict(schema='native-import-parent-fixture-exit-v1',started=started,ended=utc(),command=self.spec['command'],source_sha256=self.source_pin,entry_sha256=self.spec['entry_sha256'],worker_pid=None if p is None else p.pid,actual_child_exit=actual,release_sent=released,completion_ack_sent=ack,worker_ack_received=acked,observed_before_release=first,observed_before_completion_ack=second,completed=failure is None,failure=failure,signals_sent=signals,log_bytes=counts,overflow=overflow,packets=packets,diagnostics_readback=checked,production_dispatch=False,scientific_execution_accepted=False,automatic_retry=False)
            require(directory_id(root.lstat())==root_id,'Supervisor root changed; actual receipt unavailable.');save_once(root/'actual_exit.json',receipt)
        return receipt
