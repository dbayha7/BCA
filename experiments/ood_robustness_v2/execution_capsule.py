"""Read-only, single-use engineering bootstrap declaration and review binding.

Trusted review pins must come from an independent accepted observer. Hashes and
an exit receipt do not themselves establish that observer's provenance or the
truth of semantic gates. No defaults name real resources; no process, lock,
ledger, scientific module, model or simulator is created by this component.
"""
import copy
import datetime
import hashlib
import os
from pathlib import Path
import stat
from precommit_file import canonical,read_bound
from extension_ledger import PLAN,maximum_caps
from ancestor_guard import no_sidecars

ROLES={'plan','training_acceptance','host_checkpoint','bca_checkpoint','prepared_data',
       'support_bank','stream_acceptance','ancestor_binding','runtime_graph',
       'native_schema','storage_acceptance','supervisor_acceptance','entrypoint','interpreter'}
GATES={'training_pair','checkpoint_data_support','fresh_streams','source_runtime_graph',
       'native_factory_imports_wrappers','native_restore_schema','native_action_bounds',
       'ancestor_caps_shared_lease','storage_capacity_durability','supervisor_only_route'}
CPU_ENV={'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
INPUT_MAX=2*1024**3


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def hash_value(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'External SHA256 required.')
    return value


def path_value(value):
    require(type(value) is str,'Exact path string required.')
    p=Path(value);require(p.is_absolute() and str(p)==value and '..' not in p.parts,'Absolute canonical path required.')
    require(p.resolve(strict=False)==p,'Redirected path refused.')
    return p


def identity(path):
    p=path_value(str(path));s=p.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Unaliased regular file required.')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def pin_file(pin,*,limit=INPUT_MAX):
    require(type(pin) is dict and set(pin)=={'path','sha256','identity'},'Exact external file pin required.')
    hash_value(pin['sha256'])
    ids=pin['identity'];require(type(ids) is list and len(ids)==5 and all(type(x) is int and x>=0 for x in ids),'Exact file stat identity required.')
    p=path_value(pin['path']);before=identity(p)
    require(before==ids and 0<before[2]<=limit,'File identity/size mismatch.')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();count=0
    with os.fdopen(fd,'rb') as f:
        def fdid():
            s=os.fstat(f.fileno());return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
        require(fdid()==before,'Opened file replaced.')
        while True:
            b=f.read(1024*1024)
            if not b:break
            count+=len(b);require(count<=limit,'File grew beyond bound.');h.update(b)
        require(fdid()==before,'File changed while reading.')
    require(identity(p)==before and count==before[2] and h.hexdigest()==pin['sha256'],'File bytes/path changed.')
    return copy.deepcopy(pin)


def small_json(pin):
    pin_file(pin,limit=2_000_000)
    value=read_bound(pin['path'],pin['sha256'])
    require(identity(pin['path'])==pin['identity'],'JSON file changed during parsing.')
    return value


def directory_identity(value):
    p=path_value(value);s=p.lstat();require(stat.S_ISDIR(s.st_mode),'Existing unaliased directory required.')
    return [s.st_dev,s.st_ino]


def output_paths(paths):
    require(type(paths) is dict and set(paths)=={'parent','run_root','extension','archive','dispatch','lock','ancestor'},'Exact path roles required.')
    parsed={k:path_value(v) for k,v in paths.items()}
    require(len(set(parsed.values()))==len(parsed),'Aliased path roles.')
    parent=parsed['parent'];directory_identity(str(parent))
    root=parsed['run_root'];require(root.parent==parent,'Fresh run must be direct child of accepted parent.')
    require(not root.exists() and not root.is_symlink(),'Existing run retained; no execution reopen.')
    for k in ('extension','archive','dispatch'):
        require(parsed[k].parent==root,'Output is outside declared fresh run.')
        require(not parsed[k].exists() and not parsed[k].is_symlink(),'Existing output retained.')
    for k in ('lock','ancestor'):
        require(not parsed[k].is_relative_to(root) and parsed[k]!=parent,'Shared lock/ancestor cannot be a new output.')
    return parsed


def validate_spec(spec):
    require(type(spec) is dict and set(spec)=={'schema','phase','plan_sha256','pair','assets','paths','command','cwd','environment','caps','combined_global_cap','one_worker','automatic_retry'},'Exact execution declaration fields required.')
    require(spec['schema']=='ood-v2-engineering-capsule-v1' and spec['phase']=='engineering','Only engineering bootstrap supported; science needs separate acceptance.')
    require(spec['plan_sha256']==PLAN and spec['one_worker'] is True and spec['automatic_retry'] is False,'Frozen protocol/worker/retry policy changed.')
    pair=spec['pair'];require(type(pair) is list and len(pair)==3 and pair[0] in ('td3_bc','rebrac') and pair[1] in ('hopper','walker2d')
        and type(pair[2]) is int and pair[2] in range(202609171,202609176),'Undeclared pair.')
    require(type(spec['assets']) is dict and set(spec['assets'])==ROLES,'Missing or extra bound asset role.')
    require(spec['assets']['plan']['sha256']==PLAN,'Wrong original scientific plan bytes.')
    require(canonical(spec['caps'])==canonical(maximum_caps()) and spec['combined_global_cap']==[39998400,159993600]
        and all(type(x) is int for x in spec['combined_global_cap']),'Frozen scope/global caps changed.')
    require(type(spec['environment']) is dict and spec['environment']==CPU_ENV,'Exact CPU environment required; no inherited GPU overrides.')
    expected=[spec['assets']['interpreter']['path'],'-I',spec['assets']['entrypoint']['path'],'--execution-declaration']
    cmd=spec['command'];require(type(cmd) is list and len(cmd)==5 and cmd[:4]==expected and all(type(x) is str for x in cmd),'Exact isolated Python entrypoint argv required.')
    require(spec['cwd']==spec['paths']['parent'],'Working directory must be the declared parent.')
    return output_paths(spec['paths'])


class EngineeringCapsule:
    """Pins relationships; it neither runs the reviewer nor grants science.

    The external caller must independently accept the observer/reviewer, runtime
    graph and only-route supervisor. Synthetic receipts cannot authorize real use.
    take_bootstrap is single-use and happens before any target creation. Subsequent
    code must independently hold the shared lease and use the closed ledger/archive.
    """
    def __init__(self,*,declaration_pin,trusted_review):
        self.failed=False;self.consumed=False;self.pid=os.getpid()
        self.declaration_pin=copy.deepcopy(declaration_pin);self.trusted_review=copy.deepcopy(trusted_review)
        try:
            self.spec=small_json(self.declaration_pin);paths=validate_spec(self.spec)
            require(self.spec['command'][-1]==self.declaration_pin['path'],'Worker command binds a different declaration.')
            self.parent_identity=directory_identity(self.spec['paths']['parent'])
            self.pins={'declaration':self.declaration_pin}
            for role,pin in self.spec['assets'].items():self.pins['asset/'+role]=pin_file(pin)
            require(len({p['path'] for p in self.pins.values()})==len(self.pins),'Assets/declaration have duplicate paths.')
            require(len({tuple(p['identity'][:2]) for p in self.pins.values()})==len(self.pins),'Assets/declaration have duplicate inodes.')
            for pin in self.pins.values():
                p=Path(pin['path']);require(not p.is_relative_to(paths['run_root']) and p not in (paths['lock'],paths['ancestor']),'Input aliases mutable output/lock/ancestor.')
            self.ancestor=small_json(self.spec['assets']['ancestor_binding'])
            require(set(self.ancestor)=={'path','identity','header_sha256','entries','last_sha256','reserved','audit_receipt_sha256'}
                and self.ancestor['path']==self.spec['paths']['ancestor'],'Original ancestor binding/path differs.')
            for k in ('header_sha256','last_sha256','audit_receipt_sha256'):hash_value(self.ancestor[k])
            old=self.ancestor['reserved']['global']
            require(type(old) is list and len(old)==2 and all(type(n) is int and n>=0 for n in old)
                and old[1]==4*old[0] and all(a+b<=c for a,b,c in zip(old,self.spec['caps']['global'],self.spec['combined_global_cap'])),'Combined ancestor/new maximum exceeds original cap.')
            require(identity(paths['ancestor'])==self.ancestor['identity'],'Original ancestor file changed.');no_sidecars(paths['ancestor'])
            self._bind_review()
            self._contract=digest([self.spec,self.declaration_pin,self.trusted_review,self.pins,self.parent_identity,self.ancestor])
            self.assert_unchanged()
        except BaseException:self.failed=True;raise

    def _bind_review(self):
        tr=self.trusted_review
        require(type(tr) is dict and set(tr)=={'report','actual_exit','reviewer_source','interpreter','command'},'Independent observer pins required.')
        source=pin_file(tr['reviewer_source'],limit=2_000_000);interpreter=pin_file(tr['interpreter'])
        report=small_json(tr['report']);actual=small_json(tr['actual_exit'])
        expected=[interpreter['path'],'-I',source['path'],'--execution-declaration',self.declaration_pin['path'],'--output',tr['report']['path']]
        require(tr['command']==expected,'Reviewer actual command differs from externally bound source/inputs/output.')
        require(set(report)=={'schema','declaration_sha256','phase','assets','gates','reviewer_source_sha256'}
            and report['schema']=='ood-v2-engineering-independent-review-v1' and report['phase']=='engineering'
            and report['declaration_sha256']==self.declaration_pin['sha256'] and report['reviewer_source_sha256']==source['sha256'], 'Review report declaration/source binding failed.')
        require(report['assets']=={k:v['sha256'] for k,v in self.spec['assets'].items()},'Review omitted or changed declared asset.')
        require(type(report['gates']) is dict and set(report['gates'])==GATES and all(v is True for v in report['gates'].values()),'Independent engineering gate missing, held or refused.')
        require(set(actual)=={'schema','command','actual_returncode','timeout','interruption_signal','started','ended','pid','declaration_sha256','reviewer_source_sha256','report_sha256'}
            and actual['schema']=='ood-v2-independent-review-process-v1','Independent actual process receipt required.')
        require(actual['command']==expected and type(actual['actual_returncode']) is int and actual['actual_returncode']==0
            and actual['timeout'] is False and actual['interruption_signal'] is None,'Review did not actually close successfully.')
        require(type(actual['pid']) is int and actual['pid']>0 and actual['pid']!=self.pid,'Independent reviewer process identity required.')
        require(actual['declaration_sha256']==self.declaration_pin['sha256'] and actual['reviewer_source_sha256']==source['sha256']
            and actual['report_sha256']==tr['report']['sha256'],'Actual process receipt is stale or belongs to another review.')
        start,end=(datetime.datetime.fromisoformat(actual[k]) for k in ('started','ended'))
        require(start.utcoffset() is not None and end.utcoffset() is not None and start<=end<=datetime.datetime.now(datetime.timezone.utc),'Invalid review time ordering.')
        for k in ('report','actual_exit','reviewer_source','interpreter'):self.pins['review/'+k]=copy.deepcopy(tr[k])
        seen={}
        for role,pin in self.pins.items():
            path=Path(pin['path'])
            require(not path.is_relative_to(Path(self.spec['paths']['run_root']))
                and str(path) not in (self.spec['paths']['lock'],self.spec['paths']['ancestor']),
                'Review/input aliases a mutable resource or future output.')
            key=tuple(pin['identity'][:2])
            if key in seen:
                previous=seen[key]
                require({previous,role}=={'asset/interpreter','review/interpreter'} and self.pins[previous]==pin,
                    'Unapproved shared review/input file identity.')
            else:seen[key]=role

    def assert_unchanged(self):
        try:
            require(not self.failed and not self.consumed and os.getpid()==self.pid,'Failed, consumed or forked capsule.')
            require(digest([self.spec,self.declaration_pin,self.trusted_review,self.pins,self.parent_identity,self.ancestor])==self._contract,'Capsule contract mutated.')
            require(directory_identity(self.spec['paths']['parent'])==self.parent_identity,'Output parent replaced.')
            validate_spec(self.spec)
            for p in self.pins.values():require(identity(p['path'])==p['identity'],'Pinned asset/review changed.')
            require(identity(self.ancestor['path'])==self.ancestor['identity'],'Original ancestor changed.');no_sidecars(self.ancestor['path'])
        except BaseException:self.failed=True;raise

    def take_bootstrap(self):
        try:
            self.assert_unchanged();self.consumed=True
            return dict(schema='ood-v2-engineering-bootstrap-binding-v1',declaration_sha256=self.declaration_pin['sha256'],
                pair=copy.deepcopy(self.spec['pair']),command=copy.deepcopy(self.spec['command']),cwd=self.spec['cwd'],
                environment=copy.deepcopy(self.spec['environment']),paths=copy.deepcopy(self.spec['paths']),
                permitted_scope='engineering/'+self.spec['pair'][0]+'/'+self.spec['pair'][1],
                review_sha256=self.trusted_review['report']['sha256'],review_actual_exit_sha256=self.trusted_review['actual_exit']['sha256'],
                independently_trusted_observer_required=True,actual_supervisor_dispatched=False,
                shared_lease_acquired=False,extension_created=False,physical_calls_added=0,science_accepted=False)
        except BaseException:self.failed=True;raise
