"""Read-only pinned ancestor metadata and lifetime Linux shared-resource lease.

This is a component, not execution acceptance. No default paths, writer imports,
database repair, data/model/simulator access, or original-ledger rescan.
"""
from pathlib import Path
import fcntl,hashlib,json,math,os,re,sqlite3,stat


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def is_hash(v): return type(v) is str and re.fullmatch('[0-9a-f]{64}',v) is not None


def parse(raw):
    require(type(raw) is str and 0<len(raw.encode())<=65536,'Unbounded journal JSON.')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate journal key.');out[k]=v
        return out
    def bad(_): raise ValueError('Nonfinite journal JSON.')
    def number(v):
        x=float(v);require(math.isfinite(x),'Nonfinite journal JSON.');return x
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=number)
    require(type(value) is dict,'Expected journal object.')
    return value


def confined(path, *, exists=True):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts,'Expected absolute unredirected path.')
    require(path.parent.resolve(strict=True)==path.parent,'Redirected parent.')
    if exists:
        require(path.resolve(strict=True)==path,'Redirected file.')
        s=path.lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Expected unaliased regular file.')
    else:
        require(not path.exists() and not path.is_symlink(),'Existing target must be preserved.')
    return path


def identity(path):
    path=confined(path);s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def no_sidecars(path):
    for suffix in ('-journal','-wal','-shm'):
        side=Path(str(path)+suffix)
        require(not side.exists() and not side.is_symlink(),'Uncertain SQLite sidecar; no automatic recovery.')


class AncestorGuard:
    """Bind small metadata to a prior full independent audit and exact file stat."""
    def __init__(self, binding):
        self.binding=json.loads(canonical(binding))
        self._binding_digest=digest(self.binding)
        b=self.binding
        require(set(b)=={'path','identity','header_sha256','entries','last_sha256','reserved','audit_receipt_sha256'},'Invalid ancestor binding.')
        require(all(is_hash(b[k]) for k in ('header_sha256','last_sha256','audit_receipt_sha256')),'External ancestor hashes required.')
        require(type(b['entries']) is int and b['entries']>0 and type(b['identity']) is list
                and len(b['identity'])==5 and all(type(n) is int and n>=0 for n in b['identity']),'Invalid ancestor counts/identity.')
        totals=b['reserved']
        require(type(totals) is dict and 1<=len(totals)<=1000 and 'global' in totals
                and all(type(k) is str and type(v) is list and len(v)==2
                        and all(type(n) is int and n>=0 for n in v) for k,v in totals.items()),'Invalid ancestor totals.')
        self.path=confined(b['path']);self.unchanged()
        db=None
        try:
            db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,isolation_level=None)
            db.execute('PRAGMA query_only=ON');db.execute('PRAGMA trusted_schema=OFF');db.execute('BEGIN')
            require(db.execute('PRAGMA journal_mode').fetchone()==('delete',),'Unexpected ancestor journal mode.')
            for table in ('header','entries','totals'):
                require(db.execute('SELECT type FROM sqlite_master WHERE name=?',(table,)).fetchone()==('table',),'Ancestor table missing or redirected.')
            rows=db.execute('SELECT id,length(payload) FROM header').fetchall()
            require(len(rows)==1 and rows[0][0]==1 and 0<rows[0][1]<=65536,'Invalid ancestor header size.')
            header=parse(db.execute('SELECT payload FROM header WHERE id=1').fetchone()[0])
            require(digest(header)==b['header_sha256'],'Changed ancestor declaration/caps.')
            require(header['schema']=='ood-resource-ledger-v2'
                    and header['caps']['global']==[39998400,159993600],'Wrong original global ceiling.')
            self.global_cap=header['caps']['global'][:]
            require(db.execute('SELECT id,digest FROM entries ORDER BY id DESC LIMIT 1').fetchone()
                    ==(b['entries']-1,b['last_sha256']),'Changed ancestor reservation head.')
            require(db.execute('SELECT count(*) FROM totals').fetchone()[0]==len(totals),'Changed ancestor scope count.')
            require({s:[e,p] for s,e,p in db.execute('SELECT scope,environment,physics FROM totals')}==totals,'Changed ancestor cached totals.')
            require(all(n<=c for n,c in zip(totals['global'],self.global_cap)),'Ancestor already exceeds cap.')
        except sqlite3.DatabaseError as exc:
            raise ValueError('Read-only ancestor check failed without repair.') from exc
        finally:
            if db is not None: db.close()
        self.unchanged()

    def unchanged(self):
        require(digest(self.binding)==self._binding_digest,'Ancestor binding mutated after validation.')
        if hasattr(self,'global_cap'):
            require(self.global_cap==[39998400,159993600],'Original global cap mutated.')
        require(identity(self.path)==self.binding['identity'],'Original ledger changed; all new execution refused.')
        no_sidecars(self.path)


class ExclusiveLease:
    """Existing lock file only. Do not fork, replace or close while owning a journal."""
    def __init__(self,path):
        self.path=Path(path);self.fd=None;self.closed=True;self.attached=None

    def __enter__(self):
        require(self.fd is None,'Lease already entered.')
        self.path=confined(self.path);before=identity(self.path)
        fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            s=os.fstat(fd);require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==before,'Lock changed during acquisition.')
        except BaseException as exc:
            os.close(fd)
            if isinstance(exc,BlockingIOError): raise ValueError('Shared resource lock already held.') from exc
            raise
        self.fd=fd;self.signature=before;self.pid=os.getpid();self.closed=False
        self.assert_held();return self

    def assert_held(self):
        require(not self.closed and self.fd is not None and self.pid==os.getpid(),'Absent, closed or forked resource lease.')
        require(identity(self.path)==self.signature,'Shared resource lock replaced or changed.')
        s=os.fstat(self.fd)
        require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==self.signature,'Lease descriptor identity changed.')

    def attach(self,owner):
        self.assert_held();require(self.attached is None,'One extension owner per lease.');self.attached=owner

    def detach(self,owner):
        if self.attached is owner: self.attached=None

    def __exit__(self,*_):
        if self.fd is not None:
            fcntl.flock(self.fd,fcntl.LOCK_UN);os.close(self.fd)
        self.fd=None;self.closed=True
