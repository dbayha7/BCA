"""Separate durable v2 reservations; original ledger is never opened writable.

Only synthetic tests have accepted this component. Real binding, execution and
scientific acceptance are separate gates. An unresolved reservation stops all
further calls and cannot be refunded/retried through this interface.
"""
from pathlib import Path
import datetime,json,os,sqlite3
from ancestor_guard import require,canonical,digest,is_hash,parse,confined,identity,no_sidecars

PLAN='f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756'


def maximum_caps():
    caps={'global':[36791360,147165440]}
    for host in ('td3_bc','rebrac'):
        for env in ('hopper','walker2d'):
            caps[f'engineering/{host}/{env}']=[10000,40000]
            for seed in range(202609171,202609176):
                for kind,n in [('collection',38400),('outcomes',1792000),('repeat_checks',7168)]:
                    caps[f'{kind}/{host}/{env}/s{seed}']=[n,n*4]
    return caps


def declaration(path,lock_path,ancestor,execution_declaration_sha256):
    return dict(schema='ood-robustness-extension-ledger-v1',path=str(path),lock_path=str(lock_path),
                ancestor=json.loads(canonical(ancestor)),plan_sha256=PLAN,
                execution_declaration_sha256=execution_declaration_sha256,caps=maximum_caps(),
                combined_global_cap=[39998400,159993600],counts='reservations; completion receipts are not independent scientific acceptance')


def validate_declaration(spec,path,guard,lease):
    require(set(spec)==set(declaration(path,lease.path,guard.binding,'0'*64)),'Invalid extension declaration fields.')
    require(spec['schema']=='ood-robustness-extension-ledger-v1' and spec['path']==str(path)
            and spec['lock_path']==str(lease.path) and spec['ancestor']==guard.binding
            and spec['plan_sha256']==PLAN and is_hash(spec['execution_declaration_sha256'])
            and spec['combined_global_cap']==guard.global_cap
            and spec['counts']=='reservations; completion receipts are not independent scientific acceptance','Wrong external extension binding.')
    caps=spec['caps'];maximum=maximum_caps()
    require(type(caps) is dict and set(caps)==set(maximum),'Wrong declared scope structure.')
    for key,values in caps.items():
        require(type(values) is list and len(values)==2 and all(type(n) is int and n>=0 for n in values)
                and values[1]==4*values[0] and all(n<=c for n,c in zip(values,maximum[key])),'Expanded or invalid v2 scope cap.')
    require(all(a+b<=c for a,b,c in zip(guard.binding['reserved']['global'],caps['global'],guard.global_cap)),'Combined maximum exceeds original global ceiling.')


def cap_check(totals,caps,old_global,combined_cap,scope):
    require(scope in caps and scope!='global','Unknown physical-call scope.')
    require(set(totals)==set(caps),'Wrong cumulative scope set.')
    for key in ('global',scope):
        require(all(type(n) is int and n>=0 for n in totals[key]),'Invalid cached count.')
        require(all(n+a<=c for n,a,c in zip(totals[key],[1,4],caps[key])),'V2 reservation ceiling reached: '+key)
    require(all(old+n+a<=c for old,n,a,c in zip(old_global,totals['global'],[1,4],combined_cap)),'Combined original/v2 global ceiling reached.')


TABLES={
 'header':'CREATE TABLE header(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL)',
 'reservations':'CREATE TABLE reservations(id INTEGER PRIMARY KEY,token TEXT UNIQUE NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL)',
 'totals':'CREATE TABLE totals(scope TEXT PRIMARY KEY,environment INTEGER NOT NULL,physics INTEGER NOT NULL)',
 'completions':'CREATE TABLE completions(token TEXT PRIMARY KEY REFERENCES reservations(token),evidence_sha256 TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL)',
}
TRIGGERS={f'{table}_{action.lower()}':f"CREATE TRIGGER {table}_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'immutable v2 journal'); END"
          for table in ('header','reservations','completions') for action in ('UPDATE','DELETE')}


class ExtensionLedger:
    @classmethod
    def create(cls,path,guard,lease,spec): return cls(path,guard,lease,spec,create=True)

    @classmethod
    def open(cls,path,guard,lease,spec): return cls(path,guard,lease,spec,create=False)

    def __init__(self,path,guard,lease,spec,*,create):
        self.path=Path(path);self.guard=guard;self.lease=lease
        self.header=json.loads(canonical(spec));self.db=None;self.closed=True
        self._header_digest=digest(self.header)
        lease.assert_held();guard.unchanged();validate_declaration(self.header,self.path,guard,lease)
        require(self.path!=guard.path and self.path!=lease.path,'Extension must be separate from ancestor/lock.')
        confined(self.path,exists=not create);no_sidecars(self.path)
        lease.attach(self)
        try:
            if create:
                fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600);os.close(fd)
                directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(directory)
                finally: os.close(directory)
            initial_identity=identity(self.path)
            self.db=sqlite3.connect(self.path.as_uri()+'?mode=rw',uri=True,timeout=0,isolation_level=None)
            require(identity(self.path)==initial_identity,'Extension replaced during open.')
            self.db.execute('PRAGMA trusted_schema=OFF');self.db.execute('PRAGMA foreign_keys=ON')
            self.db.execute('PRAGMA synchronous=FULL')
            require(self.db.execute('PRAGMA journal_mode').fetchone()==('delete',),'Unexpected extension journal mode; no conversion.')
            if create:
                self.db.execute('BEGIN IMMEDIATE')
                for sql in TABLES.values(): self.db.execute(sql)
                for sql in TRIGGERS.values(): self.db.execute(sql)
                self.db.execute('INSERT INTO header VALUES(1,?)',(canonical(self.header).decode(),))
                self.db.executemany('INSERT INTO totals VALUES(?,0,0)',[(k,) for k in self.header['caps']])
                self.db.execute('COMMIT')
            self.closed=False;self.current_identity=identity(self.path)
            require(self.current_identity[:2]==initial_identity[:2],'Extension replaced during initialization.')
            report=self.audit()
            require(not report['pending'],'Unresolved reservation: completion unknown, separately accepted recovery required.')
        except BaseException:
            if self.db is not None:
                if self.db.in_transaction: self.db.execute('ROLLBACK')
                self.db.close()
            self.db=None;self.closed=True;lease.detach(self);raise

    def __enter__(self): return self

    def __exit__(self,*_):
        if self.db is not None: self.db.close()
        self.db=None;self.closed=True;self.lease.detach(self)

    def _unchanged(self):
        require(not self.closed and self.db is not None,'Closed extension journal.')
        require(digest(self.header)==self._header_digest,'Extension declaration mutated after validation.')
        self.lease.assert_held();self.guard.unchanged()
        require(identity(self.path)==self.current_identity,'Extension journal changed outside its owner.')
        no_sidecars(self.path)

    def _accept_commit(self):
        observed=identity(self.path)
        require(observed[:2]==self.current_identity[:2],'Extension file replaced during transaction.')
        self.current_identity=observed

    def _head(self):
        row=self.db.execute('SELECT id,payload,digest FROM reservations ORDER BY id DESC LIMIT 1').fetchone()
        if row is None: return 0,digest(self.header)
        v=parse(row[1]);require(v['index']==row[0] and digest(v)==row[2],'Changed reservation head.')
        return row[0]+1,row[2]

    def _totals(self):
        return {s:[e,p] for s,e,p in self.db.execute('SELECT scope,environment,physics FROM totals')}

    def _pending(self):
        # The full audit establishes that only the tail can be unresolved.
        # Each append preserves that invariant; avoid rescanning history per call.
        return [r[0] for r in self.db.execute('SELECT r.token FROM reservations r WHERE r.id=(SELECT max(id) FROM reservations) AND NOT EXISTS(SELECT 1 FROM completions c WHERE c.token=r.token)')]

    def audit(self):
        self._unchanged();self.db.execute('BEGIN')
        try:
            require(self.db.execute('PRAGMA quick_check').fetchall()==[('ok',)],'Damaged extension journal.')
            schema={n:sql for n,sql in self.db.execute("SELECT name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
            require(schema=={**TABLES,**TRIGGERS},'Changed extension schema/triggers.')
            header=self.db.execute('SELECT id,payload FROM header').fetchall()
            require(header==[(1,canonical(self.header).decode())],'Changed immutable extension declaration.')
            totals={k:[0,0] for k in self.header['caps']};head=digest(self.header);count=0
            completed=0;pending=[];last_token=None
            query='SELECT r.id,r.token,r.payload,r.digest,c.evidence_sha256,c.payload,c.digest FROM reservations r LEFT JOIN completions c ON r.token=c.token ORDER BY r.id'
            for index,token,payload,h,evidence,completion_payload,completion_hash in self.db.execute(query):
                v=parse(payload)
                require(set(v)=={'index','previous','token','scope','environment','physics','utc'}
                        and v['index']==index==count and v['previous']==head and v['token']==token
                        and type(token) is str and 0<len(token)<=1024 and digest(v)==h
                        and type(v['environment']) is int and v['environment']==1
                        and type(v['physics']) is int and v['physics']==4,'Broken extension reservation chain.')
                require(datetime.datetime.fromisoformat(v['utc']).utcoffset()==datetime.timedelta(0),'Invalid reservation UTC.')
                cap_check(totals,self.header['caps'],self.header['ancestor']['reserved']['global'],self.header['combined_global_cap'],v['scope'])
                for scope in ('global',v['scope']): totals[scope][0]+=1;totals[scope][1]+=4
                if completion_payload is None:
                    pending.append(token)
                    require(len(pending)<=1,'Multiple unresolved reservations.')
                else:
                    receipt=parse(completion_payload)
                    require(is_hash(evidence) and set(receipt)=={'token','reservation_sha256','evidence_sha256','utc'}
                            and receipt['token']==token and receipt['reservation_sha256']==h
                            and receipt['evidence_sha256']==evidence and digest(receipt)==completion_hash,'Invalid completion receipt binding.')
                    require(datetime.datetime.fromisoformat(receipt['utc']).utcoffset()==datetime.timedelta(0),'Invalid completion UTC.')
                    completed+=1
                head=h;count+=1;last_token=token
            require(totals==self._totals(),'Cached totals differ from reserved history.')
            require(not pending or pending[0]==last_token,'Out-of-order unresolved reservation.')
            require(self.db.execute('SELECT count(*) FROM completions').fetchone()[0]==completed,'Extra completion receipts.')
            require(not self.db.execute('PRAGMA foreign_key_check').fetchall(),'Orphaned completion receipt.')
            result=dict(entries=count,last_sha256=head,reserved=totals,completion_receipts=completed,pending=pending,
                        combined_reserved=[a+b for a,b in zip(self.header['ancestor']['reserved']['global'],totals['global'])],
                        physical_completion_not_inferred=True,scientific_acceptance=False)
        finally: self.db.execute('ROLLBACK')
        self._unchanged();return result

    def reserve(self,token,scope):
        self._unchanged()
        require(type(token) is str and 0<len(token)<=1024 and '\x00' not in token,'Invalid physical call token.')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            require(not self._pending(),'Unresolved previous reservation; no automatic continuation.')
            require(self.db.execute('SELECT 1 FROM reservations WHERE token=?',(token,)).fetchone() is None,'Already reserved token; no retry.')
            cap_check(self._totals(),self.header['caps'],self.header['ancestor']['reserved']['global'],self.header['combined_global_cap'],scope)
            index,previous=self._head()
            v=dict(index=index,previous=previous,token=token,scope=scope,environment=1,physics=4,
                   utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            h=digest(v)
            self.db.execute('INSERT INTO reservations VALUES(?,?,?,?)',(index,token,canonical(v).decode(),h))
            for key in ('global',scope): self.db.execute('UPDATE totals SET environment=environment+1,physics=physics+4 WHERE scope=?',(key,))
            self.guard.unchanged();self.lease.assert_held()
            self.db.execute('COMMIT');self._accept_commit()
        except BaseException:
            if self.db.in_transaction: self.db.execute('ROLLBACK')
            # A failed database operation is not silently accepted for another call.
            raise
        self._unchanged();return h

    def record_completion(self,token,evidence_sha256):
        self._unchanged();require(is_hash(evidence_sha256),'Durable output-evidence hash required; reservation stays pending.')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            require(self._pending()==[token],'Completion does not own the unique pending reservation.')
            h=self.db.execute('SELECT digest FROM reservations WHERE token=?',(token,)).fetchone()[0]
            v=dict(token=token,reservation_sha256=h,evidence_sha256=evidence_sha256,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            self.db.execute('INSERT INTO completions VALUES(?,?,?,?)',(token,evidence_sha256,canonical(v).decode(),digest(v)))
            self.guard.unchanged();self.lease.assert_held()
            self.db.execute('COMMIT');self._accept_commit()
        except BaseException:
            if self.db.in_transaction: self.db.execute('ROLLBACK')
            raise
        self._unchanged()

    def call(self,token,scope,callback):
        self.reserve(token,scope)
        self._unchanged()  # after durable reservation, immediately before physical work
        result,evidence_sha256=callback()
        self.record_completion(token,evidence_sha256)
        return result
