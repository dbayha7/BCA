"""Bounded durable wrapper for the injected frozen native artifact archive.

No model/simulator imports or default paths. Real device/runtime/execution
acceptance is external. Any uncertain append poisons this instance; no reopen,
repair, resume, deletion or retry is provided.
"""
import hashlib
import os
from pathlib import Path
import re
import shutil
from ancestor_guard import require,confined,identity,no_sidecars,is_hash


class RuntimeArchive:
    def __init__(self,*,path,lease,archive_factory,source_path,source_sha256,
                 maximum_file_bytes,minimum_free_bytes):
        self.inner=None;self.closed=True;self._failed=False;self.read_only=False
        self.path=Path(path);self.lease=lease;self.pid=os.getpid()
        require(type(maximum_file_bytes) is int and maximum_file_bytes>=65536
                and type(minimum_free_bytes) is int and minimum_free_bytes>=0,'Explicit archive/storage bounds required.')
        self.maximum_file_bytes=maximum_file_bytes;self.minimum_free_bytes=minimum_free_bytes
        self._limits=(maximum_file_bytes,minimum_free_bytes)
        lease.assert_held();confined(self.path,exists=False);no_sidecars(self.path)
        self.source=confined(source_path);self.source_identity=identity(self.source)
        require(is_hash(source_sha256) and hashlib.sha256(self.source.read_bytes()).hexdigest()==source_sha256,'Wrong frozen archive source bytes.')
        require(identity(self.source)==self.source_identity,'Archive source changed during read.')
        require(isinstance(archive_factory,type) and archive_factory.__name__=='ArtifactArchive','Exact injected archive class required.')
        for name in ('__init__','append','read'):
            method=archive_factory.__dict__.get(name)
            require(callable(method) and Path(method.__code__.co_filename).resolve()==self.source,'Wrong archive class source location.')
        self._space(65536)
        try:
            self.inner=archive_factory(self.path,maximum_record_bytes=2_000_000)
            self.current_identity=identity(self.path);self._mode();no_sidecars(self.path)
            self._sync();self.current_identity=identity(self.path);self.closed=False
        except BaseException:
            self._failed=True
            if self.inner is not None:self.inner.db.close()
            raise

    @property
    def db(self):
        require(self.inner is not None,'Archive not constructed.')
        return self.inner.db

    @property
    def failed(self): return self._failed or (self.inner is not None and self.inner.failed)

    def _space(self,extra):
        require((self.maximum_file_bytes,self.minimum_free_bytes)==self._limits,'Storage declaration mutated.')
        require(shutil.disk_usage(self.path.parent).free>=self.minimum_free_bytes+extra,'Insufficient declared storage floor.')

    def _mode(self):
        require(not self.db.in_transaction,'Unexpected active archive transaction.')
        require(self.db.execute('PRAGMA journal_mode').fetchone()==('delete',)
                and self.db.execute('PRAGMA synchronous').fetchone()==(2,),'Archive durability settings changed.')

    def _unchanged(self):
        require(not self.closed and not self.failed and self.pid==os.getpid(),'Closed/failed/forked archive owner.')
        self.lease.assert_held()
        require(identity(self.source)==self.source_identity,'Pinned archive source changed.')
        require(identity(self.path)==self.current_identity,'Archive changed outside its owner.')
        no_sidecars(self.path);self._mode()

    def _sync(self):
        # SQLite FULL/DELETE is also required. Explicit descriptor and directory
        # fsync preserve the acknowledgement boundary without rewriting payloads.
        before=identity(self.path);fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            st=os.fstat(fd)
            require([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==before,'Archive changed before fsync.')
            os.fsync(fd)
        finally:os.close(fd)
        fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        require(identity(self.path)==before,'Archive changed during fsync.')

    def append(self,name,raw,complete=True):
        try:
            self._unchanged()
            require(type(name) is str and 0<len(name)<=300 and re.fullmatch(r'[A-Za-z0-9_./-]+',name)
                    and not name.startswith('/') and '..' not in name,'Unsafe archive artifact name.')
            require(type(raw) is bytes and len(raw)<=2_000_000 and type(complete) is bool,'Bounded raw artifact required.')
            # Conservative refusal before a record near the cap; this does not
            # promise that the final device can hold the whole future campaign.
            extra=len(raw)+65536
            require(self.current_identity[2]+extra<=self.maximum_file_bytes,'Archive file ceiling reached before append.')
            self._space(extra);previous=self.current_identity
            self.inner.append(name,raw,complete=complete)
            now=identity(self.path)
            require(now[:2]==previous[:2] and now[2]<=self.maximum_file_bytes,'Archive replaced or file ceiling exceeded.')
            no_sidecars(self.path);self._mode();self._sync()
            require(self.inner.read(name,allow_partial=not complete)==raw,'Durable native archive readback differs.')
            require(identity(self.path)==now,'Archive changed during readback.')
            self.current_identity=now
            if not complete:self._failed=True
        except BaseException:
            self._failed=True
            raise

    def read(self,name,allow_partial=False):
        try:
            self._unchanged();raw=self.inner.read(name,allow_partial=allow_partial)
            require(identity(self.path)==self.current_identity,'Archive changed during read.')
            return raw
        except BaseException:
            self._failed=True
            raise

    def close(self):
        if not self.closed and self.inner is not None:self.inner.db.close()
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
