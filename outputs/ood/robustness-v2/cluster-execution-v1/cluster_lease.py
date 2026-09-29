"""Cluster NFS requires a writable descriptor for an exclusive flock.

No writes, replacement or truncation of the shared lock file. All original
lifetime, descriptor identity, PID and attached-ledger checks remain inherited.
"""
import fcntl,os
from ancestor_guard import ExclusiveLease,require,confined,identity

class ClusterLease(ExclusiveLease):
    def __enter__(self):
        require(self.fd is None,'Lease already entered.')
        self.path=confined(self.path);before=identity(self.path)
        fd=os.open(self.path,os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            s=os.fstat(fd)
            require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==before,'Lock changed during acquisition.')
        except BaseException as exc:
            os.close(fd)
            if isinstance(exc,BlockingIOError):raise ValueError('Shared resource lock already held.') from exc
            raise
        self.fd=fd;self.signature=before;self.pid=os.getpid();self.closed=False
        self.assert_held();return self
