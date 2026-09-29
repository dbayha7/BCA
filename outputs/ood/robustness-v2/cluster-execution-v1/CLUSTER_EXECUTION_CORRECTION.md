# Preserved cluster startup failure and separate correction

Job 27067667 exited 1 at 2026-09-29T06:54:01.426419Z on str-c167.
`fcntl.flock` rejected the original read-only descriptor with EBADF on the
cluster's shared filesystem. This happened before extension-ledger creation,
checkpoint loading, environment construction, or any simulator transition.
The original worker, submitted files, logs and actual exit remain immutable at
`/users/dbayha/bca-ood-v2/rebrac-hopper171-ood-v1`.

The separate v2 attempt uses an O_RDWR descriptor for the existing cluster lock.
It never writes, truncates or replaces the lock. All original identity, owner
PID, lifetime and attached-ledger checks are retained. A real second process
must be rejected while held and succeed after release, with lock identity
unchanged, before submission. No scientific source, checkpoint, seed, candidate,
normalization, loss, radius, horizon or acceptance tolerance changes.

Earlier dispatch preparer v1 also failed a Python string-quoting check before
remote staging; its failure and output remain. Dispatch v2 fixed only that
preparer and successfully submitted the now-preserved cluster v1 job. The next
dispatch is v3, targeting the separately named cluster v2 worker attempt.
