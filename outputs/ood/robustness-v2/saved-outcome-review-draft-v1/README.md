# Saved outcome review: open draft and local execution failure

The local TD3+BC Hopper worker exited **1** at 09:22:11 UTC after a reservation
commit reported `database is locked`. Our new read-only ledger audit overlapped
the failure. DELETE-mode SQLite readers can block the worker's zero-timeout
commit; that mechanism was reproduced in a separate synthetic fixture. No lock
owner trace was captured, but timing and mechanism strongly implicate the audit.
Opening the live ledger was our execution mistake. Live SQLite audits are now
refused before any database open. The original worker/source/data remain intact.

The stopped attempt's complete saved prefix now reconciles **222,854 charged
and acknowledged physical calls, 1,563,709 artifacts, 740 completed trajectories
and 52 complete panel files**. One further trajectory retains 247 ordinary steps
plus its charged exact-first-repeat check; it is not a completed return. There
are no persisted pending reservations. The failed next reservation did not reach
the physical callback. All earlier charges remain; nothing was refunded.

The independent reader checked actual native NPZ payloads against typed records,
controls, full state schemas/continuity, reward arithmetic, first repeats,
reservation and completion hashes, all 256 captures, candidate reconstruction,
support distances, saved key rows and whole-pair precommit ordering. Completed
return records were joined to their raw rewards and full final states. These
checks support saved native arithmetic and accounting; full training/support
and checkpoint/score provenance closure and a reviewed continuation route are
still required. No completed revised pair or BCA benefit is claimed.

The same reader checked a **400,000-frame prefix** of Walker171: 44,349 physical
records, 256 nominal states (248 captured, eight missing at native termination),
248 candidate banks and the start of outcomes. There are 44,348 acknowledgements
inside this bounded prefix; the last physical record's ack is beyond the prefix,
not claimed missing from the live journal. The previous failed constructor
reservation remains separately charged. This is an incomplete prefix, not final
worker acceptance. Walker172/173 also reached outcomes by 09:26 UTC; all three
worker commands/start identities and staged source pins matched. Original cluster
Hopper was still in outcomes at 09:29. Their exits remained pending.

The verifier remains **OPEN**, in the same `audit_saved_outcomes.py` file.
Executed arithmetic reader SHA `368442cf136368a6c76255a3c938d8b4fbb2fdabfccc46a6cd3320afbcaabd02`; current reader with live-SQLite
refusal SHA `d504a4a2915500e8f0624eb7183d6ef068f9713012f1563963aac473c23cb978`. Earlier four numeric fixtures, fourteen new return/alias
fixtures, one live-open refusal and one lock-mechanism fixture passed at their
recorded versions. They are not one combined final suite. Fixture syntax/order
and configuration failures remain alongside all successful attempts. A cluster
review initially refused because a login-node device ID was reused on a compute
node (same inode, devices86/75); a fresh review index stayed on str-c152 and
completed both chunks, job27068568 actual0. No scientific replay occurred.

Next: implement and independently accept the documented local continuation,
reusing verified work and cumulative accounting under the original lease. Do
not rerun the original worker, whole panels, keys, closed queries or training.
Continue framed-journal verification using saved offsets; the active original
cluster Hopper SQLite files must wait for its actual terminal exit.

[Failure evidence and continuation requirements](LOCAL_SQLITE_FAILURE_20260929.md).
The package contains exact small source/receipt bytes, excluding model weights,
SQLite databases, native NPZ, key arrays and raw scientific journals. Concurrent
CQL recovery and its training job27068529 are preserved separately.
