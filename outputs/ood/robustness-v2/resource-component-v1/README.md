# V2 resource extension component

Status:29 synthetic tests passed, actual exit0/no skips. Separate small read-only
ancestor binding also passed. **No real extension ledger was created, no real
resource lock was acquired and no physical/model call was made.** This is a
tested accounting prerequisite, not an accepted scientific executor.

The new `ancestor_guard.py` opens the original ledger only with mode=ro and
query_only. It compares the externally bound header, last entry, all65 cached
scope totals and exact audited file identity, refusing sidecars or redirection.
It reuses the successful full audit of1,298,293 entries; no full ledger rescan.
The original charged totals remain1,298,353 environment/5,193,412 physics, with
last head6bfd6380329cf7ea4ac204b80edf98fae8593c7da79411809b0132bfc3bc152c.

`extension_ledger.py` uses a separate explicit path, immutable declaration,
append-only reservations/completion receipts, FULL synchronous DELETE-journal
transactions and a lifetime exclusive lease on an existing shared lock file.
Every call reserves1 environment/4physics before the callback, checking both its
phase scope and combined original/new global ceiling. Maximum scopes are fixed
by the v2 protocol; reductions are allowed, increases are refused. Pure budget
limits still require the independently accepted actual execution declaration.

Only one unresolved reservation is allowed. A callback error, invalid output
receipt, process death after reservation, or changed ledger/lock/ancestor stops
further calls. The pending reservation stays charged with completion unknown;
there is no refund, silent journal repair, resume or retry interface. Reopening
a fully closed journal audits its chain, exact schema/triggers, totals and output
receipt ownership. Reopening a pending journal refuses execution and requires
separately designed recovery handling. A completion receipt is a producer's
hash-bound output declaration, not an independent proof of correct physics.

Runtime checks use an indexed reservation-tail query, avoiding a growing history
scan per transition. A full new-journal audit streams the join with bounded
working memory; it still detects an unresolved earlier call hidden behind a
closed tail. File identity is checked across initialization and commits; mutable
in-memory declarations/bindings are also refused after validation.

Tests use temporary synthetic SQLite files and callbacks only. They include a
subprocess that exits23 immediately after a committed reservation, confirming
that the charge persists and reopening is refused. They also cover duplicate
tokens, zero/combined scope ceilings, changed original head/file/binding,
sidecars/symlinks, lock conflict/replacement/fork misuse, immutable triggers,
cached-total/chain corruption, missing output evidence, file replacement during
callback, declaration mutation and indexed lookup behavior. The initial expected
missing-module failure and24-/26-test development passes remain preserved.
Source review prompted the indexed-tail/bounded-audit change and subsequent
immutable-binding/file-identity checks; all29 final tests were then run once.

The real integration receipt in monitor_20260928T182657Z is
`real_ancestor_check_v2.json`, backed by `real_ancestor_v2_actual_exit.json`0.
The earlier v1 small-read receipt is preserved; the second was necessary after
the new binding-immutability guard was added, not a repeat of the large audit.
Neither integration imports the extension writer or opens/acquires the real lock.

Remaining: independently review/integrate the execution declaration and actual
paths; implement v2 collector/precommit/supervisor and outcome-evidence handling;
bind accepted checkpoints/runtime/fresh streams; preserve action1e-6/reward1e-7
and full-state repeat gates; then accept execution before any new science.
The collector must give each actual physical call exactly one scope/token,
archive inputs/applied control/output/full final state and constructor evidence,
and supply the durable output hash before recording completion. This component
does not establish those scientific semantics by itself. It also does not handle
an uncooperative writer that bypasses the shared lock; identity changes refuse
further work, and exclusive ownership must be enforced by the execution layer.

The original v1 study,33 prior v2 preparation tests, frozen source/report hashes,
unknown ancestor exits/input-only output and old resource reservations remain
unchanged. Global readiness and v2 scientific/execution acceptance remain false.
