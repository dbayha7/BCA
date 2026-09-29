# Local audit interference, September 29

The local TD3+BC Hopper seed202609171 worker exited **1** at
09:22:11.785305 UTC. At 09:22:07.085316 it reported `database is locked` from
`ExtensionLedger.reserve`, specifically its `COMMIT`, before the physical
callback. This is a real interrupted OOD execution, not a harmless audit error.
The running source was not changed and the worker was not restarted.

The new verifier was reading this live ledger at that time. Its first successful
4,000-record pass ended at 09:22:08.058811 and measured 1.538549 seconds in its
review loop. The ledger uses DELETE journaling with `timeout=0`. Read-only
connections still take read locks; a conflicting reservation commit therefore
fails immediately. A separate synthetic fixture reproduces this mechanism.
There was no lock-owner trace, so the exact competing connection cannot be
proved retrospectively. The timing and mechanism strongly implicate our audit.
Starting those live ledger reads was an execution mistake.

**No more audits of active SQLite scientific databases.** The open verifier now
checks a matching actual terminal-worker receipt and that the original command
is absent before opening either evidence or ledger. A regression fixture confirms
the refusal occurs before any SQLite open. The original ReBRAC Hopper worker
also uses SQLite and must be audited only after its actual exit. Framed Walker
journals remain read as append-only prefixes without taking a SQLite lock on
the scientific writer. Their reviewer index is a different database.

After confirming the original local worker had exited, a bounded read-only
inspection found 222,854 reservations and 222,854 completion receipts; no pending
reservation. Totals remain charged: engineering5, collection38,400,
repeat_checks741 and outcomes183,708 (222,854 environment / 891,416 physics).
The failed reservation transaction rolled back according to the pinned source;
no extra persisted reservation was found. This is not a refund. There is no
evidence of a physical callback after the failed reservation commit.

The source archive retains 1,563,709 artifacts and 52 complete panel files.
Its last completed physical record is `host/state026/slot12/246` (247 steps in
that trajectory), transition222853. The immediately following token was not
saved; its step247 identity is inferred from the exact driver ordering. The
partial trajectory is not a completed return. Full saved-prefix reconciliation
completed with actual0 at 09:43 UTC using the same reviewed index and retained
reader source, behind an explicit stopped-worker wrapper. It checked every
1,563,709 artifact and 222,854 physical records/acknowledgements, all256 captures
and precommits, 740 completed trajectories, and all52 completed panel files.
The partial trajectory has247 ordinary calls plus one repeat. No model or
simulator was invoked. Remaining provenance and continuation gates are separate.

The failure evidence, source attempts, audit exits, mechanism fixture and fresh
observations are retained in
`work/standard_bca_noiw_campaign_v1/monitor_20260929T090007Z`.

## Concrete continuation work

1. Consume the now-completed native-record/state/reservation/precommit/return
   joins in `local_terminal_coverage.json` and its actual0 receipt. Complete the
   saved support/training and checkpoint/score provenance closure without
   replaying the already checked1.56M artifacts. Bind the interrupted trajectory
   and actual process exit separately; preserve all bytes and cumulative charges.
2. Implement a separately versioned, explicitly reviewed continuation which
   reuses verified completed evidence and the fixed whole-pair precommit. Do not
   replay whole panels, regenerate candidates/keys or recreate the extension.
   The current worker has no accepted resume interface; calling it again is
   prohibited.
3. Specify the interrupted trajectory disposition before dispatch. The saved
   complete native states may support a checked contiguous continuation after
   new actor/reward/restore engineering gates; that route is not yet accepted.
   Replaying the first action/repeat would consume an additional repeat beyond
   the 7,168 phase ceiling unless separately reconciled. Do not silently move
   calls between phases, refund charges or splice an unverified return.
4. Hold the original shared local lease for the complete new attempt, retain a
   permanent original-pair continuation claim, and carry the original extension
   totals. Prefer the already implemented shared journal interface for new
   evidence. Its allocation must debit the original local TD3 half; never make
   a second independent budget. Require a read-only independent execution
   disposition and unchanged native gates before any new physical call.

The four cluster OOD workers remain separate and running at the last observations.
All three Walker workers had reached outcomes by 09:26 UTC; the original cluster
Hopper was still in outcomes at 09:29. Actual exits remained pending. No completed
revised OOD pair or BCA benefit is claimed. The separately owned CQL correction
and training job27068529 continue unchanged.
