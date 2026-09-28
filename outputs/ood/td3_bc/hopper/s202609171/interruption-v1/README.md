# Outcome collection interrupted by a Windows restart

At 01:41 UTC on September 28 the original worker was absent. Windows records shutdown at **01:35:32 UTC** and startup at **01:39:15 UTC**. Its Windows supervisor was also absent. The final durable progress snapshot, at **01:32:14 UTC**, has **143 / 512 panels**, **357,500 outcome transitions** and **1,430 repeated-first-transition checks**. These counters cover closed panels only; they exclude any later reservations or partial panel. The precise total and termination time are not inferred.

Neither worker nor supervisor actual-exit receipt exists; `failure.json` and `completion.json` are also absent. Exit codes are **unknown**, not 0, 1 or an inferred signal. This is an interrupted attempt, not a successful or independently accepted harm-ranking result. The snapshot includes only the host continuation through state142; no comparison is promoted from this incomplete prefix.

All original attempt files, partial records and the cumulative resource ledger remain at their original paths. No simulator, model query, retry, resume, replacement, ledger reset or source edit was made during diagnosis. The 108 frozen training files and 12 bound execution files still match their declarations. Closed query, engineering, state and candidate gates remain accepted and are not replayed. All five training seeds and both continuations remain required; coverage and global readiness remain false.

The independent cluster worker continues: ReBRAC HalfCheetah BCA was at 485k with matching controller/worker identities at the saved observation. Eight cluster closures await independent training audits; the three accepted standard physical runs are unchanged. The prior local CQL failure and stopped training queue remain separate.

The earlier production/dispatch evidence commit `5c8d5eaa4673c06ca4d0b89ed0e316a448dfec05` is now published. New push actual exit0, matching remote main and all 80 owned blob hashes were checked. The original DNS failure receipt remains preserved. This publication concerns evidence, not accepted outcome results.

## Boundary for any later recovery

Preserve this attempt in place. A saved-record inspection may inventory durable completed/partial panels, archive records and original reservations without calling a model or simulator. It must account for reservations whose physical completion is uncertain, retain both original and any new receipts, and never turn missing actual exits into success. David has explicitly requested restart in the active chat "Analyze BCA algorithm performance (3)"; recovery preparation is already in progress there. Do not ask again for that authorization or launch a duplicate. Separately reviewed execution acceptance is still required; this monitoring turn launched no recovery.

## Evidence

The observation JSON files, actual diagnostic exits, filtered Windows restart events and source-integrity check are included here. Windows events unrelated to restart are omitted from this publication; the original diagnostic and its hash are retained. High-frequency archives and the original resource ledger are not copied or opened by this diagnostic. Publication receipts named `prior_work_*` confirm the previous commit, not the commit containing this new interruption report.

Publication copies of eight diagnostic JSON files use LF newlines to match Git blobs. `copy-normalization.json` records original and published hashes and identical parsed values; original private receipts remain unchanged. The initial pre-push mismatch was detected before publication and is retained in the monitor snapshot.
