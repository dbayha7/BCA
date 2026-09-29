# Five OOD pairs now active, with three parallel Walker workers

September 29, 2026, 08:51 UTC. Three ReBRAC Walker2d pairs (seeds
202609171, 202609172 and 202609173) are collecting states on separate CPU jobs.
They join the unchanged local TD3+BC Hopper171 and cluster ReBRAC Hopper171
workers, which are both running outcome rollouts. **No revised OOD comparison
has finished or been independently accepted.**

| Pair | CPU job | Latest explicit transitions | Stage |
| --- | --- | ---: | --- |
| ReBRAC Walker171 | 27068500 | 5,742 | Collection |
| ReBRAC Walker172 | 27068501 | 3,841 | Collection |
| ReBRAC Walker173 | 27068502 | 3,628 | Collection |
| ReBRAC Hopper171 | 27067675 | 56,803 | Outcomes |
| TD3+BC Hopper171 | Local worker444 | 179,523 | Outcomes |

These are dated observations, not final counts. Original command/start identities
and pinned worker sources matched; all five worker exits were still pending.
The original local supervisor's start-tick pin remains unavailable. Scheduler
RUNNING with exit field 0:0 does not establish successful completion.

## Real checks and the faster storage route

Each Walker worker passed the unchanged live action, reward and complete-state
repeat gates before collection. Both actors matched exactly on each pair.
Separate saved-array review processes then exited 0 at 08:50:37–38 UTC, checking
the first five acknowledged physical calls from an immutable journal prefix.
Constructor reward errors were 9.79824e-11, 1.16791e-10 and 5.25918e-11, all below
1e-7. Full first/repeat records and full resulting states matched exactly. These
are engineering checks; they do not accept collection or scientific outcomes.

The new shared sequential journal integrates the existing frozen constructor,
RecordedStep, ArchiveDirectory and EncodedStore interfaces. It retains the same
named payload bytes, native input/applied/output NPZ records and complete states.
It synchronizes reservation before the callback, before-state/input/intercepted
controls before physical execution, output/full-state/completion evidence before
the accounting acknowledgment, and that acknowledgment before returning. The
constructor has equivalent boundaries. Precommit and panel records outside a
call are individually synchronized. A local disposable SQLite lookup cache keeps
duplicate-name/token checks and offset lookups; the shared journal is the durable
authority. No node-local-only evidence or batched unreserved physics was adopted.

Final integration fixtures passed 21 checks locally and on cluster CPU job
27068417, with exact constructor/RecordedStep bodies and exact saved archive
context-manager definitions, synthetic native objects only. They include eight
deliberate exit23 cases at four durability boundaries, partial evidence/tail,
cross-process exclusion, fork/duplicate/cap refusal and fsync failures. An
independent saved-byte reviewer checked 87 frames across 17 retained journals.
Eight additional correction fixtures passed, including process death with two
cumulative charged reservations and permanent refusal to reopen the claim.
These are process-death tests, not power-loss, arbitrary-Python isolation or
general recovery acceptance. The earlier fixture setup failure (missing parent
directory, suite exit1) and 20-check development pass remain preserved.

Walker171's first approximately 150 seconds of collection recorded about 38
explicit steps/s. This is an early observed rate for this environment, not a
controlled same-workload speedup over Hopper or an outcome completion forecast.
The previous storage-only benchmark must not be reported as an end-to-end gain.
The original two scientific workers and their storage remain unchanged.

## Preserved startup failures and explicit correction

The first three new attempts, jobs 27068442/443/444, exited1 at 08:38 UTC. The new
execution wrapper mistakenly requested `walker2d-medium-v2`; the frozen adapter
only accepts the training-matched `walker2d-medium-replay-v2`. The source review
missed this mismatch. The adapter refused before `gym.make` and before creating
its native evidence directory. Each saved journal contains exactly a header
and one 1-environment/4-physics constructor reservation, with no artifacts or
acknowledgment. The actual failures, original source/claims/journals and review
remain intact. The reservation is kept charged despite the pre-constructor
refusal; its pending status is never converted into a success.

A separate saved-byte/source disposition binds that specific failure, the
adapter's actual whitelist and accepted training manifest entries. The v2
execution-only correction changes the dataset label and introduces a narrow
carry-forward claim. It acquires the ORIGINAL pair lease, creates one permanent
exclusive `correction-v2` claim, and uses a NEW journal whose header records the
prior pending charge. Cumulative engineering starts at 1, leaving 1,999 new
calls under the unchanged 2,000-call allocation. It never reopens or modifies
the failed journal or resets/refunds totals. The first corrected worker passed
its live checks before the other two were submitted. No scientific source,
checkpoint, seed assignment, numerical gate or coverage parameter changed.

## Allocation and remaining work

The immutable allocation has all ten ReBRAC pairs and totals exactly 18,395,680
environment calls. The existing Hopper171 worker retains its entire 10,000-call
Hopper engineering entitlement and a permanent legacy-pair exclusion. Other
Hopper seeds have no allocated engineering calls and are not newly dispatched.
Walker's separate 10,000-call cell entitlement is split into 2,000 per declared
seed; only the three already accepted training pairs are eligible now. All pair
collection/repeat/outcome caps stay 38,400 / 7,168 / 1,792,000. Together with the
local TD3 half and original 1,298,353 reserved calls, the maximum remains
38,089,713, below 39,998,400. All prior/uncertain charges are retained. This
parallel allocation does not authorize duplicate pairs or old claim reopening.

Each worker must finish collection and the entire 256-state precommit before
outcomes. Await actual exits and independently verify every artifact, native
payload, reservation, key/precommit/return join and source/checkpoint identity
before scientific readout. Full outcome-audit implementation/execution remains
pending. Early terminal states, missing quotas, aliases and all declared seeds
remain visible; no replacements or outcome tuning. Report absolute advantage
A and anchor-relative D together, familiar returns and warnings separately,
with honest seed uncertainty. Five active pairs are not a completed twenty-pair
tranche or evidence of BCA superiority.

The evidence package contains exact small source/receipt bytes, including failed
versions. Weights, SQLite databases, native binaries, NPZ arrays, continuation
key arrays and scientific/synthetic journal payloads are excluded. Unrelated
concurrent training-recovery edits in the repository are preserved outside this
publication. No training or closed query/fixture suite was replayed.
