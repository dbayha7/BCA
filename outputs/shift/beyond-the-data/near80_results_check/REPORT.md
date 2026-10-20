# Near-80 (version 8): what the run found, and the checks of its results (20 October 2026)

The run finished on the Framework Desktop on 19 October 2026 at 21:41 UTC. Every phase ended done, stopped or skipped
(288 done, 3 stopped, 199 skipped); nothing failed and nothing halted. The results are in
`../near80/results-20261020T115246Z/` (the owner's commit `67209d2`; the 90 data records `.gitignore` left out were
added from the owner's run-files commit `7f0fe12` in commit `5208f70`). The model files are on the branch
`claude/near80-run-files` (commit `7f0fe12`).

## The outcome, by the pre-registered rules

- **Claim 4 (step 2's result in the main setting, fresh seeds 10-14): fail.** On seeds 10, 12 and 13 the actor
  training phase of the stalling world stopped before it trained anything: BCA's switch was on, but its height was
  infinite, and the approved code refuses to build the arms from a switched-on arm with an infinite height
  (`step2lib.arm_table`). A stopped phase makes every mark that needs it fail, and the marks are pooled over all five
  seeds, so claim 4 fails.
- **Claims 1-3: not run on seeds 10-14.** The exploratory look on seeds 0-4 stopped the near-80 and random-rows
  settings: hiding the band cost the hand-set range limit 4.299068660557393 points against random rows, and the stop
  rule needed at least 5. On seeds 0-4 every other mark passed, but seeds 0-4 never carry a claim.

The exploratory and confirmatory looks (`files/looks/`) hold the full text and tables.

## Why the height was infinite

BCA's certificate draws 1,000 posterior samples of the full-throttle group's threshold. A sample is infinite when the
random weight on the new point exceeds alpha (0.1), which happens with probability 0.9 to the power of the number of
rows, whatever the critic's numbers are. The group is certified, and the height finite, only when at least 95% of the
samples are finite. That needs about 29 rows (0.9^28 = 0.052, 0.9^29 = 0.047).

| stalling world, main setting | seed 10 | seed 11 | seed 12 | seed 13 | seed 14 |
|---|---|---|---|---|---|
| full-throttle rows (`n_tail`) | 19 | 53 | 22 | 18 | 30 |
| share of finite samples | 0.879 | 0.997 | 0.903 | 0.85 | 0.964 |
| certified | no | yes | no | no | yes |
| switch | on | on | on | on | on |
| height | inf | 2.289130291938781 | inf | inf | 4.595018935203548 |

(From `files/learners/main/stall/seed1x/switch.json`.) On seeds 0-4 the same group had 35, 47, 59, 43 and 47 rows,
all certified (step 1's results, `restructure/step1/results/results.json`), so this never happened there.

This is not a coding defect. The approved code stops on it on purpose, and seeds 10-14 were made by the same generator
as seeds 0-4 (the gate's recorder, sizes 500, 200 and 300 episodes, stall slope 1.5). Amendment 1 cannot have changed
the rows: every phase rebuilt its seed's log and checked its hashes against the data record first.

## Mistakes in the pre-registration (the session's)

- **It did not plan for this stop.** Section 10 says "Only plumbing failures stop a seed-world", and section 8.3 names
  four plumbing stops. The approved code adds a fifth, "a built-in check failed", which fires on an uncertified
  full-throttle group. The outcome is the same under either reading: section 14.6 makes a mark fail when a number it
  needs is missing because a phase stopped. But "claim 4 failed" here means "claim 4 could not be tested on seeds 10,
  12 and 13", not that BCA's controller did badly on them.
- **It was foreseeable.** The gate's own records of 30 stalling-world seeds (`runs/shift/pre/data/base/stall/`, made
  on 12 October with the same generator) count 11 to 60 rows past 80% kept after the removal, and 7 of the 30 are
  below 29 (seeds 103, 106, 107, 112, 118, 901 and 902). That count is a close stand-in for the certificate's group:
  on seeds 0-4 it is 34, 45, 59, 42 and 45, within 2 rows of it. So about one fresh seed in four would stop, and the chance that at least one of five
  would was about three in four. The pre-registration's "28-59 per seed and world" came from seeds 0-4 alone, and its
  guard against too few rows (section 17, item 9) was set at 9 rows, the gap rule's minimum, not the certificate's
  roughly 29.
- **The session saw this stop on 19 October and did not report it.** While finding the cause of the halt, the cloud
  session rebuilt seed 10's real data in scratch and ran its phases with stand-in learners (section 19.6). The three
  stalling-world policy phases of that scratch run stopped with this same refusal. The session put it down to the
  stand-in critic, which was wrong: the height depends only on the number of rows, not on the critic. It should have
  been in section 19.6 and raised with the owner before the approval. It does not touch amendment 1's fix, which
  changes only where a fresh seed's log sizes come from, and the run's records were not affected.

## Description only, not a claim: seeds 11 and 14

These are the two seeds whose certificate succeeded, so they are a selected subset and say nothing about claim 4.
Per seed, through the pinned code (`step2lib.go_no_go` on one seed; the values are also in `files/looks/confirmatory.json`
under `marks.claim4` and `reported.nets.main`):

| | needed | seed 11 | seed 14 |
|---|---|---|---|
| BCA's net across worlds | | 106.28627361825103 | 109.54471459762166 |
| best world-blind rival's net (the always-on ramp) | | 76.57983041836232 | 81.45697043443035 |
| lead | at least 10 | 29.70644319988871 | 28.087744163191317 |
| keeps the gain, recorded conditions | at least -3 | 8.963949952217746 | 8.121280788357168 |
| keeps the gain, big waves | at least -3 | 19.99668080251169 | 22.455627638221102 |
| keeps the gain, big kicks | at least -3 | 20.658932936411844 | 23.028841177585186 |
| beats the matched even pull by | at least 15 | 108.22381213781182 | 105.03622605989233 |

Not every arm looks like seeds 0-4: on seed 11 most arms score lower in the no-stall world than on any of seeds 0-4,
and the split-conformal switch scores lower in the stalling world on both seeds.

## Reading the confirmatory look

Read its Claims section. Its Marks, Brackets and Validity tables print "fail", zero counts and "Check 10 ... did not
hold in 5" for the near-80 setting, which never ran on seeds 10-14 (`validity.claims_2_3.check10.unknown_seeds` is
[10, 11, 12, 13, 14]); V = 2.0 there is the exploratory value. The look's JSON writes the infinite heights as null
(`pooled.plain`); the switch records hold "inf".

## The checks (`checks.json`)

Four independent checks, each re-done by a second reviewer told to refute it:

1. **Integrity: holds.** All 2,724 lines of `SHA256SUMS`, `RESULTS.json` and all 1,200 JSON sidecars match; the 30
   training manifests without a sidecar match the fingerprints in their data records. The owner's run-files list
   agrees on all 2,724 shared paths. The 10 phases that failed before amendment 1 have attempt 1 failed (16 October,
   the type error) and attempt 2 done (19 October); no other phase has a second attempt. All 67 records written after
   the amendment's import (2026-10-19T20:27:47Z) carry its block (52fe05b92900290b, commit f56a107); none of the 222
   before do. The pinned code matches `amendment_1.json` for all 28 files it lists. All 301 attempts share one boot.
2. **The looks reproduce byte for byte** from the records with the pinned code, without the runner, once the ledger is
   limited to the attempts that had ended when each look started. Without that limit only the CPU-hours table differs.
3. **The three stops** are a genuine limit of the approved method, not a defect (above).
4. **Seeds 11 and 14** as above; every number re-derived two ways.

Small points found, none changing a result: 55 of the 199 skips were set at the runner's start-up on 19 October without a
line in `runner.txt` (they are in `status.json` and the look); the confirmatory look's input list leaves out
`status.json` and the ledger, though what it took from them matches the files; the policy's own `stop.json` files
carry no approval block (the `stops/` records do); the desktop clock stepped back about 3 seconds at the 16 October
halt; and section 19.2 gives the halt as 19:03:39 UTC, which is when the 10 phases started (the halt line is at
19:03:45 UTC).

The checkers' scripts were in the cloud session's scratch folder, which is not kept.
