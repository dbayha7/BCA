# Kill-rule and run-integrity check: arms A, B, C and F0 (EXPLORATORY)

**EXPLORATORY.** Data seeds 0-4 are seen seeds. Nothing here is pre-registered or confirmatory. A confirmatory claim would need fresh seeds 15-19, its own pre-registration and a new go from the owner.

Made 2026-10-17 06:32 UTC. Frozen rules: `routes2/arms/PASSMARKS_arms_ABC_and_F_control.md`, sections 3.3, 4.5, 5.5, 6.4 and 7.3 (SHA-256 `a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf`; `sha256sum -c` against its sidecar: OK).

## Verdict

- **Kill decisions:** every decision I rebuilt from the raw records is the same as KILLS.md's: arm A, no rule fired (all five seeds usable on all five qualities); arm B, stopped at the seed-0 screen by K1 on medium-replay, medium and medium-expert; arm C, no KB1 or K3 stop, KB0 holds, the guard is void, so B2-B4 are not scored; F0, KF1-KF4 pass on all ten seed-quality pairs, so both qualities are decided.
- **Ordering:** I found no violation. No final return of A, B or C was computed before every kill rule guarding it had been checked on all five seeds.
- **Restarts:** six jobs were interrupted (three room checks at the first reclaim and three arm-A jobs at the second). All six were redone from scratch. No partial file was reused, and no finished record was overwritten.
- **Integrity:** every SHA-256 sidecar matches. The run's own final untouched check passed, and my recheck now also passes: the sandbox code is unchanged, and so is the repository.
- **Blockers:** none. The minor points at the end change no decision.

## How this was checked

- `verify/kills_check.py` (standard library only) reads the arm records, A2's records and evaluations, the package's stage records, the runner's logs (`nohup_*.out` and per-job logs) and `state/`, and recomputes every rule. KILLS.json, SCREEN_seed0.json and results.json are read only to compare against. Output: `verify/kills_check_raw.json`.
- I also re-executed parts of the computation, reading only, with the jobs' arithmetic settings (one CPU, AVX2, one thread). Each time, `find -newer` confirmed that no file in the sandbox changed. `spot_check.py` recomputes the K2 value of all 25 seed-quality pairs from the saved 8,000-step actors, and both of KB1's critic gains from C's saved 8,000-step actors. `spot_k0_kb0.py` rebuilds the clone and the band read at the recorded commands, and recomputes K0's release shares for the 15 seed-quality pairs of medium-replay, medium and medium-expert. It also recomputes KB0's shares for 15 pairs from C's saved final actors. `spot_maps.py` rebuilds BCA's ceiling maps: B's 5 seed-0 maps and C's 25 guard maps, every rung's statistic and try count on all four thrusters. **Every recomputed number equals the recorded one bit for bit.**
- **Not re-executed:** K1, because the step-4,000 actors are not saved. I checked it from the records' refresh history and from the per-job runtime log lines, which agree with the records on all 125 chunk shares. KF1 and KF2 were checked from their recorded values only. KF3 was checked against A2's own record.

## Arm A (section 4.5)

### K0, before training: A1's release shares at refresh 0 (the clone)

Bars: at most 10% of recorder rows and at least 15% of random rows. Binding on seeds 1-4; seed 0 is information only. The rule is checked on medium-replay, medium and medium-expert only. *Recomputed* means that `spot_k0_kb0.py` rebuilt the clone (the SHA-256 equals the record's `start_sha256`) and the band read, and got the same shares bit for bit.

| Quality | Seed | Recorder rows released | Random rows released | Binding | Passes | Guard-blocked (recorder / random) | Recomputed |
|---|---|---|---|---|---|---|---|
| medium-replay | 0 | 0.07% | 24.24% | no | yes | 0 / 0 | yes |
| medium-replay | 1 | 0.05% | 22.12% | yes | yes | 0 / 0 | yes |
| medium-replay | 2 | 0.01% | 18.93% | yes | yes | 0 / 0 | yes |
| medium-replay | 3 | 0.00% | 20.31% | yes | yes | 0 / 0 | yes |
| medium-replay | 4 | 0.04% | 21.81% | yes | yes | 0 / 0 | yes |
| medium | 0 | 0.17% | 28.67% | no | yes | 0 / 0 | yes |
| medium | 1 | 0.11% | 25.87% | yes | yes | 0 / 0 | yes |
| medium | 2 | 0.06% | 22.32% | yes | yes | 0 / 0 | yes |
| medium | 3 | 0.08% | 25.95% | yes | yes | 0 / 0 | yes |
| medium | 4 | 0.20% | 28.20% | yes | yes | 0 / 0 | yes |
| medium-expert | 0 | 1.85% | 44.72% | no | yes | 0 / 0 | yes |
| medium-expert | 1 | 1.02% | 38.42% | yes | yes | 0 / 0 | yes |
| medium-expert | 2 | 0.78% | 34.99% | yes | yes | 0 / 0 | yes |
| medium-expert | 3 | 0.64% | 37.02% | yes | yes | 0 / 0 | yes |
| medium-expert | 4 | 0.87% | 36.27% | yes | yes | 0 / 0 | yes |

On random (no recorder rows) and expert (no random rows), K0 is not checked, as frozen. For information, A1 releases these shares of random rows on random: 14.1%, 12.9%, 11.0%, 15.9%, 14.4%. It releases these shares of recorder rows on expert: 17.1%, 18.5%, 13.9%, 18.0%, 15.6%.

**Verdict:** no binding K0 failure on seeds 1-4. The closest values are medium-expert seed 1's recorder share (1.02% against 10%) and medium-replay seed 2's random share (18.93% against 15%). All three qualities would also pass on seed 0. This matches KILLS.md (no K0 stop).

### K1, at the step-4,000 refresh (every seed, seed 0 included)

It fires if A1 releases more than 60% of the kept rows, or if mean D exceeds 3 times mean U. These values come from the records' refresh index 1 (`step` = 4000).

| Quality | Seed | Rows released | Mean D | Mean U | Mean D / mean U | Fires |
|---|---|---|---|---|---|---|
| random | 0 | 33.5% | 6.654 | 7.441 | 0.894 | no |
| random | 1 | 31.8% | 6.436 | 7.730 | 0.833 | no |
| random | 2 | 30.9% | 6.465 | 7.960 | 0.812 | no |
| random | 3 | 36.4% | 7.169 | 7.506 | 0.955 | no |
| random | 4 | 34.4% | 6.578 | 7.340 | 0.896 | no |
| medium-replay | 0 | 40.9% | 5.954 | 5.018 | 1.186 | no |
| medium-replay | 1 | 38.9% | 5.637 | 5.129 | 1.099 | no |
| medium-replay | 2 | 32.6% | 5.529 | 5.926 | 0.933 | no |
| medium-replay | 3 | 34.9% | 5.878 | 6.036 | 0.974 | no |
| medium-replay | 4 | 39.6% | 5.584 | 4.929 | 1.133 | no |
| medium | 0 | 36.0% | 4.069 | 3.605 | 1.129 | no |
| medium | 1 | 33.3% | 3.833 | 3.828 | 1.001 | no |
| medium | 2 | 29.9% | 3.857 | 4.196 | 0.919 | no |
| medium | 3 | 35.4% | 4.375 | 3.878 | 1.128 | no |
| medium | 4 | 36.2% | 3.891 | 3.551 | 1.096 | no |
| medium-expert | 0 | 30.8% | 2.967 | 1.717 | 1.728 | no |
| medium-expert | 1 | 26.6% | 2.881 | 1.888 | 1.526 | no |
| medium-expert | 2 | 26.6% | 2.790 | 1.972 | 1.415 | no |
| medium-expert | 3 | 24.5% | 2.754 | 1.999 | 1.378 | no |
| medium-expert | 4 | 28.1% | 2.548 | 1.814 | 1.405 | no |
| expert | 0 | 32.8% | 0.491 | 0.414 | 1.185 | no |
| expert | 1 | 35.7% | 0.495 | 0.399 | 1.240 | no |
| expert | 2 | 26.9% | 0.488 | 0.479 | 1.019 | no |
| expert | 3 | 32.2% | 0.500 | 0.419 | 1.194 | no |
| expert | 4 | 30.4% | 0.479 | 0.458 | 1.045 | no |

**Verdict:** K1 fires nowhere. The highest share is 40.9% (medium-replay seed 0) and the highest ratio is 1.728 (medium-expert seed 0). The runtime log line of each A job (`chunk 2/5, A1 released ...`) agrees with every record. This matches KILLS.md.

### K2, at 8,000 steps: A1 - none on the strong test

A quality stops if the value is -3.0 points or lower on at least 3 of 5 seeds. KILLS ran at 05:42:13, after every A job had finished (the last at 04:53:07), and before any final A actor was scored (the first `eval_A` started at 05:42:15). All 25 values were recomputed from the saved 8,000-step actors and are equal bit for bit.

| Quality | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seeds at or below -3.0 | Fires | Recomputed |
|---|---|---|---|---|---|---|---|---|
| random | 20.01 | 17.55 | 19.68 | 18.89 | 13.27 | 0 | no | yes |
| medium-replay | 14.00 | 12.76 | 10.73 | 15.02 | 10.47 | 0 | no | yes |
| medium | 7.34 | 10.33 | 6.10 | 7.51 | 7.12 | 0 | no | yes |
| medium-expert | -1.80 | -1.28 | -1.59 | -0.71 | -1.32 | 0 | no | yes |
| expert | 0.65 | 0.62 | 0.26 | 0.87 | 0.43 | 0 | no | yes |

**Verdict:** K2 fires nowhere. The lowest value is medium-expert seed 0 at -1.80. This matches KILLS.md's K2 values exactly.

### K3, at any time

- All 25 A records have status `finished`, and the final actors of every arm that a mark needs are finite.
- All 25 gapped band records have `band_valid` true. The band record's SHA-256 that each A record names equals the current file.
- No record holds a stop for a refused gate check.

**Verdict:** no K3 stop. Arm A has no stop on any quality and seeds 0-4 are usable everywhere, the same as KILLS.md.

## Arm B (section 5.5)

B was screened on seed 0 only. Its seed-0 maps were rebuilt by `spot_maps.py` and are equal to the records on all four thrusters, rung by rung. The walk (the last rung of the unbroken certified run from rung 1) reproduces every ceiling. On every rung, the `certified` flag equals 'statistic < 0 with at least one try'.

| Quality | BCA ceilings (t1-t4) | Thruster-1 tries per rung (0.50-0.55 ... 0.95-1.00) | Thruster-1 rungs certified | t1 | Tries in (0.80, 0.85] | K1 fires |
|---|---|---|---|---|---|---|
| random | 0.80, 0.75, 0.65, 0.50 | 209, 137, 80, 69, 38, 24, 12, 2, 1, 0 | 1111110000 | 0.80 | 12 | no |
| medium-replay | 0.90, 0.75, 0.70, 0.65 | 369, 272, 240, 204, 168, 199, 42, 25, 12, 2 | 1111111100 | 0.90 | 42 | yes |
| medium | 0.90, 0.75, 0.80, 0.70 | 470, 379, 386, 355, 380, 541, 54, 34, 21, 6 | 1111111100 | 0.90 | 54 | yes |
| medium-expert | 0.85, 0.80, 0.85, 0.65 | 883, 814, 859, 838, 833, 736, 78, 67, 41, 35 | 1111111000 | 0.85 | 78 | yes |
| expert | 0.80, 0.75, 0.80, 0.65 | 1120, 1089, 1095, 1096, 998, 493, 0, 0, 0, 0 | 1111110000 | 0.80 | 0 | no |

- **K1** (a thruster-1 ceiling of 0.85 or higher, with at least 30 tries in rung 7, (0.80, 0.85]) **fires on three qualities.** On medium-replay, t1 is 0.90 with 42 tries in rung 7. On medium, t1 is 0.90 with 54 tries. On medium-expert, t1 is 0.85 with 78 tries. It does not fire on random (t1 0.80, 12 tries) or expert (t1 0.80, 0 tries).
- **K0, room:** on medium-expert seed 0, (host + knee clip) - host = 6.99 points (bar +1.0), so it does not fire.
- **K2, over-refusal:** no thruster-1 ceiling is 0.60 or lower (the lowest is 0.80), so it does not fire.
- **K3:** no non-finite value, so it does not fire.
- **K4:** the map clips 28.8% of the host's thruster-1 commands on medium-expert (bar 2%), so it does not fire.
- **Which rule stopped B:** K1 at the screen (seed 0), on medium-replay, medium and medium-expert. The screen records `B_stopped_by = ['K1']`. No B job ran for seeds 1-4: `job_times.tsv` has only the five `B_*_s0` rows, and there are no B records for seeds 1-4. B's 5-seed-mean checks (K0 and K4) were never computed, which is correct for an arm stopped at the screen.
- **No B return entered a mark.** There is no `eval_B` job and no `eval/B` record. Every B number in results.json is null (each `per_seed` is all null and every mean is null). G1-G4 are not met, and every RESULTS.md row for B shows '-'. The only return-like numbers in B's records are K0's host and knee-clip points, which that rule reads.

**Verdict:** this matches KILLS.md: B was stopped by K1 on medium-replay, medium and medium-expert.

## Arm C (section 6.4)

### KB1, at 8,000 steps (every seed)

G(rho 1) > 0 everywhere, so the bar is 3 G(rho 1). The 3s fallback is never used. Still, the s that C recorded equals arm A's s for the same seed and quality exactly, in all 25 pairs. Both gains were recomputed from C's saved 8,000-step actors and are equal bit for bit.

| Quality | Seed | G(rho 1) | G(rho 0.5) | Bar 3 G(rho 1) | G(rho 0.5) / G(rho 1) | Fires |
|---|---|---|---|---|---|---|
| random | 0 | 5.405 | 7.734 | 16.214 | 1.431 | no |
| random | 1 | 4.964 | 7.392 | 14.892 | 1.489 | no |
| random | 2 | 5.125 | 7.497 | 15.374 | 1.463 | no |
| random | 3 | 5.574 | 8.109 | 16.722 | 1.455 | no |
| random | 4 | 5.473 | 7.575 | 16.419 | 1.384 | no |
| medium-replay | 0 | 4.314 | 6.061 | 12.941 | 1.405 | no |
| medium-replay | 1 | 4.278 | 5.963 | 12.833 | 1.394 | no |
| medium-replay | 2 | 4.482 | 6.270 | 13.447 | 1.399 | no |
| medium-replay | 3 | 4.452 | 6.295 | 13.357 | 1.414 | no |
| medium-replay | 4 | 4.094 | 5.871 | 12.283 | 1.434 | no |
| medium | 0 | 3.293 | 4.298 | 9.879 | 1.305 | no |
| medium | 1 | 3.242 | 4.235 | 9.725 | 1.306 | no |
| medium | 2 | 3.406 | 4.400 | 10.217 | 1.292 | no |
| medium | 3 | 3.379 | 4.587 | 10.137 | 1.357 | no |
| medium | 4 | 3.086 | 4.143 | 9.259 | 1.342 | no |
| medium-expert | 0 | 2.461 | 3.065 | 7.384 | 1.245 | no |
| medium-expert | 1 | 2.400 | 2.993 | 7.201 | 1.247 | no |
| medium-expert | 2 | 2.324 | 2.918 | 6.971 | 1.256 | no |
| medium-expert | 3 | 2.297 | 2.869 | 6.891 | 1.249 | no |
| medium-expert | 4 | 2.200 | 2.688 | 6.600 | 1.222 | no |
| expert | 0 | 0.470 | 0.554 | 1.411 | 1.178 | no |
| expert | 1 | 0.463 | 0.552 | 1.388 | 1.192 | no |
| expert | 2 | 0.464 | 0.551 | 1.391 | 1.188 | no |
| expert | 3 | 0.465 | 0.555 | 1.395 | 1.193 | no |
| expert | 4 | 0.461 | 0.547 | 1.383 | 1.186 | no |

**Verdict:** KB1 fires nowhere. The highest ratio is 1.489 (random seed 1), and the bar is 3. There is no K3 (every actor is finite and no record has non-finite values). This matches KILLS.md.

### KB0, the guard is idle

The rule needs rho 0.5's share of thruster-1 commands past float32(0.8) to be at least 2% and at least twice rho 1's, on at least 2 of random, medium-replay and medium. All 15 seed-level shares were recomputed from C's saved final actors and are equal bit for bit.

| Quality | rho 0.5 per seed | rho 1 per seed | Seed 0 passes | 5-seed mean rho 0.5 | 5-seed mean rho 1 | Twice rho 1 | Mean passes |
|---|---|---|---|---|---|---|---|
| random | 1.49%, 2.92%, 1.76%, 0.85%, 1.83% | 0.43%, 0.84%, 0.56%, 0.13%, 0.66% | no | 1.768% | 0.524% | 1.047% | no |
| medium-replay | 2.96%, 3.58%, 3.56%, 4.85%, 4.14% | 0.77%, 1.18%, 1.45%, 1.41%, 1.44% | yes | 3.819% | 1.251% | 2.502% | yes |
| medium | 8.53%, 6.71%, 8.32%, 17.28%, 7.58% | 3.19%, 1.95%, 2.25%, 3.99%, 2.42% | yes | 9.686% | 2.760% | 5.520% | yes |

**Verdict:** KB0 holds at the screen (medium-replay and medium) and on the 5-seed means (medium-replay and medium). Random fails both times (rho 0.5's mean is 1.77%, under 2%). KB0 therefore does not stop B2-B4, which matches KILLS.md (`holds: True`).

### The guard-void rule

- **By B's K1 or K2:** B's K1 fired at the screen (above), so **the guard is void on that ground alone.** B's K2 did not fire.
- **By C's own map** (a thruster-1 ceiling of 0.85 or higher, with at least 30 tries in rung (0.80, 0.85]): all 25 guard maps were rebuilt and equal the records. The rule triggers on **18 of 25** seed-quality pairs:

| Quality | Seed | Guard t1 | Tries in (0.80, 0.85] | Triggers |
|---|---|---|---|---|
| random | 0 | 0.85 | 54 | yes |
| random | 1 | 0.85 | 35 | yes |
| random | 2 | 0.80 | 36 | no |
| random | 3 | 0.85 | 30 | yes |
| random | 4 | 0.85 | 68 | yes |
| medium-replay | 0 | 0.90 | 63 | yes |
| medium-replay | 1 | 0.90 | 69 | yes |
| medium-replay | 2 | 0.80 | 85 | no |
| medium-replay | 3 | 0.85 | 125 | yes |
| medium-replay | 4 | 0.85 | 118 | yes |
| medium | 0 | 0.90 | 104 | yes |
| medium | 1 | 0.90 | 93 | yes |
| medium | 2 | 0.85 | 81 | yes |
| medium | 3 | 0.90 | 156 | yes |
| medium | 4 | 0.85 | 127 | yes |
| medium-expert | 0 | 0.85 | 118 | yes |
| medium-expert | 1 | 0.85 | 130 | yes |
| medium-expert | 2 | 0.90 | 126 | yes |
| medium-expert | 3 | 0.85 | 157 | yes |
| medium-expert | 4 | 0.85 | 141 | yes |
| expert | 0 | 0.80 | 0 | no |
| expert | 1 | 0.80 | 0 | no |
| expert | 2 | 0.80 | 0 | no |
| expert | 3 | 0.80 | 0 | no |
| expert | 4 | 0.80 | 0 | no |

  The rule triggers on random seeds 0, 1, 3 and 4 (seed 3 sits exactly at 30 tries, and 'at least 30' counts it), on medium-replay seeds 0, 1, 3 and 4, on all five seeds of medium, and on all five seeds of medium-expert. It does not trigger on random seed 2 (t1 0.80), medium-replay seed 2 (t1 0.80), or expert seeds 0-4 (t1 0.80, with no tries in the rung). This is the same list as KILLS.json's `by_own_map`.
- **Consequence:** B2, B3 and B4 are not scored (results.json `met` is None for each, with `B2_B4_not_scored` set). B1 and B5 are still scored, as frozen.

**Verdict:** this matches KILLS.md: guard void, with B2-B4 not scored.

## F0 (section 7.3)

KF3 compares F0's clone SHA-256 with `clone.actor_sha256` in A2's own record (`channel/train_f1/<Q>/A2_cmd_strong/seed<k>/record.json`). For seed 0, that record is the result F rerun's, which is in the baseline and unchanged. The A2 record each F0 read is still unchanged (its SHA-256 matches).

| Quality | Seed | KF1 disagreement (rows) | KF1 at le_none (seed 0) | KF2 (a) max diff | KF2 (b) max diff | KF3 clone F0 = A2 | KF4 finite |
|---|---|---|---|---|---|---|---|
| medium-expert | 0 | 0.0 (0) | 0.0 (0) | 0.0 | 0.0 | yes (`f87093cad7a4`) | yes |
| medium-expert | 1 | 0.0 (0) | - | 0.0 | 0.0 | yes (`f1125961a66d`) | yes |
| medium-expert | 2 | 0.0 (0) | - | 0.0 | 0.0 | yes (`dcdcb4a56350`) | yes |
| medium-expert | 3 | 0.0 (0) | - | 0.0 | 0.0 | yes (`e232b009a382`) | yes |
| medium-expert | 4 | 0.0 (0) | - | 0.0 | 0.0 | yes (`4cf20f901a33`) | yes |
| expert | 0 | 0.0 (0) | 0.0 (0) | 0.0 | 0.0 | yes (`73e03006c0f6`) | yes |
| expert | 1 | 0.0 (0) | - | 0.0 | 0.0 | yes (`3ec59d0e0b49`) | yes |
| expert | 2 | 0.0 (0) | - | 0.0 | 0.0 | yes (`41d313d1c85f`) | yes |
| expert | 3 | 0.0 (0) | - | 0.0 | 0.0 | yes (`76dabc819f38`) | yes |
| expert | 4 | 0.0 (0) | - | 0.0 | 0.0 | yes (`8d99334feb73`) | yes |

KF2 ran 300 steps. In (b), `grid`, `thresholds`, `cap`, `u_logged` and `cal_params` were replaced. **Verdict:** every check passes on all ten pairs, and both qualities are decided, as in KILLS.md.

## Ordering (sections 3.3 and 8)

All times are UTC, from `state/job_times.tsv`, the runner logs and the records' own timestamps.

| Event | Time |
|---|---|
| Seed-0 screen (`screen` job) | 2026-10-16 19:52:43 - 2026-10-16 19:52:45 |
| Last training or arm job finished | 2026-10-17 05:42:13 (F0_expert_s4) |
| Last arm record written | 2026-10-17 05:42:13 (F0 expert seed 4) |
| Kill checks (`kills` job; KILLS.json made_at 2026-10-17T05:42:13+0000) | 2026-10-17 05:42:13 - 2026-10-17 05:42:15 |
| First final-return job | 2026-10-17 05:42:15 (eval_A_random_s0) |
| Last final-return job | 2026-10-17 05:55:41 (eval_F_expert_s4) |
| Marks (`marks` job; results.json made_at 2026-10-17T05:55:42+0000) | 2026-10-17 05:55:41 - 2026-10-17 05:55:43 |

- No training, arm, host, room, band or A2 job finished after the kill checks started, and no evaluation job started before they finished. There are 60 evaluation records (A 25, C 25, F 10, B 0). Each started after KILLS.json's made_at, and each carries the current KILLS.json SHA-256 and its arm record's current SHA-256.
- **No final A, B or C return before its rules were checked.**
  - A deep scan of every A, B and C training record finds no final return. The only return-like numbers are A's K2 8,000-step difference and B's K0 host and knee-clip points, and both are what those rules read.
  - C's KB0 rollouts of the final actors keep commands only, as section 6.4 prescribes.
  - The screen holds returns only for B's K0 and for F0, A2 and le_none on seed 0. F0's seed-0 return is allowed at the screen.
  - F0's seeds 1-4 are scored inside their own jobs, after that seed's KF1-KF4. This is allowed, because no cross-seed rule guards F0.
- No `eval_*`, `kills` or `marks` job appears in the earlier runner logs, and each evaluation log shows a single attempt. The smoke test's evaluations (16 October, on smoke-trained stand-ins, deleted) were not of these actors.

**Verdict:** no ordering violation.

## Restarts

| Job | Interrupted run | First start | First attempt's last log line | Rerun's first progress line | Files now (mtime) |
|---|---|---|---|---|---|
| room_medium-replay_s4 | nohup_continue.out | 2026-10-16 21:49:17 | 21:59:34 actor-only rep 4: 5 arms, step 12000/20000 | 01:01:05 cloning rep 4: step 5000/20000, loss 0.2967 | actors.msgpack 01:19:56; record.json 01:19:56; record.json.sha256 01:19:56 |
| room_medium_s1 | nohup_continue.out | 2026-10-16 21:49:18 | 21:59:38 actor-only rep 1: 5 arms, step 12000/20000 | 01:01:05 cloning rep 1: step 5000/20000, loss 0.2617 | actors.msgpack 01:22:10; record.json 01:22:10; record.json.sha256 01:22:10 |
| room_medium_s2 | nohup_continue.out | 2026-10-16 21:56:21 | 21:56:56 cloning rep 2: step 20000/20000, loss 0.2217 | 01:01:05 cloning rep 2: step 5000/20000, loss 0.26 | actors.msgpack 01:22:16; record.json 01:22:16; record.json.sha256 01:22:16 |
| A_medium_s1 | nohup_continue2.out | 2026-10-17 03:25:59 | 03:39:39 A medium seed 1: chunk 4/5, A1 released 0.376, K0/K1 so far {'K0': True, 'K1': False, 'K2_value': None} | 03:49:53 A medium seed 1: chunk 1/5, A1 released 0.177, K0/K1 so far {'K0': True} | actors_8000.msgpack 03:52:56; actors_final.msgpack 04:02:10; record.json 04:02:10; record.json.sha256 04:02:10 |
| A_medium_s2 | nohup_continue2.out | 2026-10-17 03:38:03 | (none: stopped before its chunk 1 ended) | 03:49:53 A medium seed 2: chunk 1/5, A1 released 0.153, K0/K1 so far {'K0': True} | actors_8000.msgpack 03:52:58; actors_final.msgpack 04:02:12; record.json 04:02:12; record.json.sha256 04:02:12 |
| A_medium_s3 | nohup_continue2.out | 2026-10-17 03:38:05 | (none: stopped before its chunk 1 ended) | 03:49:54 A medium seed 3: chunk 1/5, A1 released 0.178, K0/K1 so far {'K0': True} | actors_8000.msgpack 03:52:58; actors_final.msgpack 04:02:09; record.json 04:02:09; record.json.sha256 04:02:09 |

- **First reclaim.** `nohup_continue.out` ends at 21:56:21 (16 Oct), but the job logs run on to 21:59:38, so the container went down at about 22:00. No job finished in those three minutes, which is why the runner wrote nothing more. The run resumed at 00:59:38 (`nohup_continue2.out`), skipping the 27 finished jobs.
- **Second reclaim.** The last job-log line is at 03:39:39 (17 Oct). The run resumed at 03:44:49 (`nohup_continue3.out`), skipping 68 finished jobs. The gates were rerun at 03:45:13 and all passed, including the 45-of-45 byte-identical sandbox code copy.
- **Redone from scratch.**
  - Every rerun restarts at the first step (room checks: cloning step 5,000; A: chunk 1). Its numbers equal the first attempt's step by step (the same losses, and A1's shares 0.177, 0.333, 0.347 and 0.376), as a deterministic restart should.
  - Every file in the six folders has an mtime inside its rerun window. The room stage writes its files only at the end.
  - A_medium_s1's first attempt had written `actors_8000.msgpack` at about 03:33. The rerun rewrote it at 03:52:56, and the record's `actors_8000_sha256` matches the current file. arm_a.py never reads an existing actor file.
  - A_medium_s2 and A_medium_s3 left nothing behind. No A2 job was interrupted, and there is no `state/quarantine`.
- **No finished record was overwritten.**
  - `job_times.tsv` has 228 rows for 228 distinct jobs, all with exit 0, so no job finished twice.
  - Each of the 822 files this run created under `runs/` has its mtime inside the single window of the job that owns it.
  - The inputs of SCREEN_seed0.json (22), KILLS.json (71) and results.json (106) all still have the SHA-256 recorded in them.
  - The 820 files in the EXPLORATORY label (in both `runs/` roots) match their listed SHA-256, and no file created under `runs/` is missing from the label.
- **Sidecars and hashes.** 265 `.sha256` sidecars checked, with 0 mismatches. They cover the arm, eval, A2/B2/C2 and package stage records, plus KILLS.json, results.json and SCREEN_seed0.json. All 65 arm records carry the label and the frozen digest. Every actor file (A, C, F0, A2, B2, C2) matches the SHA-256 in its record.
- **The final untouched check passed.**
  - `state/untouched.json` (checked at 2026-10-17T05:55:43+0000, logged at 05:56:21) says all_untouched = True. Sandbox: changed 0, removed 0, new outside `runs/` 0, new under `runs/` 822 (820 files plus the 2 label files). Repository: changed 0, removed 0, new 0.
  - My recheck now: all 536 sandbox and 240 repository baseline files are hash-identical to the baseline. No new sandbox file exists outside `runs/`, and no new file exists under the repository's read roots. 0 files under `/home/user/bca` (`.git` excluded) were modified after the run's start marker (2026-10-16T18:40:53+0000).
  - The sandbox holds 7 `__pycache__` folders. All predate the run (15-16 October), and none holds a file newer than the start marker.

**Verdict:** I found no partial-file reuse, no overwritten record and no failed integrity check.

## Minor points (none changes a decision)

1. **KILLS.md's B rule block is misleading.** KILLS.md (and KILLS.json `B.rules` and `C_rules.guard_void.by_B_rules`) shows B's K1 as `{"where": [], "fires": false}` and K0/K4 with `mean: null`. Those dicts cover only seeds 1-4, whose maps were never built. B's actual stop, K1 at the screen, appears only in the 'Stopped by: K1' line and in `B.stops`. A reader can be confused by 'K1 fires: false' sitting next to 'Stopped by K1' and `by_B: true`.
2. **B's stop line omits the seed.** It reads 'Stopped by K1 on medium-replay; medium; medium-expert', but the frozen form is '<quality, seed>'. All three are seed 0.
3. **RESULTS.md, arm C (outside the kill checks, for the wording check).** The B2-B4 rows show numbers with 'Met: no', but they are not scored (results.json: None). The sentence says '(KB0 failed or the guard is void)', yet KB0 held and only the guard-void rule applies. The C sentence ('Halving the pull, guarded by BCA's ceiling, met B1 ...') is not one of section 6.5's frozen sentences, and it names BCA's ceiling as the guard although the guard is void.
4. **The first reclaim was later than the runner log suggests.** The runner log for the first reclaim ends three minutes before the jobs stopped, so the reclaim was at about 22:00, not 21:56. There is no 'runner was stopped' line and no untouched check at that point, as expected for a killed container. The final untouched check covers it.
5. **Scope of re-execution.** K1 and KF1/KF2 were verified from recorded values, not re-executed (see 'How this was checked').

Files: `verify/kills_check.py`, `verify/kills_check_raw.json`, `verify/spot_check.py` and `spot_check.json`, `verify/spot_k0_kb0.py` and `spot_k0_kb0_k0.json` / `spot_k0_kb0_kb0.json`, `verify/spot_maps.py` and `spot_maps.json`, and this file's generator `verify/make_kills_check_md.py`.
