# Step 1 of the restructured BCA: the decisive test (pre-specification)

**Exploratory, designed after seeing seeds 0-4; written before any step-1 critic is trained.**

This file says in plain words what `prespec.json` fixes in machine-readable form. If the two ever differ,
`prespec.json` is what the code reads. Both are frozen by `freeze.py` (SHA-256 in `prespec.json.sha256` and
`code_manifest.json`); the training and evaluation scripts refuse to run if any frozen file has changed, and
`freeze.py` refuses to run again once any official step-1 output exists.

It sits outside the gate. It changes no pre-registered rule, and no file of version 6 or version 7. It writes only
under `runs/shift_explore/restructure/step1/`.

## What was approved

The user's words (14 Oct 2026):

> Train 10 fixed-controller critics (seeds 0-4, both worlds) on the gapped log without its surviving full-throttle
> rows, using the exact gate protocol (about 70 CPU-minutes). Then, read-only, recompute the certificate, the height
> and the deployment check from those rows' held-out residuals. Go/no-go, fixed now: the certificate is on in 5 of 5
> stalling-world seeds and off in 5 of 5 no-stall seeds; the deployment net is at least 90; with outcomes shuffled,
> the net is at most half of that; the plain rank test is reported beside it. If any fails, BCA's own machinery
> cannot carry this in ThrusterChain; report BCA as no better than a range limit here and move the claim to a mixed
> world.

Nothing else is approved: no actor training, no policy test, and no band fit other than BCA's posterior on the
residuals.

## Why this test

The redesign makes BCA read outcomes. Each recorded step gets a score: how much the critic over-predicted what that
step actually delivered. Steps are then split into full-throttle steps and the rest. BCA's own Bayesian conformal
posterior asks whether the critic is credibly more over-optimistic on the full-throttle steps. If it is, caution
switches on for that world.

In-sample, using critics that had trained on those steps, this rule netted 106 points across the two worlds. But
in-sample residuals fade as a critic fits the rows it trained on. Step 1 therefore scores the full-throttle steps
with critics that never saw them.

## Data

**What the gate's critic trained on.** The gate's fixed-controller critic (`stall_aware_gap`) trained only on the
gapped training rows:

- training-split episodes 0-499, minus the gate's gap removal;
- 42,500 rows per seed and world, the surviving full-throttle rows included;
- inputs normalised with the mean and standard deviation (+0.001) of those rows' observations.

It never trained on the calibration split (episodes 500-699, which holds BCA's calibration bank: 5 rows per episode,
827-851 rows after the gap removal). It never trained on the evaluation split (episodes 700-999).

**The gap rule (finds the survivors from commands alone).**

1. Throttle index of a row: `t = max over the four thrusters of |command|`.
2. Take the gapped training rows whose `t` lies strictly above its 99th percentile (numpy's default quantile).
3. Sort them. The largest step between neighbouring values is the gap: `t_lo` is its lower end, `t_up` its upper end.
   If two steps tie, the first (lowest) wins.
4. The rule holds only if the gap is at least 0.05 wide and at least 9 rows lie at or above `t_up`. If it does not
   hold, that seed-world has no full-throttle group and its certificate is off.
5. Grouping of any row: full-throttle if `t > t_lo`, otherwise the rest. Rows tied at `t_lo` go to the rest.

No reward or next state is read. On the design's seeds the rule found the gap at 0.78-0.79 in all 10 seed-worlds,
with 28-59 survivors. `prespec.json` lists the exact `t_lo` and survivor count for each seed-world, and the code
refuses if the rule finds anything else.

**What each step-1 critic trains on.** The gate's gapped training rows of that seed and world, minus its survivors.
Everything else is kept from the gate:

- the bank and the evaluation rows are unchanged;
- the input normalisation is unchanged: the gate's, computed from the gapped rows including the survivors'
  observations (observations only, no outcomes).

**Which residuals are compared. Both groups are held out from the critic that scores them.**

- **Full-throttle group:** the survivors (removed from this critic's training), plus the 0-2 bank rows past `t_lo`.
- **Rest:** the bank rows at or below `t_lo`, 825-850 rows. No critic ever trained on them.

One critic per seed-world is enough to hold out both groups.

> **Flag for the user: the two groups come from different episodes.** The survivors come from training episodes,
> where their neighbouring steps stayed in training. The rest comes from calibration episodes that no critic saw.
> Holding out a comparison group from the training episodes would mean removing more rows, which the approval does
> not cover. The no-stall world is the control: there, the survivors' outcomes are ordinary. An artefact that made
> the full-throttle group look worse would switch the certificate on there, and condition 1 would fail. An artefact
> in the other direction would make the stalling world harder to certify, not easier.

## Training

The call is the gate's own:

```
learners.fixed_critic(prepared, 'stall_aware', p.config.learn, seed, placebo=False, out_dir=...)
```

`acceptance.get_critic(p, sd, 'gap', 'stall_aware')` makes the same call. Every argument is the gate's:

- 100,000 updates, batch 256;
- discount 0.98, soft target rate 0.005, learning rate 3e-4;
- target smoothing noise N(0, 0.2²) clipped at 0.5; policy delay 2 (the fixed controller's actor step is discarded);
- the gate's seed rule. The critic's initial parameters equal the gate critic's, and each record checks this.

The controller is the stall-aware controller `clip(-K x, -0.8, 0.8)`.

The only change is the prepared rows, built as `shift.Removal('gap_minus_survivors', ...)` and passed through
`learners.prepare` with the gate's normalisation. Because the number of rows changes, the random batch rows differ
from the gate's run. That follows from removing rows.

**Runs.**

- Each training runs pinned to one CPU (`taskset`), with `OMP_NUM_THREADS=1`, `JAX_PLATFORMS=cpu` and no bytecode.
  The gate's numbers depend on the thread count.
- Output: `runs/shift_explore/restructure/step1/critics/<world>/seed<k>/`, write-once.
- Each record holds the file's SHA-256, the survivors, the training-row hash, the comparison with the gate's critic,
  the CPU, the software and timings.

**Failures.**

- A critic whose training goes non-finite is reported and not replaced. Its seed-world then fails condition 1, so the
  verdict is NO-GO.
- A run interrupted before writing anything may be rerun; the computation is deterministic.
- Before any update is run, the script refuses unless the learner arguments it will use equal the gate critic's
  recorded ones. After training, the evaluation refuses any record whose arguments or initial parameters differ from
  the gate critic's, whose data checks are not all true, or whose training rows are not the gapped rows minus the
  survivors.

**Reproduction check (done while preparing; scratch folder).** The script was run for stalling world, seed 0,
without removing any rows. It reproduced the gate's saved critic bit for bit:

- file SHA-256 `3a99f159…54b0`;
- byte-identical by `cmp`;
- equal parameter-tree hash and equal initial parameters;
- 517 s on one CPU, against 402.7 s for the gate. Other jobs were running on the machine at the time.

## Scores

For each held-out row, with the step-1 critic's saved parameters:

`s = min_h Q_h(o, a) − [ r + 0.98 · (1 − done) · min_h Q_h(o′, c(o′)) ]`

- **Sign.** Positive means the critic predicted more than the one-step outcome delivered (over-optimism). `s` is
  minus the TD residual.
- **Heads.** The minimum of the two heads, at both ends.
- **Next command.** `c(o′)` is the stall-aware controller's command, without noise, as in the calibration design.
- **Done.** `done` is always False in ThrusterChain.
- **Units.** Value units. No rescaling is needed.

## Certificate

1. BCA's own posterior, `calibration/wbcp.calibrate`, runs separately on the full-throttle group and on the rest:
   - miscoverage 0.1, credibility 0.95, 1,000 draws, uniform weights;
   - fixed random streams: `default_rng([20261015, 9700, seed, w, 1])` and `[..., 2]`, with w = 1 for the stalling
     world and 2 for the no-stall world.
2. **Statistic.** The 5% quantile of the posterior gap (full-throttle crossing minus rest crossing), over all
   1,000 × 1,000 pairs of the two posteriors' draws.
3. **On** if that 5% quantile is above **0.10 value units**. Otherwise off.

**What was seen before this was fixed.**

- **In-sample, the design's form.** This was the full-throttle posterior against the rest's plain 90% quantile, at
  margin 0. The 5% bound was +0.28 to +0.46 in the stalling world and −0.015 to −0.029 in the no-stall world. The
  no-stall posterior median was up to +0.025.
- **In-sample, step 1's form.** With both posteriors, the stalling world gave +0.26 to +0.46 and the no-stall world
  −0.016 to −0.058, with medians up to +0.025.
- **Held out, trace test bed.** These dumps cover seeds 0-2 and use the gate's critic. The rows were random sets of
  10-45 REMOVED full-throttle rows against the bank. The no-stall values reached +0.067. The stalling values were
  0.115 at the lowest (10 rows) and 0.42 or more with 30-45 rows.

**Why 0.10.** It is above every no-stall value seen anywhere (largest +0.067) and below every stalling value
(smallest +0.115). It is stricter than the design's margin of 0.

**Two changes from the design. Both make the rule stricter.**

- The rest group now also gets BCA's posterior, instead of a plain quantile.
- The margin is 0.10 instead of 0.

The design's form is still reported, for information.

## Height and the gated deployment check

**Height.** `R_full − R_rest`, where R is BCA's deployed threshold, `max(lambda_hat, lambda_hpd)`, of each group, in
value units. It is infinite when the full-throttle group cannot be certified. The height is reported. It is not part
of the go/no-go, and the deployment check does not use it.

**Actors and starts.**

- **Actors:** the saved no-caution actor `none` of version 6's Stage 2 policy test, read-only, SHA-256 checked:
  - `runs/shift_explore/stage2/policy/seed<k>/actors.msgpack` for the stalling world;
  - `.../seed<k>/no_stall/actors.msgpack` for the no-stall world.
- **Starts:** the first 200 of each situation's fixed bank, `thruster_chain.bank_starts(situation, 1000)[:200]`.
- **Situations:** strong current (scored) and recorded conditions (reported).
- **Episodes:** 100 steps. Points are computed with the gate's anchors.

**Rule when the certificate is on.**

1. Each step, take the actor's command (clipped to ±1).
2. If its largest thruster magnitude is above `t_lo`, clip it to the recorded range. The recorded range is the
   per-thruster 0.5% and 99.5% quantiles of the gapped training commands, Stage 2's fallback.
3. Otherwise, run the command unchanged.

When the certificate is off, the check never fires.

**Random switching.** Reported beside the check: fire at the check's own rate, with coin
`default_rng([20261014, seed, world index, situation index, 993])`.

**Net across worlds.**

- **Stalling gain:** the mean over seeds 0-4 of (points with the check minus points without), in the stalling world,
  strong current.
- **No-stall loss:** the mean over seeds 0-4 of (points without the check minus points with), in the no-stall world,
  strong current.
- **NET** is the stalling gain minus the no-stall loss.

**Plain consequence.** The rollouts do not depend on any critic. Only the certificate decides whether the check
runs. So NET can take only a few values, and the code checks that its rollouts reproduce the design's numbers
exactly:

| Certificate pattern | NET |
|---|---|
| On in all 5 stalling seeds, off in all 5 no-stall seeds | 106.07 |
| One stalling seed missed | 83.4-85.9 |
| One no-stall seed falsely on | 92.5-94.7 (but condition 1 fails) |
| On everywhere (for information) | 44.4 |

In practice, condition 2 can fail only together with condition 1.

## Shuffle control

**What is shuffled.** The outcomes, as carried by the scores. Within each seed-world, the held-out scores of the two
groups are pooled and dealt back at random. Group sizes and commands stay fixed.

**How.**

- Permutation j uses `default_rng([20261015, 9800, seed, w, j + 1])`, for j = 0..199: 200 permutations.
- Every permutation uses the same posterior streams as the real certificate.
- Permutation j sets every seed-world's certificate. That pattern gives a shuffled net.

**Statistic.** The **mean** shuffled net over the 200 permutations is compared with half of NET. Also reported: the
95th percentile, the maximum, the share of permutations reaching NET, each seed-world's on-rate and, for
information, each seed-world's median shuffled height.

## Rank test

Plain one-sided Mann-Whitney U on the same held-out scores:

`scipy.stats.mannwhitneyu(full_throttle, rest, alternative='greater')`

It tests whether the full-throttle group's overestimates are stochastically larger. U, p and AUC are reported for
each seed-world beside the certificate. For information, it also reports the net that would follow if "p < 0.05"
were the switch.

Its value decides nothing. If it gives the same yes/no as the certificate, the claim stays "BCA's residual score
reads outcomes", not "the Bayesian aggregation adds value".

## Go / no-go (as approved)

- **C1.** The certificate is on in 5 of 5 stalling-world seeds and off in 5 of 5 no-stall seeds.
- **C2.** NET is at least 90.
- **C3.** With outcomes shuffled, the mean shuffled net is at most half of NET. ("Half of that" is read as half of
  the net. The other reading, at most 45, half of the 90-point bar, is reported beside it for information; it is the
  stricter one whenever C2 holds.)
- **C4.** The plain rank test is reported beside the certificate. This is reporting only.

**GO** if C1, C2 and C3 all hold. BCA's own residual posterior, fed held-out residuals, then separates the worlds
and carries the cross-world gain on seeds 0-4. This is still exploratory: it licenses asking for step 2 (the policy
pilot), not a claim.

**NO-GO** if any of C1-C3 fails. BCA's own machinery cannot carry this in ThrusterChain. Report BCA as no better
than a range limit here and move the claim to a mixed world. There is no retuning of the margin, the groups or the
residual on these seeds.

## Information only (not part of the go/no-go)

- **I1. Trace test bed dose-response (the judges' check).** The step-1 statistic and margin, applied to the saved
  trace critics:
  - seeds 0-2, both worlds;
  - shares 0 (the gate's gapped critic), 0.5, 1, 2, 3, 5 and 10%;
  - full-throttle group: each critic's training rows past `t_lo` (survivors plus added rows, in-sample);
  - rest: (a) its own training rows at or below `t_lo`, in-sample, the judges' form; (b) the bank, held out;
  - also whether the gap rule holds on the trace log's own commands.

  Trace logs are rebuilt and checked against `trace_rows.json`, and critics against `lock.json`.
- **I2.** The design's form (the rest's plain 90% quantile) on the held-out scores, at margins 0.10 and 0.
- **I3.** A noisy-target residual: 32 fixed draws of BCA's smoothing noise, with the same certificate.
- **I4.** In-sample references from the gate's own critic. The design's in-sample numbers must reproduce to 1e-9,
  or the evaluation refuses.
- **I5.** The fade: each survivor's overestimate under the gate's critic (in-sample) and under the step-1 critic
  (held out).
- **I6.** The forced-on net and the rank-test-switched net.
- **I7.** Group sizes, means and 90% quantiles.

## Files

| File | Role |
|---|---|
| `prespec.json`, `prespec.json.sha256` | the frozen specification |
| `code_manifest.json` | SHA-256 of every file below |
| `step1lib.py` | the gap rule, the score, the certificate, the net, the shuffle, the go/no-go, output guards |
| `train_critics.py` | one critic per call; official or scratch mode |
| `evaluate.py` | everything after training; writes `results.json` and `results.md`, write-once |
| `run_step1.sh` | unit tests, then 10 trainings on 4 CPUs, then the evaluation. `STEP1_SCRATCH=<dir>` runs a tiny smoke version outside the repository |
| `test_step1.py` | unit tests on synthetic data |
| `freeze.py` | writes the two hash files |

**To run step 1**, detached, so that closing the session cannot stop it half-way:

```
setsid nohup sh /home/user/bca/outputs/shift/beyond-the-data/restructure/step1/run_step1.sh </dev/null >/dev/null 2>&1 &
```

Progress is in `runs/shift_explore/restructure/step1/logs/run.log`; the verdict is the last line of that file and the
heading of `results.md`.

- Training: about 70-85 CPU-minutes, about 25-30 minutes of wall time on 4 CPUs.
- Evaluation: about 10 minutes on one CPU.

## Review notes (independent review, 14 Oct 2026, before any official step-1 critic was trained)

Changes made, and why:

1. **Protocol check before training** (`train_critics.py`). The script now refuses before running any update unless
   the learner arguments equal the gate critic's. Before, the comparison was only recorded after training, and
   nothing refused a mismatch, although this file says the protocol is the gate's.
2. **Record checks enforced at evaluation** (`step1lib.record_problems`, `evaluate.py`). An official record is now
   refused unless its arguments and initial parameters equal the gate critic's, all its data checks are true, it ran
   on one CPU, and its world, seed and training rows match. Before, only the mode, the hashes, the update count and
   the log name were checked.
3. **Information-only items can no longer block the verdict** (`evaluate.py`). An error inside the trace test bed
   check (I1) or the noisy-target check (I3) is now recorded in `results.json` and the evaluation continues. Before,
   any error there stopped the evaluation, and once official critics exist `freeze.py` forbids any code change, so
   the verdict could have become impossible to compute. The decisive items and the integrity checks still refuse.
4. **No premature write-once NO-GO** (`evaluate.py`). `--allow-missing` is refused while `run.lock` exists, so the
   evaluation cannot be run, and its write-once result fixed, while trainings are still going.
5. **Lock kept when the runner is killed** (`run_step1.sh`). Before, a killed runner released its lock while the
   trainings it had started kept running, so a second launch could start duplicate trainings into the same
   write-once folders. The script header and the section above now give a detached launch command.
6. **Report written safely** (`evaluate.py`). If `results.md` cannot be formatted, a short fallback is written
   (`results.json`, which is written first, holds everything).
7. **Two reported items added, both information only.** The median shuffled height, which matches the design's own
   shuffled-height control. And condition 3 under its other reading (at most 45).
8. **Tests.** `test_step1.py` has 4 new tests: record validation, shuffled heights, and the other reading of
   condition 3. 24 tests now, all pass.

Checked, not changed:

- **The four conditions match the approval word for word.** C2 is decided by C1 in practice (see the table above).
  C3 uses the mean over 200 fixed permutations of the real net's own calculation; at random, the certificate is
  almost always off, so C3 can only fail if shuffled certificates are on in the stalling world in about half the
  permutations or more.
- **No free parameter is left.** Gap rule, groups, residual, posterior settings, random streams, statistic, margin,
  permutations, actors, starts and clip point are fixed and checked by code. The gap position and the clip range read
  commands only, and the rollouts must reproduce the design's numbers to 1e-9 before anything else counts.
- **Both groups are held out.** The survivors are removed from training (checked by the record and again at
  evaluation). The rest group is calibration-split bank rows, which no critic trained on. The remaining asymmetry
  (survivors come from training episodes) is flagged above. Its direction is not known. The no-stall world is its
  control: an artefact that made the full-throttle group look worse would switch the no-stall certificate on and fail
  condition 1.
- **The shuffle and the rank test test what they say.** Scores are pooled and re-dealt within each seed-world, with
  the real certificate's posterior streams. This breaks the link between command group and outcome, the same way the
  design's own controls did. The rank test is plain one-sided Mann-Whitney on the same scores.
- **Reproduction.** The writer's scratch critic (stalling world, seed 0, no rows removed) is byte-identical to the
  gate's (`cmp`). The training code it used has not changed since, apart from fix 1, which runs before training.
- **Outputs.** Official outputs go only under `runs/shift_explore/restructure/step1/`. Every script turns bytecode
  writing off before any import, and `run_step1.sh` exports `PYTHONDONTWRITEBYTECODE=1`.

Limits, stated, not changed:

- **The margin's evidence.** The held-out values that placed 0.10 came from the gate's critic, which had trained on
  the survivors. A step-1 critic has no full-throttle row at all, so how it extrapolates to full throttle in the
  no-stall world has not been seen. That is exactly what condition 1 tests.
- **In-sample reference with the official random streams** (I4, from the writer's scratch evaluation): +0.22 to
  +0.46 in the stalling world and -0.040 to -0.060 in the no-stall world. The +0.26 lower end quoted above came from
  other random streams. 0.10 stays below every stalling value.
- **Compute beyond the 10 approved trainings, all verification, in scratch folders.** These runs produce no new
  evidence about the outcome:
  - the writer's reproduction critic (about 9 CPU-minutes);
  - two smoke runs of 200 updates per critic;
  - this review's second reproduction and smoke run (see below).

Verification by this review, with the final code (scratch folder
`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/step1_review/`):

- **Second reproduction.** No-stall world, seed 3, no rows removed, pinned to one CPU. It passed the new protocol
  check before training. It reproduced the gate's saved critic byte for byte (`cmp`; file SHA-256
  `8ab6190c...1065813`; equal parameter-tree hash, initial parameters and arguments). It took 549 s, against the
  gate's 372 s, while a smoke run shared the CPU for its first 3 minutes.
- **Smoke run.** `STEP1_SCRATCH=... STEP1_UPDATES=200 sh run_step1.sh` at full evaluation settings. It finished with
  exit 0, all 40 built-in checks true and no information-only error. Its numbers equal the writer's smoke run
  exactly (net 69.8, mean shuffled net 1.3), so the review edits changed no computation. Its NO-GO means nothing:
  200-update critics are untrained.
- **Refusals.** Official-mode training without `taskset`, official evaluation with no critics, and official
  `--updates` all refused without writing anything.
- **Repository.** `runs/shift_explore/restructure/` still does not exist. No new `__pycache__` appeared in the
  repository. `git status` shows only this folder.
- **Unit tests.** 24 of 24 pass.
