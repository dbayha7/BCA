# BCA lower-edge consumer (new method version), exploratory on seeds 0-4, confirmatory on fresh seeds 5-9

**Pre-registration for Stage 2 of the plan, the flags claim.** The machine-readable twin is
`runs/shift_explore/stage2/prereg.json`. A byte-identical copy is committed in
`outputs/shift/beyond-the-data/stage2/`. You approve that file's SHA-256 (its fingerprint). This document is pinned by
the code manifest, so the fingerprint you approve covers this text, the code and every number below.

The claim under test, in your words: **"BCA flags when the actor relies on commands the data do not support, and
acting on that flag repairs the harm while keeping TD3+BC's gains."** The other claim, that BCA identifies the harm
itself and not just novelty, is not tested here. It needs the trace test bed (plan section 4.6).

---

## 0. Status, and what this was written after

**Seeds 0-4 are EXPLORATORY.** Two reasons:
- The room test's futility look on seeds 0 and 1 was seen before this was written. It was seen by the orchestrating
  agent, not by the writer of this document. The file is `runs/shift_explore/room/futility.json`, dated
  2026-10-13 04:19 UTC.
- The whole plan was designed after the gate's own results on seeds 0-4.

So everything on seeds 0-4 is post hoc with respect to seeds 0 and 1, and nothing on seeds 0-4 supports a claim.

**Seeds 5-9 are CONFIRMATORY.** They use fresh seeds with new logs, critics and cloned actors, all made under this
pre-registration with every setting frozen from seeds 0-4. Only seeds 5-9 carry the confirmatory claim.

**What this was written after, exactly:**
- **The room test:** written, reviewed, amended and launched before any of its outcomes existed. Its amendments are
  in its `changes_before_any_run`.
- **The futility look on seeds 0 and 1:** seen by the orchestrating agent. The writer of this document did not read
  any of the following:
  - `futility.json`;
  - any room-test `values.json`;
  - any Stage 1 outcome: stand-in, deployment or detection (only folder listings, file names and times were seen).
- **Read by the writer:**
  - the gate's stop report;
  - the plan and its evidence (mechanism and anatomy notes on seeds 0-4);
  - the room test's specification.
- **Change A1 of the room test was known and is applied here.** It says a penalty must be read relative to the
  recorded command at the same state.
- **Version 4 was rewritten after a review of version 3**, which was never approved. The reviser read the review's
  findings, this document, the code, the plan and its evidence, and the room test's specification. The reviser did
  not read `futility.json`, any room-test `values.json` or any Stage 1 outcome either. Section 16 lists what changed.
- **Version 5 changes the approval route, at your choice, and takes up the re-locked detection script** (section
  17). Version 4 was never approved. Its reviser read this document, the Stage 2 code, your choice and the detection
  script's code manifest. The reviser did not read `futility.json`, any
  room-test `values.json` or any Stage 1 outcome. Only folder listings and file names were seen, plus one line of
  the Stage 1 results file, found by a search for "approve", which mentions only Stage 2's approval.
- **Version 6 re-pins the stand-ins' re-locked code and tightens the approval route** (section 18), after a review of
  version 5, which was never approved. Its reviser read the review's findings, this document, the Stage 2 code, and
  the stand-ins job's code, the changes in its specification and its code manifest. The reviser did not read
  `futility.json`, any room-test `values.json` or any Stage 1 outcome. Only folder listings, file names and file
  times were seen.

---

## 1. Words used

- **BCA (Bayesian Conformal Aggregation):** the add-on under test. It puts a calibrated band around the critic's
  one-step error.
- **TD3+BC:** the host learner that BCA is added to. It trains the actor to climb the critic, plus a pull toward the
  recorded command (the **cloning pull**).
- **Critic Q(s, a):** rates command a in state s. Q1 is its first head.
- **Actor pi(s):** picks the command.
- **One-step error:** r + 0.98 · Q(next state, next command) − Q(s, a). This is what BCA's band is fitted to cover.
- **Scale network eta(s, a):** BCA's small network that predicts how large the one-step error is at (s, a). After a
  refresh it is frozen; the frozen copy is eta_f.
- **Unit u_f:** BCA's running size of the one-step error, frozen at a refresh.
- **Threshold R:** BCA's calibrated multiplier, from its threshold rule (weighted Bayesian conformal prediction) on
  the calibration bank. R is +infinity when nothing can be certified.
- **Calibration bank:** recorded rows withheld from training. About 850 remain after the removal.
- **Width U(s, a):** the band's half-width, in critic value units.
- **Unsupported command:** any thruster past 80% throttle.
- **Recorded range:** per thruster, the 0.5% to 99.5% quantiles of the gapped training commands.
- **Points:** 0 is the recording controller and 100 the stall-aware controller.
- **No-caution actor:** check 2's actor trained against the gapped critic, called `none` in the code.
- **Strength:** the root-mean-square change of the command from the no-caution actor on 5,000 held-out states.
- **Process-hour:** one CPU busy for one hour.

---

## 2. The method version and its exact formula

**The signal is BCA's, unchanged.** It is fitted by `algorithms/td3_bc_bca.py`'s own functions, imported read-only
and never edited:
- the scale fit (`fit_scale`);
- the refresh (`refresh`).

It uses the gate's settings:
- miscoverage 0.1, credibility 0.95, 1,000 draws;
- scale learning rate 0.001, coverage sharpness 20, width penalty 0.005, unit averaging 0.99.

The width is BCA's own:

    U(s, a) = R · u_f · max(eta_f(s, a), 10^-6)

**What changes is the consumer, so this is a new method version.** Its results speak for this version only, not for
BCA as published (a dose of at most 1.5 on the cloning pull), and the reverse holds too. The changes:
1. **The width is read at the actor's own command.** The gradient flows through the frozen scale network's command
   input, and its parameters get no gradient. This breaks the design sentence "host optimisation does not
   backpropagate through the calibrator" (`ALGORITHMS.md`, lines 127-130).
   - **In the weighted version the gradient goes further.** It also flows through the frozen classifier's command
     input, log w(s, a), in both of the weighted width's terms: the tabulated threshold R_c(w) and the softplus
     refusal term. So host optimisation also backpropagates through the threshold calculation, which the same
     sentence forbids. Only the classifier's parameters are frozen.
   - Why: the threshold's rise with w is BCA's own way of becoming wary away from the data. Holding R_c(w) fixed in
     the gradient would leave only the scale network and the refusal term to steer the actor. (Section 14 offers
     the other choice.)
2. **It is used in critic value units, with no 1.5 cap and no dose.**
3. **"Cannot certify" means maximum caution.** Where R is +infinity, the width is U_max, never 0.
   - R_c is R when finite, else the larger finite one of BCA's two candidate thresholds.
   - U_max = R_c · u_f · max over the bank rows of max(eta_f, 10^-6).
   - **This acts command by command only in the weighted version.** With uniform weights it is all or nothing, and
     it changes nothing (see below).
4. **Change A1.** The primary penalty is read relative to the recorded command at the same state.

**Primary penalty (paired, one-sided):**

    P(s_i, a) = [ U(s_i, a) − U(s_i, a_i) ]_+        ([x]_+ means max(x, 0); a_i is row i's recorded command)

It is zero at the recorded command and never rewards a command.

**The actor's objective (TD3+BC host, alpha = 5):**

    L(phi) = −lambda · (1/B) Σ_i [ Q1(s_i, pi(s_i)) − k · P(s_i, pi(s_i)) ] + (1/B) Σ_i (1/4) Σ_j (pi(s_i)_j − a_ij)²
    lambda = alpha / stopgrad( (1/B) Σ_i |Q1(s_i, pi(s_i))| )        computed on Q1 alone

- **k = 1, primary:** the lower edge of the band, fixed by meaning and never tuned on returns. k = 2 is a declared
  secondary arm.
- **Declared secondary arm, unpaired:** Q1(s_i, pi(s_i)) − 1 · U(s_i, pi(s_i)). This is the plan's literal form.
- **Unchanged:** the cloning pull keeps weight 1 on every row, and the critic loss is unchanged.

**Uniform weights are primary.** This is BCA as written. Its threshold is one number for all inputs, so "cannot
certify" is all or nothing. If it ever happened, every width would be U_max, and "maximum caution" would do nothing:
- the paired penalty would be 0, with zero gradient;
- the unpaired penalty would be a constant, with zero gradient;
- in the deployment check, tau would equal U_max, so the check would never fire.

With about 850 bank rows it cannot happen, because BCA's own check passes for 36 rows or more.

**The weighted version is the declared fallback.** It is used only if the detection test's direction or size mark
fails with uniform weights on seeds 0-4, and is never chosen on returns. Its width:

    U~(s, a) = R_c(w(s, a)) · u_f · max(eta_f(s, a), 10^-6) + U_max · softplus(log w(s, a) − l*)

- **The weights w(s, a):** the likelihood ratio between the actor's pairs and the recorded pairs, from a classifier.
  - Its definition restates the stand-in classifier ratio's: the shape of BCA's scale network, with the logit as
    log w.
  - Target pairs: the recorder's own scatter re-centred on the actor's command, (s_i, clip(pi(s_i) + a_i − c(s_i))),
    where c is the cloned actor.
  - Fit: a balanced logistic loss, 2,000 Adam steps of 512 rows, from fixed seeds.
- **When the classifier is refitted:**
  - in the detection test, once, with the no-caution actor;
  - in the policy test, before every 4,000-step chunk, with each arm's own actor;
  - in the full host, at every refresh.
- **The threshold:** BCA's own threshold rule with bank weights w_j and test mass t.
  - It is tabulated on 48 test masses evenly spaced in log t, using the same posterior draws for every t.
  - R_c(t) is R(t) with +infinity replaced by its largest finite value, read by straight-line interpolation in log t.
  - l* is the log of the smallest tabulated t at which R is +infinity.
- **U_max:** the largest finite width over the bank's threshold half.
- **The bank is split in two.** One half sets the threshold; the other only checks coverage, because the actor trains
  against the threshold.
- **eta_f and u_f** are the uniform fit's, from the same seed.

---

## 3. The detection test (plan 5.2)

The band is fitted and read. No actor is trained with BCA.

**Fit**, per seed and world:
- **Primary: the frozen fixed-controller gapped critic.** This is check 2's critic, the one the policy test trains
  against.
  - The procedure is `learners.bca_band`'s: 20,000 scale-fit steps, then one refresh on the gapped bank.
  - The one-step target uses the stall-aware controller's next command plus TD3's smoothing noise (standard deviation
    0.2, clipped at 0.5), as that critic was trained.
  - Only the critic's online parameters were saved, so the frozen critic is its own target.
- **Reported: as written, on check 3's frozen co-trained plain TD3+BC critic** (stalling world).
- **Plumbing check:** BCA's own bank scores are recomputed and must give its threshold exactly.

**Readout.** The width is read only at the locked detection script's pair sets T1-T4 (see section 8), through the
script's declared hook for BCA. Widths and labels are hashed separately and joined only afterwards by the script.
- **Seeds 0-4** reuse the rehearsal's own locked pair sets, labels and stand-in scores, all made before any BCA run.
- **Seeds 5-9** get their own, made by the same locked code in Stage 2's folder. The labels are computed in a process
  that holds no signal code.

**Marks.** Each is pooled (the mean over seeds of the per-seed value) and must also hold in at least 4 of 5 seeds.
Every seed is needed:
- a seed that failed on record fails every mark. That is a band fit that failed, or a fresh seed whose data checks
  or learners failed;
- a seed whose inputs merely do not exist yet stops the marks: they are written once, so the program refuses until
  every seed's width scores, tau, labels and stand-in scores exist.

The marks:
- **Direction:** on harmful past-80% pairs, the width at the command exceeds the width at its 80% cap on at least 75%
  of T1 pairs and at least 65% of T3 pairs. Chance is 50%.
- **Size:** the median over harmful past-80% T1 pairs of
  ([U(command) − U(recorded)]_+ − [U(cap) − U(recorded)]_+) / (the critic's over-preference of the command over its
  cap) is at least 1. Then k = 1 cancels the false preference.
  - This reads the penalty the actor actually feels (change A1).
  - The plan's raw (U(command) − U(cap)) / over-preference is reported.
- **Ranking:** a ranking score of at least 0.75 on T1 and T2 together, and at least 0.70 on T3. The ranking score is
  the chance that a random harmful pair scores above a random harmless one.
- **Calibration sanity:** 5% ± 1.5% of T4 rows lie above tau (section 5). Reported as a mark.
- **Selectivity:** flag rate at past-80% pairs, stalling world minus no-stall world. Reported only.

**Claim rule.** "BCA flags the actor's reliance on commands the critic cannot vouch for" may be claimed only if all
of these hold:
- direction, size and ranking pass;
- BCA's pooled ranking score is no more than 0.02 below the best data-only stand-in's, on T1 with T2 and on T3.

**The detection rivals** are five data-only stand-ins, scored by the same locked script on the same pairs:
1. distance to data;
2. behaviour density;
3. the classifier ratio;
4. two-head disagreement;
5. residual evidence: minus the mean one-step error of the 10 nearest training rows.

The range hinge is not one of them (it is a hand-set limit), and neither is the perfect band (it is the sanity
check). This set differs from the policy test's rivals (section 8). All five must have finite ranking scores on
every seed. If any is missing or not finite, this part of the claim rule fails: a rival that cannot be scored counts
against BCA, as in the policy test.

**Only seeds 5-9 can carry the claim.** On seeds 0-4 every output writes the claim as empty and says "exploratory,
not a claim".

**Version decision.**
- If direction or size fails with uniform weights on seeds 0-4, the weighted version is declared and the detection
  test is repeated. No uniform-weight policy test runs.
- If the weighted version also fails, no BCA training arm runs. The deployment check and this report remain.

---

## 4. The policy test (plan 5.3)

**Protocol: the room test's, unchanged.** Its harness is executed from the exact bytes its code manifest pins:
- check 2's actor-only protocol on the frozen gapped critic;
- start from the saved cloned actor;
- alpha 5;
- 20,000 actor steps in 4,000-step chunks;
- check 2's stream of batch rows;
- the arms of one kind run side by side.

The room's runner is extended with two penalty kinds (uniform and weighted BCA). Every other kind runs through the
room's own program.

**Arms (stalling world):**

| Arm | Penalty | Role |
|---|---|---|
| none | 0 | check 2's saved actor (fresh seeds: their own) |
| BCA with k = 0 | 0 through the BCA program | must match `none` within 10^-5 on every parameter, else the seed stops |
| **BCA, k = 1** | P(s, pi(s)) | **primary** |
| BCA, k = 2 | 2 · P(s, pi(s)) | declared secondary |
| BCA, unpaired | U(s, pi(s)) | declared secondary |
| BCA, state-shuffled | [U(s_σ(i), pi) − U(s_σ(i), a_σ(i))]_+ | reported; σ is the room's fixed permutation |
| Perfect band | the room's one-step oracle [δ*(s_i, a_i) − δ*(s_i, a)]_+ | reference |
| Recorded-range hinge | 100 · Σ_j ([a_j − hi_j]_+² + [lo_j − a_j]_+²) | reported |
| Even extra pull, matched | cloning pull × (1 + ν) | gating control |
| Random surfaces 1 and 2, matched | random networks of the scale network's shape | gating controls |
| Push less everywhere, matched | c · mean_j pi_j² | reported |
| Best data-only stand-in, matched | section 8 | rival |

**Matching.** It is blind to the strength of BCA with k = 1, uses strengths only, and happens before any return.
- **Ladders:**
  - even pull: ν = 0.1, 0.25, 0.5, 1, 2;
  - the others: knobs 0.1, 0.3, 1, 3, 10, on the room's own slope scale, so the controls do not depend on BCA.
- **Then:** the room's blind ladder extension, then the gate's matcher (tolerance 0.01, at most 5 rounds).
- **The stand-in** uses its own 4-rung ladder and the same matcher.

**Lock before returns.** `lock.json` holds every actor's parameter fingerprint. Only then are exact returns computed:
- on the full 1,000-start banks of recorded conditions, strong current, big waves and big kicks;
- with the share past 80% per thruster;
- with shift cards for `none` and BCA with k = 1, reported.

The gate's report-card bug is fixed in these shift cards: they receive one situation's anchors.

**No-stall world**, at the stalling world's locked settings, not re-matched:
- none;
- BCA with k = 1, using the no-stall world's own band;
- the even pull at the locked ν;
- the range hinge;
- the perfect band;
- the stand-in at its locked coefficient.

**Marks** (strong current unless stated). Each is the pooled mean of the paired per-seed difference, and every seed is
needed:
- **M1:** BCA (k = 1) beats the matched even pull by at least 15 points, and is ahead in at least 4 of 5 seeds.
- **M2:** BCA (k = 1) beats each matched random surface by at least 15 points.
- **M3, keeps the improvement:**
  - recorded conditions: at least none − 3, and at least the cloned actor + 25;
  - big waves and big kicks: at least none − 3.
- **M4:** BCA (k = 1) is at least the matched best stand-in − 5.
- **M5:** the deployment check on the no-caution actor beats random switching at its own rate by at least 10 points,
  and costs at most 2 points in recorded conditions.

**Reported:**
- the no-stall cost of every arm;
- the share past 80% per thruster;
- the distance of visited pairs from the data (the shift cards);
- region strengths;
- the BCA penalty's shape past 80%.

**Runs only if:**
- a version passed detection direction and size; and
- the room test's pooled look passed its room, recovery and keeps-improvement marks (plan 4.8).

---

## 5. The deployment check (Variant B)

    a_exec(s) = pi(s)                                 if U(s, pi(s)) ≤ tau
                clip_j(pi(s)_j, lo_j, hi_j)          otherwise, and whenever U is not finite

- **tau:** the split-conformal 95% level of U at every held-out recorded row. That is every row of the calibration
  split that lies outside the removed region, outside the bank and outside the detection script's T4 rows.
  - T4 is left out so that the calibration-sanity mark reads T4 on rows tau never saw.
  - Its rate on 5,000 other held-out rows is reported.
- **lo_j, hi_j:** the recorded range.
- **The band:** the chosen version's band on the frozen critic of the actor being checked.
  - The no-caution actor and BCA with k = 1 use the fixed-controller critic's band.
  - Plain TD3+BC uses the as-written (uniform) band on its own critic.
  - The check waits for the detection test's version decision. It uses the uniform band only if neither version
    passed.
  - With uniform weights, a band that cannot certify has the same width everywhere, so tau equals it and the check
    never fires. This cannot happen with about 850 bank rows.
- **Compared with:**
  - never switching;
  - random switching at the check's own firing rate;
  - the fallback diagnostic's distance check, and random switching at its rate.
- **Setup:** all with the clip fallback, on the first 200 starts of each situation, in exact simulation, in both
  worlds.
- **Mark M5** is read on the no-caution actor. The others are reported.
- **It always runs**, even when no training arm does.

---

## 6. The full host (plan 5.4)

**What runs:** TD3+BC is co-trained with BCA's lower edge (k = 1, the chosen version) for 100,000 updates, critic and
actor together, on BCA's own schedule:
- the scale fit runs every update;
- the frozen reference is first refreshed after update 10,000, then every 5,000 updates;
- the width is 0 before the first refresh;
- U_max, and the weighted version's classifier and table, are recomputed at every refresh;
- the initial parameters and the batch stream are the same as plain TD3+BC's.

**Compared with:**
- **plain TD3+BC:** the gate's saved run, or the fresh run on seeds 5-9;
- **TD3+BC plus the range hinge:** matched blind to the BCA host's strength, measured against plain TD3+BC's actor.
  - Ladder: 0.1, 0.3, 1, 3, 10, extension × √10, then the gate's matcher.
- **In the no-stall world:**
  - the BCA host;
  - plain TD3+BC, trained here because the gate trained none there;
  - the hinge host at the locked coefficient.

**Check:** the hinge program with coefficient 0 must reproduce the gate's TD3+BC trainer for 2,000 updates within
10^-5. Results are reported, with the claim sentence's numbers. The run conditions are those of the policy test.

---

## 7. Confirmation on fresh seeds 5-9

**New data and learners, all with the gate's unchanged code** (no BCA):
- **Logs:** new logs of both worlds (base world, stall slope 1.5), with the gate's recorder settings and sizes.
- **Removal:** the gap, its removal, the gate's leak checks, and the world marks T1-T5, which are required.
- **Critics:** fixed-controller critics on the gapped log, the full log and the no-stall gapped log.
- **Cloned actors**, in both worlds.
- **Actors:** the no-caution actor and the full-log critic's actor, which defines the harm on these seeds.
- **Plain TD3+BC**, alpha 5.

**Frozen from seeds 0-4:**
- the version;
- k;
- every ladder, tolerance and rule;
- the best stand-in's identity;
- tau's rule.

**Marks:** the detection marks and claim rule, plus M1-M5, on seeds 5-9.

**A fresh seed that fails** its data checks, or whose learner fails, fails every mark. It is never replaced.
- The failure is written down: `data.json` with "passed: false", or `learners.json` with "failed: true".
- Every later phase refuses to use that seed.
- The detection marks count it as a failed seed, and the policy marks count it as a stopped seed.
- So the confirmatory pooled look can still be written.

---

## 8. The comparison set and the detection script

There are **two sets of rivals**, one for each test.

**The policy test's rivals** are the data-only stand-ins built in `scratchpad/ood/standins/` (Stage 1, plan 4.2):
1. the range hinge;
2. behaviour density;
3. the classifier ratio;
4. distance to data;
5. two-head disagreement.

You choose their form in the stand-ins test's own approval:
- "approve F1-F3": density, distance and two-head are read from the recorded command, as change A1 requires;
- "decline F1-F3": the plan's literal forms.

Stage 2 follows that choice, as the stand-ins' results file records it. **The best of the five** is chosen by
`select_standin` before any BCA run, from the stand-ins' own marks on seeds 0-4:
- among the five, take those that capture the room, and choose the one with the largest pooled matched gain over no
  caution in strong current;
- if none captures the room, choose the largest gain among all five.

**The detection claim's rivals** are the five stand-ins the detection script scores (section 3):
1. distance to data;
2. behaviour density;
3. the classifier ratio;
4. two-head disagreement;
5. residual evidence.

The range hinge is not among them. The best of them is the one with the largest pooled ranking score on the same
pairs, taken separately on T1 with T2 and on T3. All five are needed on every seed.

**The detection script** is locked by the other job. It did not exist when this contract was first written, so
`detect_contract.py` defines what it must expose. The locked script implements it and adds a declared hook for BCA's
width:
- the pair sets T1-T4;
- scoring of any signal without labels;
- labels from exact simulation;
- the measures, including the paired size;
- the stand-ins' signals.

`prereg.json` pins it by fingerprint. Stage 2 only reads the rehearsal's own pair sets, labels and stand-in scores for
seeds 0-4. If they do not exist, it refuses; it never builds or writes them in the rehearsal's folder.

---

## 9. Approval

**You approve one exact file.** Approval names `prereg.json`'s SHA-256, its fingerprint. Through the code manifest it
covers this document and every pinned script too, and nothing else.

**The only route this version accepts: a terminal on your own machine.** You chose it. This session gives you no
terminal, so you approve on your own computer and send the approval through your own git push.

1. Pull the branch:

       git fetch origin && git checkout claude/main-bca-connection-x4o46s && git pull

2. Read this document in `outputs/shift/beyond-the-data/stage2/`.
3. From the repository root, run:

       python3 outputs/shift/beyond-the-data/stage2/approve_local.py

   It needs Python 3.8 or later and nothing else. In order, it:
   - checks that `prereg.json` matches its fingerprint file, that the code manifest is the one `prereg.json` pins,
     and that every file in `code/`, this document and the script itself match the manifest;
   - refuses if an approval of this fingerprint is already there (it never overwrites one);
   - refuses unless you are typing at a terminal (nothing can be piped in);
   - prints the fingerprint, what approval allows, and a pointer to this document;
   - asks you to type the fingerprint's first 16 characters. Spaces at either end are ignored; otherwise they must
     match exactly;
   - writes `approval_<first 16 characters>.json` and its `.sha256` file in that folder.

   Any failure stops it, and nothing is written. The approval file holds the fingerprint, the characters you typed,
   the mode `local-terminal`, the UTC time, your computer's name, your user name, the git commit you were on, and a
   statement that the approval covers this exact pre-registration only.
4. Commit and push those two files with the commands the script prints. After the push, do not delete, edit or
   re-add them.

**One file per fingerprint.** Each version's approval file is named after its own fingerprint. If a new version
replaces this one before the import, for example because another job's pin moved, you get a new fingerprint. You then
pull, run the script again and push the new file. The earlier file stays where it is and approves only the earlier
version.

**Then, in this session.** The orchestrating agent pulls your commit and runs `approve.py import`. Import reads git
itself, with read-only queries; nobody types the commit details in. It refuses unless all of these hold:
- exactly one commit on the branch, as fetched from GitHub, added your approval file. That commit is the one
  recorded;
- the branch on GitHub still holds the file unchanged, and the file in that commit, the file in the folder and its
  fingerprint file all match;
- it is a `local-terminal` approval of the current `prereg.json`: the same fingerprint, and the characters typed are
  exactly its first 16;
- neither the commit's author nor its committer is this cloud session's own git identity;
- the `prereg.json`, code manifest, code copies, this document and `approve_local.py` stored in that commit are the
  current ones;
- no approval has been imported yet;
- every check at approval (below) passes.

Import then writes two files:
- `runs/shift_explore/stage2/approval.json`, the approval every BCA entry point checks. It holds your file verbatim
  and the commit details. It stays in this session, because `runs/` is not committed.
- `import_<first 16 characters>.json` and its `.sha256` file, in the committed folder. The record names your commit,
  its author and committer, and the fingerprint of the runs-side approval. The orchestrating agent commits and pushes
  it.

**No BCA run starts until both are in place.** Every BCA entry point checks, with git, that your commit is still on
the branch on GitHub with your file unchanged, and that the import record has been pushed there. So you can see on
GitHub which commit was imported, without relying on a message from the agent.

**What this route shows, and what it cannot.**
- Nothing in this session types or writes your approval file. It is typed at your terminal and arrives through your
  push.
- Import reads the commit details from git itself, and its record is on GitHub. **This added strength holds only if
  you check it:** open `import_<first 16 characters>.json` on GitHub and confirm that the commit it names is yours. A
  signed commit shows as verified.
- It cannot prove who typed the approval. The script checks only that standard input is a terminal. Git's author and
  committer fields are set by whoever makes the commit, and git does not check them. Anyone able to push to the
  repository, an agent with push access included, could make such a commit under your name. Refusing this session's
  own git identity guards against a mistake, not against forgery.
- The code cannot stop the orchestrating agent from going around it. The agent could write the runs-side approval
  file by hand, or make and push a commit in your name. The git checks make such a forgery visible on GitHub, but do
  not prevent it. So this route still rests on the orchestrating agent following the procedure, and on your check of
  the commit.

**Other modes are not accepted.** Earlier versions declared two alternatives. Version 6 records your choice instead:
`cloud_terminal_allowed` and `chat_mode_allowed` are false. Their code remains, but `approve.py terminal` and
`approve.py chat` refuse, and so does every BCA entry point, even for an approval file written by hand. Using either
would need a new version and a new fingerprint. Their weaknesses, for the record:
- **(a) A terminal in this session**, as the gate: standard input is a terminal and the characters typed equal the
  first 16 characters of the fingerprint. Any agent with a shell can satisfy both checks through a pseudo-terminal
  (for example the `script` command), and you have no terminal here.
- **(b) Chat:** the exact sentence "approve <the first 16 characters>, accept chat mode", written to a file by the
  orchestrating agent. The program cannot tell your turn from a file an agent wrote.

**Approving accepts every default.** Approving this fingerprint accepts every default listed in section 14. Any other
choice needs a new version of `prereg.json` and a new fingerprint.

**Checks at approval.** `approve.py` refuses, at import and in every mode, unless:
- `prereg.json` is final and matches its sidecar;
- it accepts the mode;
- the code matches the code manifest;
- every other job's pinned component still matches the pin `prereg.json` records: the room test, the fallback
  diagnostic, the stand-ins and the detection script. A pin that changed between writing and approving refuses the
  approval.

`approve.py show` runs the same code and pin checks and says whether the fingerprint can be approved now. The
staging folder is rebuilt only when it can.

**Every BCA entry point re-checks the approval against the current `prereg.json`** and refuses otherwise:
- editing `prereg.json` after approval voids it;
- so does changing any pinned script;
- it re-checks your file and its fingerprint, the commit with git, that your committed file is still there unchanged
  here and on GitHub, and that the import record is on GitHub;
- every BCA phase also re-checks the other jobs' pins.

A draft (a missing pin) cannot be approved.

---

## 10. Order of work

0. **Pins and approval.** `prereg.json` pins the room test's code, the stand-ins' code and the detection script by
   their code manifests. All three were locked when this version was written. You approve by the route in section 9,
   and no BCA phase runs before the approval is imported and the import record is on GitHub.
   - If any of them is rewritten before the import, approval is refused and every BCA phase refuses to run.
     `write_prereg.py` is then rerun (and `write_manifest.py` too if Stage 2's own code changed), and you approve the
     new version with a new approval file (section 9).
   - After the import, `write_prereg.py` and `write_manifest.py` refuse: the approved files are frozen. A pin that
     moves then stops every BCA phase until you decide what to do.
   - You may approve before Stage 1's outcomes exist. Each phase reads what it needs when it runs: the room's pooled
     look (the run condition) and the stand-ins' marks (the rival).
   - `select_standin` refuses until the stand-ins' marks exist, unless the room test rules the stand-ins out.
1. **Choose the stand-in:** `select_standin` (no BCA).
2. **Detection, seeds 0-4:** verify the rehearsal's pair sets, labels and stand-in scores (no BCA; each step refuses
   if the rehearsal's file is absent); band fits; width scores; marks (refused until every seed's inputs exist).
   - Then the weighted version, only if declared.
3. **The rest, seeds 0-4:**
   - the deployment check, once the version decision exists;
   - if a version passed and the room passed: the policy test and the full host;
   - the exploratory pooled look, once the detection marks and every seed's deployment values exist.
4. **Seeds 5-9:** fresh data and learners (no BCA; these may start once `prereg.json` is final), then the same stages,
   and the confirmatory pooled look.

---

## 11. Compute

| Part | Process-hours |
|---|---|
| Detection, seeds 0-4 (15 band fits; the rehearsal's labels and stand-in scores are reused) | about 1 (+2 if the weighted version runs) |
| Policy test, seeds 0-4 | about 15 |
| Deployment check, seeds 0-4 | about 3 |
| Full host, seeds 0-4 | about 13-22 |
| Fresh seeds 5-9: data, critics, clones, actors, plain TD3+BC | about 3 |
| Fresh seeds 5-9: their pair sets, labels and stand-in scores, then the same BCA stages | about 35-45 |
| **Total** | **about 72-92**: about 36-46 hours on 2 processes. The first decisive look, detection on seeds 0-4, takes about 1 hour. |

---

## 12. The claim, word for word (plan 5.5)

> "In strong current, BCA's lower edge recovers X of H points and keeps Y of TD3+BC's gain over cloning in the
> recorded conditions. A hand-set limit to the recorded range recovers Z. In the no-stall world, where the missing
> commands help, BCA costs W points and the range limit V."

| Number | Meaning |
|---|---|
| H | full-log critic's actor minus no caution, strong current (98.9 on seeds 0-4) |
| X | BCA (k = 1) minus no caution, strong current |
| Y | (BCA − cloned actor) / (no caution − cloned actor), recorded conditions |
| Z | range hinge (k = 1) minus no caution, strong current |
| W | no-stall world: no caution minus BCA, strong current |
| V | no-stall world: no caution minus range hinge, strong current |

**The flags claim is supported (confirmatory) only if** both of these hold on fresh seeds 5-9, with every setting
frozen:
- the detection claim rule;
- all of M1-M5.

On seeds 0-4 the pooled look writes the claim as empty, with the words "exploratory, not a claim".

---

## 13. Departures from the plan, fixed now

1. **Change A1 applies to BCA.** The paired form is primary; the plan's unpaired form is a declared secondary arm.
2. **The size mark reads the paired penalty's difference;** the raw width difference is reported.
3. **The fixed-controller critic** has BCA's one-step target with the stall-aware controller and TD3's smoothing
   noise, and is its own target critic.
4. **Uniform "cannot certify" is all or nothing;** refusal per command exists only in the weighted version.
5. **The weighted table's grid and shared posterior draws** are fixed here.
6. **Both random surfaces** are matched, and M2 must hold against each.
7. **Push less everywhere** is matched and reported.
8. **The controls' ladders use the room's slope scale.**
9. **The weighted classifier restates the stand-in classifier ratio's estimator.** The locked detection script uses
   the same one (same network, rows, start and steps).
10. **Plan 4.2 names no rule for "the best stand-in";** section 8 fixes one.
11. **M5 is read on the no-caution actor.**
12. **Fresh seeds** have no random-removal critic. The no-caution and full-log actors are trained as a pair.
13. **The no-stall plain TD3+BC** is trained here.
14. **Shift cards** are produced with the report-card bug fixed.
15. **tau uses every held-out recorded row except the detection script's T4 rows.**
16. **Detection on seeds 0-4 reuses the rehearsal's locked pair sets, labels and stand-in scores.**
17. **The host critic's detection readout and plain TD3+BC's deployment check always use the as-written (uniform)
    band**, whichever version is chosen.
18. **The weighted version's gradient also flows through the classifier's command input** (section 2). The plan's
    formula does not say.
19. **The policy test's rival follows your choice of forms in the stand-ins test** (section 8).

---

## 14. Choices for you

1. **How to approve:** you chose a terminal on your own machine, with `approve_local.py`, then your own commit and
   push. This version accepts no other mode. The route cannot authenticate the commit's author or committer, and it
   rests on the orchestrating agent following the procedure and on your check of the commit on GitHub (section 9).
2. **Accept change A1 for BCA:** the paired form primary, the plan's form secondary.
3. **When the full host runs:** whenever detection and the room passed (the default), or only after the policy marks
   pass.
4. **The classifier:** the locked detection script restates `weights.py`'s estimator exactly (same network, rows,
   start and steps), so both use one estimator. Nothing to choose unless you want a different estimator.
5. **Which actor M5 is read on:** the no-caution actor (the default), or the BCA (k = 1) actor.
6. **M2:** against both random surfaces (the default), or one.
7. **Detection labels for seeds 0-4:** reuse the rehearsal's locked labels (the default), or recompute them.
8. **The compute budget:** about 72-92 process-hours.
9. **The weighted version's gradient** (it runs only if the uniform version fails direction or size): through the
   tabulated threshold as well as the refusal term (the default), or with the threshold held fixed in the gradient.

Approving this version's fingerprint accepts every default above.

---

## 15. Files

**Code** (in `scratchpad/ood/stage2/`, copies in `runs/shift_explore/stage2/code/`):

| File | What it holds |
|---|---|
| `stage2_common.py` | paths, output guard, approval check (named so that it cannot collide with the detection script's `common.py`) |
| `approve.py` | the import of your approval, and `show`; its terminal and chat modes refuse in this version |
| `approve_local.py` | approval on your own machine (Python standard library only) |
| `band.py` | the band fit, width readout and weighted version |
| `weights.py` | the classifier |
| `policy.py` | the policy test |
| `deploy_check.py` | Variant B |
| `host.py` | the full host |
| `fresh.py` | the seed contexts and the fresh seeds |
| `detect_contract.py` | the detection-script contract and the detection marks |
| `run_stage2.py` | the phases |
| `write_manifest.py`, `write_prereg.py`, `prereg_payload.py` | the manifest and the pre-registration |
| `test_stage2.py` | the synthetic tests |
| `launch.sh` | the launcher |

**The committed copy** is in `outputs/shift/beyond-the-data/stage2/`: `prereg.json` and its fingerprint file, this
document, `approve_local.py`, the code manifest and its fingerprint file, `code/` with a copy of every pinned script,
and a README with your steps. Your approval file `approval_<first 16 characters>.json` arrives there, and so does the
import record `import_<first 16 characters>.json`.

**Outputs** go to `runs/shift_explore/stage2/`. The one exception is the import record, which import writes into the
committed folder. Nothing is written under `experiments/`, `runs/shift/` or `runs/shift_explore/room/`, or in the
detection rehearsal's folder.

---

## 16. Changes in version 4, after the review

Version 3 was never approved. These changes were made after a review of it.

**The method, declared:**
- The weighted version's gradient also flows through the classifier's command input, into both the threshold and
  the refusal term (section 2).
- Uniform "cannot certify" changes nothing anywhere: penalty, gradient or deployment check (sections 2 and 5).

**The rules:**
- The detection claim needs all five detection rivals, with finite scores on every seed (section 3).
- No output on seeds 0-4 states the claim (sections 3 and 12).
- A fresh seed whose data or learners failed is recorded as failed, so the confirmatory look can still be written
  (section 7).
- The detection marks, the deployment check and the pooled look refuse while their inputs are incomplete, instead
  of writing a permanent failure (sections 3, 5 and 10).
- Stage 2 never builds pair sets in the rehearsal's folder (section 8).

**The approval:**
- Chat mode takes exactly one sentence (section 9).
- Approval re-checks every other job's pin, and so does every BCA phase (section 9).
- Both modes' reliance on the orchestrating agent is stated (section 9).

**The documents:** the two sets of rivals are named separately (sections 3 and 8).

**Outside the review:** the stand-ins test now asks you to choose its stand-ins' forms ("approve F1-F3" or "decline
F1-F3"). The policy test's rival now follows that choice (section 8).

---

## 17. Changes in version 5, at your choice of approval route

Version 4 was never approved. Version 5 changes the approval:
- **The primary route is now approval at a terminal on your own machine**, sent by your own git push (section 9). You
  chose it.
- **`approve_local.py` is new.** It uses only Python's standard library, and the code manifest pins it.
- **`approve.py` has a new `import` step.** The approval check that every BCA entry point runs accepts the new mode
  `local-terminal`.
- **The terminal in this session and chat mode stay as declared alternatives**, with their weaknesses stated as
  before. (Version 6 no longer accepts them; section 18.)
- **`prereg.json` now lists what approval allows and does not allow.** The script prints that list.

The method, the detection, policy and deployment tests, the marks and the settings are version 4's. **One pin
changed, outside Stage 2.** The
detection rehearsal's job re-locked its detection script (its spec version 5) at 07:00 UTC on 13 October, after
version 4 was written. So version 4 could no longer be approved. Version 5 pins the re-locked script, which still
meets the contract in `detect_contract.py` (checked).

---

## 18. Changes in version 6, after a review of version 5

Version 5 was never approved. Its fingerprint was not sent to you.

**One pin changed, outside Stage 2.** The stand-ins job re-locked its code at 07:18 UTC on 13 October (its
specification version 3, change C6). Your choice of stand-in forms now gates only its marks look, not its training.
Version 5 pinned the earlier code, so it could no longer be approved. Version 6 pins the new code manifest,
specification and `standins.py`. The functions Stage 2 calls in `standins.py` and the fields it reads from the
stand-ins' `marks.json` are unchanged (checked against the code). No other pin moved: the detection script is the one
re-locked at 07:00 UTC, and the room test and fallback diagnostic are unchanged.

**The approval route:**
- **Only your chosen route is accepted.** The terminal in this session and chat mode now refuse, in `approve.py` and
  at every BCA entry point (section 9).
- **Import reads git itself.** Nobody types the commit details in. It also checks the committer, not only the author,
  against this session's identity.
- **The import record is committed and pushed** to the branch, and every BCA entry point refuses until it is on
  GitHub. So you can check which commit was imported.
- **One approval file per fingerprint**, so a later version can be approved without deleting an earlier file.
- **`approve.py show` and the staging build check the code and every pin now**, so an out-of-date fingerprint is
  flagged before it reaches you.
- **The documents now say plainly** that the route still rests on the orchestrating agent following the procedure,
  and on your check of the commit.

The method, the detection, policy and deployment tests, the marks and the settings are version 5's, and so version
4's.
