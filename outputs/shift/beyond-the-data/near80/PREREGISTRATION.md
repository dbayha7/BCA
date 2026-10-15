# Hiding the commands near 80%: pre-registration (version 8, near-80)

**FINAL, awaiting the owner's approval. Nothing has been trained.**

No learner, critic or actor was trained for this version, and no BCA band was fitted. You accepted every default of
section 16 on 15 October 2026 at 19:24 UTC, in a comment on the ThrusterChain page ("accept the defaults"). The
machine-readable twin is `prereg.json` beside this file; its status is "final, awaiting approval". The draft it was
built from is kept in `prereg_history/`. You approve the fingerprint of `prereg.json` as section 11 says.

**Your request** (a comment on the stall chart of the ThrusterChain page): "what about the values close to 80%. I
think they should be deliberately left out of the training data and introduced during testing so we can really see
OOD". OOD means out of distribution: unlike anything in the training data. To the offer of "the main test's setting
with a band around 80% removed as well", you answered "yes i do".

**The one question this answers:** when the good commands just below the 80% stall are hidden from training, does
BCA still let the controller use them while keeping it from going past 80%? And does it do better than simply
staying inside the recorded data?

---

## 0. Status, and what this was written after

**Seeds.**
- Seeds 0-4 are seen. Everything on them is exploratory.
- Seeds 10-14 are fresh. Only they can carry a claim. Nobody has generated, loaded or looked at them.
- Seeds 5-9 belong to version 6/7. This version never touches them.

**Written after:**
- the main test (version 6) and its results on seeds 0-4 (13 and 14 October);
- restructured BCA's step 1 and step 2 on seeds 0-4. Both met their go rules (exploratory). Step 2's go "licenses
  drafting version 8 on fresh seeds (10-14)";
- a fact base made for the draft on seeds 0-4. It holds row counts for four bands, and fixed controllers and the
  exact best plan on the four fixed test start sets. It took about 25 CPU-minutes and trained nothing;
- two small checks on seeds 0-4 that trained nothing: the gate's world checks after the band (about 1 CPU-minute),
  and a scripted rule clipped to the recorded range (about 2.3 CPU-minutes);
- three candidate designs. The draft took the cheapest as its base and added the best parts of the other two;
- one cross-check by the draft's own code on seed 0, stalling world (about 13 CPU-seconds). It reproduced the fact
  base's counts exactly;
- an independent review of the draft. It reran the draft's code on seeds 0-4, both worlds, and counted the
  hard-water rows among the removed rows. It trained nothing. It found that the first random-rows control was not
  matched on hard water (section 3.7);
- after the review, this code's no-training facts on seeds 0-4, both worlds, rerun with the control drawn by kind of
  water (about 2.3 CPU-minutes, nothing trained);
- your answer accepting every default of section 16 (15 October 2026, 19:24 UTC, "accept the defaults");
- the final code, frozen by its manifest, reran the no-training checks on seeds 0-4 before approval. Its record is
  pinned in `prereg.json` (`no_training_checks_frozen`). Nothing was trained.

**Not seen by anyone:** any learner trained with the band hidden, and any number from seeds 10-14.

**So:** every mark below was fixed before any learner saw the new data. Some checks that train nothing were set
after their seeds 0-4 values were known. Each such check says "seen" and gives the value.

---

## 1. Words used

**The world**
- **Throttle**: the size of one thruster's command, from 0% to 100% (|a| from 0 to 1), whatever its sign.
- **The stall**: past 80% throttle, a thruster's push falls (0.65 at 90%, 0.5 at 100%), while the throttle still
  costs energy. A command of exactly 80% does not stall.
- **Stalling world**: the test world. **No-stall world**: a comparison world where push keeps rising to full
  throttle. Each has its own recorded log, made by the same recording controller.
- **Recording controller**: the scripted controller that made the logs. Its **normal driving** approaches 80% but
  never reaches it. Half the time when the water pushes hard, it gives a **full-throttle command** instead.
- **Hard-water rows**: recorded rows where the water pushes hard enough (push above 0.565) that the recording
  controller may give its full-throttle command. The rule reads only the row's state. Every other row is **calm**.
  Hard water and calm are the two **kinds of water**.

**Earlier work**
- **The gate**: the main test's accepted data (its logs, its ball and its removals) and the checks they passed.
- **Step 1 and step 2**: restructured BCA's two exploratory tests on seeds 0-4. Step 1 tested BCA's switch. Step 2
  trained controllers with it.
- **Gap rule**: step 1's rule that finds the empty stretch of throttle in the recorded commands. Its **lower end** is
  where the recorded throttles stop. Its **upper end** is where the survivors begin.

**The data**
- **Training rows**: 50,000 recorded rows the learners train on.
- **Calibration bank** (or just **bank**): 1,000 held-out recorded rows that BCA reads.
- **Evaluation split**: 30,000 held-out recorded rows, never trained on and never removed.
- **The ball**: the main test's removal. It takes the densest 15% of recorded commands, a ball in the space of the
  four throttles. It holds 99% of the full-throttle commands.
- **Survivors**: full-throttle commands outside the ball. They stay in training (28-59 per seed and world).
- **The band**: every throttle from 70% to 90%, both ends included, on any thruster, either sign.
- **Three settings of the data**:
  - **main setting**: the ball removed (version 6 and step 2);
  - **near-80 setting**: the ball and the band removed (new; the primary setting);
  - **random-rows setting**: the ball, and as many randomly chosen rows as the band removes, in each kind of water
    (new; the control).

**The learners**
- **Critic**: the trained network that rates a command. **Actor**: the trained controller.
- **Host**: the learner BCA is added to. **TD3+BC** is the host here. Its actor climbs the critic, with a pull toward
  the recorded command.
- **No-caution actor**: trained against the critic with no penalty. **Full-log actor**: the same, but against a
  critic trained on the full log. **Cloned actor**: copies the recorded commands.
- **Harm**: the full-log actor's points minus the no-caution actor's points, in strong current.
- **Value unit**: the critic's own unit (one unit of predicted discounted reward).

**The methods**
- **BCA (Bayesian Conformal Aggregation)**: the add-on under test.
- **BCA's switched ramp**: restructured BCA, step 2's primary arm (in the code: `obca_k1`). It is a training penalty
  on throttle. It is zero up to where the recorded throttles stop (the gap rule's lower end), rises in a straight line
  across the empty stretch above, and is flat beyond. BCA's own check on held-out recorded outcomes (its **switch**)
  turns it on or off in each world and sets its **height**.
- **BCA's posterior, 5% bound and margin**: a second critic, trained without the survivors, is scored on held-out
  rows. BCA's **posterior** is its estimate of how much more that critic over-predicts the full-throttle rows than
  the rest. The **5% bound** is that estimate's 5% lower bound. The switch is on when the bound is above the
  **margin**, 0.10 value units.
- **k (the dose)**: a multiplier on the ramp's height. k = 1 is BCA's own height, k = 2 twice it, and k = 0 no ramp.
- **Version 6's BCA**: the main test's method. Its actor climbs the lower edge of BCA's band around the critic's
  value: the value minus how much wider BCA's band is at the command than at the recorded command. Reported here, not
  marked.
- **Range limit**: a penalty for leaving each thruster's recorded range (0.5% to 99.5% of the kept training
  commands). It reads only where the data are. **Hand-set**: at a fixed strength.
- **Matched**: tuned, before any score is computed, to move the actor exactly as far as BCA's penalty does.
  **Ladders and matcher**: version 6's fixed lists of strengths to try, and its rule for picking the strength that
  matches.
- **Controls**: the **even pull** (a stronger pull toward the recorded commands everywhere), two **random surfaces**
  (penalties of random shape) and **push less** (a penalty on throttle everywhere).
- **World-blind rival**: a penalty that cannot turn itself off in either world. The **always-on ramp** is one: BCA's
  ramp with the switch removed. It keeps BCA's stalling-world height, which BCA set from held-out outcomes.
- **Perfect band**: a reference penalty that knows the critic's true error. No real method has it.

**The scores**
- **Points**: 0 is the recording controller and 100 the world's top scripted controller, from exact 100-step returns
  on the same 1,000 starts.
- **Stall-aware controller**: the textbook rule capped at 80% (100 points in the stalling world). **Rising-thrust
  controller**: the same rule allowed full throttle (100 points in the no-stall world). **Textbook rule**: the
  standard linear feedback for this task.
- **Best plan**: the exact best list of 100 commands from one start (`outputs/shift/oracle/`).
- **Test start sets**: four fixed sets of 1,000 starting states, one per **situation**: recorded conditions, big
  kicks, big waves and strong current (no, small, medium and large shift). They are not the calibration bank.
- **Unseen-throttle gain**: an actor's points as trained, minus its points with every thruster held to at most 70%
  (the band's lower edge) at test time.
- **Overshoot loss**: its points with every thruster held to at most 80% at test time, minus its points as trained.
- **Band cost** of an arm: its points in the random-rows setting minus its points in the near-80 setting. Positive
  means hiding the band hurt it.
- **Net across worlds**: points gained over the no-caution actor in the stalling world, minus points lost to it in the
  no-stall world (strong current, mean over seeds).
- **Pooled**: the mean over the five seeds of each seed's paired difference. Its **standard error** is how much it
  would typically move with other seeds.
- **Seed-world**: one seed in one world.

**Checks and compute**
- **Fingerprint**: a SHA-256 checksum of a file. It changes if one byte of the file changes.
- **Plumbing**: checks that the code runs as intended.
- **Phase**: one unit of the run, such as training one critic. A **policy phase** trains every actor of one
  seed-world. A **lane** is a slot in which one phase runs, on its own processor.
- **CPU-hour**: one processor busy for one hour. **GiB**: 1,073,741,824 bytes of memory.

---

## 2. The short version

- **Left out of training:** every recorded row in which any thruster's throttle is from 70% to 90%. This comes on
  top of the main test's removal (the ball). It applies to the training rows and BCA's calibration bank, in both
  worlds.
- **Introduced during testing:** nothing needs adding. The test situations already call for these commands. In
  strong current, the 100-point controller holds thruster 1 in the band on 81% of steps.
- **What the change really does:** after the ball, nothing is left between 80% and 90% anyway. So the band hides the
  stretch just below the stall, 70% to 80%. That is 1,310-1,531 rows of normal driving per seed in the stalling
  world (3.1-3.6% of the kept rows). The data now stop at 70%. The best command in strong current, the 80% hold,
  sits in the middle of a stretch the data never show.
- **Why that is the real test:** in the main test, "unseen" and "harmful" were nearly the same thing. A hand-set
  limit to the recorded range scored 93.1 points in strong current, against version 6's BCA's 24.9 (restructured
  BCA's switched ramp scored 88.6 there in step 2, exploratory). Now the unseen commands
  split in two: 70% to 80% is good, past 80% is bad. A scripted rule that stays inside the recorded data loses
  17.5-19.0 points once the band is hidden.
- **What is tested:** BCA's switched ramp on TD3+BC (step 2's protocol), against step 2's rivals and controls, with
  step 2's three pass marks. Every controller also gets two readouts that answer your question directly: what the
  throttles it never saw earned it, and what going past 80% cost it.
- **Cost:** about 31-43 CPU-hours, or about 10-13 hours of wall time on your Framework Desktop in three sittings (at
  most 2 policy phases at once under version 7's memory rule).

---

## 3. The data change, exactly

### 3.1 The rule

A recorded row is **in the band** when, for at least one thruster k:

    0.70 <= |a_k| <= 0.90

- a_k is the stored float32 command, compared at its exact float64 value. This is the main test's own convention
  for its ball.
- The edges are the float64 numbers 0.70 and 0.90. Both are included.
- Both signs count. Thruster 1's band rows are almost all negative (0-3 positive rows per seed).
- So a stored 70% (float32, 0.699999988) is just outside the band, on the seen side. A stored 90% (0.899999976) is
  inside it.

### 3.2 The order

1. The main test's ball is found first, exactly as before (`shift.define_gap`, `shift.make_removal`). So it is the
   same ball, bit for bit.
2. Then every band row leaves too. The removed set is the ball **or** the band.
3. The code builds it as a `shift.Removal`. `shift.py` is imported, never edited.

### 3.3 Where it applies

| Rows | Band removed? |
|---|---|
| Training rows | yes |
| Calibration bank (BCA reads it) | yes |
| Calibration rows outside the bank | never (no learner reads them) |
| Evaluation split | never |
| Test episodes on the four test start sets | never (nothing is recorded there) |

- Input scaling stays the gate's: from the ball-kept training rows' states, as step 1 did. It reads states only.
- Each world gets the band on its own log.

### 3.4 With the ball, not instead of it

- You said yes to this form.
- Without the ball, 6,187-6,795 full-throttle rows per seed would return to training. The critic would see the
  stall's real cost, and the harm would disappear. That is a different, easier question.
- With the ball kept, the only change from the main test is the band.

### 3.5 Every thruster, not thruster 1 only

The session's example named thruster 1. This version uses every thruster (choice 1, default accepted).
- **Thruster 3 would give the stall away.** With thruster 1 alone, thrusters 2-4 still record commands up to about
  0.78. BCA's ramp reads the largest throttle over all four thrusters, so it would still start at 0.773-0.781, just
  below the stall. Any good result could then come from thruster 3's data.
- With every thruster, no thruster has data from 70% to the first survivor (above 0.987). BCA's ramp then starts at
  the band's edge (0.6999-0.7000 on every seed and world).
- Thruster 3 also needs the band in strong current: the stall-aware controller has it there on 31% of steps.
- "Values close to 80%" covers every thruster. The stall is the same on each.
- The cost is small: 1,310-1,531 extra rows instead of 1,017-1,217 (stalling world).

### 3.6 Why 70% to 90%

- It is the session's example, centered on the stall.
- The 90% top changes nothing on seeds 0-4. On a fresh seed, it also catches any full-throttle row between 80% and
  90% that the ball missed.
- The hidden stretch is worth a lot at test time. In strong current, the textbook rule held to 70% scores 70.7
  points instead of 100. The best plan held to 70% loses 21.8 points.
- It is still a small cut: 3.1-3.6% of the kept rows, against the ball's 15%.
- Narrower is too weak: 75%-85% removes only 151-207 rows. Wider starts to test data shortage instead: 65%-95%
  removes 9.4-10.2% of the kept rows.

### 3.7 The control: as many random rows, from the same kinds of water

Removing 3% of the rows changes every training batch and the rows each critic and actor sees, for no reason linked
to the band. (The input scaling stays the gate's; section 3.3.) So every band cost is measured against a random
removal of the same size.

The band's extra rows are mostly hard-water rows: 725-924 of them per seed (55-60%) in the stalling world, and
49-55% in the no-stall world. Hard water is where the 80% hold matters. So the random rows are drawn separately in
each kind of water:
- Each row is calm or hard water, by the recording controller's own rule on the row's state
  (`RC.Recorder(spec, physics).in_region`).
- **Extra training rows:** in each kind of water, exactly as many as the band removes there on that seed and world,
  drawn uniformly from the ball-kept training rows of that kind, with `default_rng([20261012, rep, 99, 80, 2 + s])`,
  where s = 0 for calm and s = 1 for hard water.
- **Extra bank rows:** the same rule on the bank, with `default_rng([20261012, rep, 99, 80, 4 + s])`.
- No tag list ends in 0 (numpy drops a trailing 0).
- So the control keeps exactly the near-80 setting's share of hard-water rows (38.5-40.6% on seeds 0-4, stalling
  world; seen after this rule was set). It differs from the near-80 setting only in which commands it hides.

**Why not a plain uniform draw.** The first draft drew the extra rows uniformly. That took only 154-241 hard-water
rows per seed against the band's 725-924, and kept 42.9-45.4% of the hard-water rows against the near-80 setting's
38.5-40.6%. A band cost would then mix two effects: hiding the 70%-80% commands, and thinning the hard-water states
that matter most in strong current. The uniform draw is not used.

---

## 4. The facts on seeds 0-4, before any training

From the fact base, the two design checks, the review and this code's reruns. Ranges are over seeds 0-4. Any
thruster, band 70%-90%.

### 4.1 What the band removes

| | stalling world | no-stall world |
|---|---|---|
| removed by the ball | 7,500 | 7,500 |
| band rows in training | 1,570-1,756 | 1,254-1,391 |
| of which already in the ball | 225-279 | 174-210 |
| extra rows removed | 1,310-1,531 | 1,044-1,217 |
| of which hard water | 725-924 (55-60%) | 514-665 (49-55%) |
| share of the ball-kept rows | 3.1-3.6% | 2.5-2.9% |
| kept after both | 40,969-41,190 | 41,283-41,456 |
| bank rows kept (ball alone) | 805-826 (827-851) | 808-832 (829-850) |
| band rows in the evaluation split | 984-1,072 | 775-827 |

- Almost every band row is normal driving at or below 80%.
- Per thruster (stalling, training rows): thruster 1 1,264-1,428; thruster 2 31-46; thruster 3 209-252; thruster 4
  64-81.

### 4.2 What the learners and BCA will see

| | ball alone | ball and band |
|---|---|---|
| largest kept throttle below full throttle | 0.784-0.791 | 0.6999-0.7000 |
| thruster 1's recorded range, lower end | about -0.75 | about -0.69 |
| BCA's ramp start (gap rule, lower end) | 0.783-0.791 | 0.6999-0.7000 |
| BCA's ramp end (upper end) | 0.977-0.995 | 0.987-0.995 |
| rows at or above the ramp end | 28-59 | 28-58 |
| kept rows past 80% (the survivors) | 28-59 | the same, less 0-2 |

### 4.3 The main test's world checks after the band (stalling world)

| check | the gate's mark | ball alone | ball and band |
|---|---|---|---|
| full-throttle commands removed | at least 99% | 99.1-99.5% | 99.1-99.5% |
| kept rows past 80% | at most 0.2% | 0.08-0.14% | 0.08-0.14% |
| rows where both worlds behave the same | at least 99.8% | 99.86-99.92% | 99.86-99.92% |
| hard-water rows kept | at least 40% | 44.5-47.2% | **38.5-40.6%** |

- The last check would fail on seeds 3 and 4 after the band (39.5% and 38.5%). That is the intended effect: the 80%
  hold is given in hard water.
- So the gate's checks are read on the ball alone, as the gate defines them. The band gets its own floor of 35%
  (section 8.1). **That 35% was set after seeing these values.**
- The band takes 12.5-14.3% of the ball-kept hard-water rows.

### 4.4 How much the test needs the band (fixed test start sets, stalling world, nothing trained)

| | recorded | big kicks | big waves | strong current |
|---|---|---|---|---|
| stall-aware: thruster 1 in the band | 10.3% | 14.5% | 25.2% | **81.1%** |
| stall-aware: thruster 1 at exactly 80% | 4.6% | 9.5% | 18.8% | 71.9% |
| textbook rule held to 70%: points | 97.7 | 88.5 | 93.0 | **70.7** |
| best plan held to 70%: points lost | 1.4 | 9.4 | 3.9 | **21.8** |

- The band matters most in strong current. That is where the marks are read.
- In the no-stall world the band and above are worth almost the whole scale in strong current: the textbook rule held
  to 70% scores -1.3 there, held to 80% 55.0.

### 4.5 What the band does to a rule that reads only the data

The textbook rule, clipped to each seed's recorded range (a scripted controller; nothing trained), mean over seeds
0-4:

| points | recorded | big kicks | big waves | strong current |
|---|---|---|---|---|
| stalling, ball alone | 98.6 | 86.8 | 96.5 | 86.8 |
| stalling, ball and band | 97.1 | 82.6 | 92.1 | **68.5** |
| stalling, random rows (seen after the rule was set) | 98.5 | 86.7 | 96.4 | 86.4 |
| no-stall, ball alone | 96.9 | 73.9 | 88.4 | 24.9 |
| no-stall, ball and band | 95.4 | 70.3 | 82.8 | **-6.7** |

Per seed, the band costs this rule 17.5-19.0 points in strong current (stalling world) against the ball alone, and
17.2-18.4 against the random-rows setting (its band cost). Staying inside the data is no longer nearly free.

### 4.6 This version's code on seeds 0-4

`code/facts_near80.py` counts rows and runs the leak checks. It trains nothing.
- **Seed 0, stalling world (before the review):** 1,444 extra rows, 41,056 kept, 821 bank rows kept, 1,051
  evaluation band rows; gap rule 0.69990 to 0.99532, 33 rows above; hard-water rows kept 40.15%; band share of the
  ball-kept hard-water rows 13.98%. These match the fact base exactly.
- **The review's rerun, seeds 0-4, both worlds:** every no-training rule passed (117 rules). With the band empty,
  the removal reproduced the gate's recorded hashes on all 10 seed-worlds.
- **The rerun with the control drawn by kind of water, seeds 0-4, both worlds (seen after the rule was set):** all
  117 rules pass. On all 10 seed-worlds the control removes exactly the band's numbers of training and bank rows in
  each kind of water. It keeps 38.5-40.6% of the hard-water rows, exactly the near-80 setting's share. Its gap rule
  stays where the main setting's is: lower end 0.783-0.791, upper end 0.977-0.998.

---

## 5. Arms

### 5.1 The host

**TD3+BC in step 2's actor-only protocol, unchanged.**
- Each setting and world gets its own critic, trained on its own kept rows with the gate's call (100,000 updates),
  then frozen.
- Actors start from that setting's cloned actor and train for 20,000 steps in 4,000-step chunks, with alpha 5 and
  the gate's fixed batch stream.
- Arms of one kind train side by side, one processor per phase.

Why only this host:
- It is the only host with a frozen protocol, saved references and an exact plumbing check in this world.
- With a frozen critic, the band's effect on the critic and its effect on the actor can be read apart.

### 5.2 Near-80 setting, both worlds (the primary setting)

| arm | role |
|---|---|
| BCA's switched ramp, k = 1 | **primary** |
| the ramp with height 0 and k = 0 | plumbing check: must equal the no-caution actor within 0.00001 |
| the ramp at k = 2 | reported only (dose): no test, no claim, no sentence |
| step 2's three attribution switches | reported only: no test, no claim, no sentence |
| new: a plain mean-difference switch | reported only: no test, no claim, no sentence |
| the always-on ramp (at BCA's stalling-world height) | world-blind rival |
| range limit, matched | world-blind rival |
| even pull and two random surfaces, matched | world-blind rivals |
| push less, matched | reported |
| hand-set range limit | reported; used in claim 3 |
| version 6's BCA | reported (choice 8) |
| perfect band | reference |
| new: the ramp started at the stall (80%) | reference that knows the physics |
| new: the range limit placed at 80% | reference that knows the physics |
| no-caution, cloned and full-log actors | references |

- **Where BCA's ramp comes from, in each setting and world:** step 1's gap rule on that setting's kept commands
  places the ramp. A second critic, trained on the kept rows minus the survivors, scores how much it over-predicted
  the survivors and the kept bank rows, all held out from it. BCA's posterior on those scores (margin 0.10) gives the
  switch and the height.
- **Step 2's three attribution switches** give the ramp a different height with (almost) BCA's switch: a fixed height
  of 1.0, plain split conformal, and step 1's rank test with the raw average difference. Step 2 tested them with a
  5-point rule that chose its claim's wording. That rule is not carried over (section 15, departure 10): their
  numbers, the k = 2 arm's and the new mean-difference switch's are reported only.
- **Matching:** blind to returns, to the primary arm's stalling-world strength, with version 6's ladders and
  matcher. The no-stall world uses the stalling world's locked settings, as in step 2.
- **The full-log actor** trains against the full-log critic from this setting's cloned actor and batches. So the harm
  compares like with like. The full-log critic itself does not change (the full log is unchanged): on seeds 0-4 the
  gate's saved one is reused, checked by fingerprint.

### 5.3 Random-rows setting, both worlds (the control)

A reduced set, all unmatched:
- cloned and no-caution actors;
- the plumbing check;
- BCA's switched ramp, k = 1, with its own switch and height from this setting's data;
- the hand-set range limit.

**One training program for every setting.** The code trains the ramp arms of one batch side by side, and the batch
size changes the last bits of each actor. So BCA's switched ramp and the plumbing check train in step 2's own batches
in every setting and world: where full throttle stalls, the plumbing check alone, then six ramp arms with BCA's
switched ramp first; in the no-stall world, one batch of seven with the plumbing check first. The random-rows setting
fills the other five places with copies of its own switched ramp. They are checked equal to it and never saved or
scored. The near-80 setting's new mean-difference switch trains in a batch of its own. The hand-set range limit
trains alone in every setting, as in step 2. So band costs and the noise floor compare actors from the same program,
and the reproduction check (section 8.2) runs that program.

### 5.4 Main setting

- **Seeds 0-4:** step 2's saved actors, read-only, checked by fingerprint. They are re-evaluated with the 70% hold.
  Nothing is retrained. The reproduction check (section 8.2) ties them to the new runner.
- **Seeds 10-14:** step 2's full arm set, if choice 6 (the main setting on seeds 10-14) keeps its default of yes.
  This is the version 8 confirmation that step 2's go licensed, run on the same fresh seeds as a separate claim
  (claim 4). Step 2's code refuses every seed but 0-4, so its runner is restated for seeds 10-14 in new code, never
  edited.

### 5.5 Fixed references (no seed, nothing trained)

- the recording controller, the stall-aware and rising-thrust controllers (the points' anchors, in both worlds);
- the textbook rule held to 70% (check 9; the runner computes it in the stalling world);
- the textbook rule clipped to each setting's recorded range (check 10 and [Q]; the runner computes it in the
  stalling world, per seed);
- the best plan within 80%, 70% and 100% (first 300 starts per test start set; reported, never in a mark).

The best plans, and the no-stall rows of sections 4.4 and 4.5, come from the fact base (`outputs/shift/oracle/` and
the session's facts); the runner does not recompute them. No mark reads them.

### 5.6 Left out, and why

Three other learners (CQL, ReBRAC, IQL) could host BCA. None is in this version.

| left out | why |
|---|---|
| CQL host | no configuration on record; not offered |
| ReBRAC host | no configuration on record; about +32 CPU-hours per stage (choice 11) |
| IQL host | its actor has no slot for the ramp without a new method version |
| co-trained full host | step 2 did without it; about +4 CPU-hours per stage (choice 11) |
| version 6's detection test | it tests version 6's method; its 80% comparison command is now hidden too |
| version 6's deployment check | its mark failed on seeds 0-4; step 2 had none |
| the band alone, without the ball | a different question (section 3.4); choice 3 |

---

## 6. Situations

The main test's four fixed test start sets of 1,000 starts, 100 steps each, in both worlds:
- **recorded conditions** (no shift);
- **big kicks** (small shift);
- **big waves** (medium shift);
- **strong current** (current 0.7 to 1.1, above anything recorded; large shift). **The marks are read here.**

No situation is added. Strong current already needs the band on 81% of steps, and a new test start set would break
comparison with version 6 and step 2.

---

## 7. Measures

For every scored actor, setting, world and situation:
1. **Points** on the full test start set.
2. **Unseen-throttle gain** and **overshoot loss**, from two extra rollouts: every command clipped to 70% (the band's
   lower edge), and to 80%, at test time only.
3. **Zone shares** of visited steps:
   - every thruster below 70% (seen);
   - some thruster from 70% to 80%, none past 80% (unseen, below the stall);
   - some thruster past 80% (unseen, past the stall).
4. **Band cost**, for every arm the near-80 and random-rows settings share.
5. **Net across worlds**, strong current.
6. **Strengths and matches**, as in step 2.
7. **BCA's switch, height, 5% bound, ramp ends and group sizes**, per seed and world.
8. The worst-100-starts mean (the lower tail), and time and peak memory per phase.

Reported readouts at the evaluation split's band rows (no actor involved), near-80 and random-rows settings:
9. **The critic's error at band commands**: the frozen critic's value minus the exact value of the stall-aware
   controller after that command, beside the same state with the command clipped to 70%. It shows whether hiding the
   band made the critic wrong there, or only made the commands unseen.
10. **Version 6's width at the same pairs**: does its width rise where the band was hidden, and more past 80% than
    below?

A note for the code: the shift card's "removed share" is restated for the ball or the band.

A caution: an actor held softly near 70% can sit at 70.1% and count as "unseen, below the stall". So the marks use
points (items 1 and 2). The zone shares only explain them.

---

## 8. The tests, in order (kill rules first)

Every threshold is fixed now. None moves after its numbers exist, except through a labeled post-hoc
pre-registration that you approve.

### 8.1 The no-training checks (kill rules before any training)

They train nothing. On seeds 0-4 they run before approval, with the frozen code. If any fails there, no approval is
asked and the design is redrafted. On each fresh seed they run when its data are built. A fresh seed that fails one
is a failed seed for claims 1-3: it fails every mark of claims 1-3 and is never replaced. It fails claim 4 only if
the failed check is one of the gate's own (section 9.5).

| # | check | pass | seen on 0-4 |
|---|---|---|---|
| 1 | the band is really missing: the leak check passes; the random-rows control is matched, in total and in each kind of water, for training rows and bank | all pass | pass, seeds 0-4, both worlds (the control's per-kind match seen after the rule was set) |
| 2 | band rows left in the evaluation split | at least 300 | 775-1,072 |
| 3 | extra rows, as a share of the ball-kept rows | 1% to 6% | 2.5-3.6% |
| 4 | the gate's five world checks on the ball alone | all pass | pass |
| 4 | on the near-80 setting: full-throttle removed, kept past 80%, same-behavior share | the gate's marks | pass |
| 4 | on the near-80 setting: hard-water rows kept | at least 35% (set after seeing) | 38.5-40.6% |
| 5 | band rows among the ball-kept hard-water rows | at least 10% | 12.5-14.3% |
| 6 | BCA's gap rule is defined in both new settings; near-80 lower end | in [0.695, 0.70] | near-80 0.6999-0.7000; random rows defined, lower end 0.783-0.791 (seen after the rule was set) |
| 7 | calibration bank rows kept | at least 500 | 805-832 |
| 8 | with the band empty, the removal reproduces the gate's | hashes equal | pass, seeds 0-4, both worlds (seen after the rule was set) |
| 9 | fixed test start sets, strong current: stall-aware has thruster 1 in the band | at least 50% of steps | 81.1% |
| 9 | fixed test start sets, strong current: the textbook rule held to 70% loses | at least 15 points | 29.3 |
| 10 | per seed, strong current: the rule clipped to the recorded range loses | at least 10 points | 17.5-19.0 |

- **The leak check** (check 1): no kept training row and no kept bank row is in the band or the ball. The arrays a
  learner would receive (`dataset.d4rl_dict`) show no band command. The removed set is exactly the ball or the band.
  No evaluation or calibration row is touched. For the control: it removes the whole ball, and exactly the band's
  number of extra rows in each kind of water, for the training rows and for the bank, all from ball-kept rows.
- **The training file** (part of check 1, before any training phase): the leak check reads the arrays in memory. So
  the runner also re-reads each written training file and checks that its six arrays, the command array included,
  have the same SHA-256 as `dataset.d4rl_dict` of the kept rows (`near80lib.training_file_check`). A mismatch is a
  plumbing stop for every phase that would read that file.
- Checks 4 and 5 are read in the stalling world, as the gate reads its world checks.
- Check 9 is about the fixed test start sets, so it runs once.
- On fresh seeds, check 10 is a condition for the claims about the band (at least 4 of 5 seeds), not a seed failure.
- Why these marks: checks 1, 4 and 8 make sure the band is the only change. Checks 2, 3, 5 and 7 catch a coding error
  or a recorder change. Check 6 makes sure BCA's own rule still applies. Checks 9 and 10 make sure the test really
  needs the hidden commands.

### 8.2 Plumbing, after approval (scratch only)

- The unit tests.
- A 40-step smoke run of every phase.
- **The reproduction check:** with the band empty, the near-80 runner must reproduce, bit for bit, on seed 0 in both
  worlds: the gate's critic, cloned actor and no-caution actor; and step 2's saved actors for the plumbing check, BCA's
  switched ramp (k = 1) and the hand-set range limit. The seeds 0-4 band costs and the noise floor compare new actors
  with those saved ones, so they must come from the same code path: the check runs the official policy phases' own
  function, with step 2's batches (section 5.3). If any differs, nothing official starts, ever, in this version.
- The first peak-memory measurements of each phase kind.
- **Which plumbing failures can be retried.** Only a reproduction difference is final. Any other failure can be
  retried once its cause is fixed: a unit test that fails, a smoke phase that does not finish, a reproduction check
  that does not finish (a crash, or memory), a missing or damaged input (also a saved file whose fingerprint does not
  match, or a training file that fails its check, inside the reproduction check), or a Ctrl-C. Each attempt is recorded and kept. The
  reproduction check runs only after the unit tests and the smoke have passed.

### 8.3 The exploratory stage: seeds 0-4

In order:
1. The three settings' data for all 10 seed-worlds. The no-training checks are repeated by the official code.
2. The critics: for the near-80 and random-rows settings, each with its second critic for BCA's switch.
3. BCA's switches and heights; the cloned, no-caution and full-log actors.
4. The policy phases, with matching. Each seed-world is locked (every actor's fingerprint written) before any score is
   computed.
5. The evaluations and readouts. Step 2's saved actors get the 70% hold as well.

Then, **stop rules** (fixed now):
- **The harm is there.** Pooled, near-80 setting, stalling world, strong current: the full-log actor beats the
  no-caution actor by at least 30 points, and the no-caution actor goes past 80% on at least 25% of steps.
  - Why 30: twice the largest margin a mark asks (15), so every mark can be met.
  - On the main setting these were 98.9 points and 73.2%.
  - If not: stop the near-80 and random-rows settings before seeds 10-14, and report.
- **The band changes the rival's problem.** Pooled: the hand-set range limit's band cost is at least 5 points
  (stalling, strong current).
  - If not, the band adds nothing to the main test: stop the near-80 and random-rows settings before seeds 10-14,
    and report.
  - If choice 7 (the random-rows control) is set to no, this stop uses the main setting minus the near-80 setting
    instead of the band cost.
- **Claim 4 is not stopped by the two rules above.** They are about the near-80 setting. The main setting runs on
  seeds 10-14 whenever choice 6 (the main setting on seeds 10-14) keeps its default.
- **The noise floor.** Pooled, stalling, strong current: for the hand-set range limit and BCA's switched ramp, the
  random-rows setting and the main setting differ by at most 5 points.
  - If either differs by more, claim 3 is dropped and only reported: its 5-point margin would sit inside the noise of
    changing 3% of the rows. Nothing else changes.
  - The no-caution actor's difference is reported, not used: it is the noisiest arm, and claim 3 does not use it.
  - If a number it needs is missing on any of seeds 0-4, it cannot be read, and claim 3 is dropped in the same way
    (section 14.3, "Dropped on seeds 0-4, not measured").
  - If choice 7 (the random-rows control) is set to no, the noise floor is not run.
- **Plumbing stops, per phase:** the height-0 check, a non-finite network, a saved input whose fingerprint does not
  match, or a training file that fails its check. A stopped phase is written once and never rerun. Every mark that
  needs it fails.
- **Every seed is needed by the stop rules.** If a number that the harm rule or the rival rule needs is missing on any
  of seeds 0-4 (a phase stopped, or an actor's numbers are missing), that rule cannot be read. The near-80 and
  random-rows settings then stop before seeds 10-14, with their own sentence (section 14.5, "A stop rule cannot be
  read"). Claim 4 is not stopped by it.

**What does not stop the fresh seeds:** BCA's own result on seeds 0-4 (choice 9). Both outcomes answer your
question, so both are worth confirming.

Then the **exploratory look**: every mark of section 9 is computed on seeds 0-4 and labeled "exploratory, not a
claim". Nothing is tuned after it.

### 8.4 The confirmatory stage: seeds 10-14

The near-80 and random-rows settings run only if the stop rules above allow. The main setting (claim 4) runs if
choice 6 keeps its default. Section 10 gives the details.

---

## 9. Pass marks, with reasons and power

Each mark is pooled over seeds 10-14: the mean of each seed's paired difference. Stalling world, strong current,
unless stated. Every seed is needed. A failed or stopped seed fails every mark that needs it.

### 9.1 Validity conditions (without them, no claim either way)

- **For claims 1-3:** the harm is at least 30 points, and the no-caution actor goes past 80% on at least 25% of
  steps (near-80 setting, pooled).
- **For claims 2 and 3 (about the band):** check 10 holds in at least 4 of 5 fresh seeds. So the band really moved
  the edge of the data below the stall.
- **For claim 3 also:** the hand-set range limit's band cost is at least 5 points.
  - Power: fixed controllers predict a band cost of 12-19 points. The textbook rule clipped to each setting's
    recorded range has a band cost of 17.2-18.4 on seeds 0-4 (seen after the rule was set). With a per-seed spread of
    about 4-6, a true value of 10 reaches 5 with a chance of 97% or more.
- **Claim 4** has no validity condition from the near-80 setting (section 9.5).
- **Failed seeds:** the conditions above are computed over the fresh seeds that did not fail. A failed seed fails
  every mark of claims 1-3 by the pre-set rule (section 10), and claim 1's sentence names it. It never makes the
  validity conditions fail. A seed whose evaluation lacks a number these conditions or claims 2 and 3 need from an
  actor that is always trained, with no failure or stop record, fails in the same way (its learner failed; section
  14.6). Claims 2 and 3 then say that they fail because of the failed seed, rather than report pooled numbers that
  cannot be computed. Check 10's count leaves out a failed seed whose band data were never built.

### 9.2 Claim 1: BCA's result holds with the band hidden

Step 2's three go marks, unchanged, so the result compares directly with the main setting:

| mark | pass | reason | on the main setting, seeds 0-4 |
|---|---|---|---|
| net lead | BCA's net across worlds is at least the best world-blind rival's plus 10 | step 2's margin; about 4 standard errors | +25.3 (standard error 2.3) |
| keeps the gain | BCA minus no caution is at least -3 in recorded conditions, big waves and big kicks | version 6's margin; about the noise of 1,000 starts | +8.0, +21.2, +19.8 |
| beats the even pull | BCA minus the matched even pull is at least 15 | version 6's margin | +100.8 (standard error 5.1) |

- Power: even if BCA gave up everything the band is worth to the best plan in the other three situations (1.4, 9.4
  and 3.9 points), "keeps the gain" stays above +6. The main risk is the switch: BCA's second critic now extrapolates
  from 70%, not 79%, and could switch caution on in the no-stall world.

### 9.3 Claim 2: what BCA's controller does with commands it never saw

This answers your question directly. Two marks choose which of four fixed sentences is written (section 14):
- **Gain from unseen throttles:** BCA's unseen-throttle gain is at least 5 points, and positive in at least 4 of 5
  seeds.
  - Why 5: it is paired (the same actor on the same 1,000 starts), so its noise is low. It is version 6's tolerance
    unit, and about a sixth of the 29.3 points the stall-aware controller gets from the same throttles.
  - A positive value can come only from throttles of 70% and more. Training showed none of them, apart from the
    0.1% of rows at full throttle.
  - Power: if a seed's gain varies by 3-6 points, a true gain of 8 passes about 85-99% of the time and a true gain of
    5 about half the time.
- **No overshoot:** BCA's overshoot loss is at most 5 points.
  - Power: if a seed's overshoot loss varies by 3-6 points, a true loss of 2 passes about 87-99% of the time, a true
    loss of 5 about half the time, and a true loss of 8 only about 1-13% of the time.

### 9.4 Claim 3: hiding the band costs BCA less than the range limit

- **Mark:** the hand-set range limit's band cost minus BCA's band cost is at least 5 points, and positive in at least 4
  of 5 seeds.
- **Why the hand-set limit:** band cost compares each arm with itself, so no matching is needed, and the hand-set
  limit is far steadier than the matched one.
- **Power:** if each seed's difference varies by 6-8 points, the chance of passing is about 74-85% when the true
  difference is 8, about 87-96% when it is 10, and about 45% when it is 5.
- **What it means:** where full throttle stalls and BCA's switch is on, BCA's ramp and the always-on ramp are the same
  actor. So this compares two shapes of caution that cannot switch off: a ramp from the data's edge (at BCA's height)
  and a hard limit at it. It does not credit reading outcomes.
- It can be dropped by the noise floor on seeds 0-4 (section 8.3), also when the noise floor cannot be measured
  there.

### 9.5 Claim 4: step 2 confirmed in the main setting (if choice 6 keeps its default)

Step 2's three go marks on seeds 10-14, main setting. It is a separate claim. The near-80 result neither rescues nor
sinks it:
- **Marks:** only step 2's three go marks (section 9.2's table), in the main setting.
- **Validity:** nothing from the near-80 setting. The harm and check-10 conditions of section 9.1 are for claims 1-3.
- **A failed seed:** a fresh seed fails claim 4 only if it fails one of the gate's own data checks (the five world
  checks on the ball alone, or the ball's leak checks), or a main-setting plumbing stop. A seed that fails a check
  only the near-80 or random-rows settings need (checks 1-3 and 5-8, and the near-80 rows of check 4) still counts
  for claim 4.
- **The stops:** the two exploratory stops of section 8.3 (too little harm, rival unaffected) do not cancel it.
  Claim 4 runs on seeds 10-14 whenever choice 6 (the main setting on seeds 10-14) keeps its default.

### 9.6 Rules about the marks

- No mark moves after its numbers exist.
- The four claims are read separately. None rescues another.
- Seeds 0-4 never carry a claim. Every output there writes the claim as empty, with "exploratory, not a claim".

---

## 10. Confirmation on fresh seeds 10-14

- **Data:** new logs of both worlds, with the gate's recorder, sizes and settings (stall slope 1.5). Then the ball,
  its leak checks and the gate's five world checks (required for every claim). Then, unless the exploratory stops
  cancelled them, the band, the random-rows control and the no-training checks 1-8 for that seed.
- **A fresh seed that fails** a data check, or whose learner fails, fails every mark of claims 1-3. For claim 4, only
  the gate's data checks and the main setting's own phases count (section 9.5). A failed seed is written down and
  never replaced, as in version 6.
- **Learners:** every setting is made fresh: critics (including the full-log critic), BCA's second critics, cloned
  actors, the full-log actors, every arm of section 5.
- **Frozen from this pre-registration:** the band, the scope, the control's random streams, every arm, ladder,
  matching rule and mark. Seeds 0-4 change nothing.
- **Nothing on seeds 10-14 stops early on an outcome.** Only plumbing failures stop a seed-world.
- **The confirmatory look:** first the validity conditions, then claims 1 to 4. The code fills in the sentences of
  section 14.
- **Old code that refuses these seeds is restated, never edited.** Version 6's fresh-seed code refuses every seed but
  5-9, and step 2's code (`step2lib.py`) refuses every seed but 0-4. So both patterns, version 6's fresh-seed data
  and step 2's runner, are restated for seeds 10-14 in new code.

---

## 11. Approval

The session has written `prereg.json` (status "final, awaiting approval"), its code manifest (`code_manifest.json`),
the frozen code and the approval script (`approve_local_near80.py`), and committed them. Then, the same route as
versions 6 and 7:

1. On the Framework Desktop, open Ubuntu as root (`wsl -d Ubuntu-24.04 -u root` in PowerShell). The script needs the
   repository at `/home/user/bca` with version 7's restored `runs/`, because the code manifest pins files there.
   Then type `cd /home/user/bca` and `git pull --ff-only`.
2. Read `PREREGISTRATION.md` in `outputs/shift/beyond-the-data/near80/`.
3. At that terminal, type:

       python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check
       python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py

   It checks the files against their fingerprints (the code manifest, every file it lists, the repository files the
   code uses, and the frozen code's no-training checks on seeds 0-4), refuses any file in the folder or in `code/`
   that the manifest does not list (other than the frozen files and this fingerprint's own approval and import
   records), refuses a draft, refuses if an approval already exists, and refuses unless you are typing at a terminal.
   It shows a summary and asks you to type the first 16 characters of the fingerprint. The `-I` (isolated mode) keeps
   Python from loading anything from the script's own folder or from Python's environment settings, so no stray file
   there can run before the checks.
4. It writes `outputs/shift/beyond-the-data/near80/approval_near80_<first 16 characters>.json` and its `.sha256`
   file.
5. You commit and push those two files under your own GitHub identity, with the exact commands it prints (as in
   version 7's step 10, with its `pull --rebase` line if the push is rejected).
6. The session imports your approval with `code/import_approval.py import` and pushes an import record,
   `import_near80_<first 16 characters>.json`. Every training phase refuses without the import record. The import
   refuses if the approval commit's author or committer is this session's own git identity (as in version 6) or is
   not you, and the desktop runner refuses if root's git user.email is yours or the session's (as in version 7).
7. You bring the import record to the desktop: `git -C /home/user/bca pull --ff-only`.

**What the fingerprint covers.** The final `prereg.json` copies section 14's sentences and its bracket table
verbatim, so the claims live in the approved file itself, as version 6's `prereg.json` did. It also pins the code
manifest, which lists `PREREGISTRATION.md` and `README.md` with the code, as versions 6 and 7 did. So, as in
version 6, through the code manifest it covers this document.

**What this route shows, and what it cannot.**
- No agent writes your approval file or makes that commit. No message counts as approval.
- It cannot prove who typed it. Git does not check author names. The protection rests on the session following the
  procedure, and on you checking on GitHub that the imported commit is yours.

**Approving accepts every default** in section 16. Any other choice makes a new fingerprint.

---

## 12. Order of work

1. **Done:** the draft, its machine-readable twin, and the code that can be tested without training (the band, the
   control, the leak check, the training-file check, the facts script, and their unit tests).
2. **Done:** you accepted every default of section 16 (15 October 2026, 19:24 UTC, "accept the defaults").
3. **Done:** the session wrote the final version (the runner, the import, the freeze, the manifest and the approval
   script), ran the unit tests and the no-training checks on seeds 0-4 with the frozen code (about 5 CPU-minutes,
   nothing trained), and committed and pushed.
4. **You approve** (section 11). The session imports it.
5. **Plumbing** (section 8.2), in scratch.
6. **The exploratory stage** on seeds 0-4, then its stop rules (section 8.3).
7. **The confirmatory stage** on seeds 10-14: the main setting if choice 6 keeps its default, and the near-80 and
   random-rows settings if the stop rules allow.
8. Results are committed with their fingerprints (`run_near80.sh results`, run by itself at the end of a finished
   run: `near80/results-<utc>/`, with `SHA256SUMS`). Model files go to the run-files branch `claude/near80-run-files`
   (`push_runfiles_near80.sh`), as in version 7.

The exact commands for steps 5-8, and what to do for every message the runner prints, are in `README.md` ("How to
run").

Timing (choice 12, default accepted): after version 7's run on the desktop has finished. The near-80 runner refuses to
start while version 7's runner is running, and unless version 7's results are gathered (or its status shows every
phase final). If version 7 stopped for good without finishing, you type `run_near80.sh version7-ended` once; it is
recorded. While the near-80 run runs, it starts nothing new whenever version 7's runner or one of its phases runs.

---

## 13. Compute

On your Framework Desktop (AMD Ryzen AI MAX+ 395, 16 cores), in Ubuntu 24.04 under WSL (Windows Subsystem for Linux)
with 24 GB, on the processor alone.

| part | CPU-hours |
|---|---|
| no-training checks and plumbing | about 2-2.5 |
| exploratory, seeds 0-4 | about 11-16 |
| confirmatory, seeds 10-14: near-80 and random rows | about 12-16 |
| confirmatory, seeds 10-14: main setting (choice 6) | about 6-8.5 |
| **total** | **about 31-43** |

- **Unit costs**, one processor: a critic of 100,000 updates 6-10 minutes (measured in the cloud; the desktop is
  measured in plumbing); a stalling-world policy phase 30-40 minutes and a no-stall phase 10-12 (step 2 on the
  desktop: 27-36 and 9-10.5, plus the new arms); a random-rows phase about 8-10 minutes (its switched ramp trains in
  step 2's six-arm batch, section 5.3); an evaluation about 1.5 minutes per seed-world. The plumbing runs the unit
  tests, about 60 smoke phases (mostly 2 at a time until each kind is measured) and the full-length reproduction
  check on one lane (both worlds' critics, second critics, cloned and reference actors, and the full policy program).
- **Version 6 was estimated at 72-92 process-hours** (version 7 keeps that as an estimate that stops nothing). This
  version is cheaper because it reuses step 2's protocol and saved actors, and leaves out the detection test, the
  deployment check and the full host.
- **Lanes:** 10 by default, one phase per processor. Lanes change no result.
- **Memory: version 7's rule, unchanged.** A phase starts only if the memory reserved by the running phases plus its
  own is at most WSL's total memory (MemTotal) minus 8 GiB, and the free memory (MemAvailable) is at least its own
  reservation plus 8 GiB. A phase kind not yet measured reserves 7 GiB. A measured kind reserves 1.25 times its
  measured peak, except the policy phases, which reserve the larger of 7 GiB and twice their measured peak.
  - With WSL's total of about 23.5 GiB, that leaves about 15.5 GiB. So at most 2 policy phases run at once, whatever
    their peak (1.2 GB on the desktop in step 2). The 10 lanes speed up the critics, evaluations and other phases
    only.
  - **The plumbing's first measurements** (a departure from version 7, which measured only full-size phases): the
    reproduction check's peak, measured at full length, serves the critics, BCA's second critic and switch, the
    cloned and reference actors and the policy phases. Every other kind reserves the larger of 1.25 times its 40-step
    smoke peak and 2 GiB. Either counts only until an official attempt of that kind has been measured; from then on,
    official measurements alone set the reservation, as in version 7.
  - A phase whose own reservation is more than WSL's total memory minus 8 GiB can never start: the run halts and
    says so, instead of waiting.
- **Wall time:** about 10-13 hours in three sittings (no-training and plumbing, about 2-2.5 hours; exploratory;
  confirmatory), with at most 2 policy phases at once under version 7's memory rule. About 12-16 CPU-hours of policy
  phases run 2 at a time; the critics and evaluations come on top.
- **Budget guard:** a stage that uses more than twice its forecast's upper end pauses and reports; so does the whole
  run past the cap of 50 CPU-hours. The plumbing has no phase of the official plan, so it never pauses. A pause lasts
  until you type `run_near80.sh budget-continue <stage>`; each typed record (kept, write-once) allows one more
  forecast of that stage, and raises the cap by the same amount. No result is read differently because of a pause or
  a continuation.

---

## 14. The claim, word for word, for each outcome

The code fills in the brackets. Numbers are rounded to one decimal. Every bracket is pooled over seeds 10-14, in
strong current, unless stated:

| bracket | meaning |
|---|---|
| S, N | seeds where BCA's switch was on (stalling world), off (no-stall world) |
| X, H | BCA minus no caution; the full-log actor minus no caution (the harm) |
| W | no-stall world: no caution minus BCA |
| NET, D, R | BCA's net across worlds; its lead over the best world-blind rival; that rival's name |
| E, L | BCA minus the matched even pull; BCA's largest loss to no caution elsewhere |
| G, O | BCA's unseen-throttle gain and overshoot loss |
| G0 and O0, GZ and OZ, GV and OV | the same for no caution, the hand-set range limit, version 6's BCA |
| Q | the textbook rule clipped to the near-80 recorded range, in points |
| A, B, C, K | band costs of BCA and of the hand-set range limit; B minus A; seeds where B minus A was positive |
| M | fresh seeds where check 10 failed |
| P, V | no caution's share of steps past 80%; the largest random-rows-minus-main difference on seeds 0-4 (hand-set range limit or BCA's switched ramp) |
| dB, dR, d0 | near-80 minus main setting on the same fresh seeds: BCA, range limit, no caution |

**Always attached:** "One simulated task, two worlds, five fresh seeds (10-14), four fixed test start sets of 1,000
starts each. In strong current the test states also lie outside the recorded currents, so the band's effect is read
where states are new too."

**On seeds 0-4,** every claim sentence below (sections 14.1-14.4) starts with "Exploratory, seeds 0-4, designed after
seeing them; not a claim:".

### 14.1 Claim 1

**All three marks pass:**
> "On fresh seeds 10-14, with every recorded command that had a throttle from 70% to 90% on any thruster left out of
> training and out of BCA's calibration rows, on top of the main test's removal of the densest 15% of commands, BCA's
> switched ramp met all three pre-set marks. Its switch turned caution on in [S] of 5 seeds where full throttle stalls
> and off in [N] of 5 where it does not. In strong current it gained [X] points over no caution where full throttle
> stalls (the full-log controller gained [H]) and lost [W] where full throttle helps: a net of [NET], [D] more than
> the best rule that cannot switch itself off ([R]). It stayed within 3 points of no caution or better in the other
> three situations, and beat the matched even pull by [E]. Where full throttle stalls and its switch was on, its
> controller is the same as the always-on ramp's, so that gain comes from where the ramp starts and from its height,
> not from the switch."

**Any mark fails:**
> "On fresh seeds 10-14, with every recorded command that had a throttle from 70% to 90% on any thruster left out of
> training, BCA's switched ramp did not meet its pre-set marks: [one clause per missed mark]. [last sentence]"

The last sentence, exactly one:
- if claim 4 passed: "Its result in the main setting does not carry over to this one.";
- if claim 4 ran and did not pass: "Step 2's exploratory result was not reproduced on these seeds in either
  setting.";
- if claim 4 did not run (choice 6 set to no): "Step 2's exploratory result, from the main setting on seeds 0-4, was
  not reproduced in this setting."

The clauses:
- "its net across worlds of [NET] was [D] above the best rule that cannot switch itself off ([R]), below the 10
  required";
- "it lost [L] points to no caution in [situation], more than the 3 allowed";
- "it beat the matched even pull by [E] in strong current, below the 15 required";
- "seed [s] failed [check] and, by the pre-set rule, counts as failing every mark".

**Validity fails:**
> "On fresh seeds 10-14, the near-80 setting did not produce the harm it was built to test ([reason]). It supports no
> conclusion about BCA either way."

The reasons: "the controller trained without caution lost only [H] points to the full-log controller (at least 30
needed)", or "the controller trained without caution went past 80% on only [P]% of strong-current steps (at least 25%
needed)".

### 14.2 Claim 2: the sentence about unseen commands

G is BCA's unseen-throttle gain and O its overshoot loss, pooled, strong current. Exactly one is written, unless
check 10 failed or a seed failed (both below):

| gain mark | overshoot mark | sentence |
|---|---|---|
| pass | pass | "BCA's controller used throttles from 70% to 80% that it never saw in training, to good effect: they earned [G] points in strong current over the same controller held to 70%, and holding it to 80% would have added only [O]." |
| fail | pass | "BCA's controller gained little from throttles it never saw ([G] points in strong current over the same controller held to 70%; at least 5 needed). Holding it to 70% changed little: it scored about the same as a controller limited to the edge of the recorded data." |
| pass | fail | "BCA's controller earned [G] points in strong current from throttles from 70% to 80% that it never saw, but it went past 80% often enough that holding it to 80% would have added [O] (at most 5 allowed). Its caution did not stop it at the stall." |
| fail | fail | "BCA's controller gained little from throttles it never saw ([G]) and went past 80% often enough that holding it to 80% would have added [O]. Without data near 80%, its caution neither used the safe stretch nor stopped at the stall." |

**Always added, for scale:**
> "For scale, in strong current: the stall-aware controller gains 29.3 points from the same throttles. The controller
> trained without caution gained [G0] and lost [O0] past 80%. The hand-set range limit gained [GZ] and lost [OZ].
> Version 6's BCA gained [GV] and lost [OV]. A rule that only stays inside the recorded data scores [Q] points, against
> 100 for the stall-aware controller."

**If check 10 fails in more than one fresh seed:**
> "On fresh seeds 10-14, hiding the band moved the edge of the recorded data too little in [M] of 5 seeds. By the
> pre-set rule, this version makes no claim about the band. Its numbers are reported as description."

**If a seed failed:**
> "By the pre-set rule, claim 2 fails because seed [seeds] failed (claim 1 says why); the numbers of the other seeds
> are reported as description only."

### 14.3 Claim 3: the band's cost

**Holds:**
> "Against removing the same number of random rows, drawn from calm and hard water in the same numbers, hiding the
> band cost BCA's switched ramp [A] points and the hand-set range limit [B] points in strong current. The ramp lost
> [C] points less, in [K] of 5 seeds (at least 5 points and 4 of 5 seeds were required). Where full throttle stalls,
> this compares two shapes of caution that cannot switch off: a ramp from the data's edge (at BCA's height) and a hard
> limit at it."

**Fails:**
> "Against removing the same number of random rows, drawn from calm and hard water in the same numbers, hiding the
> band cost BCA's switched ramp [A] points and the hand-set range limit [B] points in strong current. The pre-set
> advantage of 5 points in at least 4 of 5 seeds was not shown (difference [C], in [K] of 5 seeds)."

**If a seed failed:**
> "By the pre-set rule, claim 3 fails because seed [seeds] failed (claim 1 says why); the numbers of the other seeds
> are reported as description only."

**Not valid on seeds 10-14:**
> "On fresh seeds 10-14, hiding the band cost the hand-set range limit only [B] points against random rows (at least
> 5 needed). By the pre-set rule, this version makes no claim about the band's cost. Its numbers are reported as
> description."

**Dropped on seeds 0-4:**
> "The comparison of the band's costs was dropped before the fresh seeds, because removing the same number of rows at
> random moved [arm]'s points by [V] on seeds 0-4 (at most 5 allowed). Its numbers are reported as description."

**Dropped on seeds 0-4, not measured:**
> "The comparison of the band's costs was dropped before the fresh seeds, because on seeds 0-4 the noise floor (how
> far removing the same number of rows at random moves the points) could not be measured: a number it needs is
> missing on seed [seeds]. Its numbers are reported as description."

### 14.4 Claim 4: the main setting (only if it runs)

**Holds:**
> "On fresh seeds 10-14, in the main setting (only the densest 15% of commands removed), step 2's result held: BCA's
> switched ramp met all three pre-set marks, with a net across worlds of [NET], [D] more than the best rule that
> cannot switch itself off ([R])."

**Fails:**
> "On fresh seeds 10-14, in the main setting, step 2's result did not hold: [clauses as in claim 1]."

**Always, when both settings ran:**
> "Hiding the band changed strong-current points where full throttle stalls, against the main setting on the same
> fresh seeds: BCA's switched ramp by [dB], the hand-set range limit by [dR], the controller trained without caution
> by [d0]."

### 14.5 Stops (exactly the one that applies)

- **A no-training check fails on seeds 0-4:** "On seeds 0-4, before any training, the near-80 data failed [check]
  ([value]; needed [threshold]). The design was redrafted. Nothing was trained."
- **Reproduction fails:** "Stopped: with the band empty, the new code did not reproduce [the main test's removal /
  critic / cloned actor / no-caution actor, or step 2's plumbing-check / switched-ramp / hand-set range-limit actor]
  on seed 0. No result of this version stands until it does."
- **A stop rule cannot be read:** "Stopped before the fresh seeds: on exploratory seeds 0-4, [what] could not be
  measured on seed [seeds], because a phase stopped or an actor's numbers are missing there. By the pre-set rule every
  seed is needed, so nothing on seeds 10-14 was generated for the near-80 and random-rows settings."
- **Too little harm:** "Stopped before the fresh seeds: on exploratory seeds 0-4, with the band hidden, the controller
  trained without caution lost only [H] points in strong current to the full-log controller (at least 30 needed)
  [or: went past 80% on only [P]% of steps (at least 25% needed)]. There is too little harm to test. Nothing on seeds
  10-14 was generated for the near-80 and random-rows settings."
- **The rival is unaffected:** "Stopped before the fresh seeds: on exploratory seeds 0-4, hiding the band cost the
  hand-set range limit only [B] points against random rows (at least 5 needed). The band does not change what a range
  limit can do, so this version adds nothing to the main test. Nothing on seeds 10-14 was generated for the near-80
  and random-rows settings."
- **Added to any of the three stops above, when choice 6 keeps its default:** "This stop does not touch claim 4: the
  main setting ran on seeds 10-14 as planned, and its result is reported separately."
- **A phase stops:** "Seed [s], [world]: [phase] stopped ([reason]). It was not rerun. Every mark that needs it
  fails."

### 14.6 How the code fills in the sentences

- Numbers are written with one decimal ("{:.1f}"), with a minus sign when negative and no plus sign; "-0.0" is
  written "0.0". [P] is a share of steps written as a percentage. [S], [N], [K], [M] and [s] are whole numbers.
- A number that cannot be computed, because a seed failed, a phase stopped or an arm is missing, is written "not
  available". A mark that needs it fails.
- [R] and [arm] are written as "the always-on ramp", "the matched range limit", "the matched even pull", "the first
  matched random surface", "the second matched random surface", "the hand-set range limit" or "BCA's switched
  ramp". When two rules that cannot switch themselves off have the same net, the first in step 2's order is named:
  the matched range limit, the matched even pull, the first random surface, the second random surface, the
  always-on ramp.
- [situation] is "recorded conditions", "big waves" or "big kicks". [world] is "stalling world" or "no-stall
  world". [phase] is the phase's plain name in the runner's phase table.
- [one clause per missed mark] lists the clauses of the missed marks, joined by "; ", in this order: the net lead;
  keeping the gain (one clause per situation that misses, in the order recorded conditions, big waves, big kicks);
  beating the even pull; then one clause per failed seed, in seed order. [clauses as in claim 1] is the same for the
  main setting. When a seed failed, a missed mark whose number is not available because of that seed is told by the
  seed's own clause only.
- [check] is the name of the first check that failed, in the order of section 8.1; "its [phase] phase" for a
  stopped phase; or "its evaluation (numbers missing)" for a seed whose evaluation lacks a number (the validity
  bullet below). [value] is written with four significant digits, and [threshold] as section 8.1 writes it.
- [reason] in claim 1's validity sentence is the harm reason when the harm is below 30 points, the past-80 reason
  when the share is below 25%, and both, joined by "; ", when both fail.
- In the harm stop, "[or: ...]" is left out when only the harm fails; the second form replaces the first when only
  the share fails; when both fail, the two forms are joined by " and ".
- In the reproduction stop, the bracket lists every item that differed, written "A", "A and B" or "A, B and C".
- Claims 2 and 3 are written only when claim 1's validity holds, except that in the exploratory look the dropped
  sentence of claim 3 is written whenever the noise floor fails. When a number the noise floor needs is missing on
  any of seeds 0-4, the noise floor cannot be read: claim 3 is dropped as if it had failed, and the "not measured"
  form of the dropped sentence is written instead, with [seeds] listing the seeds where a number is missing. When
  check 10 fails in more than one fresh seed, the check-10 sentence is written once, for both claims, in place of
  claim 2's sentence and claim 3's. The "for scale" sentence is written whenever the near-80 setting ran on the
  group's seeds.
- When a seed failed for claims 1-3, claim 2's sentence and claim 3's "Holds" or "Fails" sentence are replaced by
  their "If a seed failed" sentences, with [seeds] listing the failed seeds; claim 1's sentence says why each one
  failed. The check-10 sentence, claim 3's "Not valid" sentence and its dropped sentences keep their precedence. So a
  pooled number that is "not available" only because a seed failed is never written as a finding.
- The validity conditions of section 9.1 are computed over the seeds that did not fail. A failed seed fails every mark
  of claims 1-3 and is told by its own clause in claim 1's sentence ("seed [s] failed [check] ..."); it never makes
  claim 1's validity fail. A seed with no failure or stop record that lacks a number the validity conditions or
  claims 2 and 3 read from the four actors that are always trained (no caution, the full-log actor, BCA's switched
  ramp and the hand-set range limit; stalling world, strong current) fails in the same way, with [check] "its
  evaluation (numbers missing)": its learner failed (section 10). [M] leaves out a failed seed that has no check-10
  record.
- Marks and conditions are decided on the unrounded numbers, so a sentence can show 5.0 next to a mark of at least 5
  that was missed.
- On seeds 0-4, the "Always attached" sentence is not written; every claim sentence (sections 14.1-14.4) starts with
  the exploratory prefix, and in them "fresh seeds 10-14" reads "seeds 0-4" and "fresh seeds" reads "seeds". The stop
  sentences of section 14.5 are written as they stand.
- In the stop "A stop rule cannot be read", [what] is "the harm" (the harm or the past-80 share is missing), "the
  band's cost to the hand-set range limit", or both joined by " and "; [seeds] lists the seeds where a number is
  missing, written "2", "2 and 3" or "1, 2 and 3". Wherever a sentence says "seed [seeds]", it reads "seed 2" for one
  seed, and "seeds 2 and 3" or "seeds 1, 2 and 3" for more.
- Exactly one stop sentence is written, the first that applies in this order: a no-training check, reproduction, a
  stop rule that cannot be read, too little harm, the rival unaffected. The claim-4 note follows any of the last three.
  Each stopped phase adds its own sentence.

---

## 15. Departures from earlier versions, fixed now

1. **The band** is removed on top of the ball, from the training rows and the bank.
2. **The gate's world checks are read on the ball alone.** The near-80 setting gets its own 35% floor on hard-water
   rows, set after seeing seeds 0-4 (38.5-40.6%).
3. **The primary method is restructured BCA** (step 2's switched ramp), not version 6's lower edge. Version 6's BCA
   failed two of its marks on seeds 0-4; step 2 met its go rule. Version 6's BCA is reported.
4. **A new control setting:** as many random rows as the band removes, drawn separately in calm and in hard water,
   with new random-stream tags (section 3.7).
5. **New arms:** a plain mean-difference switch; the ramp started at the stall; the range limit placed at the stall.
6. **New readouts:** the 70% hold, the zone shares, the critic's error and version 6's width at band rows.
7. **A harm stop at 30 points** before the fresh seeds (step 2 had none). It applies to the near-80 and random-rows
   settings only.
8. **Seeds 10-14 for both settings.** Version 6's fresh-seed code (seeds 5-9 only) and step 2's runner (seeds 0-4
   only) are restated, not reused.
9. **No detection test and no deployment check.**
10. **Step 2's three attribution tests are not carried over.** Step 2 compared BCA's net with three other switches
    (a fixed height of 1.0, plain split conformal, and the rank test with the raw average difference), with a
    5-point rule that chose its claim's wording. Here those arms, the k = 2 arm and the new mean-difference switch
    are reported only: no test, no claim, no sentence. Why: step 2's wordings rest on facts about each arm's switch
    and height in the main setting on seeds 0-4, which are not known for the near-80 setting, and this version's
    claims credit nothing to BCA's posterior or to calibration either way.

---

## 16. Choices (every default accepted on 15 October 2026)

You accepted every default below on 15 October 2026 at 19:24 UTC ("accept the defaults"). Approving the final version
accepts them; any other choice would make a new fingerprint.

1. **Which thrusters.** Default: **any thruster**. Or thruster 1 only, as in the session's example. Thruster 3 would
   then still show data up to 0.78, and BCA's ramp would start there (section 3.5). Check 6 can then ask only that
   the gap rule is defined (table below).
2. **The band's edges.** Default: **70% to 90%**. Each alternative moves some fixed checks with it:

   | band (thrusters) | extra rows, share of the ball-kept rows: stalling; no-stall | check 3 pass range | check 6: BCA's ramp start must be | hold level (unseen-throttle gain, check 9) | textbook rule held there, strong current | fixed checks it fails as drafted |
   |---|---|---|---|---|---|---|
   | **70%-90% (any; default)** | **3.1-3.6%; 2.5-2.9%** | **1% to 6%** | **in [0.695, 0.70]** | **70%** | **70.7 points (loses 29.3)** | **none** |
   | 70%-90% (thruster 1 only) | 2.4-2.9%; 1.8-2.1% | 1% to 6% | defined only (it starts at 0.773-0.781) | 70% | 70.7 points (loses 29.3) | none |
   | 72%-88% (any) | 1.6-1.9%; 1.2-1.5% | 0.5% to 4% | in [0.715, 0.72] | 72% | 77.8 points (loses 22.2) | none |
   | 75%-85% (any) | 0.36-0.49%; 0.23-0.31% | 0.1% to 1% | in [0.745, 0.75] | 75% | 87.3 points (loses 12.7) | check 2 (90-164 evaluation band rows; 300 needed) and check 9 (loses 12.7; 15 needed) |
   | 65%-95% (any) | 9.4-10.2%; 8.1-8.8% | 4% to 15% | in [0.645, 0.65] | 65% | 49.8 points (loses 50.2) | none known |

   - The rule behind the check 6 column: the ramp must start in [lo - 0.005, lo], where lo is the band's lower edge
     (any thruster only). The hold level is always lo.
   - The check 3 ranges for the alternatives were set after seeing these values. The 75%-85% band would also need
     checks 2 and 9 restated before approval, which was not chosen.
   - Checks 4, 5 and 10 were computed for the default band only. For another band the session computes them on seeds
     0-4 before writing the final version, and section 8.1's rule applies: any failure means a redraft and no
     approval is asked.
3. **With the ball or instead.** Default: **with the ball**. Or also run the band alone as a reported third setting
   (about +6-8 CPU-hours per stage).
4. **Signs.** Default: **both signs** (the stall is the same either way). Or the negative side only, where almost all
   of thruster 1's band rows lie (0-3 positive rows per seed); positive commands from 70% to 90% would then stay in
   training.
5. **The hard-water floor on the near-80 setting.** Default: **35%**. Keeping the gate's 40% would fail seeds 3 and 4
   on seeds 0-4, so the design could not run as drafted.
6. **The main setting on seeds 10-14** (version 8's confirmation of step 2). Default: **yes**, as a separate claim
   (about +6-8.5 CPU-hours). Or no: step 2's confirmation would then need other fresh seeds (15-19).
7. **The random-rows control.** Default: **yes** (about +6-9 CPU-hours in all). Without it, claim 3 is lost, the
   "rival unaffected" stop uses the main setting minus the near-80 setting, and the noise floor is not run.
8. **Version 6's BCA as a reported arm.** Default: **yes** (about +1-1.5 CPU-hours per stage). It is the method of the
   chart you commented on.
9. **After a poor BCA result on seeds 0-4.** Default: **run seeds 10-14 anyway**, if the stop rules allow. Or stop the
   near-80 and random-rows settings there and save about 11-15 CPU-hours (claim 4 still runs).
10. **A checkpoint before seeds 10-14.** Default: **none beyond the approval**. Or a "go" file that you type and push
    after reading the exploratory results.
11. **Hosts.** Default: **TD3+BC, actor-only**. Options: the co-trained full host (about +4 CPU-hours per stage);
    ReBRAC (about +32 per stage, and a configuration must be chosen first). IQL needs a new method version. CQL is
    not offered.
12. **When.** Default: **after version 7's run on the desktop has finished**. Or alongside it, with the lanes split.
13. **Lanes.** Default: **10**. Any number from 1 to 16; lanes change no result.
14. **The name.** Default: **"version 8, near-80"**. It takes the version 8 that step 2's go licensed. Or a separate
    name, leaving "version 8" for a later version.
15. **The two readout marks of claim 2.** Default: **5 points each**. Or report them with no mark.

---

## 17. What could fool us, and what we expect

**Limits, with the guard against each:**
1. **The band's upper half was already empty.** After the ball, 80%-90% held only 1-5 full-throttle rows per seed,
   all inside the ball. Guard: every sentence says what was hidden; this document says the band adds only 70%-80%.
2. **Fewer rows, or fewer hard-water rows, not these rows.** Guard: every band cost is against the same number of
   random rows, drawn from calm and from hard water in the same numbers as the band's. So the control keeps the same
   share of hard-water rows. Check 1 tests that match per kind of water, and the noise floor can drop claim 3.
3. **Rivals getting worse is not BCA getting better.** Hiding the band may hurt the world-blind rivals in the no-stall
   world, so BCA's lead can grow while BCA itself does not improve. Guard: each arm's own points and band cost are
   reported in both worlds, and claim 1's sentence states BCA's own gain.
4. **Where full throttle stalls, the gain comes from the ramp's start and height, not from the switch.** There, BCA's
   actor is the always-on ramp's, which keeps BCA's height. Guard: claims 1 and 3 say so in words.
5. **New states and new commands together.** Strong current lies outside the recorded currents. Guard: stated in
   every claim.
6. **No method that reads only data can find the stall inside the unseen stretch.** Nobody has an outcome there. So
   the test measures how methods trade the good unseen commands against the bad ones. It cannot reward a method for
   knowing where the stall is. The two references that know the physics show what that knowledge would be worth.
7. **The band is thin.** It is 0.10 wide in one command. A smooth critic may simply bridge it. Guard: the critic's
   error at band commands is reported. Either answer is reported; neither is a failure.
8. **BCA's own inputs change.** Its second critic now has nothing above 70% before the survivors. Guard: the switch is
   named in claim 1's sentence in both worlds, and the heights are reported.
9. **Too few survivors on a fresh seed.** BCA's group needs at least 9 rows. Guard: check 6; such a seed is written
   down and never replaced.
10. **Holding to 70% is a what-if.** It changes later states too. Guard: the sentences say "held to 70%".
11. **The 35% floor was set after seeing seeds 0-4.** Guard: it is labeled wherever it appears.
12. **Fixed test start sets.** The strong-current set is 1.6-1.9 standard deviations on the hard side. Guard: the
    claims say "four fixed test start sets".

**Predictions, written before any training (not marks):**
- The no-caution actor still drives past 80% in strong current and scores near -14 points.
- The hand-set range limit loses about 10-19 points in strong current where full throttle stalls, and more where it
  helps.
- BCA's switched ramp loses some points where full throttle stalls: its ramp now starts at 70% and charges the good
  80% hold. It loses nothing where full throttle helps, if its switch stays off.
- The perfect band and the two physics references barely change.

---

## 18. Files

**The final version** (in `outputs/shift/beyond-the-data/near80/`; nothing trains before your approval is imported):

| file | what it holds |
|---|---|
| `PREREGISTRATION.md` | this document |
| `README.md` | a one-page summary |
| `approve_local_near80.py` | the approval script you run at your own terminal (standard library only) |
| `run_near80.sh` | the desktop entry: the official run, plumbing, status, the self-test, the plan, a retry, a budget continuation, the record that version 7 ended, the halt report, the results, and a stub run on a test fixture |
| `push_runfiles_near80.sh` | pushes the desktop's run files (model files included) to the branch `claude/near80-run-files`, as version 7's `push_runfiles_v7.sh` |
| `prereg.json` (+ `.sha256`, its fingerprint) | the machine-readable twin of this document, status "final, awaiting approval"; it copies section 14's sentences and bracket table verbatim and pins the code manifest |
| `code_manifest.json` (+ `.sha256`) | the fingerprint of every file in this table and of every repository file the code imports or executes, as in versions 6 and 7 |
| `prereg_history/` | the draft the final version was built from (`prereg_draft_<first 16 characters>.json`, with its fingerprint) |
| `pre_approval/no_training_checks_seeds0_4.json` (+ `.sha256`) | the frozen code's no-training checks on seeds 0-4, run before approval (nothing trained) |
| `code/near80_constants.py` | every shared constant: paths, seeds, settings, arms, marks, the runner's limits, the output layout |
| `code/near80lib.py` | the band, the near-80 removal, the random-rows control (drawn by kind of water), the leak check, the training-file check, the world card, the no-training checks |
| `code/facts_near80.py` | the no-training facts script (seeds 0-4 only) |
| `code/test_near80.py` | unit tests of the band, the removals, the leak checks and the facts, on synthetic logs |
| `code/approval_check.py` | the approval check every phase calls first, the git identity refusal, the test fixture |
| `code/import_approval.py` | the session's import of your approval (`show`, `import`) |
| `code/near80_prereg.py` | builds `prereg.json` from the draft, the choices taken, the pins and the claims word for word |
| `code/freeze.py` | `check` and `write`: the draft into history, the code manifest, the unit tests, the frozen code's no-training checks, `prereg.json` |
| `code/test_approval.py` | unit tests of the approval script, the import, the approval check and the freeze, on temporary copies and throwaway git repositories |
| `code/near80_runner.py` | the lane scheduler: plan, admission, launch, memory rule, ledger, safe stop and rerun, status, self-test |
| `code/phases.py` | the phase table: kinds, keys, dependencies, gating, done files, plain names |
| `code/phase_main.py` | the one entry every phase process runs (its checks, then the phase) |
| `code/settingdata.py` | the data of the three settings, fresh data for seeds 10-14, the training files |
| `code/learner_phases.py` | the critics, the full-log critic, BCA's second critic, switch and height, the cloned and reference actors |
| `code/policy_phase.py` | the actor phases (near-80 full set, random-rows reduced set, main setting on seeds 10-14) and the lock |
| `code/evaluate_phase.py` | the evaluations (as trained, held to 70%, capped at 80%), zone shares, lower tail, readouts, step 2's saved actors |
| `code/v6bca.py` | version 6's BCA band fit and width, restated on version 6's own functions |
| `code/ledger.py` | attempt records, peak memory, interruption classes, moving interrupted outputs aside |
| `code/stubs.py` | the stub mode: synthetic logs and learner stand-ins that take no training step |
| `code/test_runner.py` | unit tests of the runner, including a stub run of every phase kind |
| `code/pooled.py` | the looks: the stop rules, the marks, the validity conditions, the brackets, and the sentences of section 14 |
| `code/claims_text.py` | reads section 14 of this document word for word; fills the brackets |
| `code/test_pooled.py` | unit tests of the looks on synthetic results, one per sentence, stop and branch |

`run_near80.sh` follows version 7's pattern:
- `run`, `plumbing`, `status`, `selftest`, `plan`, `retry PHASE_ID`, `budget-continue STAGE`, `version7-ended`,
  `report`, `results` and `stub --fixture DIR`;
- one processor per phase, 10 lanes by default, version 7's memory rule and peak-memory ledger;
- safe stops and reruns of interrupted phases, a status command, a short self-test before every start;
- every phase checks the imported approval and refuses seeds 5-9; the near-80 and random-rows settings on seeds 10-14
  only after the exploratory stop rules pass;
- the training-file check before every phase that reads a training file;
- phases: data and no-training checks; critics; BCA's switches and heights; cloned actors; actor phases (near-80 full
  set, random-rows reduced set, main setting on seeds 10-14); the lock; evaluations (as trained, held to 70%, capped
  at 80%); readouts; the pooled looks.

Run the tests from `/tmp`, with the repository's own Python environment (numpy 2.2.6), one file at a time:

    cd /tmp && for t in test_near80.py test_approval.py test_runner.py test_pooled.py; do
      PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \
        /tmp/claude-0/bcavenv/bin/python /home/user/bca/outputs/shift/beyond-the-data/near80/code/$t || break; done

**Outputs** go to `runs/shift_explore/near80/`, write-once. Committed copies of results go to `near80/results-*/`
with their fingerprints (`SHA256SUMS`); the model files go to the branch `claude/near80-run-files`. Nothing is written
under `experiments/`, `runs/shift/`, `stage2/`, `stage2_gpu/`, `stage2_v7/` or any pinned folder.

**Sources:** the fact base and its scripts in the session's scratch folder (`near80/facts.md`, `facts/`,
`design_checks/`, `design_compute/`), the review's reruns (`near80/review/`) and the rerun after it (`near80/fixes/`);
version 6's `stage2/PREREGISTRATION.md` and `stage2/prereg.json`; version 7's `stage2_v7/README.md` and
`stage2_v7/prereg_v7.json` (its memory rule); step 2's `restructure/step2/PRESPEC.md` and
`restructure/step2_results/README.md`; the main test's results in `results-2026-10-13/` and `results-2026-10-14/`.
