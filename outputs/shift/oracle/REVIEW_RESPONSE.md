# Response to the external code review of PR #1

The review audited commit 10138a1. This response was written on 14 October 2026. Nothing was trained for it, no BCA
band was fitted, and no file of version 6, version 7, step 1 or step 2 was changed. Every claim was checked twice, once
by a checker and once by a skeptic trying to overturn it. The optimal reference was built once and then re-derived
independently.

## Words used here

- **Return**: the score of one 100-step episode, which is the sum of its rewards. Each step's reward is minus the sum
  of squares of the 10 float numbers (the five floats' positions and speeds), minus 0.3 times the sum of squares of
  the four throttles. Closer to 0 is better.
- **Points**: the project's score scale. 0 points is the recording controller, the controller that made the training
  log. 100 points is a fixed reference controller:
  - in the stalling world, the **stall-aware controller** (the textbook linear rule cut off at 80% throttle);
  - in the no-stall world, the **rising-thrust controller** (the same rule, allowed full throttle).
- **Stalling world, no-stall world**: in the stalling world, a thruster pushes less, not more, past 80% throttle. That
  point is called the **knee**. In the no-stall world, push keeps rising up to full throttle.
- **Bank**: a fixed set of 1,000 starting states used to score every controller. There is one bank per situation:
  recorded conditions, strong current, big waves and big kicks.
- **Best possible plan**: the best list of 100 commands from one start, computed exactly. The **optimal reference** is
  that plan for every start.
- **Regret**: the best possible score minus a controller's score, on the same starts. 0 is perfect.
- **Actor**: a trained network that picks the command. **Critic**: a trained network that predicts how good a command
  is.
- **The actors compared here**:
  - The **no-caution actor** (code name `none`) is trained against a critic built from the log with the full-throttle
    commands removed. It has no extra caution.
  - The **full-data actor** (`full_none` in version 6, `full` in the gate) is trained the same way, but its critic
    comes from the complete log.
  - **BCA's actor** (`bca_k1`) uses BCA's caution at strength 1.
  - The **range hinge** is a penalty on commands outside the recorded range. It looks only at where the data are,
    never at what happened. It is a "support-only" rule.
  - The **perfect band** takes its caution from the true error. It is a reference, not a method.
- **The gate**: the project's pre-registered acceptance checks (`acceptance.py`). Its saved actors and anchors are
  reused everywhere.
- **Seed**: one full repeat of the experiment, with its own log and its own networks. The gate has seeds 0-4. Version 6
  (Stage 2) has seeds 0-1 saved so far.
- **Version 6's pass marks** are called M1-M5. **Step 2's score** is called NET: what an arm gains where full throttle
  harms, minus what it costs where full throttle helps. Its go/no-go conditions are called G1-G3.
- **Discounted value**: what the learners and the truth code estimate. The task never ends, and a reward t steps
  ahead counts 0.98^t as much.
- **Standard error**: how much an average would typically move with another set of starts. **Paired**: the
  difference between two controllers measured on the same starts, which cancels most of the luck of the starts.
- **Support**: the range of situations the training data cover.
  - A **disjoint** test situation lies entirely outside that range.
  - An **overlapping** one lies inside it and differs only in how often each situation comes up.
- **Calibration bank**: the held-out recorded steps that BCA uses to set its band.
- **Design effect**: how much the likeness of steps from the same episode inflates the uncertainty. 1 means no
  inflation.
- **float32, float64**: single and double precision numbers. The deployed controller runs in float32 and the scoring
  code runs in float64.

## 1. Bottom line

The review is right in substance on all six points. Points 1, 2 and 5 are confirmed outright. Point 3 is confirmed by
the skeptic, although the first checker rated it "partly". Points 4 and 6 are partly right.

No computed number in version 6, version 7 or step 1 is wrong, and none of the checks moves a mark or a verdict. That
includes re-scoring the saved actors under the other objective, with paired standard errors, and in float32. What the
review exposes is interpretation and missing reporting.

The biggest new fact is the exact optimal reference, now built and independently re-derived:
- The 100-point reference is almost perfect in calm water: 0.8 points short of the best possible in recorded
  conditions.
- It is 9.0 points short in strong current, where the best possible score is 109.0 points. So an actor at 85 points
  there is 24 points from perfect, not 15.
- In the stalling world the best plan never goes past 80%. A rule that simply refuses unrecorded throttle therefore
  already has the right shape, which is why support-only rivals must be read first.

Everything that changes the environment, the objective or the metrics goes to version 8. Step 2 and version 7 only
get reporting added afterwards.

## 2. Point by point

| Review point | Verdict | What shows it | Already handled? |
|---|---|---|---|
| **1. The world is fully known.** True values belong to one controller and are not the best possible; the fixed banks are samples | Confirmed | All 13 state numbers are observed (float32 rounding at most about 2e-7). Two runs from the same start match bit for bit. Every "true value" is the 1,000-step discounted value of one controller (`truth.py:98-108`). The fixed strong-current bank sits about 1.6-1.9 standard deviations on the hard side | Mostly. The code and the page already say that a true value belongs to one controller and that 100 points is "not proven best", and bank standard errors are computed. Missing: paired standard errors, and the words "on these fixed starts" in version 6's marks and claim sentence. Step 2's frozen text calls 100 points "the best known controller" |
| **2. Commands past 80% are always beaten, so an exact optimum exists** | Confirmed | Full throttle pushes 0.50000003. The command equal to its own push gives a bit-identical next state for less energy (200,000 random cases checked). The best possible is 100.8-109.0 points | No. Only the warning "not proven best" existed. There was no optimum and no regret before this work |
| **3. Training, true values and the headline score use different objectives** | Confirmed. This is the skeptic's verdict; the first checker said "partly" (see below) | Training and truth use the never-ending task with discount 0.98. The headline is the plain 100-step sum, and the step number is not observed. Re-scoring version 6 seeds 0-1 under the discounted objective changes no pass mark and does not change step 1's GO. Rank agreement between the two scores is 0.96-0.98 | Disclosed on the page and in `SHIFT_ENV.md`, but never analysed until now. The page's "discounted return" is a third quantity: discounted, but cut at 100 steps |
| **4. The shifts answer different questions** | Partly | The training current is 0.10-0.70 and the wave size 0.10-0.40. The strong-current bank covers currents 0.70-1.10 and the big-waves bank wave sizes 0.40-0.70, so neither overlaps the training range. The weighting test uses currents 0.1-1.1 in both training and test. In big kicks, 98.6-99.1% of starts lie beyond every recorded float state and are back inside by about step 30. No analysis reports coverage at the states the deployed actor visits | Partly. The page already separates the fully-outside situations from the partly-outside one. Missing: a statement that the weighting test cannot show recovery under the main shifts, and coverage where the actor actually goes |
| **5. Keep the reward and add metrics** | Confirmed | The reward has no stall term (`thruster_chain.py:215-221`). No evaluator reports a cost split, peak or tail for learned actors. In strong current, BCA's mean gain came with a worse worst-10% in seed 0 and a better one in seed 1 | Partly. The cost split exists only for fixed controllers, and per-start points are saved only in the deployment files. "Risk" never names a measured outcome. `PLAN.md:138, 535` already admits that a range limit solves this world |
| **6. Calibration and numerical checks** | Partly | BCA takes 5 calibration rows per episode and treats them as independent; the measured design effect is 1.0-1.4. Taking one row per episode instead flips none of step 1's 10 certificates. In the weighting study's exploratory results E-F, the exact weights know each episode's random-command share. Float32 against float64 on trained actors: at most 0.0075 points per start | Partly. Taking few rows per episode is deliberate and documented (`calibration/bank.py:3-8`, `ALGORITHMS.md:699-710`). Support-only rivals exist in version 6, step 1 and step 2, but not in the weighting study |

### Where the review needs a correction

- **Point 2** holds only in the stalling world, and only for stall slopes up to 8 (the repository uses 1.5 and 3.0).
  - In the no-stall world the best plan does use full throttle (0.5-9.4% of its commands).
  - A reference there must allow full throttle. On 15 strong-current starts, capping it at 80% cost about 22 points.
- **Point 3**: in practice, training and truth share one objective (the truth leaves out only 0.98^1000, about 2e-9).
  That is why the first checker said "two objectives, not three". The skeptic upheld the review anyway: the repository
  itself reports the third quantity, and the advice to choose one objective stands.
- **Point 4**: big kicks is not just a heavier tail.
  - In theory its start law covers the same Gaussian range as the training starts.
  - In practice almost every start lies beyond every recorded float state, and the excursion lasts about 10-30 steps.
  - Weighting from the recorded start law to big kicks has infinite variance.
- **Point 6** is smaller than it sounds.
  - Rows from one episode are only slightly to moderately alike (design effect up to 1.4). They are nowhere near
    five copies of one row.
  - The problem of exact weights seeing labels the estimator cannot see affects only the exploratory results E-F.
    The pre-registered findings weight by the current, which the estimator can see.
  - Float32 effects are negligible.
  - Support-only rivals already exist everywhere except the weighting study.

## 3. The optimal reference

### What it is

The optimal reference is the exact best list of 100 commands from every start, in every bank, in both worlds. Three
facts make it exact:

1. Nothing is random after the start. A controller that reacts to what it sees therefore cannot beat the best fixed
   plan from the same start.
2. In the stalling world, a command past 80% is always beaten by the command equal to its own push. The floats move
   exactly the same way and the throttle cost is lower. So the best plan stays within 80%. In the no-stall world the
   limit is full throttle.
3. Within those limits, the floats move as a straight-line function of the commands and the cost is a sum of squares.
   The best plan is therefore the single answer of a bounded least-squares problem, which standard solvers find
   exactly.

That makes 8,000 plans (2 worlds × 4 banks × 1,000 starts), solved in about 3.5 minutes. Only the scoring code uses
them. No learner and no rule sees them.

### Did the independent check pass?

Yes, all four parts. The check used none of the build's code.

- **A different method gives the same answer.** A general-purpose optimiser on the raw commands, with slopes from
  automatic differentiation, matched the build on 1,000 starts to 3.4e-15 relative.
- **A guarantee on all 8,000 plans.** Each plan is within 2.5e-10 relative of the best possible.
- **The full throttle range was searched too, with the true stall.** 40,000 random plans and 640 local searches were
  tried, and none was cheaper.
- **The scores were re-done from scratch.** The anchors and all 568 saved points were re-scored, and they match
  exactly (largest difference 0.0).
- **The code was read line by line.** No bug was found.
- **Re-run end to end on the final code (15 October 2026).** `oracle.py` had been edited
  (21:03:57 local time) after its saved results were made, so it was re-run once in full (`oracle.py all`: solve,
  validate, nonconvex, discounted, score, float32, report; about 16 minutes). The best possible scores came out the
  same to the printed digit, so the confirmed numbers stand and can be cited.

### The best possible score, in points (confirmed)

| world | recorded | strong current | big waves | big kicks |
|---|---|---|---|---|
| stalling | 100.8 | 109.0 | 102.6 | 106.7 |
| no-stall | 100.6 | 104.2 | 101.0 | 103.9 |

### How far each controller falls short: regret in points (confirmed)

The learned actors here are version 6's seeds 0-1, so every row uses the same seeds. Smaller is better.

**Stalling world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| reference (100 points) | 0.8 | 9.0 | 2.6 | 6.7 |
| no-caution actor | 15.7 | 126.2 | 33.4 | 54.9 |
| full-data actor | 9.7 | 28.4 | 15.6 | 42.4 |
| BCA's actor | 9.8 | 86.9 | 20.4 | 46.3 |
| range hinge | 7.1 | 15.7 | 10.3 | 31.7 |
| perfect band (reference) | 7.3 | 29.0 | 12.6 | 36.9 |

- Over the gate's seeds 0-4, the regret is 15.3 / 123.3 / 32.6 / 54.2 for the no-caution actor and
  9.5 / 24.3 / 16.0 / 42.4 for the full-data actor.
- In strong current, measured from the no-caution actor's regret:
  - BCA's actor closes 31% of the gap to the best possible;
  - the full-data actor closes 78%;
  - the range hinge closes 88%.
- Per seed, in strong current: BCA's actor has regret 95 against the no-caution actor's 122 in seed 0, and 79 against
  131 in seed 1.

**No-stall world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| reference (100 points) | 0.6 | 4.2 | 1.0 | 3.9 |
| no-caution actor | 7.2 | 9.0 | 13.2 | 12.8 |
| full-data actor (gate, seeds 0-4) | 7.1 | 8.9 | 13.3 | 12.5 |
| BCA's actor | 10.8 | 26.6 | 20.1 | 19.5 |
| range hinge | 8.3 | 42.8 | 17.0 | 19.1 |

- Version 6 has no full-data actor in this world, so that row comes from the gate.
- Here the missing full-throttle commands would help, and caution costs regret:
  - BCA's actor gives up 3.6-17.6 points against the no-caution actor;
  - the range hinge gives up up to 33.8 points.

The best possible score is the same number for every controller on a given bank. So a difference between two
controllers is the same in regret as in points, and regret changes no ranking and no pass mark. What it changes is
how far "good" is from "best".

### Was "100 points" near-optimal?

- **In calm conditions, yes.** It is 0.8 points short in recorded conditions and 2.6 points short in big waves.
- **In the hard situations, no.**
  - It is 9.0 points short in strong current. That is 13.5 return per episode, about 13% of its cost.
  - It is 6.7 points short in big kicks.
- **Part of that gap needs a clock.** The best plan knows the step number, and no actor sees it.
  - A controller with no clock that re-plans 100 steps ahead from every state fell 0.3-0.7 points short of the
    per-start best (first 30 starts of each bank).
  - In recorded conditions, that is most of the 0.8-point gap.

### Where the loss is

Stalling world, strong current, per episode, in return units. **Tracking** is the cost of the floats being away from
rest. **Energy** is the throttle cost.

| controller | tracking | energy |
|---|---|---|
| best possible | 58.3 * | 32.1 * |
| reference (100 points) | 70.8 | 33.0 |
| no-caution actor, seeds 0-1 | 240.8 | 39.0 |
| BCA's actor, seeds 0-1 | 189.9 | 31.0 |

- **Almost all of the loss is tracking.** Energy barely differs between the controllers.
- **Where these numbers come from.**
  - The reference's split and both actors' splits match a separate replay done for review point 5.
  - The best plan's total (90.3) is confirmed. Its split (*) comes from the build only.
- **The lower tail moves either way.** The worst-10% mean return in strong current, from the same replays:
  - seed 0: no-caution actor -494.4, BCA's actor -509.7;
  - seed 1: no-caution actor -534.0, BCA's actor -472.4.

  So a mean alone can hide changes in the tail, in either direction.

### Numbers flagged as not independently confirmed

The independent check re-derived the best possible scores, the regret tables, the anchors and the saved points. It
did not recompute the following:

- **From the build only:**
  - the best plan's split into tracking and energy;
  - the splits for the gate's seed 0-4 actors;
  - the "late peak" (the largest float deviation after step 10);
  - the worst-5% means.
- **The discounted comparison** (section 7 of `results.md`), including "a perfect discounted optimiser loses only
  0.3-1.2 points on the 100-step score". The objectives check (review point 3) found the same thing by another route
  on version 6 seeds 0-1: the ranking agrees and no mark changes.
- **The float32 numbers.** The numerics check (review point 6) found the same order of size by another route: at most
  0.0075 points per start.

One more number was confirmed: in strong current, the best plan holds thruster 1 at exactly 80% in 74.1% of steps.

### Limits

- **The banks are fixed.** All scores are on the four fixed banks. The strong-current bank is about 1.6-1.9 standard
  deviations harder than an average bank of the same size.
- **Some actors were not scored.** The actors of version 6's best-case caution check were not scored.
- **The 100-step score depends slightly on knowing the time**, which no actor knows (see above).

## 4. Where each recommendation goes

### Step 2, now: reporting only, no retraining

Step 2's specification is frozen (`prespec.json.sha256`), its code refuses any change, and another agent owns its
folder. So everything here belongs in a separate report:
- it runs after the fact and reads step 2's locked actors and values files;
- it is labelled exploratory and decides nothing (not NET and not G1-G3);
- it should be written down before step 2's results are seen.

1. **Fix one sentence in the results write-up.** Step 2's pre-specification calls 100 points "the best known
   controller". It is the stall-aware reference. The best possible is 100.8 / 109.0 / 102.6 / 106.7 points in the
   stalling world and 100.6 / 104.2 / 101.0 / 103.9 in the no-stall world.
2. **Report regret beside points**: the best possible points minus the arm's points. This needs no rollout, and it
   leaves NET and G1-G3 unchanged by construction.
3. **Report the paired standard error of every pooled difference.**
   - Step 2's evaluator does not save per-start returns, so this needs one more rollout of each locked actor, about
     2 CPU-minutes per seed and world.
   - On version 6 seeds 0-1 these errors are about 0.6-0.7 points, too small to move a 10- or 15-point decision.
4. **Report more from the same rollout**: tracking cost, energy cost, the late peak, and the 10th percentile and the
   worst-10% mean of per-start return. Do not call any of it "risk".
5. **Label the situations.**
   - Strong current and big waves lie entirely outside the training range.
   - Big kicks starts beyond every recorded float state and settles back by about step 30. It is an excursion, not a
     heavier tail.
6. **Read the attribution ladder support-first.**
   - The "support rule alone" rungs are the range hinge and the matched stand-in.
   - The always-on ramp (`obca_forced`) keeps BCA's height, and in the stalling world it is exactly BCA's arm. It
     therefore tests BCA's on/off switch, not the support rule.
   - The fixed penalty is `fixed_gated`, the uncalibrated error scale is `meangap_k1`, and plain conformal is
     `conformal_k1`.
   - In the stalling world the best plan never exceeds 80%, so refusing unrecorded throttle is already the right
     shape. The stalling half of NET can be won without any calibration. The no-stall half is what tests whether BCA
     knows when not to refuse.
   - In version 6 seeds 0-1 the range hinge had regret 15.7 in strong current against BCA's 86.9. But in the no-stall
     world it had 42.8 against the no-caution actor's 9.0.
7. **Add two sentences on calibration.**
   - BCA treats the 5 recorded steps it takes from each calibration episode as independent. The measured design
     effect is 1.0-1.4.
   - Step 1's certificate survives thinning to one step per episode in all 10 combinations of seed and world. This
     was measured with the gate's critic as a stand-in, because step 1's critics are not on this machine.
8. **Add one sentence on units.** Penalties are in the critic's discounted value units, while scores are in
   undiscounted 100-step points. A penalty's size therefore does not translate into points.
9. **Say that all points are measured on the four fixed banks.**
10. **Not needed.** A float32 re-run (at most 0.0075 points per start) and a discounted re-scoring (the rankings agree)
    can be added as low-priority columns if wanted.

Throughout, the simulator stays out of every method arm. The perfect band stays labelled as a reference.

### Version 7: nothing changes

Why not:
- **It is frozen.** Version 7 is a frozen draft about where the frozen, approved version 6 runs. Its value as a
  confirmation depends on changing nothing after seeing seeds 0-4.
- **The checks give no reason to change it.**
  - On seeds 0-1, no version 6 pass mark (M1-M5) changes sign or crosses its threshold under the discounted objective.
  - Paired standard errors are about 0.7 points, against marks of 10-15 points.
  - Float32 moves a start's score by at most 0.0075 points.
- **What must not happen:**
  - re-anchoring 100 points to the best possible;
  - adding intervals to the marks;
  - changing the evaluator.

Allowed only after the confirmatory verdict, outside version 7's files, labelled exploratory, and never feeding a
mark:
- **A sentence on how the marks are computed.** They are pooled means of per-seed paired differences on the fixed
  1,000-start banks, so their spread over seeds leaves out the luck of the starts.
- **The same information table as step 2**, for seeds 5-9: regret, paired standard errors, cost split and lower tail.
- **Notes on what the results do and do not cover:**
  - the bank's coverage check covers only recorded steps at recorded commands;
  - the weighted fallback, if it triggers, reweights commands at recorded states, not at the states the actor visits;
  - big kicks is a short excursion.
- **Never calling 100 points optimal.**

### Version 8: design changes (fresh seeds 10-14)

1. **One primary objective.**
   - The cheapest consistent choice is the never-ending discounted task. Training, truth, BCA's band, step 1's
     certificate and the perfect band already use it.
     - It needs a new 0-point anchor: a recording controller with noise streams long enough for 1,000-step rollouts.
     - It needs the best possible discounted reference on the full banks. That is already solved for 200 starts per
       situation.
   - The alternative is a true 100-step task.
     - The time remaining becomes a 14th observation number, and the last step is not bootstrapped.
     - The truth uses the same remaining horizon, with no discount.
     - Every critic is retrained.
     - At the reference level this buys under 1 point.
   - Either way, report the other objective as secondary.
2. **The optimal reference as standard.**
   - Report regret, or the share of the achievable gain, beside points.
   - Use limits of ±0.8 in the stalling world and ±1 in the no-stall world.
   - Keep the stall slope at 8 or below. Above 8, the "never past 80%" proof breaks.
3. **One calibration row per episode.**
   - Today that would leave about 200 rows and raise BCA's threshold by 11-26%. Record about 4-5 times as many
     calibration episodes instead. Data are cheap: about 8 minutes per seed for a whole log in version 7.
   - Or keep 5 rows per episode, add a pre-registered dependence audit, and add a ThrusterChain entry to
     `calibration/dependence_evidence.json`. There is none today, so by the repository's own rule the bank counts as
     "not validated".
4. **Weights that see the same information.**
   - If an exact weight uses an episode label (such as the random-command share), also report the exact weight
     averaged over that label for each state and command. Or let the estimator see the label.
   - Put a support-rule-alone arm beside every weighted band, because refusing removed commands is a support rule, not
     calibration.
5. **Overlapping and disjoint shifts, labelled separately and paired.**
   - Keep strong current and big waves as the disjoint stress tests.
   - Add an overlapping test: the same recorder, the same removed command and the same physics, with the training
     current spread over 0.1-1.1 instead of 0.1-0.7, tested on the same strong-current bank. Compare full and gapped
     data under both.
   - Decide what big kicks is for.
     - As a shift that weighting can handle, its start spread must be at most about 0.6 against the recorded 0.5,
       which keeps an effective bank of about 340 of 1,000. At 0.65 only about 40 remain, and at today's 1.5 the
       variance is infinite.
     - Otherwise, call it an "excursion beyond the data" test.
6. **Coverage at three kinds of pairs, side by side.** Use the exact one-step error for each:
   - recorded states with recorded commands;
   - recorded states with the actor's command;
   - the deployed actor's own visited states with its own command.
7. **A mixed-context world.**
   - An observed number tells the controller whether this episode's thrusters stall. Full throttle then helps in some
     episodes and hurts in others, and a universal 80% cap loses in one of them.
   - Building it from the two existing worlds keeps the optimal reference exact: per episode, the limit is ±0.8 if it
     stalls and ±1 if not. This is a suggestion and has not been tested.
8. **A pre-registered metrics panel and failure event.**
   - The panel: tracking cost, energy cost, late peak (from a fixed step on), and the 10th percentile and worst-10%
     mean of per-start return, with intervals over starts and seeds.
   - Define a physical failure event with its threshold fixed in advance. Only that event may be called "risk".
   - State which numeric path it uses: `env.step` rounds commands to float32, and the truth rollout does not.
9. **Claims about all starts, not one bank.** Either draw fresh evaluation starts per seed, or pre-register paired
   per-start standard errors combined with the spread over seeds.
10. **A standing float32 check** of trained actors. For example, require a per-start difference under 0.01 points and
    report the 99th percentile.
11. **Mandatory support-only rivals**: the range hinge, the always-on rule that ignores the world, and the distance
    check.

### The page

Already right:
- a true value belongs to one controller;
- 100 points is "good but not proven best";
- the reading of the anchor's standard error;
- the split between fully-outside and partly-outside situations;
- the reward definition.

Text fixes, none needing a new experiment:

1. **Give the numbers behind "not proven best"** (`build.py:1836-1838`). Add the best possible scores: in the stalling
   world about 101 (recorded), 109 (strong current), 103 (big waves) and 107 (big kicks) points. The page already says
   "not proven best" (`template.html:1354-1355`); what it lacks is the numbers.
2. **Correct "the best linear rule if push always equalled throttle" or "if there were no stall"** (`build.py:1615`,
   `build.py:1883`, `template.html:798`, `symbols.py:158`). Change it to "the best rule for a never-ending, discounted
   version of the task with no throttle limit".
3. **Rename the "discounted return"** (`template.html:1346-1348`, `build.py:837-846`) to "discounted over the 100
   steps". Or replace it with the never-ending discounted value, which is what the critic estimates.
4. **Add one sentence on objectives.** Learners are trained for the never-ending discounted objective but scored on
   the plain 100-step sum. On the two saved version 6 seeds, the arm order and every mark are the same under both.
5. **Qualify "These 13 numbers are the whole state"** (`template.html:1238`). The step number is not observed, and
   the 100-step score depends on it a little.
6. **Add a sentence on the banks.** Every score is measured on four fixed sets of 1,000 starts. Differences between
   two controllers on the same starts are known to within about 0.2-2 points.
7. **Correct the weighting test section.**
   - Its training and test currents span the same range, so it tests weighting but cannot show recovery under strong
     current or big waves.
   - It uses a different recording controller, and it removes a good command (holding at 80%), not the harmful
     full-throttle command.
   - The sentence that the exact weight "depends only on its current" (`template.html:2930-2933`) holds only outside
     the removed commands. The "infinite weight" sentence (`template.html:3068`) is right.
   - The refusal that finding 3 credits to BCA (`template.html:3069`), and the gain of the fixed fallback in result F,
     both come from a support rule. Add a comparison with the support rule alone, or reword.
   - In results E-F the exact command weights know each episode's random-command share, which no estimator can see
     from a state and a command. The gap there measures privileged information, not estimator quality.
   - The static placeholder "0.14 - 7.36" (`template.html:2964`) should read 0.135-7.389.
8. **Describe big kicks precisely** (`template.html:1676-1682, 1708-1710`): "the floats start farther from rest than
   in any recorded step, then settle back within about 10-30 steps".
9. **Add a calibration sentence.** BCA treats the 5 steps it takes from each calibration episode as independent, and
   the measured design effect is 1.0-1.4. The weighting study's calibration set also takes 5 rows per episode.
10. **Add a float32 sentence.** Trained actors run in float32 match the float64 scores within 0.0075 points per start.
11. **Watch the names.**
    - Keep the word "risk" out until a failure event is defined.
    - A new rollout peak needs a name other than `peak_distance`, which is a steady wave swing.
    - A new energy cost needs a name other than `no_stall_cost`, which is the points lost to caution.
12. **Add a section on the restructured BCA (steps 1 and 2) and the optimal reference.** The page has neither yet.
    When it is added:
    - Show the best possible table and the regret table.
    - Put step 1's support-only comparisons beside its 106-point net. With the check always on, the net was 44.4.
      Random switching at the same rate scored 68.9 points, against 91.1.
    - Say that the stalling-world gain comes from an outcome-free rule that clips commands to the recorded range. BCA
      decides, per world, whether that rule runs.

## 5. Recommended order of work, with rough costs

This follows the review's order (align the objective, add the optimal reference, test full against gapped data under
unchanged and overlapping shifts, then the disjoint stress tests). Two adjustments: the optimal reference already
exists, and step 2 and version 7 run first, unchanged.

| # | What | Rough cost |
|---|---|---|
| 1 | Re-run the optimal reference end to end on its final code | done (about 16 minutes); numbers unchanged |
| 2 | Write step 2's reporting add-on as a separate script, outside step 2's frozen folder, before its results exist | a few hours of writing; no training |
| 3 | Run step 2 as frozen | 5-5.5 CPU-hours (1-2 hours of wall time) |
| 4 | Run the add-on on step 2's locked actors | regret: seconds; rollouts: about 20-30 CPU-minutes in all |
| 5 | Run the version 7 confirmation unchanged, then the same add-on, labelled exploratory | version 7's own budget (its longest chain is about 4.3 hours per seed); add-on about 20-30 minutes |
| 6 | Make the page text fixes | writing and a page rebuild; no compute |
| 7 | Settle version 8's design: objective, anchor, banks, calibration design, weights, metrics panel, failure event | writing; the best possible reference for the chosen objective on fresh banks takes minutes |
| 8 | Version 8 pilot: full against gapped data, under unchanged and overlapping shift | new logs, about 8 minutes per seed; training comparable to the gate (hours per seed; not costed here) |
| 9 | Disjoint stress tests (strong current, big waves) on the same actors | evaluation only, minutes |
| 10 | Mixed-context world | the largest item: new world code, logs, training and references; days of work |

## 6. Open questions for you

1. **Version 8's primary objective.**
   - Option one: the never-ending discounted task. This is my recommendation: it is the cheapest, and it matches
     training and BCA.
   - Option two: a true 100-step task with the time in the observation. That is a bigger change, because every critic
     is retrained.
2. **Version 8's points.**
   - Option one: keep 100 points as the reference controller and add regret beside it. This is my recommendation,
     because it keeps versions 6-8 comparable.
   - Option two: re-anchor 100 points to the best possible.
3. **Version 8's evaluation starts.** Fresh starts per seed, so that claims cover all starts, or the fixed banks plus
   paired standard errors?
4. **Calibration rows.** One per episode, with 4-5 times more calibration episodes, or 5 per episode with a
   pre-registered dependence audit?
5. **What counts as a physical failure?** For example, a float more than a set distance from rest after step 10. The
   threshold needs your choice before any data are seen.
6. **When the mixed-context world comes.** Is it part of version 8, or a later version? It is the largest piece of
   work, and the only one that stops a universal 80% cap from solving caution.
7. **Step 2's add-on.** Shall the add-on report be written now, before step 2's results exist, as a separate script
   outside step 2's frozen folder? And which session should run it?
8. **The page.** Should the text fixes, and a short section with the best possible scores, be made now or after
   step 2?
9. **The weighting study.** Should it get a support-rule-alone arm and weights that see the same information (a
   re-run), or only reworded text on the page?

## Files

- `/home/user/bca/outputs/shift/oracle/README.md`: the optimal reference, in plain language
- `/home/user/bca/outputs/shift/oracle/results.md` and `results.json`: every table and number
- `/home/user/bca/outputs/shift/oracle/oracle.py`: the code. To re-run, from /tmp, with
  `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=/home/user/bca`:
  `/root/bcavenv/bin/python /home/user/bca/outputs/shift/oracle/oracle.py all`
- `/home/user/bca/outputs/shift/oracle/cache/optimum100.npz`: the 8,000 best plans
- `/home/user/bca/outputs/shift/oracle/check/CHECK.md` and `check/results.json`: the independent check
- The claim-check scripts and their outputs are in the session scratchpad:
  `C:\Users\DAVIDB~1\AppData\Local\Temp\claude\C--Users-David-Bayha-BCA\b28b19ef-ba5a-4c06-a991-c30ce1f38aac\scratchpad`
