# Importance weighting for BCA: the pass marks, version 5

**In short.** Nothing has run yet. Version 5 builds in your two decisions:
- **Five dataset qualities**, as in D4RL: random, medium-replay, medium, medium-expert and expert. The expert is version
  4's data, exactly. Every mark is judged for each quality on its own, with the same numbers.
- **A second use of BCA's band, the lower edge.** Weighting barely moves BCA's own use of its band, a capped pull
  toward the recorded command. In the lower edge the band's width enters directly, so weighting moves it about ten
  times more.

If you approve, the orchestrator freezes the specification and records your approval. That approval covers fitting
BCA's band and the coverage checks, for every quality. The band is still fitted only after the first report says the
design holds; the code refuses otherwise. Training any BCA arm needs a further go from you: one go per kind of arm (the
capped pull, or the lower edge), naming the qualities it covers. The code refuses a kind or a quality your go does not
name.

All the numbers below are fixed in `spec.json` before any run, and every stage refuses a different specification.
BCA's stages (the band, coverage and the BCA arms) also refuse code other than the code you approve. The report refuses
results made by code other than the code running now, and says whether that code is the code you approved. "Code"
means every code file the run loads from this repository: this experiment's own files and the 32 files it reuses from
earlier work. A test checks that none is left out. Installed libraries, such as JAX and NumPy, are not fingerprinted,
but every result records their versions.

## What is compared

- **The world.** ThrusterChain: five floats in a channel, pushed by a current of strength c (fixed for an episode) and
  by waves, held by four thrusters. Past 80% throttle a propeller stalls. Every value is known exactly by simulation.
- **The known shift.** Only how often each current strength appears changes.
  - Training data and the calibration bank (the held-back rows BCA calibrates on) are mostly weak currents. The test
    is mostly strong currents.
  - So the exact weight of any recorded step is w(c) = test density / training density at its current. Under the
    strong test mix it runs from 0.14 at the weakest current to 7.4 at the strongest.
  - Three test mixes: none (the same as training), mild and strong.
- **The five dataset qualities.** One recording controller, one knob. The expert is version 4's recording controller.
  Each lower quality replaces some of its commands with uniform random ones (the random share).

  | Quality | Random share | Its data in points: training mix / mild / strong |
  |---|---|---|
  | random | every command | 0 / 0 / 0 |
  | medium-replay | from all to 65%, a learning curve | 25.8 / 24.7 / 23.7 |
  | medium | 65% | 48.7 / 47.0 / 45.6 |
  | medium-expert | half the episodes medium, half expert | 70.0 / 69.8 / 69.7 |
  | expert | none: version 4's data, exactly | 93.7 / 94.7 / 95.3 |

- **The action gap.** The recording controller often holds thruster 1 at exactly 80% when the water pushes hard. That
  is a good, common command. Because it repeats exactly, the standard rule that cuts the densest 15% of commands removes
  it from training and calibration data. At test it is common. The expert's gap is cut from every quality, so all five
  share it. Weighted calibration must refuse to certify that region, since no training row looks like it.
  - The random quality never holds, so it has no gap to speak of. It is labelled "weighting only".
- **The learner.** TD3+BC: an offline actor-critic learner (the actor picks commands, the critic scores them) pulled
  toward the recorded commands. Per quality and seed: one run on the gapped data (main) and one on the full data
  (control: what the gap costs).
- **BCA's band.** BCA is the method under test: an interval around the critic's score, meant to contain the critic's
  one-step error 90% of the time. Its width U = threshold × the band's unit × a scale network's output at the state and
  command. Weighting changes only the threshold.
- **Two uses of the band.**
  - **The capped pull** (BCA's own). The width sets a dose: how much harder the actor is pulled toward the recorded
    command, from 1 (no extra pull) to 1.5 (the most). The dose saturates, so weighting moves it by only about
    0.005-0.03.
  - **The lower edge** (new). A penalty in the actor's objective: the band's width at the actor's command minus its
    width at the recorded command at the same state, counted only where positive. The width enters directly, so
    weighting scales the penalty at each state by its own threshold over the unweighted one: 1.05-1.37 on synthetic
    banks built like ours.
    - Measured from the recorded command because a penalty measured from zero can reward going just past 80%, as the
      earlier cloud study found.
    - If a state's threshold is infinite ("cannot certify"), it uses the largest finite threshold of its table: the
      most caution that stays finite. This is my reading of "the largest finite width over the bank"; please check it.
- **The band's threshold**, set three ways on the same band:
  - unweighted (BCA as it is);
  - exact weights, step by step (the true w(c));
  - estimated weights, step by step (w from a classifier that only sees unlabelled training and test observations).
- **Step by step.** Each step's threshold uses its own weight, so the band is wider where the current is common at test
  and rare in training.
  - Computing the threshold afresh for every step is too slow, so it is tabulated on a fine grid of weights, 1% apart.
    Each step reads the next grid value at or above its own weight, which can only make the band slightly wider.
  - If a step's weight is too large to certify at all, that step is refused.
  - With no shift every weight is 1, and step by step is exactly the unweighted band.

**Terms used below.**
- **Coverage**: the share of steps whose critic error falls inside the band.
- **Refused**: a step the band declines to certify. It never counts as covered.
- **Return units**: the raw return over 100 steps, averaged over 1,000 fixed starts. Every mark on return reads these.
- **Points**: the same return on a 0-100 scale, shown beside it. 0 is uniform random commands, 100 the stall-aware
  controller (the textbook controller capped at 80%). 1.8 return units is version 4's 3 points.
- **Pooled**: the mean over a quality's five seeds.
- **Lower bound**: the one-sided 95% bound over the five seeds.
- **Holm's correction**: when a mark is tested on several qualities, each p-value must clear a stricter bar, so that
  the chance of any false pass across qualities stays at 5%. The group of qualities is fixed in advance: every quality
  whose own marks allow that mark's BCA arms. It is judged only once each of them has run its arms or you have
  declined it. A declined quality counts as a fail, so declining never makes the bar easier for the others.
- **Single threshold**: version 3's way, one threshold from the average weight for every step. Shown beside each
  step-by-step result; it judges nothing.

## The pass marks

Each mark is judged for each quality, on that quality's five seeds, with the same numbers. Nothing is pooled across
qualities.

### The data design (judged on the data alone)
1. **The design holds.** The gap removes at least 99% of the 80% holds. 82-87% of the expert's training commands lie
   outside the gap. In-gap share is 23-28% of the expert's test steps under mild and 33-39% under strong. The gap and
   this check are the expert's, shared by every quality, so this is judged once per seed. (Lower qualities' own
   commands lie outside the gap more often, because their random commands rarely land in it; that is reported.)
   - Pass: the data are as designed.
   - Fail: stop, for every quality. BCA's band is not fitted (the code refuses), and the design is fixed first.

### BCA's mechanics, no shift
2. **Coverage is right without a shift.** On held-out training-mix steps outside the gap: pooled coverage 89-94%, no
   seed below 86%.
   - Pass: the band is calibrated on this quality.
   - Fail: stop for this quality. Nothing later can be read.
3. **Width tracks error.** Within each tenth of the steps, sorted by current strength, the rank correlation between
   width and the exact critic error is averaged; its lower bound must be above 0.
   - Pass: the band is wider where the critic is more wrong, not just where the current is stronger.
   - Shown beside it: the same correlation after removing what both share with how far the floats have drifted. At
     low qualities the floats drift far, and width and error can both grow with drift. If the mark passes but this is
     near 0, the report says the width follows drift, not error beyond it.
   - Fail: coverage can still hold, but the width does not point at errors.
4. **The dose follows the width.** With steps sorted by width, the dose never falls by more than 1e-6 (rounding: BCA
   computes the dose in 32-bit floats).
   - Fail: a bug in the dose readout.

### Weighting under the known shift
5. **Weights of 1 change nothing.** Weights of exactly 1 give the unweighted threshold bit for bit. With no shift, the
   step-by-step band equals the unweighted band on every certified step.
   - Fail: a bug in the weighting code.
6. **Exact weights restore coverage.** Exact weights, step by step, on test steps outside the gap: pooled coverage
   89-96% under mild and under strong.
   - Fail: the weights or the bank size are wrong.
7. **The shift breaks the unweighted band.** Unweighted, strong test, outside the gap: pooled coverage below 88%.
   - Pass, with 6: the headline. Under a known shift, unweighted coverage falls short and exact weights restore it.
   - Fail: the band already adapts to the current on this quality, so weighting has nothing to fix there. That is a
     finding about the quality, not a broken method.
8. **The gap is flagged.** Both weighted versions refuse 100% of test steps inside the gap, at every mix. Outside it,
   exact weights refuse no step; estimated weights refuse only steps whose own weight is too large to certify. And
   unweighted coverage is lower inside the gap than outside.
   - That last comparison is judged only where every seed's test log has at least 500 steps inside the gap. The random
     quality has 30-69, so there it reads "not judged: too few steps inside the gap". The other qualities have enough.
   - Pass: BCA flags the unseen common command, and shows the unweighted band is over-confident there.
   - Fail of the refusal part: a bug.
9. **Estimated weights match exact ones.** Step by step, under strong, estimated-weight coverage is within 2 points of
   exact-weight coverage. With no shift, a classifier fitted on two training-mix samples gives coverage within 2
   points of unweighted.
   - Fail: estimated weights are only as good as their classifier.

### Return through BCA's capped pull
10. **Room for caution** (no BCA). An actor given extra caution placed by the exact critic error must beat an actor
    with no extra caution by at least 1.8 return units on the strong starts, with lower bound above 0. Even caution of
    the same strength is shown beside it.
    - Pass: caution of BCA's kind can raise return on this quality.
    - Fail: the capped pull cannot help here.
11. **Can weighting move return?** (no training). Predicted return change = how much step-by-step exact weighting moves
    each training row's dose, averaged × the return per unit of dose from the room check's even-caution actors. It
    must be at least 1.8 return units in size.
    - Pass: worth training the capped-pull BCA arms.
    - Fail: weighting moves BCA's dose too little to change return. Coverage is the result for this use of the band.
    - **Expected for the expert: fail.** The dose moves by 0.005-0.03 on average, which predicts well under 1 return
      unit. Below expert it is unknown: passing needs 62-360 return units per unit of dose, which is plausible for
      learners that sit far below the stall-aware controller.
12. **Improvement** (only if 2, 10 and 11 pass, and after your go for the capped pull on this quality). On the strong
    starts, BCA trained with step-by-step exact weights minus BCA unweighted, paired by seed. Each quality records its
    p-value; the summary applies Holm's correction across every quality where 2, 10 and 11 pass. A seed whose return
    is not a number fails the mark.
    - Pass: calibrating to the known test distribution raises test return through the capped pull.
    - Fail: the predicted effect did not appear.

### Return through the lower edge (new)
13. **Room for the lower edge** (no BCA). The lower edge, but with a perfect width: the critic's exact one-step error
    from the simulator, measured from the recorded command in the same way. An actor trained with it must beat an actor
    with no penalty by at least 1.8 return units on the strong starts, with lower bound above 0. An even pull of the
    same strength is shown beside it.
    - Before training, each seed checks that this perfect penalty actually rises past 80% throttle and pulls the
      best command back below 80%. If not, that seed stops, the report names which part failed, and this mark fails.
    - **Please confirm one choice.** If the critic on its own never prefers a throttle past 80% on the states checked,
      there is nothing past 80% to pull back, and the "pulls the best command back" part could never pass, whatever
      the penalty does. In that case I record that part as not applicable, and the seed trains as long as the penalty
      rises past 80% and does not itself create a preference past 80%. The report lists those seeds.
    - Pass: a lower edge can raise return on this quality, given a perfect width.
    - Fail: even a perfect lower edge does not help here.
14. **Can weighting move return through the lower edge?** (no BCA training). Predicted return change = (how much
    return rises per unit of extra penalty, measured between the perfect penalty at full strength and at 1.5 times
    full strength) × (how much step-by-step exact weighting raises the penalty at each training row, averaged). It must
    be at least 1.8 return units in size.
    - This is an approximation: it treats BCA's unweighted lower edge as if it acted like the perfect penalty, and
      weighting as turning that penalty up row by row.
    - So beside this mark, before your go, you also see how far that holds: on the training states, BCA's unweighted
      penalty against the perfect one (how large it is next to it, and how closely the two move together), and whether
      BCA's own penalty rises past 80%. This is shown only; it does not change the mark. If BCA's penalty is much
      smaller or larger than the perfect one, or does not rise past 80%, the prediction may say too much or too little.
    - Pass: worth training the lower-edge BCA arms.
    - Fail: weighting moves the lower edge too little, or return has already levelled off at the unweighted strength.
    - **Expected: unknown, for every quality.** Weighting raises the penalty by 5-30% on average, so the mark passes
      if return rises by about 6-36 return units per unit of extra penalty. The earlier cloud study found the perfect
      penalty recovered about 90 of 99 lost points in strong current, but half strength did about as well. If return
      has levelled off by full strength, this mark fails.
15. **Improvement through the lower edge** (only if 2, 13 and 14 pass, and after your go for the lower edge on this
    quality, which is separate from the go for the capped pull). On the strong starts, the lower edge with step-by-step
    exact weights minus the lower edge unweighted, paired by seed. Holm's correction across every quality where 2, 13
    and 14 pass, separately from mark 12. A seed whose return is not a number fails the mark.
    - Pass: calibrating to the known test distribution raises test return through the lower edge.
    - Fail: the predicted effect did not appear: BCA's width is not the perfect one, or the approximation in mark 14
      overstated it.

**Claims across qualities**, made by the summary only where the named marks pass on that quality:
- "BCA's mechanics work on this quality": marks 2, 3, 4 and 5.
- "Weighting restores coverage on this quality": marks 6 and 7.
- "Weighting raises return on this quality": mark 12, after Holm's correction.
- "Weighting raises return through the lower edge on this quality": mark 15, after Holm's correction.
- The last two claims wait, reading "pending", until every quality in their group has run its arms or you have
  declined it, so a claim is never made and then withdrawn.
- The table across qualities describes the results and judges nothing.

**Rules for every mark.**
- Marks pooled over seeds are judged only on all five seeds of the quality.
- If BCA's band fails on a seed, every mark that needs the band fails, including marks 11 and 14.
- The report refuses results made by code other than the code running now, and results that belong to another quality
  or seed than the folder they sit in.

**Shown without a mark.**
- Each quality's data in points, with and without the hold, beside its learner's points.
- Coverage with the full-data control; the forecast of unweighted coverage under the shift.
- How much weighting raises the threshold, and the share of steps refused because their own weight was too large.
- Step-by-step coverage beside the single threshold's; exact minus unweighted coverage beside its no-shift expectation.
- The cost of the gap: full-data TD3+BC minus gapped, on the strong starts.
- For the lower edge: the check on each seed's perfect penalty; how far the no-penalty actor of the new training
  program sits from the room check's (64-bit rounding); BCA's unweighted penalty next to the perfect one (beside mark
  14); each BCA arm's check that its own penalty rises past 80%.

## What to expect by quality

| Quality | Design | Mechanics and coverage | Capped pull | Lower edge |
|---|---|---|---|---|
| random | Shared gap; "weighting only" | Coverage marks as elsewhere; the inside-versus-outside comparison is not judged | Unknown | Unknown |
| medium-replay, medium, medium-expert | Shared gap | Unknown whether the shift breaks the unweighted band | Unknown; reach plausible | Unknown |
| expert (version 4's experiment) | Expected to hold | Version 4's expectations | Reach expected to fail | Unknown |

## The BCA arms, if they ever run

Each kind waits for its own go from you, naming the qualities, and runs only where its own marks pass. A go for one kind
does not start the other, and a go for one quality starts no other. If you decide not to run a kind on a quality where
its marks pass, that decision is recorded in your words, and that quality counts as a fail in mark 12's or 15's group.
- **Capped pull** (version 4's, marks 2, 10 and 11). Three arms per seed: unweighted; exact weights step by step (each
  training row reads its own threshold, a declared change to BCA; a row too heavy to certify gets the most pull,
  1.5); exact weights with the single threshold (comparison). Trained with the learner, 100,000 updates.
- **Lower edge** (marks 2, 13 and 14). Three arms per seed: unweighted; exact weights step by step under
  strong; the same under mild (shown only). They train only the actor, against the frozen gapped critic, like the room
  check. Before training, each arm's own penalty is checked for rising past 80%, and the result is shown.
- An estimated-weight arm of either kind runs only if you ask for it.

## Seeds and compute

- Five seeds (0-4) for every quality. Each has its own data, learner runs, band and room checks. All random streams
  derive from 20261016.
- One process per CPU, at most 20 at once on the 24 CPUs. CPU only, no GPU.
- Per run, measured:
  - data: about 1 minute per quality and seed;
  - learner: about 6.5 minutes per run (two per quality and seed);
  - room check: about 12-15 minutes per quality and seed;
  - room check for the lower edge: about 7-10 minutes per quality and seed (measured on synthetic arrays of the full
    size);
  - band and coverage: about 1 minute each;
  - evaluation: about 2.5 minutes per quality and seed.
- In total, about 990-1,140 process-minutes (16.5-19 process-hours): about 1.5 hours of wall clock on 20 CPUs. The
  lower edge's room check is 175-250 of these; the expert column is about a fifth.
- BCA arms, only where allowed and after your go: the capped pull about 200 process-minutes per quality (about 15
  minutes of wall clock); the lower edge about 14 process-minutes per quality (about 3 minutes per seed).

## Order of stages

1. You approve version 5 (or ask for changes first). The orchestrator freezes the specification: its SHA-256 (a
   fingerprint of the file's exact contents) is written beside it.
2. Your approval is written to `approval.json`, with the SHA-256 of the specification and of the code. Scope: band and
   coverage. One approval covers every quality.
3. The shared start banks, then data for every quality and seed, with exact truth for every row.
4. The TD3+BC learner on gapped and full data.
5. The room check, then the room check for the lower edge (neither uses BCA).
6. Exact returns of the learners and the room checks' actors.
7. First report per quality: "The design holds", "Room for caution" and "Room for the lower edge" are judged; the rest
   is pending. If the design check fails, stop. The code enforces this: BCA's band refuses to start unless the
   report, made by the approved code on all five seeds, says the design holds.
8. BCA's band on each frozen learner, with every threshold, and BCA's unweighted lower edge next to the perfect one.
9. Coverage of every version on held-out and test steps.
10. Full report per quality, then the summary across qualities, with the claims.
11. Only where a quality's marks allow, we ask you again, separately for each kind of arm. With your go, the BCA arms
    of that kind train on the qualities you name and are evaluated. Once every quality in a group has run or been
    declined, the improvement marks are judged with Holm's correction.
