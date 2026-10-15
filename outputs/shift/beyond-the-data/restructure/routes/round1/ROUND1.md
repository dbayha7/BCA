# Route log, round 1 (2026-10-14)

**Integrity.** Every test here is exploratory. It was designed after seeing seeds 0-4 and run on gate seeds 0-4 only. No seed 5-14 or fresh data was used, nothing was written into a pinned folder, and no pre-registered pass mark was moved. Each kill rule was written before its result existed, hash-checked after the run, and the rule was not loosened. A separate verifier recomputed every deciding number. The five tests used about 63 CPU-minutes in total, and each stayed within its 90-minute allowance.

**How to read sources.** In each entry, `results.json` or `summary.json` means the file in `/home/user/bca/runs/shift_explore/routes/<that route>/`. Other files are given in full. A number with a source is measured. Anything marked [inferred] or [speculative] is not.

**Terms used below**
- **Critic:** the network that predicts how many points a command will earn.
- **Over-prediction score:** how much the critic's one-step prediction exceeded what the step actually delivered, in value units.
- **Full-throttle step:** a command above the top edge of the recorded throttle range, about 0.79. The training log had almost all of these removed; 28-59 "survivors" remain per seed-world.
- **Certificate:** step 1's frozen on/off rule. It switches caution on when the 5% lower bound of BCA's posterior gap (full-throttle scores minus all other scores) is above 0.10 value units. The 0.10 is the "margin".
- **Clip:** when caution is on, commands above the edge are cut back to the recorded range.
- **Net across worlds:** points gained in the stalling world at strong current minus points lost in the no-stall world, both measured against never clipping. A perfect switch scores 106.1. Clipping always scores 44.4.
- **Rank test:** a one-sided Mann-Whitney test at p < 0.05.
- **Plain split conformal:** comparing the two groups' ordinary 90th percentiles with the same 0.10 margin, with no Bayesian posterior.

**Round 1 in brief:** 3 routes alive, 2 shelved.
- Every gain this round came from **reading outcomes**: the over-prediction scores of full-throttle steps the critic did not train on, compared against a **fixed 0.10 margin**.
- **BCA's Bayesian posterior** added little at realistic evidence sizes. The one hint in its favour is that it was less sensitive to where the margin sits. That hint comes from an after-the-fact sweep, so it is not a finding.

---

### Held-out calibration rows as the certificate's evidence - ALIVE: next step - check that the no-stall world stays off when more rows exist (forward passes only, on the trace test bed's saved critics)

- **The idea:** Each seed-world's calibration episodes already hold 15-30 full-throttle rows that no critic ever trained on, but BCA's calibration bank keeps only 0-2 of them. Feed those rows to step 1's certificate, so the 10 retrained critics step 1 needed (about 70 CPU-minutes) are not needed.
- **What we tested:** We ran step 1's frozen certificate on those rows. The rows were scored by the gate's own critic, with forward passes only (4.4 CPU-minutes). The kill rule, written before the test (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/heldout_calibration_rows/KILL_RULE.md`), shelves the route if any of these holds:
  - the certificate is not on in all 5 stalling seeds and off in all 5 no-stall seeds;
  - the net across worlds is below 90;
  - shuffled outcomes keep more than half the net.
- **What happened:** No kill condition fired.
  - **Result:** the certificate was on in all 5 stalling seeds, with lower bounds +0.36 to +0.84 against the 0.10 margin. It was off in all 5 no-stall seeds, with lower bounds −0.01 to +0.02 [results.json:per_seed_world.*.certificate.q05].
  - **Net:** 106.1, which equals step 1 and the perfect switch [results.json:net]. With shuffled outcomes the mean net was −0.06 [results.json:shuffle.mean].
  - **Rivals:** clipping always scored 44.4. The rank test scored 93.6, because it raised a false alarm in no-stall seed 0 (p = 0.021) [results.json:information_only].
  - **The verifier's correction:** without the 0.10 margin, BCA's certificate nets only 55.7, with false alarms in no-stall seeds 0-3. With the margin, a plain difference of means, of medians or of 90th percentiles also nets 106.1 [`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/heldout_calibration_rows/verify/out.json`:summary].
  - **Size of caution:** the "height" was finite in only 2 of 10 seed-worlds [results.json:per_seed_world.*.certificate.height].
- **Why:**
  - **Stalling world:** full throttle delivers less than the critic expects. Rows the critic never trained on show this plainly: they are over-predicted by +0.15 to +0.33, against about −0.16 for ordinary rows [results.json:per_seed_world.*.groups].
  - **No-stall world:** these rows are only slightly worse than ordinary ones, by +0.02 to +0.06 in 4 of 5 seeds (verifier). The certificate stays off only because of the 0.10 margin plus the extra uncertainty from having just 15-30 rows. With more rows it might switch on [inferred, untested].
  - **What this adds:** a cheaper, cleaner source of the same 106, not a new gain. The in-sample survivors also reach 106 in this test bed, so the general advantage of held-out rows is not shown here. The credit goes to reading outcomes with a fixed margin, not to the Bayesian posterior.

### Run-time monitor on the deployed actor's own outcomes - SHELVED: dead end

- **The idea:** After deployment, keep scoring the actor's own steps, and start clipping once its full-throttle steps turn out worse than the critic predicted. This could catch harm that lies only beyond the recorded log, which no offline method can see.
- **What we tested:** Four worlds, built exactly from saved files:
  - stalling;
  - no-stall;
  - "harm only beyond the log": the no-stall log, deployed into stalling physics;
  - "harm only inside the log": the mirror of the previous world.

  The saved actors trained with no caution ran 200 strong-current episodes. The certificate was re-checked after every episode, and once on it stayed on. The kill rule, written before the test (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/runtime_monitor/KILL_RULE.md`), shelves the route if any of these holds:
  - it ever switches on in a world with no-stall physics;
  - it is not on by episode 6 in a world with stalling physics;
  - its four-world net is below 85% of the oracle's;
  - a score-shuffled control keeps half its net.
- **What happened:** Three of the four kill conditions fired.
  - **What worked:** it caught the stall fast, after episode 2 or 3 in all 10 stalling-physics seed-worlds. It earned 112.3 in the "harm only beyond the log" world, where step 1's offline certificate earns 0 [summary.json:latch_after_episode_real, nets.real, nets.offline_step1].
  - **What failed:** it also switched on in all 10 no-stall-physics seed-worlds, after episodes 2-11 [summary.json:latch_after_episode_real]. So in practice it became "clip always": four-world net 102.1, against 103.3 for clipping always.
  - **Kill-rule numbers:** the oracle reaches 219.7 (85% bar: 186.7). The shuffled control kept 74.7 (bar: 51.1) [summary.json:nets].
  - **Not an artefact of looking 200 times:** a single look at all 200 episodes still gives a statistic of 0.24-0.65 in every no-stall world (verifier, `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/runtime_monitor/verify_adversarial/fulldata.py`, printed output).
- **Why:**
  - **The critic is guessing everywhere.** At strong current every state lies outside the log. Its misses at full throttle are somewhat larger than at gentler throttle even when nothing stalls: +0.06 to +0.24, against −0.10 to +0.02 [seed*/prep.json:worlds.*.score_mean_tail, score_mean_rest].
  - **The margin is too small for this setting.** The 0.10 margin was set in recorded conditions, and this ordinary extrapolation error clears it within a few episodes.
  - **The size is there, but the rule ignores it.** A real stall produces gaps about ten times larger: the statistic was 3.8-4.5 at switch-on, against 0.10-0.46 without a stall [summary.json:at_switch.*.q05]. The frozen rule only asks "above 0.10?".
  - **A rescue would be a new route.** A size threshold picked now would be fitted to these seeds. It would need its own kill rule and unseen seeds [speculative].

### Stall-strength ladder ("does caution switch on where clipping starts to pay?") - ALIVE: next step - rerun the same ladder with step 1's 10 held-out critics and the survivors added (forward passes only, about 5 CPU-minutes), margin left at 0.10

- **The idea:** Vary the stall strength in small steps from no stall to a harsh stall. Rules that look only at where data lie make the same choice at every strength, because the recorded commands are identical. An outcome reader can follow the harm as it grows. Does BCA switch on near the strength where clipping starts to pay?
- **What we tested:** Nine stall strengths (sigma), all set by the physics code's own setting:
  - −1 is no stall;
  - 1.5 is the stalling world;
  - 3 is harsher still.

  At each strength we recomputed exactly the one-step outcomes of the no-stall log's 15-27 held-out full-throttle rows, ran step 1's certificate unchanged, and measured the value of clipping by rollouts. The kill rule, written before the test (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/harm_dose/KILL_RULE.md`), shelves the route if either holds:
  - BCA's total over all 9 strengths is below 80% of the oracle's;
  - over the 7 middle strengths, BCA beats the better of "always clip" and "never clip" by less than 10 points.
- **What happened:** Neither kill condition fired.
  - **Value of clipping:** it rises smoothly, from −32 points with no stall to +114 in the stalling world and +381 at sigma 3. It starts to pay near sigma 0.21 [results.json:value_seed_mean, switching.break_even_linear_crossing].
  - **Totals:** over all 9 strengths BCA scores 541.1, against the oracle's 570.3 (95%) [results.json:kill]. Over the middle 7, BCA scores 427.4 against always-clip's 404.4, which is +23.0 against a bar of 10 [results.json:kill].
  - **Switch-on points scatter by seed:** −0.25, −0.25, 0.25, 1.5 and 1.0. Seed 3 only barely switches on (0.102 against 0.10) [results.json:switching.bca_switch_on_sigma, per_seed_sigma.3.1.5.cert.q05].
  - **Rivals:**
    - Plain split conformal at the same 0.10 margin: 538.7 over 9 strengths and 425.1 over the middle 7, within 2.4 points of BCA [results.json:ladder_nets].
    - Rank test: 349.7 over the middle 7, worse than always clipping [results.json:ladder_nets].
  - **Robustness (verifier):**
    - Dropping any one seed keeps the +23 between +15.7 and +33.1.
    - One seed of five is negative on its own (seed 3: −17.2).
  - **After the fact, not deciding:** across margins from 0 to 0.20, BCA's lead over the middle 7 stays between +13.7 and +37.4. Plain split conformal's lead is +0.0 at margin 0, +4.4 at 0.04-0.067, +9.4 at 0.08, +20.7 at 0.10 and +29.1 at 0.20 [`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/harm_dose/after_the_fact/margin_sweep.json`].
- **Why:**
  - **How the switch works.** The few recorded full-throttle steps are the only place the stall shows. The critic's over-prediction on them grows with the stall, and BCA clips once that gap clears 0.10.
  - **The margin landed near break-even.** So BCA avoided most of the cost of clipping in mild worlds (41.1 points avoided) and kept most of the gain in harsh ones (18.1 points missed by the two late seeds) [derived from results.json:per_seed_sigma].
  - **The switch point is noisy.** With 15-27 rows per seed it moved between seeds.
  - **A possible role for the posterior.** Its 5% lower bound behaves like an automatic allowance for small samples, which made it less sensitive to the margin than plain percentiles [inferred from an after-the-fact sweep].
  - **Caveats:**
    - Sigma 3 dominates the 9-strength total: always-clip alone reaches 85% of the oracle.
    - The +23 rests on 4 strengths near break-even and on 5 seeds.
    - The middle worlds are hypothetical: only one-step outcomes were changed, and no actor or critic was retrained.

### BCA's posterior against plain tests on thin evidence (12-45 rows) - SHELVED: dead end

- **The idea:** If BCA's Bayesian posterior earns anything, it should show when evidence is thin. So compare it with plain tests at the evidence sizes ThrusterChain actually has.
- **What we tested:** For each seed-world, 400 random draws of n rows from the 6,100-6,750 removed training rows the gate critic never saw. Four switches were compared:
  - step 1's certificate;
  - plain split conformal;
  - the rank test;
  - the certificate with margin 0.

  Expected nets came from the saved deployment values (11.9 CPU-minutes, no rollouts). The kill rule, written before the test (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/small_evidence/KILL_RULE.md`): for each rival, average BCA's net minus the rival's over n = 12, 20, 30 and 45. If that average is under 5 points, shelve the claim that BCA beats that rival; if both claims fall, shelve the route.
- **What happened:** Both claims were shelved.
  - **The deciding numbers:** BCA beat plain split conformal by 1.78 and the rank test by 4.29, both under 5 [results.json:decision.D_C, D_R].
  - **Nets at n = 12:** BCA 105.6, plain split conformal 102.3, rank test 99.1, margin 0 96.9 [results.json:nets.*.12].
  - **Where the rules differ:** every rule catches the stall almost every time from n = 12. They differ only in no-stall false alarms at n = 12: BCA 0-0.5%, plain split conformal 5.5-7.2%, rank test 2-11%. With margin 0, BCA's false-alarm rate is 4.5-43% [results.json:on_rate].
  - **Verifier's breakdown:**
    - Half of BCA's lead over the rank test (2.14 of 4.29) comes from no-stall seed 1 alone.
    - Of that lead, about 40% is the posterior and about 60% is the margin. Dropping the margin inside BCA costs 9.65 points (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/small_evidence/verify/decomp.py`, printed output).
  - **Where the posterior does help:** only at 5-8 rows, where BCA beats plain split conformal by up to 12.3. ThrusterChain never has that few [results.json:nets.*.8].
- **Why:**
  - **Detecting a stall is easy.** A real stall makes full-throttle outcomes so much worse than predicted that any sensible test notices with about a dozen rows.
  - **Avoiding false alarms is what counts.** What separates the rules is false alarms in the no-stall world, and each one costs about 60 points.
  - **The margin, not the posterior, prevents them.** False alarms are mostly prevented by the fixed 0.10 margin: the posterior without the margin was the worst of the four.

### Mixed world split by wave size (stalls in big-wave episodes, helps in small-wave ones) - ALIVE: next step - add plain throttle caps (0.75-0.95) and a local-cap arm as rivals, because a cap at the true knee already nearly matches it; only then the spliced-log critics (about 70 CPU-minutes, needs a separate go-ahead)

- **The idea:** Build a world where full throttle hurts in some episodes and helps in others. Any single yes/no for the whole world must lose on one side. BCA's weighted posterior, localised by wave size, could clip only where it hurts.
- **What we tested:** Two mixed worlds spliced exactly from whole saved episodes, which is possible because wave size never changes within an episode:
  - one stalls when wave size is above 0.25;
  - the other is the mirror.

  Per episode, BCA's posterior weighted evidence rows by closeness in wave size and applied step 1's statistic and 0.10 margin; when on, that episode clips. The evidence was either the held-out calibration rows ("primary": 18-32 per world) or about 6,500 removed training rows ("rich"). The kill rule, written before the test (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/mixed_waves/KILL_RULE.md`):
  - shelve the route if, with rich evidence, it beats the best of {world-level certificate, clip always, never} by under 10 points;
  - shelve the claim that it works with the evidence a gapped log leaves if the primary margin is under 10, or if an outcome-shuffled control keeps half of it.
- **What happened:** No kill condition fired, but a rival outside the kill rule nearly matches it.
  - **Local rule:** 43.2 with primary evidence and 50.3 with rich evidence. The best named rival (clip always) scores 24.4, and an oracle that knows which region stalls scores 52.6 [results.json:summary, best_rival]. The margins are 18.8 and 25.9 [results.json:margins].
  - **Shuffle control:** shuffling outcomes drops the net to 14.5, below clipping always [results.json:permuted_control].
  - **Where it clips:** with primary evidence it clipped 91% of starts in the stalling region and 17% in the helping region [results.json:on_rates_by_region].
  - **The verifier's correction, outside the kill rule:** a plain throttle cap, which reads no outcomes and no wave size, nets:
    - 41.5 at 0.80 (the true knee, so it is placed using knowledge of the physics);
    - 41.2 at 0.82;
    - 40.0 at 0.84.

    Across the plan's 0.75-0.95 range it nets 17.9-41.5 [seed-weighted mean of `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/mixed_waves/verify/verify_0_1_2_3.json` and `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/mixed_waves/verify/verify_4.json`:mixed_world_means.cap_*].
  - **Not told about wave size:** a nearest-neighbour version over the full state scores 15-20, no better than the world-level rivals [results.json:summary.knn_primary, knn_rich].
- **Why:**
  - **Real, outcome-driven localising.** Only the outcomes of full-throttle steps tell harmful episodes from helpful ones. Weighting the evidence by wave size lets BCA ask "does the critic overshoot near this wave size?". The shuffle control shows the gain comes from outcomes, not from where the rows lie.
  - **First limit:** the designer told the rule which variable to localise on.
  - **Second limit:** all the named rivals use the same blunt action, clipping back to the recorded range. A cap at the knee closes most of the gap without reading any data, so part of the gain is the action, not the evidence [inferred from the nets].
  - **Bottom line:** the route beat the rules its kill rule named, but not yet the best rule that reads no data.

---

### Not tested this round
- **Caution height scaled by the critic's own posterior, and its coverage at strong current:** its scale measures novelty, not harm, which is close to shelved routes. The test is only a proxy. Its motivating premise (that the height is too small) is in doubt, since step 1's held-out heights are 1.6-6.0 (runs/shift_explore/restructure/step1/logs/evaluate.log).
- **Correcting the critic with the survivors (a last-layer update, or a survivor-weighted retrain):** adding 2-5% real full-throttle rows back still left 71.5-85.8 of 99.5 harm points (runs/shift_explore/trace/pooled_v3.json). Little of any gain would be BCA's own.
- **Multi-step residuals:** dead by construction. The one-step score already carries the stall's cost, and the log never holds two stalls in a row (0 consecutive survivor pairs in 10 of 10 seed-worlds).
- **Residuals of the actor's own critic:** only the stalling world's version is saved, so it cannot show a gain across worlds without new training.
- **Outcome-priced selection among range rules:** it restates a known result: the gated hand-set hinge already gives 107.4 (results-2026-10-14/README.md). That gain belongs to the hinge, and its kill rule could not be blind.
- **Separate online outcome monitor:** merged into the run-time monitor above, now shelved.
- **Caution height used as penalty strength:** this is the planned step-2 pilot, which needs the user's own approval. Its key input has already been seen, so its kill rule could not be blind.
- **Current-local strength:** it depends on deferred routes. Its evidence sits only at currents 0.2-0.7, with 2-8 rows at low currents, so it cannot see strong current.
