# Deployment-time caution with the saved ThrusterChain actors

**EXPLORATORY. No training, no BCA.** No BCA band, calibrator or dose was fitted or read. Every number below comes from
exact simulation of actors that the gate already saved. Nothing under `experiments/shift` or `runs/shift` was written,
and no pre-registered rule changes. Gate seeds 0–4, base world, alpha 5.

## Short answer

1. **A check on new (state, command) pairs recovers about half of the strong-current harm, provided the fallback
   clips the command into the recorded range.** The check fires when the 10th-nearest-neighbour distance is above
   the 95% split-conformal threshold from held-out recorded rows. With it, the gapped critic's actor goes from −15.0
   to 42.1 points in strong current: +57.1 ± 2.5, which is 57% of the 99.5-point harm. It switches on 27% of steps.
   At the 99% threshold it recovers only +21.1 (21%).
2. **About a third of that recovery comes from targeting.** Random switching at the same rate gives +36.9, so
   targeting adds +20.2 ± 1.3 points in strong current. In big kicks, targeting is nearly the whole effect: +24.6
   from the check against +6.0 from random switching.
3. **Inside the recorded conditions it costs nothing.** With the clip fallback the check changes 2% of steps and
   *adds* 2.6 ± 0.1 points. The actor's 30-point gain over cloning, which plain TD3+BC also shows, is kept in full.
   With the cloned actor as the fallback, the same check costs 7.3 points there and 41.6 in big kicks.
4. **The limit is detection.** At the 95% level the check catches only 35% of the actor's past-80% proposals in
   strong current, although 96% of the times it fires are on such proposals. The distance is driven by the 13 state
   numbers: a check on the state alone does almost as well (+52.9), and swapping the harmful command for a recorded
   one barely changes the distance. A detector that caught every past-80% proposal would recover the whole harm
   (+106.1).
5. **This test bed rewards any limit to the recorded range. It cannot give credit for detection.** Clipping every
   command to the recorded range, with no check at all, scores 91.0 in strong current and 93.0 in recorded
   conditions. That matches the perfect detector and beats every real check. In this world every command past 80% is
   worse than 80%, and the recorded range ends just below 80%.
6. **The no-stall control shows the price.** In the no-stall comparison world, the missing commands are good. There
   the same 95% check costs 25.7 points in strong current (21.2 more than random switching), and clipping every step
   costs 61.8. The two worlds are identical up to 80% throttle, so the gapped data are nearly the same in both; only
   the few dozen surviving rows past 80% differ. So a novelty check finds commands the data do not show, not
   commands that are harmful.

## What was run

- **Actors** (saved by the gate's check 2, read through its verified loaders):
  - the gapped critic's actor (`gapped`);
  - plain TD3+BC on the gapped log (`td3bc_gapped`);
  - the full-log critic's actor (`full`);
  - as a control, the gapped critic's actor in the no-stall world (`no_stall_gapped`).
- **Switch rule.** At every step the actor proposes a command. A trigger decides whether to replace it with a
  fallback command for that step. The trigger is checked again at the next step.
- **Fallbacks.**
  - *Cloned actor:* the command of the behaviour-cloned actor that every check-2 actor started from.
  - *Clip to recorded range:* the actor's command, clipped per thruster to the range that holds 99% of the gapped
    training commands (0.5% trimmed at each end; about −0.75 to 0.51 for thruster 1).
  - *Nearest recorded command:* of the commands recorded at the 10 training rows nearest in state, the one closest
    to the actor's command.
  - *Clip to the full recorded range (minimum to maximum).* Reported only "every step".
- **Triggers.**
  - (a) *New state:* the distance from the state to its 10th-nearest training state, in standard units, is above
    the 95th or 99th split-conformal quantile of the same distance on held-out recorded rows.
  - (b) *New (state, command) pair:* the same test for (state, actor's command) against the training (state,
    command) rows.
  - (c) *Exact one-step check (reference):* switch when the fallback's command, followed by the actor for the rest of
    the episode, has a higher exact return than the actor's own command followed by the actor.
  - (d) *Random:* a coin with the rate at which (b) fired for the same actor, situation, seed and fallback.
  - Extras: (e) *command outside the commands recorded at nearby states*: how far the command lies outside the
    per-thruster range of the commands recorded at the 10 nearest training states, with conformal thresholds of the
    same kind, plus a random control at its rate; and (f) a *perfect detector* that fires exactly when the actor
    proposes past 80%. (f) uses knowledge of the gap, so it is a reference, not a method.
- **Held-out recorded rows.** The calibration split: 200 recorded episodes, about 16,800 rows, that no learner saw,
  with the same removal rule applied. On a further 5,000 held-out evaluation-split rows the thresholds fire on
  4.9–5.1% (95% level) and 0.9–1.1% (99% level) of rows, as designed.
- **Starts.** A fixed subset: the first 200 of each 1,000-start evaluation bank, the same 200 for every rule.
  Comparisons are paired by start and seed. The whole bank was too slow for about 1,500 rollouts with a
  neighbour search at every step.
  - Check: on the whole bank, "never switch" and "cap at 80%" reproduce the gate's check-2 points exactly (difference
    0 in all 80 actor × situation × seed cases).
  - On the subset, the gapped actor alone scores 86.0 / −15.0 / 70.3 / 57.0 in the four situations, against 85.5 /
    −14.3 / 70.0 / 52.5 on the whole bank.
- **Scores.** Points as in the gate: 0 is the recording controller and 100 the stall-aware controller (the
  rising-thrust controller in the no-stall world). Means are over the 5 seeds. "±" is the standard error over seeds
  of the paired difference.

## 1. The gapped critic's actor

Points, with the change from the actor alone in brackets. The "harm recovered" column gives the strong-current gain
as a share of the harm on these starts (the full-log critic's actor minus this actor, 99.5 points). The last column
is the share of steps on which the command was actually changed.

| Rule | Recorded conditions | Strong current | Harm recovered | Big waves | Big kicks | Changed (recorded / strong) |
|---|---|---|---|---|---|---|
| Actor alone | 86.0 | −15.0 | – | 70.3 | 57.0 | 0 / 0 |
| Every step → cloned actor (the cloned actor itself) | 56.0 (−29.9) | 53.9 (+68.9) | 69% | 43.2 (−27.2) | 5.1 (−51.8) | 1 / 1 |
| Every step → clip to recorded range | **93.0 (+7.0)** | **91.0 (+106.0)** | 107% | 90.0 (+19.7) | 84.9 (+28.0) | 0.06 / 0.77 |
| Every step → clip to full range (min to max) | 86.6 (+0.6) | −13.9 (+1.1) | 1% | 71.2 (+0.9) | 72.0 (+15.1) | 0.01 / 0.08 |
| Every step → nearest recorded command | 72.9 (−13.1) | 48.1 (+63.0) | 63% | 66.0 (−4.3) | 37.9 (−19.1) | 1 / 1 |
| New state 95% → clip | 88.3 (+2.3) | 37.9 (+52.9) | 53% | 74.6 (+4.3) | 81.3 (+24.3) | 0.02 / 0.25 |
| **New (state, command) 95% → clip** | **88.5 (+2.6)** | **42.1 (+57.1)** | **57%** | 75.1 (+4.8) | 81.6 (+24.6) | 0.02 / 0.27 |
| New (state, command) 99% → clip | 87.2 (+1.2) | 6.2 (+21.1) | 21% | 72.1 (+1.8) | 79.5 (+22.6) | 0.01 / 0.08 |
| New (state, command) 95% → cloned actor | 78.6 (−7.3) | 27.7 (+42.7) | 43% | 65.3 (−5.0) | 15.3 (−41.6) | 0.05 / 0.35 |
| New (state, command) 95% → nearest recorded command | 84.0 (−1.9) | 22.9 (+37.9) | 38% | 70.0 (−0.3) | 42.6 (−14.3) | 0.04 / 0.39 |
| Random at the 95% check's rate → clip | 86.4 (+0.5) | 21.9 (+36.9) | 37% | 71.8 (+1.4) | 62.9 (+6.0) | 0.00 / 0.21 |
| Random at the 99% check's rate → clip | 86.0 (+0.1) | −3.5 (+11.5) | 12% | 70.8 (+0.5) | 61.7 (+4.7) | 0.00 / 0.06 |
| Nearby-commands check 95% → clip (extra) | 89.9 (+3.9) | 24.8 (+39.7) | 40% | 75.9 (+5.6) | 80.3 (+23.3) | 0.01 / 0.24 |
| Perfect detector of past-80% proposals → clip (reference) | 93.1 (+7.2) | 91.2 (+106.1) | 107% | 90.2 (+19.9) | 85.2 (+28.2) | 0.04 / 0.70 |
| Exact one-step check → clip (reference) | 93.0 (+7.0) | 68.9 (+83.9) | 84% | 89.1 (+18.8) | 85.7 (+28.8) | 0.03 / 0.50 |
| Exact one-step check → cloned actor (reference) | 90.2 (+4.3) | 50.1 (+65.1) | 65% | 81.5 (+11.2) | 70.7 (+13.7) | 0.16 / 0.50 |

On the whole bank, the gate's own "cap at exactly 80%" gives this actor 93.9 in recorded conditions and 101.8 in
strong current.

### Targeting against random switching at the same rate

| Targeted minus random (same rate, same fallback) | Recorded | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| New (state, command) 95% → clip | +2.1 ± 0.1 | **+20.2 ± 1.3** | +3.3 ± 0.2 | +18.7 ± 0.2 |
| New (state, command) 99% → clip | +1.1 ± 0.1 | +9.7 ± 1.2 | +1.3 ± 0.1 | +17.8 ± 0.2 |
| Nearby-commands check 95% → clip | +3.7 ± 0.4 | +8.0 ± 2.5 | +4.8 ± 0.6 | +19.1 ± 0.5 |
| New (state, command) 95% → cloned actor | −6.4 ± 0.2 | +14.5 ± 1.6 | −3.3 ± 0.3 | −34.3 ± 2.0 |
| New (state, command) 95% → nearest recorded command | −1.5 ± 0.2 | +9.7 ± 1.5 | +0.4 ± 0.3 | −12.3 ± 0.1 |

In strong current the actor pushes past 80% on 73% of steps, so even random switching usually lands on a harmful
step. Targeting therefore adds less there than in big kicks, where only 16% of steps are past 80%.

### Which harmful commands each check catches (strong current, clip fallback)

| Trigger | Past-80% proposals caught | Firings that were on past-80% proposals | Points recovered |
|---|---|---|---|
| Random at the 99% rate | 8% | 73% | +11.5 |
| New state 99% | 9% | 97% | +18.0 |
| New (state, command) 99% | 11% | 97% | +21.1 |
| Random at the 95% rate | 28% | 73% | +36.9 |
| New state 95% | 32% | 96% | +52.9 |
| Nearby-commands check 95% | 33% | 99% | +39.7 |
| New (state, command) 95% | 35% | 96% | +57.1 |
| Exact one-step check | 67% | 97% | +83.9 |
| Perfect detector / every step | 100% | 100% / 71% | +106 |

When the checks fire, they are almost always right; they fire too rarely. Seed 0 shows why:

- The harmful command is almost always thruster 1 at full throttle (−1.0). At the states visited, the commands
  recorded for thruster 1 at the 10 nearest training states go down to about −0.72 (10th–90th percentile −0.78 to
  −0.66). So the harmful command lies about 0.28 throttle units outside them. That is exactly the 95% threshold of the
  nearby-commands check (0.28), set by the recording controller's own wobble on held-out rows.
- In the (state, command) distance the command barely registers. At those states, swapping the harmful command for
  its clipped version lowers the median distance only from 4.26 to 4.06 standard units, against a 95% threshold of
  4.30. The distance is set by where the state is, which is why the state-only check does almost as well.

The exact one-step check fires on only half the steps because it is myopic. It judges one clipped step followed by
the unclipped actor, and a single clipped step often does not help an actor that will stall again on the next step.

## 2. Plain TD3+BC on the gapped log and the full-log critic's actor

The same rules (the main ones; seeds 0–4, same 200 starts). For plain TD3+BC, "harm recovered" is the share of its
102.1-point gap to the full-log critic's actor in strong current.

| Rule | TD3+BC: recorded | TD3+BC: strong current | TD3+BC: big kicks | Full-log actor: recorded | Full-log actor: strong current | Full-log actor: big kicks |
|---|---|---|---|---|---|---|
| Actor alone | 86.4 | −17.6 | 52.8 | 91.3 | 84.5 | 68.4 |
| Every step → clip | +7.0 | +107.2 (105%) | +31.5 | +0.1 | +5.1 | +11.7 |
| New (state, command) 95% → clip | +3.1 | +59.7 (58%) | +28.1 | +0.2 | +3.6 | +11.8 |
| New (state, command) 99% → clip | +1.5 | +22.5 (22%) | +25.8 | +0.1 | +1.3 | +11.7 |
| Random at the 95% rate → clip | +0.4 | +39.1 (38%) | +6.2 | +0.0 | +0.9 | +2.0 |
| New (state, command) 95% → cloned actor | −6.6 | +46.1 (45%) | −37.2 | −8.5 | −10.8 | −51.1 |
| Every step → cloned actor | −30.4 | +71.5 | −47.7 | −35.3 | −30.6 | −63.3 |
| Exact one-step check → clip | +7.1 | +82.4 (81%) | +32.0 | +0.5 | +7.8 | +13.1 |

Plain TD3+BC behaves like the gapped critic's actor. For the full-log critic's actor, switching to the clipped
command costs nothing anywhere and helps in strong current and big kicks. The 95% check changes its command on 14%
of strong-current steps, against 27% for the gapped critic's actor. Falling back to the cloned actor costs every actor
7–9 points in recorded conditions and 37–51 in big kicks, because the cloned actor itself is poor there (5 points).
A fallback has to keep what policy optimisation gained; the cloned actor does not.

## 3. Control: the no-stall world, where the missing commands are good

These are the same rules for the gapped critic's actor of the no-stall comparison world. The removed full-throttle
commands are a strong hold in this world, not a stall. Change from the actor alone (93.5 / 95.1 / 88.0 / 93.7):

| Rule | Recorded | Strong current | Big waves | Big kicks | Same rule in the stalling world, strong current |
|---|---|---|---|---|---|
| Every step → clip | −2.0 | **−61.8** | −8.3 | −21.3 | +106.0 |
| New (state, command) 95% → clip | −1.3 | **−25.7** | −2.7 | −20.8 | +57.1 |
| New (state, command) 99% → clip | −0.7 | −6.5 | −1.5 | −20.2 | +21.1 |
| Random at the 95% rate → clip | −0.0 | −4.5 | −0.2 | −2.6 | +36.9 |
| Nearby-commands check 95% → clip | −0.9 | −18.4 | −1.8 | −19.1 | +39.7 |
| Perfect detector of past-80% proposals → clip | −1.9 | −61.6 | −8.0 | −21.0 | +106.1 |
| Exact one-step check → clip | +0.0 | +0.4 | +0.2 | +0.2 | +83.9 |
| New (state, command) 95% → cloned actor | −11.8 | −59.1 | −19.3 | −69.3 | +42.7 |

Here targeting hurts: the 95% check costs 21.2 ± 1.1 points more than random switching at the same rate in strong
current, and 18.2 in big kicks. It finds the unrecorded commands in both worlds. Only the exact check, which uses
the true outcome, can tell that they are harmful in one world and helpful in the other.

## Caveats

- **This test bed favours a blanket limit.** Every command past 80% is worse than 80% here: it gives less push for
  more energy. The recording controller's normal commands stop just short of 80%. So "never leave the recorded range"
  is free, and it beats every detector. Any future arm that uses a detector must be compared with this blanket clip,
  and the test should also score a world where the missing commands help.
- **A literal range check fails.** 34–59 training rows past 80% (0.08–0.14% of kept rows) survive the removal, so
  the full recorded range still allows thruster 1 to reach full throttle. Clipping to it changes nothing (+1.1 in
  strong current). A support check must ignore its most extreme 0.5% or so.
- The one-step check is not an upper bound. The perfect detector and the blanket clip beat it.
- 200 starts per bank, 5 seeds. The standard errors over seeds are small next to the effects reported. This is a
  diagnostic, with no pre-registration.

## What this suggests for BCA (interpretation, not tested)

- **Act on the actor's own command, where it acts.** Check 3 put extra pull on recorded rows during training and
  moved strong current by about 2 points. A limit on the actor's proposed command at the state it is in moves it by
  57–106 points. BCA's band is already a function of state and command, so it could be read at (state, actor's
  command): either as a deployment-time switch, or as a penalty in the actor's loss at the actor's own command,
  instead of a weight on recorded rows. Fitting the band and reading it needs the user's typed approval.
- **The band would need to be sharp about commands.** The stand-in checks are calibrated to 5% false alarms. They
  catch about a third of the harmful proposals and recover about 57 points; catching two thirds recovers about 84,
  and catching all of them recovers about 106. These numbers set the bar for BCA's detection: the share of the actor's
  past-80% proposals at which the band is wider than its 95% level on held-out recorded rows.
- **The fallback must keep the policy's gains.** Clip into the supported range; do not hand control to a cloned
  policy.
- **Be precise about what can be claimed.** Apart from the few dozen surviving rows past 80%, the gapped data are the
  same whether or not the thrusters stall. So from the data alone, no method (BCA included) can reliably tell that
  the missing commands are harmful rather than helpful.
  What BCA can claim to find is where the policy relies on commands the data do not support. Its caution there pays
  off when the policy's extrapolation is wrong and costs when it was right. A test that credits BCA should therefore:
  (i) score the stalling and the no-stall worlds together (or a world where the missing commands help in some places
  and harm in others); (ii) compare BCA with a blanket clip to the recorded range and with random switching at the
  same rate. The 34–59 surviving past-80% rows are the only evidence of the stall in the gapped log. Whether BCA's
  band grows near them is the detection question the gate never reached.

## Files

- Scripts: `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/fallback/fallback.py` (one
  seed, world and actor; also re-checks the gate's numbers), `summarise.py` (pooling), `worker2.sh` (job queue).
- Outputs: `/home/user/bca/runs/shift_explore/fallback/`:
  - `seed{0..4}_stall_{gapped,td3bc_gapped,full}.json` and `seed{0..4}_no_stall_no_stall_gapped.json`, each with a
    SHA-256 sidecar, the per-start points of every rule, the thresholds and the gate check;
  - `summary.json` and `tables.md` (every rule, with standard errors, catch rates and thresholds);
  - `logs/`.
- The gapped critic's actor ran every rule. Plain TD3+BC and the full-log critic's actor ran 21 of the rules. The
  no-stall control ran 13.
