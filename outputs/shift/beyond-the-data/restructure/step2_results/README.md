# Step 2 results: GO (exploratory)

Restructured BCA, step 2 (the policy pilot), official run on seeds 0-4. The pass marks, the arms and the code were
frozen and committed before any step-2 actor was trained (`../step2/PRESPEC.md`, prespec.json SHA-256
`839ed716...cd8da04`, code manifest `3222aedc...819fe8`). Nothing in `../step2/` was changed.

**Exploratory.** Step 2 was designed after seeing seeds 0-4 and step 1. A GO licenses drafting version 8 on fresh
seeds (10-14). It is not a claim.

## Words used here

- **Actor**: the controller trained from the recorded log. **Critic**: the learned model that predicts how good a
  command is. **"none"**: the actor trained against the critic with no caution.
- **The two worlds.** In the **stalling world**, a thruster's push collapses past 80% throttle, so full throttle
  hurts, but the critic (which saw almost no full-throttle commands) over-predicts it. In the **no-stall world**,
  push keeps rising, so full throttle helps. Each **seed-world** is one seed in one world: 10 in all.
- **Situations**: recorded conditions (the log's own range); **strong current** (currents 0.70-1.10, against the
  log's 0.10-0.70); big waves (wave sizes 0.40-0.70, against 0.10-0.40); big kicks (starts beyond every recorded
  state that settle back in about 10-30 steps: a short excursion, not a heavier tail). Each situation has a fixed
  **bank** of 1,000 starting states, the same for every arm.
- **Points**: 0 is the controller that made the log. 100 is that world's reference controller: the textbook rule
  cut off at 80% throttle in the stalling world, the same rule allowed full throttle in the no-stall world. It is
  not the best possible.
- **Best possible**: the exact best list of 100 commands from each start (`outputs/shift/oracle/`). It knows the
  step number, which no actor sees, so part of every shortfall is out of any actor's reach.
- **BCA's penalty** (arm `obca_k1`): a training penalty on high throttle. The log has almost no commands between
  about 0.79 and 0.98-0.995 throttle (the **gap**). The penalty is zero below the gap, rises steadily across it to
  its **height**, and stays there above it.
  - The height is BCA's step-1 estimate of how much more the critic over-predicts full throttle than other
    commands. It is in the critic's value units, not points, so a height does not translate into points.
  - The **dose** is a multiplier k times the height (k = 1 for BCA).
- **BCA's switch**: BCA's check from step 1. It turns the penalty on in a seed-world only if held-out recorded
  outcomes show the critic over-predicting full throttle by more than the rest, by a **margin** of 0.10 value
  units. It uses the 5% lower bound of BCA's Bayesian estimate of that difference.
- **Strength**: how far a penalty moves an actor's commands away from none's. **Matched**: tuned, without seeing
  any score, to move the actor as far as BCA's penalty does. **Even pull**: a stronger pull toward the recorded
  commands everywhere. **Range hinge**: a penalty on commands outside the recorded range; it reads where the data
  are, never an outcome. **Stand-in**: the range hinge matched to BCA's strength.
- **World-blind rival**: one that never reads an outcome, so it cannot switch itself off in the no-stall world.
- **NET**: points gained in the stalling world minus points lost in the no-stall world, both against "none",
  strong current, mean over the 5 seeds.
- **SE**: standard error (from the add-on report): how much a number would typically move with other seeds and
  another bank of starts.

## The verdict: all three pass marks were met

| mark (fixed before training) | needed | result |
|---|---|---|
| G1: BCA's NET beats the best world-blind rival | by 10 or more | +102.9 against +77.7: a lead of +25.3 (from unrounded values; SE 2.3) |
| G2: stalling world, BCA loses no more than 3 points to "none" in recorded conditions, big waves and big kicks | -3 or more each | +8.0, +21.2, +19.8 |
| G3: stalling world, strong current, BCA beats the matched even pull | by 15 or more | +100.8 (SE 5.1), ahead in 5 of 5 seeds |

The best world-blind rival was `obca_forced`: BCA's own penalty with the switch removed. It uses that seed's
stalling-world BCA height in both worlds, so it is always on.

## What the numbers say (strong current unless stated)

**BCA's penalty helped where full throttle hurts and did nothing where it helps.** In the stalling world it gained
+102.9 points over "none" (88.6 points against -14.3). In the no-stall world the switch stayed off in all 5
seeds, so BCA's actor scored exactly what "none" scored (94.9) and lost 0.0.

**The whole lead over the always-on rival comes from switching off.** In the stalling world BCA's trained actor is
bit-identical to the always-on rival's in every seed. The always-on rival then loses 25.3 points in the no-stall
world. That loss is BCA's lead.

**Rules that only look at where the data are lose in the no-stall world.** The stand-in netted +69.9: it gained
+92.7 in the stalling world and lost 22.8 in the no-stall world. BCA led it by 33.1 (SE 4.0). The unmatched range
hinge scored best of all in the stalling world (93.1 points against BCA's 88.6), because there the best possible
plan never goes past 80% throttle anyway. It lost 30.9 points in the no-stall world.

**Nothing is credited to calibration or to the Bayesian part.** The pre-set attribution tests (A1-A3) compared BCA
with three rivals. Each rival's switch made the same on/off decisions as BCA's, but its height was lower than
BCA's in the stalling world in every seed:

- `fixed_gated` uses BCA's own switch with a fixed height of 1.0.
- `conformal_k1` uses plain split conformal: the two groups' ordinary 90th percentiles, same 0.10 margin, no
  Bayesian estimate. Its switch matched BCA's in all 10 seed-worlds.
- `meangap_k1` uses step 1's rank test and the raw average difference, uncalibrated. Its switch matched BCA's in
  all but one seed-world (no-stall seed 2, where its height was a tiny 0.048).

| rival | NET | NET minus BCA's |
|---|---|---|
| plain split conformal (`conformal_k1`) | +97.3 | -5.6 (SE 0.9) |
| fixed height 1.0 (`fixed_gated`) | +89.3 | -13.6 (SE 2.5) |
| raw average difference (`meangap_k1`) | +57.2 | -45.7 (SE 3.1) |

Each test held if the rival came within 5 of BCA's NET; they decide only the wording of the claim. All three
failed. Plain split conformal missed narrowly: it was within 5 in 2 of the 5 seeds (seeds 1 and 4). The other two
were within 5 in none.

The fixed wording is therefore "a larger dose did better". BCA's own penalty at twice the dose (`obca_k2`,
k = 2) netted +106.6, 3.7 more than BCA (SE 1.0), and did better in all 5 seeds. That is within the pre-set 5-point
"plateau" band: BCA's height is large enough, which is not the same as calibrated. These seeds cannot tell whether
BCA's height did better because it is calibrated or Bayesian, or only because it is larger.

The perfect band, a reference penalty that knows the true one-step error, netted +92.6. Its shape differs from
BCA's and it is not matched in strength, so it says nothing about calibration either way.

**Against the best possible.** The best possible scores are 109.0 points in the stalling world and 104.2 in the
no-stall world. Total shortfall across both worlds: "none" 132.5 points, BCA 29.5, and the two 100-point reference
controllers (each built knowing its world) 13.1. In the stalling world alone BCA's shortfall is 20.3 against
"none"'s 123.3.

**Not only the average.** The **return** is the raw 100-step score (minus the total cost; closer to 0 is better,
and not on the points scale). In the stalling world, the mean return of the worst 100 of the 1,000 starts was
-250.3 for BCA against -509.2 for "none". In the no-stall world it was the same as "none".

## What this does not show

- **One task, two worlds, five seeds**, designed after seeing those seeds. For the NET comparisons most of the
  uncertainty is seed to seed.
- **The switch may not need BCA's Bayesian estimate.** In this run, plain split conformal's switch made BCA's
  on/off decision in all 10 seed-worlds. Route search round 1 (`../routes/round1/ROUND1.md`, on held-out rows
  scored by the gate's critic) found that a plain difference of means with the same 0.10 margin also switches
  perfectly, and credited the switch to reading outcomes against a fixed margin. Version 8 should include such a
  rival.
- **BCA's whole lead rests on one fragile decision.** In the no-stall world the critic still over-predicts full
  throttle slightly (+0.02 to +0.06 beyond the rest in 4 of 5 seeds, round 1). The switch stays off because of the
  0.10 margin and because there are only 15-30 full-throttle rows. Round 1 found that without the margin BCA's
  check raises false alarms in no-stall seeds 0-3, and that with more rows it might switch on (untested).
- **No mixed world.** Each world here is all-stall or all-no-stall. A world where some situations stall and
  others do not is still untested (round 1's mixed-world route).
- **Fixed banks.** All scores are on the same four banks of 1,000 starts. The strong-current bank is about 1.6-1.9
  standard deviations harder than an average bank of that size.
- **Calibration rows are treated as independent.** BCA uses 5 rows from each recorded episode as if they were
  unrelated. The measured design effect (how much rows from one episode inflate the uncertainty; 1 means none) is
  1.0-1.4, and step 1's switch decisions survive thinning to one row per episode. That check used the gate's critic
  as a stand-in for step 1's.
- **Coverage at the states the actor visits was not measured**: whether BCA's error bounds hold where the trained
  actor actually goes. Only scores and the share of visited steps past the gap's lower end were.

## How the run went

- The first launch at 01:45 UTC (5 parallel jobs) was stopped by low memory. Its last log lines are at 02:18 UTC
  (the restart note in `logs/run.log` estimated about 02:15). Its stalling-world actors had been trained in
  memory, but no lock (the write-once file of every trained actor's fingerprint), no scores and no results had
  been written.
- With the user's go-ahead it was restarted at 02:22 UTC with 4 parallel jobs and the same code. The 38 unit tests
  passed at both starts.
- Training finished at 03:34 UTC, scoring at 03:37 UTC. Peak memory per job was about 1.2 GB.
- During training, the penalty run with height 0 reproduced the earlier "none" actor bit for bit in all 10
  seed-worlds. That run comes first in the stalling world and beside the other ramp arms in the no-stall world.
  In the no-stall world every switched-off arm, BCA's included, reproduced it too.
- The add-on report (`addon_report/`; code in `../step2_report/`, committed before these results existed) decides
  nothing. It reproduced all 620 of step 2's bank means exactly, and every one of its checks passed.

## Files

- `results.json`, `results.md`: step 2's official results, copied unchanged from
  `runs/shift_explore/restructure/step2/`. `results.json.source.sha256` is the run's own checksum file.
- `addon_report/report.json`, `addon_report/report.md`: the add-on's regret, paired SEs, cost split and lower
  tail, copied unchanged from `runs/shift_explore/restructure/step2_report/`.
  `addon_report/report.json.source.sha256` is its own checksum file. Its `per_start.npz` (19 MB) is not committed.
- `logs/`: the run's logs. The lane logs hold both launches.
- `SHA256SUMS`: checksums of every file in this folder except itself.
