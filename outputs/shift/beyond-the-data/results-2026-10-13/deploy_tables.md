# Deployment-check channel (plan 4.3): pooled tables

EXPLORATORY. No training and no BCA: every trigger is a stand-in or an oracle. Saved check-2 actors, exact simulation, the first 200 starts of each evaluation bank, gate seeds 0-4. Differences are paired by seed; ± is the standard error over seeds.

Launch condition, shared by every job: Launched after the room test's pooled look passed its room, recovery and keeps-improvement marks (room pooled.json SHA-256 dde4ac9ff4da3f31bd92784cc5ffde0e99d4103f61f7a5bc1eb8f1e7c74a97c1).

## The mark: is the channel open?

Perfect detector minus random switching at its own rate (gapped critic's actor, stalling world, clip fallback): strong current +23.2 ± 1.0, big kicks +23.0 ± 0.3 (need at least +15 in one). Cost in recorded conditions: -7.2 ± 0.5 (need at most 3). **The channel is open.**

## gapped critic's actor, stalling world: fallback clip to recorded range

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | +6.6 ± 0.4 (+7.2) | +23.2 ± 1.0 (+106.1) | +16.3 ± 0.3 (+19.9) | +23.0 ± 0.3 (+28.2) | -7.2 | 1.00 | 0.70 | yes | yes |
| one-step oracle (re-referenced) | +6.5 ± 0.5 (+7.1) | +26.9 ± 1.5 (+106.1) | +16.5 ± 0.3 (+19.7) | +23.0 ± 0.2 (+28.3) | -7.1 | 0.93 | 0.67 | yes | yes |
| behaviour density | +0.4 ± 0.1 (+0.4) | +1.9 ± 0.9 (+3.6) | +0.5 ± 0.1 (+0.5) | +10.0 ± 0.3 (+12.3) | -0.4 | 0.02 | 0.01 | no | no |
| classifier ratio | +6.1 ± 0.4 (+7.0) | +13.9 ± 2.1 (+95.0) | +14.9 ± 0.4 (+19.5) | +15.0 ± 0.2 (+20.9) | -7.0 | 0.92 | 0.69 | yes | yes |
| two-head disagreement | +2.3 ± 0.2 (+2.6) | +15.7 ± 0.9 (+66.5) | +4.8 ± 0.3 (+7.0) | +17.7 ± 0.2 (+22.5) | -2.6 | 0.48 | 0.39 | yes | yes |
| one-step oracle, plan's literal |delta*| (reported) | +6.1 ± 0.4 (+6.6) | +29.1 ± 1.1 (+105.0) | +15.3 ± 0.4 (+18.5) | +22.3 ± 0.4 (+27.8) | -6.6 | 0.86 | 0.63 | yes | yes |
| one-step oracle re-referenced to the cloned actor (reported) | +6.6 ± 0.5 (+7.1) | +27.3 ± 1.6 (+106.0) | +16.3 ± 0.4 (+19.7) | +22.8 ± 0.2 (+27.7) | -7.1 | 0.93 | 0.66 | yes | yes |
| classifier ratio at the refusal limit l* (reported) | +6.3 ± 0.5 (+6.6) | +12.3 ± 3.8 (+57.5) | +11.9 ± 2.1 (+14.1) | +13.5 ± 0.7 (+16.6) | -6.6 | 0.51 | 0.35 | no | yes |
| distance check (carried over, reported) | +2.1 ± 0.1 (+2.6) | +20.2 ± 1.4 (+57.1) | +3.5 ± 0.1 (+4.8) | +18.7 ± 0.3 (+24.6) | -2.6 | 0.35 | 0.28 | yes | yes |

## gapped critic's actor, stalling world: fallback cloned actor

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | +0.0 ± 0.6 (-1.0) | +14.7 ± 0.5 (+73.8) | +7.7 ± 0.9 (+4.1) | -25.0 ± 1.8 (-30.4) | +1.0 | 1.00 | 0.79 | no | yes |
| one-step oracle (re-referenced) | +0.3 ± 0.5 (-0.7) | +19.0 ± 0.8 (+74.9) | +8.7 ± 0.7 (+5.3) | -22.2 ± 1.8 (-27.8) | +0.7 | 0.96 | 0.74 | yes | yes |
| behaviour density | -1.8 ± 0.2 (-1.8) | -1.5 ± 0.7 (-0.0) | -1.9 ± 0.2 (-2.0) | -38.5 ± 2.4 (-41.3) | +1.8 | 0.02 | 0.02 | no | no |
| classifier ratio | -4.9 ± 0.7 (-7.2) | +4.5 ± 1.6 (+62.2) | +4.2 ± 1.3 (-0.9) | -17.6 ± 0.5 (-24.1) | +7.2 | 0.93 | 0.78 | no | no |
| two-head disagreement | -2.6 ± 0.2 (-3.0) | +12.2 ± 1.3 (+45.6) | -0.3 ± 0.6 (-2.5) | -25.6 ± 1.7 (-30.7) | +3.0 | 0.49 | 0.41 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | +0.6 ± 0.3 (-0.2) | +21.8 ± 0.6 (+75.6) | +8.5 ± 0.6 (+5.3) | -28.4 ± 2.4 (-34.5) | +0.2 | 0.91 | 0.70 | yes | yes |
| one-step oracle re-referenced to the cloned actor (reported) | +0.1 ± 0.4 (-0.8) | +17.9 ± 0.7 (+74.8) | +8.4 ± 0.7 (+4.8) | -12.9 ± 2.1 (-17.8) | +0.8 | 0.96 | 0.75 | yes | yes |
| classifier ratio at the refusal limit l* (reported) | +1.2 ± 0.4 (+0.4) | +4.6 ± 2.7 (+36.4) | +7.3 ± 1.5 (+5.5) | -8.4 ± 0.9 (-11.2) | -0.4 | 0.55 | 0.41 | no | no |

## plain TD3+BC, stalling world: fallback clip to recorded range

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | +6.8 ± 0.3 (+7.2) | +26.1 ± 0.8 (+107.4) | +16.3 ± 0.5 (+20.2) | +25.8 ± 0.5 (+31.7) | -7.2 | 1.00 | 0.68 | yes | yes |
| one-step oracle (re-referenced) | +6.7 ± 0.4 (+7.1) | +30.2 ± 0.7 (+107.4) | +16.4 ± 0.7 (+20.1) | +25.7 ± 0.6 (+31.7) | -7.1 | 0.92 | 0.63 | yes | yes |
| behaviour density | +1.1 ± 0.0 (+1.2) | +5.0 ± 1.8 (+14.5) | +1.2 ± 0.2 (+1.6) | +15.4 ± 0.3 (+19.3) | -1.2 | 0.08 | 0.06 | yes | no |
| classifier ratio | +6.3 ± 0.3 (+7.0) | +16.0 ± 3.1 (+98.7) | +15.2 ± 0.5 (+19.8) | +17.7 ± 1.0 (+24.0) | -7.0 | 0.94 | 0.69 | yes | yes |
| two-head disagreement | +2.5 ± 0.1 (+2.8) | +17.4 ± 1.0 (+66.3) | +4.8 ± 0.3 (+7.2) | +19.9 ± 0.4 (+25.3) | -2.8 | 0.46 | 0.37 | yes | yes |
| one-step oracle, plan's literal |delta*| (reported) | +6.3 ± 0.3 (+6.7) | +32.9 ± 1.2 (+106.6) | +15.8 ± 0.6 (+19.0) | +25.3 ± 0.6 (+31.3) | -6.7 | 0.85 | 0.60 | yes | yes |
| one-step oracle re-referenced to the cloned actor (reported) | +6.6 ± 0.4 (+7.1) | +30.6 ± 0.8 (+107.3) | +16.2 ± 0.7 (+20.0) | +25.3 ± 0.5 (+30.8) | -7.1 | 0.92 | 0.63 | yes | yes |
| classifier ratio at the refusal limit l* (reported) | +6.2 ± 0.4 (+6.5) | +15.2 ± 5.7 (+57.1) | +11.6 ± 1.8 (+13.4) | +15.3 ± 1.5 (+18.7) | -6.5 | 0.47 | 0.31 | yes | yes |
| distance check (carried over, reported) | +2.6 ± 0.1 (+3.1) | +20.6 ± 0.8 (+59.7) | +4.2 ± 0.2 (+5.5) | +21.6 ± 0.5 (+28.1) | -3.1 | 0.37 | 0.29 | yes | yes |

## plain TD3+BC, stalling world: fallback cloned actor

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | -0.5 ± 0.3 (-1.2) | +16.0 ± 1.4 (+77.2) | +7.6 ± 1.2 (+4.9) | -24.5 ± 1.7 (-29.6) | +1.2 | 1.00 | 0.78 | yes | yes |
| one-step oracle (re-referenced) | -0.3 ± 0.3 (-1.0) | +21.2 ± 1.2 (+78.3) | +7.6 ± 1.2 (+5.5) | -22.1 ± 1.4 (-26.9) | +1.0 | 0.96 | 0.73 | yes | yes |
| behaviour density | -3.4 ± 0.3 (-3.5) | -2.4 ± 1.9 (+5.6) | -3.9 ± 0.2 (-4.2) | -38.1 ± 2.6 (-41.1) | +3.5 | 0.12 | 0.09 | no | no |
| classifier ratio | -5.5 ± 0.3 (-7.1) | +5.6 ± 2.4 (+68.0) | +4.1 ± 1.3 (+0.5) | -19.6 ± 1.1 (-24.8) | +7.1 | 0.95 | 0.80 | no | no |
| two-head disagreement | -2.4 ± 0.2 (-2.6) | +13.7 ± 1.8 (+47.1) | -1.0 ± 0.9 (-2.2) | -22.3 ± 1.8 (-27.0) | +2.6 | 0.48 | 0.39 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | +0.5 ± 0.2 (+0.0) | +24.1 ± 1.5 (+79.0) | +8.7 ± 1.0 (+6.5) | -25.5 ± 1.6 (-30.8) | -0.0 | 0.90 | 0.68 | yes | yes |
| one-step oracle re-referenced to the cloned actor (reported) | -0.5 ± 0.3 (-1.1) | +20.5 ± 1.1 (+78.4) | +7.1 ± 1.3 (+4.5) | -13.5 ± 1.9 (-18.2) | +1.1 | 0.95 | 0.73 | yes | yes |
| classifier ratio at the refusal limit l* (reported) | +0.8 ± 0.3 (+0.4) | +8.0 ± 3.9 (+41.0) | +6.0 ± 1.3 (+4.8) | -9.5 ± 0.9 (-12.0) | -0.4 | 0.54 | 0.39 | no | no |

## gapped critic's actor of the no-stall world, no-stall world: fallback clip to recorded range

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | -1.8 ± 0.1 (-1.9) | -26.5 ± 0.7 (-61.6) | -7.4 ± 0.4 (-8.0) | -18.7 ± 0.2 (-21.0) | +1.9 | 1.00 | 0.72 | no | no |
| one-step oracle (re-referenced) | -0.3 ± 0.0 (-0.3) | -1.5 ± 0.6 (-2.4) | -0.5 ± 0.1 (-0.6) | -4.7 ± 0.1 (-5.5) | +0.3 | 0.05 | 0.04 | no | no |
| behaviour density | -0.3 ± 0.0 (-0.3) | -0.5 ± 0.1 (-0.6) | -0.3 ± 0.1 (-0.3) | -9.2 ± 0.3 (-10.1) | +0.3 | 0.01 | 0.00 | no | no |
| classifier ratio | -1.8 ± 0.1 (-1.9) | -9.8 ± 1.8 (-46.3) | -6.6 ± 0.4 (-7.3) | -7.4 ± 0.7 (-10.0) | +1.9 | 0.94 | 0.74 | no | no |
| two-head disagreement | -0.7 ± 0.0 (-0.7) | -20.5 ± 1.4 (-32.0) | -1.7 ± 0.1 (-2.0) | -15.5 ± 0.2 (-17.4) | +0.7 | 0.46 | 0.34 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | -0.6 ± 0.0 (-0.7) | -29.9 ± 1.7 (-35.7) | -1.2 ± 0.1 (-1.3) | -17.7 ± 0.2 (-19.7) | +0.7 | 0.30 | 0.20 | no | no |
| one-step oracle re-referenced to the cloned actor (reported) | -0.2 ± 0.0 (-0.2) | -3.0 ± 1.3 (-4.7) | -0.5 ± 0.1 (-0.6) | -3.3 ± 0.2 (-4.0) | +0.2 | 0.08 | 0.07 | no | no |
| classifier ratio at the refusal limit l* (reported) | -1.0 ± 0.2 (-1.0) | -8.1 ± 2.3 (-17.2) | -1.8 ± 0.5 (-1.9) | -4.7 ± 0.7 (-5.6) | +1.0 | 0.37 | 0.25 | no | no |
| distance check (carried over, reported) | -1.2 ± 0.0 (-1.3) | -21.4 ± 1.1 (-25.7) | -2.6 ± 0.1 (-2.7) | -18.3 ± 0.2 (-20.8) | +1.3 | 0.23 | 0.16 | no | no |

## gapped critic's actor of the no-stall world, no-stall world: fallback cloned actor

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | -11.3 ± 0.4 (-12.4) | -30.4 ± 3.7 (-109.5) | -25.0 ± 2.3 (-31.5) | -53.9 ± 0.7 (-62.3) | +12.4 | 1.00 | 0.78 | no | no |
| one-step oracle (re-referenced) | -2.0 ± 0.1 (-2.2) | -3.6 ± 1.0 (-5.9) | -2.8 ± 0.1 (-3.7) | -13.2 ± 0.4 (-15.6) | +2.2 | 0.06 | 0.05 | no | no |
| behaviour density | -2.0 ± 0.2 (-2.1) | -2.0 ± 0.2 (-2.2) | -2.4 ± 0.3 (-2.5) | -39.4 ± 0.7 (-42.6) | +2.1 | 0.01 | 0.01 | no | no |
| classifier ratio | -16.8 ± 0.5 (-19.0) | -13.9 ± 2.5 (-96.0) | -26.7 ± 1.7 (-34.9) | -30.3 ± 1.9 (-39.9) | +19.0 | 0.93 | 0.80 | no | no |
| two-head disagreement | -4.3 ± 0.2 (-4.7) | -28.0 ± 3.7 (-52.4) | -6.9 ± 0.5 (-9.4) | -44.0 ± 0.8 (-50.5) | +4.7 | 0.48 | 0.35 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | -3.7 ± 0.2 (-4.1) | -34.2 ± 3.0 (-46.1) | -4.4 ± 0.3 (-5.4) | -51.8 ± 0.6 (-58.4) | +4.1 | 0.32 | 0.21 | no | no |
| one-step oracle re-referenced to the cloned actor (reported) | -2.0 ± 0.1 (-2.2) | -6.9 ± 3.5 (-11.5) | -3.0 ± 0.2 (-3.9) | -7.2 ± 0.3 (-9.2) | +2.2 | 0.12 | 0.09 | no | no |
| classifier ratio at the refusal limit l* (reported) | -5.9 ± 1.3 (-6.3) | -14.6 ± 3.8 (-31.0) | -6.6 ± 1.9 (-7.7) | -16.2 ± 2.7 (-19.7) | +6.3 | 0.36 | 0.25 | no | no |

## plain TD3+BC (stalling-world actor), no-stall world: fallback clip to recorded range

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | -2.0 ± 0.1 (-2.1) | -27.3 ± 0.7 (-58.4) | -7.2 ± 0.7 (-7.8) | -19.4 ± 0.2 (-21.8) | +2.1 | 1.00 | 0.68 | no | no |
| one-step oracle (re-referenced) | -0.3 ± 0.0 (-0.3) | -1.0 ± 0.5 (-2.3) | -0.6 ± 0.1 (-0.7) | -4.7 ± 0.1 (-5.7) | +0.3 | 0.06 | 0.05 | no | no |
| behaviour density | -0.5 ± 0.0 (-0.5) | -1.5 ± 0.2 (-1.8) | -0.6 ± 0.1 (-0.6) | -12.7 ± 0.3 (-14.1) | +0.5 | 0.02 | 0.01 | no | no |
| classifier ratio | -2.0 ± 0.1 (-2.1) | -13.4 ± 1.5 (-43.9) | -6.4 ± 0.7 (-7.2) | -8.0 ± 0.4 (-10.6) | +2.1 | 0.88 | 0.67 | no | no |
| two-head disagreement | -0.8 ± 0.0 (-0.8) | -18.6 ± 0.8 (-29.2) | -1.8 ± 0.2 (-2.2) | -16.0 ± 0.2 (-17.9) | +0.8 | 0.44 | 0.32 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | -0.7 ± 0.0 (-0.7) | -28.2 ± 1.4 (-34.8) | -1.3 ± 0.1 (-1.5) | -18.2 ± 0.1 (-20.2) | +0.7 | 0.34 | 0.21 | no | no |
| one-step oracle re-referenced to the cloned actor (reported) | -0.3 ± 0.0 (-0.3) | -2.6 ± 1.3 (-4.8) | -0.6 ± 0.1 (-0.8) | -3.7 ± 0.1 (-4.6) | +0.3 | 0.09 | 0.08 | no | no |
| classifier ratio at the refusal limit l* (reported) | -1.3 ± 0.2 (-1.3) | -9.4 ± 3.0 (-19.2) | -3.2 ± 1.0 (-3.4) | -5.1 ± 0.8 (-6.2) | +1.3 | 0.41 | 0.26 | no | no |
| distance check (carried over, reported) | -1.4 ± 0.0 (-1.4) | -20.3 ± 0.7 (-25.0) | -2.9 ± 0.1 (-3.1) | -19.0 ± 0.1 (-21.5) | +1.4 | 0.25 | 0.16 | no | no |

## plain TD3+BC (stalling-world actor), no-stall world: fallback cloned actor

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | -10.8 ± 0.4 (-11.7) | -35.5 ± 4.5 (-100.5) | -23.9 ± 1.2 (-28.4) | -56.4 ± 1.2 (-64.2) | +11.7 | 1.00 | 0.74 | no | no |
| one-step oracle (re-referenced) | -2.2 ± 0.1 (-2.4) | -3.3 ± 1.3 (-5.3) | -3.1 ± 0.1 (-3.8) | -12.7 ± 0.4 (-15.0) | +2.4 | 0.07 | 0.05 | no | no |
| behaviour density | -4.1 ± 0.3 (-4.3) | -5.3 ± 0.6 (-6.1) | -5.1 ± 0.5 (-5.4) | -47.9 ± 1.3 (-52.7) | +4.3 | 0.03 | 0.02 | no | no |
| classifier ratio | -17.2 ± 0.7 (-19.1) | -18.9 ± 3.9 (-86.6) | -29.4 ± 2.1 (-36.5) | -33.6 ± 1.4 (-43.0) | +19.1 | 0.91 | 0.76 | no | no |
| two-head disagreement | -4.1 ± 0.2 (-4.5) | -26.4 ± 4.7 (-46.1) | -7.1 ± 0.6 (-9.0) | -43.4 ± 0.9 (-49.3) | +4.5 | 0.48 | 0.34 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | -3.5 ± 0.2 (-3.8) | -36.2 ± 4.4 (-47.1) | -4.2 ± 0.3 (-5.1) | -51.8 ± 0.8 (-58.0) | +3.8 | 0.35 | 0.22 | no | no |
| one-step oracle re-referenced to the cloned actor (reported) | -2.3 ± 0.2 (-2.5) | -5.8 ± 2.5 (-9.8) | -3.6 ± 0.1 (-4.6) | -7.7 ± 0.2 (-9.6) | +2.5 | 0.13 | 0.09 | no | no |
| classifier ratio at the refusal limit l* (reported) | -7.0 ± 1.5 (-7.5) | -17.5 ± 4.8 (-35.2) | -10.4 ± 2.5 (-11.6) | -19.0 ± 2.8 (-22.4) | +7.5 | 0.42 | 0.28 | no | no |

## gapped critic's actor (stalling-world actor), no-stall world: fallback clip to recorded range

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | -1.6 ± 0.1 (-1.7) | -26.1 ± 1.1 (-58.4) | -6.8 ± 0.2 (-7.2) | -18.6 ± 0.2 (-20.7) | +1.7 | 1.00 | 0.71 | no | no |
| one-step oracle (re-referenced) | -0.2 ± 0.0 (-0.2) | -0.4 ± 0.3 (-1.3) | -0.5 ± 0.1 (-0.5) | -4.4 ± 0.3 (-5.3) | +0.2 | 0.05 | 0.04 | no | no |
| behaviour density | -0.4 ± 0.0 (-0.4) | -1.1 ± 0.2 (-1.3) | -0.5 ± 0.0 (-0.5) | -11.5 ± 0.3 (-12.6) | +0.4 | 0.02 | 0.01 | no | no |
| classifier ratio | -1.6 ± 0.1 (-1.7) | -10.8 ± 2.3 (-49.8) | -6.1 ± 0.1 (-6.7) | -7.7 ± 0.7 (-10.0) | +1.7 | 0.97 | 0.79 | no | no |
| two-head disagreement | -0.7 ± 0.0 (-0.7) | -18.8 ± 1.6 (-29.0) | -1.6 ± 0.1 (-1.9) | -15.4 ± 0.2 (-17.1) | +0.7 | 0.44 | 0.32 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | -0.6 ± 0.0 (-0.6) | -26.1 ± 1.6 (-30.9) | -1.2 ± 0.1 (-1.3) | -17.6 ± 0.1 (-19.3) | +0.6 | 0.29 | 0.18 | no | no |
| one-step oracle re-referenced to the cloned actor (reported) | -0.2 ± 0.0 (-0.2) | -1.2 ± 0.7 (-2.5) | -0.4 ± 0.1 (-0.5) | -3.5 ± 0.1 (-4.1) | +0.2 | 0.07 | 0.06 | no | no |
| classifier ratio at the refusal limit l* (reported) | -1.0 ± 0.2 (-1.0) | -13.8 ± 4.3 (-27.1) | -2.3 ± 0.7 (-2.5) | -4.9 ± 1.2 (-5.8) | +1.0 | 0.53 | 0.36 | no | no |
| distance check (carried over, reported) | -1.2 ± 0.0 (-1.2) | -20.6 ± 1.0 (-24.8) | -2.6 ± 0.1 (-2.7) | -18.0 ± 0.1 (-20.6) | +1.2 | 0.24 | 0.16 | no | no |

## gapped critic's actor (stalling-world actor), no-stall world: fallback cloned actor

Each cell: the trigger minus random switching at its own rate (points); in brackets the change from the actor alone. Then: cost in recorded conditions (actor alone minus trigger), share of past-80% proposals caught in strong current, share of steps switched in strong current, and the two bars: 15/3 is the channel-open bar (at least 15 over random in strong current or big kicks, cost at most 3; plan 4.3), Stage 2 is plan 5.3's deployment-check bar (at least 10 over random in strong current, cost at most 2).

| Trigger | Recorded conditions | Strong current | Big waves | Big kicks | Recorded cost | Caught (strong) | Switched (strong) | Clears 15/3 | Clears Stage 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect detector of past-80% proposals | -10.7 ± 0.6 (-11.4) | -34.1 ± 3.5 (-104.7) | -23.2 ± 1.8 (-28.0) | -54.6 ± 0.7 (-62.1) | +11.4 | 1.00 | 0.76 | no | no |
| one-step oracle (re-referenced) | -2.1 ± 0.1 (-2.2) | -2.1 ± 0.3 (-3.6) | -2.8 ± 0.1 (-3.4) | -12.9 ± 0.5 (-14.9) | +2.2 | 0.05 | 0.04 | no | no |
| behaviour density | -3.4 ± 0.1 (-3.5) | -4.4 ± 0.5 (-4.9) | -4.3 ± 0.3 (-4.6) | -45.2 ± 0.9 (-49.1) | +3.5 | 0.02 | 0.02 | no | no |
| classifier ratio | -17.4 ± 0.9 (-19.2) | -16.0 ± 2.7 (-102.6) | -28.5 ± 2.6 (-36.5) | -32.6 ± 2.1 (-41.7) | +19.2 | 0.97 | 0.85 | no | no |
| two-head disagreement | -4.1 ± 0.1 (-4.4) | -26.3 ± 3.8 (-45.3) | -6.7 ± 0.5 (-9.0) | -44.5 ± 0.8 (-50.2) | +4.4 | 0.46 | 0.33 | no | no |
| one-step oracle, plan's literal |delta*| (reported) | -3.6 ± 0.2 (-3.8) | -32.7 ± 3.2 (-41.4) | -4.3 ± 0.3 (-5.3) | -52.1 ± 0.8 (-58.2) | +3.8 | 0.31 | 0.20 | no | no |
| one-step oracle re-referenced to the cloned actor (reported) | -2.0 ± 0.1 (-2.3) | -4.2 ± 1.5 (-6.9) | -3.1 ± 0.2 (-4.1) | -7.3 ± 0.3 (-9.2) | +2.3 | 0.09 | 0.07 | no | no |
| classifier ratio at the refusal limit l* (reported) | -6.6 ± 1.5 (-7.0) | -22.3 ± 6.2 (-51.3) | -8.9 ± 2.6 (-10.2) | -18.2 ± 4.0 (-21.7) | +7.0 | 0.57 | 0.40 | no | no |

## Selectivity between the worlds

Stalling-world gain minus no-stall loss: (trigger minus actor alone) summed over the two worlds, strong current, clip fallback. 'Targeting' sums trigger minus random. 'Clip every step' is the same sum for the plain range limit. Catch-rate selectivity: share of past-80% proposals caught, stalling minus no-stall.

| Pair | Trigger | Gain minus loss | Targeting | Clip every step | Catch-rate selectivity |
|---|---|---|---|---|---|
| gapped critic's actor, each world's own | perfect detector of past-80% proposals | +44.5 | -3.3 | +44.2 | +0.00 |
| gapped critic's actor, each world's own | one-step oracle (re-referenced) | +103.7 | +25.4 | +44.2 | +0.89 |
| gapped critic's actor, each world's own | behaviour density | +3.0 | +1.4 | +44.2 | +0.01 |
| gapped critic's actor, each world's own | classifier ratio | +48.7 | +4.1 | +44.2 | -0.02 |
| gapped critic's actor, each world's own | two-head disagreement | +34.5 | -4.8 | +44.2 | +0.03 |
| gapped critic's actor, each world's own | one-step oracle, plan's literal |delta*| (reported) | +69.4 | -0.8 | +44.2 | +0.56 |
| gapped critic's actor, each world's own | one-step oracle re-referenced to the cloned actor (reported) | +101.3 | +24.3 | +44.2 | +0.84 |
| gapped critic's actor, each world's own | classifier ratio at the refusal limit l* (reported) | +40.3 | +4.2 | +44.2 | +0.14 |
| gapped critic's actor, each world's own | distance check (carried over, reported) | +31.4 | -1.2 | +44.2 | +0.12 |
| gapped critic's actor, the same actor in both worlds | perfect detector of past-80% proposals | +47.7 | -2.9 | +47.3 | +0.00 |
| gapped critic's actor, the same actor in both worlds | one-step oracle (re-referenced) | +104.8 | +26.4 | +47.3 | +0.89 |
| gapped critic's actor, the same actor in both worlds | behaviour density | +2.3 | +0.8 | +47.3 | -0.00 |
| gapped critic's actor, the same actor in both worlds | classifier ratio | +45.2 | +3.1 | +47.3 | -0.05 |
| gapped critic's actor, the same actor in both worlds | two-head disagreement | +37.5 | -3.1 | +47.3 | +0.04 |
| gapped critic's actor, the same actor in both worlds | one-step oracle, plan's literal |delta*| (reported) | +74.1 | +2.9 | +47.3 | +0.57 |
| gapped critic's actor, the same actor in both worlds | one-step oracle re-referenced to the cloned actor (reported) | +103.5 | +26.0 | +47.3 | +0.85 |
| gapped critic's actor, the same actor in both worlds | classifier ratio at the refusal limit l* (reported) | +30.5 | -1.5 | +47.3 | -0.02 |
| gapped critic's actor, the same actor in both worlds | distance check (carried over, reported) | +32.3 | -0.4 | +47.3 | +0.11 |
| plain TD3+BC, the same actor in both worlds | perfect detector of past-80% proposals | +48.9 | -1.2 | +48.3 | +0.00 |
| plain TD3+BC, the same actor in both worlds | one-step oracle (re-referenced) | +105.0 | +29.2 | +48.3 | +0.86 |
| plain TD3+BC, the same actor in both worlds | behaviour density | +12.7 | +3.6 | +48.3 | +0.06 |
| plain TD3+BC, the same actor in both worlds | classifier ratio | +54.8 | +2.6 | +48.3 | +0.06 |
| plain TD3+BC, the same actor in both worlds | two-head disagreement | +37.2 | -1.2 | +48.3 | +0.02 |
| plain TD3+BC, the same actor in both worlds | one-step oracle, plan's literal |delta*| (reported) | +71.7 | +4.7 | +48.3 | +0.51 |
| plain TD3+BC, the same actor in both worlds | one-step oracle re-referenced to the cloned actor (reported) | +102.5 | +28.0 | +48.3 | +0.82 |
| plain TD3+BC, the same actor in both worlds | classifier ratio at the refusal limit l* (reported) | +37.9 | +5.8 | +48.3 | +0.06 |
| plain TD3+BC, the same actor in both worlds | distance check (carried over, reported) | +34.7 | +0.3 | +48.3 | +0.12 |

## Does a trigger respond to the command, or only to the state? (reported, gates nothing)

At 2,000 full-throttle-region training states of the deployment world (the room's check-(e) states), the actor's command with thruster 1 pushed to each throttle. 'Falls' is the share of states whose score drops from 80% to 81% (the room allows at most 0.25 for a penalty; triggers are not held to it). The other columns are firing rates at each throttle and at the actor's own command. A rate that rises past 80% means the trigger reads the command; a flat rate means its catches come from the state.

| Actor and world | Trigger | Falls 80->81 | Fires at 70% | Fires at 80% | Fires at 81% | Fires at 85% | Fires at 90% | Fires at 100% | Fires at own command |
|---|---|---|---|---|---|---|---|---|---|
| gapped critic's actor, stalling world | perfect detector of past-80% proposals | 0.00 | 0.05 | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 | 0.42 |
| gapped critic's actor, stalling world | one-step oracle (re-referenced) | 0.06 | 0.07 | 0.07 | 0.11 | 0.37 | 0.59 | 0.74 | 0.37 |
| gapped critic's actor, stalling world | behaviour density | 0.00 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 | 0.01 |
| gapped critic's actor, stalling world | classifier ratio | 0.01 | 0.23 | 0.76 | 0.80 | 0.90 | 0.94 | 0.97 | 0.56 |
| gapped critic's actor, stalling world | two-head disagreement | 0.42 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 |
| gapped critic's actor, stalling world | one-step oracle, plan's literal |delta*| (reported) | 0.82 | 0.04 | 0.05 | 0.04 | 0.06 | 0.15 | 0.40 | 0.25 |
| gapped critic's actor, stalling world | one-step oracle re-referenced to the cloned actor (reported) | 0.06 | 0.06 | 0.06 | 0.09 | 0.34 | 0.56 | 0.73 | 0.36 |
| gapped critic's actor, stalling world | classifier ratio at the refusal limit l* (reported) | 0.01 | 0.03 | 0.10 | 0.12 | 0.27 | 0.56 | 0.86 | 0.33 |
| gapped critic's actor, stalling world | distance check (carried over, reported) | 0.01 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 | 0.07 |
| plain TD3+BC, stalling world | perfect detector of past-80% proposals | 0.00 | 0.06 | 0.06 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 |
| plain TD3+BC, stalling world | one-step oracle (re-referenced) | 0.06 | 0.08 | 0.08 | 0.11 | 0.37 | 0.59 | 0.75 | 0.34 |
| plain TD3+BC, stalling world | behaviour density | 0.00 | 0.01 | 0.02 | 0.02 | 0.02 | 0.03 | 0.04 | 0.01 |
| plain TD3+BC, stalling world | classifier ratio | 0.02 | 0.25 | 0.78 | 0.81 | 0.90 | 0.94 | 0.97 | 0.55 |
| plain TD3+BC, stalling world | two-head disagreement | 0.42 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 |
| plain TD3+BC, stalling world | one-step oracle, plan's literal |delta*| (reported) | 0.81 | 0.05 | 0.06 | 0.05 | 0.06 | 0.16 | 0.40 | 0.24 |
| plain TD3+BC, stalling world | one-step oracle re-referenced to the cloned actor (reported) | 0.06 | 0.07 | 0.07 | 0.10 | 0.34 | 0.55 | 0.73 | 0.34 |
| plain TD3+BC, stalling world | classifier ratio at the refusal limit l* (reported) | 0.02 | 0.03 | 0.09 | 0.11 | 0.26 | 0.51 | 0.84 | 0.29 |
| plain TD3+BC, stalling world | distance check (carried over, reported) | 0.01 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 | 0.07 |
| gapped critic's actor of the no-stall world, no-stall world | perfect detector of past-80% proposals | 0.00 | 0.07 | 0.07 | 1.00 | 1.00 | 1.00 | 1.00 | 0.26 |
| gapped critic's actor of the no-stall world, no-stall world | one-step oracle (re-referenced) | 0.20 | 0.06 | 0.07 | 0.07 | 0.08 | 0.09 | 0.11 | 0.06 |
| gapped critic's actor of the no-stall world, no-stall world | behaviour density | 0.00 | 0.05 | 0.06 | 0.06 | 0.06 | 0.07 | 0.09 | 0.01 |
| gapped critic's actor of the no-stall world, no-stall world | classifier ratio | 0.12 | 0.27 | 0.61 | 0.64 | 0.71 | 0.75 | 0.82 | 0.46 |
| gapped critic's actor of the no-stall world, no-stall world | two-head disagreement | 0.41 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 |
| gapped critic's actor of the no-stall world, no-stall world | one-step oracle, plan's literal |delta*| (reported) | 0.50 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| gapped critic's actor of the no-stall world, no-stall world | one-step oracle re-referenced to the cloned actor (reported) | 0.20 | 0.05 | 0.06 | 0.06 | 0.07 | 0.08 | 0.10 | 0.05 |
| gapped critic's actor of the no-stall world, no-stall world | classifier ratio at the refusal limit l* (reported) | 0.12 | 0.06 | 0.10 | 0.11 | 0.19 | 0.32 | 0.51 | 0.17 |
| gapped critic's actor of the no-stall world, no-stall world | distance check (carried over, reported) | 0.01 | 0.07 | 0.08 | 0.08 | 0.08 | 0.08 | 0.09 | 0.07 |
| plain TD3+BC (stalling-world actor), no-stall world | perfect detector of past-80% proposals | 0.00 | 0.07 | 0.07 | 1.00 | 1.00 | 1.00 | 1.00 | 0.22 |
| plain TD3+BC (stalling-world actor), no-stall world | one-step oracle (re-referenced) | 0.19 | 0.06 | 0.06 | 0.07 | 0.07 | 0.08 | 0.10 | 0.05 |
| plain TD3+BC (stalling-world actor), no-stall world | behaviour density | 0.00 | 0.07 | 0.08 | 0.08 | 0.08 | 0.09 | 0.14 | 0.02 |
| plain TD3+BC (stalling-world actor), no-stall world | classifier ratio | 0.09 | 0.33 | 0.66 | 0.68 | 0.75 | 0.79 | 0.85 | 0.48 |
| plain TD3+BC (stalling-world actor), no-stall world | two-head disagreement | 0.42 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 |
| plain TD3+BC (stalling-world actor), no-stall world | one-step oracle, plan's literal |delta*| (reported) | 0.51 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.03 |
| plain TD3+BC (stalling-world actor), no-stall world | one-step oracle re-referenced to the cloned actor (reported) | 0.20 | 0.05 | 0.06 | 0.06 | 0.07 | 0.08 | 0.10 | 0.05 |
| plain TD3+BC (stalling-world actor), no-stall world | classifier ratio at the refusal limit l* (reported) | 0.09 | 0.07 | 0.13 | 0.15 | 0.26 | 0.40 | 0.56 | 0.16 |
| plain TD3+BC (stalling-world actor), no-stall world | distance check (carried over, reported) | 0.01 | 0.07 | 0.08 | 0.08 | 0.08 | 0.08 | 0.10 | 0.07 |
| gapped critic's actor (stalling-world actor), no-stall world | perfect detector of past-80% proposals | 0.00 | 0.06 | 0.06 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 |
| gapped critic's actor (stalling-world actor), no-stall world | one-step oracle (re-referenced) | 0.19 | 0.05 | 0.07 | 0.07 | 0.07 | 0.08 | 0.11 | 0.05 |
| gapped critic's actor (stalling-world actor), no-stall world | behaviour density | 0.00 | 0.06 | 0.06 | 0.07 | 0.07 | 0.08 | 0.12 | 0.02 |
| gapped critic's actor (stalling-world actor), no-stall world | classifier ratio | 0.08 | 0.32 | 0.68 | 0.71 | 0.77 | 0.81 | 0.86 | 0.49 |
| gapped critic's actor (stalling-world actor), no-stall world | two-head disagreement | 0.42 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 |
| gapped critic's actor (stalling-world actor), no-stall world | one-step oracle, plan's literal |delta*| (reported) | 0.50 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.03 |
| gapped critic's actor (stalling-world actor), no-stall world | one-step oracle re-referenced to the cloned actor (reported) | 0.19 | 0.05 | 0.06 | 0.06 | 0.07 | 0.08 | 0.10 | 0.05 |
| gapped critic's actor (stalling-world actor), no-stall world | classifier ratio at the refusal limit l* (reported) | 0.08 | 0.05 | 0.12 | 0.14 | 0.25 | 0.39 | 0.58 | 0.18 |
| gapped critic's actor (stalling-world actor), no-stall world | distance check (carried over, reported) | 0.01 | 0.07 | 0.08 | 0.08 | 0.08 | 0.08 | 0.09 | 0.07 |
