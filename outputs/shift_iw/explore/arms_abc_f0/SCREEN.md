# Seed-0 screen: arms A, B, C and F0 (EXPLORATORY)

**EXPLORATORY.** EXPLORATORY: arms A/B/C and result F's no-band control; seen seeds 0-4; pass marks frozen in routes2/arms/PASSMARKS_arms_ABC_and_F_control.md (SHA-256 a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf) before this code was written. Data seed 0 is a seen seed. Nothing here is a confirmatory result.

Made 2026-10-16T19:52:43+0000. Pass marks and kill rules: `routes2/arms/PASSMARKS_arms_ABC_and_F_control.md` (SHA-256 `a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf`), checked before this report was written.

This screen computes only what the kill rules read, plus arm B's room check. **No final return of arm A, B or C is shown or computed.** F0's seed-0 score is shown, because no later rule guards it.

**Complete:** yes.

## What goes on to seeds 1-4

- **Arm A** (credible-target pull): random, medium-replay, medium, medium-expert, expert.
- **Arm B** (certified throttle ceiling): none (stopped by K1).
- **Arm C** (guarded release): random, medium-replay, medium, medium-expert, expert.
- **F0** (result F's no-band control) and result F's A2: medium-expert, expert.

## Data for random, medium-replay and medium (seeds 1-4)

| Quality | D1 identical to the committed report (seeds 0-4) | Report says D1 | Band fits allowed |
|---|---|---|---|
| random | yes | pass | yes |
| medium-replay | yes | pass | yes |
| medium | yes | pass | yes |

## Arm A: kill checks

K0 is information only on seed 0 (change 1). K1 and K3 bind on every seed. K2 is judged once all five seeds have reached 8,000 steps; seed 0's value is recorded here.

| Quality | Status | K0: recorder rows released (at most 10%) | K0: random rows released (at least 15%) | K0 would pass | K1: rows released at 4,000 (above 60% fires) | K1: mean D / mean U (above 3 fires) | K2 seed 0: A1 - none at 8,000 (points) | s | mean U / s |
|---|---|---|---|---|---|---|---|---|---|
| random | finished | - | 14.1% | not checked | 33.5% | 0.89 | 20.01 | 3.376 | 2.204 |
| medium-replay | finished | 0.1% | 24.2% | yes | 40.9% | 1.19 | 14.00 | 2.389 | 2.100 |
| medium | finished | 0.2% | 28.7% | yes | 36.0% | 1.13 | 7.34 | 1.639 | 2.199 |
| medium-expert | finished | 1.9% | 44.7% | yes | 30.8% | 1.73 | -1.80 | 0.915 | 1.877 |
| expert | finished | 17.1% | - | not checked | 32.8% | 1.19 | 0.65 | 0.265 | 1.563 |

## Arm B: kill checks

- **K0, room** ((host + knee clip) - host, strong test, medium-expert, below +1.0): 6.99 points; fires: no.
- **K1** (BCA's thruster-1 ceiling 0.85 or higher with at least 30 tries in rung (0.80, 0.85]): fires: yes ['medium-replay', 'medium', 'medium-expert'].
- **K2** (BCA's thruster-1 ceiling 0.60 or lower in 2 or more seed-quality pairs): pairs so far: none; fires: no.
- **K3** (two seeds with a non-finite gain, weight, actor or return): 0; fires: no.
- **K4** (BCA's map clips under 2% of the host's thruster-1 commands (strong starts, medium-expert)): 28.8%; fires: no.

| Quality | BCA ceilings (thrusters 1-4) | Thruster-1 tries per rung (0.50-0.55 ... 0.95-1.00) | Share of the host's thruster-1 commands the map clips |
|---|---|---|---|
| random | 0.80, 0.75, 0.65, 0.50 | 209, 137, 80, 69, 38, 24, 12, 2, 1, 0 | 0.6% |
| medium-replay | 0.90, 0.75, 0.70, 0.65 | 369, 272, 240, 204, 168, 199, 42, 25, 12, 2 | 0.2% |
| medium | 0.90, 0.75, 0.80, 0.70 | 470, 379, 386, 355, 380, 541, 54, 34, 21, 6 | 1.7% |
| medium-expert | 0.85, 0.80, 0.85, 0.65 | 883, 814, 859, 838, 833, 736, 78, 67, 41, 35 | 28.8% |
| expert | 0.80, 0.75, 0.80, 0.65 | 1120, 1089, 1095, 1096, 998, 493, 0, 0, 0, 0 | 45.8% |

## Arm C: kill checks

- **K3** (two seeds with a non-finite gain, weight, actor or return stop C (as in B)): none; fires: no.
- **KB0** (rho 0.5's share of thruster-1 commands past 80% at least 2% and at least twice rho 1's, on at least 2 of random, medium-replay, medium (seed 0); if it fails, B2-B4 are not scored): passing on ['medium-replay', 'medium']; holds: yes.
- **Guard void:** by B's K1 or K2: yes; by C's own map: ['random', 'medium-replay', 'medium', 'medium-expert'].

| Quality | Status | KB1: G(rho 1) | KB1: G(rho 0.5) | KB1 bar | KB1 fires | rho 0.5 past 80% | rho 1 past 80% | Guard ceilings (thrusters 1-4) |
|---|---|---|---|---|---|---|---|---|
| random | finished | 5.405 | 7.734 | 16.214 | no | 1.5% | 0.4% | 0.85, 0.65, 0.65, 0.50 |
| medium-replay | finished | 4.314 | 6.061 | 12.941 | no | 3.0% | 0.8% | 0.90, 0.85, 0.80, 0.70 |
| medium | finished | 3.293 | 4.298 | 9.879 | no | 8.5% | 3.2% | 0.90, 0.85, 0.90, 0.70 |
| medium-expert | finished | 2.461 | 3.065 | 7.384 | no | 39.3% | 30.8% | 0.85, 0.80, 0.85, 0.70 |
| expert | finished | 0.470 | 0.554 | 1.411 | no | 50.8% | 45.6% | 0.80, 0.75, 0.80, 0.70 |

## F0 (result F's no-band control): checks and seed-0 score

No cross-machine bar is used. A2 and F0 are compared in this sandbox only. The mark needs all five seeds, so the seed-0 difference below decides nothing.

| Quality | Status | KF1 disagreement | KF2 (a) / (b) | KF3 same clone | F0 strong points | A2 strong points | A2 - F0 | F0 - no extra caution | A2 - no extra caution |
|---|---|---|---|---|---|---|---|---|---|
| medium-expert | finished | 0.0000 | yes / yes | yes | 94.64 | 94.69 | 0.04 | 1.41 | 1.46 |
| expert | finished | 0.0000 | yes / yes | yes | 96.76 | 97.04 | 0.28 | 6.01 | 6.29 |

Seeds 1-4 start only with `bash run_arms.sh continue`, which refuses unless this screen is complete and verified.
