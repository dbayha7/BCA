# Stage 1 room test, seeds 0 and 1, from the cloud run (EXPLORATORY, no BCA)

**The result files are lost.** The numbers below are copied from the session transcript. They were printed from
`runs/shift_explore/room/seed{0,1}/values.json` on the 4-CPU cloud container, which was reclaimed before the files
were committed.

The run used:
- spec v2 (`spec.json`, SHA-256 `56bbda24…`);
- code manifest `469ada26…`;
- gate artifacts from pre-registration `68d9e8779e148b7b…`.

The futility look read the two values files:
- seed 0: SHA-256 `4be846672ce1c80c168c4b5491e05d72097e9332a680dbce83a56174ac370096`;
- seed 1: SHA-256 `cf20326aac145f2baabc5a44323fc3e24d4054e82630b87a5da6df1225f4f9e8`.

**Futility look:** "continue: run seeds 2-4 and the no-stall arms". In strong current the perfect band beat the
matched even pull by 91.7 points on seed 0 and by 99.8 on seed 1. The threshold is 5.

The runs for seeds 2–4, the no-stall world and the replication had started but had not finished when the container
was lost. Stage 1 is being rerun on the local machine, on a fresh local run of the gate. The numbers here are kept as
a record of what the cloud run showed.

**How to read the tables**
- Points: 0 is the recording controller, 100 the stall-aware controller.
- "Past 80%" is the share of strong-current steps that push any thruster past 80% throttle.
- A control written `@x` is that arm at knob value x, matched blind to the perfect band's strength.

## Seed 0

| Arm | Big starting pushes | Big waves | Recorded conditions | Strong current | Past 80% (strong current) |
|---|---|---|---|---|---|
| clone | −9.2 | 39.9 | 51.5 | 51.5 | 0.08 |
| even_pull@0.8157 | 44.6 | 60.7 | 81.0 | −14.2 | 0.68 |
| long_run_harm | 71.7 | 89.9 | 93.0 | 86.5 | 0.50 |
| none | 52.1 | 68.1 | 84.2 | −13.0 | 0.76 |
| overrating_half_literal | 70.0 | 88.7 | 92.9 | 81.0 | 0.60 |
| **perfect_band** | 69.7 | 88.7 | 92.7 | **77.6** | 0.60 |
| perfect_band_centred | 68.6 | 87.7 | 92.5 | 74.4 | 0.59 |
| perfect_band_half | 70.7 | 88.6 | 92.7 | 81.1 | 0.60 |
| perfect_band_literal | 68.5 | 84.7 | 91.5 | 76.3 | 0.78 |
| push_less@0.5099 | 54.3 | 69.0 | 82.3 | −8.2 | 0.64 |
| random_1@0.6696 | 49.2 | 68.4 | 81.7 | −14.5 | 0.75 |
| random_2@0.5176 | 50.3 | 63.7 | 81.6 | −16.9 | 0.78 |
| range_hinge | 74.9 | 91.7 | 93.1 | 93.6 | 0.34 |
| state_shuffled_band | 55.8 | 73.5 | 88.4 | 5.7 | 0.61 |
| underrating_half | 51.1 | 63.5 | 83.6 | −16.7 | 0.80 |

## Seed 1

| Arm | Big starting pushes | Big waves | Recorded conditions | Strong current | Past 80% (strong current) |
|---|---|---|---|---|---|
| clone | 4.1 | 40.0 | 57.7 | 61.8 | 0.07 |
| even_pull@0.9094 | 46.4 | 69.1 | 84.4 | −17.4 | 0.67 |
| long_run_harm | 70.9 | 90.3 | 94.2 | 84.7 | 0.47 |
| none | 51.5 | 70.3 | 86.1 | −21.5 | 0.75 |
| overrating_half_literal | 68.2 | 89.1 | 93.7 | 71.9 | 0.58 |
| **perfect_band** | 69.8 | 91.5 | 94.4 | **82.4** | 0.54 |
| perfect_band_centred | 68.1 | 89.9 | 93.4 | 73.9 | 0.56 |
| perfect_band_half | 67.6 | 88.9 | 93.7 | 75.7 | 0.56 |
| perfect_band_literal | 68.5 | 87.9 | 92.1 | 82.4 | 0.62 |
| push_less@0.6471 | 53.8 | 70.6 | 82.6 | −17.7 | 0.65 |
| random_1@0.6988 | 51.0 | 72.9 | 83.5 | −20.6 | 0.70 |
| random_2@0.7130 | 51.7 | 68.6 | 84.5 | −15.6 | 0.71 |
| range_hinge | 75.1 | 93.0 | 94.3 | 93.0 | 0.17 |
| state_shuffled_band | 54.6 | 75.8 | 89.1 | −7.1 | 0.68 |
| underrating_half | 50.7 | 67.5 | 85.2 | −19.5 | 0.78 |

## What it showed

- **The targeted penalty recovered most of the harm.** The penalty placed by the exact overrating at the actor's
  command recovered about 90–100 of the 99 lost points in strong current.
- **Untargeted caution recovered almost nothing.** This held for every control matched to the same strength: the even
  pull, pushing less, and both random surfaces.
- **The same penalty taken from another state recovered little.**
- **It also helped where the data are.** It raised the score in the recorded conditions (93–94, against 84–86 with no
  caution) and in the other two shifted situations.
- **Clipping every command into the recorded range (range_hinge) did as well or better in this stalling world.** The
  no-stall runs would have shown what that clip costs, but they did not finish.
