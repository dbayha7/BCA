# Results of 14 October 2026: Stage 2 policy test, deployment check and full host, seeds 0-4

`stage2_pooled_exploratory.json` is a copy of `runs/shift_explore/stage2/pooled_exploratory.json` (gitignored),
written by the approved pre-registration version 6 (`9b38c7c4…`) at 09:21 UTC. Its source sidecar is kept beside it
as `*.source.sha256`, and the copy's own hash is in `SHA256SUMS`. The tables below are pooled means over gate seeds
0-4, read from that file and from each seed's `policy/seed<K>/values.json` and `no_stall/values.json`.

**Exploratory.** Gate seeds 0-4 cannot carry a claim: the file's own status reads "nothing here supports a claim".
Only fresh seeds 5-9 can, and they run next (pre-registration section 10, step 4), now on the user's laptop under
version 7 (`../stage2_v7/`).

All five seeds ran every step in the cloud between 13 Oct 13:56 and 14 Oct 09:21 UTC. No seed stopped: the check that
BCA's program at strength 0 reproduces the no-caution actor (check (b)) passed on every seed.

## The policy marks (strong current unless stated)

BCA here means BCA's lower edge with k = 1, the primary arm. "None" is the actor trained with no caution. Each mark
compares BCA with one rival at the same strength (matched before any return was computed).

| Mark | What it asks | Result | Passed |
|---|---|---|---|
| M1 | BCA beats the matched even extra pull by at least 15 points, ahead in at least 4 of 5 seeds | +37.7, ahead in 5 of 5 | yes |
| M2 | BCA beats each of two matched random penalty surfaces by at least 15 points | +44.7 and +39.7 | yes |
| M3 | BCA keeps the improvement: in recorded conditions at least none − 3 and at least cloning + 25; in big waves and big kicks at least none − 3 | +5.5 over none and +34.7 over cloning (recorded); +13.6 (waves); +8.7 (kicks) | yes |
| M4 | BCA is at least the matched best data-only rival − 5 (the recorded-range hinge, chosen before any BCA run) | −45.6 | **no** |
| M5 | The deployment check on the no-caution actor beats random switching at its own rate by at least 10 points, and costs at most 2 points in recorded conditions | +1.5 over random switching; +7.1 in recorded conditions (no cost) | **no** |

So the marks are not all met. BCA clearly beats caution that is not aimed (M1, M2) and does not spoil what training
gained (M3). But at the same strength, the simple data-only rival, a penalty for leaving the recorded range, does far
better in strong current (M4), and the deployment check is no better than switching at random (M5).

## The claim sentence's numbers (pre-registered wording, filled with pooled numbers)

> In strong current, BCA's lower edge recovers 39.2 of 98.9 points and keeps 119% of TD3+BC's gain over cloning in
> the recorded conditions. A hand-set limit to the recorded range recovers 107.4. In the no-stall world, where the
> missing commands help, BCA costs 14.8 points and the range limit 30.9.

## Returns by arm (pooled means, points)

Stalling world:

| Arm | Recorded | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| none (no caution) | 85.5 | −14.3 | 70.0 | 52.5 |
| full-log actor (nothing removed; reference) | 91.3 | 84.6 | 86.6 | 64.3 |
| cloning | 56.3 | 54.4 | 43.6 | 2.0 |
| **BCA, k = 1** | 91.0 | 24.9 | 83.6 | 61.3 |
| BCA, k = 2 | 90.2 | 30.1 | 83.5 | 60.6 |
| BCA, unpaired | 90.5 | 23.6 | 83.5 | 61.0 |
| BCA, state-shuffled | 90.4 | 20.8 | 84.1 | 59.1 |
| perfect band (exact error; reference) | 93.4 | 79.7 | 89.7 | 69.4 |
| recorded-range hinge, hand-set | 93.3 | 93.1 | 91.6 | 74.9 |
| best data-only rival (range hinge), matched | 92.8 | 70.5 | — | — |
| even extra pull, matched | 83.3 | −12.8 | — | — |
| random surface 1 / 2, matched | 82.5 / 83.4 | −19.7 / −14.8 | — | — |
| push less everywhere, matched | 82.8 | −9.9 | — | — |

No-stall world (where full throttle helps, so caution only costs):

| Arm | Recorded | Strong current |
|---|---|---|
| none | 93.5 | 94.9 |
| BCA, k = 1 | 90.0 | 80.1 |
| perfect band | 93.1 | 93.6 |
| recorded-range hinge, hand-set | 92.4 | 64.0 |
| best data-only rival, at its locked strength | 92.6 | 73.7 |
| even extra pull, at its locked strength | 88.0 | 86.7 |

## The full host (TD3+BC trained together with BCA's lower edge, 100,000 updates)

| World | Arm | Recorded | Strong current | Big waves | Big kicks |
|---|---|---|---|---|---|
| stalling | TD3+BC with BCA | 92.8 | 28.0 | 87.6 | 60.1 |
| stalling | plain TD3+BC | 86.1 | −16.8 | 70.9 | 48.7 |
| stalling | cloning | 56.3 | 54.4 | 43.6 | 2.0 |
| no-stall | TD3+BC with BCA | 92.7 | 77.3 | 86.8 | 86.6 |
| no-stall | plain TD3+BC | 94.9 | 93.5 | 90.5 | 93.3 |

The range hinge in the full host (reported, seeds 1-4 only in the stalling world) averages 35.4 in strong current
(seeds: 3.6, 37.9, 94.7, 5.3).

## Detection (unchanged from 13 October)

The weighted version passed direction and size and the uniform version failed (see `../results-2026-10-13/`). The
detection claim rule is not met.

## What this means for seeds 5-9

Nothing changes. The pre-registration runs fresh seeds 5-9 with every setting frozen on seeds 0-4, and their pooled
look is the only one that can carry Stage 2's claim. Every step for seeds 0-4 is now finished in the cloud, so the
laptop inherits all of them and needs no reference re-made (version 7, C7).
