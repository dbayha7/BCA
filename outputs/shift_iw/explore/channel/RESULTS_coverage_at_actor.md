# Results: BCA's band where the lower edge reads it

EXPLORATORY and post hoc: outside the version-5 pre-registration. The expectations were written first, at
2026-10-14T02:32Z, in `EXPECTATIONS_coverage_at_actor.md`. This was run at 02:33Z with `coverage_at_actor.py`, and the
table comes from `summarise_coverage_at_actor.py`. No training was done, and nothing in `runs/shift_iw_v5` changed.
Results are in `runs/shift_iw_v5_explore/channel/<quality>.json`.

## Setup

- **Rows:** gapped host, seeds 0–4. Every kept training row: 42,500 rows (expert) to 49,919 (random).
- **Actor:** the lower-edge room check's no-caution actor.
- **Truth at the actor's command:** one exact float64 simulator step, with BCA's own target rule. The step reproduces
  the recorded transitions to 4e-6 in reward and 1.4e-6 in the next observation.
- **Strong-law figures:** rows reweighted by w(c).
- **"Past 80%":** the actor's command stalls at least one thruster.
- **No refusals:** every row's exact-strong threshold was finite.

## Results (mean over seeds, with the min to max range)

| | random | medium-replay | medium | medium-expert | expert |
|---|---|---|---|---|---|
| Actor commands past 80%, share of rows (strong law) | 3.2% | 8.8% | 15.3% | 39.1% | 38.6% |
| Same, share of hard-water rows | 3.5% | 9.8% | 17.9% | 52.6% | 61.8% |
| Coverage at recorded commands, training law (sanity) | 97.2% | 97.0% | 97.4% | 97.5% | 94.7% |
| Coverage at the actor's commands, strong law, unweighted | 92.9% | 92.5% | 91.2% | 76.8% | 68.3% |
| … with exact-strong weighting | 97.1% | 96.3% | 96.8% | 86.9% | 74.0% |
| … within 80%, unweighted | 93.2% | 93.1% | 93.2% | 95.0% | 95.7% |
| … past 80%, unweighted | 85.2% | 85.9% | 81.2% | 48.4% (41.8–54.0) | 24.7% (19.7–30.2) |
| … past 80%, exact-strong weighting | 92.5% | 92.0% | 90.4% | 67.8% | 34.9% |
| Critic's error y − q at the actor's past-80% commands | +1.64 | +0.21 | −0.97 | −3.37 | −1.68 |
| Band width U there (unweighted) | 9.03 | 6.94 | 4.75 | 2.33 | 0.49 |
| Band width U within 80% (unweighted) | 7.92 | 5.82 | 4.04 | 1.87 | 0.42 |
| Widening needed for 90% coverage past 80% | 1.16× | 1.13× | 1.34× | 3.42× (3.21–3.77) | 8.13× (6.41–9.27) |
| Widening the exact-strong weighting gives (mean R_w / R_u) | 1.22× | 1.19× | 1.34× | 1.75× | 1.45× |

A negative y − q means the critic rates the command higher than the one-step truth.

## Scored against the expectations

- **E1 (sanity), met.** At recorded commands the band covers 93.2–98.4% on every quality and seed (bar: at least 85%).
- **E2, partly met (the top two qualities only).**
  - The band covers below 90% at the actor's commands on medium-expert (73.2–79.4%) and expert (63.0–74.6%).
  - It does not on random (92.9%), medium-replay (92.5%) or medium (91.2%, two of five seeds below 90%).
  - At past-80% commands it covers 24.7% on expert, with every seed below 50%. On medium-expert it covers 48.4%, with
    four of five seeds below 50% and seed 2 at 54.0%.
  - On the three lower qualities it covers 81–86% there, not below 50%.
- **E3, partly met.**
  - The needed widening past 80% is above 3 on the top two qualities (3.42× and 8.13×): met.
  - The weighting's gain in coverage at the actor's commands stayed under 5 points only on random (+4.2) and
    medium-replay (+3.8). It was +5.6 on medium, +10.1 on medium-expert and +5.7 on expert, so that part was missed.
    I underestimated how much the weighting helps, but it still leaves past-80% coverage at 35% (expert) and 68%
    (medium-expert).
- **Why the lower qualities differ from what I expected:** random data make the critic's errors large everywhere.
  Its band is 10–18 times wider than on expert data, and those actors rarely push past 80% (3–15% of rows). So the band
  is already roughly honest at their commands.

## What it means

- **The band is honest where it was calibrated and not where the lower edge reads it, on the two qualities where the
  lower edge has room.**
  - Within 80%, the band covers 95–96% at the actor's commands.
  - Past 80%, on 53–62% of hard-water rows, the critic overrates the actor's command by 1.7 (expert) and 3.4
    (medium-expert). There the band widens only 15–25% over its within-80% width, and it covers 25% and 48%.
  - So the lower edge's penalty, the width at the actor's command minus the width at the recorded command, stays near
    zero, which is what the exploratory arms found.
- **Weighting by the current law multiplies every row's band by about the same factor** (1.45× and 1.75×, nearly
  constant across rows). It cannot create the jump past 80% that is missing (8× and 3.4×).
- **The room checks show that scale alone cannot move return either.** On the top two qualities, under the strong
  shift, scaling caution by 1.5 changes return by 0.0–0.1 points for the perfectly informed lower edge (oracle ×1 against ×1.5)
  and 0.4–0.7 points for the even pull (×1.5 against none). A near-uniform 1.2–1.8× rescale cannot raise return through either channel. Weighting has to
  change where the band is wide, not only how wide it is.
- **Consequence for the design:** for weighting to reach return, its weights must describe the distribution at which
  the band is read, which is the actor's commands under the test law. Both parts are known exactly in this world: the
  current law, and the recording controller's command density in closed form.
