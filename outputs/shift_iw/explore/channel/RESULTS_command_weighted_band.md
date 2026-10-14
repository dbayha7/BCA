# Results: BCA's band weighted for the actor's own commands (no training)

EXPLORATORY: outside the version-5 pre-registration.

- **Owner's go (2026-10-14):** "Yes, build and check it (Recommended)".
- **Pass marks:** `PASSMARKS_command_weighted_band.md`, written at 02:43Z, before anything ran.
- **Scripts and outputs:**
  - The check ran at 02:46–02:47Z with `command_band_check.py`; `summarise_command_band.py` scored it.
  - A post hoc diagnostic, `no_credit_shape.py`, ran at 02:50Z. It is not a pass mark.
  - Results are in `runs/shift_iw_v5_explore/channel/command_band/`.
- **Nothing trained:** no training was done, and nothing in `runs/shift_iw_v5` changed.

## What changed from version 5

Only the weights change. A (row, command) now gets the exact ratio

w(c) × π_h(a | s) × Z_ε(s) / β_ε(a | s)

- **w(c):** the strong current law's weight.
- **π_h:** the actor's spread, BCA's own target noise of 0.2, clipped at ±0.5.
- **β_ε:** the recording controller's command density, exact for the episode's random share.
- **Z:** the share outside the removed ball.

The scale network, the unit, the bank scores, the stream and the posterior are BCA's own, unchanged. The actor is the
lower-edge room check's no-caution actor.

## Results (mean over seeds, with the min to max range)

| | random | medium-replay | medium | medium-expert | expert |
|---|---|---|---|---|---|
| Bank's effective size (Kish share) | 1.4% | 1.2% | 1.2% | 3.0% | 15.8% |
| **C1** draws from the test distribution: covered or refused | 98.8% | 99.1% | 98.2% | 98.5% | 96.7% |
| … of which refused | 76.7% | 68.3% | 60.0% | 58.1% | 49.1% |
| **C2** at the actor's command within 80%: covered or refused | 100.0% | 99.9% | 99.3% | 99.5% | 98.0% |
| **C2** at the actor's command past 80%: covered or refused | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| … past 80%, refused | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| **C3** at the actor's command within 80%: refused | 99.2% | 89.2% | 71.5% | 47.6% | 25.0% |
| … because the command is inside the removed ball | 1.3% | 2.3% | 3.6% | 8.9% | 14.6% |
| … because the test mass is past the onset of refusal | 97.9% | 86.8% | 67.9% | 38.7% | 10.4% |
| Within-80% certified width / version 5's exact-strong width | 0.72 | 1.27 | 0.85 | 0.82 | 0.80 |
| Largest finite threshold / R_u | 1.38 | 1.80 | 1.54 | 1.70 | 3.00 |
| **C4** check (e) passes, version 5's refusal rule (a) | 2 of 5 | 1 of 5 | 1 of 5 | 1 of 5 | 0 of 5 |
| Best throttle past 80%: Q1 − W with rule (a) | 75.7% | 78.0% | 69.9% | 75.5% | 36.4% |
| Best throttle past 80%: Q1 alone | 72.8% | 78.0% | 80.1% | 85.1% | 80.4% |

## Scored against the pass marks

- **C1 (the weights are right): met on all five qualities.** At draws from the test distribution, the band covers or
  refuses 94.3–100% of commands, by seed (bar: 88% as a mean).
- **C2 (honest where the lower edge reads it): met** on medium-expert and expert. At the actor's own commands it
  covers or refuses 98.0–99.5% within 80% and 100% past 80%. Every past-80% command is refused. Version 5's band
  covered 25% (expert) and 48% (medium-expert) there.
- **C3 (targeted, not blanket): missed** on both, because within 80% the band refuses 25.0% (expert) and 47.6%
  (medium-expert) of the actor's commands (bar: 10%). The width part passes: certified widths are 0.80× and 0.82×
  version 5's.
  - **Inside the removed ball (14.6% and 8.9%):** the actor is asking for the removed hold. Refusing it is the action
    gap working as designed, and the version-5 coverage test refuses those steps too.
  - **Past the onset (10.4% and 38.7%):** the test distribution (the actor's command ± 0.2) is far narrower than the
    recorded commands. The weights therefore concentrate on few bank rows (an effective size of 3–16% of the bank),
    and large test masses cannot be certified. On the three lower qualities, where the data are mostly uniform random
    commands, this refuses 68–98% of the actor's commands: a blanket refusal.
- **C4 (the penalty has the right shape): missed** on both. Check (e) passes on 0 of 5 seeds (expert) and 1 of 5
  (medium-expert).
  - Version 5's refusal rule (a) gives a refused command the table's largest finite threshold, which is only 3.0×
    (expert) and 1.7× (medium-expert) the unweighted threshold. The channel check found that 8.1× and 3.4× are needed.
  - Every command from 80% up is refused and gets that same capped threshold, so the penalty does not rise from 80% to
    81% (e1–e3 fail).
  - Q1 − W still prefers a throttle past 80% on 36% of hard-water rows on expert (80% for Q1 alone) and 76% on
    medium-expert (85%).

## Post hoc diagnostic: a different refusal rule (not a pass mark)

Rule: a refused command gets no value credit over the row's recorded command, W = [Q1(s, a) − Q1(s, a_i)]_+, so
Q1 − W = min(Q1(s, a), Q1(s, a_i)). Certified commands keep [U(s, a) − U(s, a_i)]_+. The rows, grid, actor and rule
are those of check (e).

| Best throttle past 80% (hard-water rows) | medium-expert | expert |
|---|---|---|
| Q1 alone | 85.1% | 80.4% |
| Version 5's BCA lower edge (from the exploratory arms' record) | about the same as Q1 alone | about the same as Q1 alone |
| The exact one-step oracle (the room check, +5.8 and +8.4 points of return) | 51.7% | 34.3% |
| Command-weighted band, rule (a) | 75.5% | 36.4% |
| **Command-weighted band, no credit when refused** | **16.5%** (14.4–19.6) | **1.8%** (1.2–2.5) |

- Check (e) passes on 5 of 5 seeds for both qualities with this rule.
- On the three lower qualities e2 fails on every seed, but the share still falls: 73% to 58% (random), 78% to 49%
  (medium-replay) and 80% to 41% (medium).

## What it means

- **The weighting does what the channel needs.** Weighted for the commands where the band is read, BCA becomes
  honest there (C1, C2): it refuses every command past 80%, where the critic overrates the actor's command, and
  certifies most commands within 80%.
- **The refusal has to reach the actor.** Version 5's rule turns a refusal into a modestly larger threshold, which is
  too little. With "no credit over the recorded command", the over-preference for past-80% throttles falls below even
  the oracle's on the frozen critic.
  - This is a surface check, not return. The return test is still to run.
- **The cost to watch is refusal within 80%.** With no credit, a refused command earns only the cloning term.
  - On expert, 25% of the actor's within-80% commands are refused, 15 points of it at the removed hold.
  - On medium-expert, 48% are refused. Half its episodes are random, so cloning there partly pulls toward random
    commands.
  - The training test must show whether the gain past 80% outweighs this.
- **On the three lower qualities the weighting refuses nearly everything** (the effective bank is about 1% of its
  size). There is also no room for the lower edge there, even with the oracle. They stay out of the return test.
