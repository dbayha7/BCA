# ThrusterChain harm checks: the shift causes real harm, but caution of BCA's kind cannot reach it

Run 2026-10-12 on a 4-CPU cloud container. Pre-registration `outputs/shift/preregistration.json`, SHA-256
`68d9e8779e148b7b…` (every file of `experiments/shift/` hashed into it). Base world, gate seeds 0–4. **No BCA arm, no
BCA band fit and no BCA dose readout were run.**

## Verdict

**The checks stopped where the pre-registration says they must: before any BCA run, with the stop "no room".**

> No shifted situation is room-certified: caution placed by the exact harm, inside BCA's dose range, does not beat even
> caution there, so no BCA result could be informative.

What the run shows, in order:

1. **The environment does what it was built to do.** Removing the most common command (the D4RL densest-15% rule)
   makes the critic wrong exactly at the removed commands and nowhere else. Every control behaves as designed: random
   removal and a second initialisation explain none of it, and in the no-stall comparison world, where the removed
   commands hide nothing, the same removal makes the critic only 1.18× as wrong (2.27× in the stalling world). All six
   marks of check 1 pass.
2. **Acting on that error costs a great deal of return under shift.** In strong current, the actor trained on the
   gapped critic scores 99 points lower than the same actor trained on the full-log critic. That is almost the whole
   range from the stall-aware controller (100) down to the recording controller (0). Random removal and the no-stall
   world explain none of it, and the loss sits at the steps that push past 80% throttle. All five marks of check 2
   pass at alpha 5. At alpha 2.5 one mark (how much more often the actor pushes past 80%) misses by a hair: 1.46
   against 1.5.
3. **Caution of the kind BCA applies cannot recover it.** Check 3 places extra pull toward the recorded commands on
   the training rows by the exact harm, in BCA's own form (a multiplier between 1 and 1.5 per row). Outside strong
   current it beats even caution of the same strength by under 1 point (the mark is 3). In strong current, where the
   harm is, it does slightly worse than even caution. No situation is certified, so the power check (check 4)
   correctly refused to run, and the BCA stage is closed.

The stop is informative. Check 3 uses the exact harm, which is a better placement than any signal BCA could learn.
If even that placement cannot move the score, no per-row caution in BCA's dose range can, however well it detects
the harm. Here the limit is in the mechanism (extra pull toward the recorded commands on training rows), not in the
detection signal. The detection question itself (do BCA's doses fall on the rows where the critic is wrong?) was not
asked, because the protocol opens it only after the gate passes and the user approves.

## The world in one paragraph

Five floats in a channel are pushed downstream by a current and waves. Four thrusters hold them in place. Past 80%
throttle a thruster stalls and its push falls (0.8 at 80%, 0.5 at full throttle), but it still costs energy. The
recording controller's most common command is full throttle, so the D4RL rule removes almost all of it, and the
training data then show almost nothing past 80% (0.04–0.14% of kept rows, against 12–14% before). The learned
controllers are tested in the recorded conditions and in three shifted situations: strong current and big waves
(every start outside the recorded range) and big kicks (large starting pushes; 11–18% of the steps outside it). Scores are in points: 0 is the recording controller
and 100 is the stall-aware controller, which never pushes past 80%. Exact returns come from deterministic
simulation. Full description: `experiments/shift/SHIFT_ENV.md`.

## Check 1: the critic is wrong where data are missing (passed, 6 of 6 marks)

These critics are fixed-controller critics, each trained on one log with everything else equal. The error measured is
the median absolute error of the critic's advantage, against exact truth.

| Mark | Rule | Result (seeds 0–4) |
|---|---|---|
| a | gapped-log critic over full-log critic, at the removed commands: at least 2× pooled, 1.5× in every seed | 2.15–2.40× (pooled 2.27×) |
| b | the same ratio far from the removed commands: between 1/1.5 and 1.5 | 0.68–0.80× |
| c | random removal of the same number of rows does not explain it (its ratio) | 0.96–1.21× |
| d | another initialisation of the same critic does not explain it, on every pool of rows (its ratio) | 0.87–1.09× over the three pools (0.89–1.09× at the removed commands) |
| e | the stall makes it at least 1.5× worse than in the no-stall comparison world | 1.75–2.23× (pooled 1.92×) |
| f | the gapped critic overrates a pull past 80% (against capping at 80%) by at least 25% of its true price more than the full-log critic, in at least 4 of 5 seeds | 5 of 5: overrated by +0.33 to +0.36 (gapped) against −0.06 to −0.01 (full); the true price is 0.33–0.35 |

In the full-throttle region, pushing past 80% truly costs 0.33–0.35 against capping at 80%. The gapped critic
overrates the push by almost exactly that amount, so it sees no cost in it. The full-log critic is within 0.06 of the
truth. Reported only: the D4RL protocol's version of this check gives 2.27× at the removed commands and 0.75× far
from them. The far ratio falls below that protocol's band of 1–3: away from the gap, the gapped critic is less wrong
than the full-log critic.

## Check 2: acting on the error costs return (passed at alpha 5)

Every actor starts from the same cloned actor and is trained only against a frozen critic. "Harm" is the full-log
critic's actor minus the gapped critic's actor, in points. Means over seeds 0–4, 95% intervals in brackets.

| Situation | Cloned actor | Full-log critic's actor | Gapped critic's actor | Harm | Random removal's harm | No-stall world's harm |
|---|---|---|---|---|---|---|
| Recorded conditions | 56.3 | 91.3 | 85.5 | 5.8 [4.8, 6.8] | 0.1 | 0.0 |
| **Strong current** | 54.4 | 84.6 | **−14.3** | **98.9 [96.2, 101.7]** | 3.1 | 0.4 |
| Big waves | 43.6 | 86.6 | 70.0 | 16.6 [14.5, 18.7] | 0.0 | −0.4 |
| Big kicks | 2.0 | 64.3 | 52.5 | 11.8 [10.2, 13.4] | 0.3 | 0.1 |

Share of steps that push past 80% on any thruster:

| Situation | Cloned actor | Full-log critic's actor | Gapped critic's actor |
|---|---|---|---|
| Recorded conditions | 0.00 | 0.01 | 0.05 |
| Strong current | 0.09 | 0.41 | 0.73 |
| Big waves | 0.01 | 0.04 | 0.16 |
| Big kicks | 0.07 | 0.11 | 0.16 |

The marks are taken in strong current:

| Mark | Rule | Alpha 2.5 | Alpha 5 |
|---|---|---|---|
| a | harm of at least 5 points | 86.5 | 98.9 |
| b | at least 3 points more than random removal explains | 88.3 | 95.8 |
| c | at least 3 points more than the no-stall comparison world explains | 86.1 | 98.6 |
| d | pushes past 80% at least 1.5× as often as the full-log critic's actor | **1.46 (failed)** | 1.77 |
| e | the loss sits at the past-80% steps | 2.00 per step there, 0.00 elsewhere | 2.03 per step there, −0.01 elsewhere |

The alpha ladder (2.5, then 5, then 10) was fixed in advance and is the same for every method. It stopped at 5, the
first alpha where every mark held.

**Plain TD3+BC, reported only.** Ordinary TD3+BC trained on the gapped log (critic and actor together) shows the same
pattern. It beats cloning by 30 points in the recorded conditions, 27 in big waves and 47 in big kicks, but loses 71
points to cloning in strong current (−16.8 against 54.4). So policy optimisation helps wherever the missing commands do
not matter, and it is badly hurt where they do.

## Check 3: could caution placed at the harm help? (no situation certified)

Each arm is an actor trained against the gapped critic at alpha 5, with extra pull toward the recorded command on
each training row, in BCA's own form (the BC term multiplied by a per-row factor):

- **none:** no extra caution;
- **even ×m:** the same factor m on every row;
- **harm-placed:** factors in [1, 1.5) placed by the exact harm of each row (the largest factor on the most harmful row);
- **strength-matched even:** even caution tuned, before any return is computed, to change the actor exactly as much as
  harm-placed caution does;
- **amount-matched:** the same total extra caution as harm-placed, spread evenly;
- **shuffled:** the same factors as harm-placed, in random row order (two shuffles);
- **reversed:** the same factors placed backwards (the largest on the least harmful row);
- **cap 2:** harm-placed with factors up to 2 instead of 1.5.

Points, means over seeds 0–4:

| Arm | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| none | 87.6 | −10.1 | 72.7 | 53.7 |
| harm-placed | 88.2 | −7.6 | 73.7 | 54.1 |
| strength-matched even | 87.7 | −5.8 | 72.8 | 53.3 |
| amount-matched | 87.6 | −7.0 | 72.6 | 53.3 |
| shuffled (mean of two) | 87.6 | −7.1 | 72.6 | 53.5 |
| reversed | 87.4 | −6.4 | 72.4 | 53.1 |
| cap 2 | 88.1 | −5.6 | 74.2 | 53.6 |
| even ×1.5 | 87.2 | −4.5 | 72.3 | 51.9 |
| even ×2 | 85.7 | −0.7 | 70.7 | 48.3 |
| even ×3 | 82.4 | +7.1 | 67.3 | 41.0 |

The marks: harm-placed must beat strength-matched even, amount-matched and shuffled by at least 3 points each.

| Situation | Over strength-matched even | Over amount-matched | Over shuffled | Certified |
|---|---|---|---|---|
| Recorded conditions | +0.5 | +0.6 | +0.6 | no |
| Strong current | −1.9 | −0.6 | −0.6 | no |
| Big waves | +0.9 | +1.1 | +1.1 | no |
| Big kicks | +0.8 | +0.8 | +0.7 | no |

Outside strong current, placement does matter in the expected direction: harm-placed caution beats every control on
all 5 seeds and reversed placement loses to it on all 5. But the difference is about 1 point against a mark of 3.

In strong current, where 99 points are at stake, no caution in BCA's dose range moves the actor much. The share of
steps past 80% stays between 0.61 and 0.68 for every arm with factors of 1.5 or less. The pre-registration asks for
the cap-2 arm in this case, to show whether BCA's dose range is the limit. It is not: harm-placed factors up to 2 beat
strength-matched even caution by only 0.2 points in strong current (1.4 in big waves, 0.3 in big kicks). Only strong,
uniform caution (even ×3) lowers the share past 80%, to 0.50. That recovers 17 of the 99 points, still leaves the
actor 47 points below simple cloning, and costs 5–13 points in the other situations.

**Why the caution cannot reach the harm.** Every strong-current start lies outside the recorded range (current above
0.7). The pull toward the recorded commands acts only on the recorded rows. In strong current the actor is
extrapolating. What it carries over from training is a critic that sees no cost in pushing past 80%: the gapped data
never show the stall and cannot tell the stalling world from the no-stall comparison world. A larger pull on
the training rows barely changes how far that belief reaches. The cloned actor shows the alternative: with no critic
at all it pushes past 80% on 9% of strong-current steps and scores 54.

## Shift cards (computed outside the gate, reported only)

The gate's report-card stage has a one-line bug. `acceptance.report_cards` passes the table of reference returns for
all four situations where `shift.shift_card` expects the table for one situation, so every report-card run raised
`KeyError: 'top'`. The file is hashed into the pre-registration, so it was not edited. The bug touches no pass rule:
it is in the stage that only describes the stop.

`cards/cards_seed{0..4}.json` repeat that stage line for line, with the one difference (`anchors[situation]` instead
of `anchors`). They are produced by `cards_outside_gate.py` (kept next to them) and read every gated output through
the gate's own verified loaders. Each file says so in its `note` field. They describe, for every reference controller
and learned actor in every situation, where the actor goes: how much of the time it is outside the recorded
conditions, how often it commands past 80%, and what share of the recording controller's commands at those states
fell in the removed region. The bug will be fixed only under a later pre-registration.

Check: all 120 learned-actor scores on the cards (6 actors × 4 situations × 5 seeds) equal the gated evaluations to
within 3.2e-14 points, and the reference controllers score exactly 0 and 100. Means over seeds 0–4 are in
`cards/cards_summary.md` (made by `cards/summarise_cards.py`). What they show:

- **The shifted situations call for the removed command.** At the states visited in strong current, the recording
  controller would have given the removed full-throttle command 57% of the time, against 17% in the recorded
  conditions (big waves 24%, big kicks 17%). So the command missing from the data is the one strong current needs most.
- **Strong current and big waves lie entirely outside the recorded conditions** for every controller (share 1.00).
  Big kicks lie outside them on only 11–18% of steps, which is why its harm is smaller.
- **The gapped critic's actor goes far from the data.** Its visited (state, command) pairs in strong current sit at a
  median distance of 3.93 from the training rows (standard units; held-out recorded rows sit at 2.49). The full-log
  critic's actor sits at 3.03 and the stall-aware controller at 2.82. It loses 116 points at its past-80% steps.
- **Caution placed at the harm barely changes this.** The harm-placed check-3 actor pushes past 80% on 65% of
  strong-current steps (68% with no extra caution) and still loses 108 points at those steps (111 without).

## Files

| Path | Holds |
|---|---|
| `check1_critic/pooled.json` | check 1, pooled (copy of `runs/shift/68d9e8779e148b7b/harm_critic/base/pooled.json`, with its SHA-256 sidecar) |
| `check2_return/alpha2.5/pooled.json`, `check2_return/alpha5/pooled.json` | check 2 at each alpha of the ladder, pooled |
| `check3_room/pooled.json` | check 3, pooled |
| `check3_room/points_by_arm.json` | the per-seed points of every check-3 arm (drawn from the per-seed `values.json`; reported) |
| `cards/` | the shift cards computed outside the gate, and the script that made them |
| `runner_progress.txt` | the run log, step by step |
| `run_shift_gate.py` | the runner that called the gate's stages in order (it decides nothing; the gate's own code picks each next step) |

**One runner incident, refused by the gate.** At 16:51 a bug in the runner (not in the gate) made it misread the
gate's state after check 1 and ask for the pre-registered retune. The gate refused all six commands ("the retune runs
only when the base world's harm checks failed") and wrote nothing. The runner was fixed and restarted, and check 2
began at 16:52 on the base world, as the rules require. The log shows both attempts.

The per-seed outputs, trained actors and critics stay in `runs/shift/68d9e8779e148b7b/` (not committed: model files).

## What could come next (the user decides)

1. **Accept the stop.** The world is sound and the harm is real, but BCA's mechanism cannot help in it. Record that
   and move on.
2. **Exploratory, no BCA:** test which kind of caution can reach a harm that lies outside the recorded range. For
   example, larger dose caps, caution placed on the training rows nearest the shifted states, or a limit on the
   commands themselves (stay inside the recorded commands' range) instead of a stronger pull. This would be labelled
   exploratory and would not touch the gate.
3. **Exploratory BCA detection only:** fit BCA's band and read its doses, with no BCA-trained actor, to ask whether
   BCA finds the rows where the critic is wrong (the detection half of the claim). Under this pre-registration that
   needs the user's typed approval at a terminal. This cloud session has no terminal for the user, so it needs either
   a run on the user's machine or a new, labelled mechanism for approval.
4. **A new pre-registration with a redesigned test:** for example, shifted situations that stay inside the recorded
   states but need the removed command there, so that pulling toward the recorded commands on training rows can
   matter. The report-card bug would be fixed in the same step.
