# Stage 1 result: the value-penalty route can carry the harm, and placement is what matters

**EXPLORATORY.** This test uses no BCA, and none of its signals was fitted or read. It runs outside the gate, on the
gate's saved artifacts, and changes no pre-registered rule.
- **Spec:** `spec.json` (`56bbda24…`), with its code pinned by `code_manifest.json` (`469ada26…`). Both were committed
  in a1ea28e, before any return was computed.
- **Results:** `results/` holds the pooled look (`pooled.json`), the futility look (`futility.json`), and each seed's
  lock and values for the stalling world, the no-stall world and the replication.
- **Setup:** gate seeds 0–4, alpha 5. Every actor starts from the gate's saved cloned actor and trains for 20,000 steps
  against check 2's frozen gapped critic, which never saw a command past 80%. That critic estimates the value of
  following the stall-aware controller, so it is a fixed-controller critic.
- **Order of steps:** every untargeted control was matched in strength to the main arm, blind, before any return was
  computed, and a lock file was written. Only then were exact returns computed.

## The question

The gate's check 3 showed that adding caution as extra pull toward the recorded command, on training rows, cannot
reach this harm. However that pull is placed, it barely changes the actor in strong current.

Stage 1 asks a different question: can the harm be reached through the value the actor climbs, if the actor is
penalised at its *own* command?

The main arm uses the best information there is, an exact oracle. Its penalty is
[delta*(s, a_i) − delta*(s, a)]_+: how much more the critic overrates the actor's command a than the row's recorded
command a_i, measured by the critic's exact one-step error. If even this oracle cannot beat untargeted caution of
the same strength, no BCA version of it can.

## Marks (fixed in advance): all pass

| Mark | Rule | Result |
|---|---|---|
| Room | in strong current, the oracle beats each untargeted control by at least 20 points, positive in at least 4 of 5 seeds | **passed**: by 92.4 over even extra pull, 89.4 over push-less-everywhere, and 99.0 and 94.2 over the two random penalty surfaces; positive in 5 of 5 seeds against each |
| Recovery | in strong current, at least 49.5 points above no caution (half of the 98.9-point harm) | **passed**: +93.9 (seeds 83.4 to 103.9) |
| Keeps the policy improvement | recorded conditions: at least no caution − 2 and at least cloning + 25; big waves and big kicks: at least no caution − 3 | **passed**: recorded conditions +7.9 over no caution and +37.0 over cloning; big waves +19.6; big kicks +16.9 |

## Points by arm (means over seeds 0–4; 0 = recording controller, 100 = stall-aware controller)

Stalling world, against check 2's frozen gapped critic:

| Arm | Recorded | Strong current | Big waves | Big kicks | Share of strong-current steps past 80% |
|---|---|---|---|---|---|
| Cloned actor (the starting point) | 56.3 | 54.4 | 43.6 | 2.0 | |
| No extra caution | 85.5 | −14.3 | 70.0 | 52.5 | 0.73 |
| **Oracle penalty at the actor's command** | **93.4** | **79.7** (72.7 to 83.5) | **89.7** | **69.4** | 0.54 |
| Oracle, half strength | 93.0 | 75.5 | 88.8 | 68.1 | |
| Oracle, two-sided form | 92.9 | 75.7 | 88.8 | 68.1 | |
| Oracle, the plan's literal form (size of the error) | 91.4 | 77.5 | 85.9 | 67.2 | 0.68 |
| Exact long-run harm as the penalty | 93.6 | 82.2 | 89.9 | 70.8 | 0.50 |
| Oracle taken from a *different* training state | 88.7 | −0.3 | 76.0 | 55.6 | 0.65 |
| The underrating side of the error (wrong direction) | 84.3 | −17.0 | 65.9 | 50.9 | |
| Even extra pull, matched strength | 83.3 | −12.7 | 66.8 | 47.3 | 0.66 |
| Push less everywhere, matched | 82.9 | −9.7 | 70.0 | 55.0 | |
| Random penalty surface 1, matched | 82.7 | −19.4 | 68.0 | 50.2 | |
| Random penalty surface 2, matched | 83.5 | −14.5 | 67.5 | 51.4 | |
| Clip to the recorded range (no signal) | 93.3 | 93.1 | 91.6 | 74.9 | 0.26 |

No-stall comparison world. Here the missing commands are good, so any caution about them can only cost. The table
gives the cost in strong current: no caution minus the arm.

| Arm | Cost in strong current | Cost in the other three situations |
|---|---|---|
| Oracle penalty (from this world's own critic) | **1.3** (0.4 to 2.3) | 0.4 to 0.7 |
| Oracle, scored with the next command of the controller that assumes thrust keeps rising (`perfect_band_rising`) | 4.6 | |
| Even extra pull, at its stalling-world strength | 8.1 | |
| **Clip to the recorded range** | **30.9** (24.8 to 34.6) | 1.0 to 6.0 |

Replication on check 3's critic, the co-trained TD3+BC critic from the room check: the oracle penalty scores 84.7 in
strong current (77.9 to 89.1), against −2.8 for matched even pull.

## What it means

1. **The route through the value the actor climbs reaches harm outside the data.** The oracle penalty is applied
   only at training states, at the actor's own command. It recovers 94 of the 99 points lost in strong current, a
   situation the data never contain. It also improves every other situation, by 8 to 20 points over no caution.
   The extra pull toward recorded commands could not do this.
2. **Placement is almost the whole effect.** Untargeted caution of the same strength recovers nothing: even pull,
   push-less-everywhere and random penalty surfaces all leave strong current between −19 and −10.
   - The correct information taken from a different training state recovers little (−0.3).
   - The underrating side of the same error recovers nothing.
   - So a signal that finds the actor's overrated commands, at the state where they occur, is what repairs the harm.
3. **A perfect signal is cautious only where caution is needed.** In the no-stall world, where the missing commands
   help, the oracle costs 1.3 points. The plain clip to the recorded range costs 30.9. In the stalling world the clip
   does slightly better than the oracle (93.1), because there every command past 80% is bad. A signal that only
   detects novelty can therefore look as good as a perfect one in this world, and pay for it in the other world.
   This is why the claim that BCA identifies the harm needs the trace test bed.
4. **The reading must be paired.** The plan's literal oracle (the size of the error) still works here (77.5), but
   check (e) showed that it rewards the first points past 80%. The paired form, the error at the actor's command
   minus the error at the recorded command, is what Stage 2 uses for BCA's width.

## What comes next

- **Data-only stand-ins in the same slot** (plan 4.2): distance to data, behaviour density, classifier ratio,
  two-head disagreement and the range hinge. They set the bar BCA has to clear.
- **Deployment-check additions** (plan 4.3) and the **locked detection rehearsal** (plan 4.4).
- **Stage 2, the flags claim, with BCA itself.** Its pre-registration is written. It runs only after the user approves
  a fingerprinted version of it.
- **The trace test bed** (plan 4.6, the harm claim): running.
