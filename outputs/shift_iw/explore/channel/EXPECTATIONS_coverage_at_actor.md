# Expectations, written before running: BCA's band where the lower edge reads it

Written 2026-10-14T02:32Z, before `coverage_at_actor.py` was run. EXPLORATORY and post hoc: outside the version-5
pre-registration. No training, and nothing in `runs/shift_iw_v5` is changed.

## Why

The owner wants a return gain from weighting, and asks which channel lets the penalty change the policy. The perfect
information room checks already rank the two channels:

- **Capped pull:** even perfectly placed, it gains at most 0.2 points.
- **Lower edge:** with the exact one-step error, it gains +5.8 points on medium-expert and +8.4 on expert under the
  strong shift.

So the lower edge is the channel with room. Yet BCA's lower edge moved return by less than 4 return units. This check
asks why, in terms of coverage: is BCA's band honest at the commands the actor proposes, which is where the lower edge
reads it? Version 5 measured coverage only at recorded commands.

## What is measured

Gapped host, seeds 0–4, all five qualities. The rows are every kept training row. The actor is the lower-edge room
check's no-caution actor (`le_none`).

At each row's state s and the actor's command a = π(s):

- **Truth:** y = r(s, a) + 0.98 · min over the two heads of the target critic at (s', target actor(s') + one draw of
  clipped target noise). Here s' and r come from one exact float64 step of the simulator, and this is the same target
  rule BCA calibrates (`learners_iw.critic_parts`).
- **Critic:** q = min(Q1, Q2) at (s, a).
- **Covered:** |y − q| ≤ U(s, a), with U = R × unit × scale(s, a). R is the unweighted threshold R_u, or each row's
  exact-strong per-step threshold, capped at the largest finite one as in the lower edge.
- **Strong-law figures:** each row is reweighted by w(c) = p_strong(c) / p_train(c).
- **Groups:** all rows; rows where the actor's command stalls a thruster (past 80%); hard-water rows.
- **Needed multiplier:** the strong-law weighted 90% quantile of |y − q| / U(s, a), which is how much wider the band
  would have to be to cover 90% there.
- **Sanity check:** the same coverage at the recorded commands, using the recorded transitions.

## Expected

- **E1 (sanity):** at the recorded commands under the training law, the unweighted band covers at least 85% of rows
  on every quality and seed.
- **E2:** at the actor's own commands under the strong law, the unweighted band covers below 90% on every quality.
  At the actor's past-80% commands it covers below 50% on every quality where those commands occur on at least 1% of
  rows.
- **E3:** exact-strong weighting raises the strong-law coverage at the actor's commands by less than 5 points, as a
  mean over seeds, on every quality. At the actor's past-80% commands, the needed multiplier is above 3 on the top two
  qualities. The largest weighting factor seen in version 5 was 1.8.

## What the outcomes would mean

- **If E2 and E3 hold:** the lower edge is the right channel, but it reads the band where the band was never
  calibrated. Weighting by the current law cannot fix that. The weights would have to describe where the band is read:
  the actor's commands under the test law.
- **If E2 fails** (the band is honest at the actor's commands): the lower edge's failure has another cause, such as
  the shape of the paired form or the gradient through the scale network, and the channel question stays open.
