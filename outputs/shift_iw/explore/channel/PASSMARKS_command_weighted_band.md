# Pass marks, written before running: BCA's band weighted for the actor's own commands

Written 2026-10-14T02:43Z, before `command_band_check.py` was run. This is the owner's go of 2026-10-14 ("Yes, build
and check it (Recommended)"): build the band, check it without training, and show the results before any training
test. EXPLORATORY: outside the version-5 pre-registration. Nothing in `runs/shift_iw_v5` is changed.

## Why

The channel check (`RESULTS_coverage_at_actor.md`) found three things:

- The lower edge is the channel with room.
- BCA's band is honest within 80% at the actor's commands but covers only 25% (expert) and 48% (medium-expert) past
  80%, where the critic overrates the actor's command.
- Weighting by the current law only rescales the band.

The weights must therefore describe where the band is read, which is the actor's commands under the test law.

## The band (one change from version 5: the weights)

The scale network, the unit, the bank scores, the random stream and the posterior (wbcp.calibrate, α 0.1, β 0.95,
1,000 draws) are BCA's frozen ones, unchanged. Only the weights and the test masses differ.

**Covariate:** a row's (episode type, state, command). The training distribution is the recording process, with its
known random share ε per episode (from the episode's pick). The test distribution keeps the episode and state
distribution, reweighted to the strong current law, and draws the command from the actor's command plus BCA's own
target-noise spread (Gaussian, standard deviation 0.2, clipped at ±0.5, then clipped to [−1, 1]).

**Exact ratio** of a (row, command a):

ω = w(c) × π_h(a | s) × Z_ε(s) / β_ε(a | s)

- **w(c):** p_strong(c) / p_train(c).
- **π_h(a | s):** the actor's spread density, ∏_k φ((a_k − π_k(s)) / 0.2) / 0.2 where every |a_k − π_k| < 0.5, and 0
  elsewhere.
- **β_ε(a | s):** the recording controller's command density, (1 − ε)(1 − 0.5 · hard_water(s)) n(a | s) + ε / 16.
  - n is normal driving's density: a = 0.8 tanh(u / 0.8), u ~ N(μ(s), 0.3² I), with μ(s) = −(0.25 K_s s + K_e e)
    the recorder's own gain. It is 0 when any |a_k| ≥ 0.8.
  - The 80% hold is a point mass in thruster 1, so its density is 0. A kept hold row (the ball missed it) gets weight
    0, since the test distribution has no point mass there.
- **Z_ε(s):** the share of the recording process's commands at s that fall outside the removed ball, by Monte Carlo:
  - 256 draws of normal driving per row;
  - 20,000 draws of the hold, which is the same at every state;
  - 200,000 draws for the random command's share.
- **Refusals:** a command inside the ball, or with β = 0 (for example past 80% in an episode with no random share),
  has ω = +∞ and is refused, which is the version-5 rule.

**Bank weights:** ω at each kept bank row's own recorded command, with the actor fixed to the lower-edge room check's
no-caution actor of the seed.

**Test mass** at a row: ω at that row and the command being read. Masses are floored at 1e-4 × the mean bank weight,
where the test atom is negligible. The threshold R(ω) is read from a table on a log grid, with ratio 1.05, coarser
than version 5's 1.01. Each lookup takes the grid point at or above the mass, which is the conservative direction.
Masses beyond the onset of +∞ are refused.

**Width:** U = R(ω) × unit × scale(s, a). In the lower edge, a refused command uses the table's largest finite
threshold: version 5's rule (a), unchanged.

## What is measured

All five qualities, seeds 0–4, gapped host, every kept training row. Strong-law figures reweight rows by w(c).
Truth is one exact float64 simulator step and BCA's own target rule, as in `coverage_at_actor.py`.

- **(a) Guarantee check:** one draw from the test distribution per row (the actor's command plus the spread):
  covered, refused or missed.
- **(b) Where the lower edge reads:** the actor's own command: covered, refused or missed. The groups are within 80%,
  past 80% and hard water.
- **(c) Targeting:** refusal share within and past 80%, and the within-80% width against version 5's exact-strong
  width at the same command.
- **(d) Check (e)** on the new lower-edge penalty [U(s, a) − U(s, a_i)]_+, with the spec's rule, rows, grid and
  actor. U(s, a_i) uses the recorded command's own mass.
- **Diagnostics:** the Kish effective size of the bank weights, and the onset of refusal.

## Pass marks

"Covered or refused" counts a refusal as the infinite band the guarantee allows.

- **C1 (the weights are right):** at draws from the test distribution, covered or refused is at least 88% under the
  strong law, as a mean over seeds, on every quality. The target is 90%, with 2 points allowed for the finite bank.
- **C2 (honest where the lower edge reads):** at the actor's own command, covered or refused is at least 88% under
  the strong law, within 80% and past 80% separately, on medium-expert and expert (mean over seeds).
- **C3 (targeted, not blanket):** within 80% on medium-expert and expert, two conditions hold (mean over seeds):
  - at most 10% of the actor's commands are refused;
  - the certified width is at most 1.5× version 5's exact-strong width at the same command.
- **C4 (the penalty has the right shape):** on medium-expert and expert, two conditions hold:
  - check (e) passes on at least 4 of 5 seeds;
  - the share of hard-water rows on which Q1 − W prefers a throttle past 80% is at most half the share for Q1 alone
    (mean over seeds).

## What happens next

- **If C1–C4 all pass on medium-expert and expert:** I propose the training test, the actor-only lower edge on those
  two qualities, and ask the owner first.
- **If C1 fails:** the weights are wrong, so stop and find out why.
- **If C3 fails:** the weighting acts as a blanket refusal and widening, which is a support constraint rather than
  a targeted band. Report it.
- **If C4 fails while C1–C3 pass:** the band is honest but the penalty's form does not use it. That points to how
  refusal enters the lower edge (version 5's rule (a)), not to the weights.
