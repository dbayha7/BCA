# Kill rule: mixed world split by wave size (written 2026-10-14, before any result of this test exists)

EXPLORATORY, designed after seeing seeds 0-4. Seeds 0-4 only. No training in the decisive test. Proposed, not yet run.

## Test
Wave amplitude never changes within an episode and both worlds' logs hold the same episodes from the same starts
(routes/_where/feas_amp_0_1_2_3_4.json). So these logs are exact, built by splicing whole episodes:
- W1: full throttle stalls when amplitude > 0.25 (the midpoint of the recorded law U[0.1, 0.4]), helps otherwise.
- W2: the mirror (stalls when amplitude <= 0.25).
Strong-current starts draw amplitude from the same U[0.1, 0.4], so each deployment mixes both regions.
Per start: points of 'never' and 'clip past t_lo' in the physics that matches its amplitude (Stage 2 'none' actors of
both worlds; primary = the stalling world's actor).
Local BCA: for each episode, BCA's WBCP posterior (calibration/wbcp.calibrate) with weights = Gaussian kernel in
amplitude around the episode's amplitude (bandwidth by Silverman's rule on the full-throttle evidence rows'
amplitudes; test mass 1) on the full-throttle group and on the rest; step 1's statistic and margin 0.10; on -> clip
past t_lo for that episode.
Evidence, primary: the held-out calibration-split rows past t_lo (6-18 per region) plus the bank, residuals from the
gate critic of the world each row came from. Rich (mechanism check): the removed full-throttle training rows
(~2,500-4,000 per region), held out from the gate critic.

## Shelve
- The whole route, if with RICH evidence the local certificate's mixed-world net (mean of W1 and W2, seeds 0-4) does
  not beat the best of {world-level certificate (step 1 rule on all tail rows), clip-always, never} by >= 10 points.
- The claim "BCA can resolve region-specific harm at the evidence a gapped log leaves", if with PRIMARY evidence it
  does not beat that best rival by >= 10 points.

## Reported, not deciding
State-kNN form (k = 20 nearest full-throttle rows in the normalised 13-number state; does not know amplitude matters);
plain local Mann-Whitney; outcome-permuted control; the swap-the-critic gap (approximation error of using each world's
gate critic instead of a critic trained on the spliced log).

## Triage (selected, rank 5 of 5)
Written 2026-10-14 22:17 UTC by the triage step, before any result of this route exists (runs/shift_explore/routes/ does not exist; this folder held only the proposer's rule above, written before step 1's first evaluation output at 22:07). Seen before writing: step 1's finished evaluation (runs/shift_explore/restructure/step1/logs/evaluate.log: verdict GO, net 106.1; per seed-world statistic, height, Mann-Whitney p). This route uses none of step 1's critics or scores.

Sharpened, fixed now. Decisive arm: the amplitude-kernel local certificate exactly as specified above (it is told that
amplitude matters but not where the cut is; the claim must say so). Mixed-world net = mean over W1 and W2 and seeds
0-4 of (points with the rule minus 'never'), each start scored in the physics that matches its amplitude (cut 0.25).
Best rival = the best of {world-level certificate, clip-always, never} on the same starts. Shelve:
1. The whole route, if with RICH evidence the local arm beats the best rival by less than 10 points (unchanged).
2. The claim "with the evidence a gapped log leaves", if with PRIMARY evidence the local arm beats the best rival by
   less than 10 points (unchanged).
3. NEW, decisive for the same claim: an outcome-permuted control with primary evidence. The full-throttle evidence
   rows' scores are permuted among themselves, keeping each row's amplitude (this keeps the world-level gap and removes
   any link between amplitude and outcome); rng default_rng([20261015, 9950, seed, mixed_world_index]). Shelve the
   claim if this control keeps at least half of the local arm's margin over the best rival (then the local switching
   comes from where evidence lies, not from outcomes).
Primary evidence stays as written (held-out calibration-split rows past t_lo plus the bank); step 1's held-out
survivors are NOT added (that would change the design after seeing step 1's output). Approximations to state in any
report: each row is scored by the gate critic of the world it came from, not a critic trained on the spliced log;
the actor is the stalling world's 'none' actor in both regions.
Reported only: the state-kNN arm (not told amplitude); plain local Mann-Whitney; the no-stall 'none' actor.
CPU cap 60 minutes (estimate 30).

## Precision addendum (written 2026-10-14 by the test runner, BEFORE any score, decision or rollout of this route exists)
Nothing above is changed or loosened; this section only fixes details the text leaves open. No result of this route
existed when it was written (runs/shift_explore/routes/mixed_waves/ does not exist; only a synthetic-number timing of
wbcp.calibrate was run). Not read before writing: the results of the other route tests running in parallel.

Spliced worlds. W1 (index m = 1): a log row or deployment start with wave amplitude > 0.25 belongs to the stalling
world (its log rows, its gate critic, its physics, its anchor); amplitude <= 0.25 to the no-stall world. W2 (m = 2):
the mirror (amplitude <= 0.25 stalls). Amplitude = TC.wave_amplitude of the float64 state (log: infos/state).

t_lo and recorded range, per seed and spliced world: step1lib.gap_rule (0.99, 0.05, 9; step 1's prespec settings) on
the throttle of the spliced log's kept training commands (each region's rows from its own world's gapped prepared
data); recorded range = per-thruster 0.5% / 99.5% quantiles of those same commands. If the rule is undefined: stop and
report (plumbing), no verdict.

Evidence (each row scored by step 1's evaluate.overestimates, noiseless controller target, with the gate's gapped
fixed-controller critic, normalisation and stall-aware controller of the world the row came from):
- Primary full-throttle group: every calibration-split row of the spliced log whose command is outside its own
  world's removed ball and has throttle > t_lo (held-out calibration rows past t_lo; the bank's full-throttle rows
  included) - the construction of routes/heldout_calibration_rows/run_route.py, applied per region.
- Rich full-throttle group: every training-split row removed by its own world's gap removal with throttle > t_lo.
- Rest group (both evidence types): the bank rows (after removal) of the spliced log with throttle <= t_lo.

Local arm (amplitude-kernel local certificate), per deployment start with amplitude a0:
- weights w_j = exp(-0.5 ((amp_j - a0) / h)^2) on the rows of both groups; h by Silverman's rule
  0.9 * min(sd (ddof=1), IQR / 1.34) * n^(-1/5) on the amplitudes of that seed / spliced world / evidence type's
  full-throttle group; the same h for the rest group.
- calibration.wbcp.calibrate(scores, rng, weights=w, test_mass=1.0, alpha=0.1, beta=0.95, draws=1000) separately on
  each group; rng default_rng([20261015, 9960, seed, m, e, 1]) for the full-throttle group (e = 1 primary, 2 rich,
  common to every start) and default_rng([20261015, 9960, seed, m, 3, 2]) for the rest (shared by both evidence types).
- statistic = step1lib.pairwise_quantile(tail crossings, rest crossings, 0.05); on iff statistic > 0.10.
- on: that episode runs the check (each step: fire when max_k |a_k| > t_lo; a fired command is clipped to the recorded
  range; step 1's rollout rule); off: never fires.
World-level certificate rival: step1lib.certificate (uniform weights, step 1's statistic and margin) on the SAME
evidence type's full-throttle group vs the rest, tags [20261015, 9961, seed, m, e]; on -> check for every start.
Clip-always: check for every start. Never: no check (net 0 by definition).

Rollouts: the stalling world's Stage 2 'none' actor (runs/shift_explore/stage2/policy/seed<k>, SHA-256 checked via
step 1's evaluate.none_actor), L.mlp_policy with the stalling world's prepared normalisation, first 200 strong-current
bank starts, in both physics, never and with the check (t_lo and range of each spliced world). Per-start points with the
anchor of the world whose physics ran the start. Plumbing check before use: the same rollout code with the stalling
world's own t_lo and range must reproduce step 1's saved stalling-world 'never' and 'on' points (prespec
deployment.seen_design_values) to 1e-9.

Nets. Arm net for (seed, m) = mean over the 200 starts of (arm points - never points), each start in the physics its
amplitude selects in W_m. Mixed-world net = mean over m in {1, 2} and seeds 0-4. Best rival (per evidence type) = the
highest mixed-world net among {world-level certificate on that evidence type, clip-always, never (0)}.
Margin = local net - best rival net.

Decisions (unchanged in substance):
1. Shelve the whole route if margin_rich < 10.
2. Shelve the claim if margin_primary < 10.
3. Outcome-permuted control: per seed and m, the primary full-throttle group's scores are permuted among its rows
   with default_rng([20261015, 9950, seed, m]).permutation (amplitudes and the rest group unchanged; the local arm is
   rerun with the same h, weights and posterior tags). Control margin = control local net - best rival primary net
   (the rivals do not depend on the permutation; recomputed to confirm). Shelve the claim if
   control margin >= 0.5 * margin_primary (this also fires whenever margin_primary <= 0).
Verdict mapping for the report, fixed now: any of 1, 2, 3 fires -> "dead end" (if only 2 or 3 fire, the report says
the mechanism works with evidence a gapped log does not have, and the route's claim is shelved); none fires -> "alive".
A plumbing stop -> "inconclusive".

Reported only (fixed now): the oracle region arm (on iff the start's region stalls in W_m: the ceiling); the state-kNN
arm (the 20 nearest full-throttle rows of that evidence type in the 13-number observation normalised with the stalling
world's gapped normalisation, uniform weights, vs the whole rest group, step1lib.certificate, tags
[20261015, 9962, seed, m, e]); the local Mann-Whitney arm (one-sided, full-throttle rows vs rest rows with
|amp - a0| <= h; on iff p < 0.05; off if no full-throttle row in the window); everything again with the no-stall
world's 'none' actor (decisions unchanged); on-rates of each arm by region (stalling region vs helping region);
20 further outcome permutations (tags [20261015, 9951, seed, m, j]) as information.
CPU cap 60 minutes (estimate 15-30).
