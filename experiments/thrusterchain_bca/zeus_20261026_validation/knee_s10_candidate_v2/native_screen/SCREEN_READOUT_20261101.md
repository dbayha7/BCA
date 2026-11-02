# Native screen for candidate knee_s10_v2: readout of attempt 3 (2026-11-01, 14:01 to 14:05 EST)

**Reading: INCONCLUSIVE, because G1 is NOT REPRESENTABLE.** The screen stopped at its second unit, before any
warmup, branch or evaluation. GPU time spent by the screen: none. CPU: 197 s and 7,968,000 exact simulator steps
(the labels and the anchor values), under the 3,600 s and 9,000,000 caps.

Record: `screen_run/attempt_3/` (`SCREEN_RESULT.json` sha256 b784e6b0…, `g1/report.json` sha256 beae3bcd…), under
manifest e2d09d22… (amendment A1) and the user's approval naming attempt 3. Attempts 1 and 2 are INCOMPLETE records
of two launch faults fixed before any unit ran (`SCREEN_RULE.md`, Amendments). The preflight is attempt 1's, reused.

## What G1 asked, and what it found

G1 is the positive control on the representation (`../g1/G1_RULE.md`): Unifloral's `SoftQNetwork`, line for line,
fit by mean-squared error to exact Q^{pi_ref} labels on all 76,800 fitting rows of each formal G2 dataset, at the
host's settings (Adam 3e-4, batch 256, 20,000 updates, three fit seeds), then read at the 96 gated anchors through
the contrast error KC(slot) = [Q1(slot) − Q1(edge_knee)] − [Q^{pi_ref}(slot) − Q^{pi_ref}(edge_knee)]. It trains
nothing by temporal difference and says nothing about withholding or BCA. Its question: can this critic hold the
value departure at s10 when told the truth?

Instrument checks passed exactly: I0 (the recomputed median exact rho at s10 equals the formal report's
1.7652868718841468 to the last digit, H1 identity 0.0; the Linux recomputation agrees with the Windows formal G2),
F0 (every loss and every Q1 finite), and the upstream source match.

| Read (stall3 fit, 20,000 updates) | seed 0 | seed 1 | seed 2 | bar |
|---|---|---|---|---|
| median KC(s10) over the 72 valid anchors (rho > 0) | 1.559 | 1.403 | 1.437 | −0.15 ≤ KC ≤ 0.3 median rho + 0.15 = 0.980 |
| L1 share (per-anchor band −0.15 ≤ KC ≤ 0.3 rho + 0.15) | 0.111 | 0.222 | 0.236 | ≥ 0.60 each; pooled 0.190 against ≥ 0.70 |
| over-capture share (KC above the band: the critic values s10 too high) | 0.764 | 0.681 | 0.708 | reported |
| under-capture share (KC ≤ −0.15) | 0.125 | 0.097 | 0.056 | reported |
| anchor MSE (14 slots, exact reference) | 28.1 | 21.1 | 24.6 | reported |

Median exact rho over the 72 valid anchors is 2.767 (over all 96, 1.765). The critic recovers about half of the
departure: its contrast at s10 sits about 1.4 to 1.6 return units above the truth, in the over-valuing direction,
in every seed. q* = 0.111 (k = 8 of 72). The in-support resolution band, the 90th percentile of |KC(edge_700)|
pooled over seeds, is 1.843 (median 0.812): one slot inside the knee, in support, the stall3 fit's contrast error
is already the size of the departure it is asked to resolve.

**The smooth placebo** (no-stall fit, 20,000 updates): median KC(s10) −0.02 to +0.05, 10th to 90th percentiles
about ±0.25, and 55 to 64 percent of anchors within the plain tolerance 0.15. In the smooth world the same critic
resolves the s10 contrast to a few tenths; the tolerance is tight but achievable there. The difficulty is the stall
world's value function, not the slot or the anchors.

**The under-training diagnostic** (60,000 updates, reported only): median KC(s10) 0.936 / 0.915 / 1.100 (two seeds
inside the median band), shares 0.389 / 0.431 / 0.347, pooled 0.389; anchor MSE 9.9 / 10.2 / 8.0; training MSE
0.84 to 0.91 against 2.3 at 20k. Three times the budget halves the miss and still falls well short of every bar.

**The profile.** The miss is proportional along the probe line: pooled stall3 median KC at 20k is 0.39 at s05,
0.84 at s10, 1.35 at s15, 1.91 at s20, against median exact rho 0.84, 1.77, (s15 not scored), 3.80. The critic
smooths the kink at the knee and reproduces roughly half the drop at every distance past it; moving the slot
further out would not change the ratio (and the rule forbids a slot change for this candidate anyway).

## What this means

- Under definition v4 (section 5, G1): **stop this candidate on this host configuration.** No learning run on this
  host can be read as "withholding caused the miss at s10", and no BCA run as "BCA repaired it", because the critic
  cannot hold the departure even with exact labels. The departure (median 1.8, valid-median 2.8 return units) lies
  at or below the in-support resolution of this critic at 20,000 updates (band 1.8).
- The miss is a representational over-valuation: given the truth, the critic still values s10 about half a drop
  too high. That is the optimistic error BCA is meant to counter, arising here from function-class smoothing, with
  no data withheld. Any TD-learned critic at these settings carries this error before withholding adds anything,
  which is why the two cannot be told apart on this host.
- Not changed by this result: G2 PASS (the exact value departure across the physics worlds exists); RS1 as recorded;
  the user's correction that G2 established a value difference, not learned-critic over-estimation.
- Not said by this result: anything about BCA, kappa 63, the RANDOM control, or the Unifloral learner's TD behavior.
  Nothing trained.

## What the rule allows next

A changed host setting or feature is a new candidate in the ledger (G1_RULE.md, "Labels"); the data, the slot and
the tolerance may not change for this candidate. The two settings G1 points at: the critic's update budget (60k
helped, slowly) and its input representation (the kink at |a0| = 0.7 is what the raw-action MLP smooths). Both are
host changes and both touch the "Unifloral base only" rule; neither is started here. The screen's remaining units
never ran and its approval is spent.

## Ledger entry

Candidate knee_s10_v2, sigma s10, native screen under r4 at kappa 63, attempt 3: INCONCLUSIVE (G1 NOT
REPRESENTABLE at 20,000 updates: pooled L1 share 0.190, median KC 1.40 to 1.56 against the 0.980 band; q* 0.111;
resolution band 1.843; placebo 55 to 64 percent within 0.15; 60k diagnostic share 0.389). GPU hours 0; exact
transitions 7,968,000 (CPU).
