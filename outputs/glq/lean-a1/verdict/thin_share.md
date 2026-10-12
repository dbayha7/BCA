# Gapped LQ acceptance verdict (lean path, world thin_share)

Pre-registration SHA-256 `8528a61385157a764e53b9b37fad764ed53bdc0b1089c9755344454429a81f32` (preset real). Host alpha 2.5 (R3 fails at every alpha of the ladder). Seeds: [5, 6, 7, 8, 9].

Amendment A1 (2026-10-12), POST HOC: its thresholds were chosen after seeing seeds 0-4 of pre-registration `255369889b400c2b...` (which keep their original verdict); seeds 5-14 are fresh. R2 is gated on the FQE critic's concentration only (accuracy and the HOST critic: reported, not gated); R3's thin-shortfall mark is 1.5 x the thin mix share; matching runs up to 10 rounds and R4's level needs >= 80% of seeds matched.

**Outcome:** step(s) 5, 6 fail at this world (thin_share): next pre-registered retune: wrong_gain (T_wrong's actuator-3-4 gain x 1.5 (WorldSpec.wrong_gain = 1.5))

| Step | Requirement | Pass mark | Measured | Result | Seeds |
|---|---|---|---|---|---|
| 1 | Bellman identity per context | max relative residual of V = r + gamma E V <= 1e-12 (K*, K2, behaviour, K = 0); another policy's (P, h) gives > 1e-3 | max residual 1.3e-15; control 0.0044 | pass |  |
| 1 | Monte Carlo against the closed form | \|z\| <= 3 per context: 100-step J of the behaviour clone and of the noisy behaviour | max \|z\| 2.0191 (3000 episodes) | pass |  |
| 1 | Finite-horizon recursion | forward vs backward recursion <= 1e-10 relative; 4000 steps vs Lyapunov <= 1e-9 | 5.582e-16; 7.215e-15 | pass |  |
| 1 | Riccati optimum gradient | max \|grad J_inf\| at K* <= 1e-5 x at 0.5 K*, per context | max ratio 9.025e-09 | pass |  |
| 1 | Harness steps equal td3_bc_update | bit for bit: harness_update hooks off (it 1 and 2), with a bc_multiplier, and Tier A's loop against td3_bc_update with the critic restored | 4/4 bitwise | pass |  |
| 1 | Oracle-weight identities | OR-wrong, OR-thin, ANTI, EVEN, SHUF multisets, knob 0 = NONE, CAP in [1, 1.5), OR-D mean 1, OR-D = [-e_A]_+ | 13/13 identities | pass |  |
| 1 | Unit test suites | test_gapped_lq and test_glq_train: 0 failures and 0 errors | ran 66, failures 0, errors 0 | pass |  |
| 2 | Rich behaviour is mediocre | (J(f K*) - J(0)) / (J(K*) - J(0)) <= 0.5 in every rich context | 0.4586 | pass |  |
| 2 | Thin contexts hold room | thin mix share x g >= 0.15 of the total gap | 0.3129 | pass |  |
| 2 | Transfer is costly in T_wrong | K*_T_right in T_wrong: discounted-stable and loses >= 2 g against the behaviour | 8.3959 g, radius 0.9869 | pass |  |
| 2 | Action scale | >= 95% of logged and of optimal actions in [-1, 1]^4 | 0.9704 | pass |  |
| 2 | Decoy noise | decoy median aleatoric target SD >= 3x the rich median | 6.7498 | pass |  |
| 2 | R0 in closed form | the R0 marks on the exact behaviour state laws | min PR 9.2852, transient 0.3400, IACT 16.8163 | pass |  |
| 2 | Ideal transferred policy leaves the log | \|K*_Tr,34 s\| >= 0.3 on >= 30% of thin behaviour states | 0.4218 | pass |  |
| 2 | Every policy used is stable | sqrt(gamma) rho(A - B K) < 1 for all of them | transplant sqrt(gamma) rho 0.9869 | pass |  |
| 3 | R0 participation ratio | >= 6.0 of 10 in every context, every seed | min 9.1025 | pass | 5/5 (need 5) |
| 3 | R0 transient rows | pooled transient share >= 0.30 | 0.3500 to 0.3900 | pass | 5/5 (need 5) |
| 3 | R0 autocorrelation time | pooled IACT in [10, 50] steps | 16.6010 to 17.0289 | pass | 5/5 (need 5) |
| 3 | R0 state scale | >= 99% of rows with every \|s_i\| <= 5 | 0.9998 to 1.0000 | pass | 5/5 (need 5) |
| 4 | R1(a) rich actions explore | all 4 action-covariance eigenvalues >= 0.05 in R0, R1, DECOY | 0.0879 to 0.0899 | pass | 5/5 (need 5) |
| 4 | R1(a) thin actions at the rung | thin actuator-3-4 SD within 5% of the rung (rung 0: rank 2 and exact zeros), every rung | 5 rungs; SD / rung 0.9894 to 1.0113 | pass | 5/5 (need 5) |
| 4 | R1(b) thin share | thin share of training rows = thin mix share +/- 0.03 | 0.5000 to 0.5000 | pass | 5/5 (need 5) |
| 5 | R2 error sits in T_wrong | FQE critic: mean E in T_wrong >= 3x the rich (R0, R1) mean (HOST: reported, not gated) | FQE 2.7542 to 6.0813 (gated on FQE; a seed passes when both concentration marks hold) | pass | 4/5 (need 4) |
| 5 | R2 error sits in T_wrong - HOST critic | FQE critic: mean E in T_wrong >= 3x the rich (R0, R1) mean (HOST: reported, not gated) | HOST 1.0157 to 1.2233 (alpha 2.5) | reported, not gated | 0/5 |
| 5 | R2 thin share of E^2 | FQE critic: the thin contexts' share of sum E^2 (pooled over all 5 contexts) >= 1.5 x their share of rows (base world 0.45, thin_share 0.75) (HOST: reported, not gated) | FQE 0.5187 to 0.7834 (threshold 0.7500) (gated on FQE; a seed passes when both concentration marks hold) | FAIL | 2/5 (need 4) |
| 5 | R2 thin share of E^2 - HOST critic | FQE critic: the thin contexts' share of sum E^2 (pooled over all 5 contexts) >= 1.5 x their share of rows (base world 0.45, thin_share 0.75) (HOST: reported, not gated) | HOST 0.2280 to 0.2787 (threshold 0.7500) (alpha 2.5) | reported, not gated | 0/5 |
| 5 | R2 rich critic level accuracy | reported, not gated: rich median \|Q1(s, pi(s)) - Q_td3(s, pi(s))\| / \|Q^pi\|, Q_td3 = Q^pi + c_c the fixed point of TD3's smoothed target (a perfect critic reads 0; against Q^pi alone it reads 0.16-0.18; the former mark was <= 5%); the deterministic-reference value is reported too | FQE 0.9147 to 1.2476; HOST 0.5437 to 0.5883 | reported, not gated | 5 seeds |
| 5 | R2 rich critic slope accuracy | reported, not gated: rich (R0, R1) median \|H_aa^-1 (grad_a Q1 - grad_a Q^pi)\| / 2 at (s, pi(s)), H_aa of the exact Q^pi of the policy the diagnostics evaluate: how far the critic's error shifts the recommended action | FQE 3.2139 to 4.8406; HOST 1.9265 to 2.4325 (rich logging noise SD 0.3 for scale) | reported, not gated | 5 seeds |
| 5 | R2 dose-response | FQE critic: thin mean E falls in >= 3 of the 4 rung steps (HOST: reported, not gated) | FQE falls 4/4; HOST falls -/4 (reported, not gated) | pass | 3 ladder seeds |
| 5 | R2 dose-response top rung | FQE critic: at rung 0.3, thin / rich mean E within [1/1.5, 1.5] (HOST: reported, not gated) | FQE 1.0194; HOST - (reported, not gated) (reported HOST runs missing: 12) | pass | 3 ladder seeds |
| 6 | R3 off-support cost | w_Tw x off-support cost in T_wrong >= 0.05 x the mixture gap | 0.0790 to 0.2683 (alpha 10) (a seed passes when all four R3 marks hold) | pass | 5/5 (need 4) |
| 6 | R3 host below EXACT | NS(host) <= NS(EXACT) - 0.10 | NS(EXACT) - NS(host) 0.4885 to 0.6575 (alpha 10) (a seed passes when all four R3 marks hold) | pass | 5/5 (need 4) |
| 6 | R3 shortfall in thin contexts | the thin contexts' share of the host-vs-EXACT shortfall >= 1.5 x their mix share (base world 0.45, thin_share 0.75) | 0.5298 to 0.7103 (threshold 0.7500) (alpha 10) (a seed passes when all four R3 marks hold) | FAIL | 0/5 (need 4) |
| 6 | R3 critic useful where rich | host NS in the rich contexts (R0, R1) >= 0.30 | 0.4572 to 0.5754 (alpha 10) (a seed passes when all four R3 marks hold) | pass | 5/5 (need 4) |
| 6 | Host alpha (ladder 2.5, 5, 10) | smallest alpha whose R3 passes | 2.5: fail, 5: fail, 10: fail | FAIL |  |
| 7 | R6(a) thin off-support use | \|pi(s)_34\| >= 0.3 on >= 30% of thin states | - | not run | 0/5 seeds run |
| 7 | R6(b) rich shift | >= 20% of rich states beyond 2 logged-noise SDs | - | not run | 0/5 seeds run |
| 7 | R6(c) Kish ESS | ESS fraction in [0.05, 0.8] in at least one context | - | not run | 0/5 seeds run |
| 7 | R6(d) BC start does not shift | BC start: <= 5% on (a) and on (b) | - | not run | 0/5 seeds run |
| 7 | R1(c) host leaves thin support | >= 20% of the host's thin actions below 1% of the behaviour's peak density | - | not run | 0/5 seeds run |
| 8 | R5(i) confounds | \|Spearman(E, aleatoric SD / \|s\| / distance)\| <= 0.5, each context and pooled | - | not run | 0/5 seeds run |
| 8 | R5(i) T_right below T_wrong | mean E in T_right <= 0.5 x T_wrong | - | not run | 0/5 seeds run |
| 8 | R5(ii) band tracks E | Spearman(population band, E) >= 0.3 | - | not run | 0/5 seeds run |
| 8 | R5(ii) band finds T_wrong | AUROC of the band, T_wrong vs rich rows >= 0.7 | - | not run | 0/5 seeds run |
| 8 | R5(ii) band is not noise | AUROC of the band, thin vs decoy rows >= 0.7 | - | not run | 0/5 seeds run |
| 8 | R5 ideal regressor beats noise | AUROC(high-error rows) of the \|t - Q1\| regressor >= the noise-only one + 0.10 | - | not run | 0/5 seeds run |
| 9 | Level-1 screen (not a stop) | OR-wrong first-order gain - EVEN's at m = 2 > 0 | - (screen, not a stop) | not run | 0/5 seeds run |
| 9 | R4 strength level | S2 (EVEN m = 2) when OR-wrong and OR-thin are both matched there in >= 80% of seeds, else S1 (m = 1.3) under the same rule, else R4 fails; at the level a seed with an unmatched family counts as non-positive and is excluded from means and t-tests (recorded) | - | not run |  |
| 9 | R4(1) OR-wrong beats EVEN | mean NS(OR-wrong) - NS(EVEN) >= 0.05, p < 0.01, >= 80% of seeds positive (an unmatched seed is not positive) | - | not run |  |
| 9 | R4(2) OR-thin beats EVEN | mean NS(OR-thin) - NS(EVEN) >= 0.03, p < 0.05 | - | not run |  |
| 9 | R4(3) best OR-thin vs best EVEN | mean best-minus-best >= 0.02 | - | not run |  |
| 9 | R4(3) caution helps at all | mean best OR-wrong - NONE >= 0.03 | - | not run |  |
| 9 | R4(4) anchors | OR-thin - SHUF >= 0.05 and SHUF - ANTI >= 0.05 | - | not run |  |
| 9 | Capped oracle (report) | report NS(CAP-wrong) - NS(EVEN) at the R4 level (best - best when CAP-wrong cannot reach it); < 0.02 means the BCA test uses P1L placement | - | not run |  |
| 10 | R7(a) minimum detectable effect | MDE <= 0.02 NS at 10 seeds (else the seeds needed, <= 20) | - | not run |  |
| 10 | R7(b) anchors | OR-wrong - SHUF >= 0.05 and SHUF - ANTI >= 0.05 | - | not run |  |
| 10 | R7(c) expert-data null world | \|mean NS(OR-wrong) - NS(EVEN)\| < 0.02 | - | not run |  |
| 10 | Global-linear class | mean NS(OR-wrong) - NS(EVEN) < 0.02 | - | not run |  |
| 10 | R7(d) strength matching | \|S/S* - 1\| <= 5% for >= 90% of (family, target, seed) | - | not run |  |
| 10 | Null SHUF-vs-SHUF pairs | <= 1 pair beyond 2 SE | - | not run |  |
| 10 | R7(e) Monte Carlo SE | not applicable: no Monte Carlo actor class in this gate | - | n/a |  |
| 11 | Tier B co-trained confirmation | NS(OR-wrong) - NS(EVEN) > 0 and >= 0.5 x Tier A's mean, in >= 70% of seeds | - (lean path: Tier A only) | not run |  |

| Step | Status |
|---|---|
| 1 build tests | pass |
| 2 Stage 0 | pass |
| 3 R0 state richness | pass |
| 4 R1 thinness | pass |
| 5 R2 learned errors | FAIL |
| 6 R3 errors cost return | FAIL |
| 7 R6 policy shift | not run |
| 8 R5 world gates | not run |
| 9 R4 Tier A | not run |
| 10 R7 measurement validity | not run |
| 11 R4 Tier B | not run |

Every number is from a pre-registered stage of this digest. Seeds: a mark 'in >= f of 10 seeds' needs >= ceil(f n) of the path's n seeds. 'not run' means the stage output is missing.
