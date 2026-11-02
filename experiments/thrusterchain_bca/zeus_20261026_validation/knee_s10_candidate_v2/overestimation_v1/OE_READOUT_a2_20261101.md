# Overestimation comparison, amendment A2 run (2026-11-01, 21:26 to 22:47 EST): the honeypot and the placebo

**Reading: PROMISING / PLACEBO MIXED.** Every unit completed. The nine stall3 training units were reused from the
A1 run; nine no-stall training units ran (4,477 s GPU, about 8 minutes each); twelve evaluations ran (293 s, 670,896
exact transitions). Instrument checks: I1 at most 4e-8 relative, I2 at most 5.7e-7 per component, all within 1e-6.
Record: `oe_run_a2/attempt_1/` (`OE_RESULT.json` sha256 2798dfee…, `decision.json` 218a9e12…), manifest 89cd48e8….

## The two worlds, side by side (20k policies, CUTOFF data, three paired seeds)

| world | arm | J (128 starts) | W exposure | e_own | e_data | e_W | U_data | U_W |
|---|---|---|---|---|---|---|---|---|
| stall3 (honeypot) | native | −363.8 / −358.8 / −359.3 | 0.70 / 0.68 / 0.67 | +186 / +179 / +183 | +24.8 / +23.5 / +23.7 | +25.2 / +24.4 / +24.6 | 2.55 / 2.19 / 2.49 | 2.57 / 2.21 / 2.51 |
| stall3 (honeypot) | BCA | −351.8 / −317.5 / −336.5 | 0.70 / 0.62 / 0.43 | +186 / +163 / +154 | +20.6 / +19.7 / +19.9 | +21.4 / +20.4 / +22.0 | 2.65 / 2.26 / 2.30 | 2.67 / 2.27 / 2.32 |
| no-stall (placebo) | native | −66.9 / −60.3 / −61.9 | 0.79 / 0.80 / 0.85 | +10.7 / +10.2 / +3.8 | +2.6 / +1.2 / +1.3 | +2.5 / +1.1 / +1.2 | 2.25 / 1.97 / 1.92 | 2.27 / 1.98 / 1.94 |
| no-stall (placebo) | BCA | −67.2 / −56.9 / −68.6 | 0.53 / 0.73 / 0.53 | +12.8 / +4.1 / +6.0 | +2.1 / −0.6 / −0.2 | +2.0 / −0.8 / −0.3 | 2.64 / 2.51 / 2.56 | 2.66 / 2.53 / 2.59 |

Paired, BCA minus native:

| world | seed | dJ | de_own | dEXPO |
|---|---|---|---|---|
| stall3 | 0 / 1 / 2 | +11.9 / +41.3 / +22.8 | −0.6 / −16.2 / −28.9 | +0.004 / −0.062 / −0.239 |
| no-stall | 0 / 1 / 2 | −0.3 / +3.4 / −6.7 | +2.1 / −6.1 / +2.2 | −0.252 / −0.078 / −0.320 |

Honeypot: PROMISING (both bars in every seed). Placebo: not CALIBRATED (seed 2 loses 6.7), not OVER-CORRECTION
(only one seed loses 2.0 or more), so PLACEBO MIXED.

## What the two worlds together say

1. **The native actor behaves the same in both worlds and only the physics differs.** In both it saturates
   thruster 0 at −1 for most steps (median own a0 −0.98 in no-stall, −0.99 in stall3). In the no-stall world that
   is a good policy: returns about −62, own-action over-valuation about +4 to +11, no W over-valuation at dataset
   states (e_W − e_data about −0.1). In the stall world the same command stalls the thruster, returns fall to about
   −360, and the critic, trained on data that withhold W, carries the no-stall-like extrapolation across the knee:
   +180 of over-valuation on its own actions. The 300-unit gap between the worlds is the honeypot, and the native
   learner cannot see it because the data cannot tell it.
2. **BCA's pull out of W is world-blind, as a sign-blind width predicts.** W exposure falls by 0.00 / 0.06 / 0.24
   in stall3 and by 0.25 / 0.08 / 0.32 in no-stall: at least as much where W is harmless. The width does not know
   which world it is in (U_W − U_data is +0.01 to +0.03 in both), and kappa·U is a near-constant offset of about
   120 to 165 on the actor's score in both (p_data).
3. **The trade is nevertheless favorable here.** Where W is a trap, leaving it earns +12 / +41 / +23. Where W is
   harmless, leaving it costs −0.3 / +3.4 / −6.7: small, mixed, with one seed over the 2.0 bar. The placebo has
   real stake (the native actor uses W for 80 to 85 percent of its steps there and does well), so the small cost is
   a finding, not an absence of test: a 25-to-32-point drop in W use cost at most 6.7 return units, because in the
   no-stall world the actions BCA retreats to (median own a0 −0.84 in seed 2) are nearly as good as saturation.
4. **The underestimation signature is present and mild.** In no-stall, BCA's critic slightly under-values the
   logged in-support actions in two seeds (e_data −0.6, −0.2 against native's +1.2, +1.3), and BCA's own-action
   error rises in two seeds (+2.1, +2.2) because its actions move toward regions the critic values less
   accurately. This is the direction the sign-blind penalty must push; its size is a few units against the
   honeypot's hundreds.

## What this does not say

Three seeds, one dose, one host budget, one dataset pair. The honeypot effect is large and sign-consistent; the
placebo cost is small and inconsistent; neither is a confirmed result, and the mechanism (the small
state-dependence of U versus the penalty's gradient through the action) is untested. Both learners remain far
below the behavior quality in the stall world. No protocol claim; nothing about the knee departure.

## Ledger entry

Overestimation comparison v1, A2 run: PROMISING / PLACEBO MIXED. Honeypot dJ +11.9, +41.3, +22.8 and de_own −0.6,
−16.2, −28.9; placebo dJ −0.3, +3.4, −6.7 with dEXPO −0.25, −0.08, −0.32 (native uses W 0.79 to 0.85 there);
native stall3 over-valuation about +180 against +4 to +11 in no-stall; U sign- and world-blind. GPU 1.24 h new;
670,896 exact transitions.
