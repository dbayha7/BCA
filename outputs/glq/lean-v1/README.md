# Gapped LQ acceptance gate, lean path: the world is rejected

Run 2026-10-11 on a 4-CPU cloud container. Pre-registration `experiments/signal/glq_preregistration.json`,
SHA-256 `255369889b400c2b…`. Seeds 0–4 (the lean path). No BCA arm and no BCA band fit were run.

## Verdict

**The Gapped LQ world fails its own acceptance gate, so BCA is not run in it.**

- Steps 1–4 pass in the base world and in the retune. The build checks, Stage 0, state richness (R0) and thin data
  (R1) all pass.
- Step 6 (R3, "errors cost return") fails at every alpha on the pre-registered ladder (2.5, 5, 10).
  - Base world: 0, 1 and 0 of 5 seeds pass; 4 are needed.
  - The one usable retune (`thin_share`, each thin context 0.15 → 0.25 of episodes): 0, 3 and 3 of 5 seeds pass.
- The second pre-registered retune (`wrong_gain`, T_wrong's actuator-3-4 gain × 1.5) fails Stage 0. The transplanted
  policy is unstable (√γ·ρ = 1.0023), so it cannot be run. No retune is left, and under the stop rule the world is
  rejected.

Files: `verdict/base.md`, `verdict/thin_share.md` (the gate's own tables), `r3/`, `stage0/wrong_gain.json`.

## Why R3 fails

Three of R3's four marks pass at alpha 10 in both worlds:
- the critic's error costs return in T_wrong: 0.07–0.12 of the gap (base) and 0.11–0.32 (retune), against a mark
  of 0.05;
- the host trails the EXACT arm by 0.33–0.43 (base) and 0.48–0.72 (retune) of the gap, against 0.10;
- the host is useful where data is rich: rich NS 0.49–0.60 and 0.46–0.59, against 0.30.

The mark that fails is where the loss sits. At least 60% of the host's shortfall against EXACT must be in the thin
contexts; it is 0.40–0.60 (base) and 0.56–0.74 (retune) at alpha 10. The learned critic is also wrong in the rich
contexts and in the noisy decoy, so the host loses return there too.

The critic's errors point the way the design predicted. Using actuators 3–4 costs return in T_wrong (reversed) and
gains return in T_right (correct). T_wrong pays in all 15 seed × alpha cells of each world. T_right gains in 14 of
15 base cells (seed 3 at alpha 5 is the exception) and in 15 of 15 retune cells.

## R2, reported for information (step 5; incomplete because the density ladder was not run)

At alpha 10 the HOST critic fails all three main R2 marks in every seed, in both worlds:
- E in T_wrong is 1.0–1.6× the rich contexts' (mark ≥ 3×);
- the thin contexts carry 13–37% of ΣE² (mark ≥ 60%);
- the rich-context median |e1| / |Q^π| is 0.46–0.57 (mark ≤ 0.05).

The FQE critic shows the designed pattern on the first two marks. E in T_wrong is 3.4–6.2× the rich contexts', and the
thin contexts carry 38–80% of ΣE². But its relative error is 0.75–1.15.

The relative-error mark is unreachable for both critics here. Most of the error is a level offset, the critic sitting
well below the truth everywhere. That fits clipped double-Q's underestimation compounded over a long horizon
(γ = 0.98). A level offset does not change the actor's gradient. This was seen after the outcomes, so changing the mark
would need a new pre-registration.

## Room to win: R4 on the base world, alpha 10, INFORMATION ONLY

The gate never reached step 9. Tier A was run on the base world's alpha-10 HOST critics with an explicit
`--alpha 10`, which the gate allows only for information. The gate does not count these numbers.

| Arm (matched strength S1 = EVEN at m = 1.3) | NS minus EVEN, 5 seeds | Pre-registered mark |
|---|---|---|
| OR-wrong (caution exactly in T_wrong) | +0.060 (0.049–0.073), t = 12.4, p = 0.0001 | ≥ 0.05, p < 0.01 |
| OR-thin (caution in both thin contexts) | +0.045 (0.039–0.057), p = 0.0001 | ≥ 0.03, p < 0.05 |
| best OR-thin − best EVEN | +0.088 (0.057–0.131) | ≥ 0.02 |
| best OR-wrong − NONE | +0.103 | ≥ 0.03 |
| OR-thin − SHUF / SHUF − ANTI | +0.045 / +0.028 | each ≥ 0.05 (fails) |

At the stronger level S2 (EVEN at m = 2) the margins grow: OR-wrong +0.147 (4 of 5 seeds matched) and OR-thin +0.111.
The shuffled controls sit at 0.000 and ANTI at −0.067.

- **Room to win exists here,** unlike the 3-state toy world, where perfect placement added −1.6 to +0.04 points.
- **Row-by-row error placement captures little of it.** OR-err (rank-normal of the exact |∇ₐe1|) gains +0.016 at S2
  and OR-D (the advantage error) +0.024. That is a seventh to a fifth of OR-thin's +0.111.
- **BCA's dose range limits it.** Capped at [1, 1.5), the oracle cannot reach the matched strength at all. Best
  against best, it beats EVEN by +0.043.

Files: `r4_information_only/base_alpha10.json`, `tierA_information_only/`.

## What ran

- 15 HOST/FQE/EXACT critics per world at alpha 2.5, plus 10 HOST/EXACT critics per world at each of alpha 5 and 10.
  That is 70 runs of 100k updates, about 4.5 minutes each on 2 CPUs.
- Tier A on the base world at alpha 10: 5 seeds, about 21 minutes each.
- Not run: the density-ladder critics, R5 world gates, R6, the null world and the global-linear class.
- Matching hit its 5-round cap with 4–8 unmatched pairs per seed.
