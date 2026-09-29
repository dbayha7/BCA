# BCA advisor update

**Live snapshot: September 28, 2026, 11:46 p.m. EDT / September 29, 03:46 UTC.**

[Open the 20-slide PowerPoint](BCA_Advisor_Update_2026-09-29_Reviewed.pptx) · [Speaker notes and source references](SPEAKER_NOTES.md)

## Current experiment status

- Cluster training: 64/140 physical runs have successful worker and learner exit receipts, split 32 ReBRAC and 32 IQL. The live worker was third-seed ReBRAC HalfCheetah host at 960k. Controller and worker identities matched. Their final actual exits were still pending.
- Local training: two TD3+BC Hopper runs have successful closures. The CQL lane remains stopped after its final evaluation on a result-schema error. Its exit 1 remains preserved.
- Independent scientific acceptance: 14/280 physical runs, forming seven pairs. ReBRAC Hopper and Walker each have 3/5 accepted paired seeds. TD3+BC Hopper has 1/5. Fifty-two cluster closures still await full independent audits.
- OOD outcomes: one completed TD3+BC Hopper training seed. The prospective v2 study has no new outcomes yet because actual runtime and simulator integration remain incomplete.
- Weighted-conformal correction: mathematical reference and synthetic validation complete. Real-data sampling, justified ratios and four-host training integration remain incomplete.

## What the results mean

The first Hopper OOD panel gives BCA width a within-state harm AUROC of 0.529, versus 0.566 for support distance, 0.513 for fixed random scores and 0.500 for a constant. This provides little demonstrated harm separation in one seed. It is not a test of a conformal coverage theorem.

The post-hoc distant-action bridge gives BCA-minus-host continuation return -5.046 and a degradation advantage of -0.019 against the common anchor. It does not demonstrate an action-stress benefit. Those rows are post hoc and do not substitute for prospective v2 outcomes.

Verified training differences vary: ReBRAC Hopper is nearly tied, while Walker final differences are +4.98, -2.99 and +33.64. Every seed and episode remains included. The slide table and CSV distinguish final means from averages across periodic evaluations.

In a separate known-ratio scalar simulation, ordinary coverage under a constructed shift is 61.05%, corrected weighted conformal is 90.70%, and the Bayesian maximum is 93.95%, against nominal 90%. That validates the stated arithmetic in the toy setting, not RL coverage or practical OOD benefit.

## Advisor questions covered

1. Why the OOD study adds evidence beyond return curves.
2. What training, calibration and simulator evaluation each do.
3. Where BCA enters each host, and what the calibrator does not know.
4. What good or bad warning, coverage and behavioral results imply.
5. Why support-balanced action panels, a common anchor and independent seeds are needed.
6. What was missing from the earlier fitting-IW path, why weighted thresholds and query mass matter, and which integration gates remain.

Slides 1–6 explain objective, setup and status. Slides 7–11 show results. Slides 12–18 explain changes, interpretation and next steps. Slides 19–20 are calibration and arithmetic appendices. Sources and detailed caveats are in the speaker notes.

## Immediate critical path

Complete actual runtime/simulator gates for the v2 OOD worker, then collect the declared comparisons. Document the CQL saved-result validation recovery without rerunning completed training. Continue the original cluster queue and independent saved-result audits. Keep coverage unchanged and test corrected IW separately under a justified sampling contract.

## Evidence and reproduction

`data.json` retains all 14 accepted learning series at 200 periodic banks, derived three-seed descriptive means, all 28 earlier two-seed cluster comparisons, exact synthetic results and source hashes. Chart workbook values round to eight decimal places for portability; the source numbers remain intact. The preliminary cluster table is from the saved September 28 22:37 UTC extraction, with newer audit status explicitly distinguished. The live count uses the new small-metadata snapshot.

This update adds no training, model queries, simulator steps, new outcomes, or queue changes. Original scientific evidence and native inventories remain untouched.
