# G6 ledger: every candidate screened under OOD definition v4

Required by DRAFT_DEFINITION.md (G6): every candidate screened, its sigma, mode and outcome, with any later efficacy
claim. Development screens run on development entropy with outcomes known; they are seen data and never confirmatory.
A formal G2 runs once on fresh sealed streams under a rule written first.

| Candidate | sigma | mode | Development (seen) | Formal G2 | Efficacy claim |
|---|---|---|---|---|---|
| knee_s10_v1 (`knee_s10_candidate_v1/`) | s10 | exact, pi_ref v2 (LQR, support-projected) | G2 v1 INCONCLUSIVE (item 5 at 43 of 48); G2 r3 PASS (development entropy) | attempt 1, 2026-11-01 05:19 UTC, approval `5742605a`, manifest 0ba3e4b5...: **NOT DATA-OOD** (no-stall D3 anchors 84 of 96, bar 87; all else passed; H2 median rho(s10) 1.536 vs 0.779). Readout `formal_g2/G2_FORMAL_READOUT_20261101.md`. | none; candidate stopped by its rule |
| knee_s10_v2 (`knee_s10_candidate_v2/`: v1 with the in-support anchor prefix, `prefix_rule.py`) | s10 | exact, pi_ref v2 (unchanged) | development G2 v2, 2026-11-01 05:49 UTC, rule `G2_DEV_RULE_v2.md` (sha 4e9242e4...), v1's development entropy and stream, 48 gated anchors: **PASS** (D3 anchors 48 of 48 in both worlds, v1 had 44 of 48 in no-stall; H1 0.0; H2 median rho(s10) 2.198 vs 0.779; P1 every prefix identical in both worlds). Record `g2_development_result_v2.json`. | not yet sealed; the rule's PASS leads to a formal G2 for v2 (new namespace, fresh streams, 96 anchors, the user's approval) | none |
