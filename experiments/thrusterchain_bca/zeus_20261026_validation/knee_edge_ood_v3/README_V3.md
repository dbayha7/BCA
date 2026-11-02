# knee_edge_ood_v3: the v2 package with the audit's repairs R1-R7 (PROTOCOL_V3.md); not approved for any run

`README.md` is the carried v1/v2 README, byte-identical because the A1 manifest (`MANIFEST.json`) hashes it. What v3
changes, and why, is in `PROTOCOL_V3.md`; the accounting correction for v2's step 0.4 attempt is in
`ACCOUNTING_CORRECTION_v2_A3_attempt1.json`; the v3 tests are `test_v3_repairs.py` plus the carried suite;
`amended_analysis.py` re-reads an existing Stage 1 attempt under the v3 rules without changing it.

Gating status: A4 still requires a step 0.4 PASS and the only step 0.4 record (v2's, carried and pinned) is
INCONCLUSIVE, so no v3 run can be requested as the package stands. The GPU smoke of v3 has not been run.
