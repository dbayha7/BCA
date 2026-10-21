# Unifloral hosts v4 — completed October 20, 2026

Every approved v4 run finished. The driver completed at 9:04:09 p.m. America/New_York. The approved supplement export completed afterward. No training or evaluation process remained active at the completion check.

- Training: 118/118 unique completed records (20 stage-1 checks, 98 stage-2 runs).
- Evaluation: 416/416 planned lanes (20 stage-1 precision checks; 232 medium-expert and 164 medium stage-2 lanes).
- Medium-expert: 56/56 stage-2 runs; medium: 42/42.
- Export: 98/98 planned stage-2 records, none outside the plan, none excluded, unscored, or halted.
- Version 3's judged results are canonically unchanged. All 41 available cross-version reproduction comparisons matched; the other 77 runs had no version-3 record to compare.
- Frozen code and terminal approval fingerprint: d3538fd0b39698361e3318666e28d8316c37df9175769d5a2cb54225e5e2f71a.

## Descriptive five-seed results

Mean normalized-point differences on the strong overlapping-current evaluation bank, at the frozen final checkpoint of 100,000 outer steps. Each row uses seeds 0–4. Positive values mean higher score. These summarize the supplement, without new statistical judgments or checkpoint selection.

| Data quality | Learner | Standard BCA minus alone | IW BCA minus standard BCA | IW BCA minus alone |
|---|---|---:|---:|---:|
| medium-expert | TD3+BC | +0.515 | +0.203 | +0.717 |
| medium-expert | IQL | +0.654 | +0.198 | +0.853 |
| medium-expert | ReBRAC | -0.146 | +0.104 | -0.042 |
| medium | TD3+BC | -4.748 | -2.684 | -7.432 |
| medium | IQL | +0.472 | -0.049 | +0.424 |

The positive medium-expert averages for TD3+BC and IQL did not carry over uniformly to medium data. On medium, TD3+BC lost performance with both additions in all five seeds. IQL improved on average, with gains in three of five seeds. ReBRAC's medium-expert IW result remained slightly below its baseline. This table covers the supplemented scopes; CQL and other scopes remain in the unchanged version-3 portion of the complete export.

## Precision limitations and integrity

This is a post-hoc, exploratory supplement. S7 failures were recorded and did not drop any run. Displayed scores use the float64 scoring adapter and may differ from the learner's own float32 policy by more than 0.05 points per start. The supplement also found S7 failures on medium-expert TD3+BC standard-BCA final actors for seeds 3 and 4, beyond version 3's narrower precision checks. These are explicit caveats on the carried judgments; no version-3 judgment was rewritten.

The export reports S1–S4, S6, S6b, S8, S9 and S11 passed; S7 is a recorded outcome, and S10 is a readout under the frozen rule. S0 passed in its separate exact-code suite verification. S5 has no recorded stop. The earlier S6 runtime stop and approved resume are documented in operations/S6_RECOST_20261020.md.

## Compute

| Quality | v4 GPU-hours | v4 CPU-hours | Approved GPU / CPU cap |
|---|---:|---:|---:|
| medium-expert | 2.944 | 6.401 | 14 / 21 |
| medium | 1.216 | 2.604 | 12 / 16 |
| Total | 4.160 | 9.004 | Separate quality caps above |

These are v4 ledger totals, including the operational resume; stage 0 is booked separately. The full export also discloses earlier-version ledgers.

## Saved results

The complete export is ../shift_hosts/results.json with its .sha256 sidecar, copied byte-for-byte from /root/bca-hosts/outputs/shift_hosts/results.json. Its SHA-256 is aee3bbce94281df5b8c6bb84a9c201229216d493bc6cd8d5ff5da41d5efc2dc4. Completion checks are saved in operations/COMPLETION_VERIFICATION_20261020.json. Files are saved locally; this completion step did not commit or push them.