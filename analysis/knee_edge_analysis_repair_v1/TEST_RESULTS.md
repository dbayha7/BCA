# Verification receipt

All scientific model updates, calibration updates and simulator transitions for this work: **zero**.

| Check | Actual result |
| --- | --- |
| Independent math/source review | Float32 coordinates, predictor arithmetic, item-5 masks, one-sided V0(f) and decomposition signs match the written/source definitions. |
| Focused math suite | 24 tests passed; explicit synthetic fixtures only. |
| Full isolated suite after final fixes | 45 tests passed, exit 0, 1.165 seconds unittest-reported runtime. |
| Real CLI against approved attempt | Exit 0; `waiting_for_terminal_saved_arrays`; `empirical: false`; `protocol_decision: null`; `can_unlock_stage2: false`. |

Full-suite command:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s analysis/knee_edge_analysis_repair_v1 -p 'test_*.py' -v
```

The independent review identified and the isolated implementation fixed: missing no-stall receipt validation, unrestricted input reads, malformed JSON mapping handling, and unenforced declared lineage hashes. The attempt, approval and prior audit hashes are now checked before and after analysis, alongside source hashes. The producer-fingerprint claim was narrowed explicitly: recorded source hashes are checked, but the AST fingerprint is not independently authenticated. No live source was changed.

An earlier adapter-only run had 18 successes and one fixture setup error: the sandbox denied renaming a temporary synthetic report file. No experiment file was involved. That denied move was not retried. The missing-file test now creates a fresh empty fixture attempt, avoiding any move; it passed in both subsequent suites (43 tests, then 45 after the lineage review). There is no remaining access blocker for this pure analysis. WSL execution remains denied and was not attempted.

Test fixtures use manufactured finite 64-by-14 arrays and temporary JSON reports. Their numerical passes are software checks only, not empirical evidence. The target GPU, scientific policies and simulator were not executed by these tests.
