# Results of 13 October 2026: rivals, trace sweep, deployment check and Stage 2 detection

Copies of the result files from `runs/shift_explore/` (gitignored), with each source's SHA-256 sidecar kept beside it
as `*.source.sha256` and the copies' own hashes in `SHA256SUMS`. Everything here is on gate seeds 0-4 and is
**exploratory**: no claim can be made from it. Only fresh seeds 5-9 can carry Stage 2's claim.

## Data-only rival methods in the lower-edge slot (no BCA) — `standins_marks.json`

Points won back in strong current over no caution (stalling world), and points lost in strong current in the no-stall
world, where the missing commands are good:

| Signal | Won back | Lost (no-stall) | Captures the room |
|---|---|---|---|
| perfect signal (exact error; Stage 1) | 93.9 | 1.3 | — |
| range hinge | 84.6 | 20.4 | yes |
| classifier ratio | 36.8 | 11.0 | no |
| distance to data | 9.3 | 5.5 | no |
| behaviour density | 4.8 | 7.3 | no |
| two-head disagreement | 1.2 | 4.0 | no |

Stage 2's `select_standin` picked the range hinge as BCA's rival before any BCA run (`stage2_best_standin.json`).

## Trace test bed (harm claim; no BCA) — `trace_pooled_v3.json`

Seeds 0-2. Shares 1%, 3%, 5% and 10% were added after the 0.5% and 2% results (post hoc). No share passes:

| Share of full-throttle rows kept | 0% | 0.5% | 1% | 2% | 3% | 5% | 10% |
|---|---|---|---|---|---|---|---|
| harm in strong current (points; mark 50) | 99.5 | 96.5 | 92.6 | 85.8 | 74.9 | 71.5 | 23.4 |
| residual-evidence separation (mark 0.2) | 0.02 | 0.08 | 0.07 | 0.07 | −0.01 | 0.05 | −0.05 |

The harm claim needs a different test bed.

## Deployment-check channel (no BCA) — `deploy_summary.json`, `deploy_tables.md`

The channel is open: a perfect detector of past-80% proposals, clipping to the recorded range, beats random switching
at its own rate by 23.2 ± 1.0 points in strong current, and gains 7.2 in recorded conditions. The classifier ratio
and two-head disagreement clear the bars with the clip fallback; behaviour density does not.

## Stage 2 detection (BCA's band, pre-registration v6 `9b38c7c4…`)

| Mark | Pass mark | Uniform version | Weighted version |
|---|---|---|---|
| direction, training states / strong current | ≥ 0.75 / ≥ 0.65, and 4 of 5 seeds | 0.50 / 0.73 (1 and 2 seeds): fail | 1.00 / 1.00 (5 and 5 seeds): pass |
| size (paired penalty ÷ false preference) | ≥ 1, and 4 of 5 seeds | 0.0004 (0 seeds): fail | 1.46 (4 seeds): pass |
| ranking, training states / strong current | ≥ 0.75 / ≥ 0.70 | 0.80 / 0.62: fail | 0.99 / 0.86: pass |
| calibration (share above tau) | 5% ± 1.5% | 4.9%: pass | 4.9%: pass |
| within 0.02 of the best data-only signal | both sets | 0.80 vs 0.99; 0.62 vs 0.90: fail | 0.989 vs 0.990; 0.86 vs 0.90: fail on strong current |

- The uniform version failed direction and size, so the pre-registered weighted version was declared and run.
- The weighted width adds a term built from the same classifier as the classifier-ratio rival
  (`U_max · softplus(log w − l*)`, pre-registration section 4). It ranks like that classifier on training states and
  a little worse in strong current, so its detection mostly reflects the classifier.
- The claim rule is not met (and seeds 0-4 cannot carry a claim). Because a version passed direction and size, the
  policy test, deployment check and full host started on seeds 0-4 at 13:56 UTC, as the pre-registration orders.
