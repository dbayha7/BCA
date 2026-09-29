# Saved cluster input and score provenance, September 29

All five scientific executions remain active. This work joins their saved
inputs and advances existing evidence reviews. It does not report a completed
OOD pair, a BCA advantage or a finish-date estimate. No scientific source,
checkpoint, key, candidate, numerical gate, allocation or lease changed.

The four cluster pairs now have a saved input-provenance review with an actual
zero process exit at 11:23:58 UTC (job27069278). It checks the frozen 108-file
training source manifest, original successful training receipts and accepted
1M checkpoints, accepted preparation/result hashes, raw HDF5 identity, exact
native row and episode mapping, training-only support membership, support
reference/calibration selection and existing accepted query receipts. Hopper
joins998895 training rows and1103 held-out rows; each Walker joins300549 and1149.
Each support reference has32768 rows. Hopper has437 calibration rows and each
Walker219. The checkpoint network and GPU calibration statistics were not
recomputed; no training or accepted model query was rerun.

The initial input checker incorrectly widened native training IDs to int64
before hashing. Its eleven synthetic checks missed that native dtype and passed;
the real saved-input check then refused the first Hopper input. The original
checker, fixtures, failure and outputs remain intact. The separate v3 checker
uses the original native int32 training, held-out and normalization-fit IDs.
Three new regressions establish the old refusal, corrected exact-byte acceptance
and refusal of widened bytes. They passed locally and on the cluster. An earlier
submission was rejected by Slurm because its batch file had Windows line endings;
that attempt launched no job. Both failed preparation attempts are preserved.

A separate saved-score check uses only an inactive reviewer cache on its
original node, after binding its actual successful prefix receipt, source,
configuration, file identity and committed checkpoint. It joins every captured
state's bank and score bytes to the whole-pair barrier, accepted training and
checkpoint pins, support bank and original query scalars. Saved width arithmetic
is exact, both frozen components remain bound, native width stays unavailable
and score slots preserve missing slots. Seven new score-corruption tests passed.
Neural predictions are provenance-bound, not recomputed. Dose validity is
checked; a new per-state dose-arithmetic recomputation is not claimed.

Walker171's score review actually exited zero at11:29:18 UTC (job27069327).
It covers248 captured states, eight native-terminal missing states and3472 score
slots, reusing the761600-frame checkpoint without replaying the journal. The
review is saved at `saved-semantic-review-110839Z-v1/202609171` on the cluster.
Its `final-semantic-review.json` closes saved input/score provenance only.
It is not final outcome acceptance, and the old verifier configuration has not
been silently changed to reference it.

Walker172/173's first200000-frame reviewers both completed with actual zero
exits. Their exact boundaries were byte290824775 and290730983. Separate jobs
27069328/27069329 now each consume200000 new frames using the same source,
configuration, cache and str-c152 identity, then run the saved-score check. Their
completion must be established from actual receipts. The local continuation
advanced by200000 new frames from605000 to805000, with an actual zero exit at
11:33:08 UTC. Its new boundary is byte1124160953, with89281 complete
physical records and mirror acknowledgements. Six buffered artifacts belong to
the next partially included call; they are not a missing-live-output finding.
Its old complete prefix,740 trajectories and exact interrupted-trajectory join
remain reused. Refer to the accompanying validation receipt for the latest
completed boundaries at publication.

Original cluster Hopper's terminal-review configuration is staged using regular
saved files and its accepted input receipt. No Hopper archive/ledger reader was
started. Before any scientific SQLite open, require the exact successful worker
exit, absence of its original command and exclusive ownership of the original
cluster lifetime lease. The local active extension is also forbidden to external
readers under every SQLite flag. Framed evidence and reviewer caches are separate.

Next, bind completed semantic receipts through an explicit checked configuration
migration that preserves the prior configuration and exact cache checkpoint.
Continue only new framed bytes. Full worker exits, final native records, all
returns and panels, final source/checkpoint pins and accounting remain required;
the local continuation also needs its original cumulative-ledger reconciliation.
Readable action comparisons follow that final acceptance. All originals and
preparation failures are retained in
`work/standard_bca_noiw_campaign_v1/monitor_20260929T110839Z`.
