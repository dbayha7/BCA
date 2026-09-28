# Saved transition joins in progress; scientific acceptance pending

The new joined_actions.py component consumes bounded archive envelopes in exact
input/applied/output order and joins them to independently audited reservation
ranges. It checks original and repeated reset states, precommitted first actions,
all available controls and reward arithmetic, full saved state schemas and
unchanged model/RNG fields, keys, repeated-first contents and every reward in a
completed return. Both interrupted partials remain excluded, and the one
input-only historical call remains unknown and charged. Replay prefix records
and saved inputs are checked against the separately pinned recovery plans.

Sixteen synthetic tests passed with actual exit0 and no skips. They exercise
missing/swapped/extra records, wrong reset/candidate/key/shape/control/reward,
partial ownership, input-only uncertainty and changed replay prefixes. The
initial expected missing-module failure and earlier14-test pass remain saved.
Existing closed component sources/tests remain unchanged and were not rerun.

The new real integration reads the three saved archives with mode=ro/query_only
and uses the previously accepted zero-header journal binding. It has been
started once, separately from the now-closed scientific collection. It is a
saved-only audit process: no scientific worker, model query, simulator, shared
worker lock, database repair or ledger writer is used. Its actual exit and full
result are pending. Do not start a duplicate or edit its pinned source while it
runs. Monitor directory: monitor_20260928T164055Z. audit_dispatch.json and
join_wrapper_dispatch.json identify the process; join_progress.json is only
partial progress. real_joined_saved_v1_actual_exit.json exists only on closure.

Final full-state contents generally were not retained by the collector. For a
completed action, the first/repeated-first and intermediate following states can
be checked, but the last full state cannot be invented from its saved hash. The
join explicitly counts unavailable final following-state checks. The first
interrupted partial likewise has no saved following state for its last output;
the second partial has a saved input-only state that can check the preceding
output, while its own historical physical completion remains unknown.

This component/integration is not full scientific acceptance. Constructor
receipts and final independent review of source/checkpoint/bank bindings and
evidence coverage remain separate. No harm/AUROC/regret claim is made by a
partial join or synthetic tests. The real report adapter and separately tested
secondary regret remain pending. Primary BCA continuation, within-state versus
pooled AUROC, strict harm>1, shared-reset/candidate dependence and all five seeds
remain unchanged; this first seed is exploratory. Coverage/global readiness
remain false. No weights, SQLite or NPZ files are included in this package.
