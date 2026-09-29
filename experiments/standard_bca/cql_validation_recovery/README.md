# CQL validation correction: untouched 137-row continuation

This is an execution amendment to the September 27 standard/no-IW BCA manifest.
Every one of the 137 remaining scientific rows is byte-for-byte equivalent as
parsed JSON to the original local lane after its first three rows. Only
`runtime/validation.py` and its `train.py` call site differ in the source inventory.
Decompress `manifest.json.gz`; the uncompressed SHA256 is
`9ecf480c560871832ec6384045e5646de3238de8697bb56d6ee9dec85172b3cf`.
Use its explicit 108-file source hashes; do not substitute current moving HEAD.

The frozen source is at
`/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v1/source`.
Slurm **27068516** runs the existing supervisor/standard runner in the new root,
with one exclusive lock on its separately allocated A100. ReBRAC/IQL job
27045057 retains the other allocated GPU. Total GPU cap remains two.

`supervise.py` first validates all 14 TD3/CQL dataset preparations for both
methods against the already accepted fingerprints. Four small GPU fixtures
then exercise host/BCA updates, BCA refresh and detached checkpoint decoding:
12 synthetic updates total, zero simulator steps. These are execution tests,
not experimental runs or replacements. Failure stops the job and retains actual
exits. Only successful gates allow the existing controller to start the first
unstarted row, `cql-hopper-bca-noiw-s202609171`.

The manifest lane label `local` is retained to preserve its original rows;
physical execution is now on the cluster. `ownership.json` records the move.
The old local output remains untouched. The staged local continuation has
`delegated-to-cluster.json` and a launch claim, with no local worker launched.
Never launch a second controller for these same rows.

The recovery excludes two completed TD3 runs and the saved CQL host recovered
without retraining. Original CQL process exit 1 remains; see the
[saved-result audit](../../../docs/validation/standard-cql-host-saved-recovery.json).
Its first host/BCA pair crosses RTX/A100 hardware, which must remain explicit
in downstream comparisons. Configuration equality is not cross-device bitwise
equivalence. Remaining four seeds for that pair use A100 for both methods.

`execution_freeze.json` hashes files using their deployment paths. Packaged
`manifest.json.gz` restores `source/manifest.json`; the two changed source files
are the repository's corrected `train.py` and `runtime/validation.py`, identified
by the manifest hashes. All original checkpoint weights stay on their hosts.

Read actual `training_slurm_actual_exit.json`, `compute_acceptance.json`,
`controller-process-v1/actual_exit.json`, `queue-v1/queue_status.json` and each
attempt's `actual_exit.json` before reporting progress/completion. Submission or
training closure alone does not establish a verified OOD pair.
