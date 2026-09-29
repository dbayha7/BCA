# Current execution: job 27068529

The first cluster preflight job27068516 stopped with actual exit1 before any
model initialization, fixture or scientific training. Seven TD3 dataset cells
passed. The wrapper then tried to call TD3/ReBRAC's `validate_prepared` helper
on CQL, which has a different native interface. The original source, log,
seven acceptances and actual exits remain in [attempt1_receipts](../attempt1_receipts/).

This separate v2 uses CQL's actual pre-initialization checks: protocol,
metadata/settings/source identity, training/heldout/reference/normalization
hashes and exact evaluation-bank identity. It uses the original CQL preparation
fingerprint (a digest of named array hashes), not the TD3 tree fingerprint.
Seven TD3 cell acceptances are hash-pinned and reused without replaying them.
Only the seven CQL cells are prepared again. These operations perform no
learner updates, model queries or simulator steps.

The manifest and all 108 scientific/execution source files remain identical to
the first recovery snapshot. The same 137 unstarted rows are owned exclusively
by job **27068529**, deployed at
`/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v2`.
The successful data gate is followed by the already declared four GPU fixtures
and then CQL Hopper BCA seed202609171 training. Any failure stops the pipeline.
No completed experiment, training seed or OOD outcome is retried.

The first v2 staging script had a generated-Python newline syntax error before
any remote writes or submission. Its source/exit/log are retained. Version2b
corrected that serialization only, then submitted this job once.

Hardware qualifications, two-GPU cap, disabled local launch, original CQL
worker exit1 and saved-result audit remain as in the [parent declaration](../README.md).
The old v1 job must not be reported as current. Actual process receipts in this
v2 root govern completion; Slurm submission alone is not accepted training.
