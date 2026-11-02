# Experimental eager CQL candidate

This isolated adapter reuses the native Polyak conditional branches while
retaining native eager CQL updates, sampling, Adam, losses, and finite-state
commit behavior. Its checkpoint metadata binds the candidate source hash and
native update source hash, package versions, JAX arithmetic configuration,
device/backend details, and compiler/runtime environment settings. This
candidate rejects native v6 checkpoints and candidate checkpoints carrying a
different candidate or runtime identity. The native
adapter shares the checkpoint adapter ID and does not enforce the additional
candidate identity, so a study-level verifier must prevent cross-variant resume.

The candidate is not integrated into the full-study launcher. CPU checks do
not establish GPU parity or practical million-update runtime; GPU validation
and matched full-study checks remain pending. Do not launch a full benchmark
with it until those checks have completed.
