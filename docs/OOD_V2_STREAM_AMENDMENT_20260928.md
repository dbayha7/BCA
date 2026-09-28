# Approved prospective amendment: one OOD candidate-generation seed

David explicitly approved on September 28, 2026: "Approve the one-seed correction".
This implements the separately reviewed proposal in
`docs/superpowers/plans/2026-09-28-ood-v2-single-stream-amendment-proposal.md`.
It is an amendment after v1 results and before any v2 outcomes, not part of the
original preregistration. The original protocol, proposal, sources, stream map
and failed collision receipt remain unchanged.

Only `(td3_bc, walker2d, 202609171, candidate, 54)` changes:
`404735174 -> 404735181`. Index54 is host-collector reset block27, capture100.
The original integer overlapped a declared training evaluation seed. The chosen
replacement is the reviewed first larger unused uint32 value. No model query,
candidate pool or outcome was used to select it. This is an identifier-collision
correction; it does not demonstrate numerical dependence or invalidate v1.

Training seeds, models, data, action-generation rules, support thresholds,
calibration/coverage settings, endpoints, budgets and all other stream integers
remain unchanged. No future collision may be automatically repaired.

Separate `_amended_v1` source files bind this document and the original protocol
by SHA256. The original entrypoints continue to generate the original map.
Full-map and refusal checks are required before use. This amendment alone does
not authorize bypassing the remaining runtime, storage, shared-lock, complete
archive, process, checkpoint or simulator-adapter acceptance gates. No training
retry or scientific outcome execution is part of this amendment implementation.
