# Exact numeric environment contract implementation plan

**Goal:** Close the observed Python/NumPy/JAX environment mismatch with an explicitly versioned, narrowly scoped contract and a real import-only CPU process.

**Architecture:** The new contract supplies all eight already observed fields at launch and checks both the kernel's original process environment and Python's effective environment at each fixed import phase. It never merges inherited variables or removes fields. Existing scientific capsule/supervisor sources remain frozen; this numeric profile does not accept native simulator imports or authorize scientific dispatch. No old key generator or review runs again.

**Tech stack:** stdlib contract/observer/reviewer and pinned Python3.10, NumPy2.2.6, JAX0.6.2 for the single import-only probe.

- [ ] Write explicit eight-field policy and guard; exercise missing/extra/changed values, invalid stage transitions, process change and persistent refusal with new focused tests.
- [ ] Record current installed JAX environment setters and MuJoCo cached-extension/import requirements through source reads only.
- [ ] Observe one fresh import-only process under the exact policy, retaining parent observations, effective environments, source/runtime hashes, stdout/stderr and actual exit.
- [ ] Independently review saved policy, phase observations and actual exit; accept only Python/NumPy/JAX CPU import compatibility if they match. Keep failed original producer1 and key execution hold.
- [ ] Publish small exact-byte evidence, preserve prior artifacts and update the active monitor quietly.

User instructions prohibit subagents and routine closed reruns. Native imports, complete runtime graph, scientific capsule integration, storage/lease/physical routes and saved-key acceptance disposition remain separate required work.
