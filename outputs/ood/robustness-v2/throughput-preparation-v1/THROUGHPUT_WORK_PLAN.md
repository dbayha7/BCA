# Bounded OOD storage and parallel allocation implementation plan

**Goal:** Measure the logging bottleneck and prepare independent, already accepted pairs without changing either running worker or scientific protocol.

**Architecture:** One small storage-only benchmark compares the current SQLite transaction pattern with framed sequential writes, including a synchronously replicated node-local variant. A separate saved-data allocation document binds accepted pairs and identifies the accounting and exclusivity checks required before additional dispatch. No general observer framework or new simulator is involved.

**Tech stack:** Python standard library, SQLite FULL/DELETE, fsync, SHA256, scheduled CPU Slurm allocation, existing saved training acceptances.

Execution uses the writing-plans workflow in this session. The user's no-subagent and no-repeated-approval instructions override the skill's delegation and handoff suggestions. Running sources and closed tests remain untouched.

- [x] Create `storage_benchmark.py`: identical seven synthetic evidence payloads and two accounting records per fake call; compare nine SQLite transactions against four ordered append synchronization boundaries. Reserve before the synthetic action marker, flush input/control before it, flush output/full state before completion. No scientific callback, key, model, or simulator.
- [x] Execute a bounded local test and a separate one-CPU cluster job. Measure shared storage, node-local storage and node-local plus synchronous shared replica. Retain files and actual exits. Node-local-only speed is diagnostic, never execution acceptance.
- [x] Exercise deliberate child exit 23 after reservation, before action, after output and after completion; verify charged/pending disposition from saved bytes. Include incomplete-frame detection with no repair or truncation. This tests process death, not power loss or node-loss guarantees.
- [x] Review saved benchmark outputs independently and select only a measured candidate for future integration. Report storage-only rates and bounds, not an OOD completion promise.
- [x] Write a disjoint parallel allocation proposal for the five unstarted accepted ReBRAC pairs. Bind exact training acceptance hashes, protect the live pair, and account for the shared engineering budget. Do not claim allocation/dispatch acceptance while original engineering commitments remain unverified.
- [x] Execute CPU checkpoint query gates for all five accepted unstarted pairs and independently inspect saved arrays and actual exits; no simulator or new OOD keys.
- [x] Prepare the concrete saved-outcome verification schedule and streaming joins so full-state, reward/control, key, and reservation validation can run incrementally after immutable completed work, with final acceptance only after actual worker exit. Implementation and full execution remain pending.
- [ ] Preserve receipts, publish small owned evidence only, and update the ACTIVE monitor without dropping concurrent direction.
