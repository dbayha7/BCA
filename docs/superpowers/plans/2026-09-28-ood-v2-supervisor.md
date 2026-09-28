# One-shot engineering supervisor and worker route implementation plan

**Goal:** Observe one actual child process using the closed engineering capsule,
and make its cooperating engineering callback wait for durable launch evidence.

**Architecture:** Add separate supervisor and worker-route modules. Preserve every
closed source. The supervisor consumes the capsule, exclusively claims a fixed
evidence directory, saves intent, starts the exact isolated command with the exact
CPU environment, checks a bounded worker handshake, and durably saves identity
before release. The worker independently checks the capsule and process context
before its one callback. Actual exit and partial evidence persist; no restart.

**Tech stack:** Python standard library, Linux procfs, subprocess pipes, fsync,
the existing immutable EngineeringCapsule and bounded canonical JSON reader.

- [ ] Create one_shot_supervisor.py and engineering_worker_route.py under
  work/ood_robustness_v2. No default names a real resource or scientific entrypoint.
- [ ] Bind parent/worker PID and Linux start ticks, exact argv/environment,
  isolated interpreter, child process group/session, pins and declaration.
- [ ] Save exclusive claim, launch intent, observed identity and actual exit;
  never invent an exit on missing observation. Refuse reuse, mutation, aliasing,
  timeout and invalid/oversized/duplicate-key handshake. Capture raw output.
- [ ] Test synthetic children with temporary invented passing review receipts,
  fake checkpoint/runtime/ancestor bytes and the unchanged capsule fixture helper.
  Tests cover successful handshake, actual nonzero process death, timeout,
  held gate before spawn, missing handshake, mutation and second invocation.
  No prior suite or scientific audit is rerun. No real lock/ledger/key/model/physics.
- [ ] Review refusal paths and run only new tests, retaining every actual exit.
- [ ] Record scope explicitly: this is a cooperating process route, not a sandbox,
  proof of review semantics, full descendant containment, actual native runtime
  compatibility, shared-lease ownership or permission for real physics. Real
  entrypoint/import/schema/storage/source/runtime acceptance remains pending.
- [ ] Verify frozen sources/receipts, package small owned code/evidence, publish
  with actual push/remote/blob receipts, and update the active monitor with a
  concurrent-prompt guard. Stream amendment remains pending and unmodified.

The user prohibits delegation and already authorized engineering preparation;
execute and review in this session without subagents or another approval question.
No preimplementation red result has been captured; do not invent one.
