# Native import v3 worker bootstrap implementation

**Goal:** Implement the declared worker-owned root creation and bounded diagnostic routes inside the first concrete v3 dispatcher, with one audit/profile owner and shared first refusal.

**Architecture:** `native_import_dispatch_v3.py` implements only the bootstrap and diagnostic phases of the declared route. The remaining native/registration/capability phases refuse; there is no full import entry point or dispatch acceptance. New isolated children use an explicit fixture binding with different named roots. They exercise real directory/file descriptors, ownership, native calls, fsync, retained failures and parent release, not model or package imports.

**Tech stack:** Pinned Linux Python3.10 standard library; JSON receipts; Windows receipt/publishing helpers.

User instructions override skill delegation/handoff suggestions. No subagents, closed suite reruns, target imports, actual v3 root creation or scientific dispatch.

- [x] Implement one-use source/contract/process binding, exact root creation and four diagnostic files, native call tickets, refusal persistence and retained hooks.
- [x] Test new fixtures: parent releases only after current process/environment identity; only the child creates roots; direct native aliases/outside writes/registration/env mutation, stale source/paths/mode, repeat creation and byte limits refuse; deliberate exit23 retains fsynced bytes.
- [x] Review saved child exits, path/file bytes and evidence independently without importing the dispatcher or replaying children. Preserve every development version/failure.
- [ ] Verify immutable previous inputs; publish only owned source, tests and small evidence. Document bootstrap-only scope and full-route blockers; guard automation update and keep ACTIVE/quiet.

No preimplementation red is claimed unless actually captured. Synthetic fixture bindings are not the unlaunched production fifteen-field profile. Partial implementation must not be presented as a composed native route or a completed launch prereview.
