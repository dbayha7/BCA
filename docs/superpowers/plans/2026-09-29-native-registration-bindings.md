# Concrete registration callback and live state bindings Implementation Plan

> **For agentic workers:** Execute in this chat. The user's no-subagent and quiet
> heartbeat instructions override the writing-plans skill's delegation/handoff
> defaults. Existing closed sources and installed dependencies remain unchanged.

**Goal:** Implement the missing concrete callback/state forms required by the
actual filelock registration route, without authorizing its still-held import.

**Architecture:** A source-bound binder has no hook installer and never invokes
callbacks. It verifies real Python/class/native bound methods, captured positional
and keyword defaults, the Random.seed __class__ cell, filelock's actual state
class forms, logging weak-handler state and threading's natural partial append.
The later unified dispatcher will call this binder at its registration boundaries;
this turn cannot accept a complete transitive graph or launch the actual package.

**Tech Stack:** Pinned Python3.10, stdlib source compilation without execution,
explicit native owners/descriptors, separate isolated process fixtures and a
stdlib saved-evidence review.

## Tasks

- [ ] Create `work/ood_robustness_v2/native_registration_bindings_v1.py` with
  explicit stable-window bindings and persistent first-refusal state. Recheck
  defining owner, compiled source, defaults/closure cells, namespace/source,
  actual state container identities and native bound method descriptors.
- [ ] Create `work/ood_robustness_v2/test_native_registration_bindings_v1.py`.
  Use actual stdlib logging/random/threading/native-lock objects and exact saved
  filelock state definitions in separately identified synthetic namespaces.
  Do not preimport filelock, replace capability results, invoke callbacks, fork,
  or run old tests. Preserve each new source version and actual child exits.
- [ ] Exercise captured-reference substitutions, nonempty retained registries,
  native-owner/descriptor changes, defaults/closure changes and actual internal
  threading registration partials. Children use os._exit to avoid shutdown work.
  Tests are implementation fixtures, not full filelock namespace acceptance.
- [ ] Independently review saved source/receipts and exact callback bodies, then
  verify prior closed inputs. Publish only owned small code/evidence/docs.
- [ ] Preserve the production launch hold and all previous instructions in the
  ACTIVE monitor. Next integrate these bindings into the unified actual route;
  the binder's direct references alone cannot satisfy full graph acceptance.
