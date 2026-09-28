# OOD v2 resource-extension implementation plan

**Goal:** Add a separately versioned reservation component that preserves the original ledger and enforces the combined old/new ceilings before any physical call.

**Architecture:** A read-only, externally pinned ancestor guard; a lifetime exclusive lease on the existing shared lock; and a separate append-only SQLite reservation/completion journal. Use only temporary synthetic files this step. Do not create a real extension ledger, acquire the real resource lock, query a model or simulate. Execution acceptance and independent collector review remain separate.

**Tech Stack:** Python3.10 standard library, SQLite mode=ro/query_only for the ancestor, FULL synchronous DELETE journaling for the new journal, Linux flock. No imports from frozen scientific collectors/ResourceLedger.

Standing no-subagent direction applies. Implement and review locally. Preserve all earlier sources/tests/receipts, including expected initial failures.

## Files and tests

- `work/ood_robustness_v2/ancestor_guard.py`: check confined absolute paths, bounded external header/head/totals and accepted file identity; refuse all sidecars and changed/redirected originals. Small metadata only; no full original-ledger scan.
- `work/ood_robustness_v2/extension_ledger.py`: immutable declaration, exact fixed scope structure, cap reductions only, durable reserve-before-call, unique token, one unresolved reservation maximum, append-only completion-receipt hashes, no refund/resume/repair path. A completion receipt is producer evidence, not an independent scientific acceptance.
- `work/ood_robustness_v2/test_extension_ledger.py`: temporary SQLite/lock fixtures only. Cover original mutation, path/lock replacement, simultaneous lease, duplicate/cap/wrong-charge refusal, callback failure, process death after commit, pending-call reopen refusal, trigger/chain/total tampering, completion evidence and exact preserved ancestor bytes.
- `work/standard_bca_noiw_campaign_v1/monitor_20260928T182657Z`: actual exits/stdout/stderr and reviewed source/test hashes.

## Checklist

- [ ] Write synthetic known-answer and refusal tests; preserve initial missing-module exit.
- [ ] Implement ancestor guard and exclusive lease without opening any real resource file.
- [ ] Implement extension journal. Verify original identity and lease before reservation, after durable commit and before callback. On uncertainty leave reservation charged and stop. Never return budget after a callback or process failure.
- [ ] Reopen only a fully closed journal after an audit; an unresolved reservation requires a separately designed recovery, not automatic continuation. SQLite sidecar recovery is not silently performed.
- [ ] Verify chain, cached totals, exact immutable schemas/triggers and completion ownership on open. During ownership reject file replacement/mutation or foreign process/fork use.
- [ ] Require independent execution-layer acceptance for real paths, plan/source/checkpoint/runtime pins and actual worker/supervisor exits; component tests do not satisfy these gates.
- [ ] Freeze new tested sources, publish only owned small code/evidence after original-source checks and exact push/remote/blob verification; update monitor with the remaining collector work.

The prospective v2 scientific plan SHA remains f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756. No change to sampling, endpoints, seeds or actual global cap. Maximum additional reservations remain36,791,360 environment/147,165,440 physics. Original1,298,353/5,193,412 remain charged. Ancestor exits and the input-only physical output remain unknown. Budget enforcement does not infer completion.
