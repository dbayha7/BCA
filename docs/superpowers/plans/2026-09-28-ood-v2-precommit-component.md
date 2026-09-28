# OOD v2 precommit component implementation plan

**Goal:** Make the fixed capture schedule and all14 nominal action slots reviewable and immutable before any outcome or warning score is observed.

**Architecture:** Add separate pure preparation modules beside the frozen candidate selector. Preserve missing slots rather than compacting them; bind every populated slot to proposed/sent/applied coordinates, support labels and earliest exact applied-action owner. Bind actual supplied keys and external source/state identities, then write an exclusive, hash-bound artifact. These components do not query policies, generate JAX keys, construct simulators, acquire the resource lock or open a resource ledger.

**Tech Stack:** Python3.10, NumPy, standard-library hashing/JSON and exclusive durable file writes. Existing frozen candidate_design.py is reused unchanged.

The standing no-subagent instruction overrides delegation suggestions. Implement and review locally. This is an engineering component plan under the frozen prospective scientific protocol, not a change to its sampling/endpoints or execution acceptance.

## Files and checks

- Create work/ood_robustness_v2/precommit_bank.py for the256 nominal captures, fresh stream collision refusal, fixed14-slot banks and externally checked precommit equality.
- Create work/ood_robustness_v2/precommit_file.py for bounded duplicate-key-refusing JSON, exact hash checks, file identity and exclusive durable publication.
- Create work/ood_robustness_v2/test_precommit.py with synthetic inputs only. Preserve the expected missing-module failure before implementation.
- Save actual receipts in work/standard_bca_noiw_campaign_v1/monitor_20260928T184657Z.

## Checklist

- [ ] Test64 paired reset blocks, two collectors, captures100/300 and fixed state indices; reject undeclared host/environment/seed/index.
- [ ] Test complete and quota-deficient fixed banks. Anchor0, other near1..3, moderate4..7, strong8..11, host12 and BCA13 never move. Missing slots remain explicit. Alias ownership uses earliest numerically equal applied float32 action, including signed-zero equality, without discarding proposed/sent bytes.
- [ ] Reuse the frozen8192-proposal generator/selector without tuning or candidate score inputs. Check all proposed/sent/applied fields, support distances, clipping and training-only boundaries against supplied bound inputs.
- [ ] Require exact256x250x2 uint32 key tables for a pair, finite/bounded predecessor seed/key inventories and no collisions, with no resampling. Key derivation and predecessor-inventory completeness still require independent runtime/source acceptance; this component cannot infer them from an array.
- [ ] Bind exact external receipt/source/runtime/checkpoint/support/state hashes. Rebuild expected bank from independently supplied inputs and compare canonical bytes; a manifest cannot self-accept its own hashes.
- [ ] Test durable exclusive write, no overwrite, exact external hash, bounded reads, duplicate JSON keys, symlinks, mutation and missing pins. A partial write remains visible and is never silently replaced.
- [ ] Freeze tested sources, verify earlier sources/receipts unchanged, and publish only owned code/small evidence with actual push/remote/blob receipts.

Remaining after this component: actual source/path/execution declaration integration, collector and outcome driver, constructor/full-final-state evidence, native control/repeat parity, full-state/runtime/checkpoint gates and independent execution acceptance. No real extension ledger or physical call is authorized by a passing synthetic test.
