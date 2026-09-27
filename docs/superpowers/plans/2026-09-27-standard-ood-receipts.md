# Standard-study OOD receipt binding implementation plan

> **For agentic workers:** Execute in this session without subagents, as David requested. The existing standard-study authorization includes OOD adapter preparation; no new execution approval is needed for this read-only metadata unit.

**Goal:** Bind the standard runner's actual worker exit to its declared run and result without mistaking metadata validation for accepted OOD evidence.

**Architecture:** Add a standalone standard-library bridge in `experiments/ood/standard_receipt.py`. It reads only explicitly named lane/source/manifest paths, verifies the frozen manifest, dispatch identity, matching actual-exit timestamps, learner completion, resolved declaration and reported source pins, then produces the existing `ood-process-exit-v1` receipt. It never loads weights, decodes event journals, queries models, steps simulators, restarts jobs, or marks collection ready. Initial support is the existing TD3+BC/ReBRAC Hopper/Walker adapter scope.

**Tech Stack:** Python standard library and unittest; existing OOD metadata schema.

---

### Task 1: Specify failure cases before implementation

Create `experiments/ood/test_standard_receipt.py`. Synthetic metadata fixtures must test successful host/BCA binding while retaining pending scientific status; failed, timed-out, interrupted, mismatched-start and reversed-time exits; wrong dispatch arguments/output/source; changed manifest/resolved/source/result; non-1M result; unsupported host/dataset; and exclusive receipt output. No synthetic fixture is a scientific checkpoint.

- [x] Run `C:/Python314/python.exe -m unittest experiments.ood.test_standard_receipt -v` before implementation and preserve the failure log.

### Task 2: Implement only the process-receipt bridge

Create `experiments/ood/standard_receipt.py` with `bind_receipt(manifest_path, lane_directory, source_directory, run_id)` and exclusive `write_receipt(path, receipt)`. Pin the uncompressed study manifest SHA256. Derive the attempt and run directories from the unique frozen row; validate exact algorithm/dataset/method/seed/device/output arguments and the frozen train.py path. Require an actual integer zero exit, no timeout/interruption, identical dispatch/exit start time and an aware end timestamp at or after start. Require learner completion, exact resolved row and reported source pins, a 1M result with matching declared counters, and the final checkpoint's declared identity. Hash and retain all supporting metadata. Return `ready_for_collection=false`, `checkpoint_decoded=false`, `events_verified=false`, `data_contents_verified=false`.

- [x] Run the targeted suite with actual exit capture.
- [x] Confirm all 108 frozen source files and the compressed manifest remain unchanged.

### Task 3: Record and publish this bounded preparation

Update only unfrozen `docs/STANDARD_BCA_STATUS.md` with the independently checked 14-cell cluster data gate and its actual exit0. Add `docs/validation/standard-bca-live-gate.json` and `docs/validation/standard-ood-receipts.json` with hashes, actual test exits and explicit limitations. Preserve the monitor's initial login-node process lookup and direct compute-node SSH timeout; corrected process identity comes from a read-only Slurm step in the existing allocation, with no GPU request.

Publication procedure: review diff, run `git diff --check`, commit owned changes and publish to the existing private repository.
Record actual publication exit and remote head equality. No training completion or OOD result is implied.

Publication closure is recorded separately in the originating workspace monitoring snapshot `work/standard_bca_noiw_campaign_v1/monitor_20260927T193850Z/publication_receipt.json`, alongside a separate actual-exit receipt. This plan alone does not assert publication.
