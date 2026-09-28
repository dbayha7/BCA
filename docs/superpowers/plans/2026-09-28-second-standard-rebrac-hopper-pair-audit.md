# Second-seed ReBRAC Hopper saved-training audit implementation plan

**Goal:** Independently audit the declared ReBRAC Hopper host and BCA runs for
training seed202609172 in fixed host-then-BCA order, without a new learner,
model query, simulator call, GPU request, OOD key or stream amendment.

**Architecture:** Reuse the closed audit contracts as byte-pinned parents; create
separate explicit second-seed auditors and retain derivative diffs. First inspect
the current controller/worker identity and the fixed pair's metadata. Bind the
new input hashes, actual partitions and seeded reference order, then verify all
sources/data/checkpoints/counters/metrics/evaluation/refresh evidence. The host
must close successfully before the BCA auditor uses its newly bound exports.

- [ ] Save current monitor bytes; obtain a fresh bounded cluster identity probe.
- [ ] Inspect only the fixed second-seed pair metadata and closed-run identities.
- [ ] Derive separate auditors, with exact seed/partition/dimension checks and
  externally bound new preparation/result hashes; preserve all parent sources.
- [ ] Run new synthetic guard tests only, retaining actual exit/source hashes.
- [ ] Dispatch one read-only CPU audit per run, host then BCA. Preserve actual
  process/transport exits and all failures; no duplicate dispatch or retry.
- [ ] Independently review all saved exported arithmetic, sources and receipts
  before accepting a comparison; all five seeds remain required for inference.
- [ ] Publish owned small code/evidence and update current status with exact
  push/remote/blob verification. Keep stream and resource holds unchanged.
- [ ] Guard and merge the latest automation prompt, keep ACTIVE, notify only if
  a new comparison is independently accepted or another meaningful change occurs.

No subagents/messages. No closed audit/test rerun or routine evidence repull.
The user authorized this fixed-order audit; no additional confirmation is needed.
