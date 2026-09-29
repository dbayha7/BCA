# Independent saved-outcome verification for the two live attempts

This is the concrete audit schedule, not a completed outcome audit. The reviewer
must read saved bytes only and never load a policy, simulator or scientific
writer. A successful worker exit is necessary but does not certify outcomes.
The current SQLite workers remain unchanged. A later append writer must expose
the same logical evidence names and exact payload bytes to these joins.

1. Bind the actual supervisor exit, original process/dispatch identity, exact
   `declaration.json`, `execution-complete.json`, source hashes, checkpoint
   identities, accepted training inputs and key-file disposition. Require all
   512 nominal continuation panels or explicitly justified missing captures.
   Scheduler `RUNNING|0:0` is never a successful worker exit. Failed attempts
   retain every partial record and charged reservation.
2. Open evidence and extension databases through read-only SQLite URIs and
   `query_only`, without calling either production constructor. After the writer
   exits, bind file identities and stream artifacts in increasing `id` order.
   Rebuild canonical metadata hashes, previous-hash chain, complete flags, raw
   decompressed lengths and SHA256. Match count/head to the completion manifest.
   Bound decompression and memory to one record plus one current trajectory.
   Do not copy multi-gigabyte databases to Windows or run a repeated full scan.
3. Stream the extension reservation/completion join below. Reconstruct every
   scope total including engineering, collection, repeats, outcomes and any
   pending/uncertain charge. Check the exact declaration and original ancestor
   attestation, `1 environment / 4 physics` per call, chained reservation digest,
   completion-to-reservation digest and evidence hash. A completion receipt is
   joined to evidence; it is not independent proof of physical completion.

```sql
SELECT r.id, r.token, r.payload, r.digest,
       c.evidence_sha256, c.payload, c.digest
FROM reservations AS r
LEFT JOIN completions AS c ON c.token = r.token
ORDER BY r.id;
```

4. For each `calls/<owner>/completed`, join the token/scope to that reservation,
   and the saved hashes to `/before`, `/output`, `/after` and all three
   `transition-<number>{-input,-applied,}.npz` records. Independently decode typed
   arrays with exact dtype/shape/bytes and NPZ with pickle disabled. Match native
   input snapshot to full before state, proposed/sent action to the declaration,
   intercepted applied action to the fixed float32 wrapper transform, and
   applied action to native float64 controls. Native output fields must match
   the typed output, with finite arrays and exact full after-state contents.
   Recompute forward reward from before/after position and the bound simulator
   timestep; action cost, alive bonus, health predicates, elapsed step and native
   time-limit flags must agree. Preserve absolute action `1e-6`, reward `1e-7`
   and exact full-state/record repeat gates. Constructor evidence has its own
   namespace and joins; count its physical call separately.
5. Reconstruct the declared 256 capture schedule, terminal/missing chronology,
   training-only support neighbors (stable distance ordering and normalization),
   q95/q99 bands, fixed 8192 candidate pool and generation-order selection.
   Bind original/amended stream map, saved key bytes and all 14 nominal slots.
   Check exact aliases and missing quotas. Reconstruct these from accepted saved
   inputs, never by regenerating keys or running a policy. Candidate policy slots
   and warning scores retain checkpoint/query provenance. Do not infer candidate
   or score correctness solely from an intact archive hash chain.
6. Require every bank and score record plus the whole-pair manifest before the
   first outcome-start record. Join `panels/<continuation>/stateNNN/slotK/started`
   to its bank, capture and `(250,2)` key row. Join `/first` and `/repeat` to their
   separate charged calls; compare complete first record and full first end state
   exactly. Join each numbered outcome call in order, exclude the repeated check
   from returns, and stop at 250 total steps or native termination/time limit.
   Recompute raw undiscounted sums, lengths, final flags and full final-state
   hashes from the saved rewards/states. Alias slots reference one accepted owner
   result and contribute no duplicate physical calls. Missing slots stay missing.
7. Only after all joins and actual exit acceptance, derive the declared primary
   `A = G_BCA(a) - G_host(a)` and `D = A(a) - A(anchor)` with the same recorded
   anchor. Co-report familiar returns and first-action quality. Keep warning
   ranking separate; distinguish within-state from pooled ranking, state-near
   from joint shift, collector/capture strata, reset dependence and seed counts.
   A single completed seed is descriptive, not a five-seed BCA benefit claim.

Persist small verified chunk receipts against an immutable archive prefix to
avoid rescanning accepted bytes. Final coverage must include every record and
reservation; partial prefix checks cannot declare pair completion. This audit
implementation and its real execution remain to be completed. The closed
collection-only review and engineering review remain inputs, not substitutes.
