# Desktop restart: OOD outcome recovery

David reported a desktop restart and explicitly requested starting again.
The original CPU worker was absent in a newer boot and the shared lock was free.
Its actual exit remains **unknown**, not an invented failure code or successful exit.

The first attempt is unchanged. All **143 complete panels and 1,438 complete
action rollouts** are reused by hash. The sole unfinished action was
`host/state143/slot8`, with 236 saved outcome steps plus one repeat. Recovery
restarted only that action from its original state/action/keys; all **237 saved
records matched exactly** before new outcome steps were permitted. No numerical
gate, model, policy, scale, radius, dataset or scientific setting changed.

At **2026-09-28 01:54:24 UTC / September 27 9:54 p.m. Eastern**, the new worker's
command/start/group identity and shared lock matched. Progress had advanced to
**144/512 panels**. Worker and supervisor exits are pending. This is progress in
one exploratory training seed, not accepted harm/regret rankings or coverage.

The recovery uses the original cumulative ledger and unchanged caps. Its maximum
new physical work is 924,183 environment transitions / 3,696,732 physics steps.
The replayed 236 steps, one repeat and one new constructor consume **238 existing
engineering-allowance transitions**. Original reservations are never erased or
refunded. Primary estimates retain one complete outcome per action; the interrupted
partial trace is kept separately, never counted as another sample or spliced in.

The original archive's 1,083,525 records passed a sequential chain/payload-hash
check. Saved complete-slot rewards/counts, repeats, snapshots, actions and keys
were bound before dispatch. Full independent scientific outcome verification
still follows successful closure of the combined collection.

**Checks:** 41 focused recovery/original rollout/resource/collector tests passed;
the pre-implementation missing-module test and inspection correction are retained.
Existing 103 regression checks remain earlier evidence. CPU plugin discovery again
emitted its preserved no-GPU warning; execution was verified on CPU. No training
retry or additional GPU was launched. The cluster job was independently live;
the local CQL schema failure remains separately stopped.

The exact recovery plan is losslessly stored in `recovery-plan.json.gz`.
`export.json` records each small evidence file's stored/original byte hashes.
The original and recovery high-frequency archives remain at their exact WSL
paths in the plan; neither those databases nor model weights are copied here.
Source: `experiments/ood/restart.py`; the original `outcome_stage.py` and
`rollouts.py` are unchanged. This new recovery has no automatic retry path.
