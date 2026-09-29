# Measured storage bottleneck and next accepted OOD pairs

September 29, 2026. A bounded storage-only test found a substantially faster
durable write pattern. Neither running scientific worker has been changed, and
no faster OOD rollout rate or completion date has yet been demonstrated.

| Synthetic storage path | Local calls/s | Cluster calls/s |
| --- | ---: | ---: |
| Current SQLite transaction shape | 25.23 | 14.74 |
| Sequential shared journal | 131.78 (5.22x) | 321.46 (21.80x) |
| Node-local plus synchronous shared replica | 78.11 (3.10x) | 205.75 (13.95x) |

Each figure is the median of three 32-call trials. Every synthetic call retained
the same seven evidence payloads (18,688 raw bytes), reservation and completion
record. The SQLite baseline models seven archive commits plus two accounting
transactions using FULL/DELETE; it does not execute the production guards or
simulator. The sequential alternative flushes reservation, pre-action evidence,
post-action evidence and accounting acknowledgment at four ordered boundaries.
The replicated alternative synchronizes both copies at every boundary.

The node-local-only diagnostic reached 133.72 local and 517.38 cluster calls/s;
it is **not acceptable as the sole durable record** because job/node loss can
remove it. Local `/tmp` and the local shared-test directory are both on the WSL
host; this does not establish independent physical devices. Cluster trials used
one CPU on str-c145 (job 27068273), independently of the active str-c130 worker.
Setup, readback review and evidence copying were outside the timing loop.
Payload compression, real states, policy queries and guard overhead can change
end-to-end gains. No GPU or simulator was used by the benchmark.

Both benchmark processes exited 0. At each location, 17 child processes exited
the deliberate code 23 at selected durability boundaries or with an incomplete
tail. A separate saved-byte reviewer exited 0 for both locations: it reconstructed
3,456 timed frames per location, matched the same logical payloads across modes,
and checked charged/pending versus acknowledged disposition and retained partial
tail. This is process-death evidence, not power-loss or node-loss proof. No
restart/recovery writer was implemented or accepted. All original files remain.

The measured candidate is the simple shared sequential journal. A separately
versioned execution adapter must preserve the actual constructor/call evidence,
every pre-physics reserve/input/control flush, full output/state records and
post-evidence completion before it can run real physics. The existing workers
keep their original storage and sources.

Five additional accepted ReBRAC pairs have now passed CPU checkpoint queries:
Hopper seeds 202609172/173 and Walker2d seeds 202609171/172/173. Jobs 27068295
through 27068299 all exited 0. Both actors matched exactly for each pair; saved
target arrays and direct width/dose arrays matched. The independent saved-array
review checked actual exits, source pins, checkpoint identities and these arrays.
Checkpoint-saved calibrator statistics were retained. These checks generated no
new OOD keys, simulator steps or training updates. They are preparation for OOD,
not five newly running outcome studies. Existing accepted training was not rerun.

The parallel allocation proposal preserves the local TD3 / cluster ReBRAC split
and the existing 18,395,680-environment-call limit per half. It protects the live
Hopper171 pair against duplicate dispatch and retains its full 10,000-call Hopper
engineering entitlement. It therefore does not allocate that same entitlement
again to Hopper172/173. Walker's separate 10,000 engineering calls can be split
as 2,000 for each of its five declared seeds, with only already accepted training
pairs eligible for dispatch. Phase caps remain 1,837,568 per pair. Permanent
atomic claims, pair exclusion/crash checks and actual storage/simulator gates
remain required. No allocator or additional physics worker is accepted yet.

At the separate 08:09 UTC check, the original local worker had completed 27/512
panels (132,503 explicit transitions); the cluster worker was still collecting
at 36,202 explicit transitions. Both original command/start identities and source
hashes matched, no failure was recorded, and both actual exits remained pending.
The local supervisor's original start-tick pin remains unavailable. The next
execution work is the concrete storage adapter plus disjoint Walker allocation,
then unchanged actor/reward/full-state gates before new rollouts. The accompanying
outcome verification schedule defines the saved-data joins needed for results.
