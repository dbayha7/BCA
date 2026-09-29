# Cluster OOD execution, September 29

David explicitly reminded us to use the cluster and ensure correctness. Keep the
active GPU training queue and the local TD3 Hopper OOD worker running. Use a
separate scheduled CPU allocation for cluster OOD; no third GPU or login-node
scientific computation. Begin with the independently accepted ReBRAC Hopper
seed 202609171 pair, followed by accepted declared ReBRAC Hopper/Walker pairs.

The cluster inspection confirms the five numerical package versions and seven
audited Hopper/Walker simulator/wrapper source hashes match the local runtime.
This is runtime identity evidence, not simulator or outcome acceptance.

First execute a zero-transition checkpoint/query gate on the cluster. Bind the
original 108 training sources, accepted data and preparation hashes, final 1M
checkpoint hashes/counters, and actual training exits. Restore ReBRAC's saved
calibrator normalization into its query model. Those fixed statistics are part
of the trained checkpoint: recomputing their population reduction on a different
device is not an acceptable replacement. Compare saved statistic hashes to the
already accepted preparation, and record any difference from CPU recomputation.
Do not change the checkpoint, underlying loss, calibrator parameters or radii.
Run native actor and target arithmetic checks and frozen width/dose queries on
fixed training rows only. No simulator, training update or new random seed.

Before cluster physics, partition the original v2 budget by pair into disjoint
local and cluster scopes. The current local worker is restricted in executed
code to TD3 Hopper seed171. Cluster workers must never run that pair or duplicate
one another. Their reserved maxima plus all earlier reservations must fit the
original envelope; a separate remote ledger is not an extra scientific budget.
The remote lease, live constructor/restore/action/reward gates, state/candidate
precommit and actual exit checks must be passed before remote outcomes. Reuse
the accepted pure components; do not rebuild the import-hook framework.

Preserve every execution attempt and the active local worker's immutable files.
No completed study claim until independent outcome verification. The full twenty
pair tranche is not yet queued. CQL/IQL adapter semantics remain separate work.
