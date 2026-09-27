# First standard ReBRAC Hopper host readout

Audit the newly closed `rebrac-hopper-host-s202609171` from saved evidence. The
existing controllers continue the authorized matrix. This work adds no model
query, simulator step, learner update, GPU worker or scientific retry.

1. Bind the actual worker exit to the frozen declaration, source and result.
2. Check all 108 source hashes, raw/converted data, paired training fingerprint,
   three decoded checkpoints, complete journal and all 201 evaluation banks.
3. Aggregate ReBRAC metrics with zero-based even actor rows; keep dataset-action
   Q, coordinate-summed BC and coordinate-averaged action errors distinct.
4. Render and inspect a host-only plot. Report every final episode, calibration
   as N/A, and pending BCA pairs, further training seeds and OOD acceptance.
5. Verify frozen integrity, publish only owned code/readouts and retain separate
   actual publication exit and remote-head receipts. Preserve prior attempts.

The audit runs on CPU in the existing Slurm allocation with no requested GPU.
Checkpoint weights and the full original journal stay on the execution host.
No subagents are used. The previous TD3 audit is reused without repulling.
