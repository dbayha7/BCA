# V6 bounded validation plan

This plan fixes expected software outcomes before integrated smoke results are
examined. It is not a scientific study or a final evaluation protocol.

1. Preserve v4 and v5 source identities. Run the v6 invariant and contract
   tests, recording historical-v4 artifact skips separately.
2. Generate small, independently seeded episodes for each new declared behavior
   dataset. Check exact step and episode provenance, executed actions,
   finite-horizon terminal masks, train/calibration group separation, save/load,
   and rejection of tampering. Do not predict or select on return rankings.
3. Check new shift laws with fixed seeds, including baseline parity, reproducible
   process noise, command/reward semantics, physical bounds, and explicit
   observation and hidden-regime assumptions. No simulation validates real
   hardware performance.
4. For one explicitly resolved unified UniFloral composition, compare its
   fixed-batch update and PRNG splits against the pinned host, including actor,
   critics, target models and counters. Verify exact checkpoint resume and a
   frozen short evaluation. Record runtime and any compatibility changes.
5. Run a bounded integration smoke using a new feedback dataset, at most two
   unified actor updates (four critic updates for the chosen composition) and
   at most one 100-step evaluation episode per new scenario. Check all episodes
   complete and frozen policy state stays unchanged. These observations test
   wiring, not relative method performance. If the runtime or parity gate fails,
   report the failure and do not claim this integration smoke passed.

No million-update training, tuning sweep, policy-selection experiment, or
scientific performance comparison is authorized by this plan.

The first smoke receipt remains under `unified_smoke_01`. Afterwards the copied
standalone adapters' dataset labels and adapter IDs were versioned for v6 so
their metadata cannot describe new behavior datasets as v5. The same bounded
check is repeated as `unified_smoke_02` against the resulting source snapshot;
the earlier receipt is retained as an intermediate, not relabeled.
