# Training protocol and adapter review — 20 October 2026

No outstanding material findings in the reviewed core. The full comparison is 30 jobs: five independent trained replicates, three critical-retention levels, and native TD3+BC versus BCA. Each job has 100,000 critic updates and 50,000 delayed actor updates. This budget follows the actual ThrusterChain `LearnSettings` convention; it is distinct from the general 1,000,000-update D4RL configuration.

The reviewer inspected the unchanged repository host/BCA initialization, update, target, reference, and dose functions; the new training adapter; source-data construction; continuous evaluation; checkpoint/resume handling; and the full-run controller. The reviewer authored `TRAINING_PROTOCOL.md` and the separate evaluation-only `training_calibration.py`, but did not author the training adapter, data generator, policy/Q evaluator, or full-run controller. A second reviewer independently read the calibration evaluator and authored its focused tests.

Verified scientific properties:

- Native initialization and minibatch/target-noise streams match within each host/BCA pair. Extra BCA randomness uses separate folds. Floating-point BC reduction order can differ, so warmup is not claimed bitwise identical.
- The new target uses gamma one, 15-coordinate observations, raw rewards, and `done = 1 - bootstrap_mask`, including zero continuation at horizon 100.
- Exactly 128 source episodes are reserved before retention from the 1,024-episode balanced source log. One retained row per reserved episode enters calibration. Both methods share the retained training complement and normalizer. Removed rows are not bridged or replenished.
- Validation, final evaluation, and designed critic queries have separate fixed namespaces. Terminal checkpoints are primary, midpoint checkpoints descriptive, and no return-based selection is implemented.
- Native unready and infinite-threshold behavior uses unit BC multiplier. The native usability fraction is not certified or population support. Numerical calibration sufficiency and adaptive bank reuse are disclosed.
- Source generation and evaluation preserve the finite-horizon physical contract. Continuous actor evaluation, fixed-start numerical QP bounds, and the finite companion's exact dynamic programming remain separate claims.
- Failed/refused jobs remain represented; completed paired comparisons are aggregated over all five planned replicates only. Final evaluation requires the declared terminal checkpoint. Fixed controls can complete while learner or oracle work remains pending.

Resolved review findings:

- The adapter originally compared the entire objective metadata dictionary with only horizon/gamma keys, rejecting the actual new generator's richer `Objective.as_dict()` metadata. It now validates the relevant objective semantics.
- Evaluation originally clipped infinite action outputs into finite bounds. It now preserves nonfinite outputs as failures before clipping.
- Critic-diagnostic exceptions now retain valid policy-return arrays and receive an explicit failed diagnostic status. The ranking tie rule is declared.
- Critical counts are recorded by context; numerical calibration sufficiency is explicit; historical checkpoint loading authenticates the requested snapshot.

Verification evidence:

- The final adapter suite passed all seven tests in 45.330 seconds; see `../../outputs/training_adapter_tests.txt`. It covers real CPU updates, exact native host trace, terminal target equality, reserved trajectory isolation, BCA refresh, optimizer/PRNG checkpoint resume, authenticated snapshot reload, and baseline completion when BCA refuses.
- The independent calibration suite passed all seven tests in 5.783 seconds, CPU only with four-core affinity and two-thread numerical-library limits. It covers absent/refused outputs, unready and infinite-band semantics, terminal residual targets, six action candidates/current twin-min Q, deterministic evaluation noise, and unchanged learner/query data.
- Adapter SHA-256: `935db9f71326379f452b89c1f69de2cbd4920c005048c5b5d45cd3a3cbafa2b7`.
- Calibration evaluator SHA-256: `aa0679d565938425a70222d8078266fed6796a37aa52fde1ee975f58b625da02`.
- Calibration tests SHA-256: `e362656fa8ac08cade2b0ef68282dc19ed657c9001b6a6327f0aa0ecf7a1b44b`.

`configs/planned_comparison.json` and its metadata-only audit deliberately retain unknown actual runtime, data hashes/counts, source revision, and earlier development budgets as null. That declaration is not a completed-run report or a fairness certificate; the frozen run manifest supplies execution evidence.

This review does not establish learned-method benefit, successful completion of the 30 jobs, or the accuracy of a generalization guarantee. No full training job was launched by this reviewer. Launch, progress, and final completion must be established by the parent run's own artifacts.
