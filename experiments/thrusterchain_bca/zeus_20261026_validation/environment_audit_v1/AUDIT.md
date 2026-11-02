# Environment and conditional OOD study audit

Audit date: 28 October 2026. Scope: the pinned Titan UniFloral/ThrusterChain path and the completed central-quartile development gate. Training-package work is paused. This document does not authorize training or a new experiment.

## Readiness verdict

The existing evidence supports a **controlled synthetic, finite-horizon, conditional tuple-withholding study**. It does not establish a physically calibrated hydrodynamics benchmark, dense empirical joint state-action support, a missing-label learning effect, or BCA benefit. The development gate remains passed under its frozen criteria. The earlier negative-band gate remains failed. Neither result is revised here.

No confirmed environment implementation defect has been found in the inspected production path. No physics, reward, action bound, source, dataset, mask, model, protocol or completed receipt is changed by this audit. New artifacts are bounded regression checks, a validation receipt and this interpretation/claim map. Final test measurements are recorded in `PHYSICS_RESULT.json` and `AUDIT_RESULT.json` rather than inferred from code inspection.

## Verified semantics and deliberate design choices

| Area | What the implementation actually does | Implication |
|---|---|---|
| Physical state and units | Ten oscillator coordinates plus current and two rotating wave coordinates; 13 float64 state values. The frozen plant is damped rotations plus seeded dense coupling. There is no SI-unit, mass or timestep calibration. | A dimensionless control surrogate. The floats/springs/water narrative is an interpretation, not a validated physical model. Equal position/speed squared weights are a task cost, not joules. |
| Stability | The physical 10D plant has spectral norm below one; current is constant and the wave rotates without intended amplitude loss. Input is bounded. | The physical subsystem is contractive, but the augmented 13D matrix has unit modes. Do not claim all state coordinates decay or mechanical energy is conserved. |
| Actuators | Four commands affect the velocity-labelled coordinates of neighboring oscillator pairs. Commands are deployed as float32 then float64 and clipped to [-1,1]. Thrust is linear through the float32 knee near0.8 and falls beyond it. | Requested versus executed actions and the knee boundary matter. Native tanh actions already lie within bounds. Changing clipping, gain or knee would define a new environment. |
| Current and wave | The additive force uses current + first wave coordinate; current and both wave-phase coordinates are observed. | Strong-current resets change an observed initial-state distribution, not hidden dynamics. The high-load mask can also be reached through wave phase; it is not the strong-current scenario. |
| Reward | Negative pre-action oscillator squared norm minus0.3 times clipped command squared norm. It penalizes throttle, not generated thrust. | Control can improve future state costs while paying immediate energy. At t99 only immediate cost remains, so zero action is optimal there; this is intentional finite-horizon behavior. |
| Terminal behavior | The time-aware adapter terminates at100, with zero bootstrap. It explicitly differs from the archival continuing-time-limit mode, which truncates and bootstraps. | All evaluated data/reference semantics must remain finite-horizon. Premature numerical failures are errors, not rewarded escape terminals. |
| Observation sufficiency | All physical coordinates plus remaining time are supplied to the policy as float32. | No intended hidden current/wave state. Exact numerical state is float64, so float32 observations introduce small aliasing; never reconstruct exact physical state by promoting an observation. |
| Restoration | Public reset of a supplied state restarts at t0 and enforces start-set bounds. It does not accept an arbitrary saved time. | Interior-time Q-pi references correctly use exact prefix reconstruction plus explicit-time dynamics; private time mutation or a t0 reset is not a substitute. |
| Randomness | Dynamics and rewards are deterministic. Reset states and behavior actions use RNG. Recorder reset, action and split streams have separate named SeedSequence domains. | Repeating the same continuation is not independent Monte Carlo replication. Independent units are collection/start episodes and, later, independent training seeds. |

Source: `titan_20261026/payload/transfer_bundle/core_v4/thrusterchain/{environment.py,datasets.py,canonical/thrusterchain.py,canonical/gapped_lq.py}`. The canonical loader verifies source and matrix hashes and extracts unchanged definitions. The parent Gapped-LQ file contains other worlds/noise settings; those are not imported as ThrusterChain dynamics.

## What the OOD construction establishes

At the verified recorder epsilon0.25, requested actions have uniform density1/16 on[-1,1]^4 independently of the current state and time. Executed commands are quantized float32 values; the continuous density is not their discrete probability mass and is not a state-occupancy density.

The fixed hole is `abs(a0)<=0.25` and observed current + wave1 >=0.4. In the ideal population, removing it retains75% of high-load action probability and all low-load probability. Thus state support remains but state density changes. Symmetry preserves the conditional mean command only in expectation, not exactly in finite data.

The entire hole and adjacent candidate commands+-0.5 lie inside the linear-thrust region. This is interpolation across a deliberately unlabelled action interval, with observed load conditioning. It does not test extrapolation through the stall kink or a changed physical law. A critic may interpolate successfully; that is a valid negative result for an uncertainty penalty, not a reason to redesign until BCA wins.

Stalled commands also have an analytic limitation: in continuous-command arithmetic, a command beyond the knee has a smaller feasible unstalled command equal to its thrust, giving the same successor and a smaller immediate throttle cost. Float32 deployment can round that inverse-thrust command, so exact successor equality is not universal for deployed commands. Two exactly representable deployed pairs were checked: identical successors, with reward improvements0.22499993 and0.11625003. This is a useful known-bad-action mechanism, not a realistic claim that full throttle is always bad in physical robots. It is distinct from the central-quartile study.

The known sampler and exact mask define conditional support removal. The new64-neighbor check asks whether a query is typical of training-state geometry and whether marginal a0 alternatives appear nearby. It **does not repair or pass the former absolute radius0.5 requirement**, establish dense13D sampling, or certify the other three action coordinates at an identical state. A strict same-state, joint-action empirical study would need separately designed repeated-state branches.

## Complete data-to-reference path

1. **Generation and roles.** The immutable dataset binds source/adapter/physics, requested and executed actions, exact states, rewards, masks, group IDs and named RNG streams. The role-safe reader checks the pinned file/bindings and only decodes train or tuning-validation episode tuples. Protected selection/final-calibration tuple reads are rejected before opening or decoding the requested transition member; the ZIP archive may already be open. Whole-file hash bytes and provenance vectors are nevertheless read.
2. **Development reuse.** The gate used768 training +128 calibration +64 probe episodes. All were already development-consumed. The new64 probes differ from the earlier gate probes but are not untouched confirmation. Its16 free start seeds are a subset of these probes.
3. **Mask and controls.** Full76800 rows; hole and random67108 each. The random control exactly matches retained count within episode, ten-step bin and load side, and legitimately retains6184 hole labels. Hole replay retains zero. Maximum mean-state difference is0.001786 original SD. This checks coarse matching and first moments; it is not equality of the complete continuous-state distribution.
4. **Trajectory history.** Tuple filtering does not recollect trajectories. Retained states and successors can reflect earlier removed actions. This is not direct access to a removed action/reward label for a transition-only learner, but it does rule out claiming no historical exposure to the action region. Later replay, BC and auxiliary fitting must use the same condition-retained tuple IDs.
5. **Normalization.** The current frozen checkpoint legitimately retains its original broad-training normalization, because it is a diagnostic of that checkpoint. A later removal-training comparison must initialize from scratch and use the declared common normalization from the hole/random retained intersection, without calibrator/probe fitting. That integration has not yet been implemented or validated.
6. **Calibration plan.** Thirteen original selected rows were in the new hole and were replaced within their own128 calibration episodes by the frozen outcome-independent rule. All128 selected rows are unique; none is in the hole; no episode needed backfill. This validates a retained-support selection plan, not a fitted calibrator or conditional coverage. Repeated adaptive use in future training requires explicit caveats; no fresh guarantee follows automatically.
7. **Model and operator.** The tested model is the authenticated broad-trained Titan UniFloral checkpoint, not CORL or a hole-trained critic. Q1 is compared to full deterministic current-actor Q-pi. Native TD3+BC's smoothed target action, target network and twin-min bootstrap are different operators; their residual is not automatically full-Q-pi error.
8. **Truth/reference.** Prefix replay reconstructs exact state and time; every saved observation is checked. Six first actions hold the other actor coordinates fixed, then continue the same actor until100 with gamma0.98 and no bootstrap. Reward timing and t0/25/50/99 parity passed in the actual installed runtime. This is Q-pi, not Q-star.
9. **Fair comparisons later.** The current bank is shared within one policy. Across trained methods, own-policy Q-pi necessarily changes. A common-action/common-continuation panel is a distinct estimand. Same-critic raw versus adjusted score must not be conflated with separately trained native/BCA critics. Identical evaluation budgets, starts and action banks are needed for paired claims.

## Passed gate and what remains untested

The completed gate's hash is `91833bd4acf5a7657b2d43751c3800ffbe7b6b60c4c7c2573202b88faa522743`.

- State typicalness plus retained alternatives:22/23 high-load and40/41 low-load anchors passed; all20 reference strata were populated.
- Practical full-return separation:20/23 high-load anchors had a center/flank gap >=1; median3.009.
- Free exposure:576/773 high-load steps entered the hole, across14/16 episodes.
- Runtime accounting:28104 transitions,81 resets,zero updates,unchanged167-file prepared identity.

These measurements establish a plausible development assay. They do not show that removing labels causes extra Q error, that BCA reduces it, that overestimation dominates at every probe, or that adjusted scores improve policy return. Indeed the broad-trained critic had285 overestimates and99 underestimates among384 candidates. A downward penalty can harm the latter. Current free-exposure and logged-anchor value panels use different state distributions; their separate passes do not show every free hole visit has a measured value gap or nearby alternatives.

Strong-current performance, ordinary/strong-current learning curves, independent-training-seed uncertainty, selected-action validity and a Bayesian-specific contribution remain untested by this gate. Gaussian scale changes such as big_kicks have overlapping support; `outside_recorded_conditions` combines hard current/wave support checks with a99th-percentile tail heuristic and must not be treated as one exact support label.

## Fixes, blockers and next boundary

**Checks actually run in this audit:** six pure recorder/role unit tests passed; three physics tests passed covering11 fixed properties of the actual authenticated NumPy/JAX dynamics. All10 archival transition fixtures and scalar/batch results matched exactly. Current was preserved exactly; wave squared-amplitude residual was zero on the fixed fixtures. The physical matrix norm was0.987108 and spectral radius0.981659. A finite power sum plus norm-bounded tail gives a float-state radius bound256.3696 below the declared300 limit. CPU JAX float64/NumPy parity residual was zero; the maximum tested JVP/finite-difference discrepancy was1.826e-10 against a2e-8 tolerance. This is one smooth direction at three finite-difference scales, not an exhaustive gradient proof. Clipping/stall cusps do not have a unique derivative.

The physics test invocation and saved receipt invocation together used104 fixed NumPy one-step rows and eight JAX rows. No environment episodes, resets, trained-model calls, dataset tuple decoding or optimizer/calibration updates occurred during this audit. The completed central-quartile gate and its39 earlier preparation tests are existing evidence, not newly rerun experiments.

No semantic change to the benchmark is proposed as a repair for these limitations. Correct the study's description and keep separate truth labels for support removal, signed value error and action quality. The old negative-band failure remains part of the record.

The current environment/reference foundation is suitable for a separately reviewed synthetic missing-label study within these limits. Calling the whole efficacy study "completely validated" would be premature: the matched training implementation, common-normalization path, restored full-state/RNG behavior, actual penalty controls, independent evaluation plan and expanded resource budget still need validation. Training remains paused. The Windows runtime lacks Gymnasium; this audit's new pure-physics import fixture uses a clearly marked stub only to load definitions. It does not claim a newly completed Gymnasium environment-checker run or CUDA boundary suite. The actual CUDA gate's saved replay/parity evidence is retained as separate evidence.
