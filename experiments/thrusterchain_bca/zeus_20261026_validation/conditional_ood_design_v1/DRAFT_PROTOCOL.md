# ThrusterChain conditional OOD design, draft v1

Status: **design only, not an executable or an approved experiment budget**. Prepared on 2026-10-27 in the Zeus checkout at `50099436611b35df377923c5d9c508caafe65094`. This document preserves all earlier protocols, attempts and results. No training, simulation, dataset modification, installation or new result calculation is part of this draft.

The question is whether original BCA improves policy-specific value accuracy, action decisions and learned-policy return when particular actions are missing at otherwise represented states. The target is **Q^pi, not Q-star**. Readiness, a valid exposure design, and a frozen comparison come before method revisions. The proposed geometry and numerical design choices below have not passed a new coverage audit.

## The five components at a glance

| Component | Concrete artifact to prepare before execution | Validation gate | Advisor question |
|---|---|---|---|
| 1. Original distribution and coordinates | `source_distribution.json`, `episode_roles.json`, source hashes | Exact host, physical task, sampler, action conversion and episode roles identified | What distribution and value function are we studying? |
| 2. Conditional withheld region | `region_definition.json`, immutable source-row eligibility masks, `coverage_before_after.csv` | Positive original coverage, zero retained labeled tuples inside the hole, consistent exposure rules | Is this an actual conditional support hole rather than a label attached to arbitrary test points? |
| 3. Alternatives and controls | `control_row_plan.json`, `conditional_support.csv`, `balance_report.json` | Same retained counts within fixed state/time strata; supported alternatives and residual imbalance quantified | Does an effect reflect missing action information rather than fewer samples or changed states? |
| 4. Independent test distribution | `test_design.json`, reserved seed ledger, shared probe bank | Train/test episode separation, geometry-only labels, separate current/action factors, near/far membership | Does the effect persist on independently drawn tests and specified shifts? |
| 5. Validation and outcomes | `evaluation_contract.json`, complete per-probe table, curves, accounting | Correct Q^pi references, complete signed-error accounting, paired controls, no evaluation-driven tuning | Does BCA improve accuracy, decisions and actual return? |

These are planned artifact names, not files already generated. `SOURCE_RECEIPT.json` beside this document records the source files actually read and their hashes.

## 1. Original data-generating distribution and coordinates

### Established source facts

- Host: standalone UniFloral TD3+BC revision `f2dc1278eae18ed3e22c92255119c54885414d17`; native source SHA-256 `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee`. The earlier CORL-derived host is not a substitute.
- Canonical plant: 13 carried float64 coordinates `(p1,v1,...,p5,v5,c,w1,w2)`. Networks receive float32 observations of these coordinates plus remaining fraction `(100-t)/100`, for 14 inputs. Four executed action coordinates lie in `[-1,1]`.
- Actions are requested in float64, cast to float32, then used by the environment. Region membership uses the **executed** command. Preserve both representations and count boundary disagreements.
- Recorded starts: ten oscillator coordinates drawn as `0.5 * N(0,I)`; current `c ~ Uniform[0.1,0.7]`; wave amplitude `A ~ Uniform[0.1,0.4]`; phase uniform on `[0,2*pi)`, with `(w1,w2)=A*(cos phase,sin phase)`. The exact generator's validity checks remain active; no silent clipping or rejection resampling.
- Current is observed and constant during an episode. Waves rotate by 0.27 radians per step. The stall world has knee `k=0.800000011920929`, slope 1.5, gain 0.3 and throttle cost 0.3. Transition and reward have no process noise. Reward uses the pre-transition state and executed command.
- Finite horizon 100, discount 0.98, terminal reward 0 and no bootstrap beyond the final transition. Preserve exact simulator state and time for reference-value replay; rounded observations alone are insufficient for exact replay.
- The existing dataset has 1,280 episodes, root seed 2026102301 and SHA-256 `4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226`. Its metadata confirms behavior epsilon 0.25: a state-independent uniform requested-action law on the four-dimensional cube, density `1/16`. The implementation expresses this as masses 0.25 and 0.75 on volumes 4 and 12 split at `a0=0.5`.

The exact requested-action density is not an executed float32 mass function, state occupancy density, or joint state-action density. Those latter quantities are not supplied by the collector. Random Gaussian starts do not make the full rollout distribution Gaussian.

### Roles and proposed reuse

Preserve existing episode membership: 768 train episodes (76,800 transition rows), 256 tuning-validation episodes, 128 selection episodes and 128 final-calibration episodes. The earlier BCA workflow used an identified 128-episode subset of tuning-validation for calibration. Reuse only those identified calibration IDs if the source-role audit confirms them; do not equate them with the separately protected `final_calibration` split. The other tuning episodes remain unassigned until that audit. Protected selection/final-calibration transition rows remain unopened.

For a development pilot, the 768 training episodes can supply all conditions using immutable row-index views; auxiliary fitting shares the eligible training pool. This is already-inspected development data, not a newly untouched confirmation experiment. Fresh test episodes must use independently reserved random streams after design freeze. If a later confirmatory claim needs fresh data-collection seeds, that is a separately budgeted stage.

`source_distribution.json` must bind native source, environment and matrix hashes, behavior law, dataset content/hash, action conversion, seeds and role IDs. Its gate rejects mismatched sources, overlap between fitting and calibration episode IDs, ambiguous time/terminal handling, or use of protected records.

## 2. State-conditional withheld regions and retained coverage

### Proposed primary geometry

Define observed load `g(o)=float64(o[10])+float64(o[11])`, the current plus the first wave coordinate. For half-width `h`, define

`R_h(o,a) = [g(o) >= 0.4] AND [abs(float64(a[0]) + k) <= h]`.

Propose `h=0.05` for the primary design; reserve `h in {0.025,0.05,0.10}` as explicitly declared severity levels, not an automatic sweep. Other action coordinates remain unrestricted. This removes a band around negative first-thruster stall-knee commands at higher observed load. The same action band stays eligible at lower load, and other first-thruster values stay eligible at higher load. Negative commands oppose positive loading, but the plant's coupled state means this does **not** assert that the band contains the optimal action at every state.

The threshold 0.4 and widths are **proposed, unvalidated choices**. Their rationale is observable plant geometry; they were not chosen using Q errors, penalties or returns. Freeze their float representations and inclusive boundary rule. Audit original training geometry before freezing the experiment. If coverage is insufficient, report the failed design and version a new proposal; do not silently search regions for favorable BCA outcomes.

For source row `i`, set `keep_h[i] = not R_h(o_i,a_i)`. Retain the complete tuple `(o_i,a_i,r_i,o_next_i,terminal_i,t_i,episode_i,row_i)`. Do not replace an action while keeping its old reward or successor. Keep excluded tuples in the immutable original source, inaccessible to fitting paths; they are not an independent test set.

### Exposure contract

| Consumer | Permitted input under the hole condition |
|---|---|
| Critic replay and actor BC term | Retained original training tuples only |
| Auxiliary scale fitting and residual-unit updates | Same retained training eligibility; original residual target and update method |
| Calibration | Separate identified calibration episodes, with the same region mask; one eligible row per episode under a fixed outcome-independent selection rule |
| Penalty-strength matching | Retained training probes only; no test values or return-based coefficient choice |
| Learner normalization | Fit the unchanged native formula on retained training **current observations** only; freeze within the data condition across methods/checkpoints |
| Audit standardization | Separate common transform fitted to original training-state observations only; never fed into the learner as a replacement normalization |
| Native actor and bootstrap queries | May query arbitrary feasible actions, including the hole; these are model predictions, not restored logged labels |
| Evaluation | Independent frozen probes/episodes may enter the hole; values and trajectories never return to fitting, calibration or matching |

Do not reuse a broad-trained model, normalization or calibrated threshold as initialization for a claim about learning without the excluded labels. Fresh paired initial parameters are required for that claim; warmup, if used, must obey its data condition from its first update. An unavailable or infinite BCA reference remains a refusal/unready state; never silently replace it with a zero penalty.

Tuple filtering does not generate avoidance-policy trajectories: retained states can depend on previously excluded actions, and retained successors remain valid even when later actions would be excluded. This intervention removes observed conditional labels, not all causal influence of those actions or all model queries there.

`coverage_before_after.csv` records original/retained/excluded rows, distinct episodes, state/time strata, terminal rows, boundary cases and eligible calibration episodes. Require zero forbidden logged labels in every fitting/calibration/matching bank. Require positive original hole coverage and retained alternatives; proposed minimum sample thresholds remain subject to the geometry-only audit. A calibration episode with no eligible row is an explicit design failure, not silently replaced.

## 3. Supported alternatives and matched controls

Use three data conditions: original full training pool; conditional-hole pool; and state-stratified random-removal pool. The full pool establishes the effect of any removal. The random-removal pool has the same retained count as the hole pool but removes rows independently of actions within fixed strata.

Proposed strata are `(episode_id, ten-step time bin, high/low g)` with the load boundary 0.4. For each stratum, uniformly sample without replacement exactly the number retained by the hole mask, using a frozen independent RNG address. Preserve zero-retention strata and report them. This exactly matches those coarse counts, not all continuous state coordinates, rewards or successor distributions. Report remaining standardized state/time/reward differences. Never match using Q errors or future returns, and never claim a perfectly isolated action-only causal effect.

Apply the analogous action-independent count-matching plan within the separate calibration episodes; then select one row per eligible episode. All methods within a data condition share its rows and normalization. Auxiliary work and calibration budgets are matched across method arms where the implementation permits shadow work. Match any simple penalty only on eligible training data. The existing frozen-anchor distance control has zero gradient at its own actor anchor, so a logged-probe RMS match cannot establish equal on-policy strength; retain that limitation rather than invent a new control.

At common probe states, compare candidate actions inside the band, just below it, just above it, and farther away while holding the other three action coordinates fixed. For random banks, sample those three coordinates from the original uniform law; for policy-action diagnostics, label the anchor separately. Include actual actor actions without selecting states based on BCA benefit. Analytic eligibility alone does not establish empirical support.

For support diagnostics, use one common diagonal standardization of the 13 physical observation coordinates fitted only on original training rows; constant-coordinate scale is one. Match the same ten-step time bin and load side. Proposed neighborhood settings: `k_neighbors=64`, maximum standardized RMS distance `rho=0.5`, with at least 32 rows from 8 distinct episodes to call a neighborhood sufficiently populated. These thresholds are provisional, not existing measurements; otherwise label support **unknown**. Publish the 64th-neighbor distance even when the gate fails.

For every eligible neighborhood and action band, report original/retained row counts, distinct episodes, nearest joint state/action distance, and action-neighborhood counts at proposed L-infinity radius 0.05 across all four actions. Report zero observed counts separately from a structural exclusion under `R_h`; finite samples cannot prove support absence outside the mask. Same-state candidate comparisons plus neighborhood evidence support conditional interpretation without asserting exact repeated continuous states.

`control_row_plan.json` and `balance_report.json` bind sampling seeds/IDs, exact stratum counts and imbalances. `conditional_support.csv` records all probes, including unsupported or unknown cases. No density estimator is required. A Gaussian model could bridge the deliberate hole and must not override its geometric label.

## 4. Independent tests, action/current factors and severity

Create independent episode streams for development exposure checks and final reporting, disjoint from training/calibration and previously consumed experiments. Freeze the seed ledger after reconciling the missing Titan manifests; no seed values are assigned by guess here. Propose independent test-episode families for each training seed, with the same starts and candidate banks paired across methods and checkpoints within that seed. Each episode is independent of other episode draws conditional on the generator, while multiple states, candidates and checkpoints within it are repeated measurements. If a later frozen design instead shares test families across training seeds, record that crossed dependence explicitly.

The core comparison crosses the following factors:

| State distribution | Outside the proposed action band | Inside the band at high observed load |
|---|---|---|
| Recorded current `Uniform[0.1,0.7]` | ID/reference probes, subject to local-support checks | Conditional action-gap probes |
| Strong current `Uniform[0.7,1.1]` | Current-shift probes | Combined current shift and geometric action-band probes |

Keep wave amplitude `Uniform[0.1,0.4]`, kick SD 0.5, action conversion, plant, reward and horizon unchanged. Strong current is an **observed state-distribution shift**, not changed hidden dynamics. At strong-current states, the action-band mask is an extension of the geometry: those states were already outside recorded current support, so do not claim their exact state-action labels were selectively removed from an originally covered population. Disentangle all four conditions rather than pool them as one OOD result.

Use independent starts with a common, method-neutral reference behavior to obtain diagnostic state banks at proposed times `{0,25,50,75}`. The default proposed behavior is the original uniform recorder. This produces shared occupied states without using BCA outcomes; any simulator cost must be included in the future budget. Keep whole-policy evaluation separate, since trained actors generate different state occupancies. A geometry-only development exposure check must confirm enough high/low-load states and useful candidate spread before final banks are frozen; outcome labels do not determine which probes survive.

Predeclare near/far strata. Inside the hole, define normalized depth `d_in=(h-abs(a0+k))/h`: near boundary `0 <= d_in < 0.5`, deeper interior `0.5 <= d_in <= 1`. Outside, classify the immediate ring `h < abs(a0+k) <= 2h` separately from more distant alternatives. Record volumes and counts; handle float32 boundaries explicitly. Current severity divides the existing strong-current interval into proposed near `[0.7,0.9]` and far `(0.9,1.1]`; report the shared 0.7 boundary separately. These are geometry strata, not evidence of equal difficulty.

Borrow covariance and sample-count sensitivity as **separate planned axes**: a half-sized training-episode subset versus the full available pool, and a doubled kick SD (0.5 to 1.0, variance multiplied by four) versus the ordinary reset. Both are provisional secondary designs requiring their own budget; do not silently form a Cartesian sweep with hole widths/current shifts. Sample subsets are whole episodes with paired masks; calibration roles stay separate. A kick-covariance change is another observed start-distribution shift, not a change to the plant. Increasing trained-seed replication is desirable for uncertainty but does not authorize a 20-seed sweep.

`test_design.json` binds all distributions, proposed severities, geometry, requested/executed actions and independent RNG streams. Its gate checks no split/seed collision, feasible candidate actions, complete strata and no outcome-driven filtering. No policy entering the gap during free rollouts means the free-policy OOD mechanism is untested, even when forced probes provide diagnostics.

## 5. Validation, value errors, decisions and learning curves

For exact probe `(x,t,a)` and a frozen checkpoint actor `pi`, define

`v^pi(x,t,a) = sum_{j=0}^{99-t} 0.98^j * r_(t+j)`,

with the candidate executed once, then that same actor, exact state/time restoration, pre-transition rewards and no terminal bootstrap. Current deterministic continuations have no Monte Carlo transition noise; uncertainty across starts and trained seeds remains. A smoothed twin-min TD target is not this reference.

Keep two panels separate:

1. **Same-critic adjustment:** compare `q=Q1` and `z=Q1-kappa*U` against the identical `v^pi`. Original BCA uses proposed fixed `kappa=3`, live action gradients in its actor objective, unchanged BC and native critic targets. This measures the score's effect, not a change to stored critic predictions.
2. **Separately trained native/BCA/control policies:** each critic is evaluated against its own actor's `v^pi` on common probes. These measure accuracy relative to different continuation policies; their difference is not a correction against one common target. A common-policy reference would be a separately labeled additional diagnostic, not an implicit substitution.

Propose fixed checkpoint reporting at critic updates `{0,10000,15000,20000}`, aligned with the existing original calibration schedule at 10k/15k/20k. Record exactly whether the checkpoint is after a refresh. Before reference readiness, show raw critic/return metrics and BCA score unavailability. Evaluate every declared checkpoint after all training is frozen; never choose the best checkpoint or extend training using the curves. This grid and the 20k endpoint are proposals, not a budget authorization or a convergence claim. The final method/seed count and compute ceiling must be approved separately.

Every probe row records condition, severity, train seed, checkpoint counters and hashes, actor/reference identity, normalization, source episode/state-family ID, exact state/time, requested/executed action, geometric labels, support metrics, `Q1,Q2,U,kappa,z,v^pi`, and calibration/refusal status. Use common candidate/tie definitions across methods.

For each prespecified stratum and pooled population report:

- `e=q-v`, `e_adjusted=z-v`; over/near-zero/under counts, denominators, signed mean, MAE, RMSE and conditional magnitude means/quantiles. Proposed numerical class tolerance is `tau(v)=1e-4+1e-5*abs(v)`, pending precision validation before new labels; it is not a practical-benefit threshold.
- Paired absolute-error change `Delta=abs(e)-abs(e_adjusted)`: improved/worsened/tied counts and magnitudes under a separately frozen numerical comparison rule. Keep all probes, not only initially overestimated cases.
- The full 3-by-3 sign-transition table, including over-to-under crossings and departures from near-zero; positive error removed alongside underestimation added.
- Fixed-bank regret `max_a v^pi(s,a)-v^pi(s,selected_a)`, choice switches, gain, harm and ties versus raw Q and simple controls. This is regret within the candidate bank under that continuation policy, not regret to Q-star.
- Whole-policy discounted returns under ordinary/strong-current starts, learning curves, and actual state/action-hole exposure. Forced candidate actions are not policy return evidence.

Require accounting identities and synthetic arithmetic fixtures before execution: kappa zero produces equal scores/errors; for nonnegative widths subtraction cannot turn underestimation into overestimation; an already underestimated raw score worsens in absolute error by exactly `kappa*U`. Invalid/nonfinite/refused cases retain IDs and denominators with availability explicitly reported. Residual-calibration coverage, empirical Q^pi accuracy and selected-action validity are different claims; calibration does not automatically guarantee either OOD or selected-action coverage.

Freeze aggregation weights before outcomes: equal weights across independent training seeds; within each declared stratum, equal episode weights, then equal scheduled-state weights within an episode and equal candidate weights within each candidate family. Report strata separately; any pooled diagnostic summary uses equal weights across the declared nonempty strata, with empty strata reported as a design failure rather than silently dropped. Actor-action rows form a separate family and do not gain weight from the number of forced candidates. An intentionally balanced or oversampled hole bank measures accuracy on that diagnostic design, not recorded-distribution or deployed-policy accuracy. Whole-policy returns average complete episodes under the declared reset law separately.

Use trained seeds as method-level independent units, with the proposed independent test episodes/state families nested within seed and paired across methods. If test families are reused across training seeds, they are crossed/shared test units: preserve both episode identity and seed pairing in intervals, or report intervals conditional on that fixed bank. Candidates, checkpoints and posterior draws are not additional trained replicas. Report per-seed curves and effects; any interval must preserve the actual dependence. Readiness/learning-adequacy, mechanism and efficacy decisions remain separate. Insufficient exposure or learning is inconclusive for the intended mechanism, not a license to tune or automatically rerun.

## Missing decisions and next authorized step

1. Import the missing Titan manifests/protocols/reports and reusable model identities before deciding which completed work already answers a component. Keep held-out outcomes sealed for selection. Required named evidence includes explicit OOD diagnostics, direct held-out BCA, timeout recovery, adaptive-signal records and the stall-boundary assay. Do not rerun them to fill a transfer gap.
2. Review the proposed load threshold, primary width, severity strata, support thresholds and checkpoint grid. A later authorized metadata/geometry-only audit must measure original and retained coverage; this draft contains **no new measured conditional-hole counts**. The old 57,752 retained rows refer only to the different global cutoff and are not evidence for this mask.
3. Freeze exact eligible source-row IDs, calibration episode IDs, random-control plan and test seeds without inspecting new evaluation outcomes. Establish an explicit numerical precision/tie rule and practical effect criterion.
4. Specify and approve training seeds, final method arms, checkpoint/probe/episode counts, calibration schedule, simulator/optimizer counts, runtime ceiling and terminal/resume rules. Include every sensitivity axis and reference-value rollout in the budget; nothing launches from this document.
5. Verify Zeus runtime through permitted access. The historical WSL denial remains; general design authorization does not bypass it.

## Sources and limits of the borrowed design

Local evidence: `titan_20261026/payload/transfer_bundle/core_v4/contracts/environment_contract.json`, `scenario_registry.json`, `core_v4/docs/regular_mode.md`, `core_v4/thrusterchain/datasets.py`, canonical `thrusterchain.py`, original `unifloral_checkpoint_v1/training.py`, and the earlier support pilot's protocol/count receipts. The dataset read for this draft was its scalar manifest only; no protected transition arrays or new outcome labels were decoded.

Conceptual borrowing follows the supplied read-only review of [bnp4ood's synthetic notebook](https://github.com/rwl93/bnp4ood/blob/main/SyntheticExperiments.ipynb), [tied_dpmm.py](https://github.com/rwl93/bnp4ood/blob/main/bnp4ood/tied_dpmm.py), [rmds.py](https://github.com/rwl93/bnp4ood/blob/main/bnp4ood/rmds.py), and [the paper](https://arxiv.org/html/2502.08695v1): independent ID test draws, training-only preprocessing, explicit severity/sample-size/covariance axes and seed-level replication. These are design principles, not imported algorithms or new performance claims.

That synthetic benchmark's novel Gaussian component/prior-predictive provenance differs from deliberately withholding a conditional action region. Overlapping full-support Gaussians do not establish the same hard support-hole truth. Its posterior-predictive/prior-predictive inlier score is not signed Q error; an optional AUROC would need an explicitly stated positive class (the reviewed benchmark uses ID-positive). No BNP model is installed, fitted or added to BCA here.
