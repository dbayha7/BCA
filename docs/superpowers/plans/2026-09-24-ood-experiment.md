# BCA OOD experiment design and implementation plan

**Status: Task A implemented September 27, 2026; Tasks B–D remain pending.
No experiment has been launched.**
The original Unifloral source is now bundled. This document specifies the next
experiment without adding model variants to the four host/BCA pairs.

**Goal:** determine whether BCA's residual width identifies harmful action
extrapolation, while separately measuring residual coverage and whole-policy return.

**Architecture:** read immutable 1M checkpoints through small, read-only adapters.
Freeze candidate actions, state banks, continuation policies and all random streams
before outcomes. Analyze paired outcomes at the training-seed and reset-episode
levels. Keep learner code and the original Unifloral snapshot unchanged.

**Tools:** the existing JAX hosts and MuJoCo/D4RL environments; NumPy/SciPy for
offline statistics; Matplotlib for exportable figures. New experiment code belongs
in one `experiments/ood/` directory, with one proposed `configs/ood.yaml`.

Implementation proceeds locally, without subagents. Checked items below describe
CPU protocol checks only; they do not imply that real experiments exist or passed.

## 1. Questions and permissible conclusions

| Question | Measurement | What a favorable result would support |
| --- | --- | --- |
| Does the score respond to unfamiliar inputs? | Score versus declared perturbation severity and a separately defined support proxy | Sensitivity to those shifts; empirical support novelty, not true density |
| Does the band cover its target? | Fresh-test coverage of the exact frozen host residual | Coverage for that score and sampling population |
| Does width rank harmful actions? **Primary behavioral question** | Within-state action ranking against paired continuation return losses | Conditional harm ranking under the named later policy |
| Does BCA improve the learned policy? | Paired 1M final returns and learning-curve averages across training seeds | A host/BCA recipe difference on the declared tasks |
| Is improvement caused by assigning weights to the right examples? | A separately declared training intervention preserving strength but breaking assignment | A mechanism comparison; the first four rows alone cannot establish it |

Residual error, unfamiliarity and harm are different labels. A large Q/loss is
not an OOD ground truth. A Bayesian bootstrap of residual scores is not a
posterior over the true Q function. An action can be unfamiliar and beneficial,
or familiar and harmful. Retain all four support/harm quadrants in the report.

For a frozen positive global radius and unit, `U(s,a) = R * unit * scale(s,a)`.
Within-state width ordering is therefore the scale ordering. Increasing `R`
cannot repair reversed ordering or action ties. Coverage and ranking must be
reported separately. Both Bayesian and conformal components remain in BCA.

## 2. Scope and comparison identities

**First confirmatory tranche:** TD3+BC and ReBRAC on Hopper-medium-v2 and
Walker2d-medium-replay-v2, using all five seeds in `configs/experiment.yaml`.
These four cells provide both early-termination and locomotion tests and have
direct actor-BC interventions. They are chosen before new results, not by score.

**Expansion, after the same harness passes:** complete IQL/CQL locomotion and
HalfCheetah, then Maze and the three Pen datasets, ultimately the four hosts ×
seven configured datasets. Freeze each expansion's exact checkpoint list and
budget before outcomes. Preserve results from the first tranche; do not recollect
them to create a more favorable expanded table.

| Family | Role and budget | Restrictions |
| --- | --- | --- |
| Root `host` vs `bca` | Primary paired comparison, 1M host/critic updates; TD3/ReBRAC 500k actor updates | Same source family, dataset complement, normalization, seed and evaluation banks |
| IQL paired actors | Host/BCA actors from the same 1M BCA run share one Q/V | This is the primary IQL pair; standalone host is additional context, not a second independent seed |
| `baselines/unifloral` | Exact original baseline reference | Standalone TD3/ReBRAC 1M outer steps imply 2M critic/1M actor updates; original CQL has ten critics and sampled evaluation |
| Published means | Separately labeled reference lines | Never replace with a local candidate or infer a paired/budget-matched win |

The active minimal matrix is 280 physical declarations and 315 actor trajectories.
The 18 completed runs of the earlier campaign are different frozen recipes;
they are not automatically results of this matrix. A missing compatible verified
1M checkpoint stays missing. This plan does not resume the halted queue or the
old IQL Maze/CQL HalfCheetah recoveries.

Unifloral has **native width N/A**. Include its source and original budget in the
reader and report existing verified returns under their original provenance.
Applying a BCA score to its actions would be a cross-policy diagnostic, clearly
labeled with the BCA checkpoint that produced that score. New Unifloral training
or checkpoint instrumentation is not required by the primary experiment.

## 3. Freeze before seeing new outcomes

Create one machine-readable manifest containing:

- BCA Git revision; checkpoint, source, config, data and split hashes; actual
  successful exits; actor/critic/calibrator counters; frozen radius and unit.
- Every seed, collector, capture time, action generator, perturbation level,
  continuation policy, reward definition, horizon and metric below.
- Simulator model/wrapper hashes, observation and action transforms, native
  termination and time-limit rules, target-noise rules and precision/device.
- Disjoint random streams for preparation, collection, candidate construction,
  continuation, fresh calibration, test, bootstrap and plotting fixtures.
- Complete intended matrix with explicit `missing`, `failed`, `pending` and
  `verified` states. No selection of the best seed, checkpoint or episode.

Generate and save integer seeds/keys before evaluation. Hash the candidate arrays
and all continuation streams before the first candidate outcome. A sampled policy
uses the same base random stream for competing first actions within a state;
when the same policy appears in another contrast, reuse the declared coupling.
Different policy functions can map those random numbers to different actions.
Deterministic continuations do not gain independent evidence from identical reruns.

Never use the final bank for selecting an action generator, severity, stopping
rule, radius, checkpoint or score. Engineering fixtures use disjoint seeds and
are excluded from statistical results.

## 4. State banks and controlled action shift

For each cell and training seed, collect **64 independently reset episodes per
collector**, one host-policy collector and one BCA-policy collector. Use paired
reset seeds across collectors. Capture at elapsed steps **0 and 100**. Stop the
collector once capture 100 is reached; its continuation experiment restores that
state independently. Preserve early termination and missing captures without
replacement. A capture at 100 describes survivors at 100, not all initial states.

This produces at most 256 states per cell/seed, in 64 paired reset blocks. The
two collectors' reset-time states may coincide: mark them as shared states,
reuse verified outcomes where identical, and never count them as independent.
Give each collector/capture stratum equal weight when presenting a balanced
summary, and also show each stratum separately with its denominator. Episode-
weighted outcomes and pooled rows are not interchangeable.

At every state predeclare ten action slots, all in normalized action coordinates:

1. Frozen host action, the reference `a0`.
2. Frozen BCA action.
3. Recorded action from the nearest training observation (a support comparator,
   not proof that the action is in-support at this exact simulator state).
4. One uniform action from the valid action box.
5. Six perturbations `clip(a0 ± rho*v)`, with `rho` = 0.05, 0.15, 0.30 and one
   saved direction `v` of RMS norm one for that state.

Record proposed and applied actions, pre/post-clipping RMS distances, clipping
rates, support proxy, width, scale, both radii and consumed multiplier. Duplicated
applied actions remain in the archive but receive one vote in ranking, with slot
aliases retained. Do not choose a direction by maximizing width or observed harm.
The unperturbed actor actions and all perturbation families remain separately
identifiable. These candidates probe local action shift and a broad random
stress case; they do not exhaust the action space.

**Support proxy:** use the training complement only, its existing observation
normalization, 32 nearest reference observations and minimum RMS action distance
to their recorded actions. Fix the reference indices/size before querying tests.
Use 32,768 deterministically sampled rows, or every row if fewer exist; hash the
indices. Thresholds for `support-proximal`/`support-distant` come from the 95th
percentile on an episode-disjoint, training-only reference-validation partition.
Assign recorded episodes by a saved seeded permutation: the first 80% (rounded
down) form the neighbor-bank pool, the rest form reference validation. Require
both partitions to be nonempty. Take one uniformly selected recorded transition
per validation episode, with saved indices. Build the neighbor bank from the
other partition to exclude self-matches and same-episode leakage. If episode
boundaries cannot be established, block the
thresholded analysis; raw distance can still be reported as a qualified proxy.
No test outcome or BCA width defines these labels. Both partitions remain within
the existing training pool; no learner split or normalization changes.

Report score/severity relationships and support-proxy quadrants. Do not call
perturbed actions definitively OOD just because their perturbation is nonzero.
Do not use the same proxy-defined labels to claim that proxy is a validated
novelty detector. True support labels are available only in the synthetic oracle
fixture described in the implementation checklist.

**Separate later shift axes:** changed collector visitation is a state/population
shift; changed mass/friction is a dynamics/conditional shift; changed observation
noise is a measurement shift. None is mixed into the primary action panel.
A dynamics extension must specify its model parameters, feasibility and budget
in a new manifest before outcomes; it cannot borrow a covariate-shift guarantee.

## 5. Behavioral outcome and score comparisons

Restore the same full simulator state for every candidate. Execute the candidate
once and then one of two fixed continuations: **host** or **BCA**. The latter is
the primary harm-ranking panel because BCA's action risk depends on its later
policy. Host continuation is a prespecified sensitivity analysis. Score every
candidate with the same frozen BCA scorer, irrespective of its action source.

For the initial locomotion tranche, define `G_H(s,a;c)` as the **undiscounted raw
reward over at most 250 environment transitions**, stopping at natural termination
or the original remaining time limit. There is no learned tail bootstrap and no
invented post-termination reward. Store discounted return using the exact host
discount as a separate diagnostic. Raw simulator reward and transformed training
reward occupy different fields.

Define `harm(s,a;c) = G_H(s,a0;c) - G_H(s,a;c)`. The primary harmful label is
`harm > 1` raw reward unit; retain `> 0` and the continuous loss as sensitivities.
Show loss distributions in raw units. Threshold 1 has different practical meaning
across tasks, so do not aggregate raw loss or harm prevalence across environments
as if they had a common scale. For later Maze/Pen panels use the original remaining
episode horizon, explicitly bounded and frozen before their launch; otherwise a
short horizon can miss goal occupancy. Those panels form separate task families.

Save forward/alive/action-cost or orientation/goal/drop components where the exact
source supports them. Reward sums must independently reconstruct raw reward from
the **applied control vector**. More reward from longer goal occupancy is not the
same as reaching the goal in more states. Report survival, termination, time limits,
ever-goal and goal-duration separately. Local conditional gains do not explain a
whole-policy learning difference by themselves.

Compare three predictive scores on exactly the same candidate/outcome pairs:

- Full frozen BCA width (larger predicts more harm).
- The fixed support-distance proxy above.
- A constant score and a saved random ordering as sanity baselines.

Also archive bare scale, Q1/Q2 disagreement where defined, and each radius component
as descriptive diagnostics. Twin-critic disagreement is not a Bayesian variance;
IQL shared nuisance and Unifloral's different ensemble size preclude a pooled
"uncertainty" comparison. Observed target residual and realized return are outcome
diagnostics, never pre-outcome predictive scores.

The primary statistic is **mean within-state AUROC for width under BCA continuation**,
excluding `a0` and duplicate applied actions. Half credit for score ties; a state
needs both harmful and nonharmful alternatives. Average valid states within each
collector/capture stratum, then equally across available prespecified strata.
Report missing strata instead of silently redefining an all-strata estimand.
Width-minus-support AUROC is the primary comparative statistic. Also report:

- Valid states / captured states / attempted captures, harmful alternatives and
  prevalence, per training seed and per stratum.
- Within-state average precision, loss-versus-score ranks and a plot of all tied
  action groups. Constant-score AUROC is 0.5 only for a valid two-class state.
- Pooled AUROC in a **separate** column, never substituted for within-state AUROC.
- Risk versus retained fraction when dropping the highest scores. This is an
  offline diagnostic selection rule, not a policy BCA actually executed.

If a seed has fewer than 30 valid states or 30 harmful alternatives, mark its
ranking estimate as sparse; retain it but withhold a cell-level generalization
claim. If no valid states exist, AUROC is N/A, never zero, perfect or evidence of
safety. These thresholds are reporting rules, not power guarantees. Do not collect
until significant or replace the primary threshold after seeing class counts.

## 6. Separate residual-coverage audit

Freeze actor, online/target critics, IQL V, scale, unit, transforms and complete
target-generating code. Use **200 fresh calibration episodes and 500 disjoint
test episodes per cell/seed**, collected by the frozen BCA policy. A saved random
choice selects one transition uniformly from each realized episode, independently
of its values. This defines an episode-weighted transition population. It is not
stationary visitation or a claim that all trajectory rows are IID. Do not reuse
the adaptive training calibration bank as fresh split-conformal evidence.

Audit these radii on the same frozen scores without feeding anything back to the
actor or scale: the existing training-time radius, a fresh ordinary finite-rank
conformal radius, and fresh Bayesian-plus-conformal `max` using the current code.
Keep both components in production BCA; these are post-hoc radius comparisons.

For exact unweighted IID reference scores, report the tolerance-rank diagnostic:
the smallest `k <= n` with `Pr[Binomial(n, 0.9) <= k-1] >= 0.95`; return infinity
if none exists. This stronger conditional-risk reference is not the same claim
as ordinary marginal conformal coverage. Save infinite radii and abstentions.
Never clip infinity to a convenient finite radius.

| Host | Preserve in the residual target |
| --- | --- |
| IQL | Its identified online twin-Q/V residual; V is not automatically the emitted actor's policy value |
| TD3+BC | Frozen target actor/critics and clipped smoothing-noise distribution |
| ReBRAC | Its critic-BC target and exact next-action semantics |
| CQL | Its frozen entropy/backup configuration and target sampling |

**ReBRAC gating detail:** if a live transition lacks the recorded next action
required by the training target, do not substitute a policy action and call the
target unchanged. First specify a continuation-generated next-action population
for both new calibration and test (a separately named operational target), or
mark same-training-target live coverage unavailable. The original training-radius
comparison under that changed population is descriptive. Validate target adapters
against saved training examples before making either claim.

Report miscoverage, width, sample counts, and exact binomial test-bank intervals
conditional on the frozen model/calibration bank. These require independent test
episodes. Episode intervals do not express training-seed uncertainty. Report
each seed and the between-seed aggregate separately. A single calibration bank
cannot empirically establish the frequency of success over new calibration banks.

The existing radius uses unweighted held-out scores. Action-affinity or policy-
density **fitting** weights are not a source/target density ratio at the radius
stage. A weighted shift guarantee would require a separately implemented measure,
overlap assumptions and weight-error analysis. No such theorem is inferred here.

The uniformly selected transition may depend on future episode length. It is
suitable for the stated residual population, not automatically the original
MDP's conditional Bellman operator. An optional center-reliability audit instead
uses first transitions from independent resets, with explicit policy/target and
bins fitted without test targets. Return prediction intervals need their own
return-level target and data; a TD residual band cannot be renamed a Q interval.

## 7. Statistical analysis and decision rules

Pair host/BCA by training seed and pair all scores by identical saved outcomes.
Use a hierarchical paired bootstrap with 10,000 draws and a predeclared seed:
resample training seeds first, then paired reset blocks within seed, keeping
collectors, captures, candidates and both continuations together. Where reset
seeds are shared across training seeds, use common block resampling to preserve
that crossed dependence. Fresh coverage banks use distinct episode IDs per
training seed. Never treat candidate actions or simulator timesteps as independent
replicates. Publish all five seed estimates alongside 95% intervals; five seeds
give limited precision and a bootstrap interval is not an exact small-sample test.

Primary tranche inference comprises two contrasts per cell: width AUROC minus
0.5, and width-minus-support AUROC. Publish their estimates and descriptive 95%
intervals. A simultaneous positive claim requires a predeclared family correction
across these eight contrasts, using conservative Bonferroni-adjusted bootstrap
intervals as approximate intervals, explicitly qualified by five-seed sampling.
If evidence is inconclusive, report inconclusive; do not change the family after
seeing results. Broad cross-task claims wait for the declared expansion.

Whole-policy performance remains a different endpoint: exact reserved final mean
and the mean of the common 5k-through-1M periodic scores, paired by seed. Show
all seed learning curves and separate final episode distributions. Use host updates
on the main x-axis, with actor/per-critic counts visible. Unifloral budget/semantics
appear in a separate panel. Episode variation is never drawn as training-seed
confidence. Secondary aggregate task comparisons may use IQM/performance profiles,
but always retain task-level results and the actual number of independent seeds.

| Observed pattern | Permitted interpretation / next decision |
| --- | --- |
| Residual coverage adequate, harm ranking weak | The calibrated object is not reliably ordering behavioral risk in this panel; changing a global radius cannot fix that ordering |
| Novelty response strong, harm ranking weak | Score detects the chosen support shift more than harmfulness |
| Harm ranking strong, whole-policy return weak | Useful local risk information has not established an effective training intervention |
| Return stronger, ranking weak or unmeasurable | Policy benefit does not establish the claimed uncertainty mechanism |
| Both strong across seeds, but no assignment control | Evidence for utility of this recipe; aligned weighting mechanism still unidentified |

If a mechanism experiment becomes necessary, keep it a separate declaration.
For TD3/ReBRAC/CQL a useful control permutes the **consumed per-example dose**
within each training minibatch while preserving its multiset and generating
permutation keys from a separate stream. Fitting-weight permutation answers a
different question. An equal-current-mean dose removes sample assignment while
preserving that minibatch's average strength, but still allows trajectories to
diverge; it is not an exact fixed-strength historical control. IQL requires an
independent design that respects post-cap AWR shrinkage, masks and shared Q/V.
Do not add these training arms, a tuned constant, new gain sweeps or new models
to the two-method interface just to complete the first OOD evaluation.

## 8. Budget and execution gates

For each initial cell/seed (ceilings, before early termination or exact reuse):

| Item | Maximum environment transitions |
| --- | ---: |
| Two collectors × 64 episodes × 100 steps | 12,800 |
| 256 states × 10 actions × 2 continuations × 250 steps | 1,280,000 |
| One extra first-transition replay per candidate/continuation | 5,120 |
| Behavioral subtotal | **1,297,920** |
| Independent coverage collection: 700 episodes × original 1,000-step horizon | 700,000 |
| Combined ceiling per cell/seed | **1,997,920** |

Initial four cells × five seeds: **39,958,400 environment transitions**, including
coverage. Reserve a separate engineering ceiling of 10,000 transitions per cell,
40,000 total, disjoint from science; grand ceiling **39,998,400**. Training is not
included. Log both environment transitions and underlying MuJoCo integration
steps (`frame_skip`); do not label them interchangeably. Repeated checks and
collector overhead are real resource use. Expansion budgets must use each
environment's actual horizon and frame skip; these locomotion numbers are not
permission to run every extension.

The actor and checkpoint compatibility checks can run before any outcomes. No
training seed is replaced after failure. A failed restore, parity or arithmetic
gate stops the affected stage; preserve its logs, arrays and actual process exit.
Execution corrections use a new attempt identity and are reviewed separately.

Before a behavioral stage may execute, require:

- Every captured field satisfies the **actual chosen restore function**: exact
  field names, shapes and dtypes, including simulator time, qpos/qvel, activation,
  warmstart, controls, applied forces, mutable model/body/site fields, wrapper
  elapsed steps, RNG and any environment-specific state. The function determines
  the full schema; this list is not a substitute for inspection.
- Checkpoint actions match their verified saved exports under unchanged absolute
  `1e-6`. Capture actual/reference/error/device arrays before testing the gate.
  Do not search backend/compiler choices after a failing gate to erase failure.
- Restoring and replaying the first transition reproduces the state/reward under
  declared unchanged state tolerances; source-derived reward reconstruction from
  transformed applied actions passes absolute `1e-7`. Preserve both proposed and
  applied action/dtype and actual simulator controls.
- All candidate actions, scorer arrays and stochastic streams are hashed before
  outcomes. Collector/protocol failures cannot be relabeled as scientific results.
- GPU execution respects one local worker, the existing shared
  `/home/dbayha/bca-work/resource-locks/local-rtx5070ti.lock`, and at most two GPUs
  total. No competing controller or silent retry; verify live process ownership.

## 9. Results the advisor can read

One landing page, **host → dataset → question**, with a completeness table always
visible. Each intended cell links to its report, static figures, CSVs and provenance.
Missing/failed/sparse results get a visible status instead of a broken image or
an empty line. No aggregate panel combines incompatible reward/Q/width units.

Required figures: (1) return curves with all seeds and seed uncertainty;
(2) separate final return distributions; (3) Q/critic/actor losses with precise
query/loss definitions and skipped actor rows excluded; (4) accepted/abstained
scale fits, ESS and consumed doses; (5) both radii, their maximum and frozen unit;
(6) width versus support shift; (7) within-state harm ranking with counts, plus
pooled results separately; (8) risk/retention and reward-component/termination
contrasts; (9) fresh residual coverage and width with radius-rule identities.

Every plot records the manifest hash and number of seeds/episodes/states. Axes
include all data and published reference lines; negative values, ties, infinities
and N/A are explicit. No smoothing for reported metrics. If a visual smoother is
used, show raw curves and label it. Assert plotted arrays equal the source tables,
check every link/asset, then inspect both an actual complete cell and synthetic
missing/failed/zero-harm/tied-score fixtures at narrow and wide screen sizes.
Export PDF/PNG/SVG and CSV per figure so presentation does not depend on a browser.
The existing main thesis PDF stays unchanged.

## 10. Implementation sequence (no launches here)

Keep the implementation small. The configuration, protocol, initial adapters,
oracle and their tests exist; collection, analysis and reporting remain planned:

| File | Single responsibility |
| --- | --- |
| `configs/ood.yaml` | One experiment definition; refer to existing host/dataset configs |
| `experiments/ood/protocol.py` | Validate/freeze matrix, identities, estimands and resource ceilings |
| `experiments/ood/adapters.py` | Read-only checkpoint actions/targets/scales and saved-action gates |
| `experiments/ood/simulator.py` | Audited Hopper/Walker state restore and applied-action reward checks |
| `experiments/ood/oracle.py` | Tiny independent environment with exactly enumerable returns |
| `experiments/ood/collect.py` | Capture states, precommit candidates/streams, execute bounded continuations |
| `experiments/ood/analyze.py` | Offline coverage, paired ranking and clustered statistics |
| `experiments/ood/report.py` | Tables, figures and completeness-aware landing page |
| `experiments/ood/test_protocol.py` | Leakage, wrong-source, wrong-budget and missing-checkpoint refusal fixtures |
| `experiments/ood/test_outcomes.py` | Known support/harm oracle, restore and arithmetic fixtures |
| `experiments/ood/test_analysis.py` | Ties, missing classes, clusters, infinity and plot-data consistency |

### Task A — freeze identities and statistical contract

- [x] Add protocol refusal fixtures before the adapter: wrong checkpoint/config,
  missing counters/actual exit, overlapping calibration/test IDs, adaptive feedback,
  undeclared seed or continuation, changed horizon, and exceeded step budget.
- [x] Implement the single YAML resolver and manifest hash. Keep outcome paths out
  of identity selection. Print both update and simulation accounting.
- [x] Validate all intended rows offline; missing checkpoints remain pending.
- [x] Run `python -m unittest experiments.ood.test_protocol`; require every adverse
  fixture to fail closed and the valid frozen manifest to pass. Commit this unit.

Task A resolves all 20 pairs/40 checkpoint requirements without importing a
training host or querying a model. All rows remain pending until evidence is
explicitly bound. Synthetic metadata tests do not verify actual checkpoints,
event journals, simulator states or scientific outcomes. Accepted metadata alone
cannot promote a row to verified. The future collector must persist the cumulative
resource ledger and precommit actual candidate arrays and continuation step keys;
Task A allocates base seeds and checks ceilings without acquiring a GPU lock.

### Task B — independent oracle and read-only adapters

September 27 execution scope: implement the two initial host adapters (TD3+BC
and ReBRAC), a separate Hopper/Walker simulator adapter, and a tiny exact oracle.
Keep these in `adapters.py`, `simulator.py`, `oracle.py`, with tests in
`test_outcomes.py`. First write refusal/round-trip tests. Reuse the archived
initialization hashes from the earlier CPU parity fixtures, save direct host
forward/target/width references before loading the new adapter, and compare the
restored outputs without optimizer updates. Test missing state fields, action
transforms, termination/time-limit handling and shared continuation keys.

Use the existing WSL CPU environment. Any actual Hopper/Walker smoke execution
is a separately labeled engineering fixture: at most 64 explicit transitions
per environment plus one constructor transition per environment (130 total),
fixed test seeds 1900927001/1900927002, no trained policy outcomes or dataset
collection. Save the test declaration, source/model identities and pre-gate
arrays in a new ignored `runs/ood/` directory; retain failed attempts. Broader
host/environment adapters and real 1M checkpoint acceptance remain later work.
Run `python -m unittest experiments.ood.test_outcomes` in that CPU environment,
then the protocol suite and `python check.py`; publish only code and small
readouts, never fixture/checkpoint weights.

- [x] Implement a tiny deterministic bounded-action fixture with known behavior
  support, exact transition/reward rules and finite-horizon return enumeration.
  Include unfamiliar beneficial actions, familiar harmful actions, constant widths,
  reversed rankings and a correctly ranked oracle. It tests the harness, not BCA
  performance, and adds no learned model family.
- [x] Implement one adapter per initial-tranche host identity, preserving every target
  convention. Compare its outputs against the original saved forward/target
  fixtures. Add Unifloral only as an explicitly named external baseline adapter
  when needed; never infer shared architecture from an algorithm label.
  This unit covers TD3+BC/ReBRAC only; IQL/CQL and external Unifloral adapters
  remain outside the initial tranche.
- [x] Require full collector/restore schema agreement before simulation. Exercise
  deletion of each required field, action transforms, terminal/time-limit state,
  and stochastic-key reuse in negative tests.
- [x] Run `python -m unittest experiments.ood.test_outcomes`; archive all deviations,
  then commit the adapter/fixture unit. Do not count fixtures as research outcomes.

The [adapter test record](../../OOD_ADAPTER_TESTS.md) reports actual engineering
exits and arithmetic checks. The read-only checkpoint API preserves frozen
networks/scale/radii; it does not independently accept a training run. All 40
scientific checkpoint requirements remain pending. YAML placeholder values for
collection-specific adapters/keys/tolerances remain unchanged until those exact
artifacts are bound in Task D. The tested simulator tolerances are exact state
and repeated-observation equality, action `1e-6`, and reward arithmetic `1e-7`.

### Task C — metric and reporting correctness before collection

- [ ] Implement ranking with explicit valid-state masks, duplicate-action aliases
  and half-credit ties. Verify constant score = 0.5 on two-class states and N/A
  on single-class states. Check a fixture where pooled and within-state disagree.
- [ ] Implement finite-rank and tolerance-rank checks independently, including
  infinite bounds and zero/all failures. Preserve all radius components.
- [ ] Verify clustered resampling keeps paired scores, reset-time duplicates,
  collectors and continuations together; it must not bootstrap actions as seeds.
- [ ] Generate the complete/missing/failed/sparse synthetic reader and static
  figures; assert every plotted number against its CSV and inspect the renders.
- [ ] Run `python -m unittest experiments.ood.test_analysis` and `python check.py`.
  Commit the offline analysis/reporting unit; the report must say synthetic.

### Task D — bounded engineering validation, then experimental handoff

- [ ] Inventory compatible verified 1M checkpoints without retraining/relabeling
  old runs. Bind exact hashes and mark missing cells. Provision and record the
  actual environment/driver/model identities; a requirements file alone is not
  fresh-environment reproducibility evidence.
- [ ] Freeze the engineering manifest, maximum 10,000 transitions per initial
  cell, and its disjoint seeds before a separately instructed execution. Save
  pre-gate action arrays, collector fields and first-transition arithmetic.
- [ ] Preserve failures. Accept the harness only after all scientific/numerical
  gates pass unchanged; document any execution correction separately.
- [ ] Freeze the scientific manifest and exact resource ledger; present the
  runnable commands and expected output layout as the execution handoff. There
  is no automatic dispatch or queue resumption from this design document.
- [ ] Following explicit execution direction, collect once, verify actual exits
  and all rows, analyze without outcome-dependent changes, inspect every final
  figure, then publish source/readouts excluding checkpoint weights. Preserve
  incomplete outcomes and do not claim a full grid from a pilot.

## 11. Research basis and boundaries

These papers motivate distinct checks, not a license to merge their guarantees:

- [Tibshirani et al., Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019v3):
  weighted conformal inference needs the appropriate change-of-measure weights
  and shift assumptions. Our fitting affinities do not establish those conditions.
- [Lou and Luo, Weighted Bayesian Conformal Prediction](https://arxiv.org/abs/2604.06464v3):
  the deployed-risk posterior is conditional on its weight function; weight
  estimation and conditional shift have separate qualifications. Preserve tail
  treatment and Monte Carlo uncertainty when auditing any stronger risk claim.
- [van der Laan and Kallus, Bellman Calibration](https://arxiv.org/abs/2512.23694v2):
  center reliability is a separate criterion from residual width and value error.
  The first experiment adds no post-hoc critic correction.
- [Gan et al., Conformal Prediction Beyond the Horizon](https://arxiv.org/abs/2510.26026v1):
  return prediction requires an explicit return/tail and policy-shift construction;
  it is not supplied by renaming a one-step residual band.
- [Agarwal et al., Statistical Precipice](https://arxiv.org/abs/2108.13264v4):
  few-run RL comparisons need uncertainty and transparent task-level/aggregate
  summaries. This plan uses paired seed-level results rather than episode counts
  as apparent independent training replications.

The supplied conservative-value-prior, quantile-Q and Bayesian-flow papers suggest
other value or generative models. Adding them now would change the object under
test. They remain separate research directions rather than extra host variants.
