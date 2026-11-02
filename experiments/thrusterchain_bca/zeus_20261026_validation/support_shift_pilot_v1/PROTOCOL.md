# Proposed fresh-initialization action-support pilot

Status: exact bounded scope approved by the user at 2026-10-27 03:08 UTC. The executable is frozen separately in PREPARED.json after integration checks. No experiment model was initialized, no attempt was reserved, and no GPU or WSL command was issued during preparation. The prior pilot and every prior receipt remain unchanged.

The question is whether original direct-Q BCA's learned-policy benefit repeats from independent fresh initializations when logged actions have an explicit support gap, against native UniFloral and a simple control with a predeclared training-only strength match. This is the main proposed experiment. Actuator-noise evaluation of saved policies is optional and excluded from this budget.

## Data intervention and identity

Use the same pinned Titan standalone UniFloral source, canonical ThrusterChain, original local uncertainty architecture/objective, Bayesian/conformal calibration, BC term, twin critics, target smoothing, gamma 0.98 and horizon 100. No CORL model/code or failed one-sided revision. Initialize all model parameters and optimizer states afresh; do not restore the broad-trained 20k checkpoint.

The original behavior sampler has support throughout the action cube. Restricting replay after broad-data pretraining would test adaptation, not learning without the excluded action labels. An evaluation action-noise wrapper would test actuator robustness, not this offline-support question.

Preserve original episode roles: 768 training episodes; the original 128 calibration episodes within tuning-validation. Protected selection/final-calibration transition records remain unread. The two conditions are:

- **Restricted:** retain complete recorded tuples with executed float32 first action coordinate `a0 <= 0.5`. This leaves **57,752 rows** across all 768 training episodes, with 62–89 rows per episode. Rewards, successors and finite-horizon masks stay attached to their original actions. Never rewrite an action while retaining its old successor.
- **Broad support, count matched:** for each of those same episodes, select the same number of intact original rows without replacement using fixed PCG64 seed 2026103051. This also has **57,752 rows**, including **14,269 rows above 0.5**. Match the source pool size and episode counts, not just update counts. Both conditions use identical row plans across initialization seeds.

Every derived fitting input must obey its condition: native replay, auxiliary fitting, calibration rows and gradient-matching probes. Rebuild derived banks using the saved DATA_PLAN.json; never reuse broad-support original scale-bank indices or calibrated thresholds in the restricted condition. Source review during integration confirms that original BCA initializes residual scale to 1.0 and updates its EMA from replay minibatches. Preserve that exact behavior: the selected 256-row scale bank is eligibility/provenance evidence only, not an extra residual initialization or fit. All 128 calibration episodes have eligible restricted rows (9,591 candidates, 64–85 per episode); select one eligible row per episode using fixed seed 2026103054. Restricted calibration/probes contain no action above the cutoff. Broad calibration similarly selects one original row per same episode.

Apply the native normalization formula, fitting its statistics only on each condition's retained training observations. Freeze those statistics within that condition across all branches and evaluation. This preserves the preprocessing method; the old checkpoint's numerical normalization is not imported. Action coordinates remain the original [-1,1] scale.

The intervention also changes state/time/reward composition. Retained states originated in trajectories whose earlier actions could be excluded. Therefore this is a **count-matched, action-conditioned logged transition-support restriction**, not a pure action-only causal intervention or newly collected narrow-behavior trajectory dataset.

## Seeds, shared prefix and arms

Use fresh native initialization seeds **2026103001 and 2026103002**, paired across the two data conditions; auxiliary seeds **2026103011 and 2026103012**. These create two independent network/optimizer initializations, not RNG continuations from one trained model. Within each seed, initial parameter values match across conditions before training. Different data pools and their normalization then produce different trained states.

For each seed/condition (four cells), run one shared unpenalized native warmup with matching original auxiliary learning: **10,000 critic, 5,000 actor, 10,000 auxiliary updates**, followed by the original first calibration at critic step 10,000. This retains the original earliest calibration boundary; it is a shorter pilot than the previous 20k warmup and does not claim convergence or exact replication of that training duration.

Fork the complete 10k state into three arms, preserving optimizer moments, targets and RNG pairing. Each receives **2,000 more critic, 1,000 actor and 2,000 auxiliary updates**, with one calibration at absolute critic step 11,000. All arms do the same shadow auxiliary work; only BCA uses its calibrated width in the actor objective. Final checkpoint is fixed at step 12,000; no intermediate online evaluation or outcome-based selection.

1. **Native:** original UniFloral TD3+BC, no additional Q penalty.
2. **BCA:** original direct-Q actor score `Q1 - 3*U`, live action gradient; unchanged native BC term and unpenalized critic targets.
3. **Matched distance:** same simple control definition as the prior pilot, `Q1 - c*D`, where `D = 19.29619572569095 * mean_4((a - pi_anchor(s))^2)`. Anchor is the frozen unpenalized **10k** actor for that cell, explicitly different from the previous 20k anchor. It is not an extra learned penalty or replacement BCA method.

Continuation fork seeds are native 2026103031/2026103032 and auxiliary 2026103041/2026103042 for the two initialization seeds, identical across arms within each cell. Pairing across data conditions means common RNG conventions, not identical source-row identities from different pools. Replay samples 256 rows at each outer step and reuses that batch for two native critic steps, as before.

## Honest strength matching

At its anchor, distance and its live action gradient are exactly zero. No finite coefficient can match nonzero BCA on-policy gradients there. Do not infer strength from scalar penalty levels: an action-independent width offset does not affect actor gradients.

Instead, freeze **one coefficient per 10k base** using one logged training action per episode (768 rows), selected only from that condition's eligible training pool with PCG64 seed 2026103052. With the base's fixed calibrated reference:

`G_B = sqrt(mean_i ||grad_a [3 U_base(s_i,a_i)]||_2^2)`

`G_D = sqrt(mean_i ||grad_a D_base(s_i,a_i)||_2^2)`

`c = G_B / G_D`.

The common detached Q-normalization coefficient cancels for this shared matching reference. Compute the four coefficients before branch training and before any evaluation. Stop on nonfinite gradients, either RMS norm <=1e-6, an invalid calibration, or an unrepresentable coefficient. Never clip, sweep, backfill, rematch at halfway, or tune against returns. Values are deliberately **not yet computed** because no real model execution is authorized in preparation.

This matches RMS action-gradient energy on declared logged actions at one reference. It does not match the initially zero distance gradient at the live actor, actor-parameter gradient directions, curvature, later trajectory gradients, or the calibration refresh's later scale. It does not isolate Bayesian calibration's value from ordinary quantiles or all generic regularizers. Report it as **training-probe gradient matched**, not fully strength-equivalent.

## Evaluation and fixed interpretation

Evaluate the 12 declared final policies plus the four fixed **untrained initial actor references**, once each on the same **32 nominal reset seeds 2026104000–2026104031**, only after all training is complete. Save each cell's untrained parameters before any learner update, with that cell's fixed training-only normalization. These reference actors are predeclared baselines, not selected intermediate checkpoints. Known pinned logged/development/confirmation/prior-pilot metadata show no seed collision; no protected outcomes were read. Use the same full environment action space, original dynamics and reward; no action-noise wrapper and no enforcement of the training cutoff at deployment.

Each actor controls its whole 100-step episode. Primary score is float64 sum of `0.98**t * reward_t`, including the last reward with no terminal bootstrap. Record each actor's requested/executed actions and the fraction above `0.5 + 1e-6`; the restricted replay contains no labels for those actions. Also record excess distance to the support boundary. These descriptive quantities measure actual extrapolation exposure; calibration within the retained support supplies no coverage guarantee outside it.

**Learning-adequacy gate:** the final native policy must improve mean paired return over its own frozen untrained initial actor by more than the same numerical band, separately in all four seed/condition cells. Nonfinite training, invalid calibration, a timeout or a failed native-versus-initial comparison stops advancement. Label a failed comparison an inadequate-learning screen for this budget; do not infer that the full BCA method is disproved and do not extend training based on the outcome. Passing demonstrates learning above initialization, not convergence, optimality or adequacy for a full benchmark.

**Go:** after that gate passes, BCA's mean paired return gain exceeds the existing numerical band (`1e-6 + 1e-6*max(abs(return_BCA),abs(return_control))`, averaged over episodes) against both native and matched distance, separately in each of two seeds and each of two conditions: all eight comparisons positive. This is an exploratory advancement screen, not statistical superiority. If any comparison fails, stop expansion of this fixed specification; no coefficient changes or automatic retry.

Always report per-seed means, paired harms/ties, and the restricted-minus-broad change in BCA's relative benefit. A restricted-condition win alone does not prove that restriction increases BCA's benefit. If neither simple baseline visits the excluded action region in a restricted cell, mark the extrapolation mechanism untested there even if returns improve; do not claim an OOD-action result. Use 2,000 paired episode bootstrap draws with PCG64 seed 2026103091 only as descriptive intervals conditional on final actors, and report the two independent initialization outcomes separately.

## Maximum authorized scope being proposed

| Work | Maximum |
|---|---:|
| Fresh shared warmup states | 4 |
| Final arm policies | 12 |
| Fixed untrained reference policies | 4 |
| Total critic updates | **64,000** |
| Total actor updates | **32,000** |
| Updates of each target network | **32,000** |
| Total original auxiliary updates | **64,000** |
| Calibration calls / posterior draws | **16 / 16,000** |
| Replay row draws, with replacement | **8,192,000** |
| Critic row uses / auxiliary row uses | **16,384,000 / 16,384,000** |
| Strength-matching probe rows | **3,072** |
| Final evaluation episodes / simulator transitions | **512 / 51,200** |
| Training simulator transitions / new data collection | **0 / 0** |
| Global runtime, including setup/compile/evaluation | **9,000 seconds (150 minutes)** |

Repeated replay draws and row uses are not unique data coverage; record realized unique source-row counts. A twin-critic joint optimizer update counts once here, as in the prior pilot. Shared warmups are counted once, not once per branch. Invalid/partial runs retain evidence and stop; no automatic restart or budget extension. Stage processes inherit the global remaining deadline. Individual ceilings: 1,800 seconds per warmup including matching, 300 per branch, and 600 for evaluation; the 9,000-second global limit always dominates their sum.

Cost evidence: the historical Titan 20k warmup took 2,338.96 seconds; the completed Zeus 2k continuations took about 138–150 seconds each. These support a larger bounded stage, not a guaranteed 150-minute completion time. Fresh initialization cannot be established by the cheap saved-policy test. This proposed stage makes it feasible to test two independent initializations under a hard ceiling without claiming full training or a broad benchmark.

## Preparation evidence and remaining work

Verified role-safe counts are in SUPPORT_COUNTS.json. DATA_PLAN.json fixes source-row hashes, probe/calibration/scale-bank row IDs, selection seeds and fresh evaluation seeds. Nine synthetic CPU fixtures passed for intact-tuple filtering, per-episode count matching, eligible-row selection, gradient formula, zero-anchor/degenerate matching, constant-offset invariance and exact budget arithmetic. No real gradients, normalization statistics, model updates, calibrations or simulator calls were computed.

Integration now uses a separate seed/data/budget adapter for fresh initialization (the original frozen wrapper pins one seed and 20k), the training-only coefficient computation, and bounded training/evaluation wiring. CPU fixtures cover native parity, live BCA/distance gradients, reference timing, complete-state serialization, export handoff and accounting. Runtime gates additionally require identical initial parameter hashes across paired conditions, zero optimizer counts, exact filtered-data identities and valid calibration. Actual CUDA execution remains untested until the approved manual launch. Use only this folder's launcher; the previous pilot launcher is not interchangeable.

Assistant-side WSL access remains denied and will not be retried; the approved GPU launch must be manual. No denied attachment access is needed. Nothing has been committed or pushed. The authorization covers this exact 150-minute main pilot, not an optional actuator-noise experiment or expansion.
