# BCA action-support robustness v2

**Goal:** Directly test whether standard BCA earns better outcomes after the same unfamiliar action, whether its own first-action choice is better, and whether its warning score identifies harm.

**Architecture:** A separate, prospective, support-stratified experiment. Preserve the completed v1 study and frozen sources. Pure candidate selection and contrasts live in `work/ood_robustness_v2`; a future versioned execution layer must pass new engineering, identity and cumulative-resource gates before dispatch. No delegation or scientific dispatch is part of protocol preparation.

**Tech stack:** Python 3.10, NumPy, immutable JSON manifests, existing frozen checkpoints and simulator gates. No model, simulator, checkpoint loading or ledger writer in preparation code.

## Timing and scope

David requested this follow-up after inspecting the first-seed v1 results on September 28. This protocol is specified after those results, before any v2 outcomes. It is not the original preregistration. Existing support-near/distant v1 rows can supply a clearly labeled post-hoc bridge; they cannot validate prospective v2 performance or determine its thresholds, pool sizes or endpoint choices.

The original v1 width/harm endpoint, reports and all completed checks remain unchanged. Future unstarted v1 OOD collection is held in favor of this version; this does not stop or alter training queues. Keep the original planned first tranche: TD3+BC and ReBRAC, Hopper and Walker2d, seeds 202609171 through 202609175 (20 pairs). Only the first TD3+BC Hopper pair currently has accepted paired checkpoints. A missing or failed declared seed is not replaced. All five are required for each cell's replicated conclusion; no broad BCA claim from one cell.

## What the comparisons answer

For a full state s, imposed first action a and frozen continuation c, G_c(s,a) is the raw undiscounted return over 250 total transitions including the imposed action, preserving native termination/time limit. Let a0 be the nearest-recorded support anchor, identical for both continuations.

1. **Primary behavior under action stress:** A(s,a) = G_BCA(s,a) - G_host(s,a). Positive means higher measured return after exactly the same imposed action.
2. **Co-reported degradation contrast:** L_c(s,a) = G_c(s,a0) - G_c(s,a); D(s,a) = L_host(s,a) - L_BCA(s,a) = A(s,a) - A(s,a0). Positive means BCA loses less relative to its outcome from the same familiar first action. Do not clip negative losses: an unfamiliar action may help. D alone does not establish better stressed performance; a worse BCA familiar-action return can make D positive. Report A, D and both familiar returns together.
3. **Policy first-action quality:** G_c(s,a_BCA) - G_c(s,a_host), separately for each fixed continuation c. Label both actions by support and measured harm. Frequency of choosing a support-distant action is not itself a failure; harm is separate. This measures a single decision at a captured state, not whole-policy deployment safety.
4. **Secondary warning quality:** Frozen width and support distance versus measured host-reference harm >1, separately for each continuation, with BCA continuation retained as the primary harm readout. Within-state AUROC/AP and pooled AUROC stay separate. Compare width, support distance, fixed random and constant scores. Report both the balanced mixture and each distance band; otherwise support separation could explain a nominal ranking advantage. Ties get half credit; single-class panels are N/A.

The strong-support-distance band is the primary stress group. Moderate and near groups describe severity and nominal behavior. Never use each continuation's own best tested return as the primary robustness benchmark. V1 finite-bank regret remains an exploratory secondary result.

## Action support and state support

These are empirical support proxies relative to the accepted offline training data, not exact distribution membership or proof an action is outside true mathematical support. Environment-valid actions can be poorly represented at a particular state.

Reuse the accepted training-only, episode-disjoint reference/calibration split and native observation normalization, with at most 32,768 reference rows and 32 nearest reference observations. Action distance is the minimum RMS native-action distance to those 32 actions. Calibrate q95 and q99 with linear empirical quantiles of the recorded calibration actions' distances. No held-out outcome, width, critic or return is used. Freeze both thresholds and the calibration counts. Require finite 0 < q95 < q99; if unavailable/degenerate, stop that cell's preparation and report it. Never move boundaries to fill quotas.

Bands: near d<=q95; moderate q95<d<=q99; strong d>q99. Near means support-near by this proxy, not guaranteed ID. Independently compute nearest-reference-state RMS distance in normalized observation coordinates and its calibration q95. The primary action-shift comparison uses state-support-near captures only. State-support-distant captures form a separately labeled joint state/action-shift sensitivity analysis. Do not silently pool them.

## Fresh state and candidate banks

Use 64 paired reset episodes under both frozen collectors, capturing at steps 100 and 300: four collector/capture strata, 256 nominal captures. Preserve early termination/missing captures; no replacement or score-based selection. These later captures reduce reliance on the special reset distribution. Use fresh domain-separated streams derived from a fixed protocol namespace, host/environment/training seed, reset-block index and purpose. Save actual reset seeds, candidate seeds and all 250 JAX continuation keys before outcome collection. Check fresh streams against prior declarations; any collision is an engineering failure, not silently resampled.

At each captured state create a fixed 8,192-proposal pool: 4,096 uniform native-box actions and 4,096 perturbations around the nearest-recorded anchor, interleaved. Perturbations cycle RMS scales .01,.025,.05,.10,.20,.40,.80,1.60 with independent RMS-unit Gaussian directions. Clip once, cast to float32, and apply the actual native action transform once. Select by generation order within bands after exact applied-action deduplication, with no width/critic/return access.

Collect at most 14 nominal slots: anchor plus three other near actions, four moderate, four strong, and the host/BCA native choices. Policy slots are not selected for support distance. Exact policy aliases reuse the same recorded trajectory and retain slot attribution; do not call physics twice or count them as independent. Record proposed/sent/applied coordinates, clipping, distance, band, proposal index and aliases before scoring. Distances use applied coordinates. If any quota is unfilled, keep the missing slots visible, do not expand the pool and do not replace the state. Such states remain in coverage accounting; the complete-band comparison is explicitly conditional on feasibility. No blanket OOD claim if required groups are unavailable.

## Dependence, aggregation and interpretation

Within each state average the four fixed stress actions per band; treat all actions, both captures, both collectors and both continuations from one reset block as dependent. For each training seed report each of the four stratum means and their equal-weight mean. Empty strata make the balanced estimate N/A; never silently reweight. Report all state-support exclusions, missing states, quota failures and native-policy aliases by stratum and seed.

For each host/environment cell show all five paired seed estimates for A_strong and D_strong, the mean, and a paired-seed t interval (df=4) clearly labeled approximate with only five independent training seeds. Actions are never the independent n. To support simultaneous directional claims across four cells and these two endpoints, use Bonferroni 99.375% two-sided intervals for each of the eight cell/endpoint means; show ordinary 95% descriptive intervals separately. A cell-level claim of better action-stress robustness requires both adjusted lower bounds >0, complete declared seeds, and disclosed near-action results/coverage. Otherwise describe the actual pattern, including smaller degradation but lower absolute stressed return. Do not average raw rewards across environments. Any cross-cell generalization must state exactly which cells support it; fresh residual coverage remains separately unavailable.

No tuning after new outcomes. No optional stopping, new seed, action replacement, tolerance weakening or outcome-based checkpoint selection. Keep action absolute 1e-6, reward 1e-7 and full-state first-repeat gates.

## Resource envelope and execution gates

The old audited ledger remains immutable at 1,298,353 environment /5,193,412 physics reservations, including unknown calls and all interruption overhead. Existing first-pair outcome/repeat caps are full; it cannot be reused for v2. Preserve header, 1,298,293 entries and final head `6bfd6380329cf7ea4ac204b80edf98fae8593c7da79411809b0132bfc3bc152c`.

New maximum across 20 pairs: 35,840,000 outcomes +143,360 first-repeat checks +768,000 state-collection steps +40,000 engineering steps =36,791,360 environment /147,165,440 physics steps. Combined maximum is38,089,713 /152,358,852, below the original global ceiling39,998,400 /159,993,600. Remaining1,908,687 environment steps are unallocated, not permission for concurrent old collection or coverage. Short trajectories, aliases and missing slots reduce actual reservations without enlarging any cap.

A separate extension ledger must bind the exact prior ledger head/totals/declaration, enforce combined global ceilings, refuse changed ancestors and reserve before every physical call. It cannot mutate, reset or refund the original ledger. All execution shares the original local lock, CPU/zero GPU for this OOD phase, at most one local scientific worker, no automatic retry. New source/plan hashes, checkpoint/data audits, native-control parity, simulator restore, runtime, stream uniqueness, cap arithmetic and failure/actual-exit handling require separate acceptance. Save full final-state contents and constructor evidence in this version; preserve v1's unavailable evidence honestly.

## Implementation checklist

- [ ] Freeze this protocol before any new outcomes or post-hoc bridge values.
- [ ] Implement pure support-band selection, fresh stream derivation and common-anchor contrasts; known-answer/refusal tests include the own-benchmark ceiling trap, alias/missing-band behavior and exact band boundaries.
- [ ] Perform saved training-bank feasibility only, with external hash binding; no model query or simulator. Record threshold/quotas, not scientific outcomes. Do not tune the protocol from old outcome values.
- [ ] Produce an explicitly post-hoc, file-bound v1 bridge using accepted observations and nearest-recorded anchor; unchanged v1 report is the source, never a new confirmatory result.
- [ ] Implement/test separate extension ledger and v2 execution harness, all engineering/precommit gates and actual exit supervision before any dispatch. Independent review must check all limits and lack of outcome selection.
- [ ] Accept each declared trained pair independently, then collect once under the frozen v2 declaration. Missing pairs remain pending; do not restart failed training from this plan.
- [ ] Independently audit saved outcomes, return arrays, controls, states, keys, source identities and cumulative reservations before the new readout.
- [ ] Publish owned protocol/code/small evidence with separate actual-push and remote/blob receipts; no checkpoint weights, original inventory or thesis edits.

References informing the distinction: Kumar et al., [BEAR](https://arxiv.org/abs/1906.00949) (offline action-support mismatch); Agarwal et al., [RL evaluation uncertainty](https://arxiv.org/abs/2108.13264). Neither paper validates the particular proxy or the new experiment in advance.
