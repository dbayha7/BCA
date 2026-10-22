# Source-mapped stages and proposed fixtures

Status: original source audit and fixture specification, followed by implementation of the simple-width mechanical subset. The simple primitive now exists in mechanics/objective.py; Option C remains unimplemented. Line numbers in code references are actual source lines; panel numbers are explanatory stages, not claims about an existing module numbering scheme.

## Existing native path

Pinned matching source: commit 9fe4f2acb93fe52326ba274af5d71f0fe193db65. Three inspected live files match the pinned Git blobs and the preserved experiment's SHA256 identities.

| Stage | Actual source | Input → output | Verified meaning |
|---|---|---|---|
| A1 target | algorithms/td3_bc.py:106–115 | next observation, target actor/critics, smoothed action → detached Bellman label | Native target; not full-return truth |
| A2 critic | algorithms/td3_bc.py:117–125 | logged action, label → critic squared-error gradient | No BCA subtraction here |
| A3 reference | algorithms/td3_bc_bca.py:234–279 | held-out residual/scale → frozen posterior threshold | One-step residual target; minimum current critic head |
| A4 width/dose | algorithms/td3_bc_bca.py:283–310; calibration/dose.py:40–62,65–94 | logged-action eta, frozen reference → detached BC dose | Does not evaluate actor-action uncertainty |
| A5 actor | algorithms/td3_bc.py:128–144 | actor action, raw Q1, logged action, dose → actor update | Dose multiplies BC; raw-Q scaling; dQ/da retained |

Screenshot-ready existing pseudocode:

    A01  y <- stop_gradient(reward + mask * gamma * min_target_Q(next_state, noisy_target_action))
    A02  critic <- minimize sum_heads mean((Q(state, logged_action)-y)^2)
    A03  R, eta_frozen, unit <- refresh one-step residual reference on held-out rows
    A04  U_logged <- R * unit * max(eta_frozen(state, logged_action), floor)
    A05  m <- stop_gradient(bounded_dose(U_logged)); unusable band -> m=1
    A06  action <- actor(state)
    A07  c <- alpha_host / stop_gradient(mean(abs(Q1(state, action))))
    A08  actor_loss <- -c*mean(Q1(state, action)) + mean(m*BC_distance(action, logged_action))

Takeaway: A08 changes cloning strength. The multiplier is detached but multiplies an action-dependent loss, so its actor effect is real. Copying its detached width into an additive Q penalty would remove the intended uncertainty derivative.

## Simple direct-width candidate (proposed)

    S01  freeze critic parameters, eta parameters, threshold R and unit
    S02  action <- actor_theta(state)                       # retain action differentiation
    S03  raw_Q <- Q1_frozen(state, action)
    S04  c <- alpha_host / stop_gradient(mean(abs(raw_Q))) # raw Q, same host normalization
    S05  U <- stop_gradient(R*unit) * max(eta_frozen(state, action), floor)
    S06  score <- raw_Q - kappa*U
    S07  actor_loss <- -c*mean(score) + mean(BC_distance(action, logged_action))
    S08  differentiate only actor parameters; do not detach S05 as a whole

Purpose: isolate direct candidate-action score placement. Premise: residual-derived width may identify problematic action values. The premise is unproven outside the calibration law. kappa=0 and detached-U controls must reproduce baseline gradients from identical states.

## Exact Option C proposal (not the same candidate)

Source: [DESIGN.md:622–647](https://github.com/dbayha7/BCA/blob/743eba669e368bdd9241738d849a7194b6d648b3/outputs/shift/beyond-the-data/placement_test/DESIGN.md#L622-L647).
Status: [design-only](https://github.com/dbayha7/BCA/blob/743eba669e368bdd9241738d849a7194b6d648b3/outputs/shift/beyond-the-data/placement_test/DESIGN.md#L1-L9); dedicated implementation [to write](https://github.com/dbayha7/BCA/blob/743eba669e368bdd9241738d849a7194b6d648b3/outputs/shift/beyond-the-data/placement_test/DESIGN.md#L1434-L1454).

    C01  action <- actor_theta(state)
    C02  query <- clip(action + Normal(0,0.2^2), -1,1)     # no inner noise clipping
    C03  raw_Q <- Q1_frozen(state, action)                # unsmoothed actor action
    C04  c <- stop_gradient(2.5/(mean(abs(raw_Q))+1e-7))
    C05  if band usable: W <- positive_part(U(state,query)-U(state,logged_action))
    C06  else:           W <- positive_part(raw_Q-Q1_frozen(state,logged_action))
    C07  actor_loss <- -c*mean(raw_Q-k*W) + mean(BC_distance(action,logged_action))
    C08  k=1; freeze model/reference parameters but retain the required action derivative

The draft calls the usable branch certified. That code label is not proof of support or a true-Q coverage certificate in this new pilot. The support decision must be explicit. Its refusal branch can cancel the Q gradient where the hinge is active and k=1.

Historical differentiable width: [experiments/shift_iw/lower_edge.py:196–208](https://github.com/dbayha7/BCA/blob/743eba669e368bdd9241738d849a7194b6d648b3/experiments/shift_iw/lower_edge.py#L196-L208). It freezes reference parameters while retaining eta's input-action derivative.
Historical threshold lookup: [command_le_train.py:92–112](https://github.com/dbayha7/BCA/blob/743eba669e368bdd9241738d849a7194b6d648b3/outputs/shift_iw/explore/channel/command_le_train.py#L92-L112). It stops action before discrete lookup; no smooth derivative through lookup mass is claimed.
Non-equivalent benchmark: [bench_td3bc.py:170–187](https://github.com/dbayha7/BCA/blob/743eba669e368bdd9241738d849a7194b6d648b3/outputs/shift/beyond-the-data/placement_test/bench/bench_td3bc.py#L170-L187). It evaluates smoothed Q and normalizes the penalized score. A detached penalty can change the numerical Q/BC scaling there; it is not the specified raw-Q control.

## Deterministic fixture specification

The table below preserves the initial analytic specification. Subsequent execution is mapped in mechanics/REPORT.md: simple-objective checks passed; Option C and real-host critic/target isolation remain unrun. Expected values in this table are analytic.

Use scalar actor a=theta, Q1(a)=2+a, eta(a)=1+a, frozen R*unit=2, logged a0=0, theta=.4 and noise=.1. BC=(a-a0)^2. Treat c as a fixed number during differentiation. For Option C use its specified c=2.5/(2.4+1e-7).

| ID | Input/isolated change | Expected output/invariant | Source stage |
|---|---|---|---|
| F01 | Baseline | grad_theta L=-c+0.8 | A5 |
| F02 | Simple U=2(1+theta), kappa=1 | U=2.8; grad_theta L=c+0.8 | S05–S08 |
| F03 | Option C usable with query=.5 | U(query)=3, U(logged)=2, W=1; grad=c+0.8 | C02,C05,C07 |
| F04 | Fully detach U or W, raw-Q c held identical | grad=-c+0.8, baseline identity | S08,C07 |
| F05 | kappa/k=0 | baseline forward loss and gradient, same RNG | S06,C07 |
| F06 | Constant additive uncertainty offset | different loss value; baseline gradient | S06 |
| F07 | Option C refusal, active Q advantage | W=.4; k=1 cancels Q gradient; grad=.8 (BC only) | C06–C07 |
| F08 | Option C inactive width hinge: query=-.1, logged=0 | W=0; baseline gradient away from kink | C05 |
| F09 | Frozen parameter ownership | critic/eta/reference params unchanged; eta action Jacobian is1 and width Jacobian is2 in fixture | S01,S05 |
| F10 | Critic/target placement isolation | identical target, critic loss and critic gradient across actor-only candidates at same pre-state | A1,A2 |
| F11 | Refusal/invalid arithmetic | no 0*infinity/NaN; refusal branch counted; controls keep running | S05,C05–C06 |
| F12 | Clip, eta floor, hinge and lookup boundaries | explicit inside/outside derivatives; separately documented at-kink convention; no invented lookup derivative | C02,C05 |
| F13 | BC broadcasting | one dose per row, action dimensions reduced once; no batch-by-batch accidental product | A4,A5 |
| F14 | Raw versus penalized normalizer | raw-Q control identities pass; deliberately penalized normalization breaks identity and is labeled a different method | S04,C04 |

Analytic tolerance proposal: float64 atol/rtol 1e-9; float32 1e-6. Check finite differences away from kinks with c, reference state, random noise and branch decision frozen numerically. Recomputing stopped coefficients for perturbed inputs tests a different derivative. Compare raw gradients as well as isolated optimizer updates: optimizer state must be identical and reset.

Future file targets (not created in this audit):
- fixtures/test_actor_score_gradients.py
- fixtures/test_option_c_definition.py
- fixtures/test_placement_isolation.py
- direct_q_pilot/protocol.json, freeze.json, paired_probe_records.json

These original names are future implementation guidance. The executed subset uses mechanics/test_objective.py and mechanics/audit_checkpoint_width.py; see their separate receipts for actual passes.
