# Candidate knee_s10_v1: development G2 feasibility rule (written before any computation)

Written 2026-10-31 07:20 EDT, before the screen code ran. The user chose this candidate ("Knee at s10, new data") in
answer to "Which candidate should go through the no-training feasibility check next?". The definition is
`../ood_definition_v4/DRAFT_DEFINITION.md` r2 (a draft; not approved).

## The candidate

- **Worlds:** stall3 and its exact no-stall twin (unchanged; `knee_edge_ood_v3/worlds.py`).
- **The only change is the recorder's high-load window.** At load g >= 0.4, a0 ~ U[-1.0, -0.6] with probability 0.98
  (was U[-0.9, -0.7]). Otherwise a0 ~ U[-1, 1], and a1..a3 ~ U[-0.7, 0.7]. The reset law knee_mix15 is unchanged.
  - Density in the window is 0.98 / 0.4 + 0.01 = 2.46 per unit a0, and the window is symmetric about KNEE.
  - 25% of high-load window rows are at depth >= 0.1 (a0 <= -0.9); the old window had none there.
- **Arms (DRAFT r2 section 3.3):** CUTOFF removes every |a0| > KNEE row. The twin is the superset: CUTOFF's rows plus
  every W row.
- **Scored slot:** sigma = s10 (a0 = float32 -0.9), named now. s05 and s20 are reported only.
- **Continuation:** pi_ref, the finite-horizon discounted LQR on the core's (A, B), with cost |x_floats|^2 + 0.3|a|^2,
  gamma 0.98 and horizon 100, time-varying gains, every command clipped to +-KNEE (`pi_ref.py`). It is deterministic and
  needs no checkpoint. Inside +-KNEE thrust equals the command in both worlds, so pi_ref never stalls.

## What this screen is

It is a development estimate of G2 on a test-entropy collection and a test-only anchor stream. Nothing is trained.
It answers whether the candidate is worth sealing; it is not the formal G2.
- The formal G2 runs on the candidate's own sealed entropy and anchor streams, with sealed Monte Carlo bounds.
- These seeds are never confirmatory.

## Checks and bars (fixed now)

| id | Check | Bar | Source of the bar |
|---|---|---|---|
| D1 | CUTOFF's fitting rows with \|a0\| > KNEE, in both worlds | exactly 0 | construction |
| D2 | Distance from s10 (a0 = -0.9) to the nearest retained a0 among CUTOFF's high-load rows | >= 0.0999 | construction (one-sided) |
| D3 | L1.5 as amended, in each world, against all fitting rows (as sealed), for 128 E2 starts and the 48 gated E2 anchors | starts >= 95%, anchors >= 90% | sealed L1.5 bars |
| D3b | The same, against CUTOFF's retained fitting rows only | reported | (stricter variant) |
| D4 | The twin's fitting-row density in a0 in [-0.95, -0.85], per high-load band: count / (band rows x 0.1) | mid >= 2.00, deep >= 1.82 | the sealed L1.4 lower bounds; this is at least the label density the old twin had at the onset |
| H1 | item 5 under pi_ref at the 48 gated anchors: max_k \|P_k - Q_nostall\| <= 0.10 / 0.15 / 0.30 at s05 / s10 / s20 | >= 90% of anchors | sealed item 5 |
| H2 | median rho_hat(s10) under pi_ref, over predictor-valid gated anchors | >= (0.5 + 0.3 x 0.15) / (1 - 0.3) = 0.7786 | DRAFT r2 H2, SD-free part |

## Labels

- **PASS:** every row passes. The candidate may proceed to sealing, to G1 and to the variance pilot, each needing its
  own approval.
- **NOT TESTABLE:** D4 or H2 fails. Stop; the knee at s10 is not a candidate with this recorder.
- **NOT DATA-OOD:** D1, D2 or D3 fails.
- **INCONCLUSIVE:** H1 fails (the estimator is invalid under pi_ref).

Every label is reported. The headline follows INCONCLUSIVE > FAIL-type > PASS, and nothing is hidden.

## Reported, never used to decide

- rho_hat at s05 and s20, and the sign control p10.
- The share of anchors where pi_ref is clipped at -KNEE.
- Raw counts of high-load rows at depth >= 0.1.
