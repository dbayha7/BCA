# Formal G2, candidate knee_s10_v1, attempt 1: readout

Run by the user at their terminal, 2026-11-01 05:19 UTC (01:19 EDT), approval commit `5742605a`, manifest
`0ba3e4b56016522ef7326b163fea469feb8622b4bf9dcd7480f5049862e8aaf9`, integrity re-check after the run clean. Record:
`formal_g2_run/attempt_1/report.json` and `collection.json` (datasets git-ignored; digests in the report). This readout
reads the report against `G2_FORMAL_RULE.md` as written before the run. It adds no number the report does not hold.

## Headline

**NOT DATA-OOD**, one cause: D3 (anchors) in the no-stall world. Every other check passed, including both hard
conditions H1 and H2. By the rule's table, NOT DATA-OOD leads to: stop this candidate. There is no retry after it.

## Every check

| Check | stall3 | no-stall | bar | result |
|---|---|---|---|---|
| D1 CUTOFF holds no W row; predicates agree | exact | exact | exact | pass |
| D2 retained high-load a0 at or past s10; nearest distance | 0 rows; 0.1000 | 0 rows; 0.1000 | 0 rows; >= 0.0999 | pass |
| D3 starts inside their group's 0.99 quantile | 122 of 128 (0.953) | 125 of 128 (0.977) | >= 0.95 | pass |
| D3 gated anchors inside their group's 0.99 quantile | 95 of 96 (0.990) | **84 of 96 (0.875)** | >= 0.90 (87 of 96) | **no-stall fails** |
| D4 label density, mid band | 2.452 | 2.452 | [2.328, 2.587] | inside |
| D4 label density, deep band | 2.491 | 2.491 | [2.114, 2.813] | inside |
| B1 pi_ref inside CUTOFF support at next states | 1.000 | 1.000 | >= 0.99 | pass |
| H1 max relative |Q_stall - Q_nostall| on S9 and anchors | 0.0 | | <= 1e-9 | pass (exact) |
| H2 median exact rho(s10), all 96 gated anchors | 1.536 | | >= 0.779 | pass |

Budget: 485,696 exact simulator steps of 1,000,000; 30.6 s of 1,800.

## What the failing check measured, and why one world only

D3 is the amended L1.5 state-shift test: each anchor state's 64-nearest-neighbour distance to the world's fitting rows,
against the 0.99 quantile of the leave-one-episode-out distances of deep-load training rows at the same time step. It
asks whether the anchor states are inside the data, so that the only out-of-data thing about an anchor is its action.

The anchors are the same states for both worlds: the sealed draw rule resets, then runs the widened recorder's
commands for t steps in stall3 (`build_anchor`). The two fitting datasets are not the same states. The recorder's
commands depend only on the load band and the sealed uniforms, and load is exogenous, so both worlds executed the
same commands row for row (which is why the D4 densities agree to every digit). What differs is the thrust those
commands produced. At high load the recorder draws a0 uniformly in [-1.0, -0.6] with probability 0.98; half of those
draws are past the knee at 0.8, where stall3 cuts thrust to 0.8 - 3(|a0| - 0.8), that is 0.2 at full throttle instead
of 1.0. Over a 25- or 50-step prefix that shortfall carries the anchor into states the no-stall episodes, which got
full thrust on the same commands, seldom visit.

The report supports that reading without per-anchor distances (which it does not record): the t = 0 anchors are reset
states, identical in both worlds, and their thresholds agree (4.020 vs 4.012); the prefixed groups have tighter
no-stall thresholds (t = 25: 3.383 vs 3.534; t = 50: 3.106 vs 3.269). So the 12 no-stall failures sit in the t = 25
and t = 50 groups. The exact split by time is not on file.

## What the passing checks say (reported, not claims)

- The value gap the candidate was built for is there: median exact rho(s10) 1.536 against the bar 0.779, with
  quantiles 0.44 (25%) and 4.02 (75%). By anchor time: 0.761 at t = 0, 4.195 at t = 25, 0.902 at t = 50. Where
  pi_ref's first continuation command is projected (81% of anchors) the median is 2.197; elsewhere -0.681.
- The twin identity is exact (H1 = 0.0), and pi_ref stays inside CUTOFF's support at every next state (B1 = 1.000).
- The widened recorder realized its designed label density (2.46 designed; 2.452 mid, 2.491 deep, inside the sealed
  bounds) and left no retained row at or past s10 (D2 exact).
- pi_ref's mean return from the 128 starts in stall3 is -69.9; the zero policy's is -715.2.

## What this leads to

By the rule the user approved, candidate knee_s10_v1 (this anchor bank, this prefix rule) stops here, and the ledger
entry is written (`../../ood_definition_v4/LEDGER.md`). The definition (v4 r3), the recorder, pi_ref v2 and the bed
are unchanged. The next step of the experiment is the next candidate in the ledger.

The evidence points at the anchor prefix rule, not at sigma, the recorder or pi_ref: anchors reached through stall3
dynamics are state-shifted relative to no-stall data for 12 of 96. A candidate whose prefix never commands past the
knee (prefix a0 drawn inside the retained high-load support) has the same prefix trajectory in both worlds, so each
anchor is one physical state for both D3 checks. That moves the anchor states, and this run's rho(s10) by time (0.761
at t = 0 against the bar 0.779) says H2 is open for such a candidate until its own G2 runs. Everything in this readout
is seen data for that design and is ledgered as such.
