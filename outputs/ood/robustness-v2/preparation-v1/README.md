# Direct action-support robustness follow-up

David requested a test that better answers whether BCA handles OOD actions better.
The new protocol compares both continuations from the **same full state and same
imposed first action**, with equal planned numbers of near, moderate and strongly
support-distant actions. It separately evaluates native first-action choice and
whether frozen width predicts measured harm. Support labels use training data
only; unfamiliar does not mean harmful, and legal action bounds do not imply ID.

Status: **protocol and preparation tested; no v2 scientific collection dispatched**.
The versioned extension ledger, collector integration and their independent
engineering acceptance remain necessary before physical execution. They are the
next implementation work under the user's new direction. Only the first TD3+BC
Hopper checkpoint pair is currently accepted; other declared pairs need their
own completed training audits. Do not reopen failed CQL or duplicate cluster work.

## Completed preparation

- Frozen prospective protocol (specified after v1, before v2 outcomes), SHA256
  `f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756`.
- Twenty pure candidate/contrast tests, five post-hoc bridge tests and eight
  bounded support-reader tests passed with saved actual exit0 receipts, no skips.
- Training-only feasibility filled all 12 support slots at all437 calibration
  states using the fixed8,192-proposal pool; no threshold or pool-size tuning.
  Reference bank32,768 rows;1,748 reference episodes and437 disjoint calibration
  episodes. Action q95=0.21134613219046514, q99=0.3026651477868097;
  normalized state-distance q95=0.19383700099992648. These are empirical quantiles,
  not true distribution-support guarantees. Future captured-state feasibility
  is still unknown.
- A post-hoc v1 bridge and independent arithmetic check passed. It reuses only
  the previously accepted export and does not re-read archives/weights/ledger,
  query a model, simulate or change the original result.

No resource ledger has been opened or changed. The maximum proposed v2 envelope
plus all original charged history is38,089,713 environment transitions, below
the original39,998,400 ceiling, **only if unstarted v1 OOD collection remains on
hold and a separately tested extension enforces the combined bound**. This
arithmetic is not an executable resource acceptance.

## What the existing data say under the fairer comparison

This is a **post-hoc descriptive result from one TD3+BC Hopper training seed**.
Both policies receive the same first action. The common familiar anchor is the
nearest-recorded action (slot2). Within each state, average the available actions
in that support group, then weight the four collector/capture stratum means
equally. The support-distant group has233 actions across193 of256 states; it is
not a balanced prospective OOD bank and has no independent state-support split.

| v1 support group | Actions | BCA minus host return | Host loss from anchor | BCA loss from anchor | BCA degradation advantage |
|---|---:|---:|---:|---:|---:|
| Near |2071| -4.331688 |0.914647|0.157422|+0.757225|
| Distant |233| -5.045585 |-4.892260|-4.873551|-0.018709|

Negative loss means the alternative improved on the anchor. Thus these distant
actions improve average returns relative to this anchor under both continuations,
while BCA's absolute returns are lower and its average degradation advantage is
essentially zero. **This does not establish better OOD handling by BCA.** It also
does not establish a replicated disadvantage: this is one inspected seed with
shared reset/candidate dependence, no confidence interval and selective support
group coverage. The near and distant groups have different action/state mixtures;
do not interpret their difference as a randomized causal effect of OOD status.

The separately exported action-pooled distant contrast is -4.699876 return units
and +0.014479 degradation advantage. It is a different weighting, not a replacement
for the state/stratum-balanced values above. All2,304 rows and60 aggregate checks
passed an independent arithmetic review. Its1e-12 roundoff allowance applies only
to independently rearranged descriptive arithmetic; no scientific tolerance changed.

## Next execution work and limits

1. Implement a separate ledger extension bound to the original exact audited
   head, retaining every prior charge and unknown call; test interruption,
   duplicate reservation, mutation and combined-cap refusal.
2. Implement the new state/candidate/outcome adapter and precommit path without
   editing accepted v1 scientific sources. Verify exact native applied controls,
   full-state restore/repeat, action/reward tolerances, source/checkpoint/runtime
   identities and fresh saved streams before accepting execution.
3. Save complete final-state contents and constructor evidence. Run one accepted
   pair at a time under the existing shared local lock, CPU/zero GPU, with actual
   worker/supervisor receipts and no automatic retry. Keep aliases/missing groups
   explicit; select no replacements based on outcomes.
4. Independently audit all new saved outcomes and cumulative reservations before
   calculating the predeclared comparisons. All five seeds per cell are required;
   four host/environment cells remain distinct. This balanced stress experiment
   estimates performance on its specified action bank, not natural OOD prevalence,
   all possible actions, or universal robustness.

The frozen plan is `docs/superpowers/plans/2026-09-28-ood-robustness-v2.md` in the
workspace/repository. The plan's unchecked execution items intentionally remain
unchanged after its hash freeze; this README and subsequent receipts record progress.
Original v1 first-pair results, source/model/bank identities, partial exclusions,
unknown ancestor exits, input-only uncertainty, final-state/constructor limitations,
global readiness=false and residual coverage=false remain intact.
