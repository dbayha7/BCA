# Two-scenario repair and staged A + B plan

Status: saved-data construction verified; same-state danger diagnostic prepared but not executed; training held. User selected both reused v2 and a separate one-action-hazard version on November 1, 2026 at 11:01:54 p.m. EST. That selects preparation of both paths, not an additional execution budget or training launch.

## A. Reuse v2 with an actual missing connection

The original capsule removed only a local subset of connector labels: 677 complete successful fitting trajectories remained. The replacement is one fixed separating slab, defined before its saved-data analysis in `CUT_RULE.json`:

`d = C[:10] - B[:10]`; `q(x) = dot(x[:10] - B[:10], d) / dot(d, d)`.

Remove an original transition when its closed source-to-next projection interval intersects `[0.2, 0.8]`, in either direction. This also removes direct jumps across the slab. The boundaries inherit the original connector's 20%/80% trim; they were not searched against model results. The first ten coordinates are benchmark physical state coordinates, including velocities, not metres. Current, wave coordinates and time remain in the full states and support checks.

The single saved-data analysis returned:

| Quantity | Result |
|---|---:|
| Original / retained / removed rows | 92,875 / 89,538 / 3,337 |
| Fitting rows retained / removed | 53,669 / 1,988 |
| Genuine upstream and downstream goal-fragment pairs in fitting data | 716 |
| Complete successful fitting trajectories retained | 0 |
| Retained distinct fitting parents near B at time 25 / C at time 30 | 93 / 167 |
| Calibration parents with eligible retained rows | 256 |
| Simulator calls, model calls, optimizer or calibration updates | 0 |

Support uses the frozen metric: exact time; ten-coordinate physical L2 distance at most 0.10; current difference at most 0.005; two-coordinate wave L2 distance at most 0.02. The entire original start box lies before the slab (maximum projected coordinate -6.844493), and the entire goal ball lies after it (minimum 8.353008). Thus every start-to-goal path crosses the removed slab under the existing discrete chord-edge convention. This does not establish a unique physical route or that an offline learner can infer the missing dynamics.

Original transition IDs, complete states, next states, times, rewards and terminal labels are preserved. All 716 downstream fitting goal-terminal labels remain. Fragments follow original parent/time adjacency; no transition or next action is invented across a cut. No filtered archive was written. The same exclusion must precede fitting, auxiliary fitting, calibration and normalization. Calibration outside the slab provides no conditional guarantee inside it. This bank is inspected development data, including the reserved role's construction metadata; it is not fresh confirmation.

The learner is not yet bound to this rule. `two_scenario_v2/integration.py:74` deliberately verifies the old `data.retained_mask`, which is capsule based. Before training, the smallest required integration is a versioned binding to this explicit slab mask and its hash, with all-stream exclusion and fragment-preservation regression checks. Evaluation support labels and connector probes must use that same declared slab, not silently retain capsule labels. Do not feed the slab to the old adapter or edit the sealed v2 files in place. Honeypot retains its source/next/edge danger exclusion.

## One remaining physical check for A

`danger_diagnostic.py` compares actions from all 15 saved time-29 danger-approach states. For each identical complete state and time, it first replays the saved fatal action exactly, then separately follows the unchanged tracking controller from the safe first action through true termination or horizon 100. It is a fixed-controller return check, not learned-policy Q or a BCA efficacy test.

The frozen checks require all 15 fatal steps to replay exactly; nominal safe continuation to reach the goal with at least 5 greater discounted return; and at least 12 of 15 safe continuations to reach the goal. Recovery can finish through horizon 100. This is explicitly different from the original start-to-goal witness requirement of arrival by time 75. Every case and transition is persisted, including failures; an integrity mismatch stops without new anchors. No criterion will be adjusted after results.

Budget: at most 1,080 transitions, 30 state initializations, zero collection, model updates, calibration or Gym resets. Internal deadline 45 seconds; external timeout 55 seconds plus 5-second kill grace. Preparation and manual execution charge 90 seconds in total. Previous 189,042 transitions and 1,800 charged seconds remain accounted for. After the maximum diagnostic: 190,122 transitions, 1,890 charged seconds, 70,678 setup transitions and 12,510 seconds remaining, with 237,600 evaluation transitions still reserved. This is conservative allocation, not a claim that preparation or shell overhead was measured exactly.

Run once manually from PowerShell on Zeus:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/David/dev/BCA && timeout --signal=TERM --kill-after=5s 55s /home/david/bca-operator-audit-20261026/bin/python -B -u analysis/two_scenario_v2_repair_v1/danger_diagnostic.py --execute'
```

Expected evidence: `danger_attempt_1/ATTEMPT.json`, `ANCHOR_BANK.json`, `transitions.jsonl`, all available `case_*.json`, and `RESULT.json`. An existing attempt blocks another. No final receipt means incomplete work, with saved calls a lower bound and 1,080 an upper bound; do not retry automatically. A v3 setup reservation also blocks this command because shared accounting would need reconciliation. These are local checks, not a distributed lock.

## B. Separate one-action hazard and recollection

Preserve `two_scenario_v3` as the selected second design. It places the absorbing hazard around the first already-prescribed opposing-action successor from B, retains B/C, plant, controller and reward form, and uses the explicit connector radius 0.01. It changes task geometry, not evidence about a chaotic physical mechanism. Its cached nominal route avoids repeating construction calls. It must independently demonstrate safe and harmful choices from the same complete state, endpoint clearance and retained approach support; a failure remains a failure without another geometry search.

The current v3 draft is not launch-ready: its capsule-based data rule must receive the same separating-slab specification, and its pre-v2 budget documents are stale. The new slab is a data exclusion; it must not silently change plant rewards, goals or termination. Reuse task/evaluator/learner components and validation fixtures, not trained policies or data from A. Preserve old drafts and receipts with explicit amendment lineage.

The smallest proposed additional B scope is:

| Work | Maximum |
|---|---:|
| Physical geometry witnesses | 4,502 transitions |
| Collection: 1,280 new parents | 128,000 transitions |
| One exact replay | 128,000 transitions |
| Later frozen evaluation | 237,600 transitions |
| Total additional B physical work | 498,102 transitions |
| Proposed additional wall-time ceiling | 4 hours, pending approval |

The setup allocation alone is 260,502 transitions and a proposed 840-second process cap plus 30 seconds for preparation; the remainder of the proposed four hours would cover later training and evaluation. These are ceilings, not measured B runtime predictions. The completed v2 setup took 20.001877985 seconds for 188,832 transitions; that does not predict learner runtime. No B collection, new budget approval or training approval has been created.

## Later comparison, held for review

Within each version: two dataset conditions, native UniFloral versus original BCA actor score `Q1 - 3U`, unchanged BC, and three paired independent initialization seeds. Each condition/seed receives its own 10,000-update native warmup and two 10,000-update continuations: 18 jobs, 12 final policies, 180,000 critic / 90,000 actor / 180,000 auxiliary updates and 18 calibrations. This is 20,000 cumulative critic updates per final policy. Planned evaluation is 1,536 return episodes plus the existing matched Q-reference panels within 237,600 transitions. No extra control arms, methods or sweeps are added.

Primary outcome is paired return within version/condition/seed. Report unsupported exposure, raw critic error against its own policy's forced-first-action continuation value, adjusted-score error, over/under/near-zero counts, ranking and helpful/harmful adjustments separately. The downward-only BCA penalty cannot directly raise a fixed underestimated Q. Bottleneck measures unsupported generalization and possible pessimism harm, not guaranteed stitching. BCA benefit and Bayesian uniqueness are not established by setup success.

A's scoped total, including prior work, is at most 427,722 transitions. A plus the proposed B is 925,824: 427,424 above the original 498,400 ceiling. The unused 70,678 A allowance is shown explicitly, not silently reclaimed as permission for B. If both later comparisons are approved, their combined learning work is 36 jobs / 24 policies, 360,000 critic / 180,000 actor / 360,000 auxiliary updates, 36 calibrations and 3,072 return episodes. Keep data and results separate; differences across task geometries are not paired treatment effects.

## Verification and lineage

Root verification: 24 synthetic/array fixture tests passed in 0.861 seconds; CLI `--help`, source-only `--prepare` and `--check` passed. No diagnostic attempt exists and no simulator or learner was executed by this task. Independent review verified cut arithmetic, complete-state support, failure persistence, source authentication and budget arithmetic. These checks establish implementation preparation, not physical success or end-to-end training readiness.

Working tree inspection found concurrent unrelated commit `425c64de5c4ae053a0703ccf043d785adacd594c`, dated November 1, 2026 at 11:01:32 p.m. EST. It was not modified or treated as authority for this study. All bound v2 sources still pass their original seal check. Historical v1 failures and completed v2 evidence remain unchanged.

| Artifact | SHA256 |
|---|---|
| v2 manifest | `c4e3198289e9ea8b329bbaedbf8cdc5e1c7cea19249beb121ffaa100af53336d` |
| v2 result | `1e9ba131c35b3e4cafbb8815e3f6338ba6001cc35e1f3e05d2be7518f89941f1` |
| v2 dataset archive | `0b9d51777803f2c0f5bf003f9b3fd0043427485856ebd33dcae4baecc184f426` |
| v2 canonical dataset content | `194457ae13a172cb7e3af301b17f52abb8f17eb7a8c54cb27cef05064d2794ff` |
| frozen cut rule | `d9b1be4ddd96fcaf4572442ec87eb0511e8973598fad9b867d3033e415fe8e4b` |
| saved cut report | `3d2213ea3ada844e92b8ed645e8f1e0e2a20592cfc1b78cd22146d8bc6094d9d` |
| diagnostic contract | `49e83ef970ede060b8ed9d2210252a0fbaf239685d2c8b72cd5053e9a66a9dfe` |

Remaining gates are concrete: manual same-state evidence for A, explicit slab integration before training, final geometry/data/protocol review, and a separate execution-budget decision for B. Assistant-side WSL execution remains denied; no alternate execution route was used.
