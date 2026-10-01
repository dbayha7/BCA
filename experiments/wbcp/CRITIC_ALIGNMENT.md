# Calibrating min(Q1, Q2) versus the Q1 TD3+BC's actor climbs

## Question

TD3+BC's actor maximizes the online Q1 alone (`algorithms/td3_bc.py:130`). BCA calibrates the residual of the online
min(Q1, Q2) (`algorithms/td3_bc_bca.py:173`, `:255`). Does that hide error in the head the actor uses?

The other hosts:
- ReBRAC and CQL calibrate the min of their online heads, which is what their actors maximize.
- IQL calibrated the min of its online twin heads, while its actor weights use the min of the Polyak target heads
  minus V(s) (`algorithms/iql.py:259-268`). This is now fixed (below).

## Method

`critic_alignment.py` uses the seven frozen TD3+BC pools: 100k updates each, 2,587 to 1,992,027 held-out rows. No
training is involved. These short-train pools never refreshed and ran with BC multiplier 1, so their actors are plain
TD3+BC.

- **Restore and gate.** Each checkpoint is restored byte for byte, and both heads are recomputed at the logged action
  and at pi(s). The min of the heads must reproduce the frozen residual, scale and policy action exactly, or nothing
  is analyzed. All seven pools reproduced with maximum difference 0.
- **Scores**, all under the frozen scale sigma:
  - s_min = |t − min(Q1, Q2)| / σ, BCA's score today;
  - s_1 = |t − Q1| / σ, the actor-aligned score;
  - s_max = max(s_min, s_1).
- **Banks.** 1,000 WBCP banks per pool, each 1,024 independent rows, with α = 0.1 and β = 0.95. A bank fails for a
  residual when its threshold's population miscoverage exceeds 10%.
- **Expectations** were written before the script existed: `runs/wbcp_critic_alignment/expectations.md`, SHA-256
  `070ced71…c852`.

**Two pools are excluded post hoc.** The pre-registration covered all seven, and both exclusions were decided after
the runs.
- **pen-human: the critic diverged.** Critic loss reached 1.7×10¹⁹, mean Q 1.5×10¹⁰, and the target is above Q on
  every row.
- **pen-cloned: the critic is still diverging.** Pen's rewards are at most 61 with γ = 0.99, so Q cannot exceed about
  6,100. Its mean Q reaches 9,160, is still rising, and its critic loss is increasing. The session building the
  host sweep confirmed this independently.

Their numbers are shown below for completeness. The verdicts use the five converged pools, and the notes say what
including the excluded two would change.

## Results

| Pool | Q1 > Q2, logged / π(s) | median \|Q1−Q2\| ÷ median \|r_min\| | Q1 miscoverage at λ*_min | Calibrate s_min: banks failing for min / for Q1 | Calibrate s_1: failing for Q1 / for min | Calibrate s_max: failing for Q1 / for min | Threshold, s_max ÷ s_1 |
|---|---|---|---|---|---|---|---|
| hopper-medium | 50.6% / 51.5% | 0.93 | 11.4% | 4.8% / **37.4%** | 4.6% / 0.1% | 0.0% / 0.0% | 1.068 |
| walker2d-medium-replay | 54.2% / 57.8% | 1.03 | 12.4% | 4.0% / **70.2%** | 4.2% / 0.0% | 0.0% / 0.0% | 1.066 |
| halfcheetah-medium-expert | 54.3% / 58.9% | 0.52 | 10.6% | 4.5% / **14.4%** | 4.4% / 1.0% | 0.2% / 0.0% | 1.050 |
| maze2d-large | 46.4% / 46.2% | 1.15 | 10.5% | 3.7% / **10.7%** | 4.2% / 1.3% | 0.5% / 0.1% | 1.069 |
| pen-expert | 47.2% / 54.4% | 0.64 | 10.3% | 5.2% / **8.2%** | 5.9% / 2.9% | 0.0% / 0.0% | 1.073 |
| *pen-cloned (unconverged)* | 47.0% / 75.1% | 0.37 | 11.3% | 4.7% / 38.1% | 4.8% / 0.1% | 0.3% / 0.0% | 1.052 |
| *pen-human (diverged)* | 99.7% / 100% | 0.04 | 9.8% | 4.0% / 3.9% | 3.9% / 6.0% | 3.9% / 4.0% | 1.040 |

## Verdicts (five converged pools)

| # | Expectation | Result | Met? |
|---|---|---|---|
| 1 | Gate: exact reproduction | Exact on all seven pools | Yes |
| 2 | Logged actions: Q1 > Q2 on 35-65% of rows; median \|Q1−Q2\| below 25% of median \|r_min\| | 46-54%; the ratio is 0.52-1.15 | Partly. The sign is random, but the heads disagree by about one residual |
| 3 | π(s): the Q1 > Q2 share rises by at least 5 points on at least 5 of 7 pools | Only pen-expert, +7 points | No. The mean (Q1−Q2)/σ is larger at π(s) on 4 of 5 pools, but the shift is small |
| 4 | Q1 miscoverage at λ*_min is 8-13%; hidden misses at most 3% | 10.3-12.4%; hidden misses 1.9-4.5% | Partly. hopper (3.5%) and walker2d (4.5%) exceed 3% |
| 5a | Calibrating s_min: Q1's failure rate within 5 points of the min's | Min 3.7-5.2%; Q1 8.2-70.2% | No: within 5 points only on pen-expert. Above the 5% budget on all five |
| 5b | Calibrating s_1: Q1 fails at most about 6% | 4.2-5.9% | Yes |
| 5c | Calibrating s_max: both failure rates at most 5%, threshold under 5% above s_1's | Failures 0.0-0.5%; thresholds 5.0-7.3% wider | Partly. Valid, but wider than expected |

Including the excluded pools:
- **pen-cloned** follows the same pattern, and its π(s) rise of +28 points would make item 3 two of six.
- **pen-human** fails item 2's sign test (Q1 > Q2 on 99.7% of rows) and meets items 4 and 5a.

## What it means

**My forecast was wrong.** I expected the mismatch to cost little at logged actions, assuming the heads agree where
the data are. They do not: the typical |Q1 − Q2| at a logged action is 0.5-1.1 times the typical Bellman residual.

**Mechanism.** Q1's miscoverage at the min's true threshold is only 0.3-2.4 points above 10%. A WBCP threshold for
the min sits about 1.5 points of miscoverage inside the target, at a mean of 8.4-8.5%. The Q1 shift eats most of
that slack, so the small excess turns into a large failure rate.

**Size of the effect.** These are independent banks of 1,024 held-out rows, at logged actions. A threshold
calibrated for the min score has nominal 95% confidence. For Q1, that confidence drops to:

| Pool | Confidence for Q1 |
|---|---|
| pen-expert | 92% |
| maze2d | 89% |
| halfcheetah | 86% |
| hopper | 63% |
| walker2d | 30% |

Q1's average miscoverage under those thresholds stays at 8.8-9.7%, except on walker2d (10.6%). So the failures range
from mild to severe rather than being catastrophic on average. Whether this carries over to BCA's thinned banks and
to π(s) is untested.

**Decision rule.** Items 4 and 5a failed. That triggers the pre-registered rule: Q1 alignment (s_1, with sigma refit
on Q1's residual) should go into the 7-dataset × 4-host sweep for the TD3+BC host now.

## The side question: a max over the min and Q1?

s_max = max(|t − min|, |t − Q1|) / σ. Since min ≤ Q1:
- it equals s_min where the target is at or above Q1;
- it equals s_1 where the target is at or below the min;
- where Q2 < Q1, it is the larger absolute residual of the two heads.

Its extra width over s_1 comes only from rows with Q2 < Q1 and a target above (Q1 + Q2)/2, which are about a
quarter of the rows.

- **It is valid for both heads.** Q1 fails 0.0-0.5% and the min 0.0-0.1%, far inside the 5% budget, so it
  over-covers.
- **It is wider than needed.** Its thresholds are 5-7% wider than calibrating Q1 alone.
- **Q1 alone is enough.** Calibrating s_1 brings the actor's head to 4.2-5.9%, the nominal level, and still covers
  the min residual at 0-2.9%.
- **Where Q2 reaches the actor.**
  - The actor's value term uses only the online Q1. The online Q2 enters neither that term nor the target, which
    takes the min of the Polyak-averaged target heads.
  - Today the online Q2 does reach the actor through BCA's BC dose, because sigma and λ are fitted on min(Q1, Q2).
  - Calibrating s_1 with sigma refit on Q1 removes that path; calibrating s_max keeps it.

**Recommendation: calibrate Q1 (s_1) on TD3+BC.** Use the max only if a guarantee for both heads at once were
needed, and BCA does not need one.

## Limits

- **Scale.** sigma was fitted on the min residual, and a Q1-aligned BCA would refit it on Q1's residual. Any fixed
  scale fitted on the training rows gives a valid score on held-out rows, so refitting changes the band's width,
  not its validity. Whether it narrows the band is untested.
- **Bank design.** These banks are independent rows; BCA's are thinned to K rows per episode (Changes 4-10). Their
  absolute failure rates differ, and the point here is the comparison between scores.
- **Action domain.** Everything here is at logged actions. Whether alignment improves the actor's decisions needs
  training runs. The action-domain gap remains: the bank never sees π(s).

## Fix: IQL now calibrates the Q its actor reads

**Change.** The per-step scale fit (`calibration/iql_targets.py`) and the refresh residual
(`calibration/iql_reference.py`) both read `qf_target`, the Polyak copy IQL's actor advantage reads, instead of
the online heads. The native IQL host is untouched. INTEGRATION.md, ALGORITHMS.md and the source pins in
`configs/sources.json` are updated. The configuration identity is unchanged.

**Expectations** were written before the change: `runs/wbcp_critic_alignment/iql_fix_expectations.md`, SHA-256
`c5fa8ea0…852f`.

| # | Expectation | Result | Met? |
|---|---|---|---|
| 1 | The existing refresh test fails on the new code until its recomputation reads qf_target | It fails on the old code with the updated recomputation (threshold 2.44 vs 2.52) and passes on the new | Yes |
| 2 | New test: the recorded residual matches the target heads and not the online heads; the fitter's inputs read the target heads | Passes on the new code; fails on the old (fingerprint mismatch) | Yes |
| 3 | Everything else passes | 13 IQL host tests; 84 others (WBCP, reference, configs, TD3+BC, ReBRAC and CQL hosts, bank, freeze_scores, critic_alignment) | Yes |
| 4 | check.py passes with re-pinned sources; matrix_sha256 unchanged | Accepted, 53 code excerpts verified; `1dd66a57…` unchanged | Yes |
| 5 | Magnitude small at 100k updates | Not measured: needs the trained IQL pool. In the 4-step smoke test the threshold moved 3%, early in training when the copies differ most | Open |
