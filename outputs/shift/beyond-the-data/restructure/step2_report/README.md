# Step 2 add-on report: regret, paired standard errors, cost split and lower tail

**Exploratory. Reporting only. It decides nothing.**

This folder holds a report that runs after step 2 of the restructured BCA (the policy pilot) and puts more
information beside step 2's own numbers. It never changes NET, G1-G3, A1-A3, version 6's marks or step 2's verdict.
Those are step 2's alone, fixed in its frozen `prespec.json` before any actor was trained.

## When it was written

It was written and committed before step 2's official results existed, while the official step 2 run was still in
progress. The session that wrote it never opened any file under `runs/shift_explore/restructure/step2/` (step 2's
official output folder). It was developed and tested only on the scratch smoke run in
`/root/step2_review_scratch/smoke`, whose 40-step actors mean nothing.

- Committed: 14 October 2026, about 22:15 EDT (WSL clock), on top of commit `9aae4df`.
  `git log -- outputs/shift/beyond-the-data/restructure/step2_report` gives the exact time and the commit hash.
- Its code is bound to step 2 as frozen: `report.py` pins the SHA-256 of step 2's `prespec.json` (`839ed716...`) and
  of its final code manifest (`3222aedc...`), and refuses any other.

## What it adds

These are the reporting items of `outputs/shift/oracle/REVIEW_RESPONSE.md`, section 4, "Step 2, now".

1. **Regret beside points.** Regret is the best possible points minus an arm's points, on the same bank. The best
   possible points come from the exact optimal reference in `outputs/shift/oracle/results.json` (bank means; nothing
   of the oracle is recomputed). They are 100.8 / 109.0 / 102.6 / 106.7 points in the stalling world and 100.6 /
   104.2 / 101.0 / 103.9 in the no-stall world (recorded conditions / strong current / big waves / big kicks). The
   report gives regret for every arm, situation, world and seed, and the total regret across both worlds in strong
   current. Regret changes no ranking and no difference. In particular NET(x) = total regret of none minus total
   regret of x (checked).
2. **Paired standard errors** of every pooled difference step 2 reports: every arm's NET and its two parts, G1's
   comparisons (NET of obca_k1 against each world-blind rival), G2, G3 (= M1), A1-A3 and the dose readout, M2,
   M3's comparison with the clone, M4, and NET in recorded conditions. They come from one extra exact rollout of
   every scored locked actor on the full banks, keeping each start's return. Three numbers are given for each
   difference:
   - **SE starts**: how much the value would move with another bank of 1,000 starts. The same bank serves every
     seed, so it is the standard deviation over starts of the per-start difference averaged over seeds, divided by
     the square root of 1,000.
   - **SE seeds**: the plain seed-to-seed standard error (standard deviation of the per-seed differences over the
     square root of the number of seeds).
   - **SE**: both together, the square root of SE starts squared plus SE seeds squared. This is slightly
     conservative; `report.json` also gives the exact two-way estimate.
3. **Where the cost comes from, and the lower tail**, from the same rollouts: tracking cost (the floats away from
   rest), energy cost (0.3 times the squared throttles), the late peak (the largest distance of the floats from rest
   over steps 10-99), and the 10th percentile and worst-10% mean of per-start return.
4. **Labels of the situations.** Strong current and big waves lie entirely outside the training range. Big kicks
   starts beyond every recorded float state and settles back within about 10-30 steps.
5. **A fixed text block**:
   - 100 points is the stall-aware reference, not "the best known controller".
   - Points are measured on four fixed banks of 1,000 starts.
   - Penalties are in the critic's discounted value units, scores in undiscounted 100-step points.
   - BCA treats its 5 calibration rows per episode as independent (design effect 1.0-1.4). Step 1's certificate
     survives thinning to one row per episode, measured with the gate's critic.
   - The attribution ladder is read support-first: the range hinge and the matched stand-in are the support rule
     alone, and obca_forced is BCA's switch test (identical to obca_k1 in the stalling world).

## What it checks before it reports

- Step 2's `prespec.json` and code manifest equal the pinned SHA-256, and every file it imports equals the manifest.
- Official mode refuses unless step 2's `results.json` exists (sidecar-checked, official, the frozen specification).
- Every lock is checked as step 2 checks it, and every actor in every `actors.msgpack` is checked against the SHA-256
  that `lock.json` records. Every `values.json` must be the evaluation of its lock.
- The extra rollouts must reproduce step 2's bank means exactly, the pooled values from the per-start rollouts must
  equal those from `values.json`, and those must equal `results.json`.
- Step 2's anchors must equal the oracle's, and the oracle's best possible points must still round to the numbers
  stated above.

A check that fails is written at the top of `report.md`. It is never hidden, and it changes nothing in step 2.

## How to run it

Official, once step 2's `results.json` exists (one CPU; the output is written once to
`runs/shift_explore/restructure/step2_report/`: `report.json` with its `.sha256`, `report.md` and `per_start.npz`):

```
cd /tmp && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \
  taskset -c 20 /root/bcavenv/bin/python \
  /home/user/bca/outputs/shift/beyond-the-data/restructure/step2_report/report.py
```

Scratch (both folders outside the repository; `--allow-no-results` only for testing without `results.json`):

```
... report.py --root /root/step2_review_scratch/smoke --out /root/step2_report_scratch --allow-no-results
```

Unit tests (synthetic numbers only): `... /root/bcavenv/bin/python .../step2_report/test_report.py`.

## Tested before commit

All in scratch folders outside the repository, on the laptop, pinned to CPUs 20-23 (the official step 2 run was
using CPUs 0, 2, 4, 6 and 8).

- **Unit tests**: 22 of 22 pass (`test_report.py`, synthetic numbers only). They cover the standard errors against
  a direct loop, shared start luck counted once, one seed, the tail statistics, the costs and the late peak's window,
  regret and the NET identity, every contrast against step2lib's own NET, go/no-go, attribution and marks, the
  guards (including the official refusal before `results.json` exists, on a temporary folder), and banned words.
- **End to end** on the smoke run `/root/step2_review_scratch/smoke` (seed 0 only, 40-step actors, so the numbers
  mean nothing), with `--allow-no-results --out /root/step2_report_scratch`. That folder does hold a scratch
  `results.json`, which was read and compared.
  - Exit 0 after 70 s of wall time on one CPU; peak memory 413 MB.
  - 62 actors checked against their locks' SHA-256.
  - 124 rollouts (31 scored actors x 4 situations), 0.38 s each on average (largest 0.48 s). Loading the gate's
    data took 9-10 s per seed-world.
  - All 124 bank means of the return equal step 2's `values.json` exactly. The pooled values equal those from
    `values.json` (largest difference 6e-14) and the scratch `results.json` (73 compared, largest 3e-14).
  - The tracking + energy identity holds to 1e-12. Step 2's anchors equal the oracle's exactly. The oracle's best
    possible points are within 0.038 of the stated ones. Every check passed.
  - In the stalling world, obca_forced's trained actor is bit-identical to obca_k1's (seed 0).
- **Without `results.json`**: a scratch root that links only the smoke run's `policy` folder gives the same numbers
  to the last digit, and the comparison with `results.json` is reported as not made.
- **Refusals**, each writing nothing:
  - no `results.json` without `--allow-no-results`;
  - an existing report;
  - a scratch `--root` or `--out` inside the repository;
  - two CPUs;
  - `OMP_NUM_THREADS=4`.

**Expected official cost.** Ten seed-worlds with about 31 scored actors per seed (16 in the stalling world, 15 in the
no-stall world) give about 620 rollouts, about 4 minutes. Loading the gate's data adds about 10 s per seed-world. In
all about 6-7 minutes on one CPU.

## Files

| File | Role |
|---|---|
| `report.py` | the add-on: checks, rollouts, regret, standard errors, costs, tails, `report.json` and `report.md` |
| `test_report.py` | unit tests on synthetic numbers (standard errors, regret, tails, costs, guards, wording) |
| `README.md` | this file |
