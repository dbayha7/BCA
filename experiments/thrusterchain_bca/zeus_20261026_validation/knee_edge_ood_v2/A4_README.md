# A4: Stage 1, the native damage gate, knee_edge_ood_v2 (user-run)

**Requires** v2's step 0.4 (A3) to have ended PASS. Your A4 approval must include the predictions' acknowledgement
(`PROTOCOL_V2.md` change 2): "I acknowledge the written predictions: Stage 1 V0(d) may end INCONCLUSIVE, and after the v1 step 0.3 INCONCLUSIVE, Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE".

**v2 changes here:** the approval words name v2; A4 pins v2's A3 manifest; the decision additionally reports, per seed
and arm, KC at s20 = -1.0 (median and share <= 0.5 over the common gated anchors), the slope-3 analogue of v1's G0.5.
It gates nothing. After the external audit of 2026-10-30 (`PROTOCOL_V2.md` change 6) every worker verifies, before it
imports any package module, that every file it will load still has the hash the supervisor recorded in its `job.json`
(`integrity.py`); every job counts its simulator steps exactly (`units.SimulatorCounter`), and an evaluation job's
exact count, not a formula, is what the stage cap is charged. Those changes touched `stage1_worker.py`, so the v1
GPU smoke no longer certifies these files: **run the smoke below from this folder before approving A4.**

**What runs** (`stage1.py`, a supervisor that never imports JAX; every GPU job is a separate `stage1_worker.py`
process in an owned process group, killed at its cap):

| Order | Units | What each does |
|---|---|---|
| 0 | preflight | rerun1's runtime receipt, the recorded device kind and worker environment (exempt from 5.0's second-failure rule) |
| 1 | 12 warmups (seed x world x arm) | 10k native updates through rerun1 `runtime.step` with the shadow auxiliary fit and the 10k calibration; the distance-arm match; the 10k gradient audit; the complete 10k state saved |
| 2 | R0 | 10k adequacy (stall CUTOFF and RANDOM: J_E1 gain >= +50 on `r0_e1_starts`) and reach (stall CUTOFF R_deep- rate on `r0_e2_starts`) |
| 3 | 12 branches | exact restore of the 10k state, `runtime.branch(..., "native")` with Stage 1's seeds, 10k more updates (calibration at 15k), audits at 15k and 20k |
| 4 | 3 BC-only | stall CUTOFF, Unifloral `bc.py`'s own train step (hash-pinned), with the CUTOFF n RANDOM normalization and its own asserted adapter |
| 5 | 3 evaluations | one per seed index: free rollouts at 15k and 20k (pi, pi_clip, pi_clipall; E2 128 and E1 64; world swap), Q probes at 64 + 32 fresh anchors, KC at 48 visited states per run, diagnostics |

Every adapter (warmup, branch, BC-only, R0 and evaluation) is built by `bindings.create`, which runs the 4.4
assertions on every bound stream, and every job re-derives its data plan from the sealed A3 datasets and requires it
to equal the plan sealed in A3's `collection.json`. Every training job records the sha256 of its phase-0 minibatch
index arrays.

**Caps** (fail closed; whichever binds first): 1,800 s per training job; 5 h of training; 1 h and 18M transitions of
evaluation (R0 plus the three evaluations).

| Outcome | Label |
|---|---|
| A job that does not complete, an exhausted cap, a probe/region assertion, stall = no-stall on S9 beyond 1e-9 | INCOMPLETE |
| R0 adequacy fails; any V0 item fails in any seed (V0(d) reachability included) | INCONCLUSIVE |
| R0: stall CUTOFF R_deep- rate < 0.05 in every seed at 10k | FAIL-early ("the host contained the extrapolation at 10k; 20k was not observed") |
| A hypothesis criterion (K1a, K1c, K2, K3a-c) fails | FAIL-consistent (same criterion in all three seeds, with a reading when its precondition holds) or FAIL-mixed |
| Every rule holds in every seed | PASS: permits only the request for Stage 2 |

**Attempts (5.0)** follow `units.py`:
- a unit is sealed only when the supervisor judged it completed;
- each report carries the fingerprint of the code its job kind runs (its function, everything it reaches, and the
  report writer `run_job`/`sanitize`), as syntax trees, so comments do not count and decorators do. A fix confined to
  one kind's function body leaves the other sealed units reusable; a change to a worker module's imports or
  constants re-fingerprints every kind of that process;
- a sealed unit whose code changed must re-run and reproduce its report bit for bit; a re-run that does not reproduce
  or does not complete leaves a marker with the differing keys, is never sealed, and is compared with the original
  again next time;
- the package may change at most once over all attempts (reverting counts as a change);
- time is charged once: a reused unit's originally charged supervisor time, a re-run's own cost.
`report.json` lists every unit's directory.

Every criterion, V0 item and companion is recorded per seed in `decision.json` whatever the label. The A5
precondition (no-stall CUTOFF J_E1 gain >= +50 at 20k in every seed) is recorded with it.

**Settled conventions (the protocol leaves them open):**
- The BC-only actor takes 10,000 updates, the actor-update count of a native 20k run.
- Q probes are taken at 20k only; 15k is evaluated by free rollouts (the protocol's 15k checkpoint is reported, not
  gated; this keeps the evaluation inside its estimate).
- The a1 = +-KNEE against +-0.69999999 probes keep a0 at the actor's own value and set a2 = a3 = 0; they are Q1-only.
- Q_nostall uses the same 14-slot probe array as the stall values, so the 1e-9 stall = no-stall check compares
  identical actor batches.
- "Valid anchor": all gated Q^pi values finite. Both evaluators raise on any bad action or state, so a failed anchor
  computation ends the evaluation unit INCOMPLETE by design; it is never silently excluded.
- The rate of target-smoothed actions in W is the exact expectation over TD3's clipped Gaussian (0.2, clip 0.5) at
  the target actor's a0 on the arm's replay next observations; no random draws.
- The policy-visited sample takes the exact state of the free rollout at (episode, t), which is the prefix replay.

## Optional GPU smoke test (not a protocol step; no approval file)

`stage1_smoke.py` runs every Stage 1 worker kind once on the real Unifloral learner and this machine's RTX 5070 Ti,
on **test-only data and streams**:
- a fresh 2 x 1,280-episode collection from a test entropy, with test data, training and visited-sample seeds
  (checked distinct from every sealed seed, Stage 2's included);
- test-only start banks and anchors with the sealed banks' structure (count, load law, anchor times) but drawn from a
  smoke stream seed, so no sealed evaluation stream is touched (smoke R0 and evaluation jobs refuse to run without it);
- output only under `~/knee_edge_smoke/<UTC time>` in WSL, outside the repository.

Research jobs refuse every smoke reduction and test override, so nothing here can reach the A4 run.

**Blinded.** The smoke trains a full-length native CUTOFF run on replicate data, so its numbers would be a one-seed
pilot of the gated measurements. `SMOKE_RESULT.json` therefore keeps only structure: which jobs completed, errors,
timings, key sets, finiteness, the V1 numerical check, the device check and a projection of A4's training time
against the 5 h and 1,800 s caps. The worker reports under the output folder do hold values: do not read them. This
disclosure is part of the sealed A4 package.

What it runs: preflight; one 10k warmup (seed 0, CUTOFF, stall3); R0 on that run; its 10k branch; the seed-0 BC-only
actor; the seed-0 evaluation of that run with 4 E2 and 4 E1 anchors (one per anchor time); then stage1.py's own
decision readers on the output. About 20-25 minutes, mostly the two training jobs. One line, from PowerShell or
Git Bash on Windows (it starts WSL itself):

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2 && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u stage1_smoke.py"
```

It prints `"status": "completed"` and per-job results, and writes `SMOKE_RESULT.json` (with each failed job's error)
plus every worker's `worker.log` and `report.json`. A failure here costs nothing under 5.0: fix the code before the
A4 package is sealed. It does not cover the no-stall world's training, the RANDOM arm, the full 96-anchor
evaluation, the three-seed decision or the 31-unit sealing logic, which run only in A4 itself (the Windows tests
cover the decision and sealing logic with fakes).

## How to run it (PowerShell, from `C:\Users\David\dev\BCA`, one command at a time)

0. **Rerun the GPU smoke from this folder first** (about 20-25 minutes; see above), then copy its blinded result into the package and commit it as the receipt:

```powershell
wsl -d Ubuntu -- bash -lc 'latest=$(ls -t /home/david/knee_edge_smoke | head -1); cp /home/david/knee_edge_smoke/$latest/SMOKE_RESULT.json /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/SMOKE_RESULT_$latest.json'
```

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/SMOKE_RESULT_*.json
```

```powershell
git commit -m "Record the knee_edge_ood_v2 GPU smoke receipt"
```

A4 requires v2's step 0.4 (A3) to have ended PASS.

1. **Approve, in your own words, at your terminal** (the text carries the required acknowledgement):

```powershell
.\.venv\Scripts\python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/';h=hashlib.sha256(open(p+'A4_MANIFEST.json','rb').read()).hexdigest();open(p+'A4_APPROVAL.json','x').write(json.dumps({'approval':'I approve A4 (Stage 1, native damage gate) for knee_edge_ood_v2, attempt 1. I acknowledge the written predictions: Stage 1 V0(d) may end INCONCLUSIVE, and after the v1 step 0.3 INCONCLUSIVE, Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

2. **Commit the approval** under your identity:

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/A4_APPROVAL.json
```

```powershell
git commit -m "Approve A4 for knee_edge_ood_v2 (attempt 1)"
```

3. **Run it once in WSL** (about 4-5 hours; keep the machine awake):

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/ && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u stage1.py --execute"
```
