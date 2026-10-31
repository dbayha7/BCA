# A5: Stage 2, the BCA comparison (user-run), knee_edge_ood_v2

**Requires** a Stage 1 PASS produced under the committed A4 package, and the A5 precondition recorded with it:
no-stall CUTOFF J_E1(20k) - J_E1(untrained) >= +50 in every seed. Otherwise the record reads "Stage 2 not requested
(precondition)". Inference is conditional on the forks that passed Stage 1.

**What runs** (`stage2.py`, a supervisor that never imports JAX; every GPU job is a separate `stage2_worker.py`
process, killed at its cap). Every fork is the authenticated Stage 1 10k warmup, restored exactly, and every adapter
runs the 4.4 assertions.

| Order | Units | What each does |
|---|---|---|
| 0 | preflight | runtime receipt, recorded device kind and worker environment (exempt from 5.0's second-failure rule) |
| 1 | precheck | BCA's width U from each stall fork's 10k shadow auxiliary at the 48 gated `d12_e2_anchors`, near-edge and past-knee slots with a1..a3 = 0. **D1:** AUROC_CUTOFF({s10, s15, s20} vs near-edge) >= 0.70; **D2:** AUROC_CUTOFF(s05) - AUROC_RANDOM(s05) >= +0.15; every seed. Either fails: **DETECTION-FAIL**, no branch is trained |
| 2 | 21 branches | per seed: stall CUTOFF BCA (kappa 3), distance (the warmup's matched coefficient) and a native replicate; stall RANDOM BCA and replicate; no-stall CUTOFF BCA and replicate. 10k updates each from the fork, with the fork's own Stage 2 branch seeds (shared by its branches: common random numbers). A replicate whose minibatch digest repeats its fork's Stage 1 native branch stops the unit |
| 3 | 3 evaluations | fresh `stage2_e2_bank` (128) and `stage2_e1_bank` (64): every branch, the forks' untrained actors and the Stage 1 natives re-evaluated; KC_s10 of Q1 - 3U (BCA) and of Q1 (replicate) at the 64 `stage2_e2_anchors` in stall CUTOFF |

**Caps** (fail closed): 1,800 s per branch; 4 h of training; 1 h and 18M transitions of evaluation (precheck and
evaluations).

| Outcome | Label |
|---|---|
| A job that does not complete, an exhausted cap, an assertion, a replicate repeating Stage 1's minibatches | INCOMPLETE |
| Any Stage 2 branch with J_E1 gain < +50 in any seed (V0-S2) | INCONCLUSIVE |
| D1 or D2 fails in any seed | DETECTION-FAIL (a finding about detection, not about the condition) |
| R1, R2 or R3 fails | NOT-PROMISING-consistent (a per-seed clause fails in all seeds, or a pooled clause fails with every seed on the failing side) or NOT-PROMISING-mixed |
| R1 (G > 0 every seed, mean >= +3.0), R2 (BCA >= distance every seed) and R3 (mean H >= -1.5) | PROMISING ("worth a larger study"), annotated when the replicate did not reproduce damage in a seed |

Reported in `decision.json`: on DETECTION-FAIL, the detection block only (D1, D2, FPR@95, the AUROC for W+ and of
Q1 - 3U, the magnitude control, D1/D2 under the actor's own a1..a3 with past-knee anchors excluded and counted);
after the evaluations, also damage reproduced, the G t-interval, recovery (with its guard), the repair screens per
seed and across seeds (with their guards), the nine replicate-minus-Stage-1-native differences (no common random
numbers), and BCA-minus-replicate entry and Delta-clip in no-stall CUTOFF. An INCOMPLETE attempt has no
`decision.json` unless the decision itself found it (a replicate repeating Stage 1's minibatches); its
`report.json` names the failed unit.

**Attempts (5.0)** follow `units.py`, shared with Stage 1 (see A4_README.md): supervisor-judged sealing; a
per-kind code fingerprint, so a fix confined to the evaluation's function body leaves the sealed precheck and
branches reusable, while a change to a worker module's imports or constants re-fingerprints every kind; bit-for-bit
reproduction of sealed units whose code changed, with markers that keep the original as the reference; at most one
package change over all attempts; time charged once.

**Settled conventions (the protocol leaves them open):**
- AUROC of Q1 - 3U counts an OOD action as detected when its score is lower (the detector is -(Q1 - 3U)).
- FPR@95 uses the largest threshold that still flags >= 95% of the D1 OOD actions (score >= threshold), pooled over
  the gated anchors.
- The magnitude control is U at a0 = 0 with a1 = +-0.69999999 against a1 = 0, at the gated anchors.
- Branch gradient audits are taken at 15k and 20k (the distance arm's with the fork actor as anchor), as in Stage 1.
- The paired t-interval uses t(0.975, 2) = 4.3027.

## How to run it (PowerShell, from `C:\Users\David\dev\BCA`, one command at a time)

A5 may be requested only after a Stage 1 PASS.

1. **Approve, in your own words, at your terminal** (the text carries the required acknowledgement):

```powershell
.\.venv\Scripts\python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/';h=hashlib.sha256(open(p+'A5_MANIFEST.json','rb').read()).hexdigest();open(p+'A5_APPROVAL.json','x').write(json.dumps({'approval':'I approve A5 (Stage 2, BCA comparison) for knee_edge_ood_v2, attempt 1','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

2. **Commit the approval** under your identity:

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/A5_APPROVAL.json
```

```powershell
git commit -m "Approve A5 for knee_edge_ood_v2 (attempt 1)"
```

3. **Run it once in WSL** (about 4-5 hours):

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/ && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u stage2.py --execute"
```
