# A2: step 0.3, frozen-checkpoint checks (user-run)

**What it does.** Forward passes and exact replay on the nine frozen native 20k cells of the central-quartile study.
No training and no new data collection. It runs in the recorded WSL GPU environment: the RTX 5070 Ti, the pinned
JAX 0.6.2 stack and the recorded worker environment (`XLA_FLAGS=--xla_gpu_triton_gemm_any=True`, matmul precision
"highest", x64 off), the setup that reproduced 143 of 143 contexts byte for byte. The driver sets that environment
itself before JAX loads, refuses a conflicting value, and asserts it in the preflight. Caps: 0.5M transitions or 30
minutes for the gated items; 200k transitions or 10 minutes for the warning-only onset contrast.

| Item | What is computed | Gate |
|---|---|---|
| Known-answer reproduction (4.8) | slots 1-5 exact; slot 0 (the actor's command); q1, twin_min, adjusted; Q^pi with the recorded batching; lambda via rerun1's own gradient audit, run live; past-knee counts per cell and pooled (117/1242, 118/1140, 126/1137); T15 identity; stall = no-stall within the knee | any mismatch: INCOMPLETE |
| G0.5, the fold test | F = Q1(-1) - Q1(-0.5) + 0.22499999 on the 16 steep high-load ordinary probes of each full-data critic | under 25% with F <= 0.5 in a seed: INCONCLUSIVE; median F > 0.5 or a share <= 0.5 in a seed: pass, flagged |
| Reach forecast | edge slope, curvature and equilibrium pi0 at the 32 strong-current probes and the 72 visited rows per cell, at the actor's own a1..a3 | probe < 0.35 or visited < 0.25 in any cell: FAIL (reach) |
| Onset contrast (warning only, last) | median \|KC15_s05\| / median rho15(s05) on the full-data critics | ratio > 0.3 in a seed: pass, flagged; median rho15 <= 0 or a sub-cap: "not computed" |

A flagged pass continues only with the written prediction "Stage 1 V0(b) or V0(d) may end INCONCLUSIVE",
acknowledged by you at A3.

**Conventions settled before the run (the protocol leaves them open):**
- At a share of exactly 50% with F <= 0.5, both the pass row and the flag row of the label table apply; the run flags
  it (the conservative reading; a flag only adds the written prediction). Medians of an even count average the two
  middle values.
- The saved 20k lambda is the mean of the three gradient-audit batches, as the sealed reach screen took it.
- The actor's own a1..a3 at a strong-current probe is the saved slot-0 command at that probe.
- A failure in the onset contrast after the gated items are sealed ends the attempt INCOMPLETE but keeps the gated
  label; a fresh approval then reruns the onset only.

**Attempts (5.0).** Each attempt writes `step03_run/attempt_<k>/`. Attempt k needs your approval naming k. A re-attempt
is allowed only after an INCOMPLETE; a preflight failure computes nothing and does not count; a second failure of the
same unit under unchanged code is INCONCLUSIVE; at most one code-only fix.

## How to run it

1. **Approve, in your own words, at your terminal.** From the repository root in Git Bash:

```bash
.venv/Scripts/python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v1/';h=hashlib.sha256(open(p+'A2_MANIFEST.json','rb').read()).hexdigest();open(p+'A2_APPROVAL.json','x').write(json.dumps({'approval':'I approve A2 (step 0.3, frozen-checkpoint checks) for knee_edge_ood_v1, attempt 1','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

2. **Commit the approval** under your identity:

```bash
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v1/A2_APPROVAL.json && git commit -m "Approve A2 for knee_edge_ood_v1 (attempt 1)"
```

3. **Run it once in WSL** (about 10-20 minutes):

```bash
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v1/step03.py --execute
```

It prints the label at the end. The full record is `step03_run/attempt_1/report.json`, with `gated.json` and one
`cell_*.json` per cell.

Files: `step03.py` (driver), `step03_math.py` (gate math), `test_step03.py` (19 tests, part of the 96-test suite),
this README, and `A2_MANIFEST.json` (their hashes).
