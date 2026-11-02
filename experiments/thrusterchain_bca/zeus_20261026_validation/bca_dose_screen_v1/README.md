# BCA dose screen v1: BCA as built at kappa_A = 63 on the knee-edge forks (exploratory)

The rules are in `PROTOCOL.md`. In short: 12 BCA branches at kappa_A = 63 (21 times the built kappa of 3), one from
each exploratory 10k fork, paired with that fork's existing native branch through shared random numbers. Then an
evaluation, and a verdict per data arm, fixed in advance. The verdicts are EFFECT (selective, and through the Q
values), BEHAVIORAL EFFECT NOT THROUGH Q, NON-SELECTIVE REDUCTION, HARM, NO EFFECT, MIXED and INCOMPLETE ARM. It
supports no protocol claim.

**It stays on Q values.** BCA's actor maximizes the penalized value Q1 - kappa x U, so EFFECT also requires the
penalized Q to rank s10 below the knee, where the exact Q^pi does, on at least 20 points more anchors than raw Q1. The
native baseline from the exploratory evaluation:
- at 40-43 of the 48 gated anchors, s10 is truly worse than the knee;
- raw Q1 ranks that correctly at only 7-21% of them (CUTOFF 0.07 / 0.12 / 0.10, RANDOM 0.13 / 0.16 / 0.21 by seed).

**The prediction before you spend the GPU time** (`DOSE.json`, from the 10k fork audits, no branch outcome):
- Even at kappa_A, BCA is predicted to reverse 25-57% of native's outward pushes past the knee in the stall forks.
- In RANDOM it shows no world selectivity in 2 of 3 seeds: reversal 0.57 / 0.33 / 0.28 in stall3 against 0.58 / 0.33 /
  0.11 in nostall.
- The most likely verdicts are therefore NON-SELECTIVE REDUCTION or MIXED.
- The screen still measures what training does, because U is refit and refreshed during each branch.

## Files

| File | What it is |
|---|---|
| `PROTOCOL.md` | Rules, written before the screen's numbers; the decision was revised after review `wf_04029561-cff` |
| `DOSE.json`, `dose.py` | kappa_A by rule, with per-fork disclosures |
| `screen_worker.py` | GPU jobs: the v3 Stage 1 worker plus four declared in-process changes |
| `screen.py` | Supervisor: preflight outside the attempt, one attempt, decision, `--decide` recompute |
| `screen_math.py` | Readouts and verdicts |
| `test_screen.py` | 18 Windows tests, including the Q-value readouts, an end-to-end run against the real native evaluations, failure paths and the real module wiring |
| `SCREEN_MANIFEST.json` | Seal: these files, the v3 manifests the screen reuses and every exploratory record it reads |

## Cost

About 2.3 GPU hours of wall time (exploratory branches ran 516-802 s each, evaluations 463-517 s each):
- 12 branches, each capped at 1,800 s, with a 6-hour training cap;
- 3 evaluations, capped at 1 hour and 18,000,000 exact simulator steps.

Do not edit, test or check out anything in `knee_edge_ood_v3/` or the rerun1 folder while it runs. The workers re-hash
those files and would refuse.

## How to run it (PowerShell, from `C:\Users\David\dev\BCA`, one command at a time)

1. **Approve, in your own words, at your terminal:**

```powershell
.\.venv\Scripts\python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/bca_dose_screen_v1/';h=hashlib.sha256(open(p+'SCREEN_MANIFEST.json','rb').read()).hexdigest();open(p+'SCREEN_APPROVAL.json','x').write(json.dumps({'approval':'I approve the exploratory BCA dose screen v1 at kappa_A = 63 on the knee-edge exploratory forks; it supports no protocol claim','screen_manifest_sha256':h,'kappa_A':63},indent=2)+chr(10))"
```

2. **Commit the package and the approval** under your identity:

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/bca_dose_screen_v1
```

```powershell
git commit -m "Approve the exploratory BCA dose screen v1 (kappa_A 63)"
```

3. **Run it once in WSL** (about 2.3 hours; the console is copied to `screen_console.log`):

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/bca_dose_screen_v1 && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u screen.py --execute 2>&1 | tee screen_console.log"
```

4. When it finishes, paste the printed JSON. To recompute the decision from the saved reports later (JSON only, no GPU):

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/bca_dose_screen_v1 && /home/david/bca-operator-audit-20261026/bin/python -B screen.py --decide"
```
