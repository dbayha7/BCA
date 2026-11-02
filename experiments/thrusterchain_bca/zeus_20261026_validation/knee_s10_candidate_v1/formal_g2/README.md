# Formal G2 for candidate knee_s10_v1: how to run it

The rule is `G2_FORMAL_RULE.md`, written before any formal number exists. The gate is exact arithmetic on a fresh
collection under the widened recorder and the committed pi_ref: no training, no GPU, no checkpoint. It runs in the
Windows venv in a few minutes (the development screen took 29 s plus the collection).

Before you start, these files exist and are sealed into `G2_FORMAL_MANIFEST.json`: `g2_seeds.json` (new entropy and
anchor streams, audited against every seed on record) and `d4_bounds.json` (two-sided six-SD bounds for this
recorder). Nothing in the package is edited after sealing; the driver checks every hash before and after the run.

## Decisions your approval records

Your approval text must contain, in your own words, the sentence below. It states the three decisions the definition
leaves to you: adopting r3 for this candidate, the primary scoring set (all 96 gated anchors; the pi_ref-conditioned
subset is reported, never decides), and the attempt number.

## Run (PowerShell, from `C:\Users\David\dev\BCA`, one command at a time)

1. Tests (Windows, about a minute):

```powershell
$env:PYTHONDONTWRITEBYTECODE=1; .\.venv\Scripts\python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v1/formal_g2 -p "test_*.py"
```

2. Approve, at your terminal, in your own words (the text carries the required sentence):

```powershell
.\.venv\Scripts\python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v1/formal_g2/';h=hashlib.sha256(open(p+'G2_FORMAL_MANIFEST.json','rb').read()).hexdigest();open(p+'G2_FORMAL_APPROVAL.json','x').write(json.dumps({'approval':'I approve the formal G2 for candidate knee_s10_v1 under OOD definition v4 r3, scoring all 96 gated anchors, attempt 1. I adopt definition v4 r3 for this candidate; the sealed rules of knee_edge_ood_v1-v3 are unchanged.','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

3. Commit the approval under your identity:

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v1/formal_g2/G2_FORMAL_APPROVAL.json
```

```powershell
git commit -m "Approve the formal G2 for candidate knee_s10_v1 (attempt 1)"
```

4. Run it once (Windows venv, a few minutes):

```powershell
$env:PYTHONDONTWRITEBYTECODE=1; .\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v1/formal_g2/g2_formal.py --execute
```

It prints the headline and every label. The record is `formal_g2_run/attempt_1/report.json`, with `collection.json`
and the two datasets (`knee_s10_v1_stall3.npz`, `knee_s10_v1_nostall.npz`, git-ignored, digests in the report).
Paste the printed JSON.

## Files

| File | What it is |
|---|---|
| `G2_FORMAL_RULE.md` | the rule, written first |
| `g2_seeds.py`, `g2_seeds.json` | the named seeds, their derivation and the collision audit |
| `d4_bounds.py`, `d4_bounds.json` | the Monte Carlo and the sealed two-sided D4 bounds |
| `g2_formal.py` | the driver (preflight, collection, checks, labels, accounting, report) |
| `test_g2_formal.py` | tests of the pure parts |
| `seal.py`, `G2_FORMAL_MANIFEST.json` | the seal |
| `../pi_ref.py`, `../recorder_wide.py` | the committed continuation and recorder (hashed in the manifest) |
