# Formal G2 for candidate knee_s10_v2: how to run it

The rule is `G2_FORMAL_RULE_v2.md`, written before any formal number exists. The gate is exact arithmetic on a fresh
collection under the widened recorder and the committed pi_ref, with anchors built by the v2 prefix rule, identical in
both worlds: no training, no GPU, no checkpoint. It runs in the Windows venv in about a minute (v1's took 31 s).

Sealed into `G2_FORMAL_MANIFEST.json`: this package, `g2_seeds_v2.json` (new namespace, new entropy and streams,
audited against every seed on record including v1's formal ones), `d4_bounds.json` (v1's sealed two-sided bounds,
byte for byte), the v2 prefix rule and its development record, and the v1 files v2 inherits (pi_ref, recorder,
formal driver, seeds, bounds, manifest, rule). Nothing in the package is edited after sealing; the driver checks
every hash before and after the run.

## Decisions your approval records

The approval text must contain the sentence below: adopting r3 for this candidate, the primary scoring set (all 96
gated anchors; the pi_ref-conditioned subset is reported, never decides), and the attempt number.

## Run (from `C:\Users\David\dev\BCA`, one command at a time; bash syntax works in WSL and Git Bash)

1. Tests, about half a minute:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/formal_g2 -p "test_*.py"
```

2. Approve, at your terminal (writes `G2_FORMAL_APPROVAL.json`; refuses to overwrite):

```bash
.venv/Scripts/python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/formal_g2/';h=hashlib.sha256(open(p+'G2_FORMAL_MANIFEST.json','rb').read()).hexdigest();open(p+'G2_FORMAL_APPROVAL.json','x').write(json.dumps({'approval':'I approve the formal G2 for candidate knee_s10_v2 under OOD definition v4 r3, scoring all 96 gated anchors, attempt 1. I adopt definition v4 r3 for this candidate; the sealed rules of knee_edge_ood_v1-v3 are unchanged.','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

3. Commit the approval under your identity (in WSL, pass your identity with `-c` as before):

```bash
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/formal_g2/G2_FORMAL_APPROVAL.json && git -c user.name="David Bayha" -c user.email="35092758+dbayha7@users.noreply.github.com" commit -m "Approve the formal G2 for candidate knee_s10_v2 (attempt 1)"
```

4. Run it once:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/formal_g2/g2_formal_v2.py --execute
```

It prints the headline and every label. The record is `formal_g2_run/attempt_1/report.json`, with `collection.json`
and the two datasets (`knee_s10_v2_stall3.npz`, `knee_s10_v2_nostall.npz`, git-ignored, digests in the report).
Paste the printed JSON.

## Files

| File | What it is |
|---|---|
| `G2_FORMAL_RULE_v2.md` | the rule, written first |
| `g2_seeds_v2.py`, `g2_seeds_v2.json` | the named seeds (namespace `knee_s10_candidate_v2/`), their derivation and the collision audit |
| `d4_bounds.json` | v1's sealed two-sided D4 bounds, byte for byte (recorder unchanged) |
| `g2_formal_v2.py` | the driver: seeds, anchors (P1), preflight, report; measurements and labels from v1's sealed driver |
| `test_g2_formal_v2.py` | tests |
| `seal.py`, `G2_FORMAL_MANIFEST.json` | the seal |
