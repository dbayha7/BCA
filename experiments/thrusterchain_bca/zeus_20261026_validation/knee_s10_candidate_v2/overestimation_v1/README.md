# Overestimation comparison, native versus BCA (knee_s10_v2 CUTOFF data): how to run it

The rule is `OVERESTIMATION_RULE.md`, written before any number exists. One sealed package, one approval, one run.
Nine GPU training jobs and six CPU-heavy evaluations inside the same run, all user-run in the recorded WSL
environment. Agents never run the worker and never open a checkpoint.

## Before the run (done by the agent, on Windows)

- `oe_seeds.json` written and audited (`oe_seeds.py`).
- Tests (about half a minute):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/overestimation_v1 -p "test_*.py"
```

- Sealed: `seal.py` writes `OE_MANIFEST.json`.

## Approve, at your terminal (PowerShell; writes `OE_APPROVAL.json`, refuses to overwrite)

Read `OVERESTIMATION_RULE.md` first: the dose, the probes, the bars and the budget are yours to accept or change. A
change is edited into the rule and re-sealed before any run. The sentences are matched exactly (case included).

```bash
.venv/Scripts/python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/overestimation_v1/';h=hashlib.sha256(open(p+'OE_MANIFEST.json','rb').read()).hexdigest();open(p+'OE_APPROVAL.json','x').write(json.dumps({'approval':'I approve the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral TD3+BC host, kappa 63, attempt 1. And the BCA arm at kappa 63 is approved, and the comparison is exploratory and supports no protocol claim.','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

## Run (PowerShell launches the `Ubuntu` distro; about 1.5 to 2 hours; nothing else of yours on the GPU)

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/overestimation_v1 && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u oe.py --execute 2>&1 | tee oe_console.log"
```

It prints the reading, the paired differences and the canonical reads. The record is `oe_run/attempt_1/`: one
folder per unit with `job.json`, `report.json`, `SUPERVISOR_RESULT.json` and `worker.log`; `decision.json`;
`OE_RESULT.json` (always written). Checkpoints are git-ignored; their digests are in the reports. Paste the printed
JSON. `--decide` recomputes the decision from the saved reports without a GPU. After an INCOMPLETE attempt 1, remove
the approval file, write one naming attempt 2 (same sentences), and launch again: completed units are reused.

## Files

| File | What it is |
|---|---|
| `OVERESTIMATION_RULE.md` | the rule, written first |
| `oe_seeds.py`, `oe_seeds.json` | the two named seeds (visited states, dataset probe rows) and their audit |
| `oe_eval.py` | the `oe_evaluate` job: rollouts, own-action probes, dataset probes, exact references |
| `oe_math.py` | the frozen reads, paired differences and readings |
| `oe_worker.py` | the native screen's worker plus the `oe_evaluate` kind |
| `oe.py` | the supervisor |
| `test_oe.py` | tests |
| `seal.py`, `OE_MANIFEST.json` | the seal |
