# Confirmation of the OOD-action overestimation comparison: how to run it

The rule is `CONFIRMATION_RULE.md`, written before any fresh-seed number exists. One sealed package, one approval,
one run: thirty GPU training jobs (about 4 hours) and twenty-four evaluations, user-run in the recorded WSL
environment. Agents never run the worker and never open a checkpoint.

## Before the run (done by the agent, on Windows)

- `cf_seeds.json` written and audited (`cf_seeds.py`): fresh training rows from the reserved confirmation namespace.
- Tests (a few seconds):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1 -p "test_*.py"
```

- Sealed: `seal.py` writes `CF_MANIFEST.json`.

## Approve, at your terminal (PowerShell; writes `CF_APPROVAL.json`, refuses to overwrite)

Read `CONFIRMATION_RULE.md` first: the doses, the bars and the budget are yours to accept or change. A change is
edited into the rule and re-sealed before any run. The sentences are matched exactly (case included).

```bash
.venv/Scripts/python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1/';h=hashlib.sha256(open(p+'CF_MANIFEST.json','rb').read()).hexdigest();open(p+'CF_APPROVAL.json','x').write(json.dumps({'approval':'I approve the confirmation of the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral TD3+BC host, doses kappa 21, 63 and 189, both worlds, attempt 1. And the BCA arms at kappa 21, 63 and 189 are approved, the confirmation seeds are fresh: no run has used them, and the dose being confirmed is 63; the other doses are a profile.','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

## Run (PowerShell launches the `Ubuntu` distro; about 4.5 hours, up to about 8.5 at Zeus's slowest pace; training cap 9 h, amendment A1; nothing else of yours on the GPU; keep the terminal open)

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1 && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u cf.py --execute 2>&1 | tee cf_console.log"
```

It prints the reading ("C1 / C2" at kappa 63), the dose profile and the paired differences. The record is
`cf_run/attempt_1/`: one folder per unit with `job.json`, `report.json`, `SUPERVISOR_RESULT.json` and `worker.log`;
`decision.json`; `CF_RESULT.json` (always written). Checkpoints are git-ignored. Paste the printed JSON. `--decide`
recomputes the decision from the saved reports without a GPU. After an INCOMPLETE attempt 1, remove the approval
file, write one naming attempt 2 (same sentences), and launch again: completed training units are reused and every
evaluation reruns.

## Attempt 2 on Titan (amendments A1 and A2), after the r5 screen finishes

The checkout is `C:\Users\David Bayha\BCA`; the WSL user is `dbayha`; the interpreter is `/home/dbayha/bca-gpu/bin/python`.
Never run this alongside the r5 screen on Titan's one GPU. In a Titan PowerShell at the checkout, remove the attempt-1
approval, then write attempt 2's with the helper (`../approve_confirmation_v1.py`, outside the sealed folder):

```powershell
Remove-Item experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1/CF_APPROVAL.json
```

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA' && /home/dbayha/bca-gpu/bin/python -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/approve_confirmation_v1.py --attempt 2"
```

Commit it, then start the run in a Titan window that stays open. It waits for the r5 screen to finish, then runs
(about 7.5 h of training):

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1' && while pgrep -f 'screen_r5.py --execute' > /dev/null; do sleep 60; done; PYTHONDONTWRITEBYTECODE=1 /home/dbayha/bca-gpu/bin/python -B -u cf.py --execute 2>&1 | tee cf_console.log"
```

## Files

| File | What it is |
|---|---|
| `CONFIRMATION_RULE.md` | the rule, written first |
| `cf_seeds.py`, `cf_seeds.json` | the fresh seeds (three training rows, visited states, probe rows) and their audit |
| `cf_worker.py` | the native screen's worker plus the confirmation rows, the per-job dose and the `cf_evaluate` kind |
| `cf_math.py` | the pre-registered readings and the dose profile |
| `cf.py` | the supervisor |
| `test_cf.py` | tests |
| `seal.py`, `CF_MANIFEST.json` | the seal |
