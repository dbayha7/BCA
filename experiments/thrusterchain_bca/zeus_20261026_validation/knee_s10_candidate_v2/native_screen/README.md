# Native screen for candidate knee_s10_v2: how to run it

The rule is `SCREEN_RULE.md`, written before any screen number exists; the definition is
`../../ood_definition_v4/REVISION_r4.md`. One sealed package, one approval, one run. Everything GPU is user-run in the
recorded WSL environment; the g1 unit is CPU and runs inside the same run. Agents never run the worker and never open
a checkpoint.

## Before the run (done by the agent, on Windows, exact arithmetic only)

- `screen_seeds.json` written and audited (`screen_seeds.py`).
- `collection/collection.json` written by `prepare.py` from the formal G2 v2 datasets by digest.
- Tests: about a minute.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen -p "test_*.py"
```

- Sealed: `seal.py` writes `SCREEN_MANIFEST.json`.

## Approve, at your terminal (writes `SCREEN_APPROVAL.json`; refuses to overwrite)

Read `SCREEN_RULE.md` first: the dose, the control, the bars, the budget and the RS1 sentence are yours to accept or
change. A change is edited into the rule and re-sealed before any run.

The sentences are matched exactly (case included); set `attempt` to the attempt the rule allows (1; 2 after an
INCOMPLETE attempt 1; 3 only under amendment A1), and remove an older approval file first, since the writer refuses to
overwrite. Both commands run in PowerShell (the Windows venv cannot run inside a WSL shell, and the `wsl` launcher
exists only outside one). The approval below names attempt 3: attempts 1 and 2 are on record as INCOMPLETE
(`screen_run/`), and amendment A1 in `SCREEN_RULE.md` allows the third. Attempt 3 reuses the completed preflight and
starts at the g1 unit.

```bash
.venv/Scripts/python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen/';h=hashlib.sha256(open(p+'SCREEN_MANIFEST.json','rb').read()).hexdigest();open(p+'SCREEN_APPROVAL.json','x').write(json.dumps({'approval':'I approve the native screen for candidate knee_s10_v2 under OOD definition v4 r4 at kappa 63, attempt 3. I adopt definition v4 r4 for this candidate. RS1: the scored anchors are retained-supported; I overrule the start-share primary miss (121 of 128 against 122, a one-draw threshold); the secondary\'s agreement with D3 is on record. And the BCA arm at kappa 63 is approved, and the screen is exploratory and supports no protocol claim.','package_manifest_sha256':h,'attempt':3},indent=2)+chr(10))"
```

Launch from Windows PowerShell with the distro named (the recorded environment lives in the `Ubuntu` distro, not the
default one). Wait until no other job of yours is on the GPU (the minimal-comparison recovery run, if still training):

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u screen.py --execute 2>&1 | tee screen_console.log"
```

Commit it under your identity (in WSL pass `-c user.name=... -c user.email=...` as before):

```bash
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen/SCREEN_APPROVAL.json && git commit -m "Approve the native screen for candidate knee_s10_v2 (r4, kappa 63, attempt 3)"
```

## Run (WSL, the recorded GPU environment; about 6 to 8 GPU hours for the training jobs plus evaluation)

```bash
/home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen/screen.py --execute
```

It prints the reading and every consensus. The record is `screen_run/attempt_<n>/`: one folder per unit with
`job.json`, `report.json`, `SUPERVISOR_RESULT.json` and `worker.log`; `decision.json`; `SCREEN_RESULT.json` (always
written). Checkpoints, labels and fitted parameters are git-ignored; their digests are in the reports. Paste the
printed JSON.

`--decide` recomputes the decision from the saved reports without a GPU.

## Files

| File | What it is |
|---|---|
| `SCREEN_RULE.md` | the rule, written first |
| `screen_seeds.py`, `screen_seeds.json` | the named seeds (training rows, data plan, g1 fits) and their audit |
| `screen_bind.py` | the seven declared in-process changes to the Stage 1 pipeline |
| `screen_data.py` | change 7 (amendment A1): the authenticated, non-regenerating loader of the two archives |
| `prepare.py`, `collection/collection.json` | the datasets by digest and the data plan the jobs re-derive |
| `screen_worker.py` | the jobs: Stage 1's kinds plus `bca_branch`, `evaluate_bca`, `g1` |
| `g1_unit.py` | the G1 unit (rule and modules in `../g1/`) |
| `screen_math.py` | the frozen reads and readings |
| `screen.py` | the supervisor |
| `test_screen.py` | tests |
| `seal.py`, `SCREEN_MANIFEST.json` | the seal |
