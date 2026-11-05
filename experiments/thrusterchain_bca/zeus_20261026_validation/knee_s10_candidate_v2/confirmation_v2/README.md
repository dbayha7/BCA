# Confirmation v2 (kappa 189, five fresh seeds, both worlds): how to run it on Titan

The rule is `CONFIRMATION_RULE.md`, written before any v2 number exists. One sealed package, one approval, one run:
thirty GPU training jobs (about 7.5 hours on Titan, up to about 13 at the slowest pace either machine has shown) and
twenty evaluations, user-run in Titan's WSL environment. Agents never run the worker and never open a checkpoint.

## Before the run (done by the agent)

- `cf_seeds.json` written and audited (`cf_seeds.py`): five fresh training rows from the new namespace
  `knee_s10_candidate_v2/confirmation_v2/`, disjoint from every integer on record, confirmation v1's seeds included.
- Tests (a few seconds; 16 tests):

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v2 -p "test_*.py"
```

- Sealed: `seal.py` writes `CF_MANIFEST.json`.

## On Titan: update the checkout (only while no run is live there)

The checkout is `C:\Users\David Bayha\BCA`; the WSL user is `dbayha`; the interpreter is `/home/dbayha/bca-gpu/bin/python`.
In a Titan PowerShell, with these commands rather than GitHub Desktop:

```powershell
cd "C:\Users\David Bayha\BCA"
git switch codex/thrusterchain-bca-handoff-titan
git pull
```

## Approve, at your terminal (writes `CF_APPROVAL.json`, refuses to overwrite)

Read `CONFIRMATION_RULE.md` first: the dose, the seeds, the bars, the I1 tolerance and the budget are yours to accept or
change. A change is edited into the rule and re-sealed before any run. The helper writes the five sentences exactly.

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA' && /home/dbayha/bca-gpu/bin/python -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/approve_confirmation_v2.py --attempt 1"
```

Commit the approval and push:

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v2/CF_APPROVAL.json
git commit -m "Approve confirmation v2 attempt 1 on Titan"
git push origin HEAD:codex/thrusterchain-bca-handoff-titan
```

## Run (Titan, about 7.5 hours; nothing else on the GPU; keep the window open)

Close GitHub Desktop first and leave the checkout alone until it prints the reading: a branch switch or stash mid-run
removes the run's untracked record files, which is what ended v1's attempt 2. Start it only when no other GPU job is
running on Titan (`nvidia-smi` in WSL lists none).

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v2' && PYTHONDONTWRITEBYTECODE=1 /home/dbayha/bca-gpu/bin/python -B -u cf.py --execute 2>&1 | tee cf_console.log"
```

It prints the reading ("C1 / C2" at kappa 189), the description and the paired differences. The record is
`cf_run/attempt_1/`: one folder per unit with `job.json`, `report.json`, `SUPERVISOR_RESULT.json` and `worker.log`;
`decision.json`; `CF_RESULT.json` (always written). Checkpoints are git-ignored. `--decide` recomputes the decision from
the saved reports without a GPU. Commit the record and push:

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v2/cf_run/attempt_1
git commit -m "Record confirmation v2 attempt 1 on Titan"
git push origin HEAD:codex/thrusterchain-bca-handoff-titan
```

After an INCOMPLETE attempt 1, remove the approval file, write one naming attempt 2 with the helper, and launch again:
completed training units are reused and every evaluation reruns (the rule's "Attempts").

## Files

| File | What it is |
|---|---|
| `CONFIRMATION_RULE.md` | the rule, written first |
| `cf_seeds.py`, `cf_seeds.json` | the fresh seeds (five training rows, visited states, probe rows) and their audit |
| `cf_worker.py` | the native screen's worker plus v2's rows, the kappa 189 dose and the `cf_evaluate` kind |
| `cf_math.py` | the pre-registered readings (C1, C2, the I1 tolerance) and the description |
| `cf.py` | the supervisor |
| `device_kinds.py` | the allowed GPU kinds (Titan, Zeus), carried from confirmation v1 |
| `test_cf.py` | tests |
| `seal.py`, `CF_MANIFEST.json` | the seal |
| `../approve_confirmation_v2.py` | the approval helper, outside the sealed folder |
