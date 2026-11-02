# Native screen for candidate knee_s10_v2 under r5: how to run it

**State after amendment A5 (2026-11-02):** attempt 2 ran on Titan and was killed 32 minutes in by a forced Windows
Update restart (INCOMPLETE, on record under `screen_run/`; no outcome value existed). Attempt 3, the last, runs on
Titan with Windows updates paused. Titan's dry run of 2026-11-02 stands (A5 changes only `screen_r5.py` and the
tests), so the order is: tests; the seal; the approval with `--attempt 3`; commit and push; the run (the Titan
command below), in a window on Titan itself.

**Earlier state, after amendment A2 (2026-11-01 late):** attempt 1 ran on Zeus and was interrupted by the user (INCOMPLETE,
on record under `screen_run/`). Attempt 2 runs on Titan. Because A2 changed the worker, the package is unsealed
until the dry run is redone **on Titan**; the order below is for Titan, from a pulled clone, with the recorded
operator environment at the same path. If Titan's interpreter lives elsewhere, substitute its path in every command.

On Titan, in order: `git pull`; tests (step 1; the WSL interpreter works for them too, with `JAX_PLATFORMS=cpu`);
the dry run (step 2; it must report `device_kind` "NVIDIA GeForce RTX 5070 Ti"); the seal (step 3); the approval
with `--attempt 2` via `../approve_native_screen_r5.py`; commit and push; the run (step 5). Nothing may run on Zeus
in the meantime.

**Titan specifics (amendment A3).** The checkout is `C:\Users\David Bayha\BCA` (quote the path in WSL), the WSL user
is `dbayha`, the interpreter is `/home/dbayha/bca-gpu/bin/python`, and **every** WSL invocation sets
`OPENBLAS_CORETYPE=Haswell`, or pi_ref's gains pin refuses in stage 0 (nothing spent). The dry run and the run:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen_r5' && OPENBLAS_CORETYPE=Haswell PYTHONDONTWRITEBYTECODE=1 /home/dbayha/bca-gpu/bin/python -B -u screen_r5.py --dry-run 2>&1 | tee dry_run_console.log"
```

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen_r5' && OPENBLAS_CORETYPE=Haswell PYTHONDONTWRITEBYTECODE=1 /home/dbayha/bca-gpu/bin/python -B -u screen_r5.py --execute 2>&1 | tee screen_console.log"
```

The rule is `SCREEN_RULE_r5.md`, written before any screen number exists; the definition is
`../../ood_definition_v4/REVISION_r5.md`; the reasons are `../../ood_definition_v4/DESIGN_AUDIT_r4_20261101.md`. One
package, one dry run, one seal, one approval, one run. Everything GPU is user-run in the recorded WSL environment;
agents never run the worker and never open a checkpoint. The r4 package (`../native_screen/`) stays sealed and
untouched; this package imports its unchanged modules.

## Order

1. **Tests** (Windows, about 15 s):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen_r5 -p "test_*.py"
```

2. **The dry run** (WSL, precondition 3 of r5; about 45 to 60 minutes, five full-length smoke training jobs on the
   GPU plus the instrument unit on the real anchors; output outside the repository; nothing counted as an attempt):

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen_r5 && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u screen_r5.py --dry-run 2>&1 | tee dry_run_console.log"
```

It prints `status`, one `completed` flag per job, and the receipt path. The receipt (`dry_run/DRY_RUN_RECEIPT.json`,
structure only) is what the seal and the approval check. If a job fails, the fix is made here, the tests rerun, and
the dry run rerun; no attempt is spent.

3. **Seal** (Windows; refuses without a completed dry run whose module hashes equal the files here):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen_r5/seal_r5.py
```

4. **Approve, at your terminal** (writes `SCREEN_APPROVAL.json`; refuses to overwrite). Read `SCREEN_RULE_r5.md`
   first: the dose, the control, the bars, the nulls, the budget and the RS1 sentence are yours to accept or change; a
   change is edited into the rule and re-sealed before any run. The sentences are matched exactly; `attempt` is 1,
   2 after an INCOMPLETE attempt 1, or 3 after an INCOMPLETE or killed attempt 2 (amendment A5; stage-0 failures
   spend no attempt). Remove an older approval file first. The
   helper lives outside the sealed folder (PowerShell quoting mangles the apostrophe in a one-liner):

```bash
.venv/Scripts/python.exe experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/approve_native_screen_r5.py --attempt 3
```

Commit the package and the approval under your identity:

```bash
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen_r5 experiments/thrusterchain_bca/zeus_20261026_validation/ood_definition_v4/REVISION_r5.md experiments/thrusterchain_bca/zeus_20261026_validation/ood_definition_v4/DESIGN_AUDIT_r4_20261101.md && git commit -m "Seal and approve the r5 native screen for candidate knee_s10_v2 (kappa 63, attempt 1)"
```

5. **Run** (WSL; stage 0 takes about 5 minutes, then about 6 to 8 GPU hours of training on Zeus or 9 to 10 on Titan, plus evaluation; training cap 14 h, amendment A4):

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/native_screen_r5 && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u screen_r5.py --execute 2>&1 | tee screen_console.log"
```

It prints the reading and the deciding reads. The record is `screen_run/run_<utc>/`: one folder per unit
(`job.json`, `report.json`, `SUPERVISOR_RESULT.json`, `worker.log`), `TRAINING_STARTED.json` once the attempt counts,
`decision.json`, `SCREEN_RESULT.json` (always written). Checkpoints, labels and fitted parameters are git-ignored;
their digests are in the reports. Paste the printed JSON. `--decide` recomputes the decision from the saved reports
without a GPU.

## Files

| File | What it is |
|---|---|
| `SCREEN_RULE_r5.md` | the rule, written first |
| `screen_math_r5.py` | the frozen reads, nulls and readings |
| `screen_worker_r5.py` | the jobs: Stage 1's kinds plus `bca_branch`, `evaluate_bca`, `i0`, `g1` |
| `i0_unit.py` | the stage-0 instrument unit |
| `g1_unit_r5.py` | the reported calibration unit (fit to a plateau, read relative to its in-support error) |
| `screen_r5.py` | the supervisor: stage 0, attempts counted from the first training job, `--dry-run`, `--decide` |
| `dry_run.py` | the WSL dry run |
| `seal_r5.py`, `SCREEN_MANIFEST.json` | the seal |
| `dry_run/DRY_RUN_RECEIPT.json` | the dry run's structure-only receipt, hashed by the manifest |
| `test_screen_r5.py` | tests |
| reused from `../native_screen/` | `screen_bind.py` (the seven bindings), `screen_data.py` (the A1 loader), `screen_seeds.json`, `collection/collection.json` |
