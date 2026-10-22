# docs/bca_mechanics: BCA verified one component at a time

BCA (Bayesian Conformal Aggregation) is the project's method for putting an error band around an offline critic and
turning that band into caution for the actor. This folder checks it on the ThrusterChain test bed module by module:
what each component computes, what "working" means as a checkable statement, a synthetic test with the truth known, a
recomputation from the recorded inputs of the studies, a figure, a verdict (works / works with a caveat / does not
work / untested) and the exact command that reproduces every number. Nothing was trained for it.

## Read in this order

1. `index.html`: everything on one self-contained page (summary, the pipeline, one section per module with labeled
   pseudocode, figure, table, verdict and reproduction box, the penalty evidence, scope, glossary). Open it in any
   browser; it needs no network.
2. `MEETING_BRIEF.md`: the one-page version to read aloud.
3. `MODULES.md`: the map of the eleven modules (M1-M11), with a glossary, the verdict per module, the consistent and
   exceptional cases, and the corrections the module pages made to it.
4. `PENALTY_EVIDENCE.md`: the answer to the two questions from the meeting (is the penalty low under IID and higher
   under a shift; is the importance weighting right), case by case, with the mechanism.
5. `modules/M1.md` to `modules/M11.md`: one page per module (verdict, the one sentence, plain words, the checkable
   statements, labeled pseudocode with `path:line`, the synthetic test, the real-data test, the figure, the evidence
   labels, the summary for the meeting, reproduction).
6. `figures/`: `M1.svg` to `M11.svg` and the two penalty figures (`penalty_iid_vs_shift.svg`,
   `penalty_iw_minus_standard.svg`); all text in them is text, not outlines.
7. `scripts/`: the code that printed every number (below). `data/`: the scripts' outputs that the pages quote
   (committed with `git add -f`, because the repository ignores folders named `data/`).

## Environments

- **Windows venv** (the desktop): Python 3.14.2, NumPy 2.5.4, SciPy 1.18.1, JAX 0.12.0 (CPU), flax 0.12.10,
  Matplotlib 3.11.2; `git` on PATH with the branches `origin/main`, `origin/claude/near80-run-files` and
  `origin/claude/shift-hosts` fetched. Set `PYTHONPATH` to the repository root (so `calibration/` imports),
  `JAX_PLATFORMS=cpu` and `PYTHONDONTWRITEBYTECODE=1`. Every `M<k>_test.py` and the penalty scripts run here.
- **The pinned environment** (the versions the studies ran under, needed whenever a recomputed number must equal a
  recorded one to the last bit): Python 3.10.20, NumPy 2.2.6, JAX 0.6.2, flax 0.10.7, SciPy 1.15.3, on CPU, in the WSL
  clone `/home/user/bca` with the venv `/tmp/claude-0/bcavenv`; the gate's pre-registered flags are
  `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1` with `JAX_ENABLE_X64`, `XLA_FLAGS` and `JAX_DEFAULT_MATMUL_PRECISION` unset.
  The `*.sh` job files run there, from Windows through the bridge
  `pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile <job.sh> -TimeoutSec <s> -Name <id>` (the bridge's
  header explains the watcher the owner starts inside WSL), or directly inside WSL as each job file's header says.
  They read the run files under `/home/user/bca/runs/shift_explore` read-only and write under `/tmp/bridge/work/`.
- **Run files**: the near-80 band records are also on branch `claude/near80-run-files`; stage 2's run files exist
  only in the WSL clone (`\\wsl.localhost\Ubuntu-24.04\home\user\bca\runs\shift_explore`), which the scripts reach
  from Windows when it is mounted (from Git Bash pass the root with forward slashes, `//wsl.localhost/...`).

## How to run every script

All commands from the repository root (the worktree). Times are from the desktop.

| Script | Command | Where | Time | Writes |
|---|---|---|---|---|
| `M1_test.py` | `python docs/bca_mechanics/scripts/M1_test.py` | Windows venv | 10 s | `figures/M1.svg`; prints S1-S4, R1-R3 |
| `M1_real.sh` (runs `M1_real.py`) | `wslrun.ps1 -ScriptFile M:\docs\bca_mechanics\scripts\M1_real.sh -TimeoutSec 900 -Name m1real` | pinned | 138 s | `data/M1/M1_real_results.json`, `M1_real_rows.npz` |
| `M1_unittest.sh` | `wslrun.ps1 -ScriptFile ...\M1_unittest.sh -TimeoutSec 1800 -Name m1unittest` | pinned | 20 s | prints `ok` |
| `M2_test.py` | `python docs/bca_mechanics/scripts/M2_test.py` (`--quick` for a smoke test) | Windows venv | 30 s | `figures/M2.svg`; prints tables A-D |
| `M2_pinned_job.sh` (runs `M2_pinned.py`) | `wslrun.ps1 -ScriptFile docs/bca_mechanics/scripts/M2_pinned_job.sh -TimeoutSec 1800` | pinned | 3 min | `data/M2_pinned_recompute.json` |
| `M3_test.py` | `python docs/bca_mechanics/scripts/M3_test.py` | Windows venv | 90 s | `figures/M3.svg`; Parts A-D |
| `M3_recompute_pinned.sh` (`M3_recompute_pinned.py`) | `wslrun.ps1 -ScriptFile docs/bca_mechanics/scripts/M3_recompute_pinned.sh -TimeoutSec 3000 -Name m3recompute` (copy the `.py` to `/tmp/bridge/work/m3/` first) | pinned | 141 s | `data/M3_pinned_recompute.json` |
| `M3_bank_rows_pinned.sh` (`M3_bank_rows_pinned.py`) | likewise | pinned | 1 min | `data/M3_bank_rows.npz`, `.json` |
| `M4_test.py` | `python docs\bca_mechanics\scripts\M4_test.py` (`--fast`, `--no-figure`) | Windows venv | 18 s | `figures/M4.svg`; S1-S6 and the real-data table |
| `M4_pinned_job.sh` (`M4_pinned.py`) | `wslrun.ps1 -ScriptFile M:\docs\bca_mechanics\scripts\M4_pinned_job.sh -TimeoutSec 1500 -Name m4pinned` | pinned | 2 min | `data/M4_real_data.json` |
| `M5_test.py` | `python docs\bca_mechanics\scripts\M5_test.py --root M:\ --figure docs\bca_mechanics\figures\M5.svg --json docs\bca_mechanics\data\M5_results.json --near80-root <WSL>\runs\shift_explore\near80 --stage2-root <WSL>\runs\shift_explore\stage2` | Windows venv | 56 s | `figures/M5.svg`, `data/M5_results.json`; 14 checks |
| `M5_pinned_job.sh` | `wslrun.ps1 -ScriptFile docs\bca_mechanics\scripts\M5_pinned_job.sh -TimeoutSec 1800` | pinned | 72 s | `data/M5_pinned_output.txt`, `M5_pinned_results.json` (15 unit tests OK) |
| `M6_test.py` | `PYTHONPATH=. python docs/bca_mechanics/scripts/M6_test.py [--stage2 DIR] [--quick]` | Windows venv | 35 s | `figures/M6.svg`; A1-A5, B1-B6 |
| `M6_fresh_rows.sh` (`M6_fresh_rows.py`) | `wslrun.ps1 -ScriptFile docs/bca_mechanics/scripts/M6_fresh_rows.sh -TimeoutSec 1800 -Name m6all` | pinned | 3 min | `data/M6_fresh_rows_coverage.json` |
| `M7_test.py` | `python docs\bca_mechanics\scripts\M7_test.py --root M:\ [--stage2 <dir>]` | Windows venv | 20 s | `figures/M7.svg`, `data/M7_table.txt` |
| `M7_near80_training_dose.sh`, `M7_pinned_unit_tests.sh`, `M7_near80_normalization_check.sh` | `wslrun.ps1 -ScriptFile <job> -TimeoutSec 900` | pinned | 2 min, 46 s, 10 s | `data/M7_near80_training_dose.json`, `M7_pinned_unit_tests.out`, `M7_near80_normalization_check.out` |
| `M8_test.py` | `python docs/bca_mechanics/scripts/M8_test.py --runs //wsl.localhost/Ubuntu-24.04/home/user/bca` (or `--copy <dir>`, or no argument) | Windows venv (NumPy, Matplotlib only) | 8 s | `figures/M8.svg`; Parts A, S, B |
| `M9_records.py` | `python docs/bca_mechanics/scripts/M9_records.py` | Windows venv | 1 s | `data/M9/M9_records.json` |
| `M9_test.py` | `python docs/bca_mechanics/scripts/M9_test.py [--runs <runs/shift_explore> \| --copy <dir>]` | Windows venv | 15 s | `figures/M9.svg`, `data/M9/M9_r2.json`; output kept as `data/M9/M9_test.out` |
| `M9_width_recompute.sh` (embeds `M9_width_recompute.py`), `M9_unit_tests.sh` | `wslrun.ps1 -ScriptFile <job> -TimeoutSec 900 -Name <id>` | pinned | 1 min, 36 s | `data/M9/width_recompute_pinned.out`, `unit_tests_pinned.out` |
| `M10_test.py` | `python docs\bca_mechanics\scripts\M10_test.py` (`--fast`; `--figure-only`; `--png <path>`) | Windows venv | 410 s | `figures/M10.svg`, `data/M10/M10_test_results.json` |
| `M10_recompute_job.sh` (`M10_recompute_pinned.py`) | `wslrun.ps1 -ScriptFile docs\bca_mechanics\scripts\M10_recompute_job.sh -TimeoutSec 1800` | pinned | 90 s per setting | `data/M10/recomputed_certificates.json`, `overestimates.npz` |
| `M11_test.py` | `python docs/bca_mechanics/scripts/M11_test.py` | Windows venv | 5 s | `figures/M11.svg`; 39 checks |
| `M11_bridge_job.sh` | `wslrun.ps1 -ScriptFile docs\bca_mechanics\scripts\M11_bridge_job.sh -TimeoutSec 1500 -Name m11ramp` | pinned | 13 s of compute | the 24 files of `data/M11/` |
| `certificate_table.py` | `python docs/bca_mechanics/scripts/certificate_table.py` | any Python 3 (standard library) | 1 s | prints the 40 certificates and the three recorded stops |
| `threshold_checks.py` | `PYTHONPATH=. python docs/bca_mechanics/scripts/threshold_checks.py` | Windows venv | 16 s | seven synthetic checks of M4-M7, all PASS |
| `penalty_cases.py` | `python docs/bca_mechanics/scripts/penalty_cases.py --root . --case all --runs <runs/shift_explore> --write-json docs/bca_mechanics/data/penalty_cases.json` (or `--hosts <results.json> --stage2-copies <dir>` instead of `--runs`) | Windows venv | 8 s | `data/penalty_cases.json`; output kept as `data/penalty_cases.out` |
| `penalty_recompute.py` | `PYTHONPATH=/home/user/bca JAX_PLATFORMS=cpu /tmp/claude-0/bcavenv/bin/python docs/bca_mechanics/scripts/penalty_recompute.py --runs /home/user/bca/runs/shift_explore --out docs/bca_mechanics/data/penalty_recompute.json` | pinned (case A reproduces the recorded stream only there) | 90 s | `data/penalty_recompute.json` |
| `penalty_real_current.py` | `python docs/bca_mechanics/scripts/penalty_real_current.py --runs <runs/shift_explore> --out docs/bca_mechanics/data/penalty_real_current.json` | Windows venv or pinned (same stream) | 20 s | `data/penalty_real_current.json` |
| `penalty_figures.py` | `python docs/bca_mechanics/scripts/penalty_figures.py --cases docs/bca_mechanics/data/penalty_cases.json --recompute docs/bca_mechanics/data/penalty_recompute.json --out docs/bca_mechanics/figures` | Windows venv | 3 s | the two penalty figures |

`<WSL>` stands for `\\wsl.localhost\Ubuntu-24.04\home\user\bca`; `wslrun.ps1` for `pwsh -NoProfile -File <scratchpad>\wslrun.ps1`.
Every script's header states its inputs, outputs and the layout it expects for copies of run files; every script is
read-only on the records and on the run folders.

## What is where in `data/`

`M1/` (M1's pinned recomputation), `M2_pinned_recompute.json`, `M2_test_output.txt`, `M2_pinned_job_output.txt`,
`M3_pinned_recompute.json`, `M3_bank_rows.{npz,json}`, `M4_real_data.json`, `M5_results.json`,
`M5_pinned_results.json`, `M5_pinned_output.txt`, `M6_fresh_rows_coverage.json`, `M7_table.txt`,
`M7_near80_training_dose.json`, `M7_pinned_unit_tests.out`, `M7_near80_normalization_check.out`, `M9/` (records,
the R2 table, the test output, the two pinned outputs), `M10/` (the recomputed certificates, the per-row overestimates,
the test results), `M11/` (the 24 bridge outputs), `penalty_cases.{json,out}`, `penalty_recompute.json`,
`penalty_real_current.json`.

## Records read (never written)

`outputs/shift_iw/v5/<quality>/report.json` and `summary.json` (branch `main`); `outputs/shift_iw/explore/**`;
`origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json`; `outputs/shift/beyond-the-data/restructure/step1/results/results.json`
and `step2_results/results.json`; `outputs/shift/beyond-the-data/near80/results-20261020T115246Z/`;
`outputs/shift/beyond-the-data/results-2026-10-13/` and `results-2026-10-14/`; the run files under
`runs/shift_explore/{near80,stage2,detect}` in the WSL clone (the near-80 ones also on branch `claude/near80-run-files`);
the toy notebooks `bca_channel_toy.ipynb` and `bca_discrete_toy.ipynb` (branch `main`).
