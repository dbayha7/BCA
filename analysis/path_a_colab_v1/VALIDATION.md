# Path A Colab notebook validation

Prepared November 2, 2026. **No scientific workload or Colab installation was executed.** This is an optional transport of the frozen Path A recovery, not a new experiment or an observed launch.

## Deliverable

- `Path_A_Colab.ipynb`: 5,062,744 bytes; SHA256 `48dc3e9404cbe004d59c9ef70023ac1e2ab8ce1b6f5cf3a0b76a7e490309e32b`.
- All 657 frozen source/config/evidence files are embedded as compressed exact bytes, with the additive Colab adapter, extraction code and full 58-package lock. There are nine code cells, all unexecuted, with no outputs.
- External binary input remains `path_a_titan_recovery.zip`: 112,349,548 bytes; SHA256 `c7f492c7e3d46a8508f8f7df30a975e216230e9e7e6289e2edb53f4793ef64fc`. It contains 671 payload files, including 14 required binary dataset/state files. The notebook is not falsely described as data-self-contained.
- Exact package lock SHA256 `017b189f4786193db03c32fd7b3d691ae320c1274eff11e1e0c3766764655d14`. Wheels require Linux x86-64 / Python 3.12 and are installed into a new isolated environment, from official PyPI, with exact hashes and no source builds or silent substitutions.
- Final additive profile SHA256 `b671b929390bd5d823c3fc8b611166bb7287f134c6cf172b977100e67f79845c`.

## Checks actually completed

Command: `.\.venv\Scripts\python.exe -B -m unittest discover -s analysis/path_a_colab_v1 -p test_*.py -v`.

**17 tests passed in 12.474 seconds on the existing Windows CPU environment.** Eleven adapter tests cover one-shot reservations, unknown/consumed Titan status, exact approval/evidence binding, cumulative setup/wait timing, boot changes, zero-update compatibility schema, unknown prior work, scoped orchestration restoration and cleanup of owned children. The real extracted-source fixture verifies all 671 archive files and actual prepare → fixture authorization → source check against inherited source/smoke/checkpoint pins and the exact remaining job list. Scientific imports, child execution and PID probes are forbidden in that fixture.

Six notebook tests check standard nbformat 4.5 structure and all Python-cell syntax, empty execution records, blocked defaults, all 657 exact embedded texts including the frozen CRLF approval, exact profile and 58-package lock, actual extraction plus idempotent verification/conflict rejection, and timeout/approval boundaries. The `nbformat` package is not installed locally; these are explicit structure checks and compilation, not a claim that its schema validator ran.

Independent reviewer inspected actual profile and notebook execution flow, compiled all nine cells, compared embedded files against the archive and rechecked the final corrected hashes. Two preparation defects were caught and fixed before delivery:

1. A transcribed bootstrap hash in the new adapter had one extra character. The real extracted-source startup fixture rejected it; the constant now matches the immutable archive. No original source or seal changed.
2. Notebook polling alone was insufficient as a strict outer cap. Commands now use independent GNU `timeout`, with validation 115 seconds plus five-second kill grace and science capped dynamically minus five seconds plus that grace. Owned-child cleanup remains.

An initial extraction fixture hit Windows MAX_PATH in the sandbox's long temporary directory. It was moved to a short, isolated workspace temporary directory; the payload and extraction rules were unchanged. No system setting was altered.

## Preserved experiment and accounting

Same two datasets, three paired seeds, native versus original actor score Q1−3U, unchanged BC, normalization, targets, fitting/calibration roles, probes, finite-horizon own-policy Qπ references and thresholds. The saved Honeypot seed-0 warmup is reused. Exactly 17 training jobs plus the existing evaluation remain; no fresh completed warmup or diagnostic rerun.

Remaining scientific work: 170,000 critic / 85,000 actor / 170,000 auxiliary updates, 17 calibrations, 1,536 return episodes and at most 237,600 evaluation transitions. Conservative cumulative caps remain 190,014 critic / 95,008 actor / 190,014 auxiliary updates, 22 calibrations and 430,575 simulator transitions. Unknown prior interruption tails remain unknown within their existing bounds.

The remaining 10,660 seconds is not reset: prior reserved preparation 60 seconds and compatibility 120 seconds, with all Colab setup/upload/install/extraction/waits also deducted. Science is at most 10,480 seconds and usually less. Runtime/kernel restart, unknown previous execution or an existing attempt blocks automatic continuation. Operator source-status evidence is not a distributed lock.

## Current status and actual blockers

Read-only Zeus snapshot at **12:42:11 p.m. EST**, November 2: `run_recovery_1` contains only `HANDOFF_HOLD.json`; no local `titan_validation_1` or `run_titan_recovery_1` was observed. Actual Titan state is unknown. No new Path A BCA policy or evaluation is established by this snapshot.

Before Colab execution the operator must establish that Titan validation and recovery never started, retain the Zeus hold, provide the exact ZIP, select Colab explicitly and pass its actual zero-update compatibility check. If Titan already consumed work, this notebook intentionally cannot migrate it without reconciliation. Actual Colab installation, GPU availability/compatibility, Drive persistence and complete study runtime remain untested. The successful real-update smoke is Zeus evidence only. A different GPU is not promised bitwise future-training equivalence.

No WSL, CUDA, simulator, training or calibration execution; no install, commit, push, frozen-file mutation, approval fabrication or experiment restart occurred during preparation.

## Documentation and skills

Unpinned readiness documents updated: `analysis/path_a_readiness_v1/README.md`, `RUNBOOK.md`, `AGENT_HANDOFF.md`, and `experiments/thrusterchain_bca/two_scenario_path_a/CURRENT_STATUS.md`. Their original verification record is preserved and explicitly predates Colab. Original Titan runbook/seals/ZIP remain intact. The three external Claude agents were not contacted or claimed aligned.

Repository skills read and applied: `.agents/skills/offline-rl-experiment-design/SKILL.md`, `.agents/skills/bca-code-and-calibration-audit/SKILL.md`, `.agents/skills/thrusterchain-benchmark-validation/SKILL.md`. Installed skills read: `C:/Users/David/.codex/skills/offline-rl-audit/SKILL.md`, `offline-rl-data-audit/SKILL.md`, `offline-rl-experiments/SKILL.md`. Applied checks concern matched objectives/budgets, original actor penalty and calibration semantics, unchanged transitions/roles and own-policy evaluation. No notebook-specific skill was available. Library skill is used only for delivery of the requested notebook.

Build/review sources: `build_notebook.py`, `colab_profile.py`, `test_colab_profile.py`, `test_notebook.py`. These are isolated from all frozen experiment files. The notebook contains the full code required to run; these local review files are not additional Colab inputs.

## Delivery

The requested notebook was successfully saved to the user's Library as `Path_A_Colab.ipynb`; exact returned identity and version are preserved in `LIBRARY_DELIVERY.json`. The Library metadata helper could not apply local extended attributes because Windows Python has no `os.setxattr`. This does not undo the confirmed Library creation. No second upload or workaround was attempted. The local notebook remains available with the verified content hash above.
