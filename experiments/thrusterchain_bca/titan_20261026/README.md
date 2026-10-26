# Titan to Zeus BCA validation handoff

This handoff preserves the original pinned UniFloral standalone TD3+BC and BCA
sources, fixed protocols, checkpoint/data identities and completed validation
receipts. Relocation is recorded in `HANDOFF_MANIFEST.json`; frozen source files
and historical receipts are copied byte for byte, including original machine
paths. Those old paths are provenance, not instructions to access Titan.

The new operator audit accepts `BCA_TRANSFER_BUNDLE_ROOT`, set by `launch.py` to
the included `payload/transfer_bundle` layout. Earlier frozen launchers retain
their original paths; use this handoff launcher for the new audit.

## Current scientific status

- UniFloral is the native host. Original BCA and negative findings are preserved.
- The saved checkpoint has 20,000 critic updates, 10,000 actor updates and 20,000
  auxiliary updates; its terminal selection was fixed before diagnostic returns.
- Completed development diagnostics found incremental predictive information in
  width, but no established reliable decision advantage over simple controls.
- The completed locked confirmation remains **inconclusive** and is retained on
  Titan. No confirmation directory or confirmation rerun is included here.
- Full policy training remains on hold. The next audit is a frozen operator
  alignment diagnostic, not a new training run or a selected BCA repair.

## Transfer and restore

The local Git branch and standalone code ZIP contain source, small receipts and the file
manifest. Runtime NPZ/msgpack assets and any file larger than 5 MiB are in the
separate ZIP named by `HANDOFF_MANIFEST.json`; Git does not contain those bytes.
Both the code ZIP (or checked-out branch) and runtime ZIP are needed on Zeus. No push or upload is
performed by this package.

An optional incremental Git bundle requires base commit
`8d228198a87d38f5bbc9714b281270517a76a227` to already exist in the Zeus repository;
it cannot be cloned as a standalone repository. The code ZIP needs no Git history.

After extracting the standalone code ZIP on Zeus and entering this directory:

```powershell
python -B restore_assets.py 'C:\path\to\bca_runtime_assets_20261026_v1.zip'
python -B restore_assets.py --verify-only
python -B launch.py --preflight
```

The restorer checks ZIP membership, sizes and SHA-256 hashes before writing. It
rejects unsafe paths, duplicate/symlink entries and unequal existing files.
Matching already-restored files are accepted. It never replaces differing data.

## Execution requires a verified Zeus runtime

**Zeus has not been tested for this audit.** Earlier Zeus evidence showed a
CPU-only JAX runtime; Titan's successful GPU validation does not establish Zeus
compatibility or speed. This package installs nothing and changes no drivers,
security settings, credentials, or global configuration.

Use an existing compatible GPU Python interpreter only after the audit's
preflight/runtime gate succeeds. From that environment, the command is:

```text
python -B -u launch.py --execute --allow-outside-upstream-constraints
```

Do not substitute Titan's `/home/dbayha/bca-gpu/bin/python` path unless that exact
environment actually exists and is authorized on Zeus. The runtime gate checks
the declared JAX/CUDA package versions, GPU backend and highest matmul precision;
failure must be reported, not bypassed with changed tolerances or silent installs.

The operator audit permits at most 1,152 one-step evaluator transitions and four
fixed native smoothing draws per candidate, plus the separate saved fitting
transition bank. It performs no fit, calibration update, policy update, new full
rollout, coefficient sweep or confirmation access. The GPU audit remains pending
until a passing execution receipt exists. See the frozen audit protocol for all
limits and abort conditions.

## Packaging verification

`test_handoff.py` exercises only synthetic ZIP, overwrite, hash and launcher
boundaries. Packaging checks do not establish GPU or scientific-runtime success.
`HANDOFF_TESTS.json` records the actual packaging test and byte-check results.
`HANDOFF_MANIFEST.json` lists payload/support bytes and their original roles;
the manifest and test receipt are metadata exclusions to avoid self-hash cycles.

Keep new result directories and external assets out of unrelated commits. The
subtree's `.gitattributes` disables text rewriting to preserve frozen SHA-256
identities on Windows and Linux.
