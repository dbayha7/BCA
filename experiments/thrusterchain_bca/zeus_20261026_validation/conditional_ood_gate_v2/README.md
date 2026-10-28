# Run the corrected design gate (v2) on Zeus

Status: **prepared for your manual launch; not run.** This is a corrected copy of `../conditional_ood_gate_v1/`.

Gate v1's attempt 1 stopped after 37 seconds in its source-identity phase. It decoded no data, made no simulator step
and reached no decision. Two code defects caused it, and v2 fixes only those two:

1. **Core check.** `reference.py` read the file attribute of a canonical module that the pinned loader builds in
   memory, where no such attribute exists. v2 hashes the source file the executed dynamics function was compiled from.
2. **Crash handler.** `worker.py` had a typo, `__name()` for `__name__`, which hid the first error.

The six-command design and every threshold, seed and cap are unchanged. `PROTOCOL.md` and `contract.json` record the
lineage. v1 and its attempt record are untouched.

This is a development design screen of the original band, not permission to train a policy on it. A failed result stays
failed. There is no automatic retry, region change or radius search.

**How the fixes were checked.**
- **Reproduced on the real core.** v1's code raises the same AttributeError on the real pinned core. v2's check passes
  and hashes the right canonical source (`f22f2e21…`).
- **All tests pass.** The 32 original tests pass on v2, and so do the 2 new regression tests in `test_v2_fixes.py`.
- **What stays unverified.** These checks ran on Windows CPU. CUDA, model inference and the real-data phases are checked
  only inside the gate's capped worker at launch.

## Launch

From PowerShell on Zeus, run this once:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_gate_v2/launch.py --execute
```

**What the launcher does.**
1. Checks the prepared hashes and package versions before reserving `run/`.
2. Points the worker at the existing portable bundle and verifies the pinned Titan Unifloral source, checkpoint and
   export.
3. Checks CUDA before decoding any data.

The environment is the previously installed isolated BCA runtime, so nothing is installed. For a metadata check that
reserves nothing, replace `--execute` with `--preflight`. That check does not test CUDA or model execution. The
assistant's WSL route stays denied and was not retried.

**Limits.**
- **Fixed ceiling:** 900 seconds and 28,104 real simulator transitions.
- **Training:** zero critic, actor, auxiliary or calibration updates.

**Where results go.**
- **Progress:** printed while running and saved to `run/progress.json`.
- **Accounting:** `run/RESULT.json`.
- **Decisions and measurements:** `run/report.json`. If the worker fails, `report.json` now records the real error type,
  message and traceback.

An interruption keeps the reservation and all saved evidence. Do not delete `run/` to repeat a failed or interrupted
attempt.

## Criteria (unchanged from v1)

- **Support:** RMS radius 0.5 and 64 original neighbors. A neighborhood needs 32 rows from eight episodes, plus evidence
  in both adjacent action bands.
- **Value:** a center-to-flank gap of at least one return unit at 25% of high-load anchors.
- **Free policy:** the actor must enter the hole on at least 1% of high-load steps, in at least two episodes.

See `contract.json` for all fixed coverage, balance and eligibility criteria. Within-band return spread is also
measured, with no pass mark.

Later training, if separately approved, would extend the existing 10k support-pilot fork. This gate does not create that
training matrix. It does not establish a missing-label causal effect, a BCA benefit, joint four-action support or
held-out confirmation.
