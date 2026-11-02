# UniFloral checkpoint correctness and bounded checkpoint generation

This separate runner preserves the pinned UniFloral baseline, earlier CORL artifacts, validation reports and paused pilot. It implements the newly requested checkpoint work. Static preparation is not a GPU test result: run the correctness gate first. No training or scientific code has been executed by the preparing agent.

## Declared first checkpoint stage

The first real stage is one seed, **20,000 actual primary critic optimizer updates / 10,000 native actor updates**, batch 256, with kappa_penalty=0. It trains the native host while separately fitting the actual BCA TD-residual proxy. It does not restart the four-arm million-update pilot and does not choose a penalty coefficient.

Native initialization, Adam epsilon 1e-5, population observation normalization with std+0.001, mean reductions, Q scaling denominator epsilon 1e-7, RNG and minibatch reuse are retained. One native outer step is two critic updates on the same sampled batch followed by one actor/target update. Each runner call advances one critic update and saves whether the second call/actor step remains pending. The task-specific overrides are gamma 0.98, 100-step finite horizon, raw rewards, time-aware 14D observations and no terminal bootstrap. This is not published-benchmark reproduction.

The declared auxiliary schedule is one width fit after each native critic call using that call's held native batch, with an independent auxiliary RNG. On even calls, native critic 2, actor and targets finish before auxiliary fitting. The auxiliary therefore sees the current native targets. Eager references refresh after actual critic counts 10,000, 15,000 and 20,000. This composition schedule is explicitly new; it is not claimed to be an unchanged historical CORL scheduler. At kappa=0 no uncertainty enters native learning, and the gate checks that independence.

Required milestones are accepted real width fitting, finite-ready calibration at the scheduled refreshes, and a terminal complete outer-step checkpoint. The terminal checkpoint is fixed at 20,000; no diagnostic returns rank checkpoints. Unready warmup is reported intentionally. An invalid fit or nonfinite/invalid ready reference stops the stage and preserves the last committed state. Such a stop means insufficient diagnostic readiness, not successful zero-penalty BCA or an efficacy finding. No automatic extension is implemented, and 20,000 is not a guarantee of policy quality.

## Data separation

The runner reads only the exact hash-pinned existing 1,280-episode offline NPZ; it generates no trajectories and imports no simulator in data ingestion. Byte identity ties it to the previously validated dataset. All 768 training episodes train the native host and auxiliary. Calibration uses one seeded time row from each of the first 128 tuning-validation groups sorted by group ID. Row-selection seed is 2026102362. The 256-row fitting-only penalty scale bank uses seed 2026102363. Selection and final-calibration groups are never used. Group membership, row IDs and complete used episode seeds are saved, and reserved usefulness-development/confirmation seeds are checked for overlap.

The widths remain a twin-min one-step TD-residual proxy, not a Q1-error coverage guarantee. No simulator value or final-evaluation label reaches the learner/calibrator. The frozen-usefulness evaluator remains a separate later stage.

## First run: tiny checkpoint correctness gate

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_checkpoint_v1/run.py" --checkpoints-test --allow-outside-upstream-constraints
```

The fixed synthetic gate uses batch 8 and six critic updates in the uninterrupted combined trajectory. It saves after critic 1, while the native batch is still held; restores and replays the remaining five calls; and compares against six critic updates in the native baseline. Maximum successful work is **17 critic updates, 9 actor updates, 11 auxiliary fits and 6 tiny-fixture refresh calls**, with zero simulator transitions. Early refresh counts 2, 4 and 6 are test-only fixtures, not the production schedule.

Resumed-versus-uninterrupted state and metrics must match exactly, including actor/critic/target parameters, optimizer moments/steps, phase, minibatch, both RNG streams, counters and auxiliary/reference state. Native-baseline floating comparisons retain the previous rtol=2e-5/atol=2e-6; RNG, counters and structure are exact. Corrupt and mismatched checkpoints must be rejected without changing the good checkpoint. Tiny fixtures are ineligible for a trained-policy export.

A second zero-update roundtrip saves critic 3 with the finite reference frozen at critic 2 and the live calibrator already advanced to critic 3. This checks that those distinct states survive serialization. The gate models interruption between committed calls; it does not claim to have injected or validated OS Ctrl+C delivery.

The command uses a fresh GPU child with highest matmul precision and the already tested provisional package versions. Upstream JAX/NumPy bounds remain exceeded; the explicit flag acknowledges that limitation, not upstream reproducibility. No installations occur. The gate has a 600-second timeout. Its report must pass and match the exact current source hashes before the runner will train.

## Separate bounded training command

After reviewing the gate report, replace its directory below with the actual `checks_...` directory:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_checkpoint_v1/run.py" --train --gate-report "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_checkpoint_v1/checks_TIMESTAMP/report.json" --allow-outside-upstream-constraints
```

This explicit command launches only the fixed bounded stage above. It saves the initial state, every 1,000 committed critic updates, and the terminal state. Progress appears every 100 updates and at refresh milestones. Runtime and source/config/data/normalization/gate identities are bound into every checkpoint. No throughput estimate is available for this combined runner.

Checkpoints are new directories containing `state.msgpack`, `metadata.json` and a final `COMMITTED.json` marker. Partial or corrupted directories are rejected; previous checkpoints are never overwritten. Serialized state includes all native and auxiliary continuation state, not just model weights. Loading reconstructs the same static native modules/optimizers from the bound configuration/data, then restores the full numeric state. Model initialization in that template step is not training and is never used as a substitute for missing saved weights.

The fixed training and calibration pools remain in the original hash-bound offline dataset. Their exact row/group selections and selection seeds are bound in checkpoint metadata; resume reconstructs them from those same bytes. They are not silently replaced by new samples, and the original dataset is required for resume. The minibatch held between the two critic calls is serialized directly.

Press Ctrl+C once in the running terminal to request a pause. The training child attempts to save its last committed state, while the supervisor allows up to 60 seconds for confirmation. Do not assume a safe pause until `worker.json` reports `paused_checkpoint_saved` and the named checkpoint contains a valid `COMMITTED.json`. If interrupted during checkpoint I/O or before the first checkpoint commits, use the latest previously committed directory; an unsaved in-flight step may be lost. No unrelated process is touched.

Resume into a new run directory with the **same gate report** and committed checkpoint:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_checkpoint_v1/run.py" --train --gate-report "/absolute/path/to/the/original/checks_TIMESTAMP/report.json" --resume "/absolute/path/to/run/checkpoints/critic_NNNNNN_interrupt" --allow-outside-upstream-constraints
```

Resume preserves the original 20,000-update endpoint; it does not add another 20,000 updates. Source, dataset, normalization, config or gate mismatch is rejected. Changing scientific settings requires a separately reviewed stage, not a resume override.

At a finite-ready terminal checkpoint the runner also writes a separate inference export for `unifloral_frozen_usefulness_v1`. Export failure cannot remove the resumable terminal checkpoint. Hashes and self-consistent receipts establish identity, not independent proof of training history; inspect counters and progress before interpreting the export. Neither checkpoint correctness nor frozen first-action reranking proves penalty-training efficacy.

The agent cannot run WSL because the earlier access denial remains binding. The provided commands are for the user; no GPU result is claimed until their saved output is inspected.
