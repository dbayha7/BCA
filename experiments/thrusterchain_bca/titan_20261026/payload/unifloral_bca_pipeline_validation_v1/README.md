# UniFloral + actual BCA: bounded pipeline validation

The previous UniFloral stage passed all 19 checks on Titan: six native outer-step boundaries matched numerically with maximum state/metric error zero, midstep resume matched exactly, and the synthetic width gradient tests passed. That run used 31 critic and 16 actor updates. It did not validate actual BCA calibration.

This new stage connects the real BCA calibrator, residual scale fitter, held-out calibration refresh and frozen width to the verified UniFloral actor-loss hook. It uses fixed synthetic banks only, at most four auxiliary optimizer attempts and three refresh calls. **It performs zero critic updates, zero actor updates, no rollouts and no full training.** Originals, stage-one results and paused pilot checkpoints are untouched.

## Run once from PowerShell

Run only this bounded diagnostic with:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_bca_pipeline_validation_v1/run_checks.py" --allow-outside-upstream-constraints
```

The runner sets and verifies `JAX_DEFAULT_MATMUL_PRECISION=highest`, `JAX_PLATFORMS=cuda`, native `XLA_FLAGS=--xla_gpu_triton_gemm_any=True`, unbuffered output and disabled preallocation before scientific imports. Conflicting inherited XLA flags cause a recorded stop. The compatibility flag acknowledges the existing environment's JAX/NumPy constraint mismatch; it does not install anything or claim upstream-supported runtime reproducibility.

Each invocation creates a unique `results_<UTC>/` directory. The supervisor writes `report.json`, with fresh runtime `probe.json` / `probe.log` and bounded `pipeline.json` / `pipeline.log`. Auxiliary checkpoint data and per-check evidence are written only there. Existing reports are never overwritten. Probe timeout is 180 seconds; pipeline timeout is 600 seconds. A timeout or failed check stops with preserved logs, no automatic rerun or fallback.

## Interpret the output

- `status: pass` means the bounded implementation checks passed on this runtime. Read the individual checks and measured gradients, not only the aggregate status.
- The `kappa_penalty` grid reports magnitude and gradient ratios; it chooses no coefficient and uses no final evaluation labels.
- Unready references and ready/infinite thresholds are separate cases. Legacy infinite-threshold zero penalty is exposed as fail-open behavior. The proposed rejection contract is diagnostic until the production policy is agreed.
- The measured width remains a twin-min TD-residual proxy, not a calibrated Q1-error guarantee.
- `scientific_validation_status: incomplete_design_review_required` remains even on success. Full training stays paused.

See `DESIGN.md`, `protocol.json`, `expected_results.json`, `output_schema.json`, `uncertainty_provenance.json`, `prepared_files.json` and `PREPARATION_RECEIPT.json` for declared choices, source identity, expected checks and static preparation evidence. No scientific tests were run during preparation.
