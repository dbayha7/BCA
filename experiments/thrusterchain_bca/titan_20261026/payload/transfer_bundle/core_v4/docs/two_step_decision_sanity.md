# Two-step decision sanity appendix

This separately evaluated, evaluator-only appendix shows that the prescribed nine-action bank has a nonzero best action at decision time 98 on two synthetic physical states. It uses privileged true-model labels with a fixed zero command at time 99. It evaluates **zero learned methods** and makes no policy-benefit claim.

Fixture ID: `symmetric_v1_t98_nine_action_bank_v1`. Its geometry was selected after the terminal audit; it was not preregistered or selected to favor a learned method. The implementation follows the supplied specification and records its own SHA256.

The frozen repaired core source hash remains `49330e5314445ccf551816a09822f2e0d90ea40475facbc8363a45bc6c62ff6b`. Appendix code and its nine unit checks live in `audits/two_step_decision_sanity_v1/`; no core source or normal test file is changed. It ships in the source/review archive as an appendix and is separate from the core wheel and the normal 151-test suite.

## Fixture and target

The states are exact float64 zero vectors of length 13 except coordinate 1 (`v1`), which is respectively -1 and +1. Both pass `environment.in_start_set`. Both have current zero and wave amplitude zero, below the recorded logger start-law minima of 0.1. They are synthetic fixed-state queries, with no reachability or closed-loop occupancy claim and no recorded-law sampling claim.

Default stalling physics has slope 1.5, throttle cost 0.3, horizon 100 and discount 0.98. The public observations are float32 `[x, 0.02]`. Let `k = float32(0.8) = 0.800000011920929`. The ordered action bank is `[0, +k e1, +k e2, +k e3, +k e4, -k e1, -k e2, -k e3, -k e4]`. Actions are quantized to float32, promoted to float64, and zero signs canonicalized. All commands lie at or inside the knee, where thrust equals executed command.

The target is

`Q98_fixed_zero(x,a) = r(x,a) + 0.98 * r(f(x,a),0)`.

Reward uses the pre-transition physical state. There is no reward at time 100 and no terminal bootstrap. The evaluator uses exactly two batched calls to `environment.dynamics` for the 18 state/action labels. It cross-checks an expanded quadratic using ordinary matrix arithmetic on the same A/B plant with a checked reference matrix hash, independently of the dynamics arithmetic; this is not an independent model.

The A/B digest is SHA256 of A followed by B as raw C-order little-endian float64 bytes, with **no shape metadata**: `289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1`. It does not use `planning.array_hash`.

| State v1 | Best bank index at t98 | Best command | Zero Q98 | Best bank Q98 | Zero regret to bank best |
|---|---:|---|---:|---:|---:|
| -1 | 1 | +k e1 | -1.9427729056507024 | -1.7867286894153778 | 0.1560442162353246 |
| +1 | 5 | -k e1 | -1.9427729056507024 | -1.7867286894153778 | 0.1560442162353246 |

All assertions use absolute tolerance `1e-12` and relative tolerance zero. Sign reversal of the state exchanges positive/negative bank commands. On these same physical states at time 99, reward alone uniquely prefers bank index 0 for both states. The t98 maximum is only over the specified nine-action bank and fixed continuation; it is neither learned Q nor global Q-star.

## Reproduction and evidence

Generate JSON without writing core or result files:

```powershell
$env:PYTHONPATH = 'C:\Users\David\.cache\bca-shift\py'
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B audits\two_step_decision_sanity_v1\two_step_decision_sanity.py --core-root .
```

Run only the appendix checks:

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B audits\two_step_decision_sanity_v1\test_two_step_decision_sanity.py --core-root .
```

The accompanying `run_appendix.py --core-root .` captures evaluator stdout without editing it, then runs the appendix checks and the **original unchanged** `tests/test_calibration.py` separately in newly created scratch working directories. Existing evidence uses exclusive file creation; the capture runner intentionally refuses to overwrite an existing receipt or output. Reproduction can use a fresh copy of the source appendix with capture artifacts omitted.

The nine focused checks cover exact fixtures, deployment quantization, signed zeros, the raw matrix digest, both numerical paths, symmetry, all 18 canonical finite-horizon API replays, fresh-process import isolation, and rejection of a preloaded calibration module. The separate original calibration run expects all 15 tests, including the rule that terminal targets reject time 98. It does not advance or modify the existing terminal calibrator.

Evidence lives beside the script:

- `evaluator.stdout.json`: all 18 labels, both states and public observations, the full bank, reward components, next states, numerical checks, import inventory and hashes.
- `appendix_tests.stderr.txt` and `original_calibration_tests.stderr.txt`: separate complete unit-test logs.
- `preservation_before.json` and `receipt.json`: before/after hashes for the frozen core and every existing result file, test counts and fresh scratch paths.
- `manifest.json`: original captured appendix file hashes; `documentation_addendum.json` records later prose-only edits, and `manifest_final.json` inventories the delivered appendix and its final source hash. Each manifest excludes itself.

The evaluator imports only the environment and reference-plant bridge from the core. Its fresh-process import check excludes calibration, regular learner, dataset and planning modules. No dataset, calibration artifact or planner fit is accessed by label generation. Preservation bookkeeping separately hashes existing files without importing their modules. The original calibration regression is a separate process. These post-audit labels support no posterior, conformal, adaptive-selection, learned-policy, or calibration-coverage claim.
