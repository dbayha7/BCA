# Approved bounded matched policy pilot

Prepared for one manual Zeus launch. This changes copied learning states only; the exact transferred UniFloral checkpoint, source and original BCA remain preserved. The failed one-sided variant and head-count study are excluded. No real training, calibration, model restoration, reset or simulation was executed during preparation.

The question is whether original BCA improves the learned policy, rather than just selecting a better action from an existing candidate bank. Three arms use native TD3+BC, original direct-Q BCA at kappa=3, and the fixed actor-distance penalty. Each has two paired replay/noise continuation streams from the same complete 20k checkpoint. These are not independent initialization replications.

Each run performs exactly 2,000 additional critic updates, 1,000 actor updates, 2,000 original auxiliary updates and one refresh after update 1,000. Shadow auxiliary/calibration work matches across controls. Native gradients, BC reduction, normalization, target smoothing and twin-critic/target cadence stay unchanged. The distance control anchors to the frozen 20k actor, with factor0.1 ×19.29619572569095 and live action gradients. The original uncertainty network and Bayesian/conformal mechanism are retained.

After all six final actor exports are sealed, evaluate each learned actor plus the unchanged actor on the same32 fresh nominal reset seeds. Each actor supplies every action throughout its trajectory; no candidate reranking. Primary return is sum(t=0..99)0.98^t ×reward_t. Maximum224 episodes /22,400 simulator transitions. The original confirmation outcomes are not accessed or reused.

Total scientific ceiling: 12,000 joint twin-critic updates, 6,000 actor updates, 12,000 auxiliary updates, six calibration calls with6,000 posterior draws, and22,400 simulator transitions. One local GPU, sequential jobs; no dependency installation or paid compute. Each worker has600seconds including restoration/compilation; the full sequence has4,200seconds including evaluation. This is a70-minute ceiling, not a predicted duration. The six final full states occupy roughly62MB before manifests/logs/trajectories; allow about150MB for ordinary results. GPU memory and runtime are not newly benchmarked during preparation.

**Launch once from PowerShell:**

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/matched_policy_pilot_v1/launch.py --execute
```

The command checks prepared/source identities, checks for scoped overlapping BCA processes, reserves the entire attempt atomically, and supervises each child. GPU/runtime/model identity checks precede each worker's first optimizer update. It refuses a second launch. A failure, timeout, invalid calibration or interrupt stops the sequence; evaluation cannot start unless all six training jobs completed. No automatic restart or resume is supported.

For a read-only preparation check, replace `--execute` with `--preflight`. To monitor from another PowerShell terminal, use:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/matched_policy_pilot_v1/launch.py --status
```

The supervisor prints current jobs and progress. Files `PROGRESS.json` and `run/<job>/progress.json` preserve updates. Training commits are journaled every20updates; after a forced kill the last record is a lower bound, with up to20 further dispatched critic/auxiliary steps potentially unrecorded. Evaluation charges each32-row step before dispatch. Partial reports distinguish attempted, committed and uncertain work. `RUN_RESULT.json` is terminal; if absent after a disconnect, inspect evidence rather than relaunching. Partial states are evidence, not permission to resume.

All complete outputs stay under `run/`: per-run state/actor exports, manifest hashes, refresh receipts, counts and reports; final evaluation trajectories, paired returns, harm counts, descriptive paired-episode intervals and separate action/state displacement. Exports are linked through supervisor report SHA → report manifest SHA → parameter hashes before evaluation. Full source/checkpoint identity is rechecked, and original files are never overwritten.

The screen is positive only if BCA beats both continued controls beyond the fixed numerical band in each stream. It is an exploratory direction screen, not statistical superiority. Calibration episodes are reused; no selected-action, true-return or OOD coverage guarantee is claimed. A failure closes this pilot without strength tuning; a pass supports proposing a larger independent study, not automatic full training.

Preparation evidence is in `IMPLEMENTATION_REVIEW.json`, `LAUNCH_TESTS.json`, `EVALUATION_REVIEW.json` and `PREPARED.json`. Synthetic fixtures cover native continuation parity, gradient/reference handling, paired RNG and refresh ordering, exact budgets, immutable originals, manifest binding, final-policy evaluation arithmetic, reserved execution and interruption/timeout accounting. Real GPU integration remains the manual run; assistant-side WSL access was not retried.

Pinned provenance and existing Apache2.0/MIT notices remain intact. No CORL host/model substitution. The existing Python3.12.14/JAX0.6.2/NumPy2.5.4 CUDA runtime remains outside upstream JAX/NumPy constraints, as previously disclosed on Titan. No commit or push is part of this pilot.
