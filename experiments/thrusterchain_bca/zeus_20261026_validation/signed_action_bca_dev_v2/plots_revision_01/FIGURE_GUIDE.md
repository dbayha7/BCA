# Frozen-checkpoint diagnostic: figure guide

Source: `diagnostic_run_01/RESULT.json`, SHA-256 `db19fb35637df3417c8c9fd9a8abf0001c3965499be0433e5dd7f35744109200`.

The actor, critic, and target models were frozen. Each plotted marker is one of three checkpoint seeds; the contexts and candidate scores use paired anchors. These results do not measure policy improvement.

1. **Accuracy and regret:** high-load fixed-band Q MAE is compared to same-policy simulator return; high-load regret compares the selected action with the best of only six candidates.
2. **Choice changes:** improvements and harms compare the combined score with raw Q1 on the same state, using a 0.1-unit tolerance. Unchanged choices are shown separately.
3. **Calibration and distance:** one-step TD checks use 64 new behavior episodes; the state-distance gate is a heuristic, not a support certificate or shifted-policy coverage guarantee.

The full ten-method per-cell numbers are in `paired_metrics.csv`. The prespecified exploratory screen:
- Ordinary: did not pass; qualifying native seeds 0/3 and BCA seeds 0/3.
- Strong Current: did not pass; qualifying native seeds 0/3 and BCA seeds 0/3.

No actor, critic, actor-target, or critic-target update was performed. This is a development screen after prior exploratory analysis.
