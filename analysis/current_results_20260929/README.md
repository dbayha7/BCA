# September 29 current standard-BCA readout

Read [the report](../../outputs/current_results/2026-09-29/index.html) or [analysis](../../outputs/current_results/2026-09-29/ANALYSIS.md).

## Rebuild from the frozen snapshot

From the repository root, with Python, NumPy and Matplotlib installed:

```sh
python analysis/current_results_20260929/build_report.py --repo . --inputs analysis/current_results_20260929/inputs --output outputs/current_results/2026-09-29
python analysis/current_results_20260929/write_page.py --repo . --output outputs/current_results/2026-09-29
python analysis/current_results_20260929/verify_report.py --repo . --output outputs/current_results/2026-09-29
```

The frozen extraction contains completed saved evaluations and selected closed CQL scalar journals. Existing independently accepted exports are reused by their recorded hashes. `collect_cluster.py` is the read-only extractor retained for provenance, not part of reproduction; rebuilding does not contact the cluster, open any live database, query a model, read checkpoint weights or run a simulator.

All 115 available actor trajectories from 94 physical results are represented. Forty-six within-seed comparisons are available; only eight training pairs have independent acceptance. The one saved CQL recovery keeps original exit 1. IQL compares its two shared-Q/V actors and separates the standalone host context. Every declared cell remains visible; missing pairs have no invented score. OOD progress is a separate timestamped status snapshot with zero revised pairs complete.

`inputs/readout_attempts.json` preserves the initial read-only schema errors and corrections; science was not rerun. Static figures were visually reviewed, arithmetic recomputed independently, and HTML links checked. Local HTML browser interaction could not be verified because the browser blocks file URLs. No main PDF or scientific source was modified.
