# Advisor metrics beyond return

The 12-slide editable deck, slide images, speaker notes, concise metric overview and exact plotted arrays are in `outputs/advisor_metrics/2026-09-29/`. The earlier advisor deck is preserved.

`extract.py` reads closed exported JSON files and the frozen September 29 cluster snapshot. It never opens a live simulator database, model checkpoint or training process. Source hashes accompany `METRIC_EVIDENCE.json`. It checks the pilot harm/regret arithmetic and the saved radius maximum, and reuses independent training checkpoint-counter receipts. This is a readout check, not new scientific acceptance.

`deck.mjs` is the exact Windows builder using the bundled `@oai/artifact-tool` runtime. The local root, repository and skill paths are declared at the top. Reproduction requires those paths (or editing them for another machine), the same runtime libraries, `RUNTIME_NODE_MODULES`, and an unused candidate, final and validation-receipt filename. Link the runtime `node_modules` inside the private build directory before running. Extracted `metrics.json` belongs one directory above that build directory.

Charts use complete saved block/refresh/snapshot series without smoothing. Numeric literals are rounded to eight decimals for embedded Excel compatibility; unrounded evidence is retained. The deck has 16 native editable charts. All 12 final slide renders were inspected. Training diagnostics from reviewed TD3/ReBRAC Walker runs are distinct from the completed TD3 Hopper pilot and from preliminary CQL Walker snapshots.

No new OOD outcomes, fresh coverage verification, true-Q target or full-grid diagnostic completion is claimed. IQL non-return traces and the other unshown combinations are not silently treated as complete. Original experiment failures, outputs, checkpoint weights and the main research PDF are unchanged.
