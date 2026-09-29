# Advisor slides, September 29, 2026

The editable 11-slide deck and slide images are in `outputs/advisor_answers/2026-09-29/`. Speaker notes explain the scientific qualifications and cite the saved evidence used for each slide.

Inputs are the frozen report at `outputs/current_results/2026-09-29/plot_data.json` (published in commit `55a741720c3a847b05196484fd79a5e55fe50358`), the labelled original pilot in `outputs/advisor_progress/2026-09-29/ADVISOR_BRIEF.md`, the revised OOD protocol, and the weighted-conformal design document. The old brief is used only for pilot results, not its superseded progress counts.

`deck.mjs` is the exact builder used on David's Windows workspace with the bundled `@oai/artifact-tool` runtime. It declares the local workspace, repository and presentation-skill paths at the top. To rebuild elsewhere, update those paths, install/link the same runtime dependencies, set `RUNTIME_NODE_MODULES`, and choose unused candidate/final/validation filenames. The finalizer intentionally refuses to overwrite its validation receipt. The builder imports and renders the finalized PPTX, and its charts have editable embedded workbook data.

Charts round numeric literals to eight decimals where needed for Excel compatibility; source data remain unchanged. The deck displays normalized-score differences to two decimals and pilot AUROC to three decimals. There is no smoothing, new simulator execution, training, checkpoint alteration or scientific-result replacement.

The local build archive preserves the initial missing-runtime-variable attempt, chart precision rejection, first rendering, and validation-receipt filename collision. The reviewed deck corrects a chart category-label overlap and adds explicit score units. These are presentation-only corrections.
