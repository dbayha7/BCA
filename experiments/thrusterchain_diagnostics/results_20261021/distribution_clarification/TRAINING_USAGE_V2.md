# Distribution clarification: results guide v2

`ThrusterChain_diagnostics_results_v2.html` adds a source-to-target distribution panel and the advisor's controlled action-only follow-up plan. The original delivered `ThrusterChain_diagnostics_results.html` and its generator/template are preserved unchanged. All training measurements remain unchanged.

The new panel separates:

- Behavior trajectory states and the 625-command categorical logger.
- Joint state/action row retention and the actual equal-episode calibration law.
- Reset-state/time-grid queries with six equally represented candidate commands.
- The additional 95k frozen-reference versus 100k residual-model drift.

It labels the current coverage result a designed state/time-and-action mismatch, not an IID, state-only, or action-only experiment. The separate nine-stage plan is explicitly **post-hoc, proposed, and not run**. It freezes one available full-retention BCA model and a finite state law, proposes fresh independent calibration/evaluation draws, and changes only the current-action law on the supported categorical grid. No new training, evaluation, calibration, or source modification was performed to produce this revision.

The reproducible builder reads the original diagnostics HTML, verified result records, the v2 template, and two additional source documents:

- `../analysis/DISTRIBUTION_SHIFT_AUDIT.md`
- `../module_fixtures/ACTION_ONLY_POSTHOC_PLAN.md`

From this directory:

```powershell
python -B update_training_guide_v2.py --output NEW_RESULTS_V2.html
```

Use `--base`, `--previous-results`, `--distribution-audit`, and `--action-plan` to supply portable paths. The remaining input flags match the original builder; `--template` defaults to `training_results_section_v2.html`. Existing output paths are refused. The build manifest hashes every input and verifies preservation against both the original diagnostics guide and the previously delivered results guide.

Static verification covers internal links, unique IDs, preserved script and SVG blocks, JavaScript syntax, and all 24 combinations of the results filter. Browser rendering, navigation, and runtime-console behavior remain unverified because no browser surface was available.
