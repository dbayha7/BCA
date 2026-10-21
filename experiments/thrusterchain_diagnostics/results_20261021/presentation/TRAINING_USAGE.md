# BCA stage verification and completed ThrusterChain results

`ThrusterChain_diagnostics_results.html` is a new, standalone results guide. It starts with the advisor's stage-by-stage verification map and code-derived pseudocode, distinguishes native and importance-weighted BCA, includes measured isolated fixtures and reproducible multiplier-ordering exceptions, then reports all 30 completed continuous training jobs. The original diagnostics HTML remains an implementation/smoke release; its no-trained-results statements apply to that release. All historical sections, existing SVGs and scripts remain in the new guide.

The presentation generator does not import a learner, rerun an evaluation, change frozen files, or upload anything. It reads these saved records:

- `../analysis/aggregate.json`: verified 30-job/60-evaluation aggregate.
- `../method_audit/evidence.json`: native/IW source-stage traces and line references.
- `../module_fixtures/verification_v2/stage_map.json` and `fixture_results.json`: isolated tests, expected and observed results, including two reproduced scientific-contract failures in the historical command-weighted branch. The original 18-case run remains preserved separately.
- `../penalty_alignment/penalty_alignment.json`: descriptive multiplier diagnostics on 11 available Zeus BCA raw-query records.
- `execution_plan.json`: byte-preserved initial Zeus launch plan; `../inputs/titan_results.json`: imported execution/report evidence.
- The preserved canonical `ThrusterChain_diagnostics.html` and sibling `training_results_section.html` template.

From this presentation directory:

```powershell
python -B update_training_guide.py --base C:\Users\David\dev\BCA\experiments\thrusterchain_diagnostics\ThrusterChain_diagnostics.html --output NEW_RESULTS.html
```

All inputs can be supplied with `--aggregate`, `--method-audit`, `--stage-map`, `--fixtures`, `--penalty-alignment`, `--execution-plan`, `--titan-report`, `--base`, and `--template`. Supply `--node PATH_TO_NODE` to run JavaScript syntax and mock-DOM checks on another machine. The default points to the known local Node runtime; if absent, the build records zero JavaScript checks. The generator uses Python's standard library only.

Final builds require `aggregate.complete == true`, all 30 distinct planned jobs, consistent paired arithmetic, the required stage records, and a new output path. Existing outputs and the base are refused. `--allow-partial` permits an explicitly marked **INTERNAL PARTIAL PREVIEW** at a separate filename; do not deliver such previews as a completed report. The generator writes a matching `.build.json` recording every input hash, output hash, structural checks, and verification limitations. Rebuild at a new filename rather than overwriting a prior release.

Verification checks all internal anchors, duplicate IDs, preservation of every existing script/SVG and ID, absence of external asset URLs, syntax of all executable JavaScript when Node is available, and all 24 combinations of the added policy-table filters using a mock DOM. Browser rendering, navigation behavior and runtime-console behavior were not checked: no browser surface was available. Pseudocode panels are labeled and suitable for later screenshots; no screenshots are claimed as captured.

## Scientific workflow

The frozen experiment uses native TD3+BC and native TD3+BC+BCA, H=100, gamma=1, 100,000 critic updates, 50,000 actor updates, three retention conditions and five matched data/optimization replicates. The 100k terminal checkpoint is primary. The 50k checkpoint is descriptive; neither checkpoint selection nor online data collection is part of the protocol. Stage fixtures and multiplier diagnostics are additional post-training analysis, not retraining or a change to the frozen study.

From the BCA repository root, the following commands document how a **new** run is prepared and executed. They are not instructions to alter the completed run:

```powershell
python -B -m experiments.thrusterchain_diagnostics.full_run config --output NEW_CONFIG.json
python -B -m experiments.thrusterchain_diagnostics.full_run prepare --output NEW_RUN --config NEW_CONFIG.json
python -B -m experiments.thrusterchain_diagnostics.full_run run --output NEW_RUN --worker-index 0 --workers 3
python -B -m experiments.thrusterchain_diagnostics.full_run run --output NEW_RUN --worker-index 1 --workers 3
python -B -m experiments.thrusterchain_diagnostics.full_run run --output NEW_RUN --worker-index 2 --workers 3
python -B -m experiments.thrusterchain_diagnostics.full_run oracle --output NEW_RUN
python -B -m experiments.thrusterchain_diagnostics.full_run evaluate --output NEW_RUN
python -B -m experiments.thrusterchain_diagnostics.full_run status --output NEW_RUN
```

Workers should be separately launched under the documented CPU resource limits. The configuration, `TRAINING_PROTOCOL.md`, run provenance and native source hashes determine the experiment. The guide is not a substitute for those records. Use the analysis scripts and commands packaged in `analysis`, `module_fixtures`, and `penalty_alignment` to reproduce the saved report.

The completed evidence combines 22 locally checked Zeus CPU jobs with eight imported Titan CPU report entries. Titan raw checkpoints/rollouts were not independently accessed or rerun in the Zeus-only analysis session. Full and zero retention confound host and arm in four of five pairs. The replication unit is the trained replicate, not the 400 shared evaluation starts. Numerical QP bounds are conditional on fixed starts, not exact continuous-population expectations.

Native residual coverage, actual policy-Q error, dose usability, BC loss, and return benefit are distinct measurements. The pooled 128-row calibration bank is adaptively reused and has rare critical counts 1, 3, 0, 0, 1; it supports no critical-group certificate. A mathematically correct stage does not establish the utility of its intervention. The new guide makes tested and untested steps explicit.
