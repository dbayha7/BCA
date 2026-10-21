# ThrusterChain: completed training and sequential algorithm verification

Open `presentation/ThrusterChain_diagnostics_results.html` in a browser. It is self-contained and preserves the previous illustrated learning guide and historical results. The new section introduces the unseen-action problem, defines the experiment, presents all five paired seed results, and walks through code-derived stages with labeled pseudocode and evidence. The separate HTML Library attachment is the same file.

The frozen core study is complete: 30 learned jobs, two methods, three coverage conditions, five paired replicates, 100,000 critic updates and 50,000 actor updates per job. All terminal and descriptive midpoint evaluations are included. The tested native BCA configuration has no pooled return benefit. This is not an importance-weighted BCA training study. The separate historical audit finds mixed intervention evidence and a confirmed clipping probability-law defect that must be repaired before that particular branch can support exact-ratio calibration claims.

## Start with these files

- `analysis/RESULTS.md`: complete numerical interpretation and every paired terminal seed.
- `analysis/aggregate.json` and CSV files: machine-readable seed, context, calibration and control results.
- `method_audit/REPORT.md` and `evidence.json`: sequential native/weighted stages, exact source hashes and lines, evidence matrix, limitations and gated next steps.
- `module_fixtures/REPORT.md` and `verification_v2/fixture_results.json`: deterministic component checks and reproducible negative cases. Twenty reproduction checks pass; eighteen scientific contracts pass and two historical command-law contracts fail. The intentional exit code 1 preserves those failures.
- `penalty_alignment/`: frozen-query multiplier/error diagnostics, including all 154 qualifying ordering counterexamples. These are not reconstructed logged-row training weights or complete loss/gradient comparisons.
- `review.md`: independent scientific/code review and final presentation review.
- `audit/`: original local checkpoint/source/evaluation verification, 84-test suite receipt and comparison metadata audit.
- `MANIFEST.json`: SHA256 and byte size for every bundled file (excluding the manifest itself).

## Reproduce the small analyses

Run from this extracted folder, using Python with NumPy and SciPy installed:

```text
python analysis/aggregate_results.py --run-dir evidence/core_20261020T2250Z --titan-report inputs/titan_results.json --source-root source --evidence-only --output analysis_reproduced
python -m unittest discover -s analysis -p "test_*.py"
python analysis_penalty_alignment.py --run-dir evidence/core_20261020T2250Z --output penalty_alignment_reproduced
```

The isolated native/weighted fixtures also need the pinned JAX/Flax/Optax/Gymnasium stack recorded in `module_fixtures/verification_v2/source_closure.json`. The recorded harness deliberately enforces Windows CPU indices 4-7 and idle priority. Unlike the NumPy/SciPy analysis and core suite, that harness is not an unchanged Linux/macOS command. Gymnasium must satisfy the original dependency-directory guard explained in `module_fixtures/REPORT.md`. In the compatible Windows environment:

```text
python module_fixtures/run_fixtures.py --repo source --output fixtures_reproduced
```

This command is expected to exit 1 because it reproduces two scientific contract failures. Inspect the separate harness and contract statuses. It does not launch training. The portable source tree includes the fixture's actual project imports. Recorded package versions are provenance, not a promise of bitwise reproducibility on another platform.

The full suite implementation, configuration and tests are in `source/experiments/thrusterchain_diagnostics`. Read its README and training documentation before invoking any training entry point. No training is necessary to inspect or reproduce the evidence above. The packaging script itself records how this archive was assembled on Zeus; its original local paths are provenance, not portable analysis requirements.

## Evidence boundaries

This compact archive intentionally excludes large offline training data and checkpoint binaries. It includes the implemented suite and imported source closure, frozen configurations/manifests, evaluation banks, oracle references, 22 Zeus jobs' raw evaluation arrays, the user-supplied eight-job Titan summary report, selected historical audit evidence, analysis code, guide generator and test receipts. `--evidence-only` explicitly records omitted original-run files; it does not claim to verify omitted checkpoint/data bytes. Full training reproduction requires the preserved original data/checkpoints or a separately prepared full archive.

Zeus's 22 jobs and 44 checkpoint files were locally decoded and checked. Titan's eight terminal/eight midpoint summaries and reported identities were imported and checked; no Titan checkpoints or raw diagnostic arrays were independently accessed, and no Titan rollouts were rerun. Both hosts trained on CPU. Host and method are partly confounded for full/zero retention; a passing numerical fixture does not eliminate that limitation.

The five training seeds are the replication units. The 400 evaluation starts are a fixed bank, not 400 independent trained agents or a population expectation. Return intervals are exploratory paired t intervals with four degrees of freedom, no multiplicity correction and small-sample limitations. The 50k midpoint was not used to select checkpoints. No data top-ups or favorable-seed exclusions occurred.

Calibration uses an adaptively reused pooled bank of 128 rows. Rare critical counts are 1, 3, 0, 0, 1; zero critical counts are all zero. There is no critical-region coverage certificate. One-step bootstrapped target coverage, true long-horizon policy Q error, band refusal, action refusal and dose usability are distinct quantities. Native BCA uses width at recorded actions to weight cloning, not a direct penalty on arbitrary actions.

The source guide's embedded SVGs and scripts were preserved. Static HTML links, JavaScript syntax and result filters were checked. No enabled browser surface was available, so actual rendering, navigation interactions, console errors and screenshots were not tested. Pseudocode panels are screenshot-ready; no browser screenshots are claimed.

All original guides, frozen training records, historical studies and unrelated work remain intact. No new training, broad sweeps, source algorithm repair, publication, push or merge was performed during this results audit. Future repair and matched ablations are proposed and gated, not reported as completed.
