# Knee decision-signal analysis

This package contains the completed retrospective CPU analysis of saved knee/stall candidate banks. It changes no experiment, model, dataset, criterion or approval. It is separate from Honeypot/Bottleneck Path A.

## Read first

- `output/report_v2/Knee_Decision_Signal_Report.pdf`: six-page report with two vector figures.
- `output/report_v2/01_native_relative_ranking.png` and `02_penalty_margin_tradeoff.png`: standalone figures.
- `output/run2/metrics.json`: all 24 policies, all seeds/strengths, exact and sensitivity results.
- `output/run2/pairs.csv`: all 13,824 candidate-versus-logged comparisons.
- `SOURCE_REVIEW.md`, `TEST_RECEIPT.json`, `VERIFICATION.json`: independent review and checks.

The three training seeds are independent replicates. Candidate rows and states are nested, reused observations. Pairwise preferences are not deployed actions or four-action-bank argmax counts. Own-policy reference values differ across models. Exact historical classifications and the original INCONCLUSIVE verdict remain unchanged.

## Reproduce

Use the existing Python environment with NumPy and matplotlib. No dependency installation is part of this package. Extract this directory back to `analysis/knee_decision_signal_v1` in an authorized BCA checkout, preserving existing files rather than overwriting unrelated work.

Focused tests (no model or simulator imports):

```text
python -B -m unittest discover -s analysis/knee_decision_signal_v1 -p test_analyze.py -v
```

Regenerate the report from the included results, using a new output directory:

```text
python -B analysis/knee_decision_signal_v1/build_report.py --input analysis/knee_decision_signal_v1/output/run2 --output analysis/knee_decision_signal_v1/output/report_reproduced
```

Recompute the saved-array analysis only when the original authorized source files listed in `output/run2/RECEIPT.json` are available:

```text
python -B analysis/knee_decision_signal_v1/analyze.py --output-dir analysis/knee_decision_signal_v1/output/analysis_reproduced
```

The ZIP contains analysis code, results, figures and provenance. It does not contain the original large datasets/checkpoints or the prior analysis package. Those dependencies are named and hash-bound in the receipt. Report regeneration from included results does not need them. `PACKAGE_MANIFEST.json` lists each packaged file's SHA256 and byte size; the manifest excludes itself.

## Scope and missing evidence

The candidate archives do not contain aligned one-step TD targets, signed residuals, eta components or candidate gradients. The report proposes a possible future measurement but does not implement or authorize it. No new model inference, simulator call, optimizer update or calibration was performed for this analysis or report delivery.
