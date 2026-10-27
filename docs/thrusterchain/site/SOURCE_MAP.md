# Source-to-destination map for the ThrusterChain BCA interface

Canonical entrypoint: `docs/thrusterchain/index.html` (self-contained; built by `site/build/build.py`).
Scope rule applied throughout: Unifloral TD3+BC host only. Pages and results produced with the repository's
CORL-derived host are preserved untouched and listed as archived; none of their content was merged.

## Destination parts and where each came from

| Part / section | Source material (repository paths) | How it was used |
|---|---|---|
| Start here · claims ladder, conventions, status | the 27 Oct brief's claim structure; `site/manifest.json` (status counts rendered at runtime) | written new; status counts come from the manifest |
| Environment · watch one episode | `experiments/thrusterchain_bca/titan_20261026/payload/transfer_bundle/core_v4/contracts/trajectory_bank.json` | five exported demo trajectories replayed; rewards/returns recomputed by `extract_environment.py` with the validated formula |
| Environment · one step worked | `core_v4/contracts/dynamics_fixtures.json` (A, B, physics, ten fixture cases) | formula validated against all ten cases and all 505 trajectory steps (max error 0.0) before export; one-step recomputation in the browser |
| Environment · state, actions, reward, episode end | `core_v4/contracts/environment_contract.json`, `core_v4/docs/environment.md`, `core_v4/thrusterchain/environment.py` | facts transcribed; bounds table from the contract |
| Environment · start laws and shift families | `environment_contract.json` (reset.start_laws), `core_v4/thrusterchain/canonical/thrusterchain.py` (START_LAWS), `core_v4/contracts/scenario_registry.json` | figure and table |
| Environment · data collection and support | `core_v4/thrusterchain/datasets.py` (box sampler, regions), `core_v4/docs/regular_mode.md`, `zeus_20261026_validation/support_shift_pilot_v1/PROTOCOL.md` and `SUPPORT_COUNTS.json`, `bca_one_sided_v1/run/data_receipt.json` (splits) | prose; retained-row count resolved from SUPPORT_COUNTS.json |
| BCA explained · host | `titan_20261026/payload/unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py`, `native_loader.py`, `candidate.py` | equations and pseudocode with line refs; full audit in `site/BCA_AUDIT.md` |
| BCA explained · components 1–5 | `unifloral_bca_pipeline_validation_v1/uncertainty_source/{historical/td3_bc_bca.py, historical/direct_extension.py, calibration/*.py}`, `bca_bridge.py`, `unifloral_checkpoint_v1/training.py`, `matched_policy_pilot_v1/continuation.py`, pipeline check receipts | equations, pseudocode, receipt numbers |
| BCA explained · worked score | `unifloral_operator_alignment_v1/results_*/raw_banks.npz` (state 0 of batch 1) | computed in the browser from the saved arrays |
| BCA explained · variants | `bca_one_sided_v1/one_sided_math.py`, `support_shift_pilot_v1/design.py`, `operator_analysis/cached_*`, `direct_q_replication_20261021_v1/SIGNAL_REDESIGN_OPTIONS.md` | table |
| Setup · chain, checkpoint, banks, operator, pilot, support | READMEs, `protocol.json` files, `PROTOCOL.md`, `PROPOSAL.md`, `MODEL_IDENTITY.md` under `experiments/thrusterchain_bca/` | prose |
| Results · side-by-side Q | `raw_banks.npz` + `raw_banks.json` (128 states × 9 candidates: saved_q, saved_u, saved_values, actions, states) | per-candidate chart and table; selections recomputed with the frozen tie rule and cross-checked against both diagnostics |
| Results · error reduction | same arrays; `unifloral_frozen_scale_comparison_v1/results_*/diagnostic.json` (analysis.q1_errors) | recomputed summaries labeled as such; official means shown beside |
| Results · decisions | `unifloral_frozen_usefulness_v1_seedfix/results_*/diagnostic.json`, `unifloral_frozen_scale_comparison_v1/results_*/diagnostic.json` (analysis.methods), `unifloral_development_mechanism_audit_v1p1/results_*/audit.json` | official counts; direction counts resolved by the build |
| Results · alignment | `zeus_20261026_validation/operator_analysis/all_prespecified_strata.csv`, `summary.json`, `RESULTS.md` | table (four-draw means from the CSV) |
| Results · privileged controls | `operator_analysis/cached_score_mechanism_v1/all_prespecified_strata.csv`, `cached_lower_only_v1/all_prespecified_strata.csv` | table |
| Results · policy return | `matched_policy_pilot_v1/run/evaluation/report.json`, `RUN_RESULT.json`, `protocol.json`, `run_analysis/RESULTS.md` | chart from returns_by_policy; comparison table; cross-checked paired differences |
| Results · support test, negatives | `support_shift_pilot_v1_retry_20261027/RUN_RESULT.json`, `HANDOFF_20261027.md`, `titan_continuation_20261027/PROPOSAL.md`, `bca_one_sided_v1/run_analysis/RESULTS.md`, user's 27 Oct report | prose; Titan-only items marked unavailable here |
| Evidence · manifest, status board, files, gaps, archive | `site/manifest.json` → `site/data/manifest_resolved.json` | rendered at runtime |

## Preserved, not merged (earlier CORL-derived host)

- `docs/thrusterchain/thrusterchain.html` (environment page version 16; main test, weighting test, critic viewer)
- `experiments/thrusterchain_diagnostics/action_only_20261021/presentation/*.html` (v3, five copies)
- `experiments/thrusterchain_diagnostics/direct_q_diagnostic_20261021_v2/presentation*/…v4….html`
- `experiments/thrusterchain_diagnostics/direct_q_replication_20261021_v1/presentation/…v5…, …v6…, v6_review/pass_0*.html, …v7_interactive.html`, `v7_ui/explorer.html` (32-state saved-bank Explorer)
- `docs/bca_mechanics/index.html` and its module pages
- `.claude/worktrees/agent-ad39c346ec7752a83` (page v27 worktree; its page is byte-identical to v16)

Reason: these report the CORL-derived `algorithms/td3_bc.py` host, 100k-update checkpoints with a 95k reference and a
gamma-1 oracle. The user's instruction on 27 Oct was to use the Unifloral base only.

## Not available on this machine (the brief names them on Titan)

`bca_direct_heldout_v1`, `bca_adaptive_signal_screen_v1`, `bca_adaptive_signal_recovery_v1`,
`unifloral_explicit_ood_diagnostic_v1`, `titan_timeout_recovery_20261027_v1`, `bca_penalty_evidence*`, and the
stall-boundary validity assay. The Codex task folder `C:\Users\David\Documents\Codex\2026-10-22\task` holds only the
22 Oct benchmark builds (v1–v4p1). Their rows in the manifest carry the user's report and no numbers.

## Build pipeline

1. `python -I site/build/extract_environment.py` → `site/data/environment.json` (validates the plant formula against every fixture).
2. `.venv\Scripts\python.exe -I site/build/extract_results.py` → `site/data/results.json` (recomputes every kappa-grid outcome and refuses to write on a mismatch with the diagnostics).
3. `python -I site/build/build.py` → `docs/thrusterchain/index.html`, `site/data/manifest_resolved.json`, `site/data/build_receipt.json` (resolves every manifest number, fills prose placeholders, checks ids, links and external references).
