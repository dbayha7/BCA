# Run the bounded design gate on Zeus

Status: prepared for manual execution; no gate attempt has been reserved or run by the assistant. This is a development design screen of the original band, not permission to train a policy on it. Both older blank decision forms remain unchanged. A failed result stays failed; there is no automatic retry, region change or radius search.

The machine-readable decision contract was frozen before decoding new gate arrays or outcomes. Its SHA-256 is:

`7b27bcd379f30b3b627f1fcdd91ec9b3537d06cd1a31bbd0855faa4eba94d877`

From PowerShell on Zeus, run this once:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_gate_v1/launch.py --execute
```

The launcher checks prepared hashes and package versions before reserving `run/`. It sets the existing portable bundle location inside the worker, verifies the pinned Titan UniFloral source/checkpoint/export, then checks CUDA before decoding data. The environment is the previously installed isolated BCA runtime; no installation is performed. The runtime remains the previously disclosed exception to upstream dependency constraints. It does not load CORL models or change checkpoint normalization.

For a non-executing metadata preflight, replace `--execute` with `--preflight`. That does not validate CUDA or model forward execution. The assistant's WSL route remains denied; it has not been retried. The Windows CPU interpreter differs from the frozen runtime and was used only for synthetic tests and source hashes.

The fixed ceiling is 900 seconds and 28,104 real simulator transitions, with zero critic, actor, auxiliary or calibration updates. Progress prints while running and is saved to `run/progress.json`. Final accounting is in `run/RESULT.json`; scientific decisions and measurements are in `run/report.json`. An interruption keeps the reservation and all saved evidence. Do not delete `run/` to repeat a failed or interrupted attempt.

The support criterion retains RMS radius 0.5, 64 original neighbors, 32 rows/eight episodes, and evidence on both adjacent action bands. The value criterion requires a one-return-unit center-to-flank gap at 25% of high-load anchors. The free actor must enter the hole on at least 1% of high-load steps and in at least two episodes. See `contract.json` for all fixed coverage, balance and eligibility criteria. Within-band return spread is also measured, without an added pass mark.

All 32 preparation tests passed on synthetic data/toy dynamics. They include the entire planned transition ledger, float32 mask boundaries, state/time/discount semantics, preservation of failed attempts and frozen identities. Real canonical parity and CUDA/model integration are pending execution. The initial test run had two Windows temporary-file rename errors in fixture setup; exclusive fixture creation resolved them without altering scientific criteria.

Later training, if separately approved, will extend the existing 10k support-pilot fork. This gate does not create that larger training matrix or establish a missing-label causal effect, BCA benefit, joint four-action support, or held-out confirmation.
