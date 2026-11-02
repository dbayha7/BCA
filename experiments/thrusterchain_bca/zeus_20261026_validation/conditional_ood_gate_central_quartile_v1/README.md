# Central-quartile development gate

Prepared under the user's approval of one replacement design. Nothing in this folder authorizes full training. The old failed gate and the unfinished alternative drafts remain unchanged.

Read [PROTOCOL.md](PROTOCOL.md) for the precise scientific scope and [contract.json](contract.json) for every numerical decision. The construction is `abs(a0)<=0.25` at observed load >=0.4. It tests relative training-state typicalness, retained action alternatives, full Q-pi differences and free baseline exposure. It does not establish dense joint state-action support or BCA efficacy.

Run once from a Zeus PowerShell terminal after the prepared source manifest has been verified:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_gate_central_quartile_v1/launch.py --execute
```

The supervisor checks the frozen input hashes and package versions before an exclusive reservation. The capped worker checks CUDA, model/source identity and the actual environment adapter. It refuses an existing `run/`; do not delete a reservation, modify frozen files or rerun automatically. A failure must remain recorded. `run/RESULT.json`, `run/report.json`, `run/progress.json` and worker logs preserve completion/accounting evidence. A scientific gate failure can still have exit code zero; inspect `decision`.

Maximum 900 seconds, 28,104 simulator transitions, 384 candidate references, 16 free trajectories, 81 resets, 640 CPU reference anchors and 54,067,200 distance pairs. Zero training or calibration updates. The previous 28,104-transition failed gate is separate consumed work, not reset by this experiment.

Assistant-side WSL access remains denied and was not retried. The command uses the previously installed isolated runtime; no install or environment modification is requested. CPU/source tests do not substitute for CUDA validation. Gymnasium is explicitly stubbed only in one import-interface test; production imports the required real package.

No new gate outcomes or dataset transition arrays were evaluated during preparation. All selected logged episodes are already development-consumed. The baseline-informed candidate shape and prior results are disclosed in the protocol. These control/calibration RNG integers are independent named roles within this protocol, not new evaluation start seeds or a claim of never-used numeric integers.
