# Approved bounded support-shift pilot

Read PROTOCOL.md for the fixed method, dataset, matching rule, learning gate and limitations. PREPARED.json pins the executable and dependencies. CPU_TESTS.json records CPU fixtures; it is not a CUDA execution receipt.

Run once from PowerShell using the existing isolated Ubuntu environment:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/support_shift_pilot_v1/launch.py --execute
```

Replace `--execute` with `--preflight` for read-only file/hash checks, or `--status` for saved status. Neither mode initializes models or reserves an attempt.

The exclusive reservation is permanent. Do not relaunch after interruption, timeout or failure; preserve ATTEMPT.json, RUN_RESULT.json and run/. There is no automatic retry, resume or budget extension. A running status without a terminal receipt is not evidence that the process is alive; inspect before taking action.

Maximum scientific work: 64,000 critic, 32,000 actor and 64,000 original auxiliary updates; 16 calibrations; 512 evaluation episodes and 51,200 simulator transitions. The 150-minute global cap includes GPU initialization, compilation, training, matching and final evaluation. Individual jobs inherit the remaining global deadline. No broader training or head sweep starts afterward.

Progress prints committed counters every 20 training updates. Four warmups run first; initial weight identity/optimizer checks must pass before branches. Each branch restores its own sealed 10k state, never the previous broad-trained checkpoint. Evaluation starts only after all sixteen training jobs succeed and verifies twelve final actors plus four untrained references.

RUN_RESULT.json is authoritative for terminal status. Results and failed-attempt evidence remain local; nothing is automatically committed or pushed.
