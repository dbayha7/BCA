# Reserved bounded operator audit

The saved readiness receipt `readiness_20261026T220834_752076Z.json` passed. It records GPU backend on the RTX 5070 Ti Laptop GPU, highest matmul precision, the frozen package versions, no system-site-package inheritance, and the exact transferred UniFloral computational module path/hash. It used zero model forwards and evaluator transitions.

Readiness receipt SHA-256: `d7cecd21fd4e8106c7c385a2188fa5e37dfd47ee52feae7e4aa286cc038b7b08`.

Run once manually in PowerShell:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/run_operator_once.py --execute
```

The wrapper requires that exact readiness receipt and rechecks the unchanged source/checkpoint package. It rejects prior audit result directories, current matching audit processes and existing wrapper attempt evidence. It atomically reserves `OPERATOR_ATTEMPT.json` in this shared Windows validation folder, then invokes the original `launch.py --execute --allow-outside-upstream-constraints`. It adds no model/runtime/protocol changes.

The original worker retains its **300-second** limit and **1,152-transition** cap: 128 existing development states with nine existing candidates, one evaluator step each, four prespecified target-smoothing draws per bank, 256 saved fitting transitions and at most 5,632 target-critic input rows. No fitting, optimizer/calibration updates, new full rollouts, penalty sweep or confirmation access is permitted. Hash/postchecks remain outside the original worker limit.

Runtime/source checks, exact checkpoint restoration, reference-state equality and all saved actor/Q/radius forward replays precede the first evaluator transition. The audit remains exploratory and does not establish BCA efficacy or authorize full training.

The wrapper log is `operator_once.log`; wrapper terminal evidence is `OPERATOR_ATTEMPT_RESULT.json`. The scientific report stays in the original operator suite's new `results_*` directory. Every reservation is permanent evidence, including on failure/interruption/timeout. Do not remove it or rerun to resolve an uncertain outcome; inspect the preserved attempt first.

The process scan is limited to this user/current Ubuntu namespace. The shared exclusive reservation prevents concurrent executions through this wrapper across callers; prior result-directory checks also catch recorded direct attempts. A legacy direct launcher does not honor this new reservation, so use only the single wrapper command for this authorized attempt.

No actual reservation or audit was created during preparation. WSL was not retried by the assistant. Fixture tests use temporary files and stub launch functions; they perform no scientific execution.
