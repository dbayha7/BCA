# Focused verification before main execution

2026-10-21, Zeus: the combined `python -B -m unittest discover -s tests -v` suite passed **21 tests in 3.606 seconds**. It covers 15 calibration fixtures and 6 grid/frozen-model fixtures. The model fixtures authenticate and read the actual existing checkpoint, preserved source closure, normalizer, and source audit.

An initial restricted-shell invocation passed the 15 calibration tests and two grid tests but could not complete the model test setup because its optional `git -C C:\Users\David\dev\BCA rev-parse HEAD` read was denied by the filesystem sandbox. The appropriately scoped read-only escalated invocation was approved and all 21 tests passed. This was an execution-permission issue, not a calibration failure, and did not change any scientific setting.

No main replication had been executed when these checks and the preflight throughput fixture completed. The fixture scored 4,096 fixed queries in 0.0069068 seconds after compilation, with full source authentication and preparation taking 3.46 seconds. It did not compute a calibration-coverage result. Posterior calibration, summary statistics, compression, and disk writes are additional costs.

Machine: Zeus, Windows build 26200, Intel Core Ultra 9 275HX, 24 logical processors, Python 3.10.11. This diagnostic process is restricted to logical CPUs 8–11 at IDLE priority with one-thread BLAS/runtime settings and the JAX CPU backend. No other process is stopped or modified.

The run is a declared custom ThrusterChain configuration. Native initialization emits its existing notice that no published TD3+BC reference configuration exists for this environment; this does not reproduce an external published benchmark score.
