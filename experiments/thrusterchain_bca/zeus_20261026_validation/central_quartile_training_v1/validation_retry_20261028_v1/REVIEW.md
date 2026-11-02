# Runtime validation recovery review, 2026-10-28

The first attempt's four files were inspected before moving. Its recorded supervisor/worker were 390/549. The user's `ps` output contained only the header for those exact PIDs. The last saved progress had not changed from 19:10:31.551810 UTC, and there was no terminal report or RESULT. No contrary overlap evidence was found. The native PowerShell move used resolved, package-contained source and destination paths and verified all four original hashes afterward. No files were deleted.

An independent code reviewer confirmed the original supervisor's double-interruption bookkeeping defect: default handlers were restored before postcheck, whose `Exception` handler does not catch `KeyboardInterrupt`. This explains a possible missing terminal receipt and is consistent with the supplied traceback; it does not prove the worker's final execution counts or scientific success.

The separate retry preserves the original executable, protocol and PREPARED seal. Its record-only handlers remain active through process creation, owned-child termination and final receipt writes. Worker outcome is saved before the bounded postcheck. Scoped process inspection includes the exact validator and retry paths as well as the existing launchers. Denied inspection stops execution. Both an exclusive retry marker and the original worker's required output directory protect against duplicate calls.

Independent review found and resolved three issues in the new wrapper before sealing:

1. The final retry-manifest check now compares against the launch-time manifest; it cannot accept a newly re-sealed replacement during execution.
2. A supervisor error prevents a pass even with exit code zero and a passed worker report.
3. Failed child cleanup reports `child_cleanup_failed_liveness_uncertain`, preserves unknown accounting and does not imply that the child exited.

The final 15 CPU fixtures passed in 0.653 seconds. They include actual harmless subprocess creation/reaping, actual repeated signal delivery, CLI startup, injected cleanup and postcheck failures, identity changes and unknown cumulative work. Windows sandbox atomic replacement was denied on the first fixture run; Windows tests explicitly substitute ordinary fixture writes. Target atomic writes, CUDA/model checks, real-source initialization and the full Gymnasium checker are not established by these tests. They remain the purpose of the manual validation retry.

The scientific training hold remains active. No production attempt or validation retry was reserved during preparation. No WSL access, install, training, real-data fitting, simulator execution or push was performed by this recovery task. The earlier attempt remains classified as interrupted with unknown final accounting, regardless of the second attempt's eventual outcome.
