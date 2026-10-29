# Amendment 1 implementation review

The original failed validation and all prior source seals are preserved. The new comparator was implemented separately and reviewed against the actual restore call sites. This amendment changes implementation checks and validation bookkeeping, not the scientific configuration.

Resolved findings before sealing:

- Restore checks compared MessagePack byte order rather than decoded state. The shared checker now ignores dictionary insertion order only; exact data, schema and type checks remain. Runtime validation, branch restoration and actor-export verification use it consistently.
- The target failure did not identify the restore stage or provide an exact-comparison receipt. The amended worker identifies `full_state_restore`; successful comparison records leaf/container counts and source checkpoint SHA. A failed comparison identifies its field path. The historical failure is not retrospectively reclassified.
- A prior preparation helper could compare the JSON plan, then silently update its prose hash. The amended helper requires byte-identical protocol JSON, scientific prose and data plan and never rewrites those files. Four new tests cover scope refusal, no refreeze after attempt 3, and public entry routing.
- The old validation supervisor could lose a terminal receipt on repeated Ctrl-C. The public entry now dispatches to the separately reviewed record-only signal supervisor, retaining its fail-closed postcheck, child cleanup and unknown-accounting behavior. The unchanged worker still requires the exact reserved output directory and its owning supervisor PID.
- Historical decision documents specify a larger workload. The reconciliation explicitly retains 36 calibrations and the current evaluated checkpoints; no silent adoption of the 63-calibration/11k alternative occurs.

133 unique CPU checks passed across the full and affected suites. Exact commands, timings and mocks are in `TEST_RESULTS.json`. These include actual JAX mapping reconstruction on a two-leaf fixture, toy-state restoration/continuation, injected corruption, actual harmless child-process cleanup and real repeated signal delivery. They do not claim target GPU validation. Independent review found no additional startup issue in the attempt-3 lineage and worker routing.

Static apply functions, model and optimizer definitions, normalization and data-role binding remain authenticated by existing manifests and templates. Checkpoint hashes remain mandatory. The decoded comparator cannot by itself certify those nonserialized properties, nor does a successful synthetic test establish scientific efficacy.

The new validation command reserves only attempt 3 under the amendment directory. Attempt 1's unknown final work remains unknown; attempt 2's known synthetic counts are recorded separately. The production hold and pending 12-hour approval remain. No WSL call, runtime validation, production run, install, merge or push was performed during amendment preparation.

Remaining target checks: complete actual pinned-model save/restore and next-step parity on CUDA, five read-only real-source initializations with matched normalization, full Gymnasium checker, and all worker accounting/identity checks. A failure remains a failed attempt; no automatic retry or production release follows.
