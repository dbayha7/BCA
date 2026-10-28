# Superseded: do not launch

This folder was a corrected copy of `../conditional_ood_gate_v1/`, prepared on 28 October 2026 at about 12:06 EDT. It
fixes the same two defects in v1's first attempt. One stopped the run: a file attribute that the in-memory canonical
module lacks. The other, in the crash handler, hid that error message.

Another agent had already made the same repair at 11:12 EDT, in `../conditional_ood_gate_v1_retry_20261028/`. That
version also checks the dynamics function's compilation path and global namespace. It ran at 11:33 EDT as attempt 2 and
completed in 63.7 seconds. Its result is the gate's record: the design gate failed on local support and free-policy
exposure. See that folder's `run/report.json`.

This folder was never frozen or prepared. It has no `DECISION_FROZEN.json` and no `PREPARED.json`, so its launcher
refuses to run. No data was decoded, no model was loaded, and the simulator was never stepped. One regression test only
imported and hashed the real pinned simulator core.

Everything in this folder is kept as a record:
- the two edited lines;
- `test_v2_fixes.py`, whose two regression tests pass;
- the lineage note in `contract.json`.

It can be deleted without losing evidence.
