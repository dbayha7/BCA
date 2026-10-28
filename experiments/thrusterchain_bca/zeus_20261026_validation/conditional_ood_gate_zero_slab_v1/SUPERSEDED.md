# Superseded: do not freeze or launch

This folder was built on 28 October 2026, between 13:00 and 13:40 EDT, as a replacement for the shelved stall-knee
band. It held a zero-thrust slab of half-width 0.1 at load at least 0.4. It was never frozen or prepared: there is no
`DECISION_FROZEN.json` and no `PREPARED.json`, so its launcher refuses to run. No data was decoded, no model was loaded,
and the simulator was never stepped.

Meanwhile, another replacement was approved by the user and run: `../conditional_ood_gate_central_quartile_v1/`, a
zero-centered slab of half-width 0.25 at load at least 0.4. It ran at 13:34 EDT and passed its design gate. That run
used the same 64 anchor episodes (the last 64 development validation groups) and the same 16 free-play start seeds
that this folder would use. Running this folder now would be a second, differently sized look at the same anchors,
chosen after seeing a result. So it must not be frozen or run.

**What it contains, kept as a record:**
- `DESIGN.md`, `contract.json`, `PROTOCOL.md` and `README.md`;
- the code changes: the band centered at zero, a radius-free support rule with a training-only reference quantile,
  and a worker check that the geometry constants match the contract;
- 44 CPU tests that pass.

**The pre-freeze review's findings were never applied, since the folder will not be frozen.** The main ones:
- the launcher does not hash `DESIGN.md`;
- the half-width rule sits exactly on its threshold and does not name its statistic;
- the agent-set numbers depart from the decision block's rule that the user sets every number.

The folder can be deleted without losing evidence.
