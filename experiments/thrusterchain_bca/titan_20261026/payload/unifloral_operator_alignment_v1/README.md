# Frozen native-operator alignment audit

This is the next bounded mechanism test, not a full BCA experiment. The protocol
was frozen before new evaluator calls. Original UniFloral/BCA sources, checkpoint,
negative findings and the completed inconclusive confirmation remain unchanged.

The audit compares the existing frozen residual radius U with its actual native
absolute twin-min one-step TD residual. It then separates that residual from
current-Q1 error against the already saved deterministic current-policy returns:

`Q1 - V = (min(Q1,Q2) - y) + (Q1 - min(Q1,Q2)) + (y - V)`.

Here y uses the saved target actor/critics and native clipped target-policy noise.
This accounting identifies operator differences; it does not prove their causal
effect on policy learning. The selected live_width function returns a radius,
not a full interval width: descriptive containment uses abs(residual) <= U.

## Fixed inputs and budget

- Exact terminal checkpoint: 20,000 critic / 10,000 actor / 20,000 auxiliary
  updates; final finite reference refresh at critic 20,000. The host was trained
  with kappa zero, so this is not a trained BCA-policy efficacy test.
- Existing two development banks: 128 starting episodes, nine candidates each.
  At most 1,152 one-step evaluator transitions; no prefixes or full rollouts.
- The existing 256 recorded training scale-bank transitions, read from the
  pinned dataset. These require no simulation. They are fitting-distribution
  diagnostics, not independent held-out calibration or coverage evidence.
- Four declared float32 native smoothing draws per bank. Target-critic row
  budget: 4 * (1,152 + 256) = 5,632. Seeds, order and clipping are fixed.
- No model fitting, optimizer/calibration updates, strength sweeps or
  confirmation access. The full state and checkpoint RNG must remain unchanged.
- A 300-second worker wall limit. Supervisor hashing/postchecks are outside
  that limit. Interrupted or failed work is preserved and never called success.

The new smoothing draws do not reconstruct the consumed original refresh key.
Four draws are a finite diagnostic bank, not an exact expected operator and not
four times as many independent episodes. Saved continuation values are
deterministic conditional on states/actions/current actor: there is no Monte
Carlo return sampling noise. New randomness here is target smoothing only.

## Execution

`run_checks.py` defaults to source/checkpoint/data preflight with no model or
simulator execution. `--execute --allow-outside-upstream-constraints` starts the
bounded GPU worker. It requires the existing validated provisional package
versions, CUDA, highest matmul precision and unchanged XLA flags. It installs
nothing. A conflicting runtime fails rather than silently falling back to CPU.

On Titan, the exact manual command is:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_operator_alignment_v1/run_checks.py" --execute --allow-outside-upstream-constraints
```

The assistant has not executed that WSL route. Local preparation tests use pure
CPU fixtures only. Actual checkpoint restoration, GPU replay and one-step audit
remain pending manual execution unless a later successful result receipt exists.

## Portable locations and limitations

Keep the required sibling suite layout and exact bytes listed in the prepared
manifests. Set BCA_TRANSFER_BUNDLE_ROOT to the relocated transfer bundle that
contains core_v4/, BUNDLE_MANIFEST.json and the pinned dataset at
research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz. The override
changes only the location; all hashes and split identities remain mandatory.

Old receipts retain their original Titan paths and timestamps as evidence.
They are not rewritten to disguise relocation. The code has not been tested
on Zeus and no usable Zeus GPU runtime is assumed. Run the portable package's
file preflight first; GPU execution must still satisfy the exact runtime gate.

No new result automatically approves production changes, Q normalization,
penalty selection or full training. Unfavorable results are retained. This audit
returns the investigation to the advisor's goal: useful conservative Q influence
and decisions beyond simple controls, not a generic supervised correction.
