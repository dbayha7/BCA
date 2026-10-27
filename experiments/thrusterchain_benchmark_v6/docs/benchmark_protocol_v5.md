# V5 comparison protocol and coverage

This is a protocol template, not a preregistered study or an assertion of
completed results. Freeze actual dataset identities, algorithms and resolved
hyperparameters, seed assignments, update and selection budgets, and reporting
code before collecting results intended for final reporting. The repository's
existing scenario families and historical seeds have been examined in prior
development; their use does not establish generalization to unseen task families.

## Independently varied conditions

| Axis | Existing condition | What it does and does not establish |
|---|---|---|
| Initial state | recorded, strong current, big waves, big kicks | Current and wave laws extend beyond recorded ranges; big kicks change Gaussian scale with overlapping support. These are start-law changes, not changed transition dynamics. |
| Requested actions | mixture mass epsilon 0.25, 0.01, or 0 in a0 > 0.5 | The uniform, rare and absent regions are exact properties of the requested-action collection law. They are not exact closed-loop state-action occupancy densities. |
| Task boundary | finite horizon or continuing time limit | Finite mode has remaining time and zero terminal bootstrap. Continuing mode has stationary physical observations and bootstraps from the actual post-step state. Do not pool their scores. |
| World | stall or no stall, separately declared | Different world choices change actuator dynamics. A single-world result does not establish adaptation or generalization to changed dynamics. |

The current collector is a state-independent random behavior policy. Its coverage
does not establish expert-quality data, stitching, partially observed control,
stochasticity, or long-horizon performance. Add new task or dataset families only
with explicit generation contracts, independent test coverage and separate results.

## Required record per attempted run

Record source and environment hashes; dataset content and file digests; full
generator configuration, trajectory groups and data split; algorithm source,
resolved scalar configuration and action/reward/observation transformations;
training seed, checkpoint and preprocessing hashes; realized critic, actor and
other optimizer counts, minibatch size and dataset draws; failures; hardware and
runtime; all candidate training and selection costs; every final episode seed,
scenario, raw return and discounted return. Include stochastic-policy action
mode and its random stream. A run is not independent merely because its seed
label differs, and a hash does not prove that final outcomes were unseen.

Predeclare checkpoint selection. In strict offline mode, no simulator feedback
may affect policy, checkpoint or hyperparameter selection. Simulator-assisted
selection must report all interactions, candidates and prior development feedback
separately from final reporting episodes. Any parameter update using collected
environment transitions is offline-to-online adaptation, with its own budget.
Use matched data and declare the matching axis for optimization/compute rather
than assuming the same update count has the same cost across algorithms.

Treat trained policies as the replication unit for method uncertainty; episode
returns describe conditional evaluation variability. Paired starts are useful
for comparing policies when the identities genuinely match. `comparison.py`
validates declared seed separation, detects missing planned episodes, and
averages episode returns within each declared training run. It does not prove
independence, correct source provenance, or the time of precommitment.

Report each dataset and scenario separately, with individual run-level results,
failed/incomplete runs, raw returns and discounted returns. Do not invent
normalized scores without independently documented reference policies and
version-specific denominators. A one-step or small-update test is evidence of
plumbing and numerical agreement only, not a performance comparison.
