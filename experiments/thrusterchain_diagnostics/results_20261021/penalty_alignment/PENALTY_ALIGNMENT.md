# Penalty alignment on frozen query pairs

The multiplier is behavior-cloning regularization, not an estimated physical cost. These diagnostics test its proposed usefulness as an error/harm signal, not a mathematical guarantee. No training or rollout was added.

| Retention | Available BCA seeds | Dose vs absolute Q-error rank correlations | Dose vs candidate opportunity-loss rank correlations |
|---|---:|---|---|
| full | 1/5 | 0.215 | 0.335 |
| rare | 5/5 | 0.563, 0.537, 0.452, 0.468, 0.372 | 0.359, 0.289, 0.160, 0.370, 0.272 |
| zero | 5/5 | 0.433, 0.580, 0.447, 0.346, 0.510 | 0.297, 0.358, 0.477, 0.210, 0.441 |

Across 528 designed states, 154 have a larger BC multiplier on the best-Q candidate than on the worst-Q candidate. Every qualifying example is saved with seed, state index, time, commands, Q values and doses. This is a counterexample to an unconditional harm-ranking interpretation, not proof that uncertainty should always be smaller for good actions.

Only 1/5 full BCA raw runs available; do not extrapolate that seed to full condition.
All five rare and zero BCA raw runs are available; each learned policy defines its own continuation Q.
Residual scale is not trained to rank true Q error or harmful actions. A weak/negative association is a utility finding for that proposed use, not proof its formula is wrong.
This measures the BC multiplier only. The actual BC loss also contains squared distance to the logged action, and its gradient depends on the actor; neither actual loss nor gradient ordering follows from multiplier ordering.
Native BCA evaluates the multiplier at recorded actions before weighting each logged row cloning loss. These six-candidate values are diagnostic readouts of the frozen function, not actual training weights or direct penalties imposed on those candidate commands.
The recorded dose at 100k is a diagnostic query evaluation, not the full history of training intervention values.
Counterexamples enumerate all qualifying states in deterministic job/state order, without outcome-based omission.

Reproduce with `python analysis_penalty_alignment.py --run-dir <core_20261020T2250Z> --output <new-directory>`. The same command works on bundled raw evidence; no checkpoint bytes are needed.
