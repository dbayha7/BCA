# ThrusterChain diagnostics

A small, runnable diagnostic suite beside the historical ThrusterChain experiments. This is implementation and numerical validation, not a trained-method comparison or evidence of demonstrated novelty. Existing source, checkpoints, and experimental results are unchanged.

## Scientific contract

Two models answer different questions:

* **Continuous ThrusterChain:** the existing 13-dimensional deterministic physics and four commands, with an observed episode-fixed actuator regime and remaining time appended to the observation. The primary objective is a 100-step undiscounted return with zero terminal continuation. The original experiment's discounted long-horizon critics and historical 100-step scores remain historical objectives. A fixed-start policy return is computed by deterministic simulation; a finite bank average is not a population expectation.
* **Finite companion:** a deliberately small MDP with its own boundary and quantization rules, observed regime/load, and remaining time. Its enumerated kernels admit finite-horizon dynamic programming and population occupancy calculations without sampling error, subject to floating-point arithmetic. It is analogous to the continuous task, not an exact discretization of its full physics.

The continuous QP minimizes the same finite-horizon objective used to score its controls. In the stall regime, an above-knee command is dominated by the smaller command producing the same thrust and lower energy cost. The remaining bounded linear-quadratic problem is convex. Solver output includes a replay check and a Frank-Wolfe gap bounding suboptimality in the mathematical float64 model. Numerical certification is not symbolic exact arithmetic or a population generalization guarantee. The last action has only an energy cost because the terminal state is uncharged.

The monotone regime permits useful commands above the stall knee. Fixed-cap and regime-aware-cap controls expose the opportunity cost of indiscriminate caution. The continuous stall dominance theorem does not automatically hold on the finite companion's five-action grid: it omits the smaller command producing the same thrust. Its harmful-versus-useful contrast is asserted in the specified critical region, not globally. Heat accumulation, delayed actuator damage, and additional stochastic continuous dynamics are deferred because they would change these oracle contracts.

## Data and support

The fixed-state track queries prescribed states independently; it is not a trajectory dataset. The trajectory track advances actual MDP states and keeps each transition's own next observation. Both record episode/time, behavior action probability, pre-outcome retention propensity, split, and provenance. They use declared full/rare/zero retention in the critical region, independent of evaluation outcomes. Full retention is not a claim of full visitation coverage.

Behavior is bounded categorical randomization, so probabilities are action masses. The continuous controls are deterministic; dividing a smooth behavior density by their point mass does not produce valid ordinary action importance weights. Context-frequency ratios, conditional action probabilities, and policy occupancy ratios are distinct quantities. Population support and realized dataset counts are reported separately. A positive target mass with zero source mass is unsupported; clipping cannot recover it.

The observed-regime tasks are separate from the **hidden paired-world negative control**. That control removes world identity from learner observations and metadata and restricts logging actions to a common-dynamics region. Equality must hold for the entire observed transition/reward/termination law on all behavior-reachable support, not merely after deleting rows. Coupled complete-log equality is a regression check alongside that support certificate. Rare probes deliberately break the zero-information construction and are analyzed separately.

Train and holdout streams have separate seeds/namespaces. Evaluator-only complete logs, support certificates, and oracle arrays are saved separately from learner rows. A reserved optimization seed does not imply any optimizer ran. There are no automatic budget top-ups, no method-dependent extra data, and no silent omission of refused seeds or rows.

## Method-neutral evaluation

Adapters accept a policy and optional predicted Q/V values, intervals, and refusal probabilities. They distinguish the Q of the proposed policy from optimal Q. Refusal requires an explicit fallback; deployed return, regret, failure, harm, and opportunity loss include its decisions. Missing optional outputs remain absent/null rather than pretending to have zero error.

Support, critical counts, visitation mismatch, true-value errors, action-ranking errors, interval coverage/width, and refusal are separate diagnostics. Coverage over a fixed query law is not a frequentist guarantee across independent trained datasets. One-step residual coverage does not establish long-horizon Q coverage. An oracle error penalty is a placement control, not an optimal intervention; its failure cannot prove impossibility.

## Relation to prior work

This suite combines existing diagnostic ideas in an interpretable control task. [Fu et al. (2019)](https://proceedings.mlr.press/v97/fu19a.html) use oracle-based Q-learning unit tests; [bsuite](https://arxiv.org/abs/1908.03568) isolates agent capabilities with small experiments; [RL Unplugged](https://arxiv.org/abs/2006.13888) standardizes offline data and policy-selection protocols; [Schweighofer et al. (2022)](https://proceedings.mlr.press/v199/schweighofer22a.html) study dataset quality and coverage. The positioning here is diagnostic attribution through explicit model, support, and identifiability contracts. Novelty and effectiveness require further comparative research.

## Deferred evaluation

TD3+BC, IQL, CQL, ReBRAC, BCA, and other learned-method comparisons have not been run by this implementation. A future study must freeze algorithm definitions, hyperparameter/checkpoint selection, data and online-selection budgets, independent training seeds, and holdout reporting. Confirmatory training requires separate authorization.

## Run and inspect

From the BCA repository root, using Python with the packages in `requirements.txt`:

```text
python -B -m unittest discover -s experiments/thrusterchain_diagnostics/tests -v
python -B -m experiments.thrusterchain_diagnostics smoke --output experiments/thrusterchain_diagnostics/artifacts/my_new_run
```

The smoke command refuses any existing output directory. `--config` accepts a JSON file; the default is `configs/smoke.json`. To include authoritative source-parity testing, set `BCA_LEGACY_ROOT` to the BCA checkout before running tests, using its existing pinned shift dependencies. Otherwise that optional integration test is explicitly skipped; the independent matrix-reconstruction and portable numerical tests still run. On Zeus the existing executable is `C:\Users\David\dev\BCA\.venv\Scripts\python.exe`.

`learner/` contains only retained rows and public generation metadata. `evaluator/` contains complete audit logs, held-out data, exact finite kernels/Q values, continuous QP plans, and hidden-world probes. Do not train on the parent output directory indiscriminately. Its `report.json` records exact finite diagnostics, fixed-start continuous results, generated counts, runtime versions, source hashes, and zero training/selection/top-up budgets. Rare-probe probabilities describe opportunities to observe differing kernels, not guaranteed successful statistical identification.

Public modules: `continuous.py` (physics, time-aware controls, QP), `finite.py` (enumerated MDP and DP), `continuous_data.py` and `data.py` (generators), `diagnostics.py` (method output adapter), and `__main__.py` (lightweight reproducible run). `ThrusterChain_diagnostics.html` is the revised standalone learning guide. Historical guide results retain their original scope; the new-suite section draws its readouts from this implementation's artifacts.

For method comparisons, `MethodOutput` requires a probability policy of shape `[H,S,A]`. Optional Q/V predictions explicitly target either that proposed policy or the optimum. Default queries are uniform over the entire finite grid, including unreachable states; supply `query_weights` for another declared law. Failure probabilities include initial and terminal visits. Local decision disadvantages use a baseline-policy continuation and are not a causal decomposition of total trajectory loss.
