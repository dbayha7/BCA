# Penalty strength and network counts

The reported Greek kappa is the multiplier in `Q1 - kappa * U`. It is not the number of critics, uncertainty heads, bootstrap samples or target draws. No head-count sweep has run. The user subsequently requested a separate K=1,2,5,10,15,20 experiment for later and explicitly prioritized the current experiment. That ablation is queued here only; its critic-versus-uncertainty-head interpretation and controlled penalty strength must be specified before design or implementation.

| Component | Actual current configuration | Meaning / aggregation |
|---|---|---|
| Native UniFloral critic | Two separately parameterized Q networks, each 256×256 hidden units and one scalar output | Native target takes twin minimum; actor/action score reads Q1 |
| BCA local scale | One state/action-dependent scalar-output network, 64×64 hidden units, Dense(1), softplus + 0.01 | No uncertainty-head ensemble or across-head aggregation |
| Target critic/actor | Frozen target copies of the native models | Target critic still has two Q networks; not an extra fitted uncertainty ensemble |
| Calibration posterior | 1,000 bootstrap draws in the one-sided validation | Distribution over calibration-risk crossings; quantile/order-statistic threshold, not Q-network heads |
| Local bootstrap | One random normalized weight vector over the 256 fitting rows per update | Data weighting for one scale network, not 256 heads |
| Operator audit smoothing | Four target smoothing draws per candidate | Monte Carlo native-target draws, not critic or uncertainty heads |
| One-sided logged labels | One fixed target smoothing draw per logged row | Reused during its 128 updates |

The transferred checkpoint was trained with kappa=0. The earlier frozen original-BCA development selector used kappa=3; the one-sided pilot used kappa=1 for both its original-BCA control and new calibrated penalty. Ordinary/raw controls differ in global threshold, with matched-effective-strength aliases also recorded. These strengths do not change the network counts.

Exact source evidence:

- `titan_20261026/payload/unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py:65-94`: two 256-unit hidden layers, Dense(1), `nn.vmap(...variable_axes={params:0}, axis_size=2)`.
- `titan_20261026/payload/unifloral_bca_pipeline_validation_v1/uncertainty_source/calibration/network.py:8-31`: Calibrator hidden=64, depth=2, scalar Dense(1), positive softplus output.
- `bca_one_sided_v1/worker.py:138-140`: runtime checks state_dep=true, hidden=64, depth=2 before copying the original parameters.
- `bca_one_sided_v1/worker.py:303-309`: exact original/new/ordinary/raw/actor-distance scoring formulas.
- `titan_20261026/payload/unifloral_checkpoint_v1/config.json`: original training kappa_penalty=0.
- `bca_one_sided_v1/protocol.json`: 1,000 posterior draws, fixed kappa=1 and target-draw rules.

A separate uncertainty-head ablation could vary K=1,2,5,10,15,20 local-scale ensemble members while keeping the native twin critics fixed. It would first need a defined independent-training/bootstrap rule, an aggregation rule for the final positive scale and recalibration of that aggregate, plus equal-total-update/parameter or explicit per-head-compute controls. That is an architecture ablation requiring new fits, not a relabeling of current kappa tests. Varying the native critic count instead would change the UniFloral host: one critic removes the twin-min target, and counts above two also change the operator/architecture and invalidate existing native parity and checkpoint comparability. No such change is implemented; the user's requested sequence defers this separate experiment.
