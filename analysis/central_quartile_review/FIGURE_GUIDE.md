# Central-quartile evaluation: figure guide

These figures visualize the completed, fixed development study. They are descriptive evidence for this setup, not confirmation of general BCA efficacy. The frozen ordinary-current policy screen failed, the high-load fixed-band Q-accuracy screen passed, and both rescue screens failed. Strong-current policy comparisons are separately reported, not substituted for the primary screen.

## How to read the study

- **Three training seeds** are the independent training replications. Each free-policy mean uses 32 paired evaluation starts in a context; rows, action candidates, times and checkpoints are not extra training replications.
- **Full** uses all fitting transitions. **Hole** withholds logged tuple labels where `|a[0]| ≤ 0.25` and observed load `obs[10] + obs[11] ≥ 0.4`. **Random** removes the same number of labels with matching coarse episode/time/load counts; it does not match every state, reward or successor distribution.
- **Strong current** changes the observed initial-current distribution from `[0.1, 0.7]` to `[0.7, 1.1]`; dynamics are unchanged. Mask-based exposure does not establish local state density or joint-action support.
- Discounted return and Q use **γ = 0.98**. In own-policy Q panels, the reference takes the forced action and then follows that snapshot's own actor through the remaining 100-step episode. Error is score minus reference. Comparisons between different policies have different own-policy Q references; the shared-reference figure answers a different question.
- High-load fixed-band Q figures use 21 ordinary or 32 strong-current anchor families per seed, with three correlated actions per anchor. Free-policy visited panels use 32 episodes sampled at four times. Forced-bank diagnostics use six candidate actions and do not measure policy-return changes or global optimality.
- The final figures use the fixed 20k checkpoint unless explicitly labeled. The learning-curve plot shows 10k, 15k and 20k, without selecting the best observed checkpoint. Gradient figures use local 20k training-probe derivatives, not executed motion or a causal decomposition of the full training trajectory.

## Policy returns and distribution shifts

1. [Final returns, all conditions and seeds](figures_policy/01_final_returns.png) · [SVG](figures_policy/01_final_returns.svg) · [PDF](figures_policy/01_final_returns.pdf)
2. [BCA minus native and distance control](figures_policy/02_policy_contrasts.png) · [SVG](figures_policy/02_policy_contrasts.svg) · [PDF](figures_policy/02_policy_contrasts.pdf)
3. [Native hole damage and BCA-specific interaction](figures_policy/03_hole_interaction.png) · [SVG](figures_policy/03_hole_interaction.svg) · [PDF](figures_policy/03_hole_interaction.pdf)
4. [Fixed post-fork learning checkpoints](figures_policy/04_post_fork_learning.png) · [SVG](figures_policy/04_post_fork_learning.svg) · [PDF](figures_policy/04_post_fork_learning.pdf)
5. [Free-policy exposure to the withheld band](figures_decisions/free_policy_hole_exposure.png) · [SVG](figures_decisions/free_policy_hole_exposure.svg) · [PDF](figures_decisions/free_policy_hole_exposure.pdf)

## Q accuracy and the effect of score adjustment

6. [Native Q1 versus BCA-adjusted score, each against its own policy](figures_q/01_cross_model_accuracy.png) · [SVG](figures_q/01_cross_model_accuracy.svg) · [PDF](figures_q/01_cross_model_accuracy.pdf)
7. [Raw versus adjusted score on the same critic](figures_q/02_same_critic_accuracy.png) · [SVG](figures_q/02_same_critic_accuracy.svg) · [PDF](figures_q/02_same_critic_accuracy.pdf)
8. [Improved and worsened errors, including sign crossings](figures_q/03_adjustment_harm.png) · [SVG](figures_q/03_adjustment_harm.svg) · [PDF](figures_q/03_adjustment_harm.pdf)
9. [Accuracy on freely visited states](figures_q/04_visited_accuracy.png) · [SVG](figures_q/04_visited_accuracy.svg) · [PDF](figures_q/04_visited_accuracy.pdf)
10. [Accuracy-rescue screen and hole interaction](figures_q/05_accuracy_rescue.png) · [SVG](figures_q/05_accuracy_rescue.svg) · [PDF](figures_q/05_accuracy_rescue.pdf)
11. [High/low-load and band/flank strata](figures_q/06_accuracy_by_stratum.png) · [SVG](figures_q/06_accuracy_by_stratum.svg) · [PDF](figures_q/06_accuracy_by_stratum.pdf)
12. [Score deviation from a common full-data native reference](figures_q/07_common_reference.png) · [SVG](figures_q/07_common_reference.svg) · [PDF](figures_q/07_common_reference.pdf)

## Decisions and local training diagnostics

13. [Forced-bank regret and changed choices](figures_decisions/forced_bank_decisions.png) · [SVG](figures_decisions/forced_bank_decisions.svg) · [PDF](figures_decisions/forced_bank_decisions.pdf)
14. [Center action versus the better retained flank](figures_decisions/center_vs_retained_flanks.png) · [SVG](figures_decisions/center_vs_retained_flanks.svg) · [PDF](figures_decisions/center_vs_retained_flanks.pdf)
15. [Actor-loss gradient components](figures_gradient/01_parameter_gradient_components.png) · [SVG](figures_gradient/01_parameter_gradient_components.svg) · [PDF](figures_gradient/01_parameter_gradient_components.pdf)
16. [Local action-gradient direction](figures_gradient/02_action_gradient_direction.png) · [SVG](figures_gradient/02_action_gradient_direction.svg) · [PDF](figures_gradient/02_action_gradient_direction.pdf)

The 10k and 15k gradient diagnostics, untrained-policy return panels, other per-action summaries and detailed raw arrays remain available in the authenticated results; they are not all pictured here. These plots are a curated suite of the major results, not an exhaustive display of every saved measurement.

## Rebuild and provenance

From `C:\Users\David\dev\BCA`, using the project environment:

```powershell
.venv\Scripts\python.exe -B analysis\central_quartile_review\policy_figures.py
.venv\Scripts\python.exe -B analysis\central_quartile_review\q_figures.py
.venv\Scripts\python.exe -B analysis\central_quartile_review\decision_figures.py
.venv\Scripts\python.exe -B analysis\central_quartile_review\gradient_figures.py
```

These commands perform no simulation, calibration or training. The evaluation figure scripts rerun the guarded extractor before plotting; it requires a successful terminal receipt, exact reproduction coverage, and matching hashes for all 144 contexts. The gradient script verifies the pin manifests and the 27 plotted diagnostic files. Plotted data and detailed source hashes are recorded alongside the figures where applicable. All PNGs are 300 dpi; SVG and PDF are vector exports.

The committed package is an analysis subset: it includes the 144 evaluation contexts, state banks, common reference, reports and saved gradient diagnostics. The training checkpoints, complete training run files and original simulator/data dependencies are not included. Consequently the analysis and figure rebuilds can run from this repository with Python, NumPy and Matplotlib, but the original training/evaluation launchers and their full input-integrity preflight cannot be rerun from this subset alone.

The study had two continuations and two evaluation attempts. The original deadline was waived; the first evaluation stopped after 143 contexts, while the completed rerun reproduced all 143 byte-for-byte and added the missing context. Both continuations were approved before outcomes were seen. Physical training totals remain lower bounds because interrupted work was not fully counted. Calibration reused development episodes adaptively and does not establish fresh conformal coverage under shift. See the [frozen protocol](../../experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/PROTOCOL.md), [terminal receipt](../../experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/RUN_RESULT.json), and [evaluation report](../../experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/run/evaluation/report.json).
