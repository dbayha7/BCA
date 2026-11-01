# Expanded action-bank diagnostic: figure guide

Source: `diagnostic_run_01/RESULT.json`, SHA-256 `3b607c1c810d75477ccc927f23000d2649d5f7585ad65838cc07745e67698387`.

The actor, critic and target are frozen. Points are three checkpoint seeds. Action pairs and shared anchor states are dependent; results do not establish trained-policy improvement.

1. **Regret:** each selected action is compared with the best of the 15 candidate vectors, not a global optimum.
2. **Pairwise accuracy:** actor versus coordinate perturbations with true Qπ gap >0.1; ties in scores are counted incorrect. Each point may have a different number of informative dependent action pairs.
3. **Choice changes:** compares symmetric Q1 BCA plus guarded compatibility with raw Q1; improvements and harms require Qπ difference beyond ±0.1.
4. **Fresh check and matching:** one-step TD prediction on 64 new behavior episodes; coefficients matched on training probes; state-distance is a heuristic, not a support certificate. Scaled signed scores have no original coverage claim.

Each context has 16 natural and 16 separately selected states. The ambiguity-selected stream is conditional on a Q1 rule and is not a natural-start estimate. Full per-method counts and metrics are in `all_metrics.csv`.

Prespecified exploratory screen:
- ordinary, natural: **failed**; qualifying native 0/3, BCA 0/3.
- ordinary, ambiguous_q1: **failed**; qualifying native 0/3, BCA 0/3.
- strong_current, natural: **failed**; qualifying native 0/3, BCA 2/3.
- strong_current, ambiguous_q1: **failed**; qualifying native 0/3, BCA 0/3.

No actor, critic, actor-target, critic-target, or auxiliary optimizer updates were performed.
