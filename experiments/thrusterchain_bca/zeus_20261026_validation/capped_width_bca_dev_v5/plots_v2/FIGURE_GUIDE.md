# Frozen-controller evaluation: figure guide

Controller result SHA-256: `6a971cbc76833077156a673fa7b936a155cb186b4aeb6f392332608e5c7e77ac`.
Paired-refit result SHA-256: `a6c7c052fb84770d8a7059a76309def35a00a71ccfa47d5e930f182cf08f3dd2`.
Fresh-start artifact SHA-256: `a89e5cf6c3368fada646943f0bea97cc1e1397071cdc679d1e600aacd3feecae`.

The prespecified engineering screen **failed**. Qualifying strong-current seeds: native 0/3, BCA 1/3; ordinary-current nonharm for all seeds: True.

1. **Paired return gains:** per checkpoint and training seed, the mean over 16 matched fresh starts of capped-controller discounted full-episode return minus each reranking comparator. Positive is better. Each controller acts repeatedly on its own evolving trajectory.
2. **Complete returns:** each point is a checkpoint-seed mean across the same 16 starts. Higher is better, including when all returns are negative.
3. **First-action changes:** for the same initial state, compare capped to raw or symmetric. Each selected first action is followed by that checkpoint’s original actor. A changed action is improved/harmed only when the Qπ difference exceeds +0.1/−0.1; same actions are separate. This is not a full-controller return.
4. **Cap and guard activity:** fraction of 1,600 selected decisions per seed on the capped and guarded controllers, respectively. The two controllers follow different state distributions, so the two percentages do not describe the same states.
5. **Paired refit variability:** on actor-visited v4 development states, show RMS of the four-refit mean native/BCA contrast, RMS of the across-refit SD of that paired difference, and RMS across-refit SDs of native and BCA contrasts separately. Every 3U contrast is anchored at the native actor action; rows include duplicates and are dependent. These are not confidence intervals or a causal estimate.
6. **Capped versus actor:** the same mean paired return gain relative to the original actor, shown on its own scale so the smaller differences among rerankers remain readable in Figure 1.

All five controllers and six checkpoint actors and critics are frozen. The score cap is computed from in-sample retained training probes and is not a calibrated or safety bound. Qπ uses the saved actor continuation, not the repeated reranker. The 16 starts are shared across checkpoints and methods; only three paired training seeds and one dataset are available. No uncertainty bars or statistical-significance claims are implied.
Each underlying row, matched reset seed, first action, episode return and first-action Qπ is in `episode_values.csv`; `first_action_comparisons.csv` contains the count and mean-gain summary. `refit_contrast_variability.csv` contains all four plotted refit quantities. Full step telemetry and model artifacts are retained in the source run directories.
