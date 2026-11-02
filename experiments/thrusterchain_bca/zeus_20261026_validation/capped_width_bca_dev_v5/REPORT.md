# Paired auxiliary refits and capped-width controller evaluation

**Result: both bounded components completed, but capped width failed its prespecified strong-current screen.** Only one of three original-BCA checkpoints and none of three native checkpoints improved by at least one discounted-return unit against *both* raw Q1 and symmetric BCA. The capped controller did improve over its own saved actor in all six seeds in each context. Raw-Q1 and symmetric reranking also improved substantially over those saved actors, so the actor comparison alone does not establish that the cap helped.

The paired auxiliary refits show appreciable variability in the action-dependent width. Under strong current on the previously used actor-visited banks, the across-refit standard deviation of the paired native/BCA contrast exceeded the RMS paired-mean contrast in all three seeds. With only four refits per checkpoint, this is evidence that the original differences are sensitive to the auxiliary fit; it does not identify the causal effect of BCA policy training or exclude a systematic checkpoint effect.

These are exploratory results for six frozen 20,000-update hole-condition checkpoints, one dataset, and 16 shared fresh starts per context. The rerankers were evaluated online as fixed controllers; no policy was trained or updated. A first-action reference follows the corresponding saved actor, so it is not a complete reranker return. The earlier diagnostics, auxiliary fits, and calibration episodes were already development-used.

## What was fixed before the fresh evaluation

The new score centers the original symmetric width at the saved actor action and clips that action contrast:

`Q1(s,a) - clip(3 * (U(s,a) - U(s,a_actor)), -cap_t, cap_t)`.

The four per-checkpoint time-bin caps were the median strictly positive top-two raw-Q1 gaps among distinct actions on designated retained training probes. Caps were persisted before fresh starts. The probes were in sample for the critic and auxiliary fits. Centering without clipping adds a statewise constant and cannot change ranking; clipping is the actual scoring change. The result is not the original calibrated bound and provides no safety or shift-coverage guarantee.

Five fixed controllers were compared on a 15-slot actor-centered bank: the saved actor, raw Q1, symmetric `Q1 - 3U`, the capped score, and the previously specified guarded compatibility hybrid. Duplicate slots were retained for controller selection; caps used distinct full action vectors. The compatibility guard is a training-distance heuristic, not a support certificate. Controllers started from identical physical states within a context but can visit different states after choosing their first actions.

The three saved training seeds were paired across native and BCA checkpoints. The strong-current success rule required at least two seeds in **each** checkpoint group to achieve capped-minus-raw and capped-minus-symmetric mean returns of at least +1, as well as no ordinary-current capped-minus-actor loss larger than 1 for any seed. This is a practical engineering screen, not a significance test.

## Complete-episode returns and the prespecified screen

Each entry below is the mean discounted 100-step return over 16 paired fresh starts; higher is better. The means are conditional on these starts and checkpoints. See [all complete returns](plots_v2/02_complete_returns.pdf), [paired changes versus the three rerankers](plots_v2/01_paired_return_gains.pdf), and [the separate comparison with the saved actor](plots_v2/06_capped_vs_actor.pdf). The underlying 960 episode values are in [episode_values.csv](plots_v2/episode_values.csv).

| Context | Checkpoint | Seed | Saved actor | Raw Q1 | Symmetric | Capped | Guarded hybrid |
|---|---|---:|---:|---:|---:|---:|---:|
| Ordinary | Native | 0 | -65.107 | -54.711 | -55.440 | -55.165 | -55.621 |
| Ordinary | Native | 1 | -73.206 | -60.709 | -61.982 | -62.005 | -63.297 |
| Ordinary | Native | 2 | -79.418 | -63.534 | -62.271 | -62.384 | -63.142 |
| Ordinary | BCA | 0 | -62.893 | -53.293 | -52.127 | -52.552 | -52.255 |
| Ordinary | BCA | 1 | -73.071 | -60.745 | -59.862 | -59.557 | -60.088 |
| Ordinary | BCA | 2 | -79.053 | -62.630 | -60.988 | -60.873 | -60.702 |
| Strong | Native | 0 | -179.234 | -146.468 | -146.802 | -146.129 | -146.552 |
| Strong | Native | 1 | -177.668 | -147.673 | -146.405 | -146.792 | -145.914 |
| Strong | Native | 2 | -205.184 | -156.781 | -158.753 | -159.291 | -157.879 |
| Strong | BCA | 0 | -172.693 | -143.222 | -142.899 | -141.684 | -143.067 |
| Strong | BCA | 1 | -174.484 | -143.623 | -142.803 | -143.263 | -143.129 |
| Strong | BCA | 2 | -201.312 | -154.372 | -155.183 | -154.846 | -153.853 |

The actual screen compares paired means before rounding:

| Checkpoint | Seed | Strong: capped − raw | Strong: capped − symmetric | Ordinary: capped − actor | Strong qualifies? |
|---|---:|---:|---:|---:|---|
| Native | 0 | +0.339 | +0.673 | +9.942 | No |
| Native | 1 | +0.881 | -0.388 | +11.202 | No |
| Native | 2 | -2.509 | -0.538 | +17.033 | No |
| BCA | 0 | +1.537 | +1.215 | +10.342 | Yes |
| BCA | 1 | +0.360 | -0.459 | +13.513 | No |
| BCA | 2 | -0.474 | +0.337 | +18.180 | No |

Ordinary nonharm against the saved actor held in all six cells, but strong-current qualifying counts were **0/3 native and 1/3 BCA**, so the overall screen **failed**. For strong-current native seed 0, the capped mean beat raw by +0.339 even though only 6 of 16 paired episodes improved by more than 0.1 and 10 were harmed by more than 0.1; larger positive changes outweighed more frequent losses. Native seed 2 had 1 improved and 15 harmed episodes relative to raw, with a worst paired loss of -8.898. This illustrates why the reported mean, per-start values, and lower tail should be read together. Per-episode improvements and harms here concern **full returns** and use a descriptive ±0.1 tolerance; they are not the prespecified gate.

| Strong-current checkpoint | Seed | Capped vs raw: improved / near-tied / harmed | 10th percentile of paired gain | Worst paired gain |
|---|---:|---:|---:|---:|
| Native | 0 | 6 / 0 / 10 | -3.067 | -3.736 |
| Native | 1 | 8 / 2 / 6 | -1.854 | -5.919 |
| Native | 2 | 1 / 0 / 15 | -5.660 | -8.898 |
| BCA | 0 | 11 / 1 / 4 | -1.585 | -3.119 |
| BCA | 1 | 7 / 2 / 7 | -3.461 | -4.053 |
| BCA | 2 | 5 / 3 / 8 | -4.347 | -5.363 |

The percentile is the linearly interpolated empirical 10th percentile of just 16 paired returns; it is descriptive, not a population tail estimate or confidence interval.

## First-action decisions and controller activity

The [first-action figure](plots_v2/03_first_action_changes.pdf) compares the selected first action followed by the same saved actor. Under strong current, capped versus raw changed the action on 11 of 48 native checkpoint/start comparisons: **6 improved and 5 harmed** by more than 0.1 in the saved-actor continuation. For BCA checkpoints it changed 10 of 48: **3 improved, 6 harmed, and 1 within ±0.1**. These pool correlated states across only three trained seeds and should not be treated as 48 independent training replications.

Compared with symmetric reranking, capped changed the first action only once in the native group, where it harmed the reference, and once in the BCA group, where it improved it. The remaining first actions were unchanged. Their complete episodes can nevertheless diverge later, because each fixed controller reranks at every state it reaches. First-action Q-pi is neither the reranker's own continuation value nor Q-star. See the per-seed [first-action table](plots_v2/first_action_comparisons.csv).

The [activity figure](plots_v2/04_cap_and_guard_activity.pdf) uses 1,600 decisions per checkpoint/context/controller. Under strong current the selected action's penalty was clipped on **727/4,800** native and **1,266/4,800** BCA capped-controller decisions. This is a frequency on the capped controller's visited states, not a causal explanation of its performance. The guarded hybrid's guard was active on approximately 79–80% of its own strong-current decisions and almost all ordinary-current decisions. Its trajectory is separate from the capped trajectory after actions differ.

## Matched auxiliary refits

Four fresh auxiliary fits were made per checkpoint. Within a training-seed/replicate pair the native and BCA fits shared initialization, 2,000 minibatches, per-step target-noise/bootstrap keys, and the calibration key. Each still had its own frozen actor, critic and residual targets. All 24 fits were refreshed using the **same 64 previously consumed v3 calibration episodes**, not newly collected calibration data. The saved v4 common-action banks below were also previously inspected.

For each of the 16 pooled actor-visited states in a seed/context, define the `3U` action contrast relative to the shared bank's native-actor slot. The paired contrast is BCA minus native. The following are RMS values across those 16 states and all 15 slots, including duplicates; the columns describe distinct quantities. See [paired-refit variability](plots_v2/05_paired_refit_variability.pdf) and its [source table](plots_v2/refit_contrast_variability.csv).

| Context | Seed | RMS paired-mean contrast | RMS SD of paired difference across refits | Native within-checkpoint RMS SD | BCA within-checkpoint RMS SD |
|---|---:|---:|---:|---:|---:|
| Ordinary | 0 | 0.0531 | 0.1078 | 0.1968 | 0.2011 |
| Ordinary | 1 | 0.0343 | 0.0858 | 0.1438 | 0.1510 |
| Ordinary | 2 | 0.0487 | 0.0938 | 0.1872 | 0.1809 |
| Strong | 0 | 0.0839 | 0.1222 | 0.2843 | 0.2720 |
| Strong | 1 | 0.0471 | 0.1082 | 0.2013 | 0.2259 |
| Strong | 2 | 0.0661 | 0.1191 | 0.2096 | 0.1801 |

On the same 48 strong-current actor-visited seed/state comparisons, the four paired-refit replicates produced **3, 4, 2, and 4 native/BCA winning-slot disagreements**, respectively. The previously analyzed separately randomized v3 widths produced 11/48. The fits use different keys, and slots can contain duplicate action vectors; this is a descriptive comparison on development data, not proof that matched randomness caused the reduction or that any one fit selects better actions. The paired differences fluctuate alongside sizable within-checkpoint variation. These results do not support attributing the original 11 disagreements solely to BCA training.

## Accounting, verification, and artifacts

- **Auxiliary refits:** 24 fits × 2,000 updates = **48,000 auxiliary updates**, 24 calibration refreshes, **23,040** saved-bank width rows, and zero simulator transitions. All corresponding minibatch and fitting-key trace hashes match within each native/BCA pair. The six checkpoint digests were checked before fitting and after each paired replicate. Elapsed time was **651.2 s**.
- **Reranker evaluation:** **96,000** complete-controller transitions, **96,000** first-action continuation transitions, and **104** simulator-parity transitions, totaling **192,104** completed simulator transitions. There were 960 complete episodes, 960 first-action references, **1,152,000** candidate-score rows, and 33 resets: 32 fresh starts plus one parity reset. The v3 references were reconstructed through 18 separately counted refreshes; there were **zero auxiliary, actor, critic, target, or other optimizer updates** in this component. Elapsed time was **467.6 s**.
- Simulator parity passed for state, reward, observation, time, termination and discount, including terminal bootstrap. Fresh reset seeds were audited against 2,261 previously used seeds; no protected transition rows were read. Each fresh initial state was shared across controllers and checkpoints, so subsequent trajectories are paired by start but not independent after they diverge.
- Both runs used CPU, JAX/JAXLIB 0.6.2, and highest JAX matmul precision. Simulator states and dynamics use float64; deployed actions, observations, and network inference use float32. The source seal and the before/after checkpoint state digests are recorded in the terminal results. An independent [post-run parity check](POST_RUN_PARITY.json) also verified exact agreement of all 192 original-actor first-action references with their full-episode returns, and of every planned first action with the executed action. These checks do not turn the exploratory design into independent confirmation.

Terminal evidence: [reranker result](reranker_run_01/RERANKER_RESULT.json), [paired-refit result](paired_refits_run_01/RESULT.json), [frozen caps](CAP_PROBES.json), [figure guide](plots_v2/FIGURE_GUIDE.md), and [plot receipt](plots_v2/PLOT_RECEIPT.json). Reported means use the terminal arrays rather than rounded cells.

The post-run [comparison manifest](comparison.json) and [metadata audit](COMPARISON_AUDIT.json) describe the five fixed controllers. The audit is **incomplete**, with 50 missing per-controller metadata entries and no flagged inconsistencies: fields such as inherited training batch/use counts and total historical online-selection budgets were not established for this manifest and remain null. This metadata check does not certify fairness or independent confirmation.

| Artifact | SHA-256 |
|---|---|
| Protocol | `5687f99e7db3bfcb26ca4f1c8d46366084ce21a0b07b86729f69ff37c7e39cf7` |
| Paired-refit source seal | `624ec0248abf4fb5dca626532eead95d242a3e3ddd5088b737063bdb160eae07` |
| Reranker source seal | `1a363ff33fc1bb0ef31e7eb415955b82c4415a203014190e1116216d1a4e0ca9` |
| Frozen cap metadata | `18de516cd315bdf1c66508034bb217e54cfdb4b22011cb7cfbca9dcdaa7bd42b` |
| Paired-refit terminal result | `a6c7c052fb84770d8a7059a76309def35a00a71ccfa47d5e930f182cf08f3dd2` |
| Reranker terminal result | `6a971cbc76833077156a673fa7b936a155cb186b4aeb6f392332608e5c7e77ac` |
| Plot receipt | `4876768c67b7b795e752a8438e878d2daad35afeff36306f16c2e5b181ad108b` |

The cap did not consistently improve returns over existing rerankers under the tested shift. The large improvement of all rerankers over the actor on these starts is a separate observation, as is the auxiliary fit sensitivity. Neither establishes a policy-training result, a safety guarantee, or generalization across datasets and independently trained seeds.
