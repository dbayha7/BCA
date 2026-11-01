# Common-action / common-continuation figure guide

Source: `diagnostic_run_01/RESULT.json`, SHA-256 `fce3ac4458e9a7a02c605a80db09d61effcbf3f491824c0261846533b63426b2`.

Actors, critics and auxiliary model parameters remain frozen. Values compare forced first actions followed by a specified saved actor; they are not trained-policy returns or Q-star.
The off-diagonal critic/continuation combinations are cross-policy diagnostics, not that critic’s own-policy accuracy. Three markers are paired training seeds, not independent state/action rows.

1. **Common-action gain:** change in selected Qπ from symmetric Q1 BCA to guarded hybrid, for each critic/continuation pair; positive is better.
2. **Width and compatibility:** paired raw-to-symmetric and symmetric-to-hybrid changes on each critic’s own visited states with its own continuation.
3. **Continuation sensitivity:** for the same critic, states, actions and selected alternatives, hybrid gain under BCA continuation minus gain under native continuation.
4. **Choice changes:** counts of gains or losses greater than 0.1 on eight own-policy visited states per seed; near ties are labeled.
5. **Guard activation:** proportion of states where the extra compatibility cost is active; shared-state results are repeated for each paired seed.

Random-prefix, native-visited and BCA-visited sources are reported separately. A state-distance gate is not a support certificate. Each seed has 16 random-prefix and eight states from each actor; states and candidate actions are dependent.
The full all/guard-on/guard-off comparison data are in `paired_comparisons.csv`. The underlying RESULT retains all action-level values and scores.

Prespecified own-visited screen:
- ordinary: **inconclusive**; qualifying native 0/3, BCA 0/3; eligible native 0/3, BCA 0/3.
- strong_current: **inconclusive**; qualifying native 0/3, BCA 0/3; eligible native 0/3, BCA 0/3.
