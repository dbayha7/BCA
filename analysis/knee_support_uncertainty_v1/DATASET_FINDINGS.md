# Completed confirmation: what the learner actually received

This is a saved-data analysis of `knee_s10_candidate_v2/confirmation_v1`, completed on November 2, 2026 (EST). It is not Path A. No simulator, model, optimizer, calibration, or WSL execution was used. Original datasets, source, receipts, and decisions are unchanged.

**The two retained datasets are not identical.** Their selected source row IDs, commands, terminal flags, seeds, and masks match, but their states, rewards, successors, and fitted normalization differ. This is not an ordering difference: a joint transition multiset comparison gives the same conclusion.

| Actual learner stream | Rows per world | Joint rows different between worlds | Joint rows shared exactly |
|---|---:|---:|---:|
| CUTOFF training | 47,978 | 46,532 (96.986%) | 1,446 |
| Calibration (`heldout`) | 128 | 122 (95.313%) | 6 |
| Training gradient probes | 768 | 746 | 22 |
| Local-scale bank | 256 | 251 | 5 |
| Normalization fixture | 34,085 | 32,927 | 1,158 |

In each stream, actions and done flags are identical. Source observations, rewards, and next observations differ on the reported rows. The learner receives float32 `(obs, action, reward, next_obs, done)` with scalar reward/done shapes `[N]`. Training has 47,978 retained rows out of 76,800 fitting transitions (62.471%). The transition multiset comparison preserves field identity, dtype, shape, and duplicate multiplicity, not just independent field histograms.

The maximum training observation difference is 1.65241295099; the maximum float32 retained reward difference is 8.90413379669. The larger 9.37894552049 reward difference from the initial inspection concerned **all fitting rows**, not only retained rows.

## Why equal commands did not produce equal retained data

The 768 fitting episodes have exactly equal initial states and episode seeds. Recorded requested/executed commands, behavior-component labels, load bands, behavior log densities, and split/group/trajectory identities also match. The unchanged exogenous current/wave coordinates are state indices 10–12; differences occur in indices 0–9. Time coordinates match.

Every one of the 46,532 differing retained source states has an earlier removed `|a0| > float32(0.8)` command in its original episode. No retained state differs without such a preceding command, and all 46,532 prior-exposed retained rows differ. The same association holds for calibration: 122 prior-exposed rows, 122 differing source/successor states, and zero differences without prior exposure.

**Consequences of earlier removed transitions remain in later retained states.** The removed action/reward tuples themselves are still excluded. This is neither evaluation-label leakage nor evidence that the original unsupported transition labels reached training. It means post hoc transition filtering did not erase the source trajectory's history. This association is measured from saved arrays; no counterfactual dynamics replay was performed. Whether a learner can infer the world or exploit these differences is not established here.

The recorder source uses the same named reset/action draws in both worlds and conditions commands on load. Identical realized commands and contexts do not prove that every underlying state/action occupancy law is identical. The worlds' physics metadata explicitly differ in the `stall` flag.

## Exact binding and preprocessing

`dataset_comparison.py` independently reproduces the pure NumPy selectors in `knee_edge_ood_v3/masks.py` and `bindings.py`. It checks all five resulting stream SHA256 values against **all 30 effective completed training receipts**, including 13 reused from attempt 2. Each world's 15 receipts agrees. All consulted reconstruction sources are authenticated by an executed job's dependency closure; the upstream UniFloral source is separately authenticated through the pinned `native_loader.py` source-hash literal, as in production.

- The CUTOFF predicate is strict exclusion of executed float32 `|a0| > 0.800000011920929`. No withheld action is present in any learner stream.
- Calibration uses one eligible row from each of the first 128 tuning episodes, independently of outcomes. Both worlds replace the same 54 originally ineligible times with eligible times from the same episode. Fitting and calibration episodes are disjoint.
- Normalization uses the within-world CUTOFF/RANDOM retained intersection. RANDOM contributes only to this normalization-selection rule in this confirmation; it is not a separately trained comparison arm here.
- The saved normalization source string saying “identical all conditions” is inherited from an older study. It must not be read as identical **across worlds**. Actual mean and std differ in the first 10 of 14 coordinates; the maximum mean difference is 0.2780889794 and std difference 0.0498521328. All native/BCA arms and seeds within a world share the exact same recorded normalization.
- Applying the model's `(obs - mean)/(std + 0.001)` expression with each world's saved float32 statistics produces different normalized observations at all 47,978 training rows, including initially equal raw states. This is an algebraic NumPy reconstruction, not a new GPU arithmetic-parity claim.
- Successors come from the original same-episode `observations[e,t+1]`, not the next retained row. No next-action field is manufactured or consumed by this TD3+BC input. Finite-horizon done semantics are unchanged.

Both complete NPZ archives are decoded to recompute their pinned canonical content hashes. Protected selection/final-calibration arrays are included only in that authentication, not in numerical comparisons, parameter fitting, or outcome selection. Unused tuning episodes are not selected into learner streams.

## Reproducible artifacts and checks

Canonical output: `output/dataset_final/dataset_summary.json`, SHA256 `4464f75ebdd0d7fde1c216e88c3e43d24e76e04a0d67c4ed0a4fe228b929e80d`. It contains complete input/source hashes, per-field row and coordinate differences, joint multiset comparisons, seeds, histories, stream binding hashes, and recorded normalization. `output/dataset_final/source_rows.npz` contains only exact `(episode,t)` membership for all five streams in both worlds; its file/content hashes are recorded in the JSON. The earlier `output/dataset/` analysis is preserved; `dataset_final` adds explicit requested-command/initial-state comparisons and the recorder source pin without changing findings.

Dataset identities:

| World | NPZ SHA256 | Canonical content SHA256 |
|---|---|---|
| Stall | `125131d77f54600497ce4a5d95abff885c1cefbf8ccbafd2d74fd4d52da4bdb2` | `c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533` |
| No-stall | `5f4933fc5d687fc0b6f60e85c96809e7f0ace7023b91209ab89a9e373111b71e` | `e3549869af13b367cd726d008935b3eb466e38c005d1d5b7856faae3397e572a` |

CPU synthetic checks: **9 passed** in 0.156 seconds. They cover permutation versus changed values, joint-row associations, duplicate multiplicity, dtype/shape/signed-zero distinctions, exact successor provenance, saved normalization, content-hash sensitivity, and disjoint calibration selection at the float32 knee boundary. They do not simulate or import experiment modules.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s analysis/knee_support_uncertainty_v1 -p dataset_tests.py -v
.\.venv\Scripts\python.exe -B analysis/knee_support_uncertainty_v1/dataset_comparison.py --output-dir analysis/knee_support_uncertainty_v1/output/dataset_recheck
```

The output directory must be new and inside this analysis directory; existing results are never overwritten. The analyzer verifies all consumed input hashes again before writing.

Skills actually read: repository `.agents/skills/offline-rl-experiment-design/SKILL.md`, `.agents/skills/bca-code-and-calibration-audit/SKILL.md`, installed `C:/Users/David/.codex/skills/offline-rl-data-audit/SKILL.md`, and its `references/data-contract.md`. The comparison applies their input-provenance, scalar-shape, original-successor, disjoint-role, and training-only normalization checks. It makes no claim that structural validity proves support adequacy, policy learnability, or BCA efficacy.
