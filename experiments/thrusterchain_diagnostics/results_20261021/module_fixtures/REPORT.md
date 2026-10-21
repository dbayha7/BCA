# BCA/IWBCA deterministic module verification

The authoritative expanded run is `verification_v2/fixture_results.json`: **20 behavior-reproduction checks passed; 18 scientific contracts passed and 2 scientific contracts failed.** Its exit code is intentionally 1 because contract failures must remain visible. The older root-level `fixture_results.json` contains the initial 18-case run and is retained unchanged for provenance; it is not the complete result.

This run executes actual native calibration/reference/dose functions and existing IW functions. It trains no model, samples no environment trajectory, and edits no canonical source or frozen experiment output. It ran on Zeus in 4.543 seconds of recorded runner time, with affinity mask 240 (cores 4–7), IDLE priority, CPU JAX, and numerical thread variables set to 1. Source HEAD was `b007052a2494dc651ac00469fa9e922b287cd1e5`; individual file hashes, rather than HEAD alone, identify the tested working-tree source.

## Sequential verification

| Stage | Premise / independent expectation | Observed status and practical takeaway |
|---|---|---|
| W1 ratios | For the named truncated-exponential laws, ratio is `exp(4*(c-.6))`; independently integrate target squared over source. Identical empirical classifier covariates with class counts 26/52 require a prior correction. | PASS. Known ratios and query mean match; fitted finite-design weights equal 1 after the class-prior correction. This is not learned-ratio generalization evidence. |
| W2 normalization | Multiply bank and query masses by the same positive constant. | PASS. Crossings and ESS are invariant. Dividing bank weights alone changes the problem: the deliberately incorrect control moves R=1 to infinity for seeds 7, 19, 20261021. |
| W3 query mass / support | A source-removed query has infinite ratio; a finite query can separately cause posterior refusal. | PASS. Actual exact-ratio helper reports infinity on the supplied removed set, and native calibration rejects infinite query mass. Hand-calculated exponential draws reproduce finite and infinite crossings. |
| W4/W5 ESS / clipping | Kish ESS is `(sum w)^2/sum(w^2)`; concentration should reduce it. | PASS. Weights `[.01,1,100]` give ESS about 1.02, versus 3 for unit weights. The tested native path does not clip ratios. A manual cap is explicitly a changed-input sensitivity control. |
| N4 frozen scores | Scores are absolute residual divided by `max(eta,1e-6)*unit`. | PASS. Actual reference freezing gives the intended thresholds and half-widths for three controlled residual/unit combinations. Scale-network learning is untested. |
| N5 threshold | Directly enumerate candidate tail risks, including the unseen query atom, then select the beta order statistic. | PASS. Independent arithmetic matches the actual posterior crossings and threshold for fixed exponentials. It is a numerical contract test, not a coverage theorem. |
| W6 per-row query lookup | Coupled draws make thresholds monotone in query mass; lookup rounds upward. | PASS on the recorded grid; the high query mass yields infinity. No running-max repair was needed. |
| N6 half-width | `U=R*max(eta,1e-6)*unit`; full interval width would be `2U`. | PASS. Measured half-widths `.1,10,10` distinguish normalized threshold from response units. |
| N7 bounded dose | Finite `dose=1+.5*U/(U+unit)` is increasing and lies in `[1,1.5]`. | PASS. Widths `[0,1,4]` give `[1,1.25,1.4]`. Native infinity is a separate refusal branch with dose 1. |
| W7 host refusal | Compare the same thresholds under different explicit policies. | PASS. At infinity, native dose is 1 and IW `most_caution` dose is 1.5. IQL uses different action-weight semantics. Unit weights do not remove this refusal-policy difference. |
| N8 coefficient / alignment | The coefficient is detached; usefulness requires separate oracle evidence. | PASS for zero dose gradients. An analytic oracle gives +1 or -1 error/harm rank correlation depending on deliberately supplied eta. Arithmetic correctness alone does not establish useful caution. Actual host optimizer updates are untested. |
| N9 invalid refresh | A zero unit cannot produce a ready reference. | PASS at the helper boundary. Preservation of an earlier reference by the caller is source-audited separately, not dynamically exercised here. |
| E1 / audit I11 exploratory command density | A clipped target distribution must account for its singular boundary mass under the stated continuous law. | **CONTRACT FAIL in two cases; harness PASS in both.** Details below. |
| N1/N2/N3 and population coverage | Reservation, Bellman target, fitted scale, whole host update, held-out population coverage require different evidence. | **UNTESTED here.** No certification claim follows from these fixtures. |

`verification_v2/stage_map.json` supplies labeled pseudocode with premises, inputs, outputs, purposes, invariants, fixture IDs, source ranges and statuses. `verification_v2/source_excerpts.md` contains the actual line-numbered function bodies. In particular, `calibration/wbcp.py:43–116` separates common rescaling, empirical threshold, posterior query atom and ESS; `calibration/reference.py:152–194` constructs the frozen scores; `calibration/dose.py:23–95` constructs the half-width/dose and neutral native fallback. The IW refusal branches are in `experiments/shift_hosts/iw.py:279–364`.

## Consistent and exceptional penalty cases

IID is a statement about source/target laws, not residual magnitude. With eta=1 and unit=1, changing the IID residual from .1 to 10 changes the measured dose from about 1.04545 to 1.45455. With residual 10 and unit 100, the half-width is still 10 but the normalized dose returns to about 1.04545. This isolates the scale premise behind a claim of “low IID penalty.”

IW need not always increase caution. Upweighting the high-residual portion moves R from 1 to 10; favoring the low-residual portion moves it from 10 to 1. Both directions reproduce with seeds 7, 19 and 20261021. These are supplied-weight controls, not evidence that a trained ratio is correct. An apparent increase at refusal can instead arise entirely from changing the fallback from native 1 to `most_caution` 1.5.

The analytic alignment case uses actions -1,0,1, true Q=-a² and Qhat=0. It compares dose against absolute Q error and action disadvantage, not policy improvement, physical stall harm, the complete actor loss, or its gradient. Reversing the supplied eta reverses the correlation. This is an intentionally constructed limitation, not a trained-agent failure measurement. The parent's separate frozen-run alignment analysis contains the empirical measurements.

## Two exploratory scientific failures

The source is `outputs/shift_iw/explore/channel/command_band_check.py`, SHA-256 `3c4d4dfa41ee804a3d2e88f556ae7a7f835ae8a01d7b972d9944e389ec148ca0`. The runner compiles exact AST function/class bodies and constants from this file, bypassing its experiment driver, imports and expensive constructors. The supplied finite lookup table isolates the ratio-to-refusal path; this is not an end-to-end experiment rerun.

The sampler at lines 261–262 clips N(0,.2²) noise to ±.5 and clips the resulting action to ±1. The formula at lines 80–85 represents only the interior Gaussian density. The query path at lines 130–136 and 160–172 then exhibits:

| Fixture | Independent continuous-law expectation | Actual source behavior |
|---|---|---|
| `exploratory_noise_clip_atom` | Centre 0 creates target coordinate mass P(Z≥2.5)=0.0062096653 at action coordinate .5. With eps=1 the source uniform continuous law gives zero mass to that coordinate hyperplane; a finite density-ratio representation cannot cover it. | At `[.5,0,0,0]`, spread log density and log ratio are -infinity. `Thresholds.read` replaces this by its floor and returns a finite threshold with `refused=False`. |
| `exploratory_action_clip_atom` | Centre .8 creates target coordinate mass P(Z≥1)=0.1586552539 at action coordinate 1. Again the stated source continuous law assigns that hyperplane zero mass. | At `[1,0,0,0]`, the ordinary finite log ratio is 5.0345862392, and the lookup does not refuse. |

The event is a whole coordinate hyperplane, not a positive-probability isolated four-dimensional point. The source has a positive *density value* there but zero continuous-law event probability; those are different quantities. These proper-refusal expectations apply to the continuous source law represented by the formulas. Actual actions are rounded to float32, so an exact executable-law analysis would instead need consistent discrete lattice probabilities for source and target. The smooth density formula is not that calculation either. No exact zero-support claim about the fully quantized executable law is made here.

This is a defect in the exploratory command-weighted density claim. It does not establish a defect in native BCA, core current-law IW, or the frozen 30-run suite. No canonical source fix was attempted. Reproduction passes must not be summarized as all scientific checks passing.

## Reproduction and portability

Run `commands.ps1` using the existing Zeus interpreter. It selects a new output directory, so the delivered evidence is preserved. `--repo` accepts a source bundle root with the same relative layout; Git and `.git` are optional, and a missing repository HEAD is recorded as null. An ancestor workspace's HEAD is not attributed to a source-only bundle. Individual file hashes remain authoritative. The executable is intentionally Windows/Zeus constrained and must find CPU indices 4–7.

`verification_v2/runner_used.py` preserves the exact runner whose hash is recorded in that result's source closure. The current `run_fixtures.py` has one later provenance-only correction: missing Git and an unrelated ancestor Git repository now produce null HEAD. The numeric fixtures are unchanged. The final artifact manifest distinguishes both scripts. The expanded scientific run was not repeated merely for that provenance correction.

The exact source closure is `verification_v2/source_closure.json`: 37 actual source files including the exploratory source slice, with hashes and runtime versions. No datasets, checkpoints or frozen experiment records are needed. Python was 3.10.11; numpy 2.2.6, scipy 1.15.3, JAX/JAXLIB .6.2, flax .10.7, optax .2.8 and chex .1.90. Actual `experiments.shift` enforces gymnasium 1.0.0 from the dedicated dependency directory, here `C:\Users\David\.cache\bca-shift\py`. A different compatible directory can be selected with `BCA_SHIFT_DEPS`; it must obey that original package's whitelist. This run installed nothing. Orbax distribution metadata was unavailable and is recorded as null, not guessed.

Every case records its full fixed inputs or deterministic construction, seeds/streams, expected and observed values, interpretation and source references. Infinity is represented by a JSON string. `run_log.txt` records the executed command, result and deliberate nonzero exit. Coverage outside the explicit arithmetic/analytic examples remains uncertified. The remaining work is scientific repair and independent validation of the exploratory mixed-measure ratio, not a training rerun.
