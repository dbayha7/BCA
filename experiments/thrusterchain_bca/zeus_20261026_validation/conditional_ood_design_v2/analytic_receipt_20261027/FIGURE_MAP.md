# Figure map: REVIEW_20261027.md and DECISION_BLOCK.md against the saved analytic scripts

Date: 27 October 2026. This map ties every analytic number in `../REVIEW_20261027.md` and the analytic references in
`../DECISION_BLOCK.md` to a saved script and its rerun output in this folder. An independent read-only check
confirmed each row against the outputs, and its corrections are applied here.

## How the scripts were run

- **Copies:** the 16 scripts in `scripts/` are byte-identical copies of the session scratchpad originals.
- **Interpreter:** each ran under the repository's `.venv` (Python 3.10.11, NumPy 2.2.6), from the repository root.
- **Two runs.**
  - The first run used `-I -B`. Its outputs are in commit ebc3283f.
  - The second run added `-X utf8`. Its changed files (`run_all.py`, the `cal.py` outputs and `RECEIPT.json`) are in
    the GitHub Desktop stash on the original branch.
- **Determinism:** every script seeds its random generator. Standard output was byte-identical across the two runs
  for 15 of 16 scripts. The exception, `cal.py`, crashed on the first run. It printed a "∩" character the Windows
  console code page could not encode, and the `-X utf8` flag fixed that. No numerical code changed.
- **Inputs:** `environment.json` and `results.json` (the site's extracted data), the one-sided `data_receipt.json`
  (normalization and calibration identity), `bindings.json`, and the support-shift retry's `data_receipt.json`.
- **Dataset file:** `coverage_check.py` also hashes the dataset file and the v1 source-receipt files. Hashing reads
  bytes and decodes no arrays.
- **Nothing else was touched.** No simulator package, model, training code, dataset array or protected split was
  read.
- **Hashes:** `RECEIPT.json` has SHA-256 hashes of every script, input and output, with exit codes and run times.

**Labels used below.**
- **Analytic:** a closed-form or Monte Carlo expectation from the documented recorder and start law. It is not a
  measured count.
- **Model:** the figure comes from the linear-Gaussian state approximation, or the unconstrained linear-quadratic
  controller, on the frozen plant. That is an approximation, not a full Q^pi value.
- **Derived:** hand arithmetic on printed values or on stated design counts.

## Figures in REVIEW_20261027.md

### Receipt checks and support radius

| Review statement | Rerun value | Script and output line | Status |
|---|---|---|---|
| The v1 draft and its 15 source-receipt hashes, plus the dataset hash, recompute | 15 listed, 0 mismatches; dataset and draft hashes match | `coverage_check`: "files listed: 15 mismatches: 0", "dataset sha256 matches: True", "draft sha256 matches: True" | Reproduced |
| Knee k = 0.800000011920929 | as stated | `coverage_check` physics line | Reproduced |
| Gaussian state model matches receipt standard deviations within about 2% | ratios 0.980 to 1.005 on all 13 coordinates | `verify_findings`: covariance table | Reproduced (model) |
| About 1 to 2 same-bin, same-side rows within distance 0.5 | 1.4 to 2.3 expected rows of 3,840 | `indep_check`: "bin 0/2/5/9 … (0.5, …)" gives 1.42 to 2.08. `geom2`: "T=25 same-side P(RMS<= 0.5) = 5.99e-04" gives 2.30 | Reproduced (model) |
| A radius of about 0.67 to 0.77 gives the intended counts (main text) | 0.670 to 0.773 for 8 episodes, before and after halving the pool | `verify_findings`: "radius for E[8 of 767]", "after halving pool" | Reproduced (model) |
| Analytic table: "Radius giving 32 to 64 such rows: about 0.67 to 0.77" | 0.663 to 0.676 for 32 rows; 0.716 to 0.732 for 64 rows | `indep_check`: "q32", "q64"; `geom2`: "quantile 32/3840: 0.665/0.672", "quantile 64/3840: 0.718/0.724" | **Misstated.** About 0.66 to 0.73 gives 32 to 64 rows. The 0.77 end is the 8-episode radius after halving the pool |

### Value gap near the band

| Review statement | Rerun value | Script and output line | Status |
|---|---|---|---|
| Linear-quadratic curvature 0.672 per unit squared | 0.6723 | `verify_geometry`: "S_0[0,0] curvature"; `lq2`: "actor-side curvature" | Reproduced (model) |
| Moving 0.05 from the optimum costs about 0.002 units | 0.00168 | same lines | Reproduced (model) |
| The 0.1-wide band "spans about 0.13 units near the optimum" | The span grows as 0.1345 × distance from the optimum: under 0.01 when the band contains the optimum, 0.134 at one unit away | `verify_geometry`: "slope one unit off optimum (2*S00): 1.3446" | **Misstated** |
| Value spread across the band and its two neighbors averages about 1 to 1.5 units | 0.86 (thruster-0 proportional continuation) and 1.47 (zero continuation) over a0 ∈ {−0.9, −0.85, −0.8, −0.75, −0.7} | `verify_findings`: "spread over 5 band/ring values" | Reproduced (model). A wider set that adds a0 = −0.6, beyond the adjacent band, gives 1.29 and 1.76 in `cond_ood_arith` |
| The sign of that spread depends on the continuation | Mean v(−0.8) − v(−0.7): +0.65 (zero), +0.06 (thruster-0 proportional), +0.51 (zero, second implementation), −0.27 (all-thruster proportional) | `verify_findings`; `cond_ood_arith` "continuation=pd" | Reproduced (model). The negative mean appears only for the all-thruster rule |
| Full-data 20k critic misjudges in-band commands at g ≥ 0.4 by +19 to +82 units | 25 in-band probes at g ≥ 0.4: 24 from +19.5 to +82.0, one at −6.5 (development batch 2, group 66, t = 50) | `verify_geometry`: "in-band at g>=0.4: n 25 … min/max: -6.54 81.96 negatives: 1"; `bank` listing | **Corrected:** "+19 to +82, except one probe at −6.5" |

### Load threshold and depth

| Review statement | Rerun value | Script and output line | Status |
|---|---|---|---|
| P(g ≥ 0.4) is exactly 0.5 at every step | 0.5000 by quadrature; per-step marginal 0.4996 to 0.5008 by sampling | `prob`, `arith`, `coverage_check`, `verify_geometry` | Reproduced (analytic) |
| About 29% of high-load rows have g < 0.5 | 29.2% | `indep_check`: "among high: … P(g<.5) 0.2918" | Reproduced (analytic) |
| A 0.1 change in g moves a state about 0.15 in standardized distance | 0.157 through current, 0.147 through wave 1 | receipt standard deviations printed by `verify_findings`: 0.1/0.1767/√13 and 0.1/0.1886/√13 | Reproduced (derived) |

### The knee and counter-thrust

| Review statement | Rerun value | Script and output line | Status |
|---|---|---|---|
| Cancelling a load of 0.4 needs thruster-1 thrust of about −0.32 jointly or −0.40 alone | Least-squares counter per unit g (−0.8, −0.4, −0.4, −0.8), so −0.32 at g = 0.4. Thruster 1 pushes 0.3 per unit of thrust against a water push of 0.3 g, so −0.40 alone | `verify_geometry`: "LS counter per unit g", "water push per unit g … 0.3" | Reproduced (derived) |
| The knee matches the load only near g = 0.8 to 1.0 | −1.0 per unit g alone and −0.8 per unit g jointly reach −k at g = 0.8 and g = 1.0 | derived from the line above | Reproduced (derived) |
| …which covers about 0.4% to 6% of recorded steps | P(g ≥ 1.0) = 0.35%, P(g ≥ 0.8) = 6.2% | `prob` | Reproduced (analytic) |
| The box-pinned optimum sits at −k for roughly a third to a half of high-load states | 0.28 to 0.56 when float terms are an independent Gaussian (`verify_geometry`, `lq2`). 0.54 to 0.87 at recorded current, P(a0* ≤ −k) among high-load states, when states are propagated (`indep_check`) | as named | **Model-dependent.** The review quoted only the simpler model |
| Development 20k actor's first command stayed between −0.48 and +0.52, nearest 0.32 from the band center | −0.483 to +0.521; 0 of 128 within 0.05; nearest 0.317 from the center (0.267 from the band edge) | `verify_geometry`, `bank` | Reproduced |

### Coverage expectations

| Review statement | Rerun value | Script and output line | Status |
|---|---|---|---|
| Expected hole rows: about 1,920 (2.5%) | 1,920 | `arith`, `coverage_check` | Reproduced (analytic) |
| Training episodes never reaching g ≥ 0.4: about 11% | 11.1% | `prob`: "P(episode always g<0.4) = 0.1111" | Reproduced (analytic) |
| Training episodes with no hole row: about 21% | 21.1% | `coverage_check`: "P(an episode has no hole row) = 0.211" | Reproduced (analytic) |
| Retained in-band low-load rows: about 1,920 | 1,920 | `cond_ood_arith`: "E[band rows at LOW load, retained] train=1920" | Reproduced (analytic) |
| Band counts are Binomial(n, 0.05): 3.2 rows per band at 64 neighbors, 1.6 at 32 | 3.2 and 1.6 | `arith` | Reproduced (analytic) |
| Chance that at least one of three bands is empty: 0.11 at 64, 0.48 at 32 | 1 − 0.8909 = 0.109; 1 − 0.5164 = 0.484 | `a2`: "P(all three a0 bands >=1)" | Reproduced (analytic) |
| The random pool keeps about 95% of in-band high-load labels | each row removed independently with expected share 0.05 | derived from the stratified removal rule | Reproduced (derived) |
| Strong-current steps with g ≥ 0.4: 99.5% | 99.48% | `prob`, `verify_findings`, `indep_check` | Reproduced (analytic) |

### Float32 edges and calibration

| Review statement | Rerun value | Script and output line | Status |
|---|---|---|---|
| Executed float32 hole runs from −0.84999996 to −0.75000006; −0.75 and −0.85000002 fall outside | as stated | `prob` edge table; `verify_geometry` lattice | Reproduced |
| An episode has no row outside the hole with probability at most 0.05^100 | each of 100 rows must fall in the band, probability 0.05 each at most | derived | Reproduced (derived) |
| About 3.2 of the 128 existing calibration rows fall inside the hole | 3.2 | `arith`: "expected hole rows among 128 old calib rows" | Reproduced (analytic) |
| Calibration identity: the first 128 tuning-validation groups sorted by group ID, IDs 10 to 668 | 128 unique sorted IDs from 10 to 668. The receipt's role text reads "first128 sorted tuning_validation groups". No overlap with fit or validation. Rows equal the bindings split receipt. The support-shift retry used the same episodes | `cal` (second run), `cal2`, `cmp`, `ss2` | Reproduced. The `data.py` line 43 selection rule itself was read during the review, not rerun |

### Decision rule and compute

| Review statement | Rerun value | Script and output line | Status |
|---|---|---|---|
| With two seeds, "both positive" has probability 0.25 under no effect | 0.5² | derived | Reproduced (derived) |
| 120,000 × S critic updates, 21 × S refreshes, 33 × S unique actor snapshots | 3 × (10k + 3 × 10k); 3 × (1 + 3 × 2); 3 × (2 + 3 × 3) | derived from the stated design. Actor half and auxiliary equal to critic per v2 `DRAFT_PROTOCOL.md` line 152 | Reproduced (derived) |
| Training time for two seeds: about 4.3 to 5.0 hours | 2.125 to 2.361 hours per seed, so 4.25 to 4.72 hours for two | `a2`. Its inputs of 637.6 s and 708.3 s are hard-coded from `support_shift_pilot_v1_retry_20261027/RUN_RESULT.json` (elapsed_seconds 637.600727173 and 708.3104125549999) | **Corrected:** about 4.3 to 4.7 hours, training only |
| That is 1.7 to 2 times the 9,000-second cap | 15,300 to 17,000 s, so 1.70 to 1.89 times | derived. The cap is in `support_shift_pilot_v1/PROTOCOL.md` line 83 | **Corrected:** 1.7 to 1.9 times |

## Analytic references in DECISION_BLOCK.md

| Section | Reference | Rerun value | Script | Status |
|---|---|---|---|---|
| 1.3 | calibration identity | as in the review table | `cal`, `cal2`, `cmp`, `ss2` | Reproduced |
| 2.2 | executed edges −0.84999996, −0.85000002, −0.75000006, −0.75 | as stated | `prob` edge table | Reproduced |
| 4.2 | about 1,920 hole rows | 1,920 | `arith`, `coverage_check` | Reproduced |
| 4.3 | about 606 episodes (79%) | 0.789 × 768 = 606 | `verify_findings`, `cond_ood_arith` | Reproduced |
| 4.4 | about 192 per ten-step bin | 192 | `verify_findings`, `cond_ood_arith` | Reproduced |
| 4.5 | about 1,920 retained low-load in-band rows | 1,920 | `cond_ood_arith` | Reproduced |
| 4.6 | depth shares 29%, 25%, 46% | 29.2%, 25.1% (54.2 − 29.2), 45.8% | `indep_check` | Reproduced |
| 5.2 | "0.5 gives about 1 to 2 rows; about 0.67 to 0.77 gives 32 to 64" | 1.4 to 2.3 rows at 0.5; 0.66 to 0.73 gives 32 to 64 | `indep_check`, `geom2` | **Misstated**, as in the review table |
| 5.7, 5.8 | 3.2 and 6.4 expected per 64 neighbors | 3.2; 64 × 0.10 = 6.4 | `arith` (3.2); 6.4 derived | Reproduced (derived) |
| 6.1 | about 3.2 of the 128 calibration rows | 3.2 | `arith` | Reproduced |
| 7.1 | about 1 to 1.5 units across band and adjacent bands | 0.86 to 1.47 over five values | `verify_findings` | Reproduced (model) |
| 7.3 | full-data development critic: +19 to +82 | 24 of 25 from +19.5 to +82.0, one at −6.5 | `verify_geometry`, `bank` | **Corrected**, as in the review table |
| 7.5 | development actor: 0 of 128, nearest 0.32 | 0 of 128; 0.317 from the band center | `verify_geometry`, `bank` | Reproduced. The wording should say "from the band center" |
| 10 | about 2.1 to 2.4 hours per seed | 2.125 to 2.361 | `a2` | Reproduced |

## Corrections to carry into the review and the form

1. **In-band critic errors.** Replace "+19 to +82 units" with "+19 to +82 units, except one probe at −6.5". This
   applies in the review's first major issue and in decision-form item 7.3.
2. **The 0.13 span.** Replace "the 0.1-wide band spans about 0.13 units near the optimum" with "under 0.01 units when
   the band contains the optimum, and about 0.13 units per unit of distance from it".
3. **Pinned-optimum share.** Replace "roughly a third to a half of high-load states" with "about 0.28 to 0.56 under a
   simple float model, and about 0.54 to 0.87 under the propagated state model at recorded current". Both are
   unconstrained linear-quadratic approximations, not Q^pi.
4. **Training time.** Replace "about 4.3 to 5.0 hours" with "about 4.3 to 4.7 hours", and "1.7 to 2 times" with "1.7
   to 1.9 times".
5. **Radius for 32 to 64 rows.** In the review's analytic table and decision-form item 5.2, replace "about 0.67 to
   0.77 gives 32 to 64" with "about 0.66 to 0.73 gives 32 to 64 rows". About 0.67 to 0.77 is the radius for 8
   episodes.
6. **Wording.** In decision-form item 7.5, write "0.32 from the band center".

None of these changes a conclusion of the review. Correction 3 strengthens the point that the hole often removes
commands that are near-optimal at high load.

## Scope limits

These scripts reproduce analytic and model figures. They do not measure coverage in the training arrays, local tails
of the state distribution, or full Q^pi under any trained actor. v3's reconciliation is right that the radius and value
figures are approximations, and the labels above say so. The scratchpad scripts `env.py`, `env2.py`, `shape.py`,
`bind.py`, `seeds.py` and `ss.py` only listed file structure, and `build_samples_titan_fsc.py` belongs to earlier,
unrelated work. None of them was copied.
