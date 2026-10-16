# TOY.md: the full toy-world results (section#toy-worlds, page version 27)

Folder: `scratchpad/wf7/toy`. Written 16 Oct 2026. Nothing here was published; nothing in `C:/Users/David/dev/BCA` or in
WSL was changed (git was only read, with `git show`, `git log` and `git rev-parse`).

The owner asked (16 Oct, 13:38 UTC): "where are the full toy world results? thats very important to include those and
explain". The live page's section#toy-worlds (28,926 characters) covered two notebooks. The new section covers all five,
in order, and replaces the old one through `apply_toy.py`.

## Files

| File | What it is |
|---|---|
| `apply_toy.py IN OUT [--fragment PATH] [--allow-unknown-base]` | the anchor-based patch: replaces section#toy-worlds with `toy_fragment.html` |
| `toy_fragment.html` | the new section (172,487 characters), built by `build_toy.py` |
| `build_toy.py`, `toylib.py` | the section's source: text, tables, worked arithmetic, charts (static SVG, page chart classes) |
| `check_numbers.py` | checks 195 quoted numbers and 24 hand-worked results against the re-runs; writes `run/out/number_checks.md` |
| `run_nb.py`, `run_all.sh`, `compare_nb.py`, `dump_nb.py` | the re-run harness (a real IPython kernel, everything written inside `run/`), the stored-against-re-run comparison, a text dump |
| `run/ties/ties_check.py` | information only: did the discrete notebook's tie rule change any sweep choice (no) |
| `src/*.ipynb` | the notebooks exactly as committed (SHA-256 below); `run/*.ipynb` the copies that were run |
| `run/out/*.json`, `*_compare.txt`, `number_checks.md` | re-run outputs, comparisons, the number table (also below) |
| `test/test_toy.mjs`, `test/shots.mjs`, `test/nav_smoke.mjs`, `test/cdp.mjs` | headless Edge tests (one Edge at a time) |
| `test/result_live.json`, `test/shots_live/`, `test/nav/` | test results and screenshots |
| `edit1.py` … `edit5.py` | one-off edits made to `build_toy.py` in this round (kept for the record; not needed again) |
| `live_toy_section.html` | the live page's old section (versions 19 to 25 and live 92ba, SHA-256 88505c7a…) |

Rebuild and apply: `python -X utf8 build_toy.py`, `python -X utf8 check_numbers.py`, then
`python -X utf8 apply_toy.py BASE OUT` (Python: `C:/Users/David/dev/BCA/.venv/Scripts/python.exe`,
`PYTHONDONTWRITEBYTECODE=1`).

## Sources (read from git, newest committed version across all branches)

`git log --all -- <file>` lists exactly these commits for the five notebooks; each file in `src/` was checked byte for
byte against `git show <commit>:<file>`.

| Notebook | Commit (author time) | Branches | SHA-256 of the file |
|---|---|---|---|
| `bca_discrete_toy.ipynb` | `2c8a0e3adf58a89c537b9155a622752e02ac4a20` (15 Oct 2026, 07:11 UTC) | forked, origin/main, the current branch and the lesson branch | `bd6b8c927de23faae446c15ba624110b4cb262e838751f40c2b5a1e12505b200` |
| `bca_channel_toy.ipynb` | `632022438e6d199403634e787189efc6c86e3a80` (15 Oct, 09:28 UTC, "update", by the owner) | the same | `cf1ba0c5f1e09a115c1a903ec0641709add085d244880a9cc91e35d7e6386413` |
| `bca_mean_gate_toy.ipynb` | `6320224` (as above) | the same | `06da25617949943f8f4630b49dec130e87abb8a674c9f9fe8cd49fc674967f33` |
| `bca_weights_toy.ipynb` | `6320224` (as above) | the same | `73bf93449fe480f47c960745aee69183755ca137aa264ae1e07cb511d4a4e701` |
| `bca_findings.ipynb` | `6320224` (as above) | the same | `dbd12d2021c0a0c55679baea6c6e55f6406f63f7eef3494dc132ace6dfe1c7a7` |
| `bca_findings.ipynb`, lesson version | `2467e293ba38fdbfc3d62b418ae6cab3be4c7595` (16 Oct, 01:44 UTC) | `claude/bca-findings-walkthrough` (local and origin; origin's later merge `66eb326` and `d2d86fb` keep this file unchanged); not on main | `b2d8f44d9e51937bec39d07746d021b4f17ff7b100a257f5e3d7088d71e28ca5` |

The weights notebook's check 2 imports `calibration/wbcp.py`; the run folder held a copy of the repository's file
(last changed in `b951ddf`, 30 Sep; SHA-256 `fdd36b42…c023`, identical to HEAD `0377dd7`), so the check ran (it prints
"skipped" otherwise; it did not).

## Re-runs (no training; toy worlds only; exact where the notebooks compute exactly)

Each notebook was run top to bottom in a fresh IPython kernel (`run_nb.py`; `PYTHONDONTWRITEBYTECODE=1`,
`JAX_PLATFORMS=cpu`, Jupyter, IPython and Matplotlib directories inside `run/_env`), then every printed line was compared
with the committed notebook's stored output (`compare_nb.py`):

| Notebook | Code cells | Identical printed lines | Lines that differ | Re-run time |
|---|---|---|---|---|
| `bca_discrete_toy.ipynb` | 7 | 84 | 1, a running time | 1.4 s |
| `bca_channel_toy.ipynb` | 13 | 137 | 2, running times | 27 s |
| `bca_mean_gate_toy.ipynb` | 13 | 134 | 2, running times | 14 s |
| `bca_weights_toy.ipynb` | 13 | 153 | 2, running times | 118 s |
| `bca_findings.ipynb` (6320224) | 46 | 508 | 7, running times | 189 s |
| `bca_findings.ipynb`, lesson (2467e29) | 135 | 1,725 | 7, running times | 164 s |

No mismatch in any printed result. Three runs went beyond the stored outputs, each on a copy, none changing the
notebook's own code:

- **The discrete notebook's sweep.** The committed notebook ships with `RUN_SWEEP = False` (its stored output says
  "Optional seed sweep skipped"). The copy `run/bca_discrete_toy_sweep.ipynb` changes only that switch. Its sweep
  (200 of 200 seeds used, 4.7 s) gives exactly the numbers in the commit message of `2c8a0e3` (68.8 / 69.0 / 78.1%,
  99.1 / 99.3 / 99.9%, −0.162 with 1 / 17 / 182) and in the notebook's own text (weighted conformal reaches 0.90 in
  191 of 200 seeds). Its other outputs match the stored ones.
- **One information-only cell after the last cell** of the discrete notebook (`run/extra_discrete.ipynb`, cell 16: seed
  7's calibration contexts, weight sums, score values, break-even multipliers) and of the weights notebook
  (`run/extra_weights.ipynb`, cell 28: the deviation rates along the rungs that the notebook only draws, the worlds'
  truths and counts). The copies' other outputs match the stored ones (84 and 153 identical lines).
- **`run/ties/ties_check.py`**: runs the discrete notebook's cells 1, 3, 6 and 8 unchanged and repeats the sweep's fits.
  The only tie (sweep seed 7, "no penalty" and "support only", first stage, context 1: 1.1000000000000003 against 1.1)
  is decided the same with or without the tie rule, so the rule changed no choice in the 200 seeds (the old section's
  statement holds).

## The five notebooks, as the section now tells them

**Shared world (experiments 1 and 2).** Two stages; a context 0 or 1 drawn per episode and seen by the controller;
safe pays 0.1 then 1; venture pays 0.3 or −0.5/1.1 (first stage), 0 or 3 (last stage, context 0); venture at the last
stage in context 1 is the gap, never recorded. Recording: context 1 is 15%, test 65%; exact weights 35/85 = 7/17 = 0.412
and 65/15 = 13/3 = 4.333; w̄ = 2.961; seed 7's 300 held-out episodes are worth 97.2 equally weighted ones.

1. **`bca_discrete_toy.ipynb`, experiment 1** (exploratory demonstration, seed 7 already seen, plus a 200-seed sweep;
   no kill rule). Question: does BCA's width, subtracted as extra caution, make the policy choose better? Result:
   weighting restored coverage (first stage, test mix: 69% and 78% unweighted, 99.3% and 99.9% weighted), but every
   calibrated width turned away venture at the last stage in context 0 (multiplier 16.82 or 14.21 against a
   break-even of 13.49), costing 0.175 per episode on seed 7; over 200 seeds BCA earned less than support only in 182
   (mean −0.162), weighted BCA in 144 (−0.129). Verdict: no width improved a choice. Why: the width is one multiplier per
   stage times the count scale, measured from reward noise, and blind to the gap.
2. **`bca_channel_toy.ipynb`, experiment 2** (kill rules fingerprinted `bb9e0325…dbc3` before the first run by the
   notebook's record; the repository cannot confirm the order, first commit after the runs). Question: can any use of
   BCA change a decision for the better? Ten worlds (advantages +1.0 to −4.6) in two shapes, 5 to 100 survivors, 200
   seeds (28,000 cases per rule). Results: the widths −2.50 and −2.09 of the room (shelved); the step-1 gate +0.58
   (alive); harm test 0.028 against 0.051 (alive); BCA agreed with the exact binomial test in 99.6% (shelved); split
   conformal 5.7% against 0.2% unsafe ventures (alive, narrowly); plain average 29.0% against 0.2% in tail worlds
   (alive). Verdict: BCA helps as a gate that reads the risky action's held-out outcomes, not as a width; with equal
   weights it is the exact binomial test. Why: the critic drops out of a gate compared with safe's value.
3. **`bca_mean_gate_toy.ipynb`, experiment 3** (kill rules `50a0f49d…b1d3`; 36 worlds, 200 seeds of 20 situations;
   seeds 0–99 tune, 100–199 score). Question: can a BCA gate raise return? Results: claims 1, 2, 3 and 5 alive (worst
   harm 0.3%; 17.4% of good proposals against 0% for Hoeffding and empirical Bernstein; t-test up to 31.3% harm; at
   equal harm 0.0845 against 0.0400), claims 4 and 6 shelved (0.0567 against 0.1829 and 0.1625; 0.0270 against 0.0453).
   Verdict: the mean gate earns the most at a fixed safety level; with no safety level the plain average earns more
   (0.158 against 0.045). Why: fresh held-out tries already give an honest average.
4. **`bca_weights_toy.ipynb`, experiment 4** (kill rules `e32e04d7…f44e`; 288 worlds, 800 seeds). Question: do BCA's
   weights keep its promises when the held-out tries come from a different mix of conditions than deployment? All six
   claims alive (unweighted tail gate up to 89.9% in unsafe worlds, 17 above the allowance; weighted at most 4.3% and,
   for the mean gate, 4.4%; 33.6% against 19.0% and 0%; 38.8% against 4.3%; 0.0975 against 0.0923, narrowly). Verdict:
   yes, at a cost in evidence (about 39 and 83 tries before any yes under the moderate and strong shift, against 29).
5. **`bca_findings.ipynb`** (and its lesson version): the four experiments in one notebook, code copied unchanged;
   14 of the 18 pre-registered claims alive; seven recommendations (gate, not width; compare with the fallback's value;
   match the gate to the promise; weights under a shift; exact equal-weight BCA; describe it plainly; budget the
   evidence); limits; three next steps, none run.

## What the section now says, in order (ids kept and added)

1. `toy-worlds` heading and status; `toy-overview` with `toy-overview-table` (notebook, question, verdict, label).
2. `toy-why`: why toy worlds; kill rules, fingerprints, alive and shelved, seeds defined.
3. `toy-world`: the shared world in six numbered steps with `toy-reward-table` and four worked blocks (venture's
   averages; the gap's average; the exact weights 35/85 and 65/15 with their check and w̄; the effective number).
4. `toy-tools`: threshold, score, crossing, posterior, split conformal; width, tail gate, mean gate, weights; worked:
   why 29 tries (0.9^28, 0.9^29, x(30) = 0, 46 tries allow one); worked: the mean gate's exact form.
5. `toy-first` (experiment 1): the finding block in the shelving format (`toy-first-brief`), three findings, worked
   coverage and the weighted multiplier's running total, worked choice and break-even, `toy-first-seed7`,
   `toy-first-sweep`, the audit note.
6. `toy-second` (experiment 2): the worlds, worked tail shape, room (0.12025 + 0.00130), the rules,
   `toy-second-brief`, `toy-pooled-table` (six tests), worked "the critic drops out", `toy-second-case` (one case
   through every gate), chart `toy-second-chart` with `toy-second-regret`, `toy-second-unsafe`, the textbook identity,
   `toy-shelved` with `toy-shelved-width` and `toy-shelved-posterior`, `toy-recheck`, `toy-more-detail` (with
   `toy-second-all` and the full re-check).
7. `toy-third` (experiment 3): the world in four steps, the rivals, worked t-test trap, `toy-third-brief`,
   `toy-third-claims`, `toy-third-results`, chart `toy-third-chart` with `toy-third-caps`, `toy-third-tuned`, two
   shelved findings (`toy-third-shelved-crowded`, `toy-third-shelved-budget`), information only.
8. `toy-fourth` (experiment 4): the world in four steps with `toy-fourth-mix`, worked rungs and weights,
   `toy-fourth-brief`, `toy-fourth-claims`, `toy-fourth-results`, two charts (`toy-fourth-chart`, the tail promise;
   `toy-fourth-chart-mean`, the mean promise) with `toy-fourth-rungs`, one decision worked by hand (seed 1: total weight
   113.879, 15.1% of the weight below, 124 of 1,000 draws), `toy-fourth-evidence`, information only.
9. `toy-together`: `toy-together-claims` (18 claims, 14 alive), the four shelved claims, the seven recommendations,
   what the toys do not show, next steps.
10. `toy-meaning`: kept from the old section and corrected where it had become false: the matching record (width 0.31
   to 0.36 against a false preference of 0.63 to 0.89); step 1 and step 2 of the restructured BCA (106.1 against 90;
   1.78 points; three pass marks); where the toy does not carry over (reference point −0.049 to −0.031; one-step
   targets; no-stall seed 3 with 29 rows and 947 of 1,000 draws, checked in
   `outputs/shift/beyond-the-data/restructure/step1/results/results.json`: `finite_share_tail` 0.947); new: the evidence
   is thin for a shift (39 and 83 against step 1's 29 to 60 rows); the weighting test (coverage restored; no return gain
   as BCA uses its band); no plan changes. "Stage 2's pre-registered seeds 5–9 still test the width" (old) became
   "Stage 2's version 7 ... has ended and will not run". No status box and no status list are added.

Kept from the old section, verified or attributed: the whole re-check of experiment 2 (its numbers come from an
independent re-check of an earlier, uncommitted copy and are labeled as the re-check's; every number it shares with the
notebook matches the committed version's re-run); the ThrusterChain statements (checked against the live page and the
step-1 results file).

Dropped from the old section, because neither the notebook nor git can confirm them: "under-reported coverage on 40 of
the 200 seeds tried" (a reconstruction of the old check on the sweep, `run/extra_discrete.ipynb`, gives 47 seeds and
122 method-stage cases, so the count depends on details of the old code that are not in the repository; the section now
says only that the old form lost boundary targets to rounding) and "Seed 7's printed numbers did not change" (the
pre-fix notebook was never committed).

## Charts and colors

Four static SVG charts in the page's chart style (`svg.chart`, grid, tick, axis, axis titles on both axes, guide lines),
each with a hover or tap target per column whose shared tooltip lists every line with its color key (a swatch with the
line's dash and marker), a legend, direct labels in ink placed where they cross no other line, and a table with the same
numbers. Tokens `--toy-c1` … `--toy-c4`, `--toy-ref` on bare `:root` and redefined in both dark blocks. Each chart uses
at most three hues plus the gray reference; every set used passes the dataviz palette validator on all pairs in light
(surface #ffffff) and dark (surface #191d1b): {blue, orange, aqua} and {orange, aqua, violet}. The validator warns that
aqua (#1baf7a, 2.82:1) is under 3:1 on white; relief: direct labels, markers and the tables. An entity keeps its hue and
marker in every chart (tail gate blue circles, plain average orange squares, split conformal aqua diamonds, mean gate
violet triangles); unweighted variants are dashed with hollow markers. Experiment 4's chart is split in two (one per
promise) because six lines in one panel failed the all-pairs check.

## Tests (all on `scratchpad/live6/core_92ba.html`, SHA-256 0f02c8fc…c062, unchanged before and after)

- `apply_toy.py`: anchors T1 and T2 found once; refuses IN = OUT, a missing section, a doubled section and a changed
  section (unless `--allow-unknown-base`); no CR; no duplicate id; every link into the old section and every link and
  tooltip reference in the new one resolves; all 8 old ids kept; a second run on its own output changes nothing. It
  also applies to our builds `build/core_v26.html` and `build/core_v27.html` (the section after the color and navigation
  patches; 13 of 13 ids kept).
- `test_toy.mjs` on wrapper + patched core, headless Edge, six views (390 px light and dark, 1280 px light and dark, dark
  by `data-theme`, light by `data-theme` on a dark system): no console errors or exceptions, no sideways overflow
  (document width equals the viewport; no element of the section past it), the 15 worked blocks fit, 4 charts drawn
  (paths with non-zero boxes, 7 / 6 / 8 / 8 hover targets, 2 to 4 distinct line colors per chart, the dashed variants sharing their gate's color), every tooltip shows
  with one color key per line and stays in view, every `#` link on the page resolves, no duplicate ids, every stacked
  table cell labeled, no summary or button under 44 px.
- `nav_smoke.mjs` on our version-27 build with the patch (one section at a time): the section shows at `#toy-worlds`,
  "In this section" lists 44 parts of it with no broken anchor, `#toy-meaning` and `#toy-fourth-chart-mean` open
  and draw, no console errors, no sideways overflow at 1280 and 390 px.
- `check_numbers.py`: 195 quoted numbers found in the section and in the re-run output of the named notebook cell (the
  same cell of the committed notebook's stored output where there is one), and 24 hand-worked results recomputed; 0
  problems. One rounding was caught and fixed (93.4% to 93.5%).

## Open issues

- Build order: `apply_toy.py` must run after the color patch (B14, B15) and the navigation patch (five static ids in
  this section); it refuses any other version of the section. If the other session edits the live toy section again,
  compare and re-run with `--allow-unknown-base`.
- The section is long (about 62,000 px at 390 px, 31,500 px at 1280 px), as asked; the overview links to each part
  and the navigation lists 44 parts.
- The experiment-2 re-check figures (fresh seeds, robustness) cannot be re-run from the repository; they are kept from
  the old section and labeled as the re-check's.
- The section's version-history line and status statements elsewhere on the page are not part of this patch.

## Every number, with its notebook cell

Cells are 0-based indexes of the notebook's cells (code and text). "same" means the committed notebook's stored output
has the same text in the same cell; "re-run only" marks the sweep and the two information-only cells, which the
committed notebook does not store.

| Part | Number | Shown as | Notebook | Cell (re-run) | Stored output | Check |
|---|---|---|---|---|---|---|
| World | w0 = 35/85 | 35 / 85 = 7 / 17 = 0.412 | bca_discrete_toy.ipynb | 6 | same | ok |
| World | w1 = 65/15 | 65 / 15 = 13 / 3 = 4.333 | bca_discrete_toy.ipynb | 6 | same | ok |
| World | w-bar | = 0.144 + 2.817 = 2.961 | bca_discrete_toy.ipynb | 6 | same | ok |
| World | seed 7 calibration contexts | 264 rows of context 0 and 36 of context 1 | bca_discrete_toy.ipynb + one information-only cell | 16 | not stored (re-run only) | ok |
| World | weight sums | 264.71² / 720.76 | bca_discrete_toy.ipynb + one information-only cell | 16 | not stored (re-run only) | ok |
| World | effective number 97.2 | = 97.2 | bca_discrete_toy.ipynb | 12 | same | ok |
| Tools | 29 rows | 0.9 28 = 0.0523 | bca_findings.ipynb, lesson version (2467e29) | 29 | same | ok |
| Tools | 46 tries allow one | 46 tries allow one | bca_findings.ipynb, lesson version (2467e29) | 42 | same | ok |
| Tools | 30 tries allow none | re:x\s*\(30\) = 0 | bca_findings.ipynb, lesson version (2467e29) | 41 | same | ok |
| Tools | smallest n | fewer than 29 tries | bca_channel_toy.ipynb | 8 | same | ok |
| Tools | mean gate g at 29 tries | g = 0.11 | bca_mean_gate_toy.ipynb | 7 | same | ok |
| Exp 1 | training counts | 396 training rows | bca_discrete_toy.ipynb | 3 | same | ok |
| Exp 1 | critic venture (1,0) | venture at 1.626 | bca_discrete_toy.ipynb | 3 | same | ok |
| Exp 1 | count scales | count scale 0.05025 | bca_discrete_toy.ipynb | 3 | same | ok |
| Exp 1 | thresholds | 16.8212 | bca_discrete_toy.ipynb | 6 | same | ok |
| Exp 1 | weighted first-stage | 6.2714 | bca_discrete_toy.ipynb | 6 | same | ok |
| Exp 1 | stage-0 scores | 16 of the 300 held-out scores | bca_discrete_toy.ipynb + one information-only cell | 16 | not stored (re-run only) | ok |
| Exp 1 | break-even | = 13.49 | bca_discrete_toy.ipynb + one information-only cell | 16 | not stored (re-run only) | ok |
| Exp 1 | BCA widths at (1,0) | 1.000 − 0.845 = 0.155 | bca_discrete_toy.ipynb + one information-only cell | 16 | not stored (re-run only) | ok |
| Exp 1 | coverage 0.675 | 67.5% | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | weighted conformal 0.8375 | 83.75% | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | weighted BCA 1 | \| 100% | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | last stage floor 0.9125 | 91.25% | bca_discrete_toy.ipynb | 7 (text) | the notebook's text | ok |
| Exp 1 | no-shift 0.925 | 92.5% | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | seed 7 harmful no penalty | −1.515 | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | seed 7 support only | 1.475 | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | seed 7 BCA | 0.175 | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | seed 7 harmless support only | 0.650 | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | seed 7 harmless BCA | 0.825 | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | seed 7 harmless no penalty | 2.125 | bca_discrete_toy.ipynb | 8 | same | ok |
| Exp 1 | sweep empirical | 68.8% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep conformal | 69.0% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep BCA | 78.1% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep weighted empirical | 99.1% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep weighted conformal | 99.3% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep weighted BCA | 99.9% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep no shift | 92.8% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep no shift BCA | 95.0% | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep returns | 1.308 | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep weighted conformal return | 1.436 | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep weighted BCA return | 1.341 | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | sweep no penalty | −1.520 harmful, 2.120 harmless | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | BCA minus support only | 182 of the 200 seeds (mean −0.162 | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | weighted BCA minus support only | weighted BCA in 144 (−0.129) | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | tie seed | One sweep seed had a tie | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 1 | refusal 0 | no method refused a single test target | bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True) | 14 | not stored (re-run only) | ok |
| Exp 2 | fingerprint | bb9e0325…dbc3 | bca_channel_toy.ipynb | 10 | same | ok |
| Exp 2 | unsafe shares | 89.4% of outcomes fall below 1.0, in the tail shape 17.9% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | tail −0.25 | 17.9% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | room | printed as 0.1215 | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | room split | 0.00130 (in 2 of the 200 seeds | bca_findings.ipynb, lesson version (2467e29) | 152 | same | ok |
| Exp 2 | test 1 | −2.50 and −2.09 of the room | bca_channel_toy.ipynb | 17 | same | ok |
| Exp 2 | test 2 | +0.58 of the room | bca_channel_toy.ipynb | 17 | same | ok |
| Exp 2 | test 3 | 0.028 against 0.051 | bca_channel_toy.ipynb | 17 | same | ok |
| Exp 2 | test 4a | 99.6% agreement | bca_channel_toy.ipynb | 17 | same | ok |
| Exp 2 | test 4b | 5.7% against 0.2% | bca_channel_toy.ipynb | 17 | same | ok |
| Exp 2 | test 4c | 29.0% against 0.2% | bca_channel_toy.ipynb | 17 | same | ok |
| Exp 2 | G2 cost | at 0.060 more regret | bca_channel_toy.ipynb | 17 | same | ok |
| Exp 2 | seed 0 critic | the critic values the gap at 1.447 and safe at 1.004 | bca_findings.ipynb, lesson version (2467e29) | 117 | same | ok |
| Exp 2 | seed 0 tail case | 5 of its 30 survivors | bca_findings.ipynb, lesson version (2467e29) | 135 | same | ok |
| Exp 2 | case step-1 | 75,000 of the 1,000,000 pairs | bca_findings.ipynb, lesson version (2467e29) | 145 | same | ok |
| Exp 2 | case harm test | 75 of 1,000 draws | bca_findings.ipynb, lesson version (2467e29) | 145 | same | ok |
| Exp 2 | case t-test | upper bound on the average outcome 1.659 | bca_findings.ipynb, lesson version (2467e29) | 145 | same | ok |
| Exp 2 | case plain average | average outcome 0.827 | bca_findings.ipynb, lesson version (2467e29) | 145 | same | ok |
| Exp 2 | case tail | threshold 6.883 | bca_findings.ipynb, lesson version (2467e29) | 145 | same | ok |
| Exp 2 | case split | 6.576, above 0.443 | bca_findings.ipynb, lesson version (2467e29) | 145 | same | ok |
| Exp 2 | case binomial | 5 of 30 below 1.004; 30 tries allow 0 | bca_findings.ipynb, lesson version (2467e29) | 145 | same | ok |
| Exp 2 | regret table: tail | 0.1215 \| 0.1215 \| 0.1215 \| 0.0485 \| 0.0411 \| 0.0390 \| 0.0373 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: split | 0.1215 \| 0.0318 \| 0.0265 \| 0.0251 \| 0.0266 \| 0.0224 \| 0.0207 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: plain | 0.0297 \| 0.0275 \| 0.0149 \| 0.0125 \| 0.0102 \| 0.0084 \| 0.0065 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: G1 | 0.0854 \| 0.0693 \| 0.0482 \| 0.0441 \| 0.0397 \| 0.0362 \| 0.0329 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: G1 local | 0.0661 \| 0.0505 \| 0.0296 \| 0.0264 \| 0.0224 \| 0.0191 \| 0.0161 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: G2R | 0.0619 \| 0.0457 \| 0.0256 \| 0.0219 \| 0.0176 \| 0.0139 \| 0.0108 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: G6 | 0.1600 \| 0.0944 \| 0.0612 \| 0.0443 \| 0.0341 \| 0.0289 \| 0.0208 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: A1 | 0.4274 \| 0.4253 \| 0.4250 \| 0.4250 \| 0.4247 \| 0.4247 \| 0.4247 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | regret table: A2 | 0.4075 \| 0.4075 \| 0.4018 \| 0.3985 \| 0.3872 \| 0.3368 \| 0.2891 | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | unsafe table: G2 | 0.0% \| 0.0% \| 0.0% \| 0.3% \| 0.3% \| 0.2% \| 0.2% | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | unsafe table: G4 | 0.0% \| 0.0% \| 0.0% \| 0.4% \| 0.0% \| 0.1% \| 0.2% | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | unsafe table: G3 | 0.0% \| 8.9% \| 7.6% \| 6.6% \| 5.0% \| 5.8% \| 5.7% | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | unsafe table: G5 | 27.2% \| 30.6% \| 26.1% \| 26.8% \| 26.5% \| 25.7% \| 26.0% | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | unsafe table: G2R | 39.2% \| 34.2% \| 27.2% \| 26.5% \| 23.9% \| 22.6% \| 19.6% | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | unsafe table: G1 | 60.2% \| 56.8% \| 52.0% \| 51.5% \| 49.1% \| 47.7% \| 44.8% | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | unsafe table: G6 | 62.5% \| 55.5% \| 50.0% \| 46.3% \| 43.8% \| 42.3% \| 39.2% | bca_channel_toy.ipynb | 19 | same | ok |
| Exp 2 | worst world split conformal | split conformal ventured in 32% to 38% of seeds | bca_channel_toy.ipynb | 25 | same | ok |
| Exp 2 | worst world tail | the BCA tail certificate in at most 3% | bca_channel_toy.ipynb | 25 | same | ok |
| Exp 2 | all rules: greedy | greedy \| 0.5505 \| −3.53 \| 100% \| 100% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: count | count penalty \| 0.1222 \| −0.01 | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: host | host BC actor \| 0.4939 \| −3.06 \| 86.5% \| 86.5% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: A1 | A1 \| 0.4253 \| −2.50 \| 0.3% \| 0.3% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: A2 | A2 \| 0.3755 \| −2.09 \| 64.9% \| 65.3% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G1 | G1 \| 0.0508 \| 0.58 \| 51.7% \| 42.4% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G1 local | G1 local reference \| 0.0329 \| 0.73 \| 24.1% \| 42.4% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G2R | G2R \| 0.0282 \| 0.77 \| 27.6% \| 42.4% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G2 | G2 \| 0.0758 \| 0.38 \| 0.2% \| 0.2% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G3 | G3 \| 0.0392 \| 0.68 \| 5.7% \| 7.8% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G4 | G4 \| 0.0753 \| 0.38 \| 0.1% \| 0.1% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G5 | G5 \| 0.0157 \| 0.87 \| 27.0% \| 29.0% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G6 | G6 \| 0.0634 \| 0.48 \| 48.5% \| 68.1% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G1 shuffled | G1 shuffled \| 0.4356 \| −2.58 \| 89.9% \| 92.0% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G2R shuffled | G2R shuffled \| 0.3543 \| −1.91 \| 71.0% \| 79.1% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | all rules: G2 shuffled | G2 shuffled \| 0.1211 \| 0.00 \| 0.4% \| 0.5% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | oracle 25% | oracle switch \| 0 \| 1 \| 25.0% \| 25.0% | bca_channel_toy.ipynb | 15 | same | ok |
| Exp 2 | width coverage | 90.2% and 93.5% | bca_channel_toy.ipynb | 23 | same | ok |
| Exp 2 | step-1 shortfall | 94% to 98% of seeds | bca_channel_toy.ipynb | 25 | same | ok |
| Exp 2 | reference group | 10.7% of them a noisy cell's low outcomes | bca_channel_toy.ipynb | 25 | same | ok |
| Exp 2 | reference swing | swung from 0.35 to 1.50 | bca_channel_toy.ipynb | 25 | same | ok |
| Exp 2 | sampler at 28 and 29 | only 4 of 8 score sets matched | bca_channel_toy.ipynb | 8 | same | ok |
| Exp 3 | fingerprint | 50a0f49d…b1d3 | bca_mean_gate_toy.ipynb | 10 | same | ok |
| Exp 3 | t-test example | 0.95 29 = 0.23 | bca_mean_gate_toy.ipynb | 7 | same | ok |
| Exp 3 | mean gate on clean samples | re:\(\s*10 / 10\.4\s*\)\s*29 = 0\.32 | bca_mean_gate_toy.ipynb | 7 | same | ok |
| Exp 3 | claim 1 | worst harm rate 0.3% | bca_mean_gate_toy.ipynb | 17 | same | ok |
| Exp 3 | claim 2 | 17.4% against 0.0% and 0.0% | bca_mean_gate_toy.ipynb | 17 | same | ok |
| Exp 3 | claim 3 | up to 31.3% of harmful proposals | bca_mean_gate_toy.ipynb | 17 | same | ok |
| Exp 3 | claim 4 | 0.0567 against 0.1829 and 0.1625 | bca_mean_gate_toy.ipynb | 17 | same | ok |
| Exp 3 | claim 5 | 0.0845 against 0.0400 | bca_mean_gate_toy.ipynb | 17 | same | ok |
| Exp 3 | claim 6 | 0.0270 against 0.0453; harm 0.000 | bca_mean_gate_toy.ipynb | 17 | same | ok |
| Exp 3 | proposals | Picking the truly best candidate in every situation would gain 0.294 | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: no gate | No gate (the critic alone) \| 0.138 \| 66% \| 100% \| 100.0% \| 100.0% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: count | 0.065 \| 31% \| 32% \| 43.2% \| 12.4% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: plain | Plain average \| 0.158 \| 76% \| 83% \| 41.4% \| 36.2% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: shrunk | Shrunk average \| 0.151 \| 73% \| 69% \| 90.0% \| 16.4% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: cautious | BCA's cautious average \| 0.136 \| 65% \| 60% \| 11.8% \| 1.7% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: t-test | t-test \| 0.080 \| 39% \| 44% \| 31.3% \| 31.3% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: mean gate | BCA mean gate \| 0.045 \| 22% \| 17% \| 0.3% \| 0.0% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: tail | BCA tail gate \| 0.003 \| 2% \| 1% \| 0.0% \| 0.0% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: split | Split-conformal tail \| 0.019 \| 9% \| 10% \| 7.6% \| 7.6% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: shared | BCA mean gate, shared budget \| 0.027 \| 13% \| 10% \| 0.0% \| 0.0% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | results: oracle | 0.208 \| 100% \| 100% \| 0.0% \| 0.0% | bca_mean_gate_toy.ipynb | 15 | same | ok |
| Exp 3 | caps: mean | 0.030 \| 0.061 \| 0.084 \| 0.103 \| 0.137 \| 0.137 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: cautious | 0.000 \| 0.021 \| 0.040 \| 0.098 \| 0.136 \| 0.136 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: shrunk | 0.001 \| 0.019 \| 0.019 \| 0.056 \| 0.087 \| 0.151 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: plain | 0.001 \| 0.007 \| 0.007 \| 0.036 \| 0.078 \| 0.158 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: t-test | none qualifies \| 0.040 \| 0.158 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: Hoeffding | 0.005 \| 0.017 \| 0.017 \| 0.017 \| 0.017 \| 0.017 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: Bernstein | 0.006 \| 0.006 \| 0.006 \| 0.006 \| 0.006 \| 0.006 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: tail | 0.003 \| 0.003 \| 0.017 \| 0.017 \| 0.027 \| 0.046 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: split | 0.006 \| 0.006 \| 0.006 \| 0.019 \| 0.019 \| 0.062 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | caps: count | 0.000 \| 0.000 \| 0.000 \| 0.011 \| 0.011 \| 0.138 | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | within cap | every tuned gate stayed within its cap | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | tuned: mean | level 0.2 \| 0.0864 \| 0.0845 \| 2.3% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: cautious | margin 0.5 \| 0.0410 \| 0.0400 \| 3.8% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: shrunk | margin 0.5 \| 0.0165 \| 0.0187 \| 1.6% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: Hoeffding | level 0.5 \| 0.0167 \| 0.0174 \| 0.1% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: tail | share 0.2 \| 0.0187 \| 0.0172 \| 2.9% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: plain | margin 1.5 \| 0.0061 \| 0.0070 \| 0.8% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: Bernstein | level 0.5 \| 0.0061 \| 0.0061 \| 0.0% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: split | share 0.05 \| 0.0064 \| 0.0056 \| 0.3% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | tuned: count | 100 tries \| 0.0001 \| 0.0003 \| 0.2% | bca_mean_gate_toy.ipynb | 19 | same | ok |
| Exp 3 | good accepted by tries | 1%, 5% and 17% of good proposals, the plain average 71%, 80% and 89% | bca_mean_gate_toy.ipynb | 21 | same | ok |
| Exp 3 | good accepted, plain | 71%, 80% and 89% | bca_mean_gate_toy.ipynb | 21 | same | ok |
| Exp 3 | cautious in hardest worlds | 0.008 to 0.048 per situation | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 3 | worlds losing return | each lost return in 4 worlds | bca_mean_gate_toy.ipynb | 23 | same | ok |
| Exp 4 | fingerprint | e32e04d7…f44e | bca_weights_toy.ipynb | 10 | same | ok |
| Exp 4 | evidence: none | None \| 1.00 \| 1.00 \| 29 \| 29 \| 216 | bca_weights_toy.ipynb | 7 | same | ok |
| Exp 4 | evidence: moderate | Moderate \| 0.73 \| 0.40 \| 39 \| 73 \| 287 | bca_weights_toy.ipynb | 7 | same | ok |
| Exp 4 | evidence: strong | Strong \| 0.34 \| 0.20 \| 83 \| 145 \| 382 | bca_weights_toy.ipynb | 7 | same | ok |
| Exp 4 | claim 1 | up to 89.9%; 17 unsafe worlds above 7.3% | bca_weights_toy.ipynb | 16 | same | ok |
| Exp 4 | claim 2 | at most 4.3% | bca_weights_toy.ipynb | 16 | same | ok |
| Exp 4 | claim 3 | 33.6% against 19.0% and 0.0% | bca_weights_toy.ipynb | 16 | same | ok |
| Exp 4 | claim 4 | 38.8% against 4.3% | bca_weights_toy.ipynb | 16 | same | ok |
| Exp 4 | claim 5 | at most 4.4% | bca_weights_toy.ipynb | 16 | same | ok |
| Exp 4 | claim 6 | 0.0975 against 0.0923 | bca_weights_toy.ipynb | 16 | same | ok |
| Exp 4 | results: always | Always deviate \| 1.000 \| 1.000 \| 1.000 \| 0.0950 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: tail unweighted | 0.899 \| 0.550 \| 0.736 \| 0.1106 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: tail weighted | 0.043 \| 0.336 \| 0.025 \| 0.0838 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: split | 0.388 \| 0.750 \| 0.266 \| 0.1423 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: rejection | 0.036 \| 0.190 \| 0.021 \| 0.0644 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: stratified | 0.000 \| 0.000 \| 0.000 \| 0.0023 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: mean unweighted | 0.986 \| 0.647 \| 0.936 \| 0.1253 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: mean weighted | 0.119 \| 0.415 \| 0.044 \| 0.0982 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: rejection mean | 0.119 \| 0.261 \| 0.026 \| 0.0789 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: t-test | 0.440 \| 0.741 \| 0.381 \| 0.1395 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: weighted average | 0.730 \| 0.927 \| 0.549 \| 0.1536 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: plain | 1.000 \| 0.888 \| 1.000 \| 0.1461 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | results: oracle | 1.000 \| 1.000 \| 0.000 \| 0.1719 | bca_weights_toy.ipynb | 15 | same | ok |
| Exp 4 | world counts | 144 unsafe worlds | bca_weights_toy.ipynb + one information-only cell | 28 | not stored (re-run only) | ok |
| Exp 4 | worst unweighted mean | the mean gate in up to 93.6% | bca_weights_toy.ipynb + one information-only cell | 28 | not stored (re-run only) | ok |
| Exp 4 | effective tries | averaged 41.8 effective tries of 120 | bca_weights_toy.ipynb | 20 | same | ok |
| Exp 4 | w-bar strong | = 2.9096 | bca_weights_toy.ipynb + one information-only cell | 28 | not stored (re-run only) | ok |
| Exp 4 | usual below 0 | 0.0062 | bca_weights_toy.ipynb + one information-only cell | 28 | not stored (re-run only) | ok |
| Exp 4 | rung truth | 0.1056 | bca_weights_toy.ipynb + one information-only cell | 28 | not stored (re-run only) | ok |
| Exp 4 | rungs: tail unweighted | 0.999 \| 0.998 \| 0.945 \| 0.769 \| 0.588 \| 0.426 \| 0.219 \| 0.045 | bca_weights_toy.ipynb + one information-only cell | 28 | not stored (re-run only) | ok |
| Exp 4 | rungs: tail weighted | 0.895 \| 0.541 \| 0.211 \| 0.072 \| 0.037 \| 0.014 \| 0.006 \| 0.003 | bca_weights_toy.ipynb + one information-only cell | 28 | not stored (re-run only) | ok |
| Exp 4 | seed 1 tries | 38, 33, 27, 14 and 8 | bca_findings.ipynb, lesson version (2467e29) | 255 | same | ok |
| Exp 4 | seed 1 total weight | = 113.879 | bca_findings.ipynb, lesson version (2467e29) | 255 | same | ok |
| Exp 4 | seed 1 share | = 0.849 | bca_findings.ipynb, lesson version (2467e29) | 255 | same | ok |
| Exp 4 | seed 1 draws | 124 of 1,000 | bca_findings.ipynb, lesson version (2467e29) | 255 | same | ok |
| Exp 4 | seed 1 unweighted | 6 ≤ 6: yes, deviate | bca_findings.ipynb, lesson version (2467e29) | 255 | same | ok |
| Exp 4 | seed 1 world rates | deviated in 42.6% and the weighted gate in 1.4% | bca_findings.ipynb, lesson version (2467e29) | 255 | same | ok |
| Exp 4 | proof-covered version | worst harmful-world rate 2.6% against 4.4% | bca_weights_toy.ipynb | 26 | same | ok |
| Exp 4 | wbcp check | in 300 of 300 random cases | bca_weights_toy.ipynb | 23 | same | ok |
| Together | findings re-run | 508 printed result lines | bca_findings.ipynb (6320224) | 9 | same | ok |
| Together | findings: 14 of 18 | 14 alive and 4 shelved | bca_findings.ipynb (6320224) | 84 (text) | the notebook's text | ok |
| Exp 4 | wbcp check: 300 cases | in 300 of 300 random cases | bca_findings.ipynb (6320224) | 79 (text) | the notebook's text | ok |
| Exp 1 | weighted conformal 191 of 200 (the notebook's text) | 191 of the 200 sweep seeds | bca_discrete_toy.ipynb | 15 (text) | the notebook's text | ok |

| Worked arithmetic | Shown as | Computed | Check |
|---|---|---|---|
| 0.9^28 | 0.0523 | 0.0523348 | ok |
| 0.9^29 | 0.0471 | 0.0471013 | ok |
| 0.9^30 | 0.0424 | 0.0423912 | ok |
| P(Bin(30,0.1) <= 1) | 0.1837 | 0.183695 | ok |
| (1/1.11)^29 | 0.0485 | 0.0484879 | ok |
| (1/1.10)^29 | 0.0630 | 0.0630394 | ok |
| tail p at -0.25 | 0.179 | 0.178571 | ok |
| room at the gap | 0.12025 | 0.12025 | ok |
| 98.9% of the room | 98.9% | 0.989305 | ok |
| ESS seed 7 | 97.2 | 97.2156 | ok |
| context-1 weight share | 59% | 0.589333 | ok |
| zero-score weight share | 73.8% | 0.738074 | ok |
| after 12 scores | 93.5% | 0.934519 | ok |
| break-even | 13.49 | 13.4887 | ok |
| w-bar strong | 2.9096 | 2.90964 | ok |
| 120 / w-bar | 41.2 | 41.2422 | ok |
| allowance | 0.073 | 0.0731166 | ok |
| total weight seed 1 | 113.879 | 113.879 | ok |
| weight share below | 15.1% | 0.151477 | ok |
| cautious / plain | 86% | 0.860759 | ok |
| t-test clean share | 0.23 | 0.225936 | ok |
| mean gate clean share | 0.32 | 0.320651 | ok |
| rung 0.10 share below | 0.1056 | 0.105589 | ok |
| rung 0.12 mean | −0.04 | -0.04 | ok |
