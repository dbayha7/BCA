# ThrusterChain page, version 27: how it is built

Page version 27 = the current live page (service version 1792156350-92ba: version 25 with its status box removed at
the owner's request, published by the other session, which called that change "version 26" in its reply but wrote no
version-history entry for it) plus this build: the toy worlds' full results (section#toy-worlds rewritten with all five
toy notebooks, at the owner's request of 16 Oct 2026, 13:38 UTC), the lesson "Learn step by step", the navigation, the
remaining charts' colors with color keys in their hover readouts, the status statements outside the removed status
box, and the TD3+BC settings wording fix. The build was first made on version 19 and called "version 20"; it was
rebased onto version 23 on 15-16 Oct 2026 as version 24, onto version 25 on 16 Oct 2026 as version 26, and onto the
live page after version 25 on 16 Oct 2026 as version 27; the toy worlds' results were then integrated into version 27
as its first stage (REBASE.md: what versions 20-25 and the later live edits added, every conflict and how it was
merged, every deliberate change to live content, the status facts, the test results; part 4 is the toy integration).
**No status box**: the owner asked for it to be removed ("I dont want this section included"); this build adds,
restores or recreates no status box and no header status list anywhere, and `apply_all.py` refuses a page with any
part of one.

Everything here is under `scratchpad/wf7`. Nothing in the repository (`C:/Users/David/dev/BCA`) is written; it is only
read (git refs with `GIT_OPTIONAL_LOCKS=0`, the saved run files, its code for inference and exact simulation; the toy
notebooks were re-run on copies, see `wf7/toy/TOY.md`). Nothing is published by this build.

## Inputs

| What | File | Notes |
|---|---|---|
| Base page (the live page after version 25, service version 1792156350-92ba) | `scratchpad/live6/core_92ba.html` | 5,849,901 bytes, SHA-256 `0f02c8fc…c062`; read, never written |
| Publish wrapper | first 531 bytes of `scratchpad/live6/live_92ba.html`, plus `"\n</body></html>"` at the end | the same bytes as version 25's |
| Toy component | `wf7/toy/apply_toy.py` with `wf7/toy/toy_fragment.html` (172,487 characters) | replaces section#toy-worlds as a whole; it knows the live page's own section (SHA-256 `88505c7a…`, versions 19-25 and 92ba) and refuses any other unless `--allow-unknown-base` is passed after a person compared the two; the fragment is built by `wf7/toy/build_toy.py` with `toylib.py`, its numbers checked by `check_numbers.py` against re-runs of the committed notebooks (`wf7/toy/TOY.md`) |
| Color component | `wf7/color/apply_color.py` | 90 anchor edits (12 optional); `--status-date`, `--status-time` and `--version` passed on; no edit on the status box (B1, B2, B4, B20, B23 dropped) and none in the toy section (B14, B15 dropped with the toy integration) |
| Navigation component | `wf7/nav/apply_nav.py` with `wf7/nav/src/` (`static_edits.json`, `nav.css`, `nav_markup.html`, `nav_side.html`, `start_bar.html`, `nav.js`) | edits E10, E9, E11, E7a-c, E1, E2, E4, E5, E6, E8, E12, E13, E14; every block it replaces is checked against the SHA-256 of the versions it knows (`KNOWN`); `static_edits.json` (127 edits) is made by `wf7/nav/rules/static_edits_v27t.py` from the 132 of `static_edits_v27.py` (version 23's 133 without the status box's label) without the five ids the old toy section got from them (the new section has them itself) |
| Lesson component | `wf7/lesson/apply_lesson.py` with `lesson_fragment.html`, `lesson.css`, `lesson.js`, `lesson_data.json` next to it | built by `wf7/lesson/build/build_lesson.py` (below); for version 27 only step 7.2's status sentence changed |
| Integration | `wf7/build/apply_all.py` | runs the four patches in order, then I1-I3 and the checks |
| The version-19 build (page version 20) | `wf7/build/component_changes/v19_based/` | 178 files; SHA-256 in `component_changes/v19_based.sha256` |
| The version-23 build (page version 24) | `wf7/build/component_changes/v23_based/` | 206 files; SHA-256 in `component_changes/v23_based.sha256`; rebuilds `core_v24.html` byte for byte from `live4/core_6eea.html` with `--wrapper live4/live_6eea.html` |
| The version-25 build (page version 26) | `wf7/build/component_changes/v25_based/` | 234 files; SHA-256 in `component_changes/v25_based.sha256` (its own SHA-256 `643651c6…0c97`); `v25_based/build/apply_all.py ../live5/core_4933.html OUT --wrapper live5/live_4933.html` rebuilds `core_v26.html` (`0765c149…d378`) byte for byte |
| Version 27 before the toy integration | `wf7/build/component_changes/v27_pretoy/` | 54 files (the patch scripts and inputs it changes or uses, the rebase edit scripts, the navigation's rules and expected lists, the version-27 test scripts, the records, the toy patch as received, `live6_work/replay_v27.py`); SHA-256 in `component_changes/v27_pretoy.sha256`; its `build/apply_all.py` rebuilds the pre-toy `core_v27.html` (`7e0d6d90…657d`) byte for byte |
| The rebases' edits | `wf7/build/rebase/edits/` | version 24: `e_color_v24.py`, `e_nav_v24.py`, `e_nav_v24b.py`; version 26: `e_color_v26.py`, `e_nav_v26.py`, `e_lesson_v26.py`, `e_apply_all_v26a.py`, `e_apply_all_v26.py`; version 27: `e_color_v27.py`, `e_color_v27b.py`, `e_color_v27c.py`, `e_nav_v27.py`, `e_lesson_v27.py`, `e_apply_all_v27.py`, `e_apply_all_v27b.py` (and `nav/rules/static_edits_v27.py`); the toy integration: `e_color_v27t.py`, `e_nav_v27t.py`, `e_apply_all_v27t.py`, `e_tests_v27t.py` (and `nav/rules/static_edits_v27t.py`, `build/toyint/make_tests_toy_block.py`); exact-replacement scripts (each old text must occur as often as stated, or nothing is written); `rep.py` their helper. `scratchpad/live6/work/replay_v27.py` replays the version-27 chain from `v25_based/`, and `replay_v27t.py` the toy integration from `v27_pretoy/`: every source and the built page come out byte-identical |

## The order of the stages

`apply_all.py BASE OUT [--version N] [--status-date "D Mon YYYY"] [--status-time HH:MM] [--preview PREVIEW]
[--wrapper LIVE] [--stages DIR]` (defaults: version 27, status 16 Oct 2026, 13:44 UTC, wrapper `live6/live_92ba.html`)
runs, each patch as its own script on the previous stage's output (kept in the stages folder; version 27's are in
`wf7/build/stages_v27/`, version 26's in `stages_v26/`, version 24's in `stages/`):

0. `wf7/toy/apply_toy.py` -> `s0_toy.html`: section#toy-worlds replaced as a whole by the full results of the five toy
   notebooks (an overview table; why toy worlds and what kill rules are; the shared two-step world and its weights
   worked out step by step; BCA's tools in toy terms; experiments 1 to 4, each as the idea, what was tested and the kill
   rule, what happened, why, with tables, worked arithmetic and the shelved claims; what the toy worlds say together;
   what they mean for ThrusterChain). 15 worked-arithmetic blocks, 19 tables, 4 charts (both axes named on each, as
   version 24 asks; a tooltip per column with a color key, one swatch with its marker shape per line, in the section's
   own tokens `--toy-c1..c4`, `--toy-ref`, defined on `:root` and in both dark blocks). All 13 ids of the old section
   (its 8 and the 5 the navigation gave it) are kept. Its log must say it replaced the live page's own section (or that
   the base already has this section); nothing after it edits the section.
1. `wf7/color/apply_color.py` -> `s1_color.html`: chart colors and color keys in hover readouts (the learned actors'
   cost chart too: K13b on version 25's readout, which names the learner, K22a-e, N7b), the status statements outside
   the removed status box (B5-B11, B13, B16-B19, B21, B22, B24, B25; B21 and B22 state the near-80 test in After the stop
   and Results' opening, B24 and B25 the end of Stage 2 version 7), the Methods chart's guide lines (N10), the
   dot-and-range tooltip's shadow token (K15), the TD3+BC settings wording (B12). No edit touches the status box, which
   the base no longer has, the live page's newer "Strong test mix" tooltip, or the toy section.
2. `wf7/nav/apply_nav.py` -> `s2_color_nav.html`: the navigation (Contents, "In this section" with every part
   clickable, the bar and its sheet with peek mode, one section at a time with the remembered Whole page switch, the
   pager, the start bar), the table scroll cue and the tooltips' Tab stops that skip hidden sections (E12, E13), and
   E14 (the switch bar's comment). E1 and E8 replace the old Choices sidebar; its rows of versions 21, 22 and 25
   ("Who drives" with "Learned actor", the learned rows, "Learner", "Table", "not in use now") are carried over into
   nav.js. Its lists are built from the page's own parts, so with the status box gone neither Contents nor "In this
   section" lists it (the header: "Start of the section" and "The process, step by step"), and the toy section's
   "In this section" lists its 44 parts (Contents keeps its one entry per section, "Toy worlds").
3. `wf7/lesson/apply_lesson.py` -> `s3_color_nav_lesson.html`: section#learn after the header, its CSS, its data block
   and script. Its L4 must report "skipped" (the navigation's start bar holds the lesson link). Its links into the toy
   section (step 5.5 and its sources, to `#toy-meaning`) resolve.
4. Integration edits in `apply_all.py`: I1 the version-history entry "Version 27, 16 Oct 2026." on top (refused when an
   entry for the version exists; four of the owner's requests, the toy-world results among them; it says in one clause
   that the page as version 26 left it had its status box removed at the owner's request, and no entry is made for the
   other session's version 26), I2 After the stop's status date and time, I3 the layout comment. Then the checks: no
   CR byte, no duplicate id (MathJax glyph ids included), every SVG `<use>` and every in-page link (`href="#id"`)
   resolves, the title unchanged, the old navigation gone, **no part of the status box** (`id="head-status"`,
   `id="head-status-h"`, `<ul class="g-status"`, a link to `#head-status`, "Status as of ", the navigation's label for
   it), the header going from its opening paragraph straight to "The process, step by step" as on the live page,
   versions 20-23's own parts present, versions 24-25's code and parts present as often as in the base outside the toy
   section (axis titles, the Learner switch and its data, the rerun's readouts), the toy section's four charts each
   naming both axes, the live page's newer "Strong test mix" tooltip once as in the base, version 25's Learner row in
   the Choices, **the toy section equal to the toy patch's fragment byte for byte, with all 13 ids of the old section**,
   exactly one version-history entry per version number (1 to 25, and 27), the status statements consistent (one
   status time in its 4 places, no out-of-date wording, the old toy sentence "seeds 5–9 still test the width" gone),
   under 16 MB with the wrapper.

## The lesson's own build (when its content, data or status sentence change)

1. Numbers (only if the readings themselves change; inference and exact simulation of saved files only, nothing
   trained): `wf7/build/fix1/work/run_numbers.sh` runs `wf7/lesson-data/judge/make_new_numbers.py`,
   `extra_numbers.py` and `verify_judge.py` with the repository's Python (`PYTHONDONTWRITEBYTECODE=1`,
   `JAX_PLATFORMS=cpu`), after checking the SHA-256 of the five saved files in `wf7/lesson-data/branch/`.
2. The spec: `python -X utf8 wf7/lesson-data/judge/build_spec.py` (content of `content_ch1..7.py`; every worked line is
   checked by rule D2 and against its exact value; formulas rendered with `wf4/understand/mj_render.cjs`, family MJXL).
3. The data block and its checks: `python -X utf8 wf7/lesson-data/build_lesson_data.py`, then the repository's Python
   on `make_test_vectors.py`, then `node test_port.mjs` (all in `wf7/lesson-data`).
4. `cp wf7/lesson-data/lesson_data.json wf7/lesson/lesson_data.json`, then
   `python -X utf8 wf7/lesson/build/build_lesson.py` (with `GIT_OPTIONAL_LOCKS=0`; reads `src/parts/*.js`,
   `src/lesson.css`, `src/port.js`, `build/symbols.py`, and the repository's git refs for step 7.2's status sentence:
   the approval of Stage 2 version 7 and its import record; since version 27 also the named constant `V7_ENDED`, the
   day version 7 was ended without running, which no git ref records). For version 27 only step 4 ran: the rebuilt
   `lesson_fragment.html` differs from version 26's in that one sentence; `lesson.js`, `lesson.css` and
   `lesson_data.json` are byte-identical.

The toy section has its own build: `python -X utf8 wf7/toy/build_toy.py`, then `python -X utf8 wf7/toy/check_numbers.py`
(every quoted number against the re-runs), as `wf7/toy/TOY.md` describes. If the fragment changes, the integration
checks compare the built section with the new fragment, and the navigation's expected lists must be made again (below).

## The status time

The status statements hold as of **16 Oct 2026, 13:44 UTC** (`STATUS_DATE`, `STATUS_TIME` in `apply_all.py` and
`apply_color.py`): GitHub's branch heads read then with `git ls-remote origin` (read only; nothing fetched, nothing
written in the repository), equal to the remote-tracking refs fetched at 12:39 UTC by another session; read again at
14:47 UTC, 16:05 UTC and, for the toy integration, 16:30 UTC: the branches the page names are unchanged (one later
commit, on `claude/shift-hosts` at 16:21 UTC, freezes the four-learner study's plan and leaves every statement true;
REBASE.md section 20). Two facts are in no ref and are the ones the workflow's task states as current: Stage 2 version
7 was ended without running (seeds 5-9 will not run under it), and the near-80 test (approved by the owner at their own
terminal at 11:50:43 UTC, committed at 11:51 UTC as `f79b89c`, the approval recorded at 11:53 UTC as `287380b`) runs
on the owner's Framework Desktop computer. The four-learner study is stated as the task gives it (plan approved in
chat; code in review, tested on made-up numbers only; the owner's approval at their own terminal still to come;
nothing trained on ThrusterChain data). Statements that carry their own earlier date (the route table, Stage 2's
results, the second claim's status, the glossary's Stage 2 entry, the proposed change to the claim) are left as they
are: they were true at their date. The toy section states its own facts (the notebooks' commits, checked against the
refs; "Stage 2's version 7, which was to test the width on fresh seeds 5–9, has ended and will not run"). A later build
that moves the status time must re-read every status statement (the color patch's B5-B11, B13, B16-B19, B21, B22, B24,
B25, the lesson's step 7.2 sentence from `build_lesson.py`, the toy section's, I2) and the checks in `apply_all.py`.

## How to rebuild and test

```sh
cd scratchpad/wf7/build
python -X utf8 apply_all.py ../../live6/core_92ba.html core_v27.html --preview page_v27_preview.html --stages stages_v27
python -X utf8 tests/make_tests_v27.py    # makes the version-27 test scripts from the version-26 ones, then adjusts them for the toy section
python -X utf8 tests/setup_tests_v27.py   # copies the components' suites (the toy section's too), builds the comparison pages
timeout 7200 sh tests/run_v27.sh > tests/run_v27.log   # about an hour; one headless Edge at a time; timing tests last
python -X utf8 tests/collect_v27.py        # one line per check -> tests/out/results_v27.json
```

Before `setup_tests_v27.py`, the navigation's expected lists are made once, on the live page after the toy patch:
`python -X utf8 nav/rules/make_with_static.py build/stages_v27/s0_toy.html ../live6/live_92ba.html nav/rules/v27_with_static.html`
then `node nav/rules/run_rules.mjs <that page> - pass2_v27` (in `wf7`; `pass2_v27.txt` lists 268 parts: the 237 of the
live page without the status box, with the toy section's 44 parts in place of the old section's 13; outside the toy
section the lists are unchanged).

`run_v27.sh` runs: the static checks (`static_v27.py`: the page contract, ids, links, spelling, the old navigation
gone, no status box, version 25's data block and the live page's newer "Strong test mix" tooltip unchanged, the header
as live; the toy section equal to its fragment, its tokens defined on `:root` and in both dark blocks, its 13 old ids),
the lesson's full and extra tests, the color plan's checks on four views (with the live page rendered as the learning
curves' reference) and its text shots (After the stop's box, the first claim, Results' opening, Stage 2's outcome,
"What happened to this plan", no status box), the old full check (`int/check3_v27.mjs`), the regression test against
the live page (`regression_live.mjs`: (a) every live element id kept except the 18 removed on purpose, (b) every line
versions 20-25 and the later live edits added unchanged except the listed deliberate changes R4-R9, counted outside the
toy section, R10 the toy section (the live section is the one the toy patch knows, version 27's is its fragment, its ids
kept), (c) their features at 390x844 and 1280x900 in light and dark: Who drives, version 25's Learner switch for all
three learners, version 24's axis titles, the newer "Strong test mix" tooltip, no status box on either page, the
rails), the toy section's own suite on the integrated page (`tests/toy/test_toy.mjs`: six views, 390 and 1280 px, light
and dark by the system and by `data-theme`; no sideways scroll, the worked blocks fitting, the four charts drawn with
both axes named, each chart's tooltip with a color key per line and inside the screen, links, stacked tables' labels,
touch targets, no console error; `tests/toy/nav_smoke.mjs`: the section inside the navigation, its 44 parts, links
into it), the navigation's T-1 to T-22 and extra checks (expected lists `nav/rules/pass2_v27.json`; T-20 checks the
static edits' anchors once in `stages_v27/s0_toy.html`), and last the timing tests (T-18, `timing_v27.mjs`: load at 390
px with a 4x CPU throttle against the live page, medians; the lesson's `perf.mjs`). The scripts of versions 20, 24 and
26 are kept as they were. Two of the navigation's tests needed the status box's removal taken into account: T-15
clicked the box's link to Results (it now clicks the header's process link and expects that link's own section;
`make_tests_v27.py` makes `setup_tests_v27.py` patch it), and T-13 can fail when the machine is heavily loaded (the
page's move back to the address's `#id` after a reload can come after the test's own scroll; REBASE.md section 16);
rerun it alone with `sh tests/nav/test/run_all.sh t_kbd` before reading a failure as a fault.

On a newer base page every patch applies as long as its anchors are still there exactly once, every block the
navigation replaces is a version it knows, and the toy section is the live page's own (or already the fragment); each
stops with an error naming the edit otherwise and writes nothing. If the other session edits the live toy section,
compare its edit with the fragment, carry it into `wf7/toy/build_toy.py` if it is still wanted, and only then pass
`--allow-unknown-base` to the toy patch (by hand; `apply_all.py` never passes it).

## Records

- `REBASE.md`: the rebases onto version 23, onto version 25 and onto the live page after version 25, and (part 4) the
  toy worlds' integration (inventories, conflicts, deliberate changes, status facts, test results).
- `FIX_LOG.md`: the fix rounds of the version-19 build.
- `wf7/toy/TOY.md`: the toy section's sources, re-runs, contents and every number with its notebook cell.
- `component_changes/`: the components before integration (`*.before_integration`), before fix round 1
  (`fix1_before/`), the integration diffs, `v19_based/`, `v23_based/`, `v25_based/` and `v27_pretoy/` (every component
  source before each rebase or integration).
- `fix1/edits/`: the edit scripts of fix round 1; `rebase/edits/`: the edit scripts of the three rebases and the toy
  integration; `rebase/work/`: their probes (`smoke_v26.mjs`: the Learner switch through the panel and the Choices);
  `toyint/`: the toy integration's probes (`anchors_in_toy.py`: which of our anchors lay inside the old toy section;
  `diff_pretoy.py`: the toy build against the pre-toy build outside the toy section; `toy_mentions.py`; the trial and
  negative builds in `toyint/work/`).
- `tests/runs/`: runs' outputs kept apart (`v24_run/`, `v26_first_run_1150/`, `v27_first_build_9beedc1f/`,
  `v27_final_build_7e0d6d90/`: the batteries on the pre-toy version 27; `v27_toy_build_2f277292/`: a copy of the
  battery's results on the toy build). The latest battery's own outputs stay where the tests write them
  (`tests/out/results_v27.json` and the files it lists).
