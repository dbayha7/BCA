# Rebases of this build: onto page version 23 (page version 24), onto page version 25 (page version 26), and onto the live page after version 25 (page version 27); the toy worlds' full results (page version 27)

Part 1 (sections 1-5) records the rebase onto version 23 (15-16 Oct 2026), part 2 (sections 6-10) the rebase onto
version 25 (16 Oct 2026), part 3 (sections 11-16, at the end) the rebase onto the live page after version 25: version 25
with its status box removed at the owner's request (16 Oct 2026), and part 4 (sections 17-22, at the very end) the
integration of the toy worlds' full results into page version 27 as its first stage (16 Oct 2026). The current build
is page version 27 with the toy worlds' results (part 4); part 3's numbers (sizes, SHA-256, ids, test results) are
those of version 27 before that integration.

# Part 1. Rebase onto page version 23 (page version 24)

The build that was called "version 20" (the lesson, the navigation, the colors and the status statements, made on
the page's version 19) is rebased onto the live page's version 23 (service version 1792101199-6eea,
`scratchpad/live4/core_6eea.html`, 5,137,780 bytes) as **page version 24**. Nothing that versions 20-23 added is lost:
every one of their element ids is kept, every line they added is in version 24 unchanged except the nine deliberate
changes listed below (R1-R9), and their features work in version 24 as in version 23 (the regression test,
`tests/regression_v23.mjs`). Nothing was published; the repository was only read.

Inputs: base `live4/core_6eea.html` (never edited), publish wrapper = first 531 bytes of `live4/live_6eea.html` plus
`"\n</body></html>"`. Outputs: `build/core_v24.html`, `build/page_v24_preview.html`, made by
`python -X utf8 build/apply_all.py ../../live4/core_6eea.html core_v24.html --preview page_v24_preview.html`
(`--version 24` and `--status-time 22:13` are the defaults). The version-19 build stays reproducible from
`build/component_changes/v19_based/` (178 component source files, SHA-256 list in `component_changes/v19_based.sha256`).

## 1. What versions 20-23 changed (inventory, from the diff of version 19 against version 23 and their history entries)

87 changed blocks, 914 distinct added lines (counted with repeats), 32 new element ids, none removed.

### Version 20 (two page comments)
- [x] **Observation Space guide**: `div.box.obs-gauge` with `h3#obs-gauge` "How to gauge these numbers" (rest, near rest,
  the size of a start, what a number costs, what the current does, float 1 in strong current as a worked example) and
  the key `p#ob-key` (hidden until the script fills it); CSS block "Observation Space (version 20)" (`.ob-bar`,
  `.ob-near`, `.ob-rng`, `.ob-zero`, `.ob-key`, `.ob-scale`).
- [x] **Range bars**: a script "Observation Space (version 20)" puts a range bar (`span.ob-bar`) under each float
  number's situation cell of `table.obs-tbl` (class `ob-on`), and a reading in words on the cell (`data-tt-read`,
  `data-tt-read-title`, and hidden `span.tt-sr` text) shown by the table-cell tooltip.
- [x] **Switch bar**: `div#simtog` "Switch while you watch" between the drawing and Play/Step, with its own `<style>`
  and script (IIFE): `#simtog-h`, the segmented controls `#simtog-l-sit`, `#simtog-l-world`, `#simtog-l-ctrl`; each
  button presses the matching big option (`#sim .opt`).
- [x] Text: the live panel's pipe note ("by default with fixed controllers"), the readouts' note.

### Version 21 (three page comments)
- [x] **Who drives** (`.simtog-ctl[data-g="who"]`, `#simtog-l-who`): fixed controllers or the learned actor; the
  learned rows `#simtog-l-lq` (data quality, with `.simtog-dot` in `--lq-1..5`), `#simtog-l-ld` (the learner's data),
  `#simtog-l-lm` (tested on), the note `details.simtog-qnote` "What a data quality is", `#simtog-now` "Now playing".
- [x] **Learned-actor replay**: `script#sim-learned-data` (about 340 KB, the 20 episodes of "The critic in action",
  speed numbers recovered by re-simulation), code `LD`, `LQ`, `LQVAR`, `learnedEp()`, `nowPlaying()`, `LTIP`,
  `simBaseTip()`, `LRO`, `modeTexts()`, `learnedUI()`, `readoutsLearned()`, `explainLearned()`, `drawCostLearned()`;
  hooks in `drawChannel`, `drawCost`, `readouts`, `explain`, `render`; `S.mode = "fixed"` on a big option's click.
- [x] Notes and legends that swap by mode: `#sim-optsnote` ("Not in use now"), `#ro-note-fixed`, `#ro-note-learned`,
  `#legend-fixed`, `#legend-learned` (five items, `--lq-1..5` swatches); key `.key.lqramp`; the drawing key's Frame,
  Words and Chart entries; tokens `--lq-1..5` on `:root` and both dark blocks; `#sim.sim-learned` CSS (frame `--lm-c`,
  explanation marks `--lq-c`).
- [x] The canvas tooltips: `channelTip` and `costTip` learned branches; `set()` falls back to `simBaseTip(cv.id)`;
  "≈" in the water's-push reading.
- [x] **The learning curves' table**: `h4#lc-th` "Every run in numbers", its own selectors `#lc-tctl` (`#lc-tq-l`,
  `#lc-tq` with `data-lc-tq` buttons, "All five side by side"; `#lc-tt-l`, `#lc-tt`), the five-quality table
  `#lc-t5box` > `table#lc-table5` (`#lc-cap5`, `#lc-h5`, `#lc-tbody5`); CSS for them; script `tctl()`, `table5()`,
  `view()`, `st.all`; quality names capitalized (`QN`).
- [x] **Choices rows**: "Who drives", the three learned-actor rows, "Table" (learning curves); rows of the mode not in
  use marked "not in use now" (`.sb-row.sb-idle`); CSS `.sb-tab.sb-tab-away`; a script that moved the Choices tab
  (`#sb-tab`) aside while `#simtog` or `#lc-tctl` was on screen.
- [x] The demo's "Watch one episode" link reads the panel's mode from `#sim`'s class.
- [x] "The critic in action": its start's current to 4 significant digits (0.3168, 0.7405; `s4`, `toPrecision(4)`).
- [x] The status box shortened (one or two sentences per line).

### Version 22 (one page comment)
- [x] **Arm tables**: `table.rows8[data-arms="stall" | "no_stall" | "replication"]`, rows `tr[data-arm]`, cells with
  `data-tt-read` readings (each score compared seed by seed); three notes `aside.note.tt-howto` "How to read the
  tooltips"; data `script#arm-seeds`; a script "Situation by situation (version 22)".
- [x] Shared tooltip script: `.tt-read` (a cell's reading under its row, CSS `white-space: pre-line`), and an arm
  table's tooltip placed beside its cell when there is room (`cur.closest("table[data-arms]")` in `place()`); its doc
  comment; CSS `.subsec > aside.tt-howto`.

### Version 23 (two page comments)
- [x] Two notes above exploratory result A's coverage table (inside `div#iw-plain.finding`): `p#iw-plain-split` "Why the
  strong test is split in two", `p#iw-plain-refuse` "Is "refuses all" a bad thing?"; the table's column tooltips (in
  `COLS`, the groups "Strong test, steps outside / inside the removed region").
- [x] The status box rewritten as "Status as of 15 Oct 2026 (page version 23)"; version-history entries Version 20 to 23.

## 2. Conflicts and how each was merged

| Where | Conflict | Merge |
|---|---|---|
| Color patch K14 (canvas tooltips' `set()`) | version 21 rewrote it (`simBaseTip`) | K14 re-anchored on version 21's `set()`; it keeps the `simBaseTip` fallback and adds the color keys (third element) |
| Color patch B1-B4, B20 (status box) | version 23 rewrote the box | B1 gives it one status time and page version 24; B2 rewrites the Stage 2 line (approved, not "waits for the owner's approval"); B4 adds two lines after the toy worlds' line (four-learner study, near-80 draft); B20 adds the weighting test's place in the repository to its pre-registered line; B3 dropped (version 23's box no longer speaks of the toy notebooks; its line is true; the Toy worlds section keeps B13-B15) |
| Navigation E10 (static edits) | version 22 added `data-arms` to the three arm tables | the three id edits re-anchored on the new opening tags (`nav/rules/static_edits_v23.py`) |
| Navigation E1 (old sidebar) | version 21 put a script (moves `#sb-tab` aside) between `</aside>` and `.wrap` | E1's end anchor is now `.wrap`; the script is removed with the tab it served (checked: only that script may sit there). Its intent is kept (`rebase/edits/e_nav_v24b.py`): below 1000 px the navigation's floating Choices button steps aside (hidden unless focused) while it would overlap `#simtog` or `#lc-tctl` on screen, measured on scroll and resize only while it is shown |
| Navigation E8 (old sidebar script -> nav.js) | versions 21-22 added rows and the idle marking to the old script | nav.js carries them: "Who drives", Data quality (with the quality's colored dot), The learner's data, Tested on, Table; `idle` rows marked "not in use now" (class `nv-idle`, same words and look as `sb-idle`) |
| Navigation rules (nav.js `detect`, rules.js) | version 23's notes sit inside a finding, which was one entry | a part with `data-nav-label` inside a listed finding, check or box is listed one level below it |
| Navigation entries | new parts | entries for `#obs-gauge` (automatic: a box), `#simtog`, `#simtog-who` (new id), `#ats-howto-stalling`, `#ats-howto-no-stall`, `#ats-howto-replication` (new ids), `#iw-plain-split`, `#iw-plain-refuse`; "Every run in numbers" moves from `#lc-table` to `#lc-th` (it stays on screen in both table views); the cost chart's entry is named "Cost so far: the controllers, or the learned actors" |
| Version 20's switch-bar comment | named the walkthrough pop-out and the old sidebar | new navigation edit E14 rewrites the comment |
| Shared tooltip script | version 22's `tt-read` and beside-the-cell placement; our K9/K10 (keys) and E13 (Tab stops) | no textual overlap: K9 is in the `data-tip` path of `build()`, version 22's code in the table-cell path and in `place()`; E13 in `fixStops()`; all kept |
| Colors | version 21's learned-actor chart | K13b (keyed readout lines in `--lq-1..5`), K22a-e (legend keys), N7b (no 80% opacity, see R7) |
| Version history | one entry per version | I1 inserts "Version 24, 15 Oct 2026." above version 23's entry and checks no entry for 24 existed; the checks require exactly one entry per version number (1 to 24) |
| Lesson | none | the lesson patch applies to version 23 unchanged |

## 3. Deliberate changes to version 23 content (each checked by the regression test, part b)

| Id | Version 23 text | Version 24 | Why |
|---|---|---|---|
| R1 | `Status as of 15 Oct 2026 (page version 23)` | `Status as of 15 Oct 2026, 22:13 UTC (page version 24)` | one status time for the whole box (B1) |
| R2 | Stage 2: the run on fresh seeds 5-9 "waits for the owner's approval" (and its tooltip: "the owner's approval ... comes next") | "has not started: the owner typed their approval at their own terminal on 15 Oct at 15:45 UTC, and the approval's check and record come next, then the run" | out of date: `approval_v7_3dfcb6140e2b8e2c.json`, approved_at_utc 15:45:42, commit de4851c (15:49 UTC) (B2) |
| R3 | weighting test (pre-registered) line ends at "(The weighting test)." | adds "Its code, specification and exploratory results have been in the shared repository since 15 Oct at 17:18 UTC; its run records are not." | pull request 3, 03726de (B20) |
| R4 | `.sb-row.sb-idle` CSS, `.sb-tab.sb-tab-away`, the coarse-pointer `.sb-b`/`.sb-x` rule | gone with the old sidebar; `.nv-row.nv-idle` rules give the new rows the same marking | E1 |
| R5 | the script that moved `#sb-tab` aside (its `(function () {` line included) | gone; nav.js `pillAway()` and the CSS `html.nv-pill-away .nv-pill:not(:focus) { visibility: hidden; }` do the same for the navigation's Choices button | the tab is gone (E1); the button steps aside only while it would actually cover the switch bar or the table selectors, so it stays usable when they are elsewhere on screen |
| R6 | switch-bar comment: "(the walkthrough pop-out moves the panel)", "the choices sidebar" | "... found afresh on every use ... the panel's choices under "In this section" ..." | both removed by the navigation (E14) |
| R7 | learned actors not on screen drawn at `globalAlpha` 0.8 | 1 (thin lines, as the fixed controllers since N7) | at 80% opacity the random-data line falls to 2.45:1 (light) and 2.78:1 (dark) against the panel, below 3:1; at full strength 3.20:1 and 3.61:1 (N7b) |
| R8 | learned cost readout as one sentence | one line per data quality, each with its line's color key (`data-tip-keys`), the sentence last | the color keys in hover readouts, as for the fixed controllers (K13b) |
| R9 | the old sidebar script's rows (Who drives, idle rows, Table) | the same rows in nav.js | E8 |

Not text changes: attributes added to version 20-23 opening tags (navigation ids and labels on `#simtog`, the "Who
drives" row, the arm tables, the three notes, the two version-23 notes, `#lc-th`; color keys `data-tip-key` on the
five `#legend-learned` items). The colors versions 20-23 chose (`--lq-1..5`, the range bars' situation colors, the
switch bar's look) are unchanged; only R7's opacity changed.

Status statements outside version 20-23 text (version 19 text, updated by the color patch with the new status time):
Results' opening (B5, and new B22: the two planned lines), After the stop's status box (B6: read from the repository as
last fetched: shared branch 7124be0, forked cab5b3e, main 1ec267e; B7; new B21: the near-80 draft), the first claim
(B8, B9), After the stop's sources (B10: head 7124be0; B11), the toy worlds (B13-B15), the weighting test (B16: "after
this page's status was read" dropped, the status time being after 17:18 UTC; B17-B19), I2.

## 4. Status statements (as of 15 Oct 2026, 22:13 UTC)

Read (read only, `GIT_OPTIONAL_LOCKS=0`) from `C:/Users/David/dev/BCA` at 23:16 UTC: remote-tracking refs as fetched
from GitHub at 22:13 UTC (`.git/FETCH_HEAD`); the local branches hold nothing newer.
- Stage 2 version 7: handed off 07:47 UTC (3a8ce5b); approved by the owner: `approval_v7_3dfcb6140e2b8e2c.json`,
  `approved_at_utc` 2026-10-15T15:45:42Z, commit de4851c (15:49 UTC) on `forked` (head cab5b3e, 21:45 UTC) and on main
  through pull request 5 (1ec267e, 21:36 UTC). No import record of the approval; no run on seeds 5-9 (no result files
  in any ref; `handoff/HANDOFF.json` lists every seed 5-9 phase "not started"). 0de1e50 (17:16 UTC) only adds a
  temporary helper that runs the approval's `--check`.
- The weighting test's code, specification and exploratory results: main since 17:18 UTC (pull request 3, 03726de); no
  `runs/` on main.
- Near-80 test: draft pre-registration, "nothing trained", no approval file (7124be0, 18:50 UTC, branch
  claude/main-bca-connection-x4o46s). One plain sentence in the status box, one line in After the stop's box, one
  clause in Results' opening.
- Four-learner study: not in any ref (its code is in WSL, which this build does not read); stated as the owner's request
  gives it: plan approved in chat on 15 Oct 2026, code built and tested on made-up numbers only, the owner's approval
  of the frozen plan at their own terminal before any run, nothing trained on ThrusterChain data.

## 5. Test results

Built page: `build/core_v24.html` 6,632,841 bytes (SHA-256 `768302de…457c`), `build/page_v24_preview.html` 6,633,387
bytes (`0a141299…bbc8`, under the 16 MB limit); 3,442 ids (2,776 MathJax glyph ids), 0 duplicates, 0 CR bytes, 4,565
SVG `<use>` references resolve, one version-history entry per version 1 to 24. The rebase's component chain was
replayed from `component_changes/v19_based/` in a scratch copy: every component file came out byte-identical (the
static edits differ only in the source path they record).

`tests/run_v24.sh` (15-16 Oct 2026, 00:52 to 01:37 UTC; one headless Edge at a time, the timing tests alone at the end),
collected by `tests/collect_v24.py` into `tests/out/results_v24.json`: **42 checks (the file holds 42 lines; "44" was written here first by mistake), all pass, no stale result file.**

- Static (`static_v24.py`): no external resource beyond the Google Fonts already linked, no CSS URL, in-page links
  only as bare `#id`, 0 CR, every new or edited script an IIFE (or JSON), localStorage only inside try, no
  alert/confirm/prompt/print, no color literal outside a token in any CSS line version 24 adds, no new British spelling,
  no duplicate id, no broken link or aria reference, the old navigation and walkthrough gone, versions 20-23's parts
  present.
- The lesson: 4 views (390 and 1280, light and dark): 39 steps walked forward and back, 58 formulas and legends, the
  panels (live panel, task, critic viewer, learning curves, demo levels) working; 176 of 176 controls, 221 worked lines
  checked by rule D2 (0 bad); 1,546 random slider states (0 bad); without `hidden="until-found"` and with reduced motion,
  with and without the navigation.
- Colors (`verify_render.mjs`, 4 views with the Whole page switch on): 371 checks passed, 0 failed (tokens, marks,
  keys on every readout line that names a series, tooltips inside the screen, no sideways scroll, every legend item
  marked); the learning curves' tokens and strokes identical to version 23's render. Text shots: the edited status
  text found, the new term tooltips (CQL, IQL, ReBRAC, Unifloral, Weighted BCA) open inside the screen at 1280 and 390.
- The old full check (`check3_v24.mjs`): 4 views, each of the 25 sections shown in turn without sideways overflow, NaN or
  console errors; the whole page clean (ids, links, aria, requests); version 24's entry and versions 20-23's present;
  check3's text probes and the Design effect tooltip as in version 23.
- **Regression against version 23 (`regression_v23.mjs`)**: (a) all 2,138 ids of version 23 are in version 24 except the
  18 removed on purpose (the old sidebar's 6, the walkthrough button and dialog's 12); (b) all 914 lines versions
  20-23 added are in version 24 unchanged (once this build's added attributes are set aside), except 40 lines that are
  R1-R9 above, each with its replacement present; (c) at 390x844 and 1280x900, light and dark: Who drives for the
  random and expert qualities on the training and strong test mixes (same "Now playing", readouts and explanation as
  version 23, drawings and the learned legend shown), the switch bar (the big options follow), the cost readout's five
  keys, back to the fixed controllers, the Choices rows and their "not in use now" marking (same rows as version 23's
  sidebar), the learning curves' Choices rows (version 23's plus this build's), the range bars (same count, key shown,
  same reading in the tooltip), the arm tables' readings (same text; beside or above the cell exactly as in version 23,
  never covering it when there is room), the learning curves' table and selectors ("All five side by side", the test
  switch, back), version 23's notes and the coverage table's column tooltip, the floating Choices control stepping
  aside over the switch bar (version 23's tab, version 24's button), no console errors, no sideways scroll.
- The navigation (T-1 to T-22 and extras): 13 widths in both themes; 25 sections and 278 entries as expected (version
  23's rules output with the static edits, plus the lesson's 40); 535 links at 390 and 1280; Previous and Next through
  all 278 stops both ways; deep links at load; Back and Forward; redraws at the shown width (compared with version 23)
  and table cues; loops stop when their section is left; 7 Choices groups, 22 rows, 73 buttons (Who drives, the
  learned-actor rows and Table included); keyboard; accessibility attributes; touch targets; Whole page; print;
  without `hidden="until-found"`; failure fallbacks; without the lesson; reduced motion; screenshots without console
  errors; every entry by click, fragment and deep link.
- Load time at 390 px with a 4x CPU throttle (alternating loads; noisy, a shared machine): `timing_v24.mjs`, 5 loads
  each, medians: DOMContentLoaded **4,075 ms (version 24) against 11,131 ms (version 23)**, load event 4,094 against
  11,376 ms, main-thread task time 4.33 s against 11.8 s; version 24 lays out only the section on screen. The
  navigation's T-18 (3 loads each): 4,037 against 14,641 ms, its own start at most 37 ms. Unthrottled (`perf.mjs`, 2 x 3
  loads): about 0.7-0.9 s against 1.8-3.2 s.

Test-side fixes during the run (paths and names only, never a check's intent): `t_panels.mjs` found the learning
curves' quality buttons by lower-case name, which version 21 capitalized ("Medium", "Random"), so the match is now
case-insensitive (recorded in `setup_tests_v24.py`; T-9 re-run alone, passed); in `regression_v23.mjs` the arm-table
check at 1280 px asked for "beside" where version 22 places the tooltip above the cell because it does not cover it
there (now: never covers its cell, as version 23), and the cost readout's keys are read while the readout shows (they
were read after the pointer left, a race). The regression test was re-run after these fixes (passed) before the timing
tests started.

---

# Part 2. Rebase onto page version 25 (page version 26)

The build of page version 24 (made on version 23) is rebased onto the live page's version 25 (service version
1792113618-4933, `scratchpad/live5/core_4933.html`, 5,857,239 bytes, 12,973 lines), published by another Claude session,
as **page version 26**. Nothing that versions 20-25 added is lost: every one of their element ids is kept (except the 18
this build removes on purpose, the same as in part 1), every line they added is in version 26 unchanged except the
deliberate changes listed below (R1-R9, section 8), and their features work in version 26 as in version 25 (the
regression test, `tests/regression_v25.mjs`). Nothing was published; the repository was only read; nothing was trained.

Inputs: base `live5/core_4933.html` (never edited), publish wrapper = first 531 bytes of `live5/live_4933.html` plus
`"\n</body></html>"`. Outputs: `build/core_v26.html`, `build/page_v26_preview.html`, made by
`python -X utf8 build/apply_all.py ../../live5/core_4933.html core_v26.html --preview page_v26_preview.html --stages stages_v26`
(`--version 26`, `--status-date "16 Oct 2026"` and `--status-time 12:39` are the defaults). The version-24 build stays
reproducible from `build/component_changes/v23_based/` (206 component source files, SHA-256 list in
`component_changes/v23_based.sha256`); its `apply_all.py`, run on `live4/core_6eea.html` with
`--wrapper live4/live_6eea.html`, gives `core_v24.html` byte for byte (checked on 16 Oct).

Work notes, probes and the trial runs are in `scratchpad/live5/work/` (`diff_v23_v25.txt`, the trial stages in
`trial1/`, `span_compare.py`, `backup_v23_based.py`, `gen_e_apply_all_v26a.py`, `replay_v26.py`).

## 6. What versions 24 and 25 changed (inventory, from the diff of version 23 against version 25 and their history entries)

100 changed blocks; 298 lines added (272 distinct) and 90 removed; the page grew by 719,417 characters, almost all of
them the data block `script#sim-rerun-data` (694,999 characters). New element ids: `simtog-l-lr`, `simtog-lr-q`,
`exLrnT`, `exLrn`, `sim-rerun-data`; none removed. Versions 24 and 25 add no part that the navigation lists: the
navigation's rules give the same 238 entries on version 25 as on version 23 (`nav/rules/pass2_v25.txt` equals
`pass2_v23.txt`).

### Version 24 (one page comment): every chart names both of its axes
- [x] **Canvas titles**: the live panel's cost chart (`canvas#cost`, `drawCost` and `drawCostLearned`: the x title
  "State t of the episode (0 to 99)" or "State t (0 to 99)" under the tick numbers, height 216, bottom margin 40) and the
  throttle gauges (`canvas#channel`: one title line `gTtl`, "Throttle (bar) and push (tick) of each thruster, from −1 at
  the top ... to 1 at the bottom", the longest that fits; `gaugeTop` 46 px lower, in `drawChannel` and in the task
  demo's drawing).
- [x] **"The critic in action"** (`#qv-svg1`, `#qv-svg2`): rotated y titles (`fitTitle`, `yTitle`: "Value change from the
  actor's command", "Value (discounted return)", "Value of the actor's command"), x titles "Thruster 1's throttle
  upstream, % ..." and "State t of the episode (0 to 99)", left margins 76-78; the aria-labels name both axes.
- [x] **"Try a command" strips** (`svg.aw-strip`, `#aw-s1..4`): 114 px high (a row for the x title), the y title "Recorded
  rows per 4% bin (up to N)" and the x title "Throttle, % of full (negative = upstream)" (each the longest wording that
  fits, `fit()`); the strip's text reading names both axes.
- [x] **The weighting test's five charts** (`#iw-svg-dens`, `#iw-svg-w`, `#iw-svg-thrust`, `#iw-svg-bars`,
  `#iw-svg-cov`): `titles()` puts an x title under the ticks and a y title across the top left; taller charts (248-264);
  aria-labels with "Horizontal axis: ... Vertical axis: ...".
- [x] **The learning curves**: the two large charts (`#lc-svg1`, `#lc-svg2`: `axisWords()` and `fit()`, the aria-label
  names both axes) and the small charts of the overview (`#lc-grid`: "Points" and "Training steps" with the end ticks 0
  and 20k; cell height 132; the button's label names both axes); CSS `#iw-curves .lc-cell svg .axlab`; the note above
  the overview.
- [x] **Other charts' titles reworded the same plain way**: push and throttle (Action Space), thruster 1 state by state
  (`.tc-phone`, `.tc-narrow`, `.tc-wide`), the recorded episode, the caution multiplier, the stall chart (`#sc-svg`: titles
  fitted, `.sc-svg text.ix-ttl` on a halo), the Stage 1 dot plot (`#beyond-chart`: the head row `div.dr-axis.dr-head` with
  "Arm" and "Mean", CSS `.dr-head`).
- [x] Version-history entry "Version 24, 15 Oct 2026." (the other session's).

### Version 25 (one page comment): the Learner switch
- [x] **The Learner row** in the switch bar's learned mode: `div.simtog-ctl[data-g="lr"]` with `span#simtog-l-lr`
  "Learner" and its hint `span#simtog-lr-q`, three buttons `data-tog-group="lr"` (`td3bc`, `bca`, `bcaiw`) with their own
  tooltips; its CSS (three in a row at every width, two lines kept for the label in a narrow bar).
- [x] **The cloud rerun's data**: `script#sim-rerun-data` (schema `thrusterchain-sim-rerun-v1`; the start-1 episodes of
  TD3+BC, BCA and BCA IW on the five data qualities and both mixes, with the 80% hold removed; each learner's average
  over each mix's 1,000 starts, `avg`; the desktop run's full-data TD3+BC averages, `desk`; notes: exploratory, iw_rerun
  of 15 Oct 2026, 22:32:50 UTC, reproduced within the desktop run's own spread, not bit for bit).
- [x] **The code**: `S.lr`, `RD`, `LRNAME`, `lrOK()`, `isRerun()`, `LRT`, `RUNT`, `LRTIP_OFF` (a Learner button that cannot
  play says why), `avgOf()`, `learnedEp()` reading the rerun when the 80% hold is removed, `nowPlaying()` naming the
  learner and the run, `nowMore()` (the run, this episode's return beside the learner's average), `ltipOf()` (the cost
  chart's general tooltip for the rerun), `LRO_RERUN_J`, `modeTexts()` (the legend items' tooltips per learner, the cost
  chart's aria-label per learner), `learnedUI()` (BCA and BCA IW disabled with all the data, the learner back to TD3+BC),
  `explainLearned()` (the explanation's new line `dt#exLrnT`/`dd#exLrn` "Learner", the bank line with the learner's
  average), `drawCostLearned()` (the title names BCA or BCA IW), the drawing's label, the cost readout naming the
  learner and the cloud rerun.
- [x] **Text**: "Who drives" button "Learned actor" (was "Learned actor (TD3+BC)"), with its tooltip; the panel's notes
  (pipe note, "What a data quality is", the readouts' note, the learned legend's entry, the figure caption's "Learner"
  paragraph).
- [x] **The old Choices sidebar's script**: "Learned actor" and a new row "Learner" (marked "not in use now" while the
  fixed controllers play).
- [x] The status box's heading "Status as of 16 Oct 2026 (page version 25)" (its lines are version 23's); the
  version-history entry "Version 25, 16 Oct 2026.".

## 7. Conflicts and how each was merged

| Where | Conflict | Merge |
|---|---|---|
| Color patch K13b (the learned actors' cost readout) | version 25 rewrote the readout: it names the learner ("for the BCA actor learned from ...") and the cloud rerun | K13b re-anchored on version 25's code: one keyed line per data quality ("Random data: 380.30"), each with its line's color (`--lq-1..5`), and version 25's sentence as the last line with the learner named ("All five BCA actors learned from data with the 80% hold removed, in the cloud rerun of seed 0, and start from ..."); version 25's `nm` and `rr` kept |
| Color patch B1 (status heading) | version 25's heading "Status as of 16 Oct 2026 (page version 25)" | B1 re-anchored: "Status as of 16 Oct 2026, 12:39 UTC (page version 26)", one status time for the box |
| Color patch B2, B4-B11, B21, B22 (status statements) | the status time moved from 15 Oct 22:13 to 16 Oct 12:39 UTC; newer facts in the git refs | re-read and reworded (section 9): the approval imported (B2, B5, B7, B9, B11), the shared branch's head 0377dd7 (B6, B10), the near-80 test's final plan approved by the owner and the approval recorded (B4, B21, B22: "Two more have no results yet"), the four-learner study's code "built and under review" (B4); a placeholder `{D}` for the status date (`--status-date`) |
| Status box | version 25's box keeps version 23's statements | kept; B2 updates the Stage 2 line, B20 adds the weighting test's place in the repository, B4 adds the four-learner and near-80 lines, new B23 adds a line on version 25's rerun of BCA and BCA IW (exploratory, one seed; its numbers recomputed by `static_v26.py` from the page's own data) |
| Color patch (charts) | version 24's axis titles in the same charts and drawing code as the color patch's keys and legends | no textual overlap (the trial run applied 92 of version 24's 94 edits unchanged; only K13b and B1 needed new anchors; B23 is new); the titles use version 24's own tokens (`--ink-2`, `.axlab`, `.ix-ttl`); the regression test checks every chart's titles (same number as version 25, inside the chart, each one of the page's own wordings) |
| Navigation E8 (old sidebar script -> nav.js) | version 25 renamed "Learned actor (TD3+BC)" and added the row "Learner" to the old script, which E8 replaces as a whole: without a port the change would have been lost silently | nav.js carries both (`e_nav_v26.py`): "Learned actor", and "Learner" (TD3+BC, BCA, BCA IW) before "Data quality", `idle: simFixed` ("not in use now" while the fixed controllers play); a disabled panel button (BCA, BCA IW with all the data) is disabled in the Choices too, as before |
| Navigation, every block edit (E11, E7a-c, E1, E2, E4, E5, E6, E8) | a block replaced as a whole hides any change a newer base makes inside it | `apply_nav.py` now checks the SHA-256 of the text each block edit removes against the versions it knows (`KNOWN`: version 23's and version 25's; they differ only in E8) and stops otherwise; `live5/work/span_compare.py` compared every removed block of version 23 and version 25 (only E8 differs) |
| Navigation entries | new parts? | none: the rules list the same 238 entries on version 25 as on version 23; the Learner row, like version 21's learned rows (data quality, the learner's data, tested on), is shown only in the learned mode and has no entry of its own: it is offered in the Choices under "Watch one episode" |
| Lesson step 7.2 | its status sentence said the approval's check and record were still to come | `build_lesson.py` reads the approval's import record (`e_lesson_v26.py`); the lesson rebuilt; only that sentence changed |
| Version history | version 25's entries 25 and 24 (the other session's), our old "Version 24" entry | I1 inserts "Version 26, 16 Oct 2026." on top, its text rewritten from our old version-24 entry (built on version 25; the Choices with the rows of versions 21, 22 and 25; the keys beside version 24's axis titles; the status as of 16 Oct); no second "Version 24" entry; the checks require exactly one entry per version number (1 to 26) |
| Integration checks | version 24's statements and parts | updated for version 26 (`e_apply_all_v26a.py`, `e_apply_all_v26.py`); new: versions 24-25's code and parts must occur in OUT as often as in BASE (23 markers: `axisWords`, `yTitle`, `titles()`, `dr-head`, the canvas titles, the Learner buttons, `sim-rerun-data`, `nowMore`, `ltipOf`, `LRO_RERUN_J`, `LRTIP_OFF`, `avgOf`, `lrOK`, "Horizontal axis: ", "Vertical axis: ", `fitTitle(` ...), version 25's Learner row in the Choices |
| Colors for the Learner switch | does the switch show BCA and BCA IW in one chart? | no: the cost chart shows the five data qualities' actors of the chosen learner (`--lq-1..5`, version 21's colors, keyed by K13b and K22a-e), never two learners side by side, so no learner needs a color of its own; the learner is named in the chart's title, its aria-label, its readout and its legend tooltips |

## 8. Deliberate changes to version 25 content (each checked by the regression test, part b)

| Id | Version 25 text | Version 26 | Why |
|---|---|---|---|
| R1 | `Status as of 16 Oct 2026 (page version 25)` | `Status as of 16 Oct 2026, 12:39 UTC (page version 26)` | one status time for the whole box (B1) |
| R2 | Stage 2: the run on fresh seeds 5-9 "waits for the owner's approval" (and its tooltip: "the owner's approval ... comes next") | "is cleared to start: the owner typed their approval at their own terminal on 15 Oct at 15:45 UTC, and it was checked and recorded on 16 Oct at 02:27 UTC; no result of the run is in the repository yet" | out of date: approval `approval_v7_3dfcb6140e2b8e2c.json` (de4851c), import record `import_v7_3dfcb6140e2b8e2c.json` (02e591b, imported_at 02:27:10 UTC) (B2) |
| R3 | weighting test (pre-registered) line ends at "(The weighting test)." | adds "Its code, specification and exploratory results have been in the shared repository since 15 Oct at 17:18 UTC; its run records are not." | pull request 3, 03726de (B20) |
| R4 | `.sb-row.sb-idle` CSS, `.sb-tab.sb-tab-away`, the coarse-pointer `.sb-b`/`.sb-x` rule | gone with the old sidebar; `.nv-row.nv-idle` rules give the new rows the same marking | E1 |
| R5 | the script that moved `#sb-tab` aside | gone; nav.js `pillAway()` and `html.nv-pill-away .nv-pill:not(:focus) { visibility: hidden; }` do the same for the navigation's Choices button | the tab is gone (E1) |
| R6 | switch-bar comment: "(the walkthrough pop-out moves the panel)", "the choices sidebar" | "... found afresh on every use ... the panel's choices under "In this section" ..." | both removed by the navigation (E14) |
| R7 | learned actors not on screen drawn at `globalAlpha` 0.8 | 1 (thin lines, as the fixed controllers since N7) | at 80% opacity the random-data line falls below 3:1 against the panel (N7b) |
| R8 | version 25's learned cost readout as one sentence, naming the learner in each part | one line per data quality with its line's color key (`data-tip-keys`), version 25's sentence last, the learner named there ("All five BCA actors learned from ...") | the color keys in hover readouts (K13b); the values are the same (checked against version 25 for all three learners) |
| R9 | the old sidebar script's rows (Who drives with "Learned actor", the idle rows, version 25's "Learner", Table) | the same rows in nav.js | E8 |

Not text changes: attributes added to version 20-25 opening tags (navigation ids and labels; color keys
`data-tip-key` on the five `#legend-learned` items, which version 25's `modeTexts()` leaves alone: it sets only
`data-tip`). The colors versions 20-25 chose are unchanged; only R7's opacity changed. New text of version 26 in the
status box: B4's two lines (four learners; the near-80 test) and B23's line (version 25's rerun of BCA and BCA IW).

## 9. Status statements (as of 16 Oct 2026, 12:39 UTC)

Read (read only, `GIT_OPTIONAL_LOCKS=0`) from `C:/Users/David/dev/BCA` after 13:23 UTC: remote-tracking refs as fetched
from GitHub at 12:39 UTC (`.git/FETCH_HEAD`, 08:39:13 -04:00, fetched by another session; this build ran no fetch). A
first reading at 12:01 UTC (refs fetched at 11:50 UTC) gave the status time 11:50 UTC and a first full test run
(`tests/runs/v26_first_run_1150/`); the owner approved the near-80 plan at 11:50:43 UTC, which that reading could not
see, so every status statement was re-read at 12:39 UTC, the page rebuilt and the whole battery run again (section 10).
- Stage 2 version 7: handed off 15 Oct 07:47 UTC (3a8ce5b); approved by the owner (`approval_v7_3dfcb6140e2b8e2c.json`,
  approved_at_utc 2026-10-15T15:45:42Z, de4851c, 15:49 UTC); main merged it through pull request 5 (1ec267e, 21:36 UTC);
  the shared branch merged main (9765564, 16 Oct 02:26 UTC) and holds the **import record**
  `import_v7_3dfcb6140e2b8e2c.json` (02e591b, 16 Oct 02:27 UTC; imported_at 02:27:10Z; "Version 7 can now run on the
  Framework Desktop (step 12)"). No result of the run on seeds 5-9 in any ref (no result files; `handoff/HANDOFF.json`
  lists every seed 5-9 phase "not started"; nothing in `stage2_v7/` changed after 02e591b). The owner's local 810ecc6
  (16 Oct 03:03 UTC), a second approval file of the same plan typed at 03:00:52 UTC and never imported, reached GitHub
  in the merge 0377dd7 (12:38 UTC), which kept origin's copy, the one the import record pins; it changes no statement
  and is not on the page.
- Near-80 test: final pre-registration "ready for the owner's approval" (e3e87c3, 16 Oct 01:33 UTC, fingerprint
  51e677ab76c8d2f7; after re-freezes 55226bb, cf56ca6, d44fc1d, f8fb40b from 23:06 UTC on 15 Oct); 0ebe036 (01:52 UTC)
  adds copies of its pinned files; **approved by the owner** at their own terminal
  (`approval_near80_51e677ab76c8d2f7.json`, approved_at_utc 2026-10-16T11:50:43Z, f79b89c, 11:51 UTC) and the approval
  **recorded** (`import_near80_51e677ab76c8d2f7.json`, imported_at 11:52:58Z, 287380b, 11:53 UTC; "Every near-80 phase
  refuses until this record is on the branch on GitHub"). No near-80 result in any ref. One line in the status box
  ("Near-80 test: approved"), one in After the stop's box, one clause in Results' opening ("Two more have no results
  yet").
- The shared branch's head: 0377dd7 (16 Oct 12:38 UTC), the merge above (After the stop's box and sources).
- The weighting test: its code, specification and exploratory results on main since 15 Oct 17:18 UTC (pull request 3,
  03726de); no `runs/shift_iw*` in any ref.
- Version 25's cloud rerun of seed 0 (iw_rerun, 15 Oct 22:32:50 UTC): in no ref (no `iw_rerun` or `REPRO.json`); its
  numbers are the page's own (`script#sim-rerun-data`, `avg`). Strong test mix, weighting-test points, random to
  expert: TD3+BC 65.29, 77.63, 86.60, 92.36, 90.26; BCA 54.66, 72.10, 83.07, 92.42, 90.44; BCA IW 53.76, 72.05, 82.23,
  91.76, 90.38. So BCA is 10.6, 5.5 and 3.5 points below TD3+BC with the three weaker qualities and 0.1 and 0.2 above it
  with the two stronger; BCA IW is 0.05 to 0.90 points below BCA. One seed, exploratory; version 25's own note says such
  small differences can change sign between runs.
- Four-learner study (Unifloral's TD3+BC, CQL, IQL and ReBRAC, each alone, with standard BCA and with importance-weighted
  BCA): not in any ref (its code is in WSL, which this build does not read); stated as the owner's request gives it:
  plan approved in chat on 15 Oct 2026; code built and under review, tested on made-up numbers only; the owner's
  approval of the frozen plan at their own terminal comes before any run; nothing has trained on ThrusterChain data.
- Unchanged and still true: main 1ec267e and forked cab5b3e as at version 24's status time; the toy notebooks on main
  (B13-B15); origin/claude/bca-findings-walkthrough (d2d86fb, 11:42 UTC) is the findings notebook's lesson, about which
  the page says nothing.

## 10. Test results (page version 26)

Built page: `build/core_v26.html` 7,354,642 bytes (SHA-256 `8f829021…0ceba`), `build/page_v26_preview.html` 7,355,188
bytes (`bd9ba674…51373`, under the 16 MB limit); 3,447 ids (2,776 MathJax glyph ids), 0 duplicates, 0 CR bytes, 4,565
SVG `<use>` references resolve, one version-history entry per version 1 to 26 (26 on top, then the other session's 25
and 24). The trial run on version 25 (`live5/work/trial1/`) applied 92 of version 24's 94 color edits, the whole navigation and
the lesson unchanged; K13b and B1 failed on their anchors and were ported (section 7); E8's silent loss of version 25's
sidebar rows was found by comparing every removed block (`live5/work/span_compare.py`). The rebase's component chain was
replayed from `component_changes/v23_based/` in a scratch copy (`live5/work/replay_v26.py`): every component source came
out byte-identical; the lesson's rebuilt fragment differs from version 24's only in step 7.2's sentence.

`tests/run_v26.sh` (16 Oct 2026, 12:31 to 13:23 UTC; one headless Edge at a time, the timing tests alone at the end; the
tested files' SHA-256 are the ones above), collected by `tests/collect_v26.py` into `tests/out/results_v26.json`:
**42 checks, all pass, no stale result file.**

- Static (`static_v26.py`): no external resource beyond the Google Fonts already linked, no CSS URL, in-page links
  only as bare `#id` (427), 0 CR, every new or edited script an IIFE (or JSON), localStorage only inside try, no
  alert/confirm/prompt/print, no color literal outside a token in any CSS line version 26 adds, no new British spelling,
  no duplicate id, no broken link or aria reference, the old navigation and walkthrough gone, versions 20-25's parts
  present; the status box's rerun line recomputed from `script#sim-rerun-data` (BCA minus TD3+BC −10.6, −5.5, −3.5,
  +0.1, +0.2; BCA IW minus BCA −0.90 to −0.05; the run's date and learners).
- The lesson: 4 views (390 and 1280, light and dark): 39 steps walked forward and back, 58 formulas and legends, the
  panels working; 176 of 176 controls, 221 worked lines checked by rule D2 (0 bad); 1,546 random slider states (0 bad);
  without `hidden="until-found"` and with reduced motion, with and without the navigation.
- Colors (`verify_render.mjs`, 4 views with the Whole page switch on, version 25 as the learning curves' reference):
  371 checks passed, 0 failed. Text shots: the edited status text found (the rerun line included), the new term
  tooltips (CQL, IQL, ReBRAC, Unifloral, Weighted BCA, BCA IW) open inside the screen at 1280 and 390, no console error.
- The old full check (`check3_v26.mjs`): 4 views, each of the 25 sections shown in turn without sideways overflow, NaN or
  console errors; the whole page clean (ids, links, aria, requests); version 26's entry and versions 20-25's present;
  check3's text probes and the Design effect tooltip as in version 25.
- **Regression against version 25 (`regression_v25.mjs`)**: (a) all 2,143 ids of version 25 are in version 26 except the
  18 removed on purpose (the old sidebar's 6, the walkthrough button and dialog's 12); (b) all 1,137 lines versions
  20-25 added are in version 26 unchanged (once this build's added attributes are set aside), except 42 lines that are
  R1-R9 above, each with its replacement present; (c) at 390x844 and 1280x900, light and dark, all 21 checks pass:
  Who drives for the random and expert qualities on both mixes (same "Now playing", readouts and explanation as
  version 25); **version 25's Learner switch** for TD3+BC, BCA and BCA IW on the random/strong and medium/training
  settings (the same "Now playing", readouts, explanation with its Learner line, legend tooltips, cost chart label and
  Learner buttons as version 25, six different episodes); the cost readout at state 50 for each learner (the same five
  values as version 25, each line keyed `--lq-1..5`, "All five BCA actors ..." and "All five BCA IW actors ..."); with
  all the data BCA and BCA IW disabled in the panel and in the Choices, as in version 25's sidebar; BCA IW pressed
  through the Choices (version 26) and through the old sidebar (version 25) with the same result; **version 24's axis
  titles**: the canvas titles drawn (the cost chart's "State t ...", the gauges' "Throttle ... and push ...", the learned
  chart's title naming BCA and BCA IW) and every SVG chart's titles (the "Try a command" strips, the stall chart, both
  charts of "The critic in action", the weighting test's five charts, the two large learning-curve charts and the
  overview's cells, the dot plot's head) present in the same number as version 25, inside their charts (within 3 px:
  version 25's own titles reach up to 2 px past an edge at 390 px, the same in version 26), each the same wording as
  version 25 or another of the page's own wordings (at 390 and 1280 all were the same); the switch bar, the cost
  readout's five keys, back to the fixed controllers, the Choices rows and their "not in use now" marking (the same
  rows as version 25's sidebar, "Learner" included), the learning curves' Choices rows, the range bars, the arm tables'
  readings, the learning curves' table and selectors, version 23's notes and the coverage table's column tooltip, the
  floating Choices control stepping aside over the switch bar, no console errors, no sideways scroll.
- The navigation (T-1 to T-22 and extras): 13 widths in both themes; 25 sections and 278 entries as expected
  (`pass2_v25.json`, plus the lesson's 40); 536 links at 390 and 1280; Previous and Next through all 278 stops both
  ways; deep links at load; Back and Forward; redraws at the shown width (compared with version 25) and table cues;
  loops stop when their section is left; the Choices: 7 groups, **23 rows, 76 buttons** (version 24's 22 and 73 plus
  version 25's Learner row); keyboard; accessibility attributes; touch targets; Whole page; print; without
  `hidden="until-found"`; failure fallbacks; without the lesson (and the order lesson-then-navigation gives the same
  page); reduced motion; screenshots without console errors; every entry by click, fragment and deep link.
- Load time at 390 px with a 4x CPU throttle (alternating loads; a shared machine): `timing_v26.mjs`, 5 loads each,
  medians: DOMContentLoaded **3,873 ms (version 26) against 14,051 ms (version 25)**, load event 3,890 against 14,381 ms,
  main-thread task time 4.32 s against 14.66 s (script 1.19 s against 1.06 s); the navigation's own start 26 ms; no
  console error in any load. Version 26 lays out only the section on screen. The navigation's T-18 (3 loads each): 3,815
  against 13,823 ms, its own start 24 ms. Unthrottled (`perf.mjs`, 2 x 3 loads): about 0.5-0.6 s against 2.0-2.3 s.

Test-side fixes during the work (never a check's intent): the regression test's first trial (stopped after three views,
`regression/regression_v25_trial.log`) failed only its own two new checks: SVG titles were required to lie strictly
inside their chart, but version 25's own titles reach 2 px past an edge at 390 px (the text's line box; the same in
both versions; now within 3 px, or no further out than version 25's), and the cost readouts were compared at the
chart's middle, which is a different state when the two versions' columns differ in width at 1280 px (now both at
state 50). One build-side fix: the navigation's new block check refused the order test (lesson first, then the
navigation), whose header bar carries the lesson's L4 link; that variant's hash was added to `KNOWN` (E2). The first
full run was stopped after its static check, before any browser test, to reword the status box's rerun line (the terms
TD3+BC, BCA and the 80% hold defined where they first appear in the box); the run above is on the final page.

---

# Part 3. Rebase onto the live page after version 25 (page version 27)

The build of page version 26 (made on version 25) is rebased onto the current live page (service version
1792156350-92ba, `scratchpad/live6/core_92ba.html`, 5,849,901 bytes, SHA-256 `0f02c8fc…c062`): version 25 with its
status box removed, published by the other Claude session, which told the owner it was "removed in version 26" but
wrote no version-history entry for it. The result is **page version 27**. Nothing that versions 20-25 and the status
box's removal did is lost or undone: every live element id is kept (except the 18 this build removes on purpose, the
same as in parts 1 and 2), every line that versions 20-25 and the later live edits added is in version 27 unchanged
except the deliberate changes R4-R9 (section 13), no part of the status box is added or restored anywhere, and the
features work in version 27 as on the live page (the regression test, `tests/regression_live.mjs`, section 16).
Nothing was published; the repository was only read; nothing was trained.

Inputs: base `live6/core_92ba.html` (never edited), publish wrapper = first 531 bytes of `live6/live_92ba.html` (the
same bytes as version 25's) plus `"\n</body></html>"`. Outputs: `build/core_v27.html` (7,343,518 bytes, 18,019 lines,
SHA-256 `7e0d6d90…657d`) and `build/page_v27_preview.html` (7,344,064 bytes, `e537b31b…5120`, under the 16 MB limit),
made by `python -X utf8 build/apply_all.py ../../live6/core_92ba.html core_v27.html --preview page_v27_preview.html
--stages stages_v27` (`--version 27`, `--status-date "16 Oct 2026"`, `--status-time 13:44` and the wrapper
`live6/live_92ba.html` are the defaults). Work notes, probes and trial runs: `scratchpad/live6/work/`
(`diff_v25_live.txt`, the trial stages in `trial1/`, `backup_v25_based.py`, `replay_v26/`, `replay_v27.py`, the
rebuild checks in `rebuild_check/` and `build2/`).

This part was written in two sittings. The first (16 Oct, 13:36 to 14:07 UTC) made the backup, the inventory, the ports
and a first build of version 27 (`9beedc1f…a9726`), and started its test battery; it was then cut off by a spend limit.
The second (from 14:45 UTC) checked that work (the first build rebuilds byte for byte from the sources, the backup
matches its list, GitHub's branch heads are unchanged), let that battery finish (section 16), changed one sentence of
this build's own (B21: the near-80 approval's commit time, section 13), rebuilt, and ran the whole battery again on the
final build.

**Before the rebase.** (1) The version-26 build was backed up: `build/component_changes/v25_based/` (234 files: every
patch script and its inputs, the rebase edit scripts, the navigation's rules and expected lists, the version-26 test
scripts, the records, and the version-26 rebase's work scripts under `live5_work/`), SHA-256 list
`component_changes/v25_based.sha256` (its own SHA-256 `643651c6…0c97`). Its `build/apply_all.py`, run on
`live5/core_4933.html` with `--wrapper live5/live_4933.html`, rebuilds `core_v26.html` byte for byte (checked on 16 Oct;
`live6/work/replay_v26/`). (2) The interrupted version-26 battery had left its regression test running (an orphaned
`node regression_v25.mjs` under two `timeout` processes whose shell was gone, with its headless Edge on the profile
`wf7/nav/tools/edge-profile-13951`); that headless Edge was ended at 13:41 UTC and the node and timeout processes exited
with it. Nothing else was ended. That stopped only the regression test: the script `run_v26.sh` went on with the
navigation's tests and the timing tests and ended at 14:13 UTC (`tests/run_v26.log`), from 14:03 UTC at the same time as
the first version-27 battery (two headless Edge at once, the limit); none of those version-26 results is used. In the
second sitting (14:45 UTC) the only test processes of this build still running were the first version-27 battery's own
(`sh run_v27.sh` under `timeout 7200`, then `node t_entries.mjs` with its headless Edge on the profile
`wf7/build/tests/nav/tools/edge-profile-13821`): this job's battery on the then-current build, not a leftover of the
interrupted review, so it was not ended and was let finish (section 16). No other headless Edge or node test process
with `scratchpad\wf7` or `scratchpad/wf7` in its command line was running, and no process was ended in the second
sitting (a `tail -f` of `tests/run_v26.log`, open since 12:31 UTC, is neither, so it was left alone; it only reads that
log). (3) A correction to section 10: the final `core_v26.html` is 7,354,792 bytes, SHA-256 `0765c149…d378` (as
`run_v26.log`'s second start, 13:28:59 UTC, lists it); section 10's 7,354,642 bytes and `8f829021…0ceba` were an
earlier build of version 26. The battery on the final one passed its static, lesson, color and old full checks; its
regression test was stopped when the live page changed; version 26 was never published.

## 11. What the live page changed after version 25 (inventory, from the diff of version 25 against the live page)

`diff live5/core_4933.html live6/core_92ba.html` (`live6/work/diff_v25_live.txt`): two changes, 26 lines removed and 1
line changed; the page shrank by 7,338 bytes; 2 element ids removed (`head-status`, `head-status-h`; 2,141 ids instead
of 2,143), none added.

- [x] **The status box is gone** (version 25's lines 1720-1745): `div.box#head-status` with its heading
  `h2#head-status-h` "Status as of 16 Oct 2026 (page version 25)", its note (the Points and Pre-registered tooltips)
  and its list `ul.g-status` of nine lines (main test, Stage 1, Stage 2 seeds 0-4 and plan version 7, restructured
  BCA, route search, the weighting test's two lines, toy worlds). The owner commented on it "I dont want this section
  included" (16 Oct 2026, 12:06 UTC). The header now goes from its opening paragraph (`p.g-one`) straight to "The
  process, step by step" (`p#pipe-h`). The other session left the box's CSS (`ul.g-status` rules, the `.box > h2` rule
  and its comment) and the version-history entries that mention the box (versions 17, 21, 23) as they were, and wrote
  no version-history entry for the removal.
- [x] **The "Strong test mix" button's tooltip** (the switch bar's learned mode, `data-tog-group="lm"`,
  `data-tog-value="strong"`) rewritten in three lines (`&#10;`): the same start in the strong test mix's 1,000 fixed
  starts, only the current differs; what makes it strong (currents above 0.7 in about 64% of its episodes, against about
  19% in training); it removes no command and adds none, but strong currents mean hard water far more often, where the
  best move is the 80% hold that training lacks; on expert-quality data about 36% of its steps have a command in the
  removed region. (The page's two other "Strong test mix" tooltips are unchanged.)

The navigation's rules list 237 parts on the live page (`nav/rules/pass2_v27.txt`): version 25's 238 without the status
box ("Status of every line of work", `#head-status`); nothing else differs.

## 12. Conflicts and how each was merged

The trial run (`apply_all.py` on the live page with `--version 27`, stages in `live6/work/trial1/`) failed on exactly
the status box: the color patch's B1, B2, B4, B20 and B23 found no anchor, and the navigation's static edit E10 for the
box's label found none. Everything else (the other 90 color edits, the rest of the navigation with every replaced block
matching version 25's SHA-256 in `KNOWN`, the lesson) applied unchanged.

| Where | Conflict | Merge |
|---|---|---|
| Color patch B1 (heading), B2 (Stage 2 and plan version 7), B4 (four learners; near-80 test), B20 (the weighting test's place in the repository), B23 (version 25's rerun of BCA and BCA IW) | all five edited the status box, which the owner asked to remove and the live page removed | dropped from the patch (`rebase/edits/e_color_v27.py`); nothing of the box is added or restored. With B4 and B23 their six term tooltips (CQL, IQL, ReBRAC, Unifloral, Weighted BCA, BCA IW) leave the page; no other part of the page used them |
| Color patch B22 (Results' opening: the two newer lines of work) | it ended "(both in the status box at the top)" with a link to `#head-status` that would now be broken | it points to After the stop (`#ats-status`), whose box B21 extends with the near-80 test |
| Navigation E10 (static edits) | the label "Status of every line of work" on `div#head-status` has no anchor | `nav/rules/static_edits_v27.py` remakes `nav/src/static_edits.json` from the version-23 edits without that one (132 edits instead of 133; every other anchor checked once in the live page); `apply_nav.py`'s docstring records it (`e_nav_v27.py`) |
| Navigation, every block it replaces (E11, E7a-c, E1, E2, E4, E5, E6, E8) | could the box's removal or the tooltip change hide inside one? | no: every removed block's SHA-256 matched version 25's (`KNOWN` unchanged); E2 replaces only the header's walkthrough bar, which the live page did not change |
| Navigation's Contents and "In this section" | they listed the status box | their entries come from the page's own parts, so with the box gone they list it no more: the header's section has "Start of the section" and "The process, step by step" (expected lists `pass2_v27.json`; the regression test checks both rails at 4 views) |
| The header | the box sat between the opening paragraph and the process | with nothing added there, version 27's header goes from `p.g-one` straight to `p#pipe-h` as the live page's does, after the navigation's start bar; `apply_all.py`, `static_v27.py` and the regression test check it |
| The "Strong test mix" tooltip | the color patch's K9 changes the shared tooltip's `data-tip` path | K9 acts only on tooltips with `data-tip-keys`; this button has none, so its tooltip is built exactly as on the live page. The tooltip's markup is byte-identical (all three "Strong test mix" tooltips occur in version 27 as often as in the live page), and the regression test compares its shown text and its lines (`innerText`) with the live page's at 4 views |
| Status statements | the facts changed after version 26's status time: Stage 2 version 7 was ended without running, and the near-80 test now runs on the owner's desktop computer (section 14); B2's and B4's statements were in the box | B5, B7, B9 (Stage 2), B21, B22 (near-80) reworded; new B24 (Stage 2's outcome: "what seeds 5–9 will decide") and B25 ("What happened to this plan": "still runs under version 7"), the only other undated statements those facts made false outside the toy section; the lesson's step 7.2 (`e_lesson_v27.py`); the status time 16 Oct 2026, 13:44 UTC |
| Integration checks | they required the box ("(page version N)", B2's, B4's, B20's and B23's statements) | rewritten (`e_apply_all_v27.py`): any part of the box refuses the page (`id="head-status"`, `id="head-status-h"`, `<ul class="g-status"`, a link to `#head-status`, "Status as of ", the navigation's label for it); the header must be the live page's; the live page's new tooltip must be there once as in BASE; version 27's statements must be there as often as stated, version 26's out-of-date ones not at all; exactly 4 status times, all 16 Oct 2026, 13:44 UTC (After the stop's sentence and box, the first claim, Results' opening) |
| Version history | our version-26 entry; the other session wrote none for its version 26 | I1 inserts "Version 27, 16 Oct 2026." on top, our version-26 entry's text rewritten: built on the page as version 26 left it "(version 25 with its status box removed at the owner's request)" (the one clause on version 26), no status box, the status statements outside it; no entry is invented for version 26; the checks require exactly one entry per version number (1 to 25, and 27) |
| Test scripts | the text shots shot the box and opened its term tooltips; the static test checked the box's rerun line; check3 read the box's heading; the regression test allowed R1-R3 (box lines) | `tests/make_tests_v27.py` makes the version-27 scripts: text shots of After the stop's box, the first claim, Results' opening, Stage 2's outcome and "What happened to this plan", and a check that no part of the box is on the page; static: no box, version 25's rerun data block and the new tooltip unchanged, the header as live; check3: no status heading on either page, no entry for version 26; the regression test against the live page (`regression_live.mjs`), R1-R3 gone, new checks (section 16) |
| The toy worlds (section#toy-worlds) | another agent is rewriting the whole section (`wf7/toy/apply_toy.py`), to be added afterwards by a third | not touched by this rebase (done in part 4: the toy patch became the build's first stage, and B14, B15 and the five static ids were dropped). Version 27 still carries this build's earlier edits inside it: B14 and B15 (two phrases on the notebook's commit, from version 20) and the navigation's static ids and labels on its headings, tables, findings and details (E10); `apply_toy.py` must replace the section as a whole on version 27, or carry those |

## 13. Deliberate changes to live content (each checked by the regression test, part b, or by apply_all.py)

Lines that versions 20-25 added (the regression test's part b: 1,115 such lines, all present unchanged except these 39,
each with its replacement present): R4-R9 as in section 8 (the old sidebar's CSS, version 21's tab-away script,
version 20's switch-bar comment, the learned actors' 80% opacity, the learned cost readout as keyed lines, the old
sidebar's script whose rows nav.js carries). Section 8's R1-R3 changed the status box and are gone with it. The live
page's "Strong test mix" tooltip is not changed.

Version-19 text that this build's status statements change (outside versions 20-25's lines; each checked by
`apply_all.py`'s statement list):

| Edit | Where | Live text | Version 27 |
|---|---|---|---|
| B5 | Results, opening | "As of 15 Oct 2026, 05:41 UTC, four lines of work continue on this environment: Stage 2's confirmatory seeds 5–9 (... version 7 ... still awaits its handoff and its own approval)" | "As of 16 Oct 2026, 13:44 UTC, this is where four lines of work on this environment stand: Stage 2's confirmatory seeds 5–9 (... version 7 ... was handed off on 15 Oct at 07:47 UTC and approved by the project owner at 15:45 UTC, and the approval was checked and recorded on 16 Oct at 02:27 UTC; later that day version 7 was ended without running, so those seeds will not run under it)" |
| B22 | Results, opening, end | "... on its own data, in the next section." | adds: "Two more have no results yet: runs of four learners, whose plan the project owner approved in chat and must still approve at their own terminal, and a test that leaves the commands near 80% throttle out of the training data, whose final plan the owner approved at their own terminal on 16 Oct and which now runs on the owner's desktop computer (After the stop)." |
| I2 | After the stop, "Where this fits" | "last brought up to date ... on 15 Oct 2026 at 05:41 UTC" | "on 16 Oct 2026 at 13:44 UTC" |
| B6 | After the stop's box, heading and note | "as of 15 Oct 2026, 05:41 UTC"; read at `e91452d` | "as of 16 Oct 2026, 13:44 UTC"; read from the repository as last fetched from GitHub: the shared branch at `0377dd7` (16 Oct, 12:38 UTC), `forked` at `cab5b3e` (the approval of version 7 is `de4851c`), main at `1ec267e` |
| B7 | After the stop's box, Stage 2 seeds 5-9 | "not run. They run under version 7 ... Still to come: the handoff, the project owner's approval typed at their own terminal, then the run." | "not run, and they will not run under version 7, which changed only where and how the rest of Stage 2 runs ... handoff at 07:47 (`3a8ce5b`) ... approval at 15:45 (`de4851c`) ... checked and recorded on 16 Oct at 02:27 (`02e591b`). Later on 16 Oct version 7 was ended without running, and that computer now runs the near-80 test (below); no record of the ending is in the repository." |
| B21 | After the stop's box, a line before the weighting test's | (none) | "Near-80 test (approved and running; no result yet): ... frozen on 16 Oct at 01:33 (`e3e87c3`), approved by the owner at their own terminal at 11:50 (committed at 11:51, `f79b89c`) and the approval recorded at 11:53 (`287380b`); it now runs on the project owner's Framework Desktop computer." (the commit time added in the second sitting, `e_color_v27c.py`: the approval was typed at 11:50:43 UTC and committed at 11:51:04 UTC, the time the workflow's task gives, so the line names both, as B7 does for version 7) |
| B8 | First claim's status | "Status, as of 15 Oct 2026, 05:41 UTC" | "Status, as of 16 Oct 2026, 13:44 UTC" (still "not shown on the exploratory seeds; the deciding test has not run") |
| B9 | First claim, the deciding test | "runs under version 7 ...: its self-test was ready ..., and the handoff, the project owner's approval at their own terminal and the run itself are still to come." | "was to run under version 7 ...: its self-test was ready on 15 Oct at 00:42 UTC, the handoff was made at 07:47 UTC, the project owner typed their approval at their own terminal at 15:45 UTC, and the approval was checked and recorded on 16 Oct at 02:27 UTC; later that day version 7 was ended without running, so the deciding test will not run under it." |
| B24 (new) | Stage 2's outcome | "Whether this holds on fresh seeds is what seeds 5–9 will decide." | "Seeds 5–9 were to decide whether this holds on fresh seeds; version 7, which was to run them, was ended on 16 Oct 2026 without running." |
| B25 (new) | Stage 1's plan, "What happened to this plan" | "The plan is not withdrawn: its deciding test, on fresh seeds 5–9, still runs under version 7." | "The plan is not withdrawn, but its deciding test, on fresh seeds 5–9, will not run under version 7, which was ended on 16 Oct 2026 without running." |
| B10, B11, B12, B13-B19 | as in version 26 | | unchanged from version 26 (the sources' head `0377dd7`; the import record's place; the TD3+BC settings' wording; the toy notebooks' commits; the weighting test's files on main since 15 Oct 17:18 UTC) |
| Lesson, step 7.2 | (this build's lesson) | "...; the run is still to come. Nothing has run on those seeds yet." | "...; later that day the run's plan (version 7) was ended without running, so those seeds will not run under it. Nothing has run on those seeds." |

Left as they are, because they are dated and were true at their date: the route table ("seeds 5–9 are to run under
version 7", caption "as of 15 Oct 2026, 05:41 UTC"), Stage 2's results ("as of 15 Oct 2026, 05:41 UTC, they have not
run"), the second claim's status ("as of 15 Oct 2026, 05:41 UTC: not shown"), the glossary's Stage 2 entry, the
proposed change to the claim ("not accepted on record as of 15 Oct 2026, 05:41 UTC"). After the stop's opening ("only
its fresh seeds 5–9, not yet run, can [carry a claim]") stays: it is still true. Inside section#toy-worlds, "Stage 2's
pre-registered seeds 5–9 still test the width" is no longer true; it is left to the toy section's rewrite (not touched
here; done in part 4: the new section says version 7 "has ended and will not run").

## 14. Status statements (as of 16 Oct 2026, 13:44 UTC)

Read (read only, `GIT_OPTIONAL_LOCKS=0`) from `C:/Users/David/dev/BCA`: the remote-tracking refs as fetched from GitHub
at 12:39 UTC (`.git/FETCH_HEAD`, 08:39:13 -04:00, by another session), and GitHub's branch heads read at 13:44:08 UTC
with `git ls-remote origin` (read only: nothing fetched, nothing written in the repository); they equal the fetch, so
nothing in the repository changed between 12:39 and 13:44 UTC (the shared branch `0377dd7`, `forked` `cab5b3e`, main
`1ec267e`, `claude/stage2-run-files` `af32a64`; no near-80 run-files branch). Read again the same way in the second
sitting, at 14:47 UTC and at its end (section 16): every head the same, `.git/FETCH_HEAD` still the 12:39 UTC fetch, so
every statement below still holds and the status time stays 13:44 UTC.
- **Stage 2 version 7: ended without running.** Handed off 15 Oct 07:47 UTC (`3a8ce5b`); approved by the owner at their
  own terminal (`approval_v7_3dfcb6140e2b8e2c.json`, approved_at_utc 15:45:42Z, `de4851c`); the approval imported on 16
  Oct at 02:27:10 UTC (`import_v7_3dfcb6140e2b8e2c.json`, `02e591b`). That it has since been ended, so seeds 5-9 will
  not run under it, is in no ref; it is the fact the workflow's task of 16 Oct (about 13:38 UTC) states as current. It
  fits the near-80 plan's own rule (`near80/PREREGISTRATION.md`, choice 12: the near-80 runner starts only after
  version 7's run has finished, or once the owner records that version 7 ended, `run_near80.sh version7-ended`). No
  result of seeds 5-9 in any ref (`handoff/HANDOFF.json`: every seed 5-9 phase "not started"). The page says "ended
  without running" and "no record of the ending is in the repository".
- **Near-80 test ("version 8, near-80"): approved and running.** Final pre-registration `e3e87c3` (16 Oct 01:33 UTC,
  fingerprint 51e677ab76c8d2f7); approved by the owner at their own terminal (`approval_near80_51e677ab76c8d2f7.json`,
  approved_at_utc 11:50:43Z, commit `f79b89c` at 11:51 UTC); the approval recorded (`import_near80_51e677ab76c8d2f7.json`,
  imported_at 11:52:58Z, `287380b`, 11:53 UTC). That it now runs on the owner's Framework Desktop computer is the
  workflow task's statement (no run files in any ref). No near-80 result in any ref.
- **Four-learner study** (Unifloral's TD3+BC, CQL, IQL and ReBRAC, each alone, with standard BCA and with
  importance-weighted BCA): in code review, tested on made-up numbers only, waiting for the owner's approval at their
  own terminal; nothing of it trained on ThrusterChain data (in no ref; its code is in WSL, which this build does not
  read). The page states it once, in Results' opening (B22: "whose plan the project owner approved in chat and must
  still approve at their own terminal"); its line in the status box went with the box.
- Unchanged and still true: the weighting test's code, specification and exploratory results on main since 15 Oct 17:18
  UTC (`03726de`), its runs not; the toy notebooks on `forked` and main; version 25's cloud rerun in no ref.

## 15. Replay

`live6/work/replay_v27.py` copies `component_changes/v25_based/` to `live6/work/replay/`, runs `e_color_v27.py`,
`e_color_v27b.py`, `e_color_v27c.py`, `e_nav_v27.py`, `e_lesson_v27.py`, `e_apply_all_v27.py`, `e_apply_all_v27b.py`
(the last two from the second sitting: B21's commit time and its check) and `nav/rules/static_edits_v27.py`, and
compares every component source with wf7's: all identical (the static edits but for the source path they record;
replayed again after the second sitting's edits, 16 Oct about 14:55 UTC). The
lesson's fragment is rebuilt by `build_lesson.py` (it needs the formula renderer next to wf7, so not in the replay): it
differs from version 26's only in step 7.2's sentence; `lesson.js`, `lesson.css` and `lesson_data.json` are
byte-identical.

## 16. Test results (page version 27, before the toy integration)

Built page: `build/core_v27.html` 7,343,518 bytes, 18,019 lines (SHA-256 `7e0d6d90…657d`), `build/page_v27_preview.html`
7,344,064 bytes (`e537b31b…5120`, under the 16 MB limit); 3,445 ids (2,776 MathJax glyph ids, 1,087 of them the
lesson's), 0 duplicates, 0 CR bytes, 4,565 SVG `<use>` references resolve, the title unchanged; one version-history
entry per version (27 on top, then the other session's 25 and 24, down to 1; no entry for version 26); no part of the
status box; the header going from its opening paragraph straight to "The process, step by step" as on the live page.
`apply_all.py` gives the same bytes on every run (re-run at 14:46 UTC on the first build's sources, at 14:59 UTC on the
final ones), and its trial run on the live page failed exactly on the status box (section 12). The component chain
replays from `component_changes/v25_based/` byte for byte (section 15). The toy section is the one `wf7/toy/apply_toy.py`
knows (the live section after the color and navigation patches, `33b80c29…`). GitHub's branch heads, read again at the
end (16:05 UTC, `git ls-remote`, read only), are unchanged, so the status statements stand.

**First battery** (on the first build, `9beedc1f…a9726`, which differs from the final build only in B21's two lines;
`tests/run_v27.sh`, 14:03:49 to 14:56:45 UTC: started in the first sitting, let finish in the second; its outputs are
kept in `tests/runs/v27_first_build_9beedc1f/`): 42 checks, 41 passed. The navigation's T-15 (the fallback without
`hidden="until-found"`) failed, and the cause was in the test: it clicks a content link in the header and expects
Results to be shown, and the header's only link to Results was in the status box; with the box gone it fell back to
the header's first link, the start bar's "Learn step by step", and (correctly) the lesson was shown. Test-side fix,
its intent kept: `make_tests_v27.py` now makes `setup_tests_v27.py` patch T-15 to click the process's first link in the
header (to "The recorded data", `#dataset`) and to expect that link's own section. Run alone on the final build, T-15,
T-16, T-17 and T-21 passed; then the whole battery ran again.

**Final battery** (on the final build; `tests/run_v27.sh`, 16 Oct 2026, 15:04:29 to 16:01:31 UTC; one headless Edge at
a time, the timing tests alone at the end; `tests/collect_v27.py` -> `tests/out/results_v27.json`, a copy in
`tests/runs/v27_final_build_7e0d6d90/`): **42 checks, 41 passed in the battery, no stale result file; the one that did
not, the navigation's T-13 (Whole page), passed in three runs of its test on their own right after (16:02 to 16:05
UTC).** T-13 reloads the page with Whole page on at `#iw-f2`, scrolls to the toy worlds and turns Whole page off,
expecting the toy worlds to stay on screen; in the battery run the page's own move back to `#iw-f2` after the reload
came after the test's scroll (the machine was loaded: the live page's throttled load took 24.5 s against 20.9 s in the
first battery), so the weighting test stayed on screen. The same test passed in the first battery and in all three
reruns (`tests/nav/test/out_rerun/`: the battery's output and the three reruns'); nothing in it touches B21, the only
change between the two builds. It is listed as a test that can fail under load (open issue), not as a fault.

- Static (`static_v27.py`): no external resource beyond the Google Fonts already linked, no CSS URL, in-page links
  only as bare `#id` (418), 0 CR, every new or edited script an IIFE (or JSON), localStorage only inside try, no
  alert/confirm/prompt/print, no color literal outside a token in any CSS line this build adds, no new British
  spelling, no duplicate id, no broken link or aria reference, the old navigation and walkthrough gone; no part of the
  status box (`#head-status`, `#head-status-h`, `ul.g-status`, a link to it, "Status as of ", the navigation's label
  "Status of every line of work", an entry "Version 26, "); version 25's data block `script#sim-rerun-data` and the
  live page's newer "Strong test mix" tooltip byte-identical; the header as on the live page.
- The lesson: 4 views (390 and 1280, light and dark): 39 steps walked forward and back, 58 formulas and legends, the
  panels working; 176 of 176 controls, 221 worked lines checked by rule D2 (0 bad); 1,546 random slider states (0 bad);
  without `hidden="until-found"` and with reduced motion, with and without the navigation; no console error.
- Colors (`verify_render.mjs`, 4 views with the Whole page switch on, the live page as the learning curves' reference):
  371 checks passed, 0 failed. Text shots (1280 and 390): After the stop's box, the first claim, Results' opening,
  Stage 2's outcome and "What happened to this plan" found with version 27's statements; no part of the status box; no
  console error.
- The old full check (`int/check3_v27.mjs`): 4 views, each of the 25 sections shown in turn without sideways overflow,
  NaN or console errors; the whole page clean (ids, links, aria, requests); version 27's entry there and no entry for
  version 26; no status heading on either page.
- **Regression against the live page (`regression_live.mjs`)**: (a) all 2,141 element ids of the live page are in
  version 27 except the 18 removed on purpose (the old sidebar's 6, the walkthrough button and dialog's 12); (b) all
  1,115 lines that versions 20-25 and the later live edits added are in version 27 unchanged (once this build's added
  attributes are set aside), except 39 lines that are R4-R9 (section 13), each with its replacement present, and no
  allowance unused; (c) at 390x844 and 1280x900, light and dark, all 24 checks pass in each view: Who drives (the
  learned actor) for two data qualities and both mixes, **version 25's Learner switch for TD3+BC, BCA and BCA IW** on
  both mixes through the panel and through the Choices (the same "Now playing", readouts, explanation with its Learner
  line, legend tooltips and cost chart label as the live page), the cost readout's five values equal to the live
  page's with a color key per line, BCA and BCA IW disabled with all the data in the panel and in the Choices, **the
  live page's newer "Strong test mix" tooltip** (the same text and lines), **version 24's axis titles** (the canvas
  titles drawn, every SVG chart's titles inside their charts, as many as on the live page, each one of the page's own
  wordings), the switch bar, the Observation Space range bars, the arm tables' tooltips, the learning curves' table and
  selectors, version 23's notes, the Choices rows with "not in use now", **no status box on either page**, the header
  going from its opening paragraph to "The process, step by step" on both, **Contents and "In this section" listing
  no status box**, no console errors, no sideways scroll.
- The navigation (T-1 to T-22 and extras): 13 widths in both themes; 25 sections and 277 entries as expected
  (`pass2_v27.json`, 237 parts, plus the lesson's 40); 527 links at 390 and 1280; Previous and Next through all 277
  stops both ways; deep links at load (32 + 7); Back and Forward; redraws at the shown width (compared with the live
  page) and table cues; loops stop when their section is left; the Choices: 7 groups, 23 rows, 76 buttons; keyboard;
  accessibility attributes; touch targets; Whole page (T-13, above); print; the fallback without `hidden="until-found"`
  (T-15, with the test-side fix above); failure fallbacks; without the lesson; reduced motion; screenshots without
  console errors; every entry by click, fragment and deep link; the live panel, the task demo's levels, the tooltips,
  the critic viewer and the table Tab stop; the start bar and the lesson's memory; the page contract.
- **Load time at 390 px with a 4x CPU throttle** (`timing_v27.mjs`, 5 loads each, alternating, on the shared machine
  while another workflow ran tests in WSL), medians: DOMContentLoaded **6,670 ms for version 27 against 24,469 ms for
  the live page** (ratio 0.27), load event 6,701 against 24,908 ms, main-thread task time 7.46 s against 25.18 s
  (script 2.12 s against 1.73 s); the navigation's own start 52 ms; no console error in any load. The navigation's T-18
  (3 loads each): 6,574 against 24,643 ms, its own start 47 ms (at most 52). Unthrottled (`perf.mjs`, 2 x 3 loads): 645
  to 790 ms against 2,647 to 3,128 ms. The first battery measured 6,053 against 20,850 ms (T-18: 5,483 against 18,588
  ms): the machine's load moves both, the ratio stays about 0.3. Version 27 lays out only the section on screen; the
  live page lays out all of it.
