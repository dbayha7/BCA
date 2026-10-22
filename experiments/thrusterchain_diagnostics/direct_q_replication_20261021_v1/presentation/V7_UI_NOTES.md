# ThrusterChain V7 interactive edition

Updated October 21, 2026. V7 extends the reviewed V6 report using the saved fresh-state bank. V5, V6 and the experiment outputs remain unchanged.

## New interactions

- Five frozen scoring rules, each displaying its mean return change for the current filter.
- World/load and assigned-time filters with explicit state counts and equal-state descriptive means.
- Thirty-two state tiles, plus shortcuts to the largest gain, largest loss and an unchanged-return case.
- State details comparing the selected rule, raw-Q1 choice and original actor.
- Candidate sorting by original ID, score, measured continuation return or width; saved winners never change when sorting.
- A compact candidate table with an optional score-ingredients view.
- A selectable width-versus-signed-error plot with candidate IDs and an actor marker.
- Full-precision CSV download for each state and scoring rule.
- Reset, previous/next state controls, keyboard controls, live status messages, and reduced-motion-aware detail scrolling.
- A separate **Restart lesson** button on the landing page and above every lesson step. It clears the saved lesson position, visited marks and quiz answers, then reloads the same document at step 1.1. Reloading also resets the Try-it controls. The recap's existing **Start again from step 1.1** link performs the same reset. Reading and display preferences are preserved.

## UI changes

A new hero, grouped controls, comparison cards, outcome tiles, readable table treatments and responsive layouts give the new explorer a clearer hierarchy. The entry page has a direct explorer link and a card-based reading path. Existing evidence sections remain available through the same navigation.

The explorer starts with all 32 states and Q1 minus width. State 24 is deliberately selected as the one harmful width example, with the full-bank summary above it. No coefficients, shuffle or labels are changed by the controls.

The bottom Contents/section/Previous/Next bar now stays visible at every screen width. Its actual height and bottom offset determine the reserved reading space. The right sidebar docks from 1,200px and Contents docks from 1,600px; smaller windows use the existing drawers. CSS and JavaScript use the same breakpoints. The reading area is wider, and explorer cards plus worked calculations respond to the reading column's width. Formula and evidence-table overflow stays inside local scroll containers. Lesson number inputs and narrow table text are more readable, and short phone windows get a wider section drawer.

## Verification

- 10,368 copied scalar fields, including 1,920 scoring-rule values, match the saved CSVs.
- Independently recomputed 160 winners, 160 concordances and 32 width/error correlations.
- All 100 filter/method combinations and 640 candidate sort combinations passed.
- All 160 CSV payloads retain full numeric precision; download handling was checked in a mocked DOM.
- Controller checks covered 142 case shortcuts, 216 previous/next transitions, candidate selection, plot keyboard controls, view switching, ingredient toggling and reset.
- All 22 executable scripts compile; all 12 embedded JSON blocks parse.
- All original 11 embedded JSON blocks, 17 executable scripts, 2,330 V6 table rows and 13 numbered source panels are preserved exactly. The other two original scripts differ only in navigation breakpoint values.
- No duplicate IDs or broken static fragment/ARIA references were found among 3,955 IDs.
- The previous finite-oracle and training-filter checks still pass all 888 combinations.
- Restart-script checks seed all 39 visited steps and a saved position at 7.3, then verify all three restart entry points clear the three lesson keys, preserve unrelated preferences, select step 1.1 and request a reload. Repeated reset, unrelated clicks and denied storage are also covered with test doubles; full browser reload/navigation remains unverified.
- Eight responsive source/controller checks passed: matching embeds, synchronized breakpoints, screen-scoped bar rules, reading-column queries, synthetic bar measurements, resize coalescing, optional browser APIs, and hidden/missing-bar handling. These are not rendered viewport tests.

## What was not verified

The in-app browser previously rejected local file URLs under its security policy. No alternate browser, local-server workaround or headless renderer was used. This release has static source checks and mocked-DOM controller tests, not live browser rendering tests. Actual viewport layout, browser keyboard behavior, screen-reader output and full navigation integration remain unverified.

The controls use saved measurements only. No new model training, simulator transitions, bootstrap draws or policy evaluations were run. Filtered summaries do not reuse the original full-bank uncertainty intervals.

## Local implementation and receipts

`v7_ui/build.py` creates the self-contained HTML from preserved V6, component HTML/CSS/JavaScript and saved CSVs. `v7_ui/data.json` contains the embedded evidence. `v7_ui/responsive.py` changes only the navigation's CSS and JavaScript breakpoints. Validation receipts are `evidence_check.json`, `explorer_validation.json`, `legacy_validation.json`, `structure_check.json`, `lesson_restart_validation.json` and `responsive_validation.json` in that folder.

Final HTML SHA256: `86db7f4ca334eac6ced1e537b76604a8a9023599d545e4eb44b0e01fd421ab5c`.
