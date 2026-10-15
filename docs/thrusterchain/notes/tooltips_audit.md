# Tooltip audit: ThrusterChain, version 10

Page audited: `thrusterchain.html`, md5 `adf3837e…21a8`, the same file as `thrusterchain.v10.html`. Neither file was changed. The full inventory is in `notes/tooltips_inventory.json`, with one record per distinct text. Each record gives where the text appears, the text itself, its source and its problems.

## Main findings

- **The page can show 902 distinct tooltip texts.** They break down as follows:
  - 475 written by hand: terms, symbols, controls, readouts, chart parts, formula legends, the controller guide and column descriptions.
  - 410 cell tuples, one per table row. They cover 1,917 cells.
  - 17 computed templates from the two drawings in the live panel.
- **Almost every hand-written text breaks the "complete sentences" rule.** 450 of the 475 have at least one fragment. Only 18 are clean.
  - Column descriptions: 228 of 229 are fragments. The descriptions in `e5_cols.py` are written as noun phrases, for example "The value used when the argument is not given." or "What the argument changes.".
  - Symbol tooltips: 114 of 122 are fragments, for example "The 10 float numbers now and one step later.".
  - Fixing the source dictionaries fixes these everywhere at once.
- **No generator in version 10 cuts text.**
  - Every cell tuple keeps every word of its cells. This was checked on all 1,917 cells.
  - Every hand-written tooltip traces to its source (`e5_cols.py`, `e6_content.py`, `e7_texts.py` or the weighting-test script), and none is a shortened copy of it.
  - Two mechanisms can still hide or distort text. One is the tooltip box's height cap, which is the only way version 10 can stop a tooltip at "…while its throttle cost". The other is how the cell-text reader `plain()` handles some markup. Both are described under "Truncation mechanisms" below.
- **Inaccurate: none confirmed.**
  - An automatic check found every number in a hand-written tooltip somewhere in the page's own text. The exceptions are the three weighting-test bar shares (15.7%, 25.8%, 36.4%). They come from the script's `GAP` constants and appear nowhere else on the page.
  - Spot checks agreed with the page's tables and formulas: 0.879, 98.6 points, −606.5 points, 60 of 42,500 rows, 5.8 steps, 23.27 steps, and the signs of w₂.

## Counts of each problem type

Each count is the number of records with at least one problem of that type.

| Problem | Total | Hand-written (475) | Computed (17) | Cell tuples (410) |
|---|---|---|---|---|
| fragment | **467** | 450 | 17 | 0 |
| code | **109** | 16 | 0 | 93 (59 are tables whose purpose is to list code names) |
| unclear | **81** | 29 | 3 | 49 |
| acronym | **72** | 29 | 0 | 43 |
| truncation | **3** | 1 | 0 | 2 |
| inaccurate | **0** confirmed | 0 | 0 | 0 |
| too long (more than 3 sentences) | 14 | 4 | 10 | not applicable |

**Fragments, by kind.** A record can have more than one kind:

| Kind of fragment | Records |
|---|---|
| No main verb, for example "The rewards of the states up to this one, added up." | 359 |
| A noun phrase before a colon, for example "The page's score scale, per situation: 0 is …" | 68 |
| A clause that starts with a question word, for example "What the split is used for." or "When the propeller stalls: past 80% throttle, …" | 64 |
| No capital letter at the start, for example "0.3 times the squared throttles …" or "z when it is positive …" | 22 |
| No final full stop: the chart bands, the weighting bars and the cost-chart lines | 16 |
| No subject, for example "Keeps every normal-driving throttle inside ±80%." or "Found with one exact simulator step …" | 6 |
| A clause after a semicolon with no verb, for example "…; the water as recorded." | 5 |

**Acronyms.** Records where the acronym is not spelled out in the same tooltip:

| Acronym | Records | Hand-written ones |
|---|---|---|
| BCA | 54 | 21 |
| TD3+BC | 13 | 10 |
| TD3 | 4 | 2 |

The rest are in cell tuples only: SHA-256, UTC, D4RL, TRIGGER, VERDICTS and B1. All of them are in the "After the stop" and "Names used in the code" tables.

**Code in hand-written text.** These are column descriptions:
- `ThrusterChainEnv`, `gym.make` and `max_episode_steps` (Arguments).
- `float32` (Action Space, the Stalls column) and `float64` (the exact starting-state table).
- `Gymnasium` (the summary table).
- `experiments/shift` (Names used in the code).
- "design seed 900" in the eight "t = n" columns.

**Code in cell tuples.** Code is expected in the Arguments, Registered ids and Names used in the code tables (59 rows). Elsewhere it is not explained:
- the Type column of the dataset table (`float32`, `int8`, `bool`);
- the summary table;
- the "After the stop" table: hashes, `superseded/v2/`, `standins.py`.

**Unclear, the main ones:**
- **Chart bands (11).** Each line puts a controller, a thruster and a number side by side with no label, for example "Stall-aware thruster 1 0.34, at its 80% limit on 0%". The stall-aware line also drops the word "throttle".
- **Page nicknames left unexplained:** "the oracle", "the gate's actor on the gapped critic", "room check", "stand-in(s)", "futility look", "the marks look", "the step-up rule".
- **The "After the stop" table (7 rows):** internal bookkeeping with no explanation (hashes, "approval file", "the user's choice message", "pins").
- **Symbols used without definition:**
  - μ = dλκ / 2;
  - p and v in √(p² + v²) in the float tooltip;
  - "hold demand" and the "new recording controller" in Hard water.
- **Two different tooltips share the heading** "The share for a straight critic".
- **"(picked)" in the cost chart** is not explained.
- **Grammar:** "This is the floats on their own".
- **Text damaged by the cell-text reader** (see the last mechanism below): the import code block, "Box(…, float32) 4 numbers", ".,", "e^(2c)" and "P_cap".

**Too long.** The four controller-guide tooltips have 4 to 6 sentences, the channel drawing's general text has 6, and the computed thruster tooltips have 5.

## The owner's cell: "stall" in the first Arguments table

The tooltip it shows now is complete; nothing in it is cut:

> (Argument: **stall**, Type: bool, Default: True, Meaning: True: the stalling world. False: the no-stall comparison world (see Action Space).)
> Argument: The argument's name, accepted by ThrusterChainEnv and by gym.make.

Its problems:
- The column description is a fragment and contains code (`ThrusterChainEnv`, `gym.make`).
- The tuple repeats code values (`stall`, `bool`, `True`).
- "Meaning: True: the stalling world" puts two colons in a row.

The phrase "…while its throttle cost keeps rising" ends the Action Space "Stalls" column description. Close variants are in the "The stall" term ("…while its throttle cost keeps rising") and the "Stalling world" option ("…while the throttle cost keeps rising"). All three are complete in the page.

## Truncation mechanisms

1. **The `cut()` helper (removed).**
   - Where it was: in `e5_tooltips.js`, line 51 defined `cut(s, n)`, which returns `s.slice(0, n − 1) + "…"`. Line 263 cut every other cell of the row to 63 characters and the hovered cell to 239.
   - It was removed in e7 (`e7_texts.py` JS1: "tuples: the whole value, never cut" and "no cut helper").
   - It never reached a published version: version 9 had no tooltips, and version 10 has no `cut()`.
   - It only touched cell values. It could not have cut a column description such as the Stalls one.
2. **The tooltip box's height cap (live).**
   - The cause: `e7_texts.py` CSS adds `.tt { max-height: calc(100vh - 16px); overflow: hidden; }`, and `place()` pins a tall box to the top of the window.
   - The effect: a tooltip taller than the window minus 16 px is cut off mid-line at the bottom. There is no scroll bar, because the box lets every pointer event through, and no sign that anything is missing.
   - On phones, `100vh` is the window height with the browser bar hidden. When the bar is showing, the last lines can also sit under it.
   - The tallest tooltip is the dynamics formula legend: 396 px at 1200 px wide and 498 px at 390 px wide. Texts the cap would cut, by window height:

     | Window height | At 1200 px wide | At 390 px wide |
     |---|---|---|
     | 600 px | 0 | 0 |
     | 500 px | 0 | 1 |
     | 400 px | 1 | 4 |
     | 360 px (a phone held sideways) | 1 | 5 |
     | 300 px | 6 | 12 |
     | 250 px | 11 | 143 |
     | 200 px | 198 | 629 |

   - A probe in a 260-px-tall window shows the cut: `shots/tt-inv-1200x260-dynamics-legend.png`. The legend stops after "…for the thrusters and the water alike." in the middle of a line.
   - The Stalls cell tooltip is 172 px tall. It loses its last line ("throttle cost keeps rising.") only in a window under about 188 px tall, such as a small embedded preview. This is the only way version 10 can show it ending at "…while its throttle cost" or "…while its".
   - To fix the cause:
     - keep every tooltip within a height budget; the formula legends are the tallest, and the full legend is already printed beside each formula;
     - make `place()` widen the box, or switch to a compact layout, before anything would be clipped;
     - add a test that fails whenever `scrollHeight > clientHeight` at short windows.
3. **Native titles come back on the weighting bars (live).**
   - What happens: `drawBars()` creates an SVG `<title>` for each bar on every redraw, but the e7 script converts titles to `data-tip` only once, at load.
   - The effect: after a lean button or the slider is used, the three bars show the browser's own tooltip. That tooltip has no full stop, and the Tooltips switch cannot turn it off.
   - Probe result: before, there were 3 `data-tip` attributes and 0 titles. After clicking "Strong", there were 0 and 3.
4. **The cell-text reader `plain()` loses information (live; not a cut, but the text reads as broken).**
   - A `<pre>` code block loses its line breaks. The summary-table "import" row becomes "import experiments.shift.thruster_chain # registers the 8 ids import gymnasium as gym env = …".
   - A code line and its explanation are joined with only a space: "Box(-1.0, 1.0, (4,), float32) 4 numbers".
   - A value that ends with a full stop is followed by ", ": "…the same for every bank., Role: 0 points". This happens in 5 rows of the reference-controllers table.
   - A superscript it cannot map becomes "e^(2c)" (weighting table), and an unmappable subscript becomes "P_cap".

## Method and files

**Scripts** (in `page/scripts/`; none of them writes the page):

| Script | What it does |
|---|---|
| `tt_harvest.py` | Loads a copy of the page in Chromium with three exported hooks: `build()`, `channelTip()` and `costTip()`. It calls the page's own generators on all 396 `data-tip` and `data-tip-ref` anchors, all 235 column headers and all 1,917 cells. It also samples the channel drawing over 2 worlds × 4 situations × 4 controllers × 100 states × 15 parts, and the cost chart at 960 points. It measures every tooltip's height at 1200 and 390 px wide. |
| `tt_probe.py` | Hovers the real cells with the mouse at 1200×800, 390×844, 1200×260 and 390×300, and checks the weighting bars. |
| `tt_audit.py` | Runs the automatic checks: sentence form, acronyms, code, markup, ellipses, numbers against the page text, and the height cap. |
| `tt_audit_notes.py` | Holds the manual judgements, made after reading every text. Each problem in the inventory records whether it came from an automatic check or a manual review. |

**Outputs:**
- `page/tmp/inv/harvest.json` and `page/tmp/inv/probe.json`;
- the screenshots `page/shots/tt-inv-*.png`.

**Caveats:**
- The fragment check is heuristic, but every text was read and the misses and false hits found were corrected by hand. A few borderline definitions ("X: Y" style) may still be counted differently by another reader.
- The accuracy check verified numbers and spot-checked claims. It did not prove every sentence.
