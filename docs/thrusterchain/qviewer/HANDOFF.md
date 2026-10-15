# Handoff: the section "The critic in action"

The section is ready to merge. It was tested on a fresh copy of the working page (the 11:01 copy, 2,364,918 bytes). Every check passes there. I did not touch anything in `page\`.

## 1. Insert it into the live working copy

Run this from Windows. It writes the page in place, on CPUs 0–3 at nice 10:

```
wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/qviewer && PYTHONDONTWRITEBYTECODE=1 nice -n 10 taskset -c 0-3 /opt/bca-shift-venv/bin/python insert_qviewer.py ../page/thrusterchain.html --in-place-page'
```

- **Write to a separate file instead:** replace `--in-place-page` with `--out ../page/thrusterchain.qviewer.html`.
- **Update an earlier insertion with newer files:** add `--replace`. It takes out the old blocks first.
- **When it refuses:** it writes nothing and exits with status 2. This happens if any anchor is missing or appears twice, or if the section is already on the page and `--replace` was not given. Anchors: `<div class="wrap">`, `<section id="methods">`, `<nav class="toc">`, and the `#methods` link inside it.
- **Taking it out:** removing the marked blocks gives back the original page byte for byte. This was tested.

## 2. What it inserts, and where

Five blocks go in, each between `qviewer:begin` and `qviewer:end` markers:

| # | Content | Where | Size |
|---|---|---|---|
| 1 | `style.css` | before the first `</style>` that comes ahead of `<div class="wrap">` | 8,281 B |
| 2 | `section.html`, unchanged: `<section id="critic-in-action">` | on its own lines, immediately before `<section id="methods">` | 64,873 B |
| 3 | `<a href="#critic-in-action">The critic in action</a>` | in the contents rail, on the line before the link to `#methods` | – |
| 4 | `qv_data.json` as the page's only `<script type="application/json" id="qv-data">` | after the page's last character | 1,242,191 B |
| 5 | `script.js` as a `<script>` | after the data | 49,917 B |

The page grows from 2,364,918 to 3,730,588 bytes.

The script has to come after the page's own scripts, and appending it at the end does that. The reason: the tables in `section.html` already hold their rows, with a dash in each cell. That way the page's own table tooltips and its card layout for narrow screens set them up when the page loads. `script.js` then changes only the text in the cells.

## 3. Files (`C:\Users\David\dev\bca-shift-work\qviewer\`)

- **Inserted:**
  - `section.html`
  - `style.css`
  - `script.js`
  - `qv_data.json`
- **Insertion:** `insert_qviewer.py`.
- **Data:**
  - `build_qv_data.py` builds `qv_data.json` from `precise\q_data.json` (7 significant digits) and `qv_extra.json`.
  - The sweeps are now stored exactly, in millionths, and decode to the computed values with no rounding.
  - `compute_q.py`, `compute_q_precise.py`, `ball.py` and `enc_probe.py` (sizes of the possible encodings) are also here.
- **Checks:**
  - `run_checks.sh` runs everything below on a fresh copy of the page.
  - `ins_test.sh` checks the refusals, `--replace` and taking the insertion out.
  - `qv_test.py` covers drawing, keyboard, play and console errors.
  - `qv_wt_test.py` covers the walkthrough pop-out.
  - `final_check.py` compares every shown number with the 7-digit file, and checks the table tooltips, Esc in the pop-out, readouts at 390 px and labels against lines.
- **Screenshots:**
  - `final_shots.sh` and `final_shots_extra.py` produce `shots\final-*.png` (390 px light and 1200 px dark).
  - `test_page.html` is today's page with the section inserted.
  - `prev\` holds the previous version of the files, for comparison only.

## 4. Checks to run after merging

These only read the page. The pop-out check saves two screenshots into `qviewer\shots`.

```
wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/qviewer && export QV_PAGE=/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html PYTHONDONTWRITEBYTECODE=1 && taskset -c 2-3 nice -n 10 /opt/shot-venv/bin/python qv_test.py | tail -3 && taskset -c 2-3 nice -n 10 /opt/shot-venv/bin/python qv_wt_test.py && taskset -c 2-3 nice -n 10 /opt/shot-venv/bin/python final_check.py'
```

**Expected output:** `ALL CHECKS PASSED`, `WALKTHROUGH CHECKS PASSED` and `ALL FINAL CHECKS PASSED`.

- **Numbers:** 252,000 table cells and 2,000 "true value lost" readouts are compared with the 7-digit file. 0 should differ.
- **Console and layout:** no console errors, and no horizontal overflow at 390 px or 1200 px.
- **Walkthrough:** the section is "Section 15 of 23" today. That number will change when sections move, and the test does not check it.

Also check these on the page:

- The page has exactly one `id="critic-in-action"`, one `id="qv-data"` and one `href="#critic-in-action"`.
- The section comes directly before `<section id="methods">`, and its rail link comes directly before the Methods link.
- Your usual page checks pass: the tooltip audit, overflow and colour checks.

## 5. Things to know

- **Notation:**
  - The section writes the critic as Q(s, a), with s the full state, and the true value as Q^π(s, a).
  - The Methods section writes Q(x, a).
  - The tooltips write "the true value" and never use "^".
- **Context sentence:** the section now opens with one sentence explaining that it uses the weighting test's data, not the data described above it. In that data:
  - The recording controller never pushes past 80% (it uses the 80% hold, not full throttle).
  - The removed ball takes in that 80% hold.
  - Points run from uniformly random commands (0) to the stall-aware controller (100).
- **Header tooltips:**
  - Column headers show their description on hover or tap, from the section's own header tooltips.
  - Cells show their whole row through the page's table tooltips.
  - Keyboard focus on a header shows no description, because the page's COLS list has no entry for `critic-in-action`. Adding one is optional.
- **Upper chart:** it opens on 60% to 100%; "0% to 100%" shows every throttle.

**Corrections to the earlier data report** (the page computes these from the data and is right):

- In the strong mix, the critic's average change from 80% to 100% runs from −0.77 to +0.59, not from −0.46.
- The "flagged states cluster early, water push 0.82 against 0.64" figures hold only for the expert, hold-removed, strong-mix episode, and that cluster starts at state 0. In the lower qualities the flagged states have less push than the rest.

## 6. For the owner, in three sentences

The new section replays one real episode of a trained actor and shows, at every state, the critic's score Q(s, a) for each throttle of thruster 1 next to the exact true value from simulation, so you can see where the critic's top-rated throttle â and the truly best throttle part ways. In the strong test mix the true value falls sharply past the 80% stall limit, by 1.65 to 2.74 on average from 80% to 100%, but the critic's value falls by at most 0.77 and with the medium, medium-expert and expert data it rises (by 0.005 to 0.59), so the critic rates the stalled 100% above 80% at 11.0% to 66.1% of the states, while the true value is lower at 100% at 75.6% to 84.5% of them. The critic trained on all the data makes the same mistake (at 11.0% to 62.5% of the states, against 26.4% to 66.1% with the 80% hold removed), so removing the hold is not the cause: the recorded data hold almost no commands past 80%, none in the expert data and only random ones, 3.5% to 10.0% of the training steps, in the others.
