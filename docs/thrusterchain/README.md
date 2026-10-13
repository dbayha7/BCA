# ThrusterChain page

`thrusterchain.html` is the ThrusterChain documentation page, version 13, published as a private claude.ai artifact
(https://claude.ai/artifact/8VbJdfNyQr1qDmr7JN1Ha7). It is an HTML fragment; publishing adds the doctype, head and body.

The page describes the environment and its controlled distribution shift, the main test (stopped before BCA, entry 16
in `EXPERIMENTS.md`), Stage 1, and the known-shift weighting test (entry 17).

## What is here

- `thrusterchain.html`: the page itself.
- `scripts/`: the Python scripts that edited the page. Every edit is an exact, counted replacement. The folder also holds
  the browser checks and the tooltip sources (`ta_texts.py`, `ta_code.py`).
- `notes/`: the owner's comment queue (`followup_queue.md`), the colour-system design (`p4_design.md`), the verified
  facts behind the page's numbers (`p4_facts.md` / `.json`), and the notation and structure notes.
- `qviewer/`: the sources of the "critic in action" section, with its data and tests.
- `mathbuild/`: the MathJax renderer that turns the page's formulas into pre-rendered SVG (run `npm ci` first).

## Checks

The checks run in WSL with `/opt/shot-venv/bin/python` (Playwright), from the folder that holds the page:

- `scripts/p3f_check.py --page thrusterchain.html` must pass in full.
- `scripts/s1_wt_test.py` runs at widths 1200, 390, and 390 in dark mode with reduced motion.
- `qviewer/qv_test.py`, `qv_wt_test.py` and `final_check.py` need `QV_PAGE` set to the page path.

The old walkthrough test, `scripts/e1_wt_test.py`, fails two checks because it expects an earlier section order.

The scripts were written in the work folder, `C:\Users\David\dev\bca-shift-work\page`, so some of them name paths
there. Screenshots, scratch output and older page copies stay in that work folder.
