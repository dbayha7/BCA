"""Edit 2 (comment 5): the task diagram plays one real episode of the stall-aware controller.

usage (WSL): python3 e2_apply.py
  The first run saves the page as it stands after edit 1 to page/tmp/e2_base.html. Every run rebuilds the page from that
  base, so it can be re-run while the demo is tuned. It refuses to write if the page was changed by anyone else since
  (its md5 must be the base's or this script's last output). The backup thrusterchain.v5.html is never touched.
What it does, every edit asserted to match exactly once:
  1. CSS for the playing drawing, its lead-in and its controls (after the drawing's own rules).
  2. id="task-counteract" on the heading "What a controller does to counteract" (the lead-in links to it).
  3. The lead-in at the top of the figure: what plays, which controller, from which start, what to watch.
  4. The controls and readouts, just before the figure's caption.
  5. The demo script (e2_demo.js) as its own <script>, right after the script that defines the data object D.
  6. The caption sentences and the two drawings' aria-labels that described the still picture.
"""
import hashlib
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e1_common import BACKUP, PAGE, data_line_hash, md5, once, replace_once, sub_ws  # noqa: E402

BASE = os.path.join(os.path.dirname(PAGE), "tmp", "e2_base.html")
STAMP = os.path.join(HERE, ".e2_last_md5")
DEMO_JS = os.path.join(HERE, "e2_demo.js")

bk = md5(BACKUP)
if not os.path.exists(BASE):
    s0 = open(PAGE, encoding="utf-8").read()
    assert 'id="td-ctrls"' not in s0, "the page already has a demo; no clean base to start from"
    os.makedirs(os.path.dirname(BASE), exist_ok=True)
    shutil.copyfile(PAGE, BASE)
    print("saved base", BASE, md5(BASE))
allowed = {md5(BASE)}
if os.path.exists(STAMP):
    allowed.add(open(STAMP).read().strip())
cur = md5(PAGE)
if cur not in allowed:
    sys.exit(f"refusing: {PAGE} was changed by someone else since edit 2 (md5 {cur})")

s = open(BASE, encoding="utf-8").read()
assert 'id="td-ctrls"' not in s, "the base already has the demo"
n0 = len(s)
data0 = data_line_hash(s)
mjx0 = s.count("<mjx-container")

# 1. CSS, after the last rule of the task diagram's drawing styles
CSS_ANCHOR = "svg.dg .dg-n { fill: var(--ink); font-weight: 700; }\n"
CSS = """/* the task diagram playing its demo episode (thruster 1's throttle fills the bar; the dark tick is its push) */
svg.dg .dg-fill { fill: var(--accent); opacity: .3; }
svg.dg .dg-fill.w { fill: var(--warn); }
svg.dg .dg-ptick { stroke: var(--ink); stroke-width: 2.5; }
svg.dg .dg-wavepart.faint { opacity: .45; }
svg.dg .dg-thr.w { fill: var(--warn); }
svg.dg .dg-thrarrow.w { stroke: var(--warn); fill: var(--warn); }
svg.dg .dg-halo { paint-order: stroke; stroke: var(--water); stroke-width: 4px; stroke-linejoin: round; }
.td-lead { display: flex; flex-direction: column; gap: 6px; font-size: 15.5px; line-height: 1.5; color: var(--ink-2); }
.td-lead p, .td-lead ul { max-width: 80ch; }
.td-lead ul { gap: 4px; padding-left: 20px; }
svg.dg .dg-sign { fill: var(--water-ink); font-weight: 700; font-size: 15px; }
.td-lead b { color: var(--ink); }
.td-k { font-family: var(--display); font-size: 16px; font-weight: 700; color: var(--accent-ink); line-height: 1.3; }
.td-ctrls { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.td-ctrls[hidden] { display: none; }
.td-ctrls .tlabel { min-width: 15ch; }
"""
i = once(s, CSS_ANCHOR, "CSS anchor") + len(CSS_ANCHOR)
s = s[:i] + CSS + s[i:]

# 2. an id for the heading the lead-in links to
s = replace_once(s, "<h3>What a controller does to counteract</h3>",
                 '<h3 id="task-counteract">What a controller does to counteract</h3>', "counteract heading")
assert 'id="task-counteract"' in s

# 3. the lead-in, first thing in the figure
FIG_OPEN = '<figure class="ovfig" id="task-diagram">\n'
LEAD = """      <div class="td-lead">
        <p class="td-k">Demo: the stall-aware controller plays one real episode</p>
        <p>The drawing below replays one recorded episode state by state, from state 0 (the start) to state 99,
          counted as in the live panel above. The throttles are set by the <i>stall-aware controller</i>, the best
          controller this page defines: it scores 100 points on the page's score scale, where the recording controller,
          which made the training data, scores 0. It sets each throttle by the textbook rule (explained below, under
          <a href="#task-counteract">What a controller does to counteract</a>) and caps it at 80%, so its propellers
          never stall. Nothing proves that it is the best possible controller. The start is fixed, in the recorded
          conditions, and is the same one the live panel uses: current 0.445, wave amplitude 0.218, the floats out of
          place.</p>
        <p><b>What to watch</b></p>
        <ul>
          <li><b>States 0 to 2:</b> thruster 1 works at its 80% limit, the most it ever gives, while the floats begin
            to come back toward the dashed rest line.</li>
          <li><b>From state 11 on:</b> every float stays within 0.3 of rest, and thruster 1's throttle (the bar at the
            bottom of the drawing) rises and falls with the water's push. The harder the water pushes downstream, the
            harder thruster 1 pushes upstream.</li>
          <li><b>All episode:</b> no throttle goes past 80%, so nothing stalls. The return, the rewards added up (closer
            to 0 is better), ends at &#8722;21.87 on this start. Over all 1,000 starts of the recorded conditions, this
            controller's average return is &#8722;26.4.</li>
        </ul>
      </div>
"""
i = once(s, FIG_OPEN, "figure opening") + len(FIG_OPEN)
s = s[:i] + LEAD + s[i:]

# 4. the controls, just before the figure's caption
CTRLS = """      <div class="td-ctrls" id="td-ctrls" hidden>
        <div class="transport">
          <button class="btn" id="td-play" type="button">Play</button>
          <button class="btn ghost" id="td-back" type="button" aria-label="Back one state">&#8592; Step</button>
          <button class="btn ghost" id="td-fwd" type="button" aria-label="Forward one state">Step &#8594;</button>
          <input type="range" id="td-slider" min="0" max="99" value="0" step="1" aria-label="State shown in the demo episode">
          <span class="tlabel" id="td-step">state 0 (start)</span>
        </div>
        <div class="readouts" aria-live="off">
          <div class="ro"><span class="k">water's push on every float</span><span class="v" id="td-push">0.626</span><span class="k" id="td-pushparts">current 0.445 + wave 0.180</span></div>
          <div class="ro"><span class="k">thruster 1's throttle (the bar's fill)</span><span class="v" id="td-thr">80% upstream</span></div>
          <div class="ro"><span class="k">thruster 1's push (the dark tick)</span><span class="v" id="td-thrpush">0.80</span></div>
          <div class="ro"><span class="k">return so far (closer to 0 is better)</span><span class="v" id="td-ret">&#8722;2.09</span></div>
        </div>
      </div>
"""
f0 = once(s, 'id="task-diagram"', "figure id")
f1 = s.index("</figure>", f0)
fig = s[f0:f1]
assert fig.count("<figcaption") == 1, "figure caption: expected exactly one"
cap = f0 + fig.index("<figcaption")
line_start = s.rindex("\n", 0, cap) + 1
assert s[line_start:cap].strip() == "", "caption must start its own line"
s = s[:line_start] + CTRLS + s[line_start:]

# 5. the script, as its own block right after the script that defines D
d0 = once(s, "\nconst D = ", "data object")
e0 = s.index("</script>", d0) + len("</script>")
js = open(DEMO_JS, encoding="utf-8").read()
assert "</script" not in js
s = s[:e0] + "\n\n<script>\n" + js.rstrip() + "\n</script>" + s[e0:]

# 6. the words that described the still picture
f0 = once(s, 'id="task-diagram"', "figure id")
f1 = s.index("</figure>", f0)
s = sub_ws(s, "The ± beside each arrow is the wave, which adds to the current's push or takes from it.",
           "The sign beside each arrow is the wave in the state shown: + when it adds to the current's push, − when "
           "it takes from it (the faint dashed part is the push it takes away).",
           "caption: the wave's sign", f0, f1)
f1 = s.index("</figure>", f0)
s = sub_ws(s, "The floats are drawn, to scale, where a steady current would leave them if the thrusters were off: "
              "float 1 is carried furthest. The thrusters' arrows show only which way a negative throttle pushes.",
           "The drawing plays the episode one state at a time. Each float is drawn where it is in that state, all "
           "five to the same scale, and each thruster's arrow is as long as the push it gives. With the thrusters "
           "off, a steady current would carry float 1 furthest from rest, so thruster 1 works hardest: the bar at "
           "the bottom fills to its throttle, and the dark tick marks the push it really gives.",
           "caption: the still picture", f0, f1)
old_aria = "The floats sit below the dashed rest line, float 1 furthest and float 5 nearest."
new_aria = ("The drawing plays one recorded episode of the stall-aware controller, state by state: the floats move to "
            "where they are in each state, each thruster's arrow is as long as its push, and the throttle bar fills "
            "to thruster 1's throttle.")
for cls_ in ("dg dg-wide", "dg dg-narrow"):
    a0 = once(s, f'<svg class="{cls_}"', f"drawing {cls_}")
    a1 = s.index(">", a0)
    tag = s[a0:a1]
    assert tag.count(old_aria) == 1, f"aria-label of {cls_}: expected exactly one match"
    s = s[:a0] + tag.replace(old_aria, new_aria) + s[a1:]

# checks: data line and math untouched, ids unique, every in-page link resolves
assert data_line_hash(s) == data0, "the data line changed"
assert s.count("<mjx-container") == mjx0, "the math changed"
ids = re.findall(r'\bid="([^"]+)"', s)
assert len(ids) == len(set(ids)), "duplicate ids"
missing = {h for h in re.findall(r'(?<![:\w])href="#([^"]+)"', s) if h not in set(ids)}
assert not missing, f"links that do not resolve: {missing}"

open(PAGE, "w", encoding="utf-8").write(s)
open(STAMP, "w").write(md5(PAGE))
assert md5(BACKUP) == bk, "the backup changed"
print(f"wrote {PAGE}: {n0} -> {len(s)} chars; md5 {md5(PAGE)}; backup md5 {bk}")
