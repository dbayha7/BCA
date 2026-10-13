"""Apply the task-diagram demo episode (comment 5) to a copy of thrusterchain.html.

Usage (WSL):  python3 demo_apply.py IN.html OUT.html [--caption]
  Without --caption: adds the CSS, the lead-in line, the controls and the demo script.
  With --caption:    also rewrites the caption sentences and the two drawings' aria-labels that describe the
                     still picture (they become wrong once the drawing plays). Skip it if the caption was rewritten.
Every edit asserts that its anchor occurs exactly once; nothing is written unless all edits succeed.
IN and OUT may be the same file. The backup thrusterchain.v5.html must never be passed as OUT.
"""
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, "demo_proto.js")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
assert len(args) == 2, __doc__
src, dst = args
assert not dst.endswith("thrusterchain.v5.html"), "never modify the backup"
s = open(src, encoding="utf-8").read()
n0 = len(s)


def once(hay, needle, what):
    c = hay.count(needle)
    assert c == 1, f"{what}: expected exactly one match, found {c}"
    return hay.index(needle)


assert 'id="td-ctrls"' not in s, "the demo is already applied"

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
.td-ctrls { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.td-ctrls[hidden] { display: none; }
.td-ctrls .tlabel { min-width: 128px; }
.td-lead { max-width: 86ch; }
"""
i = once(s, CSS_ANCHOR, "CSS anchor") + len(CSS_ANCHOR)
s = s[:i] + CSS + s[i:]

# 2. the lead-in line, first thing in the figure
FIG_OPEN = '<figure class="ovfig" id="task-diagram">\n'
LEAD = """      <p class="note td-lead">Playing: one real episode of the <i>stall-aware controller</i>, the best controller this
        page defines (the textbook rule described below, with every throttle capped at 80%) and the 100-point
        reference for scores. Nothing proves it is the best possible controller. The episode starts from one fixed start
        in the recorded conditions: current 0.445, wave amplitude 0.218, floats out of place.</p>
"""
i = once(s, FIG_OPEN, "figure opening") + len(FIG_OPEN)
s = s[:i] + LEAD + s[i:]

# 3. the controls, just before the figure's caption
CTRLS = """      <div class="td-ctrls" id="td-ctrls" hidden>
        <div class="transport">
          <button class="btn" id="td-play" type="button">Play</button>
          <button class="btn ghost" id="td-back" type="button" aria-label="Step back">&#8592; Step</button>
          <button class="btn ghost" id="td-fwd" type="button" aria-label="Step forward">Step &#8594;</button>
          <input type="range" id="td-slider" min="0" max="99" value="0" step="1" aria-label="Step of the demo episode">
          <span class="tlabel" id="td-step">step 1 of 100</span>
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

# 4. the script, as its own block right after the script that defines D
d0 = once(s, "const D = ", "data object")
e0 = s.index("</script>", d0) + len("</script>")
proto = open(PROTO, encoding="utf-8").read()
assert "</script" not in proto
s = s[:e0] + "\n\n<script>\n" + proto.rstrip() + "\n</script>" + s[e0:]

# 5. optional: the words that describe the still picture
if "--caption" in flags:
    def sub_ws(text, old_words, new, what):
        pat = r"\s+".join(re.escape(w) for w in old_words.split())
        m = list(re.finditer(pat, text))
        assert len(m) == 1, f"{what}: expected exactly one match, found {len(m)}"
        return text[:m[0].start()] + new + text[m[0].end():]

    s = sub_ws(s, "The ± beside each arrow is the wave, which adds to the current's push or takes from it.",
               "The sign beside each arrow is the wave at that step: + when it adds to the current's push, − when it "
               "takes from it (the faint dashed part is the push it takes away).",
               "caption: the wave's sign")
    s = sub_ws(s, "The floats are drawn, to scale, where a steady current would leave them if the thrusters were off: "
                  "float 1 is carried furthest. The thrusters' arrows show only which way a negative throttle pushes.",
               "The drawing plays the episode one step at a time: each float is drawn where it is at that step, all five "
               "to the same scale, and each thruster's arrow is as long as the push it gives. With the thrusters off, "
               "a steady current would carry float 1 furthest from rest, so thruster 1 works hardest: the bar at the "
               "bottom fills to its throttle, and the dark tick marks the push it really gives.",
               "caption: the still picture")
    old_aria = "The floats sit below the dashed rest line, float 1 furthest and float 5 nearest."
    new_aria = ("The drawing plays one recorded episode of the stall-aware controller: the floats move to where they "
                "are at each step, each thruster's arrow is as long as its push, and the throttle bar fills to "
                "thruster 1's throttle.")
    for cls_ in ("dg dg-wide", "dg dg-narrow"):
        a0 = once(s, f'<svg class="{cls_}"', f"drawing {cls_}")
        a1 = s.index(">", a0)
        tag = s[a0:a1]
        assert tag.count(old_aria) == 1, f"aria-label of {cls_}: expected exactly one match"
        s = s[:a0] + tag.replace(old_aria, new_aria) + s[a1:]

open(dst, "w", encoding="utf-8").write(s)
print(f"wrote {dst}: {n0} -> {len(s)} chars")
