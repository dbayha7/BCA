#!/usr/bin/env python3
"""Page pass 4, edit 3 of 4: Methods in full and the harm checks (draft B, facts checked), on top of edits 1 and 2.

- Methods (#methods), rebuilt: a map of the methods onto the offline reinforcement-learning pipeline; the words and
  scores; TD3+BC's critic (with the two kinds of critic error and a worked example) and actor (the dose as a behaviour
  constraint, the lower edge as the value-pessimism channel); TD3+BC's settings in four groups, each with what it does
  and why it matters here; how far the actor moves (problem, steps, worked example; the measured parts, the share table
  and the forecast kept); behaviour cloning; fixed-controller critics; the critic swap; caution placements; BCA in three
  steps; the lower edge; BCA's placement controls. Each piece opens with its pipeline step; every legend has one
  symbol per row (the actor's legend in groups).
- The harm checks (#harm-checks): the critic-error check as a table of question / pass mark / result with a "how it is
  measured" line; a step note on each of the other three checks; the seed-count legend in groups with a worked
  example; the stop rule with its reasons, the words it uses and an if / then / why table.
- Column descriptions (COLS) for the new and rebuilt tables; a style block for the new parts.

Every replacement is exact and counted. Math: every formula is either copied from the page (same TeX label, same
display flag) or rendered by mathbuild/render.js from the job file this script writes with --jobs; every copied or
new container gets fresh glyph ids (prefix MJXE).

Run from the page directory (WSL):
    python3 scripts/p4_e3_apply.py --jobs     (write mathbuild/jobs/p4e3.json with the formulas the page lacks)
    python3 scripts/p4_e3_apply.py --check    (build in memory, write tmp/p4/edit3/applied.html only)
    python3 scripts/p4_e3_apply.py            (apply to thrusterchain.html)
"""
import hashlib
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mjx  # noqa: E402

PAGE_DIR = os.path.dirname(HERE)
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
BEFORE = os.path.join(PAGE_DIR, "tmp", "p4", "edit3.before.html")
TRIAL = os.path.join(PAGE_DIR, "tmp", "p4", "edit3", "applied.html")
SRC = os.path.join(HERE, "p4_e3_src.html")
CSS = os.path.join(HERE, "p4_e3.css")
JOB = os.path.join(mjx.MATHBUILD, "jobs", "p4e3.json")
OUT = os.path.join(mjx.MATHBUILD, "out", "p4e3.json")
E2_MD5 = "a28a5130f69b8580e26cde1e032690b8"
JOBS = "--jobs" in sys.argv
CHECK = "--check" in sys.argv
PREFIX = "MJXE"

s = open(PAGE, encoding="utf-8").read()
ORIG = s
assert hashlib.md5(open(BEFORE, "rb").read()).hexdigest() == E2_MD5, "backup changed"
assert hashlib.md5(s.encode("utf-8")).hexdigest() == E2_MD5, "the page is not the edit-2 page this edit was written for"
QV_BLOCKS = [m.group(0) for m in re.finditer(r"<!-- qviewer:begin (\w+) -->.*?<!-- qviewer:end \1 -->|/\* qviewer:begin style \*/.*?/\* qviewer:end style \*/|<script type=\"application/json\" id=\"qv-data\">.*?</script>", s, flags=re.S)]
assert len(QV_BLOCKS) >= 3, len(QV_BLOCKS)
assert PREFIX + "-" not in s
N = {"n": 0}
MATH_BEFORE = s.count("<mjx-container")


def rep(old, new, count=1, lo=None, hi=None):
    """exact replacement, asserted count; optionally only inside s[lo:hi]"""
    global s
    lo = 0 if lo is None else lo
    hi = len(s) if hi is None else hi
    part = s[lo:hi]
    n = part.count(old)
    assert n == count, f"expected {count} of {old[:100]!r}, found {n}"
    s = s[:lo] + part.replace(old, new) + s[hi:]
    N["n"] += 1


def section(sid, page=None):
    page = s if page is None else page
    assert page.count(f'<section id="{sid}"') == 1, sid
    a = page.index(f'<section id="{sid}"')
    return a, page.index("</section>", a) + len("</section>")


def between(start, end, lo, hi, page=None, include_end=False):
    """span of page from `start` up to `end` (included if include_end), start unique inside page[lo:hi]"""
    page = s if page is None else page
    part = page[lo:hi]
    assert part.count(start) == 1, (start[:80], part.count(start))
    a = lo + part.index(start)
    b = page.index(end, a)
    assert b < hi, end[:80]
    return a, b + (len(end) if include_end else 0)


def esc(text):
    return html.escape(text, quote=True)


# ------------------------------------------------------------------------------------------------- tooltips
def page_tip(title):
    """(title attribute, data-tip attribute) for a tooltip title used on the original page, verbatim. A title can be
    written with &#x27; or with a plain apostrophe; when it has more than one text, the one in Methods or the harm
    checks wins."""
    for t in (esc(title), title.replace('"', "&quot;")):
        hits = [(m.start(), m.group(1)) for m in re.finditer(r'data-tip-title="%s" data-tip="([^"]*)"' % re.escape(t), ORIG)]
        if hits:
            texts = {x for _, x in hits}
            if len(texts) > 1:   # Methods' or the harm checks' own text, else the first (the term's definition)
                a, b = section("methods", ORIG)[0], section("harm-checks", ORIG)[1]
                inside = {x for i, x in hits if a <= i < b}
                assert len(inside) <= 1, (title, len(texts))
                return t, (inside.pop() if inside else hits[0][1])
            return t, texts.pop()
    raise AssertionError(f"no tooltip titled {title!r} on the page")


NEW_TIPS = {}


def new_tip(title, text):
    assert text.endswith(".") and text[0].isupper(), text
    NEW_TIPS[title] = text
    return esc(title), esc(text)


# ------------------------------------------------------------------------------------------------- math
CONT = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
EXISTING = {}
for m in CONT.finditer(ORIG):
    EXISTING.setdefault((html.unescape(m.group(1)), bool(m.group(2))), m.group(0))
RENDERED = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) and not JOBS else {}
NEEDED = {}


def math(tex, display=False):
    key = (tex, display)
    if key in EXISTING:
        return EXISTING[key]
    k = ("D:" if display else "I:") + tex
    NEEDED[k] = {"key": k, "tex": tex, "display": display}
    if k in RENDERED:
        c = RENDERED[k]
        m = CONT.match(c)
        assert m and html.unescape(m.group(1)) == tex and bool(m.group(2)) == display, k
        return c
    return f"<!--MISSING {esc(tex)}-->"


# ------------------------------------------------------------------------------------------------- the source text
src = open(SRC, encoding="utf-8").read()
PARTS = dict(re.findall(r"<!--PART (\w+)-->\n(.*?)\n<!--/PART-->", src, flags=re.S))
assert set(PARTS) == {"methods", "hc_critic", "hc_seed_legend", "hc_stop"}, set(PARTS)
KEEP = {}


def expand(text):
    def one(m):
        f = m.group(1).split("::")
        kind = f[0]
        if kind == "M":
            return math(f[1])
        if kind == "MD":
            return math(f[1], display=True)
        if kind in ("S", "s"):
            t, d = page_tip(f[1])
            foc = ' data-tip-focus="no"' if kind == "s" else ""
            return f'<span class="sym" data-tip-title="{t}" data-tip="{d}"{foc}>{math(f[2])}</span>'
        if kind == "SN":
            t, d = new_tip(f[1], f[2])
            return f'<span class="sym" data-tip-title="{t}" data-tip="{d}">{math(f[3])}</span>'
        if kind == "T":
            t, d = page_tip(f[1])
            return f'<span class="term" data-tip-title="{t}" data-tip="{d}">{f[2]}</span>'
        if kind == "TN":
            t, d = new_tip(f[1], f[2])
            return f'<span class="term" data-tip-title="{t}" data-tip="{d}">{f[3]}</span>'
        if kind == "KEEP":
            return KEEP[f[1]]
        raise AssertionError(m.group(0)[:80])
    out = re.sub(r"⟦(.*?)⟧", one, text, flags=re.S)
    assert "⟦" not in out and "⟧" not in out
    return out


# ------------------------------------------------------------------------------------------------- 1. Methods
ma, mb = section("methods")
OLD_METHODS = s[ma:mb]
lo, hi = ma, mb
# the kept blocks, verbatim from the current section
a, b = between("<p><b>Measuring the critic's pull", "\n    <h3>Behaviour cloning</h3>", lo, hi)
keep = s[a:b]
assert keep.count("<table>") == 2 and keep.count("Forecast harm, curving critic") == 1 and keep.rstrip().endswith("</p>")
# the forecast table's situation rows wear their situation's colour edge (the colour system; the name stays)
for name, key in [("Recorded conditions", "recorded"), ("Strong current", "strong_current"), ("Big waves", "big_waves"),
                  ("Big starting pushes", "big_kicks")]:
    old = f'<tr><td>{name}</td><td class="n">'
    assert keep.count(old) == 1, name
    keep = keep.replace(old, f'<tr data-sit="{key}"><td>{name}</td><td class="n">')
KEEP["how_far_rest"] = "    " + keep.strip()
a, b = between('<div class="chartbox">', '<div class="symkey"><p class="symkey-h">What the symbols in the doses and chart mean</p>', lo, hi)
KEEP["chartbox"] = "    " + s[a:b].strip()
assert KEEP["chartbox"].endswith("</div>") and KEEP["chartbox"].count('<div class="legend">') == 1
a, b = between('<div class="tbl"><table>\n<thead><tr><th class="n" scope="col">Over-preference Δ ÷ its mean', "</tbody></table></div>", lo, hi, include_end=True)
KEEP["dose_table"] = s[a:b]
new_methods = expand(PARTS["methods"])

# ------------------------------------------------------------------------------------------------- 2. Harm checks
ha, hb = section("harm-checks")
new_critic = expand(PARTS["hc_critic"])
new_seed_legend = expand(PARTS["hc_seed_legend"])
new_stop = expand(PARTS["hc_stop"])

if JOBS:
    os.makedirs(os.path.dirname(JOB), exist_ok=True)
    json.dump(list(NEEDED.values()), open(JOB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(NEEDED)} formulas to render -> {JOB}")
    for k in NEEDED:
        print("  ", k)
    sys.exit(0)
assert not [k for k in NEEDED if k not in RENDERED], [k for k in NEEDED if k not in RENDERED]

# fresh glyph ids for every container placed by this edit, numbered after the highest MJXE number (none yet)
counter = {"n": 0}


def retag_all(text):
    def one(m):
        c = m.group(0)
        counter["n"] += 1
        found = {(x.group(2), x.group(3)) for x in mjx.IDNUM.finditer(c)}
        assert len(found) <= 1, found
        return mjx.IDNUM.sub(lambda x: f"{x.group(1)}{PREFIX}-{counter['n']}-", c)
    return CONT.sub(one, text)


new_methods = retag_all(new_methods)
new_critic = retag_all(new_critic)
new_seed_legend = retag_all(new_seed_legend)
new_stop = retag_all(new_stop)

# 1. splice Methods (the whole section)
s = s[:ma] + new_methods + s[mb:]
N["n"] += 1

# 2a. the critic-error check block
ha, hb = section("harm-checks")
a, b = between('<div class="check">\n        <h3>The critic-error check: is the critic wrong where data are missing?</h3>',
               '      <div class="check">\n        <h3>The return-cost check', ha, hb)
old_block = s[a:b]
assert old_block.count('<ol class="marks">') == 1 and old_block.rstrip().endswith("</div>")
s = s[:a] + new_critic + "\n" + s[b:]
N["n"] += 1

# 2b. one line on the pipeline step for each of the other three checks, and a link to the method each one uses
ha, hb = section("harm-checks")
rep('''        <h3>The return-cost check: does acting on that error cost return?</h3>
        <p class="uses">Uses the critic swap and the step-by-step loss accounting, scored in strong current, in points.</p>''',
    '''        <h3 id="hc-cost">The return-cost check: does acting on that error cost return?</h3>
        <p class="stepnote"><span class="stepchip" data-step="train">Step 3</span> Train the actor only, then
          <span class="stepchip in" data-step="test">Step 5</span> test it: does an actor trained against the wrong
          critic lose return where the missing commands matter?</p>
        <p class="uses">It uses the <a href="#m-swap">critic swap and the step-by-step loss accounting</a>, scored in
          <span class="km" data-sit="strong_current">strong current</span>, in points.</p>''', lo=ha, hi=hb)
ha, hb = section("harm-checks")
rep('''        <h3>The best-case caution check: could caution placed exactly where the critic is wrong help?</h3>
        <p class="uses">Uses the caution placements on the frozen critic of a plain TD3+BC run, at the chosen ''',
    '''        <h3 id="hc-caution">The best-case caution check: could caution placed exactly where the critic is wrong help?</h3>
        <p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> BCA's lever, aimed perfectly and with
          no BCA: if it cannot help here, no result of BCA could show that it helps.</p>
        <p class="uses">It uses the <a href="#m-placements">caution placements</a> on the frozen critic of a plain TD3+BC run, at the chosen ''',
    lo=ha, hi=hb)
ha, hb = section("harm-checks")
rep('''        <h3>The seed-count check: how many repeat runs does a fair test need?</h3>
        <p class="uses">Uses plain TD3+BC runs on the log after the removal in which critic and actor learn together,''',
    '''        <h3 id="hc-seeds">The seed-count check: how many repeat runs does a fair test need?</h3>
        <p class="stepnote"><span class="stepchip" data-step="test">Step 5</span> How many repeat runs the test of BCA
          needs, so that a real gain of 3 points can be told apart from chance.</p>
        <p class="uses">It uses <a href="#m-td3bc">plain TD3+BC</a> runs on the log after the removal in which critic and actor learn together,''',
    lo=ha, hi=hb)

# 2c. the seed-count legend, in groups, with a worked example
ha, hb = section("harm-checks")
a, b = between('<div class="symkey"><p class="symkey-h">What the symbols mean</p><dl><div><dt><i>N</i></dt>', "</dl></div>",
               ha, hb, include_end=True)
assert s[a:b].count("<dt>") == 7
s = s[:a] + new_seed_legend + s[b:]
N["n"] += 1

# 2d. the stop rule
ha, hb = section("harm-checks")
a, b = between('<div class="box stop">\n      <h3>The stop rule</h3>', "\n    </div>\n  </section>", ha, hb + 1)
assert s[a:b].count("<li>") == 4
s = s[:a] + new_stop + s[b + len("\n    </div>"):]
N["n"] += 1

# ------------------------------------------------------------------------------------------------- 3. Style
css = open(CSS, encoding="utf-8").read().rstrip("\n")
anchor = "@media (forced-colors: active) { .key.ring, .key.dots { forced-color-adjust: none; } }\n</style>\n"
rep(anchor, anchor + "<style>\n" + css + "\n</style>\n")

# ------------------------------------------------------------------------------------------------- 4. Column descriptions
m = re.search(r"const COLS = (\{.*?\});\n", s, flags=re.S)
COLS_SRC = m.group(1)
COLS = json.loads(COLS_SRC)
assert json.dumps(COLS, ensure_ascii=False, separators=(",", ":")) == COLS_SRC, "COLS does not round-trip"
meth = COLS["methods"]
assert list(meth[0]["cols"]) == ["Setting", "Value", "What it means"], list(meth[0]["cols"])
TBC = "twin delayed deep deterministic policy gradient plus behaviour cloning (TD3+BC)"
MAP_ENTRY = {"cols": {
    "Method": "Each row is one method used by the checks and tests, linked to its explanation below.",
    "Pipeline step": "This is the step of the offline reinforcement-learning pipeline that the method stands for in "
                     f"ThrusterChain: collect the data, remove commands, train the learner ({TBC}), build the band of "
                     "Bayesian Conformal Aggregation (BCA), or test in the shifted situations.",
    "Used by": "This says which check or test uses the method."}}
SET_ENTRY = {"group": "What the settings control", "cols": {
    "Setting": f"Each row is one setting of the learner, {TBC}; the heading rows group the settings by what they control.",
    "Value": "This is the setting's value in the runs of the main test and the weighting test.",
    "What it does": "This says in plain words what the setting does during learning.",
    "Why it matters here": "This says why the setting matters for ThrusterChain's test of the missing commands."}}
COLS["methods"] = [MAP_ENTRY, SET_ENTRY] + meth[1:]
hc = COLS["harm-checks"]
assert len(hc) == 1 and "Spread σ (standard deviation, points)" in hc[0]["cols"]
RULES_ENTRY = {"cols": {
    "Rule": "This is the rule's letter, a to f, as in the pre-registration, the plan written down before any learner ran.",
    "The question": "This is the question the rule answers, in plain words. Each rule rules out one other explanation of "
                    "the critic's error.",
    "Pass mark": "This is what the rule requires. An error ratio is a critic's typical error divided by that of the "
                 "critic trained on the full log, so 2× means twice as wrong.",
    "Result": "This says whether the rule passed, with the pooled value over seeds 0–4, its 95% interval and the range "
              "of the five seeds."}}
STOP_ENTRY = {"cols": {
    "If": "This is the failure the rule watches for, fixed in the pre-registration before any result existed.",
    "Then": "This is what the test does when that failure happens.",
    "Why": "This explains why that failure ends the test, or allows the one fixed rerun."}}
COLS["harm-checks"] = [RULES_ENTRY] + hc + [STOP_ENTRY]
for sec, lst in COLS.items():
    for e in lst:
        for k, v in e["cols"].items():
            d = v[1] if isinstance(v, list) else v
            assert d.endswith(".") and d[0].isupper(), (sec, k)
NEW_SRC = json.dumps(COLS, ensure_ascii=False, separators=(",", ":"))
assert s.count(COLS_SRC) == 1
s = s.replace(COLS_SRC, NEW_SRC, 1)
N["n"] += 1

# ------------------------------------------------------------------------------------------------- 5. Checks
mjx.check(s)
assert "<!--MISSING" not in s and "⟦" not in s and "⟧" not in s
for blk in QV_BLOCKS:
    assert s.count(blk) == 1, "a critic-section block changed"
for m in re.finditer(r'<div class="symkey"[^>]*>', s):
    assert s[m.end():m.end() + 60].lstrip().startswith('<p class="symkey-h">'), s[m.end():m.end() + 80]


class Balance(HTMLParser):
    VOID = {"br", "img", "input", "meta", "link", "hr", "source", "wbr", "col", "area", "base", "embed", "param", "track",
            "path", "use", "rect", "line", "circle", "polyline", "polygon", "stop", "ellipse"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.bad = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.bad.append((tag, self.stack[-3:]))
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


for sid in ("methods", "harm-checks"):
    a, b = section(sid)
    p = Balance()
    p.feed(s[a:b])
    assert not p.bad and not p.stack, (sid, p.bad[:5], p.stack[-5:])
ids = re.findall(r'\sid="([^"]+)"', s)
dup = {i for i in ids if ids.count(i) > 1}
assert not dup, dup
for href in set(re.findall(r'href="#([^"]+)"', s[section("methods")[0]:section("harm-checks")[1]])):
    assert f'id="{href}"' in s, href
a, b = section("methods")
sec = s[a:b]
assert sec.count('<div class="symkey">') == 8, sec.count('<div class="symkey">')
assert sec.count('class="math-display"') == 9, sec.count('class="math-display"')
assert sec.count("<table") == 5
a, b = section("harm-checks")
assert s[a:b].count("<table") == 3 and s[a:b].count('class="stepnote"') == 4

MATH_AFTER = s.count("<mjx-container")
print(f"replacements: {N['n']}; formulas: {MATH_BEFORE} -> {MATH_AFTER}; containers placed by this edit: {counter['n']};"
      f" new formulas rendered: {len(NEEDED)}; new tooltips: {len(NEW_TIPS)}")
os.makedirs(os.path.dirname(TRIAL), exist_ok=True)
open(TRIAL, "w", encoding="utf-8", newline="").write(s)
json.dump({"new_tips": NEW_TIPS, "cols": {"methods": [MAP_ENTRY, SET_ENTRY], "harm-checks": [RULES_ENTRY, STOP_ENTRY]}},
          open(os.path.join(PAGE_DIR, "tmp", "p4", "edit3", "new_texts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("trial copy:", TRIAL, hashlib.md5(s.encode("utf-8")).hexdigest())
if not CHECK:
    open(PAGE, "w", encoding="utf-8", newline="").write(s)
    print("WRITTEN", PAGE, hashlib.md5(open(PAGE, "rb").read()).hexdigest())
