#!/usr/bin/env python3
"""Page pass 4, edit 4 of 4: Results, the weighting test's exploratory results and the critic section's column
descriptions (draft B, facts checked), on top of edits 1 to 3.

- Results (#results, up to the timeline): each check as what it asks -> how it was measured -> what was expected ->
  what happened -> what it means, every term defined where it first matters, a pipeline-step note on each check; the
  best-case caution check gets "what these results mean, and why they matter" and the four situations as difficulty
  levels (ordered by the harm the missing commands cause, with the standard terms, and why that is not raw
  difficulty); the "Already measured" table rebuilt (plain questions, values next to their words, a passed column, the
  four stall-matters numbers labelled). The tables kept from the page get situation colours where a row or column
  names a situation; the stopped/below-3 tags go neutral (red marks the stall only).
- After the stop (#beyond-the-data): a new "In short" that names the main test's removal; situation colours on the
  Stage 1 tables' situation columns. The follow-up's claims: the out-of-date "BCA runs now under way" sentence.
- The weighting test: status, claims chip and progress strip now say the exploratory runs finished; "What the results
  mean" in three parts with the plain conclusion; exploratory results A (plain TD3+BC's two-critic band against BCA,
  unweighted and weighted), B (BCA's training runs, both uses of the band, against plain TD3+BC and the actor with no
  extra caution) and C (what these data hold past 80%), replacing the "running" placeholder; the gate item and the two
  "running" result cells of the pass-marks table; situation colours on the test-setting rows.
- Column descriptions (COLS): the rebuilt and new tables of Results and the weighting test, and the critic section's
  four tables (26 columns and 3 group headings).

Every replacement is exact and counted. Math: every formula is either copied from the page (same TeX label, same
display flag) or rendered by mathbuild/render.js from the job file this script writes with --jobs; every container
placed by this edit gets fresh glyph ids (prefix MJXF).

Run from the page directory (WSL):
    python3 scripts/p4_e4_apply.py --jobs     (write mathbuild/jobs/p4e4.json with the formulas the page lacks)
    python3 scripts/p4_e4_apply.py --check    (build in memory, write tmp/p4/edit4/applied.html only)
    python3 scripts/p4_e4_apply.py            (apply to thrusterchain.html)
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
BEFORE = os.path.join(PAGE_DIR, "tmp", "p4", "edit4.before.html")
TRIAL = os.path.join(PAGE_DIR, "tmp", "p4", "edit4", "applied.html")
SRC = os.path.join(HERE, "p4_e4_src.html")
CSS = os.path.join(HERE, "p4_e4.css")
E3_CSS = os.path.join(HERE, "p4_e3.css")
JOB = os.path.join(mjx.MATHBUILD, "jobs", "p4e4.json")
OUT = os.path.join(mjx.MATHBUILD, "out", "p4e4.json")
E3_MD5 = "f376b41c008320ad4881a0e68beef8e6"
JOBS = "--jobs" in sys.argv
CHECK = "--check" in sys.argv
PREFIX = "MJXF"

s = open(PAGE, encoding="utf-8").read()
ORIG = s
assert hashlib.md5(open(BEFORE, "rb").read()).hexdigest() == E3_MD5, "backup changed"
assert hashlib.md5(s.encode("utf-8")).hexdigest() == E3_MD5, "the page is not the edit-3 page this edit was written for"
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


def splice(a, b, new):
    global s
    s = s[:a] + new + s[b:]
    N["n"] += 1


def esc(text):
    return html.escape(text, quote=True)


# ------------------------------------------------------------------------------------------------- tooltips
def page_tip(title, needle=None):
    """(title attribute, data-tip attribute) for a tooltip title used on the page, verbatim. A title may carry several
    texts on the page; then `needle` picks the one that contains it."""
    for t in (esc(title), title.replace('"', "&quot;")):
        texts = []
        for m in re.finditer(r'data-tip-title="%s" data-tip="([^"]*)"' % re.escape(t), ORIG):
            if m.group(1) not in texts:
                texts.append(m.group(1))
        if texts:
            if needle is not None:
                texts = [x for x in texts if needle in html.unescape(x)]
                assert len(texts) == 1, (title, needle, len(texts))
            assert len(texts) == 1, f"tooltip {title!r} has {len(texts)} texts on the page; give a needle"
            return t, texts[0]
    raise AssertionError(f"no tooltip titled {title!r} on the page")


NEW_TIPS = {}


def new_tip(title, text):
    assert text.endswith(".") and text[0].isupper(), text
    assert f'data-tip-title="{esc(title)}"' not in ORIG, f"title {title!r} is already on the page"
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
WANT = {"results_main", "beyond_short", "iw_status", "iw_claim_chips", "iw_pipe_tail", "iw_outcome", "iw_explore",
        "iw_gate", "iw_cell_bca", "iw_cell_le"}
assert set(PARTS) == WANT, set(PARTS) ^ WANT
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
        if kind == "Tx":
            t, d = page_tip(f[1], f[2])
            return f'<span class="term" data-tip-title="{t}" data-tip="{d}">{f[3]}</span>'
        if kind == "TN":
            t, d = new_tip(f[1], f[2])
            return f'<span class="term" data-tip-title="{t}" data-tip="{d}">{f[3]}</span>'
        if kind == "KEEP":
            return KEEP[f[1]]
        raise AssertionError(m.group(0)[:80])
    out = re.sub(r"⟦(.*?)⟧", one, text, flags=re.S)
    assert "⟦" not in out and "⟧" not in out
    return out


def keep_edit(name, old, new, count=1):
    assert KEEP[name].count(old) == count, (name, old[:80], KEEP[name].count(old))
    KEEP[name] = KEEP[name].replace(old, new)


# ------------------------------------------------------------------------------------------------- 1. Results: kept blocks
ra, rb = section("results")
res_a = s.index("<h2>Results</h2>", ra)
res_b = s.index('<h3 id="results-timeline">', ra)
assert ra < res_a < res_b < rb
TABLE_END = "</table></div>"


def grab(name, start, end=TABLE_END, include_end=True):
    a, b = between(start, end, res_a, res_b, include_end=include_end)
    KEEP[name] = s[a:b]


grab("critic_table", '<div class="tbl"><table class="stack">\n      <caption><span class="cap">The critic-error check:')
assert KEEP["critic_table"].count("<tr>") == 8 and KEEP["critic_table"].count('<span class="tag ok">pass</span>') == 7
grab("cost_table", '<div class="tbl"><table class="stack">\n      <caption><span class="cap">The return-cost check:')
keep_edit("cost_table", "The return-cost check: points in strong current, by repeat run.",
          'The return-cost check: points in <span class="km" data-sit="strong_current">strong current</span>, by repeat run.')
grab("variants", "<p><b>The variants, and why each is here.</b>")
assert KEEP["variants"].count("<tr>") == 9
grab("scores", "<p><b>How to read the scores table.</b>")
keep_edit("scores", '<th class="n">Recorded conditions</th><th class="n">Strong current</th><th class="n">Big waves</th><th class="n">Big starting pushes</th>',
          '<th class="n" data-sit="recorded">Recorded conditions</th><th class="n" data-sit="strong_current">Strong current</th>'
          '<th class="n" data-sit="big_waves">Big waves</th><th class="n" data-sit="big_kicks">Big starting pushes</th>')
grab("diffs", "<p><b>The pass rule, and the result.</b>")
for name, key in [("Recorded conditions (reported only", "recorded"), ("Strong current</td>", "strong_current"),
                  ("Big waves</td>", "big_waves"), ("Big starting pushes</td>", "big_kicks")]:
    keep_edit("diffs", f"<tr><td>{name}", f'<tr data-sit="{key}"><td>{name}')
keep_edit("diffs", '<span class="tag warn">below 3</span>', '<span class="tag">below 3</span>', count=3)
# the forecast: kept to its estimates and the most likely stop; its last paragraph is rewritten in the source
grab("forecast", "<p><b>The forecast, made before any run.</b>", "<p><b>How the forecast compares with what happened.</b>",
     include_end=False)
KEEP["forecast"] = KEEP["forecast"].rstrip()
keep_edit("forecast", '(<a href="#methods">Methods</a>, under "How far the actor moves"; a forecast, never a',
          '(<a href="#m-how-far">Methods, How far the actor moves</a>; a forecast, never a')
assert KEEP["forecast"].count("<li>") == 5 and KEEP["forecast"].endswith("</p>")

new_results = expand(PARTS["results_main"])
new_short = expand(PARTS["beyond_short"])
new_iw = {k: expand(PARTS[k]) for k in ("iw_status", "iw_claim_chips", "iw_pipe_tail", "iw_outcome", "iw_explore",
                                         "iw_gate", "iw_cell_bca", "iw_cell_le")}

if JOBS:
    os.makedirs(os.path.dirname(JOB), exist_ok=True)
    json.dump(list(NEEDED.values()), open(JOB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(NEEDED)} formulas to render -> {JOB}")
    for k in NEEDED:
        print("  ", k)
    sys.exit(0)
assert not [k for k in NEEDED if k not in RENDERED], [k for k in NEEDED if k not in RENDERED]

counter = {"n": 0}


def retag_all(text):
    def one(m):
        c = m.group(0)
        counter["n"] += 1
        found = {(x.group(2), x.group(3)) for x in mjx.IDNUM.finditer(c)}
        assert len(found) <= 1, found
        return mjx.IDNUM.sub(lambda x: f"{x.group(1)}{PREFIX}-{counter['n']}-", c)
    return CONT.sub(one, text)


new_results = retag_all(new_results)
new_short = retag_all(new_short)
new_iw = {k: retag_all(v) for k, v in new_iw.items()}

# 1a. splice Results' main line (h2 up to the timeline)
splice(res_a, res_b, new_results.rstrip() + "\n\n    ")

# 1b. After the stop: the new "In short"
ra, rb = section("results")
ba = s.index('<div class="subsec" id="beyond-the-data">', ra)
a, b = between("<h4>In short: a summary of the findings argued below</h4>", "</ul>", ba, rb, include_end=True)
assert s[a:b].count("<li>") == 5
splice(a, b, new_short.strip())

# 1c. the Stage 1 tables: situation columns carry their colour
ra, rb = section("results")
ba = s.index('<div class="subsec" id="beyond-the-data">', ra)
for name, key in [("Recorded conditions", "recorded"), ("Strong current", "strong_current"), ("Big waves", "big_waves"),
                  ("Big starting pushes", "big_kicks")]:
    rep(f'<th class="n" scope="col">{name}</th>', f'<th class="n" scope="col" data-sit="{key}">{name}</th>', count=4, lo=ba, hi=rb)
    ra, rb = section("results")
rep('<th class="n" scope="col">Strong current: share of states past 80%</th>',
    '<th class="n" scope="col" data-sit="strong_current">Strong current: share of states past 80%</th>', lo=ba, hi=rb)

# 1d. the follow-up's first claim: the BCA runs it mentions have finished
ra, rb = section("results")
rep("""No Stage 2 run of BCA has started; the BCA runs
          now under way belong to the weighting test.</p>""",
    """No Stage 2 run of BCA has started; the BCA runs that
          have run belong to the weighting test's exploratory part (<a href="#iw-arms">its results</a>).</p>""", lo=ra, hi=rb)

# ------------------------------------------------------------------------------------------------- 2. The weighting test
wa, wb = section("weighting-test")
a, b = between('<p class="status"><span class="tag ok">pre-registered results: complete</span>', "</p>", wa, wb, include_end=True)
assert "running" in s[a:b]
splice(a, b, new_iw["iw_status"].strip())
wa, wb = section("weighting-test")
a, b = between('<p><span class="iw-chip pass">coverage restored on all five</span>', "</p>", wa, wb, include_end=True)
assert s[a:b].count("iw-chip") == 2
splice(a, b, new_iw["iw_claim_chips"].strip())
wa, wb = section("weighting-test")
a, b = between('<div class="iw-step now" role="listitem"><span class="t">BCA training, exploratory</span>',
               '<span class="s">shown here when finished</span></div>', wa, wb, include_end=True)
assert s[a:b].count('class="iw-step') == 2
splice(a, b, new_iw["iw_pipe_tail"].strip())
wa, wb = section("weighting-test")
a, b = between("<p><b>The outcome.</b> On all five data qualities", "</p>", wa, wb, include_end=True)
splice(a, b, new_iw["iw_outcome"].strip())
wa, wb = section("weighting-test")
a, b = between('<h4 id="iw-explore">', '<h3 id="iw-shift">', wa, wb)
old = s[a:b]
assert old.count('<span class="iw-chip">running</span>') == 10 and old.count("<table") == 1
splice(a, b, new_iw["iw_explore"].strip() + "\n\n\n    ")
wa, wb = section("weighting-test")
a, b = between('<li class="iw-gate">', "</li>", wa, wb, include_end=True)
assert "going ahead" in s[a:b]
splice(a, b, new_iw["iw_gate"].strip())
OLD_CELL = ('<td><span class="iw-chip">not run under the pre-registered rules:</span> No quality passed both checks before '
            'it. Exploratory run: <span class="iw-chip">running</span></td>')
for row, part in (("<tr><td>Weighted BCA improves</td>", "iw_cell_bca"), ("<tr><td>Weighted lower edge improves</td>", "iw_cell_le")):
    wa, wb = section("weighting-test")
    r0 = s.index(row, wa)
    r1 = s.index("</tr>", r0)
    assert r1 < wb and s[r0:r1].count(OLD_CELL) == 1
    a = s.index(OLD_CELL, r0)
    splice(a, a + len(OLD_CELL), new_iw[part].strip())
wa, wb = section("weighting-test")
assert '<span class="iw-chip">running</span>' not in s[wa:wb] and "exploratory, running" not in s[wa:wb]
# the test settings wear the colours their tests wear in the density chart
rep('<tr><td>No shift</td><td class="n">−2</td>', '<tr data-sit="recorded"><td>No shift</td><td class="n">−2</td>', lo=wa, hi=wb)
wa, wb = section("weighting-test")
rep("<tr><td>Mild (uniform)</td>", '<tr data-iw="mild"><td>Mild (uniform)</td>', lo=wa, hi=wb)
wa, wb = section("weighting-test")
rep('<tr><td>Strong</td><td class="n">+2</td>', '<tr data-sit="strong_current"><td>Strong</td><td class="n">+2</td>', lo=wa, hi=wb)
wa, wb = section("weighting-test")
rep('<caption><span class="cap">The five data qualities, strong test, points',
    '<caption><span class="cap">The five data qualities, <span class="km" data-sit="strong_current">strong test</span>, points',
    lo=wa, hi=wb)

# ------------------------------------------------------------------------------------------------- 3. Style
css = open(CSS, encoding="utf-8").read().rstrip("\n")
e3 = open(E3_CSS, encoding="utf-8").read().rstrip("\n")
anchor = e3 + "\n</style>\n"
rep(anchor, anchor + "<style>\n" + css + "\n</style>\n")

# ------------------------------------------------------------------------------------------------- 4. Column descriptions
m = re.search(r"const COLS = (\{.*?\});\n", s, flags=re.S)
COLS_SRC = m.group(1)
COLS = json.loads(COLS_SRC)
assert json.dumps(COLS, ensure_ascii=False, separators=(",", ":")) == COLS_SRC, "COLS does not round-trip"
BCA = "Bayesian Conformal Aggregation (BCA)"
TBC = "twin delayed deep deterministic policy gradient plus behaviour cloning (TD3+BC)"

res = COLS["results"]
assert list(res[5]["cols"]) == ["Check: what it guards against", "Rule (from the pre-registration)", "Seed 900",
                                "Seeds 900–904, lowest–highest"], list(res[5]["cols"])
WORLD_ENTRY = {"cols": {
    "Check": "Each row is one of the nine world checks, run before any learning on design seeds 900 to 904, asked as "
             "the question it answers.",
    "Pass mark": "This is the pass mark fixed in the pre-registration, the plan written down before any learner ran.",
    "Seed 900": "This is the value on design seed 900, the recording this page shows. A seed fixes every random draw "
                "of a recording, and the design seeds are recordings made only to design and check the world.",
    "All five": "This gives the lowest and the highest value over the five design seeds, 900 to 904.",
    "Passed": "This says whether the check passed, and whether it passed on every seed."}}
LADDER_ENTRY = {"cols": {
    "Level and situation": "Each row is one test situation of the main test, as a difficulty level: the levels are "
                           "ordered by the harm the missing full-throttle commands cause, not by raw difficulty.",
    "Standard term": "This is the usual reinforcement-learning name for how the situation differs from the training "
                     "data.",
    "What differs from the recording": "This says what changes from the conditions of the recorded training data.",
    "States outside the recorded conditions": "This is the share of the visited states that lie outside the recorded "
                                              "conditions: a current above 0.7, a wave amplitude above 0.4, or floats "
                                              "farther from rest than 99% of the recorded training states.",
    "States where the stall-aware controller needs a removed command": "This is the share of the states the "
        "stall-aware controller visits at which its own command lies inside the main test's removed ball, so the "
        "training data no longer show that command (the shift table, design seed 900).",
    "Harm from the missing commands, points": "This is the return-cost check's loss at critic weight 5: the actor "
        "trained against the critic of the full log minus the actor trained against the critic after the removal, in "
        "points, the mean over seeds 0 to 4.",
    "Actor of the full-log critic, points": "This is the score of the actor trained against the critic of the full "
        "log, which shows how hard the situation is for a learner when no command is missing; on this scale 0 is the "
        "recording controller and 100 the stall-aware controller."}}
COLS["results"] = res[:5] + [LADDER_ENTRY, WORLD_ENTRY] + res[6:]

iw = COLS["weighting-test"]
assert "Capped pull: exact weights (each training row at its own threshold) minus unweighted" in iw[1]["cols"]
PLAIN_ENTRY = {"cols": {
    "Quality": "Each row is one data quality of the weighting test, from random to expert.",
    "No shift": "These three columns are measured on held-out steps of the training distribution.",
    "Strong test, other steps": "These three columns are measured on the strong test's steps whose command is not the "
                                "removed 80% hold.",
    "Strong test, steps that use the removed 80% hold": "These three columns are measured on the strong test's steps "
                                                        "whose command is the removed 80% hold.",
    "No shift: plain": f"This is the coverage of plain {TBC}'s band, the lower critic's value plus or minus the two "
                       "critics' disagreement, on held-out steps of the training distribution.",
    "No shift: BCA": f"This is the coverage of the unweighted band of {BCA} on the same steps.",
    "No shift: weighted": "This is the coverage of the band of Bayesian Conformal Aggregation with exact weights on the "
                          "same steps; with no shift every weight is 1, so it equals the unweighted band.",
    "Strong test, other steps: plain": "This is the coverage of plain TD3+BC's two-critic band on the strong test's "
                                       "steps that do not use the removed 80% hold.",
    "Strong test, other steps: BCA": f"This is the coverage of the unweighted band of {BCA} on those steps.",
    "Strong test, other steps: weighted": "This is the coverage of the band of Bayesian Conformal Aggregation with exact "
                                          "weights, each step at its own threshold, on those steps.",
    "Strong test, removed hold: plain": "This is the coverage of plain TD3+BC's two-critic band on the strong test's "
                                        "steps that use the removed 80% hold.",
    "Strong test, removed hold: BCA": f"This is the coverage of the unweighted band of {BCA} on those steps.",
    "Strong test, removed hold: weighted": "The weighted band refuses every one of these steps, because the removed hold "
                                           "has no chance in training, so its weight cannot be computed."}}
RET_ENTRY = {"cols": {
    "Quality": "Each row is one data quality of the weighting test, from random to expert.",
    "Capped pull: TD3+BC retrained as a whole": "These three columns retrain the whole learner, critic and actor, for "
                                                "100,000 updates.",
    "Lower edge: actor only, against the frozen critic": "These four columns train only the actor, against the same "
                                                         "frozen critic.",
    "Plain TD3+BC": f"This is plain {TBC} trained on the data with the 80% hold removed, the reference for the capped "
                    "pull, in points on the strong test.",
    "Capped pull, unweighted": f"This is TD3+BC retrained with the dose of {BCA} on its pull toward the recorded "
                               "commands, the band unweighted.",
    "Capped pull, weighted": "This is the same run with the band weighted exactly for the strong test, each training "
                             "row at its own threshold.",
    "Actor only, no extra caution": "This is the actor trained alone against the frozen critic with no extra caution, "
                                    "the reference for the lower edge.",
    "Lower edge, unweighted": f"This is the same actor-only run with the lower-edge penalty of {BCA} on the value of the "
                              "actor's own command, the band unweighted.",
    "Lower edge, weighted": "This is the same run with the band weighted exactly for the strong test.",
    "Perfect lower edge, for scale": "This is the same actor-only run with a perfect penalty, built from the critic's "
                                     "exact one-step error; no method that learns from data has it."}}
COLS["weighting-test"] = [iw[0], PLAIN_ENTRY, RET_ENTRY] + iw[2:]

assert "critic-in-action" not in COLS
QV_ENTRIES = [
    {"cols": {
        "Thruster 1, upstream": "Each row is one of the 41 throttles that thruster 1 is swept through in the state on "
                                "screen, from 0% to 100% upstream in steps of 2.5%, with thrusters 2 to 4 kept where the "
                                "actor put them. One more row, highlighted, holds the actor's own command.",
        "Critic, Q(s, a)": "This is the critic's value of the row's command: its estimate of the discounted return of "
                           "giving that command in this state and letting the actor drive afterwards, the smaller of its "
                           "two estimates. Every reward is a cost, so every value is negative and higher is better.",
        "True value": "This is the true value of the row's command, computed exactly by simulation: one step with the "
                      "command, then the actor drives for 600 steps, with each later reward discounted by 0.98 per step.",
        "Critic minus true value": "This is the critic's value minus the true value. A positive number means the critic "
                                   "rates the command better than it is, which is the kind of error that draws the actor "
                                   "toward the command.",
        "Note": "This marks the throttle the critic rates highest (â), the swept throttle with the highest true value, "
                "the 80% stall limit, the throttles inside this section's removed region, and the row of the actor's own "
                "command."}},
    {"cols": {
        "State": "Each row is one state of the episode, numbered 0 to 99. The highlighted row is the state on screen.",
        "Water's push c + w₁": "This is the water's push in the state: the current c plus the push w₁ that the wave adds "
                               "now. A positive push points downstream.",
        "Thruster 1 (actor), upstream": "This is the throttle the actor gives thruster 1 in the state, upstream unless "
                                        "marked down. Past 80% the propeller stalls and pushes less.",
        "Reward": "This is the state's reward: minus the sum of the squares of the floats' ten numbers, minus 0.3 times "
                  "the sum of the squared throttles. Every reward is a cost, so closer to 0 is better.",
        "Critic, Q(s, π(s))": "This is the critic's value of the command the actor gave in the state, the smaller of its "
                              "two estimates.",
        "True value": "This is the true value of the command the actor gave, computed exactly by simulation with the "
                      "actor driving afterwards.",
        "Critic minus true value": "This is the critic's value minus the true value at the actor's command. A positive "
                                   "number means the critic rates the command better than it is.",
        "â": "This is the swept throttle of thruster 1 that the critic rates highest in the state. The actor, which is "
             "trained to raise the critic's value, is drawn toward it.",
        "True best of the sweep": "This is the swept throttle of thruster 1 with the highest true value in the state, "
                                  "with the other three thrusters kept where the actor put them."}},
    {"cols": {
        "Dataset quality": "Each row is one of the five dataset qualities of the weighting test, from random to expert.",
        "How it was recorded": "This says how the quality's commands were recorded, and what share of its training "
                               "steps carry a random command instead of the recording controller's.",
        "Training steps kept, all data / hold removed": "This is the number of recorded training steps each critic "
                                                        "trained on: all of them, and those left after the densest-15% "
                                                        "removal, which in the weighting test takes the 80% hold.",
        "Thruster 1 held at exactly 80%, all data / hold removed": "This is the share of those training steps in which "
                                                                   "thruster 1 is held at exactly 80% upstream, the best "
                                                                   "move in hard water, in each critic's data.",
        "Thruster 1 past 80%, all data / hold removed": "This is the share of those training steps in which thruster 1 is "
                                                        "more than 80% upstream, in each critic's data. Only random "
                                                        "commands go there, so they are the only evidence of the stall "
                                                        "the critic ever sees."}},
    {"cols": {
        "Dataset quality": "Each row is one dataset quality. The highlighted row is the quality chosen above.",
        "Critic rates 100% above 80%": "These two columns give the share of the states at which the critic rates "
                                       "thruster 1 at 100% upstream above 80%, with the other three thrusters as the "
                                       "actor set them.",
        "Critic rates 100% above 80%, all the data": "This is that share for the critic trained on all the data, over the "
                                                     "1,000 states its own actor visits from the first ten starts of the "
                                                     "situation chosen above.",
        "Critic rates 100% above 80%, 80% hold removed": "This is that share for the critic trained with the 80% hold "
                                                         "removed, over the states its own actor visits.",
        "Critic rates 100% higher, true value lower": "These two columns give the share of the states at which the critic "
                                                      "rates 100% above 80% while the true value of 100% is below that "
                                                      "of 80%: the critic prefers the stall, and the truth does not.",
        "Critic rates 100% higher, true value lower, all the data": "This is that share for the critic trained on all "
                                                                    "the data.",
        "Critic rates 100% higher, true value lower, 80% hold removed": "This is that share for the critic trained with "
                                                                        "the 80% hold removed.",
        "Actor gives thruster 1 more than 80%": "These two columns give the share of the states at which the actor itself "
                                                "gives thruster 1 more than 80% upstream, into the stall.",
        "Actor gives thruster 1 more than 80%, all the data": "This is that share for the actor trained with the critic "
                                                              "on all the data.",
        "Actor gives thruster 1 more than 80%, 80% hold removed": "This is that share for the actor trained with the 80% "
                                                                  "hold removed."}},
]
# the keys must be the critic section's own column names (its data-label values) and group headings
qa, qb = section("critic-in-action")
labels = re.findall(r'<th[^>]*data-label="([^"]*)"', s[qa:qb])
heads = [html.unescape(x) for x in labels]
assert len(heads) == 26, len(heads)
need = [heads[0:5], heads[5:14], heads[14:19], heads[19:26]]
for e, cols in zip(QV_ENTRIES, need):
    assert all(c in e["cols"] for c in cols), [c for c in cols if c not in e["cols"]]
grp = re.findall(r'<th colspan="2"><span data-tip-title="([^"]*)"', s[qa:qb])
assert [html.unescape(g) for g in grp] == ["Critic rates 100% above 80%", "Critic rates 100% higher, true value lower",
                                           "Actor gives thruster 1 more than 80%"], grp
COLS["critic-in-action"] = QV_ENTRIES
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


for sid in ("results", "weighting-test"):
    a, b = section(sid)
    p = Balance()
    p.feed(s[a:b])
    assert not p.bad and not p.stack, (sid, p.bad[:5], p.stack[-5:])
ids = re.findall(r'\sid="([^"]+)"', s)
dup = {i for i in ids if ids.count(i) > 1}
assert not dup, dup
for sid in ("results", "weighting-test"):
    a, b = section(sid)
    for href in set(re.findall(r'href="#([^"]+)"', s[a:b])):
        assert f'id="{href}"' in s, href
for old_id in ("results-question", "results-short", "results-critic", "results-cost", "results-caution", "results-stop",
               "results-forecast", "results-world", "results-timeline", "beyond-the-data", "ats-stage1", "iw-explore",
               "iw-results", "iw-shift", "iw-hold"):
    assert s.count(f'id="{old_id}"') == 1, old_id
a, b = section("results")
sec = s[a:b]
assert sec.count('class="stepnote"') == 4 and 'class="tag warn"' not in sec
a0 = sec.index('<div class="subsec" id="beyond-the-data">')
assert sec[:a0].count("<table") == 7, sec[:a0].count("<table")    # critic, cost, variants, scores, diffs, ladder, world
a, b = section("weighting-test")
sec = s[a:b]
assert sec.count('class="iw-step') == 6 and sec.count("<table") == 5
for need_text in ("18:10 UTC", "refuses all", "3.25% to 10.11%", "1/30 to 1/160"):
    assert need_text in sec, need_text

MATH_AFTER = s.count("<mjx-container")
print(f"replacements: {N['n']}; formulas: {MATH_BEFORE} -> {MATH_AFTER}; containers placed by this edit: {counter['n']};"
      f" new formulas rendered: {len(NEEDED)}; new tooltips: {len(NEW_TIPS)}")
os.makedirs(os.path.dirname(TRIAL), exist_ok=True)
open(TRIAL, "w", encoding="utf-8", newline="").write(s)
json.dump({"new_tips": NEW_TIPS, "cols": {"results": [LADDER_ENTRY, WORLD_ENTRY],
                                          "weighting-test": [PLAIN_ENTRY, RET_ENTRY], "critic-in-action": QV_ENTRIES}},
          open(os.path.join(PAGE_DIR, "tmp", "p4", "edit4", "new_texts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("trial copy:", TRIAL, hashlib.md5(s.encode("utf-8")).hexdigest())
if not CHECK:
    open(PAGE, "w", encoding="utf-8", newline="").write(s)
    print("WRITTEN", PAGE, hashlib.md5(open(PAGE, "rb").read()).hexdigest())
