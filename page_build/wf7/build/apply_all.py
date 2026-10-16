"""Page version 27 of the ThrusterChain page: the live page after version 25 (version 25 with its status box removed
at the owner's request) plus this build (the toy worlds' full results, the lesson, the navigation, the colors and the
status statements), made by the four component patches and the integration edits, in order.

Usage:  python -X utf8 apply_all.py BASE.html OUT.html [--version N] [--status-date "D Mon YYYY"] [--status-time HH:MM]
                                    [--preview PREVIEW.html] [--wrapper LIVE.html] [--stages DIR]

BUILD.md next to this script describes the build (inputs, the order of the stages, the status time, how to rebuild and
test); REBASE.md lists what versions 20-25 added and what the live page changed after version 25, and how each conflict
with this build was merged; FIX_LOG.md records the fix rounds of the version-19 build. The earlier builds stay
reproducible: the version-19 build (page version 20) from component_changes/v19_based/, the version-23 build (page
version 24) from component_changes/v23_based/, the version-25 build (page version 26) from component_changes/v25_based/,
and page version 27 before the toy worlds' integration from component_changes/v27_pretoy/ (its sources' SHA-256 in
component_changes/v27_pretoy.sha256; it rebuilds core_v27.html 7e0d6d90... from live6/core_92ba.html).

BASE is the page's core HTML (no publish wrapper): the live page after version 25 (service version 1792156350-92ba,
live6/core_92ba.html) or a newer base built on it. It is read and never written. OUT receives the new version's core
HTML. --version names it (default 27) in the version-history entry, dated --status-date (default 16 Oct 2026). With
--preview, a browser preview is also written: the publish wrapper's head (the first 531 bytes of --wrapper, by default
live6/live_92ba.html), the core, and "\\n</body></html>".

Order (each component patch is run unchanged, as its own script, on the previous stage's output):
  0. wf7/toy/apply_toy.py       the Toy worlds section (section#toy-worlds) replaced as a whole by the full results of the
                                five toy notebooks (wf7/toy/toy_fragment.html, built by wf7/toy/build_toy.py; TOY.md
                                there). It replaces only a section it knows (SHA-256: the live page's own, versions 19
                                to 25 and the live service version 92ba), so a base page whose toy section changed stops
                                the build. It runs first, on the live page's own section, and nothing after it edits the
                                section: the color patch's B14 and B15 and the navigation's five ids in the old section
                                are dropped (the new section states those facts and carries those ids itself).
  1. wf7/color/apply_color.py   the remaining charts' colors (versions 21 and 25's learned actors' cost chart included),
                                color keys in their hover readouts, the status statements outside the removed status
                                box and the other text fixes. --status-date, --status-time and --version are passed on.
  2. wf7/nav/apply_nav.py       the navigation (E10, E9, E11, E7a-c, E1, E2, E4, E5, E6, E8, E12, E13, E14). Its E2
                                replaces the header's walkthrough bar, and with it the button "Full walkthrough mode"
                                (#wt-open), by the start bar, whose link #start-learn goes to the lesson. Every block it
                                replaces must be a version it knows (SHA-256), so a base page that changed one stops it.
  3. wf7/lesson/apply_lesson.py the lesson section#learn (L1), its CSS (L2), its data and script (L3). Its L4 (the
                                header button -> a lesson link) must report "skipped": the navigation's start bar
                                already holds that link.
  4. the integration edits below (I1-I3), each at an anchor that must occur exactly once.

How the overlaps are resolved (each change is applied once; REBASE.md has the merges):
  - The status box: the owner asked for it to be removed and the live page removed it. Nothing of it is added or
    restored: the color patch has no edit on it, the navigation lists no entry for it, and the checks below refuse a
    page with any part of it.
  - The walkthrough button: replaced only by the navigation's start bar (step 2); the lesson's L4 is skipped and this
    script checks that it was.
  - The toy worlds: replaced first, by the toy patch (step 0), and edited by nothing after it (checked: the section of
    OUT is the fragment byte for byte). Its 13 old ids are kept (the live section's 8, and the 5 the navigation's static
    edits gave it before: the new section has them itself); the links into it (the lesson's step 5.5 and its sources to
    #toy-meaning, Contents to #toy-worlds) and its own links resolve. The navigation lists its parts by its own rules
    (Contents and "In this section"). Its four charts have their own color keys in their hover readouts: templates
    beside each chart, one swatch per line, in the section's own tokens (defined on :root and in both dark blocks).
  - The shared tooltip script: the color patch adds the color keys (K9, K10), the navigation's E13 the Tab stops that
    skip hidden sections; version 22's cell readings (data-tt-read) and its placement beside an arm table's cell are
    in other parts of that script and stay as they are. The live page's newer tooltips (the "Strong test mix" button's,
    rewritten after version 25) are not touched.
  - The Choices: the navigation's script (E8) replaces the old sidebar's script and carries its rows, versions 21, 22
    and 25's included ("Who drives" with "Learned actor", the learned actor's rows, version 25's "Learner", "Table",
    the "not in use now" marking).
  - The charts: version 24's axis titles and version 25's learner names stay as they are; the color patch adds the
    color keys beside them (K13b on version 25's learned-actor cost readout).
  - The status statements: each place is stated once, by the patch that owns it: the color patch (Results' opening,
    After the stop, the first claim, the weighting test's notes), the toy patch (its own section: the notebooks' commits,
    and that Stage 2 version 7 has ended and will not run), the lesson's step 7.2 (built from the repository by
    wf7/lesson/build/build_lesson.py) and I2 here. The checks below refuse a page with an out-of-date status statement
    or with status times that differ.

Integration edits:
  I1  section#version-history: the entry "Version N, D Mon YYYY." at the top of its list (newest first); the page must
      have no entry for version N before, and has exactly one entry per version number after.
  I2  After the stop: "Its status was last brought up to date ... at 05:41 UTC" names the date and time of the box
      under it (#ats-status, "Where each line of work stands").
  I3  the page's top layout comment: the old right-hand contents rail is gone.

Checks (the script stops with an error and writes no OUT on any failure): each stage's patch exits 0; every anchor of
I1-I3 occurs exactly once; no CR byte; no duplicate id, counting every element id in the markup, the MathJax glyph ids
of the page (MJX...) and of the lesson (MJXL...) included; every in-page reference of an SVG <use> and every in-page
link (href="#id") resolves; the title is unchanged; the old navigation's strings are gone; no part of the status box;
the header as on the live page; the parts versions 20-25 added are there (as often as in BASE, outside the toy
section, for the code of versions 24 and 25, and for the live page's newer "Strong test mix" tooltip); the toy section
is the toy patch's fragment byte for byte, with every id of the old section and both axes named on each of its charts;
the status statements agree; the preview is under 16 MB. The intermediate stages (s0_toy, s1_color, s2_color_nav,
s3_color_nav_lesson) are kept in --stages (default: a folder "stages" next to OUT) for testing.
"""
import html.parser
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W7 = HERE.parent
SCR = W7.parent
COLOR = W7 / "color" / "apply_color.py"
NAV = W7 / "nav" / "apply_nav.py"
LESSON = W7 / "lesson" / "apply_lesson.py"
TOY = W7 / "toy" / "apply_toy.py"
TOY_FRAGMENT = W7 / "toy" / "toy_fragment.html"
# the ids that the navigation's static edits gave parts of the live page's old toy section (static_edits_v27.py); the
# new section has them itself (nav/rules/static_edits_v27t.py drops those edits)
TOY_NAV_IDS = ("toy-reward-table", "toy-pooled-table", "toy-shelved-width", "toy-shelved-posterior", "toy-more-detail")
WRAPPER = SCR / "live6" / "live_92ba.html"
WRAP_HEAD_BYTES = 531
WRAP_TAIL = b"\n</body></html>"
LIMIT = 16 * 1000 * 1000          # 16 MB, counted in decimal megabytes to stay on the safe side
PAGE_VERSION = "27"

# The time (UTC, 15 Oct 2026) at which the repository's records hold for the status statements: the remote-tracking
# refs of C:/Users/David/dev/BCA as last fetched from GitHub at 22:13 UTC (.git/FETCH_HEAD), read (read only) at 23:16
# UTC; the local branches hold nothing newer. Heads: forked cab5b3e (21:45, .claude/settings.json; the approval of
# Stage 2 version 7 is de4851c, 15:49, its file says approved_at_utc 15:45:42), main 1ec267e (21:36, pull request 5,
# which merged forked; 03726de at 17:18, pull request 3, merged the weighting test's code, specification and
# exploratory results, not its runs), claude/main-bca-connection-x4o46s 7124be0 (18:50, the near-80 test's draft
# pre-registration, nothing trained; a643094 at 18:30, push permissions). No import record of the approval and no run on
# seeds 5-9 (stage2_v7/handoff/HANDOFF.json: every seed 5-9 phase "not started"). The four-learner study's code is not
# in any of these refs (it is built in WSL, which this build does not read); its status is the one the owner's
# request states: plan approved in chat, code tested on made-up numbers only, nothing trained. Page version 20's
# statements held as of 15:53 UTC; every one was re-read for version 24 (REBASE.md).
#
# Page version 26: 16 Oct 2026, 12:39 UTC, the time the remote-tracking refs were last fetched from GitHub
# (.git/FETCH_HEAD, 08:39:13 -04:00, fetched by another session; this build fetched nothing), read (read only,
# GIT_OPTIONAL_LOCKS=0) after 13:23 UTC. (A first reading at 12:01 UTC, of refs fetched at 11:50 UTC, was overtaken by
# the near-80 approval at 11:50:43 UTC.) New since version 24's status time, on the shared branch: the near-80 test's
# final pre-registration (e3e87c3, 01:33 UTC, fingerprint 51e677ab76c8d2f7; 0ebe036 at 01:52 adds copies of its pinned
# files), the merge of main into it (9765564, 02:26), the import record of Stage 2 version 7's approval (02e591b, 02:27,
# import_v7_3dfcb6140e2b8e2c.json, imported_at 02:27:10Z: "Version 7 can now run on the Framework Desktop"), the owner's
# approval of the near-80 plan (f79b89c, approval_near80_51e677ab76c8d2f7.json, approved_at_utc 11:50:43Z, typed at the
# owner's own terminal) and its import record (287380b, imported_at 11:52:58Z), and a merge (0377dd7, 12:38) that also
# brings the owner's local 810ecc6 (a second Stage 2 approval file typed at 03:00 UTC, never imported; the merge kept
# origin's copy, the one the import record pins). origin/claude/bca-findings-walkthrough d2d86fb (11:42, the findings
# notebook's lesson; no status statement). main (1ec267e) and forked (cab5b3e) unchanged. No result of the run on seeds
# 5-9 and no near-80 result in any ref; no runs/shift_iw* in any ref; version 25's cloud rerun of seed 0 (iw_rerun, 15
# Oct 22:32:50 UTC) in no ref. The four-learner study: as the owner's request states it (plan approved in chat on 15
# Oct; code built and under review, tested on made-up numbers only; nothing trained on ThrusterChain data).
#
# Page version 27: 16 Oct 2026, 13:44 UTC. GitHub's branch heads read then (git ls-remote, read only; nothing fetched)
# equal the fetch of 12:39 UTC: nothing in the repository changed (the shared branch 0377dd7, forked cab5b3e, main
# 1ec267e; no near-80 run-files branch). Two facts are in no ref; they are the ones the workflow's task of 16 Oct (about
# 13:38 UTC) states as current: Stage 2 version 7 was ended without running, so seeds 5-9 will not run under it (the
# near-80 runner starts only once version 7 has finished or its ending is recorded on the desktop, near80/
# PREREGISTRATION.md, choice 12), and the near-80 test (plan "version 8, near-80", approved 11:50:43 UTC, committed
# 11:51, f79b89c; the approval recorded 11:52:58 UTC, 287380b) now runs on the owner's Framework Desktop computer. The
# four-learner study: in code review, tested on made-up numbers only, waiting for the owner's approval at their own
# terminal; nothing of it trained on ThrusterChain data. There is no status box (removed at the owner's request).
# The toy section's own statements (the toy patch) were read against the same refs on 16 Oct (git branch --contains,
# read only): the five toy notebooks were committed on 15 Oct in 2c8a0e3 and 6320224, both on forked, on main and on
# the shared branch; the notebook lesson's 2467e29 only on claude/bca-findings-walkthrough.
STATUS_TIME = "13:44"
STATUS_DATE = "16 Oct 2026"


def die(msg):
    sys.stderr.write("apply_all.py: " + msg + "\n")
    sys.exit(1)


def read_text(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def once(s, needle, label):
    n = s.count(needle)
    if n != 1:
        die("edit %s: its anchor %r occurs %d times (it must occur exactly once)" % (label, needle[:100], n))
    return s.index(needle)


def replace_once(s, old, new, label):
    once(s, old, label)
    return s.replace(old, new, 1)


class IdScan(html.parser.HTMLParser):
    """Every id attribute of every element in the markup (script and style contents are not markup), and every
    in-page reference of an SVG <use>."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.use_refs, self.titles, self.links = [], [], [], []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if d.get("id") is not None:
            self.ids.append(d["id"])
        if tag == "a" and (d.get("href") or "").startswith("#") and len(d["href"]) > 1:
            self.links.append(d["href"][1:])
        if tag == "use":
            for k in ("href", "xlink:href"):
                v = d.get(k)
                if v and v.startswith("#"):
                    self.use_refs.append(v[1:])
        if tag == "title":
            self._in_title = True
            self.titles.append("")

    handle_startendtag = handle_starttag

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and self.titles:
            self.titles[-1] += data


def scan(s):
    p = IdScan()
    p.feed(s)
    p.close()
    return p


def doc_title(s):
    """The document's title: the core starts with it (the charts' SVG <title> elements come later)."""
    m = re.match(r"<title>([^<]*)</title>", s)
    if not m:
        die("the page does not start with its <title>")
    return m.group(1)


def run_stage(script, src, dst, extra, label):
    cmd = [sys.executable, "-X", "utf8", str(script), str(src), str(dst)] + extra
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        die("stage %s (%s) failed with exit %d:\n%s" % (label, script.name, r.returncode, out))
    return out.strip()


def no_toy(x):
    "The page without its section#toy-worlds (the toy patch replaces it as a whole; its own checks are separate)."
    i = x.find('<section id="toy-worlds">')
    return x if i < 0 else x[:i] + x[x.index("</section>", i) + len("</section>"):]


def version_entry(v, d):
    return (
        '      <li><b>Version %s, %s.</b> Four of the project owner\'s requests, built on the page as version 26 left it\n'
        '        (version 25 with its status box removed at the owner\'s request), with everything versions 20 to 26 did kept: a\n'
        '        step-by-step way to learn the page, an easier way around it, better colors in the plots, and the toy worlds\'\n'
        '        full results. A new section,\n'
        '        "Learn step by step", replaces Full walkthrough mode: a lesson in 7 chapters and 39 steps, each with its goal,\n'
        '        a plain explanation, the rule as a formula with a key to its symbols, the arithmetic worked out line by line\n'
        '        with real numbers from the project\'s saved runs (some read for the lesson by inference and exact simulation),\n'
        '        a small control to try the rule and a question to check yourself. A new navigation: the page\'s sections\n'
        '        grouped in seven parts in a Contents list; beside each section an "In this section" list in which every part\n'
        '        of the section is a link, the parts versions 20 to 23 added among them (the guide to gauging the observed\n'
        '        numbers, the live panel\'s switch bar and "Who drives", the notes under the arm tables, the learning curves\'\n'
        '        table, the two notes on the weighting test\'s result A); Previous and Next, which step through those parts in\n'
        '        order; one section shown at a time, or the whole page (the choice is remembered); a bar at the bottom of a\n'
        '        phone\'s screen; and the Choices (every option of the live panels as a short button, the rows of versions 21,\n'
        '        22 and 25 included, version 25\'s Learner switch among them, still marked "not in use now" while the other\n'
        '        mode plays) placed under their panels in the "In this section" list, with a Choices button on a phone, which\n'
        '        steps aside rather than cover a panel\'s own switches. The remaining charts\' colors retuned for contrast and\n'
        '        for color-blind readers, and each line\'s or bar\'s color shown in their hover readouts, beside version 24\'s\n'
        '        axis titles, as version 19 did for the learning curves; now also on the learned actors\' cost chart, for each\n'
        '        of version 25\'s learners, whose lines no longer fade. The Toy worlds section rewritten with the full results\n'
        '        of all five toy notebooks (the four experiments and the notebook that gathers them); each was run again from\n'
        '        its committed version, and every number the section quotes from them was checked against those runs. It shows\n'
        '        what each notebook asks and finds at a glance, the shared two-step world and its weights worked out step by\n'
        '        step, BCA\'s tools in toy terms, and for each experiment its idea, what was tested and the result that would\n'
        '        shelve a claim (set it aside), what happened and why, with tables, worked arithmetic and four charts whose\n'
        '        hover readouts show each line\'s color. Of the 18 claims written down before the runs, 14 are alive and 4\n'
        '        shelved. The status statements in Results, After the stop and\n'
        '        the first claim brought up to date at one time: Stage 2\'s plan version 7 was approved by the project owner\n'
        '        and the approval checked and recorded, but it was then ended without running, so seeds 5 to 9 will not run\n'
        '        under it; a test that leaves the commands near 80%% throttle out of the training data, its final plan approved\n'
        '        by the owner, now runs; runs of four learners are planned, with nothing trained yet; the weighting test\'s\n'
        '        code, specification and exploratory results are in the shared repository. The TD3+BC settings table\'s\n'
        '        wording fixed: the slowly averaged\n'
        '        copies move only when the actor updates, that is every second critic update.</li>\n'
        % (v, d))


def integrate(s, t, v, d):
    log = []
    # I1: the version-history entry for this version, at the top of the list (newest first)
    sec = once(s, '<section id="version-history">', "I1 (section)")
    a = '    <ul class="leads">\n      <li><b>Version '
    i = s.find(a, sec)
    if i < 0 or s.find("</section>", sec) < i:
        die("edit I1: the version history's list was not found")
    if re.search(r"<li><b>Versions? [^<]*\b%s\b[^<]*\.</b>" % re.escape(v), s[sec:s.find("</section>", sec)]):
        die("edit I1: the version history already has an entry for version %s" % v)
    j = i + len('    <ul class="leads">\n')
    s = s[:j] + version_entry(v, d) + s[j:]
    log.append("I1 version history: entry for version %s" % v)

    # I2: After the stop: the section's own "last brought up to date" names the time of the status box under it
    s = replace_once(
        s,
        "Its status was last brought up to date from the repository\n"
        "      on 15 Oct 2026 at 05:41 UTC; the box below says where each line of work stands.",
        "Its status was last brought up to date from the repository\n"
        "      on %s at %s UTC; the box below says where each line of work stands." % (d, t),
        "I2")
    log.append("I2 After the stop: status time %s, %s UTC, as in the box under it (#ats-status)" % (d, t))

    # I3: the layout comment at the top of the page's CSS
    s = replace_once(
        s,
        "A reading column with a\n   contents rail on the right; a live channel panel as the figure;",
        "A reading column with the\n   navigation around it (Contents at the left, \"In this section\" at the right, a bar at the "
        "bottom of a narrow screen;\n   its CSS and script come from the navigation's patch); a live channel panel as the "
        "figure;",
        "I3")
    log.append("I3 layout comment")
    return s, log


def main(argv):
    pos, opts = [], {}
    k = 0
    while k < len(argv):
        a = argv[k]
        if a in ("--status-time", "--status-date", "--preview", "--wrapper", "--stages", "--version"):
            if k + 1 >= len(argv):
                die(a + " needs a value")
            opts[a] = argv[k + 1]
            k += 2
            continue
        if a.startswith("--"):
            die("unknown option " + a)
        pos.append(a)
        k += 1
    if len(pos) != 2:
        die("usage: python -X utf8 apply_all.py BASE.html OUT.html [--version N] [--status-date \"D Mon YYYY\"] "
            "[--status-time HH:MM] [--preview PREVIEW.html] [--wrapper LIVE.html] [--stages DIR]")
    base, out = Path(pos[0]), Path(pos[1])
    if base.resolve() == out.resolve():
        die("BASE and OUT must be different files")
    t = opts.get("--status-time", STATUS_TIME)
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", t):
        die("--status-time must be HH:MM (UTC)")
    d = opts.get("--status-date", STATUS_DATE)
    if not re.fullmatch(r"[1-9]\d? (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}", d):
        die("--status-date must read like 16 Oct 2026")
    v = opts.get("--version", PAGE_VERSION)
    if not re.fullmatch(r"[1-9]\d{0,3}", v):
        die("--version must be a page version number")
    stages = Path(opts.get("--stages", out.resolve().parent / "stages"))
    stages.mkdir(parents=True, exist_ok=True)
    preview = Path(opts["--preview"]) if "--preview" in opts else None
    wrapper = Path(opts.get("--wrapper", WRAPPER))

    base_s = read_text(base)
    if "\r" in base_s:
        die("BASE has CR bytes")
    base_scan = scan(base_s)
    base_title = doc_title(base_s)

    s0 = stages / "s0_toy.html"
    s1, s2, s3 = stages / "s1_color.html", stages / "s2_color_nav.html", stages / "s3_color_nav_lesson.html"
    logs = []
    toy_log = run_stage(TOY, base, s0, [], "0 toy worlds")
    logs.append("[0 toy worlds]\n" + toy_log)
    if "replacing the live page's section" not in toy_log and "already has this toy section" not in toy_log:
        die("stage 0: the toy patch was expected to replace the live page's own toy section: %s" % toy_log)
    logs.append("[1 color]\n" + run_stage(COLOR, s0, s1, ["--status-date", d, "--status-time", t, "--version", v], "1 color"))
    logs.append("[2 navigation]\n" + run_stage(NAV, s1, s2, [], "2 navigation"))
    lesson_log = run_stage(LESSON, s2, s3, [], "3 lesson")
    logs.append("[3 lesson]\n" + lesson_log)
    if "L4 skipped" not in lesson_log:
        die("stage 3: the lesson's L4 was expected to be skipped (the navigation's start bar holds the lesson link)")

    s = read_text(s3)
    s, ilog = integrate(s, t, v, d)
    logs.append("[4 integration]\n" + "\n".join(ilog))

    # checks
    if "\r" in s:
        die("the result has CR bytes")
    sc = scan(s)
    seen = {}
    for i in sc.ids:
        seen[i] = seen.get(i, 0) + 1
    dups = sorted(i for i, n in seen.items() if n > 1)
    if dups:
        die("duplicate ids: %s" % dups[:20])
    rx_count = {}
    for i in re.findall(r'\sid="([^"]+)"', s):
        rx_count[i] = rx_count.get(i, 0) + 1
    rx_dups = sorted(i for i, n in rx_count.items() if n > 1)
    if rx_dups:
        die("duplicate ids (text scan): %s" % rx_dups[:20])
    missing_use = sorted({r for r in sc.use_refs if r not in seen})
    if missing_use:
        die("SVG <use> references without a target: %s" % missing_use[:20])
    broken = sorted({r for r in sc.links if r not in seen})
    if broken:
        die("in-page links without a target: %s" % broken[:20])
    # the toy worlds: the section is the toy patch's fragment byte for byte (nothing after the toy patch edits it), and
    # every id of the live page's old toy section, with the five the navigation gave it before, is kept
    frag = read_text(TOY_FRAGMENT)
    ta = once(s, '<section id="toy-worlds">', "toy section")
    if s[ta:s.index("</section>", ta) + len("</section>")] != frag:
        die("section#toy-worlds is not the toy patch's fragment (wf7/toy/toy_fragment.html) byte for byte")
    ba = base_s.index('<section id="toy-worlds">')
    old_toy = base_s[ba:base_s.index("</section>", ba) + len("</section>")]
    toy_ids = set(re.findall(r'\sid="([^"]+)"', old_toy)) | set(TOY_NAV_IDS)
    lost = sorted(toy_ids - set(seen))
    if lost:
        die("ids of the old toy section are gone: %s" % lost)
    mjx = [i for i in seen if i.startswith("MJX")]
    mjxl = [i for i in seen if i.startswith("MJXL")]
    if doc_title(s) != base_title:
        die("the title changed: %r -> %r" % (base_title, doc_title(s)))
    for bad in ('id="wt-open"', 'id="wt"', "Full walkthrough mode</button>", 'class="toc"', "sb-panel", 'aside class="sb"',
                'id="sb-tab"', "sb-tab-away", 'id="ls-start"', "page version 17)", "(page version 23)", "(page version 25)",
                "(the walkthrough pop-out moves the panel)", 'learned: "Learned actor (TD3+BC)"',
                # the status box, which the owner asked to remove and the live page removed (page version 27)
                'id="head-status"', 'id="head-status-h"', '<ul class="g-status"', 'href="#head-status"', "Status as of ",
                'data-nav-label="Status of every line of work"'):
        if bad in s:
            die("the old %r is still in the page" % bad)
    for need, n in (('id="start-learn"', 1), ('<section id="learn"', 1), ('id="lesson-data"', 1), ('id="nv-side"', 1),
                    ("<b>Version %s, %s.</b>" % (v, d), 1),
                    # versions 20-23's own parts, which the patches must keep
                    ('id="simtog"', 1), ('id="sim-learned-data"', 1), ('id="ob-key"', 1), ('id="arm-seeds"', 1),
                    ('id="iw-plain-split"', 1), ('id="iw-plain-refuse"', 1), ('id="lc-table5"', 1), ('id="lc-tq"', 1),
                    ("<b>Version 23, 15 Oct 2026.</b>", 1), ("<b>Version 22, 15 Oct 2026.</b>", 1),
                    ("<b>Version 21, 15 Oct 2026.</b>", 1), ("<b>Version 20, 15 Oct 2026.</b>", 1),
                    # versions 24-25's version-history entries, and version 25's Learner row in the Choices (nav.js)
                    ("<b>Version 24, 15 Oct 2026.</b>", 1), ("<b>Version 25, 16 Oct 2026.</b>", 1),
                    ('row("Learner", all(\'#simtog [data-tog-group="lr"]\')', 1),
                    ('named({ fixed: "Fixed controllers", learned: "Learned actor" }, "data-tog-value")', 1)):
        if s.count(need) != n:
            die("%r occurs %d times, expected %d" % (need, s.count(need), n))
    # versions 24-25's code and parts, as often as in BASE (none of them is edited by this build except K13b's readout,
    # whose anchor holds the learner name): the axis titles, the Learner switch and its data, the rerun's readouts
    for need in ("function axisWords(group, m)", "yTitle(svg, m, H, narrow", "function titles(svg, W, H, m, xs, ys)",
                 "dr-axis dr-head", 'const xTtl = ["State t of the episode (0 to 99)"', "const gTtl = [",
                 'data-tog-group="lr"', 'id="sim-rerun-data"', 'id="simtog-l-lr"', 'id="simtog-lr-q"', 'id="exLrnT"',
                 'id="exLrn"', "function nowMore()", "function ltipOf(id)", "LRO_RERUN_J", "const LRTIP_OFF = {",
                 "function avgOf(L)", "const lrOK = () =>", "Horizontal axis: ", "Vertical axis: ",
                 'class="axlab"', "fitTitle(", "#iw-curves .lc-cell svg .axlab"):
        nb, no = no_toy(base_s).count(need), no_toy(s).count(need)
        if nb == 0 or no != nb:
            die("version 24-25 code %r occurs %d times in BASE and %d times in OUT, outside the toy section (it must be kept "
                "as it is)" % (need, nb, no))
    # ... and the toy section's charts follow version 24's rule: each names both of its axes, on the chart and for
    # screen readers
    toy = s[s.index('<section id="toy-worlds">'):]
    toy = toy[:toy.index("</section>")]
    n_toy_charts = len(re.findall(r'<svg class="chart toy-chart"', toy))
    if not n_toy_charts or toy.count('class="axlab"') != 2 * n_toy_charts or toy.count("Horizontal axis: ") != n_toy_charts \
            or toy.count("Vertical axis: ") != n_toy_charts:
        die("the toy section's %d charts must each name both axes (axis titles %d, Horizontal axis %d, Vertical axis %d)" % (
            n_toy_charts, toy.count('class="axlab"'), toy.count("Horizontal axis: "), toy.count("Vertical axis: ")))
    # the header goes from its opening paragraph straight to "The process, step by step", as on the live page
    def after_glance(x):
        i = x.index('<p class="g-one">')
        j = x.index("</p>", i) + len("</p>")
        k = x.index('<p class="pipe-h" id="pipe-h">', j)
        return x[j:k]
    if after_glance(s) != after_glance(base_s) or after_glance(s).strip():
        die("the header must go from its opening paragraph straight to the process, as on the live page: %r" % after_glance(s)[:200])
    # the live page's changes after version 25, kept as they are: its "Strong test mix" tooltip
    for need in ("data-tip-title=\"Strong test mix\" data-tip=\"The same start in the strong test mix&#x27;s 1,000 fixed starts. "
                 "Only the current differs: the floats and the wave start the same.&#10;What makes it strong: currents above "
                 "0.7 come in about 64% of its episodes, against about 19% in training.&#10;It removes no command and adds "
                 "none: the learned actor picks its own. But strong currents mean hard water far more often, and there the "
                 "best move is the 80% hold that training lacks. On expert-quality data, about 36% of its steps have a "
                 "command in the removed region.\"",):
        nb, no = base_s.count(need), s.count(need)
        if nb != 1 or no != 1:
            die("the live page's tooltip %r occurs %d times in BASE and %d times in OUT (once each expected)" % (need[:60], nb, no))
    # exactly one version-history entry per version number
    vh = s[s.index('<section id="version-history">'):]
    vh = vh[:vh.index("</section>")]
    nums = {}
    for m in re.finditer(r"<li><b>Versions? ([^<]*?)(?:, \d+ Oct 2026|\s\d+ Oct|,|\.)", vh):
        for x in re.findall(r"\d+", re.sub(r"(\d+) to (\d+)", lambda q: " ".join(str(k) for k in range(int(q.group(1)), int(q.group(2)) + 1)), m.group(1).replace(" and ", " "))):
            nums[x] = nums.get(x, 0) + 1
    twice = sorted((x for x, c in nums.items() if c > 1), key=int)
    if twice or v not in nums:
        die("the version history must have exactly one entry per version: repeated %s, version %s %s" % (twice, v, "present" if v in nums else "missing"))
    # the status statements agree with each other (REBASE.md, "Status statements"): no out-of-date wording is left, and
    # each statement of version 26 is there as often as it should be
    for stale in ("awaits the project owner's approval", "awaits its own approval", "not yet in the repository</span>",
                  "<td>the weighting test</td><td>not yet in the shared repository", "its run is next);",
                  "waits for the project owner&#x27;s approval", "waits for the owner's approval", "awaiting approval;",
                  "the project owner's approval typed at their own terminal, then the run", "<code>forked</code> only",
                  "after this page's status was read", "after this status was read", "still awaits its handoff",
                  "the owner&#x27;s approval, typed at their own terminal, comes next",
                  # version 24's statements that the import record and the near-80 test's final plan made out of date
                  "the approval's check and record come next", "next come the\n      approval's check and record",
                  "the approval's\n          check and record and the run itself are still to come",
                  "Still to come: the approval's check against the", "Still to come: the approval is checked",
                  "the approval&#x27;s check against the repository&#x27;s history and its record, then the run, are still to come",
                  "whose draft awaits the owner's approval", "Its draft pre-registration awaits", "Near-80 test: draft",
                  "a draft plan; nothing trained", "its code is being\n          built", "Status as of 16 Oct 2026 (page",
                  # the first reading (11:50 UTC), before the near-80 approval
                  "Near-80 test: awaiting approval", "whose final plan awaits the owner", "Two more are planned, with nothing trained",
                  "(<code>e3e87c3</code>), awaits the owner's", "(a final plan; nothing trained)",
                  # version 26's, made out of date when Stage 2 version 7 was ended without running (page version 27)
                  "the run comes next", "Still to come: the run.", "the run is still to come", "the run itself is still to come",
                  "is what seeds 5–9 will decide", "runs under version 7 on the project", "still runs under version 7",
                  "not run. They run under", "Stage 2 now runs only there", "Near-80 test</b> (approved; no result yet)",
                  "status box</a> at the top", "lines of work continue on this environment",
                  # the old toy section's (versions 19-25), replaced by the toy patch
                  "seeds 5–9 still test the", "because the notebook is not committed"):
        if stale in s:
            die("an out-of-date status statement is still in the page: %r" % stale)
    for need, n in (("As of %s, %s UTC, this is where four\n      lines of work on this environment stand" % (d, t), 1),
                    ("checked and recorded on 16 Oct at 02:27 UTC; later that day version 7 was ended without running", 2),
                    ("Later on 16 Oct version 7\n          was ended without running, and that computer now runs the near-80 test (below)", 1),
                    ("no record of the ending is\n          in the repository.", 1),
                    ("so the\n          deciding test will not run under it.", 1),
                    ("will not run under version 7, which was ended on 16 Oct 2026 without running.", 1),
                    ("version 7, which\n          was to run them, was ended on 16 Oct 2026 without running.</dd>", 1),
                    ("recorded on 16 Oct 2026, 02:27 UTC; later that day the run&#x27;s plan (version 7) was ended without running, so those seeds will not run under it. Nothing has run on those seeds.", 1),
                    ("typed their approval at their own terminal at 15:45 UTC", 1),
                    ("typed their approval at their own terminal\n          at 15:45 (committed at 15:49", 1),
                    ("merged into the shared repository's main branch on\n      15 Oct 2026 at 17:18 UTC (commit <code>03726de</code>)", 1),
                    ("approved by the project owner at 15:45 UTC, and the\n      approval was checked and recorded", 1),
                    ("commit\n        <code>de4851c</code>, 15:49 UTC", 1),
                    ("<code>forked</code> and merged into its main branch at 21:36 UTC)", 1),
                    ("its main branch at commit <code>1ec267e</code> (15 Oct, 21:36 UTC)", 1),
                    ("<code>claude/main-bca-connection-x4o46s</code> at commit <code>0377dd7</code> (16 Oct, 12:38 UTC)", 1),
                    ("head <code>0377dd7</code>, 16 Oct 2026, 12:38 UTC", 1),
                    ("<code>stage2_v7/import_v7_3dfcb6140e2b8e2c.json</code> (<code>02e591b</code>)", 1),
                    ("<b>Near-80 test</b> (approved and running; no result yet)", 1),
                    ("frozen on 16 Oct at 01:33", 1), ("(<code>e3e87c3</code>)", 1), ("their own terminal at 11:50 (committed at 11:51, <code>f79b89c</code>)", 1), ("(<code>287380b</code>)", 1),
                    ("(<code>287380b</code>); it now runs on the project owner's Framework Desktop computer.</li>", 1),
                    ("Two more have no results yet: runs of four", 1),
                    ("Stage 2's version 7, which was to test the width on fresh seeds 5–9, has ended and will not run.", 1),
                    ("the notebook was first committed on 15 Oct, after its runs.", 2),
                    ("whose plan the project owner approved in chat and must still approve at their own terminal", 1),
                    ("their own terminal on 16 Oct and which now runs on the owner's desktop computer\n      (<a href=\"#ats-status\">After the stop</a>).</p>", 1)):
        if s.count(need) != n:
            die("status statement %r occurs %d times, expected %d" % (need, s.count(need), n))
    stamps = re.findall(r"(\d+ Oct 2026),? (?:at )?(\d\d:\d\d) UTC(?:; the box below|</h4>|: not shown on the exploratory seeds|, this is where four)", s)
    if len(stamps) != 4:
        die("expected 4 status times (After the stop's sentence and box, the first claim, Results' opening), found %d" % len(stamps))
    stamps = set(stamps)
    if stamps != {(d, t)}:
        die("the status statements name different times: %s (expected only %s, %s)" % (sorted(stamps), d, t))
    core = s.encode("utf-8")
    head = wrapper.read_bytes()[:WRAP_HEAD_BYTES]
    if not head.rstrip(b"\n").endswith(b"<body>") or b"\r" in head:
        die("the wrapper's first %d bytes do not end with <body> (and a newline)" % WRAP_HEAD_BYTES)
    full = head + core + WRAP_TAIL
    if len(full) >= LIMIT:
        die("the page would be %d bytes, over the 16 MB limit" % len(full))

    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    if preview:
        preview.write_bytes(full)
    print("\n".join(logs))
    print("[checks]\nBASE %d bytes, %d ids -> OUT %d bytes, %d ids (%d MathJax glyph ids, %d of them the lesson's MJXL); "
          "0 duplicate ids; %d SVG <use> references resolve; 0 CR bytes; title %r unchanged (%d title elements in all); "
          "%d in-page links resolve; the toy section is the toy patch's fragment (%d characters), %d ids of the old section kept; "
          "one version-history entry per version (%d entries' numbers); no status box; page version %s, status time %s, %s UTC; published "
          "size with the wrapper %d bytes (limit %d)%s" % (
              len(base_s.encode("utf-8")), len(base_scan.ids), len(core), len(sc.ids), len(mjx), len(mjxl),
              len(sc.use_refs), base_title, len(sc.titles), len(sc.links), len(frag), len(toy_ids), len(nums), v, d, t, len(full), LIMIT,
              ("; preview " + str(preview)) if preview else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
