"""Apply the page's navigation (nav_spec.md, built on page version 19) to a copy of the page source.

Usage:  python -X utf8 apply_nav.py IN.html OUT.html

IN is the page's core HTML (without the publish wrapper); it is read and never written. OUT receives the result.
Every edit is made at an exact text anchor that must occur exactly once in the page at the moment it is applied;
otherwise the script stops with an error that names the edit, and writes nothing. Run on a newer base page, it applies
cleanly as long as every anchor is still there once. It never introduces CR bytes or duplicate ids (both checked).

The blocks it inserts are the files next to this script, in src/:
  static_edits.json   E10: ids and short labels (data-nav-label) for every listed part (regenerated on version 19)
  nav.css             the navigation's CSS
  nav_markup.html     early script, skip link, the bar, the Choices button, the dim layer, Contents, the live region
  nav_side.html       "In this section" (inside .wrap, before <main>)
  start_bar.html      the start bar in the header (replaces the walkthrough bar; the Tooltips switch moves to Contents)
  nav.js              the navigation script (stays the page's last script)
Page version 24 (rebased on version 23): static_edits.json is made by rules/static_edits_v23.py; E1 also removes
version 21's script that moved the old Choices tab aside (the tab goes with E1); E14 rewrites the comment of version
20's switch-bar script that named the walkthrough pop-out and the old sidebar (both removed here).
Page version 26 (rebased on version 25): nav.js carries version 25's Choices rows ("Learned actor", "Learner"). Every
edit that removes a block (E11, E7a-c, E1, E2, E4, E5, E6, E8) checks the SHA-256 of the text it removes against the
versions this build knows (KNOWN below: version 23's and version 25's), so that a newer base page that changed a
replaced block stops the patch instead of losing the change; carry the change into src/ and add its hash.
Page version 27 (on the live page after version 25, whose header status box was removed at the owner's request):
static_edits.json is made by rules/static_edits_v27.py, which drops the box's label from the version-23 edits (132
edits); the blocks this patch replaces are still version 25's, and the navigation lists no status box. With the
toy worlds' full results (the toy patch, wf7/toy/apply_toy.py, the build's first stage), this patch runs on the page
after the toy patch, and static_edits.json is made by rules/static_edits_v27t.py from those 132 edits without the five
that gave parts of the old toy section their ids (the new section has the ids itself): 127 edits. The toy section's
parts all carry ids, so Contents and "In this section" list them by the navigation's own rules.
Order (nav_spec 13.2): E10, E9, E11, E7a, E7b, E7c, E1, E2, E4, E5, E6, E8, then (fix round 1) E12 and E13: the page's
table scroll cue and the tooltips' Tab stops skip tables in hidden sections. The lesson (E3, section#learn right after
the header) is inserted by the lesson's own patch, before or after this one; the navigation finds it when it is there.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def die(msg):
    sys.stderr.write("apply_nav.py: " + msg + "\n")
    sys.exit(1)


def block(name):
    with open(SRC / name, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n")


def once(s, needle, label):
    n = s.count(needle)
    if n != 1:
        die("edit %s: its anchor %r occurs %d times (it must occur exactly once)" % (label, needle[:90], n))
    return s.index(needle)


# SHA-256 (first 16 hex digits) of the text each block edit removes, on the versions this build knows. Version 23's
# and version 25's blocks are the same except E8 (version 25 named the learned button "Learned actor" and added the row
# "Learner" to the old sidebar's script; nav.js carries both). E2's second hash is the header's walkthrough bar after
# the lesson's L4 (its button made a lesson link), when the lesson patch runs before this one (the order test).
KNOWN = {
    "E7a": {"c2bb1771bf83c03e"}, "E7b": {"780735490d3ae0a1"}, "E7c": {"52af8f79174455ce"},
    "E1": {"c7ee78a5d09e8a0d"}, "E2": {"6175747978f305d3", "d7210c08d6f1c341"}, "E4": {"60c929b95f33e9dc"}, "E5": {"bc49df82a2167eb9"},
    "E6": {"8bceb97dc648ed57"}, "E8": {"329dba5e8f45d88c", "8715864a03fa09f2"}, "E11": {"baf0e98bb1791b45"},
}
# The live page's version 26 (service version 1792172582-8406, 16 Oct 2026): E4's old contents with its four new entries
# (BCA's main channel, and three of the toy section's parts, which "In this section" lists).
KNOWN["E4"].add("77d6fafecdc1b7fe")
# ... and E8's old Choices script, whose "Tested on" row lists the main test's four situations too (nav.js carries them)
KNOWN["E8"].add("b0cd8cc4deca8827")
# ... and the live page after version 26 (service version 1792181606-4140): its "Learner" row names result F's learners
KNOWN["E8"].add("954959ae7c56fda0")


def known(label, removed):
    h = hashlib.sha256(removed.encode("utf-8")).hexdigest()[:16]
    if label in KNOWN and h not in KNOWN[label]:
        die("edit %s: the block it replaces (SHA-256 %s...) is not a version this build knows (%s); the base page changed "
            "it, so carry the change into src/ first and add its hash to KNOWN" % (label, h, ", ".join(sorted(KNOWN[label]))))


def span(s, start, end, new, label, inclusive=True):
    i = once(s, start, label + " (start)")
    j = once(s, end, label + " (end)")
    if j < i:
        die("edit %s: its end anchor comes before its start anchor" % label)
    stop = j + len(end) if inclusive else j
    known(label, s[i:stop])
    return s[:i] + new + s[stop:]


def ids_of(s):
    return re.findall(r'\sid="([^"]+)"', s)


def main():
    if len(sys.argv) != 3:
        die("usage: python -X utf8 apply_nav.py IN.html OUT.html")
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    if src_path.resolve() == out_path.resolve():
        die("IN and OUT must be different files")
    with open(src_path, encoding="utf-8", newline="") as f:
        s = f.read()
    if "\r" in s:
        die("the input has CR bytes")
    if 'id="nv-side"' in s or 'class="nv-pager"' in s:
        die("the navigation is already in this page")
    ids_before = ids_of(s)
    dup_before = sorted({i for i in ids_before if ids_before.count(i) > 1})
    log = []

    # E10: an id for every listed part that has none, and short labels (one exact anchor each)
    edits = json.loads((SRC / "static_edits.json").read_text(encoding="utf-8"))["edits"]
    for e in edits:
        once(s, e["old"], "E10 " + (e.get("id") or e["label"]))
    taken = set(ids_before)
    for e in edits:
        if e.get("id") and e["id"] in taken:
            die("edit E10 %s: the id is already used in the page" % e["id"])
    for e in edits:
        s = s.replace(e["old"], e["new"], 1)
    log.append("E10 %d static edits" % len(edits))

    # E9: the two captions sized by the viewport are sized by the page column
    for a in (".subsec table.rows8 tr.sub td[colspan] .grp { display: inline-block; max-width: calc(100vw - 72px); white-space: normal; }",
              ".subsec table.rows8 caption .cap { display: inline-block; max-width: calc(100vw - 72px); }"):
        once(s, a, "E9")
        s = s.replace(a, a.replace("calc(100vw - 72px)", "calc(var(--nv-main-w, calc(100vw - 40px)) - 32px)"), 1)
    log.append("E9 two caption rules")

    # E11: the page grid has one column (the navigation adds the rail's column); the old right rail's CSS goes
    s = span(s, ".wrap { max-width: 1200px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 72px;\n"
                "  display: grid; grid-template-columns: minmax(0, 1fr) 200px; gap: 48px; }\n",
             "section { display: flex; flex-direction: column; gap: 14px; scroll-margin-top: 24px; min-width: 0; }",
             ".wrap { max-width: 1200px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 72px;\n"
             "  display: grid; grid-template-columns: minmax(0, 1fr); gap: 32px; }\n"
             "main { min-width: 0; display: flex; flex-direction: column; gap: 56px; }\n", "E11", inclusive=False)
    a = ("/* the indent marks a subsection in the sidebar; the phone's row of pills has no indent */\n"
         "@media (min-width: 1001px) { nav.toc a.sub { padding-left: 24px; font-size: 13.5px; } }\n")
    once(s, a, "E11 (sub-link indent)")
    s = s.replace(a, "", 1)
    # version 26 added a rule for the old contents' links under it; it goes too
    a = "@media (min-width: 1001px) { nav.toc a { padding-block: 2px; line-height: 1.4; } }\n"
    once(s, a, "E11 (version 26's contents-link rule)")
    s = s.replace(a, "", 1)
    log.append("E11 page grid and nav.toc CSS")

    # E7: the full walkthrough mode's CSS (three blocks); the .head-body rule stays
    s = span(s, "/* full walkthrough mode: the page's sections one at a time in a pop-out, in reading order */\n",
             "/* tooltips: one shared tooltip (made by the last script) for column names, table cells, terms and controls; the",
             ".head-body { display: flex; flex-direction: column; gap: 18px; scroll-margin-top: 24px; }\n", "E7a", inclusive=False)
    s = span(s, "/* the walkthrough changes sections without moving anything sideways, so nothing is ever wider than the pop-out:",
             ".wt-stage.wt-enter-next, .wt-stage.wt-enter-back { transform: none; }\n", "", "E7b")
    s = span(s, "/* walkthrough mode: a link can point at a heading near the end of a section; room after the section's last block",
             '.wt-stage::after { content: ""; flex: none; height: min(30vh, 240px); }\n', "", "E7c")
    log.append("E7 walkthrough CSS")

    # E1: the version-18 Choices sidebar (CSS and markup) -> the navigation's CSS and markup, and the rail in .wrap.
    # Version 21 put a small script between the sidebar's markup and .wrap: it moved the sidebar's tab (#sb-tab) aside
    # while a panel's own switches were on screen. The tab goes with the sidebar, so the script goes too; anything
    # else found there stops the patch.
    a0 = "<style>\n/* the choices sidebar: every choice of the page's live panels as short buttons, with no explanations. Wide screens:"
    a1 = '<div class="wrap" lang="en">\n<main>'
    i, j = once(s, a0, "E1 (start)"), once(s, a1, "E1 (end)")
    k = s.rfind("</aside>\n", i, j)
    if j < i or k < 0:
        die("edit E1: the sidebar's </aside> was not found between its anchors")
    tail = s[k + len("</aside>\n"):j]
    if tail and not (tail.startswith("<script>\n") and tail.endswith("</script>\n") and tail.count("<script") == 1
                     and 'getElementById("sb-tab")' in tail and "sb-tab-away" in tail):
        die("edit E1: unexpected content between the sidebar and .wrap: %r" % tail[:200])
    known("E1", s[i:j + len(a1)])
    s = (s[:i] + "<style>\n" + block("nav.css") + "</style>\n" + block("nav_markup.html") + '<div class="wrap" lang="en">\n'
         + block("nav_side.html") + "<main>" + s[j + len(a1):])
    log.append("E1 navigation CSS and markup" + (" (and version 21's tab-away script)" if tail else ""))

    # E2: the walkthrough bar in the header -> the start bar
    s = span(s, '    <div class="wt-bar" hidden>\n', '    <div class="head-body" id="glance" data-wt-title="At a glance">',
             block("start_bar.html") + '    <div class="head-body" id="glance">', "E2")
    log.append("E2 start bar")

    # E4: the right-hand rail nav.toc -> the end-of-section pager, inside <main>
    s = span(s, '</main>\n\n<nav class="toc" aria-label="On this page">',
             '  <a href="#version-history">Version history</a>\n</nav>\n</div>',
             '  <nav class="nv-pager" id="nv-pager" aria-label="Previous and next section"></nav>\n</main>\n</div>', "E4")
    log.append("E4 pager")

    # E5 and E6: the walkthrough dialog and its script
    s = span(s, '<dialog class="wt" lang="en" id="wt" aria-label="Full walkthrough mode" aria-describedby="wt-count">',
             "</dialog>\n\n", "", "E5")
    s = span(s, "<script>\n/* Full walkthrough mode: the page's sections one at a time in a pop-out, in reading order. The real nodes move into",
             '  window.addEventListener("hashchange", () => { if (dlg.open) close(); });\n})();\n</script>\n\n', "", "E6")
    log.append("E5 E6 walkthrough dialog and script")

    # E8: the Choices sidebar's script -> the navigation script (the page's last script)
    s = span(s, "<script>\n/* The choices sidebar. Every button stands for one control of a live panel and presses that control, so the panel's",
             '  setOpen(wide.matches && saved() !== "0", false);\n})();\n</script>',
             block("nav.js").rstrip("\n"), "E8")
    log.append("E8 navigation script")

    # E12 and E13 (fix round 1, finding T1): the page's table scroll cue and the tooltips' Tab stops skip tables in a
    # hidden section or lesson page (hidden="until-found", the class nv-off, or any section while the page loads under
    # nv-boot): reading their layout forced Chrome to lay out every hidden section at load. A shown section is measured
    # by the window "resize" the navigation sends after showing it.
    hid = lambda v: ("%s.closest('[hidden=\"until-found\"], .nv-off') || (document.documentElement.classList.contains('nv-boot')"
                     " && %s.closest('main > section[id], main > figure.panel[id]'))" % (v, v))
    a = "    const up = () => {\n      const over = box.scrollWidth > box.clientWidth + 1;"
    once(s, a, "E12")
    s = s.replace(a, "    const up = () => {\n      if (" + hid("box") + ") return;   // hidden: measured when shown\n"
                  "      const over = box.scrollWidth > box.clientWidth + 1;", 1)
    a = "  function fixStops() {\n    tables.forEach(info => {\n      if (info.stop && info.stop.getClientRects().length) return;"
    once(s, a, "E13")
    s = s.replace(a, "  function fixStops() {\n    tables.forEach(info => {\n      const c0 = info.cells[0];   // hidden: fixed when shown\n"
                  "      if (c0 && (" + hid("c0") + ")) return;\n"
                  "      if (info.stop && info.stop.getClientRects().length) return;", 1)
    log.append("E12 E13 table cue and Tab stops skip hidden sections")

    # E14 (page version 24): the comment of version 20's switch-bar script named the walkthrough pop-out and the old
    # Choices sidebar, both removed above
    a = ("/* the quick switches: each presses the live panel's own option with the same group and value, found afresh on every\n"
         "   use (the walkthrough pop-out moves the panel), and copies the options' pressed state back whenever they change,\n"
         "   whoever changed them: the big options above, the choices sidebar or the task demo's \"Watch one episode\" links.\n")
    once(s, a, "E14")
    s = s.replace(a, "/* the quick switches: each presses the live panel's own option with the same group and value, found afresh on every\n"
                  "   use, and copies the options' pressed state back whenever they change, whoever changed them: the big options\n"
                  "   above, the panel's choices under \"In this section\" or the task demo's \"Watch one episode\" links.\n", 1)
    log.append("E14 the switch bar's comment")

    # checks
    for bad in ('id="wt-open"', 'id="wt"', "sb-panel", 'class="toc"', "wt-stage", 'aside class="sb"', "nav.toc", 'id="sb"',
                'id="sb-tab"', "sb-tab-away", "(the walkthrough pop-out moves the panel)"):
        if bad in s:
            die("after the edits %r is still in the page" % bad)
    if "\r" in s:
        die("the result has CR bytes")
    ids_after = ids_of(s)
    dup_after = sorted({i for i in ids_after if ids_after.count(i) > 1})
    if [d for d in dup_after if d not in dup_before]:
        die("the result has duplicate ids: %s" % [d for d in dup_after if d not in dup_before])
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print("\n".join(log))
    print("bytes %d -> %d; ids %d -> %d; duplicate ids %d" % (len(open(src_path, "rb").read()), len(s.encode("utf-8")),
                                                               len(ids_before), len(ids_after), len(dup_after)))


if __name__ == "__main__":
    main()
