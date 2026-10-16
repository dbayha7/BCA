"""Apply the lesson "Learn step by step" (section#learn) to a copy of the ThrusterChain page's core HTML.

Usage:  python -X utf8 apply_lesson.py IN.html OUT.html

IN is the page's core HTML (without the publish wrapper); it is read and never written. OUT receives the result.
The blocks it inserts are the files next to this script (made by build/build_lesson.py):
  lesson_fragment.html   section#learn, inserted right after </header> (before section#action-space)   [L1]
  lesson.css             a <style> block of its own, right after the page's first <style> block         [L2]
  lesson_data.json       <script type="application/json" id="lesson-data">, at the end of the page        [L3]
  lesson.js              the lesson's script (one IIFE), after it, the page's last script                  [L3]
and one edit in the header:
  L4  the header button "Full walkthrough mode" (button#wt-open) becomes the link "Learn step by step" (a#ls-start,
      href="#learn"). The walkthrough's dialog, script and CSS are the navigation's to remove; this patch changes only
      the button. When the navigation was applied first, its start bar (#start-learn) already holds the link and the
      button is gone: then L4 is skipped and says so; in any other case of a missing or repeated anchor the script stops.

Every edit is made at an exact text anchor that must occur exactly once at the moment it is applied; otherwise the
script stops with an error that names the edit, and writes nothing. It never introduces CR bytes or duplicate ids
(both checked), and it refuses a page that already holds the lesson. Run on a newer base page, it applies cleanly as
long as the anchors are still there.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def die(msg):
    sys.stderr.write("apply_lesson.py: " + msg + "\n")
    sys.exit(1)


def read(name):
    p = HERE / name
    if not p.exists():
        die("missing input file %s (run build/build_lesson.py first)" % p)
    with open(p, encoding="utf-8", newline="") as f:
        s = f.read()
    if "\r" in s:
        die("%s has CR bytes" % name)
    return s


def once(s, needle, label):
    n = s.count(needle)
    if n != 1:
        die("edit %s: its anchor %r occurs %d times (it must occur exactly once)" % (label, needle[:90], n))
    return s.index(needle)


def ids_of(s):
    return re.findall(r'\sid="([^"]+)"', s)


def main():
    if len(sys.argv) != 3:
        die("usage: python -X utf8 apply_lesson.py IN.html OUT.html")
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if src.resolve() == out.resolve():
        die("IN and OUT must be different files")
    with open(src, encoding="utf-8", newline="") as f:
        s = f.read()
    if "\r" in s:
        die("the input has CR bytes")
    if 'id="learn"' in s or 'id="lesson-data"' in s:
        die("the lesson is already in this page")
    frag, css, data, js = read("lesson_fragment.html"), read("lesson.css"), read("lesson_data.json").strip(), read("lesson.js")
    if "</script" in data.lower() or "<!--" in data:
        die("lesson_data.json is not safe inside a script element")
    if "</script" in js.lower().replace("<\\/script", ""):
        die("lesson.js holds a closing script tag")
    ids_before = ids_of(s)
    dup_before = sorted({i for i in ids_before if ids_before.count(i) > 1})
    frag_ids = set(ids_of(frag))
    clash = sorted(frag_ids & set(ids_before))
    if clash:
        die("the lesson's ids are already used in the page: %s" % clash[:10])
    log = []

    # L1: section#learn right after the header (At a glance), before Action Space
    a = '</header>\n\n  <section id="action-space">'
    i = once(s, a, "L1")
    j = i + len("</header>\n\n")
    s = s[:j] + frag + "\n" + s[j:]
    log.append("L1 section#learn after </header> (%d bytes)" % len(frag.encode("utf-8")))

    # L2: the lesson's CSS, a <style> block of its own after the page's first one
    a = "/* qviewer:end style */\n</style>\n"
    i = once(s, a, "L2")
    j = i + len(a)
    s = s[:j] + "<style>\n" + css.rstrip("\n") + "\n</style>\n" + s[j:]
    log.append("L2 lesson CSS (%d bytes)" % len(css.encode("utf-8")))

    # L4: the header button (see the docstring for the one allowed skip)
    a = ('<button type="button" class="btn wt-open" id="wt-open" aria-haspopup="dialog" aria-controls="wt">'
         'Full walkthrough mode</button>')
    n = s.count(a)
    if n == 1:
        s = s.replace(a, '<a class="btn ls-start" id="ls-start" href="#learn">Learn step by step</a>', 1)
        log.append("L4 header button -> a#ls-start \"Learn step by step\" (href #learn)")
    elif n == 0 and 'id="start-learn"' in s and 'id="wt-open"' not in s:
        log.append("L4 skipped: the navigation's start bar (#start-learn) already links to #learn and #wt-open is gone")
    else:
        die("edit L4: its anchor (the Full walkthrough mode button) occurs %d times, and the navigation's start bar is %s"
            % (n, "present" if 'id="start-learn"' in s else "absent"))

    # L3: the data block and the lesson's script, at the end of the page (after its last script)
    tail = s.rstrip("\n")
    if not tail.endswith("</script>"):
        die("edit L3: the page does not end with a script (its last characters: %r)" % tail[-40:])
    s = tail + '\n<script type="application/json" id="lesson-data">' + data + "</script>\n<script>\n" + js.rstrip("\n") + "\n</script>\n"
    log.append("L3 data block (%d bytes) and lesson script (%d bytes)" % (len(data.encode("utf-8")), len(js.encode("utf-8"))))

    # checks
    if "\r" in s:
        die("the result has CR bytes")
    ids_after = ids_of(s)
    dup_after = sorted({i for i in ids_after if ids_after.count(i) > 1})
    new_dups = [d for d in dup_after if d not in dup_before]
    if new_dups:
        die("the result has duplicate ids: %s" % new_dups[:10])
    for need in ('<section id="learn"', 'id="lesson-data"', "Learn step by step"):
        if need not in s:
            die("after the edits %r is missing" % need)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print("\n".join(log))
    print("bytes %d -> %d; ids %d -> %d; duplicate ids before %d, after %d" % (
        len(open(src, "rb").read()), len(s.encode("utf-8")), len(ids_before), len(ids_after), len(dup_before), len(dup_after)))


if __name__ == "__main__":
    main()
