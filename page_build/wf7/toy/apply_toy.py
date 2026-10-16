"""apply_toy.py IN OUT [--fragment PATH] [--allow-unknown-base]

Replaces section#toy-worlds of the ThrusterChain page (core HTML, without the publish wrapper) with the full
toy-world results of all five toy notebooks (page version 27): toy_fragment.html next to this script, built by
build_toy.py.

Anchors, each checked to occur exactly once (a clear error otherwise, and nothing is written):
  T1  '<section id="toy-worlds">'  the section's start tag (and the id "toy-worlds" occurs once in the page);
  T2  the first '</section>' after T1, with no '<section' between them (the section is not nested).
The section being replaced must be a version this patch knows (SHA-256 below): the live page's own (versions 19 to
25 and the live service version 92ba), or that section after our color patch, or after our color and navigation
patches. A section changed in any other way (for example by the other session) is refused, so that no newer edit is
overwritten unseen; --allow-unknown-base overrides this after a person has compared the two.
Checks on the result: no CR byte added, no new duplicate id, every '#id' link anywhere in the page that pointed into
the old section still resolves, every link and tooltip reference inside the new section resolves, and IN is never
modified. Re-running on a page that already has this fragment changes nothing.

Order in the build: run it AFTER the color patch (its edits B14 and B15 rewrite two sentences of the old section) and
AFTER the navigation patch (its static_edits.json gives five parts of the old section ids: toy-reward-table,
toy-pooled-table, toy-shelved-width, toy-shelved-posterior, toy-more-detail). The new section keeps all five ids and
the old section's eight others (toy-worlds, toy-why, toy-world, toy-first, toy-second, toy-shelved, toy-recheck,
toy-meaning; the lesson's step 5.5 and its sources link to #toy-meaning). Its parts all carry ids, so the navigation's
"In this section" list finds them by its own rules. To change the section's text, edit build_toy.py and run
python -X utf8 build_toy.py, then python -X utf8 check_numbers.py (every quoted number against the re-runs).
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
START = '<section id="toy-worlds">'
KNOWN = {   # SHA-256 of the whole section, '<section id="toy-worlds">' to its '</section>'
    # the live page's own toy section: versions 19 to 25 and the live service version 1792156350-92ba (28,926 chars)
    "88505c7ac804b7c7e0dc2dc047aeeba396b780fb527bdd0549d7096a7505d05f": "live page's section (versions 19-25, 92ba)",
    # the same after our color patch (its B14 and B15 on the section's two "not committed" sentences)
    "e4fc4f4b3aeba23cbee4494c084e6255f3a81f93ff5ab783f8175f2db4e40f58": "live section after the color patch",
    # the same after our color and navigation patches (ids on 2 tables, 2 findings and the details)
    "33b80c29c778d6d2ad84a9b5b1516550f436db563b6b55da6f19731e588a190f":
        "live section after the color and navigation patches",
}


def die(msg):
    sys.stderr.write("apply_toy.py: ERROR: " + msg + "\n")
    sys.exit(1)


def ids_of(html):
    """id attributes outside <script> bodies (a script's own text may spell id=\"...\")."""
    no_script = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S | re.I)
    return re.findall(r'\sid="([^"]+)"', no_script)


def main():
    frag_path = os.path.join(HERE, "toy_fragment.html")
    allow_unknown = False
    raw = sys.argv[1:]
    args, k = [], 0
    while k < len(raw):
        a = raw[k]
        if a == "--fragment":
            frag_path = raw[k + 1]
            k += 2
            continue
        if a == "--allow-unknown-base":
            allow_unknown = True
        elif a.startswith("--"):
            die(f"unknown option {a}")
        else:
            args.append(a)
        k += 1
    if len(args) != 2:
        die("usage: apply_toy.py IN OUT [--fragment PATH] [--allow-unknown-base]")
    src, out = args
    if os.path.abspath(src) == os.path.abspath(out):
        die("OUT must differ from IN (IN is never modified)")
    page = open(src, encoding="utf-8", newline="").read()
    frag = open(frag_path, encoding="utf-8", newline="").read()

    # the fragment itself
    if "\r" in frag:
        die("the fragment contains a CR byte")
    if not frag.startswith(START) or not frag.endswith("</section>"):
        die("the fragment must start with '<section id=\"toy-worlds\">' and end with '</section>'")
    if frag.count("<section") != 1 or frag.count("</section>") != 1:
        die("the fragment must hold exactly one section")

    # T1
    n = page.count(START)
    if n != 1:
        die(f"anchor T1 '{START}' occurs {n} times (expected exactly once)")
    n = page.count('id="toy-worlds"')
    if n != 1:
        die(f"the id toy-worlds occurs {n} times in IN (expected exactly once)")
    a = page.index(START)
    # T2
    b = page.find("</section>", a)
    if b < 0:
        die("anchor T2: no '</section>' after the toy section's start")
    if "<section" in page[a + len(START):b]:
        die("anchor T2: another <section> opens inside section#toy-worlds; its end cannot be found safely")
    b += len("</section>")
    old = page[a:b]
    if old == frag:
        open(out, "w", encoding="utf-8", newline="").write(page)
        print("apply_toy.py: the page already has this toy section; OUT = IN")
        return
    h = hashlib.sha256(old.encode("utf-8")).hexdigest()
    if h not in KNOWN:
        msg = (f"section#toy-worlds in IN is not a version this patch knows (SHA-256 {h[:16]}..., {len(old):,} characters). "
               "It may hold newer edits that this patch would overwrite; compare it with the live page's version-25 "
               "section and pass --allow-unknown-base only after checking.")
        if not allow_unknown:
            die(msg)
        sys.stderr.write("apply_toy.py: WARNING: " + msg + " Proceeding (--allow-unknown-base).\n")
    else:
        print(f"apply_toy.py: replacing the {KNOWN[h]} ({len(old):,} characters, SHA-256 {h[:16]}...)")

    new = page[:a] + frag + page[b:]

    # checks on the result
    if new.count("\r") > page.count("\r"):
        die("the result would gain a CR byte")
    before = ids_of(page)
    after = ids_of(new)
    dup_before = {x for x in set(before) if before.count(x) > 1}
    seen, dup_after = set(), set()
    for x in after:
        if x in seen:
            dup_after.add(x)
        seen.add(x)
    added = dup_after - dup_before
    if added:
        die("the result would have new duplicate ids: " + ", ".join(sorted(added)[:20]))
    all_ids = set(after)
    old_ids = set(ids_of(old))
    outside = page[:a] + page[b:]
    for ref in sorted(set(re.findall(r'href="#([^"]+)"', outside))):
        if ref in old_ids and ref not in all_ids:
            die(f"a link elsewhere on the page points to #{ref}, which the new section no longer has")
    for ref in sorted(set(re.findall(r'href="#([^"]+)"', frag))):
        if ref not in all_ids:
            die(f"the new section links to #{ref}, which is not on the page")
    for ref in sorted(set(re.findall(r'data-tip-ref="([^"]+)"', frag))):
        if ref not in all_ids:
            die(f"a tooltip in the new section refers to the template #{ref}, which is missing")
    lost = sorted(old_ids - all_ids)
    open(out, "w", encoding="utf-8", newline="").write(new)
    print(f"apply_toy.py: wrote {out}: {len(page):,} -> {len(new):,} characters; section {len(old):,} -> "
          f"{len(frag):,}; ids of the old section kept: {len(old_ids) - len(lost)} of {len(old_ids)}"
          + (f" (not kept, and linked from nowhere: {', '.join(lost)})" if lost else ""))


if __name__ == "__main__":
    main()
