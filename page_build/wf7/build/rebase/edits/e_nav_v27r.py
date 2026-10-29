"""Rebase onto the live page's version 26 (service version 1792172582-8406): the navigation's anchors and blocks that
version 26 changed.
E10 iw-qualities-table: version 26's toy section has a second table whose caption starts "The f" ("The four parts"),
so the anchor is lengthened to the caption's own words."""
import json
import rep
from pathlib import Path

p = rep.W7 / "nav" / "src" / "static_edits.json"
d = json.loads(p.read_text(encoding="utf-8"))
hit = [e for e in d["edits"] if e.get("id") == "iw-qualities-table"]
assert len(hit) == 1
e = hit[0]
OLD = '<table class="stack">\n      <caption><span class="cap">The f'
NEW = ('<table class="stack">\n      <caption><span class="cap">The five data qualities, <span class="km" '
       'data-sit="strong_current">strong test</span>')
if e["old"] == OLD:
    e["old"] = NEW
    e["new"] = NEW.replace('<table class="stack">', '<table id="iw-qualities-table" class="stack">', 1)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("wrote nav/src/static_edits.json (E10 iw-qualities-table)")
else:
    assert e["old"] == NEW, "unexpected anchor"
    print("already applied")

# E4: version 26 added four entries to the old right-hand contents (nav.toc), which E4 replaces by the pager: the new
# navigation lists every section itself (BCA's main channel gets its own entry below) and every part of the toy section
# under "In this section". The block's version-26 hash is added to KNOWN.
# E11: version 26 added a rule for the old contents' links ("nav.toc a { padding-block ... }") under the sub-link indent
# rule that E11 removes; it goes with it (it styles only the removed nav.toc). Version 26's "#capped-pull td.n" stays.
# Contents: an entry for version 26's section#capped-pull, after the weighting test (its own h2 is a full sentence).
rep.apply([
    ("nav/apply_nav.py",
     '"E6": {"8bceb97dc648ed57"}, "E8": {"329dba5e8f45d88c", "8715864a03fa09f2"}, "E11": {"baf0e98bb1791b45"},\n}',
     '"E6": {"8bceb97dc648ed57"}, "E8": {"329dba5e8f45d88c", "8715864a03fa09f2"}, "E11": {"baf0e98bb1791b45"},\n}\n'
     '# The live page\'s version 26 (service version 1792172582-8406, 16 Oct 2026): E4\'s old contents with its four new entries\n'
     '# (BCA\'s main channel, and three of the toy section\'s parts, which "In this section" lists).\n'
     'KNOWN["E4"].add("77d6fafecdc1b7fe")', 1),
    ("nav/apply_nav.py",
     '    once(s, a, "E11 (sub-link indent)")\n    s = s.replace(a, "", 1)\n',
     '    once(s, a, "E11 (sub-link indent)")\n    s = s.replace(a, "", 1)\n'
     '    # version 26 added a rule for the old contents\' links under it; it goes too\n'
     '    a = "@media (min-width: 1001px) { nav.toc a { padding-block: 2px; line-height: 1.4; } }\\n"\n'
     '    once(s, a, "E11 (version 26\'s contents-link rule)")\n    s = s.replace(a, "", 1)\n', 1),
    ("nav/src/nav_markup.html",
     '      <li class="nv-sec"><a href="#weighting-test" data-unit="weighting-test">The weighting test</a></li>\n',
     '      <li class="nv-sec"><a href="#weighting-test" data-unit="weighting-test">The weighting test</a></li>\n'
     '      <li class="nv-sec"><a href="#capped-pull" data-unit="capped-pull">BCA\'s main channel</a></li>\n', 1),
])

# E8: version 26 gave the old Choices script's "Tested on" row the main test's four situations (their buttons, in the
# live panel's "Tested on" control, play only with the 80% hold removed); nav.js carries the same names, and the
# block's version-26 hash is added to KNOWN.
rep.apply([
    ("nav/src/nav.js",
     '        named({ training: "Training mix", strong: "Strong test mix" }, "data-tog-value"), { idle: simFixed }) ] },\n',
     '        named({ training: "Training mix", strong: "Strong test mix", recorded: "Main test: recorded conditions (no shift)",\n'
     '          big_kicks: "Main test: big starting pushes (small shift)", big_waves: "Main test: big waves (medium shift)",\n'
     '          strong_current: "Main test: strong current (large shift)" }, "data-tog-value"), { idle: simFixed }) ] },\n', 1),
    ("nav/apply_nav.py",
     'KNOWN["E4"].add("77d6fafecdc1b7fe")',
     'KNOWN["E4"].add("77d6fafecdc1b7fe")\n'
     '# ... and E8\'s old Choices script, whose "Tested on" row lists the main test\'s four situations too (nav.js carries them)\n'
     'KNOWN["E8"].add("b0cd8cc4deca8827")', 1),
])
