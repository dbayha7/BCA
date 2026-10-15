#!/usr/bin/env python3
"""Page pass 4, edit 3: bring the tooltip text sources in line with the page (run once, after p4_e3_apply.py).

Edits 1 and 2 of pass 4 did not touch the sources, so this records all three edits so far, read from the page:

- ta_texts.py gets PASS4_EDITS_1_3, a record (ta_apply.py does not use it) of
    tips        every data-tip text on the page that the version-12 page did not have, by section, title -> text;
    legends     every symbol legend whose text pass 4 changed, as the page now has it (group heading, symbol, definition;
                the symbol and definition as HTML; the worked-example line if there is one);
    cols        the column descriptions (COLS) of every section whose descriptions pass 4 changed, as the page has them;
    superseded  the record ids of the older entries these replace: TIPS texts and LEGENDS entries no longer on the page,
                and the PASS3_FINAL sections whose COLS changed.
- ta_code.py gets PASS4_TT, the one change pass 4 made to the shared tooltip script (edit 2: a formula's tooltip shows
  every group of a grouped legend), as an (old, new) snippet like its TT list. Pass 4 changed no other snippet that
  ta_code.py holds (checked: every TT, DR and IW snippet that was on the version-12 page is still on the page).

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && python3 scripts/p4_e3_sources.py'
"""
import hashlib
import html
import json
import os
import pprint
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
NOW = open(os.path.join(PAGE_DIR, "thrusterchain.html"), encoding="utf-8").read()
V12 = open(os.path.join(PAGE_DIR, "tmp", "p4", "edit1.before.html"), encoding="utf-8").read()
assert hashlib.md5(V12.encode("utf-8")).hexdigest() == "f3736a56dc7698e0b57633acbb91de3c", "not the version-12 page"
E3_MD5 = "f376b41c008320ad4881a0e68beef8e6"
assert hashlib.md5(NOW.encode("utf-8")).hexdigest() == E3_MD5, "the page is not the edit-3 page"
TA_TEXTS = os.path.join(HERE, "ta_texts.py")
TA_CODE = os.path.join(HERE, "ta_code.py")
src_t = open(TA_TEXTS, encoding="utf-8").read()
src_c = open(TA_CODE, encoding="utf-8").read()
assert "PASS4_EDITS_1_3" not in src_t and "PASS4_TT" not in src_c, "already recorded"
ns_t, ns_c = {}, {}
exec(compile(src_t, TA_TEXTS, "exec"), ns_t)
exec(compile(src_c, TA_CODE, "exec"), ns_c)


def sections(page):
    return [(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', page)]


def sec_at(secs, i):
    r = "(outside any section)"
    for a, n in secs:
        if a <= i:
            r = n
    return r


TAG = re.compile(r'<[a-zA-Z][^>]*?\sdata-tip="([^"]*)"[^>]*>')


def tips(page):
    secs, out = sections(page), []
    for m in TAG.finditer(page):
        t = re.search(r'\sdata-tip-title="([^"]*)"', m.group(0))
        out.append((sec_at(secs, m.start()), html.unescape(t.group(1)) if t else "", html.unescape(m.group(1))))
    return out


old_texts = {x[2] for x in tips(V12)}
new_tips = {}
for sec, title, text in tips(NOW):
    if text not in old_texts:
        new_tips.setdefault(sec, {})
        key = title or "(no title)"
        if key in new_tips[sec] and new_tips[sec][key] != text:
            k = 2
            while f"{key} ({k})" in new_tips[sec]:
                k += 1
            key = f"{key} ({k})"
        new_tips[sec][key] = text

# legends: every div.symkey, as the page has it
LEG = re.compile(r'<div class="symkey"[^>]*>.*?</dl>\s*(?:<p class="symkey-ex">.*?</p>\s*)?</div>', re.S)


def legends(page):
    secs, out = sections(page), []
    for m in LEG.finditer(page):
        out.append((sec_at(secs, m.start()), m.group(0)))
    return out


def parse_legend(block):
    head = re.search(r'<p class="symkey-h">(.*?)</p>', block, re.S).group(1)
    rows, group = [], None
    for m in re.finditer(r'<p class="symkey-g">(.*?)</p>|<div[^>]*>\s*<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>\s*</div>', block, re.S):
        if m.group(1) is not None:
            group = re.sub(r"\s+", " ", m.group(1)).strip()
        else:
            rows.append([group, m.group(2), re.sub(r"\s+", " ", m.group(3)).strip()])
    ex = re.search(r'<p class="symkey-ex">(.*?)</p>', block, re.S)
    return {"heading": head, "rows": rows, "example": re.sub(r"\s+", " ", ex.group(1)).strip() if ex else None}


norm = lambda x: re.sub(r"\s+", " ", x)
old_legs = {norm(b) for _, b in legends(V12)}
leg_now = legends(NOW)
N_LEG = len(re.findall(r'<div class="symkey"[^>]*>', NOW))
assert len(leg_now) == N_LEG, (len(leg_now), N_LEG)
for _, b in leg_now:
    p = parse_legend(b)
    assert p["rows"] and len(p["rows"]) == b.count("<dt"), (p["heading"], len(p["rows"]), b.count("<dt"))
changed_legs = {}
for sec, b in leg_now:
    if norm(b) not in old_legs:
        changed_legs.setdefault(sec, []).append(parse_legend(b))


def cols(page):
    m = re.search(r"const COLS = (\{.*?\});\n", page, flags=re.S)
    return json.loads(m.group(1))


c_old, c_now = cols(V12), cols(NOW)
changed_cols = {sec: c_now[sec] for sec in c_now if c_old.get(sec) != c_now[sec]}

# superseded: older source entries whose text pass 4 took off the page
plain_now = html.unescape(NOW)
plain_old = html.unescape(V12)
sup_tips = sorted(k for k, (t, x) in ns_t["TIPS"].items() if x in plain_old and x not in plain_now)
sup_leg = sorted(k for k, lst in ns_t["LEGENDS"].items()
                 if any(x and x in V12 for x in lst) and not all((x is None) or (x in NOW) for x in lst))
sup_p3 = sorted(sec for sec in ns_t["PASS3_FINAL"] if ns_t["PASS3_FINAL"][sec] != c_now.get(sec))
rec = {"tips": new_tips, "legends": changed_legs, "cols": changed_cols,
       "superseded": {"TIPS": sup_tips, "LEGENDS": sup_leg, "PASS3_FINAL": sup_p3}}

block = ("\n\n# ------------------------------------------------------------------------------------- pass 4, edits 1 to 3\n"
         "# Page pass 4 (the colour system, the pipeline strip and the rewritten sections): what the page now has, read from\n"
         "# the page after edit 3 (p4_e3_sources.py wrote it; ta_apply.py does not use this record). tips: every data-tip text\n"
         "# the version-12 page did not have, by section, title -> text. legends: every symbol legend whose text pass 4\n"
         "# changed, as [group heading or None, symbol, definition] rows (HTML), with its worked-example line. cols: the column\n"
         "# descriptions of every section whose descriptions pass 4 changed. superseded: the older entries above that these\n"
         "# replace (TIPS and LEGENDS record ids whose text is no longer on the page; PASS3_FINAL sections that changed).\n"
         "PASS4_EDITS_1_3 = " + pprint.pformat(rec, width=118, sort_dicts=False) + "\n")
out_t = src_t.rstrip("\n") + block
ns = {}
exec(compile(out_t, TA_TEXTS, "exec"), ns)
assert ns["PASS4_EDITS_1_3"] == rec

# ta_code.py: the one change pass 4 made to the shared tooltip script
OLD_TT = '''    const dl = k && k.querySelector("dl"), h = k && k.querySelector(".symkey-h");
    if (!dl) return;
    if (!dl.id) dl.id = "symkey-" + (keys.indexOf(k) + 1);'''
m = re.search(r"    const h = k && k\.querySelector\(\"\.symkey-h\"\);\n    let dl = k && k\.querySelector\(\"dl\"\);\n.*?"
              r"    \} else if \(!dl\.id\) dl\.id = \"symkey-\" \+ \(keys\.indexOf\(k\) \+ 1\);", NOW, re.S)
NEW_TT = m.group(0)
assert OLD_TT in V12 and OLD_TT not in NOW and NOW.count(NEW_TT) == 1
for lst in ("TT", "DR", "IW"):
    for item in ns_c[lst]:
        if item[2] in V12:
            assert item[2] in NOW, f"ta_code.{lst} {item[0]!r} changed by pass 4 and not recorded"
block_c = ("\n\n# ------------------------------------------------------------------------------------- pass 4, edit 2\n"
           "# The one change page pass 4 made to the shared tooltip script (p4_e2_apply.py, section 10; recorded by\n"
           "# p4_e3_sources.py; ta_apply.py does not use it): a formula's tooltip shows the whole legend beside it, every\n"
           "# group of a grouped legend (one template per grouped legend, its group headings in bold).\n"
           "PASS4_TT = [\n    (\"a formula's tooltip: every group of a grouped legend\",\n     r'''" + OLD_TT + "''',\n     r'''"
           + NEW_TT + "'''),\n]\n")
out_c = src_c.rstrip("\n") + block_c
nc = {}
exec(compile(out_c, TA_CODE, "exec"), nc)
assert nc["PASS4_TT"][0][1] == OLD_TT and nc["PASS4_TT"][0][2] == NEW_TT

open(TA_TEXTS, "w", encoding="utf-8", newline="").write(out_t)
open(TA_CODE, "w", encoding="utf-8", newline="").write(out_c)
print("ta_texts.py: PASS4_EDITS_1_3 recorded:",
      sum(len(v) for v in new_tips.values()), "tips in", len(new_tips), "sections;",
      sum(len(v) for v in changed_legs.values()), "legends in", len(changed_legs), "sections; COLS of", sorted(changed_cols))
print("  superseded:", json.dumps(rec["superseded"]))
print("ta_code.py: PASS4_TT recorded (1 snippet)")
