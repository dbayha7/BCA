#!/usr/bin/env python3
"""Page pass 4, edit 4: bring the tooltip text sources in line with the page (run once, after p4_e4_apply.py).

ta_texts.py gets PASS4_EDIT_4, a record (ta_apply.py does not use it) read from the page, against the edit-3 page:
    tips        every data-tip text on the page that the edit-3 page did not have, by section, title -> text;
    cols        the column descriptions (COLS) of every section whose descriptions edit 4 changed, as the page has
                them (Results, the weighting test, and the critic section, which had none);
    superseded  the record ids of older entries whose text edit 4 took off the page: TIPS texts, and the
                PASS3_FINAL and PASS4_EDITS_1_3 sections whose COLS changed.
Edit 4 changed no symbol legend and no part of the shared tooltip script, so ta_code.py is unchanged (checked).

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && python3 scripts/p4_e4_sources.py'
"""
import hashlib
import html
import json
import os
import pprint
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.dirname(HERE)
NOW = open(os.path.join(PAGE_DIR, "thrusterchain.html"), encoding="utf-8").read()
E3 = open(os.path.join(PAGE_DIR, "tmp", "p4", "edit4.before.html"), encoding="utf-8").read()
assert hashlib.md5(E3.encode("utf-8")).hexdigest() == "f376b41c008320ad4881a0e68beef8e6", "not the edit-3 page"
E4_MD5 = "e24913a0187b041b7a279b10ed4b8a65"
assert hashlib.md5(NOW.encode("utf-8")).hexdigest() == E4_MD5, "the page is not the edit-4 page"
TA_TEXTS = os.path.join(HERE, "ta_texts.py")
TA_CODE = os.path.join(HERE, "ta_code.py")
src_t = open(TA_TEXTS, encoding="utf-8").read()
assert "PASS4_EDIT_4" not in src_t, "already recorded"
BK = os.path.join(PAGE_DIR, "tmp", "p4", "edit4", "ta_texts.before.py")
shutil.copyfile(TA_TEXTS, BK)
ns_t = {}
exec(compile(src_t, TA_TEXTS, "exec"), ns_t)


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


old_texts = {x[2] for x in tips(E3)}
new_tips = {}
for sec, title, text in tips(NOW):
    if text not in old_texts:
        new_tips.setdefault(sec, {})[title or "(no title)"] = text
assert sum(len(v) for v in new_tips.values()) == 3, new_tips


def cols(page):
    m = re.search(r"const COLS = (\{.*?\});\n", page, flags=re.S)
    return json.loads(m.group(1))


c_old, c_now = cols(E3), cols(NOW)
changed_cols = {sec: c_now[sec] for sec in c_now if c_old.get(sec) != c_now[sec]}
assert sorted(changed_cols) == ["critic-in-action", "results", "weighting-test"], sorted(changed_cols)

# legends and the tooltip script: untouched by edit 4
LEG = re.compile(r'<div class="symkey"[^>]*>.*?</dl>\s*(?:<p class="symkey-ex">.*?</p>\s*)?</div>', re.S)
norm = lambda x: re.sub(r"\s+", " ", x)
assert sorted(norm(b) for b in LEG.findall(E3)) == sorted(norm(b) for b in LEG.findall(NOW)), "a legend changed"
ns_c = {}
exec(compile(open(TA_CODE, encoding="utf-8").read(), TA_CODE, "exec"), ns_c)
for lst in ("TT", "DR", "IW"):
    for item in ns_c[lst]:
        if item[2] in E3:
            assert item[2] in NOW, f"ta_code.{lst} {item[0]!r} changed by edit 4"
for item in ns_c["PASS4_TT"]:
    assert item[2] in NOW

plain_now, plain_old = html.unescape(NOW), html.unescape(E3)
sup_tips = sorted(k for k, (t, x) in ns_t["TIPS"].items() if x in plain_old and x not in plain_now)
sup_p3 = sorted(sec for sec in ns_t["PASS3_FINAL"] if sec in changed_cols and ns_t["PASS3_FINAL"][sec] != c_now.get(sec))
p4 = ns_t["PASS4_EDITS_1_3"]
sup_p4_tips = sorted(f"{sec} / {title}" for sec, d in p4["tips"].items() for title, x in d.items() if x not in plain_now)
sup_p4_cols = sorted(sec for sec in p4["cols"] if sec in changed_cols)
rec = {"tips": new_tips, "cols": changed_cols,
       "superseded": {"TIPS": sup_tips, "PASS3_FINAL": sup_p3, "PASS4_EDITS_1_3": {"tips": sup_p4_tips, "cols": sup_p4_cols}}}

block = ("\n\n# ------------------------------------------------------------------------------------- pass 4, edit 4\n"
         "# Page pass 4, edit 4 (Results, the weighting test's exploratory results, the critic section's column\n"
         "# descriptions): what the page now has, read from the page after edit 4 (p4_e4_sources.py wrote it; ta_apply.py\n"
         "# does not use this record). tips: every data-tip text the edit-3 page did not have, by section, title -> text.\n"
         "# cols: the column descriptions of every section whose descriptions edit 4 changed. superseded: the older entries\n"
         "# above whose text edit 4 took off the page (TIPS ids; PASS3_FINAL sections; PASS4_EDITS_1_3 tips and sections).\n"
         "# Edit 4 changed no symbol legend and nothing in the shared tooltip script (ta_code.py holds no edit-4 change).\n"
         "PASS4_EDIT_4 = " + pprint.pformat(rec, width=118, sort_dicts=False) + "\n")
out_t = src_t.rstrip("\n") + block
ns = {}
exec(compile(out_t, TA_TEXTS, "exec"), ns)
assert ns["PASS4_EDIT_4"] == rec
open(TA_TEXTS, "w", encoding="utf-8", newline="").write(out_t)
print("ta_texts.py: PASS4_EDIT_4 recorded:", sum(len(v) for v in new_tips.values()), "tips in", sorted(new_tips),
      "; COLS of", sorted(changed_cols))
print("  superseded:", json.dumps(rec["superseded"]))
print("  ta_code.py: unchanged (no edit-4 change to the tooltip script); copy of the old ta_texts.py:", BK)
