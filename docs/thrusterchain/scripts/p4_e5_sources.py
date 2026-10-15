#!/usr/bin/env python3
"""Page pass 4, edit 5: bring the tooltip text sources in line with the page (run once, after p4_e5_apply.py).

ta_texts.py gets PASS4_EDIT_5, a record (ta_apply.py does not use it) read from the page, against the edit-4 page:
    tips     every data-tip text on the page that the edit-4 page did not have, by section, title -> text;
    cols     the column descriptions (COLS) of every section whose descriptions edit 5 changed, as the page has them;
    legends  every symbol-legend definition (dd) that edit 5 rewrote, by section: old text -> new text.
ta_code.py (the shared tooltip script's code) is unchanged: edit 5 changed only the COLS data inside that script.

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && python3 scripts/p4_e5_sources.py'
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
E4 = open(os.path.join(PAGE_DIR, "tmp", "p4", "edit5.before.html"), encoding="utf-8").read()
assert hashlib.md5(E4.encode("utf-8")).hexdigest() == "e24913a0187b041b7a279b10ed4b8a65", "not the edit-4 page"
TA_TEXTS = os.path.join(HERE, "ta_texts.py")
src_t = open(TA_TEXTS, encoding="utf-8").read()
assert "PASS4_EDIT_5" not in src_t, "already recorded"
BK = os.path.join(PAGE_DIR, "tmp", "p4", "edit5", "ta_texts.before.py")
shutil.copyfile(TA_TEXTS, BK)


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


old_texts = {x[2] for x in tips(E4)}
new_tips = {}
for sec, title, text in tips(NOW):
    if text not in old_texts:
        new_tips.setdefault(sec, {})[title or "(no title)"] = text


def cols(page):
    m = re.search(r"const COLS = (\{.*?\});\n", page, flags=re.S)
    return json.loads(m.group(1))


c_old, c_now = cols(E4), cols(NOW)
changed_cols = {sec: c_now[sec] for sec in c_now if c_old.get(sec) != c_now[sec]}

CONT = re.compile(r'<mjx-container[^>]*aria-label="(.*?)"[^>]*>.*?</mjx-container>', re.S)


def dds(page):
    secs, out = sections(page), []
    for m in re.finditer(r'<div class="symkey"[^>]*>.*?</dl>\s*(?:<p class="symkey-ex">.*?</p>\s*)?</div>', page, flags=re.S):
        for d in re.findall(r"<dd>(.*?)</dd>", m.group(0), flags=re.S):
            out.append((sec_at(secs, m.start()), html.unescape(re.sub(r"<[^>]+>", "", CONT.sub(r"\1", d)))))
    return out


d_old, d_now = dds(E4), dds(NOW)
assert len(d_old) == len(d_now), (len(d_old), len(d_now))
legends = {}
for (s0, a), (s1, b) in zip(d_old, d_now):
    if a != b:
        legends.setdefault(s1, []).append([a, b])
rec = {"tips": new_tips, "cols": changed_cols, "legends": legends}
block = ("\n\n# ------------------------------------------------------------------------------------- pass 4, edit 5\n"
         "# Page pass 4, edit 5 (the final review's findings): what the page now has, read from the page after edit 5\n"
         "# (p4_e5_sources.py wrote it; ta_apply.py does not use this record). tips: every data-tip text the edit-4 page did\n"
         "# not have, by section, title -> text. cols: the column descriptions of every section whose descriptions edit 5\n"
         "# changed (histogram rows are bins or ranges, not bands; the critics are named full-recording critic and critic\n"
         "# trained after the removal). legends: every legend definition edit 5 rewrote as a complete sentence, old -> new.\n"
         "PASS4_EDIT_5 = " + pprint.pformat(rec, width=118, sort_dicts=False) + "\n")
out_t = src_t.rstrip("\n") + block
ns = {}
exec(compile(out_t, TA_TEXTS, "exec"), ns)
assert ns["PASS4_EDIT_5"] == rec
open(TA_TEXTS, "w", encoding="utf-8", newline="").write(out_t)
print("ta_texts.py: PASS4_EDIT_5 recorded:", sum(len(v) for v in new_tips.values()), "tips in", sorted(new_tips),
      "; COLS of", sorted(changed_cols), "; legend definitions:", sum(len(v) for v in legends.values()), "in", sorted(legends))
print("  copy of the old ta_texts.py:", BK)
