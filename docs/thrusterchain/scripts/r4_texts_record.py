"""Record edit 4's new tooltip texts in ta_texts.py (EDIT4_RESULTS), as a literal dict. Run once, after r4_apply.py.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/r4_texts_record.py'
"""
import html as H
import os
import pprint
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_cols  # noqa: E402

T = os.path.join(HERE, "ta_texts.py")
src = open(os.path.join(HERE, "r4_results_src.html"), encoding="utf-8").read()
tips = {t: d for t, d in re.findall(r'<span class="term" data-tip-title="([^"]*)" data-tip="([^"]*)">', src)}
assert len(tips) == 4, tips
page = open("/mnt/c/Users/David/dev/bca-shift-work/page/tmp/r4/applied.html", encoding="utf-8").read()
g0 = page.index("const GLOSS = [")
gl = page[g0:page.index("];", g0)]
gloss = [txt for txt in re.findall(r'"([^"]*project owner[^"]*)"', gl)]
assert len(gloss) == 4, gloss
rec = {"tips": tips,
       "cols_results": r4_cols.RESULTS_NEW,
       "cols_weighting": [r4_cols.QUALITIES, r4_cols.EXPLORE, r4_cols.MARKS],
       "gloss": gloss}
block = ("\n\n# ------------------------------------------------------------------------------------------------ edit 4 (comment D)\n"
         "# New texts added by r4_apply.py, already on the page (ta_apply.py does not use them): the tooltips of the four\n"
         "# terms of \"The question, in plain words\" in Results; the column descriptions of the six new tables of Results\n"
         "# (COLS[\"results\"] entries 0-5) and of the weighting test's new tables (COLS[\"weighting-test\"] entries 0, 1 and\n"
         "# 3; entry 2 is kept); and the four glossary notes that now name the project owner instead of the user.\n"
         "EDIT4_RESULTS = " + pprint.pformat(rec, width=118, sort_dicts=False) + "\n")
s = open(T, encoding="utf-8").read()
assert "EDIT4_RESULTS" not in s, "already recorded"
open(T, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + block)
ns = {}
exec(compile(open(T, encoding="utf-8").read(), T, "exec"), ns)
assert ns["EDIT4_RESULTS"]["tips"] == tips
print("ta_texts.py: EDIT4_RESULTS recorded,", len(tips), "tips,", len(rec["cols_results"]) + len(rec["cols_weighting"]),
      "column entries,", len(gloss), "glossary notes")
