"""Reviewer: embedded data, notation in the rendered text, and a few report claims. Read only; prints only."""
import json
import re
from pathlib import Path

import numpy as np

QD = Path("/mnt/c/Users/David/dev/bca-shift-work/qviewer")
page = (QD / "test_page.html").read_text(encoding="utf-8")
qv = (QD / "qv_data.json").read_text(encoding="utf-8")
m = re.findall(r'<script[^>]*id="qv-data"[^>]*>(.*?)</script>', page, re.S)
print("qv-data tags:", len(m), "| equals qv_data.json:", bool(m) and json.loads(m[0]) == json.loads(qv),
      "| byte-identical:", bool(m) and m[0].strip() == qv.strip())
sec = (QD / "section.html").read_text(encoding="utf-8").strip()
print("section.html inside test_page:", sec in page)
print("order: critic-in-action before methods:", page.find('id="critic-in-action"') < page.find('id="methods"'))

T = json.loads(Path("/root/bca-shift/runs/qviewer/review/page_text.json").read_text(encoding="utf-8"))
blob = json.dumps(T, ensure_ascii=False)
for pat in [r"a\*", r"a\^\*", "favou?rite", r"Q\^pi", "a-hat", r"\bw1\b", r"\bw2\b", "NaN", "undefined", "Infinity"]:
    hits = re.findall(".{0,50}" + pat + ".{0,50}", blob, re.I)
    print(f"rendered text /{pat}/: {len(hits)}", hits[:3])

Q7 = json.loads(Path("/root/bca-shift/runs/qviewer/precise/q_data.json").read_text())
Q4 = json.loads(Path("/root/bca-shift/runs/qviewer/q_data.json").read_text())
for s in Q7["summaries"]:
    q, c, sit, d = s
    if sit == "strong":
        print(q, c, "rise-while-fall", d["critic_rises_while_truth_falls_states"],
              "push there", round(d["push_mean_where_critic_rises_while_truth_falls"] or float("nan"), 3),
              "elsewhere", round(d["push_mean_elsewhere"], 3),
              "steps", d["critic_rises_while_truth_falls_steps"] if (q, c) == ("expert", "gap") else "")

# "rounding was more than 2% of the curve's spread at up to 86 of 100 states" (4-digit file)
F = {f: i for i, f in enumerate(Q4["episode_fields"])}
worst = 0
for e4 in Q4["episodes"]:
    n = 0
    for key in ("q_sweep", "qpi_sweep"):
        pass
    cnt = 0
    for t in range(100):
        bad = False
        for key in ("q_sweep", "qpi_sweep"):
            v = np.asarray(e4[F[key]][t])
            spread = v.max() - v.min()
            mag = np.max(np.abs(v))
            unit = 10 ** (np.floor(np.log10(mag)) - 3)
            if unit > 0.02 * spread:
                bad = True
        cnt += bad
    worst = max(worst, cnt)
print("max states per episode where a 4-digit step exceeds 2% of a curve's spread:", worst)
