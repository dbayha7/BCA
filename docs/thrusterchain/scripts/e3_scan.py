"""Prove the plain-language pass: extract reader-visible text from a page (tags stripped, math as its label, script
and style skipped) plus the string literals of its scripts, and list every remaining short code or nickname.

usage: python3 e3_scan.py PAGE OUT_TEXT
Short codes (H1-H4, T1-T9, D1-D3, C1-C3, V-codes, P2-P4, B1, A1, stage0) are allowed only inside the section
#code-names, where they map plain names to the code. Exit status 1 if any code is found elsewhere.
"""
import re
import subprocess
import sys

page, out = sys.argv[1], sys.argv[2]
js_out = out.replace(".txt", "_js.txt")
subprocess.run([sys.executable, "/mnt/c/Users/David/dev/bca-shift-work/page/scripts/e3_extract.py", page, out, js_out],
               check=True)
rows = []
for line in open(out, encoding="utf-8"):
    m = re.match(r"\[([^|]*) \| ([^\]]*)\] (.*)", line.rstrip("\n"))
    if m:
        rows.append(m.groups())
js = [l.rstrip("\n") for l in open(js_out, encoding="utf-8")]

CODES = re.compile(r"\b(?:H[1-4]|T[1-9]|D[1-3]|C[1-3]|V[0-9]|P[2-4]|B1|A1)\b|\bstage ?0\w*")
NICK = re.compile(
    r"\bthe gate\b|\bgate(?:'s| seeds| report)\b|\broom\b|room-certified|\bcertif\w*|\barms?\b|\bgapped\b|"
    r"full-data|full critic|world card|shift.card|\bladder\b|\bretune\b|cap-2|[Ff]actors up to 2|edge.slope|"
    r"[Ee]ven caution|\bEven ×|strength-matched even|\bSpec\b|\bscout\b|\bClaim [12]\b|\bthe user\b|\bprotocol\b|"
    r"\bthe design\b(?! plan)|\bdesign's\b|[Tt]ime step|start law|calibration bank|\bthe gap\b|\bCPUs?\b|wall clock|"
    r"\b\d+(?:\.\d+)? h\b|Holm family|\bLQ\b(?! is short)|[Cc]onfirmation seeds?|\blevel [12]\b|\bLevels? [12]\b|"
    r"[Vv]erdicts? \d|\bcolumn \d|\bmeasure \d|over-preference-placed|[Pp]olicy optimisation|not gated|build checks|"
    r"thruster-steps stalled")

bad = 0
print("== short codes in reader-visible text")
for sec, near, t in rows:
    for m in CODES.finditer(t):
        where = "allowed (code-names mapping)" if sec == "code-names" else "NOT ALLOWED"
        if sec != "code-names":
            bad += 1
        print(f"  [{sec} | {near}] {m.group(0)!r} {where}: …{t[max(0, m.start() - 60):m.end() + 60]}…")
print("== short codes in script strings")
for l in js:
    for m in CODES.finditer(l):
        bad += 1
        print(f"  {l[:160]}")
print("== nicknames and shorthand (outside code-names)")
nick = 0
for sec, near, t in rows:
    if sec == "code-names":
        continue
    for m in NICK.finditer(t):
        nick += 1
        print(f"  [{sec} | {near}] {m.group(0)!r}: …{t[max(0, m.start() - 70):m.end() + 70]}…")
for l in js:
    if "skipped" in l:
        continue
    for m in NICK.finditer(l):
        nick += 1
        print(f"  [script] {m.group(0)!r}: {l[:160]}")
print(f"short codes outside code-names: {bad}; nickname or shorthand hits: {nick}")
sys.exit(1 if bad else 0)
