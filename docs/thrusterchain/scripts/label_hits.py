"""List every occurrence of cryptic labels in the extracted page text (read-only).

Reads notes/page_text.txt (from extract_text.py) and notes/js_strings.txt, splits
each block into sentences, and writes notes/label_hits.txt grouped by category:
section id | nearest element id | the sentence around the match.
"""
import re
from collections import OrderedDict

BASE = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/"
TEXT = BASE + "page_text.txt"
OUT = BASE + "label_hits.txt"

CATS = OrderedDict([
    ("H1-H4 harm checks", r"\bH[1-4]\b"),
    ("T1-T9 world-card marks", r"\bT[1-9]\b"),
    ("D1-D3 detection measures", r"\bD[1-3]\b"),
    ("C1-C3 actor checks", r"\bC[1-3]\b"),
    ("other code letters (P2-P4, B1, A1, V#)", r"\b(?:P[2-4]|B1|A1|V[0-9])\b"),
    ("stage codes", r"\bstage ?0\w*"),
    ("the gate", r"\bgate(?:'s|s)?\b"),
    ("arm / arms", r"\barms?\b"),
    ("room / room-certified / certified", r"\broom\b|room-certified|\bcertif\w*"),
    ("level 1 / level 2", r"\blevels? [12]\b|\blevel-1\b"),
    ("numbered verdicts", r"\bverdicts? [0-9]|\bverdicts \d and \d"),
    ("gapped / gap", r"\bgapped\b|\bgap\b"),
    ("full-data / full-log critic", r"full-data critic|full-log critic|full critic"),
    ("world card / shift card / marks", r"world card|shift card|shift-card|\bmarks?\b"),
    ("the design (as a noun) / design seeds / scout", r"\bthe design\b|\bdesign's\b|design seeds?|\bscout\b"),
    ("seed families", r"gate seeds|confirmation seeds|data seed|design seed"),
    ("numbered settings / claims / columns / measures", r"\bSetting [123]\b|\bClaim [12]\b|\bcolumns? \d\b|\bmeasure \d\b"),
    ("spec / protocol / the user", r"\b[Ss]pec\b|\bprotocol\b|\bthe user\b|\bthe user's\b"),
    ("placement nicknames", r"harm-placed|over-preference-placed|cap-2|[Ff]actors up to 2|strength-matched|amount-matched|Reversed|stand-ins?|\bbend(?:ing)? critic|edge-slope"),
    ("acronyms", r"\bBCA\b|TD3\+BC|\bTD3\b|\bBC\b|D4RL|WBCP|\bFQE\b|\bLQ\b|\bLQR\b|SHA-256|\bUTC\b|\bCPUs?\b|CPU-hours|\bJSON\b|hdf5|MuJoCo|\bAdam\b|\btanh\b|Holm|Student's t|bootstrap|Kish|float32|float64|int32|int8|\bbool\b|\bh\b"),
    ("jargon phrases", r"[Pp]olicy optimisation|co-trained|[Rr]ank correlation|wall clock|\bcores\b|\bdensity\b|classifier|commit [0-9a-f]{7}|build checks|[Ff]easibility"),
    ("time-step wording", r"[Tt]ime step"),
])


def sentences(t):
    # split on sentence ends, keep math markers intact
    parts = re.split(r"(?<=[.!?:;])\s+(?=[A-Z0-9(“\"])", t)
    return [p.strip() for p in parts if p.strip()]


def main():
    rows = []
    for line in open(TEXT, encoding="utf-8"):
        m = re.match(r"\[([^|]*) \| ([^\]]*)\] (.*)", line.rstrip("\n"))
        if m:
            rows.append(m.groups())
    out = []
    for cat, pat in CATS.items():
        rx = re.compile(pat)
        hits = []
        for sec, near, t in rows:
            for s in sentences(t):
                if rx.search(s):
                    found = sorted(set(x.group(0) for x in rx.finditer(s)))
                    if len(s) > 420:
                        k = rx.search(s).start()
                        s = ("…" if k > 180 else "") + s[max(0, k - 180): k + 240] + "…"
                    hits.append(f"  [{sec} | {near or '-'}] ({', '.join(found)}) {s}")
        out.append(f"## {cat}: {len(hits)} sentences")
        out.extend(hits)
        out.append("")
    open(OUT, "w", encoding="utf-8").write("\n".join(out))
    for l in out:
        if l.startswith("## "):
            print(l)


main()
