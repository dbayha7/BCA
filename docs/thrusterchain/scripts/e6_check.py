"""Check that every number in the texts edit 2 adds (e6_content.py) appears in the page as it was before the edit
(its text, its tables, its formulas' labels or its data payload). Prints any number it cannot find.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/e6_check.py'
"""
import html as H
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e6_content as C  # noqa: E402

BEFORE = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e5/thrusterchain.pre-e5.html"   # fallback
PAGE_BEFORE = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"


def texts():
    out = []
    def add(where, *xs):
        for x in xs:
            out.append((where, re.sub(r"<[^>]+>", "", H.unescape(x))))
    for d, name in ((C.SIT_TIPS, "situation"), (C.WORLD_TIPS, "world"), (C.READOUT_TIPS, "readout"),
                    (C.TRANSPORT_TIPS, "transport"), (C.TD_TRANSPORT_TIPS, "demo transport"),
                    (C.TD_READOUT_TIPS, "demo readout"), (C.T, "term"), (C.SYM, "symbol"), (C.SYM_SECTION, "symbol"),
                    (C.IW_TIPS, "weighting")):
        for k, v in d.items():
            add(f"{name} {k}", *v)
    for row in C.CTRL_PANEL:
        add(f"controller {row[0]}", *row[1:])
    for row in C.CTRL_OTHER + C.CTRL_LEARNED:
        add(f"controller {row[0]}", *[x for x in row if x != "fixed"])
    add("canvas", *C.CANVAS_CHANNEL_TIP, *C.CANVAS_COST_TIP)
    for k, v in C.COST_LEGEND_TIPS.items():
        add("cost legend", k, v)
    add("lead-in", C.LEAD_BULLET)
    add("worked example", C.WORKED_NEW)
    for anchor, heading, items in C.LEGENDS:
        for sym, text in items:
            add(f"legend {heading} / {re.sub('<[^>]+>', '', sym)}", sym, text)
    for suffix, tip in C.SVG_TIPS:
        add("svg", *tip)
    return out


NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?")


def main():
    page = open(PAGE_BEFORE, encoding="utf-8").read()
    hay = H.unescape(page).replace("−", "-").replace("&#8722;", "-")
    hay = re.sub(r"\\,|\\ ", " ", hay)
    missing, total = [], 0
    for where, t in texts():
        for m in NUM.finditer(t):
            n = m.group(0).replace("−", "-").rstrip(",")
            total += 1
            bare = n.lstrip("-")
            if len(bare.replace(",", "").replace(".", "")) <= 1:
                continue                                           # single digits: 0 to 9
            pat = r"(?<![\d.])" + re.escape(n) + r"(?![\d])"
            if not re.search(pat, hay):
                # a negative number may be written with a leading minus or as "minus" plus the number
                if n.startswith("-") and re.search(r"(?<![\d.])" + re.escape(bare) + r"(?![\d])", hay):
                    continue
                missing.append((where, n, t[:110]))
    print(f"{total} numbers checked; {len(missing)} not found in {os.path.basename(PAGE_BEFORE)}")
    for w, n, t in missing:
        print(f"  {n!r:>12}  {w}: {t}")


if __name__ == "__main__":
    main()
