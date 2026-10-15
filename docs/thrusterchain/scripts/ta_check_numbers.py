"""Check that every number in the new tooltip texts (ta_texts.py, and the notes in ta_code.py) appears in version 10 of
the page (its text, tables, formula labels, scripts or data). Prints any number it cannot find. Read-only.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/ta_check_numbers.py'
"""
import html as H
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ta_code as C  # noqa: E402
import ta_texts as T  # noqa: E402

V10 = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v10.html"
NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?")


def main():
    page = open(V10, encoding="utf-8").read()
    hay = H.unescape(page).replace("−", "-")
    hay = re.sub(r"\\,|\\ ", " ", hay)                 # LaTeX thin spaces in formula labels
    items = [(f"tip {k}", (t or "") + " " + x) for k, (t, x) in T.TIPS.items()]
    items += [(f"column {i}", new) for i, (_, new) in enumerate(T.COLS_NEW)]
    items += [(f"legend {k}", re.sub(r"<[^>]+>", "", d)) for k, v in T.LEGENDS.items() for d in v if d]
    items += [(f"guide {k}", lab + " " + txt) for k, v in T.GUIDE.items() for lab, txt in v]
    gl = re.findall(r'\[/.*?/[a-z]*, "(.*?)"', [x for x in C.TT if x[0].startswith("plain()")][0][2])
    items += [("note", g) for g in gl]
    missing, total = [], 0
    for where, t in items:
        for m in NUM.finditer(t):
            n = m.group(0).replace("−", "-").rstrip(",")
            total += 1
            bare = n.lstrip("-")
            if len(bare.replace(",", "").replace(".", "")) <= 1:
                continue
            if not re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", hay):
                if n.startswith("-") and re.search(r"(?<![\d.])" + re.escape(bare) + r"(?![\d])", hay):
                    continue
                missing.append((where, n, t[:110]))
    print(f"{len(items)} texts, {total} numbers checked against version 10; {len(missing)} not found")
    for w, n, t in missing:
        print(f"  {n!r:>12}  {w}: {t}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
