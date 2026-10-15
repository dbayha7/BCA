"""Check that every number in the texts the review fixes add (e7_texts.py, and the Play/Pause text) appears in the
page as it was before the fixes (its text, tables, formula labels or data payload). Prints any number it cannot find.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/e7_check.py'
"""
import html as H
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e7_texts import TEXTS, COLS_TEXTS  # noqa: E402

BEFORE = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e7/thrusterchain.pre-e7.html"
NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?")


def main():
    page = open(BEFORE, encoding="utf-8").read()
    hay = H.unescape(page).replace("−", "-")
    hay = re.sub(r"\\,|\\ ", " ", hay)
    items = [(w, re.sub(r"<[^>]+>", "", H.unescape(new))) for w, old, new, mode, n in TEXTS]
    items += [(w, new) for w, old, new, n in COLS_TEXTS]
    items.append(("Play button", "Pause the episode on the state on screen."))
    missing, total = [], 0
    for where, t in items:
        for m in NUM.finditer(t):
            n = m.group(0).replace("−", "-").rstrip(",")
            total += 1
            bare = n.lstrip("-")
            if len(bare.replace(",", "").replace(".", "")) <= 1:
                continue                                           # single digits: 0 to 9
            if not re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", hay):
                if n.startswith("-") and re.search(r"(?<![\d.])" + re.escape(bare) + r"(?![\d])", hay):
                    continue
                missing.append((where, n, t[:110]))
    print(f"{total} numbers checked; {len(missing)} not found in {os.path.basename(BEFORE)}")
    for w, n, t in missing:
        print(f"  {n!r:>12}  {w}: {t}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
