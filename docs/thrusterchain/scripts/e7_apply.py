"""Fixes from the review of the tooltip pass (comments 6 to 10). Exact, asserted replacements on thrusterchain.html:
each old string is matched exactly once; an old string that occurs n times is replaced at each place through a context
string that is itself matched exactly once. Never touches thrusterchain.v9.html. Saves the page as it was before
this edit to tmp/e7/thrusterchain.pre-e7.html. Run:
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/e7_apply.py'
Add --dry to write tmp/e7/dry.html instead of the page, --counts to only count each old text.
The texts and code are in e7_texts.py.
"""
import hashlib
import html as H
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e7_texts import JS1, JS2, CSS, TEXTS, COLS_TEXTS  # noqa: E402

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v9.html"
TMP = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e7"
LOG = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def esc(x):
    return H.escape(x, quote=True)


def once(s, old, new, what):
    n = s.count(old)
    assert n == 1, f"{what}: expected exactly 1 match, found {n}: {old[:100]!r}"
    LOG.append(what)
    return s.replace(old, new, 1)


def each(s, old, new, n, what):
    """Replace all n places of old, each through a context string matched exactly once."""
    found = s.count(old)
    assert found == n, f"{what}: expected {n} matches, found {found}: {old[:100]!r}"
    if n == 1:
        return once(s, old, new, what)
    pos = []
    i = s.find(old)
    while i >= 0:
        pos.append(i)
        i = s.find(old, i + len(old))
    for a in reversed(pos):
        b = a + len(old)
        k = 30
        while True:
            lo, hi = max(0, a - k), min(len(s), b + k)
            ctx = s[lo:hi]
            if s.count(ctx) == 1:
                break
            k *= 2
            assert k < 40000, f"{what}: no unique context"
        s = once(s, ctx, s[lo:a] + new + s[b:hi], f"{what} @{a}")
    return s


def main():
    dry = "--dry" in sys.argv
    b0 = md5(BACKUP)
    s = open(PAGE, encoding="utf-8").read()
    assert "firstOfItsKind" not in s and "pt-hits" not in s, "the review fixes are already applied"
    os.makedirs(TMP, exist_ok=True)
    if "--counts" in sys.argv:
        for what, old, new, mode, n in TEXTS:
            o = esc(old) if mode == "esc" else old
            print(f"{s.count(o):3d}  (expected {n})  {what}")
        for what, old, new, n in COLS_TEXTS:
            print(f"{s.count(old):3d}  (expected {n})  {what}")
        for what, old, new in JS1 + JS2:
            print(f"{s.count(old):3d}  (expected 1)  {what}")
        return
    if not dry:
        open(os.path.join(TMP, "thrusterchain.pre-e7.html"), "w", encoding="utf-8").write(s)

    # the two scripts: each snippet must sit inside its own script
    for what, old, new in JS1:
        i1, i2 = s.index("/* Tooltips: one shared tooltip"), s.index("/* e6_tooltips */")
        assert s.count(old) == 1 and i1 < s.index(old) < i2, f"{what}: not once inside the tooltip script"
        s = once(s, old, new, "js1: " + what)
    i2 = s.index("/* e6_tooltips */")
    for what, old, new in JS2:
        assert s.count(old) == 1 and s.index(old) > i2, f"{what}: not once inside the drawings script"
        s = once(s, old, new, "js2: " + what)
    s = once(s, "</style>", CSS + "</style>", "css")

    for what, old, new, mode, n in TEXTS:
        o, w = (esc(old), esc(new)) if mode == "esc" else (old, new)
        if n is None:                      # a symbol's tooltip: as many places as the symbol has
            n = s.count(o)
            assert n >= 1, f"{what}: not found"
        s = each(s, o, w, n, what)

    # column descriptions: inside the one-line COLS object, which must then equal e5_cols.py's
    head = "  const COLS = "
    a = s.index(head)
    line = s[a:s.index(";\n", a)]
    for what, old, new, n in COLS_TEXTS:
        assert line.count(old) == n == s.count(old), f"{what}: expected {n}, found {line.count(old)}"
    for what, old, new, n in COLS_TEXTS:
        s = each(s, old, new, n, what)
    a = s.index(head)
    z = s.index(";\n", a)
    from e5_cols import COLS
    page_cols = json.loads(s[a + len(head):z].replace("<\\/", "</"))
    assert page_cols == json.loads(json.dumps(COLS)), "the page's column descriptions differ from e5_cols.py"

    out = os.path.join(TMP, "dry.html") if dry else PAGE
    open(out, "w", encoding="utf-8").write(s)
    assert md5(BACKUP) == b0, "the backup changed"
    print(f"{'dry run' if dry else 'applied'}: {len(s):,} bytes -> {out}; md5 {md5(out)}; backup untouched ({b0})")
    print(f"{len(LOG)} exact replacements")


if __name__ == "__main__":
    main()
