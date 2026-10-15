"""Tooltip text pass: apply ta_texts.py and ta_code.py to thrusterchain.html with exact, asserted replacements.

Every replacement names its old text exactly and asserts how many times it occurs (a data-tip attribute as many times
as the inventory counted it, everything else exactly once, inside the script or block it belongs to). Never touches
thrusterchain.v10.html (checked by md5 before and after). Saves the page as it was before to tmp/ta/thrusterchain.pre-ta.html.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/ta_apply.py'
Add --dry to write tmp/ta/dry.html instead of the page. Add --redo to rebuild the page from version 10 (read from the
backup, which is only read) when the page already holds an earlier run of this pass and nothing else.
"""
import hashlib
import html as H
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ta_code as C  # noqa: E402
import ta_texts as T  # noqa: E402

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
BACKUP = os.path.join(PAGE_DIR, "thrusterchain.v10.html")
INV = os.path.join(PAGE_DIR, "notes", "tooltips_inventory.json")
TMP = os.path.join(PAGE_DIR, "tmp", "ta")
V10_MD5 = "adf3837e6d032f77f6c8bf56bf0221a8"
LOG = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def esc(x):
    return H.escape(x, quote=True)


def once(s, old, new, what, lo=0, hi=None):
    """Replace old by new; old must occur exactly once in s[lo:hi] and exactly once in s."""
    hi = len(s) if hi is None else hi
    n_in, n_all = s[lo:hi].count(old), s.count(old)
    assert n_in == 1 and n_all == 1, f"{what}: expected exactly 1 match (in its block {n_in}, in the page {n_all}): {old[:120]!r}"
    LOG.append(what)
    return s.replace(old, new, 1)


# ------------------------------------------------------------------------------------------------ legends
SUPM = dict(zip("0123456789+−-=()ntiTk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿᵗⁱᵀᵏ"))
SUBM = dict(zip("0123456789+−-=()aeijkmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₐₑᵢⱼₖₘₙₒₚᵣₛₜᵤᵥₓ"))
SUBM[","] = ","
SUPM.update(zip("abcdefghijklmnoprstuvwxyz,", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ'"))


def html_text(h):
    """The inventory's plain text of a legend's HTML (the same function as tt_audit.py)."""
    def conv(m, table, mark):
        inner = H.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        if inner and all(ch in table for ch in inner):
            return "".join(table[ch] for ch in inner)
        return mark + inner
    h = re.sub(r"<sub>(.*?)</sub>", lambda m: conv(m, SUBM, "_"), h, flags=re.S)
    h = re.sub(r"<sup>(.*?)</sup>", lambda m: conv(m, SUPM, "^"), h, flags=re.S)
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", h))).strip()


DL = re.compile(r'<div class="symkey[^"]*"><p class="symkey-h">[^<]*</p><dl>(.*?)</dl>', re.S)
PAIR = re.compile(r"<div><dt>(.*?)</dt><dd>(.*?)</dd></div>", re.S)


def apply_legends(s, recs):
    blocks = [(m.start(1), m.end(1), m.group(1)) for m in DL.finditer(s)]
    for rid, news in T.LEGENDS.items():
        want = [tuple(e) for e in recs[rid]["entries"]]
        hits = [b for b in blocks if [(html_text(a), html_text(d)) for a, d in PAIR.findall(b[2])] == want]
        assert len(hits) == 1, f"legend {rid}: {len(hits)} blocks match its entries"
        a, z, inner = hits[0]
        pairs = PAIR.findall(inner)
        assert len(pairs) == len(news), f"legend {rid}: {len(pairs)} entries, {len(news)} new texts"
        new_inner = inner
        for (dt, dd), nd in zip(pairs, news):
            if nd is None:
                continue
            old = f"<div><dt>{dt}</dt><dd>{dd}</dd></div>"
            assert new_inner.count(old) == 1, f"legend {rid}: entry {dt!r} not once"
            new_inner = new_inner.replace(old, f"<div><dt>{dt}</dt><dd>{nd}</dd></div>", 1)
        old_block = "<dl>" + inner + "</dl>"
        s = once(s, old_block, "<dl>" + new_inner + "</dl>", f"legend {rid}")
        blocks = [(m.start(1), m.end(1), m.group(1)) for m in DL.finditer(s)]
    return s


# ------------------------------------------------------------------------------------------------ main
def main():
    dry = "--dry" in sys.argv
    assert md5(BACKUP) == V10_MD5, "the v10 backup is not the published version 10"
    s = open(PAGE, encoding="utf-8").read()
    if "--redo" in sys.argv:
        last = os.path.join(TMP, "last_applied.md5")
        assert os.path.exists(last) and open(last).read().strip() == hashlib.md5(s.encode("utf-8")).hexdigest(), \
            "--redo: the page is not the output of the last run of this pass (it may hold other edits)"
        s = open(BACKUP, encoding="utf-8").read()
    assert "tt-note" not in s and "function fit()" not in s, "the tooltip text pass is already applied"
    assert hashlib.md5(s.encode("utf-8")).hexdigest() == V10_MD5, "the page is not version 10"
    os.makedirs(TMP, exist_ok=True)
    if not dry:
        open(os.path.join(TMP, "thrusterchain.pre-ta.html"), "w", encoding="utf-8").write(s)
    recs = {r["id"]: r for r in json.load(open(INV, encoding="utf-8"))["records"]}

    # 1. data-tip texts: the whole attribute (with its heading when it has one), every place it occurs
    for rid, (new_title, new_text) in T.TIPS.items():
        r = recs[rid]
        t, x = r["title"], r["text"]
        if t:
            old = f'data-tip-title="{esc(t)}" data-tip="{esc(x)}"'
            new = f'data-tip-title="{esc(new_title or t)}" data-tip="{esc(new_text)}"'
        else:
            assert new_title is None, rid
            old, new = f'data-tip="{esc(x)}"', f'data-tip="{esc(new_text)}"'
        n = s.count(old)
        assert n == r["count"], f"{rid}: found {n}, the inventory counted {r['count']}: {old[:120]!r}"
        s = s.replace(old, new)
        LOG.append(f"tip {rid} x{n}")

    # 2. formula legends (printed beside each formula, and the formula's tooltip)
    s = apply_legends(s, recs)

    # 3. the controller guide's four tooltips
    for tid, parts in T.GUIDE.items():
        m = re.search(r'<template id="' + re.escape(tid) + r'">(.*?)</template>', s, re.S)
        assert m, tid
        new = " ".join(f"<p><b>{esc(lab)}</b> {esc(txt)}</p>" for lab, txt in parts)
        s = once(s, m.group(0), f'<template id="{tid}">{new}</template>', f"guide {tid}")

    # 4. chart bands: each point's <title> in the task chart
    titles = [m for m in re.finditer(r"<title>(water&#x27;s push [^<]*)</title>", s)]
    assert len(titles) == 33, f"expected 33 band titles, found {len(titles)}"
    for m in reversed(titles):
        new = "<title>" + esc(T.band_title(H.unescape(m.group(1)))) + "</title>"
        assert s.count(m.group(0)) == 1, m.group(0)
        s = s[:m.start()] + new + s[m.end():]
        LOG.append("band title")

    # 5. column descriptions: the one-line COLS object of the tooltip script
    head = "  const COLS = "
    a = s.index(head)
    z = s.index(";\n", a)
    raw = s[a + len(head):z]
    cols = json.loads(raw.replace("<\\/", "</"))
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    assert dump(cols) == raw, "the COLS object does not round-trip; its serialisation differs"
    distinct = []
    for sec, lst in cols.items():
        for e in lst:
            for v in e["cols"].values():
                d = v[1] if isinstance(v, list) else v
                if d not in distinct:
                    distinct.append(d)
    assert len(distinct) == len(T.COLS_NEW), f"{len(distinct)} distinct descriptions, {len(T.COLS_NEW)} new texts"
    newdesc = {}
    for d, (start, new) in zip(distinct, T.COLS_NEW):
        assert d.startswith(start), f"column description out of order: {d[:70]!r} does not start with {start!r}"
        newdesc[d] = new
    for sec, lst in cols.items():
        for i, e in enumerate(lst):
            for name, v in list(e["cols"].items()):
                if isinstance(v, list):
                    e["cols"][name] = [v[0], newdesc[v[1]]]
                else:
                    e["cols"][name] = newdesc[v]
                if (sec, i, name) in T.RENAME:
                    e["cols"][name] = [T.RENAME[(sec, i, name)], e["cols"][name]]
    s = once(s, head + raw + ";\n", head + dump(cols) + ";\n", "column descriptions")
    json.dump(cols, open(os.path.join(TMP, "cols_after.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 6. scripts and style
    i1, i2 = s.index("/* Tooltips: one shared tooltip"), s.index("/* e6_tooltips */")
    for what, old, new in C.TT:
        i1, i2 = s.index("/* Tooltips: one shared tooltip"), s.index("/* e6_tooltips */")
        s = once(s, old, new, "tooltip script: " + what, i1, i2)
    for what, old, new in C.DR:
        i2 = s.index("/* e6_tooltips */")
        s = once(s, old, new, "drawings script: " + what, i2)
    for what, old, new in C.IW:
        i3 = s.index("/* the weighting test: every number here")
        i4 = s.index("</script>", i3)
        s = once(s, old, new, "weighting test: " + what, i3, i4)
    s = once(s, C.CSS_OLD, C.CSS_NEW, "style: the box no longer hides text")
    s = once(s, "</style>", C.CSS_ADD + "</style>", "style: tuple, note, fit")

    out = os.path.join(TMP, "dry.html") if dry else PAGE
    open(out, "w", encoding="utf-8").write(s)
    if not dry:
        open(os.path.join(TMP, "last_applied.md5"), "w").write(md5(PAGE))
    assert md5(BACKUP) == V10_MD5, "the backup changed"
    print(f"{'dry run' if dry else 'applied'}: {len(s):,} bytes -> {out}; md5 {md5(out)}; backup untouched ({V10_MD5})")
    print(f"{len(LOG)} exact replacements ({sum(1 for x in LOG if x.startswith('tip '))} tooltip texts, "
          f"{sum(1 for x in LOG if x.startswith('legend'))} legends, {sum(1 for x in LOG if x.startswith('guide'))} guide "
          f"tooltips, {sum(1 for x in LOG if x == 'band title')} band titles, 1 column-description object, "
          f"{len(C.TT) + len(C.DR) + len(C.IW) + 2} script and style snippets)")


if __name__ == "__main__":
    main()
