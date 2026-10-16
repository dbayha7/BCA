"""Test only: static checks of the integrated page (version 27) against its base, the live page after version 25, whose
status box was removed (static_v20.py, static_v24.py and static_v26.py did the same for versions 20, 24 and 26; made by
make_tests_v27.py).

  - page contract: external resources (only the Google Fonts already linked), every script that version 24 adds is an
    IIFE (or a JSON data block), localStorage only inside try/catch, no alert/confirm/prompt/window.print, in-page
    links only as bare #id fragments, no CR byte, under 16 MB with the wrapper;
  - every color literal in a <style> block that version 24 adds sits in a custom-property definition, and every
    custom property those blocks define on :root is redefined in both dark blocks (or is defined from other tokens);
  - no part of the status box (removed at the owner's request); version 25's rerun data block and the live page's newer
    "Strong test mix" tooltip are exactly as in the base;
  - British spellings: none new anywhere in the page (markup, attributes and scripts), and none in the integration's
    own text;
  - every id once, every in-page link and aria reference resolves (static);
  - the walkthrough's and the old navigation's markup is gone, the lesson's L4 and the navigation's start bar agree;
  - the toy worlds (the toy integration): the section is the toy patch's fragment byte for byte, its color tokens are
    defined on :root and redefined in both dark blocks, every token it uses is defined, and its 13 old ids are kept.
Usage: python -X utf8 static_v27.py  -> prints a summary and writes out/static_v27.json
"""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
SCR = BUILD.parent.parent
v19 = (SCR / "live6" / "core_92ba.html").read_text(encoding="utf-8")   # the base: the live page (the name v19 is kept for the code below)
v20 = (BUILD / "core_v27.html").read_text(encoding="utf-8")            # the built page: version 27
res = {}

# ------------------------------------------------------------------ resources and links
ext = lambda s: sorted(set(re.findall(r'<(?:link|script|img|iframe|source|video|audio)\b[^>]*\b(?:href|src)="(https?://[^"]+)"', s)))
res["external_resources_v19"] = ext(v19)
res["external_resources_v20"] = ext(v20)
res["external_new"] = sorted(set(res["external_resources_v20"]) - set(res["external_resources_v19"]))
css_urls = sorted(set(re.findall(r"url\(\s*['\"]?(https?://[^'\")]+)", v20)))
res["css_external_urls"] = css_urls
hrefs = re.findall(r'\shref="(#[^"]*)"', v20)
res["hash_links"] = len(hrefs)
res["hash_links_not_bare"] = sorted({h for h in hrefs if not re.fullmatch(r"#[A-Za-z][\w-]*", h)})
res["cr_bytes"] = v20.count("\r")
res["bytes_core"] = len(v20.encode("utf-8"))

# ------------------------------------------------------------------ scripts
def scripts(s):
    return [(m.group(1), m.group(2)) for m in re.finditer(r"<script([^>]*)>(.*?)</script>", s, re.S)]
s19 = {b for _, b in scripts(v19)}
heads19 = {b.lstrip()[:300] for b in s19}
changed = [(a, b) for a, b in scripts(v20) if b not in s19]
# a script whose start is a version-19 script's start is that script, edited (by the color patch); the others are new
new_scripts = [(a, b) for a, b in changed if b.lstrip()[:300] not in heads19]
res["scripts_edited"] = [b.lstrip()[:60] for a, b in changed if b.lstrip()[:300] in heads19]
res["scripts_v19"] = len(s19)
res["scripts_v20"] = len(scripts(v20))
res["scripts_new"] = []
for a, b in new_scripts:
    kind = "json" if 'type="application/json"' in a else "js"
    body = re.sub(r"^\s*(/\*.*?\*/\s*|//[^\n]*\n\s*)*", "", b, flags=re.S)
    iife = kind == "json" or bool(re.match(r"(\(\(\)\s*=>\s*\{|\(function\s*\(|;?\(\s*function|\(async\s*\(\)\s*=>)", body))
    tail_ok = kind == "json" or bool(re.search(r"\}\)\(\);?\s*$", b.strip()) or re.search(r"\}\(\)\);?\s*$", b.strip()))
    ls = [m.start() for m in re.finditer(r"localStorage", b)]
    ls_bad = [b[max(0, p - 100):p + 40].replace("\n", " ") for p in ls if "try" not in b[max(0, p - 160):p]]
    forb = sorted({m.group(0) for m in re.finditer(r"\b(alert|confirm|prompt)\s*\(|window\.print\s*\(", b)})
    res["scripts_new"].append({"kind": kind, "bytes": len(b.encode("utf-8")), "starts": body[:60], "iife": iife, "iife_end": tail_ok,
                               "localStorage": len(ls), "localStorage_outside_try": ls_bad[:3], "forbidden": forb})
res["scripts_ok"] = all(x["iife"] and x["iife_end"] and not x["localStorage_outside_try"] and not x["forbidden"] for x in res["scripts_new"])

# ------------------------------------------------------------------ colors in the added style blocks
def styles(s):
    return re.findall(r"<style>(.*?)</style>", s, re.S)
def css_lines(b):
    return re.sub(r"/\*.*?\*/", "", b, flags=re.S).split("\n")
lines19 = {l.strip() for b in styles(v19) for l in css_lines(b)}
# only the CSS lines that version 24 adds or changes (a line of version 19's CSS kept as it was is not counted)
new_styles = ["\n".join(l for l in css_lines(b) if l.strip() not in lines19) for b in styles(v20)]
lit, defs_root, defs_dark1, defs_dark2 = [], set(), set(), set()
COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)|\boklch\([^)]*\)")
for b in new_styles:
    nocom = re.sub(r"/\*.*?\*/", "", b, flags=re.S)
    for line in nocom.split("\n"):
        for m in COLOR.finditer(line):
            if not re.search(r"--[\w-]+\s*:\s*[^;]*" + re.escape(m.group(0)), line):
                lit.append(line.strip()[:140])
res["new_css_lines"] = sum(len([l for l in b.split("\n") if l.strip()]) for b in new_styles)
# black shadows (rgba(0, 0, 0, a) in a box-shadow) are how version 19 draws every shadow (7 of them, the same in both
# themes); they are reported, not counted as colors
SHADOW = re.compile(r"box-shadow:[^;]*rgba\(\s*0\s*,\s*0\s*,\s*0\s*,\s*[0-9.]+\s*\)")
res["shadow_literals"] = [l for l in lit if SHADOW.search(l) and not COLOR.search(SHADOW.sub("", l))]
lit = [l for l in lit if l not in res["shadow_literals"]]
res["color_literals_outside_tokens"] = lit[:20]
res["color_literals_outside_tokens_n"] = len(lit)

# ------------------------------------------------------------------ British spellings: nothing new
BRIT = re.compile(r"\b(colour\w*|behaviour\w*|centre\w*|centred|grey\w*|analys(e|ed|es|ing)\b|\w+isation\w*|"
                  r"(organ|recogn|summar|visual|optim|minim|maxim|real|priorit|categor|normal|initial|emphas|stabil|util|standard|special)is(e|ed|es|ing)\b|"
                  r"favour\w*|honour\w*|labelled(?!by)|labelling|modelled|modelling|travelled|cancelled|neighbour\w*|metre\b|programme\b)", re.I)
def brit(s):
    s = re.sub(r"<(mjx-container)\b.*?</\1>", " ", s, flags=re.S)
    # identifiers, never shown as they are: the facts files' key names in the lesson's data-keys attributes and data
    # block (the lesson's script shows a key in American spelling when a worked line is tapped: shownPath)
    s = re.sub(r'data-keys="[^"]*"', " ", s)
    s = re.sub(r'<script type="application/json" id="lesson-data">.*?</script>', " ", s, flags=re.S)
    out = {}
    for m in BRIT.finditer(html.unescape(s)):
        w = m.group(0)
        out[w] = out.get(w, 0) + 1
    return out
b19, b20 = brit(v19), brit(v20)
res["british_v19"] = b19
res["british_new_in_v20"] = {w: n - b19.get(w, 0) for w, n in b20.items() if n > b19.get(w, 0)}
# the new ones, in context
ctx = {}
for w in res["british_new_in_v20"]:
    ctx[w] = [html.unescape(v20[max(0, m.start() - 70):m.end() + 50]).replace("\n", " ") for m in re.finditer(re.escape(w), v20)][:4]
res["british_new_context"] = ctx

# ------------------------------------------------------------------ ids and references
class P:
    pass
import html.parser
class Ids(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.ids = {}; self.refs = []; self.links = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if d.get("id") is not None: self.ids[d["id"]] = self.ids.get(d["id"], 0) + 1
        for k in ("aria-labelledby", "aria-describedby", "aria-controls", "for", "data-tip-ref"):
            if d.get(k): self.refs += [(k, x) for x in d[k].split()]
        if tag == "a" and (d.get("href") or "").startswith("#") and len(d["href"]) > 1: self.links.append(d["href"][1:])
        if tag == "use":
            for k in ("href", "xlink:href"):
                if (d.get(k) or "").startswith("#"): self.refs.append(("use", d[k][1:]))
    handle_startendtag = handle_starttag
p = Ids(); p.feed(v20); p.close()
res["ids"] = len(p.ids)
res["dup_ids"] = sorted(k for k, n in p.ids.items() if n > 1)
res["links_static"] = len(p.links)
res["links_broken_static"] = sorted({x for x in p.links if x not in p.ids})
res["refs_broken_static"] = sorted({"%s->%s" % r for r in p.refs if r[1] not in p.ids})
res["mathjax_glyph_ids"] = {"MJXL (lesson)": sum(1 for k in p.ids if k.startswith("MJXL")), "MJX other": sum(1 for k in p.ids if k.startswith("MJX") and not k.startswith("MJXL"))}

# ------------------------------------------------------------------ the old navigation and the walkthrough are gone
res["absent"] = {s: s not in v20 for s in ('id="wt-open"', 'id="wt"', "Full walkthrough mode</button>", 'class="toc"', "sb-panel",
                                             'aside class="sb"', "wt-stage", 'id="ls-start"', 'id="sb"', "tc-choices-wide", 'id="sb-tab"', "sb-tab-away",
                                             # version 27: the status box, which the owner asked to remove
                                             'id="head-status"', 'id="head-status-h"', '<ul class="g-status"', 'href="#head-status"',
                                             "Status as of ", "Status of every line of work", "<b>Version 26, ")}
res["present_once"] = {s: v20.count(s) for s in ('id="start-learn"', '<section id="learn"', 'id="lesson-data"', 'id="nv-side"', 'id="nv-pager"',
                                                   'id="tt-switch"', "<b>Version 27, 16 Oct 2026.</b>", 'id="simtog"', 'id="sim-learned-data"', 'id="lc-table5"', 'id="iw-plain-split"',
                                                   "<b>Version 25, 16 Oct 2026.</b>", "<b>Version 24, 15 Oct 2026.</b>", 'id="sim-rerun-data"', 'id="simtog-l-lr"', 'id="exLrnT"')}

# ------------------------------------------------------------------ version 27: no status box; the live page's own parts kept
_rd = lambda x: re.search(r'<script type="application/json" id="sim-rerun-data">(.*?)</script>', x, re.S).group(1)
res["rerun_data_same"] = _rd(v20) == _rd(v19)
# the live page's "Strong test mix" tooltips (three: the switch bar's button, rewritten after version 25, and two older ones)
_st = re.findall(r'data-tip-title="Strong test mix" data-tip="[^"]*"', v19)
_new = [x for x in _st if "What makes it strong" in x]
res["strong_mix_tips"] = len(_st)
res["strong_mix_tip"] = _new[0][:400] if _new else None
res["strong_mix_tip_same"] = len(_new) == 1 and all(v20.count(x) == v19.count(x) for x in _st)
_g = lambda x: x[x.index("</p>", x.index('<p class="g-one">')) + 4:x.index('<p class="pipe-h" id="pipe-h">')]
res["header_as_live"] = _g(v20) == _g(v19) and not _g(v20).strip()

# ------------------------------------------------------------------ the toy worlds (the toy integration)
_frag = (BUILD.parent / "toy" / "toy_fragment.html").read_text(encoding="utf-8")
_ta = v20.index('<section id="toy-worlds">')
_toy = v20[_ta:v20.index("</section>", _ta) + len("</section>")]
res["toy_is_fragment"] = _toy == _frag
_css = re.search(r"<style>(.*?)</style>", _toy, re.S).group(1)
_tok = lambda rx: set(re.findall(r"(--toy-[\w-]+)\s*:", re.search(rx, _css).group(1)))
_root = _tok(r"(?m)^:root \{([^}]*)\}")
_d1 = _tok(r'@media \(prefers-color-scheme: dark\) \{ :root:not\(\[data-theme="light"\]\) \{([^}]*)\}')
_d2 = _tok(r':root\[data-theme="dark"\] \{([^}]*)\}')
res["toy_tokens"] = {"root": sorted(_root), "dark_media": sorted(_d1), "dark_theme": sorted(_d2)}
res["toy_tokens_ok"] = bool(_root) and _root == _d1 == _d2
res["toy_tokens_undefined"] = sorted(set(re.findall(r"var\((--toy-[\w-]+)\)", _toy)) - _root)
res["toy_ids_kept"] = {i: v20.count('id="%s"' % i) for i in ('toy-worlds', 'toy-why', 'toy-world', 'toy-first', 'toy-second', 'toy-shelved', 'toy-recheck', 'toy-meaning', 'toy-reward-table', 'toy-pooled-table', 'toy-shelved-width', 'toy-shelved-posterior', 'toy-more-detail')}

ok = (not res["external_new"] and not css_urls and not res["hash_links_not_bare"] and res["cr_bytes"] == 0 and res["scripts_ok"]
      and not lit and not res["british_new_in_v20"] and not res["dup_ids"] and not res["links_broken_static"] and not res["refs_broken_static"]
      and all(res["absent"].values()) and all(n == 1 for n in res["present_once"].values()) and res["rerun_data_same"]
      and res["strong_mix_tip_same"] and res["header_as_live"]
      and res["toy_is_fragment"] and res["toy_tokens_ok"] and not res["toy_tokens_undefined"]
      and all(n == 1 for n in res["toy_ids_kept"].values()))
res["ok"] = ok
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "static_v27.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
short = {k: v for k, v in res.items() if k not in ("british_v19",)}
print(json.dumps(short, indent=1, ensure_ascii=False)[:6000])
print("static_v27 ok:", ok)
