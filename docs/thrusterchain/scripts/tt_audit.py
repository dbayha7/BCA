#!/usr/bin/env python3
"""Tooltip inventory, step 2 of 2: one record per tooltip text, its locations, its source and its problems.

Reads tmp/inv/harvest.json (from tt_harvest.py) and tmp/inv/probe.json (from tt_probe.py), plus the generator
scripts (e5_cols.py, e6_content.py, e7_texts.py) to name each text's source. Read-only on the page.
Writes notes/tooltips_inventory.json and prints counts. Manual judgements are in tt_audit_notes.py.
Run (Windows or WSL python):
  python scripts/tt_audit.py
"""
import collections
import html as H
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import tt_audit_notes as NOTES  # noqa: E402

HARV = json.load(open(os.path.join(PAGE_DIR, "tmp", "inv", "harvest.json"), encoding="utf-8"))
PROBE = json.load(open(os.path.join(PAGE_DIR, "tmp", "inv", "probe.json"), encoding="utf-8"))
PAGE = open(os.path.join(PAGE_DIR, "thrusterchain.html"), encoding="utf-8").read()
OUT = os.path.join(PAGE_DIR, "notes", "tooltips_inventory.json")
R, R390 = HARV["1200"], HARV["390"]
LANDSCAPE_VH = 360          # a phone held sideways: about 360 px of page height under the browser bar

# ------------------------------------------------------------------------------------------------ sources
def strings(obj, path, out):
    if isinstance(obj, str):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            strings(v, f"{path}[{k!r}]" if not isinstance(k, str) or len(k) < 40 else f"{path}[…]", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            strings(v, f"{path}[{i}]", out)


SRC = []
import e6_content  # noqa: E402
for name in dir(e6_content):
    if name.startswith("__") or name in ("WORKED_OLD", "WORKED_NEW", "I"):
        continue
    strings(getattr(e6_content, name), "e6_content.py " + name, SRC)
import e7_texts  # noqa: E402
for what, old, new, mode, n in e7_texts.TEXTS:
    SRC.insert(0, (f"e7_texts.py TEXTS «{what}»", new))
for what, old, new, n in e7_texts.COLS_TEXTS:
    SRC.insert(0, (f"e7_texts.py COLS_TEXTS «{what}»", new.replace('\\"', '"')))
import e5_cols  # noqa: E402
strings(e5_cols.COLS, "e5_cols.py COLS", SRC)
SRC = [(p, re.sub(r"<[^>]+>", "", H.unescape(s)).replace("\n", " ")) for p, s in SRC]


def source_of(body, default):
    if not body:
        return default
    probe = body.strip()
    for key in (probe, probe[:70], probe[-60:]):
        if len(key) < 12:
            continue
        for p, s in SRC:
            if key in s:
                return p
    return default


SUPM = dict(zip("0123456789+−-=()ntiTk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿᵗⁱᵀᵏ"))
SUBM = dict(zip("0123456789+−-=()aeijkmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₐₑᵢⱼₖₘₙₒₚᵣₛₜᵤᵥₓ"))
SUBM[","] = ","
SUPM.update(zip("abcdefghijklmnoprstuvwxyz,", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ'"))


def html_text(h):
    """Plain text of a legend's HTML, with <sub> and <sup> written as Unicode indices where they exist."""
    def conv(m, table, mark):
        inner = H.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        if inner and all(ch in table for ch in inner):
            return "".join(table[ch] for ch in inner)
        return mark + inner          # (the legend shows it as a real index; this is only the inventory's text)
    h = re.sub(r"<sub>(.*?)</sub>", lambda m: conv(m, SUBM, "_"), h, flags=re.S)
    h = re.sub(r"<sup>(.*?)</sup>", lambda m: conv(m, SUPM, "^"), h, flags=re.S)
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", h))).strip()


# ------------------------------------------------------------------------------------------------ page text for numbers
def page_corpus():
    s = re.sub(r"<script\b.*?</script>|<style\b.*?</style>|<template\b.*?</template>", " ", PAGE, flags=re.S)
    labels = " ".join(H.unescape(m) for m in re.findall(r'<mjx-container[^>]*aria-label="([^"]*)"', s))
    s = re.sub(r"<title>.*?</title>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    txt = H.unescape(s) + " " + labels
    return txt.replace("−", "-").replace(",", "")


CORPUS = page_corpus()
NUM = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?%?")


def unverified_numbers(text):
    out = []
    for m in NUM.finditer(text):
        n = m.group(0).replace("−", "-").replace(",", "")
        bare = n.lstrip("-").rstrip("%")
        if bare in {"0", "1", "2", "3", "4", "5", "10", "13", "100", "80", "99"}:
            continue
        if n not in CORPUS and bare not in CORPUS:
            out.append(m.group(0))
    return out


# ------------------------------------------------------------------------------------------------ rule checks
VERBS = set("""is are was were be been being has have had does do did can cannot could will would should must may might
accepts acts adds allows answers applies asks averages behaves carries changes charged checks climbs counts covers curves
cuts damps deletes describes differs divides draws drives enters equals exceeds fades falls finds fixes flips follows
gains gives goes grows happens holds keeps lasts leans learns leaves lies limits lowers makes means moves multiplies
needs occurs overrates passes plots prefers pushes puts rates reaches reads runs says sees sends sets settles shows
shrinks sits spans stalls stands stays stops stores takes tells trains turns uses visits made ran failed passed chose
went found took kept stopped stood marks tests turned lost picks treats became needed won rises ends ranges scores
play press drag hover tap show pause read see use click choose open move keep give take add pick
go lie stay fall rise need act differ reach grow settle follow learn carry lose enter
sit stand cover sets draw stall
isn't aren't doesn't don't can't won't""".split())
VERB_PHRASES = re.compile(r"\b(?:it|they|this|that|which|push|water|wave)\s+points\b", re.I)
MARKERS = {"that", "which", "who", "whom", "whose", "where", "when", "while", "because", "if", "whether", "unless",
           "until", "so", "as", "how", "what", "why", "though", "although", "once", "since", "after", "before",
           "except"}
MODALS = {"would", "will", "can", "cannot", "could", "must", "may", "might", "should"}
LEAD_OK = {"so", "here", "there", "then", "only", "once", "since", "as", "when", "while", "if", "after", "before"}
IMPERATIVE = {"play", "press", "drag", "hover", "tap", "show", "pause", "read", "see", "use", "click", "choose", "open",
              "move", "keep", "give", "take", "add", "pick", "go"}
NOUN_CLAUSE = re.compile(r"^(What|How|Which|Whether|Where|When|Why|Who)\b")


def main_verb(sent):
    """True when the sentence has a finite verb outside brackets and outside clauses that start with a marker word
    (that, which, where, while, so, as, than, how ...), which run to the next comma, semicolon, colon or bracket."""
    s = re.sub(r"\([^()]*\)", " ", sent)
    s = re.sub(r"\([^()]*\)", " ", s)
    s = re.sub(r"\b(starting|thrusters'|thrusters’|the) pushes\b", r"\1 kicks", s)
    if VERB_PHRASES.search(s):
        return True
    toks = re.findall(r"[A-Za-z][A-Za-z'’\-]*|[,;:]", s)
    inside = False
    for i, t in enumerate(toks):
        lt = t.lower()
        if t in ",;:":
            inside = False
            continue
        if lt in MARKERS and not (i == 0 and lt in LEAD_OK and lt not in ("when", "while", "if", "as", "once", "since"))                 and not (i == 0 and lt in ("after", "before")):
            inside = True
            continue
        if (not inside and lt in VERBS) or lt in MODALS:
            return True
    # a sentence that opens with a subordinate clause and a comma: 'When x is y, z happens'
    return False


def top_level_split(sent):
    out, depth, cur = [], 0, ""
    for ch in sent:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in ":;" and depth == 0:
            out.append((cur, ch))
            cur = ""
        else:
            cur += ch
    out.append((cur, ""))
    return out


def split_sentences(body):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9−\-(“\"'ωαβγδεκλμνπρσφθΔψηΣ×√|𝒩UJQPK])", body.strip())
    return [p for p in parts if p]


def words(s):
    return re.findall(r"[A-Za-z][A-Za-z'’\-]*", s)


def fragment_problems(body):
    probs = []
    b = body.strip()
    if not b:
        return probs
    first = b[0]
    if not first.isupper():
        probs.append(("fragment", "starts with " + ("a number or symbol" if not first.isalpha() else "a lower-case word")
                      + f" ({b[:24]!r}), not a capital letter"))
    if not re.search(r"[.!?][)”\"']?$", b):
        probs.append(("fragment", f"no final full stop (ends {b[-24:]!r})"))
    for sent in split_sentences(b):
        if sent.rstrip().endswith("?"):
            continue
        if NOUN_CLAUSE.match(sent):
            probs.append(("fragment", f"a question-word clause ('What ...', 'How ...'), not a sentence: {sent[:100]!r}"))
            continue
        w0 = (re.findall(r"[A-Za-z][A-Za-z'’\-]*", sent) or [""])[0]
        if sent[:1].isupper() and w0.lower() in VERBS and w0.lower() not in IMPERATIVE and w0.lower().endswith("s")                 and w0.lower() not in ("is", "has", "does", "was"):
            probs.append(("fragment", f"no subject (starts with the verb '{w0}'): {sent[:100]!r}"))
            continue
        if not main_verb(sent):
            probs.append(("fragment", f"no main verb (a noun phrase or a clause only): {sent[:100]!r}"))
            continue
        parts = top_level_split(sent)
        if len(parts) > 1:
            lead, sep = parts[0]
            if lead.strip() and not main_verb(lead) and not re.match(r"\s*(In|At|On|For|With|Past|Above|Below|Up|Near|"
                                                                    r"Within|After|Before|From|Over|Under|Here|There|Then|So|Only|Once|When|While|If)\b", lead):
                probs.append(("fragment", f"a noun phrase before '{sep}' with no verb of its own (an implied 'X is'): {lead.strip()[:80]!r}"))
            for clause, sep2 in parts[1:]:
                pass
            for (c1, s1), (c2, s2) in zip(parts, parts[1:]):
                if s1 == ";" and c2.strip() and not main_verb(c2):
                    probs.append(("fragment", f"a clause after ';' with no verb: {c2.strip()[:80]!r}"))
    return probs


ACRO = re.compile(r"(?<![\w-])(?:[A-Z]{2,}[0-9]*(?:\+[A-Z]{2,})?|[A-Z][0-9][A-Z0-9+]*|[A-Z]{1,}[0-9]+[A-Z]*(?:\+[A-Z]+)?|"
                  r"SHA-256|H1–H4|D4RL|NaN)(?![\w])")
ACRO_OK = {"I", "II", "OK"}
SPELL = {"TD3": "twin delayed", "BC": "behaviour cloning", "BCA": "Bayesian Conformal Aggregation", "SHA": "secure hash",
         "UTC": "coordinated universal time", "D4RL": "datasets for deep data-driven reinforcement learning",
         "JAX": "", "HDF5": "hierarchical data format", "RMS": "root-mean-square"}


def acronym_problems(text):
    probs = []
    seen = set()
    for m in ACRO.finditer(text):
        a = m.group(0)
        if a in ACRO_OK or a in seen:
            continue
        if re.fullmatch(r"[A-Z][0-9]+", a) and a[0] in "NKJQPHTSW":      # symbols with an index: N0, K1, Q1, H1
            continue
        seen.add(a)
        parts = [x for x in re.split(r"\+", a) if x]
        explained = all(re.search(re.escape(p) + r"\s*\(", text) or re.search(r"\(\s*" + re.escape(p) + r"\s*\)", text)
                        or (SPELL.get(p) and SPELL[p].lower() in text.lower()) for p in parts)
        if not explained:
            probs.append(("acronym", f"{a} is not spelled out in this tooltip"))
    return probs


CODE_PATTERNS = [
    (r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", "snake_case name"),
    (r"\b[a-z_]+\.[a-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?", "dotted code name"),
    (r"\b[A-Z][a-z]+[A-Z]\w*\b", "CamelCase code name"),
    (r"(?<![\w/])[A-Za-z_][\w.-]*/[A-Za-z_][\w./-]*", "path or id with a slash"),
    (r"\b\w+\.(?:py|md|json|npz|hdf5|h5)\b", "file name"),
    (r"\b[0-9a-f]{8,}\b", "hash"),
    (r"\b(?:float32|float64|int32|int8|bool|str)\b", "programming type name"),
    (r"\b\w+=\S+", "keyword argument"),
    (r"\b(?:gym|numpy|jax|Gymnasium|Box)\b", "library name"),
]
CODE_OK = {"ThrusterChain"}


def code_problems(text):
    probs, seen = [], set()
    for pat, what in CODE_PATTERNS:
        for m in re.finditer(pat, text):
            tok = m.group(0)
            if tok in CODE_OK or tok in seen or re.fullmatch(r"[\d.]+", tok):
                continue
            if what == "dotted code name" and re.fullmatch(r"e\.g\.?|i\.e\.?", tok):
                continue
            if what == "path or id with a slash" and re.fullmatch(r"[\d.]+/[\d.]+|\w/\w|and/or", tok):
                continue
            seen.add(tok)
            probs.append(("code", f"{what}: {tok}"))
    return probs


def markup_problems(text):
    probs = []
    if re.search(r"\^\(|\\[A-Za-z]+|_\{", text):
        probs.append(("unclear", "math markup left in plain text (" + re.search(r"\S*(\^\(|\\[A-Za-z]+|_\{)\S*", text).group(0) + ")"))
    if re.search(r"[.!]\s*,|[.!]\s+\)|\s,", text):
        m = re.search(r".{0,30}([.!]\s*,|[.!]\s+\)|\s,).{0,10}", text)
        probs.append(("unclear", f"punctuation glitch where the tuple joins values: {m.group(0)!r}"))
    m = re.search(r".{0,30}\w(…|\.\.\.).{0,12}", text)
    if m or text.rstrip().endswith(("…", "...")):
        probs.append(("truncation", "contains an ellipsis that cuts a word, number or sentence short: " +
                      repr(m.group(0) if m else text[-40:])))
    return probs


def sentence_count(body):
    return len(split_sentences(body))


# ------------------------------------------------------------------------------------------------ records
RECS = []


def add(rec):
    rec.setdefault("problems", [])
    RECS.append(rec)
    return rec


def loc_of(t):
    parts = [t["section"] or "top of page"]
    if t.get("anc") and t["anc"].split(" < ")[0] not in (t["section"],):
        parts.append("in #" + t["anc"].split(" < ")[0])
    if t["heading"]:
        parts.append(t["heading"][:70])
    if t["details"]:
        parts.append("details: " + t["details"][:50])
    return " › ".join(parts)


def kind_of(t):
    c = t["cls"]
    if t["isRect"]:
        return "chart band"
    if t["ref"].startswith("tip-ctrl"):
        return "controller guide"
    if t["ref"]:
        return "formula legend"
    if t["tag"] == "canvas":
        return "drawing (canvas)"
    if "sym" in c.split():
        return "symbol"
    if "term" in c.split() or (t["tag"] in ("i", "span") and not c):
        return "term" if t["tag"] != "span" or "term" in c else "chart legend item"
    if t["tag"] in ("button", "input"):
        return "control"
    if "ro" in c.split():
        return "readout"
    if t["svg"]:
        return "chart or diagram part"
    if "iw-formula" in c:
        return "formula readout"
    return "other"


GEN_TIP = "static data-tip attribute in thrusterchain.html"
tips_by_text = collections.OrderedDict()
for t in R["tips"]:
    tips_by_text.setdefault((t["title"], t["shown"] or ""), []).append(t)
h390 = {}
for t in R390["tips"]:
    if t["box"]:
        h390[(t["title"], t["shown"] or "")] = t["box"]["h"]

n = 0
for (title, shown), lst in tips_by_text.items():
    t = lst[0]
    n += 1
    kind = kind_of(t)
    body = shown.split("\n", 1)[1] if (title and "\n" in shown) else shown
    rec = {"id": f"tip-{n:03d}", "kind": kind, "count": len(lst),
           "where": sorted({loc_of(x) + (f" (on '{x['own'][:40]}')" if x["own"] and kind not in ("chart band",) else "")
                            for x in lst})[:12],
           "title": title, "text": body}
    if kind in ("formula legend", "controller guide"):
        rh = t["refHtml"]
        if kind == "formula legend":
            pairs = re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", rh, flags=re.S)
            rec["entries"] = [[html_text(a), html_text(b)] for a, b in pairs]
            rec["text"] = "\n".join(f"{a} — {b}" for a, b in rec["entries"])
            rec["generator"] = ("legend <dl> beside the formula (" + source_of(rec["entries"][0][1] if rec["entries"] else "", "page source")
                                + "); attached as the formula's tooltip by the e7 drawings script (data-tip-ref, "
                                "e7_texts.py JS2 «chart columns, formula legends, Play labels»)")
        else:
            paras = re.findall(r"<p>(.*?)</p>", rh, flags=re.S)
            rec["entries"] = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", H.unescape(p))).strip() for p in paras]
            rec["text"] = "\n".join(rec["entries"])
            rec["generator"] = "<template id=" + t["ref"] + "> " + source_of(rec["entries"][0] if rec["entries"] else "", "") + \
                " (e6_content.py CTRL_PANEL)"
    elif kind == "chart band":
        rec["generator"] = ("e7 drawings script: joins the <title> texts of one column of chart points (page source, "
                            "task chart) into one band; the shared prefix before '): ' becomes the heading and each "
                            "point's remainder a line (e7_texts.py JS2 «chart columns ...»)")
    elif t["tag"] == "path" and t["svg"] == "iw-svg-bars":
        rec["generator"] = ("weighting-test script drawBars(): an SVG <title> per bar; the e7 drawings script turns it "
                            "into data-tip once at load")
    else:
        rec["generator"] = GEN_TIP + " — " + source_of(body, source_of(title, "page source (not in e6/e7 scripts)"))
    rec["height_px"] = {"1200": t["box"]["h"] if t["box"] else None, "390": h390.get((title, shown))}
    add(rec)

# Play buttons in the Pause state, the drawings' dynamic texts
for pb in R["play"]:
    for lab, txt in pb["states"][1:]:
        title, _, body = txt.partition("\n")
        add({"id": f"play-{pb['id']}-pause", "kind": "control (changes with its label)", "count": 1,
             "where": [f"{pb['section'] or 'top of page'} › #{pb['id']} while playing"], "title": title, "text": body,
             "generator": "e7 drawings script: the Play buttons' tooltips follow their label (e7_texts.py JS2)"})
for i, tp in enumerate(R["dyn"]["templates"]):
    ex = tp["example"]
    if not ex:
        continue                     # no tooltip of its own there: the drawing keeps its general text
    title, body = ex[0], ex[1]
    add({"id": f"dyn-{i + 1:02d}", "kind": "drawing part (computed)", "count": 1, "samples": tp["n"],
         "where": [f"live panel (#sim) › {tp['kind']} — e.g. world {tp['ctx'][0]}, {tp['ctx'][1]}, controller {tp['ctx'][2]}, state {tp['ctx'][3]}"],
         "title": title, "text": body, "template": tp["template"],
         "generator": ("e6_tooltips.js channelTip() (revised by e7_texts.py JS2)" if tp["kind"].startswith("channel")
                       else "e6_tooltips.js costTip()") if ex else "(none: the drawing keeps its general text)"})

# native browser tooltips left after the scripts, and the ones that come back
pr = PROBE.get("weighting bars", {})
if pr.get("after clicking Strong", {}).get("titles"):
    add({"id": "native-iw-bars", "kind": "native browser tooltip (SVG <title>)", "count": len(pr["after clicking Strong"]["titles"]),
         "where": ["weighting-test › How often the test needs what training lacks (bars), after a lean button or the slider is used"],
         "title": "", "text": " | ".join(pr["after clicking Strong"]["titles"]),
         "generator": "weighting-test script drawBars() re-creates <title> elements on every redraw; the e7 conversion to "
                      "data-tip ran once at load, so after any change the bars show the browser's own tooltip, which the "
                      "Tooltips switch cannot turn off, and their data-tip is gone"})

# tables: column descriptions (one per distinct text), group headings, row tuples
col_seen = collections.OrderedDict()
for tb in R["tables"]:
    tloc = f"{tb['section']} › table {tb['idxInSec'] + 1}" + (f" ({tb['heading'][:60]})" if tb["heading"] else "")
    for h in tb["headers"]:
        if not h["shown"]:
            continue
        name, _, desc = h["shown"].partition("\n")
        col_seen.setdefault((name, desc), {"tables": [], "box": h["box"]})["tables"].append(tloc)
n = 0
for (name, desc), v in col_seen.items():
    n += 1
    add({"id": f"col-{n:03d}", "kind": "column description", "count": len(v["tables"]),
         "where": sorted(set(v["tables"])), "title": name, "text": desc,
         "generator": "column-description dictionary COLS (" + source_of(desc, "e5_cols.py COLS") +
                      "), shown on the column header and under every cell tooltip of the column; also hidden "
                      "header text for screen readers ('. ' + description)",
         "height_px": {"1200": v["box"]["h"] if v["box"] else None}})

for tb in R["tables"]:
    tloc = f"{tb['section']} › table {tb['idxInSec'] + 1}" + (f" ({tb['heading'][:60]})" if tb["heading"] else "")
    for row in tb["rows"]:
        if "group" in row:
            continue
        cells = [c for c in row["cells"] if c.get("tuple")]
        if not cells:
            continue
        c0 = cells[0]
        hs = [c["box"]["h"] for c in cells if c.get("box")]
        add({"id": f"row-T{tb['ti']:02d}-r{row['ri']:02d}", "kind": "cell tuple (one per table row)",
             "count": len(cells), "where": [tloc + f", row {row['ri']}"], "title": "",
             "text": c0["tuple"], "pairs": c0["pairs"],
             "cells": [{"column": c["colName"].rstrip(": "), "value": c["value"]} for c in cells],
             "generator": "shared tooltip script build() on a data-tt=\"c\" cell (e5_tooltips.js, revised by "
                          "e7_texts.py JS1): the row's cells as (column: value, ...) with plain() text, the cell's own "
                          "value in bold, then that column's description",
             "height_px": {"1200": max(hs) if hs else None}})

# ------------------------------------------------------------------------------------------------ checks
AUTHORED = {"term", "symbol", "readout", "control", "chart or diagram part", "drawing (canvas)", "chart legend item",
            "formula readout",
            "column description", "control (changes with its label)", "drawing part (computed)", "other"}
for rec in RECS:
    k = rec["kind"]
    text = rec["text"] or ""
    full = (rec["title"] + "\n" if rec["title"] else "") + text
    if k == "formula legend":        # the symbols are typeset in the legend; check the words beside them
        full = rec["title"] + "\n" + "\n".join(m for _, m in rec.get("entries", []))
    P = []
    if k in AUTHORED:
        P += fragment_problems(text)
        if sentence_count(text) > 3:
            P.append(("long", f"{sentence_count(text)} sentences (rule: one to three for a term or column)"))
    elif k == "formula legend":
        for sym, mean in rec.get("entries", []):
            for typ, det in fragment_problems(mean):
                P.append((typ, f"entry {sym!r}: {det}"))
    elif k == "controller guide":
        for para in rec.get("entries", []):
            lead, _, rest = para.partition(": ")
            for typ, det in fragment_problems(rest or para):
                P.append((typ, f"paragraph {lead!r}: {det}"))
        if sum(sentence_count(p) for p in rec.get("entries", [])) > 3:
            P.append(("long", f"{sum(sentence_count(p) for p in rec.get('entries', []))} sentences in one tooltip"))
    elif k == "chart band":
        for line in text.split("\n"):
            P += [(t_, "line " + repr(line[:40]) + ": " + d) for t_, d in fragment_problems(line)][:2]
    elif k.startswith("native"):
        for line in text.split(" | "):
            P += [(t_, d) for t_, d in fragment_problems(line) if "full stop" in d][:1]
    P += acronym_problems(full)
    P += code_problems(full)
    P += markup_problems(full)
    if k not in ("cell tuple (one per table row)",):
        un = unverified_numbers(text)
        if un:
            rec["numbers_not_found_in_page_text"] = un
    # long tuples are allowed; the box clips anything taller than the viewport
    h = (rec.get("height_px") or {}).get("1200")
    if h:
        rec["clipped_if_viewport_shorter_than_px"] = h + 16
        if h + 16 > LANDSCAPE_VH:
            P.append(("truncation", f"the tooltip box (max-height: calc(100vh − 16px); overflow: hidden) cuts this "
                                    f"{h}-px-tall text off mid-line in any viewport under {h + 16} px tall, such as a "
                                    f"phone held sideways (about {LANDSCAPE_VH} px)"))
    rec["problems"] = [{"type": a, "detail": b, "how": "automatic check"} for a, b in P]

# manual judgements: add, or drop automatic ones that are wrong
by_id = {r["id"]: r for r in RECS}
for rec in RECS:
    for key, items in NOTES.MANUAL.items():
        if key.startswith("id:"):
            if rec["id"] not in key[3:].split(","):
                continue
        elif key not in (rec["title"] + "\n" + (rec["text"] or "")):
            continue
        for it in items:
            typ, det = it
            if typ == "drop":
                rec["problems"] = [p for p in rec["problems"] if det not in p["detail"]]
            else:
                rec["problems"].append({"type": typ, "detail": det, "how": "manual review"})
for rec in RECS:
    for typ, det in NOTES.TUPLE_RULES(rec):
        rec["problems"].append({"type": typ, "detail": det, "how": "manual rule for this table"})
    # one entry per (type, detail)
    seen, ps = set(), []
    for p in rec["problems"]:
        if (p["type"], p["detail"]) not in seen:
            seen.add((p["type"], p["detail"]))
            ps.append(p)
    rec["problems"] = ps
    rec["problem_types"] = sorted({p["type"] for p in ps})

# ------------------------------------------------------------------------------------------------ output
summary = collections.Counter()
by_kind = collections.defaultdict(collections.Counter)
for rec in RECS:
    for t_ in rec["problem_types"]:
        summary[t_] += 1
        by_kind[rec["kind"]][t_] += 1
    by_kind[rec["kind"]]["_records"] += 1
    by_kind[rec["kind"]]["_shown"] += rec["count"]
meta = {
    "page": "thrusterchain.html (md5 adf3837e6d032f77f6c8bf56bf0221a8, published version 10)",
    "how": "tt_harvest.py loads a copy of the page with three exported hooks and calls the page's own generators "
           "on every anchor, header, cell and drawing state; tt_probe.py hovers real targets; tt_audit.py checks "
           "each text (automatic checks plus manual review in tt_audit_notes.py).",
    "counting": "one record per distinct text: a term repeated in several places is one record (count = anchors); a "
                "column description is one record (count = tables); a cell tuple is one record per table row "
                "(count = cells in the row: every cell shows the same tuple with its own value in bold, then its "
                "column's description); a computed drawing text is one record per template, its numbers replaced by # "
                "(samples = how many of the 48,960 sampled pointer positions produced it: 2 worlds x 4 situations x 4 controllers x 100 states x 15 parts of the channel drawing, and 960 positions on the cost chart). Records with a problem "
                "type are counted once per type.",
    "records": len(RECS),
    "shown_texts": {"data-tip anchors": len(R["tips"]), "table cells": sum(t["nCells"] for t in R["tables"]),
                    "column headers": sum(len(t["headers"]) for t in R["tables"])},
    "records_with_problem_type": dict(summary),
    "by_kind": {k: dict(v) for k, v in by_kind.items()},
    "truncation_mechanisms": NOTES.MECHANISMS,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"meta": meta, "records": RECS}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(RECS)} records -> {OUT} ({os.path.getsize(OUT):,} bytes)")
print("records with each problem type:", dict(summary))
for k, v in by_kind.items():
    print(f"  {k:40s} {dict(v)}")
if "--list" in sys.argv:
    for rec in RECS:
        if rec["kind"].startswith("cell tuple") and "--all" not in sys.argv:
            continue
        print(f"\n{rec['id']} [{rec['kind']}] x{rec['count']} {rec['where'][0][:90]}\n  {rec['title']!r}\n  {rec['text'][:600]!r}")
        for p in rec["problems"]:
            print(f"    - {p['type']}: {p['detail'][:160]}")
        if rec.get("numbers_not_found_in_page_text"):
            print("    ? numbers not in page text:", rec["numbers_not_found_in_page_text"])
