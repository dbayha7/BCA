"""Rebase onto the live page's version 26 (service version 1792172582-8406), part 2: an id and a short label for the
one table of version 26's section#capped-pull ("BCA's main channel"), whose four headings carry ids already, so its
"In this section" entry and deep link are stable (#capped-table) rather than made at run time from the caption."""
import json
import rep

p = rep.W7 / "nav" / "src" / "static_edits.json"
d = json.loads(p.read_text(encoding="utf-8"))
OLD = ('<div class="tbl"><table class="stack">\n      <caption><span class="cap">Capped pull minus plain TD3+BC (BCA IW '
       'minus plain TD3+BC in brackets)')
LABEL = "Capped pull minus plain TD3+BC, by data quality and situation"
NEW = OLD.replace('<table class="stack">', '<table id="capped-table" class="stack" data-nav-label="%s">' % LABEL, 1)
if any(e.get("id") == "capped-table" for e in d["edits"]):
    print("already applied")
else:
    d["edits"].append({"kind": "id+label", "unit": "capped-pull", "entry": "table", "label": LABEL, "id": "capped-table",
                       "line": None, "old": OLD, "new": NEW, "count": 1})
    d["n_edits"] = len(d["edits"])
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("wrote nav/src/static_edits.json (capped-table); %d edits" % len(d["edits"]))
