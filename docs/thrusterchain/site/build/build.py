"""Assemble docs/thrusterchain/index.html from the site sources, resolving and checking every manifest number.

Run from the repository root (standard library only):
    python -I docs/thrusterchain/site/build/build.py

What it checks before writing anything:
  * every manifest headline resolves: a JSON path yields a value in its source file, or a text snippet
    occurs verbatim in its source file; missing or unresolvable entries fail the build;
  * every <span class="man" data-man="id|label"> placeholder in the parts refers to a resolved headline;
  * the data files exist, parse, and the results cross-checks all matched;
  * all in-page fragment links point at an existing id; ids are unique;
  * the final page is below 16 MB and contains no external script, style or font reference.
A build receipt with hashes and the check list is written next to the data files.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SITE = ROOT / "docs/thrusterchain/site"
OUT = ROOT / "docs/thrusterchain/index.html"
PARTS = ["start", "environment", "bca", "setup", "results", "evidence"]
_JSON_CACHE: dict[str, object] = {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def src_path(rel: str) -> Path:
    """Repository-relative by default; absolute Windows paths are allowed for the Codex task folder."""
    return Path(rel) if re.match(r"^[A-Za-z]:/", rel) else ROOT / rel


def load_json(rel: str):
    if rel not in _JSON_CACHE:
        _JSON_CACHE[rel] = json.loads(src_path(rel).read_text("utf8"))
    return _JSON_CACHE[rel]


def walk(obj, path: str):
    """Resolve a dotted JSON path with [i] indices; keys may contain dots only inside brackets, e.g. a["x.y"] is not supported, use plain keys."""
    tokens = re.findall(r'\["([^"]+)"\]|\[(\d+)\]|([^.\[\]]+)', path)
    cur = obj
    for qkey, idx, key in tokens:
        if qkey != "":
            cur = cur[qkey]
        elif idx != "":
            cur = cur[int(idx)]
        else:
            if isinstance(cur, dict):
                cur = cur[key]
            else:
                raise KeyError(key)
    return cur


def fmt(v, rnd=None, unit=None, pct=False, sign=False):
    if v is None:
        return "unavailable"
    if isinstance(v, bool):
        s = "true" if v else "false"
    elif isinstance(v, (int,)) and not isinstance(v, bool):
        s = f"{v:,}"
    elif isinstance(v, float):
        if pct:
            s = f"{100 * v:.1f}%"
        elif rnd is not None:
            s = f"{v:+,.{rnd}f}" if sign else f"{v:,.{rnd}f}"
        else:
            s = repr(v)
    else:
        s = str(v)
    return s + (" " + unit if unit else "")


def resolve_manifest(man: dict) -> tuple[dict, list]:
    checks = []
    for e in man["experiments"]:
        for h in e.get("headline", []):
            src = h["source"]
            p = src_path(src)
            if not p.exists():
                raise SystemExit(f"manifest source missing: {src} ({e['id']} / {h['label']})")
            if "path" in h:
                data = load_json(src)
                try:
                    v = walk(data, h["path"])
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(f"cannot resolve {h['path']} in {src}: {exc}")
                h["value"] = v
                disp = fmt(v, h.get("round"), None, sign=bool(h.get("sign")))
                if "pair_path" in h:
                    v2 = walk(data, h["pair_path"])
                    h["value_pair"] = v2
                    r = h.get("round", 4)
                    disp = f"{fmt(v, r if isinstance(v, float) else None, sign=bool(h.get('sign')))} / {fmt(v2, r if isinstance(v2, float) else None, sign=bool(h.get('sign')))}"
                if "ci_path" in h:
                    ci = walk(data, h["ci_path"])
                    h["ci"] = ci
                    disp += f" [{ci[0]:.{h.get('round', 3)}f}, {ci[1]:.{h.get('round', 3)}f}]"
                if h.get("unit"):
                    disp += " " + h["unit"]
                h["display"] = disp
                checks.append({"experiment": e["id"], "label": h["label"], "source": src, "path": h["path"], "resolved": True, "value": v if not isinstance(v, (dict, list)) else "(compound)"})
            elif "text" in h:
                body = p.read_text("utf8", errors="replace")
                if h["text"] not in body:
                    raise SystemExit(f"text not found verbatim in {src}: {h['text'][:80]!r}")
                h["value"] = h["text"]
                h["display"] = h["text"]
                checks.append({"experiment": e["id"], "label": h["label"], "source": src, "text": h["text"], "resolved": True})
            else:
                raise SystemExit(f"headline without path or text: {e['id']} / {h['label']}")
        for s in e.get("sources", []):
            if not src_path(s).exists():
                raise SystemExit(f"listed source does not exist: {s}")
    return man, checks


def fill_placeholders(fragment: str, man: dict) -> tuple[str, int]:
    index = {}
    for e in man["experiments"]:
        for h in e.get("headline", []):
            index[(e["id"], h["label"])] = h
    count = 0

    def rep(m):
        nonlocal count
        attrs = m.group(1)
        key = re.search(r'data-man="([^"]+)"', attrs).group(1)
        eid, label = key.split("|", 1)
        label = html.unescape(label)
        h = index.get((eid, label))
        if h is None:
            raise SystemExit(f"placeholder refers to an unknown headline: {eid} | {label}")
        pct = 'data-pct' in attrs
        v = h.get("value")
        text = fmt(v, h.get("round"), None, pct=pct, sign=bool(h.get("sign"))) if "path" in h else h["display"]
        if "pair_path" in h and not pct:
            text = h["display"]
        count += 1
        return f'<span class="man" title="{html.escape(h["source"])}">{html.escape(text)}</span>'

    out = re.sub(r'<span class="man"([^>]*)></span>', rep, fragment)
    return out, count


def main() -> int:
    css = (SITE / "site.css").read_text("utf8")
    js = (SITE / "site.js").read_text("utf8")
    env = json.loads((SITE / "data/environment.json").read_text("utf8"))
    res = json.loads((SITE / "data/results.json").read_text("utf8"))
    if not all(c["match"] for c in res["cross_checks"]):
        raise SystemExit("results.json cross-checks did not all match; rerun extract_results.py")
    if env["validation"]["max_abs_error"] > env["validation"]["tolerance"]:
        raise SystemExit("environment.json fixture validation exceeded tolerance")
    man = json.loads((SITE / "manifest.json").read_text("utf8"))
    man, checks = resolve_manifest(man)
    (SITE / "data/manifest_resolved.json").write_text(json.dumps(man, indent=1), "utf8", newline="\n")

    parts_html, placeholders = [], 0
    for name in PARTS:
        frag = (SITE / "parts" / f"{name}.html").read_text("utf8")
        frag, n = fill_placeholders(frag, man)
        placeholders += n
        parts_html.append(frag)
    body_parts = "\n".join(parts_html)

    def jsonblock(id_, obj):
        return f'<script type="application/json" id="{id_}">' + json.dumps(obj, separators=(",", ":"), allow_nan=False).replace("</", "<\\/") + "</script>"

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>ThrusterChain BCA interface</title>
<meta name="description" content="Research interface for Bayesian Conformal Aggregation on the ThrusterChain benchmark with the Unifloral TD3+BC host: environment explorer, BCA explained from source, experimental setup, results and evidence manifest.">
<style>
{css}
</style>
</head>
<body>
<a class="sr" href="#part-start">Skip to content</a>
<div class="topbar"><div class="topbar-in">
  <a class="brand" href="#part-start">ThrusterChain · BCA<small>Unifloral TD3+BC host · evidence as of 27 Oct 2026</small></a>
  <nav class="tabs" id="tabs" aria-label="Parts"></nav>
  <div class="tools"><button class="btn" id="js-all" type="button" aria-pressed="false">Show all parts</button><button class="btn" id="js-theme" type="button">Theme: auto</button></div>
</div></div>
<div class="layout">
  <aside class="side" aria-label="In this part">
    <details open>
      <summary>In this part</summary>
      <h2 id="side-part">Start here</h2>
      <ol id="side-list"></ol>
      <div class="side-foot"><button class="btn js-prev" type="button">← Previous</button><button class="btn js-next" type="button">Next →</button></div>
    </details>
  </aside>
  <main id="main">
{body_parts}
  </main>
</div>
<div class="bottombar"><button class="btn js-prev" type="button" aria-label="Previous section">←</button><div class="where" id="where">Start here</div><button class="btn js-next" type="button" aria-label="Next section">→</button></div>
<footer class="footer">Built {built} from checkout adf51269 on branch codex/thrusterchain-bca-handoff-20261026. No training, simulation beyond one validated plant step, fitting or network access happens on this page. Sources: docs/thrusterchain/site/.</footer>
{jsonblock('env-data', env)}
{jsonblock('results-data', res)}
{jsonblock('manifest-data', man)}
<script>
{js}
</script>
</body>
</html>
"""
    # Link and id checks.
    ids = re.findall(r'\sid="([^"]+)"', page)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit("duplicate ids: " + ", ".join(dupes[:20]))
    idset = set(ids)
    # h3 ids are generated at runtime; accept links to h3 slugs only if a matching heading text exists.
    hrefs = re.findall(r'href="#([^"]+)"', page)
    missing = sorted({h for h in hrefs if h not in idset})
    if missing:
        raise SystemExit("fragment links without a target: " + ", ".join(missing))
    if re.search(r'<(script|link)[^>]+(src|href)="https?:', page) or "fonts.googleapis" in page:
        raise SystemExit("external resource reference found")
    OUT.write_text(page, "utf8", newline="\n")
    size = OUT.stat().st_size
    if size > 16 * 1024 * 1024:
        raise SystemExit("page exceeds 16 MB")
    receipt = {
        "built_utc": built, "output": OUT.relative_to(ROOT).as_posix(), "bytes": size, "sha256": sha(OUT),
        "inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in [SITE / "site.css", SITE / "site.js", SITE / "manifest.json", SITE / "data/environment.json", SITE / "data/results.json", *[SITE / "parts" / f"{n}.html" for n in PARTS]]},
        "checks": {
            "manifest_headlines_resolved": len(checks), "placeholders_filled": placeholders,
            "results_cross_checks": len(res["cross_checks"]), "results_cross_checks_all_match": True,
            "environment_fixture_cases": env["validation"]["fixture_cases_checked"], "environment_max_abs_error": env["validation"]["max_abs_error"],
            "ids": len(idset), "duplicate_ids": 0, "fragment_links": len(set(hrefs)), "fragment_links_missing": 0, "external_references": 0,
        },
        "headlines": checks,
    }
    (SITE / "data/build_receipt.json").write_text(json.dumps(receipt, indent=1), "utf8", newline="\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "headlines"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
