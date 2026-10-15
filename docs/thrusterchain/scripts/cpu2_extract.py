"""CPU 2: extract Results, After the stop, and weighting-test HTML + plain text (read only)."""
import re, html, json, os
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/cpu2"
s = open(P, encoding="utf-8").read()

def section(sid):
    m = re.search(r'<section id="%s"' % re.escape(sid), s)
    assert m, sid
    i = m.start()
    # find matching </section> by depth counting
    depth = 0
    for t in re.finditer(r'<(/?)section\b', s[i:]):
        depth += -1 if t.group(1) else 1
        if depth == 0:
            return s[i:i + t.end() + len('>') + 0]
    raise SystemExit("no end " + sid)

def to_text(h):
    # replace mjx-container with its aria-label TeX
    h = re.sub(r'<mjx-container[^>]*?aria-label="([^"]*)"[^>]*>.*?</mjx-container>', lambda m: ' $' + html.unescape(m.group(1)) + '$ ', h, flags=re.S)
    h = re.sub(r'<svg\b.*?</svg>', ' [svg] ', h, flags=re.S)
    h = re.sub(r'<script\b.*?</script>', ' [script] ', h, flags=re.S)
    h = re.sub(r'<style\b.*?</style>', '', h, flags=re.S)
    h = re.sub(r'<(h[1-6])[^>]*>', r'\n\n## ', h)
    h = re.sub(r'</(p|li|tr|h[1-6]|figcaption|div|table|caption)>', '\n', h)
    h = re.sub(r'<(td|th)[^>]*>', ' | ', h)
    h = re.sub(r'<[^>]+>', '', h)
    h = html.unescape(h)
    h = re.sub(r'[ \t]+', ' ', h)
    h = re.sub(r'\n\s*\n+', '\n\n', h)
    return h.strip()

for sid in ["results", "weighting-test", "harm-checks", "bca-test", "shift", "reference-scores", "methods", "task", "dataset", "recording-controller"]:
    try:
        sec = section(sid)
    except AssertionError:
        # glance may not be a section
        m = re.search(r'id="%s"' % sid, s)
        print("not a section:", sid, bool(m)); continue
    open(os.path.join(OUT, sid + ".html"), "w", encoding="utf-8").write(sec)
    open(os.path.join(OUT, sid + ".txt"), "w", encoding="utf-8").write(to_text(sec))
    tips = re.findall(r'data-tip="([^"]*)"', sec)
    open(os.path.join(OUT, sid + "_tips.txt"), "w", encoding="utf-8").write("\n".join(html.unescape(t) for t in tips))
    print(sid, len(sec), "chars html;", len(tips), "tips")
