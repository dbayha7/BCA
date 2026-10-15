"""Merge the other session's additions (live version 1791881669-ef92) onto this session's rework, losing nothing.

Their changes relative to v5: one CSS block, a rewritten top status line (merged as one added sentence, since the rework
rewrote the same paragraph), the "After the stop" subsection inside Results, a Version History addition and a contents
link. Each is inserted at the matching anchor of the rework, with asserted positions.
"""
import re
import sys

LIVE = "/mnt/c/Users/David/.claude/projects/C--Users-David-crepes-weighted/c1e911d4-991b-4d7f-a2a7-4d2d993a1d54/tool-results/artifact-3cadefd0-1791881669-ef92.html"
MINE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
V5 = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v5.html"
OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.merged.html"


def unwrap(a):
    if a.startswith("<!doctype"):
        a = a[a.find("<body>") + 6:].lstrip("\n")
    a = a.rstrip()
    if a.endswith("</body></html>"):
        a = a[: -len("</body></html>")].rstrip("\n")
    return a


L = unwrap(open(LIVE, encoding="utf8").read()).split("\n")
V = unwrap(open(V5, encoding="utf8").read()).split("\n")
M = open(MINE, encoding="utf8").read().rstrip("\n").split("\n")


def need(cond, msg):
    if not cond:
        sys.exit("refused: " + msg)


# their pieces, located in the live file by their own markers
css = L[459:528]
need(css[0].strip().startswith("/* after the stop"), f"css block start: {css[0][:60]!r}")
need(L[458] == V[458] and L[528] == V[459], "css block not bracketed by v5 lines")
s0 = L.index('    <div class="subsec" id="beyond-the-data">')
s1 = s0
while L[s1] != "  </section>":
    s1 += 1
sub = L[s0:s1]
block = "\n".join(sub)
need(block.count("<div") == block.count("</div>"), "subsection divs unbalanced")
vh_first = [i for i, x in enumerate(L) if x.startswith("      <li><b>ThrusterChain v0 (current).</b>")]
need(len(vh_first) == 1, "live version-history item")
vh_add = L[vh_first[0] + 1: vh_first[0] + 6]
need(vh_add[0].strip().startswith("The user did not accept the stop.") and vh_add[-1].strip() == "is unchanged.",
     f"version-history lines: {vh_add[0][:50]!r} / {vh_add[-1][:50]!r}")

out = list(M)
# 1. CSS: after the v5 line that precedes it
i = [k for k, x in enumerate(out) if x == V[458]]
need(len(i) == 1, "css anchor in mine")
out[i[0] + 1: i[0] + 1] = css
# 2. Results subsection: before the closing tag of the results section
ri = [k for k, x in enumerate(out) if '<section id="results">' in x]
need(len(ri) == 1, "results section in mine")
end = ri[0]
while out[end] != "  </section>":
    end += 1
out[end:end] = sub
# 3. Version history: after the first line of the ThrusterChain v0 item
vi = [k for k, x in enumerate(out) if x.startswith("      <li><b>ThrusterChain v0 (current).</b>")]
need(len(vi) == 1, "version-history item in mine")
out[vi[0] + 1: vi[0] + 1] = vh_add
# 4. Contents rail
ci = [k for k, x in enumerate(out) if x == '  <a href="#results">Results</a>']
need(len(ci) == 1, "contents link in mine")
out[ci[0] + 1: ci[0] + 1] = ['  <a href="#beyond-the-data" class="sub">After the stop</a>']
# 5. Top status line: add their pointer sentence before the paragraph closes
st = [k for k, x in enumerate(out) if '<p class="status"><span class="tag warn">stopped before BCA</span> <b>The problem:</b> a test' in x]
need(len(st) == 1, "top status line in mine")
k = st[0]
while "</p>" not in out[k]:
    k += 1
need(out[k].rstrip().endswith("</p>"), "status line end")
out[k] = out[k].rstrip()[:-4] + (' What has happened since, with no BCA run, is in <a href="#beyond-the-data">After the '
                                  'stop</a>.</p>')
text = "\n".join(out) + "\n"
need(text.count('id="beyond-the-data"') == 1, "subsection id once")
open(OUT, "w", encoding="utf8").write(text)
print(f"wrote {OUT}: {len(text):,} characters; subsection {len(sub)} lines, css {len(css)} lines")
