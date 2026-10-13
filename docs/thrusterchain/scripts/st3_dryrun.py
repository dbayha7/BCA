"""Read-only dry run of the structure plan (notes/structure3.md) in memory: asserts every marker and prints the result.
Writes nothing."""
import hashlib, re
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
B = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v11.html"
assert hashlib.md5(open(B, "rb").read()).hexdigest() == "79f00557ba8803625b9f36bc497f7cc6"
s = open(P, encoding="utf-8").read()
n0 = len(s)
a = s.index('\n    <figure class="panel" id="sim"') + 1
b = s.index('\n  <section id="action-space">') + 1
c = s.index('\n  <section id="description">') + 1
assert s.count('\n    <figure class="panel" id="sim"') == s.count('\n  <section id="action-space">') == s.count('\n  <section id="description">') == 1 and a < b < c
assert s[:a].endswith("  </header>\n\n")
simtask, six = s[a:b], s[b:c]
assert simtask.endswith("  </section>\n\n") and six.endswith("  </section>\n\n")
assert six.count('<section id="') == 6 and simtask.count('<section id="') == 1
s = s[:a] + six + simtask + s[c:]
assert len(s) == n0
def once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80]); return s.replace(old, new, 1)
s = once(s, '  <a href="#sim">Watch one episode</a>\n  <a href="#task">The task</a>\n', '')
s = once(s, '  <a href="#episode-end">Episode End</a>\n', '  <a href="#episode-end">Episode End</a>\n  <a href="#sim">Watch one episode</a>\n  <a href="#task">The task</a>\n')
s = once(s, '  <a href="#version-history">Version History</a>\n', '')
s = once(s, 'These are the same\n        float dynamics as in the earlier world this one replaces (<a href="#version-history">Version History</a>).</li>', 'X-DYN</li>')
v0 = s.index('\n  <section id="version-history">') + 1; v1 = s.index('\n  <section id="code-names">') + 1
blk = s[v0:v1]; assert blk.count('<section id="') == 1 and blk.endswith("  </section>\n\n") and "<h2>Version History</h2>" in blk
s = s[:v0] + s[v1:]
s = once(s, '    <div class="panel">\n      <p><span class="tag warn">illustration, not a result</span></p>\n      <div class="iw-chart" id="iw-c-cov">', '    <div class="panel">\n      <div class="iw-chart" id="iw-c-cov">')
s = once(s, '(<a href="#task-strong">Why strong current is the hard case</a>).</p>\n    </div>', 'X-CALLOUT')
s = once(s, '</tbody></table></div>\n  </section>\n</main>', 'X-ROWS</tbody></table></div>\n  </section>\n</main>')
s = once(s, 'Section 1 of 22', 'Section 1 of 21')
s = once(s, 'The live panel "<a href="#sim">Watch one episode</a>" comes next, and right after it the section "<a href="#task">The task</a>" explains the whole task.', 'X-GLANCE')
assert "version-history" not in s and "Version History" not in s and "illustration, not a result" not in s
m0, m1 = s.index("<main>"), s.index("</main>")
order = re.findall(r'\n  <header class="head" id="(top)"|\n  <section id="([^"]+)">|\n    <figure class="panel" id="(sim)"', s[m0:m1])
print("main order:", [x[0] or x[1] or x[2] for x in order])
nav = s[s.index('<nav class="toc"'):s.index('</nav>')]
print("rail:", re.findall(r'href="#([^"]+)"', nav))
print("dry run OK; nothing written")
